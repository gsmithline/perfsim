"""Population + telemetry helpers for run_pokec_gated_lm.py.

Kept transformers-free (torch/numpy only) so they can be imported and mock
tested on a laptop; the LM only ever enters through call arguments. Loaded by
the pipeline via importlib, like _collapse_metrics.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

# AB population: gated Deffuant-with-bias on the Pokec graph, torch-vectorized
# (adapted from experiments/competition/16_model_mass.py, partner selection
# restricted to graph neighbors).

AB_BATCH = 75
AB_MIN_DIST = 1e-5
PPL_BATCH = int(os.environ.get("PPL_BATCH", "64"))   # micro-batch for per_agent_ppl


def peer_gate(dist, eps, mode="threshold"):
    """The ONE peer (Deffuant) confidence-gate definition (2026-08-20,
    qwen_wu_limit wave). Mirrors ai_gate below; every peer sweep and the
    offline replay call it, so the deployed acceptance rule and any
    reconstruction can never diverge.

      threshold  accept iff |x_i - x_j| < eps   (strict <, byte-identical
                 to the pre-2026-08-20 inline expression)
      all_open   accept EVERY sampled pair. NOT encoded as eps=1: the
                 threshold test is a strict inequality, so a pair at
                 (0, 1) sits at distance exactly 1 and would be REJECTED
                 under eps=1. A genuinely open peer channel therefore
                 needs its own mode, never a numeric stand-in.

    `dist` is the absolute opinion distance of the sampled pairs. The
    caller draws pairs BEFORE calling this, so the mode changes only the
    acceptance decision -- never which pairs are sampled, never how much
    RNG the sweep consumes.
    """
    if mode == "all_open":
        return torch.ones_like(dist, dtype=torch.bool)
    if mode != "threshold":
        raise ValueError(f"unknown PEER_GATE_MODE: {mode!r}")
    return dist < eps


def ab_sweep(x, adj, eps, gamma, gen=None, gate_mode="threshold",
             alpha=0.5):
    """Exactly N biased pair selections among graph neighbors; disjoint pairs
    per batch via Luby-style conflict resolution. Mutates x, returns accepted.
    `gen` isolates the population RNG from the global stream, which the HF
    trainer re-seeds every round (frozen pair patterns otherwise).

    gate_mode="threshold" (the default) reproduces every archived run
    byte-for-byte; "all_open" accepts every sampled pair (see peer_gate).
    The mode is applied AFTER pair selection, so both modes draw the same
    pairs from the same generator state.

    alpha is the symmetric Deffuant compromise rate. The production default
    alpha=0.5 retains the legacy midpoint update byte-for-byte; values in
    [0, 0.5] permit the linear peer-influence sweep used by the Section 2
    perfect-prediction baseline."""
    if not 0.0 <= float(alpha) <= 0.5:
        raise ValueError(f"Deffuant alpha must lie in [0, 0.5]; got {alpha}")
    n = x.shape[0]
    device = x.device
    bsz = min(AB_BATCH, n)
    done, accepted = 0, 0
    while done < n:
        ini = torch.randperm(n, device=device, generator=gen)[:bsz]
        ar = torch.arange(ini.shape[0], device=device)
        d = (x[ini, None] - x[None, :]).abs().clamp_min(AB_MIN_DIST)
        wts = d.pow(-gamma) * adj[ini]
        wts[ar, ini] = 0.0
        ok_row = wts.sum(1) > 0
        if not bool(ok_row.any()):
            break  # only possible off the LCC (e.g. tiny mock graphs)
        ini, wts = ini[ok_row], wts[ok_row]
        ar = torch.arange(ini.shape[0], device=device)
        par = torch.multinomial(wts, 1, generator=gen).squeeze(1)
        # Luby-style conflict resolution: keep a pair iff it holds the min
        # random priority at both endpoints
        pri = torch.rand(ini.shape[0], device=device, generator=gen)
        inc = torch.zeros(ini.shape[0], n, dtype=torch.bool, device=device)
        inc[ar, ini] = True
        inc[ar, par] = True
        best = torch.where(inc, pri[:, None], torch.tensor(2.0, device=device)).amin(0)
        keep = (pri <= best[ini]) & (pri <= best[par])
        idx = keep.nonzero().squeeze(1)[: n - done]
        i1, i2 = ini[idx], par[idx]
        ok = peer_gate((x[i1] - x[i2]).abs(), eps, gate_mode)
        a1, a2 = i1[ok], i2[ok]
        if float(alpha) == 0.5:
            # Preserve the archived midpoint computation bit-for-bit.
            mid = 0.5 * (x[a1] + x[a2])
            x[a1] = mid
            x[a2] = mid
        else:
            old1 = x[a1].clone()
            old2 = x[a2].clone()
            x[a1] = old1 + float(alpha) * (old2 - old1)
            x[a2] = old2 + float(alpha) * (old1 - old2)
        done += len(idx)
        accepted += int(ok.sum())
    return accepted


def ab_sweep_stubborn(x, adj, eps, gamma, gen, fixed, gate_mode="threshold"):
    """One-sided STUBBORN Deffuant sweep (INNATE_CLAMP_PEER_MODE=
    stubborn, 2026-08-17). Mutates x, returns (accepted, stats).

    Fixed agents participate fully in pair selection (initiators AND
    partners) but are NEVER moved by any path through this function --
    not even transiently (no move-then-reset; a later pair in the same
    sweep always sees the fixed agent at its standing value). Accepted
    R-R pairs take the ordinary midpoint; an accepted F-R pair moves
    ONLY the responsive endpoint to the midpoint (the fixed endpoint
    stays bit-exact); an accepted F-F pair moves no one.

    The selection machinery is a verbatim copy of ab_sweep (same draws,
    same Luby resolution, same generator consumption) -- only the pair
    UPDATE branches on the mask, so a no-fixed-agent call reproduces
    the legacy sweep bit-for-bit. Legacy ab_sweep above is untouched.

    gate_mode is passed straight to peer_gate, exactly as in ab_sweep.

    stats: fr_sampled / fr_accepted count kept pairs with EXACTLY one
    fixed endpoint (before / after the confidence test); touched is a
    [n] bool of responsive agents that sat in an ACCEPTED F-R pair.
    """
    n = x.shape[0]
    device = x.device
    stats = {"fr_sampled": 0, "fr_accepted": 0,
             "touched": torch.zeros(n, dtype=torch.bool, device=device)}
    bsz = min(AB_BATCH, n)
    done, accepted = 0, 0
    while done < n:
        ini = torch.randperm(n, device=device, generator=gen)[:bsz]
        ar = torch.arange(ini.shape[0], device=device)
        d = (x[ini, None] - x[None, :]).abs().clamp_min(AB_MIN_DIST)
        wts = d.pow(-gamma) * adj[ini]
        wts[ar, ini] = 0.0
        ok_row = wts.sum(1) > 0
        if not bool(ok_row.any()):
            break
        ini, wts = ini[ok_row], wts[ok_row]
        ar = torch.arange(ini.shape[0], device=device)
        par = torch.multinomial(wts, 1, generator=gen).squeeze(1)
        pri = torch.rand(ini.shape[0], device=device, generator=gen)
        inc = torch.zeros(ini.shape[0], n, dtype=torch.bool, device=device)
        inc[ar, ini] = True
        inc[ar, par] = True
        best = torch.where(inc, pri[:, None],
                           torch.tensor(2.0, device=device)).amin(0)
        keep = (pri <= best[ini]) & (pri <= best[par])
        idx = keep.nonzero().squeeze(1)[: n - done]
        i1, i2 = ini[idx], par[idx]
        one_fixed = fixed[i1] ^ fixed[i2]
        ok = peer_gate((x[i1] - x[i2]).abs(), eps, gate_mode)
        a1, a2 = i1[ok], i2[ok]
        f1, f2 = fixed[a1], fixed[a2]
        mid = 0.5 * (x[a1] + x[a2])
        x[a1[~f1]] = mid[~f1]
        x[a2[~f2]] = mid[~f2]
        fr_ok = ok & one_fixed
        stats["fr_sampled"] += int(one_fixed.sum())
        stats["fr_accepted"] += int(fr_ok.sum())
        fr_idx = fr_ok.nonzero().squeeze(1)
        if fr_idx.numel():
            r_end = torch.where(fixed[i1[fr_idx]], i2[fr_idx],
                                i1[fr_idx])
            stats["touched"][r_end] = True
        done += len(idx)
        accepted += int(ok.sum())
    return accepted, stats


def quantile_w1(a, b, grid=512):
    """1-Wasserstein between samples of possibly different sizes via
    quantile interpolation on a shared probability grid (the responsive
    and fixed cohorts differ in size)."""
    qs = torch.linspace(0.0, 1.0, grid)
    return float((torch.quantile(a.float().cpu(), qs)
                  - torch.quantile(b.float().cpu(), qs)).abs().mean())


def gated_blend(x, served, w_agent, eps):
    """x_i <- (1-w_i) x_i + w_i m_i where |m_i - x_i| < eps. Returns (x, contact).

    LEGACY pre-social operator (population_update marker absent). Kept for
    auditing archived runs; new runs use nested_presocial_update below.
    """
    gate = (served - x).abs() < eps
    x = torch.where(gate, (1.0 - w_agent) * x + w_agent * served, x)
    return x, float(gate.float().mean())


def ai_gate(served, x0, eps_ai, mode="threshold"):
    """The ONE AI-gate definition (2026-08-13, sft_icl_reach wave). Runner and
    checker both call this so the deployed update, the dry/counterfactual gate
    calculations, and the offline replay can never diverge.

      threshold  g_i = 1{|m_i - x0_i| < eps_ai}   (strict <, RNG-free)
      all_open   g_i = 1 for every agent. NOT encoded as eps_ai=1: the
                 threshold gate is a strict inequality, so |m - x| = 1 (an
                 agent at 0 served 1) would be REJECTED under eps_ai=1.

    NOTE the second argument is the gate REFERENCE, not necessarily the
    start-of-round opinion. Which vector that is comes from
    gate_reference() below (AI_GATE_REFERENCE); this function only
    measures the distance it is handed, so both semantics share one
    strict-inequality definition.
    """
    if mode == "all_open":
        return torch.ones_like(served, dtype=torch.bool)
    if mode != "threshold":
        raise ValueError(f"unknown AI_GATE_MODE: {mode!r}")
    return (served - x0).abs() < eps_ai


def gate_reference(x0, innate, k, gate_on="anchor"):
    """The ONE definition of the vector the AI gate measures distance FROM
    (AI_GATE_REFERENCE, 2026-08-22 semantic correction).

      "anchor"  x'_i = k innate_i + (1-k) x0_i -- the ANCHORED opinion the
                agent actually holds when the served value is judged. This
                is the intended model: the gate is
                    |m_i(t) - x'_i(t)| < eps_AI
                with x' the same human component the mixture blends into,
                so the acceptance test and the state it updates are the
                same vector. DEFAULT since 2026-08-22.
      "x0"      the raw start-of-round opinion. Reproduces every archived
                run bit-for-bit; kept reachable (and named) so an offline
                replay of a pre-correction trajectory is exact rather than
                approximate.

    k IS gamma in the write-up's x'_i = gamma x_innate + (1-gamma) x_i, and
    is INNATE_LAMBDA in the runner. Note k = 0 makes the two references
    IDENTICAL (x' == x0), which is why the whole k=0 archive is unaffected
    by the correction.
    """
    if gate_on == "x0":
        return x0
    if gate_on != "anchor":
        raise ValueError(f"unknown AI_GATE_REFERENCE: {gate_on!r} "
                         f"(want 'anchor' or 'x0')")
    return k * innate + (1.0 - k) * x0


def nested_presocial_update(x0, served, innate, k, w_agent, eps_ai,
                            gate_mode="threshold", gate_on="anchor"):
    """Pre-social round operator.

        h_i = k innate_i + (1-k) x0_i                      (human component)
        r_i = gate_reference(x0, innate, k, gate_on)        (gate reference)
        g_i = ai_gate(m, r, eps_ai, gate_mode)
        z_i = (1-w_i) h_i + w_i m_i  if g_i else h_i        (platform mixture)

    gate_on="anchor" (the DEFAULT since 2026-08-22, config marker
    population_update="nested_ai_anchored_then_social_v2") evaluates the
    gate against r == h: the agent judges the served value against the
    opinion it is actually holding, which is the anchored one. gate_on="x0"
    evaluates it against the raw start-of-round opinion and reproduces every
    archived run (population_update="nested_ai_then_social_v1") bit-for-bit.

    The correction is INERT wherever the distance is not read:
      * k == 0        -> h == x0, so the two references are the same vector;
      * all_open      -> ai_gate ignores the distance entirely.
    Only k > 0 together with gate_mode="threshold" can change anything.

    The innate pull dilutes only the human share either way: w_i = 1 on a
    gated agent gives z_i = m_i for every k.

    gate_mode="threshold" (the default) is the strict |m - r| < eps_ai test;
    "all_open" opens the gate for every agent (see ai_gate).

    Pure and side-effect free; peer (Deffuant) dynamics run AFTER this on z.
    Returns (z, gate) with gate the boolean acceptance mask.
    """
    h = k * innate + (1.0 - k) * x0
    gate = ai_gate(served, gate_reference(x0, innate, k, gate_on),
                   eps_ai, gate_mode)
    eff_w = torch.where(gate, w_agent, torch.zeros_like(w_agent))
    return (1.0 - eff_w) * h + eff_w * served, gate


# ---------------------------------------------------------------------------
# REFERENCE REPLAY (REF_REPLAY_*, 2026-08-22).
#
# Each round the learner is handed a FULL n-row training set
#
#     y_i^(t) = x_i(t)   if i in S_t   (live: this round's real label)
#               b_i      otherwise      (frozen reference prediction)
#
# with b a PINNED vector from another run. Every helper below is pure and
# STATELESS -- the round's live set is a function of (seed, round) alone,
# never of a generator that the loop advances -- so the runner, the tests
# and any offline checker reconstruct the identical sets and labels.

def ref_replay_n_live(n, q):
    """|S_t| = round(q n), the live-row count at fraction q.

    Depends on (n, q) only -- NOT on the round -- so the sample size is
    constant across the run and comparable across arms. q = 1 gives
    exactly n (ordinary SFT); a q so small it would empty the live set is
    an error, not a silent all-reference run."""
    q = float(q)
    if not (0.0 < q <= 1.0):
        raise ValueError(f"REF_REPLAY_Q must be in (0, 1]; got {q!r}")
    n = int(n)
    k = int(round(q * n))
    if not 0 < k <= n:
        raise ValueError(f"REF_REPLAY_Q={q!r} on n={n} gives a degenerate "
                         f"live set ({k} rows)")
    return k


def ref_replay_perm(n, seed, round_t):
    """The ONE permutation of 0..n-1 for (seed, round).

        torch.Generator().manual_seed(seed + t) -> randperm(n)

    the PROJECT'S STANDARD per-round selection stream, identical to the
    one SFT_SAMPLE_N uses and that check_pofd_sanity already replays, so
    a checker reconstructs the live set with the same one-liner rather
    than a bespoke hash. (It aliases (seed, t) with (seed+1, t-1); that
    is harmless here because a wave shares ONE REF_REPLAY_SEED across
    arms and what the design needs is only that consecutive rounds
    differ and that the draw is reconstructible.)

    STATELESS by construction: a dedicated generator seeded from
    (seed, round) only. Nothing about q enters, which is exactly what
    makes the live sets NESTED -- S_t(q=.10) is the first 72 entries of
    the same permutation whose first 145 entries are S_t(q=.20) -- and
    makes that nesting hold ACROSS ARMS at the same round, not just
    within one run. It also draws from no stream the loop shares, so a
    ref-replay run consumes the same RNG as its q=1 twin."""
    g = torch.Generator()
    g.manual_seed(int(seed) + int(round_t))
    return torch.randperm(int(n), generator=g)


def ref_replay_live(n, q, seed, round_t):
    """S_t: the first round(q n) entries of the round's permutation, kept
    IN PERMUTATION ORDER so a smaller q is a literal PREFIX of a larger
    one (subset either way; the prefix is the stronger, checkable form).

    Refreshed every round INCLUDING round 0 -- round 0 is a draw like any
    other, not the full-data round it is under REPLAY_FRAC."""
    return ref_replay_perm(n, seed, round_t)[:ref_replay_n_live(n, q)]


def ref_replay_labels(x_live, ref_vec, live_idx):
    """The exact n-row label vector handed to the learner:

        y = b everywhere, then y[S_t] <- x(t)[S_t]

    Rebuilt FROM b every call. There is no path by which a previous
    round's substituted label can survive into this one -- accumulation
    is impossible by construction, not by discipline -- and rows stay in
    CANONICAL agent order 0..n-1, so row i is always agent i."""
    x_live = x_live.detach().float()
    ref_vec = ref_vec.detach().float()
    if x_live.shape != ref_vec.shape or x_live.ndim != 1:
        raise ValueError(f"ref replay: live labels {tuple(x_live.shape)} and "
                         f"reference vector {tuple(ref_vec.shape)} must be the "
                         f"same 1-D [n] shape")
    live_idx = live_idx.long()
    y = ref_vec.clone()
    y[live_idx] = x_live[live_idx]
    return y


def ref_replay_hash(vec):
    """sha256 over b's raw float32 bytes -- the config carries it so a
    swapped or truncated reference vector is detectable without holding
    the donor run."""
    import hashlib
    return hashlib.sha256(
        vec.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def validate_ref_replay_vec(vec, n=None, where="REF_REPLAY_REF_RUN"):
    """Hard-fail unless b is n finite opinions in [0, 1].

    A nan or an out-of-range entry in b would become a training label on
    (1-q) n rows every round and be indistinguishable from a real one in
    the artifact, so it is rejected at load time rather than trained on."""
    if vec.ndim != 1:
        raise ValueError(f"{where}: reference vector must be 1-D [n]; got "
                         f"{tuple(vec.shape)}")
    if n is not None and int(vec.shape[0]) != int(n):
        raise ValueError(f"{where}: reference vector has {int(vec.shape[0])} "
                         f"agents, population has {int(n)}")
    if not bool(torch.isfinite(vec).all()):
        raise ValueError(f"{where}: reference vector has "
                         f"{int((~torch.isfinite(vec)).sum())} non-finite "
                         f"entries")
    lo, hi = float(vec.min()), float(vec.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(f"{where}: reference vector out of [0, 1] "
                         f"(min {lo:.6g}, max {hi:.6g})")
    return vec


def make_canary(n, delta, seed):
    """Fixed per-agent +/-delta pattern, seeded; zeros when delta == 0."""
    if delta <= 0:
        return torch.zeros(n)
    g = torch.Generator().manual_seed(seed + 99991)
    sign = torch.randint(0, 2, (n,), generator=g, dtype=torch.float32) * 2.0 - 1.0
    return sign * delta


def select_probe_indices(innate, n_probe):
    """n_probe real agents stratified by innate: evenly spaced ranks of the
    innate ordering (chosen once, fixed for the whole run)."""
    order = torch.argsort(innate)
    ranks = torch.linspace(0, innate.shape[0] - 1, n_probe).round().long()
    return order[ranks]


def _largest_remainder(sizes, total):
    """Integer quotas proportional to sizes summing exactly to total.
    Deterministic: leftover units go to the largest fractional remainders,
    ties broken by lower bin index."""
    n_all = sum(sizes)
    exact = [total * s / n_all for s in sizes]
    base = [int(e) for e in exact]
    left = total - sum(base)
    by_rem = sorted(range(len(sizes)),
                    key=lambda i: (-(exact[i] - base[i]), i))
    for i in by_rem[:left]:
        base[i] += 1
    return base


def innate_clamp_mask(innate, mode, frac, seed):
    """Boolean [n] cohort mask for the permanent innate clamp
    (INNATE_CLAMP_MODE, 2026-08-17 mistral_innate_clamp_nopeer wave).

    Deterministic in (innate, mode, frac, seed) ONLY: randomness comes
    from a dedicated torch.Generator, never the global stream, so the
    SFT and ICL arms and every AI gate at a given INNATE_CLAMP_SEED
    share the bit-identical cohort, and building the mask perturbs no
    RNG the loop consumes.

    Cohort size is round(frac*n) exactly (723 @ 0.20 -> 145). "bottom"
    takes the lowest innate opinions with agent id as the deterministic
    tie-break. "stratified_random" ranks agents by (innate, id), splits
    the ranking into 5 quintile bins (largest-remainder sizing), and
    samples each bin's proportional quota without replacement, so the
    frozen cohort represents the innate distribution.
    """
    n = int(innate.numel())
    n_frozen = int(round(float(frac) * n))
    if not 0 < n_frozen < n:
        raise ValueError(f"INNATE_CLAMP_FRAC={frac!r} gives a degenerate "
                         f"cohort ({n_frozen} of {n})")
    order = sorted(range(n), key=lambda i: (float(innate[i]), i))
    if mode == "bottom":
        idx = order[:n_frozen]
    elif mode == "stratified_random":
        n_bins = 5
        cuts = _largest_remainder([1] * n_bins, n)   # near-even rank bins
        bins, lo = [], 0
        for c in cuts:
            bins.append(order[lo:lo + c])
            lo += c
        quotas = _largest_remainder([len(b) for b in bins], n_frozen)
        g = torch.Generator()
        g.manual_seed(int(seed))
        idx = []
        for b, q in zip(bins, quotas):
            perm = torch.randperm(len(b), generator=g)[:q]
            idx.extend(b[int(j)] for j in perm)
    else:
        raise ValueError(f"unknown INNATE_CLAMP_MODE {mode!r} (want "
                         f"'off', 'stratified_random' or 'bottom')")
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.tensor(sorted(idx), dtype=torch.long)] = True
    assert int(mask.sum()) == n_frozen
    return mask


def innate_clamp_hash(mask):
    """sha256 over the raw cohort bytes -- the trajectory carries it so a
    tampered/truncated mask is detectable without reconstruction."""
    import hashlib
    return hashlib.sha256(
        mask.detach().cpu().to(torch.uint8).numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Telemetry helpers: everything loss/likelihood-based, from the platform seat
# (mirrors experiments/competition/18_platform_telemetry.py).

def snapshot_trainable(module):
    """CPU fp32 copy of the trainable (adapter) parameters."""
    return {k: v.detach().to("cpu", torch.float32).clone()
            for k, v in module.named_parameters() if v.requires_grad}


class swapped_params:
    """Temporarily load a trainable-param snapshot into `module` (round-0
    adapter evals); restores the current weights on exit."""

    def __init__(self, module, snap):
        self.module = module
        self.snap = snap
        self.saved = None

    def __enter__(self):
        params = dict(self.module.named_parameters())
        self.saved = {k: params[k].detach().clone() for k in self.snap}
        with torch.no_grad():
            for k, v in self.snap.items():
                params[k].copy_(v.to(params[k].device, params[k].dtype))
        return self.module

    def __exit__(self, *exc):
        params = dict(self.module.named_parameters())
        with torch.no_grad():
            for k, v in self.saved.items():
                params[k].copy_(v)
        return False


def load_trainable(module, snap):
    """Permanently copy a trainable-param snapshot into `module` (no restore).
    Used for fresh-each-round retraining: reset the adapter to its pristine
    base-behavior state before fitting a new model on the round's data."""
    params = dict(module.named_parameters())
    with torch.no_grad():
        for k, v in snap.items():
            params[k].copy_(v.to(params[k].device, params[k].dtype))


def adapter_step(cur, prev=None):
    """L2 norm of the trainable (LoRA) params, or of their change vs `prev`.
    adapter_step(cur, prev) = ||theta_t - theta_{t-1}|| = the weight-space
    performative-stability step (decays to 0 at a stable point). adapter_step(cur)
    = ||theta_t|| (so a relative step = step/norm is recoverable offline)."""
    total = 0.0
    for k in cur:
        d = cur[k] if prev is None else (cur[k] - prev[k])
        total += float((d * d).sum())
    return total ** 0.5


def _example_ids(lm, agent_i, y_i, fmt):
    """(input_ids, labels) for one prompt+completion example, prompt masked."""
    prompt = lm.build_prompt(lm.profile_at(int(agent_i)))
    target = fmt(float(y_i))
    n_prompt = lm.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    ids = lm.tokenizer(prompt + target, return_tensors="pt",
                       truncation=True).input_ids.to(lm._target_device)
    labels = ids.clone()
    labels[:, :n_prompt] = -100
    return ids, labels


@torch.no_grad()
def sft_batch_loss(lm, train_data, fmt, cap, module=None):
    """Mean completion-token CE (the SFT objective without the KL term) of
    `module` (default: current adapter) on the first `cap` rows of the batch.
    One forward pass per example, no padding."""
    module = module if module is not None else lm.inner_model
    idx = train_data["agent_idx"]
    y = train_data["y"].squeeze(-1) if train_data["y"].ndim > 1 else train_data["y"]
    n = idx.shape[0] if cap <= 0 else min(cap, idx.shape[0])
    was_cache = bool(getattr(module.config, "use_cache", False))
    module.config.use_cache = False
    total_nll, total_tok = 0.0, 0
    try:
        for i in range(n):
            ids, labels = _example_ids(lm, idx[i], y[i], fmt)
            out = module(ids, labels=labels)
            ntok = max(int((labels[:, 1:] != -100).sum()), 1)
            total_nll += float(out.loss) * ntok
            total_tok += ntok
    finally:
        module.config.use_cache = was_cache
    return total_nll / max(total_tok, 1)


@torch.no_grad()
def batched_answer_losses(module, examples, pad_id, micro=PPL_BATCH):
    """Per-example mean answer-token CE for a list of (ids[1,L], labels[1,L]),
    right-padded and run in micro-batches. Equals module(ids, labels=labels).loss
    per example: right padding plus the causal mask leaves each real token's
    logits identical to the unpadded forward, so the per-row loss matches while
    one weight-streaming pass now covers `micro` agents instead of one."""
    losses = []
    for s in range(0, len(examples), micro):
        chunk = examples[s:s + micro]
        b, lmax = len(chunk), max(e[0].shape[1] for e in chunk)
        dev = chunk[0][0].device
        input_ids = torch.full((b, lmax), pad_id, dtype=torch.long, device=dev)
        labels = torch.full((b, lmax), -100, dtype=torch.long, device=dev)
        attn = torch.zeros((b, lmax), dtype=torch.long, device=dev)
        for r, (ids, lab) in enumerate(chunk):
            length = ids.shape[1]
            input_ids[r, :length] = ids[0]
            labels[r, :length] = lab[0]
            attn[r, :length] = 1
        logits = module(input_ids=input_ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
        shift_labels = labels[:, 1:].reshape(-1)
        tok_ce = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100,
                                 reduction="none").view(b, -1)
        valid = (labels[:, 1:] != -100).float()
        row_loss = (tok_ce * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        losses.extend(row_loss.detach().float().cpu().tolist())
    return losses


def per_agent_ppl(lm, idx, y, fmt, cap=0, gen=None, module=None, micro=PPL_BATCH):
    """Per-agent answer-token perplexity: how surprised the current model is by
    each agent's target opinion y_i (prompt masked, like sft_batch_loss but kept
    per-example, not averaged). Returns (ppl_list, scored_idx) -- the empirical
    perplexity distribution over profiles. cap>0 scores a random cap-sized subset.
    Batched over agents (see batched_answer_losses) instead of one forward each."""
    module = module if module is not None else lm.inner_model
    n = len(idx)
    order = list(range(n))
    if cap and n > cap:
        order = sorted(torch.randperm(n, generator=gen)[:cap].tolist())
    if not order:
        return [], []
    pad_id = lm.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = lm.tokenizer.eos_token_id
    was = bool(getattr(module.config, "use_cache", False))
    module.config.use_cache = False
    try:
        examples = [_example_ids(lm, int(idx[k]), float(y[k]), fmt) for k in order]
        losses = batched_answer_losses(module, examples, pad_id, micro)
    finally:
        module.config.use_cache = was
    ppl = [float(np.exp(min(l, 60.0))) for l in losses]
    scored = [int(idx[k]) for k in order]
    return ppl, scored


def sft_grad_norm(lm, train_data, fmt, n_examples):
    """Trainable-grad norm of the mean CE over the first n_examples rows: the
    step-0 gradient of the SFT objective (KL term excluded -- the easy grab)."""
    module = lm.inner_model
    idx = train_data["agent_idx"]
    y = train_data["y"].squeeze(-1) if train_data["y"].ndim > 1 else train_data["y"]
    n = min(n_examples, idx.shape[0])
    if n == 0:
        return 0.0
    module.zero_grad(set_to_none=True)
    was_cache = bool(getattr(module.config, "use_cache", False))
    module.config.use_cache = False
    try:
        for i in range(n):
            ids, labels = _example_ids(lm, idx[i], y[i], fmt)
            # per-example backward accumulates grads without retaining graphs
            (module(ids, labels=labels).loss / n).backward()
        sq = 0.0
        for p in module.parameters():
            if p.requires_grad and p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum())
        return float(np.sqrt(sq))
    finally:
        module.config.use_cache = was_cache
        module.zero_grad(set_to_none=True)


def kl_grad_decompose(lm, learner, train_data, fmt, n_examples):
    """Step-0 gradient decomposition of the KL-SFT objective on the first
    n_examples rows: |g_CE|, |beta*g_KL|, ratio and cosine. Same protocol as
    sft_grad_norm (per-example backward, mean over n; grad_norm0 here equals
    its output). The KL term mirrors KLSFTLearner.compute_loss: completion
    tokens only (labels != -100), next-token shift, mean over completion
    tokens, scaled by kl_beta, same kl_direction as the learner. Rescue-by-force
    reads as ratio >> 1 with negative cosine; feedback alignment as the cosine
    rotating positive."""
    kl_beta = float(getattr(learner, "kl_beta", 0.0))
    kl_direction = str(getattr(learner, "kl_direction", "reverse"))
    if kl_beta <= 0:
        return {}
    module = lm.inner_model
    ref = learner._ensure_ref()
    idx = train_data["agent_idx"]
    y = train_data["y"].squeeze(-1) if train_data["y"].ndim > 1 else train_data["y"]
    n = min(n_examples, idx.shape[0])
    if n == 0:
        return {}
    was_cache = bool(getattr(module.config, "use_cache", False))
    module.config.use_cache = False
    try:
        module.zero_grad(set_to_none=True)
        for i in range(n):
            ids, labels = _example_ids(lm, idx[i], y[i], fmt)
            (module(ids, labels=labels).loss / n).backward()
        task_sq, g_task = 0.0, {}
        for name, p in module.named_parameters():
            if p.requires_grad and p.grad is not None:
                g = p.grad.detach().float()
                task_sq += float(g.pow(2).sum())
                g_task[name] = g.cpu()
        module.zero_grad(set_to_none=True)
        for i in range(n):
            ids, labels = _example_ids(lm, idx[i], y[i], fmt)
            logits = module(ids).logits
            with torch.no_grad():
                ref_logits = ref(input_ids=ids).logits
            logp = F.log_softmax(logits[:, :-1, :], dim=-1)
            logq = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            mask_shift = (labels != -100).float()[:, 1:]
            if kl_direction == "forward":
                kl_per_token = (logq.exp() * (logq - logp)).sum(dim=-1)
            else:
                kl_per_token = (logp.exp() * (logp - logq)).sum(dim=-1)
            kl = (kl_per_token * mask_shift).sum() / mask_shift.sum().clamp_min(1.0)
            (kl_beta * kl / n).backward()
        kl_sq, dot = 0.0, 0.0
        for name, p in module.named_parameters():
            if p.requires_grad and p.grad is not None:
                g = p.grad.detach().float()
                kl_sq += float(g.pow(2).sum())
                if name in g_task:
                    dot += float((g.cpu() * g_task[name]).sum())
        task_norm, kl_norm = float(np.sqrt(task_sq)), float(np.sqrt(kl_sq))
        return {"grad_norm0": task_norm, "grad_kl_norm0": kl_norm,
                "grad_ratio0": kl_norm / max(task_norm, 1e-12),
                "grad_cos0": dot / max(task_norm * kl_norm, 1e-12)}
    finally:
        module.config.use_cache = was_cache
        module.zero_grad(set_to_none=True)


@torch.no_grad()
def probe_predictions(lm, probe_prompts):
    """Greedy generations on the fixed probe prompts, parsed to [0,1] floats.

    Mirrors HFCausalLMModel.forward's telemetry side effects. forward() is
    what normally records _last_raw / _last_parse_fail, but the in-context
    serving path calls THIS function instead, so without the mirror every
    ICL run reports the stale __init__ values -- raw=[] and
    parse_fail_frac=0.0 -- under DEBUG_GEN=1. That silently reads as "no
    parse failures" when nothing was measured at all (found 2026-08-15
    while diagnosing mistral7b K=32 serving a constant 0.5).

    Telemetry only: the returned values are the same lm._parse over the
    same generations, so served predictions are bit-identical and no
    archived run is affected.
    """
    outputs = lm._generate(probe_prompts)
    lm._last_raw = list(outputs)
    lm._last_parse_fail = sum(
        1 for o in outputs if re.search(r"\d", o) is None
    ) / max(1, len(outputs))
    return [lm._parse(o) for o in outputs]


def append_telemetry(path, row):
    """One JSON object per line, appended and flushed each round (crash-safe)."""
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
