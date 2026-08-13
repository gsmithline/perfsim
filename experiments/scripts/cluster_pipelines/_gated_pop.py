"""Population + telemetry helpers for run_pokec_gated_lm.py.

Kept transformers-free (torch/numpy only) so they can be imported and mock
tested on a laptop; the LM only ever enters through call arguments. Loaded by
the pipeline via importlib, like _collapse_metrics.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

# AB population: gated Deffuant-with-bias on the Pokec graph, torch-vectorized
# (adapted from experiments/competition/16_model_mass.py, partner selection
# restricted to graph neighbors).

AB_BATCH = 75
AB_MIN_DIST = 1e-5
PPL_BATCH = int(os.environ.get("PPL_BATCH", "64"))   # micro-batch for per_agent_ppl


def ab_sweep(x, adj, eps, gamma, gen=None):
    """Exactly N biased pair selections among graph neighbors; disjoint pairs
    per batch via Luby-style conflict resolution. Mutates x, returns accepted.
    `gen` isolates the population RNG from the global stream, which the HF
    trainer re-seeds every round (frozen pair patterns otherwise)."""
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
        ok = (x[i1] - x[i2]).abs() < eps
        a1, a2 = i1[ok], i2[ok]
        mid = 0.5 * (x[a1] + x[a2])
        x[a1] = mid
        x[a2] = mid
        done += len(idx)
        accepted += int(ok.sum())
    return accepted


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

      threshold  g_i = 1{|m_i - x0_i| < eps_ai}   (strict <, gate on the
                 START-OF-ROUND opinion; byte-identical to the pre-2026-08-13
                 inline expression, and RNG-free either way)
      all_open   g_i = 1 for every agent. NOT encoded as eps_ai=1: the
                 threshold gate is a strict inequality, so |m - x| = 1 (an
                 agent at 0 served 1) would be REJECTED under eps_ai=1.
    """
    if mode == "all_open":
        return torch.ones_like(served, dtype=torch.bool)
    if mode != "threshold":
        raise ValueError(f"unknown AI_GATE_MODE: {mode!r}")
    return (served - x0).abs() < eps_ai


def nested_presocial_update(x0, served, innate, k, w_agent, eps_ai,
                            gate_mode="threshold"):
    """Pre-social round operator (population_update="nested_ai_then_social_v1").

        h_i = k innate_i + (1-k) x0_i                      (human component)
        g_i = ai_gate(m, x0, eps_ai, gate_mode)             (gate on x0)
        z_i = (1-w_i) h_i + w_i m_i  if g_i else h_i        (platform mixture)

    The gate is evaluated against the START-OF-ROUND opinion x0 -- not against
    h and not against any peer-updated state -- because the platform serves
    before the population moves. The innate pull therefore dilutes only the
    human share: w_i = 1 on a gated agent gives z_i = m_i for every k.

    gate_mode="threshold" (the default) reproduces the pre-2026-08-13 behavior
    byte-for-byte; "all_open" opens the gate for every agent (see ai_gate).

    Pure and side-effect free; peer (Deffuant) dynamics run AFTER this on z.
    Returns (z, gate) with gate the boolean acceptance mask.
    """
    h = k * innate + (1.0 - k) * x0
    gate = ai_gate(served, x0, eps_ai, gate_mode)
    eff_w = torch.where(gate, w_agent, torch.zeros_like(w_agent))
    return (1.0 - eff_w) * h + eff_w * served, gate


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
    """Greedy generations on the fixed probe prompts, parsed to [0,1] floats."""
    outputs = lm._generate(probe_prompts)
    return [lm._parse(o) for o in outputs]


def append_telemetry(path, row):
    """One JSON object per line, appended and flushed each round (crash-safe)."""
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
