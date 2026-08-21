#!/usr/bin/env python3
"""ADAPTER KL / SOFT-DECODE PROBE for the SFT training-dose scouts
(2026-08-21). GPU, H100 only.

THE QUESTION THIS EXISTS TO ANSWER. The dose wave measured distance in
SERVED-OPINION space, and served opinions come out of greedy decoding.
Greedy is an argmax, so it is a DISCONTINUOUS functional of the model:
if the base assigns P(0.43) = .49 and P(0.65) = .48, an update that
barely moves the distribution can flip the served number by .22, and two
very different distributions can serve the same number. Frozen Qwen on
this task serves only FIVE distinct values, 98.9% of them either 0.25 or
0.65, so the entire served map is one token-level decision -- exactly the
regime where argmax amplifies a tiny logit change into a large number.

So the dose result ("weak SFT does not stay near frozen Qwen") is
established for the GREEDY SERVED MAP, which is what the feedback loop
actually consumes, and is NOT established for the token distribution.
Small optimizer steps are a heuristic for distributional closeness, not
a constraint on it; a trust region constrains KL explicitly and SFT does
not. This probe measures the distributional statement directly.

WHAT IT COMPUTES, per agent and per adapter:

  KL over the answer span. Teacher-force the BASE's own greedy answer
  tokens and compare next-token distributions position by position, in
  both directions. Both models are scored in the SAME forward setup --
  the adapter pass and the base pass differ only by peft's
  disable_adapter() -- so nothing about padding, dtype or position ids
  can differ between them.

  SOFT-DECODED VALUE, the continuous analogue of the served opinion.
  At the decision position t* we hold the base's answer prefix fixed,
  substitute each candidate token, parse the resulting string through
  the SAME parser the simulator uses, and take the expectation of that
  value under the model's own distribution. The value map is defined ONCE
  by the base model and reused unchanged for every adapter, so the base
  and adapter soft values live in one fixed reference frame and their
  difference is purely distributional. If low-dose adapters sit near the
  base in soft value while their greedy vectors do not, the flat greedy
  distance is an argmax artifact. If the soft values are equally far,
  the implicit anchor genuinely is not operating over this range.

  t* IS NOT ASSUMED. It is chosen per agent as the answer position where
  the base's own uncertainty carries the most VALUE variance:
  leverage(t) = sd of the substituted value under the base distribution
  at t. The leverage of every position is recorded, so a reader can see
  whether t* dominates or the value uncertainty is spread out. Nothing
  here hard-codes "the digit after 0.".

  BASE TOP-2 MARGIN at t*. The mechanical test of the amplification
  story: a knife-edge base (small margin) can be flipped by a tiny
  logit change, a confident one cannot. This is a property of the BASE
  alone and is what decides whether the argmax explanation is even
  available.

WHAT IT DOES NOT ESTABLISH. KL rises with dose almost by construction --
more training moves the weights further. A rising KL curve on its own is
not evidence for the implicit anchor. The informative comparison is
between the SHAPES of the greedy and soft curves over the same dial,
which is why both are written to the same CSV.

TRUNCATED SUPPORT, STATED PLAINLY. The value map is built on a support
set S = (every token that decodes to something starting with a digit)
union (the base's top-M tokens at that position). Soft values are
expectations over S renormalized, and the mass falling OUTSIDE S is
recorded per agent per model as tail_mass. The checker refuses the run
if that mass is not negligible; it is never silently renormalized away.

SELF-CHECKS THAT MAKE THE NUMBERS TRUSTWORTHY (all hard failures):
  * the base's parsed served vector must reproduce the canonical frozen
    Qwen2.5 K=D=0 sha256 -- the same pin the dose analyzer uses. This
    proves the probe's serving path is the archived runs' path, not a
    lookalike. It is hardware-specific, hence H100 only.
  * teacher-forced base argmax must equal the generated answer token at
    every position. Left padding plus a plain forward silently uses the
    wrong RoPE positions unless position_ids are supplied; this check
    fails loudly if that (or any other batching slip) happens.
  * the adapter list is read from the ON-DISK condor config files the
    jobs actually ran from, never re-derived from a tag grammar.

Usage (cluster):
  python probe_adapter_kl.py --out-dir runs/adapter_kl_probe
  python probe_adapter_kl.py --limit-agents 32 --max-adapters 2 --smoke
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
REPO = HERE.parent.parent.parent

# canonical frozen Qwen2.5-7B-Instruct K=D=0 served vector (H100, greedy,
# max_new_tokens=6). Pinned identically in check_pofd_sanity.py, the dose
# analyzer and the mechanism audit.
CANON_SHA = "1674ee5f8d833f46de672791d933e1d3bdeefb07484c2d110dec84ce71da30bb"
N_AGENTS = 723
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 6          # MAX_NEW_TOKENS in every dose job
GEN_BATCH = 32              # GEN_BATCH_SIZE=32 in every dose job
SUPPORT_TOP_M = 16          # base top-M tokens folded into the support
DOSE_CONFIGS = ("configs_pofd_qwen_sft_update_dose.txt",
                "configs_pofd_qwen_sft_lr_dose.txt",
                "configs_pofd_qwen_sft_rank_dose.txt")
ADAPTER_SUBDIR = "round0_adapter"


# =====================================================================
# pure helpers -- no HF, no CUDA, no filesystem. Everything the numbers
# depend on lives here so the unit tests can drive it with synthetic
# tensors on a machine that never loads a language model.
# =====================================================================

def sha_vec(x) -> str:
    """sha256 of a float32 vector, matching the dose analyzer's pin."""
    a = np.asarray(x, dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def numeric_support_ids(decoded_vocab) -> list[int]:
    """Token ids whose decoded text can carry the leading digits of a
    number: optional whitespace then a digit. The parser reads the FIRST
    numeric run in the answer, so these are the tokens that can change
    the served value when substituted into a numeric answer."""
    pat = re.compile(r"^\s*\d")
    return [i for i, s in enumerate(decoded_vocab)
            if isinstance(s, str) and pat.match(s)]


def value_map_for(ans_ids, t, support_ids, decode_fn, parse_fn):
    """Value of the answer when the token at position t is replaced by
    each support token and the base's remaining tokens are kept.

    Keeping the tail fixed is deliberate: it makes the map a FUNCTION of
    the substituted token alone, so the same map applies to every model.
    Re-generating the tail per candidate would make the map depend on
    whichever model produced it, which is the thing being measured.
    """
    pre, post = list(ans_ids[:t]), list(ans_ids[t + 1:])
    seqs = [pre + [int(j)] + post for j in support_ids]
    return np.asarray([parse_fn(s) for s in decode_fn(seqs)],
                      dtype=np.float64)


def restrict(probs, support_ids):
    """(mass on the support, renormalized probabilities on it)."""
    p = np.asarray(probs, dtype=np.float64)[np.asarray(support_ids)]
    m = float(p.sum())
    if m <= 0.0:
        return 0.0, np.full(p.shape, 1.0 / max(p.size, 1))
    return m, p / m


def soft_value(probs, support_ids, values):
    """E[value] under the model, restricted to the support and
    renormalized. Returns (soft value, mass outside the support)."""
    mass, q = restrict(probs, support_ids)
    return float((q * np.asarray(values, dtype=np.float64)).sum()), 1.0 - mass


def leverage(probs, support_ids, values):
    """sd of the substituted value under the model's own distribution at
    this position: how much VALUE uncertainty this token carries. Zero
    at a position whose alternatives all parse to the same number."""
    _, q = restrict(probs, support_ids)
    v = np.asarray(values, dtype=np.float64)
    mu = float((q * v).sum())
    return float(np.sqrt(max((q * (v - mu) ** 2).sum(), 0.0)))


def pick_tstar(levs) -> int:
    """Position with the most value leverage; earliest on a tie. Ties are
    broken deterministically so two runs of the probe agree exactly."""
    a = np.asarray(levs, dtype=np.float64)
    if a.size == 0:
        raise ValueError("pick_tstar: no positions")
    return int(np.argmax(a))


def kl_rows(logp_p, logp_q):
    """KL(p || q) per row, from log-probability rows. Clamped at 0: the
    quantity is non-negative and only float error can take it below."""
    lp = np.asarray(logp_p, dtype=np.float64)
    lq = np.asarray(logp_q, dtype=np.float64)
    return np.maximum((np.exp(lp) * (lp - lq)).sum(axis=-1), 0.0)


def top2_margin(probs):
    """(top-1 prob, top-1 minus top-2 prob). The knife-edge diagnostic:
    a small margin is what lets a tiny logit change flip the argmax."""
    p = np.sort(np.asarray(probs, dtype=np.float64))[::-1]
    if p.size < 2:
        return float(p[0]), float(p[0])
    return float(p[0]), float(p[0] - p[1])


def strip_tail(ids, stop_ids):
    """Drop trailing pad/eos so the teacher-forced span is the real
    answer. Stops at the FIRST stop token: everything after it was never
    generated in any meaningful sense."""
    out = []
    for t in list(ids):
        if int(t) in stop_ids:
            break
        out.append(int(t))
    return out


def read_dose_tags(condor_dir, config_names=DOSE_CONFIGS):
    """Adapter tags read from the ON-DISK condor config files the jobs ran
    from. The tag grammar is NOT re-derived here: deriving one string in
    two places is exactly how the analyzer's 5em05/5em5 mismatch and the
    submit script's brace bug happened."""
    tags = []
    for name in config_names:
        p = Path(condor_dir) / name
        if not p.exists():
            raise SystemExit(f"[akl] missing dose config {p} -- run "
                             f"gen_pofd_sweep.py first")
        for line in p.read_text().splitlines():
            if line.strip():
                tags.append(line.split(",")[0].strip())
    if len(tags) != len(set(tags)):
        raise SystemExit(f"[akl] duplicate tags across {config_names}")
    return tags


# =====================================================================
# GPU stages
# =====================================================================

def _load_run_module():
    spec = importlib.util.spec_from_file_location(
        "run_gated", str(HERE / "run_pokec_gated_lm.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _position_ids(attn):
    """What HF's generation path feeds a left-padded batch. A plain
    forward() defaults to arange, which puts the RoPE phase in the wrong
    place for every padded row; the teacher-forced argmax self-check
    exists to catch that, and this is the fix it checks for."""
    return (attn.cumsum(-1) - 1).clamp(min=0)


def build_prompts(setup, tokenizer, n=None):
    bp = setup["build_prompt"]
    profiles = setup["profiles"]
    idx = range(len(profiles) if n is None else min(n, len(profiles)))
    return [bp(profiles.iloc[i], tokenizer) for i in idx]


@torch.no_grad()
def generate_answers(model, tokenizer, prompts, device, batch=GEN_BATCH):
    """Greedy answers with the dose jobs' exact decoding settings."""
    stop = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    stop = {int(s) for s in stop if s is not None}
    ids, texts = [], []
    model.eval()
    for i in range(0, len(prompts), batch):
        enc = tokenizer(prompts[i:i + batch], return_tensors="pt",
                        padding=True, truncation=True).to(device)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
        for row in new:
            ids.append(strip_tail(row.tolist(), stop))
    return ids, texts


def _batch_tensors(tokenizer, prompts, ans_ids, sl, device):
    """Left-padded prompt + right-padded answer span for one slice."""
    pad = tokenizer.pad_token_id
    enc = tokenizer(prompts[sl], return_tensors="pt",
                    padding=True, truncation=True).to(device)
    pids, pmask = enc["input_ids"], enc["attention_mask"]
    chunk = ans_ids[sl]
    lens = [len(a) for a in chunk]
    lmax = max(lens)
    a_ids = torch.full((len(chunk), lmax), pad, dtype=pids.dtype,
                       device=device)
    a_msk = torch.zeros((len(chunk), lmax), dtype=pmask.dtype, device=device)
    for r, a in enumerate(chunk):
        if a:
            a_ids[r, :len(a)] = torch.tensor(a, device=device)
            a_msk[r, :len(a)] = 1
    full = torch.cat([pids, a_ids], dim=1)
    attn = torch.cat([pmask, a_msk], dim=1)
    return full, attn, int(pids.shape[1]), lmax, lens


def _span_logp(model, full, attn, p, lmax):
    out = model(input_ids=full, attention_mask=attn,
                position_ids=_position_ids(attn)).logits
    # position p-1+t predicts answer token t
    return torch.log_softmax(out[:, p - 1:p - 1 + lmax, :].float(), dim=-1)


@torch.no_grad()
def span_logprobs(model, tokenizer, prompts, ans_ids, device, batch):
    """Teacher-forced log-probs over each agent's answer span.

    Yields (slice, logp[B, Lmax, V], lengths) batch by batch instead of
    materialising a [N, Lmax, V] tensor: at Qwen's 152k vocab that would
    be ~1.3 GB per position per model.
    """
    for i in range(0, len(prompts), batch):
        sl = slice(i, min(i + batch, len(prompts)))
        full, attn, p, lmax, lens = _batch_tensors(
            tokenizer, prompts, ans_ids, sl, device)
        yield sl, _span_logp(model, full, attn, p, lmax), lens


@torch.no_grad()
def dual_span_logprobs(peft_model, tokenizer, prompts, ans_ids, device,
                       batch):
    """Adapter and base log-probs from the SAME batch tensors.

    The base pass is the adapter pass with peft's adapters switched off,
    on the identical input_ids / attention_mask / position_ids. Scoring
    the base from a separately tokenised batch would let padding width
    differ between the two, which perturbs nothing mathematically but
    everything numerically in bf16 -- and the whole measurement is a
    difference between these two distributions.
    """
    for i in range(0, len(prompts), batch):
        sl = slice(i, min(i + batch, len(prompts)))
        full, attn, p, lmax, lens = _batch_tensors(
            tokenizer, prompts, ans_ids, sl, device)
        logp_a = _span_logp(peft_model, full, attn, p, lmax)
        with peft_model.disable_adapter():
            logp_b = _span_logp(peft_model, full, attn, p, lmax)
        yield sl, logp_a, logp_b, lens


def base_stage(model, tokenizer, prompts, device, args, parse_fn):
    """Everything defined by the BASE model: answers, support, value maps,
    t*, margins, soft base values."""
    t0 = time.time()
    ans_ids, ans_txt = generate_answers(model, tokenizer, prompts, device)
    served = np.asarray([parse_fn(s) for s in ans_txt], dtype=np.float32)
    print(f"[akl] base generation: {len(prompts)} agents in "
          f"{time.time() - t0:.1f}s, {len(set(np.round(served, 6)))} "
          f"distinct served values", flush=True)

    vocab = list(range(int(len(tokenizer))))
    decoded = []
    for i in range(0, len(vocab), 4096):
        decoded.extend(tokenizer.batch_decode(
            [[t] for t in vocab[i:i + 4096]], skip_special_tokens=False))
    support = numeric_support_ids(decoded)
    sup_set = set(support)
    print(f"[akl] numeric support: {len(support)} tokens of "
          f"{len(vocab)}", flush=True)

    def decode_fn(seqs):
        return tokenizer.batch_decode(seqs, skip_special_tokens=True)

    # THE SUPPORT IS FIXED, NOT PER-POSITION. Folding each position's
    # top-M into it would make the support (and so the cache key) unique
    # per agent and position, turning a few hundred decodes into millions
    # -- and worse, it would give each cell a different reference frame,
    # so soft values would not be comparable across agents or adapters.
    # Instead the support is the numeric token set alone, and whatever
    # mass falls outside it is REPORTED (tail_mass, topm_outside_support)
    # for the checker to gate rather than silently renormalized away.
    #
    # Value maps then depend only on (answer token sequence, position),
    # and the answers repeat heavily -- frozen Qwen serves five distinct
    # values over 723 agents -- so this cache collapses the work to a few
    # dozen distinct maps.
    cache = {}

    def vmap(a, t):
        key = (tuple(a), t)
        if key not in cache:
            cache[key] = value_map_for(a, t, support, decode_fn, parse_fn)
        return cache[key]

    n = len(prompts)
    tstar = np.zeros(n, dtype=np.int64)
    lev_all = [None] * n
    soft_b = np.zeros(n)
    tail_b = np.zeros(n)
    top1 = np.zeros(n)
    marg = np.zeros(n)
    val_at = [None] * n
    topm_out = np.zeros(n)
    mismatched = []

    for sl, logp, lens in span_logprobs(model, tokenizer, prompts, ans_ids,
                                        device, args.tf_batch):
        probs = logp.exp().cpu().numpy()
        idx0 = sl.start
        for r, L in enumerate(lens):
            i = idx0 + r
            a = ans_ids[i]
            if L == 0:
                raise SystemExit(f"[akl] agent {i}: empty answer span")
            levs, vals = [], []
            for t in range(L):
                pr = probs[r, t]
                if int(np.argmax(pr)) != int(a[t]):
                    mismatched.append((i, t))
                v = vmap(a, t)
                levs.append(leverage(pr, support, v))
                vals.append(v)
            k = pick_tstar(levs)
            tstar[i] = k
            lev_all[i] = levs
            val_at[i] = vals[k]
            soft_b[i], tail_b[i] = soft_value(probs[r, k], support, vals[k])
            top1[i], marg[i] = top2_margin(probs[r, k])
            # how much of the base's own top-M sits OUTSIDE the numeric
            # support: a companion to tail mass that says whether the
            # truncation is dropping anything the model actually favours
            topm = np.argpartition(-probs[r, k], SUPPORT_TOP_M)[:SUPPORT_TOP_M]
            topm_out[i] = float(sum(probs[r, k][j] for j in topm
                                    if int(j) not in sup_set))

    if mismatched:
        raise SystemExit(
            f"[akl] TEACHER-FORCING BROKEN: base argmax != generated token "
            f"at {len(mismatched)} (agent, pos) pairs, first "
            f"{mismatched[:5]} -- the teacher-forced pass is not scoring "
            f"the greedy path (check position_ids / padding)")

    return {"ans_ids": ans_ids, "ans_txt": ans_txt, "served": served,
            "support": support, "values_at_tstar": val_at,
            "tstar": tstar, "leverage": lev_all, "soft_base": soft_b,
            "tail_base": tail_b, "base_top1": top1, "base_margin": marg,
            "topm_outside_support": topm_out}


def adapter_stage(peft_model, tokenizer, prompts, device, args, base, tag):
    """One adapter: KL against the base over the answer span plus the soft
    value in the base's fixed reference frame. The base distributions are
    recomputed here under disable_adapter() rather than reloaded, so the
    two models are compared inside one identical forward."""
    n = len(prompts)
    out = {k: np.zeros(n) for k in
           ("kl_fwd_sum", "kl_rev_sum", "kl_fwd_tstar", "kl_rev_tstar",
            "soft_adapter", "tail_adapter", "greedy_tf", "flip_tstar",
            "first_div", "n_tok")}
    out["first_div"][:] = -1.0
    out["soft_base_recheck"] = np.zeros(n)
    ans_ids = base["ans_ids"]

    for sl, logp_a, logp_b, lens in dual_span_logprobs(
            peft_model, tokenizer, prompts, ans_ids, device, args.tf_batch):
        la = logp_a.cpu().numpy()
        lb = logp_b.cpu().numpy()
        i0 = sl.start
        for r, L in enumerate(lens):
            i = i0 + r
            k = int(base["tstar"][i])
            kf = kl_rows(lb[r, :L], la[r, :L])
            kr = kl_rows(la[r, :L], lb[r, :L])
            out["kl_fwd_sum"][i] = kf.sum()
            out["kl_rev_sum"][i] = kr.sum()
            out["kl_fwd_tstar"][i] = kf[k]
            out["kl_rev_tstar"][i] = kr[k]
            out["n_tok"][i] = L
            sup = base["support"]
            val = base["values_at_tstar"][i]
            pa = np.exp(la[r, k])
            out["soft_adapter"][i], out["tail_adapter"][i] = \
                soft_value(pa, sup, val)
            # the base soft value recomputed under disable_adapter(): must
            # match base_stage's, and the checker compares it ACROSS
            # adapters. If an adapter leaked into the "base" pass this
            # drifts, and every KL in that cell is measured against the
            # wrong reference.
            out["soft_base_recheck"][i] = soft_value(
                np.exp(lb[r, k]), sup, val)[0]
            am = int(np.argmax(pa))
            out["flip_tstar"][i] = float(am != int(ans_ids[i][k]))
            # value the adapter WOULD serve if it took the base's prefix
            # and its own argmax at t*: the teacher-forced greedy
            # counterpart of the soft value
            j = int(np.searchsorted(sup, am))
            out["greedy_tf"][i] = (float(val[j])
                                   if j < len(sup) and sup[j] == am
                                   else float("nan"))
            for t in range(L):
                if int(np.argmax(la[r, t])) != int(ans_ids[i][t]):
                    out["first_div"][i] = float(t)
                    break
    out["tag"] = tag
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=REPO / "runs" / "pokec_gated_lm")
    ap.add_argument("--condor-dir", type=Path,
                    default=REPO / "experiments" / "condor")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "runs" / "adapter_kl_probe")
    ap.add_argument("--ml-dir", type=Path,
                    default=REPO / "experiments/data/movielens/ml-100k")
    ap.add_argument("--target", default="Action")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--tf-batch", type=int, default=8)
    ap.add_argument("--limit-agents", type=int, default=0)
    ap.add_argument("--max-adapters", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated tag filter")
    ap.add_argument("--smoke", action="store_true",
                    help="relax the canonical-hash gate to a WARNING; "
                         "only legal with --limit-agents")
    args = ap.parse_args()

    if args.smoke and not args.limit_agents:
        raise SystemExit("[akl] --smoke requires --limit-agents (the hash "
                         "gate is only relaxed for a truncated agent set)")

    from peft import PeftModel                                # noqa: E402
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
    from perfsim.models.hf_causal_lm import HFCausalLMModel    # noqa: E402

    parse_fn = HFCausalLMModel._parse
    run_mod = _load_run_module()
    setup = run_mod.load_movielens_setup(args.ml_dir, args.target)
    if setup["n"] != N_AGENTS:
        raise SystemExit(f"[akl] setup has {setup['n']} agents != {N_AGENTS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("[akl] this probe needs a GPU (H100: the frozen "
                         "vector's hash is architecture-specific)")
    gpu = torch.cuda.get_device_name(0)
    print(f"[akl] {gpu}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16).to(device)
    base_model.config.pad_token_id = tok.pad_token_id
    base_model.eval()

    prompts = build_prompts(setup, tok,
                            args.limit_agents or None)
    print(f"[akl] {len(prompts)} prompts built", flush=True)

    base = base_stage(base_model, tok, prompts, device, args, parse_fn)
    sha = sha_vec(base["served"])
    full = args.limit_agents in (0, N_AGENTS)
    if sha != CANON_SHA:
        msg = (f"[akl] base served sha {sha[:16]}... != canonical "
               f"{CANON_SHA[:16]}... -- this probe is not reproducing the "
               f"archived frozen Qwen serving path")
        if full and not args.smoke:
            raise SystemExit(msg)
        print(f"WARNING {msg} (truncated agent set: expected)", flush=True)

    tags = read_dose_tags(args.condor_dir)
    if args.only:
        keep = {t.strip() for t in args.only.split(",") if t.strip()}
        tags = [t for t in tags if t in keep]
    if args.max_adapters:
        tags = tags[:args.max_adapters]
    paths = []
    for t in tags:
        p = args.runs_root / t / ADAPTER_SUBDIR
        if not p.is_dir():
            raise SystemExit(f"[akl] missing adapter {p}")
        paths.append(p)
    print(f"[akl] {len(tags)} adapters", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in base.items()},
               args.out_dir / "base_probe.pt")

    peft_model = None
    rows = []
    for i, (tag, p) in enumerate(zip(tags, paths)):
        t0 = time.time()
        name = f"a{i}"
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base_model, str(p),
                                                   adapter_name=name)
        else:
            peft_model.load_adapter(str(p), adapter_name=name)
        peft_model.set_adapter(name)
        peft_model.eval()
        res = adapter_stage(peft_model, tok, prompts, device, args, base, tag)
        torch.save(res, args.out_dir / f"adapter_{tag}.pt")
        rows.append({"tag": tag,
                     "kl_fwd_per_tok": float((res["kl_fwd_sum"]
                                              / res["n_tok"]).mean()),
                     "kl_fwd_tstar": float(res["kl_fwd_tstar"].mean()),
                     "kl_rev_tstar": float(res["kl_rev_tstar"].mean()),
                     "flip_rate": float(res["flip_tstar"].mean()),
                     "soft_rmse_to_base": float(np.sqrt(np.mean(
                         (res["soft_adapter"] - base["soft_base"]) ** 2))),
                     "tail_adapter_max": float(res["tail_adapter"].max()),
                     "soft_base_recheck_dev": float(np.abs(
                         res["soft_base_recheck"] - base["soft_base"]).max()),
                     "early_div_frac": float(np.mean(
                         (res["first_div"] >= 0)
                         & (res["first_div"] < base["tstar"])))})
        print(f"[akl] {tag}: KL/tok {rows[-1]['kl_fwd_per_tok']:.4f}  "
              f"KL@t* {rows[-1]['kl_fwd_tstar']:.4f}  flip "
              f"{100 * rows[-1]['flip_rate']:.1f}%  soft->base "
              f"{rows[-1]['soft_rmse_to_base']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    manifest = {
        "base_model": args.base_model, "gpu": gpu,
        "n_agents": len(prompts), "tags": tags,
        "base_served_sha256": sha, "canonical_sha256": CANON_SHA,
        "hash_gate": "enforced" if (full and not args.smoke) else "warned",
        "support_top_m": SUPPORT_TOP_M, "max_new_tokens": MAX_NEW_TOKENS,
        "tf_batch": args.tf_batch,
        "base_top1_mean": float(base["base_top1"].mean()),
        "base_margin_mean": float(base["base_margin"].mean()),
        "base_margin_median": float(np.median(base["base_margin"])),
        "tail_base_max": float(base["tail_base"].max()),
        "support_size": int(len(base["support"])),
        "topm_outside_support_max": float(base["topm_outside_support"].max()),
        "summary": rows,
        "note": ("soft values are expectations over the base-defined "
                 "numeric support, renormalized; tail_* records the mass "
                 "outside it"),
    }
    with open(args.out_dir / "probe_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[akl] wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
