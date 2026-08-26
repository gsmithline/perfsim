#!/usr/bin/env python3
"""GATE for the Figure-4 anchor trade-off wave (pofdf4a_; generator key
fig4_anchor_tradeoff). CPU only, login-node friendly (1 thread, no
transformers import).

THE GRID IS READ FROM THE GENERATOR (experiments/condor/gen_pofd_sweep.py
via importlib, exactly as check_fig3_full_loop does it): F4A_* constants,
f4a_cells(), f4a_source(), f4a_tag(), F4A_SMOKE_CELLS, f4a_ext_requests(),
F4A_ZSPRIOR / F4A_ZSPRIOR_SHA / F4A_ZSPRIOR_WARN_SHA and F4A_GPU_NAME (the
wave's single hardware class) are the single source of truth.

The wave: beta = W_PLAT {0,.25,.5,.75,1} x gamma = INNATE_LAMBDA
{1,.5,.2,0} x es {.05,.2} x {Qwen2.5-7B, Qwen3-8B} = 80 nominal cells,
60 TRAINED + 20 exact algebraic DUPS (beta=1: gamma drops out of the
operator, one trained cell per (model, es); beta=0: the population never
sees the model and equals the matched twin bit-for-bit, trained for
Qwen3-8B only). Corrected operator (AI_GATE_REFERENCE=anchor ->
population_update nested_ai_anchored_then_social_v2), AI gate all_open,
peer gate threshold at es, ONE Deffuant sweep per round, forward-KL
lambda=2, fresh r512 LoRA each round (181 optimizer steps), 30 rounds.

THE THREE NAMES THAT COLLIDE IN THIS REPO, once:
    beta   = W_PLAT (config w_plat)          gamma = INNATE_LAMBDA
    (config innate_lambda)                   lambda = kl_beta (config kl_beta)
The homophily gamma (config gamma_bias) is a DIFFERENT gamma; it is 0.

HARD-FAIL (exit 1) unless ALL of (each item fails by name, nothing warns):
  1. GRID COMPLETENESS -- every trained cell present exactly once; dup
     cells are NOT run dirs (their source must be present; a dup run dir
     is EXTRA); every extension the manifest requests is present at its
     horizon; no foreign pofdf4a_ run dir under the roots.
  2. CONFIG PINS per run -- w_plat, innate_lambda, eps (social), kl_beta=2,
     kl_direction=forward, training_style=sft_kl, kl_ref_adapter="",
     ab_sweeps=1, deffuant_alpha=0.5, n_rounds, seed=0, dataset/ml_target,
     base_model, ai_gate_mode=all_open, peer_gate_mode=threshold,
     ai_gate_reference=anchor, population_update=...anchored..._v2,
     pop_model=ab, icl_k=0, train_cap=723, n_labeled=723, lora_r=512,
     use_lora, fresh_each_round, sft_epochs=1, sft_batch_size=4,
     sft_lr=5e-5, save_raw_gen, serve_eval_mode, do_sample=False,
     parse_mode=strict, train_witness=True, witness_probe_n=64, homophily
     gamma 0, Qwen3 chat_thinking False, git_sha non-empty. ONE git_sha
     across the wave; ONE innate hash; ONE twin hash per (es, gamma) --
     the twin depends on (seed, es, gamma) only, so twin_raw must be
     bit-identical across models and betas (extensions: over the base
     horizon).
  3. ARRAYS -- op_raw/twin_raw/pred_raw shaped [>=n_rounds, 723], finite,
     op/twin in [0,1]; trajectory rows >= n_rounds; RUNTIME GATE EVIDENCE
     every round: contact == 1.0 (AI gate all_open), peer_gate_mode ==
     "threshold", peer_pairs == 723 * ab_sweeps, 0 <= accepted <=
     peer_pairs. Under the threshold peer gate the runner writes
     peer_gate_mode/peer_pairs ONLY with TRAIN_WITNESS=1 (pinned on every
     F4A job); a row lacking them FAILS naming TRAIN_WITNESS.
  4. beta = 0 runs -- op_raw == twin_raw bit-exactly on every round (the
     decoupling that licenses the cross-model dup).
  5. FULL LOOP -- telemetry carries a training record every round (l_init
     present, n_train == 723) and grad_kl_norm0 > 0 every round.
  6. TRAINING WITNESS every round -- witness_steps ==
     witness_steps_requested == 181, witness_n_rows == 723,
     witness_lora_b_norm > 0, witness_lora_ab_norm > 0,
     witness_data_loss_last / witness_kl_last finite, witness_kl_last > 0,
     witness_probe_kl_fwd > 1e-9 (else IDENTICAL TO FROZEN AT THE
     DISTRIBUTION LEVEL), witness_probe_n == 64, witness_probe_sha
     identical across rounds and across every run of the same model. A
     run lacking witness fields FAILS ("finite-lambda run skipped the
     training witness").
  7. ZERO PARSE FAILURES, STRICT -- raw_gen_log.json.gz REQUIRED (the
     parser serves a FINITE 0.5 on failure, so pred_raw can never reveal
     one); exactly rounds 0..n-1 once; parse_fail_frac == 0 every round
     (absent field = failure); 723 parsed per round; every raw string
     STARTS with a well-formed number in [0,1] (mirror of
     HFCausalLMModel._parse_strict; leading-dot ".64" allowed) and the
     logged parsed value equals it; and the logged parsed vector equals
     the served pred_raw[t] (served == written).
  8. GREEDY CARDINALITY -- informational only: distinct pred_raw values
     per round (min/median/max) and the final-round modal share.
  9. ZERO-SHOT PRIORS (F4A_ZSPRIOR) -- both A100-served _a100 artifacts
     present under the roots. NOT subjected to the item-2 pins (a frozen
     base model serving once needs none of them). Pinned only:
     training_style frozen, use_lora False/0, icl_k 0, icl_days 0,
     do_sample False, movielens/Action, the model's base_model, Qwen3
     chat_thinking False, and the item-10 hardware class. Then pred_raw
     constant across its rounds, 723 finite values in [0,1], raw log
     present with parse_fail_frac 0 and every raw string well-formed and
     equal to parsed; sha256 of pred_raw[0].float().numpy().tobytes()
     recorded. The sha is a HARD pin only where F4A_ZSPRIOR_SHA[model] is
     not None (the coordinator pins it after the serve); it is ALWAYS
     compared to the archived H100-served vector
     F4A_ZSPRIOR_WARN_SHA[model] as a WARN, never a FAIL (an A100 serve
     of the same checkpoint is expected to differ in a handful of
     agents), and both shas are printed.
 10. ONE HARDWARE CLASS -- every run dir of the wave (all cells,
     extensions and both priors) records config.hardware.gpu_name ==
     F4A_GPU_NAME ("NVIDIA A100-SXM4-80GB": the H100 pool left the cluster
     on 2026-08-26 and the wave was retargeted whole), and exactly ONE
     gpu_name is seen across the wave. A mismatch fails naming the class.
--smoke gates the two 3-round pofdf4asmk_ cells (witness_steps_requested
is still 181 per round) plus BOTH zero-shot priors (4 artifacts).

NOT GATED HERE: the lambda = inf (frozen) offline replays. The beta = 0
replay is NOT bit-equal to the runner twin (CPU vs CUDA pair stream), so
no gate compares replay arrays to run arrays; only the analyzer compares
them distributionally.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
GEN = REPO / "experiments" / "condor" / "gen_pofd_sweep.py"
DEFAULT_RUN_ROOTS = (
    Path("/home/gsmithline/perfsim/runs/pokec_gated_lm"),
    REPO / "runs" / "pokec_gated_lm",
    REPO / "notes" / "pofd" / "cluster",
)
LOG = "[check_f4a]"
N_AGENTS = 723
TOL = 1e-9
SERVED_TOL = 1e-6
WITNESS_STEPS = 181
PROBE_N = 64
PROBE_KL_MIN = 1e-9
WANT_OP = "nested_ai_anchored_then_social_v2"
PROD_PREFIX = "pofdf4a_"
SMOKE_PREFIX = "pofdf4asmk_"
WITNESS_FIELDS = (
    "witness_steps", "witness_steps_requested", "witness_n_rows",
    "witness_lora_b_norm", "witness_lora_ab_norm", "witness_data_loss_last",
    "witness_kl_last", "witness_probe_kl_fwd", "witness_probe_kl_rev",
    "witness_probe_argmax_agree", "witness_probe_n", "witness_probe_sha",
)
IDENTICAL_MSG = "IDENTICAL TO FROZEN AT THE DISTRIBUTION LEVEL"
SKIPPED_MSG = "finite-lambda run skipped the training witness"

# Mirror of HFCausalLMModel._parse_strict (perfsim/models/hf_causal_lm.py);
# duplicated so the login-node gate never imports transformers/peft.
_STRICT_RE = re.compile(r"^\s*(\d*\.\d+|\d+(?:\.\d*)?)")


# ------------------------------------------------------------ helpers
def _load_gen(path=None):
    p = Path(path) if path else GEN
    spec = importlib.util.spec_from_file_location("_gen_f4a", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gen_f4a"] = mod
    spec.loader.exec_module(mod)
    return mod


def _num(v):
    return f"{v:g}".replace(".", "p")


def _resolve(tag, roots):
    """The run dir holding <tag>/trajectory.pt, or None."""
    for r in roots:
        p = Path(r) / tag
        if (p / "trajectory.pt").exists():
            return p
    return None


def _dirs_named(tag, roots):
    return [str(Path(r) / tag) for r in roots if (Path(r) / tag).is_dir()]


def _sha_t(t):
    a = t.detach().cpu().contiguous().float().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def _finite(t):
    return bool(torch.isfinite(torch.as_tensor(t)).all())


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fin(v):
    return _isnum(v) and math.isfinite(float(v))


def _read_jsonl(path, gz=False):
    rows = []
    opener = (lambda p: gzip.open(p, "rt")) if gz else (lambda p: open(p))
    with opener(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _by_round(rows):
    """Merge JSONL rows by round (the witness fields ride the training
    row of the same round; merging keeps a split writer from hiding
    them)."""
    out = {}
    for r in rows:
        t = r.get("round")
        if t is None:
            continue
        try:
            t = int(t)
        except (TypeError, ValueError):
            continue
        out.setdefault(t, {}).update(r)
    return out


def strict_parse(text):
    """(value, ok): well-formed number in [0,1] at the start of the
    string, else (None, False)."""
    m = _STRICT_RE.match(text or "")
    if m is None:
        return None, False
    try:
        v = float(m.group(1))
    except ValueError:
        return None, False
    if not 0.0 <= v <= 1.0:
        return None, False
    return v, True


def _eq(cfg, key, want, errs):
    got = cfg.get(key, "<absent>")
    if isinstance(want, float):
        ok = _isnum(got) and abs(float(got) - want) <= TOL
    elif isinstance(want, bool):
        ok = isinstance(got, bool) and got == want
    elif isinstance(want, int):
        ok = _isnum(got) and not isinstance(got, bool) and int(got) == want \
            and float(got) == want
    else:
        ok = got == want
    if not ok:
        errs.append(f"CONFIG {key}={got!r} (want {want!r})")


# --------------------------------------------------- 10. hardware class
def check_hardware(cfg, g, errs):
    """ONE hardware class: config.hardware.gpu_name (written by the
    runner's _hardware_meta) must equal F4A_GPU_NAME on every artifact of
    the wave -- cells, extensions and both priors. Returns the recorded
    gpu_name (None when absent) so main() can prove ONE class wave-wide."""
    hw = cfg.get("hardware")
    gpu = hw.get("gpu_name") if isinstance(hw, dict) else None
    want = g.F4A_GPU_NAME
    if gpu != want:
        errs.append(f"HARDWARE config.hardware.gpu_name={gpu!r} (want "
                    f"{want!r}: the wave's single hardware class -- every "
                    f"F4A artifact must run on {want})")
    return gpu


# ------------------------------------------------------------ 2. config
def check_config(cfg, model, es, beta, gamma, rounds, tag, g, errs):
    expected = {
        "w_plat": float(beta),
        "innate_lambda": float(gamma),
        "eps": float(es),
        "kl_beta": float(g.F4A_LAMBDA),
        "kl_direction": "forward",
        "training_style": "sft_kl",
        "kl_ref_adapter": "",
        "ab_sweeps": int(g.F4A_SWEEPS),
        "deffuant_alpha": float(g.F4A_ALPHA),
        "n_rounds": int(rounds),
        "seed": int(g.F4A_SEED),
        "dataset": "movielens",
        "ml_target": "Action",
        "base_model": g.FAM_MODELS[model]["base_model"],
        "ai_gate_mode": "all_open",
        "peer_gate_mode": "threshold",
        "ai_gate_reference": "anchor",
        "population_update": WANT_OP,
        "pop_model": "ab",
        "icl_k": 0,
        "train_cap": N_AGENTS,
        "n_labeled": N_AGENTS,
        "lora_r": 512,
        "use_lora": True,
        "fresh_each_round": True,
        "sft_epochs": 1,
        "sft_batch_size": 4,
        "sft_lr": 5e-5,
        "save_raw_gen": True,
        "serve_eval_mode": True,
        "do_sample": False,
        "parse_mode": "strict",
        "train_witness": True,
        "witness_probe_n": PROBE_N,
    }
    for key, want in expected.items():
        _eq(cfg, key, want, errs)
    if cfg.get("run_tag") not in (None, tag):
        errs.append(f"CONFIG run_tag={cfg.get('run_tag')!r} != dir tag {tag!r}")
    sha = cfg.get("git_sha")
    if not isinstance(sha, str) or not sha.strip():
        errs.append("CONFIG git_sha absent or empty")
    hg = cfg.get("homophily_gamma", cfg.get("gamma_bias", "<absent>"))
    if not (_isnum(hg) and float(hg) == 0.0):
        errs.append(f"CONFIG homophily gamma={hg!r} (want 0.0)")
    if model == "qwen3_8b" and cfg.get("chat_thinking") is not False:
        errs.append(f"CONFIG Qwen3 chat_thinking={cfg.get('chat_thinking')!r} "
                    "(want False)")


# ------------------------------------------------------------ 3. arrays
def check_arrays(d, rounds, errs):
    for key in ("op_raw", "twin_raw", "pred_raw"):
        if key not in d:
            errs.append(f"ARTIFACT {key} absent")
            continue
        a = torch.as_tensor(d[key]).float()
        if a.ndim != 2 or a.shape[1] != N_AGENTS:
            errs.append(f"ARTIFACT {key} shape {tuple(a.shape)} "
                        f"(want [>={rounds}, {N_AGENTS}])")
            continue
        if a.shape[0] < rounds:
            errs.append(f"ARTIFACT {key} has {a.shape[0]} rounds "
                        f"(want >= {rounds})")
        if not _finite(a):
            errs.append(f"ARTIFACT {key} holds non-finite values")
        if key != "pred_raw" and a.numel():
            lo, hi = float(a.min()), float(a.max())
            if lo < -TOL or hi > 1.0 + TOL:
                errs.append(f"ARTIFACT {key} out of [0,1]: [{lo:.4g},{hi:.4g}]")
    if "innate" not in d:
        errs.append("ARTIFACT innate absent")
    else:
        inn = torch.as_tensor(d["innate"]).float()
        if inn.ndim != 1 or inn.shape[0] != N_AGENTS or not _finite(inn):
            errs.append(f"ARTIFACT innate shape {tuple(inn.shape)} / finite "
                        f"(want [{N_AGENTS}])")
    tr = d.get("trajectory", []) or []
    if len(tr) < rounds:
        errs.append(f"ARTIFACT trajectory has {len(tr)} rows (want >= {rounds})")


def check_runtime_gate(d, rounds, errs, notes):
    """Runtime evidence (not just config) that the AI gate was open and the
    peer gate was the threshold gate in EVERY round: the trajectory rows
    carry contact (the AI-gate open fraction, 1.0 under all_open),
    peer_gate_mode, the pair budget peer_pairs == 723 * ab_sweeps and
    the accepted count 0 <= accepted <= peer_pairs."""
    tr = d.get("trajectory", []) or []
    cfg = d.get("config", {}) or {}
    sweeps = cfg.get("ab_sweeps")
    if not isinstance(sweeps, int) or isinstance(sweeps, bool) or sweeps <= 0:
        errs.append(f"GATE-RUNTIME config ab_sweeps={sweeps!r} unusable")
        return
    want_pairs = N_AGENTS * sweeps
    bad, all_accepted, unwitnessed = [], True, []
    for t in range(min(rounds, len(tr))):
        row = tr[t] or {}
        try:
            contact_ok = abs(float(row.get("contact")) - 1.0) <= TOL
        except (TypeError, ValueError):
            contact_ok = False
        if "peer_gate_mode" not in row or "peer_pairs" not in row:
            # under PEER_GATE_MODE=threshold the runner writes these two
            # fields ONLY when TRAIN_WITNESS=1 (all_open always wrote
            # them); every F4A job pins TRAIN_WITNESS=1, so their absence
            # means the run did not carry the witness
            unwitnessed.append(t)
        mode_ok = row.get("peer_gate_mode") == "threshold"
        pairs, acc = row.get("peer_pairs"), row.get("accepted")
        pairs_ok = (isinstance(pairs, int) and not isinstance(pairs, bool)
                    and isinstance(acc, int) and not isinstance(acc, bool)
                    and pairs == want_pairs and 0 <= acc <= pairs)
        if pairs_ok and acc < pairs:
            all_accepted = False
        if not (contact_ok and mode_ok and pairs_ok):
            bad.append((t, row.get("contact"), row.get("peer_gate_mode"),
                        pairs, acc))
    if len(tr) < rounds:
        bad.append(("rows", len(tr), "<", rounds))
    if unwitnessed:
        errs.append(f"GATE-RUNTIME {len(unwitnessed)} round(s) (e.g. "
                    f"{unwitnessed[:4]}) carry no peer_gate_mode / "
                    f"peer_pairs -- under the threshold peer gate the runner "
                    f"writes them only with TRAIN_WITNESS=1, which every "
                    f"F4A job pins; this run did not carry the witness")
    if bad:
        errs.append(f"GATE-RUNTIME every round must record contact == 1.0 "
                    f"(AI gate all_open), peer_gate_mode == 'threshold' and "
                    f"0 <= accepted <= peer_pairs == {want_pairs}; "
                    f"{len(bad)} round(s) fail, e.g. {bad[:3]}")
    elif all_accepted and rounds > 0:
        notes.append("threshold peer gate rejected no pair in any round "
                     "(accepted == peer_pairs throughout)")


# ------------------------------------------------------ 4. beta = 0 twin
def check_beta0_decoupling(d, rounds, errs):
    op = torch.as_tensor(d["op_raw"]).float()[:rounds]
    tw = torch.as_tensor(d["twin_raw"]).float()[:rounds]
    bad = [t for t in range(rounds)
           if not torch.equal(op[t].contiguous(), tw[t].contiguous())]
    if bad:
        errs.append(f"BETA0 op_raw != twin_raw bit-exactly in {len(bad)} "
                    f"round(s), e.g. {bad[:6]} -- the beta=0 decoupling that "
                    f"licenses the cross-model dup does not hold")


# ------------------------------------------------------------ 5. full loop
def check_full_loop(by_round, rounds, errs):
    missing = [t for t in range(rounds)
               if t not in by_round or "l_init" not in by_round[t]]
    if missing:
        errs.append(f"FULL-LOOP telemetry carries a training record (l_init) "
                    f"for only {rounds - len(missing)} of {rounds} round(s); "
                    f"missing {missing[:8]}{'...' if len(missing) > 8 else ''}")
    bad_n = [(t, by_round[t].get("n_train")) for t in range(rounds)
             if t in by_round and not (_isnum(by_round[t].get("n_train"))
                                       and int(by_round[t]["n_train"]) == N_AGENTS)]
    if bad_n:
        errs.append(f"FULL-LOOP n_train must be {N_AGENTS} every round; "
                    f"{len(bad_n)} round(s) fail, e.g. {bad_n[:4]}")
    bad_kl = []
    for t in range(rounds):
        v = by_round.get(t, {}).get("grad_kl_norm0")
        if not (_fin(v) and float(v) > 0.0):
            bad_kl.append((t, v))
    if bad_kl:
        errs.append(f"KL-WITNESS grad_kl_norm0 must be recorded and > 0 in "
                    f"every round 0..{rounds - 1}; {len(bad_kl)} round(s) "
                    f"fail, e.g. {bad_kl[:4]}")


# ------------------------------------------------------ 6. training witness
def check_witness(by_round, rounds, errs):
    """Per-round TRAIN_WITNESS telemetry. Returns a summary dict (or None
    when the witness is absent)."""
    present = [t for t in range(rounds) if t in by_round]
    lacking = []
    for t in range(rounds):
        row = by_round.get(t, {})
        miss = [f for f in WITNESS_FIELDS if f not in row]
        if miss:
            lacking.append((t, miss[:3] + (["..."] if len(miss) > 3 else [])))
    if not present or len(lacking) == rounds:
        errs.append(f"WITNESS {SKIPPED_MSG}: no round carries the witness_* "
                    f"fields (TRAIN_WITNESS=1 is mandatory here)")
        return None
    if lacking:
        errs.append(f"WITNESS {SKIPPED_MSG} in {len(lacking)} round(s), "
                    f"e.g. {lacking[:3]}")
    full = [t for t in range(rounds)
            if t in by_round and all(f in by_round[t] for f in WITNESS_FIELDS)]
    bad_steps, bad_rows, bad_b, bad_ab, bad_loss, bad_kl = [], [], [], [], [], []
    identical, bad_rev, bad_agree, bad_pn, shas = [], [], [], [], {}
    for t in full:
        r = by_round[t]
        s, sr = r["witness_steps"], r["witness_steps_requested"]
        if not (_isnum(s) and _isnum(sr) and int(s) == int(sr) == WITNESS_STEPS):
            bad_steps.append((t, s, sr))
        if not (_isnum(r["witness_n_rows"]) and int(r["witness_n_rows"]) == N_AGENTS):
            bad_rows.append((t, r["witness_n_rows"]))
        if not (_fin(r["witness_lora_b_norm"]) and float(r["witness_lora_b_norm"]) > 0.0):
            bad_b.append((t, r["witness_lora_b_norm"]))
        if not (_fin(r["witness_lora_ab_norm"]) and float(r["witness_lora_ab_norm"]) > 0.0):
            bad_ab.append((t, r["witness_lora_ab_norm"]))
        if not _fin(r["witness_data_loss_last"]):
            bad_loss.append((t, r["witness_data_loss_last"]))
        if not (_fin(r["witness_kl_last"]) and float(r["witness_kl_last"]) > 0.0):
            bad_kl.append((t, r["witness_kl_last"]))
        v = r["witness_probe_kl_fwd"]
        if not (_fin(v) and float(v) > PROBE_KL_MIN):
            identical.append((t, v))
        if not (_fin(r["witness_probe_kl_rev"]) and float(r["witness_probe_kl_rev"]) >= 0.0):
            bad_rev.append((t, r["witness_probe_kl_rev"]))
        a = r["witness_probe_argmax_agree"]
        if not (_fin(a) and 0.0 <= float(a) <= 1.0):
            bad_agree.append((t, a))
        if not (_isnum(r["witness_probe_n"]) and int(r["witness_probe_n"]) == PROBE_N):
            bad_pn.append((t, r["witness_probe_n"]))
        sha = r["witness_probe_sha"]
        if not isinstance(sha, str) or not sha.strip():
            bad_pn.append((t, f"probe_sha={sha!r}"))
        else:
            shas[sha] = shas.get(sha, 0) + 1
    if bad_steps:
        errs.append(f"WITNESS witness_steps == witness_steps_requested == "
                    f"{WITNESS_STEPS} required every round; {len(bad_steps)} "
                    f"round(s) fail, e.g. {bad_steps[:4]}")
    if bad_rows:
        errs.append(f"WITNESS witness_n_rows must be {N_AGENTS}; "
                    f"{len(bad_rows)} round(s) fail, e.g. {bad_rows[:4]}")
    if bad_b:
        errs.append(f"WITNESS witness_lora_b_norm must be > 0 (a fresh "
                    f"adapter that never moved has B == 0); {len(bad_b)} "
                    f"round(s) fail, e.g. {bad_b[:4]}")
    if bad_ab:
        errs.append(f"WITNESS witness_lora_ab_norm must be > 0; {len(bad_ab)} "
                    f"round(s) fail, e.g. {bad_ab[:4]}")
    if bad_loss:
        errs.append(f"WITNESS witness_data_loss_last must be finite; "
                    f"{len(bad_loss)} round(s) fail, e.g. {bad_loss[:4]}")
    if bad_kl:
        errs.append(f"WITNESS witness_kl_last must be finite and > 0; "
                    f"{len(bad_kl)} round(s) fail, e.g. {bad_kl[:4]}")
    if identical:
        errs.append(f"WITNESS witness_probe_kl_fwd <= {PROBE_KL_MIN:g} in "
                    f"{len(identical)} round(s), e.g. {identical[:4]} -- the "
                    f"trained adapter is {IDENTICAL_MSG}")
    if bad_rev:
        errs.append(f"WITNESS witness_probe_kl_rev must be finite and >= 0; "
                    f"{len(bad_rev)} round(s) fail, e.g. {bad_rev[:4]}")
    if bad_agree:
        errs.append(f"WITNESS witness_probe_argmax_agree must lie in [0,1]; "
                    f"{len(bad_agree)} round(s) fail, e.g. {bad_agree[:4]}")
    if bad_pn:
        errs.append(f"WITNESS witness_probe_n must be {PROBE_N} and "
                    f"witness_probe_sha a non-empty string; {len(bad_pn)} "
                    f"round(s) fail, e.g. {bad_pn[:4]}")
    if len(shas) > 1:
        errs.append(f"WITNESS witness_probe_sha differs across rounds "
                    f"({len(shas)} distinct: {sorted(shas)[:3]}) -- the probe "
                    f"set must be the same every round")
    if not full:
        return None

    def _vals(key):
        return [float(by_round[t][key]) for t in full if _fin(by_round[t][key])]
    b, ab, kf, kr, ag, kl = (_vals("witness_lora_b_norm"),
                             _vals("witness_lora_ab_norm"),
                             _vals("witness_probe_kl_fwd"),
                             _vals("witness_probe_kl_rev"),
                             _vals("witness_probe_argmax_agree"),
                             _vals("witness_kl_last"))
    return {
        "rounds_with_witness": len(full),
        "probe_sha": (sorted(shas)[0] if len(shas) == 1 else
                      (sorted(shas) if shas else None)),
        "steps": WITNESS_STEPS if not bad_steps else None,
        "min_lora_b_norm": min(b) if b else None,
        "min_lora_ab_norm": min(ab) if ab else None,
        "min_probe_kl_fwd": min(kf) if kf else None,
        "max_probe_kl_fwd": max(kf) if kf else None,
        "mean_probe_kl_rev": (sum(kr) / len(kr)) if kr else None,
        "mean_probe_argmax_agree": (sum(ag) / len(ag)) if ag else None,
        "min_kl_last": min(kl) if kl else None,
    }


# ------------------------------------------------- 7. zero parse failures
def check_raw_generations(run_dir, rounds, errs, pred=None):
    """STRICT zero-parse-failure gate plus the served == written proof.
    Returns the generation counts (or None when the log is absent)."""
    gz = Path(run_dir) / "raw_gen_log.json.gz"
    if not gz.exists():
        errs.append("PARSE raw_gen_log.json.gz ABSENT -- parse_fail_frac is "
                    "recorded nowhere else and the parser stores a finite "
                    "0.5 on failure, so this cell's parse rate cannot be "
                    "established (SAVE_RAW_GEN=1 is mandatory here)")
        return None
    try:
        rows = _read_jsonl(gz, gz=True)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        errs.append(f"PARSE raw_gen_log.json.gz unreadable: {e}")
        return None
    got = [r.get("round") for r in rows]
    if got != list(range(rounds)):
        absent = [t for t in range(rounds) if t not in got]
        errs.append(f"PARSE raw_gen_log must carry exactly rounds "
                    f"0..{rounds - 1} in order, once; it holds "
                    f"{got[:6]}{'...' if len(got) > 6 else ''} -- "
                    f"{len(absent)} round(s) without a logged parse rate")
    bad = []
    for r in rows:
        v = r.get("parse_fail_frac")
        if not (_isnum(v) and float(v) == 0.0):
            bad.append((r.get("round"), v))
    if bad:
        errs.append(f"PARSE parse_fail_frac must be exactly 0 in every "
                    f"round (an absent field counts as a failure); "
                    f"{len(bad)} bad round(s), e.g. {bad[:4]}")
    short = [(r.get("round"), len(r.get("parsed") or []))
             for r in rows if len(r.get("parsed") or []) != N_AGENTS]
    if short:
        errs.append(f"PARSE {len(short)} round(s) parsed fewer than "
                    f"{N_AGENTS} agents, e.g. {short[:4]}")
    malformed, mismatched, unserved, total = [], [], [], 0
    for r in rows:
        raws = r.get("raw") or []
        parsed = r.get("parsed") or []
        t = r.get("round")
        if len(raws) != len(parsed):
            errs.append(f"PARSE round {t} logs {len(raws)} raw strings but "
                        f"{len(parsed)} parsed values")
            continue
        for i, (txt, pv) in enumerate(zip(raws, parsed)):
            total += 1
            v, ok = strict_parse(txt)
            if not ok:
                malformed.append((t, i, str(txt)[:20]))
            elif not _isnum(pv) or abs(float(pv) - v) > SERVED_TOL:
                mismatched.append((t, i, str(txt)[:20], pv))
        # served == written: the logged parsed vector IS the served one
        if (pred is not None and isinstance(t, int) and 0 <= t < pred.shape[0]
                and len(parsed) == N_AGENTS):
            try:
                pv = torch.tensor([float(x) for x in parsed])
                diff = (pv - pred[t]).abs().max().item()
            except (TypeError, ValueError):
                diff = float("inf")
            if not diff <= SERVED_TOL:
                unserved.append((t, diff))
    if malformed:
        errs.append(f"PARSE {len(malformed)}/{total} generation(s) are not "
                    f"a well-formed number in [0,1] at the start of the "
                    f"string, e.g. {malformed[:3]} -- the served value is "
                    f"not what the model wrote")
    if mismatched:
        errs.append(f"PARSE {len(mismatched)}/{total} served value(s) "
                    f"differ from the number the model wrote, e.g. "
                    f"{mismatched[:3]}")
    if unserved:
        errs.append(f"PARSE the logged parsed vector differs from the served "
                    f"pred_raw in {len(unserved)} round(s), e.g. "
                    f"{unserved[:3]} -- served != written")
    return {"generations": total, "malformed": len(malformed),
            "mismatched": len(mismatched), "unserved_rounds": len(unserved)}


# ------------------------------------------------- 8. greedy cardinality
def greedy_cardinality(pred, rounds):
    """Informational: distinct served values per round and the final
    round's modal share. Never a gate."""
    if pred.ndim != 2 or pred.shape[0] < rounds or rounds < 1:
        return None
    distinct = [int(torch.unique(pred[t]).numel()) for t in range(rounds)]
    vals, counts = torch.unique(pred[rounds - 1], return_counts=True)
    j = int(counts.argmax())
    return {"distinct_min": min(distinct),
            "distinct_median": float(statistics.median(distinct)),
            "distinct_max": max(distinct),
            "distinct_final": distinct[-1],
            "final_modal_value": float(vals[j]),
            "final_modal_share": float(counts[j]) / float(pred.shape[1])}


# ------------------------------------------------- 9. zero-shot priors
def check_zsprior(model, tag, g, roots):
    """The zero-shot prior artifacts (both served on the wave's hardware
    class under the _a100 tags) are NOT subjected to the wave's config
    pins -- a frozen base model serving once with no LoRA and no context
    needs none of them. Pinned: frozen, no LoRA, icl_k 0, icl_days 0,
    greedy, MovieLens/Action, the right checkpoint, Qwen3 thinking off,
    and config.hardware.gpu_name == F4A_GPU_NAME. Then: pred_raw constant
    across its rounds, 723 finite values in [0,1], raw log well-formed
    and equal to the served vector, sha256(pred_raw[0]) recorded --
    HARD-pinned only where F4A_ZSPRIOR_SHA[model] is not None (the
    coordinator pins it after the serve), and ALWAYS compared to the
    archived H100-served vector F4A_ZSPRIOR_WARN_SHA[model] as a WARN,
    never a FAIL (an A100 serve is expected to differ in a handful of
    agents); both shas are printed."""
    errs, warns = [], []
    rec = {"model": model, "tag": tag, "path": None, "status": "ABSENT",
           "errors": errs, "warnings": warns, "sha256_pred0": None,
           "expected_sha256": g.F4A_ZSPRIOR_SHA.get(model),
           "archived_sha256": getattr(g, "F4A_ZSPRIOR_WARN_SHA", {}).get(model),
           "sha_pin": None, "gpu_name": None,
           "n_rounds": None, "distinct": None, "mean": None}
    path = _resolve(tag, roots)
    if path is None:
        errs.append(f"ZSPRIOR {tag}/trajectory.pt absent under the run roots")
        return rec
    rec["path"] = str(path)
    d = torch.load(path / "trajectory.pt", map_location="cpu", weights_only=False)
    cfg = d.get("config", {}) or {}
    rec["gpu_name"] = check_hardware(cfg, g, errs)
    for key, want in (("base_model", g.FAM_MODELS[model]["base_model"]),
                      ("training_style", "frozen"), ("icl_k", 0),
                      ("icl_days", 0), ("dataset", "movielens"),
                      ("ml_target", "Action"), ("do_sample", False)):
        _eq(cfg, key, want, errs)
    if cfg.get("use_lora", "<absent>") not in (False, 0):
        errs.append(f"CONFIG use_lora={cfg.get('use_lora', '<absent>')!r} "
                    f"(want False)")
    if model == "qwen3_8b" and cfg.get("chat_thinking") is not False:
        errs.append(f"CONFIG Qwen3 chat_thinking={cfg.get('chat_thinking')!r} "
                    "(want False)")
    if "pred_raw" not in d:
        errs.append("ZSPRIOR pred_raw absent")
        rec["status"] = "FAIL"
        return rec
    pred = torch.as_tensor(d["pred_raw"]).float()
    if pred.ndim != 2 or pred.shape[0] < 1 or pred.shape[1] != N_AGENTS:
        errs.append(f"ZSPRIOR pred_raw shape {tuple(pred.shape)} "
                    f"(want [>=1, {N_AGENTS}])")
        rec["status"] = "FAIL"
        return rec
    n = int(pred.shape[0])
    rec["n_rounds"] = n
    if not _finite(pred):
        errs.append("ZSPRIOR pred_raw holds non-finite values")
    lo, hi = float(pred.min()), float(pred.max())
    if lo < -TOL or hi > 1.0 + TOL:
        errs.append(f"ZSPRIOR pred_raw out of [0,1]: [{lo:.4g},{hi:.4g}]")
    shas = {_sha_t(pred[t]) for t in range(n)}
    if len(shas) != 1:
        errs.append(f"ZSPRIOR pred_raw is not constant across its {n} "
                    f"round(s) ({len(shas)} distinct served vectors) -- a "
                    f"frozen zero-shot serve cannot change")
    sha0 = _sha_t(pred[0])
    rec["sha256_pred0"] = sha0
    want = g.F4A_ZSPRIOR_SHA.get(model)
    if want is None:
        rec["sha_pin"] = "unpinned"            # HARD pin skipped by design
    elif sha0 != want:
        rec["sha_pin"] = "MISMATCH"
        errs.append(f"ZSPRIOR sha256(pred_raw[0]) = {sha0} != the pinned "
                    f"{want} (F4A_ZSPRIOR_SHA[{model}]: the vector the "
                    f"frozen replays were built from)")
    else:
        rec["sha_pin"] = "match"
    archived = rec["archived_sha256"]
    if archived is not None and sha0 != archived:
        warns.append(f"ZSPRIOR {model} sha256(pred_raw[0]) = {sha0} differs "
                     f"from the archived H100-served prior {archived} "
                     f"(F4A_ZSPRIOR_WARN_SHA) -- a serve of the same "
                     f"checkpoint on {g.F4A_GPU_NAME} is expected to differ "
                     f"in a handful of agents; review, never a failure")
    rec["distinct"] = int(torch.unique(pred[0]).numel())
    rec["mean"] = float(pred[0].mean())
    if "innate" in d:
        inn = torch.as_tensor(d["innate"]).float()
        rec["innate_sha"] = _sha_t(inn) if inn.numel() == N_AGENTS else None
    check_raw_generations(path, n, errs, pred)
    rec["status"] = "PASS" if not errs else "FAIL"
    return rec


# -------------------------------------------------------------- per cell
def check_cell(model, es, beta, gamma, rounds, tag, g, roots, base_rounds):
    """Gate one trained (or extension) run dir. `base_rounds` is the
    horizon the twin identity is compared over (the base horizon for
    extensions)."""
    errs, notes = [], []
    rec = {"model": model, "es": es, "beta": beta, "gamma": gamma,
           "kind": "gpu", "tag": tag, "source_tag": tag, "rounds": rounds,
           "path": None, "status": "ABSENT", "errors": errs, "notes": notes,
           "git_sha": None, "innate_sha": None, "twin_sha": None,
           "op_sha": None, "pop_final_mean": None, "pop_final_sd": None,
           "twin_final_mean": None, "innate_mean": None,
           "greedy": None, "witness": None, "generations": None,
           "accepted_mean": None, "gpu_name": None}
    path = _resolve(tag, roots)
    if path is None:
        errs.append("trajectory.pt absent under the run roots")
        return rec
    rec["path"] = str(path)
    d = torch.load(path / "trajectory.pt", map_location="cpu", weights_only=False)
    cfg = d.get("config", {}) or {}
    rec["git_sha"] = cfg.get("git_sha")
    rec["gpu_name"] = check_hardware(cfg, g, errs)
    check_config(cfg, model, es, beta, gamma, rounds, tag, g, errs)
    check_arrays(d, rounds, errs)
    if any(e.startswith("ARTIFACT") for e in errs):
        rec["status"] = "FAIL"
        return rec
    op = torch.as_tensor(d["op_raw"]).float()
    tw = torch.as_tensor(d["twin_raw"]).float()
    pred = torch.as_tensor(d["pred_raw"]).float()
    inn = torch.as_tensor(d["innate"]).float()
    check_runtime_gate(d, rounds, errs, notes)
    if beta == 0.0:
        check_beta0_decoupling(d, rounds, errs)
    tel = path / "telemetry.json"
    if not tel.exists():
        errs.append("FULL-LOOP telemetry.json ABSENT -- no recorded evidence "
                    "that this cell ever trained")
        errs.append(f"WITNESS {SKIPPED_MSG}: telemetry.json absent")
    else:
        try:
            by_round = _by_round(_read_jsonl(tel))
        except (OSError, ValueError, json.JSONDecodeError) as e:
            by_round = {}
            errs.append(f"FULL-LOOP telemetry.json unreadable: {e}")
        check_full_loop(by_round, rounds, errs)
        rec["witness"] = check_witness(by_round, rounds, errs)
    rec["generations"] = check_raw_generations(path, rounds, errs, pred)
    rec["greedy"] = greedy_cardinality(pred, rounds)
    tr = d.get("trajectory", []) or []
    acc = [r.get("accepted") for r in tr[:rounds] if _isnum((r or {}).get("accepted"))]
    rec["accepted_mean"] = (sum(acc) / len(acc)) if acc else None
    rec["innate_sha"] = _sha_t(inn)
    rec["innate_mean"] = float(inn.mean())
    k = min(rounds, base_rounds)
    rec["twin_sha"] = _sha_t(tw[:k])
    rec["op_sha"] = _sha_t(op[:rounds])
    rec["pop_final_mean"] = float(op[rounds - 1].mean())
    rec["pop_final_sd"] = float(op[rounds - 1].std())
    rec["twin_final_mean"] = float(tw[rounds - 1].mean())
    if pred.shape[0] >= 2 and len({_sha_t(pred[t]) for t in range(rounds)}) == 1:
        notes.append("served map constant across rounds (legitimate for a "
                     "settled loop; the witness above is the evidence)")
    rec["status"] = "PASS" if not errs else "FAIL"
    return rec


def _wave_fail(tag, msg, **extra):
    r = {"model": "-", "es": None, "beta": None, "gamma": None,
         "kind": "wave", "tag": tag, "source_tag": None, "rounds": None,
         "path": None, "status": "FAIL", "errors": [msg], "notes": []}
    r.update(extra)
    return r


def _fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="gate the Figure-4 anchor trade-off wave (pofdf4a_); CPU only")
    ap.add_argument("--run-root", action="append", default=None,
                    help="run-dir root (repeatable; default: the cluster "
                         "path, runs/pokec_gated_lm, notes/pofd/cluster)")
    ap.add_argument("--smoke", action="store_true",
                    help="gate the two 3-round pofdf4asmk_ cells + BOTH "
                         "zero-shot priors (4 artifacts)")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the machine-readable verdict here")
    ap.add_argument("--gen", default=None,
                    help="path of gen_pofd_sweep.py (default: this checkout)")
    ap.add_argument("--ext-manifest", default=None,
                    help="override the committed "
                         "fig4_anchor_extension_request.json")
    args = ap.parse_args(argv)

    roots = [Path(r) for r in (args.run_root or DEFAULT_RUN_ROOTS)]
    for r in roots:
        if args.run_root and not r.is_dir():
            print(f"{LOG} usage error: --run-root {r} is not a directory",
                  file=sys.stderr)
            return 2
    g = _load_gen(args.gen)
    base_rounds = int(g.F4A_ROUNDS)

    # ---- what we EXPECT, read from the generator ------------------------
    if args.smoke:
        rounds = int(g.F4A_SMOKE_ROUNDS)
        cells = [(m, e, b, gm, rounds, g.f4a_tag(m, e, b, gm, rounds=rounds,
                                                  smoke=True))
                 for (m, e, b, gm) in g.F4A_SMOKE_CELLS]
        dups, ext, zs_models = [], [], sorted(g.F4A_ZSPRIOR)
        base_rounds = rounds
    else:
        rounds = base_rounds
        cells, dups = [], []
        for (m, e, b, gm, kind, src) in g.f4a_cells():
            if kind == "gpu":
                tag = g.F4A_REUSED.get((m, e, b, gm)) or g.f4a_tag(m, e, b, gm)
                cells.append((m, e, b, gm, rounds, tag))
            else:
                dups.append((m, e, b, gm, src))
        ext = [(m, e, b, gm, r, g.f4a_tag(m, e, b, gm, rounds=r))
               for (m, e, b, gm, r) in sorted(g.f4a_ext_requests(args.ext_manifest))]
        zs_models = sorted(g.F4A_ZSPRIOR)

    records = [check_cell(m, e, b, gm, r, tag, g, roots, base_rounds)
               for (m, e, b, gm, r, tag) in cells]
    for rec, (m, e, b, gm, _r, _tag) in zip(records, cells):
        rec["reused"] = (m, e, b, gm) in g.F4A_REUSED
    ext_records = [check_cell(m, e, b, gm, r, tag, g, roots, base_rounds)
                   for (m, e, b, gm, r, tag) in ext]
    for rec in ext_records:
        rec["kind"] = "ext"
    zs_records = {m: check_zsprior(m, g.F4A_ZSPRIOR[m], g, roots) for m in zs_models}
    wave = []

    # ---- 1. grid completeness / EXTRA ----------------------------------
    by_tag = {r["tag"]: r for r in records + ext_records}
    expected_tags = set(by_tag)
    if not args.smoke:
        for (m, e, b, gm, src) in dups:
            dtag = g.f4a_tag(m, e, b, gm)
            where = _dirs_named(dtag, roots)
            if where:
                wave.append(_wave_fail(dtag, f"EXTRA {dtag} is an algebraic dup "
                                       f"of {g.f4a_tag(*src)} (no run is expected "
                                       f"there) but exists as a run dir: {where[0]}"))
        seen = set()
        for root in roots:
            if not root.is_dir():
                continue
            for p in sorted(root.iterdir()):
                n = p.name
                if not p.is_dir() or not n.startswith(PROD_PREFIX) or n in seen:
                    continue
                seen.add(n)
                if n not in expected_tags:
                    why = ("an extension the manifest does not request"
                           if not n.endswith(f"_r{base_rounds}") else
                           "NOT in the expected grid")
                    wave.append(_wave_fail(n, f"EXTRA run dir {n} under {root} is "
                                           f"{why}"))
    trained = [r for r in records if r["status"] != "ABSENT"]

    # ---- 2. one provenance, one population, one twin per (es, gamma) ----
    if trained:
        # (an absent git_sha / innate is already a per-run CONFIG or
        # ARTIFACT failure; here only DISAGREEMENT fails the wave)
        shas = sorted({r["git_sha"] for r in trained if r.get("git_sha")})
        if len(shas) > 1:
            wave.append(_wave_fail("<provenance>",
                                   f"{len(shas)} distinct git SHAs across the "
                                   f"{len(trained)} trained run(s) ({shas[:4]}); "
                                   f"the wave must share ONE provenance"))
        inn = sorted({r["innate_sha"] for r in trained if r.get("innate_sha")})
        if len(inn) > 1:
            wave.append(_wave_fail("<innate>",
                                   f"{len(inn)} distinct innate hashes; every "
                                   f"cell must share one population"))
        groups = {}
        for r in trained + [x for x in ext_records if x["status"] != "ABSENT"]:
            if r.get("twin_sha"):
                groups.setdefault((r["es"], r["gamma"]), {}).setdefault(
                    r["twin_sha"], []).append(r["tag"])
        for (e, gm), shs in sorted(groups.items()):
            if len(shs) != 1:
                ex = {k[:12]: v[:2] for k, v in shs.items()}
                wave.append(_wave_fail(f"<twin es={e:g} gamma={gm:g}>",
                                       f"{len(shs)} distinct twin_raw hashes at "
                                       f"(es={e:g}, gamma={gm:g}) over the base "
                                       f"horizon; the twin depends on (seed, es, "
                                       f"gamma) only and must be bit-identical "
                                       f"across models and betas: {ex}"))
        probes = {}
        for r in trained + ext_records:
            w = r.get("witness") or {}
            if isinstance(w.get("probe_sha"), str):
                probes.setdefault(r["model"], {}).setdefault(
                    w["probe_sha"], []).append(r["tag"])
        for m, shs in sorted(probes.items()):
            if len(shs) != 1:
                ex = {k[:12]: v[:2] for k, v in shs.items()}
                wave.append(_wave_fail(f"<probe {m}>",
                                       f"witness_probe_sha differs across the "
                                       f"{m} runs ({len(shs)} distinct: {ex}); "
                                       f"the probe set must be shared by every "
                                       f"cell of a model"))

    # ---- 10. ONE hardware class across the whole wave --------------------
    # (a wrong or absent gpu_name is already a per-artifact HARDWARE
    # failure; here DISAGREEMENT between artifacts fails the wave)
    gpus = {}
    for r in (trained + [x for x in ext_records if x["status"] != "ABSENT"]
              + [z for z in zs_records.values() if z["status"] != "ABSENT"]):
        if r.get("gpu_name") is not None:
            gpus.setdefault(r["gpu_name"], []).append(r["tag"])
    if len(gpus) > 1:
        ex = {k: v[:2] for k, v in sorted(gpus.items())}
        wave.append(_wave_fail("<hardware>",
                               f"{len(gpus)} distinct gpu_name values across "
                               f"the wave ({sorted(gpus)}); every artifact "
                               f"must run on ONE hardware class, "
                               f"{g.F4A_GPU_NAME}: {ex}"))

    # ---- 80-cell view: dups resolved through their source ---------------
    cell_rows = list(records)
    for (m, e, b, gm, src) in dups:
        s = by_tag.get(g.F4A_REUSED.get(src) or g.f4a_tag(*src))
        cell_rows.append({
            "model": m, "es": e, "beta": b, "gamma": gm, "kind": "dup",
            "tag": g.f4a_tag(m, e, b, gm), "source_tag": s["tag"] if s else None,
            "rounds": rounds, "path": s["path"] if s else None,
            "status": (s["status"] if s else "ABSENT"),
            "errors": ([] if s and s["status"] == "PASS" else
                       [f"source {g.f4a_tag(*src)} is "
                        f"{s['status'] if s else 'ABSENT'}"]),
            "notes": [],
            "pop_final_mean": s.get("pop_final_mean") if s else None,
            "pop_final_sd": s.get("pop_final_sd") if s else None,
            "twin_final_mean": s.get("twin_final_mean") if s else None,
            "greedy": s.get("greedy") if s else None,
            "witness": s.get("witness") if s else None})
    cell_rows.sort(key=lambda r: (r["model"], r["es"], r["beta"], -r["gamma"]))

    ok = (all(r["status"] == "PASS" for r in records)
          and all(r["status"] == "PASS" for r in ext_records)
          and all(r["status"] == "PASS" for r in zs_records.values())
          and not wave)

    # ---- table ---------------------------------------------------------
    hdr = (f"{'cell':<38}{'verdict':<8}{'rounds':>6}{'popMean':>9}"
           f"{'popSD':>8}{'twinMean':>9}{'distinctServed':>16}{'probeKL':>10}")
    print("=" * len(hdr))
    print(f"FIGURE-4 ANCHOR TRADE-OFF GATE -- {'SMOKE' if args.smoke else 'PRODUCTION'}"
          f": {len(records)} trained + {len(dups)} dup + {len(ext_records)} ext "
          f"+ {len(zs_records)} zero-shot prior(s); hardware class "
          f"{g.F4A_GPU_NAME}")
    print("=" * len(hdr))
    print(hdr)
    for r in cell_rows + ext_records:
        cell = (f"{r['model']:<9} es{_num(r['es'])} w{_num(r['beta'])} "
                f"k{_num(r['gamma'])} {r['kind']}")
        gr = r.get("greedy") or {}
        ds = ("-" if not gr else
              f"{gr['distinct_min']}/{gr['distinct_median']:g}/{gr['distinct_max']}")
        w = r.get("witness") or {}
        pk = w.get("min_probe_kl_fwd")
        print(f"{cell:<38}{r['status']:<8}{str(r.get('rounds') or '-'):>6}"
              f"{_fmt(r.get('pop_final_mean')):>9}{_fmt(r.get('pop_final_sd')):>8}"
              f"{_fmt(r.get('twin_final_mean')):>9}{ds:>16}"
              f"{('-' if pk is None else f'{pk:.2e}'):>10}"
              + (f"  {r['tag']}" if r["kind"] != "dup" else f"  -> {r['source_tag']}"))
        for e in r.get("errors", []):
            print(f"    !! {e}")
        for n in r.get("notes", []):
            print(f"    NOTE {n}")
    for m, z in sorted(zs_records.items()):
        sha = z.get("sha256_pred0")
        print(f"ZSPRIOR {m:<9} {z['status']:<8} {z['tag']}  sha256(pred_raw[0])="
              f"{sha if sha else '-'}  distinct={z.get('distinct')}  "
              f"mean={_fmt(z.get('mean'))}  gpu={z.get('gpu_name') or '-'}")
        print(f"    sha pin={z.get('sha_pin') or '-'} "
              f"(F4A_ZSPRIOR_SHA={z.get('expected_sha256') or 'None'})  "
              f"archived_h100={z.get('archived_sha256') or '-'}")
        for e in z["errors"]:
            print(f"    !! {e}")
        for w in z.get("warnings") or []:
            print(f"    WARN {w}")
    for r in wave:
        print(f"FAIL {r['tag']}: {r['errors'][0]}")
    git_shas = sorted({r["git_sha"] for r in trained + ext_records if r.get("git_sha")})
    innate_shas = sorted({r["innate_sha"] for r in trained if r.get("innate_sha")})
    n_fail = sum(1 for r in records + ext_records if r["status"] != "PASS")
    zs_bits = ", ".join(f"{m}: {z.get('sha256_pred0') or 'ABSENT'}"
                        for m, z in sorted(zs_records.items()))
    n_warn = sum(len(z.get("warnings") or []) for z in zs_records.values())
    print(f"PROVENANCE {len(trained)}/{len(records)} trained run(s) present "
          f"@ git {git_shas}; innate {[s[:12] for s in innate_shas]}; "
          f"hardware {sorted(gpus)} (want [{g.F4A_GPU_NAME!r}]); "
          f"zero-shot priors {{{zs_bits}}}; "
          f"{len(ext_records)} extension(s); {n_fail} failing run(s); "
          f"{len(wave)} wave-level failure(s); {n_warn} WARN line(s) "
          f"(warnings never change the exit code)")

    verdict = {
        "ok": bool(ok),
        "wave": g.F4A_KEY,
        "smoke": bool(args.smoke),
        "generator": str(args.gen or GEN),
        "run_roots": [str(r) for r in roots],
        "n_cells": len(records) + len(dups),
        "n_trained": len(records),
        "n_dup": len(dups),
        "n_ext": len(ext_records),
        "n_failing": n_fail + len(wave)
                     + sum(1 for z in zs_records.values() if z["status"] != "PASS"),
        "cells": cell_rows,
        "extensions": ext_records,
        "zsprior": zs_records,
        "wave_failures": wave,
        "warnings": [w for z in zs_records.values() for w in (z.get("warnings") or [])],
        "git_sha": git_shas,
        "innate_sha": innate_shas,
        "hardware_class": g.F4A_GPU_NAME,
        "gpu_names": sorted(gpus),
        "operator_required": WANT_OP,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(verdict, indent=2, default=str))
        print(f"{LOG} verdict -> {out}")
    if ok:
        print(f"{LOG} PASS -- every trained cell matches the pinned surface "
              f"({WANT_OP}, all_open AI gate, threshold peer gate, one sweep, "
              f"forward-KL lambda=2), retrained every round with a live KL "
              f"anchor and a full training witness (181 steps, LoRA moved, "
              f"probe KL > 0), served well-formed numbers with zero parse "
              f"failures, beta=0 populations equal their twins bit-exactly, "
              f"twins agree across models and betas, and the wave shares one "
              f"provenance, one population, one hardware class "
              f"({g.F4A_GPU_NAME}) and both zero-shot priors.")
        return 0
    print(f"{LOG} FAIL -- {verdict['n_failing']} failing record(s). See the "
          f"!! / FAIL lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
