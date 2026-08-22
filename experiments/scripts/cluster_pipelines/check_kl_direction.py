#!/usr/bin/env python3
"""GATE for the forward-vs-reverse KL wave (pofdkd_, 2026-08-22).

RUN ON THE CLUSTER before anything is pulled or plotted.

The wave's whole claim is a DIRECTION contrast, so the gate is built
around the ways that contrast could be fake rather than around generic
run health:

  DIRECTION      config.kl_direction must equal the tag's own token. A
                 tag saying "rev" over a run that trained forward is the
                 single failure that would invert the headline, and it
                 is invisible in every downstream artifact. Checked per
                 run; and across a production wave BOTH directions must
                 appear, so a half-submitted wave cannot be analyzed as
                 if it were the comparison.
  TRAINING       the arm must have actually trained: sft_kl style, a
                 positive lambda, LoRA on, fresh adapter each round, and
                 finite training losses that are not all identical. A
                 KL weight so large that the optimizer no-ops would look
                 like "close to frozen" for the wrong reason, so a run
                 whose served vector never moves across rounds is
                 rejected here rather than reported as retention.
  REFERENCE      kl_ref_adapter must be EMPTY and anchor_mode "fixed":
                 the anchor is the raw base checkpoint. If one arm
                 anchored to a trained adapter its lambda would not be
                 comparable to the others'. base_model must also be
                 identical across the wave.
  SERVING        pred_raw must be [rounds, 723] and finite everywhere.
                 A partially served round silently shrinks the
                 population the figure summarises.
  PARSE          parse_fail_frac must be exactly 0 in every round. A
                 parse failure is recorded as a confident constant --
                 the Qwen3 Pokec cell emitted "<think>" for all 2163
                 agents and logged a clean 0.500 -- so a nonzero rate is
                 never a rounding matter.
  NONCONSTANT    predictions must vary ACROSS AGENTS. A collapsed served
                 map can still have a moving mean, and the question this
                 wave asks is explicitly about heterogeneity, so a
                 degenerate map is surfaced as a hard failure rather
                 than left for the reader to notice in the table.

SURFACE PINS. Everything that must be identical across arms for the
comparison to mean anything is pinned too: seed, dataset/target, agent
count, k (INNATE_LAMBDA), both gate MODES, eps, one peer sweep, LoRA
rank, learning rate, epochs, epoch size, train cap, no ICL, the H100
SKU, and the declared horizon read off the tag PREFIX (pofdkd_ = the
full horizon, pofdkdsmk_ = the 3-round smoke).

Usage:
  python check_kl_direction.py runs/pokec_gated_lm/pofdkd_*_r10
  python check_kl_direction.py --smoke runs/pokec_gated_lm/pofdkdsmk_*_r3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

N_AGENTS = 723
H100 = "NVIDIA H100 80GB HBM3"
BASE = "Qwen/Qwen2.5-7B-Instruct"
PROD_ROUNDS = 10
SMOKE_ROUNDS = 3
LAMBDAS = (0.1, 1.0, 10.0)
PROD_PREFIX = "pofdkd_"
SMOKE_PREFIX = "pofdkdsmk_"


def fail(tag, msg, out):
    out.append(f"[check_kd] FAIL {tag}: {msg}")


def check_one(d, smoke, out):
    """Returns (ok, direction, lam, w) -- direction/lam/w are None on a
    run too broken to classify."""
    tag = os.path.basename(d.rstrip("/"))
    cf = os.path.join(d, "config.json")
    tf = os.path.join(d, "trajectory.pt")
    if not os.path.exists(cf):
        fail(tag, "no config.json", out)
        return False, None, None, None
    if not os.path.exists(tf):
        fail(tag, "no trajectory.pt -- run did not finish", out)
        return False, None, None, None
    c = json.load(open(cf))
    ok = True

    # ---- prefix / horizon -------------------------------------------
    want_prefix = SMOKE_PREFIX if smoke else PROD_PREFIX
    if not tag.startswith(want_prefix):
        fail(tag, f"prefix is not {want_prefix!r}", out)
        return False, None, None, None
    if not smoke and tag.startswith(SMOKE_PREFIX):
        fail(tag, "a smoke cell cannot stand in for a production cell", out)
        return False, None, None, None
    want_rounds = SMOKE_ROUNDS if smoke else PROD_ROUNDS
    if int(c.get("n_rounds", -1)) != want_rounds:
        fail(tag, f"n_rounds={c.get('n_rounds')} != {want_rounds}", out)
        ok = False

    # ---- DIRECTION: the tag and the config must agree ---------------
    if "_fwd" in tag and "_rev" in tag:
        fail(tag, "tag names both directions", out)
        return False, None, None, None
    if "_fwd" in tag:
        tag_dir = "forward"
    elif "_rev" in tag:
        tag_dir = "reverse"
    else:
        fail(tag, "tag names no direction", out)
        return False, None, None, None
    cfg_dir = c.get("kl_direction")
    if cfg_dir != tag_dir:
        fail(tag, f"DIRECTION MISMATCH: tag says {tag_dir!r}, config "
                  f"recorded {cfg_dir!r}", out)
        ok = False
    if cfg_dir not in ("forward", "reverse"):
        fail(tag, f"kl_direction={cfg_dir!r} is neither direction "
                  f"(a 'js' or defaulted run is not this wave)", out)
        ok = False

    # ---- TRAINING really happened -----------------------------------
    if c.get("training_style") != "sft_kl":
        fail(tag, f"training_style={c.get('training_style')!r}, not sft_kl", out)
        ok = False
    lam = c.get("kl_beta")
    try:
        lam = float(lam)
    except (TypeError, ValueError):
        lam = None
    if lam is None or lam <= 0:
        fail(tag, f"kl_beta={c.get('kl_beta')!r} is not a positive lambda", out)
        ok = False
    elif lam not in LAMBDAS:
        fail(tag, f"kl_beta={lam} is not one of {LAMBDAS}", out)
        ok = False
    if not c.get("use_lora"):
        fail(tag, "use_lora is false -- nothing was trained", out)
        ok = False
    if not c.get("fresh_each_round"):
        fail(tag, "fresh_each_round is false -- the wave declares a fresh "
                  "LoRA every round", out)
        ok = False

    # ---- REFERENCE is the raw base ----------------------------------
    if c.get("kl_ref_adapter"):
        fail(tag, f"kl_ref_adapter={c.get('kl_ref_adapter')!r} -- the anchor "
                  f"must be the raw base checkpoint, so lambda means the "
                  f"same thing in every arm", out)
        ok = False
    if c.get("anchor_mode") != "fixed":
        fail(tag, f"anchor_mode={c.get('anchor_mode')!r}, not 'fixed'", out)
        ok = False
    if c.get("base_model") != BASE:
        fail(tag, f"base_model={c.get('base_model')!r}", out)
        ok = False

    # ---- surface pins ------------------------------------------------
    pins = {
        "seed": 0, "dataset": "movielens", "ml_target": "Action",
        "n_labeled": N_AGENTS, "ai_gate_mode": "all_open",
        "peer_gate_mode": "all_open", "ab_sweeps": 1, "pop_model": "ab",
        "icl_k": 0, "icl_days": 0, "lora_r": 512, "sft_epochs": 1,
        "epoch_size": 100, "train_cap": N_AGENTS, "eps": 0.2,
    }
    for k, v in pins.items():
        got = c.get(k, "<absent>")
        if got != v:
            fail(tag, f"{k}={got!r}, expected {v!r}", out)
            ok = False
    if abs(float(c.get("innate_lambda") or -1) - 1.0) > 1e-9:
        fail(tag, f"innate_lambda={c.get('innate_lambda')!r}, expected 1.0 "
                  f"(this is Celestine's gamma, the innate anchor)", out)
        ok = False
    w = c.get("w_plat")
    try:
        w = float(w)
    except (TypeError, ValueError):
        w = None
    if w not in (0.5, 1.0):
        fail(tag, f"w_plat={c.get('w_plat')!r}, expected 0.5 or 1.0 "
                  f"(this is Celestine's beta)", out)
        ok = False
    if abs(float(c.get("sft_lr") or -1) - 5e-5) > 1e-12:
        fail(tag, f"sft_lr={c.get('sft_lr')!r}, expected 5e-05", out)
        ok = False
    hw = c.get("hardware") or {}
    if hw.get("gpu_name") != H100:
        fail(tag, f"gpu={hw.get('gpu_name')!r}, expected {H100!r} -- greedy "
                  f"decoding is only bit-reproducible within an architecture",
             out)
        ok = False
    # serving provenance: this wave postdates the field, so ABSENCE is a
    # failure here (unlike the audited pre-2026-08-21 reuse cells)
    if c.get("serve_eval_mode") is not True:
        fail(tag, "serve_eval_mode is not True -- cannot certify that LoRA "
                  "dropout was off while serving", out)
        ok = False

    # ---- the artifact itself -----------------------------------------
    d_ = torch.load(tf, map_location="cpu", weights_only=False)
    pr = d_.get("pred_raw")
    if pr is None or not torch.is_tensor(pr) or pr.numel() == 0:
        fail(tag, "pred_raw missing or empty", out)
        return False, tag_dir, lam, w
    pr = pr.float()
    if tuple(pr.shape) != (want_rounds, N_AGENTS):
        fail(tag, f"pred_raw shape {tuple(pr.shape)} != "
                  f"{(want_rounds, N_AGENTS)} -- incomplete serving", out)
        ok = False
    if not torch.isfinite(pr).all():
        n_bad = int((~torch.isfinite(pr)).sum())
        fail(tag, f"{n_bad} non-finite served value(s) -- incomplete serving",
             out)
        ok = False
    op = d_.get("op_raw")
    if op is None or not torch.is_tensor(op) or tuple(op.shape) != \
            (want_rounds, N_AGENTS):
        fail(tag, f"op_raw shape "
                  f"{tuple(op.shape) if torch.is_tensor(op) else None} != "
                  f"{(want_rounds, N_AGENTS)}", out)
        ok = False
    innate = d_.get("innate")
    if innate is None or not torch.is_tensor(innate) or \
            innate.numel() != N_AGENTS:
        fail(tag, "innate vector missing -- the figure needs it as t=0", out)
        ok = False

    # ---- PARSE failures: exactly zero --------------------------------
    traj = d_.get("trajectory") or []
    pf = [r.get("parse_fail_frac") for r in traj
          if isinstance(r, dict) and "parse_fail_frac" in r]
    if not pf:
        fail(tag, "no parse_fail_frac recorded in any round", out)
        ok = False
    else:
        bad = [(i, v) for i, v in enumerate(pf)
               if v is None or not (float(v) == 0.0)]
        if bad:
            fail(tag, f"parse failures in {len(bad)} round(s), e.g. "
                      f"round {bad[0][0]} frac={bad[0][1]}", out)
            ok = False

    # ---- NONCONSTANT across agents -----------------------------------
    if torch.is_tensor(pr) and pr.numel():
        per_round_sd = pr.std(dim=1, unbiased=False)
        if float(per_round_sd.max()) == 0.0:
            fail(tag, "served map is CONSTANT across agents in every round "
                      "-- a fully collapsed map, not a retention result", out)
            ok = False
        # a served vector identical in every round means the adapter never
        # changed what the model emits: for a TRAINED arm that is the
        # frozen signature and must not be read as retention
        if pr.shape[0] > 1 and float((pr - pr[0]).abs().max()) == 0.0:
            fail(tag, "served vector is bit-identical in EVERY round -- a "
                      "trained arm that never moved (frozen signature)", out)
            ok = False

    # ---- losses finite and not degenerate ----------------------------
    li = [r.get("l_init") for r in traj if isinstance(r, dict)]
    li = [float(v) for v in li if v is not None]
    if not li:
        fail(tag, "no l_init recorded -- cannot confirm training ran", out)
        ok = False
    else:
        import math
        if not all(math.isfinite(v) for v in li):
            fail(tag, "non-finite training loss", out)
            ok = False

    return ok, tag_dir, lam, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--smoke", action="store_true",
                    help="gate a 3-round pofdkdsmk_ cell instead of the wave")
    args = ap.parse_args()

    out, seen, allok = [], [], True
    for d in args.runs:
        ok, direction, lam, w = check_one(d, args.smoke, out)
        allok &= ok
        if ok:
            seen.append((direction, lam, w))
            out.append(f"[check_kd] pass {os.path.basename(d.rstrip('/'))} "
                       f"({direction}, lambda={lam:g}, W={w:g})")

    if not args.smoke and allok:
        # a production wave must actually BE the comparison
        dirs = {d for d, _, _ in seen}
        if dirs != {"forward", "reverse"}:
            out.append(f"[check_kd] FAIL wave: directions present = "
                       f"{sorted(dirs)}; the comparison needs both. Gate the "
                       f"whole wave, not a subset.")
            allok = False
        if len(set(seen)) != len(seen):
            out.append("[check_kd] FAIL wave: duplicate (direction, lambda, "
                       "W) cells")
            allok = False
        ws = {w for _, _, w in seen}
        if ws != {0.5, 1.0}:
            out.append(f"[check_kd] FAIL wave: W values present = "
                       f"{sorted(ws)}; both columns of the figure are needed")
            allok = False

    for line in out:
        print(line)
    n = len(args.runs)
    if allok:
        print(f"[check_kd] PASS -- {n} run(s): direction matches tag, trained "
              f"with a positive lambda against the raw base, serving "
              f"complete, zero parse failures, served map nonconstant")
        return 0
    print(f"[check_kd] FAILED -- see above ({n} run(s) inspected)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
