#!/usr/bin/env python3
"""GATE for the recursive update-dose wave (pofdud_, 2026-08-25).

The wave's one claim: every arm consumes ALL 723 current live labels
exactly once per round in the same order, and ONLY the optimizer-step
frequency differs.  So the checks are exactly those:

  1. per-round sft_dose records exist (trajectory.pt["sft_dose"]) and the
     realized global_step equals the tag's u EVERY round (u1 -> 1, u5 ->
     5, u19 -> 19); n_rows == 723 every round.
  2. SAVE_SFT_ORDER artifacts: each round's sft_order_idx_raw row holds
     all 723 agent ids EXACTLY ONCE (a permutation), and the ordered
     labels equal the previous round's post-peer opinions (round 0: the
     innate opinions) -- live labels, no replay, no subsample.
  3. config: sft_grad_accum matches the arm, plain sft (kl_beta 0),
     W = 1, k = 1, S = 100, both gates all_open, anch2 operator, lr 5e-5,
     rank 512, batch 4, epochs 1, Qwen3-8B thinking off.
  4. cross-arm: the same sampler order (trainer_seed) in every arm.
  5. zero parse failures (raw_gen_log.json.gz).

--smoke gates the 2-round u1 cell: 723 consumed exactly once per round
and global_step == 1 each round, exactly as specified.

--seeds 42,43 gates the SEED REPLICATION (2026-08-28): U in {1, 5, 181}
at each named seed, ten rounds, everything else byte-identical to the
seed-0 cells.  Two checks are stricter here than on the original wave,
because these cells carry no archived-reuse excuse:
  * raw_gen_log.json.gz is MANDATORY.  The parser stores a finite 0.5 on
    failure, so an absent log means the malformed rate cannot be
    established at all -- absence is a FAILURE, not a pass.
  * every logged `parsed` vector must equal pred_raw for that round,
    agent by agent.  A well-formed log proves internal consistency; only
    this proves the log describes THIS trajectory.
The cross-arm sampler-order check runs WITHIN each seed.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import gzip
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
DEFAULT_RUN_ROOTS = (
    Path("/home/gsmithline/perfsim/runs/pokec_gated_lm"),
    REPO / "runs" / "pokec_gated_lm",
    REPO / "notes" / "pofd" / "cluster",
)
N = 723
ARMS = {1: 181, 5: 37, 19: 10}     # u -> accum, the seed-0 ladder
EXT_ARMS = {1: 181, 5: 37, 181: 1}  # u -> accum, the replicated ladder
EXT_SEEDS = (42, 43)


def _find(tag, roots):
    for r in roots:
        p = Path(r) / tag / "trajectory.pt"
        if p.exists():
            return p
    return None


def check_cell(tag, u, accum, rounds, roots, seed=0, strict_raw=False):
    errs = []
    p = _find(tag, roots)
    if p is None:
        return {"tag": tag, "status": "ABSENT",
                "errors": ["no trajectory.pt"]}
    d = torch.load(p, map_location="cpu", weights_only=False)
    cfg = d.get("config", {}) or {}
    # run_pokec_gated_lm.py:1882 writes sft_grad_accum into config ONLY
    # when it exceeds 1, so at accum 1 the key is legitimately ABSENT and
    # means the default. Read it with that default rather than skipping
    # the gate entirely (the old behaviour): an expected accum > 1 that
    # is missing still fails, so a mislabelled cell cannot slip through.
    # The binding evidence either way is the DOSE check below -- realized
    # global_step must equal the tag's u -- which no config value can fake.
    got_accum = cfg.get("sft_grad_accum", 1)
    if got_accum != accum:
        errs.append(f"CONFIG sft_grad_accum={got_accum!r} (want {accum})")
    for key, want in (("w_plat", 1.0), ("innate_lambda", 1.0),
                      ("kl_beta", 0.0), ("ab_sweeps", 100),
                      ("n_rounds", rounds), ("seed", seed),
                      ("training_style", "sft"), ("run_mode", "loop"),
                      ("data_regime", "replace"), ("train_cap", 723),
                      ("save_sft_order", True), ("save_raw_gen", True),
                      ("chat_thinking", False), ("do_sample", False),
                      ("ai_gate_reference", "anchor"),
                      ("ai_gate_mode", "all_open"),
                      ("peer_gate_mode", "all_open"),
                      ("sft_lr", 5e-5), ("lora_r", 512),
                      ("sft_batch_size", 4), ("sft_epochs", 1),
                      ("base_model", "Qwen/Qwen3-8B"),
                      ("population_update",
                       "nested_ai_anchored_then_social_v2")):
        got = cfg.get(key, "<absent>")
        ok = (abs(float(got) - want) <= 1e-9 if isinstance(want, float)
              and isinstance(got, (int, float)) else got == want)
        if not ok:
            errs.append(f"CONFIG {key}={got!r} (want {want!r})")
    # 1. realized steps + rows, EVERY round
    dose = d.get("sft_dose", [])
    if len(dose) < rounds:
        errs.append(f"DOSE only {len(dose)} sft_dose records "
                    f"(want {rounds})")
    for rec in dose[:rounds]:
        if int(rec.get("global_step", -1)) != u:
            errs.append(f"DOSE round {rec.get('round')}: global_step="
                        f"{rec.get('global_step')} (want {u})")
        if int(rec.get("n_rows", -1)) != N:
            errs.append(f"DOSE round {rec.get('round')}: n_rows="
                        f"{rec.get('n_rows')} (want {N})")
    seeds = {int(r.get("trainer_seed", -1)) for r in dose[:rounds]}
    # 2. every round a PERMUTATION of all 723 ids, labels live
    idx = d.get("sft_order_idx_raw")
    y = d.get("sft_order_y_raw")
    if idx is None or y is None:
        errs.append("ORDER sft_order artifacts absent (SAVE_SFT_ORDER)")
    else:
        idx = torch.as_tensor(idx)[:rounds]
        y = torch.as_tensor(y).float()[:rounds]
        op = torch.as_tensor(d["op_raw"]).float()
        inn = torch.as_tensor(d["innate"]).float()
        for t in range(min(rounds, idx.shape[0])):
            row = idx[t]
            if row.numel() != N or len(set(row.tolist())) != N:
                errs.append(f"ORDER round {t}: {row.numel()} rows, "
                            f"{len(set(row.tolist()))} unique -- every "
                            f"agent must appear EXACTLY once")
                continue
            src = inn if t == 0 else op[t - 1]
            if not torch.allclose(y[t], src[row.long()], atol=1e-6):
                errs.append(f"ORDER round {t}: labels are not the live "
                            f"post-peer opinions (no replay allowed)")
    # 5. zero parse failures + (strict) the log must describe THIS run
    raw = p.parent / "raw_gen_log.json.gz"
    worst_pvp = None
    if not raw.exists():
        if strict_raw:
            errs.append("PARSE raw_gen_log.json.gz ABSENT -- the parser "
                        "stores a finite 0.5 on failure, so the malformed "
                        "rate cannot be established anywhere else "
                        "(SAVE_RAW_GEN=1 is mandatory for this wave)")
    else:
        rows = []
        try:
            with gzip.open(raw, "rt") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except (OSError, ValueError, json.JSONDecodeError) as e:
            errs.append(f"PARSE raw_gen_log.json.gz unreadable: {e}")
            rows = []
        for rec in rows:
            pf = rec.get("parse_fail_frac")
            if pf is None or float(pf) > 0:
                errs.append(f"PARSE round {rec.get('round')} "
                            f"parse_fail_frac={pf!r} (want exactly 0)")
                break
        if strict_raw and rows:
            got = [r.get("round") for r in rows]
            if got != list(range(rounds)):
                errs.append(f"PARSE raw_gen_log must carry rounds "
                            f"0..{rounds - 1} once, in order; got "
                            f"{got[:6]}{'...' if len(got) > 6 else ''}")
            pred = d.get("pred_raw")
            by_round = {r.get("round"): r.get("parsed") for r in rows}
            if torch.is_tensor(pred):
                pred = pred.float()
                bad, worst_pvp = [], 0.0
                for t in range(min(rounds, pred.shape[0])):
                    vals = by_round.get(t)
                    if not isinstance(vals, list) or len(vals) != N:
                        bad.append((t, "missing/short parsed vector"))
                        continue
                    pv = torch.tensor([float(v) for v in vals],
                                      dtype=torch.float32)
                    diff = (pv.clamp(0, 1) - pred[t].clamp(0, 1)).abs()
                    worst_pvp = max(worst_pvp, float(diff.max()))
                    if float(diff.max()) > 1e-6:
                        i = int(diff.argmax())
                        bad.append((t, f"agent {i}: parsed "
                                       f"{float(pv[i]):.6f} != pred_raw "
                                       f"{float(pred[t][i]):.6f}"))
                if bad:
                    errs.append(f"PARSE-VS-SERVED parsed != pred_raw in "
                                f"{len(bad)} round(s) (max |diff| "
                                f"{worst_pvp:.3e}), e.g. round {bad[0][0]} "
                                f"{bad[0][1]} -- the raw log does not "
                                f"describe this trajectory")
    op = torch.as_tensor(d["op_raw"]).float()
    if op.shape[0] < rounds:
        errs.append(f"TRAJECTORY only {op.shape[0]} of {rounds} rounds")
    if not torch.isfinite(op).all():
        errs.append("TRAJECTORY non-finite opinions")
    out = {"tag": tag, "status": "PASS" if not errs else "FAIL",
           "errors": errs, "trainer_seeds": sorted(seeds), "seed": seed,
           "u": u, "accum": accum,
           "mean": float(op[min(rounds, op.shape[0]) - 1].mean()),
           "sd": float(op[min(rounds, op.shape[0]) - 1].std())}
    if worst_pvp is not None:
        out["parsed_vs_pred_max_abs"] = worst_pvp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", action="append", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds for the replication wave "
                         "(e.g. 42,43); gates U in {1,5,181} at each")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    roots = [Path(r) for r in (args.run_root or DEFAULT_RUN_ROOTS)]

    recs = []
    if args.seeds:
        want = [int(s) for s in args.seeds.split(",") if s.strip()]
        if set(want) - set(EXT_SEEDS):
            print(f"[check_ud] NOTE: gating seeds {want}; the generated "
                  f"replication covers {list(EXT_SEEDS)}")
        for sd in want:
            for u, acc in EXT_ARMS.items():
                recs.append(check_cell(
                    f"pofdud_qwen3_8b_sft_u{u}_sw100_eaopen_w1_k1_esopen"
                    f"_anch2_s{sd}_r10", u, acc, 10, roots, seed=sd,
                    strict_raw=True))
        # cross-arm sampler order, WITHIN each seed
        for sd in want:
            ss = {tuple(r.get("trainer_seeds", [])) for r in recs
                  if r.get("seed") == sd and r["status"] != "ABSENT"}
            if len(ss) > 1:
                recs.append({"tag": f"<cross-arm sampler order s{sd}>",
                             "status": "FAIL", "errors":
                             [f"seed {sd} arms record different trainer "
                              f"seeds {ss} -- the minibatch order must "
                              f"be shared across U at a fixed seed"]})
    elif args.smoke:
        recs.append(check_cell(
            "pofdudsmk_qwen3_8b_sft_u1_sw100_eaopen_w1_k1_esopen_anch2"
            "_s0_r2", 1, 181, 2, roots))
    else:
        for u, acc in ARMS.items():
            recs.append(check_cell(
                f"pofdud_qwen3_8b_sft_u{u}_sw100_eaopen_w1_k1_esopen"
                f"_anch2_s0_r10", u, acc, 10, roots))
        # cross-arm: identical sampler order
        seedsets = {tuple(r.get("trainer_seeds", [])) for r in recs
                    if r["status"] != "ABSENT"}
        if len(seedsets) > 1:
            recs.append({"tag": "<cross-arm sampler order>",
                         "status": "FAIL", "errors":
                         [f"arms record different trainer seeds "
                          f"{seedsets} -- the minibatch order must be "
                          f"shared"]})
    ok = True
    for r in recs:
        if r["status"] != "PASS":
            ok = False
        m = f"{r.get('mean', float('nan')):.4f}" if "mean" in r else "-"
        print(f"{r['tag']:<70}{r['status']:<8}{m}")
        for e in r["errors"]:
            print(f"    !! {e}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"ok": ok, "cells": recs}, indent=2, default=str))
    print(f"[check_ud] {'PASS' if ok else 'FAIL'} -- every arm consumed "
          f"all {N} live labels exactly once per round in the shared "
          f"order, at its exact realized step count."
          if ok else "[check_ud] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
