#!/usr/bin/env python3
"""SECTION-4 PROBE analyzer (2026-08-26): cohort-B signed/absolute
effect for the 8-cell, 5-round, seed-0, beta=0.75 channel probe
(section4_gate_anch2_probe; tags pofds4gp_*).

READ-ONLY and DESCRIPTIVE: seed 0 is the only replicate, so NO
confidence intervals are computed or reported anywhere in this file --
any claim from this probe is descriptive until a seed replication runs.

Run AFTER the gate (check_section4_gate.py --wave probe) passes; this
file trusts the gate for operator/clamp/parse integrity and re-checks
only what it consumes (shapes, one shared innate vector, the stored
fixed masks against the reconstruction).

THE REGISTERED QUESTION (written before submission, per the QUESTIONS.md
convention): the signed and absolute effect on cohort B -- the 578
responsive agents -- relative to the matched no-platform twin. Is
personal-history ICL (d8) near zero with peers CLOSED (es=0) but
STRONGER than ordinary SFT (b0) with peers OPEN (es=1)?

Metrics, all over cohort B only, all vs the run's own twin_raw:
  signed_b  mean(op[t][B] - twin[t][B])   (direction of the pull)
  abs_b     mean|op[t][B] - twin[t][B]|   (magnitude, MAD)
  w1_b      Wasserstein-1 between the two cohort-B populations
op_raw is the END-OF-ROUND POST-PEER state (peer sweeps run last), so
round r means "after round r's AI blend and Deffuant sweep".

Cohort A = the 145 lowest-innate agents under the deterministic
(innate, id) ranking -- reconstructed identically for BOTH conditions
via analyze_section4_gate.cohort_a_mask, so the evolving condition
(which stores no mask) is masked exactly like its fixed partner.

Outputs (out dir, default notes/pofd/s4g_probe/):
  s4g_probe_cohortB.csv    every (cond, arm, es, round) row
  s4g_probe_verdict.json   machine-readable summary (--json to move it)
and a printed final-round table plus the two hypothesis contrasts.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys

import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GEN_PATH = os.path.join(REPO, "experiments", "condor", "gen_pofd_sweep.py")
AN_PATH = os.path.join(HERE, "analyze_section4_gate.py")

LOG = "[s4g_probe]"
N = 723


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Section-4 probe: cohort-B effect vs twin; CPU only")
    ap.add_argument("--run-root",
                    default="/home/gsmithline/perfsim/runs/pokec_gated_lm")
    ap.add_argument("--out-dir",
                    default=os.path.join(REPO, "notes", "pofd", "s4g_probe"))
    ap.add_argument("--gen", default=GEN_PATH)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    gen = _load(args.gen, "_gen_s4gp")
    AN = _load(AN_PATH, "_an_s4gp")

    conds = tuple(gen.S4G_CONDS)
    arms = tuple(gen.S4GP_ARMS)
    ess = tuple(float(e) for e in gen.S4GP_ESS)
    ea = float(gen.S4GP_GATES[0])
    seed = int(gen.S4GP_SEEDS[0])
    rounds = int(gen.S4GP_ROUNDS)

    cells = [(cond, arm, es) for cond in conds for arm in arms
             for es in ess]
    run_of, missing = {}, []
    for cond, arm, es in cells:
        tag = gen.s4gp_tag(arm, cond, ea, es, seed)
        rd = AN.find_run(args.run_root, tag)
        (missing.append(tag) if rd is None
         else run_of.__setitem__((cond, arm, es), rd))
    print(f"{LOG} trajectories located: {len(run_of)}/{len(cells)}")
    if missing:
        for t in missing:
            print(f"{LOG}   MISSING {t}")
        print(f"{LOG} HARD FAIL: {len(missing)} of {len(cells)} cells "
              f"missing -- no output written", file=sys.stderr)
        return 1

    rows_csv, per_cell, inn_sha = [], {}, {}
    for (cond, arm, es), rd in sorted(run_of.items(), key=str):
        d = AN.load(rd)
        op = d["op_raw"].float()
        tw, tw_src = AN.twin_of(d)
        inn = d["innate"].float()
        if tuple(op.shape) != (rounds, N) or tuple(tw.shape) != (rounds, N):
            print(f"{LOG} HARD FAIL {os.path.basename(rd)}: shapes "
                  f"{tuple(op.shape)}/{tuple(tw.shape)} != {(rounds, N)}",
                  file=sys.stderr)
            return 1
        inn_sha[(cond, arm, es)] = AN.innate_sha(inn)
        mask_a = AN.cohort_a_mask(inn)
        cm = d.get("innate_clamp_mask")
        if torch.is_tensor(cm) and cm.numel() and \
                not torch.equal(cm.bool(), mask_a):
            print(f"{LOG} HARD FAIL {os.path.basename(rd)}: stored clamp "
                  f"mask != reconstructed bottom-145 cohort", file=sys.stderr)
            return 1
        b = ~mask_a
        rec_rounds = []
        for t in range(rounds):
            diff = op[t][b] - tw[t][b]
            row = {
                "cond": cond, "arm": arm, "es": f"{es:g}", "round": t + 1,
                "signed_b": float(diff.mean()),
                "abs_b": float(diff.abs().mean()),
                "w1_b": AN.w1(op[t][b], tw[t][b]),
                "mu_op_b": float(op[t][b].mean()),
                "mu_tw_b": float(tw[t][b].mean()),
                "sd_op_b": float(op[t][b].std()),
                "sd_tw_b": float(tw[t][b].std()),
                "mu_op_a": float(op[t][mask_a].mean()),
                "twin_source": tw_src,
            }
            rec_rounds.append(row)
            rows_csv.append(row)
        per_cell[(cond, arm, es)] = rec_rounds[-1]
        del d, op, tw, inn

    if len(set(inn_sha.values())) != 1:
        print(f"{LOG} HARD FAIL: {len(set(inn_sha.values()))} distinct "
              f"innate vectors across the probe -- the cohort masks are "
              f"not comparable", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "s4g_probe_cohortB.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        wtr.writeheader()
        wtr.writerows(rows_csv)
    print(f"{LOG} wrote {csv_path} ({len(rows_csv)} rows)")

    # ---- final-round table --------------------------------------------
    print(f"\n{LOG} COHORT-B EFFECT vs MATCHED TWIN, final round "
          f"({rounds} of {rounds}), post-peer. SEED 0 ONLY -- descriptive, "
          f"no intervals.")
    hdr = (f"{'cond':<9} {'es':>3} {'arm':>3} {'signed_b':>9} {'abs_b':>8} "
           f"{'w1_b':>8} {'mu_op_b':>8} {'mu_tw_b':>8} {'sd_ratio':>8}")
    print(hdr)
    print("-" * len(hdr))
    for cond in conds:
        for es in ess:
            for arm in arms:
                r = per_cell[(cond, arm, es)]
                sdr = (r["sd_op_b"] / r["sd_tw_b"]
                       if r["sd_tw_b"] > 0 else float("nan"))
                print(f"{cond:<9} {es:>3g} {arm:>3} {r['signed_b']:>+9.4f} "
                      f"{r['abs_b']:>8.4f} {r['w1_b']:>8.4f} "
                      f"{r['mu_op_b']:>8.4f} {r['mu_tw_b']:>8.4f} "
                      f"{sdr:>8.3f}")

    # ---- the registered contrasts -------------------------------------
    print(f"\n{LOG} REGISTERED QUESTION -- is ICL (d8) ~0 with peers "
          f"closed (es=0) but stronger than SFT (b0) with peers open "
          f"(es=1)?")
    verdicts = {}
    for cond in conds:
        closed_d8 = per_cell[(cond, "d8", 0.0)]["abs_b"]
        closed_b0 = per_cell[(cond, "b0", 0.0)]["abs_b"]
        open_d8 = per_cell[(cond, "d8", 1.0)]["abs_b"]
        open_b0 = per_cell[(cond, "b0", 1.0)]["abs_b"]
        verdicts[cond] = {
            "closed_d8_abs": closed_d8, "closed_b0_abs": closed_b0,
            "open_d8_abs": open_d8, "open_b0_abs": open_b0,
            "icl_stronger_when_open": open_d8 > open_b0,
        }
        print(f"{LOG}   {cond}: peers CLOSED d8 abs={closed_d8:.4f} "
              f"(b0 {closed_b0:.4f}); peers OPEN d8 abs={open_d8:.4f} vs "
              f"b0 {open_b0:.4f} -> ICL {'>' if open_d8 > open_b0 else '<='}"
              f" SFT when open")

    verdict = {
        "wave": gen.S4GP_KEY, "rounds": rounds, "seed": seed,
        "eps_ai": ea, "w_plat": float(gen.S4GP_W_PLAT),
        "cells": [{"cond": c, "arm": a, "es": e, **per_cell[(c, a, e)]}
                  for (c, a, e) in sorted(per_cell, key=str)],
        "contrasts": verdicts,
        "note": "seed 0 only -- descriptive, no intervals",
    }
    jp = args.json_out or os.path.join(args.out_dir,
                                       "s4g_probe_verdict.json")
    os.makedirs(os.path.dirname(os.path.abspath(jp)), exist_ok=True)
    with open(jp, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(f"{LOG} verdict -> {jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
