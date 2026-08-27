#!/usr/bin/env python3
"""SECTION-4 PROBE / SCOUT analyzer (2026-08-26/27): cohort-B effects
for the beta=0.75 channel family --
  --variant probe   8 cells, 5 rounds, es {0, 1}            (pofds4gp_)
  --variant scout  20 cells, 30 rounds, es {0,.1,.2,.3,1}  (pofds4gs_)
both seed 0, ea=0.7, gamma=0.2, alpha=0.5, {b0, d8} x {fixed, evolving}.
For the scout the crossover question is answered per social gate on the
LATE WINDOW (the final five post-peer rounds, the Figure-6 convention)
with the final round beside it.

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

TWO CONTRASTS, kept apart because they answer different questions:

(1) DIRECT fixed-vs-evolving contrast -- THE FIGURE-5 QUANTITY (added
    2026-08-27 after the first read conflated the two). Does cohort A's
    state reach cohort B? Each arm's fixed run is paired AGENT BY AGENT
    with its evolving run at the same (arm, es):
      mae_b_paired   mean_i |op_B(fixed)[t][i] - op_B(evolving)[t][i]|
      delta_mu_b     mean op_B(fixed) - mean op_B(evolving)   (= -T_a)
      w1_b_pair      Wasserstein-1 between the two cohort-B populations
    The d8/es=0 cell is the structural null: frozen weights, own-history
    prompts and an inert peer step give A no path to B, so mae_b_paired
    must be 0 exactly there.
(2) PLATFORM EFFECT vs the run's own twin -- agent-local: does the
    platform move B at all, relative to the matched no-platform process?
      signed_b  mean(op[t][B] - twin[t][B])   (direction of the pull)
      abs_b     mean|op[t][B] - twin[t][B]|   (magnitude, MAD)
      w1_b      Wasserstein-1 between the two cohort-B populations
    This is NOT transmission: it is identical for fixed and evolving
    wherever A cannot reach B.
op_raw is the END-OF-ROUND POST-PEER state (peer sweeps run last), so
round r means "after round r's AI blend and Deffuant sweep".

Cohort A = the 145 lowest-innate agents under the deterministic
(innate, id) ranking -- reconstructed identically for BOTH conditions
via analyze_section4_gate.cohort_a_mask, so the evolving condition
(which stores no mask) is masked exactly like its fixed partner.

Outputs (out dir, default notes/pofd/s4g_probe/):
  s4g_probe_direct_contrast.csv   (1) per (arm, es, round)
  s4g_probe_cohortB.csv           (2) per (cond, arm, es, round)
  s4g_probe_verdict.json          machine-readable summary (--json moves it)
and printed final-round tables plus the registered contrasts in BOTH
frames; the transmission frame (1) is the one the question is about.
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
    ap.add_argument("--variant", default="probe", choices=("probe", "scout"),
                    help="probe (8 cells, 5 rounds, es {0,1}; DEFAULT) or "
                         "scout (20 cells, 30 rounds, es {0,.1,.2,.3,1})")
    ap.add_argument("--run-root",
                    default="/home/gsmithline/perfsim/runs/pokec_gated_lm")
    ap.add_argument("--out-dir", default=None,
                    help="default notes/pofd/s4g_<variant>/")
    ap.add_argument("--gen", default=GEN_PATH)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = os.path.join(REPO, "notes", "pofd",
                                    f"s4g_{args.variant}")
    stem = f"s4g_{args.variant}"

    gen = _load(args.gen, "_gen_s4gp")
    AN = _load(AN_PATH, "_an_s4gp")
    fam = gen.S4G_VARIANTS[args.variant]

    conds = tuple(gen.S4G_CONDS)
    arms = tuple(gen.S4GP_ARMS)
    ess = tuple(float(e) for e in fam["ess"])
    ea = float(gen.S4GP_GATES[0])
    seed = int(gen.S4GP_SEEDS[0])
    rounds = int(fam["rounds"])
    late = list(range(max(0, rounds - 5), rounds))     # final five rounds
    print(f"{LOG} variant={args.variant} key={fam['key']} rounds={rounds} "
          f"es={list(ess)} late window op_raw {late[0]}-{late[-1]}")

    cells = [(cond, arm, es) for cond in conds for arm in arms
             for es in ess]
    run_of, missing = {}, []
    for cond, arm, es in cells:
        tag = gen.s4gv_tag(arm, cond, ea, es, seed, fam["prefix"])
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
    op_b = {}                      # (cond, arm, es) -> op_raw[:, B] (5 x 578)
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
        op_b[(cond, arm, es)] = op[:, b].clone()
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

    # ---- (1) DIRECT fixed-vs-evolving contrast: the Figure-5 quantity --
    direct_rows, direct_final = [], {}
    for arm in arms:
        for es in ess:
            f, e = op_b[("fixed", arm, es)], op_b[("evolving", arm, es)]
            for t in range(rounds):
                diff = f[t] - e[t]
                row = {"arm": arm, "es": f"{es:g}", "round": t + 1,
                       "mae_b_paired": float(diff.abs().mean()),
                       "delta_mu_b": float(diff.mean()),
                       "t_a_evolving_minus_fixed": float(-diff.mean()),
                       "w1_b_pair": AN.w1(f[t], e[t]),
                       "max_abs_diff": float(diff.abs().max()),
                       "bit_identical": bool(torch.equal(f[t], e[t]))}
                direct_rows.append(row)
            fin = dict(direct_rows[-1])
            lw = [direct_rows[-rounds + t] for t in late]
            fin["mae_b_paired_late"] = sum(r["mae_b_paired"] for r in lw) / len(lw)
            fin["delta_mu_b_late"] = sum(r["delta_mu_b"] for r in lw) / len(lw)
            direct_final[(arm, es)] = fin
    dpath = os.path.join(args.out_dir, f"{stem}_direct_contrast.csv")
    with open(dpath, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(direct_rows[0]))
        wtr.writeheader()
        wtr.writerows(direct_rows)
    print(f"{LOG} wrote {dpath} ({len(direct_rows)} rows)")
    print(f"\n{LOG} (1) DIRECT fixed-vs-evolving contrast on cohort B "
          f"(agent-paired), final round {rounds} of {rounds}, post-peer. "
          f"THE FIGURE-5 QUANTITY. SEED 0 ONLY -- descriptive.")
    hdr = (f"{'es':>4} {'arm':>3} {'mae_b_paired':>13} {'delta_mu_b':>11} "
           f"{'w1_b_pair':>10} {'max|d|':>8} {'bit-identical':>13} | "
           f"{'mae_late5':>10} {'delta_late5':>12}")
    print(hdr)
    print("-" * len(hdr))
    for es in ess:
        for arm in arms:
            r = direct_final[(arm, es)]
            print(f"{es:>4g} {arm:>3} "
                  f"{r['mae_b_paired']:>13.4f} {r['delta_mu_b']:>+11.4f} "
                  f"{r['w1_b_pair']:>10.4f} {r['max_abs_diff']:>8.4f} "
                  f"{str(r['bit_identical']):>13} | "
                  f"{r['mae_b_paired_late']:>10.4f} "
                  f"{r['delta_mu_b_late']:>+12.4f}")
    # the crossover per social gate, on the late window
    print(f"\n{LOG} CROSSOVER per social gate (late-window mae_b_paired, "
          f"rounds {late[0] + 1}-{late[-1] + 1}): SFT(b0) vs ICL(d8)")
    crossover = []
    for es in ess:
        mb = direct_final[("b0", es)]["mae_b_paired_late"]
        md = direct_final[("d8", es)]["mae_b_paired_late"]
        win = ("null (ICL exactly 0)" if md == 0.0 else
               "ICL" if md > mb else "SFT")
        ratio = (md / mb) if mb > 0 else float("nan")
        crossover.append({"es": es, "sft_mae_late": mb, "icl_mae_late": md,
                          "icl_over_sft": ratio, "larger": win})
        print(f"{LOG}   es={es:<4g} SFT {mb:.4f}  ICL {md:.4f}  "
              f"ICL/SFT={ratio:.2f}  -> {win}")
    print(f"\n{LOG} REGISTERED QUESTION in the transmission frame -- does "
          f"A reach B through ICL (d8) ~0 with peers closed, and more than "
          f"through SFT (b0) with peers open?")
    es_open = max(ess)
    closed_d8 = direct_final[("d8", 0.0)]["mae_b_paired"]
    closed_b0 = direct_final[("b0", 0.0)]["mae_b_paired"]
    open_d8 = direct_final[("d8", es_open)]["mae_b_paired"]
    open_b0 = direct_final[("b0", es_open)]["mae_b_paired"]
    direct_verdict = {
        "closed_d8_mae": closed_d8, "closed_b0_mae": closed_b0,
        "closed_d8_bit_identical": direct_final[("d8", 0.0)]["bit_identical"],
        "open_d8_mae": open_d8, "open_b0_mae": open_b0,
        "icl_transmits_more_when_open": open_d8 > open_b0,
        "icl_null_when_closed": closed_d8 == 0.0,
    }
    print(f"{LOG}   peers CLOSED: d8 mae={closed_d8:.4f} "
          f"({'EXACT structural null' if closed_d8 == 0.0 else 'NOT zero'}), "
          f"b0 mae={closed_b0:.4f} (shared-weight route)")
    print(f"{LOG}   peers OPEN (es={es_open:g}): d8 mae={open_d8:.4f} vs "
          f"b0 mae={open_b0:.4f} -> ICL {'>' if open_d8 > open_b0 else '<='}"
          f" SFT transmission at round {rounds}")

    # ---- (2) PLATFORM EFFECT vs twin: agent-local, NOT transmission ------
    csv_path = os.path.join(args.out_dir, f"{stem}_cohortB.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        wtr.writeheader()
        wtr.writerows(rows_csv)
    print(f"{LOG} wrote {csv_path} ({len(rows_csv)} rows)")

    # ---- final-round table --------------------------------------------
    print(f"\n{LOG} (2) PLATFORM EFFECT on cohort B vs the MATCHED TWIN "
          f"(agent-local, NOT transmission), final round ({rounds} of "
          f"{rounds}), post-peer. SEED 0 ONLY -- descriptive, no intervals.")
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
    print(f"\n{LOG} the same contrast in the platform-vs-twin frame (for "
          f"the record; it does NOT answer the transmission question):")
    verdicts = {}
    for cond in conds:
        closed_d8 = per_cell[(cond, "d8", 0.0)]["abs_b"]
        closed_b0 = per_cell[(cond, "b0", 0.0)]["abs_b"]
        open_d8 = per_cell[(cond, "d8", es_open)]["abs_b"]
        open_b0 = per_cell[(cond, "b0", es_open)]["abs_b"]
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
        "wave": fam["key"], "variant": args.variant, "rounds": rounds,
        "seed": seed, "late_window_op_raw": [late[0], late[-1]],
        "crossover_late_window": crossover,
        "eps_ai": ea, "w_plat": float(gen.S4GP_W_PLAT),
        "cells": [{"cond": c, "arm": a, "es": e, **per_cell[(c, a, e)]}
                  for (c, a, e) in sorted(per_cell, key=str)],
        "direct_contrast_final": [
            {"arm": a, "es": e, **direct_final[(a, e)]}
            for (a, e) in sorted(direct_final, key=str)],
        "contrasts_direct_transmission": direct_verdict,
        "contrasts_platform_vs_twin": verdicts,
        "primary_frame": "direct_transmission (mae_b_paired)",
        "note": "seed 0 only -- descriptive, no intervals",
    }
    jp = args.json_out or os.path.join(args.out_dir, f"{stem}_verdict.json")
    os.makedirs(os.path.dirname(os.path.abspath(jp)), exist_ok=True)
    with open(jp, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(f"{LOG} verdict -> {jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
