#!/usr/bin/env python3
"""No-peer innate-clamp analysis (2026-08-17, mistral_innate_clamp_nopeer).

THE TEST: with 20% of the population permanently pinned to its innate
opinions, does ordinary fresh SFT (b0) or frozen-weight live K=8 ICL
(dyn) preserve more of the RESPONSIVE population's structure -- and, for
the bottom cohort, how far does the responsive population close the gap
toward the frozen block?

Read-only, descriptive. Cells: 2 clamp modes {stratified_random(strat),
bottom} x 2 arms {b0, dyn} x eps_AI {0.05,0.1,0.2,0.4,1.0} x seeds
{0,42,43}, es=0, family pofdclamp_. All statistics are computed on the
RESPONSIVE subset (the frozen cohort is innate by construction; mixing
it in would mechanically deflate every dispersion metric).

Reported, rounds 25-29 (window mean) alongside the final round, with
SEEDS as the replicates (mean/SD/95% Student-t, t_crit(2)=4.30265):
  * responsive population SD relative to its matched twin's responsive SD
  * responsive prediction SD
  * responsive displacement from the twin (mean |op - twin|)
  * responsive AI acceptance (gate fraction) and cumulative reach (the
    fraction of responsive agents gated at least once by the round)
  * BOTTOM COHORT ONLY: responsive-to-frozen mean gap and 1-Wasserstein
    gap (quantile-interpolated -- the cohorts differ in size), plus the
    NORMALIZED GAP CLOSURE 1 - gap_late/gap_innate for both.

Outputs notes/pofd/clamp_analysis/: clamp_per_seed.csv,
clamp_summary.csv.
"""
import argparse
import csv
import importlib.util
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "analyze_reach", os.path.join(HERE, "analyze_sft_icl_reach.py"))
AN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN)

NA = "NA"
T_CRIT = {1: 12.7062, 2: 4.30265, 3: 3.18245, 4: 2.77645}
LATE_LO, LATE_HI = 25, 30
MODES = {"strat": "stratified_random", "bottom": "bottom"}
ARMS = ["b0", "dyn"]
GATES = [0.05, 0.1, 0.2, 0.4, 1.0]
SEEDS = [0, 42, 43]
METRICS = ["resp_std_ratio_late", "resp_std_ratio_final",
           "resp_pred_std_late", "resp_pred_std_final",
           "resp_disp_twin_late", "resp_disp_twin_final",
           "resp_accept_late", "resp_accept_final", "resp_reach_final",
           "gap_mean_late", "gap_mean_final", "gap_w1_late",
           "gap_w1_final", "gap_mean_closure", "gap_w1_closure"]


def _num(v):
    return f"{v:g}".replace(".", "p")


def clamp_tag(arm, mode_tok, gate, seed):
    return (f"pofdclamp_mistral7b_{arm}_{mode_tok}_ea{_num(gate)}"
            f"_w0p5_l0p2_es0_s{seed}")


def mean_ci(vals):
    vals = [v for v in vals if v not in (NA, None)]
    if not vals:
        return NA, NA, NA
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, NA, NA
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var), T_CRIT[len(vals) - 1] * math.sqrt(
        var / len(vals))


def w1_quantile(a, b, grid=2048):
    """1-Wasserstein between empirical samples of DIFFERENT sizes via
    quantile interpolation on a shared probability grid."""
    qs = torch.linspace(0.0, 1.0, grid)
    return float((torch.quantile(a.float(), qs)
                  - torch.quantile(b.float(), qs)).abs().mean())


def cell_metrics(run_dir, mode_tok):
    d = AN.load(run_dir)
    op = d["op_raw"].float()
    pred = d["pred_raw"].float()
    tw = d["twin_raw"].float()
    gates = d["gate_raw"].bool()
    innate = d["innate"].float()
    mask = d["innate_clamp_mask"].bool()
    resp, froz = ~mask, mask
    n_r = op.shape[0]
    if n_r < LATE_HI:
        raise SystemExit(f"{run_dir}: {n_r} rounds < {LATE_HI}")

    def std_ratio(t):
        s_tw = float(tw[t][resp].std())
        return float(op[t][resp].std()) / s_tw if s_tw > 0 else None

    late = range(LATE_LO, LATE_HI)
    ratios = [r for r in (std_ratio(t) for t in late) if r is not None]
    out = {
        "resp_std_ratio_late": (sum(ratios) / len(ratios)
                                if ratios else NA),
        "resp_std_ratio_final": (std_ratio(n_r - 1)
                                 if std_ratio(n_r - 1) is not None
                                 else NA),
        "resp_pred_std_late": float(torch.stack(
            [pred[t][resp].std() for t in late]).mean()),
        "resp_pred_std_final": float(pred[-1][resp].std()),
        "resp_disp_twin_late": float(torch.stack(
            [(op[t][resp] - tw[t][resp]).abs().mean()
             for t in late]).mean()),
        "resp_disp_twin_final": float((op[-1][resp]
                                       - tw[-1][resp]).abs().mean()),
        "resp_accept_late": float(torch.stack(
            [gates[t][resp].float().mean() for t in late]).mean()),
        "resp_accept_final": float(gates[-1][resp].float().mean()),
        # cumulative reach: gated at least once by the final round
        "resp_reach_final": float(gates[:, resp].any(dim=0)
                                  .float().mean()),
        "n_frozen": int(mask.sum()), "n_responsive": int(resp.sum()),
    }
    for m in ("gap_mean_late", "gap_mean_final", "gap_w1_late",
              "gap_w1_final", "gap_mean_closure", "gap_w1_closure",
              "gap_mean_innate", "gap_w1_innate"):
        out[m] = NA
    if mode_tok == "bottom":
        # responsive-to-frozen separation -- meaningful only for the
        # bottom cohort, whose innate gap is positive by construction.
        # (Stratified cohorts START at ~zero gap; a closure ratio there
        # would divide by noise.)
        g0_mean = float(innate[resp].mean() - innate[froz].mean())
        g0_w1 = w1_quantile(innate[resp], innate[froz])
        gm_late = float(torch.stack(
            [op[t][resp].mean() - op[t][froz].mean()
             for t in late]).mean())
        gw_late = sum(w1_quantile(op[t][resp], op[t][froz])
                      for t in late) / len(list(late))
        out.update({
            "gap_mean_innate": g0_mean, "gap_w1_innate": g0_w1,
            "gap_mean_late": gm_late,
            "gap_mean_final": float(op[-1][resp].mean()
                                    - op[-1][froz].mean()),
            "gap_w1_late": gw_late,
            "gap_w1_final": w1_quantile(op[-1][resp], op[-1][froz]),
            "gap_mean_closure": (1.0 - gm_late / g0_mean
                                 if g0_mean != 0 else NA),
            "gap_w1_closure": (1.0 - gw_late / g0_w1
                               if g0_w1 != 0 else NA)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "notes", "pofd", "clamp_analysis"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    per_seed, missing = [], []
    for tok in MODES:
        for arm in ARMS:
            for gate in GATES:
                for seed in SEEDS:
                    tag = clamp_tag(arm, tok, gate, seed)
                    rd = AN.find_run(args.roots, tag)
                    if rd is None:
                        missing.append(tag)
                        continue
                    per_seed.append({
                        "clamp_mode": MODES[tok], "arm": arm,
                        "gate": gate, "seed": seed, "run_tag": tag,
                        **cell_metrics(rd, tok)})
    print(f"[clamp] cells located: {len(per_seed)}/"
          f"{len(MODES) * len(ARMS) * len(GATES) * len(SEEDS)}")
    for tag in missing:
        print(f"  MISSING {tag}")

    summary = []
    for tok in MODES:
        for arm in ARMS:
            for gate in GATES:
                rows = [r for r in per_seed
                        if r["clamp_mode"] == MODES[tok]
                        and r["arm"] == arm and r["gate"] == gate]
                out = {"clamp_mode": MODES[tok], "arm": arm,
                       "gate": gate, "n_seeds": len(rows)}
                for m in METRICS:
                    mu, sd, ci = mean_ci([r[m] for r in rows])
                    out[f"{m}_mean"] = mu
                    out[f"{m}_sd"] = sd
                    out[f"{m}_ci95"] = ci
                summary.append(out)

    def write(name, rows):
        if not rows:
            print(f"[clamp] {name}: no rows")
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out_dir, name), "w",
                  newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys, restval=NA)
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"[clamp] wrote {name} ({len(rows)} rows)")

    write("clamp_per_seed.csv", per_seed)
    write("clamp_summary.csv", summary)

    print("\n== responsive std ratio vs twin, rounds 25-29 "
          "(3-seed mean +- 95% t) ==")
    for s in summary:
        mu = s["resp_std_ratio_late_mean"]
        if mu == NA:
            continue
        ci = s["resp_std_ratio_late_ci95"]
        ci_s = f"+-{ci:.3f}" if ci != NA else "(n<2)"
        acc = s["resp_accept_late_mean"]
        acc_s = f"{acc:.3f}" if acc != NA else NA
        line = (f"  {s['clamp_mode']:17s} {s['arm']:3s} "
                f"ea={s['gate']:<4g} ratio {mu:.3f}{ci_s}  "
                f"accept {acc_s}")
        if s["gap_mean_closure_mean"] != NA:
            line += f"  gap-closure {s['gap_mean_closure_mean']:+.3f}"
        print(line)


if __name__ == "__main__":
    main()
