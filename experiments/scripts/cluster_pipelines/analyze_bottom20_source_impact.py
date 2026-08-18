#!/usr/bin/env python3
"""BOTTOM-20% SOURCE-IMPACT analysis (2026-08-18,
mistral_bottom20_source_impact).

Cohort A = the 145 agents with the LOWEST innate Action opinions
(deterministic innate-then-id ranking), pinned bit-exact at innate in
population and matched twin; cohort B = the other 578. es=0 (no peer
operator), so the ONLY route from A to B is the platform: three arms
separate its channels at each AI gate ea in {0.1, 0.2, 0.4, 1}:
  b0    ordinary SFT, A's innate labels in every weight update
  b0xa  identical SFT with A excluded (volume-matched 723 rows)
  d8    frozen weights, personal-history ICL (structural null)

PRIMARY OUTCOME -- signed movement of B toward the low cohort:
  T_a = mu_B^0 - mu_{B,a}^late     (late = rounds 25-29)
positive = B moved DOWN toward A. PRIMARY CAUSAL CONTRAST:
  T_b0 - T_b0xa
positive = including A's labels in SFT pulled B farther toward A
through shared weights.

Also reported per arm: initial/late A-B mean gaps + normalized
closure, quantile-W1(A, B) + normalized W1 closure, responsive SD
ratio vs the matched twin (secondary); and paired late b0-vs-b0xa
opinion/prediction differences on B. Seeds {0, 42, 43} are
replicates: three-seed means with 95% Student-t intervals
(t_{.975, df=2} = 4.3027). All 36 conceptual cells are HARD-REQUIRED
and every run must share the bit-identical A/B partition and innate
population.

Outputs (notes/pofd/bottom20_impact_analysis/):
  bottom20_per_cell.csv   one row per (arm, gate, seed)
  bottom20_summary.csv    per (arm, gate): mean + 95% t-CI
  bottom20_contrast.csv   per gate: T_b0 - T_b0xa and the paired
                          b0-vs-b0xa opinion/prediction differences
  bottom20_impact_panels.png/pdf   signed movement toward A and the
                          late A-B mean gap across gates, three arms
"""
import argparse
import csv
import importlib.util
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "analyze_reach", os.path.join(HERE, "analyze_sft_icl_reach.py"))
AN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN)
_spec_c = importlib.util.spec_from_file_location(
    "analyze_clamp", os.path.join(HERE, "analyze_innate_clamp.py"))
AC = importlib.util.module_from_spec(_spec_c)
_spec_c.loader.exec_module(AC)

ARMS = ["b0", "b0xa", "d8"]
GATES = [0.1, 0.2, 0.4, 1.0]
SEEDS = [0, 42, 43]
LATE = range(25, 30)
T975_DF2 = 4.302652729911275
OUT_DIR_DEFAULT = os.path.join(
    REPO, "notes", "pofd", "bottom20_impact_analysis")


def _num(v):
    return f"{v:g}".replace(".", "p")


def cell_tag(arm, gate, seed):
    return (f"pofdclamp_mistral7b_{arm}_bottom_ea{_num(gate)}"
            f"_w0p5_l0p2_es0_s{seed}")


def t_ci(vals):
    """(mean, lo, hi): three-seed mean with the 95% Student-t
    interval (df = n-1)."""
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    h = T975_DF2 * sd / math.sqrt(n)
    return m, m - h, m + h


def cell_metrics(d, mask):
    """Per-run outcomes on cohort B (rounds 25-29)."""
    op = d["op_raw"].float()
    tw = d["twin_raw"].float()
    innate = d["innate"].float()
    b, a = ~mask, mask
    mu_b0 = float(innate[b].mean())
    mu_b_late = float(torch.stack(
        [op[t][b].mean() for t in LATE]).mean())
    gap0 = float(innate[b].mean() - innate[a].mean())
    gap_late = float(torch.stack(
        [op[t][b].mean() - op[t][a].mean() for t in LATE]).mean())
    w1_0 = AC.w1_quantile(innate[b], innate[a])
    w1_late = sum(AC.w1_quantile(op[t][b], op[t][a])
                  for t in LATE) / len(list(LATE))
    ratios = []
    for t in LATE:
        s_tw = float(tw[t][b].std())
        if s_tw > 0:
            ratios.append(float(op[t][b].std()) / s_tw)
    return {
        "t_move": mu_b0 - mu_b_late,
        "gap0_mean": gap0, "gap_late_mean": gap_late,
        "closure_mean": (1.0 - gap_late / gap0
                         if abs(gap0) > 1e-9 else float("nan")),
        "w1_0": w1_0, "w1_late": w1_late,
        "closure_w1": (1.0 - w1_late / w1_0
                       if w1_0 > 1e-9 else float("nan")),
        "sd_ratio_late": (sum(ratios) / len(ratios)
                          if ratios else float("nan")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    run_of, missing = {}, []
    for arm in ARMS:
        for gate in GATES:
            for seed in SEEDS:
                tag = cell_tag(arm, gate, seed)
                rd = AN.find_run(args.roots, tag)
                if rd is None:
                    missing.append(tag)
                else:
                    run_of[(arm, gate, seed)] = rd
    n_total = len(ARMS) * len(GATES) * len(SEEDS)
    print(f"[bottom20] cells located: {len(run_of)}/{n_total}")
    for tag in missing:
        print(f"  MISSING {tag}")
    if missing:
        print(f"[bottom20] HARD FAIL: {len(missing)} of {n_total} "
              f"conceptual cells missing -- no output written",
              file=sys.stderr)
        sys.exit(1)

    # shared-partition guarantee: every run must carry the identical
    # bottom-145 mask and innate population, and the mask must BE the
    # deterministic bottom ranking (innate, then agent id)
    loads = {k: AN.load(rd) for k, rd in run_of.items()}
    ref = loads[("b0", GATES[0], SEEDS[0])]
    mask = ref["innate_clamp_mask"].bool()
    innate = ref["innate"].float()
    order = sorted(range(innate.numel()),
                   key=lambda i: (float(innate[i]), i))
    want_mask = torch.zeros(innate.numel(), dtype=torch.bool)
    want_mask[torch.tensor(order[:int(mask.sum())])] = True
    if not torch.equal(mask, want_mask):
        print("[bottom20] HARD FAIL: stored mask is not the "
              "deterministic bottom-145 ranking", file=sys.stderr)
        sys.exit(1)
    if int(mask.sum()) != 145 or int((~mask).sum()) != 578:
        print(f"[bottom20] HARD FAIL: partition {int(mask.sum())}/"
              f"{int((~mask).sum())} != 145/578", file=sys.stderr)
        sys.exit(1)
    for k, d in loads.items():
        if not torch.equal(d["innate_clamp_mask"].bool(), mask) or \
                not torch.equal(d["innate"], ref["innate"]):
            print(f"[bottom20] HARD FAIL: {cell_tag(*k)} does not "
                  f"share the A/B partition / innate population",
                  file=sys.stderr)
            sys.exit(1)

    per_cell = []
    for (arm, gate, seed), d in sorted(loads.items()):
        per_cell.append({"arm": arm, "gate": gate, "seed": seed,
                         "run_tag": cell_tag(arm, gate, seed),
                         **cell_metrics(d, mask)})

    def cell(arm, gate, seed):
        return [r for r in per_cell if r["arm"] == arm
                and r["gate"] == gate and r["seed"] == seed][0]

    metrics = ["t_move", "gap_late_mean", "closure_mean", "w1_late",
               "closure_w1", "sd_ratio_late"]
    summary = []
    for arm in ARMS:
        for gate in GATES:
            row = {"arm": arm, "gate": gate,
                   "gap0_mean": cell(arm, gate, 0)["gap0_mean"],
                   "w1_0": cell(arm, gate, 0)["w1_0"]}
            for m in metrics:
                vals = [cell(arm, gate, s)[m] for s in SEEDS]
                mu, lo, hi = t_ci(vals)
                row[f"{m}_mean"], row[f"{m}_lo"], row[f"{m}_hi"] = \
                    mu, lo, hi
            summary.append(row)

    # primary causal contrast + paired late b0-vs-b0xa differences on
    # B (paired by seed; MAE = per-agent, dmean = signed mean shift
    # b0 minus b0xa)
    b = ~mask
    contrast = []
    for gate in GATES:
        d_t, op_mae, op_dm, pr_mae, pr_dm = [], [], [], [], []
        for seed in SEEDS:
            d_t.append(cell("b0", gate, seed)["t_move"]
                       - cell("b0xa", gate, seed)["t_move"])
            da = loads[("b0", gate, seed)]
            db = loads[("b0xa", gate, seed)]
            oa, ob = da["op_raw"].float(), db["op_raw"].float()
            pa = da["pred_raw"].float().clamp(0.0, 1.0)
            pb = db["pred_raw"].float().clamp(0.0, 1.0)
            op_mae.append(float(torch.stack(
                [(oa[t][b] - ob[t][b]).abs().mean()
                 for t in LATE]).mean()))
            op_dm.append(float(torch.stack(
                [(oa[t][b] - ob[t][b]).mean() for t in LATE]).mean()))
            pr_mae.append(float(torch.stack(
                [(pa[t][b] - pb[t][b]).abs().mean()
                 for t in LATE]).mean()))
            pr_dm.append(float(torch.stack(
                [(pa[t][b] - pb[t][b]).mean() for t in LATE]).mean()))
        row = {"gate": gate}
        for key, vals in (("d_t_b0_minus_b0xa", d_t),
                          ("op_mae_late", op_mae),
                          ("op_dmean_late", op_dm),
                          ("pred_mae_late", pr_mae),
                          ("pred_dmean_late", pr_dm)):
            mu, lo, hi = t_ci(vals)
            row[f"{key}_mean"], row[f"{key}_lo"], row[f"{key}_hi"] = \
                mu, lo, hi
        contrast.append(row)

    os.makedirs(args.out_dir, exist_ok=True)

    def write(name, rows):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out_dir, name), "w",
                  newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys)
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"[bottom20] wrote {name} ({len(rows)} rows)")

    write("bottom20_per_cell.csv", per_cell)
    write("bottom20_summary.csv", summary)
    write("bottom20_contrast.csv", contrast)

    def srow(arm, gate):
        return [r for r in summary if r["arm"] == arm
                and r["gate"] == gate][0]

    print("\n== signed movement of B toward A, T_a = mu_B^0 - "
          "mu_B^late (3-seed mean [95% t-CI]; cols = ea "
          + "/".join(f"{g:g}" for g in GATES) + ") ==")
    for arm in ARMS:
        cells = [srow(arm, g) for g in GATES]
        print(f"  {arm:<4}: " + "  ".join(
            f"{r['t_move_mean']:+.4f} [{r['t_move_lo']:+.4f},"
            f"{r['t_move_hi']:+.4f}]" for r in cells))
    print("\n== PRIMARY CONTRAST T_b0 - T_b0xa (positive = A's "
          "labels pulled B toward A through shared weights) ==")
    for r in contrast:
        print(f"  ea{r['gate']:<4g}: {r['d_t_b0_minus_b0xa_mean']:+.4f} "
              f"[{r['d_t_b0_minus_b0xa_lo']:+.4f},"
              f"{r['d_t_b0_minus_b0xa_hi']:+.4f}]")

    if not args.no_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        color = {"b0": "tab:red", "b0xa": "tab:blue",
                 "d8": "tab:green"}
        label = {"b0": "SFT, A included (b0)",
                 "b0xa": "SFT, A excluded (b0xa)",
                 "d8": "personal-history ICL (d8)"}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
        for arm in ARMS:
            rows = [srow(arm, g) for g in GATES]
            for ax, m in ((ax1, "t_move"), (ax2, "gap_late_mean")):
                mu = [r[f"{m}_mean"] for r in rows]
                lo = [r[f"{m}_lo"] for r in rows]
                hi = [r[f"{m}_hi"] for r in rows]
                ax.errorbar(GATES, mu,
                            yerr=[[m_ - l for m_, l in zip(mu, lo)],
                                  [h - m_ for m_, h in zip(mu, hi)]],
                            color=color[arm], marker="o", ms=4,
                            capsize=3, lw=1.5, label=label[arm])
        ax1.axhline(0.0, color="0.6", lw=0.8, ls=":")
        ax1.set_xlabel(r"$\varepsilon_{\mathrm{AI}}$")
        ax1.set_ylabel(r"movement toward $A$:"
                       r" $\mu_B^0-\mu_B^{\mathrm{late}}$")
        gap0 = summary[0]["gap0_mean"]
        ax2.axhline(gap0, color="0.6", lw=0.8, ls="--",
                    label="initial gap")
        ax2.set_xlabel(r"$\varepsilon_{\mathrm{AI}}$")
        ax2.set_ylabel(r"late $A$-$B$ mean gap")
        for ax in (ax1, ax2):
            ax.set_xscale("log")
            ax.set_xticks(GATES)
            ax.set_xticklabels([f"{g:g}" for g in GATES])
        ax1.legend(frameon=False, fontsize=8)
        ax2.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(
                args.out_dir, f"bottom20_impact_panels.{ext}"),
                dpi=200 if ext == "png" else None)
        print("[bottom20] wrote bottom20_impact_panels.png/pdf")


if __name__ == "__main__":
    main()
