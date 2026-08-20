#!/usr/bin/env python3
"""LOCK-IN RATE of feature endogenization at lambda=1 (2026-08-20,
feature_lambda1_seeds).

Across the first five seeds the natural lambda=1 cell is BIMODAL
rather than noisy-around-a-mean:

  seed  peak    round  late(25-29)  round-29 gender gap
  0     0.071     9      0.005          +0.006
  42    0.056    15      0.014          +0.009
  43    0.038    23      0.010          +0.003
  44    0.010     5      0.000          +0.000
  45    0.119    27      0.105          +0.067

Four seeds show a TRANSIENT spike at an arbitrary round that decays
back to the noise floor; one locks in late and stays there. Averaging
the curves smears four randomly-timed spikes plus one plateau into a
low broad bump, which is why the mean curve understates what is
actually happening and why its interval contains zero.

The quantity worth estimating is therefore the RATE at which a run
locks in, not the height of the averaged curve. This script pools
every available lambda=1 seed and reports that rate with an exact
(Clopper-Pearson) binomial interval.

LOCK-IN CRITERION -- PRE-SPECIFIED, and fixed before any seed beyond
45 existed:

    late-window (rounds 25-29) mean incremental R^2 of gender > 0.02

Rationale, from the five seeds above: the non-locking runs span
0.000-0.014 and the locking run sits at 0.105, a gap of ~7x. 0.02
sits inside that gap, above the frozen control's persistent level
(~0.013) and far above the lambda=0 noise floor (|.| < 0.001).
Anything in roughly [0.015, 0.09] gives the same classification on
the existing seeds; the threshold is reported alongside the rate and
a sensitivity sweep is printed so the reader can see that.

Outputs (notes/pofd/feature_lambda1_seeds_analysis/):
  lambda1_per_seed.csv   one row per seed: peak, peak round, late
                         mean, round-29 gender gap, lock-in flag
  lambda1_rate.csv       the rate, its interval, the threshold and
                         the sensitivity sweep
  lambda1_seeds.png/pdf  every seed's trajectory (locking runs
                         highlighted) + the late-window summary
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LLM = os.path.join(REPO, "experiments", "llm")
sys.path.insert(0, LLM)

LATE = list(range(25, 30))
LOCKIN_THRESHOLD = 0.02          # pre-specified; see the docstring
SENSITIVITY = [0.015, 0.02, 0.03, 0.05, 0.09]
OUT_DIR_DEFAULT = os.path.join(
    REPO, "notes", "pofd", "feature_lambda1_seeds_analysis")


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval; no scipy dependency."""
    if n == 0:
        return (float("nan"), float("nan"))

    def betainc_inv(p, a, b):
        # bisection on the regularised incomplete beta via numpy only
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _betainc(a, b, mid) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _betainc(a, b, x, n_steps=20000):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        t = np.linspace(0.0, x, n_steps)
        y = t ** (a - 1) * (1 - t) ** (b - 1)
        num = np.trapz(y, t)
        t2 = np.linspace(0.0, 1.0, n_steps)
        y2 = t2 ** (a - 1) * (1 - t2) ** (b - 1)
        return float(num / np.trapz(y2, t2))

    lo = 0.0 if k == 0 else betainc_inv(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else betainc_inv(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    import plot_feature_endogenization_main as M
    from plot_feature_endogenization_beta_final import (
        RUN_ROOT, TASTE_COLUMNS, load_run, natural_tag)

    ref = load_run(natural_tag(1.0, 0))["trajectory"]
    tastes = np.column_stack(
        [np.asarray(ref["profiles"][k], dtype=float)
         for k in TASTE_COLUMNS])
    gender = np.asarray(ref["profiles"]["gender"]) == "M"

    # every lambda=1 seed present on disk -- the original five plus
    # whatever replication has landed
    seeds = []
    for s in [0, 42, 43, 44, 45] + list(range(46, 100)):
        if (RUN_ROOT / natural_tag(1.0, s) / "trajectory.pt").exists():
            seeds.append(s)
    print(f"[l1] lambda=1 seeds available: {len(seeds)} -> {seeds}")
    if len(seeds) < 5:
        print("[l1] HARD FAIL: fewer than the five known seeds found",
              file=sys.stderr)
        sys.exit(1)

    rows, series = [], {}
    for s in seeds:
        run = load_run(natural_tag(1.0, s))
        v = M.series_for_run(run, tastes, gender)
        series[s] = v
        op = run["trajectory"]["op_raw"].float().numpy()
        g = np.asarray(run["trajectory"]["profiles"]["gender"]) == "M"
        late = float(np.mean(v[LATE]))
        rows.append({
            "seed": s, "peak": float(v.max()),
            "peak_round": int(v.argmax()),
            "late_mean_r2": late,
            "final_r2": float(v[-1]),
            "gender_gap_r29": float(op[29][g].mean()
                                    - op[29][~g].mean()),
            "locked_in": late > LOCKIN_THRESHOLD})
        print(f"  seed {s:<3} peak={v.max():+.4f} @r{int(v.argmax()):<3}"
              f" late={late:+.4f} gap={rows[-1]['gender_gap_r29']:+.4f}"
              f" {'LOCK-IN' if rows[-1]['locked_in'] else ''}")

    k = sum(r["locked_in"] for r in rows)
    n = len(rows)
    lo, hi = clopper_pearson(k, n)
    print(f"\n[l1] lock-in rate: {k}/{n} = {k / n:.1%}  "
          f"95% CI [{lo:.1%}, {hi:.1%}]  "
          f"(threshold late-window R^2 > {LOCKIN_THRESHOLD})")

    sens = []
    print("[l1] sensitivity to the threshold:")
    for th in SENSITIVITY:
        kk = sum(1 for r in rows if r["late_mean_r2"] > th)
        l2, h2 = clopper_pearson(kk, n)
        sens.append({"threshold": th, "k": kk, "n": n,
                     "rate": kk / n, "ci_lo": l2, "ci_hi": h2})
        print(f"    > {th:<6}: {kk}/{n} = {kk / n:>6.1%}  "
              f"[{l2:.1%}, {h2:.1%}]")

    os.makedirs(args.out_dir, exist_ok=True)

    def write(name, data):
        keys = []
        for r in data:
            for kk2 in r:
                if kk2 not in keys:
                    keys.append(kk2)
        with open(os.path.join(args.out_dir, name), "w",
                  newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
        print(f"[l1] wrote {name} ({len(data)} rows)")

    write("lambda1_per_seed.csv", rows)
    write("lambda1_rate.csv",
          [{"threshold_prespecified": LOCKIN_THRESHOLD,
            "k": k, "n": n, "rate": k / n,
            "ci_lo": lo, "ci_hi": hi}] + sens)

    if args.no_fig:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.4, 4.0),
        gridspec_kw={"width_ratios": (2.0, 1.0)})
    rounds = np.arange(30)
    for r in rows:
        s = r["seed"]
        locked = r["locked_in"]
        ax.plot(rounds, series[s],
                color="#0072B2" if locked else "#999999",
                linewidth=1.6 if locked else 0.9,
                alpha=1.0 if locked else 0.75,
                zorder=3 if locked else 2,
                label=("lock-in" if locked else "no lock-in")
                if r is next(x for x in rows
                             if x["locked_in"] == locked) else None)
    ax.axhline(LOCKIN_THRESHOLD, color="#D55E00", linestyle=":",
               linewidth=1.0)
    ax.text(0.4, LOCKIN_THRESHOLD, " lock-in threshold", fontsize=7,
            color="#D55E00", va="bottom")
    ax.set_xlabel("round")
    ax.set_ylabel(r"Incremental $R^2$ of gender")
    ax.set_xlim(0, 29)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    late_vals = [r["late_mean_r2"] for r in rows]
    ax2.scatter([0.04 * (i % 5) for i in range(n)], late_vals,
                c=["#0072B2" if r["locked_in"] else "#999999"
                   for r in rows], s=26, zorder=3)
    ax2.axhline(LOCKIN_THRESHOLD, color="#D55E00", linestyle=":",
                linewidth=1.0)
    ax2.set_xticks([])
    ax2.set_ylabel(r"late-window mean $R^2$ (r25-29)")
    ax2.set_title(f"lock-in {k}/{n} = {k / n:.0%}", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out_dir,
                                 f"lambda1_seeds.{ext}"),
                    dpi=220 if ext == "png" else None)
    plt.close(fig)
    print("[l1] wrote lambda1_seeds.png/pdf")


if __name__ == "__main__":
    main()
