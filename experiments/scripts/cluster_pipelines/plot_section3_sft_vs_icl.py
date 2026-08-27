#!/usr/bin/env python3
"""Compact PAIRED comparison: reference-regularized SFT vs frozen
personal-history ICL, per model, at the Section-3 open-gate surface.

Both arms sit on the SAME surface -- MovieLens/Action, 723 agents, 30
rounds, S=100 Deffuant sweeps, W=1, k=1, alpha=.5, both gates all_open,
corrected anch2 operator, seeds {0,42,43} -- and differ ONLY in the
adaptation channel, which is what makes a paired plot meaningful:

  SFT  reference-regularized, fresh r512 LoRA each round, forward KL
       lambda=2                          (wave section3_model_equilibria)
  ICL  frozen weights, ICL_K=0, ICL_DAYS=8: each agent sees only its own
       last eight post-peer opinions     (wave section3_model_icl)

THE PERFECT-PREDICTION REFERENCE IS RETAINED. At W=1 with both gates
open and mean-preserving alpha=.5 sweeps, a platform that predicts
perfectly serves each agent its own current opinion, so the population
mean never moves: the innate mean IS the perfect-prediction equilibrium,
common to both arms and requiring no GPU run. It is drawn once, as the
shared dashed reference both arms are read against.

Consumes the two analyses' outputs (model_equilibria.csv + summary.json
from each) -- it recomputes nothing, so the numbers here are by
construction the numbers in those CSVs.

Figures carry NO title (house rule); the caption block is written beside
the PDF.

  python plot_section3_sft_vs_icl.py \
      --sft-dir notes/pofd/section3_model_equilibria \
      --icl-dir notes/pofd/section3_model_icl \
      --out-dir notes/pofd/section3_sft_vs_icl
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "perfsim-s3-pair"))

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent

ORDER = ("qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b",
         "mistral7b", "ministral8b")
DISPLAY = {
    "qwen7b": "Qwen\n2.5", "qwen3_8b": "Qwen\n3",
    "olmo7b": "OLMo\n2", "olmo3_7b": "OLMo\n3",
    "mistral7b": "Mistral", "ministral8b": "Ministral",
}
SFT_C = "#4c72b0"      # the published Figure-3(a) blue
ICL_C = "#c44e52"
REFERENCE = "#696d73"
INK = "#202328"


def load_arm(d: Path, label: str):
    rows_p, sum_p = d / "model_equilibria.csv", d / "summary.json"
    if not rows_p.exists() or not sum_p.exists():
        sys.exit(f"[s3pair] {label}: analysis outputs absent under {d} -- "
                 f"run analyze_section3_model_equilibria.py first")
    with rows_p.open() as fh:
        rows = {r["model"]: r for r in csv.DictReader(fh)}
    if set(rows) != set(ORDER):
        sys.exit(f"[s3pair] {label}: expected exactly the six models, got "
                 f"{sorted(rows)}")
    summary = json.loads(sum_p.read_text())
    return rows, summary


def col(rows, model, *names):
    """First present numeric column among names (the analyzer's scalar
    names are stable, but be explicit rather than guess)."""
    r = rows[model]
    for n in names:
        if n in r and r[n] not in ("", None):
            return float(r[n])
    sys.exit(f"[s3pair] none of {names} in the analysis row for {model}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sft-dir",
                    default=str(REPO / "notes" / "pofd"
                               / "section3_model_equilibria"))
    ap.add_argument("--icl-dir",
                    default=str(REPO / "notes" / "pofd"
                               / "section3_model_icl"))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    sft_rows, sft_sum = load_arm(Path(args.sft_dir), "SFT")
    icl_rows, icl_sum = load_arm(Path(args.icl_dir), "ICL")
    out = Path(args.out_dir) if args.out_dir else (
        REPO / "notes" / "pofd" / "section3_sft_vs_icl")
    out.mkdir(parents=True, exist_ok=True)

    # THE SHARED REFERENCE. Both waves sit on one innate vector, so the
    # perfect-prediction mean must agree; if it does not, the two arms
    # are not the same population and the pairing is void.
    p_sft = float(sft_sum["perfect_prediction_mean"])
    p_icl = float(icl_sum["perfect_prediction_mean"])
    if abs(p_sft - p_icl) > 1e-9:
        sys.exit(f"[s3pair] REFUSING: the two arms report different "
                 f"perfect-prediction means ({p_sft:.8f} vs {p_icl:.8f}) -- "
                 f"they are not the same population, so they cannot be "
                 f"paired")
    perfect = p_sft

    recs = []
    for m in ORDER:
        s_mu = col(sft_rows, m, "equilibrium_mean")
        i_mu = col(icl_rows, m, "equilibrium_mean")
        rec = {"model": m, "sft_mean": s_mu, "icl_mean": i_mu,
               "perfect_prediction": perfect,
               "sft_minus_perfect": s_mu - perfect,
               "icl_minus_perfect": i_mu - perfect,
               "icl_minus_sft": i_mu - s_mu}
        for arm, rows in (("sft", sft_rows), ("icl", icl_rows)):
            for k_out, names in (("ci_lo", ("ci95_low",)),
                                 ("ci_hi", ("ci95_high",))):
                try:
                    rec[f"{arm}_{k_out}"] = col(rows, m, *names)
                except SystemExit:
                    rec[f"{arm}_{k_out}"] = float("nan")
        recs.append(rec)

    csv_p = out / "section3_sft_vs_icl.csv"
    with csv_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print(f"[s3pair] wrote {csv_p} ({len(recs)} models)")

    # ---------------------------------------------------------- figure
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": .8,
                         "xtick.major.width": .8, "ytick.major.width": .8,
                         "text.color": INK, "axes.labelcolor": INK})
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    x = np.arange(len(ORDER), dtype=float)
    off = 0.16

    ax.axhline(perfect, color=REFERENCE, lw=1.35, ls=(0, (4, 2.5)),
               zorder=1)
    for i, r in enumerate(recs):
        # the paired stem: how far each channel sits from the other
        ax.plot([x[i] - off, x[i] + off], [r["sft_mean"], r["icl_mean"]],
                color="0.72", lw=1.0, zorder=2)
    for i, r in enumerate(recs):
        for dx, key, c in ((-off, "sft_mean", SFT_C),
                           (off, "icl_mean", ICL_C)):
            arm = "sft" if key.startswith("sft") else "icl"
            lo, hi = r.get(f"{arm}_ci_lo"), r.get(f"{arm}_ci_hi")
            if lo is not None and hi is not None and math.isfinite(lo) \
                    and math.isfinite(hi):
                ax.plot([x[i] + dx, x[i] + dx], [lo, hi], color=c, lw=1.3,
                        solid_capstyle="round", zorder=3)
            ax.plot([x[i] + dx], [r[key]], "o", ms=4.6, color=c,
                    mec="white", mew=.7, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in ORDER])
    ax.set_ylabel("Equilibrium", labelpad=4)
    ax.set_xlim(-.6, len(ORDER) - .4)
    vals = [v for r in recs for v in (r["sft_mean"], r["icl_mean"])]
    lo_, hi_ = min(vals + [perfect]), max(vals + [perfect])
    pad = max(.02, .12 * (hi_ - lo_ if hi_ > lo_ else .1))
    ax.set_ylim(lo_ - pad, hi_ + pad)
    ax.text(-.52, perfect + .006, "perfect prediction", fontsize=7,
            color=REFERENCE, va="bottom")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], color=SFT_C, marker="o", ms=4.6, lw=1.3,
               label="reference-regularized SFT ($\\lambda=2$)"),
        Line2D([], [], color=ICL_C, marker="o", ms=4.6, lw=1.3,
               label="personal-history ICL (frozen, $D=8$)"),
    ], frameon=False, fontsize=7, loc="best", handlelength=1.6)
    fig.tight_layout()
    pdf, png = out / "section3_sft_vs_icl.pdf", out / "section3_sft_vs_icl.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"[s3pair] wrote {pdf} and {png}")

    cap = [
        "Paired adaptation-channel comparison at the Section-3 open-gate",
        "surface. Each model contributes two points: reference-regularized",
        "SFT (fresh r512 LoRA each round, forward KL lambda=2) and frozen",
        "personal-history ICL (ICL_K=0, ICL_DAYS=8 -- each agent sees only",
        "its own last eight post-peer opinions). Bars are across-seed 95%",
        "Student-t intervals over seeds {0,42,43}; the grey line connects",
        "the two channels for one model. Both arms share every other",
        f"setting (MovieLens/Action, 723 agents, 30 rounds, S=100 sweeps,",
        "W=1, k=1, alpha=.5, both gates all_open, anch2 operator), so the",
        "vertical distance between a pair is attributable to WHERE",
        "adaptation is stored. The dashed line is perfect prediction --",
        f"the common innate mean ({perfect:.4f}), which needs no GPU run",
        "and is shared by both arms.",
    ]
    (out / "section3_sft_vs_icl_caption.txt").write_text("\n".join(cap) + "\n")
    (out / "section3_sft_vs_icl_summary.json").write_text(json.dumps({
        "perfect_prediction_mean": perfect,
        "sft_wave": sft_sum.get("wave_key", sft_sum.get("wave")),
        "icl_wave": icl_sum.get("wave_key", icl_sum.get("wave")),
        "models": recs}, indent=2))
    print(f"[s3pair] wrote the caption and summary")

    hdr = (f"{'model':<12} {'SFT':>9} {'ICL':>9} {'ICL-SFT':>9} "
           f"{'SFT-perf':>9} {'ICL-perf':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in recs:
        print(f"{r['model']:<12} {r['sft_mean']:>9.4f} {r['icl_mean']:>9.4f} "
              f"{r['icl_minus_sft']:>+9.4f} {r['sft_minus_perfect']:>+9.4f} "
              f"{r['icl_minus_perfect']:>+9.4f}")
    print(f"\nperfect prediction (shared innate mean) = {perfect:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
