#!/usr/bin/env python3
"""Plot the compact cross-model open-gate equilibrium comparison.

One point and seed-level 95% interval per regularized pretrained model;
one dashed perfect-prediction reference.  There is deliberately no lambda=0
series: the separate regularization ladder supplies that comparison.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "perfsim-s3-model-eq"))

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
DEFAULT_IN = REPO / "notes" / "pofd" / "section3_model_equilibria"

ORDER = ("qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b",
         "mistral7b", "ministral8b")
X = np.asarray([0.0, 1.0, 2.35, 3.35, 4.7, 5.7])
LABEL = {
    "qwen7b": "Qwen\n2.5",
    "qwen3_8b": "Qwen\n3",
    "olmo7b": "OLMo\n2",
    "olmo3_7b": "OLMo\n3",
    "mistral7b": "Mistral",
    "ministral8b": "Ministral",
}
BLUE = "#4c72b0"
INK = "#202328"
REFERENCE = "#696d73"


def main():
    ap = argparse.ArgumentParser(
        description="plot six regularized model-specific consensus means")
    ap.add_argument("--in-dir", default=str(DEFAULT_IN))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = in_dir / "model_equilibria.csv"
    summary_path = in_dir / "summary.json"
    if not rows_path.exists() or not summary_path.exists():
        ap.error("analysis outputs absent; run analyze_section3_model_equilibria.py")
    with rows_path.open() as fh:
        rows = {r["model"]: r for r in csv.DictReader(fh)}
    if set(rows) != set(ORDER):
        ap.error(f"expected exactly six models; got {sorted(rows)}")
    if not all(r["all_converged"] == "True" and
               r.get("any_cyclic", "False") == "False" and
               r["all_consensus"] == "True" for r in rows.values()):
        ap.error("at least one model contains an unsettled, cyclic or "
                 "non-consensus seed")
    summary = json.loads(summary_path.read_text())
    perfect = float(summary["perfect_prediction_mean"])

    y = np.asarray([float(rows[m]["equilibrium_mean"]) for m in ORDER])
    lo = np.asarray([float(rows[m]["ci95_low"]) for m in ORDER])
    hi = np.asarray([float(rows[m]["ci95_high"]) for m in ORDER])
    yerr = np.vstack((y - lo, hi - y))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.0,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": INK,
        "axes.labelcolor": INK,
    })

    fig, ax = plt.subplots(figsize=(6.25, 2.55))
    fig.subplots_adjust(left=.105, right=.99, top=.80, bottom=.23)
    ax.axhline(perfect, color=REFERENCE, lw=1.35, ls=(0, (4, 2.5)), zorder=1)
    # Light displacement stems make the claim legible without adding an
    # unregularized series: every stem starts at perfect prediction.
    for xi, yi in zip(X, y):
        ax.plot([xi, xi], [perfect, yi], color=BLUE, alpha=.23,
                lw=1.2, zorder=1)
    ax.errorbar(X, y, yerr=yerr, fmt="o", color=BLUE, ecolor=BLUE,
                markersize=6.2, markeredgewidth=.7, capsize=2.5,
                elinewidth=1.25, zorder=3)

    ax.set_xticks(X)
    ax.set_xticklabels([LABEL[m] for m in ORDER], fontsize=9.1)
    ax.set_ylabel("Mean post-peer opinion\n(rounds 21\u201330)",
                  fontsize=10.0, labelpad=3)
    low = min(float(lo.min()), perfect)
    high = max(float(hi.max()), perfect)
    pad = max(.025, .10 * (high - low))
    ax.set_ylim(max(0.0, low - pad), min(1.0, high + pad))
    ax.set_xlim(-.45, X[-1] + .45)
    ax.grid(axis="y", color="#d9dde2", lw=.6, alpha=.7)
    ax.tick_params(axis="both", labelsize=9.0, length=2.4,
                   width=.65, pad=1.8)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([], [], marker="o", color=BLUE, lw=0, markersize=6.2,
               label="reference-regularized SFT ($\\lambda=2$)"),
        Line2D([], [], color=REFERENCE, lw=1.35, ls=(0, (4, 2.5)),
               label="perfect prediction"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .99),
               ncol=2, frameon=False,
               prop={"size": 9.0, "weight": "bold"},
               handlelength=2.0, columnspacing=1.4)

    stem = out_dir / "model_specific_open_gate_equilibria"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=.025)
    fig.savefig(f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)
    print(f"[plot_s3m] wrote {stem}.pdf / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
