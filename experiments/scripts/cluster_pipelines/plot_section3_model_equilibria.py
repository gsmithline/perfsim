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
import math
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
FAMILY = {
    "qwen7b": "qwen", "qwen3_8b": "qwen",
    "olmo7b": "olmo", "olmo3_7b": "olmo",
    "mistral7b": "mistral", "ministral8b": "mistral",
}
BLUE = "#4c72b0"
INK = "#202328"
REFERENCE = "#696d73"


def main():
    ap = argparse.ArgumentParser(
        description="plot six regularized model-specific consensus means")
    ap.add_argument("--in-dir", default=str(DEFAULT_IN))
    ap.add_argument("--arm-label",
                    default="reference-regularized SFT ($\\lambda=2$)",
                    help="legend text for the plotted arm; the ICL "
                         "analogue passes 'personal-history ICL "
                         "(frozen, $D=8$)'. The perfect-prediction "
                         "reference is unchanged.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--allow-unsettled", action="store_true",
        help="render a diagnostic preview even when the strict settling gate fails")
    ap.add_argument(
        "--invalid-model", action="append", default=[], choices=ORDER,
        help="leave a model unplotted when its scalar extraction is invalid")
    ap.add_argument(
        "--round30", action="store_true",
        help="plot the seed-averaged round-30 post-peer value")
    ap.add_argument(
        "--horizontal", action="store_true",
        help="render a compact horizontal dot-and-whisker comparison")
    ap.add_argument(
        "--bars", action="store_true",
        help="render a compact bar comparison against perfect prediction")
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
    invalid = set(args.invalid_model)
    plotted = tuple(m for m in ORDER if m not in invalid)
    if not plotted:
        ap.error("--invalid-model excludes every model")
    if not args.allow_unsettled and not all(
            rows[m]["all_converged"] == "True" and
            rows[m].get("any_cyclic", "False") == "False" and
            rows[m]["all_consensus"] == "True" for m in plotted):
        ap.error("at least one model contains an unsettled, cyclic or "
                 "non-consensus seed")
    summary = json.loads(summary_path.read_text())
    perfect = float(summary["perfect_prediction_mean"])

    if args.round30:
        cells_path = in_dir / "model_equilibrium_cells.csv"
        if not cells_path.exists():
            ap.error("model_equilibrium_cells.csv absent")
        with cells_path.open() as fh:
            cells = list(csv.DictReader(fh))
        tcrit = float(summary.get("t_crit_df2_95", 4.302652729911275))
        for model in ORDER:
            values = np.asarray([
                float(r["final_mean"]) for r in cells if r["model"] == model
            ])
            if len(values) != 3:
                ap.error(f"expected three round-30 values for {model}")
            mean = float(values.mean())
            half = tcrit * float(values.std(ddof=1)) / math.sqrt(3)
            rows[model]["equilibrium_mean"] = str(mean)
            rows[model]["ci95_low"] = str(mean - half)
            rows[model]["ci95_high"] = str(mean + half)

    plotted_positions = []
    for i, model in enumerate(plotted):
        if i == 0:
            plotted_positions.append(0.0)
        else:
            gap = 1.0 if FAMILY[model] == FAMILY[plotted[i - 1]] else 1.35
            plotted_positions.append(plotted_positions[-1] + gap)
    plotted_x = np.asarray(plotted_positions)
    y = np.asarray([float(rows[m]["equilibrium_mean"]) for m in plotted])
    lo = np.asarray([float(rows[m]["ci95_low"]) for m in plotted])
    hi = np.asarray([float(rows[m]["ci95_high"]) for m in plotted])
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

    if args.bars:
        fig, ax = plt.subplots(figsize=(5.9, 2.85))
        fig.subplots_adjust(left=.11, right=.985, top=.94, bottom=.24)

        bars = ax.bar(
            plotted_x, y, width=.70, color=BLUE, edgecolor="#355f99",
            linewidth=.65, zorder=2,
        )
        ax.errorbar(
            plotted_x, y, yerr=yerr, fmt="none", ecolor=INK,
            capsize=2.6, elinewidth=1.0, capthick=1.0, zorder=4,
        )
        ax.axhline(
            perfect, color=REFERENCE, lw=1.35, ls=(0, (4, 2.5)),
            zorder=3,
        )

        for bar, value in zip(bars, y):
            ax.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.7,
                fontweight="bold", color=INK,
            )

        ax.set_xticks(plotted_x)
        ax.set_xticklabels([LABEL[m] for m in plotted], fontsize=9.1)
        ax.set_ylabel("Equilibrium", fontsize=10.0, labelpad=4)
        ax.set_ylim(0, min(1.0, max(.80, float(hi.max()) + .075)))
        ax.set_xlim(-.48, plotted_x[-1] + .48)
        ax.grid(axis="y", color="#d9dde2", lw=.6, alpha=.72, zorder=0)
        ax.tick_params(axis="both", labelsize=9.0, length=2.4,
                       width=.65, pad=2)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontweight("bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -.40, perfect + .012, "perfect prediction",
            ha="left", va="bottom", fontsize=8.4,
            fontweight="bold", color=REFERENCE,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": .8,
                  "alpha": .9},
        )

        stem = out_dir / "model_specific_open_gate_equilibria_bars"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=.025)
        fig.savefig(f"{stem}.png", dpi=320, bbox_inches="tight",
                    pad_inches=.025)
        plt.close(fig)
        print(f"[plot_s3m] wrote {stem}.pdf / .png")
        return 0

    if args.horizontal:
        fig, ax = plt.subplots(figsize=(5.35, 2.75))
        fig.subplots_adjust(left=.23, right=.98, top=.91, bottom=.20)
        ypos = np.arange(len(plotted))[::-1]
        ax.axvline(perfect, color=REFERENCE, lw=1.35,
                   ls=(0, (4, 2.5)), zorder=1)
        for yi, value in zip(ypos, y):
            ax.plot([perfect, value], [yi, yi], color=BLUE,
                    alpha=.23, lw=1.25, zorder=1)
        ax.errorbar(y, ypos, xerr=yerr, fmt="o", color=BLUE, ecolor=BLUE,
                    markersize=6.4, markeredgewidth=.7, capsize=2.5,
                    elinewidth=1.25, zorder=3)
        for value, yi in zip(y, ypos):
            ax.annotate(f"{value:.3f}", (value, yi), xytext=(7, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=8.8, fontweight="bold", color=INK)

        low = min(float(lo.min()), perfect)
        high = max(float(hi.max()), perfect)
        span = max(high - low, .1)
        ax.set_xlim(max(0.0, low - .10 * span),
                    min(1.0, high + .16 * span))
        ax.set_ylim(-.55, len(plotted) - .35)
        ax.set_yticks(ypos)
        ax.set_yticklabels([LABEL[m].replace("\n", " ") for m in plotted],
                           fontsize=9.2)
        ax.set_xlabel("Equilibrium", fontsize=10.0, labelpad=3)
        ax.grid(axis="x", color="#d9dde2", lw=.6, alpha=.7)
        ax.tick_params(axis="both", labelsize=9.0, length=2.4,
                       width=.65, pad=2)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontweight("bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(perfect, len(plotted) - .47, "perfect prediction",
                ha="center", va="bottom", fontsize=8.7,
                fontweight="bold", color=REFERENCE)

        stem = out_dir / "model_specific_open_gate_equilibria_horizontal"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=.025)
        fig.savefig(f"{stem}.png", dpi=320, bbox_inches="tight",
                    pad_inches=.025)
        plt.close(fig)
        print(f"[plot_s3m] wrote {stem}.pdf / .png")
        return 0

    fig, ax = plt.subplots(figsize=(6.25, 2.55))
    fig.subplots_adjust(left=.105, right=.99, top=.80, bottom=.23)
    ax.axhline(perfect, color=REFERENCE, lw=1.35, ls=(0, (4, 2.5)), zorder=1)
    # Light displacement stems make the claim legible without adding an
    # unregularized series: every stem starts at perfect prediction.
    for xi, yi in zip(plotted_x, y):
        ax.plot([xi, xi], [perfect, yi], color=BLUE, alpha=.23,
                lw=1.2, zorder=1)
    ax.errorbar(plotted_x, y, yerr=yerr, fmt="o", color=BLUE, ecolor=BLUE,
                markersize=6.2, markeredgewidth=.7, capsize=2.5,
                elinewidth=1.25, zorder=3)

    ax.set_xticks(plotted_x)
    ax.set_xticklabels([LABEL[m] for m in plotted], fontsize=9.1)
    ax.set_ylabel("Equilibrium", fontsize=10.0, labelpad=3)
    low = min(float(lo.min()), perfect)
    high = max(float(hi.max()), perfect)
    pad = max(.025, .10 * (high - low))
    ax.set_ylim(max(0.0, low - pad), min(1.0, high + pad))
    ax.set_xlim(-.45, plotted_x[-1] + .45)
    ax.grid(axis="y", color="#d9dde2", lw=.6, alpha=.7)
    ax.tick_params(axis="both", labelsize=9.0, length=2.4,
                   width=.65, pad=1.8)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([], [], marker="o", color=BLUE, lw=0, markersize=6.2,
               label=args.arm_label),
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
