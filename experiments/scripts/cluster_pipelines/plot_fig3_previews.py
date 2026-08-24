#!/usr/bin/env python3
"""PREVIEW for the redesigned Figure 3 -- the intended ONE-ROW, FOUR-PANEL
figure, exactly as it will appear: one panel per gamma in {1, .5, .2, 0},
the full lambda ladder {0, .25, .5, 1, 2, 4, 8, inf} on every x-axis,
beta in {0, .25, .5, .75, 1} as line color, the POST-PEER population
outcome on y, and the perfect-prediction baseline dashed.

Reads fig3_cells.csv + fig3_summary.json from analyze_fig3_full_loop.py
-- never a trajectory -- so presentation iterates in a second.

STRUCTURAL SHAPES a reader should know (both by construction):
  * the beta = 0 line is FLAT: lambda drops out at W = 0 (that line is
    the matched no-platform twin, one value per gamma);
  * the beta = 1 line is IDENTICAL in all four panels: gamma drops out
    at W = 1.
The interesting structure is between the two edges.

Cells whose outcome is not "equilibrium" are RINGED: an unsettled value
is the state at the end of its horizon, not a converged limit, and the
analyzer's --paper mode refuses to bless the figure while any remain.
Long-run/cyclic cells keep the ring and are named in the caption block.

HOUSE RULES: NO TITLE TEXT anywhere (no set_title, no suptitle) -- the
narrative is the printed caption block.  Output goes to the analysis
dir, NEVER to paper/ -- the paper placeholder is replaced only after
check_fig3_full_loop.py passes the whole grid AND the analyzer's --paper
gate is clean.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "perfsim-f3-preview"))

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
DEFAULT_IN = REPO / "notes" / "pofd" / "fig3_full_loop"

PANEL_GAMMAS = ["1", "0.5", "0.2", "0"]        # left to right
BETAS = ["0", "0.25", "0.5", "0.75", "1"]
LAMS = ["0", "0.25", "0.5", "1", "2", "4", "8", "inf"]
LAM_LABEL = {"0": "0\nSFT", "0.25": ".25", "0.5": ".5", "1": "1",
             "2": "2", "4": "4", "8": "8", "inf": "$\\infty$\nfrozen"}
# light-to-dark with beta: the platform's weight literally darkens the line
BETA_COLOR = {"0": "#b8bdc4", "0.25": "#7fa8d9", "0.5": "#4c72b0",
              "0.75": "#2b4f86", "1": "#16294a"}
INK = "#202328"
PERFECT = "#c44e52"


def _read_cells(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _index(rows):
    """(beta_str, gamma_str, lambda_str) -> (value, outcome), with the two
    structural dedups expanded so every panel can just look cells up."""
    idx = {}
    for r in rows:
        b, gs, ls = r["beta_w_plat"], r["gamma_innate_lambda"], \
            r["lambda_kl_beta"]
        v = float(r["mean_postpeer_final"])
        oc = r["outcome"]
        if b == "0":
            # one twin per gamma serves the whole lambda ladder
            for lam in LAMS:
                idx[(b, gs, lam)] = (v, oc)
        elif gs == "dedup":
            # beta = 1: gamma drops out of the operator
            for g in PANEL_GAMMAS:
                idx[(b, g, ls)] = (v, oc)
        else:
            idx[(b, gs, ls)] = (v, oc)
    return idx


def main():
    ap = argparse.ArgumentParser(
        description="the one-row four-panel Figure 3 preview (no titles)")
    ap.add_argument("--in-dir", default=str(DEFAULT_IN))
    ap.add_argument("--out-dir", default=None,
                    help="defaults to <in-dir>/previews; may NOT be under "
                         "paper/")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "previews"
    if "paper" in out_dir.parts:
        ap.error("previews never go under paper/ -- the placeholder is "
                 "replaced only after the whole grid gates clean and the "
                 "analyzer's --paper gate passes")
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = in_dir / "fig3_cells.csv"
    summary_p = in_dir / "fig3_summary.json"
    if not cells.exists() or not summary_p.exists():
        ap.error(f"{cells} / {summary_p} not found -- run "
                 f"analyze_fig3_full_loop.py first")
    rows = _read_cells(cells)
    summary = json.loads(summary_p.read_text())
    innate_mean = summary.get("innate_mean")
    idx = _index(rows)

    x = np.arange(len(LAMS))
    fig, axes = plt.subplots(1, 4, figsize=(15.6, 3.7), sharey=True)
    ringed = []
    for ax, gs in zip(axes, PANEL_GAMMAS):
        for b in BETAS:
            xs, ys, bad = [], [], []
            for i, lam in enumerate(LAMS):
                got = idx.get((b, gs, lam))
                if got is None:
                    continue
                v, oc = got
                xs.append(i)
                ys.append(v)
                if oc != "equilibrium":
                    bad.append((i, v))
                    ringed.append((b, gs, lam, oc))
            ax.plot(xs, ys, marker="o", ms=3.8, lw=1.6,
                    color=BETA_COLOR[b], label=f"$\\beta={b}$")
            if bad:
                ax.scatter([i for i, _ in bad], [v for _, v in bad], s=64,
                           facecolors="none", edgecolors=INK,
                           linewidths=1.1, zorder=5)
        if innate_mean is not None:
            # the perfect-prediction baseline: the served map reproduces
            # the population, so the mean stays at the innate mean
            ax.axhline(innate_mean, ls="--", lw=1.3, color=PERFECT,
                       zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([LAM_LABEL[l] for l in LAMS], fontsize=8.5)
        ax.set_xlabel("$\\lambda$", fontsize=10, color=INK)
        ax.annotate(f"$\\gamma={gs}$", xy=(0.04, 0.94),
                    xycoords="axes fraction", fontsize=11, color=INK,
                    va="top", ha="left")
        ax.tick_params(labelsize=9, colors=INK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK)
    axes[0].set_ylabel("post-peer population mean", fontsize=10, color=INK)
    handles = [Line2D([], [], color=BETA_COLOR[b], marker="o", ms=3.8,
                      lw=1.6, label=f"$\\beta={b}$") for b in BETAS]
    handles.append(Line2D([], [], color=PERFECT, ls="--", lw=1.3,
                          label="perfect prediction"))
    axes[-1].legend(handles=handles, frameon=False, fontsize=8,
                    loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    stem = out_dir / "fig3_preview_onerow"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("=" * 96)
    print(f"[fig3-preview] wrote {stem}.pdf / .png")
    print("=" * 96)
    print("CAPTION BLOCK (the narrative -- this figure carries no title):")
    print("  Population equilibria under recursive post-training, Qwen3-8B")
    print("  on MovieLens/Action, 723 agents, S=100 complete Deffuant")
    print("  sweeps per round, both gates open, seed 0. One panel per")
    print("  innate re-anchor gamma; the forward-KL retention lambda on")
    print("  every x-axis (lambda=0 ordinary SFT, lambda=infinity the")
    print("  frozen model with the population loop still recursive); line")
    print("  color is the platform susceptibility beta. Every value is the")
    print("  POST-PEER population mean at the end of the cell's horizon.")
    print("  The dashed line is the perfect-prediction baseline (the")
    print("  served map reproduces the population, so the mean holds at")
    print("  the innate mean). By construction the beta=0 line is flat")
    print("  (lambda drops out at W=0: the matched no-platform twin) and")
    print("  the beta=1 line is identical in all four panels (gamma drops")
    print("  out at W=1).")
    if ringed:
        uniq = sorted({(b, gs, l, oc) for (b, gs, l, oc) in ringed})
        print(f"  RINGED markers: {len(uniq)} cell(s) not settled --")
        for b, gs, l, oc in uniq:
            print(f"    beta={b} gamma={gs} lambda={l}: {oc}")
        print("  Ringed values are the state at the end of the horizon,")
        print("  not converged limits; the analyzer's --paper gate refuses")
        print("  the figure while any 'extend_to_*' cell remains, and")
        print("  long-run/cyclic cells must be captioned as such.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
