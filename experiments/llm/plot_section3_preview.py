#!/usr/bin/env python3
"""Compact Section 3 PREVIEW figure -- the retention ladder.

CPU only, seconds to run. Set OMP_NUM_THREADS=1; this reads the CSVs
analyze_section3.py wrote and opens no run artifacts and no model.

  OMP_NUM_THREADS=1 python experiments/llm/plot_section3_preview.py \\
      --in notes/pofd/section3_analysis \\
      --out notes/pofd/figures

====================================================================
LAYOUT, PANEL BY PANEL
====================================================================
Six rows x four columns, in TWO visually separated blocks.

ROWS -- the k = 1 MAIN BLOCK (rows 1-4):
    1  qwen7b     beta = 0.5   (main)
    2  qwen3_8b   beta = 0.5   (main)
    3  qwen7b     beta = 1     (wu)
    4  qwen3_8b   beta = 1     (wu)
  then a HEAVY RULE and a wide gap, and the k = 0.2 MEMORY EXTENSION:
    5  qwen7b     beta = 0.5,  k = 0.2
    6  qwen3_8b   beta = 0.5,  k = 0.2
  The two checkpoints are separate rows throughout; the memory extension
  is a separate BLOCK, not another row of the main ladder, because it is
  a different environment and must not be read as one more rung.

COLUMNS:
  1  equilibrium distance to the MATCHED FROZEN-MODEL equilibrium,
     against lambda. lambda = infinity is the frozen model itself and
     therefore sits at 0 by construction -- it anchors the right-hand end
     of the axis rather than being an extra series.
  2  equilibrium population SD against lambda, same axis.
  3  post-peer population MEAN trajectories, rounds 0..100, for
     lambda in {0, 1, 8, infinity} plus perfect prediction.
  4  post-peer population SD trajectories, same arms.

  The lambda axis is CATEGORICAL (0, 0.1, 0.5, 1, 2, 4, 8, infinity) --
  a log axis cannot hold 0 and a linear one crushes the low rungs.

FORWARD IS THE PRIMARY CURVE: solid, thickest, filled circles, blues.
REVERSE KL (lambda in {1, 8}, k = 1 only) IS A ROBUSTNESS CHECK: red
open triangles with NO connecting line in columns 1-2, and thin
dash-dot in columns 3-4. It is never the main line, and its legend entry
says "robustness check".

EQUAL LAMBDA IS NOT EQUAL EFFECTIVE STRENGTH ACROSS DIRECTIONS, so the
reverse markers sit at their own lambda positions purely as locations on
a shared axis, NOT as matched counterparts of the forward rungs. The
footer says this on the figure.

NO TITLES (project convention). Facets are identified by left-edge row
labels and column annotations; the narrative belongs in the caption.

MONOTONICITY IS NOT ASSUMED. In columns 1-2 the forward rungs are joined
by a line ONLY when the plotted sequence is actually monotone in lambda.
When it is not, the markers are drawn unconnected and the panel is
annotated "non-monotonic in lambda", so the figure cannot imply a
variance decline the data does not show.

ABSOLUTE READABLE SCALES. Each column shares ONE y-range across all six
rows. If the memory block's data would occupy less than a quarter of
that shared range, the memory block gets its own y-range and the panel
is explicitly annotated "zoomed y-axis".

Cells still flagged as late-round states (not equilibria) by
analyze_section3.py are marked with a hollow ring around their marker
and counted in the footer -- the figure never silently calls a
late-round state an equilibrium.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(tempfile.gettempdir(), "perfsim-plot-cache"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]

ROUNDS = 100
LATE_LO, LATE_HI = 81, 100
MODELS = ("qwen7b", "qwen3_8b")
MAIN_ROWS = [("qwen7b", 0.5, 1.0), ("qwen3_8b", 0.5, 1.0),
             ("qwen7b", 1.0, 1.0), ("qwen3_8b", 1.0, 1.0)]
MEM_ROWS = [("qwen7b", 0.5, 0.2), ("qwen3_8b", 0.5, 0.2)]

FWD_LADDER = [("sft", 0.0, "0"),
              ("fwdlam0p1", 0.1, "0.1"),
              ("fwdlam0p5", 0.5, "0.5"),
              ("fwdlam1", 1.0, "1"),
              ("fwdlam2", 2.0, "2"),
              ("fwdlam4", 4.0, "4"),
              ("fwdlam8", 8.0, "8"),
              ("__frozen__", math.inf, r"$\infty$")]
REV_ARMS = [("revlam1", 1.0), ("revlam8", 8.0)]
TRAJ_ARMS = [("sft", r"SFT $\lambda=0$"),
             ("fwdlam1", r"forward $\lambda=1$"),
             ("fwdlam8", r"forward $\lambda=8$"),
             ("__frozen__", r"frozen ($\lambda\to\infty$)"),
             ("__perfect__", "perfect prediction")]
REV_TRAJ = [("revlam1", r"reverse $\lambda=1$ (robustness)"),
            ("revlam8", r"reverse $\lambda=8$ (robustness)")]

FWD_BLUE = "#1F4E9C"
REV_RED = "#C0392B"
FROZEN_GREY = "0.45"
PP_BLACK = "#111111"
SFT_GREEN = "#2A8A4A"

TRAJ_STYLE = {
    "sft": dict(color=SFT_GREEN, ls="-", lw=1.5),
    "fwdlam1": dict(color="#5B8FD4", ls="-", lw=1.6),
    "fwdlam8": dict(color=FWD_BLUE, ls="-", lw=1.9),
    "__frozen__": dict(color=FROZEN_GREY, ls=":", lw=2.0),
    "__perfect__": dict(color=PP_BLACK, ls="--", lw=1.5),
    "revlam1": dict(color="#E08E86", ls="-.", lw=1.0),
    "revlam8": dict(color=REV_RED, ls="-.", lw=1.1),
}


def read_csv(path):
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row, key):
    v = row.get(key, "")
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def env_key(row):
    return (row["model"], round(_f(row, "beta"), 6), round(_f(row, "k"), 6))


def monotone(vals):
    """True iff the finite entries are non-increasing OR non-decreasing."""
    v = [x for x in vals if np.isfinite(x)]
    if len(v) < 3:
        return True
    d = np.diff(v)
    return bool((d <= 1e-12).all() or (d >= -1e-12).all())


def main():
    ap = argparse.ArgumentParser(
        description="compact Section 3 preview figure (no titles)")
    ap.add_argument("--in", dest="indir", default="notes/pofd/section3_analysis",
                    help="directory holding analyze_section3.py's CSVs")
    ap.add_argument("--out", default="notes/pofd/figures")
    ap.add_argument("--name", default="section3_preview")
    ap.add_argument("--allow-missing", action="store_true",
                    help="draw whatever is present. OFF by default: a "
                         "silently short ladder reads as a real result")
    args = ap.parse_args()

    indir = Path(args.indir)
    if not indir.is_absolute():
        indir = ROOT / indir
    late_p = indir / "section3_late_equilibrium.csv"
    round_p = indir / "section3_per_round.csv"
    for p in (late_p, round_p):
        if not p.exists():
            print(f"[s3fig] HARD FAIL: {p} does not exist. Run "
                  f"analyze_section3.py first -- this figure is a view of "
                  f"its tables, not an independent computation.",
                  file=sys.stderr)
            return 2

    late = {(r["model"], round(_f(r, "beta"), 6), round(_f(r, "k"), 6),
             r["arm"]): r for r in read_csv(late_p)}
    per = defaultdict(dict)
    for r in read_csv(round_p):
        key = (r["model"], round(_f(r, "beta"), 6), round(_f(r, "k"), 6),
               r["arm"])
        per[key][int(_f(r, "t"))] = r

    rows = MAIN_ROWS + MEM_ROWS
    missing = []
    for (m, b, k) in rows:
        for arm, _lam, _t in FWD_LADDER:
            if (m, b, k, arm) not in late:
                missing.append(f"{m} beta={b:g} k={k:g} arm={arm}")
        if (m, b, k, "__perfect__") not in late:
            missing.append(f"{m} beta={b:g} k={k:g} arm=__perfect__")
        if k == 1.0:
            for arm, _lam in REV_ARMS:
                if (m, b, k, arm) not in late:
                    missing.append(f"{m} beta={b:g} k={k:g} arm={arm}")
    if missing and not args.allow_missing:
        print(f"[s3fig] HARD FAIL: {len(missing)} series missing from "
              f"{late_p.name} --", file=sys.stderr)
        for s in missing:
            print(f"        {s}", file=sys.stderr)
        print("[s3fig] complete the wave and re-run analyze_section3.py, or "
              "pass --allow-missing to draw a deliberately partial ladder.",
              file=sys.stderr)
        return 2
    if missing:
        print(f"[s3fig] WARNING: drawing WITHOUT {len(missing)} series. "
              f"This is a PARTIAL ladder and must be labelled as such.")

    # ---------------- figure skeleton: two separated blocks -------------
    plt.rcParams.update({"font.size": 8.4, "axes.labelsize": 8.6,
                         "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
                         "axes.linewidth": .7, "legend.fontsize": 7.8})
    fig = plt.figure(figsize=(12.6, 12.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[len(MAIN_ROWS),
                                                  len(MEM_ROWS)],
                             hspace=0.185, left=.075, right=.985,
                             top=.955, bottom=.115)
    gs_main = outer[0].subgridspec(len(MAIN_ROWS), 4, hspace=.16, wspace=.28)
    gs_mem = outer[1].subgridspec(len(MEM_ROWS), 4, hspace=.16, wspace=.28)
    axes = {}
    for i, rk in enumerate(MAIN_ROWS):
        for j in range(4):
            axes[(rk, j)] = fig.add_subplot(gs_main[i, j])
    for i, rk in enumerate(MEM_ROWS):
        for j in range(4):
            axes[(rk, j)] = fig.add_subplot(gs_mem[i, j])

    xpos = list(range(len(FWD_LADDER)))
    xlab = [t for _a, _l, t in FWD_LADDER]
    lam_of_arm = {a: l for a, l, _ in FWD_LADDER}
    pos_of_lam = {l: i for i, (_a, l, _t) in enumerate(FWD_LADDER)}
    t_axis = np.arange(0, ROUNDS + 1)

    data_range = {j: [np.inf, -np.inf] for j in range(4)}
    mem_range = {j: [np.inf, -np.inf] for j in range(4)}
    n_flagged = 0

    def track(j, vals, is_mem):
        v = np.asarray([x for x in np.atleast_1d(vals) if np.isfinite(x)],
                       dtype=float)
        if not v.size:
            return
        tgt = mem_range if is_mem else data_range
        for d in (data_range, tgt):
            d[j][0] = min(d[j][0], float(v.min()))
            d[j][1] = max(d[j][1], float(v.max()))

    for rk in rows:
        m, b, k = rk
        is_mem = rk in MEM_ROWS

        # ---- columns 1-2: equilibrium quantities against lambda -------
        for j, field in ((0, "late_w1_frozen_eq"), (1, "late_pop_sd")):
            ax = axes[(rk, j)]
            ys, flags = [], []
            for arm, _lam, _t in FWD_LADDER:
                r = late.get((m, b, k, arm))
                ys.append(_f(r, field) if r else float("nan"))
                flags.append(bool(r) and r.get("converged") == "0")
            mono = monotone(ys)
            ax.plot(xpos, ys, ls="-" if mono else "none", color=FWD_BLUE,
                    lw=1.9, marker="o", ms=4.6, zorder=4,
                    label="forward KL (primary)")
            for x, y, fl in zip(xpos, ys, flags):
                if fl and np.isfinite(y):
                    ax.plot([x], [y], marker="o", ms=9.5, mfc="none",
                            mec=FWD_BLUE, mew=1.0, ls="none", zorder=5)
            if not mono:
                ax.annotate("non-monotonic in $\\lambda$", xy=(.03, .06),
                            xycoords="axes fraction", fontsize=6.9,
                            color="0.30")
            track(j, ys, is_mem)
            if k == 1.0:
                rx, ry = [], []
                for arm, lam in REV_ARMS:
                    r = late.get((m, b, k, arm))
                    if r:
                        rx.append(pos_of_lam[lam])
                        ry.append(_f(r, field))
                if rx:
                    ax.plot(rx, ry, ls="none", marker="^", ms=6.5,
                            mfc="none", mec=REV_RED, mew=1.5, zorder=6,
                            label="reverse KL (robustness check)")
                    track(j, ry, is_mem)
            pp = late.get((m, b, k, "__perfect__"))
            if pp:
                y = _f(pp, field)
                ax.axhline(y, color=PP_BLACK, ls="--", lw=1.2, zorder=3,
                           label="perfect prediction")
                track(j, [y], is_mem)
            ax.set_xticks(xpos)
            ax.set_xticklabels(xlab)
            ax.set_xlim(-.45, len(FWD_LADDER) - .55)
            ax.grid(alpha=.25, lw=.6)
            if rk in (MAIN_ROWS[-1], MEM_ROWS[-1]):
                ax.set_xlabel(r"KL coefficient $\lambda$")

        # ---- columns 3-4: post-peer trajectories -----------------------
        for j, field in ((2, "pop_mean"), (3, "pop_sd")):
            ax = axes[(rk, j)]
            for arm, lbl in TRAJ_ARMS + (REV_TRAJ if k == 1.0 else []):
                d = per.get((m, b, k, arm))
                if not d:
                    continue
                y = np.array([_f(d[t], field) if t in d else float("nan")
                              for t in t_axis])
                ax.plot(t_axis, y, label=lbl, **TRAJ_STYLE[arm], zorder=4)
                track(j, y, is_mem)
            ax.axvspan(LATE_LO, LATE_HI, color="0.88", zorder=0, lw=0)
            ax.set_xlim(0, ROUNDS)
            ax.grid(alpha=.25, lw=.6)
            if rk in (MAIN_ROWS[-1], MEM_ROWS[-1]):
                ax.set_xlabel("round $t$ (0 = innate, post-peer after)")

        n_flagged += sum(
            1 for arm, _l, _t in FWD_LADDER
            if late.get((m, b, k, arm), {}).get("converged") == "0")
        axes[(rk, 0)].set_ylabel(
            f"{m}\n" + rf"$\beta={b:g}$, $k={k:g}$", fontsize=8.2)

    # ---- absolute readable scales, with an explicit zoom label ---------
    for j in range(4):
        lo, hi = data_range[j]
        mlo, mhi = mem_range[j]
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        span = hi - lo
        # A memory block with NO data in this column keeps the shared
        # range: an empty panel must not silently acquire a private axis.
        mem_has = bool(np.isfinite(mlo) and np.isfinite(mhi))
        mem_span = (mhi - mlo) if mem_has else 0.0
        # ...and a block that is FLAT to within FLAT_FLOOR keeps the
        # shared range too. Zooming onto a 1e-7 wobble (perfect
        # prediction is mean-preserving to float noise) would turn
        # arithmetic dust into a full-height signal, which no "zoomed"
        # label can undo.
        FLAT_FLOOR = 1e-4
        zoom_mem = bool(mem_has and span > 0 and mem_span > FLAT_FLOOR
                        and mem_span / span < 0.25)
        for rk in rows:
            ax = axes[(rk, j)]
            if zoom_mem and rk in MEM_ROWS:
                pad = .06 * mem_span if mem_span > 0 else max(1e-3, .05)
                ax.set_ylim(mlo - pad, mhi + pad)
                ax.annotate("zoomed y-axis", xy=(.97, .93),
                            xycoords="axes fraction", ha="right",
                            fontsize=6.9, color="0.30",
                            bbox=dict(fc="w", ec="0.7", lw=.5, pad=1.4))
            else:
                pad = .06 * span if span > 0 else max(1e-3, abs(hi) * .05)
                ax.set_ylim(lo - pad, hi + pad)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    for j, lbl in enumerate((
            "equilibrium $W_1$ to frozen-model equilibrium",
            "equilibrium population SD",
            "post-peer population mean",
            "post-peer population SD")):
        axes[(MAIN_ROWS[0], j)].annotate(
            lbl, xy=(.5, 1.06), xycoords="axes fraction", ha="center",
            va="bottom", fontsize=9.2)

    # ---- the block separator: memory extension is NOT another rung -----
    top_mem = axes[(MEM_ROWS[0], 0)].get_position().y1
    bot_main = axes[(MAIN_ROWS[-1], 0)].get_position().y0
    ysep = (top_mem + bot_main) / 2.0
    fig.add_artist(Line2D([0.03, 0.99], [ysep, ysep], color="0.25", lw=1.9,
                          transform=fig.transFigure))
    fig.text(0.03, ysep + 0.006,
             r"$k=1$ main result", fontsize=8.6, ha="left", va="bottom",
             color="0.25")
    fig.text(0.03, ysep - 0.007,
             r"MEMORY EXTENSION: $k=0.2$ — a different environment, "
             r"not another rung of the $k=1$ ladder",
             fontsize=8.6, ha="left", va="top", color="0.25")

    handles, labels = [], []
    for ax in (axes[(MAIN_ROWS[0], 0)], axes[(MAIN_ROWS[0], 2)]):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(.5, .045))
    fig.text(.5, .012,
             "forward KL is the PRIMARY ladder; reverse KL is a labelled "
             "robustness check — equal $\\lambda$ is NOT equal effective "
             "strength across directions, so the reverse markers share the "
             "axis without being matched counterparts.\n"
             f"Shaded band = equilibrium window (rounds {LATE_LO}–{LATE_HI}). "
             f"A hollow ring marks a cell still flagged as a LATE-ROUND "
             f"STATE, not an equilibrium ({n_flagged} of "
             f"{len(rows) * len(FWD_LADDER)} forward points). "
             "$\\lambda=\\infty$ is the frozen model, so its distance to "
             "itself is 0 by construction.",
             ha="center", va="bottom", fontsize=7.2, color="0.25")
    if missing:
        fig.text(.5, .988,
                 f"PARTIAL LADDER — {len(missing)} series missing; not the "
                 f"complete Section 3 result",
                 ha="center", va="top", fontsize=8.6, color=REV_RED)

    outp = Path(args.out)
    if not outp.is_absolute():
        outp = ROOT / outp
    outp.mkdir(parents=True, exist_ok=True)
    made = []
    for ext in ("png", "pdf"):
        p = outp / f"{args.name}.{ext}"
        fig.savefig(p, dpi=200)
        made.append(p)
    plt.close(fig)
    for p in made:
        print(f"[s3fig] -> {p}")
    print(f"[s3fig] {n_flagged} forward point(s) are late-round states, not "
          f"equilibria (hollow rings).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
