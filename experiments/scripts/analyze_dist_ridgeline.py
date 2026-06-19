"""Ridgeline (joyplot) of opinion + prediction distributions over rounds.

One fine-binned histogram per round, stacked with a vertical offset and colored
early (purple) -> late (yellow), so the distribution's evolution over generations
is visible the way the model-collapse papers show it. Reads op_raw/pred_raw from
trajectory.pt. Pure torch/numpy/matplotlib, no LLM.

rows = runs, cols = {opinions, predictions}. Pass run dirs/tags; no args =
fresh+replace at beta=0 (collapse) and beta=1 (displacement). BINS / STEP env vars
override bin count and round spacing.

Run: MPLCONFIGDIR=/tmp/mpl BINS=80 STEP=2 python experiments/scripts/analyze_dist_ridgeline.py [run_dir_or_tag ...]
"""

import os
import sys

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "runs/pokec_gated_lm"
DEFAULT = [("fresh+replace beta=0", "gfr_e040_b0_s0"),
           ("fresh+replace beta=1", "gfr_e040_b1_s0")]
BINS = int(os.environ.get("BINS", "80"))
STEP = int(os.environ.get("STEP", "2"))


def load(d):
    t = torch.load(os.path.join(d, "trajectory.pt"), map_location="cpu", weights_only=False)
    return np.asarray(t["op_raw"], dtype=np.float64), np.asarray(t["pred_raw"], dtype=np.float64)


def ridge(ax, M, title, lo, hi):
    T = M.shape[0]
    rounds = list(range(0, T, STEP))
    if T - 1 not in rounds:
        rounds.append(T - 1)
    dens = []
    for t in rounds:
        v = M[t][~np.isnan(M[t])]
        h, edges = np.histogram(v, bins=BINS, range=(lo, hi), density=True)
        dens.append((t, h, 0.5 * (edges[:-1] + edges[1:])))
    gmax = max((h.max() for _, h, _ in dens), default=1.0) or 1.0
    scale = 2.2 / gmax
    J = len(dens)
    for j, (t, h, cx) in enumerate(dens):
        ax.fill_between(cx, j, j + h * scale, color=plt.cm.viridis(j / max(J - 1, 1)),
                        alpha=0.85, lw=0.4, edgecolor="white", zorder=J - j)
    ticks = list(range(0, J, max(J // 6, 1)))
    ax.set_yticks(ticks)
    ax.set_yticklabels([dens[k][0] for k in ticks])
    ax.set(title=title, xlim=(lo, hi), ylabel="round")


def main():
    args = sys.argv[1:]
    pairs = [(os.path.basename(a.rstrip("/")), a if "/" in a else os.path.join(ROOT, a)) for a in args] \
        if args else [(n, os.path.join(ROOT, t)) for n, t in DEFAULT]
    runs = [(n, d) for n, d in pairs if os.path.exists(os.path.join(d, "trajectory.pt"))]
    if not runs:
        print("no trajectory.pt found for the requested runs (pull the .pt first)")
        return
    data = [(name, load(d)) for name, d in runs]
    cat = np.concatenate([np.concatenate([op.ravel(), pr.ravel()]) for _, (op, pr) in data])
    cat = cat[~np.isnan(cat)]
    lo = max(0.0, float(np.percentile(cat, 0.5)) - 0.05)
    hi = min(1.0, float(np.percentile(cat, 99.5)) + 0.05)
    n = len(runs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 4.2 * n), squeeze=False)
    for i, (name, (op, pr)) in enumerate(data):
        ridge(axes[i][0], op, f"{name}: opinions", lo, hi)
        ridge(axes[i][1], pr, f"{name}: predictions", lo, hi)
        if i == n - 1:
            axes[i][0].set_xlabel("value")
            axes[i][1].set_xlabel("value")
    fig.tight_layout()
    out = "experiments/competition/figs/dist_ridgeline.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out, "| bins", BINS, "step", STEP)


if __name__ == "__main__":
    main()
