"""Opinion + prediction histograms at snapshot rounds (shape over time).

Overlaid step-histograms at a few rounds, colored early (purple) -> late (yellow),
for the per-agent opinions and the model's per-agent predictions. Makes the
distribution SHAPE explicit: the spike forming at beta=0 (collapse), the shifted
multi-band spread at beta=1 (displacement). Reads op_raw/pred_raw from
trajectory.pt. Pure torch/numpy/matplotlib, no LLM.

Pass run dirs/tags. No args = fresh+replace at beta=0/0.5/1 (the beta-switch).

Run: MPLCONFIGDIR=/tmp/mpl python experiments/scripts/analyze_dist_hist.py [run_dir_or_tag ...]
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
           ("fresh+replace beta=0.5", "gfr_e040_b0p5_s0"),
           ("fresh+replace beta=1", "gfr_e040_b1_s0")]
SNAPS = [0, 5, 15, 30, 59]


def load(d):
    t = torch.load(os.path.join(d, "trajectory.pt"), map_location="cpu", weights_only=False)
    return np.asarray(t["op_raw"], dtype=np.float64), np.asarray(t["pred_raw"], dtype=np.float64)


def hist_panel(ax, M, title, lo, hi, bins=40):
    T = M.shape[0]
    snaps = [t for t in SNAPS if t < T]
    for k, t in enumerate(snaps):
        v = M[t][~np.isnan(M[t])]
        ax.hist(v, bins=bins, range=(lo, hi), histtype="step", density=True,
                color=plt.cm.viridis(k / max(len(snaps) - 1, 1)), lw=1.6, label=f"r{t}")
    ax.set(title=title, xlim=(lo, hi))
    ax.legend(fontsize=7, frameon=False)


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
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.7 * n), squeeze=False, sharex=True)
    for i, (name, (op, pr)) in enumerate(data):
        hist_panel(axes[i][0], op, f"{name}: opinions", lo, hi)
        hist_panel(axes[i][1], pr, f"{name}: predictions", lo, hi)
        if i == n - 1:
            axes[i][0].set_xlabel("value")
            axes[i][1].set_xlabel("value")
    fig.tight_layout()
    out = "experiments/competition/figs/dist_hist.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
