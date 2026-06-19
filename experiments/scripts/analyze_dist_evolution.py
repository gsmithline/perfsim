"""Population opinion + model-prediction distributions over rounds (the whole picture).

For each run, two heatmaps: the per-agent OPINION distribution and the model's
per-agent PREDICTION distribution, each round binned over [0,1], with mean and
p10/p90 overlaid. Shows collapse (predictions -> a spike), displacement (the cloud
slides up keeping its spread), and the stranded-tail floor directly. Reads
op_raw/pred_raw from trajectory.pt. Pure torch/numpy/matplotlib, no LLM.

Pass run dirs (or bare tags under runs/pokec_gated_lm). No args = the 4 corners at
eps=0.4, beta=0 (the collapse case).

Run: MPLCONFIGDIR=/tmp/mpl python experiments/scripts/analyze_dist_evolution.py [run_dir_or_tag ...]
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
DEFAULT = [("continual+replace", "ginn_e040_l00_b0_s0"), ("fresh+replace", "gfr_e040_b0_s0"),
           ("fresh+accumulate", "gfa_e040_b0_s0"), ("fresh+pristine", "gfp_e040_b0_s0")]


def load(d):
    t = torch.load(os.path.join(d, "trajectory.pt"), map_location="cpu", weights_only=False)
    return np.asarray(t["op_raw"], dtype=np.float64), np.asarray(t["pred_raw"], dtype=np.float64)


def heat(ax, M, title, lo, hi, bins=60):
    T = M.shape[0]
    H = np.zeros((bins, T))
    for t in range(T):
        v = M[t][~np.isnan(M[t])]
        h, _ = np.histogram(v, bins=bins, range=(lo, hi))
        H[:, t] = h / max(len(v), 1)
    vmax = np.percentile(H[H > 0], 98) if (H > 0).any() else 1.0
    ax.imshow(H, origin="lower", aspect="auto", extent=[0, T, lo, hi], cmap="magma", vmin=0, vmax=vmax)
    r = np.arange(T) + 0.5
    ax.plot(r, np.nanmean(M, 1), color="cyan", lw=1.2)
    ax.plot(r, np.nanpercentile(M, 10, 1), color="cyan", lw=0.6, ls="--")
    ax.plot(r, np.nanpercentile(M, 90, 1), color="cyan", lw=0.6, ls="--")
    ax.set(title=title, ylim=(lo, hi))


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
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.7 * n), squeeze=False)
    for i, (name, (op, pr)) in enumerate(data):
        heat(axes[i][0], op, f"{name}: opinions", lo, hi)
        heat(axes[i][1], pr, f"{name}: predictions", lo, hi)
        axes[i][0].set_ylabel("value")
        if i == n - 1:
            axes[i][0].set_xlabel("round")
            axes[i][1].set_xlabel("round")
    fig.tight_layout()
    out = "experiments/competition/figs/dist_evolution.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
