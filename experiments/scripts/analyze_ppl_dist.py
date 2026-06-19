"""Plot the empirical answer-perplexity distribution over model generations.

Reads ppl_raw [n_rounds, n_profiles] from a run's trajectory.pt (logged when
LOG_PPL_DIST=1) and shows the distribution deforming round by round: quantile
bands (median, p10-p90, p99) on the left, a ridgeline of snapshot rounds on the
right. The tail tracks which profiles the model can no longer reproduce -- the
model-collapse signature.

Falls back to the telemetry quantiles (ppl_p10/p50/p90/p99) for the band plot
when the full array is absent (e.g. only the JSON was pulled, not the .pt). Pure
torch/numpy/matplotlib, no LLM.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/scripts/analyze_ppl_dist.py runs/pokec_gated_lm/gca_e040_b0_s0 [more dirs ...]
"""

import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_ppl(run_dir):
    """ppl_raw as [T, M] numpy, or None if the full array was not logged/pulled."""
    tp = os.path.join(run_dir, "trajectory.pt")
    if not os.path.exists(tp):
        return None
    d = torch.load(tp, map_location="cpu", weights_only=False)
    pr = d.get("ppl_raw")
    if pr is None or not hasattr(pr, "numel") or pr.numel() == 0:
        return None
    return np.asarray(pr, dtype=np.float64)


def load_quantiles(run_dir):
    """Per-round ppl quantiles from telemetry.json (fallback when ppl_raw absent)."""
    path = os.path.join(run_dir, "telemetry.json")
    if not os.path.exists(path):
        return None
    keys = ["ppl_p10", "ppl_p50", "ppl_p90", "ppl_p99", "ppl_max"]
    out = {k: [] for k in keys}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if "ppl_p50" in r:
            for k in keys:
                out[k].append(r.get(k, np.nan))
    if not out["ppl_p50"]:
        return None
    return {k: np.array(v) for k, v in out.items()}


def _band(ax, rounds, p10, p50, p90, p99, title):
    ax.fill_between(rounds, p10, p90, alpha=0.3, color="#1f77b4", label="p10-p90")
    ax.plot(rounds, p50, color="#1f77b4", lw=2, label="median")
    ax.plot(rounds, p99, color="#d62728", lw=1, label="p99")
    ax.set(xlabel="round", ylabel="answer perplexity", yscale="log", title=title)
    ax.legend(fontsize=7, frameon=False)


def plot_run(run_dir, ax_band, ax_ridge):
    name = os.path.basename(run_dir)
    arr = load_ppl(run_dir)
    if arr is not None:
        T = arr.shape[0]
        rounds = np.arange(T)
        _band(ax_band, rounds,
              np.percentile(arr, 10, axis=1), np.percentile(arr, 50, axis=1),
              np.percentile(arr, 90, axis=1), np.percentile(arr, 99, axis=1), name)
        snaps = sorted(set([0, T // 4, T // 2, 3 * T // 4, T - 1]))
        offs = np.linspace(0, 1, len(snaps))
        for o, t in zip(offs, snaps):
            v = np.log10(np.clip(arr[t], 1e-3, None))
            hist, edges = np.histogram(v, bins=40, range=(0, 6), density=True)
            cx = 0.5 * (edges[:-1] + edges[1:])
            h = hist / max(hist.max(), 1e-9) * 0.18
            ax_ridge.fill_between(cx, o, o + h, alpha=0.75, color=plt.cm.viridis(o), label=f"r{t}")
        ax_ridge.set(xlabel="log10 perplexity", yticks=[], title="distribution / round")
        ax_ridge.legend(fontsize=6, frameon=False, loc="upper right")
        return True
    q = load_quantiles(run_dir)
    if q is None:
        ax_band.text(0.5, 0.5, "no ppl logged here\n(run with LOG_PPL_DIST=1)",
                     ha="center", va="center", transform=ax_band.transAxes)
        ax_band.set(title=name)
        ax_ridge.axis("off")
        return False
    rounds = np.arange(len(q["ppl_p50"]))
    _band(ax_band, rounds, q["ppl_p10"], q["ppl_p50"], q["ppl_p90"], q["ppl_p99"], name)
    ax_ridge.text(0.5, 0.5, "ridgeline needs ppl_raw\n(pull trajectory.pt)",
                  ha="center", va="center", transform=ax_ridge.transAxes)
    ax_ridge.axis("off")
    return True


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: python experiments/scripts/analyze_ppl_dist.py <run_dir> [run_dir ...]")
        return
    n = len(dirs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.2 * n), squeeze=False)
    for i, d in enumerate(dirs):
        plot_run(d, axes[i][0], axes[i][1])
    fig.tight_layout()
    out = "experiments/competition/figs/ppl_dist.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
