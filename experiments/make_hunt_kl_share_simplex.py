"""The market-share simplex. The 3 corners are the 3 models; the point each
round is the share vector (avg fraction of agents each model gets, sums to 1).
A corner = that model attracting all users; center = even 3-way split.
Trajectory over rounds, averaged over seeds, one triangle per KL beta.
Run: python experiments/make_hunt_kl_share_simplex.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("runs/pokec_fj_hunt_kl")
OUT = ROOT / "figs"
BETAS = [0, 1, 3, 10]
SEEDS = [0, 1, 2]
H = 3 ** 0.5 / 2
V = [(0.5, H), (0.0, 0.0), (1.0, 0.0)]          # Qwen, SmolLM2, TinyLlama
NAMES = ["Qwen", "SmolLM2", "TinyLlama"]


def bary(s):
    return (sum(s[i] * V[i][0] for i in range(3)),
            sum(s[i] * V[i][1] for i in range(3)))


def mean_shares(beta):
    per_seed = []
    for sd in SEEDS:
        tj = ROOT / f"hunt_kl{beta}_t05_s{sd}" / "trajectory.json"
        if tj.exists():
            per_seed.append([r["shares"] for r in json.load(open(tj))])
    if not per_seed:
        return None
    T = min(len(s) for s in per_seed)
    return [[sum(s[t][p] for s in per_seed) / len(per_seed) for p in range(3)]
            for t in range(T)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, beta in zip(axes.flat, BETAS):
        sh = mean_shares(beta)
        if sh is None:
            ax.set_title(f"beta={beta}: missing"); continue
        tri = V + [V[0]]
        ax.plot([p[0] for p in tri], [p[1] for p in tri], "k-", lw=1, alpha=0.5)
        for (vx, vy), nm in zip(V, NAMES):
            ax.text(vx, vy + (0.04 if vy > 0 else -0.06), nm, ha="center",
                    fontsize=9, fontweight="bold")
        cx, cy = bary([1 / 3, 1 / 3, 1 / 3])
        ax.scatter(cx, cy, color="gray", marker="+", s=80, zorder=2)   # even split
        xy = [bary(s) for s in sh]
        xs = [p[0] for p in xy]; ys = [p[1] for p in xy]
        ax.plot(xs, ys, color="#1f4e8b", lw=1.8)
        ax.scatter(xs[0], ys[0], color="#1f4e8b", marker="o", s=45, zorder=3, label="start")
        ax.scatter(xs[-1], ys[-1], color="#d8633b", marker="*", s=220, zorder=4, label="end")
        final = sh[-1]
        ax.set_title(f"beta={beta}   end shares "
                     f"[{final[0]:.2f}, {final[1]:.2f}, {final[2]:.2f}]")
        ax.set_aspect("equal"); ax.axis("off")
    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Market-share simplex: corner = one model takes all users (+ = even split)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "share_simplex.png", dpi=150)
    print(f"[figs] -> {OUT / 'share_simplex.png'}")
    for beta in BETAS:
        sh = mean_shares(beta)
        if sh:
            print(f"beta={beta:>2d}  start {[round(x,2) for x in sh[0]]} "
                  f"-> end {[round(x,2) for x in sh[-1]]}")


if __name__ == "__main__":
    main()
