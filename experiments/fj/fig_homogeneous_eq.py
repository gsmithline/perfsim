"""Phase-portrait figure: myopic competition under drift flows to the
homogeneous equilibrium from every start.
Run: python experiments/fj/fig_homogeneous_eq.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, "experiments/fj")
import run_hotelling as rh

OUT = Path("runs/fj_beach")
LR = 0.01
ROUNDS = 1200
STARTS = [
    (0.25, 0.75), (0.30, 0.45), (0.55, 0.80), (0.35, 0.40),
    (0.60, 0.72), (0.22, 0.60), (0.45, 0.55), (0.30, 0.68),
]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rh.LR_STRATEGIC = LR
    rh.ROUNDS = ROUNDS
    setup = rh.load_pokec()
    runs = [rh.run_market(setup, update="strategic", drift=True, peers=True,
                          seed=0, b0=s, beach_bins=80) for s in STARTS]

    fig = plt.figure(figsize=(13.5, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.3, wspace=0.25)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])
    cmap = plt.cm.viridis

    # A: (b1, b2) phase portrait
    axA.plot([0.15, 0.9], [0.15, 0.9], ls="--", color="black", lw=1.0, alpha=0.6)
    axA.text(0.82, 0.86, "co-location\n(b1 = b2)", fontsize=9, alpha=0.7, ha="center")
    ends = []
    for i, r in enumerate(runs):
        b = torch.tensor(r["b"])                      # (T, 2)
        c = cmap(i / (len(runs) - 1))
        axA.plot(b[:, 0], b[:, 1], color=c, lw=1.6, alpha=0.9)
        axA.scatter(b[0, 0], b[0, 1], facecolors="none", edgecolors=c, s=55, zorder=3)
        k = ROUNDS // 8
        axA.annotate("", xy=(b[k + 4, 0], b[k + 4, 1]), xytext=(b[k, 0], b[k, 1]),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4))
        ends.append(b[-1])
    end = torch.stack(ends).mean(dim=0)
    axA.scatter(*end, marker="*", s=380, color="#d8633b", edgecolors="black",
                zorder=4, label="homogeneous equilibrium")
    axA.scatter([], [], facecolors="none", edgecolors="gray", s=55, label="initial placement")
    axA.set_xlabel("platform 1 position")
    axA.set_ylabel("platform 2 position")
    axA.set_title("every starting placement flows to co-location")
    axA.set_xlim(0.15, 0.9)
    axA.set_ylim(0.15, 0.9)
    axA.legend(loc="lower right", fontsize=9, frameon=False)
    style(axA)

    # B: (gap, beach std) plane
    for i, r in enumerate(runs):
        gap = torch.tensor(r["gap"])
        std = torch.tensor(r["std"])
        c = cmap(i / (len(runs) - 1))
        axB.plot(gap, std, color=c, lw=1.4, alpha=0.9)
        axB.scatter(gap[0], std[0], facecolors="none", edgecolors=c, s=45, zorder=3)
    axB.scatter(0.0, float(torch.tensor([r["std"][-1] for r in runs]).mean()),
                marker="*", s=300, color="#d8633b", edgecolors="black", zorder=4)
    axB.axhline(float(setup["innate"].std()), ls=":", color="black", lw=1.0, alpha=0.6)
    axB.text(0.42, float(setup["innate"].std()) + 0.003, "innate diversity", fontsize=8, alpha=0.7)
    axB.set_xlabel("platform gap  |b1 - b2|")
    axB.set_ylabel("beach std (opinion diversity)")
    axB.set_title("market and population homogenize together")
    style(axB)

    # C: the beach over time, platforms overlaid (one representative run)
    r = runs[0]
    beach = torch.stack(r["beach"]).t()               # (bins, T)
    axC.imshow(beach.sqrt(), aspect="auto", origin="lower", cmap="magma",
               extent=[0, ROUNDS, 0, 1])
    b = torch.tensor(r["b"])
    axC.plot(range(ROUNDS), b[:, 0], color="#7fd3f5", lw=1.6)
    axC.plot(range(ROUNDS), b[:, 1], color="#7fb8f5", lw=1.6, ls="--")
    axC.set_xlabel("round")
    axC.set_ylabel("opinion")
    axC.set_title("the beach collapses onto the merged platforms")
    axC.set_ylim(0.2, 0.9)

    fig.suptitle("Myopic platform competition under opinion drift: the homogeneous equilibrium\n"
                 f"(Pokec N=2163, strategic gradient play, lr={LR:g}, {ROUNDS} rounds)",
                 fontweight="bold")
    fig.savefig(OUT / "homogeneous_eq.png", dpi=150, bbox_inches="tight")
    print(f"mean final position {float(end.mean()):.3f}, "
          f"final std {float(torch.tensor([r['std'][-1] for r in runs]).mean()):.3f}")
    print(f"[fj] figure -> {OUT / 'homogeneous_eq.png'}")


if __name__ == "__main__":
    main()
