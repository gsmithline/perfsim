"""3D platform-means trajectory for the all-hunters runs (hunt+hunt+hunt),
mirror of make_competition_3d.py. Each axis is one LLM's mean; dashed diagonal
is consensus. Cells differ in tau (+ lr control).
Run: python experiments/make_hunt_3d.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path("runs/pokec_fj_hunt")
OUT = ROOT / "figs"
CELLS = [
    ("hunt3_t05", "tau=0.05", "#d8633b"),
    ("hunt3_t01", "tau=0.01", "#3b6fd8"),
    ("hunt3_t05_lrlo", "tau=0.05 lr-lo", "#3bb0a0"),
]


def means(tag):
    tj = ROOT / tag / "trajectory.json"
    if not tj.exists():
        return None
    return [r["platform_means"] for r in json.load(open(tj))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    axe = fig.add_subplot(1, 2, 2)

    allv = []
    for tag, label, c in CELLS:
        tr = means(tag)
        if tr is None:
            continue
        m0 = [p[0] for p in tr]; m1 = [p[1] for p in tr]; m2 = [p[2] for p in tr]
        allv += m0 + m1 + m2
        ax.plot(m0, m1, m2, color=c, lw=2, label=label)
        ax.scatter(m0[0], m1[0], m2[0], color=c, marker="o", s=35)
        ax.scatter(m0[-1], m1[-1], m2[-1], color=c, marker="*", s=140)
        dist = [(sum((x - sum(p) / 3) ** 2 for x in p) / 3) ** 0.5 for p in tr]
        axe.plot(dist, color=c, lw=2, label=label)

    lo, hi = min(allv), max(allv)
    ax.plot([lo, hi], [lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="consensus diagonal")
    ax.set_xlabel("Qwen mean"); ax.set_ylabel("SmolLM2 mean"); ax.set_zlabel("TinyLlama mean")
    ax.set_title("platform-means trajectory (o=start, *=end)")
    ax.legend(fontsize=7); ax.view_init(elev=22, azim=-60)

    axe.set_title("distance to consensus (std of 3 means) vs round")
    axe.set_xlabel("round"); axe.set_ylabel("std of platform means")
    axe.legend(fontsize=8)
    for s in ("top", "right"):
        axe.spines[s].set_visible(False)

    fig.suptitle("Three hunters over Pokec FJ: where the platform means flow",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "trajectory_3d.png", dpi=150)
    print(f"[figs] -> {OUT / 'trajectory_3d.png'}")


if __name__ == "__main__":
    main()
