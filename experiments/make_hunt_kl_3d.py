"""3D platform-means trajectory for the all-hunters KL sweep. Each axis is one
LLM's mean; dashed diagonal is consensus. beta=0/1/3 should flow to the
diagonal (merge); beta=10 should stay off it (held by the prior).
Run: python experiments/make_hunt_kl_3d.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path("runs/pokec_fj_hunt_kl")
OUT = ROOT / "figs"
BETAS = [0, 1, 3, 10]
COLORS = {0: "#3b1f8b", 1: "#2c9e7f", 3: "#9bbf30", 10: "#f0c000"}


def means(beta, seed):
    tj = ROOT / f"hunt_kl{beta}_t05_s{seed}" / "trajectory.json"
    if not tj.exists():
        return None
    return [r["platform_means"] for r in json.load(open(tj))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    axe = fig.add_subplot(1, 2, 2)
    allv = []
    for beta in BETAS:
        for seed in (0, 1, 2):
            tr = means(beta, seed)
            if tr is None:
                continue
            m0 = [p[0] for p in tr]; m1 = [p[1] for p in tr]; m2 = [p[2] for p in tr]
            allv += m0 + m1 + m2
            solid = seed == 0
            ax.plot(m0, m1, m2, color=COLORS[beta], lw=2 if solid else 1,
                    alpha=1.0 if solid else 0.3, label=f"beta={beta}" if solid else None)
            if solid:
                ax.scatter(m0[0], m1[0], m2[0], color=COLORS[beta], marker="o", s=32)
                ax.scatter(m0[-1], m1[-1], m2[-1], color=COLORS[beta], marker="*", s=140)
            dist = [(sum((x - sum(p) / 3) ** 2 for x in p) / 3) ** 0.5 for p in tr]
            axe.plot(dist, color=COLORS[beta], lw=2 if solid else 1,
                     alpha=1.0 if solid else 0.3, label=f"beta={beta}" if solid else None)

    lo, hi = min(allv), max(allv)
    ax.plot([lo, hi], [lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="consensus diagonal")
    ax.set_xlabel("Qwen mean"); ax.set_ylabel("SmolLM2 mean"); ax.set_zlabel("TinyLlama mean")
    ax.set_title("platform-means trajectory (o=start, *=end)")
    ax.legend(fontsize=7); ax.view_init(elev=22, azim=-60)
    axe.set_title("distance to consensus (std of 3 means) vs round")
    axe.set_xlabel("round"); axe.set_ylabel("std of platform means"); axe.legend(fontsize=8)
    for s in ("top", "right"):
        axe.spines[s].set_visible(False)
    fig.suptitle("KL-anchored hunters over Pokec FJ: platform-means flow by beta",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "trajectory_3d.png", dpi=150)
    print(f"[figs] -> {OUT / 'trajectory_3d.png'}")


if __name__ == "__main__":
    main()
