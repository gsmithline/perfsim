"""Hunt-grid figures: hunters vs describers vs mixed playing the game over the
Pokec FJ population. Per-round inter-platform divergence, distance-to-consensus
(std of the 3 means), and population op_std. Single seed per cell.
Run: python experiments/make_hunt_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("runs/pokec_fj_hunt")
OUT = ROOT / "figs"

# (tag, label, color, linestyle) -- all 3 LLMs are hunters
CELLS = [
    ("hunt3_t05", "tau=0.05", "#d8633b", "-"),
    ("hunt3_t01", "tau=0.01", "#3b6fd8", "-"),
    ("hunt3_t05_lrlo", "tau=0.05 lr-lo", "#3bb0a0", "--"),
]


def load(tag):
    tj = ROOT / tag / "trajectory.json"
    return json.load(open(tj)) if tj.exists() else None


def std_means(p):
    m = sum(p) / 3
    return (sum((x - m) ** 2 for x in p) / 3) ** 0.5


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    print(f"{'cell':>20s} | {'div f->l':>14s} {'gap f->l':>14s} {'op_std f->l':>14s}")
    for tag, label, c, ls in CELLS:
        t = load(tag)
        if t is None:
            print(f"{tag:>20s} | missing"); continue
        div = [r["pred_divergence"] for r in t]
        gap = [std_means(r["platform_means"]) for r in t]
        ostd = [r["op_std"] for r in t]
        ax[0].plot(div, color=c, ls=ls, label=label)
        ax[1].plot(gap, color=c, ls=ls, label=label)
        ax[2].plot(ostd, color=c, ls=ls, label=label)
        print(f"{tag:>20s} | {div[0]:.3f}->{div[-1]:.3f}   "
              f"{gap[0]:.3f}->{gap[-1]:.3f}   {ostd[0]:.3f}->{ostd[-1]:.3f}")

    ax[0].set_title("inter-platform divergence vs round")
    ax[1].set_title("distance to consensus (std of means) vs round")
    ax[2].set_title("population op_std vs round")
    for a in ax:
        a.set_xlabel("round")
        a.legend(fontsize=7)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Three hunters (all 3 LLMs play the game) on Pokec FJ",
                 fontweight="bold")
    fig.savefig(OUT / "hunt_grid.png", dpi=150)
    print(f"[figs] -> {OUT / 'hunt_grid.png'}")


if __name__ == "__main__":
    main()
