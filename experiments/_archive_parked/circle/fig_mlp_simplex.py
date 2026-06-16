"""Market-share simplex for the MLP circle competition (block 4 machinery).

Corners = the 3 platforms, center = even split. One triangle per cell; per-seed
trajectories over rounds, fresh runs so every cell gets several seeds.

Run: python experiments/competition/fig_mlp_simplex.py
"""

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_spec = importlib.util.spec_from_file_location("b4", "experiments/competition/circle/04_mlp_circle.py")
b4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b4)

OUT = Path("experiments/competition/circle/figs")
H = 3 ** 0.5 / 2
V = [(0.5, H), (0.0, 0.0), (1.0, 0.0)]
SEEDS = [0, 1, 2, 3]
ROUNDS = 40

CELLS = [
    ("uniform", 0.0, 0.6, "uniform, shared base"),
    ("uniform", 1 / 3, 0.6, "uniform, distinct (1/3)"),
    ("bimodal", 0.0, 0.6, "bimodal, shared base"),
    ("concentrated", 0.0, 0.6, "concentrated, shared base"),
]


def bary(s):
    return (sum(s[i] * V[i][0] for i in range(3)),
            sum(s[i] * V[i][1] for i in range(3)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b4.ROUNDS = ROUNDS
    fig, axes = plt.subplots(1, len(CELLS), figsize=(4.2 * len(CELLS), 4.4),
                             constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (kind, delta, sigma, title) in zip(axes, CELLS):
        tri = V + [V[0]]
        ax.plot([p[0] for p in tri], [p[1] for p in tri], "k-", lw=1, alpha=0.5)
        cx, cy = bary([1 / 3] * 3)
        ax.plot(cx, cy, "k+", ms=10)
        cond = "shared" if delta == 0.0 else "distinct"
        for c, seed in zip(colors, SEEDS):
            traj = b4.run_cell(cond, delta, kind, seed, sigma_f=sigma)
            xy = [bary(r["shares"]) for r in traj]
            ax.plot([p[0] for p in xy], [p[1] for p in xy], c=c, lw=1.2, alpha=0.8)
            ax.plot(*xy[0], "s", c=c, ms=4)
            ax.plot(*xy[-1], "o", c=c, ms=6, mec="black")
        ax.set(title=title, xticks=[], yticks=[])
        ax.set_aspect("equal")
        print(f"done: {title}", flush=True)
    fig.suptitle("share simplex, 4 seeds (square=start, dot=end, += even split)")
    fig.savefig(OUT / "mlp_share_simplex.png", dpi=130)
    print(f"saved {OUT / 'mlp_share_simplex.png'}")


if __name__ == "__main__":
    main()
