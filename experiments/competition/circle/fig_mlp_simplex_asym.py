"""Share simplex for ASYMMETRIC cells, where shares can actually move.

Cells:
  pair        uniform pop, two platforms share a base, one sits 0.2 away
  camps 60/40 bimodal population with a majority camp, shared base
  uneven      bimodal 50/50, anchors at [-0.05, 0, +0.25] (close pair + far one)
  sticky      concentrated pop at 0.25, distinct anchors, platform 0 pays a
              chasing cost (prox 1.0), platforms 1,2 move free

Run: python experiments/competition/fig_mlp_simplex_asym.py
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np

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
SIGMA = 0.6
C0 = b4.C0

CELLS = [
    ("pair", dict(innate="uniform", centers=[C0, C0, (C0 + 0.2) % 1.0], prox=None)),
    ("camps 60/40", dict(innate="bimodal64", centers=[C0] * 3, prox=None)),
    ("uneven", dict(innate="bimodal", centers=[(C0 - 0.05) % 1.0, C0, (C0 + 0.25) % 1.0],
                    prox=None)),
    ("sticky p0", dict(innate="concentrated", centers=[(C0 + (i - 1) / 3) % 1.0 for i in range(3)],
                       prox=[1.0, 0.0, 0.0])),
]


def bary(s):
    return (sum(s[i] * V[i][0] for i in range(3)),
            sum(s[i] * V[i][1] for i in range(3)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b4.ROUNDS = 40
    res = {}
    fig, axes = plt.subplots(1, len(CELLS), figsize=(4.2 * len(CELLS), 4.4),
                             constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    print(f"{'cell':>12} {'seed':>4} | final shares (p0, p1, p2) | pdist (01, 02, 12)")
    for ax, (name, cfg) in zip(axes, CELLS):
        tri = V + [V[0]]
        ax.plot([p[0] for p in tri], [p[1] for p in tri], "k-", lw=1, alpha=0.5)
        ax.plot(*bary([1 / 3] * 3), marker="+", c="black", ms=10)
        rows = []
        for c, seed in zip(colors, SEEDS):
            traj = b4.run_cell("distinct", 0.0, cfg["innate"], seed, sigma_f=SIGMA,
                               centers=cfg["centers"], prox_w=cfg["prox"])
            rows.append(traj)
            xy = [bary(r["shares"]) for r in traj]
            ax.plot([p[0] for p in xy], [p[1] for p in xy], c=c, lw=1.2, alpha=0.8)
            ax.plot(*xy[0], "s", c=c, ms=4)
            ax.plot(*xy[-1], "o", c=c, ms=6, mec="black")
            fs = np.mean([r["shares"] for r in traj[-5:]], 0)
            pd = np.mean([r["pdist"] for r in traj[-5:]], 0)
            print(f"{name:>12} {seed:>4} | {fs[0]:.2f} {fs[1]:.2f} {fs[2]:.2f} | "
                  f"{pd[0]:.3f} {pd[1]:.3f} {pd[2]:.3f}", flush=True)
        res[name] = [[{k: v for k, v in r.items()} for r in t] for t in rows]
        ax.set(title=name, xticks=[], yticks=[])
        ax.set_aspect("equal")
    json.dump(res, open(OUT / "mlp_simplex_asym.json", "w"))
    fig.suptitle("asymmetric cells: share simplex (square=start, dot=end, += even split)")
    fig.savefig(OUT / "mlp_simplex_asym.png", dpi=130)
    print(f"saved {OUT / 'mlp_simplex_asym.png'}")


if __name__ == "__main__":
    main()
