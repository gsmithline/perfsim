"""Flow portrait of platform configurations on the spacing simplex.

State = the three gaps between platform positions around the circle (sum to 1).
Center = evenly segmented market, edge = two platforms merged, corner = all
merged. Platforms start from nets pretrained at random centers (random initial
configuration); anchors follow the condition. Many starts per panel make the
flow and its attractors visible.

Run: python experiments/competition/fig_mlp_flow.py
"""

import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch

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
N_STARTS = 12
ROUNDS = 40
SIGMA = 0.6

PANELS = [
    ("uniform", "shared", 0.0, "uniform pop, shared base"),
    ("uniform", "distinct", 1 / 3, "uniform pop, distinct (1/3)"),
    ("bimodal", "shared", 0.0, "bimodal pop, shared base"),
]


def gaps(plat_means):
    p = np.sort(np.asarray(plat_means) % 1.0)
    g = np.array([p[1] - p[0], p[2] - p[1], (p[0] + 1.0 - p[2])])
    return np.sort(g)[::-1]  # order-free: largest gap first


def bary(g):
    return (sum(g[i] * V[i][0] for i in range(3)),
            sum(g[i] * V[i][1] for i in range(3)))


def run_traj(cond, delta, innate_kind, seed):
    rng = np.random.default_rng(seed)
    pk = b4.load_pokec()
    w_graph = pk["W"]
    n = w_graph.shape[0]
    innate = torch.tensor(b4.make_innate(innate_kind, n, rng), dtype=torch.float32)
    feats = torch.tensor(b4.make_features(innate.numpy(), rng, SIGMA), dtype=torch.float32)
    anchors = b4.centers_for(cond, delta)
    bases = [b4.pretrain_base(c, rng, seed * 10 + i) for i, c in enumerate(anchors)]
    init_centers = rng.random(b4.K)
    platforms = [b4.pretrain_base(c, rng, seed * 10 + 5 + i) for i, c in
                 enumerate(init_centers)]
    base_preds = [b4.pred_vec(b, feats).detach() for b in bases]
    traj = b4._loop(platforms, base_preds, feats, innate.clone(), innate, w_graph, seed)
    return traj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b4.ROUNDS = ROUNDS
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.6 * len(PANELS), 4.8),
                             constrained_layout=True)
    for ax, (kind, cond, delta, title) in zip(axes, PANELS):
        tri = V + [V[0]]
        ax.plot([p[0] for p in tri], [p[1] for p in tri], "k-", lw=1, alpha=0.5)
        ax.plot(*bary([1 / 3] * 3), marker="+", c="black", ms=12)
        cmap = plt.cm.plasma(np.linspace(0, 0.9, N_STARTS))
        for c, seed in zip(cmap, range(N_STARTS)):
            traj = run_traj(cond, delta, kind, seed)
            pts = [bary(gaps_from_record(r)) for r in traj]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, c=c, lw=1.0, alpha=0.75)
            ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
            ax.plot(xs[0], ys[0], "s", c=c, ms=4)
            ax.plot(xs[-1], ys[-1], "o", c=c, ms=6, mec="black")
        ax.set(title=title, xticks=[], yticks=[])
        ax.set_aspect("equal")
        ax.text(*bary([1 / 3] * 3 ), "  even", fontsize=8, color="gray")
        print(f"done: {title}", flush=True)
    fig.suptitle("configuration flow on the spacing simplex "
                 "(center = evenly segmented, corner/edge = merged; 12 random starts)")
    fig.savefig(OUT / "mlp_flow_simplex.png", dpi=130)
    print(f"saved {OUT / 'mlp_flow_simplex.png'}")


def gaps_from_record(r):
    """Recover the three gaps from the recorded pairwise distances is ambiguous;
    use the platform mean angles recorded per round instead."""
    return gaps(r["plat_means"])


if __name__ == "__main__":
    main()
