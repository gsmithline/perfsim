"""Per-node opinion trajectories via NDlib's own OpinionEvolution plotter.

Reruns representative cells of 09 (AlgorithmicBias population + anchored MLP
platform) while recording full status snapshots, then renders each with
ndlib.viz.mpl.OpinionEvolution. Cells: gamma=1.5 fragmentation without the
platform vs the platform pulling everyone to its prior, plus the HK pair.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/fig_ndlib_spaghetti.py
"""

import importlib.util
import os

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")

from ndlib.viz.mpl.OpinionEvolution import OpinionEvolution

_spec = importlib.util.spec_from_file_location("b9", "experiments/competition/09_ab_mlp_loop.py")
b9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b9)

CELLS = [
    (0.3, 1.5, 0.0, "ab_spaghetti_nopf"),
    (0.3, 1.5, 0.3, "ab_spaghetti_platform"),
]


def run_recorded(eps, gamma, w, seed=0):
    pop = b9.build_pop(eps, gamma, seed)
    rng = np.random.default_rng(seed)
    x0 = np.array([pop.status[i] for i in range(b9.N)])
    feats = torch.tensor(b9.make_features(x0, rng), dtype=torch.float32)
    net = b9.pretrain_base(rng, seed)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    snaps = [{"iteration": 0, "status": dict(pop.status)}]
    for t in range(b9.ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(b9.N)])
        target = torch.tensor(x, dtype=torch.float32)
        for _ in range(b9.TRAIN_STEPS):
            opt.zero_grad()
            pred = net(feats).squeeze(1)
            loss = ((pred - target) ** 2).mean() \
                + b9.ANCHOR_W * ((pred - b9.P0) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1).numpy()
        if w > 0:
            gate = np.abs(m - x) < eps
            x = np.where(gate, (1 - w) * x + w * m, x)
            for i in range(b9.N):
                pop.status[i] = float(x[i])
            pop.sts = x.copy()
        snaps.append({"iteration": t + 1, "status": dict(pop.status)})
    return pop, snaps


def main():
    for eps, gamma, w, stem in CELLS:
        pop, snaps = run_recorded(eps, gamma, w)
        viz = OpinionEvolution(pop, snaps)
        viz.plot(f"experiments/competition/figs/{stem}.png")
        print(f"saved experiments/competition/figs/{stem}.png "
              f"(eps={eps} gamma={gamma} w={w})", flush=True)


if __name__ == "__main__":
    main()
