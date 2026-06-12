"""Block 9: MLP platform in the loop with NDlib AlgorithmicBias (Deffuant) population.

Population: AlgorithmicBiasModel on a complete graph, opinions in [0,1].
Known baselines from _ab_probe: gamma=0 is textbook Deffuant (consensus at
eps>=0.3), gamma=0.75 freezes 2-cluster polarization at eps=0.3, gamma=1.5
destroys consensus at every eps tested.
Platform: one MLP (fixed noisy profile features -> opinion), retrained every
round on the current opinions, anchored toward a constant prior P0 in opinion
space. Feedback respects bounded confidence: agent i blends toward the model
prediction only if |m_i - x_i| < epsilon, weight w.

One model.iteration() does N pair interactions = one sweep, so one round = one
iteration() call. The complete-graph branch caches opinions in model.sts (used
for biased partner selection), so the writeback updates both status and sts.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/09_ab_mlp_loop.py
"""

import json
import os
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ndlib.models.ModelConfig as mc
from ndlib.models.opinions import AlgorithmicBiasModel

N = 300
ROUNDS = 300
P0 = 0.7
ANCHOR_W = 0.3
TRAIN_STEPS = 10
SIGMA_F = 0.15
GAP = 0.02
MIN_SIZE = 5

EPS_GRID = [0.1, 0.2, 0.3, 0.4]
GAMMA_GRID = [0.0, 0.75, 1.5]
W_GRID = [0.0, 0.15, 0.3]
SEEDS = [0, 1]


def mlp():
    return nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1), nn.Sigmoid())


def make_features(x0, rng):
    noise = rng.normal(0, 1.0, (len(x0), 2))
    return np.stack([x0 + rng.normal(0, SIGMA_F, len(x0)), noise[:, 0], noise[:, 1]], 1)


def pretrain_base(rng, seed, n_corpus=4000, steps=400):
    torch.manual_seed(seed)
    y = np.clip(rng.normal(P0, 0.15, n_corpus), 0.01, 0.99)
    feats = torch.tensor(make_features(y, rng), dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)
    net = mlp()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(feats).squeeze(1) - target) ** 2).mean()
        loss.backward()
        opt.step()
    return net


def build_pop(eps, gamma, seed):
    random.seed(seed)
    np.random.seed(seed)
    g = nx.complete_graph(N)
    model = AlgorithmicBiasModel(g)
    cfg = mc.Configuration()
    cfg.add_model_parameter("epsilon", eps)
    cfg.add_model_parameter("gamma", gamma)
    model.set_initial_status(cfg)
    model.iteration(False)  # iteration 0 is a no-op
    return model


def n_clusters(x):
    s = np.sort(x)
    breaks = np.where(np.diff(s) > GAP)[0]
    sizes = np.diff(np.concatenate([[0], breaks + 1, [len(s)]]))
    return int((sizes >= MIN_SIZE).sum())


def run(eps, gamma, w, seed):
    pop = build_pop(eps, gamma, seed)
    rng = np.random.default_rng(seed)
    x0 = np.array([pop.status[i] for i in range(N)])
    feats = torch.tensor(make_features(x0, rng), dtype=torch.float32)
    net = pretrain_base(rng, seed)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    traj = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        target = torch.tensor(x, dtype=torch.float32)
        for _ in range(TRAIN_STEPS):
            opt.zero_grad()
            pred = net(feats).squeeze(1)
            loss = ((pred - target) ** 2).mean() \
                + ANCHOR_W * ((pred - P0) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1).numpy()
        gate = np.abs(m - x) < eps
        if w > 0:
            x_new = np.where(gate, (1 - w) * x + w * m, x)
            for i in range(N):
                pop.status[i] = float(x_new[i])
            pop.sts = x_new.copy()  # complete-graph cache used by partner selection
            x = x_new
        traj.append({
            "pop_mean": float(x.mean()), "pop_std": float(x.std()),
            "clusters": n_clusters(x),
            "pred_mean": float(m.mean()), "pred_std": float(m.std()),
            "loss": float(loss.item()), "contact": float(gate.mean()),
        })
    return traj


def main():
    res = {}
    print(f"AB n={N} complete graph, prior P0={P0}, anchor_w={ANCHOR_W}, "
          f"{ROUNDS} rounds, seeds {SEEDS}")
    print(f"{'eps':>5} {'gamma':>6} {'w':>5} | {'clusters':>8} {'pop_std':>7} "
          f"{'pop_mean':>8} {'contact':>7} {'pred_mean':>9} {'pred_std':>8} {'loss':>7}")
    for eps in EPS_GRID:
        for gamma in GAMMA_GRID:
            for w in W_GRID:
                rows = [run(eps, gamma, w, s) for s in SEEDS]
                res[f"{eps}|{gamma}|{w}"] = rows
                last = lambda f: np.mean(
                    [np.mean([r[f] for r in t[-5:]]) for t in rows])
                print(f"{eps:>5.2f} {gamma:>6.2f} {w:>5.2f} | "
                      f"{last('clusters'):>8.1f} {last('pop_std'):>7.3f} "
                      f"{last('pop_mean'):>8.3f} {last('contact'):>7.2f} "
                      f"{last('pred_mean'):>9.3f} {last('pred_std'):>8.3f} "
                      f"{last('loss'):>7.4f}", flush=True)
    json.dump(res, open("experiments/competition/figs/ab_mlp_loop.json", "w"))

    last = lambda f, e, g, w: np.mean(
        [np.mean([r[f] for r in t[-5:]]) for t in res[f"{e}|{g}|{w}"]])
    fig, ax = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for j, w in enumerate(W_GRID):
        for i, (f, lab, vmax) in enumerate([("clusters", "final cluster count", 5),
                                            ("contact", "contact fraction", 1)]):
            grid = np.array([[last(f, e, g, w) for e in EPS_GRID] for g in GAMMA_GRID])
            im = ax[i, j].imshow(grid, origin="lower", aspect="auto",
                                 vmin=0, vmax=vmax, cmap="viridis")
            ax[i, j].set(xticks=range(len(EPS_GRID)), xticklabels=EPS_GRID,
                         yticks=range(len(GAMMA_GRID)), yticklabels=GAMMA_GRID,
                         xlabel="epsilon", ylabel="gamma",
                         title=f"{lab}, w={w}")
            for gi in range(len(GAMMA_GRID)):
                for ei in range(len(EPS_GRID)):
                    ax[i, j].text(ei, gi, f"{grid[gi, ei]:.2f}", ha="center",
                                  va="center", color="white", fontsize=8)
            fig.colorbar(im, ax=ax[i, j], shrink=0.8)
    colors = {0.0: "#7f7f7f", 0.75: "#1f77b4", 1.5: "#d62728"}
    styles = {0.0: ":", 0.15: "--", 0.3: "-"}
    for g in GAMMA_GRID:
        for w in W_GRID:
            ax[0, 3].plot(EPS_GRID, [last("pop_mean", e, g, w) for e in EPS_GRID],
                          styles[w], marker="o", c=colors[g], ms=4,
                          label=f"gamma={g}, w={w}")
    ax[0, 3].axhline(P0, ls="--", c="black", lw=1)
    ax[0, 3].axhline(0.5, ls=":", c="gray", lw=1)
    ax[0, 3].set(xlabel="epsilon", ylabel="final population mean",
                 title="population mean (black dash = prior P0)")
    ax[0, 3].legend(frameon=False, fontsize=7)
    ax[1, 3].axis("off")
    fig.suptitle("AB population x P0-anchored MLP platform")
    fig.savefig("experiments/competition/figs/ab_mlp_loop.png", dpi=130)
    print("saved experiments/competition/figs/ab_mlp_loop.png")


if __name__ == "__main__":
    main()
