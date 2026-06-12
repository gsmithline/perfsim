"""Block 8: MLP platform in the loop with NDlib bounded-confidence (HK) population.

Population: NDlib HKModel, opinions in [-1, 1]. Known staircase: final cluster
count ~ 1/epsilon, with sharp jumps as epsilon crosses the cluster thresholds.
Platform: one MLP (fixed noisy profile features -> opinion), retrained every
round on the current population opinions (all data is platform-mediated), with
an anchor toward a base net pretrained on a prior centered at P0.
Feedback channel respects the same bounded confidence: agent i blends toward
the model's prediction only if |m_i - x_i| < epsilon, weight w.

Questions: (1) does the model's retraining dynamics jump where the population
cluster count jumps; (2) does w move the critical epsilon; (3) collapse mode:
fragmented population, model prediction outside everyone's window, contact -> 0.

Run: python experiments/competition/08_ndlib_mlp_loop.py
"""

import json
import os
import random
import sys

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ndlib.models.ModelConfig as mc

sys.path.insert(0, "experiments/competition")
from _ndlib_fixed import FixedHKModel  # noqa: E402

N = 300
P_EDGE = 0.05
ROUNDS = 80
P0 = 0.6
ANCHOR_W = 0.3
TRAIN_STEPS = 10
SIGMA_F = 0.15

EPS_GRID = [0.15, 0.25, 0.35, 0.5, 0.7, 1.0]
W_GRID = [0.0, 0.15, 0.3]
SEEDS = [0, 1]


def mlp():
    return nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1), nn.Tanh())


def make_features(x0, rng):
    noise = rng.normal(0, 1.0, (len(x0), 2))
    return np.stack([x0 + rng.normal(0, SIGMA_F, len(x0)), noise[:, 0], noise[:, 1]], 1)


def pretrain_base(rng, seed, n_corpus=4000, steps=400):
    torch.manual_seed(seed)
    y = np.clip(rng.normal(P0, 0.2, n_corpus), -0.99, 0.99)
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


def build_pop(eps, seed):
    random.seed(seed)
    np.random.seed(seed)
    g = nx.erdos_renyi_graph(N, P_EDGE, seed=seed)
    model = FixedHKModel(g)
    cfg = mc.Configuration()
    cfg.add_model_parameter("epsilon", eps)
    model.set_initial_status(cfg)
    return model


def n_clusters(x, bins=40):
    h, _ = np.histogram(x, bins=bins, range=(-1, 1))
    th = 0.05 * len(x)
    peaks, inside = 0, False
    for c in h:
        if c > th and not inside:
            peaks += 1
            inside = True
        elif c <= th:
            inside = False
    return peaks


def run(eps, w, seed):
    pop = build_pop(eps, seed)
    rng = np.random.default_rng(seed)
    x0 = np.array([pop.status[i] for i in range(N)])
    feats = torch.tensor(make_features(x0, rng), dtype=torch.float32)
    base = pretrain_base(rng, seed)
    base_pred = base(feats).squeeze(1).detach()
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
                + ANCHOR_W * ((pred - base_pred) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1).numpy()
        gate = np.abs(m - x) < eps
        if w > 0:
            x_new = np.where(gate, (1 - w) * x + w * m, x)
            for i in range(N):
                pop.status[i] = float(x_new[i])
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
    print(f"HK n={N} ER(p={P_EDGE}), prior P0={P0}, anchor_w={ANCHOR_W}, "
          f"{ROUNDS} rounds, seeds {SEEDS}")
    print(f"{'eps':>5} {'w':>5} | {'clusters':>8} {'pop_std':>7} {'pop_mean':>8} "
          f"{'contact':>7} {'pred_std':>8} {'loss':>7}")
    for eps in EPS_GRID:
        for w in W_GRID:
            rows = [run(eps, w, s) for s in SEEDS]
            res[f"{eps}|{w}"] = rows
            last = lambda f: np.mean([np.mean([r[f] for r in t[-5:]]) for t in rows])
            print(f"{eps:>5.2f} {w:>5.2f} | {last('clusters'):>8.1f} "
                  f"{last('pop_std'):>7.3f} {last('pop_mean'):>8.3f} "
                  f"{last('contact'):>7.2f} {last('pred_std'):>8.3f} "
                  f"{last('loss'):>7.4f}", flush=True)
    json.dump(res, open("experiments/competition/figs/ndlib_mlp_loop.json", "w"))

    fig, ax = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    colors = {0.0: "#7f7f7f", 0.15: "#1f77b4", 0.3: "#d62728"}
    for w in W_GRID:
        last = lambda f, eps: np.mean(
            [np.mean([r[f] for r in t[-5:]]) for t in res[f"{eps}|{w}"]])
        for i, (f, ylab) in enumerate([("clusters", "final cluster count"),
                                       ("contact", "model contact fraction"),
                                       ("pred_std", "model prediction std"),
                                       ("pop_mean", "population mean")]):
            ax[i].plot(EPS_GRID, [last(f, e) for e in EPS_GRID], "o-",
                       c=colors[w], label=f"w={w}")
            ax[i].set(xlabel="epsilon", ylabel=ylab)
    ax[3].axhline(P0, ls="--", c="black", lw=1, label="prior p0")
    ax[3].axhline(0.0, ls=":", c="gray", lw=1)
    for a in ax:
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("HK population x anchored MLP platform: transition seen from the model side")
    fig.savefig("experiments/competition/figs/ndlib_mlp_loop.png", dpi=130)
    print("saved experiments/competition/figs/ndlib_mlp_loop.png")


if __name__ == "__main__":
    main()
