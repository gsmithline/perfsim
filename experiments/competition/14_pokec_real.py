"""Block 14: the real-data arm of the AB+MLP loop (block 9) on Pokec.

Same platform loop as 09 (P0-anchored MLP retrained each round, bounded-
confidence gated feedback), but the population lives on the real Pokec graph
(n=2163, mean degree ~2.2) instead of a complete graph, and the "real" init
uses the real innate opinions (already in [0,1], concentrated around 0.53).
The uniform-init arm on the same graph isolates graph effects from
marginal-shape effects. Prior P0 = clip(real_mean + 0.2, 0, 1).

On sparse graphs AlgorithmicBiasModel.iteration reads neighbor states from
self.status (copied per sweep); the node_data sts snapshot is never used and
model.sts stays None, so the writeback only needs status (sts is updated too
when present, for complete-graph compatibility).

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/14_pokec_real.py
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
from ndlib.models.opinions import AlgorithmicBiasModel

sys.path.insert(0, "experiments/fj")
from _pokec import load_pokec

torch.set_num_threads(1)

pk = load_pokec()
GRAPH = nx.convert_node_labels_to_integers(nx.from_numpy_array(pk["adj"].numpy()))
X_REAL = np.clip(pk["innate"].numpy().astype(np.float64), 0.0, 1.0)
N = GRAPH.number_of_nodes()
P0 = float(np.clip(X_REAL.mean() + 0.2, 0.0, 1.0))

ROUNDS = 300
ANCHOR_W = 0.3
TRAIN_STEPS = 10
SIGMA_F = 0.15
GAP = 0.02
MIN_SIZE = 5
TAIL = 0.3

INIT_GRID = ["real", "uniform"]
EPS_GRID = [0.1, 0.2, 0.3]
GAMMA_GRID = [0.0, 1.5]
W_GRID = [0.0, 0.3]
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


def write_opinions(model, x):
    for i in range(N):
        model.status[i] = float(x[i])
    if model.sts is not None:
        model.sts = x.copy()


def build_pop(eps, gamma, seed, x0):
    random.seed(seed)
    np.random.seed(seed)
    model = AlgorithmicBiasModel(GRAPH)
    cfg = mc.Configuration()
    cfg.add_model_parameter("epsilon", eps)
    cfg.add_model_parameter("gamma", gamma)
    model.set_initial_status(cfg)
    write_opinions(model, x0)
    model.initial_status = model.status.copy()
    model.iteration(False)  # iteration 0 is a no-op
    return model


def n_clusters(x):
    s = np.sort(x)
    breaks = np.where(np.diff(s) > GAP)[0]
    sizes = np.diff(np.concatenate([[0], breaks + 1, [len(s)]]))
    return int((sizes >= MIN_SIZE).sum())


def run(init, eps, gamma, w, seed):
    rng = np.random.default_rng(seed)
    x0 = X_REAL.copy() if init == "real" else rng.uniform(0, 1, N)
    pop = build_pop(eps, gamma, seed, x0)
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
            write_opinions(pop, x_new)
            x = x_new
        traj.append({
            "pop_mean": float(x.mean()), "pop_std": float(x.std()),
            "clusters": n_clusters(x),
            "tail_mass": float((np.abs(x - P0) > TAIL).mean()),
            "pred_mean": float(m.mean()), "pred_std": float(m.std()),
            "loss": float(loss.item()), "contact": float(gate.mean()),
        })
    return traj


def main():
    deg = np.array([d for _, d in GRAPH.degree()])
    print(f"Pokec n={N}, mean degree {deg.mean():.2f}, "
          f"innate mean {X_REAL.mean():.3f} std {X_REAL.std():.3f} shape {X_REAL.shape}")
    print(f"prior P0={P0:.3f}, anchor_w={ANCHOR_W}, {ROUNDS} rounds, seeds {SEEDS}")
    res = {}
    print(f"{'init':>8} {'eps':>5} {'gamma':>6} {'w':>5} | {'clusters':>8} "
          f"{'pop_std':>7} {'pop_mean':>8} {'tail':>5} {'contact':>7} "
          f"{'pred_mean':>9} {'pred_std':>8} {'loss':>7}")
    for init in INIT_GRID:
        for eps in EPS_GRID:
            for gamma in GAMMA_GRID:
                for w in W_GRID:
                    rows = [run(init, eps, gamma, w, s) for s in SEEDS]
                    res[f"{init}|{eps}|{gamma}|{w}"] = rows
                    last = lambda f: np.mean(
                        [np.mean([r[f] for r in t[-5:]]) for t in rows])
                    print(f"{init:>8} {eps:>5.2f} {gamma:>6.2f} {w:>5.2f} | "
                          f"{last('clusters'):>8.1f} {last('pop_std'):>7.3f} "
                          f"{last('pop_mean'):>8.3f} {last('tail_mass'):>5.2f} "
                          f"{last('contact'):>7.2f} {last('pred_mean'):>9.3f} "
                          f"{last('pred_std'):>8.3f} {last('loss'):>7.4f}", flush=True)
    json.dump(res, open("experiments/competition/figs/pokec_real.json", "w"))

    last = lambda f, key: np.mean([np.mean([r[f] for r in t[-5:]]) for t in res[key]])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = {0.1: "#1f77b4", 0.2: "#2ca02c", 0.3: "#d62728"}
    for eps in EPS_GRID:
        for init, ls in [("real", "-"), ("uniform", "--")]:
            t = res[f"{init}|{eps}|1.5|0.3"][0]
            ax[0].plot([r["pop_std"] for r in t], ls, c=colors[eps], lw=1.2,
                       label=f"{init}, eps={eps}")
    ax[0].set(xlabel="round", ylabel="pop_std",
              title="pop_std vs round (gamma=1.5, w=0.3, seed 0)")
    ax[0].legend(frameon=False, fontsize=7)
    gcol = {0.0: "#1f77b4", 1.5: "#d62728"}
    for i, (f, lab) in enumerate([("clusters", "final cluster count"),
                                  ("pop_mean", "final population mean")]):
        for gamma in GAMMA_GRID:
            for init, ls in [("real", "-"), ("uniform", "--")]:
                ax[i + 1].plot(EPS_GRID,
                               [last(f, f"{init}|{e}|{gamma}|0.3") for e in EPS_GRID],
                               ls, marker="o", c=gcol[gamma], ms=4,
                               label=f"{init}, gamma={gamma}")
        ax[i + 1].set(xlabel="epsilon", ylabel=lab, title=f"{lab} at w=0.3")
        ax[i + 1].legend(frameon=False, fontsize=7)
    ax[2].axhline(P0, ls="--", c="black", lw=1)
    ax[2].axhline(X_REAL.mean(), ls=":", c="gray", lw=1)
    fig.suptitle("Pokec real graph x P0-anchored MLP platform "
                 f"(P0={P0:.2f}, real innate mean={X_REAL.mean():.2f})")
    fig.savefig("experiments/competition/figs/pokec_real.png", dpi=130)
    print("saved experiments/competition/figs/pokec_real.png")


if __name__ == "__main__":
    main()
