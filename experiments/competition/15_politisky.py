"""Block 15: the AB+MLP loop (block 14) on PolitiSky24 — real polarized stances
on a real Bluesky interaction graph.

Population: PolitiSky24 user-level 2024-election stances (Harris/Trump targets,
Favor/Against/Neither + confidence). Opinion in [0,1]: 0.5 + 0.3 * lean where
lean = (conf-weighted trump score - harris score)/2, plus tiny tie-breaking
jitter. Pro-Harris camp sits near 0.27, pro-Trump near 0.73; heavily imbalanced
(Bluesky skews anti-Trump, ~76% Harris camp vs ~6% Trump camp).

Graph: real quote-interaction network among labeled users, LCC capped to ~4000
nodes (all Trump-camp users kept, majority filled by degree), mean degree ~62.

Platform grid adds P0 in {0.5 centrist, 0.8 partisan}. Results are written per
cell so interrupted runs resume.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/15_politisky.py
"""

import json
import os
import random
import sys

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ndlib.models.ModelConfig as mc
from ndlib.models.opinions import AlgorithmicBiasModel

torch.set_num_threads(1)

DATA = "experiments/data/politisky24"
OUT = "experiments/competition/figs/politisky.json"
STANCE_VAL = {"Favor": 1.0, "Against": -1.0, "Neither": 0.0}
CAP = 4000

ROUNDS = 300
ANCHOR_W = 0.3
TRAIN_STEPS = 10
SIGMA_F = 0.15
GAP = 0.02
MIN_SIZE = 5
TAIL = 0.3

P0_GRID = [0.5, 0.8]
EPS_GRID = [0.1, 0.2, 0.3]
GAMMA_GRID = [0.0, 1.5]
W_GRID = [0.0, 0.3]
SEEDS = [0, 1]


def load_politisky():
    lab = {}
    with open(f"{DATA}/LLM_annotation_on_dataset_users.json") as f:
        for line in f:
            r = json.loads(line)
            v = STANCE_VAL.get(r["stance"], 0.0) * float(r["confidence"] or 0.5)
            lab.setdefault(r["did"], {})[r["is_trump"]] = v
    op = {d: 0.5 + 0.3 * (v.get(1, 0.0) - v.get(0, 0.0)) / 2.0 for d, v in lab.items()}

    df = pd.read_parquet(f"{DATA}/Quote_network.parquet")
    df = df[df["did"].isin(op) & df["quoted_by_did"].isin(op)]
    g = nx.Graph()
    g.add_edges_from(zip(df["did"], df["quoted_by_did"]))
    g.remove_edges_from(nx.selfloop_edges(g))
    g = g.subgraph(max(nx.connected_components(g), key=len))

    deg = dict(g.degree())
    trump = [d for d in g.nodes if op[d] > 0.55]
    rest = sorted((d for d in g.nodes if op[d] <= 0.55), key=lambda d: -deg[d])
    keep = set(trump) | set(rest[: CAP - len(trump)])
    g = g.subgraph(keep)
    g = g.subgraph(max(nx.connected_components(g), key=len))

    x0 = np.array([op[d] for d in g.nodes])
    x0 = np.clip(x0 + np.random.default_rng(12345).normal(0, 0.01, len(x0)), 0, 1)
    return nx.convert_node_labels_to_integers(g), x0


GRAPH, X0 = load_politisky()
N = GRAPH.number_of_nodes()
CAMP_H = X0 < 0.45
CAMP_T = X0 > 0.55
C_H = float(X0[CAMP_H].mean())
C_T = float(X0[CAMP_T].mean())


def mlp():
    return nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1), nn.Sigmoid())


def make_features(x0, rng):
    noise = rng.normal(0, 1.0, (len(x0), 2))
    return np.stack([x0 + rng.normal(0, SIGMA_F, len(x0)), noise[:, 0], noise[:, 1]], 1)


def pretrain_base(rng, seed, p0, n_corpus=4000, steps=400):
    torch.manual_seed(seed)
    y = np.clip(rng.normal(p0, 0.15, n_corpus), 0.01, 0.99)
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


def run(p0, eps, gamma, w, seed):
    rng = np.random.default_rng(seed)
    x0 = X0.copy()
    pop = build_pop(eps, gamma, seed, x0)
    feats = torch.tensor(make_features(x0, rng), dtype=torch.float32)
    net = pretrain_base(rng, seed, p0)
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
                + ANCHOR_W * ((pred - p0) ** 2).mean()
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
            "tail_mass": float((np.abs(x - p0) > TAIL).mean()),
            "pred_mean": float(m.mean()), "pred_std": float(m.std()),
            "loss": float(loss.item()), "contact": float(gate.mean()),
            "mass_h": float((np.abs(x - C_H) < 0.15).mean()),
            "mass_t": float((np.abs(x - C_T) < 0.15).mean()),
            "disp_h": float((x - x0)[CAMP_H].mean()),
            "disp_t": float((x - x0)[CAMP_T].mean()),
            "cross_h": float((x[CAMP_H] > 0.5).mean()),
            "cross_t": float((x[CAMP_T] < 0.5).mean()),
            "contact_h": float(gate[CAMP_H].mean()),
            "contact_t": float(gate[CAMP_T].mean()),
        })
    return traj


def main():
    deg = np.array([d for _, d in GRAPH.degree()])
    print(f"PolitiSky24 quote graph n={N}, mean degree {deg.mean():.2f}")
    print(f"x0 mean {X0.mean():.3f} std {X0.std():.3f}; camps H {CAMP_H.mean():.3f}"
          f" @ {C_H:.3f}, T {CAMP_T.mean():.3f} @ {C_T:.3f}")
    print(f"anchor_w={ANCHOR_W}, {ROUNDS} rounds, seeds {SEEDS}")
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(f"{'P0':>4} {'eps':>5} {'gamma':>6} {'w':>5} | {'pop_std':>7} "
          f"{'pop_mean':>8} {'mass_h':>6} {'mass_t':>6} {'cross_h':>7} "
          f"{'cross_t':>7} {'cont_h':>6} {'cont_t':>6} {'clust':>5}")
    for p0 in P0_GRID:
        for eps in EPS_GRID:
            for gamma in GAMMA_GRID:
                for w in W_GRID:
                    key = f"{p0}|{eps}|{gamma}|{w}"
                    if key not in res:
                        res[key] = [run(p0, eps, gamma, w, s) for s in SEEDS]
                        json.dump(res, open(OUT, "w"))
                    rows = res[key]
                    last = lambda f: np.mean(
                        [np.mean([r[f] for r in t[-5:]]) for t in rows])
                    print(f"{p0:>4.1f} {eps:>5.2f} {gamma:>6.2f} {w:>5.2f} | "
                          f"{last('pop_std'):>7.3f} {last('pop_mean'):>8.3f} "
                          f"{last('mass_h'):>6.2f} {last('mass_t'):>6.2f} "
                          f"{last('cross_h'):>7.2f} {last('cross_t'):>7.2f} "
                          f"{last('contact_h'):>6.2f} {last('contact_t'):>6.2f} "
                          f"{last('clusters'):>5.1f}", flush=True)

    last = lambda f, key: np.mean([np.mean([r[f] for r in t[-5:]]) for t in res[key]])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = {0.1: "#1f77b4", 0.2: "#2ca02c", 0.3: "#d62728"}
    for eps in EPS_GRID:
        t = res[f"0.5|{eps}|1.5|0.3"][0]
        ax[0].plot([r["pop_std"] for r in t], c=colors[eps], lw=1.2,
                   label=f"P0=0.5, eps={eps}")
        t = res[f"0.8|{eps}|1.5|0.3"][0]
        ax[0].plot([r["pop_std"] for r in t], "--", c=colors[eps], lw=1.2,
                   label=f"P0=0.8, eps={eps}")
    ax[0].set(xlabel="round", ylabel="pop_std",
              title="pop_std vs round (gamma=1.5, w=0.3, seed 0)")
    ax[0].legend(frameon=False, fontsize=7)
    for eps in EPS_GRID:
        t = res[f"0.5|{eps}|1.5|0.3"][0]
        ax[1].plot([r["mass_h"] for r in t], c=colors[eps], lw=1.2,
                   label=f"mass_h eps={eps}")
        ax[1].plot([r["mass_t"] for r in t], "--", c=colors[eps], lw=1.2,
                   label=f"mass_t eps={eps}")
    ax[1].set(xlabel="round", ylabel="camp mass",
              title="camp masses (P0=0.5, gamma=1.5, w=0.3, seed 0)")
    ax[1].legend(frameon=False, fontsize=7)
    for p0, ls in [(0.5, "-"), (0.8, "--")]:
        for gamma, c in [(0.0, "#1f77b4"), (1.5, "#d62728")]:
            ax[2].plot(EPS_GRID,
                       [last("pop_mean", f"{p0}|{e}|{gamma}|0.3") for e in EPS_GRID],
                       ls, marker="o", c=c, ms=4, label=f"P0={p0}, gamma={gamma}")
    ax[2].axhline(X0.mean(), ls=":", c="gray", lw=1)
    ax[2].set(xlabel="epsilon", ylabel="final population mean",
              title="final mean at w=0.3 (dotted: initial mean)")
    ax[2].legend(frameon=False, fontsize=7)
    fig.suptitle(f"PolitiSky24 quote graph x P0-anchored MLP platform "
                 f"(n={N}, camps {C_H:.2f}/{C_T:.2f})")
    fig.savefig("experiments/competition/figs/politisky.png", dpi=130)
    print("saved experiments/competition/figs/politisky.png")


if __name__ == "__main__":
    main()
