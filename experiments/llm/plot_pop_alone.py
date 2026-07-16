"""Act I fig 1: the population alone (no model anywhere in the loop).

Local Deffuant-AB (experiments/scripts/cluster_pipelines/_gated_pop.ab_sweep)
on the real ML-Action kNN graph (723 agents, LCC, innate = Action mean
rating rescaled to [0,1] -- exact replica of load_movielens_setup in
run_pokec_gated_lm.py). Sweep eps x gamma, 5 seeds, 30 rounds; report
dr(30) = op_std(30) / op_std(0).

CLAIM: heterophilic populations (gamma=-1.5) self-collapse; homophilic
ones (gamma=+1.5) stay diverse at every mixing rate. No AI involved.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/llm/plot_pop_alone.py
Pure numpy/torch -- no transformers, no model download.
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch

ML = "experiments/data/movielens/ml-100k"
OUT = "experiments/llm/figs"
EPS = [0.05, 0.10, 0.20, 0.30, 0.40]
GAMMAS = [(-1.5, "heterophily  $\\gamma=-1.5$", "#c0392b"),
          (0.0, "neutral  $\\gamma=0$", "#555555"),
          (1.5, "homophily  $\\gamma=+1.5$", "#2980b9")]
SEEDS = range(5)
ROUNDS = 30
TARGET = "Action"
CORE = ["Drama", "Romance", "Comedy", "Action", "Thriller", "War", "Crime",
        "Sci-Fi", "Adventure", "Mystery", "Children's"]

spec = importlib.util.spec_from_file_location(
    "gp", "experiments/scripts/cluster_pipelines/_gated_pop.py")
gp = importlib.util.module_from_spec(spec); spec.loader.exec_module(gp)


def ml_action_setup():
    """Replicates load_movielens_setup(target='Action', knn=10) without torch
    model deps: genre-mean profiles, cosine kNN graph on the other genres,
    LCC restriction, innate = (Action mean - 1) / 4."""
    gen = pd.read_csv(f"{ML}/u.genre", sep="|", names=["name", "gid"], encoding="latin-1")
    genres = list(gen.sort_values("gid")["name"])
    items = pd.read_csv(f"{ML}/u.item", sep="|", encoding="latin-1", header=None)
    gmat = pd.DataFrame(items.iloc[:, 5:5 + len(genres)].values,
                        index=items[0].values, columns=genres)
    rat = pd.read_csv(f"{ML}/u.data", sep="\t", names=["uid", "iid", "r", "t"]).merge(
        gmat, left_on="iid", right_index=True)
    P = pd.DataFrame({g: rat[rat[g] == 1].groupby("uid")["r"].mean() for g in CORE}).dropna()
    feats = [g for g in CORE if g != TARGET]
    Zc = P[feats].values - P[feats].values.mean(0)
    norm = Zc / (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)
    nbrs = np.argsort(-sim, axis=1)[:, :10]
    graph = nx.Graph(); graph.add_nodes_from(range(len(P)))
    for i, row in enumerate(nbrs):
        for j in row:
            graph.add_edge(i, int(j))
    lcc = sorted(max(nx.connected_components(graph), key=len))
    h = nx.relabel_nodes(graph.subgraph(lcc).copy(), {n: k for k, n in enumerate(lcc)})
    Pl = P.iloc[lcc]
    innate = ((Pl[TARGET].values - 1.0) / 4.0).astype(np.float64)
    adj = (torch.tensor(nx.to_numpy_array(h, nodelist=range(len(Pl))),
                        dtype=torch.float32) > 0).float()
    return torch.tensor(innate, dtype=torch.float32), adj


innate, adj = ml_action_setup()
init_std = float(innate.std())
print(f"ML-Action N={len(innate)}  innate std={init_std:.4f}")

results = {}   # (gamma, eps) -> list of dr(30) per seed
for gamma, _, _ in GAMMAS:
    for eps in EPS:
        drs = []
        for seed in SEEDS:
            x = innate.clone()
            g = torch.Generator().manual_seed(seed)
            for _ in range(ROUNDS):
                gp.ab_sweep(x, adj, eps, gamma, gen=g)
            drs.append(float(x.std()) / init_std)
        results[(gamma, eps)] = drs
        print(f"gamma={gamma:+.1f} eps={eps:.2f}  dr(30)={np.mean(drs):.3f} "
              f"+- {np.std(drs):.3f}")

# ---- data json next to the figure ----------------------------------------
data = {"dataset": "ML-100k Action LCC", "n": int(len(innate)),
        "innate_std": init_std, "rounds": ROUNDS, "seeds": list(SEEDS),
        "dynamics": "_gated_pop.ab_sweep (Deffuant-AB, graph neighbors)",
        "cells": [{"gamma": gm, "eps": e,
                   "dr30_per_seed": results[(gm, e)],
                   "dr30_mean": float(np.mean(results[(gm, e)])),
                   "dr30_sd": float(np.std(results[(gm, e)]))}
                  for gm, _, _ in GAMMAS for e in EPS]}
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/fig_pop_alone.json", "w") as fh:
    json.dump(data, fh, indent=2)

# ---- figure: dr(30) vs eps, three gamma lines, +-1sd band -----------------
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "font.size": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})
fig, ax = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
for gamma, label, color in GAMMAS:
    mu = np.array([np.mean(results[(gamma, e)]) for e in EPS])
    sd = np.array([np.std(results[(gamma, e)]) for e in EPS])
    ax.plot(EPS, mu, "-o", color=color, lw=2.0, ms=4.5, label=label)
    ax.fill_between(EPS, mu - sd, mu + sd, color=color, alpha=0.18, lw=0)
ax.axhline(1.0, color="#999999", lw=0.8, ls="--")
ax.set_xlabel("$\\epsilon$ (peer confidence radius)", fontsize=10)
ax.set_ylabel("diversity kept  dr(30)", fontsize=10)
ax.set_xticks(EPS)
ax.set_ylim(0, 1.08)
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_title("With no model in the loop, heterophilic\n"
             "populations self-collapse; homophilic stay wide", fontsize=10)
fig.savefig(f"{OUT}/fig_pop_alone.png", dpi=140)
print(f"saved {OUT}/fig_pop_alone.png and fig_pop_alone.json")
