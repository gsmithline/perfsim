"""Emergence of gender predictiveness: dR2_gender(t) per round, Action
world (two-mode prior) vs War world (one-mode prior), same axes. The
dotted line is the preregistered structural criterion (dR2 > 0.01, War
Tree-3 criterion 5). Action anchored/frozen arms climb past it; War arms
and all controls stay at the estimator's null point.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch
from numpy.linalg import solve

BASE = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs/qwen"
BLUE, ORANGE, GREEN, GRAY = "#0E76B4", "#C96A14", "#2E7D4F", "#9A9A92"
CORE = ["Drama", "Romance", "Comedy", "Action", "Thriller", "War", "Crime",
        "Sci-Fi", "Adventure", "Mystery", "Children's"]

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 9, "ytick.labelsize": 9})


def cvr2(X, y, lam=1.0):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    r = np.random.default_rng(0); idx = r.permutation(len(y)); yh = np.zeros_like(y, float)
    for f in range(5):
        te = idx[f::5]; tr = np.setdiff1d(idx, te)
        w = solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ (y[tr] - y[tr].mean()))
        yh[te] = y[tr].mean() + X[te] @ w
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def dr2_series(op, Xg, Xgm):
    return np.array([cvr2(Xgm, op[r]) - cvr2(Xg, op[r]) for r in range(op.shape[0])])


def world(target, runs):
    feats = [g for g in CORE if g != target]
    t0 = torch.load(f"{BASE}/{runs[0][1]}/trajectory.pt", map_location="cpu", weights_only=False)
    prof = pd.DataFrame(t0["profiles"])
    male = (prof["gender"] == "M").values.astype(float)
    Xg = prof[feats].values.astype(float)
    Xgm = np.hstack([Xg, male[:, None]])
    out = {}
    for lab, tag in runs:
        t = torch.load(f"{BASE}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
        out[lab] = dr2_series(np.asarray(t["op_raw"], float), Xg, Xgm)
    # no-AI baseline series (local Deffuant, seed 0)
    spec = importlib.util.spec_from_file_location(
        "gp", "experiments/scripts/cluster_pipelines/_gated_pop.py")
    gp = importlib.util.module_from_spec(spec); spec.loader.exec_module(gp)
    Z = Xg - Xg.mean(0); nrm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    sim = nrm @ nrm.T; np.fill_diagonal(sim, -np.inf)
    nbrs = np.argsort(-sim, axis=1)[:, :10]
    G = nx.Graph(); G.add_nodes_from(range(len(prof)))
    for i, r in enumerate(nbrs):
        for j in r:
            G.add_edge(i, int(j))
    adjb = (torch.tensor(nx.to_numpy_array(G), dtype=torch.float32) > 0).float()
    x = torch.tensor(np.asarray(t0["innate"], float), dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    traj = []
    for _ in range(30):
        gp.ab_sweep(x, adjb, 0.10, 0.0, gen=g)
        traj.append(x.numpy().copy())
    out["no-AI"] = dr2_series(np.array(traj), Xg, Xgm)
    return out

action = world("Action", [
    ("$\\beta$=0.5", "mlat_e010_a040_rep_b0p5_s0"),
    ("$\\beta$=1", "mlat_e010_a040_rep_b1_s0"),
    ("frozen", "frz_qwen_e010_s0"),
    ("$\\beta$=0", "mlat_e010_a040_rep_b0_s0"),
    ("gender removed", "mlaaC_e010_a040_rep_b1_s0")])
war = world("War", [
    ("$\\beta$=0.5", "mlaw_e010_a040_rep_b0p5_s0"),
    ("$\\beta$=1", "mlaw_e010_a040_rep_b1_s0"),
    ("gender removed", "mlawC_e010_a040_rep_b1_s0")])

STYLE = {"$\\beta$=1": (BLUE, "-"), "$\\beta$=0.5": (BLUE, "--"),
         "frozen": (ORANGE, "-"), "$\\beta$=0": (GREEN, "-"),
         "gender removed": (GREEN, "--"), "no-AI": (GRAY, "-")}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True)
rounds = np.arange(1, 31)
for ax, data, title in [(axes[0], action, "Action world (two-mode prior: 0.25 / 0.65)"),
                        (axes[1], war, "War world (one-mode prior)")]:
    ax.axhline(0, color="#666666", lw=0.8)
    ax.axhline(0.01, color="#333333", lw=1.1, ls=":")
    for lab, series in data.items():
        c, ls = STYLE[lab]
        ax.plot(rounds[:len(series)], series, color=c, ls=ls, lw=2.0, label=lab)
        ax.annotate(lab, (rounds[len(series) - 1], series[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=c, va="center")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("round", fontsize=10)
    ax.set_xlim(1, 36.5)
axes[0].set_ylabel("$\\Delta R^2_{gender}$ on current opinions", fontsize=10)
axes[0].text(1.5, 0.0113, "preregistered bar (0.01)", fontsize=8, color="#333333")
fig.suptitle("The loop makes gender predictive -- but only where the prior offers two destinations\n"
             "(genres-only vs genres+gender, 5-fold CV ridge, identical folds; seed 0)", fontsize=11.5)
fig.savefig(f"{OUT}/dr2_emergence.png", dpi=140)
print(f"saved {OUT}/dr2_emergence.png")
