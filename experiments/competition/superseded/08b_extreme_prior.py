"""Block 8b: extreme prior arm. Same HK + anchored-MLP loop as block 8, but the
base net is pretrained at P0 = 0.9, near the edge of opinion space, and the
anchor weight becomes a second axis (0.3 vs 1.0).

Question: collapse-in-on-itself (population far from prior, predictions drift
to prior, contact -> 0, loss stuck high) vs lock-in (model captures the cluster
nearest the prior and ratchets it to P0). Which cells show which mode.

Run: python experiments/competition/08b_extreme_prior.py
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
P0 = 0.9
TRAIN_STEPS = 10
SIGMA_F = 0.15

EPS_GRID = [0.2, 0.35, 0.5, 1.0]
W_GRID = [0.15, 0.3, 0.5]
ANCHOR_GRID = [0.3, 1.0]
SEEDS = [0, 1]

# representative cells (eps, w, anchor_w) for the trajectory figure
SHOW = [(0.2, 0.15, 0.3), (0.2, 0.5, 1.0), (0.5, 0.3, 1.0), (1.0, 0.5, 1.0)]


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


def run(eps, w, anchor_w, seed):
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
        x_prev = np.array([pop.status[i] for i in range(N)])
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        target = torch.tensor(x, dtype=torch.float32)
        for _ in range(TRAIN_STEPS):
            opt.zero_grad()
            pred = net(feats).squeeze(1)
            loss = ((pred - target) ** 2).mean() \
                + anchor_w * ((pred - base_pred) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1).numpy()
        gate = np.abs(m - x) < eps
        x_new = np.where(gate, (1 - w) * x + w * m, x)
        for i in range(N):
            pop.status[i] = float(x_new[i])
        x = x_new
        locked = gate & (np.abs(m - x) < np.abs(m - x_prev))
        traj.append({
            "pop_mean": float(x.mean()), "pop_std": float(x.std()),
            "clusters": n_clusters(x),
            "pred_mean": float(m.mean()), "pred_std": float(m.std()),
            "loss": float(loss.item()), "contact": float(gate.mean()),
            "dist_prior": float(abs(x.mean() - P0)),
            "pred_to_prior": float(abs(m.mean() - P0)),
            "frac_locked": float(locked.mean()),
        })
    return traj


def main():
    res = {}
    print(f"HK n={N} ER(p={P_EDGE}), extreme prior P0={P0}, "
          f"anchors {ANCHOR_GRID}, {ROUNDS} rounds, seeds {SEEDS}")
    print(f"{'eps':>5} {'w':>5} {'aw':>4} | {'clusters':>8} {'pop_std':>7} "
          f"{'pop_mean':>8} {'pred_mean':>9} {'contact':>7} {'locked':>7} "
          f"{'d_prior':>7} {'pr2pr':>6} {'pred_std':>8} {'loss':>7}")
    for eps in EPS_GRID:
        for w in W_GRID:
            for aw in ANCHOR_GRID:
                rows = [run(eps, w, aw, s) for s in SEEDS]
                res[f"{eps}|{w}|{aw}"] = rows
                last = lambda f: np.mean(
                    [np.mean([r[f] for r in t[-5:]]) for t in rows])
                print(f"{eps:>5.2f} {w:>5.2f} {aw:>4.1f} | "
                      f"{last('clusters'):>8.1f} {last('pop_std'):>7.3f} "
                      f"{last('pop_mean'):>8.3f} {last('pred_mean'):>9.3f} "
                      f"{last('contact'):>7.2f} {last('frac_locked'):>7.2f} "
                      f"{last('dist_prior'):>7.3f} {last('pred_to_prior'):>6.3f} "
                      f"{last('pred_std'):>8.3f} {last('loss'):>7.4f}", flush=True)
    json.dump(res, open("experiments/competition/figs/extreme_prior.json", "w"))

    fig, ax = plt.subplots(2, len(SHOW), figsize=(4.2 * len(SHOW), 7),
                           constrained_layout=True)
    for j, (eps, w, aw) in enumerate(SHOW):
        t = res[f"{eps}|{w}|{aw}"][0]
        rounds = range(len(t))
        ax[0, j].plot(rounds, [r["pop_mean"] for r in t], c="#1f77b4",
                      label="pop_mean")
        ax[0, j].plot(rounds, [r["pred_mean"] for r in t], c="#d62728",
                      label="pred_mean")
        ax[0, j].axhline(P0, ls="--", c="black", lw=1, label="P0")
        ax[0, j].set(title=f"eps={eps}, w={w}, anchor={aw}", xlabel="round",
                     ylim=(-1, 1))
        ax[1, j].plot(rounds, [r["contact"] for r in t], c="#2ca02c",
                      label="contact")
        ax[1, j].plot(rounds, [r["frac_locked"] for r in t], c="#9467bd",
                      label="frac_locked")
        ax[1, j].set(xlabel="round", ylim=(-0.02, 1.02))
    ax[0, 0].set_ylabel("opinion")
    ax[1, 0].set_ylabel("fraction")
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Extreme prior P0=0.9: collapse-in-on-itself vs lock-in")
    fig.savefig("experiments/competition/figs/extreme_prior.png", dpi=130)
    print("saved experiments/competition/figs/extreme_prior.png")


if __name__ == "__main__":
    main()
