"""Block 8d: constant anchor in opinion space. Same HK + MLP loop as block 8b,
but the anchor term is ANCHOR_W * (pred - P0)^2 directly, no base net. The
prior is a fixed point in opinion space, so it can sit outside every agent's
confidence window and the collapse mode is no longer structurally blocked.

Question: collapse-in-on-itself (predictions drift to P0, contact -> 0, loss
stuck high) vs lock-in (model keeps contact and ratchets the population to P0).
Which cells show which mode, and is the detachment sharp.

Run: python experiments/competition/08d_constant_anchor.py
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


def pretrain_on_pop(feats, x0, seed, steps=400):
    torch.manual_seed(seed)
    target = torch.tensor(x0, dtype=torch.float32)
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
    net = pretrain_on_pop(feats, x0, seed)
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
                + anchor_w * ((pred - P0) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1).numpy()
        gate = np.abs(m - x) < eps
        x_new = np.where(gate, (1 - w) * x + w * m, x)
        for i in range(N):
            pop.status[i] = float(x_new[i])
        x = x_new
        traj.append({
            "pop_mean": float(x.mean()), "pop_std": float(x.std()),
            "clusters": n_clusters(x),
            "pred_mean": float(m.mean()), "pred_std": float(m.std()),
            "loss": float(loss.item()), "contact": float(gate.mean()),
            "dist_prior": float(abs(x.mean() - P0)),
            "pred_to_prior": float(abs(m.mean() - P0)),
        })
    return traj


def main():
    res = {}
    print(f"HK n={N} ER(p={P_EDGE}), constant prior P0={P0}, "
          f"anchors {ANCHOR_GRID}, {ROUNDS} rounds, seeds {SEEDS}")
    print(f"{'eps':>5} {'w':>5} {'aw':>4} | {'clusters':>8} {'pop_std':>7} "
          f"{'pop_mean':>8} {'pred_mean':>9} {'contact':>7} "
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
                      f"{last('contact'):>7.2f} "
                      f"{last('dist_prior'):>7.3f} {last('pred_to_prior'):>6.3f} "
                      f"{last('pred_std'):>8.3f} {last('loss'):>7.4f}", flush=True)
    json.dump(res, open("experiments/competition/figs/constant_anchor.json", "w"))

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
        ax[1, j].set(xlabel="round", ylim=(-0.02, 1.02))
    ax[0, 0].set_ylabel("opinion")
    ax[1, 0].set_ylabel("fraction")
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Constant anchor P0=0.9: collapse-in-on-itself vs lock-in")
    fig.savefig("experiments/competition/figs/constant_anchor.png", dpi=130)
    print("saved experiments/competition/figs/constant_anchor.png")


if __name__ == "__main__":
    main()
