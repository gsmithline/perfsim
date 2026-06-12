"""Block 11: is the eps-threshold a property of mediation, or of retraining itself?

A DIRECT self-consuming: no population. Each round the MLP trains on its own
previous-round predictions (plus the P0 anchor); the predictions become next
round's data.
B MEDIATED: the block-9 loop at gamma=1.5, w=0.3, eps in {0.1, 0.3}.
C POPULATION-ONLY: gamma=1.5 AB dynamics, w=0; model trains but never feeds back.

Same MLP, anchor, and initial opinions x0 per seed (build_pop's uniform draw
depends only on the seed). Tail mass = fraction with |value - P0| > 0.3.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/11_self_consuming.py
"""

import importlib.util
import json
import os

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location(
    "ab_mlp_loop", "experiments/competition/09_ab_mlp_loop.py")
m9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m9)

ROUNDS = 300
GAMMA = 1.5
W = 0.3
EPS_LIST = [0.1, 0.3]
SEEDS = [0, 1, 2]
TAIL = 0.3


def init(eps, seed):
    pop = m9.build_pop(eps, GAMMA, seed)
    rng = np.random.default_rng(seed)
    x0 = np.array([pop.status[i] for i in range(m9.N)])
    feats = torch.tensor(m9.make_features(x0, rng), dtype=torch.float32)
    net = m9.pretrain_base(rng, seed)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    return pop, x0, feats, net, opt


def train_predict(net, opt, feats, target):
    for _ in range(m9.TRAIN_STEPS):
        opt.zero_grad()
        pred = net(feats).squeeze(1)
        loss = ((pred - target) ** 2).mean() \
            + m9.ANCHOR_W * ((pred - m9.P0) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(feats).squeeze(1).numpy()


def tail_mass(v):
    return float((np.abs(v - m9.P0) > TAIL).mean())


def run_direct(seed):
    _, x0, feats, net, opt = init(EPS_LIST[-1], seed)
    data = x0
    traj = []
    for _ in range(ROUNDS):
        m = train_predict(net, opt, feats,
                          torch.tensor(data, dtype=torch.float32))
        data = m
        traj.append({"std": float(m.std()), "mean": float(m.mean()),
                     "tail": tail_mass(m)})
    return traj


def run_loop(eps, w, seed):
    pop, x0, feats, net, opt = init(eps, seed)
    traj = []
    for _ in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(m9.N)])
        m = train_predict(net, opt, feats,
                          torch.tensor(x, dtype=torch.float32))
        gate = np.abs(m - x) < eps
        if w > 0:
            x = np.where(gate, (1 - w) * x + w * m, x)
            for i in range(m9.N):
                pop.status[i] = float(x[i])
            pop.sts = x.copy()  # complete-graph cache used by partner selection
        traj.append({"std": float(x.std()), "mean": float(x.mean()),
                     "tail": tail_mass(x), "pred_mean": float(m.mean()),
                     "pred_std": float(m.std()), "contact": float(gate.mean())})
    return traj


def half_life(traj):
    s0 = traj[0]["std"]
    for t, r in enumerate(traj):
        if r["std"] < 0.5 * s0:
            return t
    return None


def main():
    conds = {"direct": lambda s: run_direct(s)}
    for eps in EPS_LIST:
        conds[f"mediated_eps{eps}"] = lambda s, e=eps: run_loop(e, W, s)
        conds[f"pop_only_eps{eps}"] = lambda s, e=eps: run_loop(e, 0.0, s)

    res = {}
    print(f"n={m9.N}, P0={m9.P0}, anchor_w={m9.ANCHOR_W}, gamma={GAMMA}, "
          f"{ROUNDS} rounds, seeds {SEEDS}")
    print(f"{'condition':>18} | {'std0':>6} {'std_end':>7} {'mean_end':>8} "
          f"{'tail_end':>8} {'half-life':>9}")
    for name, fn in conds.items():
        rows = [fn(s) for s in SEEDS]
        res[name] = rows
        hl = [half_life(t) for t in rows]
        hl_s = "/".join("-" if h is None else str(h) for h in hl)
        avg = lambda f, k: np.mean([np.mean([r[f] for r in t[k]]) for t in rows])
        print(f"{name:>18} | {np.mean([t[0]['std'] for t in rows]):>6.3f} "
              f"{avg('std', slice(-5, None)):>7.4f} "
              f"{avg('mean', slice(-5, None)):>8.3f} "
              f"{avg('tail', slice(-5, None)):>8.3f} {hl_s:>9}", flush=True)
    json.dump(res, open("experiments/competition/figs/self_consuming.json", "w"))

    colors = {"direct": "#d62728", "mediated_eps0.1": "#1f77b4",
              "mediated_eps0.3": "#9467bd", "pop_only_eps0.1": "#7f7f7f",
              "pop_only_eps0.3": "#2ca02c"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, rows in res.items():
        std = np.mean([[r["std"] for r in t] for t in rows], 0)
        tail = np.mean([[r["tail"] for r in t] for t in rows], 0)
        lab = name + (" (pred)" if name == "direct" else " (pop)")
        ax[0].plot(std, c=colors[name], label=lab)
        ax[1].plot(tail, c=colors[name], label=lab)
    ax[0].set(xlabel="round", ylabel="std", yscale="log",
              title="dispersion vs round (seed avg)")
    ax[1].set(xlabel="round", ylabel=f"tail mass |v - P0| > {TAIL}",
              title="tail mass vs round")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("self-consuming (direct) vs mediated AB loop vs population-only")
    fig.savefig("experiments/competition/figs/self_consuming.png", dpi=130)
    print("saved experiments/competition/figs/self_consuming.png")


if __name__ == "__main__":
    main()
