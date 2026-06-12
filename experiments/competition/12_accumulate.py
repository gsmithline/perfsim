"""Block 12: does Gerstgrasser-style data accumulation rescue the AB x MLP collapse?

Same loop as block 9 (gamma=1.5, w=0.3, anchored MLP, gated feedback), three
training-data regimes: REPLACE trains on current opinions only, ACCUMULATE
samples one random past round per agent each train step, ROUND0-MIX averages
the loss 50/50 between current opinions and the pristine round-0 opinions.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/12_accumulate.py
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

N, P0, ANCHOR_W, TRAIN_STEPS = m9.N, m9.P0, m9.ANCHOR_W, m9.TRAIN_STEPS
GAMMA = 1.5
W = 0.3
ROUNDS = 300
TAIL = 0.3
EPS_GRID = [0.2, 0.3]
SEEDS = [0, 1, 2]
REGIMES = ["replace", "accumulate", "round0mix"]


def run(eps, seed, regime):
    pop = m9.build_pop(eps, GAMMA, seed)
    rng = np.random.default_rng(seed)
    x0 = np.array([pop.status[i] for i in range(N)])
    feats = torch.tensor(m9.make_features(x0, rng), dtype=torch.float32)
    net = m9.pretrain_base(rng, seed)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    t0 = torch.tensor(x0, dtype=torch.float32)

    hist = np.empty((ROUNDS + 1, N))
    hist[0] = x0
    traj = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        hist[t + 1] = x
        cur = torch.tensor(x, dtype=torch.float32)
        for _ in range(TRAIN_STEPS):
            opt.zero_grad()
            pred = net(feats).squeeze(1)
            if regime == "replace":
                fit = ((pred - cur) ** 2).mean()
            elif regime == "accumulate":  # one random past round per agent
                idx = rng.integers(0, t + 2, N)
                target = torch.tensor(hist[idx, np.arange(N)], dtype=torch.float32)
                fit = ((pred - target) ** 2).mean()
            else:  # round0mix
                fit = 0.5 * ((pred - cur) ** 2).mean() + 0.5 * ((pred - t0) ** 2).mean()
            loss = fit + ANCHOR_W * ((pred - P0) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1).numpy()
        gate = np.abs(m - x) < eps
        x_new = np.where(gate, (1 - W) * x + W * m, x)
        for i in range(N):
            pop.status[i] = float(x_new[i])
        pop.sts = x_new.copy()  # complete-graph cache used by partner selection
        x = x_new
        traj.append({
            "pop_mean": float(x.mean()), "pop_std": float(x.std()),
            "tail_mass": float((np.abs(x - P0) > TAIL).mean()),
            "pred_std": float(m.std()), "contact": float(gate.mean()),
            "clusters": m9.n_clusters(x),
        })
    return traj


def main():
    res = {}
    print(f"AB n={N}, gamma={GAMMA}, w={W}, P0={P0}, anchor_w={ANCHOR_W}, "
          f"{ROUNDS} rounds, seeds {SEEDS}")
    print(f"{'eps':>5} {'regime':>10} | {'pop_mean':>8} {'pop_std':>7} "
          f"{'tail':>6} {'pred_std':>8} {'contact':>7} {'clusters':>8}")
    for eps in EPS_GRID:
        for regime in REGIMES:
            for s in SEEDS:
                res[f"{eps}|{regime}|{s}"] = run(eps, s, regime)
            last = lambda f: np.mean(
                [np.mean([r[f] for r in res[f"{eps}|{regime}|{s}"][-5:]])
                 for s in SEEDS])
            print(f"{eps:>5.2f} {regime:>10} | {last('pop_mean'):>8.3f} "
                  f"{last('pop_std'):>7.3f} {last('tail_mass'):>6.2f} "
                  f"{last('pred_std'):>8.3f} {last('contact'):>7.2f} "
                  f"{last('clusters'):>8.1f}", flush=True)
    json.dump(res, open("experiments/competition/figs/accumulate.json", "w"))

    colors = {"replace": "#d62728", "accumulate": "#1f77b4", "round0mix": "#2ca02c"}
    fig, ax = plt.subplots(2, len(EPS_GRID), figsize=(11, 7), constrained_layout=True)
    for j, eps in enumerate(EPS_GRID):
        for regime in REGIMES:
            curves = np.array([[r["pop_std"] for r in res[f"{eps}|{regime}|{s}"]]
                               for s in SEEDS])
            ax[0, j].plot(curves.mean(0), c=colors[regime], label=regime)
            curves = np.array([[r["pop_mean"] for r in res[f"{eps}|{regime}|{s}"]]
                               for s in SEEDS])
            ax[1, j].plot(curves.mean(0), c=colors[regime], label=regime)
        ax[0, j].set(xlabel="round", ylabel="pop_std", title=f"eps={eps}")
        ax[1, j].axhline(P0, ls="--", c="black", lw=1)
        ax[1, j].set(xlabel="round", ylabel="pop_mean", title=f"eps={eps}")
        ax[0, j].legend(frameon=False, fontsize=8)
    fig.suptitle("training-data regime vs collapse to P0 (gamma=1.5, w=0.3, seed-avg)")
    fig.savefig("experiments/competition/figs/accumulate.png", dpi=130)
    print("saved experiments/competition/figs/accumulate.png")


if __name__ == "__main__":
    main()
