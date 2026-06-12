"""Block 10: model-collapse-style generation curves through the mediated loop.

Same loop as 09 (AB population, gamma=1.5, anchored MLP platform) but with
per-round collapse metrics: pop_var, normalized 40-bin entropy, tail_mass
(fraction with |x - P0| > 0.3), pred_var, minority_mass (fraction outside the
largest gap-cluster). Grid: eps x w, 3 seeds, 300 rounds.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/10_collapse_metrics.py
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

_spec = importlib.util.spec_from_file_location("b9", "experiments/competition/09_ab_mlp_loop.py")
b9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b9)

GAMMA = 1.5
EPS_GRID = [0.10, 0.15, 0.20, 0.30, 0.40]
W_GRID = [0.0, 0.15, 0.3]
SEEDS = [0, 1, 2]
ROUNDS = 300
N_BINS = 40
TAIL = 0.3


def entropy(x):
    p = np.histogram(x, bins=N_BINS, range=(0, 1))[0] / len(x)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(N_BINS))


def minority_mass(x):
    s = np.sort(x)
    breaks = np.where(np.diff(s) > b9.GAP)[0]
    sizes = np.diff(np.concatenate([[0], breaks + 1, [len(s)]]))
    return float(1.0 - sizes.max() / len(s))


def run(eps, w, seed):
    pop = b9.build_pop(eps, GAMMA, seed)
    rng = np.random.default_rng(seed)
    x0 = np.array([pop.status[i] for i in range(b9.N)])
    feats = torch.tensor(b9.make_features(x0, rng), dtype=torch.float32)
    net = b9.pretrain_base(rng, seed)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    traj = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(b9.N)])
        target = torch.tensor(x, dtype=torch.float32)
        for _ in range(b9.TRAIN_STEPS):
            opt.zero_grad()
            pred = net(feats).squeeze(1)
            loss = ((pred - target) ** 2).mean() \
                + b9.ANCHOR_W * ((pred - b9.P0) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = net(feats).squeeze(1).numpy()
        gate = np.abs(m - x) < eps
        if w > 0:
            x_new = np.where(gate, (1 - w) * x + w * m, x)
            for i in range(b9.N):
                pop.status[i] = float(x_new[i])
            pop.sts = x_new.copy()  # complete-graph cache used by partner selection
            x = x_new
        traj.append({
            "pop_var": float(x.var()), "pop_std": float(x.std()),
            "entropy": entropy(x),
            "tail_mass": float((np.abs(x - b9.P0) > TAIL).mean()),
            "pred_var": float(m.var()), "pred_std": float(m.std()),
            "minority_mass": minority_mass(x),
            "clusters": b9.n_clusters(x), "contact": float(gate.mean()),
        })
    return traj


def half_time(series, frac=0.5):
    s = np.asarray(series)
    if s[0] <= 0:
        return None
    hit = np.where(s < frac * s[0])[0]
    return int(hit[0]) if len(hit) else None


def main():
    res = {}
    print(f"AB gamma={GAMMA}, n={b9.N}, P0={b9.P0}, {ROUNDS} rounds, seeds {SEEDS}")
    print(f"{'eps':>5} {'w':>5} | {'pop_std':>7} {'entropy':>7} {'tail':>6} "
          f"{'minority':>8} {'pred_std':>8} {'clusters':>8}")
    for eps in EPS_GRID:
        for w in W_GRID:
            rows = [run(eps, w, s) for s in SEEDS]
            res[f"{eps}|{w}"] = rows
            last = lambda f: np.mean([np.mean([r[f] for r in t[-5:]]) for t in rows])
            print(f"{eps:>5.2f} {w:>5.2f} | {last('pop_std'):>7.3f} "
                  f"{last('entropy'):>7.3f} {last('tail_mass'):>6.3f} "
                  f"{last('minority_mass'):>8.3f} {last('pred_std'):>8.3f} "
                  f"{last('clusters'):>8.1f}", flush=True)
    json.dump(res, open("experiments/competition/figs/collapse_metrics.json", "w"))

    # timing: first round each metric falls below half its round-0 value
    print("\nhalf-life rounds (metric < 0.5 * round-0 value), per seed")
    print(f"{'eps':>5} {'w':>5} | {'pop_var':>16} {'tail_mass':>16} {'pred_var':>16}")
    for eps in EPS_GRID:
        for w in W_GRID:
            rows = res[f"{eps}|{w}"]
            cols = []
            for f in ["pop_var", "tail_mass", "pred_var"]:
                ts = [half_time([r[f] for r in t]) for t in rows]
                cols.append(" ".join("  --" if v is None else f"{v:>4d}" for v in ts))
            print(f"{eps:>5.2f} {w:>5.2f} | {cols[0]:>16} {cols[1]:>16} {cols[2]:>16}")

    mean_curve = lambda e, w, f: np.mean(
        [[r[f] for r in t] for t in res[f"{e}|{w}"]], axis=0)
    final = lambda e, w, f: np.mean(
        [np.mean([r[f] for r in t[-5:]]) for t in res[f"{e}|{w}"]])
    cmap = plt.get_cmap("viridis")
    ecol = {e: cmap(i / (len(EPS_GRID) - 1)) for i, e in enumerate(EPS_GRID)}
    styles = {0.0: ":", 0.15: "--", 0.3: "-"}

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for e in EPS_GRID:
        ax[0, 0].plot(mean_curve(e, 0.3, "pop_std"), c=ecol[e], label=f"eps={e}")
        ax[1, 1].plot(mean_curve(e, 0.3, "pred_std"), c=ecol[e], label=f"eps={e}")
    ax[0, 0].set(xlabel="round", ylabel="pop_std",
                 title="(a) population std vs round, w=0.3")
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[1, 1].set(xlabel="round", ylabel="pred_std",
                 title="(d) platform output std vs round, w=0.3")
    ax[1, 1].legend(frameon=False, fontsize=8)
    for i, (f, lab) in enumerate([("tail_mass", "(b) final tail_mass"),
                                  ("entropy", "(c) final entropy")]):
        a = ax[0, 1] if i == 0 else ax[1, 0]
        for w in W_GRID:
            a.plot(EPS_GRID, [final(e, w, f) for e in EPS_GRID],
                   styles[w], marker="o", c="#d62728", label=f"w={w}")
        a.set(xlabel="epsilon", ylabel=f, title=f"{lab} vs epsilon (last 5 rounds)")
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Collapse metrics through the mediated loop, gamma={GAMMA}, P0={b9.P0}")
    fig.savefig("experiments/competition/figs/collapse_metrics.png", dpi=130)
    print("saved experiments/competition/figs/collapse_metrics.png")


if __name__ == "__main__":
    main()
