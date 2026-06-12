"""Block 13: two prior-anchored MLP platforms vs the AB (Deffuant) population.

Distinct priors (PA=0.3, PB=0.7) vs shared (PA=PB=0.7). One platform is known
to collapse the population to its prior above eps~0.2 (block 9); in the linear
FJ LLM runs distinct priors rescued diversity where a shared prior gave none.
Each platform trains only on its own audience (agents inside its gate) and an
agent blends toward the mean of the in-gate platform predictions.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/13_two_priors.py
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
    "ab09", "experiments/competition/09_ab_mlp_loop.py")
ab09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab09)

N = ab09.N
ROUNDS = 300
W = 0.3
ANCHOR_W = 0.3
TRAIN_STEPS = 10
MIN_AUD = 5
TAIL = 0.35

EPS_GRID = [0.2, 0.3]
GAMMA_GRID = [0.0, 1.5]
CONDS = {"distinct": (0.3, 0.7), "shared": (0.7, 0.7)}
SEEDS = [0, 1, 2]


def pretrain(prior, rng, torch_seed, n_corpus=4000, steps=400):
    torch.manual_seed(torch_seed)
    y = np.clip(rng.normal(prior, 0.15, n_corpus), 0.01, 0.99)
    feats = torch.tensor(ab09.make_features(y, rng), dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)
    net = ab09.mlp()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(feats).squeeze(1) - target) ** 2).mean()
        loss.backward()
        opt.step()
    return net


def run(eps, gamma, priors, seed):
    pop = ab09.build_pop(eps, gamma, seed)
    rng = np.random.default_rng(seed)
    x0 = np.array([pop.status[i] for i in range(N)])
    feats = torch.tensor(ab09.make_features(x0, rng), dtype=torch.float32)
    nets = [pretrain(p, rng, seed * 7 + k) for k, p in enumerate(priors)]
    opts = [torch.optim.Adam(n.parameters(), lr=3e-3) for n in nets]

    traj = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        xt = torch.tensor(x, dtype=torch.float32)
        ms, gates = [], []
        for net in nets:
            with torch.no_grad():
                ms.append(net(feats).squeeze(1).numpy())
            gates.append(np.abs(ms[-1] - x) < eps)
        for net, opt, prior, gate in zip(nets, opts, priors, gates):
            if gate.sum() < MIN_AUD:
                continue
            aud = torch.tensor(gate)
            for _ in range(TRAIN_STEPS):
                opt.zero_grad()
                pred = net(feats[aud]).squeeze(1)
                loss = ((pred - xt[aud]) ** 2).mean() \
                    + ANCHOR_W * ((pred - prior) ** 2).mean()
                loss.backward()
                opt.step()
        n_in = gates[0].astype(float) + gates[1].astype(float)
        m_sum = np.where(gates[0], ms[0], 0.0) + np.where(gates[1], ms[1], 0.0)
        blend_to = m_sum / np.maximum(n_in, 1.0)
        x_new = np.where(n_in > 0, (1 - W) * x + W * blend_to, x)
        for i in range(N):
            pop.status[i] = float(x_new[i])
        pop.sts = x_new.copy()  # complete-graph cache used by partner selection
        x = x_new
        traj.append({
            "pop_mean": float(x.mean()), "pop_std": float(x.std()),
            "clusters": ab09.n_clusters(x),
            "tail_mass": float((np.abs(x - 0.5) > TAIL).mean()),
            "share_A": float(gates[0].mean()), "share_B": float(gates[1].mean()),
            "pred_mean_A": float(ms[0].mean()), "pred_mean_B": float(ms[1].mean()),
            "platform_gap": float(abs(ms[0].mean() - ms[1].mean())),
            "contact_any": float((gates[0] | gates[1]).mean()),
        })
    return traj


def main():
    res = {}
    print(f"AB n={N} complete graph, two platforms, w={W}, anchor_w={ANCHOR_W}, "
          f"{ROUNDS} rounds, seeds {SEEDS}")
    print(f"{'gamma':>6} {'eps':>5} {'cond':>9} | {'pop_std':>7} {'clust':>5} "
          f"{'tail':>5} {'mean':>6} {'shA':>5} {'shB':>5} {'predA':>6} "
          f"{'predB':>6} {'gap':>6} {'any':>5}")
    for gamma in GAMMA_GRID:
        for eps in EPS_GRID:
            for cond, priors in CONDS.items():
                rows = [run(eps, gamma, priors, s) for s in SEEDS]
                res[f"{eps}|{gamma}|{cond}"] = rows
                last = lambda f: np.mean(
                    [np.mean([r[f] for r in t[-5:]]) for t in rows])
                print(f"{gamma:>6.2f} {eps:>5.2f} {cond:>9} | "
                      f"{last('pop_std'):>7.3f} {last('clusters'):>5.1f} "
                      f"{last('tail_mass'):>5.2f} {last('pop_mean'):>6.3f} "
                      f"{last('share_A'):>5.2f} {last('share_B'):>5.2f} "
                      f"{last('pred_mean_A'):>6.3f} {last('pred_mean_B'):>6.3f} "
                      f"{last('platform_gap'):>6.3f} {last('contact_any'):>5.2f}",
                      flush=True)
    json.dump(res, open("experiments/competition/figs/two_priors.json", "w"))

    fig, ax = plt.subplots(2, 4, figsize=(18, 7), constrained_layout=True)
    colors = {"distinct": "#1f77b4", "shared": "#d62728"}
    for j, (gamma, eps) in enumerate(
            [(g, e) for g in GAMMA_GRID for e in EPS_GRID]):
        for cond in CONDS:
            rows = res[f"{eps}|{gamma}|{cond}"]
            for i, f in enumerate(["pop_std", "platform_gap"]):
                curves = np.array([[r[f] for r in t] for t in rows])
                ax[i, j].plot(curves.mean(0), c=colors[cond], label=cond)
                for c in curves:
                    ax[i, j].plot(c, c=colors[cond], alpha=0.25, lw=0.6)
        ax[0, j].set(title=f"gamma={gamma}, eps={eps}", ylabel="pop_std",
                     ylim=(0, 0.35))
        ax[1, j].set(xlabel="round", ylabel="platform_gap", ylim=(-0.01, 0.45))
        ax[0, j].legend(frameon=False, fontsize=8)
    fig.suptitle("Two prior-anchored platforms: distinct (0.3/0.7) vs shared (0.7)")
    fig.savefig("experiments/competition/figs/two_priors.png", dpi=130)
    print("saved experiments/competition/figs/two_priors.png")


if __name__ == "__main__":
    main()
