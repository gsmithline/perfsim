"""Waterfalls for the train-on-users model: population distribution over rounds
with the platform's prediction band overlaid, so the platform-vs-population
decoupling (state 3 = platform below, state 4 = platform wider) is visible.

Same engagement loop as toy_ols_engagement.py (platform trains only on the gated
set). gamma=0, pop_eps=0.25, three regimes x three AI gates.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/toy/toy_ols_engagement_time.py
"""
import os, random
import numpy as np
import networkx as nx
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "experiments/toy"
FIGS = f"{HERE}/figs"
os.makedirs(FIGS, exist_ok=True)

N, ROUNDS, POP_EPS, W_AI = 300, 60, 0.25, 0.5
GAMMA = float(os.environ.get("GAMMA", "0.0"))
TRAIN_ON_USERS = os.environ.get("TRAIN_ON_USERS", "1") == "1"
TAG = f"g{GAMMA}_{'users' if TRAIN_ON_USERS else 'all'}"
D = 2
A_SLOPE, NOISE_SD = 0.8, 0.15
SEEDS = [0, 1]
EPS_AI_GRID = [0.05, 0.10, 0.25]
REGIMES = ["replace", "accumulate", "pristine"]

_rng = np.random.default_rng(0)
z = _rng.uniform(0.0, 1.0, N)
z_std = (z - z.mean()) / (z.std() + 1e-9)
x0 = np.clip(0.5 + A_SLOPE * (z - 0.5) + _rng.normal(0.0, NOISE_SD, N), 0.0, 1.0)
G = nx.complete_graph(N)
F = np.column_stack([np.ones(N), z_std])
INIT_STD = float(x0.std())


def build_pop(seed):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G)
    c = mc.Configuration()
    c.add_model_parameter("epsilon", POP_EPS)
    c.add_model_parameter("gamma", GAMMA)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x0[i])
    m.sts = x0.copy()
    m.iteration(False)
    return m


def ols(Xtr, ytr, Xpr):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0), w


def run(regime, eps_ai, seed, snap=False):
    pop = build_pop(seed)
    x = x0.copy()
    keep = np.random.default_rng(seed).permutation(N)[: N // 2]
    F0keep = F[keep]
    HX, Hy = [], []
    U = np.arange(N)
    ostd, pstd, pmean, snaps = [], [], [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        tr = U if (TRAIN_ON_USERS and len(U) >= D + 2) else np.arange(N)
        if regime == "replace":
            Xtr, ytr = F[tr], x[tr]
        elif regime == "accumulate":
            HX.append(F[tr]); Hy.append(x[tr].copy())
            Xtr, ytr = np.vstack(HX), np.concatenate(Hy)
        else:
            Xtr = np.vstack([F[tr], F0keep]); ytr = np.concatenate([x[tr], x0[keep]])
        m, w = ols(Xtr, ytr, F)
        g = np.abs(m - x) < eps_ai
        U = np.where(g)[0]
        x = np.where(g, (1 - W_AI) * x + W_AI * m, x)
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
        ostd.append(float(x.std())); pstd.append(float(m.std())); pmean.append(float(m.mean()))
        if snap:
            snaps.append(x.copy())
    return np.array(ostd), np.array(pstd), np.array(pmean), snaps


def main():
    # trajectories (averaged) for op_std vs pred_std
    traj = {}
    for regime in REGIMES:
        for ea in EPS_AI_GRID:
            runs = [run(regime, ea, s) for s in SEEDS]
            traj[(regime, ea)] = (np.mean([r[0] for r in runs], 0) / INIT_STD,
                                  np.mean([r[1] for r in runs], 0) / INIT_STD)

    fig, axes = plt.subplots(1, len(EPS_AI_GRID), figsize=(5 * len(EPS_AI_GRID), 4.4), constrained_layout=True)
    colors = {"replace": "#d62728", "accumulate": "#1f77b4", "pristine": "#2ca02c"}
    for ax, ea in zip(axes, EPS_AI_GRID):
        for regime in REGIMES:
            o, p = traj[(regime, ea)]
            ax.plot(range(ROUNDS), o, c=colors[regime], lw=2, label=f"{regime} pop")
            ax.plot(range(ROUNDS), p, c=colors[regime], lw=1.5, ls="--", label=f"{regime} platform")
        ax.set(xlabel="round", ylim=(0, 1.3), title=f"eps_AI = {ea}")
    axes[0].set_ylabel("std / init  (solid=population, dashed=platform)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(f"Train-on-users: population vs platform spread (gamma={GAMMA}, pop_eps={POP_EPS})")
    fig.savefig(f"{FIGS}/engagement_traj_{TAG}.png", dpi=130)

    # waterfalls: population dist + platform prediction band
    bins = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(len(REGIMES), len(EPS_AI_GRID),
                             figsize=(4 * len(EPS_AI_GRID), 3.4 * len(REGIMES)), constrained_layout=True)
    for i, regime in enumerate(REGIMES):
        for j, ea in enumerate(EPS_AI_GRID):
            ax = axes[i][j]
            _, pstd, pmean, snaps = run(regime, ea, SEEDS[0], snap=True)
            H = np.array([np.histogram(s, bins=bins, density=True)[0] for s in snaps])
            ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma", extent=[0, ROUNDS, 0, 1])
            rr = np.arange(ROUNDS)
            ax.plot(rr, pmean, c="#00e5ff", lw=1.2)
            ax.fill_between(rr, np.clip(pmean - pstd, 0, 1), np.clip(pmean + pstd, 0, 1),
                            color="#00e5ff", alpha=0.25)
            if i == 0:
                ax.set_title(f"eps_AI = {ea}")
            if j == 0:
                ax.set_ylabel(f"{regime}\nopinion")
            if i == len(REGIMES) - 1:
                ax.set_xlabel("round")
    lab = {-1.5: "HETEROPHILY", 0.0: "NEUTRAL", 3.0: "HOMOPHILY"}.get(GAMMA, "")
    mode = "train-on-USERS" if TRAIN_ON_USERS else "train-on-EVERYONE"
    fig.suptitle(f"{mode} waterfalls, gamma={GAMMA} {lab}. magma = population, cyan band = platform pred mean +/- std")
    fig.savefig(f"{FIGS}/engagement_waterfall_{TAG}.png", dpi=130)
    print(f"saved {FIGS}/engagement_traj_{TAG}.png and {FIGS}/engagement_waterfall_{TAG}.png")


if __name__ == "__main__":
    main()
