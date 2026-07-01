"""Does the training regime stop the AI-driven convergence at gamma=0?

At gamma=0, pop_eps=0.25, even a slightly open AI gate merges the population's
clusters into one mode under replace. replace refits on the collapsing population,
so the AI chases the collapse. accumulate pools all history; pristine keeps the
original real data in the fit. This sweeps regime x eps_AI over time to see whether
the anchor holds the population open as the gate opens.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/toy/toy_ols_eps_regime_time.py
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

N, ROUNDS, GAMMA, POP_EPS, W_AI = 300, 80, 0.0, 0.25, 0.5
A_SLOPE, NOISE_SD = 0.8, 0.15
SEEDS = [0, 1, 2]
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
    ostd, gatef, snaps = [], [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        if regime == "replace":
            Xtr, ytr = F, x
        elif regime == "accumulate":
            HX.append(F); Hy.append(x.copy())
            Xtr, ytr = np.vstack(HX), np.concatenate(Hy)
        else:
            Xtr = np.vstack([F, F0keep]); ytr = np.concatenate([x, x0[keep]])
        m, w = ols(Xtr, ytr, F)
        if eps_ai > 0:
            g = np.abs(m - x) < eps_ai
            x = np.where(g, (1 - W_AI) * x + W_AI * m, x)
            gatef.append(float(g.mean()))
        else:
            gatef.append(0.0)
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
        ostd.append(float(x.std()))
        if snap:
            snaps.append(x.copy())
    return np.array(ostd), np.array(gatef), snaps


def noai_baseline(seed):
    pop = build_pop(seed)
    out = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        out.append(float(x.std()))
    return np.array(out)


def main():
    print(f"gamma={GAMMA} pop_eps={POP_EPS} W_AI={W_AI} rounds={ROUNDS} seeds={SEEDS}  init op_std={INIT_STD:.3f}")
    base = np.mean([noai_baseline(s) for s in SEEDS], axis=0) / INIT_STD
    print(f"no-AI baseline dr: r0={base[0]:.2f} r20={base[20]:.2f} r79={base[-1]:.2f}\n")
    print(f"  {'regime':10s} {'eps_AI':>6} {'gate~':>6} | r0   r10  r20  r40  r79")

    dr_mean, snaps0 = {}, {}
    for regime in REGIMES:
        for ea in EPS_AI_GRID:
            runs = [run(regime, ea, s) for s in SEEDS]
            dr = np.mean([r[0] for r in runs], axis=0) / INIT_STD
            gate = np.mean([r[1] for r in runs], axis=0).mean()
            dr_mean[(regime, ea)] = dr
            snaps0[(regime, ea)] = run(regime, ea, SEEDS[0], snap=True)[2]
            cells = "  ".join(f"{dr[r]:.2f}" for r in [0, 10, 20, 40, ROUNDS - 1])
            print(f"  {regime:10s} {ea:>6} {gate:>6.2f} | {cells}")

    # Trajectories: one panel per eps_AI, a line per regime, no-AI dotted
    fig, axes = plt.subplots(1, len(EPS_AI_GRID), figsize=(5 * len(EPS_AI_GRID), 4.4), constrained_layout=True)
    colors = {"replace": "#d62728", "accumulate": "#1f77b4", "pristine": "#2ca02c"}
    for ax, ea in zip(axes, EPS_AI_GRID):
        for regime in REGIMES:
            ax.plot(range(ROUNDS), dr_mean[(regime, ea)], lw=2, c=colors[regime], label=regime)
        ax.plot(range(ROUNDS), base, ls=":", c="gray", lw=1.5, label="no AI")
        ax.set(xlabel="round", ylim=(0, 1.1), title=f"eps_AI = {ea}")
    axes[0].set_ylabel("op_std(t) / init")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Spread over time by regime as the AI gate opens (gamma={GAMMA}, pop_eps={POP_EPS})")
    fig.savefig(f"{FIGS}/eps_regime_traj.png", dpi=130)

    # Waterfall grid: rows = regime, cols = eps_AI
    bins = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(len(REGIMES), len(EPS_AI_GRID),
                             figsize=(4 * len(EPS_AI_GRID), 3.4 * len(REGIMES)), constrained_layout=True)
    for i, regime in enumerate(REGIMES):
        for j, ea in enumerate(EPS_AI_GRID):
            ax = axes[i][j]
            H = np.array([np.histogram(s, bins=bins, density=True)[0] for s in snaps0[(regime, ea)]])
            ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma", extent=[0, ROUNDS, 0, 1])
            if i == 0:
                ax.set_title(f"eps_AI = {ea}")
            if j == 0:
                ax.set_ylabel(f"{regime}\nopinion")
            if i == len(REGIMES) - 1:
                ax.set_xlabel("round")
    fig.suptitle(f"Opinion distribution over rounds by regime x eps_AI (gamma={GAMMA}, pop_eps={POP_EPS})")
    fig.savefig(f"{FIGS}/eps_regime_waterfall.png", dpi=130)
    print(f"\nsaved {FIGS}/eps_regime_traj.png and {FIGS}/eps_regime_waterfall.png")


if __name__ == "__main__":
    main()
