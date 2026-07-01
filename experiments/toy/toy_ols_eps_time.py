"""Time-resolved view at gamma=0 (no homophily/heterophily): how the population
evolves over rounds for different epsilon settings, not just the endpoint.

Separate gates: pop_eps drives the Deffuant population, eps_AI gates the AI.
Five settings contrast a narrow vs wide population gate with no AI / tight AI /
wide AI, to see WHEN and HOW the population collapses (or holds).

Outputs op_std(t) trajectories and an opinion-distribution waterfall per setting.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/toy/toy_ols_eps_time.py
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

N, ROUNDS, GAMMA, W_AI = 300, 50, 0.0, 0.5
A_SLOPE, NOISE_SD = 0.8, 0.15
SEEDS = [0, 1, 2]
REGIME = "replace"

# (label, pop_eps, eps_AI)
SETTINGS = [
    ("pop 0.08 / no AI",   0.08, 0.00),
    ("pop 0.08 / AI 0.25", 0.08, 0.25),
    ("pop 0.25 / no AI",   0.25, 0.00),
    ("pop 0.25 / AI 0.05", 0.25, 0.05),
    ("pop 0.25 / AI 0.25", 0.25, 0.25),
]

_rng = np.random.default_rng(0)
z = _rng.uniform(0.0, 1.0, N)
z_std = (z - z.mean()) / (z.std() + 1e-9)
x0 = np.clip(0.5 + A_SLOPE * (z - 0.5) + _rng.normal(0.0, NOISE_SD, N), 0.0, 1.0)
G = nx.complete_graph(N)
F = np.column_stack([np.ones(N), z_std])
INIT_STD = float(x0.std())


def build_pop(eps, seed):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G)
    c = mc.Configuration()
    c.add_model_parameter("epsilon", eps)
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


def run(pop_eps, eps_ai, seed, snap=False):
    pop = build_pop(pop_eps, seed)
    x = x0.copy()
    ostd, omean, gatef, snaps = [], [], [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        m, w = ols(F, x, F)                        # replace regime: fit current pop
        if eps_ai > 0:
            g = np.abs(m - x) < eps_ai
            x = np.where(g, (1 - W_AI) * x + W_AI * m, x)
            gatef.append(float(g.mean()))
        else:
            gatef.append(0.0)
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
        ostd.append(float(x.std())); omean.append(float(x.mean()))
        if snap:
            snaps.append(x.copy())
    return np.array(ostd), np.array(omean), np.array(gatef), snaps


def main():
    print(f"gamma={GAMMA} regime={REGIME} W_AI={W_AI} seeds={SEEDS}  init op_std={INIT_STD:.3f}")
    print(f"dr = op_std(t)/init.  rounds shown: 0 5 10 20 30 49\n")
    print(f"  {'setting':22s} {'gate~':>6} | " + "  ".join(f"r{r:<3}" for r in [0, 5, 10, 20, 30, 49]))

    traj_mean, snaps_seed0 = {}, {}
    for label, pe, ea in SETTINGS:
        runs = [run(pe, ea, s) for s in SEEDS]
        dr = np.mean([r[0] for r in runs], axis=0) / INIT_STD
        gate = np.mean([r[2] for r in runs], axis=0)
        traj_mean[label] = dr
        _, _, _, snaps_seed0[label] = run(pe, ea, SEEDS[0], snap=True)
        cells = "  ".join(f"{dr[r]:.2f}" for r in [0, 5, 10, 20, 30, 49])
        print(f"  {label:22s} {gate.mean():6.2f} | {cells}")

    # Figure 1: dr(t) trajectories
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label, _, _ in SETTINGS:
        ax.plot(range(ROUNDS), traj_mean[label], marker="", lw=2, label=label)
    ax.axhline(1.0, ls=":", c="gray", lw=1)
    ax.set(xlabel="round", ylabel="op_std(t) / init", ylim=(0, 1.1),
           title=f"Population spread over time (gamma={GAMMA}, {REGIME}, W_AI={W_AI})")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(f"{FIGS}/eps_time_traj.png", dpi=130)

    # Figure 2: opinion-distribution waterfall per setting (seed 0)
    bins = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(1, len(SETTINGS), figsize=(4 * len(SETTINGS), 4.2), constrained_layout=True)
    for ax, (label, _, _) in zip(axes, SETTINGS):
        snaps = snaps_seed0[label]
        H = np.array([np.histogram(s, bins=bins, density=True)[0] for s in snaps])  # rounds x bins
        ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, ROUNDS, 0, 1])
        ax.set(xlabel="round", title=label)
    axes[0].set_ylabel("opinion")
    fig.suptitle(f"Opinion distribution over rounds (gamma={GAMMA}, {REGIME})")
    fig.savefig(f"{FIGS}/eps_time_waterfall.png", dpi=130)
    print(f"\nsaved {FIGS}/eps_time_traj.png and {FIGS}/eps_time_waterfall.png")


if __name__ == "__main__":
    main()
