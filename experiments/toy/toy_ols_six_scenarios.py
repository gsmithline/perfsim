"""Six-scenario figure: population type (homophilic / neutral / heterophilic)
crossed with platform off vs on, at a fixed moderate AI gate.

Columns: no platform (clean Deffuant counterfactual) vs platform on (replace,
train on everyone, eps_AI fixed). Rows: gamma = 3 (homophily), 0 (neutral),
-1.5 (heterophily). magma = population opinion distribution over rounds; cyan
band (platform-on column) = platform prediction mean +/- std.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/toy/toy_ols_six_scenarios.py
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

N, ROUNDS, POP_EPS, W, EPS_AI = 300, 60, 0.25, 0.5, 0.10
A_SLOPE, NOISE_SD, SEED = 0.8, 0.15, 0
GAMMAS = [3.0, 0.0, -1.5]
GLAB = {3.0: "Homophilic ($\\gamma=3$)", 0.0: "Neutral ($\\gamma=0$)", -1.5: "Heterophilic ($\\gamma=-1.5$)"}

_rng = np.random.default_rng(0)
z = _rng.uniform(0.0, 1.0, N)
z_std = (z - z.mean()) / (z.std() + 1e-9)
x0 = np.clip(0.5 + A_SLOPE * (z - 0.5) + _rng.normal(0.0, NOISE_SD, N), 0.0, 1.0)
G = nx.complete_graph(N)
F = np.column_stack([np.ones(N), z_std])
INIT_STD = float(x0.std())


def build_pop(gamma, seed):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G)
    c = mc.Configuration()
    c.add_model_parameter("epsilon", POP_EPS)
    c.add_model_parameter("gamma", gamma)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x0[i])
    m.sts = x0.copy()
    m.iteration(False)
    return m


def ols(Xtr, ytr, Xpr):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0), w


def run(gamma, ai_on, seed):
    pop = build_pop(gamma, seed)
    x = x0.copy()
    snaps, pmean, pstd, drs = [], [], [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        if ai_on:
            m, _ = ols(F, x, F)                    # replace, train on everyone
            g = np.abs(m - x) < EPS_AI
            x = np.where(g, (1 - W) * x + W * m, x)
            for i in range(N):
                pop.status[i] = float(x[i])
            pop.sts = x.copy()
            pmean.append(float(m.mean())); pstd.append(float(m.std()))
        else:
            pmean.append(float("nan")); pstd.append(float("nan"))
        snaps.append(x.copy()); drs.append(float(x.std()) / INIT_STD)
    return snaps, np.array(pmean), np.array(pstd), drs


def main():
    bins = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(len(GAMMAS), 2, figsize=(9.5, 11), constrained_layout=True)
    for i, gamma in enumerate(GAMMAS):
        for j, ai_on in enumerate([False, True]):
            ax = axes[i][j]
            snaps, pmean, pstd, drs = run(gamma, ai_on, SEED)
            H = np.array([np.histogram(s, bins=bins, density=True)[0] for s in snaps])
            ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma", extent=[0, ROUNDS, 0, 1])
            if ai_on:
                rr = np.arange(ROUNDS)
                ax.plot(rr, pmean, c="#00e5ff", lw=1.2)
                ax.fill_between(rr, np.clip(pmean - pstd, 0, 1), np.clip(pmean + pstd, 0, 1),
                                color="#00e5ff", alpha=0.25)
            ax.text(0.97, 0.06, f"dr$_{{final}}$={drs[-1]:.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", color="white", fontsize=9)
            if i == 0:
                ax.set_title("No platform (clean Deffuant)" if not ai_on
                             else f"With platform (replace, $\\epsilon_{{AI}}$={EPS_AI})")
            if j == 0:
                ax.set_ylabel(f"{GLAB[gamma]}\nopinion")
            if i == len(GAMMAS) - 1:
                ax.set_xlabel("round")
    fig.suptitle(f"Six scenarios: population type $\\times$ platform (N={N}, pop $\\epsilon$={POP_EPS}, W={W})")
    fig.savefig(f"{FIGS}/six_scenarios.png", dpi=130)
    print(f"saved {FIGS}/six_scenarios.png")


if __name__ == "__main__":
    main()
