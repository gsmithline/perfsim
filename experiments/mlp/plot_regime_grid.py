"""Regime grid: rows = training regime (replace / accumulate / pristine),
columns = population type (homophily / neutral / heterophily). Fixed gate,
strong feature, OLS. Each panel: population waterfall + platform band, with
dr/vr and the classified equilibrium. Shows how the regime sets the equilibrium.
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

FIGS = "experiments/toy/figs"
N, ROUNDS, POP_EPS, W, EPS_AI, SEED = 300, 60, 0.25, 0.5, 0.10, 0
A_SLOPE, NOISE_SD = 0.8, 0.15
G = nx.complete_graph(N)
_rng = np.random.default_rng(0)
z = _rng.uniform(0.0, 1.0, N)
z_std = ((z - z.mean()) / (z.std() + 1e-9)).astype(np.float32)
x0 = np.clip(0.5 + A_SLOPE * (z - 0.5) + np.random.default_rng(1).normal(0.0, NOISE_SD, N), 0.0, 1.0).astype(np.float32)
F = np.column_stack([np.ones(N), z_std])
INIT = float(x0.std())


def build_pop(gamma):
    random.seed(SEED); np.random.seed(SEED)
    m = AlgorithmicBiasModel(G); c = mc.Configuration()
    c.add_model_parameter("epsilon", POP_EPS); c.add_model_parameter("gamma", gamma)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x0[i])
    m.sts = x0.copy(); m.iteration(False)
    return m


def ols(Xtr, ytr, Xpr):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0), w


def run(gamma, regime):
    pop = build_pop(gamma); x = x0.copy()
    keep = np.random.default_rng(SEED).permutation(N)[: N // 2]
    F0, x0k = F[keep], x0[keep]
    HX, Hy = [], []
    snaps, pmean, pstd = [], [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)], dtype=np.float32)
        if regime == "replace":
            Xtr, ytr = F, x
        elif regime == "accumulate":
            HX.append(F); Hy.append(x.copy()); Xtr, ytr = np.vstack(HX), np.concatenate(Hy)
        else:
            Xtr = np.vstack([F, F0]); ytr = np.concatenate([x, x0k])
        m, w = ols(Xtr, ytr, F)
        g = np.abs(m - x) < EPS_AI
        x = np.where(g, (1 - W) * x + W * m, x)
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
        snaps.append(x.copy()); pmean.append(float(m.mean())); pstd.append(float(m.std()))
    dr = float(x.std()) / INIT
    vr = pstd[-1] / float(x.std()) if x.std() > 1e-9 else float("nan")
    return snaps, np.array(pmean), np.array(pstd), dr, vr


def classify(dr, vr):
    wide = (not np.isnan(vr)) and vr >= 0.7
    div = dr >= 0.5
    return ("2 both diverse" if (div and wide) else "3 plat below" if div else
            "4 plat wide" if wide else "1 both collapsed")


GAMMAS = [("Homophily ($\\gamma$=3)", 3.0), ("Neutral ($\\gamma$=0)", 0.0), ("Heterophily ($\\gamma$=-1.5)", -1.5)]
REGIMES = ["replace", "accumulate", "pristine"]

bins = np.linspace(0, 1, 41)
fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
for i, regime in enumerate(REGIMES):
    for j, (glab, gamma) in enumerate(GAMMAS):
        ax = axes[i][j]
        snaps, pmean, pstd, dr, vr = run(gamma, regime)
        H = np.array([np.histogram(s, bins=bins, density=True)[0] for s in snaps])
        ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma", extent=[0, ROUNDS, 0, 1])
        rr = np.arange(ROUNDS)
        ax.plot(rr, pmean, c="#00e5ff", lw=1.2)
        ax.fill_between(rr, np.clip(pmean - pstd, 0, 1), np.clip(pmean + pstd, 0, 1), color="#00e5ff", alpha=0.25)
        ax.text(0.96, 0.05, f"dr={dr:.2f} vr={vr:.2f}\n-> {classify(dr, vr)}", transform=ax.transAxes,
                ha="right", va="bottom", color="white", fontsize=8)
        if i == 0:
            ax.set_title(glab, fontsize=11)
        if j == 0:
            ax.set_ylabel(f"{regime}\nopinion")
        if i == 2:
            ax.set_xlabel("round")
fig.suptitle(f"Regime grid (OLS, strong feature, $\\epsilon_{{AI}}$={EPS_AI}): "
             f"rows = training regime, columns = population type", fontsize=12)
fig.savefig(f"{FIGS}/regime_grid.png", dpi=130)
print(f"saved {FIGS}/regime_grid.png")
