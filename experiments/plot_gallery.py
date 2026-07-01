"""Four-equilibria gallery: one waterfall per state, population distribution (magma)
plus platform prediction band (cyan). Synthetic OLS-in-AB loop, train on everyone.
States 1/2/4 use a strong feature; state 3 uses a weak feature (the platform
cannot represent the population), which is what produces platform-below-population.
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
N, ROUNDS, POP_EPS, W, SEED = 300, 60, 0.25, 0.5, 0
G = nx.complete_graph(N)
_rng = np.random.default_rng(0)
z = _rng.uniform(0.0, 1.0, N)
z_std = (z - z.mean()) / (z.std() + 1e-9)
F = np.column_stack([np.ones(N), z_std])


def make_x0(a_slope, noise_sd):
    x0 = np.clip(0.5 + a_slope * (z - 0.5) + np.random.default_rng(1).normal(0.0, noise_sd, N), 0.0, 1.0)
    w0, *_ = np.linalg.lstsq(F, x0, rcond=None)
    R2 = float(1 - ((x0 - F @ w0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum())
    return x0, R2


def build_pop(gamma, x0):
    random.seed(SEED); np.random.seed(SEED)
    m = AlgorithmicBiasModel(G)
    c = mc.Configuration()
    c.add_model_parameter("epsilon", POP_EPS); c.add_model_parameter("gamma", gamma)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x0[i])
    m.sts = x0.copy(); m.iteration(False)
    return m


def ols(Xtr, ytr, Xpr):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0), w


def run(gamma, eps_ai, regime, a_slope, noise_sd):
    x0, R2 = make_x0(a_slope, noise_sd)
    init = float(x0.std())
    pop = build_pop(gamma, x0)
    x = x0.copy()
    keep = np.random.default_rng(SEED).permutation(N)[: N // 2]
    F0, x0k = F[keep], x0[keep]
    HX, Hy = [], []
    snaps, pmean, pstd = [], [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        if regime == "replace":
            Xtr, ytr = F, x
        elif regime == "accumulate":
            HX.append(F); Hy.append(x.copy()); Xtr, ytr = np.vstack(HX), np.concatenate(Hy)
        else:
            Xtr = np.vstack([F, F0]); ytr = np.concatenate([x, x0k])
        m, w = ols(Xtr, ytr, F)
        g = np.abs(m - x) < eps_ai
        x = np.where(g, (1 - W) * x + W * m, x)
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
        snaps.append(x.copy()); pmean.append(float(m.mean())); pstd.append(float(m.std()))
    dr = float(x.std()) / init
    vr = pstd[-1] / float(x.std()) if x.std() > 1e-9 else float("nan")
    return snaps, np.array(pmean), np.array(pstd), R2, dr, vr


SCEN = [
    ("1  both collapsed",                 0.0, 0.25, "replace",  0.8, 0.15),
    ("2  both diverse (platform tracks)", 3.0, 0.10, "replace",  0.8, 0.15),
    ("3  population diverse, platform below", 3.0, 0.10, "replace", 0.3, 0.25),
    ("4  population collapsed, platform wide", 0.0, 0.25, "pristine", 0.8, 0.15),
]

bins = np.linspace(0, 1, 41)
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for ax, (label, gamma, ea, regime, a, nz) in zip(axes.flat, SCEN):
    snaps, pmean, pstd, R2, dr, vr = run(gamma, ea, regime, a, nz)
    H = np.array([np.histogram(s, bins=bins, density=True)[0] for s in snaps])
    ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma", extent=[0, ROUNDS, 0, 1])
    rr = np.arange(ROUNDS)
    ax.plot(rr, pmean, c="#00e5ff", lw=1.3)
    ax.fill_between(rr, np.clip(pmean - pstd, 0, 1), np.clip(pmean + pstd, 0, 1), color="#00e5ff", alpha=0.25)
    ax.set_title(f"{label}\n$\\gamma$={gamma}, $\\epsilon_{{AI}}$={ea}, {regime}, $R^2$={R2:.2f}  "
                 f"(dr={dr:.2f}, vr={vr:.2f})", fontsize=10)
    ax.set(xlabel="round", ylabel="opinion")
fig.suptitle("Four equilibria: population distribution (magma) and platform band (cyan)", fontsize=13)
fig.savefig(f"{FIGS}/four_equilibria_gallery.png", dpi=130)
print(f"saved {FIGS}/four_equilibria_gallery.png")
