"""Why heterophily retains more variance than neutral (population property).
A) std retained vs gamma, pure population and coupled-with-platform overlaid.
B) acceptance rate vs gamma: fraction of selection mass inside the eps bound.
   Heterophily starves interactions (few merges) -> variance frozen high.
C) eps-robustness: std retained vs pop_eps for neutral vs heterophily. The
   ordering rides on neutral sitting in its consensus regime (eps near/above
   the bounded-confidence threshold); heterophily is stable across eps.
Complete graph N=300, same x0 as regime_grid, 3 seeds.
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
N, ROUNDS, POP_EPS, W, EPS_AI = 300, 100, 0.25, 0.5, 0.10
A_SLOPE, NOISE_SD = 0.8, 0.15
SEEDS = [0, 1, 2]
GAMMAS = [3.0, 1.5, 0.5, 0.0, -0.5, -1.5, -3.0]
EPS_GRID = [0.15, 0.20, 0.25, 0.30, 0.35]
G = nx.complete_graph(N)

z = np.random.default_rng(0).uniform(0.0, 1.0, N)
z_std = ((z - z.mean()) / (z.std() + 1e-9)).astype(np.float32)
x0 = np.clip(0.5 + A_SLOPE * (z - 0.5) + np.random.default_rng(1).normal(0.0, NOISE_SD, N), 0.0, 1.0).astype(np.float32)
F = np.column_stack([np.ones(N), z_std]); INIT = float(x0.std())


def build_pop(gamma, seed, eps=POP_EPS):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G); c = mc.Configuration()
    c.add_model_parameter("epsilon", eps); c.add_model_parameter("gamma", gamma)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x0[i])
    m.sts = x0.copy(); m.iteration(False)
    return m


def accept_rate(x, gamma, eps):
    D = np.abs(x[:, None] - x[None, :])
    Wt = np.maximum(D, 1e-5) ** (-gamma); np.fill_diagonal(Wt, 0.0)
    p = Wt / (Wt.sum(1, keepdims=True) + 1e-12)
    within = (D < eps).astype(np.float32); np.fill_diagonal(within, 0.0)
    return float((p * within).sum(1).mean())


def ols(Xpr, ytr):
    w, *_ = np.linalg.lstsq(F, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0)


def run_pure(gamma, seed, eps=POP_EPS):
    pop = build_pop(gamma, seed, eps); accs = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)], dtype=np.float32)
        if t >= ROUNDS - 40 and t % 5 == 0:
            accs.append(accept_rate(x, gamma, eps))
    return float(x.std()) / INIT, float(np.mean(accs))


def run_coupled(gamma, seed):
    pop = build_pop(gamma, seed); x = x0.copy()
    for t in range(60):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)], dtype=np.float32)
        m = ols(F, x); g = np.abs(m - x) < EPS_AI
        x = np.where(g, (1 - W) * x + W * m, x)
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
    return float(x.std()) / INIT


pure_sd, pure_ac, coup = {}, {}, {}
for g in GAMMAS:
    sd, ac = zip(*[run_pure(g, s) for s in SEEDS])
    pure_sd[g] = np.array(sd); pure_ac[g] = np.array(ac)
    coup[g] = np.array([run_coupled(g, s) for s in SEEDS])

eps_curve = {0.0: [], -1.5: []}
for e in EPS_GRID:
    for g in eps_curve:
        eps_curve[g].append(np.mean([run_pure(g, s, e)[0] for s in SEEDS]))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
gx = np.array(GAMMAS)
ps = np.array([pure_sd[g].mean() for g in GAMMAS]); pssd = np.array([pure_sd[g].std() for g in GAMMAS])
cs = np.array([coup[g].mean() for g in GAMMAS])
ax[0].errorbar(gx, ps, yerr=pssd, fmt="-o", c="k", capsize=3, label="pure population")
ax[0].plot(gx, cs, "-s", c="#1f77b4", label="coupled + platform")
ax[0].axvline(0, ls=":", c="gray"); ax[0].axhline(1, ls="--", c="gray", lw=1)
imin = int(np.argmin(ps)); ax[0].scatter([gx[imin]], [ps[imin]], s=140, facecolors="none", edgecolors="r", zorder=5)
ax[0].annotate("peak mixing\n(min variance)", (gx[imin], ps[imin]), textcoords="offset points",
               xytext=(6, 22), color="r", fontsize=8)
ax[0].annotate("lock-in", (2.2, 1.0), textcoords="offset points", xytext=(0, -16), color="dimgray", fontsize=8, ha="center")
ax[0].annotate("starvation", (-2.2, 0.55), textcoords="offset points", xytext=(0, 10), color="dimgray", fontsize=8, ha="center")
ax[0].set(title="A. std retained vs gamma (U-shape)", xlabel="gamma  (>0 homophily, <0 heterophily)", ylabel="std / init")
ax[0].legend(fontsize=9)

ac = np.array([pure_ac[g].mean() for g in GAMMAS])
ax[1].semilogy(gx, ac, "-o", c="#b30000"); ax[1].axvline(0, ls=":", c="gray")
ax[1].set(title="B. acceptance rate vs gamma\n(heterophily starves merges)",
          xlabel="gamma", ylabel="mass within eps (log)")

for g, c, lab in [(0.0, "#1f77b4", "neutral $\\gamma$=0"), (-1.5, "#d62728", "heterophily $\\gamma$=-1.5")]:
    ax[2].plot(EPS_GRID, eps_curve[g], "-o", c=c, label=lab)
ax[2].axvline(POP_EPS, ls=":", c="gray")
ax[2].set(title="C. eps-robustness\n(ordering holds where neutral reaches consensus)",
          xlabel="pop_eps (confidence bound)", ylabel="std / init")
ax[2].legend(fontsize=9)
fig.suptitle(f"Heterophily retains variance by interaction starvation, not diversity-seeking. "
             f"complete graph N={N}, {len(SEEDS)} seeds.", fontsize=11)
fig.savefig(f"{FIGS}/gamma_variance.png", dpi=130)
print(f"saved {FIGS}/gamma_variance.png")
for g in GAMMAS:
    print(f"gamma {g:>5.1f}  pure={pure_sd[g].mean():.3f}  coupled={coup[g].mean():.3f}  acc={pure_ac[g].mean():.4f}")
print("eps  neutral  hetero")
for i, e in enumerate(EPS_GRID):
    print(f"{e:.2f}  {eps_curve[0.0][i]:.3f}   {eps_curve[-1.5][i]:.3f}")
