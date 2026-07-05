"""Waterfall comparison of OLS / MLP-mean / MLP-sampler.

Rows = predictor, columns = scenario. Each panel: population distribution over
rounds (magma) plus platform prediction band (cyan). Shows that model class
barely changes the population, and that the sampler's wide band is injected
noise, not preserved diversity. Synthetic OLS-in-AB loop, replace, train on all.
"""
import os, random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(4)
FIGS = "experiments/toy/figs"
N, ROUNDS, POP_EPS, W, STEPS, SEED = 300, 60, 0.25, 0.5, 80, 0
G = nx.complete_graph(N)
_rng = np.random.default_rng(0)
z = _rng.uniform(0.0, 1.0, N)
z_std = ((z - z.mean()) / (z.std() + 1e-9)).astype(np.float32)


def make_x0(a, nz):
    return np.clip(0.5 + a * (z - 0.5) + np.random.default_rng(1).normal(0.0, nz, N), 0.0, 1.0).astype(np.float32)


def build_pop(gamma, x0):
    random.seed(SEED); np.random.seed(SEED)
    m = AlgorithmicBiasModel(G); c = mc.Configuration()
    c.add_model_parameter("epsilon", POP_EPS); c.add_model_parameter("gamma", gamma)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x0[i])
    m.sts = x0.copy(); m.iteration(False)
    return m


def mlp(d, out):
    return nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, out))


def predict(kind, x):
    if kind == "ols":
        Fm = np.column_stack([np.ones(N), z_std])
        w, *_ = np.linalg.lstsq(Fm, x, rcond=None)
        return np.clip(Fm @ w, 0, 1).astype(np.float32)
    Xt = torch.tensor(z_std).unsqueeze(1); Yt = torch.tensor(x).unsqueeze(1)
    net = mlp(1, 1 if kind == "mlp_mean" else 2)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(STEPS):
        opt.zero_grad(); o = net(Xt)
        if kind == "mlp_mean":
            loss = ((o - Yt) ** 2).mean()
        else:
            mu = o[:, 0:1]; lv = o[:, 1:2].clamp(-6, 2)
            loss = (0.5 * ((Yt - mu) ** 2 / torch.exp(lv) + lv)).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        o = net(Xt)
        if kind == "mlp_mean":
            return o.squeeze(1).clamp(0, 1).numpy()
        mu = o[:, 0:1]; sd = torch.exp(0.5 * o[:, 1:2].clamp(-6, 2))
        return (mu + sd * torch.randn_like(mu)).squeeze(1).clamp(0, 1).numpy()


def run(kind, gamma, eps_ai, a, nz):
    x0 = make_x0(a, nz); init = float(x0.std())
    torch.manual_seed(SEED)
    pop = build_pop(gamma, x0); x = x0.copy()
    snaps, pmean, pstd = [], [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)], dtype=np.float32)
        m = predict(kind, x)
        g = np.abs(m - x) < eps_ai
        x = np.where(g, (1 - W) * x + W * m, x).astype(np.float32)
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
        snaps.append(x.copy()); pmean.append(float(m.mean())); pstd.append(float(m.std()))
    dr = float(x.std()) / init
    vr = pstd[-1] / float(x.std()) if x.std() > 1e-9 else float("nan")
    return snaps, np.array(pmean), np.array(pstd), dr, vr


SCEN = [("diverse\n($\\gamma$=3, strong, $\\epsilon_{AI}$=0.1)", 3.0, 0.10, 0.8, 0.15),
        ("collapse\n($\\gamma$=0, strong, $\\epsilon_{AI}$=0.25)", 0.0, 0.25, 0.8, 0.15),
        ("weak feature\n($\\gamma$=3, weak, $\\epsilon_{AI}$=0.1)", 3.0, 0.10, 0.3, 0.25)]
KINDS = [("OLS", "ols"), ("MLP-mean", "mlp_mean"), ("MLP-sampler", "mlp_sample")]

bins = np.linspace(0, 1, 41)
fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
for i, (klab, kind) in enumerate(KINDS):
    for j, (slab, gamma, ea, a, nz) in enumerate(SCEN):
        ax = axes[i][j]
        snaps, pmean, pstd, dr, vr = run(kind, gamma, ea, a, nz)
        H = np.array([np.histogram(s, bins=bins, density=True)[0] for s in snaps])
        ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma", extent=[0, ROUNDS, 0, 1])
        rr = np.arange(ROUNDS)
        ax.plot(rr, pmean, c="#00e5ff", lw=1.2)
        ax.fill_between(rr, np.clip(pmean - pstd, 0, 1), np.clip(pmean + pstd, 0, 1), color="#00e5ff", alpha=0.25)
        ax.text(0.96, 0.05, f"dr={dr:.2f}\nvr={vr:.2f}", transform=ax.transAxes, ha="right",
                va="bottom", color="white", fontsize=8)
        if i == 0:
            ax.set_title(slab, fontsize=10)
        if j == 0:
            ax.set_ylabel(f"{klab}\nopinion")
        if i == 2:
            ax.set_xlabel("round")
fig.suptitle("Waterfalls: OLS vs MLP-mean vs MLP-sampler (population magma, platform band cyan)", fontsize=13)
fig.savefig(f"{FIGS}/mlp_ols_waterfalls.png", dpi=130)
print(f"saved {FIGS}/mlp_ols_waterfalls.png")
