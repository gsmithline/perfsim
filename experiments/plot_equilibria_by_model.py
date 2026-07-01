"""Equilibria by model: rows = OLS / MLP-mean / MLP-sampler, columns = the four
target states (each with the parameterization that yields it for a faithful
predictor). Each panel is a population waterfall + platform band. The actual
state reached is classified and annotated, so where the sampler cannot reach a
narrow-platform state (1 or 3) it shows up as a mismatch.
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


def predict(kind, Xtr, ytr, Xpr):
    if kind == "ols":
        Ftr = np.column_stack([np.ones(len(ytr)), Xtr[:, 0]])
        Fpr = np.column_stack([np.ones(len(Xpr)), Xpr[:, 0]])
        w, *_ = np.linalg.lstsq(Ftr, ytr, rcond=None)
        return np.clip(Fpr @ w, 0, 1).astype(np.float32)
    Xt = torch.tensor(Xtr); Yt = torch.tensor(ytr).unsqueeze(1); Xp = torch.tensor(Xpr)
    net = mlp(Xtr.shape[1], 1 if kind == "mlp_mean" else 2)
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
        o = net(Xp)
        if kind == "mlp_mean":
            return o.squeeze(1).clamp(0, 1).numpy()
        mu = o[:, 0:1]; sd = torch.exp(0.5 * o[:, 1:2].clamp(-6, 2))
        return (mu + sd * torch.randn_like(mu)).squeeze(1).clamp(0, 1).numpy()


def run(kind, gamma, eps_ai, regime, a, nz):
    x0 = make_x0(a, nz); init = float(x0.std())
    feat = z_std.reshape(-1, 1)
    keep = np.random.default_rng(SEED).permutation(N)[: N // 2]
    feat_keep, x0_keep = feat[keep], x0[keep]
    torch.manual_seed(SEED)
    pop = build_pop(gamma, x0); x = x0.copy()
    snaps, pmean, pstd = [], [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)], dtype=np.float32)
        if regime == "replace":
            trf, ty = feat, x
        else:
            trf = np.vstack([feat, feat_keep]); ty = np.concatenate([x, x0_keep])
        m = predict(kind, trf, ty, feat)
        g = np.abs(m - x) < eps_ai
        x = np.where(g, (1 - W) * x + W * m, x).astype(np.float32)
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
        snaps.append(x.copy()); pmean.append(float(m.mean())); pstd.append(float(m.std()))
    dr = float(x.std()) / init
    vr = pstd[-1] / float(x.std()) if x.std() > 1e-9 else float("nan")
    return snaps, np.array(pmean), np.array(pstd), dr, vr


def classify(dr, vr):
    wide = (not np.isnan(vr)) and vr >= 0.7
    div = dr >= 0.5
    if div and wide:
        return "2 both diverse"
    if div and not wide:
        return "3 plat below"
    if (not div) and wide:
        return "4 plat wide"
    return "1 both collapsed"


# target state -> (gamma, eps_ai, regime, a, nz)
STATES = [("State 1\nboth collapsed", 0.0, 0.25, "replace", 0.8, 0.15),
          ("State 2\nboth diverse", 3.0, 0.10, "replace", 0.8, 0.15),
          ("State 3\nplatform below", 3.0, 0.10, "replace", 0.3, 0.25),
          ("State 4\nplatform wide", 0.0, 0.25, "pristine", 0.8, 0.15)]
KINDS = [("OLS", "ols"), ("MLP-mean", "mlp_mean"), ("MLP-sampler", "mlp_sample")]

bins = np.linspace(0, 1, 41)
fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
for i, (klab, kind) in enumerate(KINDS):
    for j, (slab, gamma, ea, regime, a, nz) in enumerate(STATES):
        ax = axes[i][j]
        snaps, pmean, pstd, dr, vr = run(kind, gamma, ea, regime, a, nz)
        H = np.array([np.histogram(s, bins=bins, density=True)[0] for s in snaps])
        ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma", extent=[0, ROUNDS, 0, 1])
        rr = np.arange(ROUNDS)
        ax.plot(rr, pmean, c="#00e5ff", lw=1.2)
        ax.fill_between(rr, np.clip(pmean - pstd, 0, 1), np.clip(pmean + pstd, 0, 1), color="#00e5ff", alpha=0.25)
        got = classify(dr, vr)
        target = slab.split("\n")[0]
        mismatch = not got.startswith(target.split()[-1])
        ax.text(0.96, 0.05, f"dr={dr:.2f} vr={vr:.2f}\n-> {got}", transform=ax.transAxes, ha="right",
                va="bottom", color=("#ff5555" if mismatch else "white"), fontsize=8,
                fontweight=("bold" if mismatch else "normal"))
        if i == 0:
            ax.set_title(slab, fontsize=11)
        if j == 0:
            ax.set_ylabel(f"{klab}\nopinion")
        if i == 2:
            ax.set_xlabel("round")
fig.suptitle("Equilibria reachable by each model (rows) for each target state (columns). "
             "Red label = model landed in a DIFFERENT state than the target.", fontsize=12)
fig.savefig(f"{FIGS}/equilibria_by_model.png", dpi=130)
print(f"saved {FIGS}/equilibria_by_model.png")
