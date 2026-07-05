"""Real-data versions of the two best grids.
 A) regime grid: rows = regime, cols = gamma, OLS, on MovieLens-Children's.
 B) model grid: rows = OLS/MLP-mean/sampler, cols = dataset (Action/Children's/Yelp),
    homophily so the population stays diverse and feature strength (the dataset)
    sets the platform width; shows the sampler cannot go narrow on weak features.
Each panel: population waterfall + platform band. Captures per-round snapshots.
"""
import os, random
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(4)
FIGS = "experiments/toy/figs"
W, ROUNDS, KNN, STEPS, SEED = 0.3, 50, 10, 80, 0
EPS_AI = 0.10

ML = "experiments/data/movielens/ml-100k"
CORE = ["Drama", "Romance", "Comedy", "Action", "Thriller", "War", "Crime",
        "Sci-Fi", "Adventure", "Mystery", "Children's"]
_gen = pd.read_csv(f"{ML}/u.genre", sep="|", names=["name", "gid"], encoding="latin-1")
GEN = list(_gen.sort_values("gid")["name"])
_items = pd.read_csv(f"{ML}/u.item", sep="|", encoding="latin-1", header=None)
_gmat = pd.DataFrame(_items.iloc[:, 5:5 + len(GEN)].values, index=_items[0].values, columns=GEN)
_users = pd.read_csv(f"{ML}/u.user", sep="|", names=["uid", "age", "gender", "occ", "zip"])
_rat = pd.read_csv(f"{ML}/u.data", sep="\t", names=["uid", "iid", "r", "t"]).merge(
    _gmat, left_on="iid", right_index=True)
_pref = {g: _rat[_rat[g] == 1].groupby("uid")["r"].mean() for g in CORE}
P = pd.DataFrame(_pref).reindex(_users["uid"]).dropna()


def ml_dataset(target):
    feats = [g for g in CORE if g != target]
    Z = P[feats].values; Zc = Z - Z.mean(0)
    _, idx = NearestNeighbors(n_neighbors=KNN + 1, metric="cosine").fit(Zc).kneighbors(Zc)
    g = nx.Graph(); g.add_nodes_from(range(len(P)))
    for i, nbrs in enumerate(idx):
        for j in nbrs[1:]:
            g.add_edge(i, int(j))
    lcc = sorted(max(nx.connected_components(g), key=len))
    h = nx.relabel_nodes(g.subgraph(lcc).copy(), {n: k for k, n in enumerate(lcc)})
    Pl = P.iloc[lcc].reset_index(drop=True)
    x0 = (Pl[target].values - 1.0) / 4.0
    Zf = Pl[feats].values; Zf = (Zf - Zf.mean(0)) / (Zf.std(0) + 1e-9)
    F = np.column_stack([np.ones(len(Pl)), Zf]); w0, *_ = np.linalg.lstsq(F, x0, rcond=None)
    R2 = float(1 - ((x0 - F @ w0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum())
    return dict(name=f"ML-{target}", G=h, feat=Zf.astype(np.float32), x0=x0.astype(np.float32), R2=R2, pop_eps=0.25)


def yelp_dataset():
    d = np.load("experiments/yelp/yelp_acme_lcc.npz", allow_pickle=True)
    edges, x0 = d["edges"], d["opinion"].astype(np.float32)
    avg = d["avg_stars"].astype(float); avg_z = ((avg - avg.mean()) / (avg.std() + 1e-9)).astype(np.float32).reshape(-1, 1)
    N = len(x0); G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(edges.tolist())
    F = np.column_stack([np.ones(N), avg_z]); w0, *_ = np.linalg.lstsq(F, x0, rcond=None)
    R2 = float(1 - ((x0 - F @ w0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum())
    return dict(name="Yelp", G=G, feat=avg_z, x0=x0, R2=R2, pop_eps=0.40)


def build_pop(G, eps, gamma, x0):
    random.seed(SEED); np.random.seed(SEED)
    m = AlgorithmicBiasModel(G); c = mc.Configuration()
    c.add_model_parameter("epsilon", eps); c.add_model_parameter("gamma", gamma)
    m.set_initial_status(c)
    for i in range(len(x0)):
        m.status[i] = float(x0[i])
    m.iteration(False)
    return m


def mlp(d, out):
    return nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, out))


def predict(kind, Xtr, ytr, Xpr):
    if kind == "ols":
        Ftr = np.column_stack([np.ones(len(ytr)), Xtr]); Fpr = np.column_stack([np.ones(len(Xpr)), Xpr])
        w, *_ = np.linalg.lstsq(Ftr, ytr, rcond=None)
        return np.clip(Fpr @ w, 0, 1).astype(np.float32)
    if len(ytr) > 4000:
        idx = np.random.default_rng(0).permutation(len(ytr))[:4000]; Xtr, ytr = Xtr[idx], ytr[idx]
    Xt = torch.tensor(Xtr); Yt = torch.tensor(ytr).unsqueeze(1); Xp = torch.tensor(Xpr)
    net = mlp(Xtr.shape[1], 1 if kind == "mlp_mean" else 2)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(STEPS):
        opt.zero_grad(); o = net(Xt)
        if kind == "mlp_mean":
            loss = ((o - Yt) ** 2).mean()
        else:
            mu = o[:, 0:1]; lv = o[:, 1:2].clamp(-6, 2); loss = (0.5 * ((Yt - mu) ** 2 / torch.exp(lv) + lv)).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        o = net(Xp)
        if kind == "mlp_mean":
            return o.squeeze(1).clamp(0, 1).numpy()
        mu = o[:, 0:1]; sd = torch.exp(0.5 * o[:, 1:2].clamp(-6, 2))
        return (mu + sd * torch.randn_like(mu)).squeeze(1).clamp(0, 1).numpy()


def run(ds, kind, gamma, regime):
    G, feat, x0, eps = ds["G"], ds["feat"], ds["x0"], ds["pop_eps"]
    N = len(x0); init = float(x0.std()); torch.manual_seed(SEED)
    keep = np.random.default_rng(SEED).permutation(N)[: N // 2]; fk, xk = feat[keep], x0[keep]
    pop = build_pop(G, eps, gamma, x0); x = x0.copy(); HX, Hy = [], []
    snaps, pmean, pstd = [], [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)], dtype=np.float32)
        if regime == "replace":
            trf, ty = feat, x
        elif regime == "accumulate":
            HX.append(feat); Hy.append(x.copy()); trf, ty = np.vstack(HX), np.concatenate(Hy)
        else:
            trf = np.vstack([feat, fk]); ty = np.concatenate([x, xk])
        m = predict(kind, trf, ty, feat)
        g = np.abs(m - x) < EPS_AI
        x = np.where(g, (1 - W) * x + W * m, x).astype(np.float32)
        for i in range(N):
            pop.status[i] = float(x[i])
        snaps.append(x.copy()); pmean.append(float(m.mean())); pstd.append(float(m.std()))
    dr = float(x.std()) / init; vr = pstd[-1] / float(x.std()) if x.std() > 1e-9 else float("nan")
    return snaps, np.array(pmean), np.array(pstd), dr, vr


def waterfall(ax, snaps, pmean, pstd, dr, vr):
    bins = np.linspace(0, 1, 41)
    H = np.array([np.histogram(s, bins=bins, density=True)[0] for s in snaps])
    ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma", extent=[0, len(snaps), 0, 1])
    rr = np.arange(len(snaps))
    ax.plot(rr, pmean, c="#00e5ff", lw=1.2)
    ax.fill_between(rr, np.clip(pmean - pstd, 0, 1), np.clip(pmean + pstd, 0, 1), color="#00e5ff", alpha=0.25)
    ax.text(0.96, 0.05, f"dr={dr:.2f} vr={vr:.2f}", transform=ax.transAxes, ha="right", va="bottom",
            color="white", fontsize=8)


def main():
    action, child, yelp = ml_dataset("Action"), ml_dataset("Children's"), yelp_dataset()

    # A) regime grid on MovieLens-Children's (OLS)
    GAM = [("Homophily ($\\gamma$=3)", 3.0), ("Neutral ($\\gamma$=0)", 0.0), ("Heterophily ($\\gamma$=-1.5)", -1.5)]
    REG = ["replace", "accumulate", "pristine"]
    fig, ax = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for i, regime in enumerate(REG):
        for j, (glab, gamma) in enumerate(GAM):
            waterfall(ax[i][j], *run(child, "ols", gamma, regime))
            if i == 0:
                ax[i][j].set_title(glab, fontsize=11)
            if j == 0:
                ax[i][j].set_ylabel(f"{regime}\nopinion")
            if i == 2:
                ax[i][j].set_xlabel("round")
    fig.suptitle(f"Real-data regime grid: MovieLens-Children's (R2={child['R2']:.2f}), OLS, $\\epsilon_{{AI}}$={EPS_AI}",
                 fontsize=12)
    fig.savefig(f"{FIGS}/real_regime_grid.png", dpi=130); print(f"saved {FIGS}/real_regime_grid.png")

    # B) model grid: rows = model, cols = dataset (feature strength), homophily, replace
    DS = [action, child, yelp]
    KINDS = [("OLS", "ols"), ("MLP-mean", "mlp_mean"), ("MLP-sampler", "mlp_sample")]
    fig, ax = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for i, (klab, kind) in enumerate(KINDS):
        for j, ds in enumerate(DS):
            waterfall(ax[i][j], *run(ds, kind, 3.0, "replace"))
            if i == 0:
                ax[i][j].set_title(f"{ds['name']} (R2={ds['R2']:.2f})", fontsize=11)
            if j == 0:
                ax[i][j].set_ylabel(f"{klab}\nopinion")
            if i == 2:
                ax[i][j].set_xlabel("round")
    fig.suptitle(f"Real-data model grid: rows = model, cols = dataset (feature strength). "
                 f"homophily, replace, $\\epsilon_{{AI}}$={EPS_AI}", fontsize=12)
    fig.savefig(f"{FIGS}/real_model_grid.png", dpi=130); print(f"saved {FIGS}/real_model_grid.png")


if __name__ == "__main__":
    main()
