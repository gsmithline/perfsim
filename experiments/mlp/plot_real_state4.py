"""Real state-4 panel: MovieLens-Action (strong feature), neutral, pristine.
Left = no platform (the population collapses on its own). Right = pristine
platform (the population is held up and the platform stays wide, vr>1).
Shows state 4 / stable capture / the AI-as-diversity-source effect on real data.
"""
import os, random
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = "experiments/toy/figs"
W, ROUNDS, KNN, SEED, EPS_AI, GAMMA = 0.3, 50, 10, 0, 0.10, 0.0
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
P = pd.DataFrame({g: _rat[_rat[g] == 1].groupby("uid")["r"].mean() for g in CORE}).reindex(_users["uid"]).dropna()

feats = [g for g in CORE if g != "Action"]
Zc = P[feats].values - P[feats].values.mean(0)
_, idx = NearestNeighbors(n_neighbors=KNN + 1, metric="cosine").fit(Zc).kneighbors(Zc)
g = nx.Graph(); g.add_nodes_from(range(len(P)))
for i, nbrs in enumerate(idx):
    for j in nbrs[1:]:
        g.add_edge(i, int(j))
lcc = sorted(max(nx.connected_components(g), key=len))
G = nx.relabel_nodes(g.subgraph(lcc).copy(), {n: k for k, n in enumerate(lcc)})
Pl = P.iloc[lcc].reset_index(drop=True)
x0 = ((Pl["Action"].values - 1.0) / 4.0).astype(np.float32)
Zf = Pl[feats].values; Zf = (Zf - Zf.mean(0)) / (Zf.std(0) + 1e-9)
F = np.column_stack([np.ones(len(Pl)), Zf]); N = len(Pl); INIT = float(x0.std())


def build_pop():
    random.seed(SEED); np.random.seed(SEED)
    m = AlgorithmicBiasModel(G); c = mc.Configuration()
    c.add_model_parameter("epsilon", 0.25); c.add_model_parameter("gamma", GAMMA)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x0[i])
    m.iteration(False)
    return m


def ols(Xtr, ytr, Xpr):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0), w


def run(with_platform):
    pop = build_pop(); x = x0.copy()
    keep = np.random.default_rng(SEED).permutation(N)[: N // 2]; F0, x0k = F[keep], x0[keep]
    snaps, pmean, pstd = [], [], []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)], dtype=np.float32)
        if with_platform:
            Xtr = np.vstack([F, F0]); ytr = np.concatenate([x, x0k])   # pristine
            m, w = ols(Xtr, ytr, F)
            gate = np.abs(m - x) < EPS_AI
            x = np.where(gate, (1 - W) * x + W * m, x).astype(np.float32)
            for i in range(N):
                pop.status[i] = float(x[i])
            pmean.append(float(m.mean())); pstd.append(float(m.std()))
        snaps.append(x.copy())
    dr = float(x.std()) / INIT
    vr = (pstd[-1] / float(x.std())) if (with_platform and x.std() > 1e-9) else float("nan")
    return snaps, np.array(pmean), np.array(pstd), dr, vr


bins = np.linspace(0, 1, 41)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
for ax, (wp, title) in zip(axes, [(False, "No platform\n(population collapses on its own)"),
                                  (True, "Pristine platform\n(state 4: population held, platform wide)")]):
    snaps, pmean, pstd, dr, vr = run(wp)
    H = np.array([np.histogram(s, bins=bins, density=True)[0] for s in snaps])
    ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma", extent=[0, ROUNDS, 0, 1])
    if wp:
        rr = np.arange(ROUNDS)
        ax.plot(rr, pmean, c="#00e5ff", lw=1.3)
        ax.fill_between(rr, np.clip(pmean - pstd, 0, 1), np.clip(pmean + pstd, 0, 1), color="#00e5ff", alpha=0.25)
        ax.text(0.96, 0.05, f"dr={dr:.2f}  vr={vr:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                color="white", fontsize=10)
    else:
        ax.text(0.96, 0.05, f"dr={dr:.2f}", transform=ax.transAxes, ha="right", va="bottom", color="white", fontsize=10)
    ax.set(title=title, xlabel="round", ylabel="opinion")
fig.suptitle(f"Real state 4: MovieLens-Action (R2={float(1 - ((x0 - F @ np.linalg.lstsq(F, x0, rcond=None)[0]) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum()):.2f}), "
             f"neutral, pristine. Platform widens a self-collapsing population and stays wider than it.", fontsize=11)
fig.savefig(f"{FIGS}/real_state4.png", dpi=130)
print(f"saved {FIGS}/real_state4.png")
