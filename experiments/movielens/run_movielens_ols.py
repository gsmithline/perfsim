"""OLS-in-AB loop on MovieLens 100K taste.

Node = user. Evolving label = taste for a TARGET genre (avg rating, normalized to
[0,1]). Fixed features = the user's round-0 taste for the OTHER core genres (the
strong predictor: taste->taste). Graph = kNN on the feature-genre taste vectors
(MovieLens has no native social graph), LCC. Population = NDlib AlgorithmicBias.
Platform = OLS of current target-taste on the fixed other-genre features; data
regimes replace/accumulate/pristine; gated feedback. We sweep eps x regime for a
strong-feature target (Action) and a weak one (Children's) to see the corners on
real data.
"""
import os, random
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

ML = "experiments/data/movielens/ml-100k"
W, ROUNDS, SEED, KNN = 0.3, 50, 0, 10
EPS_GRID = [0.08, 0.15, 0.25, 0.40, 0.60]
REGIMES = ["replace", "accumulate", "pristine"]
TARGETS = ["Action", "Children's"]
CORE = ["Drama", "Romance", "Comedy", "Action", "Thriller", "War", "Crime",
        "Sci-Fi", "Adventure", "Mystery", "Children's"]

genres = pd.read_csv(f"{ML}/u.genre", sep="|", names=["name", "gid"], encoding="latin-1")
GEN = list(genres.sort_values("gid")["name"])
items = pd.read_csv(f"{ML}/u.item", sep="|", encoding="latin-1", header=None)
gmat = pd.DataFrame(items.iloc[:, 5:5 + len(GEN)].values, index=items[0].values, columns=GEN)
users = pd.read_csv(f"{ML}/u.user", sep="|", names=["uid", "age", "gender", "occ", "zip"])
rat = pd.read_csv(f"{ML}/u.data", sep="\t", names=["uid", "iid", "r", "t"]).merge(
    gmat, left_on="iid", right_index=True)
pref = {g: rat[rat[g] == 1].groupby("uid")["r"].mean() for g in CORE}
P = pd.DataFrame(pref).reindex(users["uid"])
P = P.dropna()                              # keep users with real taste in all core genres
print(f"users with full core-genre coverage: {len(P)}")


def build_graph(featgenres):
    Z = P[featgenres].values
    Zc = Z - Z.mean(0)
    nn = NearestNeighbors(n_neighbors=KNN + 1, metric="cosine").fit(Zc)
    _, idx = nn.kneighbors(Zc)
    g = nx.Graph()
    g.add_nodes_from(range(len(P)))
    for i, nbrs in enumerate(idx):
        for j in nbrs[1:]:
            g.add_edge(i, int(j))
    lcc = max(nx.connected_components(g), key=len)
    h = g.subgraph(lcc).copy()
    mapping = {n: k for k, n in enumerate(sorted(h.nodes))}
    return nx.relabel_nodes(h, mapping), sorted(lcc)


def build_pop(G, eps, x0, seed):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G)
    c = mc.Configuration()
    c.add_model_parameter("epsilon", eps); c.add_model_parameter("gamma", 0.0)
    m.set_initial_status(c)
    for i in range(len(x0)):
        m.status[i] = float(x0[i])
    m.iteration(False)
    return m


def ols(Xtr, ytr, Xpr):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0), w


def run(G, F, x0, regime, eps):
    N = len(x0)
    pop = build_pop(G, eps, x0, SEED)
    x = x0.copy()
    keep = np.random.default_rng(SEED).permutation(N)[: N // 2]
    F0keep = F[keep]
    HX, Hy = [], []
    traj = []
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
        g = np.abs(m - x) < eps
        x = np.where(g, (1 - W) * x + W * m, x)
        for i in range(N):
            pop.status[i] = float(x[i])
        ostd = float(x.std())
        traj.append(dict(op_std=ostd, pred_std=float(m.std()),
                         vr=float(m.std()) / ostd if ostd > 1e-9 else float("nan")))
    return traj


def fin(traj, k):
    v = [r[k] for r in traj[-5:] if not (isinstance(r[k], float) and np.isnan(r[k]))]
    return sum(v) / len(v) if v else float("nan")


for target in TARGETS:
    feats = [g for g in CORE if g != target]
    G, lcc = build_graph(feats)
    Pl = P.iloc[lcc].reset_index(drop=True)
    x0 = ((Pl[target].values - 1.0) / 4.0)                       # normalize rating to [0,1]
    Zf = Pl[feats].values
    Zf = (Zf - Zf.mean(0)) / (Zf.std(0) + 1e-9)
    F = np.column_stack([np.ones(len(Pl)), Zf])
    w0, *_ = np.linalg.lstsq(F, x0, rcond=None)
    R2 = 1 - ((x0 - F @ w0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum()
    print(f"\n===== target={target}  N={len(Pl)}  graph mean_deg={2*G.number_of_edges()/len(Pl):.1f}  "
          f"init op_std={x0.std():.3f}  static feat R2={R2:.3f} =====")
    print(f"  {'eps':>5} {'regime':10s} | {'op_std':>7} {'pred_std':>8} {'vr':>6}")
    for eps in EPS_GRID:
        for regime in REGIMES:
            t = run(G, F, x0, regime, eps)
            print(f"  {eps:>5} {regime:10s} | {fin(t,'op_std'):7.3f} {fin(t,'pred_std'):8.3f} {fin(t,'vr'):6.2f}")
        print()
