"""MovieLens OLS-in-AB loop, full grid: target genre (feature strength) x gamma x
eps x regime x sampling budget, averaged over seeds. Budget is a redraw on the
fit only; predictions go to everyone. Mirrors the synthetic budget grid on real
taste data.
"""
import json, os, random
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

ML = "experiments/data/movielens/ml-100k"
OUT = "experiments/movielens/results"
os.makedirs(OUT, exist_ok=True)
GRAPH_TYPE = os.environ.get("GRAPH_TYPE", "taste")   # taste (kNN) | random (k-regular)
W, ROUNDS, KNN = 0.3, 50, 10
RAND_DEG = 14                                         # ~matches taste mean degree (15)
EPS_GRID = [0.08, 0.25, 0.60]
GAMMA_GRID = [0.0, 1.5, 3.0]
BUDGETS = [None, 100, 30]            # None = full population
REGIMES = ["replace", "accumulate", "pristine"]
TARGETS = ["Action", "Children's"]
SEEDS = [0, 1]
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
P = pd.DataFrame(pref).reindex(users["uid"]).dropna()


def build_graph(featgenres):
    if GRAPH_TYPE == "random":
        g = nx.random_regular_graph(RAND_DEG, len(P), seed=0)   # neutral, no taste structure
    else:
        Z = P[featgenres].values
        Zc = Z - Z.mean(0)
        _, idx = NearestNeighbors(n_neighbors=KNN + 1, metric="cosine").fit(Zc).kneighbors(Zc)
        g = nx.Graph(); g.add_nodes_from(range(len(P)))
        for i, nbrs in enumerate(idx):
            for j in nbrs[1:]:
                g.add_edge(i, int(j))
    lcc = sorted(max(nx.connected_components(g), key=len))
    h = nx.relabel_nodes(g.subgraph(lcc).copy(), {n: k for k, n in enumerate(lcc)})
    return h, lcc


def build_pop(G, eps, gamma, x0, seed):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G)
    c = mc.Configuration()
    c.add_model_parameter("epsilon", eps); c.add_model_parameter("gamma", gamma)
    m.set_initial_status(c)
    for i in range(len(x0)):
        m.status[i] = float(x0[i])
    m.iteration(False)
    return m


def ols(Xtr, ytr, Xpr):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0), w


def run(G, F, x0, regime, eps, gamma, B, seed):
    N = len(x0)
    B = N if B is None else min(B, N)
    pop = build_pop(G, eps, gamma, x0, seed)
    x = x0.copy()
    keep = np.random.default_rng(seed).permutation(N)[: N // 2]
    srng = np.random.default_rng(1000 + seed)
    F0keep = F[keep]
    HX, Hy = [], []
    traj = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        samp = srng.permutation(N)[:B]
        Fs, xs = F[samp], x[samp]
        if regime == "replace":
            Xtr, ytr = Fs, xs
        elif regime == "accumulate":
            HX.append(Fs); Hy.append(xs)
            Xtr, ytr = np.vstack(HX), np.concatenate(Hy)
        else:
            Xtr = np.vstack([Fs, F0keep]); ytr = np.concatenate([xs, x0[keep]])
        m, w = ols(Xtr, ytr, F)
        g = np.abs(m - x) < eps
        x = np.where(g, (1 - W) * x + W * m, x)
        for i in range(N):
            pop.status[i] = float(x[i])
        ostd = float(x.std())
        traj.append(dict(op_std=ostd, pred_std=float(m.std()),
                         vr=float(m.std()) / ostd if ostd > 1e-9 else float("nan")))
    return traj


def fin(trajs, k):
    vals = []
    for t in trajs:
        v = [r[k] for r in t[-5:] if not (isinstance(r[k], float) and np.isnan(r[k]))]
        if v:
            vals.append(sum(v) / len(v))
    return float(np.mean(vals)) if vals else float("nan")


allruns, meta = {}, {}
for target in TARGETS:
    feats = [g for g in CORE if g != target]
    G, lcc = build_graph(feats)
    Pl = P.iloc[lcc].reset_index(drop=True)
    x0 = (Pl[target].values - 1.0) / 4.0
    Zf = Pl[feats].values
    Zf = (Zf - Zf.mean(0)) / (Zf.std(0) + 1e-9)
    F = np.column_stack([np.ones(len(Pl)), Zf])
    w0, *_ = np.linalg.lstsq(F, x0, rcond=None)
    R2 = float(1 - ((x0 - F @ w0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum())
    meta[target] = dict(N=len(Pl), init_std=float(x0.std()), R2=R2,
                        mean_deg=2 * G.number_of_edges() / len(Pl))
    for eps in EPS_GRID:
        for gamma in GAMMA_GRID:
            for regime in REGIMES:
                for B in BUDGETS:
                    allruns[f"{target}|{eps}|{gamma}|{regime}|{B}"] = [
                        run(G, F, x0, regime, eps, gamma, B, s) for s in SEEDS]

json.dump({"meta": meta, "graph_type": GRAPH_TYPE, "eps": EPS_GRID, "gamma": GAMMA_GRID,
           "budgets": [str(b) for b in BUDGETS], "regimes": REGIMES, "seeds": SEEDS,
           "runs": {k: v for k, v in allruns.items()}},
          open(f"{OUT}/movielens_grid_{GRAPH_TYPE}.json", "w"))

print(f"MovieLens OLS-in-AB grid  GRAPH={GRAPH_TYPE}  W={W} rounds={ROUNDS} seeds={SEEDS}")
print("cells = op_std / vr  (mean over seeds).  B=full is the whole population.\n")
for target in TARGETS:
    mt = meta[target]
    print("=" * 70)
    print(f"TARGET={target}  N={mt['N']}  init op_std={mt['init_std']:.3f}  "
          f"static feat R2={mt['R2']:.3f}  mean_deg={mt['mean_deg']:.1f}")
    print("=" * 70)
    bcols = "  ".join(f"B={'full' if b is None else b:<4}".rjust(13) for b in BUDGETS)
    print(f"  {'eps':>4} {'gamma':>5} {'regime':10s} |{bcols}")
    print("  " + "-" * (24 + 13 * len(BUDGETS)))
    for eps in EPS_GRID:
        for gamma in GAMMA_GRID:
            for regime in REGIMES:
                cells = []
                for B in BUDGETS:
                    k = f"{target}|{eps}|{gamma}|{regime}|{B}"
                    cells.append(f"{fin(allruns[k],'op_std'):.3f}/{fin(allruns[k],'vr'):.2f}".rjust(13))
                print(f"  {eps:>4} {gamma:>5} {regime:10s} |{'  '.join(cells)}")
            print()
print(f"saved {OUT}/movielens_grid.json")
