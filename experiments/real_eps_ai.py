"""Real-data OLS-in-AB loop with a SEPARATE AI gate (eps_AI != population eps).

Three datasets spanning feature strength: MovieLens-Action (R2 ~ 0.79),
MovieLens-Children's (R2 ~ 0.36), Yelp (R2 ~ 0.067). Population eps fixed per
dataset; eps_AI swept (eps_AI=0 is the no-platform baseline). Train on everyone.
Question: does a tighter AI gate re-protect the population on real data, as it
does in the synthetic toy?

Run: MPLCONFIGDIR=/tmp/mpl python experiments/real_eps_ai.py
"""
import json, os, random
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

OUT = "experiments/toy/results"
os.makedirs(OUT, exist_ok=True)
W, ROUNDS, KNN = 0.3, 50, 10
EPS_AI_GRID = [0.0, 0.05, 0.10, 0.25, 0.50]
GAMMAS = [0.0, 3.0]
REGIMES = ["replace", "pristine"]
SEEDS = [0, 1]

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
    Z = P[feats].values
    Zc = Z - Z.mean(0)
    _, idx = NearestNeighbors(n_neighbors=KNN + 1, metric="cosine").fit(Zc).kneighbors(Zc)
    g = nx.Graph(); g.add_nodes_from(range(len(P)))
    for i, nbrs in enumerate(idx):
        for j in nbrs[1:]:
            g.add_edge(i, int(j))
    lcc = sorted(max(nx.connected_components(g), key=len))
    h = nx.relabel_nodes(g.subgraph(lcc).copy(), {n: k for k, n in enumerate(lcc)})
    Pl = P.iloc[lcc].reset_index(drop=True)
    x0 = (Pl[target].values - 1.0) / 4.0
    Zf = Pl[feats].values
    Zf = (Zf - Zf.mean(0)) / (Zf.std(0) + 1e-9)
    F = np.column_stack([np.ones(len(Pl)), Zf])
    w0, *_ = np.linalg.lstsq(F, x0, rcond=None)
    R2 = float(1 - ((x0 - F @ w0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum())
    return dict(name=f"MovieLens-{target}", G=h, F=F, x0=x0, R2=R2, pop_eps=0.25)


def yelp_dataset():
    d = np.load("experiments/yelp/yelp_acme_lcc.npz", allow_pickle=True)
    edges, x0 = d["edges"], d["opinion"].astype(float)
    avg = d["avg_stars"].astype(float)
    avg_z = (avg - avg.mean()) / (avg.std() + 1e-9)
    N = len(x0)
    G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(edges.tolist())
    F = np.column_stack([np.ones(N), avg_z])
    w0, *_ = np.linalg.lstsq(F, x0, rcond=None)
    R2 = float(1 - ((x0 - F @ w0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum())
    return dict(name="Yelp", G=G, F=F, x0=x0, R2=R2, pop_eps=0.40)


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


def run(ds, eps_ai, gamma, regime, seed):
    G, F, x0, pop_eps = ds["G"], ds["F"], ds["x0"], ds["pop_eps"]
    N = len(x0)
    pop = build_pop(G, pop_eps, gamma, x0, seed)
    x = x0.copy()
    keep = np.random.default_rng(seed).permutation(N)[: N // 2]
    F0, x0k = F[keep], x0[keep]
    traj = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        if regime == "replace" or eps_ai == 0.0:
            Xtr, ytr = F, x
        else:
            Xtr = np.vstack([F, F0]); ytr = np.concatenate([x, x0k])
        m, w = ols(Xtr, ytr, F)
        if eps_ai > 0.0:
            g = np.abs(m - x) < eps_ai
            x = np.where(g, (1 - W) * x + W * m, x)
            for i in range(N):
                pop.status[i] = float(x[i])
            gate = float(g.mean())
        else:
            gate = 0.0
        ostd = float(x.std())
        traj.append(dict(op_std=ostd, pred_std=float(m.std()),
                         vr=float(m.std()) / ostd if ostd > 1e-9 else float("nan"), gate=gate))
    return traj


def fin(t, k):
    v = [r[k] for r in t[-5:] if not (isinstance(r[k], float) and np.isnan(r[k]))]
    return sum(v) / len(v) if v else float("nan")


def main():
    datasets = [ml_dataset("Action"), ml_dataset("Children's"), yelp_dataset()]
    out = {}
    for ds in datasets:
        init = float(ds["x0"].std())
        print(f"\n############ {ds['name']}  N={len(ds['x0'])}  R2={ds['R2']:.3f}  "
              f"init_std={init:.3f}  pop_eps={ds['pop_eps']} ############")
        estr = "  ".join(f"{e:>4.2f}" for e in EPS_AI_GRID)
        print(f"  {'gamma':>5} {'regime':8s} |  dr at eps_AI [{estr}]   (eps_AI=0 = no platform)")
        for gamma in GAMMAS:
            for regime in REGIMES:
                cells, vrs, gts = [], [], []
                for ea in EPS_AI_GRID:
                    runs = [run(ds, ea, gamma, regime, s) for s in SEEDS]
                    dr = np.mean([fin(t, "op_std") for t in runs]) / init
                    vr = np.mean([fin(t, "vr") for t in runs])
                    gt = np.mean([fin(t, "gate") for t in runs])
                    cells.append(f"{dr:4.2f}"); vrs.append(f"{vr:4.2f}"); gts.append(f"{gt:4.2f}")
                    out[f"{ds['name']}|{gamma}|{regime}|{ea}"] = dict(dr=float(dr), vr=float(vr), gate=float(gt))
                print(f"  {gamma:>5} {regime:8s} |  " + "  ".join(cells)
                      + "   vr[" + " ".join(vrs) + "]")
    json.dump(out, open(f"{OUT}/real_eps_ai.json", "w"))
    print(f"\nsaved {OUT}/real_eps_ai.json")


if __name__ == "__main__":
    main()
