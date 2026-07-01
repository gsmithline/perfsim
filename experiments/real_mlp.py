"""Does an expressive model escape the projection collapse OLS forces?

Same OLS-in-AB loop, three platform predictors:
  ols         linear conditional mean  E[y|x]   (baseline, strips residual)
  mlp_mean    nonlinear conditional mean E[y|x]  (still a mean predictor)
  mlp_sample  Gaussian head, feeds back a SAMPLE from P(y|x)  (re-injects residual)

Hypothesis: mean predictors (ols, mlp_mean) collapse the within-feature spread;
the sampler preserves it, because it feeds back variance, not just the mean.
This is the cheap stand-in for an LLM that samples diverse outputs.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/real_mlp.py
"""
import json, os, random
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

torch.set_num_threads(4)
W, ROUNDS, KNN, STEPS, CAP = 0.3, 30, 10, 80, 4000
GAMMA = float(os.environ.get("GAMMA", "0.0"))
REGIMES = ["replace", "accumulate", "pristine"]
SEEDS = [0, 1]
N_SYN = 300

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
    F = np.column_stack([np.ones(len(Pl)), Zf])
    w0, *_ = np.linalg.lstsq(F, x0, rcond=None)
    R2 = float(1 - ((x0 - F @ w0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum())
    return dict(name=f"ML-{target}", G=h, feat=Zf.astype(np.float32), x0=x0.astype(np.float32),
                R2=R2, pop_eps=0.25)


def yelp_dataset():
    d = np.load("experiments/yelp/yelp_acme_lcc.npz", allow_pickle=True)
    edges, x0 = d["edges"], d["opinion"].astype(np.float32)
    avg = d["avg_stars"].astype(float)
    avg_z = ((avg - avg.mean()) / (avg.std() + 1e-9)).astype(np.float32).reshape(-1, 1)
    N = len(x0); G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(edges.tolist())
    F = np.column_stack([np.ones(N), avg_z]); w0, *_ = np.linalg.lstsq(F, x0, rcond=None)
    R2 = float(1 - ((x0 - F @ w0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum())
    return dict(name="Yelp", G=G, feat=avg_z, x0=x0, R2=R2, pop_eps=0.40)


N_SYN = 300


def synth_dataset(a_slope, noise_sd, name, pop_eps=0.25):
    rng = np.random.default_rng(0)
    zz = rng.uniform(0.0, 1.0, N_SYN)
    zs = ((zz - zz.mean()) / (zz.std() + 1e-9)).astype(np.float32)
    x0 = np.clip(0.5 + a_slope * (zz - 0.5) + rng.normal(0.0, noise_sd, N_SYN), 0.0, 1.0).astype(np.float32)
    Fm = np.column_stack([np.ones(N_SYN), zs]); w0, *_ = np.linalg.lstsq(Fm, x0, rcond=None)
    R2 = float(1 - ((x0 - Fm @ w0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum())
    return dict(name=name, G=nx.complete_graph(N_SYN), feat=zs.reshape(-1, 1), x0=x0, R2=R2, pop_eps=pop_eps)


def build_pop(G, eps, gamma, x0, seed):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G); c = mc.Configuration()
    c.add_model_parameter("epsilon", eps); c.add_model_parameter("gamma", gamma)
    m.set_initial_status(c)
    for i in range(len(x0)):
        m.status[i] = float(x0[i])
    m.iteration(False)
    return m


def mlp(d, out):
    return nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, out))


def predict(kind, Xtr, ytr, Xpr, sd_seed):
    if kind == "ols":
        Ftr = np.column_stack([np.ones(len(ytr)), Xtr])
        Fpr = np.column_stack([np.ones(len(Xpr)), Xpr])
        w, *_ = np.linalg.lstsq(Ftr, ytr, rcond=None)
        return np.clip(Fpr @ w, 0, 1).astype(np.float32), None
    if len(ytr) > CAP:
        idx = np.random.default_rng(sd_seed).permutation(len(ytr))[:CAP]
        Xtr, ytr = Xtr[idx], ytr[idx]
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
            return o.squeeze(1).clamp(0, 1).numpy(), None
        mu = o[:, 0:1]; sd = torch.exp(0.5 * o[:, 1:2].clamp(-6, 2))
        samp = (mu + sd * torch.randn_like(mu)).squeeze(1).clamp(0, 1).numpy()
        return samp, float(sd.mean())


def run(ds, kind, regime, gamma, seed):
    G, feat, x0, eps = ds["G"], ds["feat"], ds["x0"], ds["pop_eps"]
    N = len(x0); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    pop = build_pop(G, eps, gamma, x0, seed)
    x = x0.copy()
    keep = rng.permutation(N)[: N // 2]
    feat_keep, x0_keep = feat[keep], x0[keep]
    HXf, Hy = [], []
    traj = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)], dtype=np.float32)
        if regime == "replace":
            trf, ty = feat, x
        elif regime == "accumulate":
            HXf.append(feat); Hy.append(x.copy())
            trf, ty = np.vstack(HXf), np.concatenate(Hy)
        else:
            trf = np.vstack([feat, feat_keep]); ty = np.concatenate([x, x0_keep])
        m, sd = predict(kind, trf, ty, feat, seed * 1000 + t)
        g = np.abs(m - x) < eps
        x = np.where(g, (1 - W) * x + W * m, x).astype(np.float32)
        for i in range(N):
            pop.status[i] = float(x[i])
        ostd = float(x.std())
        traj.append(dict(op_std=ostd, pred_std=float(m.std()),
                         vr=float(m.std()) / ostd if ostd > 1e-9 else float("nan"),
                         sigma=sd if sd is not None else float("nan")))
    return traj


def fin(t, k):
    v = [r[k] for r in t[-5:] if not (isinstance(r[k], float) and np.isnan(r[k]))]
    return sum(v) / len(v) if v else float("nan")


OUT = "experiments/toy/results"
GAMMAS_ALL = [3.0, 0.0, -1.5]


def main():
    os.makedirs(OUT, exist_ok=True)
    datasets = [synth_dataset(0.8, 0.15, "Synth-strong"), synth_dataset(0.3, 0.25, "Synth-weak"),
                ml_dataset("Action"), ml_dataset("Children's"), yelp_dataset()]
    results = {"meta": dict(W=W, rounds=ROUNDS, steps=STEPS, seeds=SEEDS,
                            gammas=GAMMAS_ALL, regimes=REGIMES,
                            note="dr=op_std/init, vr=pred_std/op_std, sigma=learned residual sd"),
               "datasets": {}, "runs": {}}
    for ds in datasets:
        init = float(ds["x0"].std())
        results["datasets"][ds["name"]] = dict(N=len(ds["x0"]), R2=ds["R2"], init_std=init, pop_eps=ds["pop_eps"])
        print(f"### {ds['name']}  R2={ds['R2']:.3f} ###")
        for gamma in GAMMAS_ALL:
            for kind in ["ols", "mlp_mean", "mlp_sample"]:
                for regime in REGIMES:
                    runs = [run(ds, kind, regime, gamma, s) for s in SEEDS]
                    dr = np.mean([fin(t, "op_std") for t in runs]) / init
                    ps = np.mean([fin(t, "pred_std") for t in runs])
                    vr = np.mean([fin(t, "vr") for t in runs])
                    sg = np.mean([fin(t, "sigma") for t in runs])
                    results["runs"][f"{ds['name']}|g{gamma}|{kind}|{regime}"] = dict(
                        dr=float(dr), pred_std=float(ps), vr=float(vr), sigma=float(sg))
            print(f"  gamma={gamma} done")
    json.dump(results, open(f"{OUT}/real_mlp_results.json", "w"))
    print(f"saved {OUT}/real_mlp_results.json  ({len(results['runs'])} cells)")


if __name__ == "__main__":
    main()
