"""Yelp Acme, Jiduan-style: predict the LABEL (opinion) from the DATA (observed
opinions), no side feature. Predictor is an opinion-estimator of accuracy b:

    m_i = target + b * (x_i - target),   target = mean of the pooled training data

b=1 is perfect prediction (Jiduan's homogenizing-force case, neutral here),
b=0 is mean estimation (Jiduan's limited-info case). The data regime sets which
opinions are pooled into `target` (replace=current, accumulate=history,
pristine=+original). Population is OUR conditional-bias (AB/homophily) model.
"""
import json, os, random
import numpy as np
import networkx as nx
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

HERE = "experiments/yelp"
OUT = f"{HERE}/results"
os.makedirs(OUT, exist_ok=True)
EPS, W, ROUNDS, SEED = 0.4, 0.3, 60, 0
GAMMAS = [0.0, 1.5]
BS = [1.0, 0.8, 0.5, 0.2, 0.0]
REGIMES = ["replace", "accumulate", "pristine"]

d = np.load(f"{HERE}/yelp_acme_lcc.npz", allow_pickle=True)
x0 = d["opinion"].astype(float); edges = d["edges"]
N = len(x0)
G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(edges.tolist())


def build(gamma, seed, x):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G); c = mc.Configuration()
    c.add_model_parameter("epsilon", EPS); c.add_model_parameter("gamma", gamma)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x[i])
    return m


def run(b, regime, gamma, use_platform=True):
    pop = build(gamma, SEED, x0.copy()); x = x0.copy()
    keep = np.random.default_rng(SEED).permutation(N)[: N // 2]
    hist = []
    traj = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        if use_platform:
            hist.append(x.copy())
            if regime == "replace":
                pool = x
            elif regime == "accumulate":
                pool = np.concatenate(hist)
            else:
                pool = np.concatenate([x, x0[keep]])
            target = pool.mean()
            m = np.clip(target + b * (x - target), 0.0, 1.0)   # label estimator, accuracy b
            g = np.abs(m - x) < EPS
            x = np.where(g, (1 - W) * x + W * m, x)
            for i in range(N):
                pop.status[i] = float(x[i])
            pstd, pmean = float(np.std(m)), float(np.mean(m))
        else:
            pstd = pmean = float("nan")
        traj.append(dict(round=t, op_std=float(x.std()), op_mean=float(x.mean()),
                         pred_std=pstd, pred_bias=(pmean - float(x0.mean())) if use_platform else float("nan")))
    return traj


def main():
    init = float(x0.std())
    runs = {}
    for gamma in GAMMAS:
        runs[f"noplatform_g{gamma}"] = run(0, "replace", gamma, use_platform=False)
        for b in BS:
            for regime in REGIMES:
                runs[f"b{b}_{regime}_g{gamma}"] = run(b, regime, gamma)
    json.dump({"meta": dict(N=N, eps=EPS, W=W, init_std=init, init_mean=float(x0.mean())),
               "runs": runs}, open(f"{OUT}/yelp_label_results.json", "w"))

    def fin(k, key):
        v = [r[key] for r in runs[k][-5:] if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return sum(v) / len(v) if v else float("nan")
    print(f"Yelp Acme  N={N}  eps={EPS} W={W}  init op_std={init:.3f} (mean {x0.mean():.3f})")
    print("predict label-from-data, accuracy b (b=1 perfect, b=0 mean-estimation)")
    for gamma in GAMMAS:
        npl = fin(f"noplatform_g{gamma}", "op_std")
        print(f"\n=== gamma={gamma}   no-platform op_std -> {npl:.3f} (divratio {npl/init:.2f}) ===")
        print(f"  {'b':>4s} | " + " ".join(f"{r:>20s}" for r in REGIMES))
        for b in BS:
            cells = []
            for regime in REGIMES:
                k = f"b{b}_{regime}_g{gamma}"
                cells.append(f"os{fin(k,'op_std'):.3f} dr{fin(k,'op_std')/init:.2f}")
            print(f"  {b:>4.1f} | " + " ".join(f"{c:>20s}" for c in cells))
    print(f"\nsaved {OUT}/yelp_label_results.json")


main()
