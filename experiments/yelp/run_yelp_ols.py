"""Yelp Acme loop, OLS predictor on OUR conditional-bias (AB/homophily) population.

This is NOT Jiduan's FJ setup. The population is the algorithmic-bias bounded-
confidence model (gamma = homophily, eps = confidence gate). The 1789-user LCC,
opinions = Acme star ratings in [0,1], real feature = average_stars (R2~0.067).

Grid: predictor {blind, real, perfect} x data-regime {replace, accumulate,
pristine} x gamma {0.0, 1.5}, eps fixed. Plus a no-platform baseline per gamma.
Tracks per-round op/pred std+mean, displacement, l_c0, and the fitted slope.
Writes everything to experiments/yelp/results/.
"""
import json, os, random
import numpy as np
import networkx as nx
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

HERE = "experiments/yelp"
OUT = f"{HERE}/results"
os.makedirs(OUT, exist_ok=True)

EPS, W, ROUNDS = 0.4, 0.3, 60
GAMMAS = [0.0, 1.5]
REGIMES = ["replace", "accumulate", "pristine"]
PREDICTORS = ["blind", "real", "perfect"]
SEED = 0

d = np.load(f"{HERE}/yelp_acme_lcc.npz", allow_pickle=True)
edges, x0_op = d["edges"], d["opinion"]
avg = d["avg_stars"].astype(float)
avg_z = (avg - avg.mean()) / (avg.std() + 1e-9)   # standardized real feature
N = len(x0_op)
G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(edges.tolist())


def build_pop(gamma, seed, x0):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G)
    c = mc.Configuration()
    c.add_model_parameter("epsilon", EPS)
    c.add_model_parameter("gamma", gamma)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x0[i])
    return m


def ols(Xtr, ytr, Xpr):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0), w


def design(pred, x):
    """feature matrix for a predictor given the current opinion x."""
    if pred == "blind":
        return np.ones((N, 1))
    if pred == "real":
        return np.column_stack([np.ones(N), avg_z])
    return np.column_stack([np.ones(N), x])           # perfect: current opinion


def run(pred, regime, gamma, use_platform=True):
    pop = build_pop(gamma, SEED, x0_op.copy())
    x = x0_op.copy()
    keep = np.random.default_rng(SEED).permutation(N)[: N // 2]   # pristine retained set
    HX, Hy = [], []
    traj = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        slope = float("nan")
        if use_platform:
            Fpr = design(pred, x)
            if regime == "replace":
                Xtr, ytr = Fpr, x
            elif regime == "accumulate":
                HX.append(Fpr); Hy.append(x)
                Xtr, ytr = np.vstack(HX), np.concatenate(Hy)
            else:  # pristine: current + retained original
                F0 = design(pred, x0_op)[keep]
                Xtr = np.vstack([Fpr, F0]); ytr = np.concatenate([x, x0_op[keep]])
            m, w = ols(Xtr, ytr, Fpr)
            if len(w) > 1:
                slope = float(w[1])
            g = np.abs(m - x) < EPS
            x = np.where(g, (1 - W) * x + W * m, x)
            for i in range(N):
                pop.status[i] = float(x[i])
            # l_c0: current model's error reproducing the original opinions
            m0, _ = ols(Xtr, ytr, design(pred, x0_op))
            lc0 = float(((m0 - x0_op) ** 2).mean())
            pstd, pmean = float(np.std(m)), float(np.mean(m))
        else:
            lc0 = float("nan"); pstd = pmean = float("nan")
        traj.append(dict(round=t, op_std=float(x.std()), op_mean=float(x.mean()),
                         pred_std=pstd, pred_mean=pmean,
                         pred_bias=(pmean - float(x0_op.mean())) if use_platform else float("nan"),
                         l_c0=lc0, slope=slope))
    return traj


def main():
    init_std = float(x0_op.std())
    results = {"meta": dict(N=N, eps=EPS, W=W, rounds=ROUNDS, init_op_std=init_std,
                            init_op_mean=float(x0_op.mean()), feat_R2_static=0.067)}
    runs = {}
    for gamma in GAMMAS:
        runs[f"noplatform_g{gamma}"] = run("blind", "replace", gamma, use_platform=False)
        for pred in PREDICTORS:
            for regime in REGIMES:
                key = f"{pred}_{regime}_g{gamma}"
                runs[key] = run(pred, regime, gamma)
    results["runs"] = runs
    json.dump(results, open(f"{OUT}/yelp_ols_results.json", "w"))

    # summary table
    def fin(key, k):
        tr = runs[key][-5:]
        v = [r[k] for r in tr if not (isinstance(r[k], float) and np.isnan(r[k]))]
        return sum(v) / len(v) if v else float("nan")
    print(f"Yelp Acme LCC  N={N}  eps={EPS} W={W}  init op_std={init_std:.3f} (mean {x0_op.mean():.3f})")
    for gamma in GAMMAS:
        print(f"\n===== gamma={gamma} =====")
        npl = fin(f"noplatform_g{gamma}", "op_std")
        print(f"  no-platform: op_std {init_std:.3f} -> {npl:.3f}  (divratio {npl/init_std:.2f})")
        print(f"  {'predictor':9s} {'regime':11s} | {'op_std':>7s} {'pred_std':>8s} {'divratio':>8s} "
              f"{'pred_bias':>9s} {'l_c0':>7s} {'slope':>7s}")
        for pred in PREDICTORS:
            for regime in REGIMES:
                k = f"{pred}_{regime}_g{gamma}"
                print(f"  {pred:9s} {regime:11s} | {fin(k,'op_std'):7.3f} {fin(k,'pred_std'):8.3f} "
                      f"{fin(k,'op_std')/init_std:8.2f} {fin(k,'pred_bias'):9.3f} {fin(k,'l_c0'):7.4f} "
                      f"{fin(k,'slope'):7.3f}")
    print(f"\nsaved {OUT}/yelp_ols_results.json")


main()
