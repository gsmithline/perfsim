"""Sampling-budget OLS-in-AB loop, full grid: feature strength x eps x gamma x
regime x budget, averaged over seeds. Budget is a redraw on the fit only;
predictions go to everyone; the population carries the feedback.
"""
import json, os, random
import numpy as np
import networkx as nx
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

HERE = "experiments/toy"
OUT = f"{HERE}/results"
os.makedirs(OUT, exist_ok=True)

N, W, ROUNDS = 300, 0.3, 50
FEATURES = {"strong": (0.8, 0.15), "weak": (0.25, 0.25)}
EPS_GRID = [0.08, 0.25, 0.60]
GAMMA_GRID = [0.0, 1.5, 3.0]
BUDGETS = [300, 30, 10]
REGIMES = ["replace", "accumulate", "pristine"]
SEEDS = [0, 1, 2]

_rng = np.random.default_rng(0)
z = _rng.uniform(0.0, 1.0, N)
z_std = (z - z.mean()) / (z.std() + 1e-9)
base_noise = _rng.normal(0.0, 1.0, N)          # fixed unit noise, scaled per feature
G = nx.complete_graph(N)
F = np.column_stack([np.ones(N), z_std])


def make_x0(a, sd):
    return np.clip(0.5 + a * (z - 0.5) + sd * base_noise, 0.0, 1.0)


def feat_R2(x0):
    w, *_ = np.linalg.lstsq(F, x0, rcond=None)
    fit = F @ w
    return 1.0 - ((x0 - fit) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum()


def build_pop(x0, gamma, eps, seed):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G)
    c = mc.Configuration()
    c.add_model_parameter("epsilon", eps); c.add_model_parameter("gamma", gamma)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x0[i])
    m.sts = x0.copy(); m.iteration(False)
    return m


def ols(Xtr, ytr, Xpr):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0), w


def run(x0, regime, eps, gamma, B, seed):
    pop = build_pop(x0, gamma, eps, seed)
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
        pop.sts = x.copy()
        ostd = float(x.std())
        traj.append(dict(op_std=ostd, op_mean=float(x.mean()), pred_std=float(m.std()),
                         vr=(float(m.std()) / ostd) if ostd > 1e-9 else float("nan")))
    return traj


def finmean(trajs, k):
    vals = []
    for t in trajs:
        v = [r[k] for r in t[-5:] if not (isinstance(r[k], float) and np.isnan(r[k]))]
        if v:
            vals.append(sum(v) / len(v))
    return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))


def main():
    allruns = {}
    meta_feat = {}
    for fname, (a, sd) in FEATURES.items():
        x0 = make_x0(a, sd)
        meta_feat[fname] = dict(init_std=float(x0.std()), init_R2=float(feat_R2(x0)),
                                init_mean=float(x0.mean()))
        for eps in EPS_GRID:
            for gamma in GAMMA_GRID:
                for regime in REGIMES:
                    for B in BUDGETS:
                        key = f"{fname}|{eps}|{gamma}|{regime}|{B}"
                        allruns[key] = [run(x0, regime, eps, gamma, B, s) for s in SEEDS]

    json.dump({"meta": dict(N=N, W=W, rounds=ROUNDS, features=meta_feat, eps=EPS_GRID,
                            gamma=GAMMA_GRID, budgets=BUDGETS, regimes=REGIMES, seeds=SEEDS),
               "runs": {k: v for k, v in allruns.items()}},
              open(f"{OUT}/toy_ols_budget_grid.json", "w"))

    print(f"N={N} complete graph  W={W}  rounds={ROUNDS}  seeds={SEEDS}")
    print(f"cells = op_std / vr  (mean over seeds).  vr=pred_std/op_std.  budget B = refit sample size.\n")
    for fname in FEATURES:
        mf = meta_feat[fname]
        print("=" * 78)
        print(f"FEATURE = {fname.upper()}   init op_std={mf['init_std']:.3f}  init R2={mf['init_R2']:.3f}")
        print("=" * 78)
        bh = "  ".join(f"B={B:<3}".rjust(13) for B in BUDGETS)
        print(f"  {'eps':>4} {'gamma':>5} {'regime':10s} |{bh}")
        print("  " + "-" * (24 + 13 * len(BUDGETS)))
        for eps in EPS_GRID:
            for gamma in GAMMA_GRID:
                for regime in REGIMES:
                    cells = []
                    for B in BUDGETS:
                        os_, _ = finmean(allruns[f"{fname}|{eps}|{gamma}|{regime}|{B}"], "op_std")
                        vr_, _ = finmean(allruns[f"{fname}|{eps}|{gamma}|{regime}|{B}"], "vr")
                        cells.append(f"{os_:.3f}/{vr_:.2f}".rjust(13))
                    print(f"  {eps:>4} {gamma:>5} {regime:10s} |{'  '.join(cells)}")
                print()
    print(f"saved {OUT}/toy_ols_budget_grid.json")


if __name__ == "__main__":
    main()
