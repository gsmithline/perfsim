"""Sampling-budget version of the OLS-in-AB loop.

Same population (strong feature, complete graph) but each round the platform
fits its OLS on a fresh random sample of B agents (the redraw), then predicts
for everyone. B<N injects finite-sample noise into the fit, which the gate feeds
into the population so the population carries it forward. This is the model-
collapse channel (errors compound across generations) added on top of the
bounded-confidence geometry. B=N recovers the noiseless toy.

Regime x budget is the point: replace carries the full per-round noise,
accumulate pools B*t samples so the noise averages down, pristine anchors it.
"""
import json, os, random
import numpy as np
import networkx as nx
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "experiments/toy"
OUT, FIGS = f"{HERE}/results", f"{HERE}/figs"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

N, W, ROUNDS = 300, 0.3, 50
A_SLOPE, NOISE_SD = 0.8, 0.15
EPS_GRID = [0.08, 0.25, 0.60]
BUDGETS = [300, 100, 30, 10]
REGIMES = ["replace", "accumulate", "pristine"]
SEEDS = [0, 1, 2]
GAMMA = 0.0

_rng = np.random.default_rng(0)
z = _rng.uniform(0.0, 1.0, N)
z_std = (z - z.mean()) / (z.std() + 1e-9)
x0 = np.clip(0.5 + A_SLOPE * (z - 0.5) + _rng.normal(0.0, NOISE_SD, N), 0.0, 1.0)
G = nx.complete_graph(N)
F = np.column_stack([np.ones(N), z_std])
INIT_STD, INIT_MEAN = float(x0.std()), float(x0.mean())


def build_pop(eps, seed):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G)
    c = mc.Configuration()
    c.add_model_parameter("epsilon", eps); c.add_model_parameter("gamma", GAMMA)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x0[i])
    m.sts = x0.copy(); m.iteration(False)
    return m


def ols(Xtr, ytr, Xpr):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0), w


def run(regime, eps, B, seed):
    pop = build_pop(eps, seed)
    x = x0.copy()
    keep = np.random.default_rng(seed).permutation(N)[: N // 2]
    srng = np.random.default_rng(1000 + seed)      # independent sampling stream
    F0keep = F[keep]
    HX, Hy = [], []
    traj = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        samp = srng.permutation(N)[:B]             # the budget redraw (fit only)
        Fs, xs = F[samp], x[samp]
        if regime == "replace":
            Xtr, ytr = Fs, xs
        elif regime == "accumulate":
            HX.append(Fs); Hy.append(xs)
            Xtr, ytr = np.vstack(HX), np.concatenate(Hy)
        else:
            Xtr = np.vstack([Fs, F0keep]); ytr = np.concatenate([xs, x0[keep]])
        m, w = ols(Xtr, ytr, F)                    # predict for ALL N
        g = np.abs(m - x) < eps
        x = np.where(g, (1 - W) * x + W * m, x)
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
        ostd = float(x.std())
        traj.append(dict(op_std=ostd, op_mean=float(x.mean()), pred_std=float(m.std()),
                         vr=(float(m.std()) / ostd) if ostd > 1e-9 else float("nan")))
    return traj


def fin(traj, k):
    v = [r[k] for r in traj[-5:] if not (isinstance(r[k], float) and np.isnan(r[k]))]
    return sum(v) / len(v) if v else float("nan")


def main():
    runs = {}   # (regime,eps,B) -> list of trajs over seeds
    for regime in REGIMES:
        for eps in EPS_GRID:
            for B in BUDGETS:
                runs[(regime, eps, B)] = [run(regime, eps, B, s) for s in SEEDS]
    json.dump({"meta": dict(N=N, init_std=INIT_STD, init_mean=INIT_MEAN, budgets=BUDGETS,
                            eps=EPS_GRID, regimes=REGIMES, seeds=SEEDS, gamma=GAMMA),
               "runs": {f"{r}|{e}|{b}": v for (r, e, b), v in runs.items()}},
              open(f"{OUT}/toy_ols_budget.json", "w"))

    def agg(regime, eps, B, k):
        vals = [fin(t, k) for t in runs[(regime, eps, B)]]
        return np.mean(vals), np.std(vals)

    print(f"STRONG feature, complete graph, gamma=0.  N={N}, init op_std={INIT_STD:.3f} (mean {INIT_MEAN:.3f})")
    print(f"budget B = how many agents the platform refits on each round (B={N} is full/noiseless).")
    print(f"values = mean +/- std over seeds {SEEDS}.  vr = pred_std/op_std.\n")

    for eps, label in [(0.08, "LOW eps (population stays wide on its own)"),
                       (0.60, "HIGH eps (population collapses on its own)")]:
        print(f"===== eps={eps}  {label} =====")
        print(f"  {'regime':10s} {'B':>4} | {'op_std':>13} {'op_mean':>13} {'pred_std':>9} {'vr':>6}")
        for regime in REGIMES:
            for B in BUDGETS:
                om, os_ = agg(regime, eps, B, "op_std")
                mm, ms = agg(regime, eps, B, "op_mean")
                pm, _ = agg(regime, eps, B, "pred_std")
                vm, _ = agg(regime, eps, B, "vr")
                print(f"  {regime:10s} {B:>4} | {om:6.3f}+/-{os_:5.3f} {mm:6.3f}+/-{ms:5.3f} {pm:9.3f} {vm:6.2f}")
            print()

    # figure: op_std vs round at eps=0.08, one panel per regime, line per budget
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    cmap = {300: "#000000", 100: "#1f77b4", 30: "#ff7f0e", 10: "#d62728"}
    for a, regime in zip(ax, REGIMES):
        for B in BUDGETS:
            curves = np.array([[r["op_std"] for r in t] for t in runs[(regime, 0.08, B)]])
            a.plot(range(ROUNDS), curves.mean(0), c=cmap[B], label=f"B={B}")
        a.axhline(INIT_STD, ls=":", c="gray", lw=1)
        a.set(xlabel="round", title=f"{regime} (eps=0.08)", ylim=(0, INIT_STD * 1.1))
    ax[0].set_ylabel("population op_std")
    ax[0].legend(frameon=False, fontsize=8, title="budget")
    fig.suptitle("Does finite sampling make the population carry the collapse?  (strong feature, eps=0.08)")
    fig.savefig(f"{FIGS}/toy_ols_budget.png", dpi=130)
    print(f"saved {OUT}/toy_ols_budget.json and {FIGS}/toy_ols_budget.png")


if __name__ == "__main__":
    main()
