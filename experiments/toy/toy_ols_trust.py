"""Trust-weighting toy: one shared epsilon gate, AI weight as the swept axis.

Question: with a single source-agnostic confidence gate (you accept any opinion,
person or AI, that is within epsilon), does varying the AI weight W_AI change the
outcome, or does the AI dominate the population at every weight because it acts on
all agents every round while a peer is one stochastic partner per step?

  W_AI=0      reproduces the no-platform population (AI is gated but moves nobody).
  dr flat low across W_AI  -> weight does not matter, AI dominates by reach,
                              motivates a separate (tighter) epsilon_AI.
  dr declines with W_AI    -> weight carries the difference, shared epsilon is enough.

gate = fraction of agents whose gate is open to the AI (|m_i - x_i| < epsilon). If
gate ~ 1 at low epsilon, the shared gate is not filtering the AI, only weight is.

Population = ndlib AlgorithmicBiasModel (Deffuant bounded confidence + gamma
homophily). Predictor = OLS of current opinion on the standardized feature z.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/toy/toy_ols_trust.py
"""
import json, os, random
import numpy as np
import networkx as nx
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

HERE = "experiments/toy"
OUT = f"{HERE}/results"
os.makedirs(OUT, exist_ok=True)

SMOKE = os.environ.get("SMOKE", "0") == "1"
N, ROUNDS = 300, 50
A_SLOPE, NOISE_SD = 0.8, 0.15
EPS_GRID = [0.08, 0.25]
W_AI_GRID = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
GAMMA_GRID = [-1.5, 0.0, 1.5, 3.0]
REGIMES = ["replace", "accumulate", "pristine"]
SEEDS = [0, 1]
if SMOKE:
    EPS_GRID, W_AI_GRID, GAMMA_GRID, SEEDS, ROUNDS = [0.08], [0.0, 0.5, 0.9], [0.0], [0], 20

_rng = np.random.default_rng(0)
z = _rng.uniform(0.0, 1.0, N)
z_std = (z - z.mean()) / (z.std() + 1e-9)
x0 = np.clip(0.5 + A_SLOPE * (z - 0.5) + _rng.normal(0.0, NOISE_SD, N), 0.0, 1.0)
G = nx.complete_graph(N)
F = np.column_stack([np.ones(N), z_std])
INIT_STD, INIT_MEAN = float(x0.std()), float(x0.mean())


def build_pop(gamma, eps, seed):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G)
    c = mc.Configuration()
    c.add_model_parameter("epsilon", eps)
    c.add_model_parameter("gamma", gamma)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x0[i])
    m.sts = x0.copy()
    m.iteration(False)
    return m


def ols(Xtr, ytr, Xpr):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0), w


def run(regime, gamma, eps, w_ai, seed):
    pop = build_pop(gamma, eps, seed)
    x = x0.copy()
    keep = np.random.default_rng(seed).permutation(N)[: N // 2]
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
        x = np.where(g, (1 - w_ai) * x + w_ai * m, x)
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
        ostd = float(x.std())
        traj.append(dict(op_std=ostd, op_mean=float(x.mean()), pred_std=float(m.std()),
                         vr=(float(m.std()) / ostd) if ostd > 1e-9 else float("nan"),
                         gate=float(g.mean())))
    return traj


def fin(traj, k):
    v = [r[k] for r in traj[-5:] if not (isinstance(r[k], float) and np.isnan(r[k]))]
    return sum(v) / len(v) if v else float("nan")


def agg(runs, regime, gamma, eps, w_ai, k):
    return float(np.mean([fin(t, k) for t in runs[(regime, gamma, eps, w_ai)]]))


def main():
    runs = {}
    for eps in EPS_GRID:
        for gamma in GAMMA_GRID:
            for regime in REGIMES:
                for w_ai in W_AI_GRID:
                    runs[(regime, gamma, eps, w_ai)] = [run(regime, gamma, eps, w_ai, s) for s in SEEDS]

    json.dump({"meta": dict(N=N, rounds=ROUNDS, init_std=INIT_STD, init_mean=INIT_MEAN,
                            eps=EPS_GRID, w_ai=W_AI_GRID, gamma=GAMMA_GRID, regimes=REGIMES, seeds=SEEDS),
               "runs": {f"{r}|{g}|{e}|{w}": v for (r, g, e, w), v in runs.items()}},
              open(f"{OUT}/toy_ols_trust.json", "w"))

    print(f"N={N} rounds={ROUNDS} seeds={SEEDS}  init op_std={INIT_STD:.3f} mean={INIT_MEAN:.3f}")
    print("dr = op_std/init (population spread retained; <1 collapsed).  W_AI=0 is the no-AI baseline.")
    wstr = "  ".join(f"{w:>4.1f}" for w in W_AI_GRID)
    for eps in EPS_GRID:
        for metric, label in [("op_std", "dr = op_std/init"), ("gate", "gate-open fraction"),
                              ("op_mean", "op_mean (displacement from init)")]:
            print(f"\n===== eps={eps}   {label}   cols = W_AI [{wstr}] =====")
            for regime in REGIMES:
                for gamma in GAMMA_GRID:
                    cells = []
                    for w_ai in W_AI_GRID:
                        v = agg(runs, regime, gamma, eps, w_ai, metric)
                        if metric == "op_std":
                            v = v / INIT_STD
                        cells.append(f"{v:>4.2f}")
                    print(f"  {regime:10s} g={gamma:>4} | " + "  ".join(cells))
    print(f"\nsaved {OUT}/toy_ols_trust.json")


if __name__ == "__main__":
    main()
