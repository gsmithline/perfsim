"""Separate AI gate: population keeps epsilon, the AI gets its own epsilon_AI.

Prior result (toy_ols_trust.py): at a wide population gate (pop_eps=0.25) the AI
reaches ~all agents and collapses the population at any trust weight, because the
shared gate does not filter it. Trust weight W_AI was a weak lever. This asks the
fix directly: give the AI its own threshold epsilon_AI and sweep it.

  epsilon_AI = 0     gate never opens = no-AI baseline (population alone).
  epsilon_AI = 0.25  reproduces the shared-gate case (= pop_eps here).
  dr climbs back toward the baseline as epsilon_AI shrinks  -> a separate, tighter
                                                              AI gate re-protects.

Population = ndlib AlgorithmicBiasModel (Deffuant bounded confidence + gamma
homophily) at pop_eps. AI gate uses epsilon_AI. W_AI fixed (weight is weak).

Run: MPLCONFIGDIR=/tmp/mpl python experiments/toy/toy_ols_eps_ai.py
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
POP_EPS = float(os.environ.get("POP_EPS", "0.25"))
W_AI = float(os.environ.get("W_AI", "0.5"))
EPS_AI_GRID = [0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50]
GAMMA_GRID = [-1.5, 0.0, 1.5, 3.0]
REGIMES = ["replace", "accumulate", "pristine"]
SEEDS = [0, 1]
if SMOKE:
    EPS_AI_GRID, GAMMA_GRID, SEEDS, ROUNDS = [0.0, 0.05, 0.25], [0.0, 3.0], [0], 20

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


def run(regime, gamma, pop_eps, eps_ai, w_ai, seed):
    pop = build_pop(gamma, pop_eps, seed)
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
        g = np.abs(m - x) < eps_ai                 # AI gate uses its OWN epsilon
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


def agg(runs, regime, gamma, eps_ai, k):
    return float(np.mean([fin(t, k) for t in runs[(regime, gamma, eps_ai)]]))


def main():
    runs = {}
    for gamma in GAMMA_GRID:
        for regime in REGIMES:
            for eps_ai in EPS_AI_GRID:
                runs[(regime, gamma, eps_ai)] = [run(regime, gamma, POP_EPS, eps_ai, W_AI, s) for s in SEEDS]

    json.dump({"meta": dict(N=N, rounds=ROUNDS, init_std=INIT_STD, init_mean=INIT_MEAN,
                            pop_eps=POP_EPS, w_ai=W_AI, eps_ai=EPS_AI_GRID, gamma=GAMMA_GRID,
                            regimes=REGIMES, seeds=SEEDS),
               "runs": {f"{r}|{g}|{e}": v for (r, g, e), v in runs.items()}},
              open(f"{OUT}/toy_ols_eps_ai.json", "w"))

    print(f"N={N} rounds={ROUNDS} seeds={SEEDS}  pop_eps={POP_EPS}  W_AI={W_AI}")
    print(f"init op_std={INIT_STD:.3f} mean={INIT_MEAN:.3f}.  eps_AI=0 is the no-AI baseline.")
    estr = "  ".join(f"{e:>4.2f}" for e in EPS_AI_GRID)
    for metric, label, scale in [("op_std", "dr = op_std/init (spread retained; 1=baseline, 0=collapsed)", INIT_STD),
                                 ("gate", "gate-open fraction (how many agents accept the AI)", 1.0),
                                 ("op_mean", "op_mean (displacement from init)", 1.0)]:
        print(f"\n===== {label}   cols = epsilon_AI [{estr}] =====")
        for regime in REGIMES:
            for gamma in GAMMA_GRID:
                cells = "  ".join(f"{agg(runs, regime, gamma, e, metric) / scale:>4.2f}" for e in EPS_AI_GRID)
                print(f"  {regime:10s} g={gamma:>4} | {cells}")
    print(f"\nsaved {OUT}/toy_ols_eps_ai.json")


if __name__ == "__main__":
    main()
