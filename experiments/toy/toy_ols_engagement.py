"""Platform trains on its USERS, not everyone.

train_on_users=False reproduces the earlier "train on everyone" runs.

Readouts: dr = op_std/init (population spread), vr = pred_std/op_std (platform vs
population, < 1 means platform narrower), ufrac = fraction of nodes engaged.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/toy/toy_ols_engagement.py
"""
import os, random
import numpy as np
import networkx as nx
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

HERE = "experiments/toy"
OUT = f"{HERE}/results"
os.makedirs(OUT, exist_ok=True)

N, ROUNDS, POP_EPS, W_AI = 300, 60, 0.25, 0.5
D = 2
A_SLOPE, NOISE_SD = 0.8, 0.15
EPS_AI_GRID = [0.05, 0.10, 0.25]
GAMMA_GRID = [-1.5, 0.0, 1.5, 3.0]
REGIMES = ["replace", "accumulate", "pristine"]
SEEDS = [0, 1]

_rng = np.random.default_rng(0)
z = _rng.uniform(0.0, 1.0, N)
z_std = (z - z.mean()) / (z.std() + 1e-9)
x0 = np.clip(0.5 + A_SLOPE * (z - 0.5) + _rng.normal(0.0, NOISE_SD, N), 0.0, 1.0)
G = nx.complete_graph(N)
F = np.column_stack([np.ones(N), z_std])
INIT_STD = float(x0.std())


def build_pop(gamma, seed):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G)
    c = mc.Configuration()
    c.add_model_parameter("epsilon", POP_EPS)
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


def run(regime, gamma, eps_ai, seed, train_on_users):
    pop = build_pop(gamma, seed)
    x = x0.copy()
    keep = np.random.default_rng(seed).permutation(N)[: N // 2]
    F0keep = F[keep]
    HX, Hy = [], []
    U = np.arange(N)                                  # round 0: everyone uses it
    traj = []
    for t in range(ROUNDS):
        pop.iteration(node_status=False)
        x = np.array([pop.status[i] for i in range(N)])
        tr = U if (train_on_users and len(U) >= D + 2) else np.arange(N)
        if regime == "replace":
            Xtr, ytr = F[tr], x[tr]
        elif regime == "accumulate":
            HX.append(F[tr]); Hy.append(x[tr].copy())
            Xtr, ytr = np.vstack(HX), np.concatenate(Hy)
        else:
            Xtr = np.vstack([F[tr], F0keep]); ytr = np.concatenate([x[tr], x0[keep]])
        m, w = ols(Xtr, ytr, F)                 
        g = np.abs(m - x) < eps_ai
        U = np.where(g)[0]            
        x = np.where(g, (1 - W_AI) * x + W_AI * m, x)
        for i in range(N):
            pop.status[i] = float(x[i])
        pop.sts = x.copy()
        ostd = float(x.std()); pstd = float(m.std())
        traj.append(dict(op_std=ostd, pred_std=pstd,
                         vr=(pstd / ostd) if ostd > 1e-9 else float("nan"),
                         op_mean=float(x.mean()), ufrac=float(g.mean())))
    return traj


def fin(traj, k):
    v = [r[k] for r in traj[-5:] if not (isinstance(r[k], float) and np.isnan(r[k]))]
    return sum(v) / len(v) if v else float("nan")


def agg(regime, gamma, eps_ai, tou, k):
    return float(np.mean([fin(run(regime, gamma, eps_ai, s, tou), k) for s in SEEDS]))


def main():
    print(f"N={N} pop_eps={POP_EPS} W={W_AI} rounds={ROUNDS} seeds={SEEDS} init_std={INIT_STD:.3f}")
    print("dr=op_std/init  vr=pred_std/op_std (platform vs population)  ufrac=engaged fraction")
    print("LEFT = train on everyone   RIGHT = train on users only\n")
    for regime in REGIMES:
        print(f"===== regime = {regime} =====")
        print(f"  {'gamma':>5} {'eps_AI':>6} | {'all_dr':>6} {'all_vr':>6} || {'usr_dr':>6} {'usr_vr':>6} {'ufrac':>6}")
        for gamma in GAMMA_GRID:
            for ea in EPS_AI_GRID:
                adr = agg(regime, gamma, ea, False, "op_std") / INIT_STD
                avr = agg(regime, gamma, ea, False, "vr")
                udr = agg(regime, gamma, ea, True, "op_std") / INIT_STD
                uvr = agg(regime, gamma, ea, True, "vr")
                uf = agg(regime, gamma, ea, True, "ufrac")
                print(f"  {gamma:>5} {ea:>6} | {adr:6.2f} {avr:6.2f} || {udr:6.2f} {uvr:6.2f} {uf:6.2f}")
            print()


if __name__ == "__main__":
    main()
