"""Freeze test: is the population the engine of the collapse, or just the conduit?

Same fixed feature design F, same AI gate + blend, replace regime. The only knob
is whether the population runs its own Deffuant/AB step between the model's
projections.

  frozen : no autonomous dynamics. Opinions change only through the AI gate.
           The projection P_F is then a fixed map, the feature-explained part of x
           is invariant, so this can only strip the residual (1 - R^2) and must
           plateau at dr = sqrt(R^2). One-shot, bounded (Dohmatob dependent case).
  full   : Deffuant/AB on. Averaging keeps shrinking the slope between projections,
           so the projection never becomes idempotent and the collapse continues.

If frozen plateaus at sqrt(R^2) while full grinds below it, the population's own
dynamics are what sustain the collapse past the one-shot projection bound.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/toy/toy_ols_freeze.py
"""
import os, random
import numpy as np
import networkx as nx
from ndlib.models.opinions import AlgorithmicBiasModel
import ndlib.models.ModelConfig as mc

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "experiments/toy"
FIGS = f"{HERE}/figs"
os.makedirs(FIGS, exist_ok=True)

N, ROUNDS, GAMMA, POP_EPS, W_AI = 300, 80, 0.0, 0.25, 0.5
A_SLOPE, NOISE_SD = 0.8, 0.15
SEEDS = [0, 1, 2]
EPS_AI_GRID = [0.05, 0.10, 0.25]

_rng = np.random.default_rng(0)
z = _rng.uniform(0.0, 1.0, N)
z_std = (z - z.mean()) / (z.std() + 1e-9)
x0 = np.clip(0.5 + A_SLOPE * (z - 0.5) + _rng.normal(0.0, NOISE_SD, N), 0.0, 1.0)
G = nx.complete_graph(N)
F = np.column_stack([np.ones(N), z_std])
INIT_STD = float(x0.std())

_w0 = np.linalg.lstsq(F, x0, rcond=None)[0]
INIT_R2 = float(1.0 - ((x0 - F @ _w0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum())
FROZEN_PLATEAU = float(np.sqrt(INIT_R2))   # predicted frozen dr


def build_pop(seed):
    random.seed(seed); np.random.seed(seed)
    m = AlgorithmicBiasModel(G)
    c = mc.Configuration()
    c.add_model_parameter("epsilon", POP_EPS)
    c.add_model_parameter("gamma", GAMMA)
    m.set_initial_status(c)
    for i in range(N):
        m.status[i] = float(x0[i])
    m.sts = x0.copy()
    m.iteration(False)
    return m


def ols(Xtr, ytr, Xpr):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return np.clip(Xpr @ w, 0.0, 1.0), w


def run(frozen, eps_ai, seed):
    pop = None if frozen else build_pop(seed)
    x = x0.copy()
    ostd = []
    for t in range(ROUNDS):
        if not frozen:
            pop.iteration(node_status=False)
            x = np.array([pop.status[i] for i in range(N)])
        m, w = ols(F, x, F)                 # replace: fit current pop
        g = np.abs(m - x) < eps_ai
        x = np.where(g, (1 - W_AI) * x + W_AI * m, x)
        if not frozen:
            for i in range(N):
                pop.status[i] = float(x[i])
            pop.sts = x.copy()
        ostd.append(float(x.std()))
    return np.array(ostd)


def main():
    print(f"gamma={GAMMA} pop_eps={POP_EPS} W_AI={W_AI} replace  rounds={ROUNDS} seeds={SEEDS}")
    print(f"init op_std={INIT_STD:.3f}  init R2={INIT_R2:.3f}  predicted frozen plateau dr=sqrt(R2)={FROZEN_PLATEAU:.3f}\n")
    print(f"  {'cond':7s} {'eps_AI':>6} | r0    r5    r10   r20   r40   r79")

    curves = {}
    for frozen in [True, False]:
        for ea in EPS_AI_GRID:
            dr = np.mean([run(frozen, ea, s) for s in SEEDS], axis=0) / INIT_STD
            curves[(frozen, ea)] = dr
            cells = "  ".join(f"{dr[r]:.2f}" for r in [0, 5, 10, 20, 40, ROUNDS - 1])
            print(f"  {'frozen' if frozen else 'full':7s} {ea:>6} | {cells}")

    fig, axes = plt.subplots(1, len(EPS_AI_GRID), figsize=(5 * len(EPS_AI_GRID), 4.4), constrained_layout=True)
    for ax, ea in zip(axes, EPS_AI_GRID):
        ax.plot(range(ROUNDS), curves[(True, ea)], lw=2, c="#1f77b4", label="frozen (AI only)")
        ax.plot(range(ROUNDS), curves[(False, ea)], lw=2, c="#d62728", label="full (AI + Deffuant)")
        ax.axhline(FROZEN_PLATEAU, ls="--", c="gray", lw=1.2, label=f"sqrt(R2)={FROZEN_PLATEAU:.2f}")
        ax.set(xlabel="round", ylim=(0, 1.1), title=f"eps_AI = {ea}")
    axes[0].set_ylabel("op_std(t) / init")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Freeze test: does the population sustain the collapse? (gamma={GAMMA}, replace)")
    fig.savefig(f"{FIGS}/freeze_test.png", dpi=130)
    print(f"\nsaved {FIGS}/freeze_test.png")


if __name__ == "__main__":
    main()
