"""Pure-population probe of ndlib AlgorithmicBiasModel (no platform).
Complete graph n=300, sweep epsilon x gamma, report final cluster count and std.
One model iteration = n pairwise interactions (a sweep).
"""

import numpy as np
import networkx as nx
import ndlib.models.ModelConfig as mc
from ndlib.models.opinions import AlgorithmicBiasModel

N = 300
SWEEPS = 600
SEEDS = [0, 1, 2]
EPSILONS = [0.1, 0.2, 0.3, 0.4]
GAMMAS = [0.0, 0.75, 1.5]
GAP = 0.02        # opinions closer than this are one cluster
MIN_SIZE = 5      # ignore stragglers below this size


def cluster_count(ops):
    s = np.sort(ops)
    breaks = np.where(np.diff(s) > GAP)[0]
    sizes = np.diff(np.concatenate([[0], breaks + 1, [len(s)]]))
    return int((sizes >= MIN_SIZE).sum()), int((sizes < MIN_SIZE).sum())


def run_cell(eps, gamma, seed):
    np.random.seed(seed)
    g = nx.complete_graph(N)
    model = AlgorithmicBiasModel(g)
    cfg = mc.Configuration()
    cfg.add_model_parameter("epsilon", eps)
    cfg.add_model_parameter("gamma", gamma)
    model.set_initial_status(cfg)
    for _ in range(SWEEPS + 1):
        model.iteration(False)
    ops = np.array(list(model.status.values()))
    nc, nout = cluster_count(ops)
    return nc, nout, ops.std()


def main():
    print(f"n={N} complete graph, {SWEEPS} sweeps, seeds={SEEDS}")
    print(f"{'eps':>5} {'gamma':>6} {'clusters':>14} {'outlier_grps':>12} {'std':>14}")
    for eps in EPSILONS:
        for gamma in GAMMAS:
            res = [run_cell(eps, gamma, s) for s in SEEDS]
            ncs = [r[0] for r in res]
            nouts = [r[1] for r in res]
            stds = [r[2] for r in res]
            print(f"{eps:>5} {gamma:>6} {str(ncs):>14} {str(nouts):>12} "
                  f"{np.mean(stds):>7.4f}+-{np.std(stds):.4f}")


if __name__ == "__main__":
    main()
