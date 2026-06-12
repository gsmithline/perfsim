"""Block 7: one anchored platform inside NDlib's attraction-repulsion HK model.

Population: ARWHK (bounded confidence, optional repulsion/backfire), opinions in
[-1, 1]. Platform: one hub node connected to everyone with heavier edge weight;
between sweeps its opinion is set by the measured LLM reduced form
    m = (1 - a) * mean(population) + a * p0
with anchor strength a and prior p0. All training data is platform-mediated
human opinion (the AI-mediated-human-data setting, single platform).

Run: python experiments/competition/07_ndlib_one_platform.py
"""

import os
import random

import networkx as nx
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op

N = 400
EPS = 0.3
P0 = 0.6
ROUNDS = 150
PEER_W = 0.1
HUB_W = 0.4
SEED = 0


def build(variant, with_platform):
    random.seed(SEED)
    np.random.seed(SEED)
    g = nx.erdos_renyi_graph(N, 0.03, seed=SEED)
    hub = N
    if with_platform:
        g.add_node(hub)
        g.add_edges_from((hub, i) for i in range(N))
    model = op.ARWHKModel(g)
    cfg = mc.Configuration()
    cfg.add_model_parameter("epsilon", EPS)
    cfg.add_model_parameter("method_variant", variant)
    for e in g.edges():
        w = HUB_W if hub in e else PEER_W
        cfg.add_edge_configuration("weight", e, w)
    model.set_initial_status(cfg)
    return model, hub


def run(variant, anchor, with_platform=True):
    model, hub = build(variant, with_platform)
    traj = []
    for t in range(ROUNDS):
        model.iteration_bunch(N, node_status=False)
        opin = np.array([v for k, v in model.status.items() if k != hub])
        if with_platform:
            m = float(np.clip((1 - anchor) * opin.mean() + anchor * P0, -0.999, 0.999))
            model.status[hub] = m
        else:
            m = float("nan")
        traj.append((opin.mean(), opin.std(), m,
                     float((np.abs(opin - P0) < EPS).mean()),
                     float((opin < -0.3).mean())))
    return traj, opin


def n_clusters(opin, bins=40):
    h, _ = np.histogram(opin, bins=bins, range=(-1, 1))
    th = 0.05 * len(opin)
    peaks, inside = 0, False
    for c in h:
        if c > th and not inside:
            peaks += 1
            inside = True
        elif c <= th:
            inside = False
    return peaks


def main():
    rows = []
    print(f"ARWHK n={N} eps={EPS} prior p0={P0}; platform hub weight {HUB_W}")
    print(f"{'variant':>8} {'anchor':>6} | {'mean_T':>6} {'std_T':>6} {'near_p0':>7} "
          f"{'opp_mass':>8} {'clusters':>8}")
    finals = {}
    for variant in (1, 3):
        for label, a, wp in [("none", 0.0, False), ("mirror", 0.0, True),
                             ("a=0.5", 0.5, True), ("a=1.0", 1.0, True)]:
            traj, opin = run(variant, a, wp)
            last = traj[-1]
            finals[(variant, label)] = (traj, opin)
            print(f"{variant:>8} {label:>6} | {last[0]:>6.2f} {last[1]:>6.2f} "
                  f"{last[3]:>7.2f} {last[4]:>8.2f} {n_clusters(opin):>8}", flush=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), constrained_layout=True)
    for i, variant in enumerate((1, 3)):
        for j, label in enumerate(("none", "mirror", "a=0.5", "a=1.0")):
            traj, opin = finals[(variant, label)]
            ax = axes[i, j]
            ax.hist(opin, bins=40, range=(-1, 1), color="#1f77b4")
            ax.axvline(P0, c="#d62728", ls="--", lw=1.2)
            mt = [r[2] for r in traj]
            if not np.isnan(mt[-1]):
                ax.axvline(mt[-1], c="black", lw=1.2)
            vname = "attraction" if variant == 1 else "attr+repulsion"
            ax.set(title=f"{vname}, {label}", xticks=[-1, 0, 1])
    fig.suptitle("final opinion distributions (red dashed = prior p0, black = platform)")
    fig.savefig("experiments/competition/figs/ndlib_one_platform.png", dpi=130)
    print("saved experiments/competition/figs/ndlib_one_platform.png")


if __name__ == "__main__":
    main()
