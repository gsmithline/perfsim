"""Transition figure: competition collapse on the Pokec graph (uniform innate).

Sweeps malleability on the real Pokec social graph with a uniform circular
innate population (topology only). Plots platform separation and population
diversity vs malleability, averaged over seeds with the seed spread shaded.
"""

import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "experiments/fj")
from _pokec import load_pokec  # noqa: E402

_spec = importlib.util.spec_from_file_location("b2", "experiments/competition/circle/02_phase_diagram.py")
b2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b2)

K = 3
SEEDS = [0, 1, 2, 3]
MALL = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]


def main():
    w_graph = load_pokec()["W"]
    sep = np.zeros((len(MALL), len(SEEDS)))
    div = np.zeros((len(MALL), len(SEEDS)))
    for i, m in enumerate(MALL):
        for j, s in enumerate(SEEDS):
            pos, x = b2.run(m, k=K, graph=w_graph, seed=s)
            sep[i, j] = b2.plat_min_gap(pos) * K  # 1.0 = max segmentation
            div[i, j] = b2.circ_var(x)
        print(f"mall={m:.2f}  sep={sep[i].mean():.3f}  div={div[i].mean():.3f}", flush=True)

    mall = np.array(MALL)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arr, color, label in [
        (sep, "C0", "platform separation"),
        (div, "C3", "population diversity"),
    ]:
        mu, sd = arr.mean(1), arr.std(1)
        ax.plot(mall, mu, "-o", color=color, label=label)
        ax.fill_between(mall, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=color, alpha=0.2)
    ax.set_xlabel("malleability (how far users move toward platforms)")
    ax.set_ylabel("normalized (1 = diverse, 0 = collapsed)")
    ax.set_title("Competition collapse on the Pokec graph (uniform innate, K=3)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = "experiments/competition/circle/figs/phase_pokec.png"
    import os

    os.makedirs("experiments/competition/circle/figs", exist_ok=True)
    fig.savefig(out, dpi=150)
    print("saved", out)
    np.savez("experiments/competition/circle/figs/phase_pokec.npz", mall=mall, sep=sep, div=div)


if __name__ == "__main__":
    main()
