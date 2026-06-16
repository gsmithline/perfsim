"""Block 5: attractor phase diagram over (feature informativeness, anchor spacing).

For each (sigma_f, delta) cell, run from two starts:
  spread: population at innate, platforms at their own anchors.
  merged: population concentrated at C0, all platforms initialized from a
          C0-pretrained net (anchor losses still pull toward their own bases).
Classify the endpoint:
  A segmented          platforms apart, population spread
  B product-monoculture platforms together, population spread
  C collapse           platforms together, population narrow
Cells where the two starts disagree are bistable (path dependence).

Uniform innate, real Pokec graph, K=3 learning MLP platforms (block 4 machinery).

Run: python experiments/competition/05_phase_attractors.py
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_spec = importlib.util.spec_from_file_location("b4", "experiments/competition/circle/04_mlp_circle.py")
b4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b4)

OUT = Path("experiments/competition/circle/figs")

SIGMAS = [0.05, 0.15, 0.30, 0.60, 2.0]
DELTAS = [0.0, 0.05, 0.10, 0.20, 1 / 3]
SEEDS = [0, 1]
ROUNDS = 40
INNATE_KIND = sys.argv[1] if len(sys.argv) > 1 else "uniform"

PLAT_APART = 0.08
POP_WIDE = 0.5


def run_start(delta, sigma_f, seed, start):
    """One run; returns (mean pairwise platform dist, pop circ-var) at the end."""
    b4.ROUNDS = ROUNDS
    centers = b4.centers_for("shared" if delta == 0.0 else "distinct", delta)
    if start == "spread":
        traj = b4.run_cell("distinct", delta, INNATE_KIND, seed,
                           sigma_f=sigma_f, centers=centers)
    else:
        traj = b4.run_cell_merged(centers, seed, sigma_f, innate_kind=INNATE_KIND)
    last = traj[-5:]
    spread = float(np.mean([np.mean(r["pdist"]) for r in last]))
    var = float(np.mean([r["pop_var"] for r in last]))
    return spread, var


def classify(spread, var):
    if spread > PLAT_APART and var > POP_WIDE:
        return "A"
    if var > POP_WIDE:
        return "B"
    return "C"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    print(f"phase grid [{INNATE_KIND}]: sigma {SIGMAS} x delta {[round(d,3) for d in DELTAS]}, "
          f"2 starts, {len(SEEDS)} seeds")
    print(f"{'sigma':>6} {'delta':>6} | {'spread-start':>14} | {'merged-start':>14} | bistable")
    for sf in SIGMAS:
        for d in DELTAS:
            cls = {}
            for start in ("spread", "merged"):
                sv = [run_start(d, sf, s, start) for s in SEEDS]
                spread = np.mean([v[0] for v in sv])
                var = np.mean([v[1] for v in sv])
                cls[start] = (classify(spread, var), spread, var)
            bist = cls["spread"][0] != cls["merged"][0]
            res[f"{sf}|{d:.3f}"] = {k: v for k, v in cls.items()} | {"bistable": bist}
            a, b = cls["spread"], cls["merged"]
            print(f"{sf:>6.2f} {d:>6.3f} | {a[0]} sp={a[1]:.3f} v={a[2]:.2f} | "
                  f"{b[0]} sp={b[1]:.3f} v={b[2]:.2f} | {'YES' if bist else ''}", flush=True)

    json.dump(res, open(OUT / f"phase_attractors_{INNATE_KIND}.json", "w"), default=str)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    colors = {"A": 2, "B": 1, "C": 0}
    for ax, start, title in [(axes[0], "spread", "from spread start"),
                             (axes[1], "merged", "from merged start")]:
        grid = np.zeros((len(SIGMAS), len(DELTAS)))
        for i, sf in enumerate(SIGMAS):
            for j, d in enumerate(DELTAS):
                grid[i, j] = colors[res[f"{sf}|{d:.3f}"][start][0]]
        ax.imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=2, aspect="auto")
        for i, sf in enumerate(SIGMAS):
            for j, d in enumerate(DELTAS):
                cell = res[f"{sf}|{d:.3f}"]
                mark = cell[start][0] + ("*" if cell["bistable"] else "")
                ax.text(j, i, mark, ha="center", va="center", color="white", fontsize=11)
        ax.set(xticks=range(len(DELTAS)), xticklabels=[f"{d:.2f}" for d in DELTAS],
               yticks=range(len(SIGMAS)), yticklabels=[f"{s:.2f}" for s in SIGMAS],
               xlabel="anchor spacing delta", ylabel="feature noise sigma", title=title)
    fig.suptitle("A=segmented  B=product monoculture (people safe)  C=collapse  *=bistable")
    fig.savefig(OUT / f"phase_attractors_{INNATE_KIND}.png", dpi=130)
    print(f"saved {OUT}/phase_attractors_{INNATE_KIND}.png")


if __name__ == "__main__":
    main()
