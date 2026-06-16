"""Does innate-graph alignment (homophily) change the collapse transition?

Real Pokec graph, alcohol innate mapped to the arc [0, 0.5] so the ordinal
extremes stay antipodal (no wraparound). Aligned = labels on their true nodes
(edge assortativity ~0.165); shuffled = same values permuted across labeled
nodes (assortativity ~0). Missing labels imputed by sampling the labeled
marginal. Identical marginal + graph, only alignment differs.

Run: python experiments/competition/fig_innate_alignment.py
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

sys.path.insert(0, "experiments/fj")
from _pokec import load_pokec  # noqa: E402

_spec = importlib.util.spec_from_file_location("b2", "experiments/competition/circle/02_phase_diagram.py")
b2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b2)

OUT = Path("experiments/competition/circle/figs")

ALC = {"abstinent": 0.0, "nepijem": 0.0, "uz nepijem": 0.17,
       "abstinent, pijem prilezitostne": 0.33, "pijem prilezitostne": 0.5,
       "prilezitostne": 0.5, "pijem pravidelne": 1.0}
MALL = [0.2, 0.3, 0.4, 0.5]
SEEDS = [0, 1]
K = 3


def load_alcohol():
    import pickle
    with open("examples/pokec/lcc_profiles_relation_to_smoking.pk", "rb") as fh:
        df = pickle.load(fh)
    return df["relation_to_alcohol"].map(ALC).values.astype(float)


def make_innate(alc, shuffle, seed):
    rng = np.random.default_rng(seed)
    v = alc.copy()
    lab = ~np.isnan(v)
    if shuffle:
        v[lab] = rng.permutation(v[lab])
    v[~lab] = rng.choice(v[lab], size=(~lab).sum())
    return torch.tensor(v * 0.5, dtype=torch.float32)  # arc: extremes antipodal


def edge_assort(v, adj):
    r, c = adj.nonzero(as_tuple=True)
    return float(np.corrcoef(v[r].numpy(), v[c].numpy())[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pk = load_pokec()
    alc = load_alcohol()
    res = {}
    print(f"Alignment test: real Pokec graph, alcohol-on-arc innate, K={K}, {len(SEEDS)} seeds.")
    print(f"{'cond':>9} {'mall':>5} | {'sep':>6} | {'div':>6} | innate assort")
    for shuffle, cond in [(False, "aligned"), (True, "shuffled")]:
        for m in MALL:
            sep, div, ass = [], [], []
            for s in SEEDS:
                innate = make_innate(alc, shuffle, s)
                ass.append(edge_assort(innate, pk["adj"]))
                pos, x = b2.run(m, k=K, graph=pk["W"], innate=innate, seed=s)
                sep.append(b2.plat_min_gap(pos) * K)
                div.append(b2.circ_var(x))
            res[f"{cond}_{m}"] = {"sep": sep, "div": div, "assort": ass}
            print(f"{cond:>9} {m:>5.2f} | {np.mean(sep):>6.3f} | {np.mean(div):>6.3f} | "
                  f"{np.mean(ass):.3f}", flush=True)

    json.dump(res, open(OUT / "innate_alignment.json", "w"))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for cond, color in [("aligned", "#3b6fd8"), ("shuffled", "#d8633b")]:
        for i, key in enumerate(["sep", "div"]):
            mu = [np.mean(res[f"{cond}_{m}"][key]) for m in MALL]
            sd = [np.std(res[f"{cond}_{m}"][key]) for m in MALL]
            ax[i].errorbar(MALL, mu, yerr=sd, color=color, marker="o", label=cond, capsize=3)
    ax[0].set(xlabel="malleability", ylabel="platform sep", title="segmentation")
    ax[1].set(xlabel="malleability", ylabel="pop circ-var", title="population diversity")
    ax[0].legend(frameon=False)
    fig.savefig(OUT / "innate_alignment.png", dpi=150)
    print(f"saved {OUT / 'innate_alignment.png'}")


if __name__ == "__main__":
    main()
