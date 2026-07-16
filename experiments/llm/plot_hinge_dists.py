"""Act I/II hinge, companion figure: the r30 opinion DISTRIBUTIONS behind the
2x2 dr numbers -- the mechanism the heatmap summarizes.

Same four corners and cell as plot_hinge_2x2.py (Qwen ML-Action, eps=0.4,
replace/continual, beta=0, seed 0). Each cell = the population's opinion
histogram at round 30, with the innate distribution outlined for reference:

  no-AI / no feedback (serve off): population self-collapses to a narrow blob
      -- and the two are ~identical, because TRAINING alone never touches the
      population (that's the point).
  frozen (serve on, train off): the model SORTS agents onto its two prior
      modes -> a BIMODAL distribution -> that is why frozen's dr=0.71 is high
      (spread across two peaks, not one wide band).
  closed (serve on, train on): the trained model sharpens and pulls the
      population into ONE collapsed spike -> dr back down to 0.27.

no-AI opinions are regenerated locally (Deffuant-AB ab_sweep, gamma=0,
eps=0.4, seed 0, 30 rounds) -- the exact machinery of plot_pop_alone.py.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/llm/plot_hinge_dists.py
Pure numpy/torch -- no transformers, no download.
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch

ML = "experiments/data/movielens/ml-100k"
RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs"
TAIL = 5
EPS = 0.4
CORE = ["Drama", "Romance", "Comedy", "Action", "Thriller", "War", "Crime",
        "Sci-Fi", "Adventure", "Mystery", "Children's"]

spec = importlib.util.spec_from_file_location(
    "gp", "experiments/scripts/cluster_pipelines/_gated_pop.py")
gp = importlib.util.module_from_spec(spec); spec.loader.exec_module(gp)


def ml_action_setup():
    """Exact replica of plot_pop_alone.ml_action_setup (genre-mean profiles,
    cosine kNN graph on non-Action genres, LCC, innate=(Action-1)/4)."""
    gen = pd.read_csv(f"{ML}/u.genre", sep="|", names=["name", "gid"], encoding="latin-1")
    genres = list(gen.sort_values("gid")["name"])
    items = pd.read_csv(f"{ML}/u.item", sep="|", encoding="latin-1", header=None)
    gmat = pd.DataFrame(items.iloc[:, 5:5 + len(genres)].values,
                        index=items[0].values, columns=genres)
    rat = pd.read_csv(f"{ML}/u.data", sep="\t", names=["uid", "iid", "r", "t"]).merge(
        gmat, left_on="iid", right_index=True)
    P = pd.DataFrame({g: rat[rat[g] == 1].groupby("uid")["r"].mean() for g in CORE}).dropna()
    feats = [g for g in CORE if g != "Action"]
    Zc = P[feats].values - P[feats].values.mean(0)
    norm = Zc / (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)
    nbrs = np.argsort(-sim, axis=1)[:, :10]
    graph = nx.Graph(); graph.add_nodes_from(range(len(P)))
    for i, row in enumerate(nbrs):
        for j in row:
            graph.add_edge(i, int(j))
    lcc = sorted(max(nx.connected_components(graph), key=len))
    h = nx.relabel_nodes(graph.subgraph(lcc).copy(), {n: k for k, n in enumerate(lcc)})
    Pl = P.iloc[lcc]
    innate = ((Pl["Action"].values - 1.0) / 4.0).astype(np.float64)
    adj = (torch.tensor(nx.to_numpy_array(h, nodelist=range(len(Pl))),
                        dtype=torch.float32) > 0).float()
    return torch.tensor(innate, dtype=torch.float32), adj


def run_ops(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    return op[-1], float(op[-TAIL:].std(1).mean() / (inn.std() + 1e-9))


# ---- no-AI opinions, regenerated locally (gamma=0, eps=0.4, seed 0) -------
innate, adj = ml_action_setup()
x = innate.clone()
g = torch.Generator().manual_seed(0)
for _ in range(30):
    gp.ab_sweep(x, adj, EPS, 0.0, gen=g)
noai_op = np.clip(x.numpy(), 0, 1)
noai_dr = float(x.std() / (innate.std() + 1e-9))
inn_np = innate.numpy()

frz_op, frz_dr = run_ops("frz_qwen_e040_s0")
nf_op, nf_dr = run_ops("mlanf_e040_a040_rep_b0_s0")
cl_op, cl_dr = run_ops("mla2dv2_e040_a040_b0_s0")

# rows = train off/on, cols = serve off/on   (matches the hinge heatmap)
CELLS = [[("no-AI", noai_op, noai_dr), ("frozen", frz_op, frz_dr)],
         [("no feedback", nf_op, nf_dr), ("closed", cl_op, cl_dr)]]

with open(f"{OUT}/fig_hinge_dists.json", "w") as fh:
    json.dump({"cell": "Qwen ML-Action e040 rep b0 s0",
               "bins": np.linspace(0, 1, 31).tolist(), "density": True,
               "innate": np.round(inn_np, 4).tolist(),
               "innate_mean": float(inn_np.mean()),
               "dr": {c[0]: c[2] for row in CELLS for c in row},
               "op_r30": {c[0]: np.round(c[1], 4).tolist() for row in CELLS for c in row},
               "note": "op_r30 = per-agent opinion at round 30 (the histogrammed "
                       "data); no-AI regenerated via ab_sweep gamma0 eps0.4 seed0"}, fh, indent=2)

# ---- figure: 2x2 of r30 opinion histograms, innate outlined --------------
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "font.size": 9})
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey=True,
                         constrained_layout=True)
bins = np.linspace(0, 1, 31)
# shared corner palette (matches plot_hinge_traj / _regime):
# no-AI gray · frozen blue · no-feedback green · closed red
COLOR = {"no-AI": "#95a5a6", "no feedback": "#27ae60",
         "frozen": "#2980b9", "closed": "#c0392b"}
for i in range(2):
    for j in range(2):
        ax = axes[i, j]
        name, op, dr = CELLS[i][j]
        ax.hist(inn_np, bins=bins, color="#000000", histtype="step", lw=1.0,
                alpha=0.35, label="innate", density=True)
        ax.hist(op, bins=bins, color=COLOR[name], alpha=0.75, density=True,
                label="round 30")
        ax.axvline(inn_np.mean(), color="#000000", lw=0.7, ls=":", alpha=0.5)
        serve = "serve ON" if j else "serve off"
        train = "train ON" if i else "train off"
        ax.text(0.03, 0.95, f"{name}\n({serve}, {train})\ndr = {dr:.2f}",
                transform=ax.transAxes, fontsize=9, va="top",
                color=COLOR[name], fontweight="bold", linespacing=1.4)
        if i == 1:
            ax.set_xlabel("opinion (Action taste)", fontsize=9)
        if j == 0:
            ax.set_ylabel("density", fontsize=9)
        ax.set_xlim(0, 1)
axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.text(0.5, -0.02, "Qwen ML-Action, $\\epsilon$=0.4, replace, $\\beta$=0, "
         "seed 0.  Dotted line = innate mean; black outline = innate distribution.",
         ha="center", va="top", fontsize=7.5, color="#555555")
fig.savefig(f"{OUT}/fig_hinge_dists.png", dpi=140, bbox_inches="tight")
print(f"saved {OUT}/fig_hinge_dists.png and fig_hinge_dists.json")
print("dr:", {c[0]: round(c[2], 3) for row in CELLS for c in row})
