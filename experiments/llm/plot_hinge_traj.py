"""Act I/II hinge, dynamics companion: WHEN the four corners diverge.

Per-round trajectories over 30 rounds for the same cell (Qwen ML-Action,
eps=0.4, replace, beta=0, seed 0):
  left  panel: population diversity dr(t) = op_std(t)/innate_std  (all 4 corners)
  right panel: model perplexity ppl(t) = median per-agent ppl     (3 trained/served
               corners; no-AI has no model, frozen weights never move)

no-AI dr(t) regenerated locally (Deffuant-AB ab_sweep, gamma0 eps0.4 seed0).

READING:
  * dr(t): no-AI and no-feedback decay together (self-collapse); frozen RISES
    (sorting onto two modes) then plateaus; closed decays like no-AI early but
    the trained model keeps pulling it down.
  * ppl(t): frozen flat (never trains); no-feedback and closed ramp UP as their
    training corpora homogenize round by round -- rot is progressive, not a
    step change.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/llm/plot_hinge_traj.py
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
EPS = 0.4
CORE = ["Drama", "Romance", "Comedy", "Action", "Thriller", "War", "Crime",
        "Sci-Fi", "Adventure", "Mystery", "Children's"]

spec = importlib.util.spec_from_file_location(
    "gp", "experiments/scripts/cluster_pipelines/_gated_pop.py")
gp = importlib.util.module_from_spec(spec); spec.loader.exec_module(gp)


def ml_action_setup():
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


def traj(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    ppl = np.asarray(d.get("ppl_raw", []), np.float32)
    dr_t = op.std(1) / (inn.std() + 1e-9)
    ppl_t = np.median(ppl, axis=1) if ppl.size else None
    return dr_t, ppl_t


# no-AI dr(t)
innate, adj = ml_action_setup()
x = innate.clone(); g = torch.Generator().manual_seed(0)
noai = []
for _ in range(30):
    gp.ab_sweep(x, adj, EPS, 0.0, gen=g)
    noai.append(float(x.std() / (innate.std() + 1e-9)))
noai = np.array(noai)

frz_dr, frz_ppl = traj("frz_qwen_e040_s0")
nf_dr, nf_ppl = traj("mlanf_e040_a040_rep_b0_s0")
cl_dr, cl_ppl = traj("mla2dv2_e040_a040_b0_s0")

C = {"no-AI": "#95a5a6", "frozen": "#2980b9", "no feedback": "#27ae60", "closed": "#c0392b"}

with open(f"{OUT}/fig_hinge_traj.json", "w") as fh:
    json.dump({"cell": "Qwen ML-Action e040 rep b0 s0",
               "dr": {"no-AI": noai.tolist(), "frozen": frz_dr.tolist(),
                      "no feedback": nf_dr.tolist(), "closed": cl_dr.tolist()},
               "ppl": {"frozen": frz_ppl.tolist(), "no feedback": nf_ppl.tolist(),
                       "closed": cl_ppl.tolist()}}, fh, indent=2)

# ---- figure: dr(t) | ppl(t) ----------------------------------------------
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "font.size": 9})
fig, (axd, axp) = plt.subplots(1, 2, figsize=(7.6, 3.4), constrained_layout=True)
R = np.arange(1, 31)

for name, y in [("no-AI", noai), ("frozen", frz_dr), ("no feedback", nf_dr), ("closed", cl_dr)]:
    axd.plot(R, y, "-", color=C[name], lw=2, label=name)
axd.axhline(1.0, color="#999", lw=0.8, ls="--")
axd.set_xlabel("round", fontsize=9)
axd.set_ylabel("population diversity  dr(t)", fontsize=9)
axd.set_title("population: serve moves it; frozen sorts UP", fontsize=9.5)
axd.legend(frameon=False, fontsize=7.5, loc="upper right")
axd.set_xlim(1, 30)

for name, y in [("frozen", frz_ppl), ("no feedback", nf_ppl), ("closed", cl_ppl)]:
    if y is not None:
        axp.plot(R, y, "-", color=C[name], lw=2, label=name)
axp.set_yscale("log")
axp.set_xlabel("round", fontsize=9)
axp.set_ylabel("model perplexity  ppl(t) (log)", fontsize=9)
axp.set_title("model: train rots it, progressively", fontsize=9.5)
axp.legend(frameon=False, fontsize=7.5, loc="upper left")
axp.set_xlim(1, 30)

fig.suptitle("When the corners diverge: the population departs only under SERVE "
             "(frozen sorts up early),\nthe model rots only under TRAIN and does "
             "so gradually as its corpus homogenizes", fontsize=10)
fig.text(0.5, -0.03, "Qwen ML-Action, $\\epsilon$=0.4, replace, $\\beta$=0, seed 0.  "
         "no-AI has no model; frozen weights never move.",
         ha="center", va="top", fontsize=7.5, color="#555555")
fig.savefig(f"{OUT}/fig_hinge_traj.png", dpi=140, bbox_inches="tight")
print(f"saved {OUT}/fig_hinge_traj.png and fig_hinge_traj.json")
print("dr end:", {k: round(float(v[-1]), 3) for k, v in
                  [("no-AI", noai), ("frozen", frz_dr), ("nf", nf_dr), ("closed", cl_dr)]})
print("ppl end:", {k: round(float(v[-1]), 1) for k, v in
                   [("frozen", frz_ppl), ("nf", nf_ppl), ("closed", cl_ppl)] if v is not None})
