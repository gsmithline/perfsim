"""Tree-3 headline lines chart (the R2-dial grammar, applied to manufacture):
x = round, y = War-opinion gender gap (M-F), one line per arm.

Left panel: population gap D_x(t) from op_raw.
Right panel: the model's expressed gap D_m(t) from pred_raw.
Gray band: no-AI Deffuant baseline (10 seeds, mean +/- 2sd), computed
locally. Dotted line: preregistered success threshold +0.05 (QUESTIONS.md
Q7). Arms render as their runs appear in runs/pokec_gated_lm/ -- missing
arms are listed in the footnote, so the figure is valid at any stage.
"""
import glob
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch

ROOT = Path(".")
RUNS = ROOT / "runs/pokec_gated_lm"
OUT = "experiments/llm/figs/qwen"
BLUE, ORANGE, GREEN, GRAY = "#0E76B4", "#C96A14", "#2E7D4F", "#9A9A92"
THRESH = 0.05

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 9, "ytick.labelsize": 9})


def gap_series(arr, male):
    return np.array([np.nanmean(a[male]) - np.nanmean(a[~male]) for a in arr])


# ---- no-AI baseline band (local Deffuant, War world) -------------------------
spec = importlib.util.spec_from_file_location(
    "gp", ROOT / "experiments/scripts/cluster_pipelines/_gated_pop.py")
gp = importlib.util.module_from_spec(spec); spec.loader.exec_module(gp)
ml = ROOT / "experiments/data/movielens/ml-100k"
core = ["Drama", "Romance", "Comedy", "Action", "Thriller", "War", "Crime",
        "Sci-Fi", "Adventure", "Mystery", "Children's"]
gen = pd.read_csv(ml / "u.genre", sep="|", names=["name", "gid"], encoding="latin-1")
genres = list(gen.sort_values("gid")["name"])
items = pd.read_csv(ml / "u.item", sep="|", encoding="latin-1", header=None)
gmat = pd.DataFrame(items.iloc[:, 5:5 + len(genres)].values, index=items[0].values, columns=genres)
users = pd.read_csv(ml / "u.user", sep="|", names=["uid", "age", "gender", "occ", "zip"]).set_index("uid")
rat = pd.read_csv(ml / "u.data", sep="\t", names=["uid", "iid", "r", "t"]).merge(
    gmat, left_on="iid", right_index=True)
P = pd.DataFrame({g: rat[rat[g] == 1].groupby("uid")["r"].mean() for g in core}).dropna()
feats = [g for g in core if g != "War"]
Z = P[feats].values - P[feats].values.mean(0)
nrm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9); sim = nrm @ nrm.T
np.fill_diagonal(sim, -np.inf)
nbrs = np.argsort(-sim, axis=1)[:, :10]
G = nx.Graph(); G.add_nodes_from(range(len(P)))
for i, r in enumerate(nbrs):
    for j in r:
        G.add_edge(i, int(j))
lcc = sorted(max(nx.connected_components(G), key=len))
h = nx.relabel_nodes(G.subgraph(lcc).copy(), {n: k for k, n in enumerate(lcc)})
Pl = P.iloc[lcc]; demo = users.reindex(Pl.index)
innate = torch.tensor((Pl["War"].values - 1) / 4, dtype=torch.float32)
male0 = (demo.gender == "M").values
adjb = (torch.tensor(nx.to_numpy_array(h, nodelist=range(len(Pl))), dtype=torch.float32) > 0).float()
noai = []
for seed in range(10):
    x = innate.clone(); g = torch.Generator().manual_seed(seed)
    traj = []
    for _ in range(30):
        gp.ab_sweep(x, adjb, 0.10, 0.0, gen=g)
        traj.append(x.numpy().copy())
    noai.append(gap_series(np.array(traj), male0))
noai = np.array(noai)

# ---- arms ---------------------------------------------------------------------
ARMS = [("B  real feature, $\\beta$=1", "mlaw_e010_a040_rep_b1_s*", BLUE, "-", "o"),
        ("B  real feature, $\\beta$=0.5", "mlaw_e010_a040_rep_b0p5_s*", BLUE, "--", "o"),
        ("C  gender removed", "mlawC_e010_a040_rep_b1_s*", GREEN, "-", "s"),
        ("D  gender permuted (true labels)", "mlawD_e010_a040_rep_b1_s*", ORANGE, "-", "^"),
        ("D  scored by permuted labels", None, ORANGE, "--", "^")]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True, constrained_layout=True)
pending = []
rounds = np.arange(1, 31)
for ax, key, title in [(axes[0], "op_raw", "population: gender gap in War opinion"),
                       (axes[1], "pred_raw", "model: expressed gender gap in predictions")]:
    ax.fill_between(rounds, noai.mean(0) - 2 * noai.std(0), noai.mean(0) + 2 * noai.std(0),
                    color=GRAY, alpha=0.35, lw=0, label="no-AI band ($\\pm2$sd)" if key == "op_raw" else None)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.axhline(THRESH, color="#333333", lw=1.1, ls=":")
    if key == "op_raw":
        ax.text(29.7, THRESH - 0.0025, "preregistered success (+0.05)", fontsize=8,
                ha="right", va="top", color="#333333")
    for label, pat, c, ls, mk in ARMS:
        if pat is None:   # permuted-label scoring of arm D
            dirs = sorted(glob.glob(str(RUNS / "mlawD_e010_a040_rep_b1_s*")))
            series = []
            for d in dirs:
                pj = Path(d) / "permute_cols.json"
                if not (Path(d) / "trajectory.pt").exists() or not pj.exists():
                    continue
                import json
                perm = np.array(json.load(open(pj))["perm"])
                t = torch.load(Path(d) / "trajectory.pt", map_location="cpu", weights_only=False)
                g_perm = (pd.DataFrame(t["profiles"])["gender"].values == "M")  # already permuted in profiles
                series.append(gap_series(np.asarray(t[key], float), g_perm))
            # profiles in D runs carry the permuted gender; the TRUE-label arm
            # below rebuilds truth from the unpermuted user table order
        else:
            dirs = sorted(glob.glob(str(RUNS / pat)))
            series = []
            for d in dirs:
                if not (Path(d) / "trajectory.pt").exists():
                    continue
                t = torch.load(Path(d) / "trajectory.pt", map_location="cpu", weights_only=False)
                if label.startswith("D"):
                    m = male0          # TRUE labels regardless of what prompts said
                else:
                    m = (pd.DataFrame(t["profiles"])["gender"].values == "M") if \
                        "gender" in pd.DataFrame(t["profiles"]).columns else male0
                series.append(gap_series(np.asarray(t[key], float), m))
        if not series:
            if key == "op_raw":
                pending.append(label.strip())
            continue
        S = np.vstack([s[:30] for s in series])
        for s in S:
            ax.plot(rounds[:len(s)], s, color=c, ls=ls, lw=0.8, alpha=0.35)
        ax.plot(rounds[:S.shape[1]], S.mean(0), color=c, ls=ls, lw=2.2, marker=mk,
                markevery=5, ms=4, label=label if key == "op_raw" else None)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("round", fontsize=10)
axes[0].set_ylabel("mean(M) $-$ mean(F)", fontsize=10)
axes[0].legend(frameon=False, fontsize=8, loc="center left")
note = ("thin lines = individual seeds; bold = arm mean. " +
        (f"PENDING ARMS (no runs on disk yet): {'; '.join(pending)}" if pending else
         "all arms present"))
fig.suptitle("Tree 3: does the model's false War $\\times$ gender belief become a real gap?\n"
             "truth starts at $-$0.004 (no real gap); model believes +0.13", fontsize=11.5)
fig.text(0.01, -0.03, note, fontsize=8, color="#666666")
fig.savefig(f"{OUT}/tree3_gap_lines.png", dpi=140, bbox_inches="tight")
print(f"saved {OUT}/tree3_gap_lines.png | pending: {pending or 'none'}")
