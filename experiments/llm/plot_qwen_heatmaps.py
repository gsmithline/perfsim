"""Qwen ML-Action campaign as annotated heatmaps (single seed, values shown).

Fig 1 world plane: eps x eps_AI closed-loop dr (b0, b3) + platform-added
effect vs a local W=0 Deffuant baseline (same graph, seed, ab_gen protocol).
Fig 2 knob plane: gate dial and feature dial as beta x level strips,
dr beside log10 perplexity.
Fig 3 atlas slab: regime x cell dr at beta {0, 0.5, 1}.
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import torch

sys.path.insert(0, "experiments/scripts/cluster_pipelines")
import _gated_pop as gp

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs/pokec_gated_lm")
FIGS = Path("experiments/llm/figs/qwen")
TAIL = 5

EPS_LEVELS = [("e010", 0.1), ("e020", 0.2), ("e040", 0.4)]
AI_LEVELS = [("a010", 0.1), ("a020", 0.2), ("a040", 0.4)]
GATE_COLS = [("a005", 0.05), ("a010", 0.10), ("a015", 0.15), ("a020", 0.20),
             ("a030", 0.30), ("a040", 0.40), ("a070", 0.70), ("a100", 1.00)]
FEAT_KNOBS = ["p100", "p065", "p040", "p033", "p015", "p005", "nat",
              "q025", "q050", "q100"]
CELLS = ["e040_a010", "e040_a020", "e040_a040", "e020_a040"]
REGIMES = ["rep", "acc", "pri", "frep", "facc"]
BETAS = ["b0", "b0p5", "b1"]

LEGACY = {}
for c in CELLS:
    LEGACY[(c, "rep", "b0")] = f"mla2dv2_{c}_b0_s0"
for c in ["e040_a010", "e040_a040", "e020_a040"]:
    for b in ["b0p5", "b1"]:
        LEGACY[(c, "rep", b)] = f"mla2bv2_{c}_{b}_s0"
LEGACY[("e040_a040", "acc", "b0")] = "mla2drv2_e040_a040_b0_acc_s0"
LEGACY[("e040_a040", "pri", "b0")] = "mla2drv2_e040_a040_b0_pri_s0"
LEGACY[("e040_a040", "frep", "b0")] = "mla2dfv2_e040_a040_b0_rep_s0"
LEGACY[("e040_a040", "facc", "b0")] = "mla2dfv2_e040_a040_b0_acc_s0"


def stats(tag):
    path = RUNS / tag / "trajectory.pt"
    if not path.exists():
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    ppl = np.asarray(d["ppl_raw"], np.float32)
    return dict(
        dr=float(op[-TAIL:].std(1).mean() / (inn.std() + 1e-9)),
        bias=float(op[-TAIL:].mean() - inn.mean()),
        pmed=float(np.median(ppl[-TAIL:])),
        d=d)


def realized_r2(d):
    prof = pd.DataFrame(d["profiles"])
    y = np.asarray(d["innate"], float)
    cols = []
    for c in prof.columns:
        v = prof[c]
        if pd.api.types.is_numeric_dtype(v):
            cols.append(v.values.astype(float)[:, None])
        else:
            cols.append(pd.get_dummies(v).values.astype(float))
    X = np.hstack(cols); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); yh = np.zeros_like(y)
    for f in range(5):
        te = idx[f::5]; tr = np.setdiff1d(idx, te)
        w = np.linalg.solve(X[tr].T @ X[tr] + np.eye(X.shape[1]),
                            X[tr].T @ (y[tr] - y[tr].mean()))
        yh[te] = y[tr].mean() + X[te] @ w
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def ml_baseline_dr(eps, seed=0, rounds=30):
    """W=0 Deffuant on the cluster's ML graph: no platform, same ab_gen stream."""
    ml = Path("experiments/data/movielens/ml-100k")
    core = ["Drama", "Romance", "Comedy", "Action", "Thriller", "War", "Crime",
            "Sci-Fi", "Adventure", "Mystery", "Children's"]
    gen = pd.read_csv(ml / "u.genre", sep="|", names=["name", "gid"], encoding="latin-1")
    gnames = list(gen.sort_values("gid")["name"])
    items = pd.read_csv(ml / "u.item", sep="|", encoding="latin-1", header=None)
    gmat = pd.DataFrame(items.iloc[:, 5:5 + len(gnames)].values,
                        index=items[0].values, columns=gnames)
    rat = pd.read_csv(ml / "u.data", sep="\t", names=["uid", "iid", "r", "t"]).merge(
        gmat, left_on="iid", right_index=True)
    P = pd.DataFrame({g: rat[rat[g] == 1].groupby("uid")["r"].mean() for g in core}).dropna()
    feats = [g for g in core if g != "Action"]
    Zc = P[feats].values - P[feats].values.mean(0)
    norm = Zc / (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)
    nbrs = np.argsort(-sim, axis=1)[:, :10]
    g = nx.Graph(); g.add_nodes_from(range(len(P)))
    for i, row in enumerate(nbrs):
        for j in row:
            g.add_edge(i, int(j))
    lcc = sorted(max(nx.connected_components(g), key=len))
    h = nx.relabel_nodes(g.subgraph(lcc).copy(), {n: k for k, n in enumerate(lcc)})
    Pl = P.iloc[lcc]
    innate = ((Pl["Action"].values - 1.0) / 4.0).astype(np.float32)
    adj = torch.tensor(nx.to_numpy_array(h, nodelist=range(len(Pl))) > 0, dtype=torch.float32)
    out = {}
    for e in eps:
        ab_gen = torch.Generator().manual_seed(seed + 424243)
        x = torch.tensor(innate.copy())
        stds = []
        for _ in range(rounds):
            gp.ab_sweep(x, adj, e, 0.0, gen=ab_gen)
            stds.append(float(x.std()))
        out[e] = float(np.mean(stds[-TAIL:]) / (innate.std() + 1e-9))
    return out


DR_CMAP = plt.get_cmap("viridis").with_extremes(over="crimson")


def annotate(ax, M, fmt="{:.2f}", vmin=None, vmax=None, cmap="viridis"):
    lo = np.nanmin(M) if vmin is None else vmin
    hi = np.nanmax(M) if vmax is None else vmax
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    for (i, j), v in np.ndenumerate(M):
        if np.isnan(v):
            ax.text(j, i, "--", ha="center", va="center", color="0.5", fontsize=10)
            continue
        r, g, b, _ = cm((v - lo) / (hi - lo + 1e-12))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        ax.text(j, i, fmt.format(v), ha="center", va="center",
                color="black" if lum > 0.55 else "white", fontsize=10)


plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "xtick.labelsize": 11, "ytick.labelsize": 11})
FIGS.mkdir(parents=True, exist_ok=True)


# Fig 1: world plane
base = ml_baseline_dr([v for _, v in EPS_LEVELS])
print("W=0 baselines:", {k: round(v, 3) for k, v in base.items()})

world = {}
for blab, bkey in [("0", "b0"), ("3", "b3")]:
    M = np.full((len(EPS_LEVELS), len(AI_LEVELS)), np.nan)
    for i, (ec, _) in enumerate(EPS_LEVELS):
        for j, (ac, _) in enumerate(AI_LEVELS):
            s = stats(f"mla2dv2_{ec}_{ac}_{bkey}_s0")
            if s:
                M[i, j] = s["dr"]
    world[blab] = M

fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.0), constrained_layout=True)
for j, blab in enumerate(["0", "3"]):
    M = world[blab]
    im = axes[0, j].imshow(M, cmap=DR_CMAP, vmin=0, vmax=1, aspect="auto")
    annotate(axes[0, j], M, vmin=0, vmax=1, cmap=DR_CMAP)
    axes[0, j].set_title(f"closed loop, $\\beta$={blab}", fontsize=13)
    A = M - np.array([[base[e]] for _, e in EPS_LEVELS])
    im2 = axes[1, j].imshow(A, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    annotate(axes[1, j], A, fmt="{:+.2f}", vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    axes[1, j].set_title(f"platform-added effect, $\\beta$={blab}", fontsize=13)
for ax in axes.flat:
    ax.set_xticks(range(len(AI_LEVELS)), [f"{v:.1f}" for _, v in AI_LEVELS])
    ax.set_yticks(range(len(EPS_LEVELS)), [f"{v:.1f}" for _, v in EPS_LEVELS])
    ax.set_xlabel("AI gate width $\\epsilon_{AI}$", fontsize=12)
    ax.set_ylabel("peer confidence $\\epsilon$", fontsize=12)
fig.colorbar(im, ax=axes[0, :], label="final $d_r$ (red: $d_r>1$, overshoot)",
             shrink=0.85, extend="max")
fig.colorbar(im2, ax=axes[1, :], label="$d_r$ closed $-$ $d_r$ no platform", shrink=0.85)
fig.suptitle("Qwen ML-Action, replace/continual, seed 0 (baseline: same-seed W=0 Deffuant)",
             fontsize=12)
fig.savefig(FIGS / "heatmap_world_plane.png", dpi=150)
plt.close(fig)
print("saved", FIGS / "heatmap_world_plane.png")


# Fig 2: knob plane
def gate_tag(ac, beta):
    for t in (f"mlatA_e040_{ac}_rep_{beta}_s0",
              f"mlat_e040_{ac}_rep_{beta}_s0",
              LEGACY.get((f"e040_{ac}", "rep", beta), "")):
        if t and (RUNS / t / "trajectory.pt").exists():
            return t
    return None


def feat_tag(knob, beta):
    if knob == "nat":
        return LEGACY.get(("e040_a040", "rep", beta))
    return f"mlatF_{knob}_e040_a040_rep_{beta}_s0"


gate_dr = np.full((2, len(GATE_COLS)), np.nan)
gate_ppl = np.full((2, len(GATE_COLS)), np.nan)
for i, beta in enumerate(["b0", "b1"]):
    for j, (ac, _) in enumerate(GATE_COLS):
        s = stats(gate_tag(ac, beta) or "")
        if s:
            gate_dr[i, j] = s["dr"]; gate_ppl[i, j] = np.log10(s["pmed"])

feat = []
for knob in FEAT_KNOBS:
    row = {}
    for beta in ["b0", "b1"]:
        s = stats(feat_tag(knob, beta) or "")
        if s:
            row[beta] = s
    if row:
        row["r2"] = realized_r2(next(iter(row.values()))["d"])
        row["knob"] = knob
        feat.append(row)
feat.sort(key=lambda r: r["r2"])
feat_dr = np.full((2, len(feat)), np.nan)
feat_ppl = np.full((2, len(feat)), np.nan)
for j, row in enumerate(feat):
    for i, beta in enumerate(["b0", "b1"]):
        if beta in row:
            feat_dr[i, j] = row[beta]["dr"]
            feat_ppl[i, j] = np.log10(row[beta]["pmed"])

fig, axes = plt.subplots(2, 2, figsize=(13.5, 5.6), constrained_layout=True)
panels = [
    (axes[0, 0], gate_dr, [f"{v:.2f}" for _, v in GATE_COLS],
     "gate dial: final $d_r$", "viridis", 0, 1, "{:.2f}"),
    (axes[0, 1], gate_ppl, [f"{v:.2f}" for _, v in GATE_COLS],
     "gate dial: $\\log_{10}$ ppl", "magma", 0, 3.3, "{:.1f}"),
    (axes[1, 0], feat_dr, [f"{r['r2']:.2f}" for r in feat],
     "feature dial: final $d_r$", "viridis", 0, 1, "{:.2f}"),
    (axes[1, 1], feat_ppl, [f"{r['r2']:.2f}" for r in feat],
     "feature dial: $\\log_{10}$ ppl", "magma", 0, 3.3, "{:.1f}"),
]
for ax, M, xl, title, cmap, vmin, vmax, fmt in panels:
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    annotate(ax, M, fmt=fmt, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(M.shape[1]), xl)
    ax.set_yticks([0, 1], ["$\\beta$=0", "$\\beta$=1"])
    fig.colorbar(im, ax=ax, shrink=0.9)
axes[0, 0].set_xlabel("$\\epsilon_{AI}$"); axes[0, 1].set_xlabel("$\\epsilon_{AI}$")
axes[1, 0].set_xlabel("realized probe $R^2$"); axes[1, 1].set_xlabel("realized probe $R^2$")
fig.suptitle("Qwen ML-Action knob plane (trap-cell world, replace/continual, seed 0)",
             fontsize=13)
fig.savefig(FIGS / "heatmap_knob_plane.png", dpi=150)
plt.close(fig)
print("saved", FIGS / "heatmap_knob_plane.png")


# Fig 3: atlas slab
def slab_tag(cell, regime, beta):
    t = f"mlat_{cell}_{regime}_{beta}_s0"
    if (RUNS / t / "trajectory.pt").exists():
        return t
    return LEGACY.get((cell, regime, beta))


fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
for k, beta in enumerate(BETAS):
    M = np.full((len(REGIMES), len(CELLS)), np.nan)
    for i, reg in enumerate(REGIMES):
        for j, cell in enumerate(CELLS):
            s = stats(slab_tag(cell, reg, beta) or "")
            if s:
                M[i, j] = s["dr"]
    im = axes[k].imshow(M, cmap=DR_CMAP, vmin=0, vmax=1, aspect="auto")
    annotate(axes[k], M, vmin=0, vmax=1, cmap=DR_CMAP)
    axes[k].set_title(f"$\\beta$={beta.replace('b', '').replace('p', '.')}", fontsize=13)
    axes[k].set_xticks(range(len(CELLS)),
                       [c.replace("e0", "$\\epsilon$.").replace("_a0", " / $\\epsilon_{AI}$.")
                        for c in CELLS], fontsize=9)
    axes[k].set_yticks(range(len(REGIMES)), REGIMES)
fig.colorbar(im, ax=axes, label="final $d_r$ (red: $d_r>1$, overshoot)",
             shrink=0.85, extend="max")
fig.suptitle("Qwen ML-Action atlas slab: training regime $\\times$ world cell, seed 0",
             fontsize=13)
fig.savefig(FIGS / "heatmap_atlas_slab.png", dpi=150)
plt.close(fig)
print("saved", FIGS / "heatmap_atlas_slab.png")
