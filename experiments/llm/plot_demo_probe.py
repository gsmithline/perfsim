"""Tree-2 probe figures (Q7): the stereotype map (model beliefs vs human
data, one shared diverging scale), the eleven age-belief curves, and the
nf mediation ladder. Reads runs/pokec_gated_lm/demo_probe/probe_*.json +
experiments/llm/demo_probe_human_gaps.json; ladder numbers from the scored
mlanf_*/mlatR_/mla2*v2/mlatE_ runs (QUESTIONS.md Q7 outcome, 2026-07-15).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm/demo_probe"
OUT = "experiments/llm/figs/qwen"
ML = "experiments/data/movielens/ml-100k"
AGES = [20, 30, 40, 50, 60]
BLUE, ORANGE = "#0E76B4", "#C96A14"   # M, F (validated pair)

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 9, "ytick.labelsize": 9})

human = json.load(open("experiments/llm/demo_probe_human_gaps.json"))

# ---- load probe, compute effects -------------------------------------------
probe = {}
for f in sorted(glob.glob(f"{RUNS}/probe_*.json")):
    d = json.load(open(f))
    V = {k: np.array([np.nan if v is None else v for v in vv], float)
         for k, vv in d["variants"].items()}
    probe[d["target"]] = {"V": V, "innate": np.array(d["innate"])}

# human interaction + age direction from raw data (old-young to match m(60)-m(20))
core = list(probe.keys())
gen = pd.read_csv(f"{ML}/u.genre", sep="|", names=["name", "gid"], encoding="latin-1")
genres = list(gen.sort_values("gid")["name"])
items = pd.read_csv(f"{ML}/u.item", sep="|", encoding="latin-1", header=None)
gmat = pd.DataFrame(items.iloc[:, 5:5 + len(genres)].values, index=items[0].values, columns=genres)
users = pd.read_csv(f"{ML}/u.user", sep="|", names=["uid", "age", "gender", "occ", "zip"]).set_index("uid")
rat = pd.read_csv(f"{ML}/u.data", sep="\t", names=["uid", "iid", "r", "t"]).merge(
    gmat, left_on="iid", right_index=True)
P = pd.DataFrame({g: rat[rat[g] == 1].groupby("uid")["r"].mean() for g in core}).dropna()
demo = users.reindex(P.index)
male = (demo.gender == "M").values
old = (demo.age > 31).values

rows = {}
for tgt, d in probe.items():
    V = d["V"]
    g_at = {a: np.nanmean(V[f"a{a}_M"] - V[f"a{a}_F"]) for a in AGES}
    age_F = np.nanmean(V["a60_F"] - V["a20_F"])
    age_M = np.nanmean(V["a60_M"] - V["a20_M"])
    inter = g_at[60] - g_at[20]
    y = (P[tgt].values - 1) / 4
    h_g = y[male].mean() - y[~male].mean()
    h_a = y[old].mean() - y[~old].mean()
    h_i = (y[male & old].mean() - y[~male & old].mean()) - \
          (y[male & ~old].mean() - y[~male & ~old].mean())
    rows[tgt] = dict(model=[g_at[20], g_at[40], g_at[60], age_F, age_M, inter],
                     human=[h_g, h_a, h_i],
                     pooled_g=np.nanmean([g_at[a] for a in AGES]))
order = sorted(rows, key=lambda t: -abs(rows[t]["pooled_g"]))

# ---- fig 1: stereotype map ---------------------------------------------------
MCOLS = ["M$-$F\n@20", "M$-$F\n@40", "M$-$F\n@60", "60$-$20\n@F", "60$-$20\n@M", "inter-\naction"]
HCOLS = ["M$-$F", "old$-$\nyoung", "inter-\naction"]
VLIM = 0.15
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), constrained_layout=True,
                         gridspec_kw={"width_ratios": [6, 3]})
for ax, key, cols, title in [(axes[0], "model", MCOLS, "what the model believes\n(frozen Qwen, tastes held fixed)"),
                             (axes[1], "human", HCOLS, "what the data says\n(MovieLens groups)")]:
    M = np.array([rows[t][key] for t in order])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-VLIM, vmax=VLIM, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    labels = [("* " if t == "War" else "") + t for t in order]
    ax.set_yticks(range(len(order)), labels if key == "model" else [""] * len(order))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.09 else "#333333")
    ax.set_title(title, fontsize=11)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
fig.colorbar(im, ax=axes, shrink=0.75, label="prediction / opinion gap")
fig.suptitle("The stereotype map: large demographic beliefs, null demographic data\n"
             "(* = Tree-3 selection by pre-committed rule: War x gender; "
             "cells beyond $\\pm$0.15 are clipped, numbers are exact)", fontsize=11)
fig.savefig(f"{OUT}/demo_probe_map.png", dpi=140)
plt.close(fig)

# ---- fig 2: age curves --------------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(12, 7.6), sharex=True, sharey=True,
                         constrained_layout=True)
for k, tgt in enumerate(order):
    ax = axes.flat[k]
    V = probe[tgt]["V"]
    for g, c, mk in [("M", BLUE, "o"), ("F", ORANGE, "s")]:
        ax.plot(AGES, [np.nanmean(V[f"a{a}_{g}"]) for a in AGES],
                color=c, marker=mk, ms=4, lw=1.8, label=g if k == 0 else None)
    ax.axhline(probe[tgt]["innate"].mean(), color="#888888", lw=1.0, ls="--")
    ax.set_title(tgt, fontsize=10)
    ax.tick_params(labelsize=8)
axes.flat[11].axis("off")
axes.flat[11].text(0.1, 0.65, "solid: model prediction\nby stated age/gender\n"
                   "(tastes held fixed)\n\ndashed: population's\ntrue mean opinion",
                   fontsize=9, va="top")
fig.legend(loc="lower right", bbox_to_anchor=(0.97, 0.06), frameon=False, fontsize=10)
fig.supxlabel("age stated in the prompt", fontsize=11)
fig.supylabel("mean predicted opinion", fontsize=11)
fig.suptitle("The model's age beliefs are large and cliff-shaped (drop at 50-60), "
             "not monotone -- every age candidate failed rule (c)", fontsize=11)
fig.savefig(f"{OUT}/demo_probe_age_curves.png", dpi=140)
plt.close(fig)

# ---- fig 3: nf mediation ladder ----------------------------------------------
arms = [("pristine (closed, 50% innate data held in)", 1.6),
        ("reset (closed, s=0.3, static targets)", 2.0),
        ("NO-FEEDBACK slow mix (s=0, drifting)", 19.7),
        ("NO-FEEDBACK wide mix (s=0, drifting)", 26.7),
        ("replace (closed, s$\\to$1)", 55.5),
        ("equilibrated (closed, s=1)", 82.0)]
fig, ax = plt.subplots(figsize=(9.2, 3.8), constrained_layout=True)
ys = np.arange(len(arms))[::-1]
vals = [v for _, v in arms]
ax.barh(ys, vals, height=0.62, color="#2A7FA8")
for y, (lab, v) in zip(ys, arms):
    ax.text(v * 1.06, y, f"{v:g}", va="center", fontsize=9)
ax.set_yticks(ys, [a for a, _ in arms], fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("final median per-agent perplexity (log)", fontsize=10)
ax.set_title("Model damage tracks corpus diversity + stationarity,\n"
             "not contamination share: zero model-made data (no-feedback)\n"
             "still rots the model 13x; closing the loop doubles it",
             fontsize=10, loc="left")
fig.savefig(f"{OUT}/nf_ladder.png", dpi=140)
plt.close(fig)
print(f"saved {OUT}/demo_probe_map.png, demo_probe_age_curves.png, nf_ladder.png")
