"""Cross-model Tree-2 figures: per-model stereotype map + age curves
(same layout as the Qwen originals in plot_demo_probe.py), for
{llama, olmo, gemma}. Reads runs/pokec_gated_lm/demo_probe_<m>/ +
demo_probe_human_gaps.json + demo_probe_crossmodel.json (for the
per-model rule winner to star)."""
import glob
import json
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "experiments/llm/figs/qwen"
ML = "experiments/data/movielens/ml-100k"
AGES = [20, 30, 40, 50, 60]
BLUE, ORANGE = "#0E76B4", "#C96A14"
MODELS = {"llama": ("demo_probe_llama", "Llama-3.1-8B (zero-shot: constant 0.50 -- degenerate, not unbiased)"),
          "olmo": ("demo_probe_olmo", "OLMo-2-7B"),
          "gemma": ("demo_probe_gemma", "Gemma-3-12b-it")}

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 9, "ytick.labelsize": 9})

cross = json.load(open("experiments/llm/demo_probe_crossmodel.json"))
core = ["Drama", "Romance", "Comedy", "Action", "Thriller", "War", "Crime",
        "Sci-Fi", "Adventure", "Mystery", "Children's"]
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

MCOLS = ["M$-$F\n@20", "M$-$F\n@40", "M$-$F\n@60", "60$-$20\n@F", "60$-$20\n@M", "inter-\naction"]
HCOLS = ["M$-$F", "old$-$\nyoung", "inter-\naction"]
VLIM = 0.15

for mname, (d, label) in MODELS.items():
    probe = {}
    for f in sorted(glob.glob(f"runs/pokec_gated_lm/{d}/probe_*.json")):
        j = json.load(open(f))
        probe[j["target"]] = {"V": {k: np.array([np.nan if v is None else v for v in vv], float)
                                    for k, vv in j["variants"].items()},
                              "innate": np.array(j["innate"])}
    rows = {}
    for tgt, dd in probe.items():
        V = dd["V"]
        g_at = {a: np.nanmean(V[f"a{a}_M"] - V[f"a{a}_F"]) for a in AGES}
        y = (P[tgt].values - 1) / 4
        rows[tgt] = dict(
            model=[g_at[20], g_at[40], g_at[60],
                   np.nanmean(V["a60_F"] - V["a20_F"]), np.nanmean(V["a60_M"] - V["a20_M"]),
                   g_at[60] - g_at[20]],
            human=[y[male].mean() - y[~male].mean(), y[old].mean() - y[~old].mean(),
                   (y[male & old].mean() - y[~male & old].mean())
                   - (y[male & ~old].mean() - y[~male & ~old].mean())],
            pooled_g=np.nanmean([g_at[a] for a in AGES]))
    order = sorted(rows, key=lambda t: -abs(rows[t]["pooled_g"]))
    winners = [t for t, r in cross[mname].items() if r["passes"]]
    star = max(winners, key=lambda t: cross[mname][t]["S"]) if winners else None

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [6, 3]})
    for ax, key, cols, title in [(axes[0], "model", MCOLS, f"what the model believes\n({label})"),
                                 (axes[1], "human", HCOLS, "what the data says\n(MovieLens groups)")]:
        M = np.array([rows[t][key] for t in order])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-VLIM, vmax=VLIM, aspect="auto")
        ax.set_xticks(range(len(cols)), cols, fontsize=8)
        labels = [("* " if t == star else "") + t for t in order]
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
    stitle = (f"* = this model's own rule-winner: {star} x gender" if star
              else "no cell passes the binding rule for this model")
    fig.suptitle(f"Stereotype map, {label.split(' (')[0]}: {stitle}\n"
                 "(cells beyond $\\pm$0.15 clipped, numbers exact)", fontsize=11)
    fig.savefig(f"{OUT}/demo_probe_map_{mname}.png", dpi=140)
    plt.close(fig)

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
    fig.legend(loc="lower right", bbox_to_anchor=(0.97, 0.06), frameon=False, fontsize=10)
    fig.supxlabel("age stated in the prompt", fontsize=11)
    fig.supylabel("mean predicted opinion", fontsize=11)
    fig.suptitle(f"Age-belief curves, {label}", fontsize=11)
    fig.savefig(f"{OUT}/demo_probe_age_curves_{mname}.png", dpi=140)
    plt.close(fig)
    print(f"saved {mname}: map + age curves")
