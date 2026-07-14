"""Q6 diagnostic figure: frozen-reference perplexity on each round's
training corpus, PPL_ref(D_t), per beta at both mixing cells. Falling
curve = the population's data becomes higher-probability under the
never-trained reference (feedback alignment); flat = the corpus stays as
hard as the innate targets (gradient domination carries the rescue).
"""
import json, os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs/qwen"
CELLS = [("wide mixing $\\epsilon$=0.40 (displacement form)",
          [("mla2dv2_e040_a040_b0_s0", "0"), ("mla2bv2_e040_a040_b0p5_s0", "0.5"),
           ("mla2bv2_e040_a040_b1_s0", "1")]),
         ("slow mixing $\\epsilon$=0.10 (mode-capture form)",
          [("mlat_e010_a040_rep_b0_s0", "0"), ("mlat_e010_a040_rep_b0p5_s0", "0.5"),
           ("mlat_e010_a040_rep_b1_s0", "1")])]
RAMP = plt.cm.GnBu([0.45, 0.7, 0.95])

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True, constrained_layout=True)
for ax, (title, runs) in zip(axes, CELLS):
    for (tag, blab), c in zip(runs, RAMP):
        d = json.load(open(f"{RUNS}/{tag}/ref_ppl.json"))
        ys = [r["p50"] for r in d["rounds"]]
        ax.plot(np.arange(1, len(ys) + 1), ys, color=c, lw=2.2, label=f"$\\beta$={blab}")
    ax.axhline(d["innate"]["p50"], color="#888888", lw=1.2, ls="--")
    ax.text(29.5, d["innate"]["p50"] * 1.05, "innate targets", fontsize=8,
            color="#777777", ha="right")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("round", fontsize=11)
    ax.set_yscale("log")
axes[0].set_ylabel("frozen-reference perplexity on the round's corpus", fontsize=11)
axes[0].legend(frameon=False, fontsize=10)
fig.suptitle("Does the world become like the model? Only when mixing lets it: the anchored corpus\n"
             "stays hard at wide $\\epsilon$ (rescue = regularization) and becomes trivial at slow "
             "$\\epsilon$ (ppl 13.9 to 1.4)", fontsize=12)
fig.savefig(f"{OUT}/ref_ppl_diagnostic.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/ref_ppl_diagnostic.png")
