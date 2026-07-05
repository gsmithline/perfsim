"""Model-collapse style perplexity histograms for the training-regime factorials:
per arm, the per-agent distribution at snapshot rounds overlaid (light = early,
dark = late), log-scale x. Replace should show the rightward dissolution,
pristine should stay pinned. ML uses the post-fix v2 runs; Pokec and Yelp arms
exist pre-fix only (compare arms within a figure, not levels across datasets).
"""
import os
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
FIGS = "experiments/llm/figs"
SNAPS = [0, 4, 9, 19, 29]

DATASETS = [
    ("mlv2", "e040_a040", "mla2dv2", "mla2drv2", "mla2dfv2", 3.0,
     "MovieLens-Action e040_a040 (post-fix v2)"),
    ("pokec", "e020_a040", "e2d", "e2dr", "e2df", 4.0,
     "Pokec e020_a040 (pre-fix, quenched dynamics)"),
    ("yelp", "e040_a040", "ylp2d", "ylp2dr", "ylp2df", 4.0,
     "Yelp-Acme e040_a040 (pre-fix, quenched dynamics)"),
]


def panel(ax, tag, bins, label, show_legend=False):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    ppl = np.clip(np.asarray(d["ppl_raw"], np.float32), 1.0, None)
    colors = plt.cm.viridis(np.linspace(0.92, 0.05, len(SNAPS)))
    for t, c in zip(SNAPS, colors):
        ax.hist(ppl[t], bins=bins, density=True, histtype="stepfilled",
                alpha=0.15, color=c)
        ax.hist(ppl[t], bins=bins, density=True, histtype="step",
                color=c, lw=1.8, label=f"round {t + 1}" if show_legend else None)
    ax.set_xscale("log")
    ax.set_title(f"{label}   (final med={np.median(ppl[-5:]):.1f})", fontsize=9)
    if show_legend:
        ax.legend(fontsize=8, frameon=False)


for stem, cell, g, gr, gf, xmax, label in DATASETS:
    # replace tags: grid runs have no arm suffix, fresh replace does
    rep = f"{g}_{cell}_{{b}}_s0"
    arms = [("continual", "replace", rep),
            ("continual", "accumulate", f"{gr}_{cell}_{{b}}_acc_s0"),
            ("continual", "pristine", f"{gr}_{cell}_{{b}}_pri_s0"),
            ("fresh", "replace", f"{gf}_{cell}_{{b}}_rep_s0"),
            ("fresh", "accumulate", f"{gf}_{cell}_{{b}}_acc_s0"),
            ("fresh", "pristine", f"{gf}_{cell}_{{b}}_pri_s0")]
    bins = np.logspace(0, xmax, 45 if xmax == 3.0 else 55)
    for b, bv in [("b0", 0), ("b3", 3)]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True,
                                 constrained_layout=True)
        for k, (w, dta, pat) in enumerate(arms):
            ax = axes[k // 3, k % 3]
            panel(ax, pat.format(b=b), bins, f"{w} / {dta}", show_legend=(k == 0))
            if k // 3 == 1: ax.set_xlabel("per-agent perplexity")
            if k % 3 == 0: ax.set_ylabel("density")
        fig.suptitle(f"{label}, KL beta={bv}: per-agent perplexity histograms "
                     "across rounds (light = early, dark = late).", fontsize=11)
        out = f"{FIGS}/{'mlv2_ppl_hist' if stem == 'mlv2' else 'ppl_hist_' + stem}_{b}.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"saved {out}")
