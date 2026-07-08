"""Model-health trajectories: per-round log10 perplexity as line charts,
one figure per (model arm, statistic). Panels: rows = beta, cols = cell;
lines = training regime (color = data recipe, dashed = fresh weights).
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
MODELS = {
    "qwen": "mlat",
    "qwen-reset": "mlatR",
    "qwen-eq": "mlatE",
    "llama": "mlatL",
    "gemma": "mlatG",
    "mistral": "mlatM",
}
CELLS = ["e040_a010", "e040_a020", "e040_a040", "e020_a040"]
CELL_LABEL = {"e040_a010": "capture", "e040_a020": "mid",
              "e040_a040": "trap", "e020_a040": "smear"}
REGIMES = ["rep", "acc", "pri", "frep", "facc"]
STYLE = {"rep": ("#1b9e77", "-"), "acc": ("#d95f02", "-"), "pri": ("#7570b3", "-"),
         "frep": ("#1b9e77", "--"), "facc": ("#d95f02", "--")}
BETAS = [("b0", "0"), ("b0p5", "0.5"), ("b1", "1")]

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


def find_tag(model, cell, regime, beta):
    tag = f"{MODELS[model]}_{cell}_{regime}_{beta}_s0"
    if os.path.exists(f"{RUNS}/{tag}/trajectory.pt"):
        return tag
    if model == "qwen":
        leg = LEGACY.get((cell, regime, beta))
        if leg and os.path.exists(f"{RUNS}/{leg}/trajectory.pt"):
            return leg
    return None


def load_lp(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    return np.log10(np.clip(np.asarray(d["ppl_raw"], np.float32), 1.0, None))


plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.0, "xtick.labelsize": 9, "ytick.labelsize": 9})

for model in MODELS:
    if not any(find_tag(model, c, r, b) for c in CELLS for r in REGIMES
               for b, _ in BETAS):
        continue
    os.makedirs(f"{FIGS}/{model}", exist_ok=True)
    for stat, fn in [("median", lambda a: np.median(a, axis=1)),
                     ("mean", lambda a: a.mean(axis=1))]:
        fig, axes = plt.subplots(len(BETAS), len(CELLS), figsize=(13.5, 8.2),
                                 sharex=True, sharey=True, constrained_layout=True)
        for ri, (bc, blab) in enumerate(BETAS):
            for ci, cell in enumerate(CELLS):
                ax = axes[ri, ci]
                for regime in REGIMES:
                    tag = find_tag(model, cell, regime, bc)
                    if tag is None:
                        continue
                    y = fn(load_lp(tag))
                    color, ls = STYLE[regime]
                    ax.plot(np.arange(1, len(y) + 1), y, color=color, ls=ls,
                            lw=1.8, label=regime)
                if ri == 0:
                    ax.set_title(CELL_LABEL[cell], fontsize=12)
                if ci == 0:
                    ax.set_ylabel(f"$\\beta$={blab}\n{stat} $\\log_{{10}}$ ppl",
                                  fontsize=11)
                if ri == len(BETAS) - 1:
                    ax.set_xlabel("round", fontsize=11)
        axes[0, 0].legend(fontsize=9, frameon=False, ncols=2)
        fig.suptitle(f"ML-Action atlas slab, {model} (seed 0): per-agent "
                     f"{stat} log10 perplexity per round; dashed = fresh weights",
                     fontsize=12)
        out = f"{FIGS}/{model}/ppl_lines_{stat}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"saved {out}")
