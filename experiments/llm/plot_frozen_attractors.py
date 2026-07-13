"""Frozen-editor attractors (Q5, seed 0): final opinion distributions for
four editors x social eps {0.10, 0.40}, innate distribution as reference.
Each deployed prior becomes the population attractor: Llama 0.50, Qwen
splits 0.25/0.65, Gemma blob ~0.70, OLMo 0.73-0.78. Dashed line = the
editor's round-0 output mean (nu).
"""
import os, numpy as np, torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs/qwen"
EDITORS = [("llama", "Llama-3.1-8B", "#2f6f9f"),
           ("qwen", "Qwen2.5-7B", "#e08214"),
           ("gemma", "Gemma-3-12b", "#7b3294"),
           ("olmo", "OLMo-2-7B", "#1b7837")]
EPS = [("e010", "0.10"), ("e040", "0.40")]
YCAP = 250  # spike panels annotate their true peak instead of rescaling

def load(tag): return torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})

bins = np.linspace(0, 1, 51)
fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.8), sharex=True, constrained_layout=True)
for ci, (ek, elab, col) in enumerate(EDITORS):
    for ri, (ec, eplab) in enumerate(EPS):
        ax = axes[ri, ci]
        d = load(f"frz_{ek}_{ec}_s0")
        innate = np.asarray(d["innate"], float)
        fin = np.asarray(d["op_raw"][-1], float)
        nu = float(np.asarray(d["pred_raw"][0], float).mean())
        bias = fin.mean() - innate.mean()
        dr = fin.std() / innate.std()
        cnt, _, _ = ax.hist(fin, bins=bins, color=col, alpha=0.8)
        ax.hist(innate, bins=bins, histtype="step", color="#555555", lw=1.4)
        ax.axvline(nu, color="#333333", lw=1.2, ls="--")
        if cnt.max() > YCAP:
            ax.set_ylim(0, YCAP)
            pk = bins[int(np.argmax(cnt))] + 0.01
            ax.text(pk, YCAP * 0.93, f"peak {int(cnt.max())}", fontsize=8,
                    ha="center", color="#333333")
        ax.text(0.02, 0.96, f"bias {bias:+.2f}\ndr {dr:.2f}", transform=ax.transAxes,
                va="top", fontsize=9)
        if ri == 0:
            ax.set_title(elab, fontsize=12)
        if ri == 1:
            ax.set_xlabel("opinion", fontsize=11)
        if ci == 0:
            ax.set_ylabel(f"social $\\epsilon$={eplab}\nagents / bin", fontsize=11)
handles = [Line2D([], [], color="#555555", lw=1.4, label="innate"),
           Line2D([], [], color="#999999", lw=6, alpha=0.8, label="final (round 30)"),
           Line2D([], [], color="#333333", lw=1.2, ls="--", label="editor mean $\\nu$")]
axes[0, 0].legend(handles=handles, frameon=False, fontsize=9, loc="center left")
fig.suptitle("Frozen editors (Q5, seed 0): four models, four attractors set by each deployed prior",
             fontsize=13)
fig.savefig(f"{OUT}/frozen_attractor_dists.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/frozen_attractor_dists.png")
