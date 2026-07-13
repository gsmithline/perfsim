"""Cross-model drift (Q5 stage 3, trap cell e040_a040 replace, seed 0).
One panel per model. x = round; y = mean opinion / mean model output.
Lines: population mean and model output mean at beta 0 and beta 1, plus
the frozen editor's output mean as the beta -> inf reference.
Reading: unanchored, the model follows the crowd (its mean drifts to the
population); anchored, the crowd follows the model (population drifts to
the frozen prior line). Gemma absent: no retraining runs (torch 2.6).
"""
import os, numpy as np, torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs/qwen"
MODELS = [
    ("Qwen2.5-7B",  {"b0": "mla2dv2_e040_a040_b0_s0", "b1": "mla2bv2_e040_a040_b1_s0"},  "frz_qwen_e040_s0"),
    ("Llama-3.1-8B", {"b0": "mlatL_e040_a040_rep_b0_s0", "b1": "mlatL_e040_a040_rep_b1_s0"}, "frz_llama_e040_s0"),
    ("OLMo-2-7B",   {"b0": "olmo_e040_a040_rep_b0_s0", "b1": "olmo_e040_a040_rep_b1_s0"}, "frz_olmo_e040_s0"),
]
CB = {"b0": "#2f6f9f", "b1": "#e08214"}

def load(tag): return torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
def means(tag):
    d = load(tag)
    op = np.asarray(d["op_raw"], float)
    pred = np.asarray(d["pred_raw"], float)
    return np.nanmean(op, 1), np.nanmean(np.where(np.isfinite(pred), pred, np.nan), 1)

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True, sharey=True,
                         constrained_layout=True)
rounds = np.arange(1, 31)
for ax, (name, tags, frz) in zip(axes, MODELS):
    innate_mu = float(np.asarray(load(tags["b0"])["innate"], float).mean())
    ax.axhline(innate_mu, color="#bbbbbb", lw=1)
    _, frz_pred = means(frz)
    ax.plot(rounds, frz_pred, color="#555555", lw=1.6, ls="--")
    for bc in ("b0", "b1"):
        opm, prm = means(tags[bc])
        ax.plot(rounds, opm, color=CB[bc], lw=2.2)
        ax.plot(rounds, prm, color=CB[bc], lw=1.8, ls=":")
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("round", fontsize=11)
axes[0].set_ylabel("mean opinion / mean model output", fontsize=11)
handles = [Line2D([], [], color=CB["b0"], lw=2.2, label="population, $\\beta$=0"),
           Line2D([], [], color=CB["b0"], lw=1.8, ls=":", label="model, $\\beta$=0"),
           Line2D([], [], color=CB["b1"], lw=2.2, label="population, $\\beta$=1"),
           Line2D([], [], color=CB["b1"], lw=1.8, ls=":", label="model, $\\beta$=1"),
           Line2D([], [], color="#555555", lw=1.6, ls="--", label="frozen editor"),
           Line2D([], [], color="#bbbbbb", lw=1, label="innate mean")]
axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="best")
fig.suptitle("Who follows whom: at $\\beta$=0 the model drifts to the crowd; "
             "at $\\beta$=1 the crowd drifts to the model's frozen prior (seed 0)",
             fontsize=13)
fig.savefig(f"{OUT}/crossmodel_drift.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/crossmodel_drift.png")
