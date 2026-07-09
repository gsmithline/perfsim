"""s(t) across anchor doses and models: rows = {Qwen, Llama}, cols = beta
{0, 0.5, 1}. Qwen panels show all three memory arms; Llama only has the
carry arm (no reset/eq slabs were run for it).
"""
import os
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
CELLS = ["e040_a010", "e040_a020", "e040_a040", "e020_a040"]
REGIMES = ["rep", "acc", "pri", "frep", "facc"]
BETAS = [("b0", "0"), ("b0p5", "0.5"), ("b1", "1")]
COLOR = {"reset": "#1b9e77", "carry": "#d95f02", "equilibrated": "#7570b3"}

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


def find_tag(stem, cell, reg, beta, legacy_ok):
    tag = f"{stem}_{cell}_{reg}_{beta}_s0"
    if os.path.exists(f"{RUNS}/{tag}/trajectory.pt"):
        return tag
    if legacy_ok:
        leg = LEGACY.get((cell, reg, beta))
        if leg and os.path.exists(f"{RUNS}/{leg}/trajectory.pt"):
            return leg
    return None


def s_curve(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    return np.array([row.get("s_tag", np.nan) for row in d["trajectory"]], np.float32)


plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})
fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), sharex=True, sharey=True,
                         constrained_layout=True)

ROWS = [("qwen", {"reset": ("mlatR", False), "carry": ("mlat", True),
                  "equilibrated": ("mlatE", False)}),
        ("llama", {"carry": ("mlatL", False)})]
for ri, (model, arms) in enumerate(ROWS):
    for ci, (bc, blab) in enumerate(BETAS):
        ax = axes[ri, ci]
        for arm, (stem, legacy_ok) in arms.items():
            curves = []
            for cell in CELLS:
                for reg in REGIMES:
                    tag = find_tag(stem, cell, reg, bc, legacy_ok)
                    if tag is None:
                        continue
                    s = s_curve(tag)
                    if np.isnan(s).all():
                        continue
                    curves.append(s)
                    ax.plot(np.arange(1, len(s) + 1), s, color=COLOR[arm],
                            alpha=0.10, lw=0.8)
            if curves:
                L = min(len(c) for c in curves)
                med = np.nanmedian(np.stack([c[:L] for c in curves]), axis=0)
                ax.plot(np.arange(1, L + 1), med, color=COLOR[arm], lw=2.6,
                        label=f"{arm} (n={len(curves)})")
        if ri == 0:
            ax.set_title(f"$\\beta$={blab}", fontsize=13)
        if ci == 0:
            ax.set_ylabel(f"{model}\n$s$: model share of opinion", fontsize=11)
        if ri == 1:
            ax.set_xlabel("round", fontsize=12)
        ax.set_ylim(0, 1.02); ax.set_xlim(1, 30)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.suptitle("Contamination dynamics by anchor dose and model (ML-Action, seed 0; "
             "Llama has no reset/eq arms)", fontsize=12)
out = "experiments/llm/figs/qwen/stag_by_beta.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print("saved", out)
