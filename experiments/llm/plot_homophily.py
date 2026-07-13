"""Homophily resists the anchor (Q3 revision, Qwen trap cell, seed 0).
x = homophily gamma {-1.5, 0, +1.5}; lines = beta {0, 0.5, 1}.
Left: final population diversity dr. Right: absolute mean displacement.
Data: mlatgx gp15/gn15 runs + the gamma=0 rep cells.
"""
import os, numpy as np, torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs/qwen"
GAMMAS = [-1.5, 0.0, 1.5]
TAGS = {
    (-1.5, "b0"): "mlatgx_gn15_e040_a040_rep_b0_s0",
    (-1.5, "b0p5"): "mlatgx_gn15_e040_a040_rep_b0p5_s0",
    (-1.5, "b1"): "mlatgx_gn15_e040_a040_rep_b1_s0",
    (0.0, "b0"): "mla2dv2_e040_a040_b0_s0",
    (0.0, "b0p5"): "mla2bv2_e040_a040_b0p5_s0",
    (0.0, "b1"): "mla2bv2_e040_a040_b1_s0",
    (1.5, "b0"): "mlatgx_gp15_e040_a040_rep_b0_s0",
    (1.5, "b0p5"): "mlatgx_gp15_e040_a040_rep_b0p5_s0",
    (1.5, "b1"): "mlatgx_gp15_e040_a040_rep_b1_s0",
}
BETAS = [("b0", "$\\beta$=0"), ("b0p5", "$\\beta$=0.5"), ("b1", "$\\beta$=1")]
RAMP = plt.cm.GnBu([0.45, 0.7, 0.95])

def stats(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.asarray(d["op_raw"], float)[-1]
    innate = np.asarray(d["innate"], float)
    return op.std() / innate.std(), abs(op.mean() - innate.mean())

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
for (bc, blab), col in zip(BETAS, RAMP):
    drs, biases = zip(*[stats(TAGS[(g, bc)]) for g in GAMMAS])
    axA.plot(GAMMAS, drs, "-o", color=col, lw=2.2, ms=7, label=blab)
    axB.plot(GAMMAS, biases, "-o", color=col, lw=2.2, ms=7, label=blab)
axA.axhline(1.0, color="#bbbbbb", lw=1, ls=":")
axA.set_ylabel("final population diversity  $d_r$", fontsize=12)
axB.set_ylabel("absolute mean displacement  $|\\bar{x}_T - \\bar{x}_0|$", fontsize=12)
for ax in (axA, axB):
    ax.set_xticks(GAMMAS)
    ax.set_xticklabels(["-1.5\nheterophilic", "0\nneutral", "+1.5\nhomophilic"], fontsize=10)
    ax.set_xlabel("population homophily  $\\gamma$", fontsize=12)
axA.legend(frameon=False, fontsize=10, loc="upper left")
fig.suptitle("Homophilic populations resist the anchor: diversity survives ($d_r$ up, left)\n"
             "and displacement shrinks (right) as $\\gamma$ rises (Qwen, trap cell, seed 0)",
             fontsize=12)
fig.savefig(f"{OUT}/homophily_resists.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/homophily_resists.png")
