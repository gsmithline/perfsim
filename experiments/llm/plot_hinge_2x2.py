"""Act I/II hinge: the causal 2x2 {serve x train}.

Two one-way wires (Act I) plus their absence and their combination (Act II),
on ONE fixed cell so the corners are comparable: Qwen ML-Action, eps=0.4
(wide mixing), replace/continual, beta=0, seed 0.

  serve = the model DEPLOYS predictions into the population (the gate).
  train = the model FITS on the population's data.

  corner            serve train   run
  no-AI              off   off    population-alone Deffuant (fig_pop_alone, gamma0 e040)
  frozen             ON    off    frz_qwen_e040_s0          (weights never move)
  nf (no-feedback)   off   ON     mlanf_e040_a040_rep_b0_s0 (trains, serves nothing)
  closed             ON    ON     mla2dv2_e040_a040_b0_s0   (post-RNG-fix closed loop)

Two metrics, two objects:
  population diversity  dr = op_std(tail)/innate_std   (TAIL=5, read_atlas_slab convention)
  model health          ppl = median per-agent perplexity over the tail
Base/untrained model ppl (the do-nothing reference) = the FROZEN run's ppl,
since frozen never trains -> its weights are the base model's.

READING (double dissociation, with one honest wrinkle):
  * model rots ONLY when it TRAINS (ppl varies down the rows: 13.8 -> 29/46);
    serving amplifies the rot within the train-on row (29 -> 46).
  * the population only leaves its free self-collapse when the model SERVES
    (dr moves across the columns). WRINKLE: frozen serve SORTS onto two prior
    modes and props diversity UP (0.71), while closed serve+train collapses it
    back DOWN (0.27) -- direction within serve-on is set by training.
  * only the closed corner (bottom-right) degrades BOTH objects at once.
    Act II (anchoring) is the intervention that trades one for the other.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/llm/plot_hinge_2x2.py
Pure numpy/torch on already-pulled runs -- no transformers, no download.
"""
import json
import os

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs"
TAIL = 5
EPS = 0.4  # wide-mixing cell


def metrics(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    ppl = np.asarray(d["ppl_raw"], np.float32)
    return (float(op[-TAIL:].std(1).mean() / (inn.std() + 1e-9)),
            float(np.median(ppl[-TAIL:])))


# ---- gather the four corners --------------------------------------------
noai_cells = json.load(open(f"{OUT}/fig_pop_alone.json"))["cells"]
noai_dr = next(c["dr30_mean"] for c in noai_cells
               if abs(c["gamma"]) < 1e-9 and abs(c["eps"] - EPS) < 1e-6)

frz_dr, frz_ppl = metrics("frz_qwen_e040_s0")
nf_dr, nf_ppl = metrics("mlanf_e040_a040_rep_b0_s0")
cl_dr, cl_ppl = metrics("mla2dv2_e040_a040_b0_s0")
base_ppl = frz_ppl  # frozen never trains -> its ppl IS the do-nothing model

# rows = train (off top, on bottom); cols = serve (off left, on right)
DR = np.array([[noai_dr, frz_dr],
               [nf_dr,   cl_dr]])
PPL = np.array([[base_ppl, frz_ppl],
                [nf_ppl,   cl_ppl]])
NAME = np.array([["no-AI", "frozen"],
                 ["nf", "closed"]])

data = {"cell": "Qwen ML-Action e040 rep b0 s0", "tail": TAIL,
        "rows": "train off/on", "cols": "serve off/on",
        "dr": DR.tolist(), "ppl": PPL.tolist(), "names": NAME.tolist(),
        "note": "base_ppl = frozen ppl (frozen never trains)"}
with open(f"{OUT}/fig_hinge_2x2.json", "w") as fh:
    json.dump(data, fh, indent=2)

# ---- figure: two 2x2 heatmaps sharing serve x train axes -----------------
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.linewidth": 1.1, "font.size": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), constrained_layout=True)


def panel(ax, M, title, cmap, norm, fmt, good_high):
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="equal")
    for i in range(2):
        for j in range(2):
            # label contrast: dark cells get white text
            frac = norm(M[i, j])
            txt = "white" if (frac > 0.55) else "black"
            ax.text(j, i - 0.14, NAME[i, j], ha="center", va="center",
                    color=txt, fontsize=10, fontweight="bold")
            ax.text(j, i + 0.16, fmt(M[i, j]), ha="center", va="center",
                    color=txt, fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["serve off", "serve ON"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["train off", "train ON"])
    ax.set_title(title, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    return im


# population diversity: sequential; low dr = collapsed (bad for population)
panel(axes[0], DR, "population diversity  dr(30)\n(serve moves it; frozen SORTS up)",
      cmap="YlGnBu", norm=plt.Normalize(0.0, 1.0),
      fmt=lambda v: f"{v:.2f}", good_high=True)
# model health: log ppl; high = rotted (bad for model)
panel(axes[1], PPL, "model perplexity  ppl\n(train rots it; serve amplifies)",
      cmap="OrRd", norm=LogNorm(vmin=10, vmax=60),
      fmt=lambda v: f"{v:.0f}", good_high=False)

fig.suptitle("The hinge — each object degrades through a different wire:\n"
             "train rots the model · serve moves the population · "
             "only the closed loop fails both",
             fontsize=10.5)
fig.text(0.5, -0.02, "Qwen ML-Action, $\\epsilon$=0.4 (wide mixing), replace, "
         "$\\beta$=0, seed 0.  no-AI/frozen ppl = untrained base.",
         ha="center", va="top", fontsize=7.5, color="#555555")
fig.savefig(f"{OUT}/fig_hinge_2x2.png", dpi=140, bbox_inches="tight")
print(f"saved {OUT}/fig_hinge_2x2.png and fig_hinge_2x2.json")
print("DR grid  (rows train off/on, cols serve off/on):\n", np.round(DR, 3))
print("PPL grid:\n", np.round(PPL, 1))
