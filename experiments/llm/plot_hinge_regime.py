"""Act I/II hinge, robustness companion: is the dramatic ppl in the 2x2 an
artifact of the REPLACE regime? Expand the trained corners along the data
regime axis {replace, accumulate, pristine} at the same cell (Qwen ML-Action,
eps=0.4, beta=0, seed 0).

Corners re-used: no feedback (serve off, train on) and closed (serve on,
train on). Frozen/no-AI don't train, so regime is irrelevant to them.

  regime      no feedback (serve off)   closed (serve on)
  replace       ppl 29.0 / dr 0.23        ppl 46.1 / dr 0.27
  accumulate    ppl  2.6 / dr 0.23        ppl  2.8 / dr 0.52
  pristine      (not run)                 ppl  1.7 / dr 0.62

READING (honest, two parts):
  * MODEL rot is a REPLACE story: accumulate/pristine keep ppl ~2-3 whether
    or not the model serves. The 29/46 in the hinge is specific to overwrite-
    every-round training, not to closing the loop per se.
  * POPULATION: under serve-off the dr is regime-invariant (0.23 -- training
    never touches the population); under serve-on a HEALTHIER model (accum/
    pristine) also spares the population (dr 0.27 -> 0.62). Better data
    regimes protect both objects at once.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/llm/plot_hinge_regime.py
Pure numpy/torch on pulled runs -- no transformers, no download.
"""
import json
import os

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs"
TAIL = 5

# regime -> (no-feedback tag, closed tag); None where the run was not made
CORNERS = {
    "replace":    ("mlanf_e040_a040_rep_b0_s0", "mla2dv2_e040_a040_b0_s0"),
    "accumulate": ("mlanf_e040_a040_acc_b0_s0", "mla2drv2_e040_a040_b0_acc_s0"),
    "pristine":   (None,                        "mla2drv2_e040_a040_b0_pri_s0"),
}
REGIMES = list(CORNERS)


def metrics(tag):
    if tag is None:
        return (np.nan, np.nan)
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    ppl = np.asarray(d["ppl_raw"], np.float32)
    return (float(np.median(ppl[-TAIL:])),
            float(op[-TAIL:].std(1).mean() / (inn.std() + 1e-9)))


nf_ppl, nf_dr, cl_ppl, cl_dr = [], [], [], []
for r in REGIMES:
    (p, dr) = metrics(CORNERS[r][0]); nf_ppl.append(p); nf_dr.append(dr)
    (p, dr) = metrics(CORNERS[r][1]); cl_ppl.append(p); cl_dr.append(dr)

with open(f"{OUT}/fig_hinge_regime.json", "w") as fh:
    json.dump({"regimes": REGIMES, "no_feedback": {"ppl": nf_ppl, "dr": nf_dr},
               "closed": {"ppl": cl_ppl, "dr": cl_dr},
               "cell": "Qwen ML-Action e040 b0 s0"}, fh, indent=2)

# ---- figure: two panels (model ppl | population dr) vs regime -------------
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "font.size": 9})
fig, (axp, axd) = plt.subplots(1, 2, figsize=(7.4, 3.4), constrained_layout=True)
x = np.arange(len(REGIMES))
NF, CL = "#7f8c8d", "#c0392b"

axp.plot(x, nf_ppl, "-o", color=NF, lw=2, ms=6, label="no feedback (serve off)")
axp.plot(x, cl_ppl, "-s", color=CL, lw=2, ms=6, label="closed (serve on)")
axp.axhline(2.0, color="#2980b9", lw=0.8, ls="--")
axp.text(2.02, 2.0, "healthy ~2", color="#2980b9", fontsize=7, va="bottom", ha="right")
axp.set_yscale("log")
axp.set_xticks(x); axp.set_xticklabels(REGIMES)
axp.set_ylabel("model perplexity  ppl (log)", fontsize=9)
axp.set_title("model rot is a REPLACE story", fontsize=9.5)
axp.legend(frameon=False, fontsize=7.5, loc="upper right")

axd.plot(x, nf_dr, "-o", color=NF, lw=2, ms=6, label="no feedback (serve off)")
axd.plot(x, cl_dr, "-s", color=CL, lw=2, ms=6, label="closed (serve on)")
axd.set_xticks(x); axd.set_xticklabels(REGIMES)
axd.set_ylim(0, 1.0)
axd.set_ylabel("population diversity  dr(30)", fontsize=9)
axd.set_title("a healthier model also spares the population", fontsize=9.5)
axd.legend(frameon=False, fontsize=7.5, loc="upper left")

fig.suptitle("The hinge's damage is regime-specific: accumulate/pristine keep the "
             "model healthy (ppl ~2-3)\nand — when served — leave the population "
             "far more diverse than replace does", fontsize=10)
fig.text(0.5, -0.03, "Qwen ML-Action, $\\epsilon$=0.4, $\\beta$=0, seed 0.  "
         "no-feedback pristine not run (serve-off pop is regime-invariant anyway).",
         ha="center", va="top", fontsize=7.5, color="#555555")
fig.savefig(f"{OUT}/fig_hinge_regime.png", dpi=140, bbox_inches="tight")
print(f"saved {OUT}/fig_hinge_regime.png and fig_hinge_regime.json")
print("nf ppl", np.round(nf_ppl, 1), "dr", np.round(nf_dr, 3))
print("cl ppl", np.round(cl_ppl, 1), "dr", np.round(cl_dr, 3))
