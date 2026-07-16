"""Act I.3 punchline: homophily protects the model that only WATCHES the
population (train, never serve = no-feedback).

Four no-feedback runs at the same cell (ML-Action, a040, replace, beta=0,
seed 0), crossing mixing {slow e010, wide e040} x social structure
{neutral gamma=0, homophily gamma=+1.5}:

  arm            ppl(tail)   dr(end)
  neutral e010    15.4        0.72
  homo    e010     5.8        0.99
  neutral e040    29.0        0.23
  homo    e040     9.2        0.95

READING (direction confirmed, magnitude honest): homophily keeps the
population diverse (dr ~0.95-0.99 vs 0.23-0.72), and the watching model
stays ~3x healthier (ppl 9.2/5.8 vs 29.0/15.4). The same social structure
that protects the population (Act I.1) protects a model open-loop-watching
it -- open-loop health is a property of the society. BUT homophily
PROTECTS, it does not fully RESCUE: homo ppl 9.2 (wide) / 5.8 (slow) still
sit above the pristine floor ~2, because even homophily lets the population
drift slightly and that residual non-stationarity rots the model.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/llm/plot_nf_homophily.py
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
# (label, eps) -> (neutral tag, homophily tag)
CELLS = [("slow  $\\epsilon$=0.1", "mlanf_e010_a040_rep_b0_s0", "mlanfH_e010_a040_rep_b0_s0"),
         ("wide  $\\epsilon$=0.4", "mlanf_e040_a040_rep_b0_s0", "mlanfH_e040_a040_rep_b0_s0")]
NEU, HOM = "#555555", "#2980b9"  # matches plot_pop_alone (neutral gray, homophily blue)
FLOOR = 2.0


def metrics(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    ppl = np.asarray(d["ppl_raw"], np.float32)
    return (float(np.median(ppl[-TAIL:])),
            float(op[-TAIL:].std(1).mean() / (inn.std() + 1e-9)))


labels = [c[0] for c in CELLS]
neu_ppl, neu_dr, hom_ppl, hom_dr = [], [], [], []
for _, nt, ht in CELLS:
    p, dr = metrics(nt); neu_ppl.append(p); neu_dr.append(dr)
    p, dr = metrics(ht); hom_ppl.append(p); hom_dr.append(dr)

with open(f"{OUT}/fig_nf_homophily.json", "w") as fh:
    json.dump({"cell": "no-feedback, a040 replace b0 s0",
               "labels": labels, "floor": FLOOR,
               "neutral": {"ppl": neu_ppl, "dr": neu_dr},
               "homophily": {"ppl": hom_ppl, "dr": hom_dr}}, fh, indent=2)

# ---- figure: model ppl | population dr, grouped bars ---------------------
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "font.size": 9})
fig, (axp, axd) = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
x = np.arange(len(labels)); w = 0.36

axp.bar(x - w / 2, neu_ppl, w, color=NEU, label="neutral  $\\gamma$=0")
axp.bar(x + w / 2, hom_ppl, w, color=HOM, label="homophily  $\\gamma$=+1.5")
for xi, v in zip(x - w / 2, neu_ppl):
    axp.text(xi, v + 0.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
for xi, v in zip(x + w / 2, hom_ppl):
    axp.text(xi, v + 0.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
axp.axhline(FLOOR, color="#c0392b", lw=0.9, ls="--")
axp.text(-0.42, FLOOR + 0.9, "pristine floor ~2", color="#c0392b", fontsize=7,
         va="bottom", ha="left")
axp.set_xticks(x); axp.set_xticklabels(labels)
axp.set_ylabel("model perplexity  ppl", fontsize=9)
axp.legend(frameon=False, fontsize=8, loc="upper left")

axd.bar(x - w / 2, neu_dr, w, color=NEU)
axd.bar(x + w / 2, hom_dr, w, color=HOM)
for xi, v in zip(x - w / 2, neu_dr):
    axd.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
for xi, v in zip(x + w / 2, hom_dr):
    axd.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
axd.set_xticks(x); axd.set_xticklabels(labels)
axd.set_ylim(0, 1.1)
axd.set_ylabel("population diversity  dr(30)", fontsize=9)

fig.text(0.5, -0.03, "No-feedback (train, never serve), ML-Action a040 replace "
         "$\\beta$=0 seed 0.  Protects, not full rescue: homo ppl still > floor.",
         ha="center", va="top", fontsize=7.5, color="#555555")
fig.savefig(f"{OUT}/fig_nf_homophily.png", dpi=140, bbox_inches="tight")
print(f"saved {OUT}/fig_nf_homophily.png and fig_nf_homophily.json")
print("neutral ppl", neu_ppl, "dr", neu_dr)
print("homo    ppl", hom_ppl, "dr", hom_dr)
