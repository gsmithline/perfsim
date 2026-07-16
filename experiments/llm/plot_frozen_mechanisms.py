"""Act I.2 mechanism panel: frozen serving (model -> population, never train)
fails in THREE distinct ways, one per model family.

8 frozen runs (frz_<model>_<eps>_s0, 4 models x eps {0.1,0.4}, seed 0). Each
cell = the population's opinion distribution at round 30 under a frozen model
that only serves. Innate outlined for reference. Two diagnostics separate the
mechanisms:
  corr(pred, innate) -- does the model preserve the individual? (model-level)
  the distribution SHAPE -- point mass / two modes / one shifted mode

Ordered left->right by corr (individual preservation), which IS the mechanism axis:

  Llama  corr~0    POINT MASS: serves 0.50 to everyone -> population collapses
                   to one spike; dr -> 0.00 at wide mixing. Deploys nothing.
  Qwen   corr 0.04 SORTING:    ignores the individual but has TWO prior modes
                   (0.25/0.65) -> assigns people to modes -> BIMODAL; dr can
                   exceed 1 (spread onto two camps) at slow mixing.
  OLMo   corr 0.36 NUDGING:    individual signal survives; prior shifts the
  Gemma  corr 0.51 NUDGING:    level UP (+0.07..+0.10) but keeps the ordering.

So "frozen serve = collapse" is too flat: it is collapse (Llama), sorting
(Qwen), or nudging (Gemma/OLMo) depending on the prior's geometry.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/llm/plot_frozen_mechanisms.py
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
# ordered by corr(pred,innate): point-mass -> sorting -> nudging
MODELS = [("llama", "Llama", "POINT MASS", "#c0392b"),
          ("qwen", "Qwen", "SORTING", "#8e44ad"),
          ("olmo", "OLMo", "NUDGING", "#27ae60"),
          ("gemma", "Gemma", "NUDGING", "#16a085")]
EPS = [("e040", 0.4), ("e010", 0.1)]


def cell(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)
    pr = np.clip(np.asarray(d["pred_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    a, b = pr[-1], inn
    m = np.isfinite(a) & np.isfinite(b)
    corr = float(np.corrcoef(a[m], b[m])[0, 1]) if (m.sum() > 5 and a[m].std() > 1e-6) else np.nan
    return dict(op=op[-1], inn=inn,
                dr=float(op[-TAIL:].std(1).mean() / (inn.std() + 1e-9)),
                disp=float(op[-TAIL:].mean() - inn.mean()), corr=corr)


plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.0, "font.size": 8.5})
fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.6), sharex=True, sharey=True,
                         constrained_layout=True)
bins = np.linspace(0, 1, 31)
out = {}
for c, (mkey, mname, mech, color) in enumerate(MODELS):
    for r, (etag, eps) in enumerate(EPS):
        ax = axes[r, c]
        d = cell(f"frz_{mkey}_{etag}_s0")
        out[f"{mkey}_{etag}"] = {"dr": d["dr"], "disp": d["disp"], "corr": d["corr"]}
        ax.hist(d["inn"], bins=bins, histtype="step", lw=0.9, color="#000",
                alpha=0.35, density=True)
        ax.hist(d["op"], bins=bins, color=color, alpha=0.75, density=True)
        ax.axvline(float(d["inn"].mean()), color="#000", lw=0.6, ls=":", alpha=0.5)
        cc = "n/a" if not np.isfinite(d["corr"]) else f"{d['corr']:.2f}"
        head = f"{mname} · {mech}\n" if r == 0 else ""
        ax.text(0.04, 0.95, f"{head}dr {d['dr']:.2f}   corr {cc}",
                transform=ax.transAxes, fontsize=8.5, va="top", linespacing=1.5,
                color=color, fontweight="bold")
        ax.set_xlim(0, 1)
        if c == 0:
            ax.set_ylabel(f"$\\epsilon$={eps}\ndensity", fontsize=9)
        if r == 1:
            ax.set_xlabel("opinion", fontsize=8.5)

with open(f"{OUT}/fig_frozen_mechanisms.json", "w") as fh:
    json.dump({"cell": "frozen serve, 4 models x eps{0.4,0.1}, s0",
               "innate_mean": 0.63, "metrics": out}, fh, indent=2)

fig.text(0.5, -0.02, "ML-Action, replace, seed 0.  Black outline = innate "
         "distribution; dotted = innate mean (~0.63).  Rows: wide vs slow mixing.",
         ha="center", va="top", fontsize=7.5, color="#555555")
fig.savefig(f"{OUT}/fig_frozen_mechanisms.png", dpi=140, bbox_inches="tight")
print(f"saved {OUT}/fig_frozen_mechanisms.png and fig_frozen_mechanisms.json")
for k, v in out.items():
    print(f"  {k:14} dr {v['dr']:.2f}  disp {v['disp']:+.2f}  "
          f"corr {v['corr'] if np.isfinite(v['corr']) else float('nan'):.2f}")
