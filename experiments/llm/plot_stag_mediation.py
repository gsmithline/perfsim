"""Q2 evidence figure: contamination is endogenous and damage tracks it.
Left: s_tag(t) per memory arm (reset / carry / equilibrated), beta=0 runs,
thin per-run lines + bold arm medians. Right: end ppl (log) vs realized end
s_tag, beta=0 colored by arm (the dose-response); anchored runs (beta>=0.5)
in gray showing the anchor severs the relation.
"""
import os
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
FIGS = "experiments/llm/figs/qwen"
CELLS = ["e040_a010", "e040_a020", "e040_a040", "e020_a040"]
REGIMES = ["rep", "acc", "pri", "frep", "facc"]
BETAS = ["b0", "b0p5", "b1"]
ARMS = {"reset": "mlatR", "carry": "mlat", "equilibrated": "mlatE"}
COLOR = {"reset": "#1b9e77", "carry": "#d95f02", "equilibrated": "#7570b3"}
MARKER = dict(zip(CELLS, ["o", "s", "^", "D"]))
TAIL = 5

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


def find_tag(arm, cell, regime, beta):
    tag = f"{ARMS[arm]}_{cell}_{regime}_{beta}_s0"
    if os.path.exists(f"{RUNS}/{tag}/trajectory.pt"):
        return tag
    if arm == "carry":
        leg = LEGACY.get((cell, regime, beta))
        if leg and os.path.exists(f"{RUNS}/{leg}/trajectory.pt"):
            return leg
    return None


def load(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    s = np.array([row.get("s_tag", np.nan) for row in d["trajectory"]], np.float32)
    ppl = np.asarray(d["ppl_raw"], np.float32)
    return s, float(np.median(ppl[-TAIL:]))
# dose = time-averaged s over the run: eq spends all 30 rounds at full
# contamination, carry only the later ones -- end-s cannot separate them


plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.9), constrained_layout=True)

n_used = {}
for arm in ARMS:
    curves = []
    for cell in CELLS:
        for reg in REGIMES:
            tag = find_tag(arm, cell, reg, "b0")
            if tag is None:
                continue
            s, _ = load(tag)
            if np.isnan(s).all():
                continue
            curves.append(s)
            axL.plot(np.arange(1, len(s) + 1), s, color=COLOR[arm], alpha=0.14, lw=0.9)
    n_used[arm] = len(curves)
    if curves:
        L = min(len(c) for c in curves)
        med = np.nanmedian(np.stack([c[:L] for c in curves]), axis=0)
        axL.plot(np.arange(1, L + 1), med, color=COLOR[arm], lw=2.8, label=f"{arm} (n={len(curves)})")
axL.axhline(0.5, color="0.45", ls="--", lw=1.2)
axL.text(29.5, 0.52, "classical setting: synthetic share is a fixed dial",
         ha="right", fontsize=9, color="0.35")
axL.set_xlabel("round", fontsize=12)
axL.set_ylabel("$s$: model share of opinion", fontsize=12)
axL.set_ylim(0, 1.02); axL.set_xlim(1, 30)
axL.legend(fontsize=10, frameon=False, loc="lower right")
axL.set_title("the loop sets its own contamination rate ($\\beta$=0)", fontsize=12)

# right panel: replace recipe only -- acc/pri add Q1's data-anchor
# protection on top and would smear the dose-response
for arm in ARMS:
    for cell in CELLS:
        for reg in ["rep"]:
            for beta in BETAS:
                tag = find_tag(arm, cell, reg, beta)
                if tag is None:
                    continue
                s, pend = load(tag)
                send = float(np.nanmean(s))
                if np.isnan(send):
                    continue
                if beta == "b0":
                    axR.scatter(send, pend, color=COLOR[arm], marker=MARKER[cell],
                                s=34, alpha=0.85, edgecolors="none", zorder=3)
                else:
                    axR.scatter(send, pend, color="0.72", marker=MARKER[cell],
                                s=18, alpha=0.6, edgecolors="none", zorder=2)
axR.set_yscale("log")
axR.set_xlabel("realized dose: run-mean $s$", fontsize=12)
axR.set_ylabel("end median perplexity", fontsize=12)
axR.set_title("damage tracks dose, not route (replace); anchor severs it",
              fontsize=12)
handles = [plt.Line2D([], [], color=COLOR[a], marker="o", ls="", label=a) for a in ARMS]
handles.append(plt.Line2D([], [], color="0.72", marker="o", ls="",
                          label="anchored ($\\beta\\geq0.5$)"))
axR.legend(handles=handles, fontsize=9, frameon=False, loc="upper left")
fig.suptitle("Contamination of the human data channel is endogenous "
             "(ML-Action slab, seed 0; arms differ only in population memory)",
             fontsize=12)
out = f"{FIGS}/stag_mediation.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print("saved", out, "| runs per arm at b0:", n_used)
