"""Q2 evidence figure. (a) corpus model-share s(t) per memory arm, beta=0.
(b) end median perplexity vs mean corpus model share (replace recipe),
with a fitted dose-response line.
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
ARMS = {"reset": "mlatR", "carry": "mlat", "equilibrated": "mlatE"}
COLOR = {"reset": "#0173b2", "carry": "#de8f05", "equilibrated": "#cc2529"}
MARK = {"reset": "o", "carry": "^", "equilibrated": "s"}
LABEL = {"reset": "reset (memoryless, shallow contact)",
         "carry": "carry (population memory)",
         "equilibrated": "equilibrated (deep contact)"}
TAIL = 5

LEGACY = {}
for c in CELLS:
    LEGACY[(c, "rep", "b0")] = f"mla2dv2_{c}_b0_s0"
LEGACY[("e040_a040", "acc", "b0")] = "mla2drv2_e040_a040_b0_acc_s0"
LEGACY[("e040_a040", "pri", "b0")] = "mla2drv2_e040_a040_b0_pri_s0"
LEGACY[("e040_a040", "frep", "b0")] = "mla2dfv2_e040_a040_b0_rep_s0"
LEGACY[("e040_a040", "facc", "b0")] = "mla2dfv2_e040_a040_b0_acc_s0"


def find_tag(arm, cell, regime):
    tag = f"{ARMS[arm]}_{cell}_{regime}_b0_s0"
    if os.path.exists(f"{RUNS}/{tag}/trajectory.pt"):
        return tag
    if arm == "carry":
        leg = LEGACY.get((cell, regime, "b0"))
        if leg and os.path.exists(f"{RUNS}/{leg}/trajectory.pt"):
            return leg
    return None


def load(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    s = np.array([row.get("s_tag", np.nan) for row in d["trajectory"]], np.float32)
    ppl = np.asarray(d["ppl_raw"], np.float32)
    return s, float(np.median(ppl[-TAIL:]))


plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.9), constrained_layout=True)

for arm in ARMS:
    curves = []
    for cell in CELLS:
        for reg in REGIMES:
            tag = find_tag(arm, cell, reg)
            if tag is None:
                continue
            s, _ = load(tag)
            if np.isnan(s).all():
                continue
            curves.append(s)
            axL.plot(np.arange(1, len(s) + 1), s, color=COLOR[arm], alpha=0.13, lw=0.9)
    if curves:
        L = min(len(c) for c in curves)
        med = np.nanmedian(np.stack([c[:L] for c in curves]), axis=0)
        axL.plot(np.arange(1, L + 1), med, color=COLOR[arm], lw=2.8, label=LABEL[arm])
axL.axhline(0.5, color="0.45", ls="--", lw=1.2)
axL.text(29.5, 0.52, "classical setting: synthetic share is a fixed dial",
         ha="right", fontsize=9, color="0.35")
axL.axhline(0.30, color="0.55", ls=":", lw=1.1)
axL.text(29.5, 0.315, "one-blend floor ($W$=0.3)", ha="right", fontsize=8.5,
         color="0.45")
axL.set_xlabel("round", fontsize=12)
axL.set_ylabel("corpus model-share $s(t)$", fontsize=12)
axL.set_ylim(0, 1.02); axL.set_xlim(1, 30)
axL.legend(fontsize=9.5, frameon=False, loc="lower right")
axL.set_title("(a) loop sets its own contamination rate", fontsize=12)

xs, ys = [], []
for arm in ARMS:
    for cell in CELLS:
        tag = find_tag(arm, cell, "rep")
        if tag is None:
            continue
        s, pend = load(tag)
        sbar = float(np.nanmean(s))
        if np.isnan(sbar):
            continue
        xs.append(sbar); ys.append(pend)
        axR.scatter(sbar, pend, color=COLOR[arm], marker=MARK[arm], s=52,
                    alpha=0.9, edgecolors="none", zorder=3)
a, b = np.polyfit(xs, np.log10(ys), 1)
grid = np.linspace(min(xs) - 0.03, max(xs) + 0.03, 100)
axR.plot(grid, 10 ** (a * grid + b), color="0.35", lw=1.4, ls="-", zorder=2)
axR.set_yscale("log")
axR.set_xlabel("mean corpus model share", fontsize=12)
axR.set_ylabel("end median perplexity", fontsize=12)
axR.set_title("(b) damage tracks share", fontsize=12)
handles = [plt.Line2D([], [], color=COLOR[a2], marker=MARK[a2], ls="", label=a2)
           for a2 in ARMS]
axR.legend(handles=handles, fontsize=9.5, frameon=False, loc="upper left")

out = f"{FIGS}/stag_mediation.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print("saved", out, f"| fit: log10(ppl) = {a:.2f} s + {b:.2f}")
