"""Gate boundary determines reach (Q3 gate dial, Qwen trap cell, seed 0).
Left: captured share (within 0.05 of the near mode 0.65) vs gate width
eps_AI, one line per beta (0.5 and 1; no capture without an anchor).
Right: geometry at beta=1: share at the near mode vs share at the far
mode 0.25. The far-mode camp appears once the gate reaches it; vertical
line at half the mode separation D/2 = 0.20.
"""
import os, numpy as np, torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs/qwen"
GATES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.70, 1.00]
def tag(a, bc):
    acode = f"a{int(round(a * 100)):03d}"
    if acode in ("a005", "a015", "a030", "a070", "a100"):
        return f"mlatA_e040_{acode}_rep_{bc}_s0"
    if acode == "a020":
        return f"mlat_e040_{acode}_rep_{bc}_s0"
    return f"mla2bv2_e040_{acode}_{bc}_s0"   # a010, a040 legacy

def share_near(t, mode):
    d = torch.load(f"{RUNS}/{t}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.asarray(d["op_raw"], float)[-1]
    return float(np.mean(np.abs(op - mode) <= 0.05))

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
RAMP = {"b0p5": plt.cm.GnBu(0.6), "b1": plt.cm.GnBu(0.95)}
for bc, blab in (("b0p5", "$\\beta$=0.5"), ("b1", "$\\beta$=1")):
    ys = [share_near(tag(a, bc), 0.65) for a in GATES]
    axA.plot(GATES, ys, "-o", color=RAMP[bc], lw=2.2, ms=7, label=blab)
axA.set_ylabel("captured share (within 0.05 of mode 0.65)", fontsize=11)
axA.legend(frameon=False, fontsize=10)
axA.set_title("Capture reaches wider when the anchor is weaker", fontsize=12)

near = [share_near(tag(a, "b1"), 0.65) for a in GATES]
far = [share_near(tag(a, "b1"), 0.25) for a in GATES]
mid = [share_near(tag(a, "b1"), 0.45) for a in GATES]
axB.plot(GATES, near, "-o", color="#2f6f9f", lw=2.2, ms=7, label="near mode 0.65")
axB.plot(GATES, mid, "-^", color="#1b7837", lw=2.2, ms=7, label="between modes 0.45")
axB.plot(GATES, far, "--s", color="#e08214", lw=2.2, ms=7, label="far mode 0.25")
axB.axvline(0.20, color="#999999", lw=1.2, ls=":")
axB.text(0.205, 0.72, "D/2 = 0.20", fontsize=9, color="#777777", va="top")
axB.set_ylabel("population share at each destination", fontsize=11)
axB.legend(frameon=False, fontsize=10, loc="center left")
axB.set_title("Geometry at $\\beta$=1: past the boundary the mass lands\nBETWEEN the modes, not at the far one (wide social $\\epsilon$)", fontsize=12)
for ax in (axA, axB):
    ax.set_xlabel("AI gate width  $\\epsilon_{AI}$", fontsize=12)
    ax.set_xscale("log"); ax.set_xticks(GATES)
    ax.set_xticklabels([f"{g:g}" for g in GATES], fontsize=9)
    ax.set_ylim(-0.03, 1.03)
fig.suptitle("Gate width sets the AI's reach: geometry decides where capture lands, dose decides how far it extends "
             "(Qwen, trap cell, seed 0)", fontsize=12)
fig.savefig(f"{OUT}/gate_boundary.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/gate_boundary.png")
