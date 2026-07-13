"""Gate-dial version of the dr/vr grid: same axes as the feature-dial twin
(fdial_llm_dr_vr*.png) but lines = gate width eps_AI (natural features,
continual weights, replace, trap-cell social eps 0.40, seed 0). Rows =
beta {0, 0.5, 1}. Dark = wide gate.
"""
import os, numpy as np, torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs/qwen"
GATES = [1.00, 0.70, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05]
BETAS = [("b0", "0"), ("b0p5", "0.5"), ("b1", "1")]

def tag(a, bc):
    acode = f"a{int(round(a * 100)):03d}"
    if acode in ("a005", "a015", "a030", "a070", "a100"):
        return f"mlatA_e040_{acode}_rep_{bc}_s0"
    if bc == "b0":
        return f"mla2dv2_e040_{acode}_b0_s0"
    if acode == "a020":
        return f"mlat_e040_{acode}_rep_{bc}_s0"
    return f"mla2bv2_e040_{acode}_{bc}_s0"

def ex(t): return os.path.exists(f"{RUNS}/{t}/trajectory.pt")

def curves(t):
    d = torch.load(f"{RUNS}/{t}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.asarray(d["op_raw"], float)
    pred = np.asarray(d["pred_raw"], float)
    innate = np.asarray(d["innate"], float)
    dr = op.std(1) / innate.std()
    pstd = np.array([np.nanstd(np.where(np.isfinite(p), p, np.nan)) for p in pred])
    vr = pstd / np.maximum(op.std(1), 1e-9)
    return dr, vr

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})

fig, axes = plt.subplots(3, 2, figsize=(12.6, 12.6), constrained_layout=True, sharex=True)
ts = np.arange(1, 31)
ramp = plt.cm.GnBu(np.linspace(0.95, 0.35, len(GATES)))
for ri, (bc, blab) in enumerate(BETAS):
    axA, axB = axes[ri]
    for a, c in zip(GATES, ramp):
        t = tag(a, bc)
        if not ex(t):
            continue
        dr, vr = curves(t)
        axA.plot(ts, dr, color=c, lw=2.2, label=f"{a:g}")
        axB.plot(ts, vr, color=c, lw=2.2)
    axA.set(ylabel=f"$\\beta$={blab}\nDiversity ratio  $d_r$", ylim=(0, 1.05))
    axB.set_ylabel("Platform/population spread  $v_r$  (log)")
    axB.set_yscale("log")
    axB.axhline(1.0, color="#bbbbbb", lw=1, ls=":")
    if ri == 0:
        axA.legend(title="gate width $\\epsilon_{AI}$", frameon=False, fontsize=9,
                   title_fontsize=10, ncols=2)
for ax in axes[-1]:
    ax.set_xlabel("Timestep")
fig.suptitle("LLM gate dial by anchor strength: Qwen, continual weights, natural features, "
             "replace, trap cell, seed 0", fontsize=13)
fig.savefig(f"{OUT}/gdial_llm_dr_vr.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/gdial_llm_dr_vr.png")
