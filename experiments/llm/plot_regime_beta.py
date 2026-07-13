"""Training-regime figures at the trap cell (e040_a040, seed 0).
Fig regime_vs_beta: model degradation log10(end/start median ppl) vs
anchor strength beta, one line per recipe (rep/acc/pri + fresh variants),
Qwen. Data anchors and the KL anchor are substitutes: every recipe is on
the floor by beta=1.
Fig unanchored_regime_bars: beta=0 only, groups replace/accumulate/
pristine, bars Qwen and Llama; left = model degradation, right = final
population diversity dr. OLMo omitted (replace-only runs).
"""
import os, numpy as np, torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs/qwen"
BETAS = [("b0", 0.0), ("b0p5", 0.5), ("b1", 1.0)]
QWEN = {
    "rep":  {"b0": "mla2dv2_e040_a040_b0_s0",       "b0p5": "mla2bv2_e040_a040_b0p5_s0",   "b1": "mla2bv2_e040_a040_b1_s0"},
    "acc":  {"b0": "mla2drv2_e040_a040_b0_acc_s0",  "b0p5": "mlat_e040_a040_acc_b0p5_s0",  "b1": "mlat_e040_a040_acc_b1_s0"},
    "pri":  {"b0": "mla2drv2_e040_a040_b0_pri_s0",  "b0p5": "mlat_e040_a040_pri_b0p5_s0",  "b1": "mlat_e040_a040_pri_b1_s0"},
    "frep": {"b0": "mla2dfv2_e040_a040_b0_rep_s0",  "b0p5": "mlat_e040_a040_frep_b0p5_s0", "b1": "mlat_e040_a040_frep_b1_s0"},
    "facc": {"b0": "mla2dfv2_e040_a040_b0_acc_s0",  "b0p5": "mlat_e040_a040_facc_b0p5_s0", "b1": "mlat_e040_a040_facc_b1_s0"},
}
LLAMA = {r: {bc: f"mlatL_e040_a040_{r}_{bc}_s0" for bc, _ in
             (("b0", 0), ("b0p5", 0.5), ("b1", 1))}
         for r in ("rep", "acc", "pri", "frep", "facc")}
LLAMA_B0 = {r: LLAMA[r]["b0"] for r in ("rep", "acc", "pri")}
# recipe -> (label, hue, linestyle); fresh variants share the hue, dashed
STYLE = {"rep":  ("replace", "#2f6f9f", "-"), "frep": ("fresh replace", "#2f6f9f", "--"),
         "acc":  ("accumulate", "#e08214", "-"), "facc": ("fresh accumulate", "#e08214", "--"),
         "pri":  ("pristine", "#1b7837", "-")}

def load(tag): return torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
def degrade(d):
    ppl = np.asarray(d["ppl_raw"], float)
    med = lambda r: np.nanmedian(np.where(np.isfinite(r), r, np.nan))
    return np.log10(med(ppl[-1]) / med(ppl[0]))
def final_dr(d):
    op = np.asarray(d["op_raw"], float); innate = np.asarray(d["innate"], float)
    return op[-1].std() / innate.std()

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})

# ---- Fig 3: recipe lines vs beta, Qwen | Llama ----
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True, constrained_layout=True)
for ax, (mname, tagset) in zip(axes, (("Qwen2.5-7B", QWEN), ("Llama-3.1-8B", LLAMA))):
    for rec, tags in tagset.items():
        lab, col, ls = STYLE[rec]
        ys = [degrade(load(tags[bc])) for bc, _ in BETAS]
        ax.plot([b for _, b in BETAS], ys, ls, marker="o", color=col, lw=2.2, ms=7, label=lab)
    ax.axhline(0, color="#bbbbbb", lw=1)
    ax.set_xticks([0, 0.5, 1]); ax.set_xlabel("anchor strength $\\beta$", fontsize=12)
    ax.set_title(mname, fontsize=12)
axes[0].set_ylabel("model degradation:  $\\log_{10}$(end ppl / round-0 ppl)", fontsize=12)
axes[0].legend(frameon=False, fontsize=10)
fig.suptitle("Data-regime advantage exists only unanchored: the anchor equalizes the continual\n"
             "recipes by $\\beta$=0.5; fresh replace is the unstable one (trap cell, seed 0)",
             fontsize=13)
fig.savefig(f"{OUT}/regime_vs_beta.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/regime_vs_beta.png")

# ---- Fig 4: unanchored bars, Qwen vs Llama ----
GROUPS = ["rep", "acc", "pri"]
GLAB = ["replace", "accumulate", "pristine"]
MODCOL = {"Qwen2.5-7B": "#e08214", "Llama-3.1-8B": "#2f6f9f"}
vals = {"Qwen2.5-7B": {r: (degrade(load(QWEN[r]["b0"])), final_dr(load(QWEN[r]["b0"]))) for r in GROUPS},
        "Llama-3.1-8B": {r: (degrade(load(LLAMA_B0[r])), final_dr(load(LLAMA_B0[r]))) for r in GROUPS}}
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
xg = np.arange(len(GROUPS)); w = 0.34
for i, (m, col) in enumerate(MODCOL.items()):
    axA.bar(xg + (i - 0.5) * w, [vals[m][r][0] for r in GROUPS], w * 0.92, color=col, label=m)
    axB.bar(xg + (i - 0.5) * w, [vals[m][r][1] for r in GROUPS], w * 0.92, color=col, label=m)
for ax, ylab in ((axA, "model degradation:  $\\log_{10}$(end ppl / round-0 ppl)"),
                 (axB, "final population diversity  $d_r$")):
    ax.set_xticks(xg); ax.set_xticklabels(GLAB, fontsize=11); ax.set_ylabel(ylab, fontsize=11)
axA.axhline(0, color="#bbbbbb", lw=1)
axB.axhline(0.22, color="#777777", lw=1.2, ls="--")
axB.text(-0.45, 0.23, "no-AI baseline", fontsize=8, color="#777777", va="bottom")
axA.legend(frameon=False, fontsize=10)
fig.suptitle("Unanchored MMHD ($\\beta$=0): replace > accumulate > pristine in model damage,\n"
             "and the data anchor buys back population diversity too (trap cell, seed 0)",
             fontsize=12)
fig.savefig(f"{OUT}/unanchored_regime_bars.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/unanchored_regime_bars.png")
