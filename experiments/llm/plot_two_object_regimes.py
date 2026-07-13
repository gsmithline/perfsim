"""Two-object plane by training regime (appendix to two_object_plane).
Same axes as the headline figure, one panel per model, one beta-arrow per
data recipe. Replace walks the L (model collapse -> population capture).
Accumulate/pristine never leave the healthy x range, so their arrows are
nearly vertical: with data hygiene the anchor buys nothing on the model
side and still displaces the population. Llama's anchored endpoint is
identical (D_pop 4.40) in all three recipes: the anchor erases regime.
Note D_pop is direction-agnostic: at beta 0 part of acc/pri "deformation"
is EXTRA diversity kept relative to the endogenously collapsing no-AI
world, not displacement.
"""
import importlib.util, os
import numpy as np, torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

spec = importlib.util.spec_from_file_location("rm", "experiments/real_mlp.py")
rm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rm)

RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs/qwen"
K_WORLDS = 20
# recipe -> (label, hue, linestyle, filled marker); fresh shares its
# continual counterpart's hue, drawn dashed with open markers
REGIMES = {"rep": ("replace", "#2f6f9f", "-", True),
           "frep": ("fresh replace", "#2f6f9f", "--", False),
           "acc": ("accumulate", "#e08214", "-", True),
           "facc": ("fresh accumulate", "#e08214", "--", False),
           "pri": ("pristine", "#1b7837", "-", True)}
TAGS = {
    ("Qwen2.5-7B", "rep"): ["mla2dv2_e040_a040_b0_s0", "mla2bv2_e040_a040_b0p5_s0", "mla2bv2_e040_a040_b1_s0"],
    ("Qwen2.5-7B", "acc"): ["mla2drv2_e040_a040_b0_acc_s0", "mlat_e040_a040_acc_b0p5_s0", "mlat_e040_a040_acc_b1_s0"],
    ("Qwen2.5-7B", "pri"): ["mla2drv2_e040_a040_b0_pri_s0", "mlat_e040_a040_pri_b0p5_s0", "mlat_e040_a040_pri_b1_s0"],
    ("Qwen2.5-7B", "frep"): ["mla2dfv2_e040_a040_b0_rep_s0", "mlat_e040_a040_frep_b0p5_s0", "mlat_e040_a040_frep_b1_s0"],
    ("Qwen2.5-7B", "facc"): ["mla2dfv2_e040_a040_b0_acc_s0", "mlat_e040_a040_facc_b0p5_s0", "mlat_e040_a040_facc_b1_s0"],
    ("Llama-3.1-8B", "rep"): [f"mlatL_e040_a040_rep_{b}_s0" for b in ("b0", "b0p5", "b1")],
    ("Llama-3.1-8B", "acc"): [f"mlatL_e040_a040_acc_{b}_s0" for b in ("b0", "b0p5", "b1")],
    ("Llama-3.1-8B", "pri"): [f"mlatL_e040_a040_pri_{b}_s0" for b in ("b0", "b0p5", "b1")],
    ("Llama-3.1-8B", "frep"): [f"mlatL_e040_a040_frep_{b}_s0" for b in ("b0", "b0p5", "b1")],
    ("Llama-3.1-8B", "facc"): [f"mlatL_e040_a040_facc_{b}_s0" for b in ("b0", "b0p5", "b1")],
}

ds = rm.ml_dataset("Action")
G, x_init = ds["G"], np.asarray(ds["x0"], float)
N = len(x_init)
worlds = []
for k in range(K_WORLDS):
    pop = rm.build_pop(G, 0.40, 0.0, x_init, k)
    for _ in range(30):
        pop.iteration(node_status=False)
    worlds.append(np.array([pop.status[i] for i in range(N)]))

def metrics(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.asarray(d["op_raw"], float); ppl = np.asarray(d["ppl_raw"], float)
    med = lambda r: np.nanmedian(np.where(np.isfinite(r), r, np.nan))
    x = np.log10(med(ppl[-1]) / med(ppl[0]))
    return x, float(np.mean([np.linalg.norm(op[-1] - w) / np.linalg.norm(w - w.mean()) for w in worlds]))

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})

fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), sharey=True, constrained_layout=True)
for ax, model in zip(axes, ("Qwen2.5-7B", "Llama-3.1-8B")):
    ax.axvline(1.0, color="#aaaaaa", lw=1, ls="--", zorder=0)
    ax.axhline(1.0, color="#aaaaaa", lw=1, ls="--", zorder=0)
    for rec, (lab, col, ls, filled) in REGIMES.items():
        pts = [metrics(t) for t in TAGS[(model, rec)]]
        xs, ys = zip(*pts)
        for i in range(2):
            if abs(xs[i] - xs[i + 1]) < 1e-9 and abs(ys[i] - ys[i + 1]) < 1e-9:
                continue
            ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, ls=ls,
                                        shrinkA=6, shrinkB=6))
        ax.plot(xs, ys, "o", color=col, ms=8, zorder=5, label=lab,
                mfc=col if filled else "white", mec=col)
        if filled:
            ax.annotate("$\\beta$=0", (xs[0], ys[0]), textcoords="offset points",
                        xytext=(8, -4), fontsize=8, color=col)
    ax.set_title(model, fontsize=12)
    ax.set_xlabel("model degradation:  $\\log_{10}$(end ppl / round-0 ppl)", fontsize=11)
    ax.set_xlim(-0.4, 2.5)
axes[0].set_ylabel("population deformation  $D_{pop}$", fontsize=12)
axes[0].legend(frameon=False, fontsize=10, loc="upper right", title="data recipe")
fig.suptitle("The trade only exists when the data pipeline is broken: replace walks the L;\n"
             "accumulate/pristine arrows are vertical (the anchor still displaces the population, "
             "the model was never sick)", fontsize=12)
fig.savefig(f"{OUT}/two_object_plane_regimes.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/two_object_plane_regimes.png")
