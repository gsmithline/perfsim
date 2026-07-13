"""Two-object failure plane at the LIVE-baseline cell (social eps=0.10,
no-AI counterfactual keeps dr ~0.70). Same axes and layout as
plot_two_object_plane.py (trap cell). Attribution is clean here: the
baseline population does not collapse on its own, so beta-0 contraction
and beta-1 displacement are both the AI's doing. Note D_pop values are
smaller than at eps=0.40 because the live baseline has ~3x the dispersion
in the denominator, not because the AI does less.
"""
import importlib.util, os
import numpy as np, torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("rm", "experiments/real_mlp.py")
rm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rm)

RUNS = "runs/pokec_gated_lm"
OUT = "experiments/llm/figs/qwen"
K_WORLDS = 20
MODELS = [
    ("Qwen2.5-7B", "#e08214", [f"mlat_e010_a040_rep_{b}_s0" for b in ("b0", "b0p5", "b1")]),
    ("Llama-3.1-8B", "#2f6f9f", [f"mlatL_e010_a040_rep_{b}_s0" for b in ("b0", "b0p5", "b1")]),
    ("OLMo-2-7B", "#1b7837", [f"olmo_e010_a040_rep_{b}_s0" for b in ("b0", "b0p5", "b1")]),
]
BLAB = ["$\\beta$=0", "$\\beta$=0.5", "$\\beta$=1"]

ds = rm.ml_dataset("Action")
G, x_init = ds["G"], np.asarray(ds["x0"], float)
N = len(x_init)
worlds = []
for k in range(K_WORLDS):
    pop = rm.build_pop(G, 0.10, 0.0, x_init, k)
    for _ in range(30):
        pop.iteration(node_status=False)
    worlds.append(np.array([pop.status[i] for i in range(N)]))

def metrics(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.asarray(d["op_raw"], float); ppl = np.asarray(d["ppl_raw"], float)
    med = lambda r: np.nanmedian(np.where(np.isfinite(r), r, np.nan))
    x = np.log10(med(ppl[-1]) / med(ppl[0]))
    ys = [np.linalg.norm(op[-1] - w) / np.linalg.norm(w - w.mean()) for w in worlds]
    return x, float(np.mean(ys)), float(np.std(ys))

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})

XTH, YTH = 1.0, 1.0
fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
ax.axvline(XTH, color="#aaaaaa", lw=1, ls="--", zorder=0)
ax.axhline(YTH, color="#aaaaaa", lw=1, ls="--", zorder=0)
ax.add_patch(plt.Rectangle((-0.05, -0.1), XTH + 0.05, YTH + 0.1, color="#eaf4ea", zorder=-2))
for name, col, tags in MODELS:
    pts = [metrics(t) for t in tags]
    xs, ys, sds = zip(*pts)
    ax.errorbar(xs, ys, yerr=sds, fmt="none", ecolor=col, elinewidth=1.2,
                capsize=3, alpha=0.6, zorder=4)
    for i in range(2):
        if abs(xs[i] - xs[i + 1]) < 1e-9 and abs(ys[i] - ys[i + 1]) < 1e-9:
            continue
        ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, shrinkA=6, shrinkB=6))
    ax.plot(xs, ys, "o", color=col, ms=8, zorder=5)
    for i, bl in enumerate(BLAB):
        off = (-10, 4) if i == 2 else (8, 6)
        ha = "right" if i == 2 else "left"
        ax.annotate(bl, (xs[i], ys[i]), textcoords="offset points", xytext=off,
                    ha=ha, fontsize=9, color=col)
    nx, nha = {"Qwen2.5-7B": ((-10, -16), "right")}.get(name, ((10, -16), "left"))
    ax.annotate(name, (xs[0], ys[0]), textcoords="offset points",
                xytext=nx, ha=nha, fontsize=10, color=col, fontweight="bold")
opts = dict(fontsize=10, color="#888888", style="italic", transform=ax.transAxes)
ax.text(0.03, 0.02, "both healthy", **opts)
ax.text(0.98, 0.02, "model collapse", ha="right", **opts)
ax.text(0.03, 0.96, "population capture", va="top", **opts)
ax.text(0.98, 0.96, "joint failure", ha="right", va="top", **opts)
ax.set_xlabel("model degradation:  $\\log_{10}$(end ppl / round-0 ppl)", fontsize=12)
ax.set_ylabel("population deformation  $D_{pop}=\\Vert x_{AI}-x_0\\Vert_2 \\, / \\, \\Vert C x_0\\Vert_2$",
              fontsize=12)
ax.set_xlim(-0.05, 2.3); ax.set_ylim(-0.1, 2.4)
ax.set_title("Live-baseline cell (social $\\epsilon$=0.10): the same walk with clean attribution",
             fontsize=13)
fig.savefig(f"{OUT}/two_object_plane_e010.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/two_object_plane_e010.png")
