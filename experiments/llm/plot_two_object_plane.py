"""Two-object failure plane (trap cell e040_a040, replace, seed 0).
x = model degradation, log10(end median ppl / round-0 median ppl).
y = population deformation D_pop = ||x_AI(T) - x_0(T)|| / ||C x_0(T)||,
    distance from the no-AI counterfactual population, normalized by that
    population's own centered dispersion. Counterfactual = matched no-AI
    world (same init, same Deffuant dynamics), not a paired pair-sequence.
    NOTE: NDlib's AlgorithmicBiasModel is NOT seed-deterministic (its
    iteration draws from an internal RNG that build_pop's reseeding does
    not control), so x_0(T) is a random draw. D_pop is therefore averaged
    over K counterfactual worlds and plotted with a whisker for the spread.
Each model draws beta {0, 0.5, 1} as connected points with arrows.
Alt x (appendix fig): excess prediction error RMSE(pred_T, innate) -
RMSE(pred_0, innate). It fails as a degradation axis: a rotting model
still emits plausible numbers, so point error stays flat while ppl
detonates. Llama's b0.5 and b1 coincide exactly (anchor saturates).
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
ROUNDS = 30
MODELS = [
    ("Qwen2.5-7B", "#e08214",
     ["mla2dv2_e040_a040_b0_s0", "mla2bv2_e040_a040_b0p5_s0", "mla2bv2_e040_a040_b1_s0"]),
    ("Llama-3.1-8B", "#2f6f9f",
     ["mlatL_e040_a040_rep_b0_s0", "mlatL_e040_a040_rep_b0p5_s0", "mlatL_e040_a040_rep_b1_s0"]),
    ("OLMo-2-7B", "#1b7837",
     ["olmo_e040_a040_rep_b0_s0", "olmo_e040_a040_rep_b0p5_s0", "olmo_e040_a040_rep_b1_s0"]),
]
BLAB = ["$\\beta$=0", "$\\beta$=0.5", "$\\beta$=1"]

K_WORLDS = 20

ds = rm.ml_dataset("Action")
G, x_init = ds["G"], np.asarray(ds["x0"], float)
N = len(x_init)
worlds = []
for k in range(K_WORLDS):
    pop = rm.build_pop(G, 0.40, 0.0, x_init, k)
    for _ in range(ROUNDS):
        pop.iteration(node_status=False)
    worlds.append(np.array([pop.status[i] for i in range(N)]))
print(f"{K_WORLDS} no-AI worlds: final std {np.mean([w.std() for w in worlds]):.4f} "
      f"+- {np.std([w.std() for w in worlds]):.4f}")

def metrics(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.asarray(d["op_raw"], float)
    pred = np.asarray(d["pred_raw"], float)
    ppl = np.asarray(d["ppl_raw"], float)
    innate = np.asarray(d["innate"], float)
    med = lambda row: np.nanmedian(np.where(np.isfinite(row), row, np.nan))
    x1 = np.log10(med(ppl[-1]) / med(ppl[0]))
    def rmse(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))
    x2 = rmse(pred[-1], innate) - rmse(pred[0], innate)
    ys = [np.linalg.norm(op[-1] - w) / np.linalg.norm(w - w.mean()) for w in worlds]
    return x1, x2, float(np.mean(ys)), float(np.std(ys))

pts = {name: [metrics(t) for t in tags] for name, _, tags in MODELS}
for name, _, tags in MODELS:
    for (x1, x2, y, ysd), tag in zip(pts[name], tags):
        print(f"{name:14s} {tag:28s} x1={x1:+.2f} x2={x2:+.3f} D_pop={y:.2f}+-{ysd:.2f}")

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "xtick.labelsize": 10, "ytick.labelsize": 10})

XTH, YTH = 1.0, 1.0  # 10x ppl growth; deformation = full counterfactual dispersion

def draw(ax, xi, xlabel, regions=True, names=True):
    for name, col, _ in MODELS:
        p = pts[name]
        xs = [q[xi] for q in p]; ys = [q[2] for q in p]
        ax.errorbar(xs, ys, yerr=[q[3] for q in p], fmt="none",
                    ecolor=col, elinewidth=1.2, capsize=3, alpha=0.6, zorder=4)
        for i in range(2):
            if xs[i] == xs[i + 1] and ys[i] == ys[i + 1]:
                continue
            ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                        shrinkA=6, shrinkB=6))
        ax.plot(xs, ys, "o", color=col, ms=8, zorder=5)
        for i, bl in enumerate(BLAB):
            if i == 2:
                ax.annotate(bl, (xs[i], ys[i]), textcoords="offset points",
                            xytext=(-10, 4), ha="right", fontsize=9, color=col)
            else:
                ax.annotate(bl, (xs[i], ys[i]), textcoords="offset points",
                            xytext=(8, 6), fontsize=9, color=col)
        if names:
            ax.annotate(name, (xs[0], ys[0]), textcoords="offset points",
                        xytext=(0, -16), ha="center", fontsize=10, color=col,
                        fontweight="bold")
    if regions:
        ax.axvline(XTH, color="#aaaaaa", lw=1, ls="--", zorder=0)
        ax.axhline(YTH, color="#aaaaaa", lw=1, ls="--", zorder=0)
        ax.add_patch(plt.Rectangle((-0.05, -0.15), XTH + 0.05, YTH + 0.15,
                                   color="#eaf4ea", zorder=-2))
        opts = dict(fontsize=10, color="#888888", style="italic", zorder=1)
        ax.text(0.03, 0.02, "both healthy", **opts)
        ax.text(0.98, 0.02, "model collapse", ha="right", **opts)
        ax.text(0.03, 0.96, "population capture", va="top", **opts)
        ax.text(0.98, 0.96, "joint failure", ha="right", va="top", **opts)
        for t in ax.texts[-4:]:
            t.set_transform(ax.transAxes)
    ax.set_xlabel(xlabel, fontsize=12)

fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
draw(ax, 0, "model degradation:  $\\log_{10}$(end ppl / round-0 ppl)")
ax.set_ylabel("population deformation  $D_{pop}=\\Vert x_{AI}-x_0\\Vert_2 \\, / \\, \\Vert C x_0\\Vert_2$",
              fontsize=12)
ax.set_xlim(-0.05, 2.5); ax.set_ylim(-0.15, 7.6)
ax.set_title("Anchoring stabilizes the model but shifts failure onto the population",
             fontsize=13)
fig.savefig(f"{OUT}/two_object_plane.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/two_object_plane.png")

# appendix: excess prediction error is not a usable degradation axis
fig, ax = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
draw(ax, 1, "excess prediction error:  RMSE(pred$_T$, innate) - RMSE(pred$_0$, innate)",
     regions=False, names=False)
ax.legend(handles=[Line2D([], [], color=c, marker="o", ls="-", label=n)
                   for n, c, _ in MODELS], frameon=False, fontsize=10, loc="upper left")
ax.set_ylabel("population deformation  $D_{pop}$", fontsize=12)
ax.set_title("Alt x axis fails: point error stays flat while ppl detonates\n"
             "(a rotting model still emits plausible numbers)", fontsize=12)
fig.savefig(f"{OUT}/two_object_plane_altx.png", dpi=140); plt.close(fig)
print(f"saved {OUT}/two_object_plane_altx.png")
