"""ML atlas slab overview figures, seed 0, one output dir per model arm:
  figs/<model>/grid_opinions_atlas.png : opinion histogram over rounds +
                            platform p10/50/90, rows = regimes, cols = cell x beta
  figs/<model>/grid_ppl_atlas.png      : per-agent perplexity at snapshot rounds
Models with no runs on disk are skipped. Panel style mirrors plot_grid_overview.py.
"""
import os
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

RUNS = "runs/pokec_gated_lm"
FIGS = "experiments/llm/figs"
MODELS = {
    "qwen": "mlat",
    "qwen-reset": "mlatR",
    "llama": "mlatL",
    "gemma": "mlatG",
    "mistral": "mlatM",
}
CELLS = ["e040_a010", "e040_a020", "e040_a040", "e020_a040"]
CELL_LABEL = {"e040_a010": "capture", "e040_a020": "mid",
              "e040_a040": "trap", "e020_a040": "smear"}
REGIMES = ["rep", "acc", "pri", "frep", "facc"]
BETAS = [("b0", "0"), ("b0p5", "0.5"), ("b1", "1")]
TAIL = 5
OPBINS = 50
SNAPS = [0, 4, 9, 19, 29]
PPLBINS = np.logspace(0, 4, 45)

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


def find_tag(model, cell, regime, beta):
    tag = f"{MODELS[model]}_{cell}_{regime}_{beta}_s0"
    if os.path.exists(f"{RUNS}/{tag}/trajectory.pt"):
        return tag
    if model == "qwen":
        leg = LEGACY.get((cell, regime, beta))
        if leg and os.path.exists(f"{RUNS}/{leg}/trajectory.pt"):
            return leg
    return None


def load(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)
    pr = np.clip(np.asarray(d["pred_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    ppl = np.clip(np.asarray(d["ppl_raw"], np.float32), 1.0, None)
    return op, pr, inn, ppl


def op_panel(ax, tag):
    op, pr, inn, _ = load(tag)
    R = op.shape[0]
    H = np.stack([np.histogram(op[t], bins=OPBINS, range=(0, 1))[0] for t in range(R)], axis=1)
    ax.imshow(H + 1, origin="lower", aspect="auto", extent=[0, R, 0, 1],
              cmap="Blues", norm=LogNorm(vmin=1, vmax=max(H.max(), 2)))
    q = np.percentile(pr, [10, 50, 90], axis=1)
    ax.plot(np.arange(R) + 0.5, q[1], color="#d95f02", lw=1.1)
    ax.plot(np.arange(R) + 0.5, q[0], color="#d95f02", lw=0.6, ls="--")
    ax.plot(np.arange(R) + 0.5, q[2], color="#d95f02", lw=0.6, ls="--")
    op_stdF = op[-TAIL:].std(1).mean()
    dr = op_stdF / (inn.std() + 1e-9)
    vr = pr[-TAIL:].std(1).mean() / (op_stdF + 1e-9)
    ax.set_title(f"dr={dr:.2f} vr={vr:.2f}", fontsize=7)
    ax.set_xlim(0, R); ax.set_ylim(0, 1)


def ppl_panel(ax, tag, show_legend=False):
    _, _, _, ppl = load(tag)
    colors = plt.cm.viridis(np.linspace(0.92, 0.05, len(SNAPS)))
    for t, c in zip(SNAPS, colors):
        ax.hist(ppl[t], bins=PPLBINS, density=True, histtype="stepfilled", alpha=0.12, color=c)
        ax.hist(ppl[t], bins=PPLBINS, density=True, histtype="step", color=c, lw=1.0,
                label=f"r{t + 1}" if show_legend else None)
    ax.set_xscale("log")
    ax.set_title(f"med={np.median(ppl[-TAIL:]):.1f}", fontsize=7)
    if show_legend:
        ax.legend(fontsize=6, frameon=False)


for model in MODELS:
    if not any(find_tag(model, c, r, b) for c in CELLS for r in REGIMES
               for b, _ in BETAS):
        continue
    os.makedirs(f"{FIGS}/{model}", exist_ok=True)
    for kind, panel in [("opinions", op_panel), ("ppl", ppl_panel)]:
        ncol = len(CELLS) * len(BETAS)
        fig, axes = plt.subplots(len(REGIMES), ncol, figsize=(2.1 * ncol, 10.5),
                                 sharex=True, sharey=(kind == "opinions"),
                                 constrained_layout=True)
        for ri, regime in enumerate(REGIMES):
            for ci, cell in enumerate(CELLS):
                for bi, (bc, blab) in enumerate(BETAS):
                    ax = axes[ri, ci * len(BETAS) + bi]
                    tag = find_tag(model, cell, regime, bc)
                    if tag is None:
                        ax.set_axis_off()
                        continue
                    if kind == "ppl":
                        panel(ax, tag, show_legend=(ri == 0 and ci == 0 and bi == 0))
                    else:
                        panel(ax, tag)
                    if ri == 0:
                        ax.set_title(f"{CELL_LABEL[cell]} $\\beta$={blab}\n" + ax.get_title(),
                                     fontsize=7)
                    if ci * len(BETAS) + bi == 0:
                        ax.set_ylabel(regime, fontsize=9)
                    if ri == len(REGIMES) - 1:
                        ax.set_xlabel("round" if kind == "opinions" else "ppl", fontsize=7)
                    ax.tick_params(labelsize=6)
        fig.suptitle(f"ML-Action atlas slab, {model} (seed 0): rows = training "
                     "regime, columns = cell x beta -- "
                     + ("opinion distribution/round + platform p10/p50/p90"
                        if kind == "opinions" else
                        "per-agent perplexity at rounds 1/5/10/20/30 (light to dark)"),
                     fontsize=11)
        out = f"{FIGS}/{model}/grid_{kind}_atlas.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"saved {out}")
