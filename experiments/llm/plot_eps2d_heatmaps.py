"""eps x eps_AI heatmaps from the decoupled LLM grids (e2d_*, mla2d_*, ylp2d_*).
Rows = peer bound eps, cols = AI gate eps_AI; the diagonal (outlined) is the old
coupled setting. Per dataset: beta=0 vs beta=3, three readouts (last-5-round means):
  dr  = op_std / innate_std        diverging, midpoint 1 = innate preserved
  vr  = pred_std / op_std          diverging, midpoint 1 = platform tracks
  ppl = median perplexity          sequential, log scale
Norms are shared across the three datasets so the figures compare directly.
"""
import os
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
from matplotlib.patches import Rectangle

RUNS = "runs/pokec_gated_lm"
FIGS = "experiments/llm/figs"
DATASETS = [("e2d", "pokec", "Pokec (no usable feature, R2~0), pre-fix grid"),
            ("mla2dv2", "mlaction_v2", "MovieLens-Action (strong feature, R2=0.79), post-fix v2"),
            ("ylp2d", "yelp", "Yelp-Acme (weak feature, R2=0.07), pre-fix grid")]
EPS = [("e010", 0.1), ("e020", 0.2), ("e040", 0.4)]
EAI = [("a010", 0.1), ("a020", 0.2), ("a040", 0.4)]
BETA = [("b0", 0), ("b3", 3)]
TAIL = 5

NORMS = {"dr": TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=1.6),
         "vr": TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.2),
         "ppl": LogNorm(vmin=1.5, vmax=400)}
CMAPS = {"dr": "RdBu", "vr": "RdBu", "ppl": "Blues"}
TITLES = {"dr": "diversity retained  dr = op_std/innate_std",
          "vr": "platform spread  vr = pred_std/op_std",
          "ppl": "perplexity (median)"}
FMT = {"dr": "{:.2f}", "vr": "{:.2f}", "ppl": "{:.1f}"}


def stats(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)
    pr = np.clip(np.asarray(d["pred_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    ppl = np.asarray(d["ppl_raw"], np.float32)
    op_stdF = op[-TAIL:].std(1).mean()
    return {"dr": op_stdF / (inn.std() + 1e-9),
            "vr": pr[-TAIL:].std(1).mean() / (op_stdF + 1e-9),
            "ppl": float(np.median(ppl[-TAIL:]))}


def cell_ink(im, val):
    r, g, b, _ = im.cmap(im.norm(val))
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"


for pre, stem, label in DATASETS:
    grids = {(b, m): np.zeros((3, 3)) for b, _ in BETA for m in NORMS}
    for i, (ec, _) in enumerate(EPS):
        for j, (ac, _) in enumerate(EAI):
            for bc, _ in BETA:
                st = stats(f"{pre}_{ec}_{ac}_{bc}_s0")
                for m in NORMS:
                    grids[(bc, m)][i, j] = st[m]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True)
    for bi, (bc, bv) in enumerate(BETA):
        for mi, m in enumerate(NORMS):
            ax = axes[bi, mi]
            G = grids[(bc, m)]
            im = ax.imshow(G, cmap=CMAPS[m], norm=NORMS[m], origin="lower", aspect="equal")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, FMT[m].format(G[i, j]), ha="center", va="center",
                            fontsize=11, color=cell_ink(im, G[i, j]))
                ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="black", lw=1.6))
            ax.set_xticks(range(3), [f"{v:g}" for _, v in EAI])
            ax.set_yticks(range(3), [f"{v:g}" for _, v in EPS])
            ax.set_xlabel("AI gate  $\\epsilon_{AI}$  (usage)")
            if mi == 0:
                ax.set_ylabel(f"KL $\\beta$={bv}\n\npeer bound  $\\epsilon$")
            if bi == 0:
                ax.set_title(TITLES[m], fontsize=10)
            if bi == 1:
                cb = fig.colorbar(im, ax=axes[:, mi], shrink=0.75, pad=0.02)
                cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"{label} -- LLM loop (Qwen2.5-7B, replace, 30 rounds, seed 0). "
                 "Outlined diagonal = coupled $\\epsilon=\\epsilon_{AI}$ baseline.", fontsize=11)
    out = f"{FIGS}/eps2d_heatmaps_{stem}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")
