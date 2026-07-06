"""Whole-grid overview figures, one panel per (eps, eps_AI, beta) cell.
Per dataset two figures:
  grid_opinions_<ds>.png : opinion histogram over rounds (blue) + platform
                           p10/p50/p90 (orange), dr/vr in each title
  grid_ppl_<ds>.png      : per-agent perplexity histograms at snapshot rounds
                           (light=early, dark=late), final median in title
Layout: 6 rows (beta 0 block then beta 3 block, peer eps 0.1/0.2/0.4 within)
x 3 columns (gate eps_AI 0.1/0.2/0.4). ML = post-fix v2 grid; Pokec and Yelp =
their pre-fix grids (only versions with all 18 cells).
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
FIGS = "experiments/llm/figs/qwen"
EPS = [("e010", 0.1), ("e020", 0.2), ("e040", 0.4)]
EAI = [("a010", 0.1), ("a020", 0.2), ("a040", 0.4)]
BETA = [("b0", 0), ("b3", 3)]
TAIL = 5
OPBINS = 50
SNAPS = [0, 4, 9, 19, 29]
PPLBINS = np.logspace(0, 4, 45)

DATASETS = [("mlaction", "mla2dv2", "MovieLens-Action (strong feature) -- post-fix v2"),
            ("pokec", "e2d", "Pokec (no usable feature) -- pre-fix grid"),
            ("yelp", "ylp2d", "Yelp-Acme (weak feature) -- pre-fix grid")]


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
    ax.set_title(f"dr={dr:.2f}  vr={vr:.2f}", fontsize=8)
    ax.set_xlim(0, R); ax.set_ylim(0, 1)


def ppl_panel(ax, tag, show_legend=False):
    _, _, _, ppl = load(tag)
    colors = plt.cm.viridis(np.linspace(0.92, 0.05, len(SNAPS)))
    for t, c in zip(SNAPS, colors):
        ax.hist(ppl[t], bins=PPLBINS, density=True, histtype="stepfilled", alpha=0.12, color=c)
        ax.hist(ppl[t], bins=PPLBINS, density=True, histtype="step", color=c, lw=1.2,
                label=f"r{t + 1}" if show_legend else None)
    ax.set_xscale("log")
    ax.set_title(f"med={np.median(ppl[-TAIL:]):.1f}", fontsize=8)
    if show_legend:
        ax.legend(fontsize=7, frameon=False)


for kind, panel in [("opinions", op_panel), ("ppl", ppl_panel)]:
    for stem, g, label in DATASETS:
        fig, axes = plt.subplots(6, 3, figsize=(11, 17),
                                 sharex=True, sharey=(kind == "opinions"),
                                 constrained_layout=True)
        for bi, (bc, bv) in enumerate(BETA):
            for ei, (ec, ev) in enumerate(EPS):
                r = bi * 3 + ei
                for ai, (ac, av) in enumerate(EAI):
                    ax = axes[r, ai]
                    tag = f"{g}_{ec}_{ac}_{bc}_s0"
                    if kind == "ppl":
                        panel(ax, tag, show_legend=(r == 0 and ai == 0))
                    else:
                        panel(ax, tag)
                    ax.set_title(f"$\\beta$={bv} $\\epsilon$={ev} gate={av}   " + ax.get_title(),
                                 fontsize=8)
                    if r == 5:
                        ax.set_xlabel("round" if kind == "opinions" else "perplexity", fontsize=8)
                    if ai == 0:
                        ax.set_ylabel("opinion" if kind == "opinions" else "density", fontsize=8)
                    ax.tick_params(labelsize=7)
        fig.suptitle(f"{label}: all 18 cells, "
                     + ("opinion distribution/round + platform p10/p50/p90"
                        if kind == "opinions" else
                        "per-agent perplexity at rounds 1/5/10/20/30 (light to dark)"),
                     fontsize=11)
        out = f"{FIGS}/grid_{kind}_{stem}.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"saved {out}")
