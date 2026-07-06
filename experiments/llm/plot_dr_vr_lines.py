"""dr(t) and vr(t) trajectory lines for the eps x eps_AI grids, per dataset.
Layout: rows = beta, cols = peer eps, lines = gate eps_AI (light -> dark, wider
gate = darker). Reference line at 1.0 (dr: innate preserved / vr: tracking).
ML = post-fix v2 grid; Pokec and Yelp = pre-fix grids (only full-grid versions).
The trajectory companion to the 18-cell waterfall grids: shows speed, plateaus,
and divergence that endpoint heatmaps hide.
"""
import os
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/pokec_gated_lm"
FIGS = "experiments/llm/figs"
EPS = [("e010", 0.1), ("e020", 0.2), ("e040", 0.4)]
EAI = [("a010", 0.1), ("a020", 0.2), ("a040", 0.4)]
BETA = [("b0", 0), ("b3", 3)]
GATE_COLORS = plt.cm.Blues([0.45, 0.7, 0.95])


def curves(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)
    pr = np.clip(np.asarray(d["pred_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    dr = op.std(1) / (inn.std() + 1e-9)
    vr = pr.std(1) / (op.std(1) + 1e-9)
    return dr, vr


DATASETS = [("mla2dv2", "mlaction", "MovieLens-Action (post-fix v2)"),
            ("e2d", "pokec", "Pokec (pre-fix grid)"),
            ("ylp2d", "yelp", "Yelp-Acme (pre-fix grid)")]

for g, stem, label in DATASETS:
    for metric, ylab, yscale in [("dr", "dr = op_std / innate_std", "linear"),
                                 ("vr", "vr = pred_std / op_std", "log")]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True,
                                 constrained_layout=True)
        for bi, (bc, bv) in enumerate(BETA):
            for ei, (ec, ev) in enumerate(EPS):
                ax = axes[bi, ei]
                for (ac, av), c in zip(EAI, GATE_COLORS):
                    dr, vr = curves(f"{g}_{ec}_{ac}_{bc}_s0")
                    y = dr if metric == "dr" else vr
                    ax.plot(np.arange(1, len(y) + 1), y, color=c, lw=2,
                            label=f"$\\epsilon_{{AI}}$={av}")
                ax.axhline(1.0, color="gray", lw=1, ls="--")
                ax.set_title(f"$\\beta$={bv}   peer $\\epsilon$={ev}", fontsize=10)
                ax.set_yscale(yscale)
                if bi == 1: ax.set_xlabel("round")
                if ei == 0: ax.set_ylabel(ylab)
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=9, frameon=False)
        fig.suptitle(f"{label}, replace/continual: "
                     f"{'population diversity' if metric == 'dr' else 'platform-to-population spread'} "
                     "over rounds. Lines = AI gate width (darker = wider).", fontsize=11)
        out = f"{FIGS}/lines_{metric}_{stem}.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"saved {out}")
