"""Distribution-over-rounds waterfalls for the post-fix ML factorial (mla2d*v2).
Each panel: opinion histogram per round (x=round, y=opinion, log-count shading)
with the platform's p10/p50/p90 prediction bands overlaid, so population
collapse and platform dissociation are visible as shapes, not summary stats.
Figures: b0 factorial, b3 factorial (2x3 weights x data at e040_a040), and the
b3 dissociation cells (e040_a010 / e040_a020, replace/continual).
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
BINS = 60
TAIL = 5


def load(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)
    pr = np.clip(np.asarray(d["pred_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    return op, pr, inn


def panel(ax, tag, label):
    op, pr, inn = load(tag)
    R = op.shape[0]
    H = np.stack([np.histogram(op[t], bins=BINS, range=(0, 1))[0] for t in range(R)], axis=1)
    ax.imshow(H + 1, origin="lower", aspect="auto", extent=[0, R, 0, 1],
              cmap="Blues", norm=LogNorm(vmin=1, vmax=max(H.max(), 2)))
    q = np.percentile(pr, [10, 50, 90], axis=1)
    ax.plot(np.arange(R) + 0.5, q[1], color="#d95f02", lw=1.4)
    ax.plot(np.arange(R) + 0.5, q[0], color="#d95f02", lw=0.8, ls="--")
    ax.plot(np.arange(R) + 0.5, q[2], color="#d95f02", lw=0.8, ls="--")
    op_stdF = op[-TAIL:].std(1).mean()
    dr = op_stdF / (inn.std() + 1e-9)
    vr = pr[-TAIL:].std(1).mean() / (op_stdF + 1e-9)
    tru = np.sqrt(((pr[-TAIL:] - inn[None]) ** 2).mean())
    ax.set_title(f"{label}\ndr={dr:.2f}  vr={vr:.2f}  true={tru:.3f}", fontsize=9)
    ax.set_xlim(0, R); ax.set_ylim(0, 1)


ARMS = [("continual", "replace", "mla2dv2_e040_a040_{b}_s0"),
        ("continual", "accumulate", "mla2drv2_e040_a040_{b}_acc_s0"),
        ("continual", "pristine", "mla2drv2_e040_a040_{b}_pri_s0"),
        ("fresh", "replace", "mla2dfv2_e040_a040_{b}_rep_s0"),
        ("fresh", "accumulate", "mla2dfv2_e040_a040_{b}_acc_s0"),
        ("fresh", "pristine", "mla2dfv2_e040_a040_{b}_pri_s0")]

for b, bv in [("b0", 0), ("b3", 3)]:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True,
                             constrained_layout=True)
    for k, (w, dta, pat) in enumerate(ARMS):
        ax = axes[k // 3, k % 3]
        panel(ax, pat.format(b=b), f"{w} / {dta}")
        if k // 3 == 1: ax.set_xlabel("round")
        if k % 3 == 0: ax.set_ylabel("opinion")
    fig.suptitle(f"MovieLens-Action e040_a040, KL beta={bv} (post-fix). Blue = opinion "
                 "distribution per round; orange = platform prediction p10/p50/p90.", fontsize=11)
    out = f"{FIGS}/mlv2_waterfall_{b}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, constrained_layout=True)
for ax, (tag, label) in zip(axes, [
        ("mla2dv2_e040_a010_b3_s0", "gate 0.1: dissociation"),
        ("mla2dv2_e040_a020_b3_s0", "gate 0.2: dissociation"),
        ("mla2dv2_e040_a040_b3_s0", "gate 0.4: inflation + displacement")]):
    panel(ax, tag, label)
    ax.set_xlabel("round")
axes[0].set_ylabel("opinion")
fig.suptitle("MovieLens-Action beta=3, peer eps=0.4 (post-fix): tight gate -> population consensus "
             "under a spread-out platform; wide gate -> anchored platform floods the population.", fontsize=10)
out = f"{FIGS}/mlv2_dissociation.png"
fig.savefig(out, dpi=130); plt.close(fig)
print(f"saved {out}")
