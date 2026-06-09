"""Each model as a point on the opinion simplex. Bin each model's per-agent
predictions into 3 bins (low/mid/high) -> a histogram = barycentric point on
the 2-simplex. One triangle per beta; three model trajectories (o=start,
*=end). beta=0/1/3 should collapse the 3 models to one point; beta=10 should
keep them at distinct points.
Run: python experiments/make_hunt_kl_simplex.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path("runs/pokec_fj_hunt_kl")
OUT = ROOT / "figs"
BETAS = [0, 1, 3, 10]
EDGES = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0001])
MODEL_COLORS = ["#d8633b", "#3bb0a0", "#8d3bd8"]
MODEL_NAMES = ["Qwen", "SmolLM2", "TinyLlama"]
# triangle vertices: low=(0,0), mid=(1,0), high=(0.5, h)
H = 3 ** 0.5 / 2


def bary(hist):
    a, b, c = hist
    return b * 1.0 + c * 0.5, c * H        # x, y


def hist3(v):
    counts = torch.histogram(v, bins=EDGES).hist
    return (counts / counts.sum()).tolist()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, beta in zip(axes.flat, BETAS):
        tj = ROOT / f"hunt_kl{beta}_t05_s0" / "trajectory.pt"
        if not tj.exists():
            ax.set_title(f"beta={beta}: missing"); continue
        preds = torch.load(tj, map_location="cpu", weights_only=False)["preds_raw"].float()
        T, P, _ = preds.shape
        # triangle frame
        tri = [(0, 0), (1, 0), (0.5, H), (0, 0)]
        ax.plot([p[0] for p in tri], [p[1] for p in tri], "k-", lw=1, alpha=0.5)
        ax.text(-0.04, -0.04, "low", fontsize=8, ha="right")
        ax.text(1.04, -0.04, "mid", fontsize=8, ha="left")
        ax.text(0.5, H + 0.03, "high", fontsize=8, ha="center")
        for p in range(P):
            xs, ys = [], []
            for t in range(T):
                x, y = bary(hist3(preds[t, p]))
                xs.append(x); ys.append(y)
            ax.plot(xs, ys, color=MODEL_COLORS[p], lw=1.5,
                    label=MODEL_NAMES[p] if beta == BETAS[0] else None)
            ax.scatter(xs[0], ys[0], color=MODEL_COLORS[p], marker="o", s=35, zorder=3)
            ax.scatter(xs[-1], ys[-1], color=MODEL_COLORS[p], marker="*", s=160, zorder=4)
        ax.set_title(f"beta={beta}  (o=start, *=end)")
        ax.set_aspect("equal"); ax.axis("off")
    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Each model on the opinion simplex (low/mid/high pred histogram), by KL beta",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "simplex.png", dpi=150)
    print(f"[figs] -> {OUT / 'simplex.png'}")


if __name__ == "__main__":
    main()
