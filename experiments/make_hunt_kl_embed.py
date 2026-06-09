"""Each model as a point. Take each model's per-agent prediction vector (R^N)
each round, embed all of them (all betas, rounds, models) into a shared 2D PCA
space, and draw one path per model per beta. Convergence of the 3 paths = the
models merge; separated endpoints = they stay distinct.
Run: python experiments/make_hunt_kl_embed.py
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
MODEL_COLORS = ["#d8633b", "#3bb0a0", "#8d3bd8"]
MODEL_NAMES = ["Qwen", "SmolLM2", "TinyLlama"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preds = {}
    rows = []
    for b in BETAS:
        tj = ROOT / f"hunt_kl{b}_t05_s0" / "trajectory.pt"
        if not tj.exists():
            continue
        p = torch.load(tj, map_location="cpu", weights_only=False)["preds_raw"].float()
        preds[b] = p                       # (T, 3, N)
        rows.append(p.reshape(-1, p.shape[-1]))   # (T*3, N)
    X = torch.cat(rows, dim=0)
    Xc = X - X.mean(dim=0, keepdim=True)
    U, S, V = torch.linalg.svd(Xc, full_matrices=False)
    comp = V[:2]                            # (2, N)

    def to2d(vec):
        return ((vec - X.mean(dim=0)) @ comp.t()).tolist()

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    for ax, b in zip(axes.flat, BETAS):
        if b not in preds:
            ax.set_title(f"beta={b}: missing"); continue
        P = preds[b]; T = P.shape[0]
        for m in range(3):
            xy = [to2d(P[t, m]) for t in range(T)]
            xs = [c[0] for c in xy]; ys = [c[1] for c in xy]
            ax.plot(xs, ys, color=MODEL_COLORS[m], lw=1.5,
                    label=MODEL_NAMES[m] if b == BETAS[0] else None)
            ax.scatter(xs[0], ys[0], color=MODEL_COLORS[m], marker="o", s=40, zorder=3)
            ax.scatter(xs[-1], ys[-1], color=MODEL_COLORS[m], marker="*", s=200, zorder=4)
        ax.set_title(f"beta={b}  (o=start, *=end)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes.flat[0].legend(fontsize=8)
    var = (S[:2] ** 2 / (S ** 2).sum()).tolist()
    fig.suptitle(f"Three models as points (per-agent pred vectors, shared PCA; "
                 f"PC1+PC2={100*(var[0]+var[1]):.0f}% var)", fontweight="bold")
    fig.supxlabel("PC1"); fig.supylabel("PC2")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "model_embed.png", dpi=150)
    print(f"[figs] -> {OUT / 'model_embed.png'}  PC var {[round(v,3) for v in var]}")


if __name__ == "__main__":
    main()
