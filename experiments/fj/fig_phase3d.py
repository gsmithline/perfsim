"""Phase diagram across choice-noise regimes: merge spiral vs partial merge
vs freeze, in platform space and the (gap, diversity) plane.
Run: python experiments/fj/fig_phase3d.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, "experiments/fj")
import run_hotelling as rh

OUT = Path("runs/fj_beach")
ROUNDS = 1200
LR = 0.01
TAUS = {0.10: ("#3b6fd8", "tau=0.10 (flippers): merge spiral"),
        0.02: ("#3bb0a0", "tau=0.02: partial merge"),
        0.01: ("#d8633b", "tau=0.01 (loyalists): frozen")}
STARTS = [(0.30, 0.50, 0.70), (0.25, 0.40, 0.80), (0.35, 0.60, 0.75)]


def run3(temp: float, b0, seed: int = 0):
    setup = run3.setup
    innate = setup["innate"]
    n = innate.shape[0]
    world = rh.FJWorld(innate, setup["W"], setup["peer_sus"],
                       platform_sus=setup["platform_sus"])
    world.reset(seed=seed)
    gen = torch.Generator()
    gen.manual_seed(seed + 1000)
    b = torch.tensor(b0)
    x = innate.clone()
    traj, stds = [], []
    for _ in range(ROUNDS):
        probs = torch.softmax(-(b.unsqueeze(0) - x.unsqueeze(1)).abs() / temp, dim=1)
        assign = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        data = world.run(rh.FixedPredictions(b[assign]), n_steps=10)
        x = data["y"].reshape(-1)
        probs = torch.softmax(-(b.unsqueeze(0) - x.unsqueeze(1)).abs() / temp, dim=1)
        g = torch.zeros(3)
        for p in range(3):
            mask = assign == p
            if int(mask.sum()):
                g[p] = ((1 - probs[mask, p]) * -torch.sign(b[p] - x[mask]) / temp).sum() / n
        b = (b + LR * g).detach()
        traj.append([float(v) for v in b])
        stds.append(float(x.std()))
    return torch.tensor(traj), torch.tensor(stds)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run3.setup = rh.load_pokec()

    fig = plt.figure(figsize=(14, 6.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    ax.plot([0.2, 0.85], [0.2, 0.85], [0.2, 0.85], ls="--", color="black",
            lw=1.2, alpha=0.7)
    ax.text(0.78, 0.78, 0.9, "consensus diagonal\nb1 = b2 = b3", fontsize=8, alpha=0.7)

    for temp, (color, label) in TAUS.items():
        first = True
        for b0 in STARTS:
            traj, stds = run3(temp, b0)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, lw=1.5,
                    alpha=0.9, label=label if first else None)
            ax.scatter(*traj[0], facecolors="none", edgecolors=color, s=45)
            marker = "*" if temp == 0.10 else "s"
            ax.scatter(*traj[-1], marker=marker, s=120 if marker == "*" else 45,
                       color=color, edgecolors="black", zorder=5)
            gap = traj.max(dim=1).values - traj.min(dim=1).values
            ax2.plot(gap, stds, color=color, lw=1.5, alpha=0.9,
                     label=label if first else None)
            ax2.scatter(gap[0], stds[0], facecolors="none", edgecolors=color, s=40)
            ax2.scatter(gap[-1], stds[-1], marker=marker,
                        s=110 if marker == "*" else 40, color=color,
                        edgecolors="black", zorder=5)
            first = False

    ax.set_xlabel("platform 1")
    ax.set_ylabel("platform 2")
    ax.set_zlabel("platform 3")
    ax.set_title("platform space: spiral to consensus vs freeze in place")
    ax.view_init(elev=20, azim=40)
    ax.legend(fontsize=8, loc="upper left")

    ax2.axhline(0.131, ls=":", color="black", lw=1.0, alpha=0.6)
    ax2.text(0.02, 0.134, "innate diversity", fontsize=8, alpha=0.7)
    ax2.set_xlabel("market differentiation (max platform gap)")
    ax2.set_ylabel("population diversity (opinion std)")
    ax2.set_title("joint dynamics: where the market goes, the people follow")
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.legend(fontsize=8)

    fig.suptitle("Phase diagram of platform-population dynamics by choice noise "
                 f"(Pokec N=2163, 3 platforms, own-data strategic, lr={LR:g})",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "phase3d.png", dpi=150, bbox_inches="tight")
    print(f"[fj] figure -> {OUT / 'phase3d.png'}")


if __name__ == "__main__":
    main()
