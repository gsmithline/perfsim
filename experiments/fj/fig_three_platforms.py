"""3D phase portrait: three data-driven strategic platforms flow to the
consensus diagonal.
Run: python experiments/fj/fig_three_platforms.py
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
TEMP = 0.05
LR = 0.01
ROUNDS = 1500
STARTS = [
    (0.30, 0.50, 0.70), (0.25, 0.35, 0.80), (0.20, 0.60, 0.75),
    (0.40, 0.45, 0.55), (0.25, 0.70, 0.78), (0.35, 0.55, 0.85),
    (0.22, 0.40, 0.62), (0.50, 0.65, 0.72),
]


def run3(b0: tuple[float, float, float], seed: int = 0) -> torch.Tensor:
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
    traj = []
    for _ in range(ROUNDS):
        probs = torch.softmax(-(b.unsqueeze(0) - x.unsqueeze(1)).abs() / TEMP, dim=1)
        assign = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        data = world.run(rh.FixedPredictions(b[assign]), n_steps=10)
        x = data["y"].reshape(-1)
        probs = torch.softmax(-(b.unsqueeze(0) - x.unsqueeze(1)).abs() / TEMP, dim=1)
        g = torch.zeros(3)
        for p in range(3):
            mask = assign == p
            if int(mask.sum()):
                g[p] = ((1 - probs[mask, p]) * -torch.sign(b[p] - x[mask]) / TEMP).sum() / n
        b = (b + LR * g).detach()
        traj.append([float(v) for v in b])
    return torch.tensor(traj)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run3.setup = rh.load_pokec()
    trajs = [run3(s) for s in STARTS]

    fig = plt.figure(figsize=(13.5, 6.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    cmap = plt.cm.viridis
    ax.plot([0.15, 0.9], [0.15, 0.9], [0.15, 0.9], ls="--", color="black",
            lw=1.2, alpha=0.7)
    ends = []
    for i, t in enumerate(trajs):
        c = cmap(i / (len(trajs) - 1))
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color=c, lw=1.4, alpha=0.9)
        ax.scatter(*t[0], facecolors="none", edgecolors=c, s=50)
        ends.append(t[-1])
    end = torch.stack(ends).mean(dim=0)
    ax.scatter(*end, marker="*", s=320, color="#d8633b", edgecolors="black",
               zorder=5)
    ax.text(float(end[0]), float(end[1]), float(end[2]) + 0.06,
            "homogeneous\nequilibrium", fontsize=9, ha="center")
    ax.text(0.84, 0.84, 0.95, "b1 = b2 = b3", fontsize=9, alpha=0.7)
    ax.set_xlabel("platform 1")
    ax.set_ylabel("platform 2")
    ax.set_zlabel("platform 3")
    ax.set_title("three strategic platforms, opinion drift on:\n"
                 "every start flows to the consensus diagonal")
    ax.view_init(elev=20, azim=38)

    ax2 = fig.add_subplot(1, 2, 2)
    for i, t in enumerate(trajs):
        gap = (t.unsqueeze(2) - t.unsqueeze(1)).abs().amax(dim=(1, 2))
        ax2.plot(gap, color=cmap(i / (len(trajs) - 1)), lw=1.4, alpha=0.9)
    ax2.set_xlabel("round")
    ax2.set_ylabel("max pairwise gap")
    ax2.set_title("differentiation decays from every start")
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.grid(alpha=0.25, lw=0.5)

    fig.suptitle("The homogeneous equilibrium with three data-driven strategic platforms "
                 f"(Pokec N=2163, own-customer gradients, lr={LR:g})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "three_platforms.png", dpi=150, bbox_inches="tight")
    print(f"mean end position: ({end[0]:.3f}, {end[1]:.3f}, {end[2]:.3f})")
    print(f"[fj] figure -> {OUT / 'three_platforms.png'}")


if __name__ == "__main__":
    main()
