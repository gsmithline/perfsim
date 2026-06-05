"""Platform paths over the population density terrain, one panel per
choice-noise regime.
Run: python experiments/fj/fig_platform_paths.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, "experiments/fj")
import run_hotelling as rh

OUT = Path("runs/fj_beach")
ROUNDS = 500
LR = 0.01
BINS = 90
START = (0.30, 0.50, 0.70)
REGIMES = ((0.10, "flippers (tau = 0.10): platforms and population merge"),
           (0.02, "tau = 0.02: partial merge"),
           (0.01, "loyalists (tau = 0.01): platforms freeze, population condenses onto them"))
PCOLORS = ("#d8633b", "#3bb0a0", "#3b6fd8")


def run3(temp: float, seed: int = 0):
    setup = run3.setup
    innate = setup["innate"]
    n = innate.shape[0]
    world = rh.FJWorld(innate, setup["W"], setup["peer_sus"],
                       platform_sus=setup["platform_sus"])
    world.reset(seed=seed)
    gen = torch.Generator()
    gen.manual_seed(seed + 1000)
    b = torch.tensor(START)
    x = innate.clone()
    traj, beach = [], []
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
        beach.append(torch.histc(x, bins=BINS, min=0.0, max=1.0))
    return torch.tensor(traj), torch.stack(beach)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run3.setup = rh.load_pokec()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True,
                             constrained_layout=True)
    im = None
    for ax, (temp, title) in zip(axes, REGIMES):
        traj, beach = run3(temp)
        im = ax.imshow(beach.t().sqrt(), aspect="auto", origin="lower",
                       cmap="Blues", extent=[0, ROUNDS, 0, 1], vmin=0,
                       vmax=0.55 * float(beach.sqrt().max()))
        kernel = torch.ones(11) / 11.0
        for p in range(3):
            sm = torch.nn.functional.conv1d(
                traj[:, p].view(1, 1, -1), kernel.view(1, 1, -1), padding=5
            ).reshape(-1)
            sm[:5] = traj[:5, p]
            sm[-5:] = traj[-5:, p]
            line, = ax.plot(range(ROUNDS), sm, color=PCOLORS[p], lw=2.4,
                            label=f"platform {p + 1}")
            line.set_path_effects([pe.Stroke(linewidth=3.6, foreground="white"),
                                   pe.Normal()])
            ax.scatter(0, traj[0, p], color=PCOLORS[p], edgecolors="white",
                       s=55, zorder=5)
            ax.scatter(ROUNDS - 1, traj[-1, p], color=PCOLORS[p],
                       edgecolors="black", s=70, zorder=5, marker="D")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("round")
        ax.set_ylim(0.15, 0.85)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("opinion space")
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cbar.set_label("population density (sqrt scale)", fontsize=9)

    fig.suptitle("Where the platforms go and where the people pile up, by choice noise\n"
                 "(Pokec N = 2163, three platforms, own-customer strategic gradients)",
                 fontweight="bold", fontsize=13)
    fig.savefig(OUT / "platform_paths.png", dpi=150)
    print(f"[fj] figure -> {OUT / 'platform_paths.png'}")


if __name__ == "__main__":
    main()
