"""Sweep the two performativity channels: alpha (labels) x eta (population);
push = learning - frozen, per run_performative_push.
Run: python experiments/recommender/run_push_sweep.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from run_performative_push import PROBES, trajectory

OUT = Path("runs/recommender")

ALPHAS = (0.0, 1.0, 2.0, 4.0, 8.0)
ETAS = {0.0: ("#3b6fd8", "eta=0 (no taste drift)"), 0.15: ("#d8633b", "eta=0.15")}
SEEDS = (0, 1, 2)


def push(alpha: float, eta: float, seed: int) -> dict[str, float]:
    learn = trajectory(None, alpha=alpha, eta=eta, seed=seed)
    frozen = trajectory(1, alpha=alpha, eta=eta, seed=seed)
    return {name: learn[name][-1] - frozen[name][-1] for name in PROBES}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = {(a, e): [push(a, e, s) for s in SEEDS] for e in ETAS for a in ALPHAS}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, probe_name in zip(axes.flat, PROBES):
        for eta, (color, label) in ETAS.items():
            means = [torch.tensor([p[probe_name] for p in grid[(a, eta)]]).mean()
                     for a in ALPHAS]
            ax.plot(ALPHAS, means, "o-", color=color, label=label)
            for a in ALPHAS:
                pts = [p[probe_name] for p in grid[(a, eta)]]
                ax.scatter([a] * len(pts), pts, color=color, s=10, alpha=0.4)
        ax.axhline(0.0, color="black", lw=0.7)
        ax.set_title(f"push in {probe_name}")
        ax.set_xlabel("alpha (exposure bias)")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Push of retraining (learning - frozen, final round) by performativity channel",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "push_sweep.png", dpi=140)

    header = f"{'eta':>5s} {'alpha':>6s}" + "".join(f"  {n[:24]:>24s}" for n in PROBES)
    print(header)
    for (a, e), runs in grid.items():
        row = f"{e:5.2f} {a:6.1f}"
        for name in PROBES:
            row += f"  {torch.tensor([p[name] for p in runs]).mean():24.3f}"
        print(row)
    print(f"[recommender] figure -> {OUT / 'push_sweep.png'}  (mean over seeds {SEEDS})")


if __name__ == "__main__":
    main()
