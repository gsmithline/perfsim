"""Distortion surface faithfulness(alpha*beta, eta) rendered from four
view/axis permutations to pick the clearest. Run:
    python experiments/recommender/run_phase_3dviews.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.scenarios.recommender import build_recommender_env, engagement_faithfulness
from perfsim.simulator import Simulator

OUT = Path("runs/recommender")
SEEDS = tuple(range(10))
PRODUCTS = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0)
ETAS = (0.0, 0.05, 0.15, 0.3)


def cell(prod: float, eta: float) -> float:
    vals = []
    for s in SEEDS:
        env = build_recommender_env(n_items=30, n_users=200, dim=8,
                                    alpha=prod / 4.0, beta=4.0, eta=eta, seed=s)
        model = LinearModel(8, 1)
        sim = Simulator(env=env, learner=ERMLearner(model, MSELoss()), loss=MSELoss())
        last = {"v": float("nan")}
        sim.run(n_rounds=20, epoch_size=8, seed=s,
                on_round=lambda t, r: last.update(v=engagement_faithfulness(env)))
        vals.append(last["v"])
    return sum(vals) / len(vals)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    Z = torch.tensor([[cell(p, e) for p in PRODUCTS] for e in ETAS])   # (eta, prod)

    # log-spaced product axis so the threshold isn't crushed at the origin
    px = torch.log2(torch.tensor([max(p, 1.0) for p in PRODUCTS]))
    ex = torch.tensor(ETAS)

    fig = plt.figure(figsize=(13, 10))
    views = [
        ("product front", (28, -60), False),
        ("eta front", (28, -150), False),
        ("high angle", (50, -75), False),
        ("product front, eta swapped to x", (28, -60), True),
    ]
    for k, (title, (elev, azim), swap) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, k, projection="3d")
        if swap:
            X, Y = torch.meshgrid(ex, px, indexing="xy")
            ax.plot_surface(X, Y, Z.t(), cmap="viridis", edgecolor="gray", lw=0.3)
            ax.set_xlabel("eta")
            ax.set_ylabel("log2 alpha*beta")
        else:
            X, Y = torch.meshgrid(px, ex, indexing="xy")
            ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="gray", lw=0.3)
            ax.set_xlabel("log2 alpha*beta")
            ax.set_ylabel("eta")
        ax.set_zlabel("faithfulness")
        ax.set_title(f"{title}  (elev={elev}, azim={azim})", fontsize=9)
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle("Distortion surface — view permutations (mean of 10 seeds)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "phase_3dviews.png", dpi=150)
    print(f"[recommender] figure -> {OUT / 'phase_3dviews.png'}")


if __name__ == "__main__":
    main()
