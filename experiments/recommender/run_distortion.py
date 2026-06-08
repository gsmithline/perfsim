"""The honest headline: exposure bias erodes the faithfulness of the platform's
own data to true relevance.

Compares alpha=0 (no exposure bias, baseline) against alpha>0 over the PP loop,
tracking corr(induced engagement, true relevance). eta=0 isolates the label
(exposure) channel from taste drift.

Run from repo root:
    python experiments/recommender/run_distortion.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.scenarios.recommender import (
    build_recommender_env,
    engagement_faithfulness,
    exposure_faithfulness,
)
from perfsim.simulator import Simulator

OUT = Path("runs/recommender")


def trajectory(alpha: float, *, beta: float = 8.0, eta: float = 0.0, seed: int = 0,
               n_rounds: int = 20, epoch_size: int = 8) -> tuple[list[float], list[float]]:
    env = build_recommender_env(
        n_items=30, n_users=200, dim=8, alpha=alpha, beta=beta, eta=eta, seed=seed,
    )
    model = LinearModel(8, 1)
    loss = MSELoss()
    sim = Simulator(env=env, learner=ERMLearner(model, loss), loss=loss)
    yq: list[float] = []
    uq: list[float] = []

    def on_round(t: int, record: dict) -> None:
        yq.append(engagement_faithfulness(env))
        uq.append(exposure_faithfulness(env))

    sim.run(n_rounds=n_rounds, epoch_size=epoch_size, seed=seed, on_round=on_round)
    return yq, uq


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    alphas = {0.0: "#888888", 4.0: "#d8633b", 8.0: "#8d3bd8"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for alpha, color in alphas.items():
        yq, uq = trajectory(alpha)
        rounds = range(len(yq))
        label = "alpha=0 (no bias, baseline)" if alpha == 0 else f"alpha={alpha:g}"
        ax[0].plot(rounds, yq, color=color, label=label)
        ax[1].plot(rounds, uq, color=color, label=label)
        print(f"alpha={alpha:4.1f}  corr(engagement,truth): {yq[0]:+.3f} -> {yq[-1]:+.3f}   "
              f"corr(exposure,truth): {uq[0]:+.3f} -> {uq[-1]:+.3f}")

    ax[0].set_title("Data faithfulness erodes under exposure bias")
    ax[0].set_xlabel("round"); ax[0].set_ylabel("corr(induced engagement, true relevance)")
    ax[0].set_ylim(0.0, 0.5); ax[0].legend()

    ax[1].set_title("Exposure misallocation under bias")
    ax[1].set_xlabel("round"); ax[1].set_ylabel("corr(exposure, true relevance)")
    ax[1].set_ylim(0.0, 0.5); ax[1].legend()

    fig.suptitle("Performative recommender: acting on predictions degrades the data's link to truth",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "distortion.png", dpi=140)
    print(f"[recommender] figure -> {OUT / 'distortion.png'}")


if __name__ == "__main__":
    main()