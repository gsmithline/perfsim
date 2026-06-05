"""Does RETRAINING push the world, beyond merely deploying a ranker?

Same world, same seed, three deployment policies:
  - learning:  ERM refit every round on the engagement it induced (the PP loop)
  - frozen:    one ERM fit on round-0 engagement, then deployed unchanged
  - untrained: random-init theta deployed unchanged (no information baseline)

All three bias exposure (alpha>0) and drift tastes (eta>0); only `learning`
closes the retraining feedback loop. The performative push of learning is the
gap between the learning and frozen trajectories.

Run from repo root:
    python experiments/recommender/run_performative_push.py
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
from perfsim.scenarios.recommender import (
    build_recommender_env,
    engagement_faithfulness,
    exposure_faithfulness,
    exposure_gini,
    interest_drift,
)
from perfsim.simulator import Simulator

OUT = Path("runs/recommender")

PROBES = {
    "corr(engagement, truth)": engagement_faithfulness,
    "corr(exposure, truth)": exposure_faithfulness,
    "exposure gini": exposure_gini,
    "interest drift": interest_drift,
}


class FreezeAfter(ERMLearner):
    """ERM for the first `train_rounds` fits, then a frozen deploy."""

    def __init__(self, model, loss, *, train_rounds: int) -> None:
        super().__init__(model, loss)
        self._budget = train_rounds

    def train(self, data) -> None:
        if self._budget > 0:
            self._budget -= 1
            super().train(data)


def trajectory(train_rounds: int | None, *, alpha: float = 4.0, beta: float = 8.0,
               eta: float = 0.15, seed: int = 0, n_rounds: int = 40,
               epoch_size: int = 8) -> dict[str, list[float]]:
    env = build_recommender_env(
        n_items=30, n_users=200, dim=8, alpha=alpha, beta=beta, eta=eta, seed=seed,
    )
    torch.manual_seed(seed)  # identical theta_0 across arms
    model = LinearModel(8, 1)
    loss = MSELoss()
    if train_rounds is None:
        learner: ERMLearner = ERMLearner(model, loss)
    else:
        learner = FreezeAfter(model, loss, train_rounds=train_rounds)
    sim = Simulator(env=env, learner=learner, loss=loss)
    out: dict[str, list[float]] = {name: [] for name in PROBES}

    def on_round(t: int, record: dict) -> None:
        for name, probe in PROBES.items():
            out[name].append(probe(env))

    sim.run(n_rounds=n_rounds, epoch_size=epoch_size, seed=seed, on_round=on_round)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arms = {
        "learning (refit every round)": (None, "#d8633b"),
        "frozen (one fit, then fixed)": (1, "#3b6fd8"),
        "untrained (random theta)": (0, "#888888"),
    }
    seeds = (0, 1, 2)

    runs = {name: [trajectory(tr, seed=s) for s in seeds] for name, (tr, _) in arms.items()}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, probe_name in zip(axes.flat, PROBES):
        for arm_name, (_, color) in arms.items():
            curves = torch.tensor([r[probe_name] for r in runs[arm_name]])
            mean = curves.mean(dim=0)
            ax.plot(range(len(mean)), mean, color=color, label=arm_name)
            ax.fill_between(range(len(mean)), curves.min(dim=0).values,
                            curves.max(dim=0).values, color=color, alpha=0.15)
        ax.set_title(probe_name)
        ax.set_xlabel("round")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Performative push of retraining vs deploying a fixed ranker",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "performative_push.png", dpi=140)

    print(f"{'probe':28s}  {'learning':>9s}  {'frozen':>9s}  {'untrained':>9s}  {'push':>7s}")
    for probe_name in PROBES:
        finals = {a: torch.tensor([r[probe_name][-1] for r in runs[a]]).mean().item()
                  for a in arms}
        learn, frozen, untrained = finals.values()
        print(f"{probe_name:28s}  {learn:9.3f}  {frozen:9.3f}  {untrained:9.3f}  "
              f"{learn - frozen:+7.3f}")
    print(f"[recommender] figure -> {OUT / 'performative_push.png'}  "
          f"(final round, mean over seeds {seeds}; push = learning - frozen)")


if __name__ == "__main__":
    main()
