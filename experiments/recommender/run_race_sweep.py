"""Duel verdict vs taste drift, one line per user switching speed.
Run: python experiments/recommender/run_race_sweep.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from run_performative_push import FreezeAfter

from perfsim.competition import run_competition
from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.scenarios.recommender import (
    CompetingRecommendersWorld,
    sample_corpus,
    sample_user_population,
)

OUT = Path("runs/recommender")
SEEDS = (0, 1, 2)
ETAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.15)
MOBS = {0.25: "#3b6fd8", 1.0: "#3bb0a0", 4.0: "#d8633b"}


def duel_share(eta_mob: float, eta: float, seed: int) -> float:
    gen = torch.Generator()
    gen.manual_seed(seed)
    corpus = sample_corpus(30, 8, generator=gen)
    users = sample_user_population(200, 8, n_communities=4, generator=gen)
    idx = users.community % 2
    shares0 = torch.full((200, 2), 0.1)
    shares0[torch.arange(200), idx] = 0.9
    env = CompetingRecommendersWorld(
        corpus, users, n_platforms=2, alpha=4.0, beta=4.0, eta=eta,
        eta_mob=eta_mob, shares0=shares0,
    )
    frozen, retrain = LinearModel(8, 1), LinearModel(8, 1)
    learners = [FreezeAfter(frozen, MSELoss(), train_rounds=1),
                ERMLearner(retrain, MSELoss())]
    run_competition(env, learners, n_rounds=40, epoch_size=8, seed=seed)
    return float(env.shares[:, 1].mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    xs = range(len(ETAS))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for mob, color in MOBS.items():
        per_seed = [[duel_share(mob, eta, s) for s in SEEDS] for eta in ETAS]
        means = [sum(v) / len(v) for v in per_seed]
        ax.plot(xs, means, "o-", color=color, label=f"switching speed {mob:g}")
        for x, vals in zip(xs, per_seed):
            ax.scatter([x] * len(vals), vals, color=color, s=12, alpha=0.4)
        print(f"eta_mob={mob:g}: " + " ".join(f"{m:.3f}" for m in means))
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.text(0.05, 0.503, "parity", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{e:g}" for e in ETAS])
    ax.set_xlabel("eta (taste drift)")
    ax.set_ylabel("retrainer's final market share")
    ax.set_title("Drift rescues the self-corrupting retrainer;\n"
                 "user switching speed cannot move the threshold", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "race_sweep.png", dpi=140)
    print(f"[recommender] figure -> {OUT / 'race_sweep.png'}  (seeds {SEEDS})")


if __name__ == "__main__":
    main()
