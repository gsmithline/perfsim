"""Duel verdict vs manipulation strength alpha*beta (a 1D curve: alpha and
beta enter only via their product; see choice.py), by eta.
Run: python experiments/recommender/run_phase.py
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
PRODUCTS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
BETA = 4.0
ETAS = {0.0: "#3b6fd8", 0.01: "#3bb0a0", 0.05: "#b89b3b", 0.15: "#d8633b"}


def duel_share(product: float, eta: float, seed: int) -> float:
    gen = torch.Generator()
    gen.manual_seed(seed)
    corpus = sample_corpus(30, 8, generator=gen)
    users = sample_user_population(200, 8, n_communities=4, generator=gen)
    idx = users.community % 2
    shares0 = torch.full((200, 2), 0.1)
    shares0[torch.arange(200), idx] = 0.9
    env = CompetingRecommendersWorld(
        corpus, users, n_platforms=2, alpha=product / BETA, beta=BETA, eta=eta,
        eta_mob=0.25, shares0=shares0,
    )
    frozen, retrain = LinearModel(8, 1), LinearModel(8, 1)
    learners = [FreezeAfter(frozen, MSELoss(), train_rounds=1),
                ERMLearner(retrain, MSELoss())]
    run_competition(env, learners, n_rounds=40, epoch_size=8, seed=seed)
    return float(env.shares[:, 1].mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    xs = range(len(PRODUCTS))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for eta, color in ETAS.items():
        per_seed = [[duel_share(p, eta, s) for s in SEEDS] for p in PRODUCTS]
        means = [sum(v) / len(v) for v in per_seed]
        ax.plot(xs, means, "o-", color=color, label=f"eta={eta:g}")
        for x, vals in zip(xs, per_seed):
            ax.scatter([x] * len(vals), vals, color=color, s=12, alpha=0.4)
        print(f"eta={eta:g}: " + " ".join(f"{m:.3f}" for m in means))
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.text(0.1, 0.503, "parity", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{p:g}" for p in PRODUCTS])
    ax.set_xlabel("alpha * beta (manipulation strength)")
    ax.set_ylabel("retrainer's final market share")
    ax.set_title("Markets punish moderate manipulation hardest;\n"
                 "extreme manipulation and any drift both escape", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "phase_product.png", dpi=140)
    print(f"[recommender] figure -> {OUT / 'phase_product.png'}  (beta={BETA:g}, seeds {SEEDS})")


if __name__ == "__main__":
    main()
