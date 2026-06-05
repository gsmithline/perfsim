"""Both platforms retraining: welfare, felt/true inflation, and feed
concentration vs alpha*beta, by eta.
Run: python experiments/recommender/run_phase_market.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from perfsim.competition import run_competition
from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.scenarios.recommender import (
    CompetingRecommendersWorld,
    sample_corpus,
    sample_user_population,
    satisfaction_current,
    welfare_innate,
)

OUT = Path("runs/recommender")
SEEDS = (0, 1, 2)
PRODUCTS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
BETA = 4.0
ETAS = {0.0: "#3b6fd8", 0.01: "#3bb0a0", 0.05: "#b89b3b", 0.15: "#d8633b"}


def _gini(x: torch.Tensor) -> float:
    xs, _ = x.reshape(-1).clamp_min(0.0).sort()
    n = xs.numel()
    idx = torch.arange(1, n + 1, dtype=xs.dtype)
    return float((2.0 * (idx * xs).sum()) / (n * xs.sum()) - (n + 1) / n)


def market(product: float, eta: float, seed: int) -> dict[str, float]:
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
    learners = [ERMLearner(LinearModel(8, 1), MSELoss()) for _ in range(2)]
    run_competition(env, learners, n_rounds=40, epoch_size=8, seed=seed)
    felt, true = satisfaction_current(env), welfare_innate(env)
    return {
        "welf": true,
        "infl": felt / true if true > 1e-9 else float("nan"),
        "gini": (_gini(env.last_exposure[0]) + _gini(env.last_exposure[1])) / 2,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    xs = range(len(PRODUCTS))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    panels = (("welf", "true welfare (vs innate tastes)"),
              ("infl", "inflation: felt / true"),
              ("gini", "feed concentration (gini)"))
    for eta, color in ETAS.items():
        per_seed = {k: [] for k, _ in panels}
        for p in PRODUCTS:
            runs = [market(p, eta, s) for s in SEEDS]
            for k, _ in panels:
                per_seed[k].append([r[k] for r in runs])
        for ax, (k, _) in zip(axes, panels):
            means = [sum(v) / len(v) for v in per_seed[k]]
            ax.plot(xs, means, "o-", color=color, label=f"eta={eta:g}")
            for x, vals in zip(xs, per_seed[k]):
                ax.scatter([x] * len(vals), vals, color=color, s=10, alpha=0.4)
        print(f"eta={eta:g} welf: " + " ".join(f"{sum(v)/len(v):.2f}" for v in per_seed["welf"]))
    for ax, (k, title) in zip(axes, panels):
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{p:g}" for p in PRODUCTS], fontsize=8)
        ax.set_xlabel("alpha * beta")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("Both platforms retraining: what users get vs manipulation strength",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "phase_market.png", dpi=140)
    print(f"[recommender] figure -> {OUT / 'phase_market.png'}  (beta={BETA:g}, seeds {SEEDS})")


if __name__ == "__main__":
    main()
