"""Three market arms: both frozen, one retraining, both retraining, at
eta 0 and 0.15.
Run: python experiments/recommender/run_arms.py
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
    satisfaction_current,
    welfare_innate,
)

OUT = Path("runs/recommender")
SEEDS = (0, 1, 2, 3, 4)
ARMS = {
    "both frozen": ("frozen", "frozen"),
    "one retrains": ("frozen", "learn"),
    "both retrain": ("learn", "learn"),
}
COLORS = {"both frozen": "#3b6fd8", "one retrains": "#3bb0a0", "both retrain": "#d8633b"}


def _gini(x: torch.Tensor) -> float:
    xs, _ = x.reshape(-1).clamp_min(0.0).sort()
    n = xs.numel()
    idx = torch.arange(1, n + 1, dtype=xs.dtype)
    return float((2.0 * (idx * xs).sum()) / (n * xs.sum()) - (n + 1) / n)


def run_arm(kinds: tuple[str, str], eta: float, seed: int) -> dict[str, float]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    corpus = sample_corpus(30, 8, generator=gen)
    users = sample_user_population(200, 8, n_communities=4, generator=gen)
    idx = users.community % 2
    shares0 = torch.full((200, 2), 0.1)
    shares0[torch.arange(200), idx] = 0.9
    env = CompetingRecommendersWorld(
        corpus, users, n_platforms=2, alpha=4.0, beta=4.0, eta=eta,
        eta_mob=0.25, shares0=shares0,
    )
    learners = []
    for kind in kinds:
        m = LinearModel(8, 1)
        learners.append(ERMLearner(m, MSELoss()) if kind == "learn"
                        else FreezeAfter(m, MSELoss(), train_rounds=1))
    run_competition(env, learners, n_rounds=40, epoch_size=8, seed=seed)
    return {
        "sat": satisfaction_current(env),
        "welf": welfare_innate(env),
        "gini": (_gini(env.last_exposure[0]) + _gini(env.last_exposure[1])) / 2,
    }


def _bars(ax, results, key, title, ylabel):
    for i, (name, runs) in enumerate(results.items()):
        vals = [r[key] for r in runs]
        ax.bar(i, sum(vals) / len(vals), color=COLORS[name], width=0.6)
        ax.scatter([i] * len(vals), vals, color="black", s=12, zorder=3)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results, fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {eta: {name: [run_arm(kinds, eta, s) for s in SEEDS]
                 for name, kinds in ARMS.items()} for eta in (0.0, 0.15)}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    _bars(axes[0], res[0.0], "gini", "feed concentration (no drift)", "exposure gini")
    _bars(axes[1], res[0.0], "welf", "user welfare (no drift)", "true-taste affinity")
    axes[1].set_ylim(3.5, None)

    ax = axes[2]
    width = 0.35
    for i, name in enumerate(ARMS):
        runs = res[0.15][name]
        sat = [r["sat"] for r in runs]
        welf = [r["welf"] for r in runs]
        ax.bar(i - width / 2, sum(sat) / len(sat), width, color=COLORS[name], alpha=0.45)
        ax.bar(i + width / 2, sum(welf) / len(welf), width, color=COLORS[name])
        ax.scatter([i - width / 2] * len(sat), sat, color="black", s=12, zorder=3)
        ax.scatter([i + width / 2] * len(welf), welf, color="black", s=12, zorder=3)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(ARMS, fontsize=8)
    ax.set_title("with drift: felt (light) vs true (solid)")
    ax.set_ylabel("affinity of consumption")

    fig.suptitle("Retraining concentrates feeds and taxes welfare; drift hides the difference",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "arms.png", dpi=140)

    for eta, by_arm in res.items():
        for name, runs in by_arm.items():
            m = lambda k: sum(r[k] for r in runs) / len(runs)
            print(f"eta={eta:.2f} {name:>13s}: sat={m('sat'):6.3f} welf={m('welf'):6.3f} "
                  f"gini={m('gini'):.3f}")
    print(f"[recommender] figure -> {OUT / 'arms.png'}  (seeds {SEEDS})")


if __name__ == "__main__":
    main()
