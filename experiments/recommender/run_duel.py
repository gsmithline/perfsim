"""Duel: honest-frozen platform vs continually retraining one, with and
without taste drift.
Run: python experiments/recommender/run_duel.py
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


def _gini(x: torch.Tensor) -> float:
    xs, _ = x.reshape(-1).clamp_min(0.0).sort()
    n = xs.numel()
    idx = torch.arange(1, n + 1, dtype=xs.dtype)
    return float((2.0 * (idx * xs).sum()) / (n * xs.sum()) - (n + 1) / n)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1) - a.mean()
    b = b.reshape(-1) - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def trajectory(eta: float, *, seed: int = 0, n_rounds: int = 40,
               epoch_size: int = 8) -> dict[str, list[float]]:
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
    frozen, retrain = LinearModel(8, 1), LinearModel(8, 1)
    learners = [FreezeAfter(frozen, MSELoss(), train_rounds=1),
                ERMLearner(retrain, MSELoss())]
    q = env.hidden_quality()
    out: dict[str, list[float]] = {k: [] for k in (
        "share_retrain", "truth_frozen", "truth_retrain",
        "gini_frozen", "gini_retrain", "sat_frozen", "sat_retrain",
    )}

    def on_round(t: int, record: dict) -> None:
        with torch.no_grad():
            sA = frozen(env.features).reshape(-1)
            sB = retrain(env.features).reshape(-1)
        sat = env.last_satisfaction
        out["share_retrain"].append(float(env.shares[:, 1].mean()))
        out["truth_frozen"].append(_pearson(sA, q))
        out["truth_retrain"].append(_pearson(sB, q))
        out["gini_frozen"].append(_gini(env.last_exposure[0]))
        out["gini_retrain"].append(_gini(env.last_exposure[1]))
        out["sat_frozen"].append(float(sat[:, 0].mean()))
        out["sat_retrain"].append(float(sat[:, 1].mean()))

    run_competition(env, learners, n_rounds=n_rounds, epoch_size=epoch_size,
                    seed=seed, on_round=on_round)
    return out


def _band(ax, runs, key, color, label):
    curves = torch.tensor([r[key] for r in runs])
    mean = curves.mean(dim=0)
    ax.plot(range(len(mean)), mean, color=color, label=label)
    ax.fill_between(range(len(mean)), curves.min(dim=0).values,
                    curves.max(dim=0).values, color=color, alpha=0.15)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs0 = [trajectory(0.0, seed=s) for s in SEEDS]
    runs15 = [trajectory(0.15, seed=s) for s in SEEDS]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("truth_frozen", "truth_retrain", "corr(scores, true relevance)"),
        ("gini_frozen", "gini_retrain", "exposure gini"),
        ("sat_frozen", "sat_retrain", "mean satisfaction"),
    ]
    for ax, (kA, kB, title) in zip(axes.flat, panels):
        _band(ax, runs0, kA, "#3b6fd8", "frozen (honest fit)")
        _band(ax, runs0, kB, "#d8633b", "retraining")
        ax.set_title(f"{title}  (eta=0)")
        ax.set_xlabel("round")
    axes.flat[0].legend(fontsize=8)

    ax = axes.flat[3]
    _band(ax, runs0, "share_retrain", "#d8633b", "eta=0 (no drift)")
    _band(ax, runs15, "share_retrain", "#8d3bd8", "eta=0.15 (drift rescue)")
    ax.axhline(0.5, color="black", lw=0.7)
    ax.set_title("retrainer's market share")
    ax.set_xlabel("round")
    ax.legend(fontsize=8)

    fig.suptitle("Retraining on self-biased labels loses the duel; taste drift rescues it",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "duel.png", dpi=140)

    for label, runs in (("eta=0.00", runs0), ("eta=0.15", runs15)):
        share = torch.tensor([r["share_retrain"][-1] for r in runs]).mean()
        truth = torch.tensor([r["truth_retrain"][-1] for r in runs]).mean()
        print(f"{label}: final retrainer share {share:.3f}, truth-alignment {truth:.3f}")
    print(f"[recommender] figure -> {OUT / 'duel.png'}  (mean over seeds {SEEDS})")


if __name__ == "__main__":
    main()
