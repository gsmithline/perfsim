"""Performative recommender loop: theta retrains on the engagement it induces.

Runs the PP loop on RecommenderEcosystemWorld and records, per round:
  - alignment_induced: corr(theta, induced engagement)   -> rises (fits its world)
  - alignment_truth:   corr(theta, hidden true relevance) -> falls (performative)
  - exposure_gini:     feed concentration                 -> rises
  - interest_drift:    population taste movement           -> rises
  - engagement_entropy: consumption diversity              -> falls

Headline: the ranker looks increasingly "accurate" on logged engagement while
diverging from true relevance -- accuracy on the world it built, not the truth.

Run from repo root:
    python experiments/scripts/recommender/run_ecosystem_loop.py
Outputs a figure to runs/recommender/ecosystem_loop.png and prints a summary.
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
    alignment_induced,
    alignment_truth,
    build_recommender_env,
    engagement_entropy,
    exposure_gini,
    induced_calibration,
    interest_drift,
)
from perfsim.simulator import Simulator

OUT = Path("runs/recommender")


def run(
    *,
    n_rounds: int = 40,
    epoch_size: int = 8,
    alpha: float = 4.0,
    beta: float = 4.0,
    eta: float = 0.15,
    n_items: int = 30,
    n_users: int = 200,
    dim: int = 8,
    seed: int = 0,
) -> dict[str, list[float]]:
    env = build_recommender_env(
        n_items=n_items, n_users=n_users, dim=dim,
        alpha=alpha, beta=beta, eta=eta, seed=seed,
    )
    model = LinearModel(dim, 1)
    loss = MSELoss()
    sim = Simulator(env=env, learner=ERMLearner(model, loss), loss=loss)

    log: dict[str, list[float]] = {
        k: [] for k in (
            "align_induced", "align_truth", "gini", "drift", "entropy", "calib",
        )
    }

    def on_round(t: int, record: dict) -> None:
        data = {"x": env.features, "y": env.last_engagement.reshape(-1, 1)}
        log["align_induced"].append(alignment_induced(model, data))
        log["align_truth"].append(alignment_truth(model, env))
        log["gini"].append(exposure_gini(env))
        log["drift"].append(interest_drift(env))
        log["entropy"].append(engagement_entropy(data))
        log["calib"].append(induced_calibration(model, data))

    sim.run(n_rounds=n_rounds, epoch_size=epoch_size, seed=seed, on_round=on_round)
    return log


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = run()
    rounds = range(len(log["align_induced"]))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    ax[0].plot(rounds, log["align_induced"], label="align(theta, engagement)", color="#3b7dd8")
    ax[0].plot(rounds, log["align_truth"], label="align(theta, true relevance)", color="purple")
    ax[0].set_title("Performative divergence")
    ax[0].set_xlabel("round")
    ax[0].set_ylim(-1.05, 1.05)
    ax[0].legend()

    ax[1].plot(rounds, log["gini"], label="exposure Gini", color="#d8633b")
    ax[1].plot(rounds, log["entropy"], label="engagement entropy", color="#3bd86b")
    ax[1].set_title("Feed concentration / diversity")
    ax[1].set_xlabel("round")
    ax[1].legend()

    ax[2].plot(rounds, log["drift"], label="mean interest drift", color="#8d3bd8")
    ax[2].set_title("Population taste drift")
    ax[2].set_xlabel("round")
    ax[2].legend()

    fig.suptitle(
        "Performative recommender: theta fits induced engagement while diverging from truth",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "ecosystem_loop.png", dpi=140)

    print(
        f"[recommender] round 0 -> {len(log['align_induced']) - 1}: "
        f"align_induced {log['align_induced'][0]:+.3f} -> {log['align_induced'][-1]:+.3f}; "
        f"align_truth {log['align_truth'][0]:+.3f} -> {log['align_truth'][-1]:+.3f}; "
        f"gini {log['gini'][0]:.3f} -> {log['gini'][-1]:.3f}; "
        f"drift {log['drift'][0]:.3f} -> {log['drift'][-1]:.3f}"
    )
    print(f"[recommender] figure -> {OUT / 'ecosystem_loop.png'}")


if __name__ == "__main__":
    main()