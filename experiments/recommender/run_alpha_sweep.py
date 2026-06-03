"""Exposure-bias sweep: does fitting induced engagement cost truth alignment?

Holds beta/eta/seed fixed and sweeps alpha (the exposure bias). For each alpha we
run the full PP loop to its stable point and record the FINAL:
  - align_induced: corr(theta, induced engagement)
  - align_truth:   corr(theta, true relevance)
  - gini:          exposure concentration

Expected: at alpha=0 (no exposure bias) engagement reflects true affinity, so
theta recovers truth (align_truth high). As alpha grows, induced engagement is
driven by exposure feedback; theta still fits it (align_induced stays high) but
align_truth degrades -- the performative cost made causal.

Run from repo root:
    python experiments/scripts/recommender/run_alpha_sweep.py
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
    exposure_gini,
    build_recommender_env as _build,  # alias, keeps lints calm if unused
)
from perfsim.simulator import Simulator

OUT = Path("runs/recommender")
ALPHAS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]


def final_metrics(
    alpha: float,
    *,
    beta: float = 8.0,
    eta: float = 0.15,
    n_rounds: int = 40,
    epoch_size: int = 8,
    n_items: int = 30,
    n_users: int = 200,
    dim: int = 8,
    seed: int = 0,
) -> tuple[float, float, float]:
    env = build_recommender_env(
        n_items=n_items, n_users=n_users, dim=dim,
        alpha=alpha, beta=beta, eta=eta, seed=seed,
    )
    model = LinearModel(dim, 1)
    loss = MSELoss()
    sim = Simulator(env=env, learner=ERMLearner(model, loss), loss=loss)
    sim.run(n_rounds=n_rounds, epoch_size=epoch_size, seed=seed)
    data = {"x": env.features, "y": env.last_engagement.reshape(-1, 1)}
    return (
        alignment_induced(model, data),
        alignment_truth(model, env),
        exposure_gini(env),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    induced, truth, gini = [], [], []
    for a in ALPHAS:
        ai, at, g = final_metrics(a)
        induced.append(ai)
        truth.append(at)
        gini.append(g)
        print(f"[alpha={a:5.1f}] align_induced={ai:+.3f}  align_truth={at:+.3f}  gini={g:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(ALPHAS, induced, "o-", label="align(theta, engagement)", color="#3b7dd8")
    ax[0].plot(ALPHAS, truth, "o-", label="align(theta, true relevance)", color="purple")
    ax[0].set_xlabel("exposure bias alpha")
    ax[0].set_ylabel("correlation")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title("Truth alignment degrades as exposure bias grows")
    ax[0].legend()

    ax[1].plot(ALPHAS, gini, "o-", color="#d8633b")
    ax[1].set_xlabel("exposure bias alpha")
    ax[1].set_ylabel("exposure Gini")
    ax[1].set_title("Feed concentration vs exposure bias")

    fig.tight_layout()
    fig.savefig(OUT / "alpha_sweep.png", dpi=140)
    print(f"[recommender] figure -> {OUT / 'alpha_sweep.png'}")


if __name__ == "__main__":
    main()