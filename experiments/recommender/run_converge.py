"""Direct convergence read on the both-retrain recommender competition (same
setup as run_arms): population interest dispersion (variance analog), mean
drift, and inter-platform feed divergence over rounds. Tests whether the
recommender shows the FJ-style mean/variance contraction + platform merge.
Run: python experiments/recommender/run_converge.py
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
)

OUT = Path("runs/recommender")
SEEDS = (0, 1, 2)
ROUNDS = 40


def dispersion(interest):
    c = interest - interest.mean(dim=0, keepdim=True)
    return float((c ** 2).sum(dim=1).mean().sqrt())          # RMS distance to centroid


def run_one(eta, seed):
    gen = torch.Generator(); gen.manual_seed(seed)
    corpus = sample_corpus(30, 8, generator=gen)
    users = sample_user_population(200, 8, n_communities=4, generator=gen)
    idx = users.community % 2
    shares0 = torch.full((200, 2), 0.1)
    shares0[torch.arange(200), idx] = 0.9
    env = CompetingRecommendersWorld(
        corpus, users, n_platforms=2, alpha=4.0, beta=4.0, eta=eta,
        eta_mob=0.25, shares0=shares0,
    )
    learners = [ERMLearner(LinearModel(8, 1), MSELoss()) for _ in range(2)]
    disp, drift, pdiv = [], [], []
    i0_mean = env.interest0.mean(dim=0)

    def on_round(t, rec):
        I = env.current_interest
        disp.append(dispersion(I))
        drift.append(float((I.mean(dim=0) - i0_mean).norm()))
        u = env.last_exposure
        pdiv.append(float((u[0] - u[1]).abs().mean()))

    run_competition(env, learners, n_rounds=ROUNDS, epoch_size=8, seed=seed, on_round=on_round)
    return torch.tensor(disp), torch.tensor(drift), torch.tensor(pdiv)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    print(f"{'eta':>5s} | {'dispersion f->l':>18s} {'mean-drift f->l':>18s} {'plat-div f->l':>16s}")
    for eta, col in ((0.0, "#3b6fd8"), (0.15, "#d8633b")):
        runs = [run_one(eta, s) for s in SEEDS]
        disp = torch.stack([r[0] for r in runs]).mean(0)
        drift = torch.stack([r[1] for r in runs]).mean(0)
        pdiv = torch.stack([r[2] for r in runs]).mean(0)
        ax[0].plot(disp, color=col, label=f"eta={eta:g}")
        ax[1].plot(drift, color=col, label=f"eta={eta:g}")
        ax[2].plot(pdiv, color=col, label=f"eta={eta:g}")
        print(f"{eta:>5g} | {disp[0]:.3f}->{disp[-1]:.3f}        "
              f"{drift[0]:.3f}->{drift[-1]:.3f}        {pdiv[0]:.4f}->{pdiv[-1]:.4f}")
    for a, t in zip(ax, ("population dispersion (variance analog)",
                         "population mean drift from innate",
                         "inter-platform feed divergence")):
        a.set_title(t); a.set_xlabel("round")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].legend(fontsize=8)
    fig.suptitle("Recommender both-retrain: does it show FJ-style moment convergence + platform merge?",
                 fontweight="bold")
    fig.savefig(OUT / "converge.png", dpi=150)
    print(f"[recommender] figure -> {OUT / 'converge.png'}")


if __name__ == "__main__":
    main()
