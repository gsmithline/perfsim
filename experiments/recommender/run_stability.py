"""Direct performative-stability check for the recommender loop: the per-round
model change ||theta_{t+1} - theta_t|| (and the change in the model's item
scores) going to zero = the deployed model has reached a fixed point of
retraining. Same setup as run_distortion (alpha 0/4/8, eta=0).
Run: python experiments/recommender/run_stability.py
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
from perfsim.scenarios.recommender import build_recommender_env
from perfsim.simulator import Simulator

OUT = Path("runs/recommender")


def trajectory(alpha, *, beta=8.0, eta=0.0, seed=0, n_rounds=20, epoch_size=8):
    env = build_recommender_env(n_items=30, n_users=200, dim=8,
                                alpha=alpha, beta=beta, eta=eta, seed=seed)
    model = LinearModel(8, 1)
    sim = Simulator(env=env, learner=ERMLearner(model, MSELoss()), loss=MSELoss())
    feats = env.features
    thetas, preds = [], []

    def on_round(t, rec):
        thetas.append(model.get_params().detach().clone())
        with torch.no_grad():
            preds.append(model(feats).reshape(-1).clone())

    sim.run(n_rounds=n_rounds, epoch_size=epoch_size, seed=seed, on_round=on_round)
    dtheta = [float((thetas[i + 1] - thetas[i]).norm()) for i in range(len(thetas) - 1)]
    dpred = [float((preds[i + 1] - preds[i]).norm()) for i in range(len(preds) - 1)]
    return dtheta, dpred


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    alphas = {0.0: "#888888", 4.0: "#d8633b", 8.0: "#8d3bd8"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for alpha, c in alphas.items():
        dtheta, dpred = trajectory(alpha)
        label = "alpha=0 (baseline)" if alpha == 0 else f"alpha={alpha:g}"
        ax[0].plot(range(1, len(dtheta) + 1), dtheta, color=c, marker=".", label=label)
        ax[1].plot(range(1, len(dpred) + 1), dpred, color=c, marker=".", label=label)
        print(f"alpha={alpha:4.1f}  ||dtheta|| {dtheta[0]:.4f} -> {dtheta[-1]:.4f}   "
              f"||dpred|| {dpred[0]:.4f} -> {dpred[-1]:.4f}")

    ax[0].set_title("model parameter change  ||theta_{t+1} - theta_t||")
    ax[0].set_ylabel("L2 step");
    ax[1].set_title("model prediction change  ||scores_{t+1} - scores_t||")
    ax[1].set_ylabel("L2 step")
    for a in ax:
        a.set_xlabel("round"); a.set_yscale("log"); a.legend()
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Performative stability: the model stops moving under retraining (step -> 0)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "stability.png", dpi=140)
    print(f"[recommender] figure -> {OUT / 'stability.png'}")


if __name__ == "__main__":
    main()
