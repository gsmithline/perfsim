"""Mediation-strength (gamma) sweep for the social-dilemma PP loop (PPO pop).

Outer performative loop, per gamma:
  - theta (MLP) trains on the previous epoch's induced (x_i, y_i)
  - the PPO population trains for `ppo_cycles` in the public-goods substrate
    under the frozen theta (which enters via reward shaping, strength gamma)
  - read off equilibrium cooperation y_i, log the four welfare probes

The headline to look for: as gamma rises, the platform stays calibrated
(calibration error stays low) while the induced population degrades -- mean
cooperation and the type-signal R^2 fall, behavioral variance collapses. That is
"accurate on the world it helped create, but the world is worse."

Run from repo root:  python experiments/scripts/social_dilemma/run_gamma_sweep.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perfsim.losses import MSELoss
from perfsim.learners import GradientLearner
from perfsim.models import MLPModel
from perfsim.scenarios.social_dilemma import (
    PublicGoodsMARL,
    calibration_error,
    cooperation_variance,
    mean_cooperation,
    type_r2,
)

OUT = Path("runs/social_dilemma")
N = int(os.environ.get("N_AGENTS", 24))
D = 3
N_ROUNDS = int(os.environ.get("N_ROUNDS", 12))
PPO_CYCLES = int(os.environ.get("PPO_CYCLES", 20))
GAMMAS = [float(g) for g in os.environ.get("GAMMAS", "0.0,0.3,0.6,1.0").split(",")]
SEED = int(os.environ.get("SEED", 0))


def make_population(seed: int):
    g = torch.Generator().manual_seed(seed)
    half = N // 2
    type_baseline = torch.cat(
        [
            0.75 + 0.10 * torch.randn(half, generator=g),
            0.25 + 0.10 * torch.randn(N - half, generator=g),
        ]
    ).clamp(0.0, 1.0)
    # x[:,0] carries the type signal (+ noise); x[:,1:] are nuisance features.
    x = torch.stack(
        [
            type_baseline + 0.10 * torch.randn(N, generator=g),
            torch.randn(N, generator=g),
            torch.randn(N, generator=g),
        ],
        dim=1,
    )
    return x.float(), type_baseline.float()


def run_one(gamma: float):
    x, type_baseline = make_population(SEED)
    model = MLPModel(D, [64, 64], 1, final_activation="sigmoid", init_seed=SEED)
    learner = GradientLearner(model, MSELoss(), lr=1e-2, steps_per_round=50, optimizer="adam")
    env = PublicGoodsMARL(x, type_baseline, gamma=gamma, horizon=16, ppo_epochs=4)
    env.reset(seed=SEED)

    traj = []
    prev = None
    for t in range(N_ROUNDS):
        if prev is not None:
            learner.train(prev)
        data = env.run(model, n_steps=PPO_CYCLES)
        row = {
            "round": t,
            "ybar": mean_cooperation(data),
            "var": cooperation_variance(data),
            "type_r2": type_r2(data, type_baseline),
            "calib_err": calibration_error(model, data),
        }
        traj.append(row)
        print(
            f"  [gamma={gamma:.2f}] r{t:02d}  ybar={row['ybar']:.3f}  "
            f"var={row['var']:.4f}  type_r2={row['type_r2']:.3f}  "
            f"calib_err={row['calib_err']:.4f}",
            flush=True,
        )
        prev = data
    return traj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for gamma in GAMMAS:
        print(f"=== gamma = {gamma} ===", flush=True)
        results[gamma] = run_one(gamma)

    (OUT / "gamma_sweep.json").write_text(json.dumps(results, indent=2))

    panels = [
        ("ybar", "mean cooperation  (welfare)"),
        ("var", "Var(y)  (homogenization)"),
        ("type_r2", "R^2 type -> y  (signal survival)"),
        ("calib_err", "calibration error  (platform accuracy)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
    for ax, (key, title) in zip(axes, panels):
        for gamma in GAMMAS:
            rounds = [r["round"] for r in results[gamma]]
            ax.plot(rounds, [r[key] for r in results[gamma]], marker="o", ms=3,
                    label=f"gamma={gamma}")
        ax.set_title(title)
        ax.set_xlabel("round")
    axes[0].legend()
    fig.suptitle("Social dilemma (PPO population): mediation-strength sweep", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "gamma_sweep.png", dpi=140)
    print(f"saved {OUT / 'gamma_sweep.png'} and {OUT / 'gamma_sweep.json'}", flush=True)


if __name__ == "__main__":
    main()
