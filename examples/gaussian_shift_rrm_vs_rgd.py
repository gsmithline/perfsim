"""

RRM vs RGD convergence demo on GaussianShiftWorld.

Both Learners chase the same closed-form fixed point θ* = (I - A)^-1 b, on a stateless, MSE-regression world. 
ERMLearner solves to convergence each round (RRM), GradientLearner takes ``k`` SGD steps per round (RGD). 
Prints the distance to θ* per round and saves a log-scale convergence plot.

Run:
    python examples/gaussian_shift_rrm_vs_rgd.py
    python examples/gaussian_shift_rrm_vs_rgd.py --d 5 --n-rounds 40
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl_cache")
)

import matplotlib.pyplot as plt
import torch

from perfsim.history import History
from perfsim.learners import ERMLearner, GradientLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.simulator import Simulator
from perfsim.worlds.gaussian_shift import GaussianShiftWorld


def _make_world(d: int, contraction: float, seed: int) -> GaussianShiftWorld:
    """Construct a contractive Gaussian-shift world.

    A is set so ||A||_2 == `contraction` (< 1 for RRM convergence) by drawing
    a random matrix, normalizing by its spectral norm, then scaling.
    """
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(d, d, generator=g)
    spectral = torch.linalg.matrix_norm(raw, ord=2)
    A = raw / spectral * contraction
    b = torch.randn(d, generator=g) * 0.5
    return GaussianShiftWorld(A=A, b=b, sigma_noise=0.01, batch_size=512)


def _run_one(
    world: GaussianShiftWorld,
    learner_kind: str,
    *,
    n_rounds: int,
    lr: float,
    steps: int,
    seed: int,
) -> tuple[History, list[float]]:
    """Run one simulator and return (history, ||θ_t - θ*||_2 trajectory)."""
    model = LinearModel(in_features=world.dim, out_features=1, bias=False)
    loss = MSELoss()
    if learner_kind == "rrm":
        learner = ERMLearner(model, loss, max_iter=200)
    elif learner_kind == "rgd":
        learner = GradientLearner(
            model, loss, lr=lr, steps_per_round=steps, optimizer="sgd"
        )
    else:
        raise ValueError(f"unknown learner {learner_kind!r}")
    theta_star = world.closed_form_fp()
    distances: list[float] = []
    sim = Simulator(world=world, learner=learner, loss=loss)
    history = sim.run(n_rounds=n_rounds, seed=seed)
    for r in history.records:
        theta = r["theta"].flatten()
        distances.append((theta - theta_star).norm().item())
    return history, distances


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--contraction", type=float, default=0.5, help="||A||_2 in [0, 1)")
    p.add_argument("--n-rounds", type=int, default=20)
    p.add_argument("--rgd-lr", type=float, default=0.1)
    p.add_argument("--rgd-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "gaussian_shift_rrm_vs_rgd.png",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    world = _make_world(args.d, args.contraction, args.seed)
    theta_star = world.closed_form_fp()
    print(f"# GaussianShift: d={args.d} ||A||_2={args.contraction}")
    print(f"# theta* = {theta_star.tolist()}")

    _, rrm_dist = _run_one(
        world, "rrm", n_rounds=args.n_rounds, lr=args.rgd_lr, steps=args.rgd_steps, seed=args.seed
    )
    world.reset(seed=args.seed)
    _, rgd_dist = _run_one(
        world, "rgd", n_rounds=args.n_rounds, lr=args.rgd_lr, steps=args.rgd_steps, seed=args.seed
    )

    print(f"# {'round':>5}  {'||θ_RRM - θ*||':>16}  {'||θ_RGD - θ*||':>16}")
    for t, (a, b) in enumerate(zip(rrm_dist, rgd_dist)):
        print(f"  {t:>5d}  {a:>16.6e}  {b:>16.6e}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(rrm_dist)), rrm_dist, marker="o", markersize=4, label="RRM (ERM)")
    ax.plot(range(len(rgd_dist)), rgd_dist, marker="s", markersize=4,
            label=f"RGD (SGD, lr={args.rgd_lr}, k={args.rgd_steps})")
    ax.set_yscale("log")
    ax.set_xlabel("round t")
    ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$")
    ax.set_title(f"RRM vs RGD on GaussianShift (d={args.d}, $\\|A\\|_2$={args.contraction})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"# saved figure -> {args.out}")


if __name__ == "__main__":
    main()
