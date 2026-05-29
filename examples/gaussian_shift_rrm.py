import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def _():
    """
    RRM vs RGD convergence demo on GaussianShiftWorld.

    Both learners chase the same closed-form fixed point theta* = (I - A)^-1 b on
    a stateless MSE-regression world. ERMLearner solves to convergence each round
    (RRM); GradientLearner takes k SGD steps per round (RGD). Prints the distance
    to theta* per round.

    Run: marimo edit examples/gaussian_shift_rrm.py  (or python examples/gaussian_shift_rrm.py)
    """
    return


@app.cell
def _():
    import argparse
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torch

    from perfsim.history import History
    from perfsim.learners import ERMLearner, GradientLearner
    from perfsim.losses import MSELoss
    from perfsim.models import LinearModel
    from perfsim.simulator import Simulator
    from perfsim.environments.dynamics.gaussian_shift import GaussianShiftWorld

    return (
        ERMLearner,
        GaussianShiftWorld,
        GradientLearner,
        History,
        LinearModel,
        MSELoss,
        Simulator,
        torch,
    )


@app.cell
def _(GaussianShiftWorld, torch):
    def make_world(d: int, contraction: float, seed: int) -> GaussianShiftWorld:
        """Contractive Gaussian-shift world with ||A||_2 == contraction (<1)."""
        g = torch.Generator().manual_seed(seed)
        raw = torch.randn(d, d, generator=g)
        spectral = torch.linalg.matrix_norm(raw, ord=2)
        A = raw / spectral * contraction
        b = torch.randn(d, generator=g) * 0.5
        return GaussianShiftWorld(A=A, b=b, sigma_noise=0.01, batch_size=512)

    return (make_world,)


@app.cell
def _(
    ERMLearner,
    GaussianShiftWorld,
    GradientLearner,
    History,
    LinearModel,
    MSELoss,
    Simulator,
):
    def run_one(
        world: GaussianShiftWorld,
        learner_kind: str,
        *,
        n_rounds: int,
        lr: float,
        steps: int,
        seed: int,
    ) -> tuple[History, list[float]]:
        """Run one simulator; return (history, ||theta_t - theta*||_2 trajectory)."""
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

    return (run_one,)


@app.cell
def _(make_world, run_one):
    d = 3
    contraction = .5
    n_rounds = 20
    rgd_lr = 0.05
    rgd_steps = 5
    seed = 0


    world = make_world(d, contraction, seed)
    theta_star = world.closed_form_fp()
    print(f"# GaussianShift: d={d} ||A||_2={contraction}")
    print(f"# theta* = {theta_star.tolist()}")

    _, rrm_dist = run_one(
        world, "rrm", n_rounds=n_rounds, lr=rgd_lr, steps=rgd_steps, seed=seed
    )
    world.reset(seed=seed)
    _, rgd_dist = run_one(
        world, "rgd", n_rounds=n_rounds, lr=rgd_lr, steps=rgd_steps, seed=seed
    )

    print(f"#{'round':>5}  {'||theta_RRM - theta*||':>16}  {'||theta_RGD - theta||':>16}")
    for t, (a, b) in enumerate(zip(rrm_dist, rgd_dist)):
        print(f"  {t:>5d}  {a:>16.6e}  {b:>16.6e}")
    return


if __name__ == "__main__":
    app.run()
