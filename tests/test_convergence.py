"""Validation test 1 (gating): RRM and gradient Learners converge to the
closed-form fixed point on GaussianShiftWorld.

For a linear model with MSE on D(theta) = N(A theta + b, Sigma), the
RRM fixed point is theta* = (I - A)^-1 b. This is the canonical PP smoke
test (Perdomo, Mendler-Dunner et al.) and gates every commit to the
numerical core.
"""

from __future__ import annotations

import torch

from perfsim.learners import ERMLearner, GradientLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.environments.dynamics import GaussianShiftWorld


def _make_world_and_model(
    d: int = 3,
    scale: float = 0.5,
    sigma: float = 0.01,
    batch: int = 512,
    seed: int = 0,
) -> tuple[GaussianShiftWorld, LinearModel, torch.Tensor]:
    A = scale * torch.eye(d)
    b = torch.tensor([1.0, 0.5, -0.5])[:d].clone()
    world = GaussianShiftWorld(A=A, b=b, sigma_noise=sigma, batch_size=batch)
    world.reset(seed=seed)
    model = LinearModel(in_features=d, out_features=1, bias=False)
    return world, model, world.closed_form_fp()


def _rrm_rounds(world, model, learner, n_rounds: int) -> None:
    for _ in range(n_rounds):
        data = world.step(model)
        learner.step(data)


def test_erm_converges_to_closed_form_fp() -> None:
    world, model, fp = _make_world_and_model(d=3, sigma=0.005, batch=1024)
    learner = ERMLearner(model, MSELoss(), max_iter=200, tolerance_grad=1e-9)
    _rrm_rounds(world, model, learner, n_rounds=30)
    recovered = model.get_params()
    assert torch.allclose(recovered, fp, atol=0.05), (
        f"ERM did not converge to FP: recovered={recovered.tolist()}, "
        f"fp={fp.tolist()}"
    )


def test_gradient_converges_to_closed_form_fp() -> None:
    world, model, fp = _make_world_and_model(d=3, sigma=0.005, batch=1024)
    learner = GradientLearner(model, MSELoss(), lr=0.1, steps_per_round=1, optimizer="sgd")
    _rrm_rounds(world, model, learner, n_rounds=400)
    recovered = model.get_params()
    assert torch.allclose(recovered, fp, atol=0.1), (
        f"Gradient did not converge to FP: recovered={recovered.tolist()}, "
        f"fp={fp.tolist()}"
    )


def test_erm_converges_under_seed_reset() -> None:
    """Re-running with the same seed must produce identical trajectories."""
    world, model_a, fp = _make_world_and_model(d=3, sigma=0.005, batch=512, seed=0)
    learner_a = ERMLearner(model_a, MSELoss(), max_iter=200)
    _rrm_rounds(world, model_a, learner_a, n_rounds=10)
    params_a = model_a.get_params().clone()

    world.reset(seed=0)
    model_b = LinearModel(in_features=3, out_features=1, bias=False)
    learner_b = ERMLearner(model_b, MSELoss(), max_iter=200)
    _rrm_rounds(world, model_b, learner_b, n_rounds=10)
    params_b = model_b.get_params().clone()

    assert torch.allclose(params_a, params_b, atol=1e-5)
