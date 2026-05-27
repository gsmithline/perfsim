"""Metrics: PR, DPR, stability_gap, optimality_gap, convergence detection."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from perfsim.core.loss import Loss
from perfsim.core.model import Model
from perfsim.core.environment import Environment as World


def performative_risk(world: World, model: Model, loss: Loss) -> Tensor:
    """Single-batch performative risk estimate."""
    with torch.no_grad():
        data = world.sample(model)
        return loss(model, data, reduction="mean")


def decoupled_risk(
    world: World,
    model_deploy: Model,
    model_eval: Model,
    loss: Loss,
) -> Tensor:
    """Single-batch decoupled performative risk estimate."""
    with torch.no_grad():
        data = world.sample(model_deploy)
        return loss(model_eval, data, reduction="mean")


def stability_gap(theta_prev: Tensor, theta_curr: Tensor) -> Tensor:
    """||theta_t - theta_{t-1}||_2."""
    return (theta_curr - theta_prev).norm()


def optimality_gap(
    world: World,
    model: Model,
    optimal_model: Model,
    loss: Loss,
) -> Tensor:
    """PR(model) - PR(optimal_model)."""
    pr_current = performative_risk(world, model, loss)
    pr_optimal = performative_risk(world, optimal_model, loss)
    return pr_current - pr_optimal


def has_converged(
    thetas: Sequence[Tensor],
    *,
    tol: float = 1e-4,
    window: int = 5,
) -> bool:
    """True if all pairwise gaps within the last `window` thetas are below `tol`.

    Returns False if fewer than `window + 1` thetas are available.
    """
    if len(thetas) < window + 1:
        return False
    recent = thetas[-(window + 1) :]
    gaps = torch.stack(
        [(recent[i + 1] - recent[i]).norm() for i in range(len(recent) - 1)]
    )
    return bool((gaps < tol).all())


def _theta_diff_norm(model_a: Model, model_b: Model) -> Tensor:
    return (model_a.get_params().flatten() - model_b.get_params().flatten()).norm()


def sensitivity_paired(world: World, model_a: Model, model_b: Model) -> Tensor:
    """Lipschitz estimate of D(theta) under the paired (identity) coupling."""
    with torch.no_grad():
        denom = _theta_diff_norm(model_a, model_b)
        if denom.item() == 0.0:
            raise ValueError(
                "sensitivity is undefined when theta_a == theta_b (zero denominator)"
            )
        x_a = world.sample(model_a)["x"]
        x_b = world.sample(model_b)["x"]
        if x_a.shape != x_b.shape:
            raise ValueError(
                f"paired sensitivity requires equal sample shapes; got "
                f"{tuple(x_a.shape)} vs {tuple(x_b.shape)}"
            )
        diff = (x_a - x_b).reshape(x_a.shape[0], -1)
        return diff.norm(dim=-1).mean() / denom


def sensitivity_sliced(
    world: World,
    model_a: Model,
    model_b: Model,
    *,
    n_proj: int = 50,
    seed: int = 0,
) -> Tensor:
    """Sliced-Wasserstein-1 Lipschitz estimate of D(theta)."""
    with torch.no_grad():
        denom = _theta_diff_norm(model_a, model_b)
        if denom.item() == 0.0:
            raise ValueError(
                "sensitivity is undefined when theta_a == theta_b (zero denominator)"
            )
        x_a = world.sample(model_a)["x"]
        x_b = world.sample(model_b)["x"]
        if x_a.shape != x_b.shape:
            raise ValueError(
                f"sliced sensitivity requires equal sample shapes; got "
                f"{tuple(x_a.shape)} vs {tuple(x_b.shape)}"
            )
        flat_a = x_a.reshape(x_a.shape[0], -1)
        flat_b = x_b.reshape(x_b.shape[0], -1)
        feat_dim = flat_a.shape[-1]
        g = torch.Generator(device=flat_a.device).manual_seed(int(seed))
        directions = torch.randn(
            n_proj, feat_dim, generator=g, dtype=flat_a.dtype, device=flat_a.device
        )
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        proj_a = flat_a @ directions.T  # (B, n_proj)
        proj_b = flat_b @ directions.T
        sorted_a, _ = proj_a.sort(dim=0)
        sorted_b, _ = proj_b.sort(dim=0)
        per_proj_w1 = (sorted_a - sorted_b).abs().mean(dim=0)  # (n_proj,)
        sliced_w1 = per_proj_w1.mean()
        return sliced_w1 / denom
