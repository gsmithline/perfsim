"""Helpers shared across multiple World implementations."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from perfsim.core.model import Model


def validate_strat_features(
    strat_features: Iterable[int] | None, *, dim: int
) -> Tensor | None:
    """Validate and canonicalize a `strat_features` argument."""
    if strat_features is None:
        return None
    idx = torch.tensor(list(strat_features), dtype=torch.long)
    if idx.numel() == 0:
        raise ValueError(
            "strat_features cannot be empty when set; pass None for all-features"
        )
    if int(idx.min().item()) < 0:
        raise ValueError(
            f"strat_features must be non-negative; got min={int(idx.min().item())}"
        )
    if int(idx.max().item()) >= dim:
        raise ValueError(
            f"strat_features max index {int(idx.max().item())} >= d={dim}"
        )
    if idx.unique().numel() != idx.numel():
        raise ValueError(f"strat_features must be unique; got {idx.tolist()}")
    return idx


def input_gradient(model: Model, x0: Tensor, *, expected_n: int) -> Tensor:
    """Compute d(sum_i f(x_0_i; theta)) / dx_0 via autograd."""
    with torch.enable_grad():
        x = x0.detach().clone().requires_grad_(True)
        scores = model(x)
        if scores.shape[0] != expected_n:
            raise ValueError(
                f"model output leading dim {scores.shape[0]} does not match "
                f"population size {expected_n}"
            )
        grad_x = torch.autograd.grad(scores.sum(), x, create_graph=False)[0]
    return grad_x.detach()


def apply_strategic_shift(
    x0: Tensor,
    direction: Tensor,
    *,
    epsilon: float,
    strat_features: Tensor | None,
) -> Tensor:
    """Apply epsilon * direction to x0, optionally restricted to a subset of feature columns."""
    if strat_features is None:
        return x0 + epsilon * direction
    shift = torch.zeros_like(x0)
    shift[:, strat_features] = epsilon * direction[:, strat_features]
    return x0 + shift
