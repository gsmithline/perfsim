"""Helpers shared across multiple World implementations.

Kept internal (`_common.py`) to avoid implying a public extension point.
These are utilities that happen to be useful inside the worlds module.
Promote to a public location only when an external caller needs them.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from perfsim.core.model import Model


def validate_strat_features(
    strat_features: Iterable[int] | None, *, dim: int
) -> Tensor | None:
    """Validate and canonicalize a `strat_features` argument.

    Returns:
        None if `strat_features` is None (all features are strategic).
        Otherwise a `LongTensor` of unique, in-range indices.

    Raises:
        ValueError if the list is empty, has negative indices, has indices
        out of range for `dim`, or contains duplicates.
    """
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
    """Compute ∂(sum_i f(x_0_i; θ)) / ∂x_0 via autograd.

    Wraps the autograd block in `torch.enable_grad()` so this is correct
    whether or not the caller is in a `torch.no_grad()` block (the metrics
    path is: `performative_risk` wraps `world.sample` in `no_grad`).

    Args:
        model: differentiable predictor.
        x0:    (N, D) population features at which to evaluate the gradient.
        expected_n: expected leading dim of model output; raises if mismatched.

    Returns:
        (N, D) detached gradient tensor.
    """
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
    """Apply ε · direction to x0, optionally restricted to a subset of feature columns.

    Args:
        x0:             (N, D) baseline features.
        direction:      (N, D) shift direction (e.g., predictor weight w, or
                        the input gradient ∂f/∂x).
        epsilon:        scalar magnitude (Perdomo's ε; sign-laden).
        strat_features: column indices that may be shifted, or None for all.

    Returns:
        (N, D) shifted features.
    """
    if strat_features is None:
        return x0 + epsilon * direction
    shift = torch.zeros_like(x0)
    shift[:, strat_features] = epsilon * direction[:, strat_features]
    return x0 + shift
