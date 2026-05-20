"""Loss implementations: MSE, CE, BCE, BCEWithLogits, Hinge.

Each Loss is a callable `(model, data, *, reduction) -> Tensor`. Reduction is
"mean" (default), "sum", or "none" (per-example).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from perfsim.core.loss import Loss
from perfsim.core.model import Model
from perfsim.core.types import Data


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"unknown reduction {reduction!r}; expected 'mean', 'sum', or 'none'")


def _align(y: Tensor, target_shape: torch.Size) -> Tensor:
    return y if y.shape == target_shape else y.view(target_shape)


class MSELoss(Loss):
    """Mean squared error. Per-example: ||y_hat - y||^2 (sum over output dims)."""

    def __call__(
        self, model: Model, data: Data, *, reduction: str = "mean"
    ) -> Tensor:
        y_hat = model(data["x"])
        y = _align(data["y"].to(y_hat.dtype), y_hat.shape)
        sq = (y_hat - y).pow(2)
        if sq.ndim > 1:
            sq = sq.sum(dim=tuple(range(1, sq.ndim)))
        return _reduce(sq, reduction)


class CrossEntropyLoss(Loss):
    """
    Multinomial cross-entropy from logits. 
    Need Integer Labels
    """

    def __call__(
        self, model: Model, data: Data, *, reduction: str = "mean"
    ) -> Tensor:
        logits = model(data["x"])
        y = data["y"].long()
        per_ex = F.cross_entropy(logits, y, reduction="none")
        return _reduce(per_ex, reduction)


class BCELoss(Loss):
    """Binary cross-entropy from probabilities in (0, 1"""

    def __call__(
        self, model: Model, data: Data, *, reduction: str = "mean"
    ) -> Tensor:
        p = model(data["x"])
        y = _align(data["y"].to(p.dtype), p.shape)
        per_ex = F.binary_cross_entropy(p, y, reduction="none")
        if per_ex.ndim > 1:
            per_ex = per_ex.sum(dim=tuple(range(1, per_ex.ndim)))
        return _reduce(per_ex, reduction)


class BCEWithLogitsLoss(Loss):
    """Binary cross-entropy from logits. Numerically stable."""

    def __call__(
        self, model: Model, data: Data, *, reduction: str = "mean"
    ) -> Tensor:
        logits = model(data["x"])
        y = _align(data["y"].to(logits.dtype), logits.shape)
        per_ex = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        if per_ex.ndim > 1:
            per_ex = per_ex.sum(dim=tuple(range(1, per_ex.ndim)))
        return _reduce(per_ex, reduction)


class HingeLoss(Loss):
    """Hinge: max(0, 1 - y * f(x)) with y in {-1, +1}."""

    def __call__(
        self, model: Model, data: Data, *, reduction: str = "mean"
    ) -> Tensor:
        score = model(data["x"])
        y = _align(data["y"].to(score.dtype), score.shape)
        per_ex = torch.clamp(1.0 - y * score, min=0.0)
        if per_ex.ndim > 1:
            per_ex = per_ex.sum(dim=tuple(range(1, per_ex.ndim)))
        return _reduce(per_ex, reduction)


class L2RegularizedLoss(Loss):
    """Wrap a base Loss with L2 weight decay on the model's parameters.

    Adds `0.5 * weight_decay * sum_p ||p||^2` to the base loss for the
    `mean` and `sum` reductions. The L2 term is NOT added for `reduction =
    "none"` (per-example losses); the regularization is a population-level
    quantity, not per-example.

    The L2 term retains its autograd dependency on thr model params so the gradient based learners use it 
    """

    def __init__(
        self,
        base: Loss,
        weight_decay: float = 0.0,
        *,
        decay_bias: bool = True,
    ) -> None:
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0; got {weight_decay}")
        self.base = base
        self.weight_decay = float(weight_decay)
        self.decay_bias = bool(decay_bias)

    def __call__(
        self, model: Model, data: Data, *, reduction: str = "mean"
    ) -> Tensor:
        base = self.base(model, data, reduction=reduction)
        if self.weight_decay == 0.0 or reduction == "none":
            return base
        reg = torch.zeros((), dtype=base.dtype, device=base.device)
        for name, p in model.named_parameters():
            if not self.decay_bias and (name.endswith(".bias") or name == "bias"):
                continue
            reg = reg + (p * p).sum()
        return base + 0.5 * self.weight_decay * reg
