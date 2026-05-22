"""StrategicGradientWorld: non-linear strategic best-response via autograd.

Generalizes `StrategicLinearWorld` to arbitrary differentiable predictors.
Each agent has fixed initial features `x_0` and a fixed label `y`. On each
round, agents shift their features along the gradient of the predictor's
scalar output wrt the input:

    x_t = x_0 + epsilon * ∂f(x_0; θ) / ∂x

For a linear predictor f(x) = w·x (e.g. `LinearModel(out_features=1)`)
this gradient is just `w`, so `StrategicGradientWorld` reduces exactly to
`StrategicLinearWorld`. For non-linear predictors (MLPs etc.) the
gradient is a non-trivial function of `x_0`, giving each agent a
location-dependent strategic shift.

Sign convention (matches StrategicLinearWorld and Perdomo's notebook):
positive `epsilon` shifts agents *up* the predictor's gradient. To get
the Perdomo strategic-loan setup where agents try to *lower* their
predicted default probability, pass `epsilon = -mu` for `mu > 0`.

The predictor's output for each agent must be a scalar; for vector-valued
predictors the world sums over output dims before computing the
gradient (matches the behavior of `model(x).sum().backward()`).
"""

from __future__ import annotations

from typing import ClassVar, Iterable

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.core.environment import StatefulDynamics
from perfsim.environments.dynamics._common import (
    apply_strategic_shift,
    input_gradient,
    validate_strat_features,
)


class StrategicGradientWorld(StatefulDynamics):
    """Strategic best-response via the predictor's input gradient.

    One-shot best-response to a deployed classifier; N>1 inner steps either
    repeat or compound trivially. Forces `epoch_size = 1` (DESIGN.md §8).
    """

    max_meaningful_epoch_size: ClassVar[int] = 1

    def __init__(
        self,
        x0: Tensor,
        y: Tensor,
        epsilon: float = 1.0,
        strat_features: Iterable[int] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if x0.ndim != 2:
            raise ValueError(f"x0 must be 2-D (N, D); got {tuple(x0.shape)}")
        if y.shape[0] != x0.shape[0]:
            raise ValueError(
                f"y leading dim {y.shape[0]} does not match x0 leading dim {x0.shape[0]}"
            )
        self._x0 = x0.to(dtype=dtype).clone()
        self._y = y.clone()
        self._epsilon = float(epsilon)
        self._dtype = dtype
        self._n, self._d = x0.shape
        self._strat_features = validate_strat_features(strat_features, dim=self._d)
        # Per-agent index for LM-Learner profile lookup after train_mask filtering.
        self._agent_idx = torch.arange(self._n)

    @property
    def produces_schema(self) -> DataSchema:
        return SUPERVISED_SCHEMA

    @property
    def n_agents(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._d

    @property
    def strat_features(self) -> tuple[int, ...] | None:
        if self._strat_features is None:
            return None
        return tuple(int(i) for i in self._strat_features.tolist())

    def reset(self, seed: int = 0) -> None:
        """No RNG state to reset; population is fixed at __init__."""
        return

    def sample(self, model: Model) -> Data:
        grad_x = input_gradient(model, self._x0, expected_n=self._n).to(self._dtype)
        x = apply_strategic_shift(
            self._x0,
            grad_x,
            epsilon=self._epsilon,
            strat_features=self._strat_features,
        )
        return {"x": x, "y": self._y, "agent_idx": self._agent_idx.clone()}

    def step(self, model: Model) -> Data:
        return self.sample(model)
