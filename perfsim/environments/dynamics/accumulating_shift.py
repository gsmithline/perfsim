"""AccumulatingShiftWorld: strategic agents whose baseline features drift.

Relaxes static strategic classification: each round x_0 drifts toward the
strategic position, modeling agents who internalize past manipulations.

    grad        = d f(x_0; theta) / dx
    x_strategic = x_0 + epsilon * grad                 (this round's data)
    x_0_next    = (1 - eta) * x_0 + eta * x_strategic  (drift)

eta=0 recovers static StrategicGradientWorld; eta=1 fully adopts each round's
strategic position. Positive epsilon shifts up the input gradient; pass
epsilon=-mu for Perdomo-style risk-lowering agents.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.environments.dynamics._common import (
    apply_strategic_shift,
    input_gradient,
    validate_strat_features,
)
from perfsim.environments.dynamics.stateful_population import State, StatefulPopulationWorld


class AccumulatingShiftWorld(StatefulPopulationWorld):
    """Strategic-shift world where x_0 drifts toward x_strategic each round."""

    def __init__(
        self,
        x0: Tensor,
        y: Tensor,
        epsilon: float = 1.0,
        eta: float = 0.0,
        strat_features: Iterable[int] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if x0.ndim != 2:
            raise ValueError(f"x0 must be 2-D (N, D); got {tuple(x0.shape)}")
        if y.shape[0] != x0.shape[0]:
            raise ValueError(
                f"y leading dim {y.shape[0]} does not match x0 leading dim {x0.shape[0]}"
            )
        if not (0.0 <= eta <= 1.0):
            raise ValueError(f"eta must be in [0, 1]; got {eta}")
        x0_t = x0.to(dtype=dtype).detach().clone()
        super().__init__({"x0": x0_t}, dtype=dtype)
        self._y = y.clone()
        self._epsilon = float(epsilon)
        self._eta = float(eta)
        self._n, self._d = x0_t.shape
        self._strat_features = validate_strat_features(strat_features, dim=self._d)

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
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def eta(self) -> float:
        return self._eta

    @property
    def strat_features(self) -> tuple[int, ...] | None:
        if self._strat_features is None:
            return None
        return tuple(int(i) for i in self._strat_features.tolist())

    def _step(self, model: Model) -> tuple[Data, State]:
        x0 = self._state["x0"]
        grad_x = input_gradient(model, x0, expected_n=self._n).to(self._dtype)
        x_strategic = apply_strategic_shift(
            x0, grad_x, epsilon=self._epsilon, strat_features=self._strat_features
        )
        x0_next = (1.0 - self._eta) * x0 + self._eta * x_strategic
        return {"x": x_strategic, "y": self._y}, {"x0": x0_next}
