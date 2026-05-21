"""AccumulatingShiftWorld: strategic agents whose initial features drift.

The Hardt-Megiddo-Papadimitriou-Wootters (2016) / Perdomo et al. (2020)
strategic-classification setup assumes agents' "natural" features x_0 are
fixed across rounds; only their strategic response x_t depends on the
deployed predictor. This world relaxes that: at the end of each round,
x_0 partially drifts toward the strategic position x_t, modeling agents
who gradually internalize their past manipulations:

    grad         = ∂ f(x_0^t; θ_t) / ∂x
    x_strategic  = x_0^t + epsilon * grad                  (this round's data)
    x_0^{t+1}    = (1 - eta) * x_0^t + eta * x_strategic   (drift)

`eta = 0` recovers the static strategic-classification setup (this
reduces to `StrategicGradientWorld`). `eta = 1` means the population
fully adopts each round's strategic position as the new baseline.
Intermediate values give a sticky population.

Sign convention is identical to `StrategicGradientWorld`: positive
`epsilon` shifts agents up the predictor's input gradient. For
Perdomo-style strategic-loan setups where agents try to *lower* their
predicted default risk, pass `epsilon = -mu`.
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
