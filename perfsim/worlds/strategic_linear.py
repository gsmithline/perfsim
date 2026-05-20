"""StrategicLinearWorld: stateful agent-based world with linear strategic
best-response (Perdomo et al. ICML 2020).

Each agent has fixed initial features `x_0` and a fixed label `y`. On each
round, agents shift their features by `epsilon * w` (first-order
best-response of a linear utility against a quadratic feature-shift cost),
where `w` is the deployed linear classifier's weight vector. Returns the
shifted features as the round's training data.

If `strat_features` is set, only those feature indices are shifted; the
others remain at their initial values. This matches Perdomo's notebook
where only three of the GMSC columns can be strategically manipulated.

"Stateful" in the agent-based sense: the population (x_0, y) is
materialized once at init and persists across rounds. State does not evolve
beyond the strategic shift.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.core.world import StatefulWorld


class StrategicLinearWorld(StatefulWorld):
    """Perdomo-style linear strategic best-response."""

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
        if strat_features is None:
            self._strat_features: Tensor | None = None
        else:
            idx = torch.tensor(list(strat_features), dtype=torch.long)
            if idx.numel() == 0:
                raise ValueError("strat_features cannot be empty when set; pass None for all-features")
            if int(idx.min().item()) < 0:
                raise ValueError(
                    f"strat_features must be non-negative; got min={int(idx.min().item())}"
                )
            if int(idx.max().item()) >= self._d:
                raise ValueError(
                    f"strat_features max index {int(idx.max().item())} >= d={self._d}"
                )
            if idx.unique().numel() != idx.numel():
                raise ValueError(f"strat_features must be unique; got {idx.tolist()}")
            self._strat_features = idx

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

    def _weight_vector(self, model: Model) -> Tensor:
        """Extract the linear weight vector from a model.

        Convention: model exposes a `linear` attribute of type nn.Linear
        (`LinearModel`, `LogisticModel`).
        """
        if not hasattr(model, "linear"):
            raise TypeError(
                "StrategicLinearWorld expects a model with a `.linear` "
                "attribute (LinearModel, LogisticModel)."
            )
        return model.linear.weight.detach().reshape(-1).to(self._dtype)

    def sample(self, model: Model) -> Data:
        w = self._weight_vector(model)
        if w.numel() != self._d:
            raise ValueError(
                f"model weight has {w.numel()} elements but population dim is {self._d}"
            )
        if self._strat_features is None:
            x = self._x0 + self._epsilon * w
        else:
            shift = torch.zeros_like(self._x0)
            shift[:, self._strat_features] = self._epsilon * w[self._strat_features]
            x = self._x0 + shift
        return {"x": x, "y": self._y}

    def step(self, model: Model) -> Data:
        return self.sample(model)
