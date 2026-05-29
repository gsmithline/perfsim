"""StrategicLinearWorld: linear strategic best-response (Perdomo et al. 2020).

Each round agents shift features by epsilon * w (first-order best-response of a
linear utility against a quadratic shift cost), w = the deployed classifier's
weights. If strat_features is set, only those indices shift. The population
(x_0, y) is fixed at init and persists across rounds.
"""

from __future__ import annotations

from typing import ClassVar, Iterable

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.core.environment import StatefulDynamics
from perfsim.environments.dynamics._common import apply_strategic_shift, validate_strat_features


class StrategicLinearWorld(StatefulDynamics):
    """Perdomo-style linear strategic best-response; forces epoch_size=1."""

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

    def _weight_vector(self, model: Model) -> Tensor:
        """Extract the linear weight vector (model must expose `.linear`)."""
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
        direction = w.expand_as(self._x0)
        x = apply_strategic_shift(
            self._x0, direction, epsilon=self._epsilon, strat_features=self._strat_features
        )
        return {"x": x, "y": self._y, "agent_idx": self._agent_idx.clone()}

    def step(self, model: Model) -> Data:
        return self.sample(model)
