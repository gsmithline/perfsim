"""Strategic classification map (Perdomo et al. 2020; survey eq 13).

Linear utility against quadratic cost gives the closed-form best response
x = x_base + epsilon * w, optionally restricted to mutable features. Labels
persist; only features move. Map version of StrategicLinearWorld.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from perfsim.maps.base import ModelView, TransformationMap
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.maps._common import apply_strategic_shift, validate_strat_features


class StrategicLinearMap(TransformationMap):
    """Quadratic-cost linear best response on a fixed labeled population."""

    model_channel = "parameters"

    def __init__(
        self,
        x0: Tensor,
        y: Tensor,
        *,
        epsilon: float = 1.0,
        strat_features: Iterable[int] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
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

    @property
    def produces_schema(self) -> DataSchema:
        return SUPERVISED_SCHEMA

    @property
    def dim(self) -> int:
        return self._d

    @property
    def n_population(self) -> int:
        return self._n

    def sample_base(self, n: int, *, generator: torch.Generator) -> Data:
        idx = torch.randint(self._n, (n,), generator=generator)
        return {"x": self._x0[idx], "y": self._y[idx], "agent_idx": idx}

    def transform(self, z_base: Data, model: ModelView) -> Data:
        w = self._weights(model)
        direction = w.expand_as(z_base["x"])
        out = dict(z_base)
        out["x"] = apply_strategic_shift(
            z_base["x"], direction, epsilon=self._epsilon, strat_features=self._strat_features
        )
        return out

    def _weights(self, model: ModelView) -> Tensor:
        # Flat layout for LinearModel is weight then bias; drop a trailing bias.
        theta = model.params.detach().to(self._dtype)
        if theta.numel() == self._d:
            return theta
        if theta.numel() == self._d + 1:
            return theta[: self._d]
        raise ValueError(
            f"model has {theta.numel()} params but population dim is {self._d}"
        )
