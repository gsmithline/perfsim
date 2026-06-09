"""A configurable performative map: a Gaussian-mixture base whose features get
a theta-dependent shift. Two independent knobs:

  epsilon     -> sensitivity (how far the distribution moves per unit theta)
  n_modes,    -> modality; n_modes=1 is a single Gaussian (location-scale, so
  separation     mixture dominance holds), n_modes>=2 separated breaks it

It exposes all three levels on one object: samples, the mechanism (base +
transform), and a closed-form density. So the same environment can be handed to
any method, and the two knobs move independently.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor

from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.maps._common import apply_strategic_shift, validate_strat_features
from perfsim.maps.base import DensityMap, ModelView, TransformationMap


class MixtureShiftMap(TransformationMap, DensityMap):
    model_channel = "parameters"

    def __init__(
        self,
        dim: int,
        *,
        n_modes: int = 1,
        separation: float = 3.0,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        label_noise: float = 0.1,
        strat_features: Iterable[int] | None = None,
        w_true: Tensor | None = None,
        weights: Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1; got {dim}")
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1; got {n_modes}")
        if sigma <= 0 or label_noise <= 0:
            raise ValueError("sigma and label_noise must be positive")
        self._d = int(dim)
        self._k = int(n_modes)
        self._sep = float(separation)
        self._epsilon = float(epsilon)
        self._sigma = float(sigma)
        self._label_noise = float(label_noise)
        self._dtype = dtype
        self._strat = validate_strat_features(strat_features, dim=self._d)

        offsets = (torch.arange(self._k, dtype=dtype) - (self._k - 1) / 2) * self._sep
        centers = torch.zeros(self._k, self._d, dtype=dtype)
        centers[:, 0] = offsets
        self._centers = centers
        self._w_true = (torch.ones(self._d, dtype=dtype) if w_true is None
                        else w_true.to(dtype=dtype).clone())
        if weights is None:
            weights = torch.full((self._k,), 1.0 / self._k, dtype=dtype)
        if weights.shape != (self._k,):
            raise ValueError(f"weights must be ({self._k},); got {tuple(weights.shape)}")
        self._weights_p = (weights / weights.sum()).to(dtype=dtype)

    @property
    def produces_schema(self) -> DataSchema:
        return SUPERVISED_SCHEMA

    @property
    def dim(self) -> int:
        return self._d

    @property
    def n_modes(self) -> int:
        return self._k

    @property
    def mode_centers(self) -> Tensor:
        return self._centers.clone()

    def sample_base(self, n: int, *, generator: torch.Generator) -> Data:
        comp = torch.multinomial(self._weights_p, n, replacement=True, generator=generator)
        x = self._centers[comp] + self._sigma * torch.randn(
            n, self._d, generator=generator, dtype=self._dtype
        )
        y = x @ self._w_true + self._label_noise * torch.randn(
            n, generator=generator, dtype=self._dtype
        )
        return {"x": x, "y": y.unsqueeze(-1)}

    def transform(self, z_base: Data, model: ModelView) -> Data:
        direction = self._weights_of(model).expand_as(z_base["x"])
        x = apply_strategic_shift(
            z_base["x"], direction, epsilon=self._epsilon, strat_features=self._strat
        )
        return {"x": x, "y": z_base["y"]}

    def log_prob(self, z: Data, model: "object") -> Tensor:
        view = self.view(model)
        direction = self._weights_of(view).expand_as(z["x"])
        shift = apply_strategic_shift(
            torch.zeros_like(z["x"]), direction, epsilon=self._epsilon,
            strat_features=self._strat,
        )
        x_base = z["x"] - shift
        diff = x_base.unsqueeze(1) - self._centers.unsqueeze(0)          # (n, K, d)
        sq = diff.pow(2).sum(dim=-1)                                     # (n, K)
        log_comp = (-0.5 * sq / self._sigma ** 2
                    - 0.5 * self._d * math.log(2 * math.pi * self._sigma ** 2)
                    + torch.log(self._weights_p).unsqueeze(0))
        log_x = torch.logsumexp(log_comp, dim=1)                        # (n,)
        mean = x_base @ self._w_true
        y = z["y"].reshape(-1)
        log_y = (-0.5 * (y - mean) ** 2 / self._label_noise ** 2
                 - 0.5 * math.log(2 * math.pi * self._label_noise ** 2))
        return log_x + log_y

    def _weights_of(self, view: ModelView) -> Tensor:
        theta = view.params.detach().to(self._dtype)
        if theta.numel() == self._d:
            return theta
        if theta.numel() == self._d + 1:
            return theta[: self._d]
        raise ValueError(
            f"model has {theta.numel()} params but map dim is {self._d}"
        )
