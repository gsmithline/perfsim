"""Gaussian regression map with a closed-form RRM fixed point.

x ~ N(0, I), y = x . (A theta + b) + sigma eps. Map version of
GaussianShiftWorld; exposes all three levels (samples, mechanism, density) and
the analytic fixed point theta* = (I - A)^-1 b for gating tests.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from perfsim.maps.base import DensityMap, ModelView, TransformationMap
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.core.model import Model


class GaussianShiftMap(TransformationMap, DensityMap):
    """Linear location shift in regression targets, Gaussian everywhere."""

    model_channel = "parameters"

    def __init__(
        self,
        A: Tensor,
        b: Tensor,
        sigma_noise: float = 0.01,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {tuple(A.shape)}")
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise ValueError(
                f"b must be 1-D with length matching A; got A {tuple(A.shape)}, b {tuple(b.shape)}"
            )
        if sigma_noise <= 0.0:
            raise ValueError(f"sigma_noise must be positive; got {sigma_noise}")
        self._d = int(A.shape[0])
        self._A = A.to(dtype=dtype).clone()
        self._b = b.to(dtype=dtype).clone()
        self._sigma = float(sigma_noise)
        self._dtype = dtype

    @property
    def produces_schema(self) -> DataSchema:
        return SUPERVISED_SCHEMA

    @property
    def dim(self) -> int:
        return self._d

    def sample_base(self, n: int, *, generator: torch.Generator) -> Data:
        x = torch.randn(n, self._d, generator=generator, dtype=self._dtype)
        eps = torch.randn(n, generator=generator, dtype=self._dtype)
        return {"x": x, "eps": eps}

    def transform(self, z_base: Data, model: ModelView) -> Data:
        target = self._target(model)
        y = z_base["x"] @ target + self._sigma * z_base["eps"]
        return {"x": z_base["x"], "y": y.unsqueeze(-1)}

    def log_prob(self, z: Data, model: "Model | ModelView") -> Tensor:
        target = self._target(self.view(model))
        x = z["x"].to(self._dtype)
        y = z["y"].reshape(-1).to(self._dtype)
        mean = x @ target
        var = self._sigma**2
        lp_y = -0.5 * ((y - mean) ** 2 / var + math.log(2.0 * math.pi * var))
        lp_x = -0.5 * (x**2).sum(dim=1) - 0.5 * self._d * math.log(2.0 * math.pi)
        return lp_y + lp_x

    def closed_form_fp(self) -> Tensor:
        """RRM fixed point theta* = (I - A)^-1 b."""
        eye = torch.eye(self._d, dtype=self._dtype)
        return torch.linalg.solve(eye - self._A, self._b)

    def _target(self, model: ModelView) -> Tensor:
        theta = model.params.detach().to(self._dtype)
        if theta.numel() != self._d:
            raise ValueError(
                f"model has {theta.numel()} params but map expects d={self._d}"
            )
        return self._A @ theta + self._b
