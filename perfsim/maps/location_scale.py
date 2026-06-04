"""Location-scale family (Miller et al. 2021; survey eq 17).

z = (Sigma0 + Sigma(theta)) z_base + mu0 + M theta, with Sigma(theta) linear
in theta. The base population is a fixed dataset resampled with replacement.
"""

from __future__ import annotations

import torch
from torch import Tensor

from perfsim.maps.base import ModelView, TransformationMap
from perfsim.core.types import FEATURES_SCHEMA, SUPERVISED_SCHEMA, Data, DataSchema


class LocationScaleMap(TransformationMap):
    """Linear-in-theta location and scale shift of the feature block."""

    model_channel = "parameters"

    def __init__(
        self,
        x0: Tensor,
        y: Tensor | None = None,
        *,
        M: Tensor | None = None,
        mu0: Tensor | None = None,
        S: Tensor | None = None,
        Sigma0: Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if x0.ndim != 2:
            raise ValueError(f"x0 must be 2-D (N, D); got {tuple(x0.shape)}")
        if y is not None and y.shape[0] != x0.shape[0]:
            raise ValueError(
                f"y leading dim {y.shape[0]} does not match x0 leading dim {x0.shape[0]}"
            )
        if M is None and S is None:
            raise ValueError("at least one of M (location) or S (scale) must be set")
        self._n, self._d = x0.shape
        self._x0 = x0.to(dtype=dtype).clone()
        self._y = None if y is None else y.clone()
        self._dtype = dtype

        p: int | None = None
        if M is not None:
            if M.ndim != 2 or M.shape[0] != self._d:
                raise ValueError(f"M must be (D, P) with D={self._d}; got {tuple(M.shape)}")
            p = int(M.shape[1])
        if S is not None:
            if S.ndim != 3 or S.shape[1:] != (self._d, self._d):
                raise ValueError(
                    f"S must be (P, D, D) with D={self._d}; got {tuple(S.shape)}"
                )
            if p is not None and S.shape[0] != p:
                raise ValueError(f"S has P={S.shape[0]} but M has P={p}")
            p = int(S.shape[0])
        assert p is not None
        self._p = p
        self._M = None if M is None else M.to(dtype=dtype).clone()
        self._S = None if S is None else S.to(dtype=dtype).clone()
        self._mu0 = (
            torch.zeros(self._d, dtype=dtype) if mu0 is None else mu0.to(dtype=dtype).clone()
        )
        self._Sigma0 = (
            torch.eye(self._d, dtype=dtype)
            if Sigma0 is None
            else Sigma0.to(dtype=dtype).clone()
        )
        if self._mu0.shape != (self._d,):
            raise ValueError(f"mu0 must be (D,)={self._d}; got {tuple(self._mu0.shape)}")
        if self._Sigma0.shape != (self._d, self._d):
            raise ValueError(
                f"Sigma0 must be (D, D)={self._d}; got {tuple(self._Sigma0.shape)}"
            )

    @property
    def produces_schema(self) -> DataSchema:
        return FEATURES_SCHEMA if self._y is None else SUPERVISED_SCHEMA

    @property
    def dim(self) -> int:
        return self._d

    @property
    def theta_dim(self) -> int:
        return self._p

    def sample_base(self, n: int, *, generator: torch.Generator) -> Data:
        idx = torch.randint(self._n, (n,), generator=generator)
        out: Data = {"x": self._x0[idx]}
        if self._y is not None:
            out["y"] = self._y[idx]
        return out

    def transform(self, z_base: Data, model: ModelView) -> Data:
        theta = model.params.detach().to(self._dtype)
        if theta.numel() != self._p:
            raise ValueError(
                f"model has {theta.numel()} params but map expects P={self._p}"
            )
        sigma = self._Sigma0
        if self._S is not None:
            sigma = sigma + torch.einsum("p,pij->ij", theta, self._S)
        mu = self._mu0
        if self._M is not None:
            mu = mu + self._M @ theta
        out = dict(z_base)
        out["x"] = z_base["x"] @ sigma.T + mu
        return out
