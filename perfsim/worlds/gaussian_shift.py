"""GaussianShiftWorld: stateless location-shift world for the canonical
closed-form fixed-point convergence test.

Performative map: given deployed parameter vector theta in R^d,

    x ~ N(0, I_d)
    y = x . (A theta + b) + sigma * N(0, 1)

For a linear regressor model(x) = x . theta_model with MSE, the population
risk minimizer is `A theta_deployed + b`. RRM iterates theta_{t+1} =
A theta_t + b; the closed-form fixed point is `theta* = (I - A)^-1 b`.

Implements the `DifferentiableWorld` trait via `grad_sample`, exposing
partial-D / partial-theta through a reparameterized sample (noise drawn
without grad, theta path autograd-traceable).
"""

from __future__ import annotations

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.core.world import StatelessWorld


class GaussianShiftWorld(StatelessWorld):
    """Stateless D(theta) = N(A theta + b, Sigma) over regression targets.

    The deployed theta is read from the model as a flat parameter vector
    (matching `Model.get_params()` order). For `LinearModel(in_features=d,
    out_features=1, bias=False)`, theta is the d-dim weight vector and the
    closed-form FP is `(I - A)^-1 b`.
    """

    def __init__(
        self,
        A: Tensor,
        b: Tensor,
        sigma_noise: float = 0.01,
        batch_size: int = 256,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {tuple(A.shape)}")
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise ValueError(
                f"b must be 1-D with length matching A; got A {tuple(A.shape)}, b {tuple(b.shape)}"
            )
        self._d = int(A.shape[0])
        self._A = A.to(dtype=dtype).clone()
        self._b = b.to(dtype=dtype).clone()
        self._sigma = float(sigma_noise)
        self._batch_size = int(batch_size)
        self._dtype = dtype

    @property
    def produces_schema(self) -> DataSchema:
        return SUPERVISED_SCHEMA

    @property
    def dim(self) -> int:
        return self._d

    def _sample_batch(self, model: Model, generator: torch.Generator) -> Data:
        theta = model.get_params().detach().to(self._dtype)
        if theta.numel() != self._d:
            raise ValueError(
                f"model has {theta.numel()} params but World expects d={self._d}"
            )
        x = torch.randn(
            self._batch_size, self._d, generator=generator, dtype=self._dtype
        )
        target = self._A @ theta + self._b  # (d,)
        eps = torch.randn(self._batch_size, generator=generator, dtype=self._dtype)
        y = x @ target + self._sigma * eps  # (B,)
        return {"x": x, "y": y.unsqueeze(-1)}

    def grad_sample(self, model: Model) -> Data:
        """Autograd-traceable sample. Noise drawn without grad; theta path
        retains autograd so derivative-aware Learners can backprop.
        """
        if self._gen is None:
            self.reset(seed=0)
        assert self._gen is not None
        forked = torch.Generator()
        forked.set_state(self._gen.get_state())
        with torch.no_grad():
            x = torch.randn(
                self._batch_size, self._d, generator=forked, dtype=self._dtype
            )
            eps = torch.randn(
                self._batch_size, generator=forked, dtype=self._dtype
            )
        theta = torch.cat([p.reshape(-1) for p in model.parameters()])
        if theta.numel() != self._d:
            raise ValueError(
                f"model has {theta.numel()} params but World expects d={self._d}"
            )
        target = self._A @ theta + self._b
        y = x @ target + self._sigma * eps
        return {"x": x, "y": y.unsqueeze(-1)}

    def closed_form_fp(self) -> Tensor:
        """Closed-form RRM fixed point: theta* = (I - A)^-1 b."""
        eye = torch.eye(self._d, dtype=self._dtype)
        return torch.linalg.solve(eye - self._A, self._b)
