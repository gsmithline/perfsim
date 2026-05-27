"""ERM Learner: solves to convergence each round via L-BFGS."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor

from perfsim.core.learner import Learner
from perfsim.core.loss import Loss
from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema


class ERMLearner(Learner):
    """Solves empirical risk minimization to convergence each round via L-BFGS."""

    accepted_schemas: ClassVar[tuple[DataSchema, ...]] = (SUPERVISED_SCHEMA,)

    def __init__(
        self,
        model: Model,
        loss: Loss,
        *,
        max_iter: int = 100,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
    ) -> None:
        super().__init__(model, loss)
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.history_size = history_size
        self._initial_params = self.model.get_params().clone()

    def train(self, data: Data) -> None:
        opt = torch.optim.LBFGS(
            list(self.model.parameters()),
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
            history_size=self.history_size,
        )

        def closure() -> Tensor:
            opt.zero_grad()
            value = self.loss(self.model, data, reduction="mean")
            value.backward()
            return value

        opt.step(closure)

    def reset(self) -> None:
        self.model.set_params(self._initial_params.clone())
