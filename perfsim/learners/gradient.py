"""Gradient-step Learner: k SGD or Adam steps per round."""

from __future__ import annotations

from typing import ClassVar

import torch

from perfsim.core.learner import Learner
from perfsim.core.loss import Loss
from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema


class GradientLearner(Learner):
    """k SGD or Adam steps per round."""

    accepted_schemas: ClassVar[tuple[DataSchema, ...]] = (SUPERVISED_SCHEMA,)

    def __init__(
        self,
        model: Model,
        loss: Loss,
        *,
        lr: float = 0.01,
        steps_per_round: int = 1,
        optimizer: str = "sgd",
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ) -> None:
        super().__init__(model, loss)
        self.lr = lr
        self.steps_per_round = steps_per_round
        self.optimizer_name = optimizer.lower()
        self.weight_decay = weight_decay
        self.momentum = momentum
        self._initial_params = self.model.get_params().clone()
        self._opt = self._make_optimizer()

    def _make_optimizer(self) -> torch.optim.Optimizer:
        params = list(self.model.parameters())
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        if self.optimizer_name == "adam":
            return torch.optim.Adam(
                params, lr=self.lr, weight_decay=self.weight_decay
            )
        raise ValueError(
            f"unknown optimizer {self.optimizer_name!r}; expected 'sgd' or 'adam'"
        )

    def train(self, data: Data) -> None:
        for _ in range(self.steps_per_round):
            self._opt.zero_grad()
            value = self.loss(self.model, data, reduction="mean")
            value.backward()
            self._opt.step()

    def reset(self) -> None:
        self.model.set_params(self._initial_params.clone())
        self._opt = self._make_optimizer()
