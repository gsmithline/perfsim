"""Performative gradient descent learners."""

from __future__ import annotations

import copy
from collections import deque
from typing import Callable, ClassVar

import torch
from torch import Tensor

from perfsim.core.environment import Environment
from perfsim.core.learner import Learner
from perfsim.core.loss import Loss
from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema

PopulationLossFn = Callable[[Environment], Tensor]


class PerfGDLearner(Learner):
    """Performative gradient descent via backprop through a differentiable environment.

    Each train() call runs the env with gradients live, computes a
    population-level loss on the outcome, and backprops to the model params.
    Requires the environment to support grad_run().
    """

    accepted_schemas: ClassVar[tuple[DataSchema, ...]] = (SUPERVISED_SCHEMA,)

    def __init__(
        self,
        model: Model,
        loss: Loss,
        env: Environment,
        population_loss_fn: PopulationLossFn,
        *,
        lr: float = 0.01,
        steps_per_round: int = 1,
        env_steps: int = 1,
        optimizer: str = "adam",
    ) -> None:
        super().__init__(model, loss)
        self.env = env
        self.population_loss_fn = population_loss_fn
        self.lr = lr
        self.steps_per_round = steps_per_round
        self.env_steps = env_steps
        self._optimizer = self._build_optimizer(optimizer)

    def _build_optimizer(self, name: str) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if name == "adam":
            return torch.optim.Adam(params, lr=self.lr)
        if name == "sgd":
            return torch.optim.SGD(params, lr=self.lr)
        raise ValueError(f"unknown optimizer: {name!r}")

    def train(self, data: Data) -> None:
        for _ in range(self.steps_per_round):
            self._optimizer.zero_grad()
            if hasattr(self.env, "grad_run"):
                self.env.grad_run(self.model, n_steps=self.env_steps)
                pop_loss = self.population_loss_fn(self.env)
            else:
                data = self.env.grad_sample(self.model)
                pop_loss = self.population_loss_fn(self.env, data)
            pop_loss.backward()
            self._optimizer.step()

    def reset(self) -> None:
        pass


class PerfGDFiniteDiffLearner(Learner):
    """Performative gradient descent via finite differences (Perdomo et al. 2020).

    Estimates the performative gradient by perturbing each model parameter,
    running the environment, and computing central differences on the
    population loss. No differentiable environment needed.
    """

    accepted_schemas: ClassVar[tuple[DataSchema, ...]] = (SUPERVISED_SCHEMA,)

    def __init__(
        self,
        model: Model,
        loss: Loss,
        env: Environment,
        population_loss_fn: PopulationLossFn,
        *,
        lr: float = 0.01,
        eps: float = 0.1,
        env_steps: int = 1,
        n_seeds: int = 3,
    ) -> None:
        super().__init__(model, loss)
        self.env = env
        self.population_loss_fn = population_loss_fn
        self.lr = lr
        self.eps = eps
        self.env_steps = env_steps
        self.n_seeds = n_seeds
        self._step_count = 0

    def _eval_loss(self, seed: int) -> float:
        self.env.reset(seed=seed)
        self.env.run(self.model, n_steps=self.env_steps)
        with torch.no_grad():
            return float(self.population_loss_fn(self.env).item())

    def _eval_loss_averaged(self) -> float:
        total = sum(self._eval_loss(s) for s in range(self.n_seeds))
        return total / self.n_seeds

    def train(self, data: Data) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        flat = torch.cat([p.detach().reshape(-1) for p in params])
        grad = torch.zeros_like(flat)

        for i in range(flat.numel()):
            flat_plus = flat.clone()
            flat_plus[i] += self.eps
            self._set_flat_params(params, flat_plus)
            loss_plus = self._eval_loss_averaged()

            flat_minus = flat.clone()
            flat_minus[i] -= self.eps
            self._set_flat_params(params, flat_minus)
            loss_minus = self._eval_loss_averaged()

            grad[i] = (loss_plus - loss_minus) / (2 * self.eps)

        # Restore original params and apply gradient step
        new_flat = flat - self.lr * grad
        self._set_flat_params(params, new_flat)
        self._step_count += 1

    @staticmethod
    def _set_flat_params(params: list[torch.nn.Parameter], flat: Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.copy_(flat[offset:offset + n].reshape(p.shape))
                offset += n

    def reset(self) -> None:
        self._step_count = 0
