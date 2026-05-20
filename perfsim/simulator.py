"""Simulator: round-loop orchestration for performative-prediction simulations.

v0 entry point. Drives the World/Learner round loop directly, no Executor or
agent shell yet (those land in v1 alongside InProcessExecutor and
CoordinatorAgent per DESIGN.md Section 17 phasing).

Binding (at __init__): checks Learner.accepted_schemas against World.produces_schema
and raises SchemaError on mismatch. Single source of truth, caught early.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import Tensor

from perfsim.core.dataset import Dataset
from perfsim.core.learner import Learner
from perfsim.core.loss import Loss
from perfsim.core.types import SchemaError
from perfsim.core.world import World
from perfsim.history import History
from perfsim.metrics import stability_gap


MetricFn = Callable[["Simulator"], Any]


class Simulator:
    """Round-loop orchestration."""

    def __init__(
        self,
        world: World,
        learner: Learner,
        loss: Loss,
        *,
        metrics: Optional[dict[str, MetricFn]] = None,
        history: Optional[History] = None,
        dataset: Optional[Dataset] = None,
    ) -> None:
        self.world = world
        self.learner = learner
        self.loss = loss
        self.metrics: dict[str, MetricFn] = metrics or {}
        self.history = history or History()
        self.dataset = dataset
        self._prev_theta: Tensor | None = None
        self._current_round: int = -1
        self._bind()

    def _bind(self) -> None:
        """Validate the World's produced schema is accepted by the Learner."""
        produces = self.world.produces_schema
        if not type(self.learner).accepts(produces):
            raise SchemaError(
                f"Binding error: Learner {type(self.learner).__name__} does not "
                f"accept World's schema {produces.name!r}. Learner accepts: "
                f"{[s.name for s in self.learner.accepted_schemas]}."
            )

    @property
    def current_round(self) -> int:
        return self._current_round

    def run(self, n_rounds: int, *, seed: int = 0) -> History:
        """Run the PP loop for `n_rounds`. Returns the History."""
        self.world.reset(seed=seed)
        self._prev_theta = None
        self._current_round = -1
        for t in range(n_rounds):
            self._current_round = t
            data = self.world.step(self.learner.model)
            self.learner.step(data)
            self._record_round(t)
        return self.history

    def _record_round(self, t: int) -> None:
        theta = self.learner.model.get_params().detach().cpu()
        record: dict[str, Any] = {"round": t, "theta": theta}
        if self._prev_theta is not None:
            record["stability_gap"] = stability_gap(self._prev_theta, theta)
        with torch.no_grad():
            for name, fn in self.metrics.items():
                record[name] = fn(self)
        if self.dataset is not None:
            record["dataset_hash"] = self.dataset.hash()
        self.history.append(**record)
        self._prev_theta = theta
