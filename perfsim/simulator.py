"""Simulator: epoch-loop orchestration for performative-prediction simulations.

v0 entry point. Drives the Environment / Predictor epoch loop directly; the
agent-shell layer (`agents/`, `core/executor.py`) is preserved scaffolding
and not on this hot path (DESIGN.md §18).

Binding (at __init__): checks Learner.accepted_schemas against World.produces_schema
and raises SchemaError on mismatch (DESIGN.md §6 #7).

Epoch semantics (DESIGN.md §8):
    for t in range(n_rounds):
        for _ in range(epoch_size):                # inner loop, theta frozen
            final_data = world.step(learner.model)
        learner.train(final_data)                  # one update at epoch end
        record(t)

With `epoch_size=1` the loop reduces to classical lockstep PP and matches
the prior round-loop semantics tensor-for-tensor.

"Round" and "epoch" are synonymous in this design; `n_rounds` is retained
as the public parameter name for backward compatibility.
"""

from __future__ import annotations

import math
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
    """Epoch-loop orchestration."""

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

    def _validate_epoch_size(self, epoch_size: int) -> None:
        if not isinstance(epoch_size, int) or epoch_size < 1:
            raise ValueError(f"epoch_size must be a positive int; got {epoch_size!r}")
        max_size = getattr(self.world, "max_meaningful_epoch_size", math.inf)
        if epoch_size > max_size:
            raise ValueError(
                f"epoch_size={epoch_size} exceeds {type(self.world).__name__}."
                f"max_meaningful_epoch_size={max_size}. This World's inner step is "
                f"not meaningful for N>1 under fixed theta (DESIGN.md §8)."
            )

    @property
    def current_round(self) -> int:
        return self._current_round

    def run(
        self,
        n_rounds: int,
        *,
        epoch_size: int = 1,
        seed: int = 0,
    ) -> History:
        """Run the PP loop for `n_rounds` epochs of `epoch_size` env steps each.

        With `epoch_size=1` (default), each round is one world.step + one
        learner.train, matching the prior lockstep semantics tensor-for-tensor.
        With `epoch_size>1`, world.step is invoked N times under the frozen
        deployed model; only the final step's data is passed to learner.train
        (DESIGN.md §8, final-state-only training).
        """
        self._validate_epoch_size(epoch_size)
        self.world.reset(seed=seed)
        self._prev_theta = None
        self._current_round = -1
        for t in range(n_rounds):
            self._current_round = t
            final_data: dict[str, Tensor] | None = None
            for _ in range(epoch_size):
                final_data = self.world.step(self.learner.model)
            assert final_data is not None  # epoch_size >= 1 ensures this
            self.learner.train(final_data)
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
