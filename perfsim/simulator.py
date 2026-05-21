"""Simulator: epoch-loop orchestration for performative-prediction simulations.

v0 entry point. Drives the Environment / Predictor epoch loop directly; the
agent-shell layer (`agents/`, `core/executor.py`) is preserved scaffolding
and not on this hot path (DESIGN.md §18).

Binding (at __init__): checks Learner.accepted_schemas against
Environment.produces_schema and raises SchemaError on mismatch
(DESIGN.md §6 #7).

Epoch semantics (DESIGN.md §8):
    for t in range(n_rounds):
        handle = predictor.deploy()                # snapshot of theta_t
        for _ in range(epoch_size):                # inner loop, theta frozen
            final_data = env.step(handle)
        predictor.train(final_data)                # one update at epoch end
        record(t)

With `epoch_size=1` the loop reduces to classical lockstep PP and matches
the prior round-loop semantics tensor-for-tensor.

"Round" and "epoch" are synonymous in this design; `n_rounds` is retained
as the public parameter name for backward compatibility.

Constructor accepts two equivalent forms:
- `Simulator(env, learner, loss, ...)`: legacy triplet; a `Predictor` is
  constructed internally from `(learner.model, loss, learner)`.
- `Simulator(env, predictor=..., ...)`: canonical form (DESIGN.md §4).

Either form is fine; they produce identical behavior. Tests still using the
triplet form continue to work without modification.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from perfsim.core.dataset import Dataset
from perfsim.core.environment import Environment
from perfsim.core.learner import Learner
from perfsim.core.loss import Loss
from perfsim.core.predictor import Predictor
from perfsim.core.types import SchemaError
from perfsim.history import History
from perfsim.metrics import stability_gap


MetricFn = Callable[["Simulator"], Any]


class Simulator:
    """Epoch-loop orchestration."""

    def __init__(
        self,
        world: Environment | None = None,
        learner: Learner | None = None,
        loss: Loss | None = None,
        *,
        env: Environment | None = None,
        predictor: Predictor | None = None,
        metrics: Optional[dict[str, MetricFn]] = None,
        history: Optional[History] = None,
        dataset: Optional[Dataset] = None,
    ) -> None:
        env_arg = env if env is not None else world
        if env_arg is None:
            raise TypeError("Simulator requires `env=` (or legacy positional `world=`)")
        if predictor is None:
            if learner is None or loss is None:
                raise TypeError(
                    "Simulator: pass either `predictor=` or both `learner=` and `loss=`"
                )
            predictor = Predictor(model=learner.model, loss=loss, learner=learner)
        elif learner is not None or loss is not None:
            raise TypeError(
                "Simulator: pass either `predictor=` or `(learner=, loss=)`, not both"
            )

        self.env: Environment = env_arg
        self.predictor: Predictor = predictor
        self.metrics: dict[str, MetricFn] = metrics or {}
        self.history = history or History()
        self.dataset = dataset
        self._prev_theta: Tensor | None = None
        self._current_round: int = -1
        self._bind()

    # ---- Backward-compatible properties ------------------------------------
    # These let existing code (tests, scenarios) continue to read sim.world,
    # sim.learner, sim.loss without modification.

    @property
    def world(self) -> Environment:
        return self.env

    @property
    def learner(self) -> Learner:
        return self.predictor.learner

    @property
    def loss(self) -> Loss:
        return self.predictor.loss

    # ---- Binding and validation --------------------------------------------

    def _bind(self) -> None:
        """Validate the Environment's produced schema is accepted by the Learner."""
        produces = self.env.produces_schema
        learner = self.predictor.learner
        if not type(learner).accepts(produces):
            raise SchemaError(
                f"Binding error: Learner {type(learner).__name__} does not "
                f"accept Environment's schema {produces.name!r}. Learner accepts: "
                f"{[s.name for s in learner.accepted_schemas]}."
            )

    def _validate_epoch_size(self, epoch_size: int) -> None:
        if not isinstance(epoch_size, int) or epoch_size < 1:
            raise ValueError(f"epoch_size must be a positive int; got {epoch_size!r}")
        max_size = getattr(self.env, "max_meaningful_epoch_size", math.inf)
        if epoch_size > max_size:
            raise ValueError(
                f"epoch_size={epoch_size} exceeds {type(self.env).__name__}."
                f"max_meaningful_epoch_size={max_size}. This Environment's inner "
                f"step is not meaningful for N>1 under fixed theta (DESIGN.md §8)."
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

        With `epoch_size=1` (default), each round is one env.step + one
        predictor.train, matching the prior lockstep semantics tensor-for-tensor.
        With `epoch_size>1`, env.step is invoked N times under the frozen
        deployed handle; only the final step's data is passed to predictor.train
        (DESIGN.md §8, final-state-only training).
        """
        self._validate_epoch_size(epoch_size)
        self.env.reset(seed=seed)
        self._prev_theta = None
        self._current_round = -1
        for t in range(n_rounds):
            self._current_round = t
            handle = self.predictor.deploy()
            final_data: dict[str, Tensor] | None = None
            for _ in range(epoch_size):
                final_data = self.env.step(handle)
            assert final_data is not None  # epoch_size >= 1 ensures this
            self.predictor.train(final_data)
            self._record_round(t)
        return self.history

    def _record_round(self, t: int) -> None:
        theta = self.predictor.model.get_params().detach().cpu()
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
