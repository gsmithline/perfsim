"""Learner ABC.

Owns the Model and any optimizer state. Updates Model in place each round
via `step(data)`. Declares `accepted_schemas` as a class attribute so
Simulator.bind() can check World-Learner compatibility at binding time
rather than at the first hot-path call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from perfsim.core.loss import Loss
from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema


class Learner(ABC):
    """Base Learner. Owns Model + optimizer state.

    Subclasses declare `accepted_schemas` (class attribute) and implement
    `step(data)` and `reset()`. Lifecycle:

    - `__init__(model, loss)`: bind components.
    - `step(data)`: many calls; mutates `self.model`.
    - `reset()`: optional, restores model params and clears optimizer state.
      ERM doesn't need it across rounds; gradient-based learners with
      stateful optimizers (Adam, SGD-momentum) need it for fresh
      retraining.
    """

    accepted_schemas: ClassVar[tuple[DataSchema, ...]] = (SUPERVISED_SCHEMA,)

    def __init__(self, model: Model, loss: Loss) -> None:
        self.model = model
        self.loss = loss

    @abstractmethod
    def train(self, data: Data) -> None:
        """Run one update; mutates `self.model` in place.

        Called once per epoch by the Simulator (DESIGN.md §8). For epoch_size=1
        this is the classical per-round PP update; for epoch_size>1 the `data`
        argument is the final state of the environment's inner N-step run.
        """

    def step(self, data: Data) -> None:
        """Legacy alias for `train`. Prefer `train` in new code.

        Retained so existing callsites and tests continue to work during the
        v0 transition (DESIGN.md §17).
        """
        self.train(data)

    @abstractmethod
    def reset(self) -> None:
        """Re-initialize model parameters and any optimizer state."""

    @classmethod
    def accepts(cls, produces: DataSchema) -> bool:
        """True if a World producing `produces` is compatible with this Learner."""
        return any(produces.covers(s) for s in cls.accepted_schemas)
