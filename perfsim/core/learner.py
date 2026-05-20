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
    def step(self, data: Data) -> None:
        """Run one update; mutates `self.model` in place."""

    @abstractmethod
    def reset(self) -> None:
        """Re-initialize model parameters and any optimizer state."""

    @classmethod
    def accepts(cls, produces: DataSchema) -> bool:
        """True if a World producing `produces` is compatible with this Learner."""
        return any(produces.covers(s) for s in cls.accepted_schemas)
