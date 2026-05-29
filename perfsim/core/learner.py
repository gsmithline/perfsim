"""Learner ABC: owns the Model and optimizer state, updates it each round."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from perfsim.core.loss import Loss
from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema


class Learner(ABC):
    """Base Learner. Owns Model + optimizer state; mutates the model in place.

    Subclasses declare `accepted_schemas` and implement `train(data)` and
    `reset()`. `reset()` restores params and clears optimizer state for fresh
    retraining; ERM can skip it, stateful optimizers (Adam, SGD-momentum) need it.
    """

    accepted_schemas: ClassVar[tuple[DataSchema, ...]] = (SUPERVISED_SCHEMA,)

    def __init__(self, model: Model, loss: Loss) -> None:
        self.model = model
        self.loss = loss

    @abstractmethod
    def train(self, data: Data) -> None:
        """Run one update, mutating `self.model`. Called once per epoch.

        For epoch_size>1, `data` is the final state of the env's inner run.
        """

    def step(self, data: Data) -> None:
        """Legacy alias for `train`."""
        self.train(data)

    @abstractmethod
    def reset(self) -> None:
        """Re-initialize model parameters and any optimizer state."""

    @classmethod
    def accepts(cls, produces: DataSchema) -> bool:
        """True if an Environment producing `produces` is compatible."""
        return any(produces.covers(s) for s in cls.accepted_schemas)
