"""Predictor: facade over (Model, Loss, Learner), the deployed "platform"."""

from __future__ import annotations

from torch import Tensor

from perfsim.core.learner import Learner
from perfsim.core.loss import Loss
from perfsim.core.model import Model
from perfsim.core.types import Data


class Predictor:
    """Facade over (Model, Loss, Learner).

    The Learner must own the same Model instance, since Learner.train mutates
    it in place; distinct instances would split predictions from updates.
    """

    def __init__(self, model: Model, loss: Loss, learner: Learner) -> None:
        if learner.model is not model:
            raise ValueError(
                "Predictor: learner.model must be the same object as the "
                "supplied model; got distinct instances. Pass the same Model "
                "to both the Learner and the Predictor."
            )
        self._model = model
        self._loss = loss
        self._learner = learner

    @property
    def model(self) -> Model:
        return self._model

    @property
    def loss(self) -> Loss:
        return self._loss

    @property
    def learner(self) -> Learner:
        return self._learner

    def predict(self, x: Tensor) -> Tensor:
        """Forward pass through the model. Convenience wrapper around `model(x)`."""
        return self._model(x)

    def train(self, data: Data) -> None:
        """One training call on epoch-final data; mutates the model."""
        self._learner.train(data)

    def deploy(self) -> Model:
        """Return the deployed handle (the model) for the Environment to query."""
        return self._model
