"""Predictor: facade over (Model, Loss, Learner).

The "platform" in performative-prediction terms (DESIGN.md §4-5). Owns the
parameters theta that get deployed to the Environment each epoch, and the
training step that updates theta at epoch end.

The internal Model/Loss/Learner split is preserved because:

- Loss is used by metrics standalone (`metrics.py`).
- Learners are swappable for the same (Model, Loss) pair (ERM, Gradient,
  Proximal, DerivativeAware, RL/LM later).
- Model is reusable across Predictors with different optimizers.

The Predictor adds one externally-visible API on top of that split:

    predict(x) -> y       forward pass through the model
    train(data) -> None   one training call on epoch-final data
    deploy() -> handle    snapshot for the Environment to consume

`deploy()` returns the model directly in v0. The Environment is expected
to query this handle ONCE at the start of `env.run(...)` to produce initial
conditions, then evolve autonomously for the rest of the epoch
(DESIGN.md §8). Inside the inner N-step loop, nothing calls Predictor.train,
so theta is frozen by loop structure (DESIGN.md §6 #9).
"""

from __future__ import annotations

from torch import Tensor

from perfsim.core.learner import Learner
from perfsim.core.loss import Loss
from perfsim.core.model import Model
from perfsim.core.types import Data


class Predictor:
    """Facade over (Model, Loss, Learner).

    The Predictor is constructed by passing in the three components. The
    Learner must own the same Model instance the Predictor is given, since
    Learner.train mutates `learner.model` in place; accepting both invites
    the bug where predictions and updates go to different objects.
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
        """One training call on epoch-final data. Mutates the model.

        Delegated to `learner.train(data)`. Called exactly once per epoch by
        the Simulator at the end of the inner N-step loop.
        """
        self._learner.train(data)

    def deploy(self) -> Model:
        """Return the deployed-predictor handle for the Environment.

        v0: returns the underlying Model. The Environment queries this
        handle once at the start of `env.run(...)` to produce initial
        conditions (predictions, classifier coefficients, fitness landscape),
        then evolves autonomously without re-querying for the remainder of
        the epoch (DESIGN.md §8 publishing pattern).
        """
        return self._model
