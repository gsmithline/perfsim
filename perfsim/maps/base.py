"""Distribution maps: the central object D: Theta -> Delta(Z).

A DistributionMap is stateless and side-effect free; sampling never mutates
it. Access is the type hierarchy itself: DistributionMap.sample gives samples,
TransformationMap adds the mechanism (base distribution + transform), and
DensityMap adds the density. Maps see the deployed model through a ModelView,
which opens the parameter channel only if the map declares it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor

from perfsim.core.types import Data, DataSchema

if TYPE_CHECKING:
    from perfsim.core.model import Model

_CHANNELS = ("predictions", "parameters")


class AccessError(Exception):
    """Raised when a map or learner touches a channel it did not declare."""


class ModelView:
    """Restricted window onto a deployed model.

    The prediction channel is always open. The parameter channel opens only
    when the owning map declares model_channel = "parameters". All reads are
    detached; gradient access for oracle baselines is a separate trait.
    """

    def __init__(self, model: "Model", *, expose_params: bool) -> None:
        self._model = model
        self._expose_params = bool(expose_params)

    def predict(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self._model(x)

    @property
    def params(self) -> Tensor:
        if not self._expose_params:
            raise AccessError(
                "parameter channel is closed; the map must declare "
                "model_channel='parameters' to read theta"
            )
        return self._model.get_params()


def validate_n(n: int) -> None:
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive int; got {n!r}")


class DistributionMap(ABC):
    """Performative map D(theta). Implementing sample() is the samples level."""

    model_channel: ClassVar[str] = "predictions"

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.model_channel not in _CHANNELS:
            raise TypeError(
                f"model_channel must be one of {_CHANNELS}; got {cls.model_channel!r}"
            )

    @property
    @abstractmethod
    def produces_schema(self) -> DataSchema: ...

    @abstractmethod
    def sample(
        self, model: "Model | ModelView", n: int, *, generator: torch.Generator
    ) -> Data:
        """Draw n samples from D(theta) using the given generator."""

    def view(self, model: "Model | ModelView") -> ModelView:
        """Coerce a model to a view matching this map's declared channel."""
        if isinstance(model, ModelView):
            return model
        return ModelView(model, expose_params=self.model_channel == "parameters")


class TransformationMap(DistributionMap):
    """The mechanism level: a base distribution plus a theta-dependent transform.

    z = phi(z_base; theta) with z_base ~ D_base. sample() is derived, so
    subclasses only write sample_base and transform.
    """

    @abstractmethod
    def sample_base(self, n: int, *, generator: torch.Generator) -> Data:
        """Draw n samples from the model-independent base distribution."""

    @abstractmethod
    def transform(self, z_base: Data, model: ModelView) -> Data:
        """Apply the theta-dependent transformation to base samples."""

    def sample(
        self, model: "Model | ModelView", n: int, *, generator: torch.Generator
    ) -> Data:
        validate_n(n)
        return self.transform(self.sample_base(n, generator=generator), self.view(model))


class DensityMap(DistributionMap):
    """The density level: adds a parameterized per-sample log density."""

    @abstractmethod
    def log_prob(self, z: Data, model: "Model | ModelView") -> Tensor:
        """Per-sample log p(z; theta), shape (n,). Detached in theta."""


def access_levels(map_: DistributionMap) -> frozenset[str]:
    """What a map exposes: samples, mechanism (base + transform), density."""
    levels = {"samples"}
    if isinstance(map_, TransformationMap):
        levels.add("mechanism")
    if isinstance(map_, DensityMap):
        levels.add("density")
    return frozenset(levels)
