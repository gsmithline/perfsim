"""World ABCs and capability traits.

`World` is the performative map D(theta): given a deployed Model, produce a
data dict the Learner can train on. Two flavors:

- `StatelessWorld`: D(theta) is history-independent. Each round samples IID
  from D(theta_t). The base provides a forked-generator pattern so `sample`
  (peek) does not advance the RNG that `step` uses, keeping off-policy
  evaluation hermetic.
- `StatefulWorld`: D may depend on history or carry population state.
  Subclasses fully implement `sample` (peek) and `step` (advance).

Capability traits below are runtime-checkable Protocols. Learners and
Metrics use them at binding time to declare optional requirements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

from perfsim.core.types import Data, DataSchema

if TYPE_CHECKING:
    from perfsim.core.model import Model


class World(ABC):
    """Base World: performative map D(theta).

    Subclasses implement `sample` (peek; no state mutation) and `step`
    (advance state + return data) plus `reset(seed)`.
    """

    @property
    @abstractmethod
    def produces_schema(self) -> DataSchema: ...

    @abstractmethod
    def reset(self, seed: int = 0) -> None:
        """Re-initialize RNG and any internal state."""

    @abstractmethod
    def sample(self, model: "Model") -> Data:
        """Sample from D(theta) without mutating internal state.

        Used for off-policy evaluation (decoupled performative risk).
        """

    @abstractmethod
    def step(self, model: "Model") -> Data:
        """Sample from D(theta) and advance state."""


class StatelessWorld(World):
    """Base for stateless worlds: D(theta) is history-independent.

    Subclasses implement `_sample_batch(model, generator)`; the base provides
    `sample` (forked-generator peek) and `step` (advance).
    """

    def __init__(self) -> None:
        self._gen: torch.Generator | None = None

    @abstractmethod
    def _sample_batch(self, model: "Model", generator: torch.Generator) -> Data:
        """Produce one batch of data from D(theta) using the given generator."""

    def reset(self, seed: int = 0) -> None:
        self._gen = torch.Generator()
        self._gen.manual_seed(int(seed))

    def sample(self, model: "Model") -> Data:
        if self._gen is None:
            self.reset(seed=0)
        assert self._gen is not None
        forked = torch.Generator()
        forked.set_state(self._gen.get_state())
        return self._sample_batch(model, forked)

    def step(self, model: "Model") -> Data:
        if self._gen is None:
            self.reset(seed=0)
        assert self._gen is not None
        return self._sample_batch(model, self._gen)


class StatefulWorld(World):
    """Base for stateful worlds. Subclasses implement sample, step, reset fully."""


@runtime_checkable
class DifferentiableWorld(Protocol):
    """World whose single-step sample is differentiable wrt theta."""

    def grad_sample(self, model: object) -> "Data": ...


@runtime_checkable
class FullyDifferentiableWorld(Protocol):
    """World whose step and any internal population updates are end-to-end
    autograd-traceable across multiple rounds.

    Stronger than DifferentiableWorld. Implies differentiable surrogates for
    every primitive in the World (Gumbel-softmax for discrete sampling,
    softmax-with-temperature for argmax, reparameterized continuous
    distributions, soft population selection).
    """

    def grad_sample(self, model: object) -> "Data": ...

    def grad_step(self, model: object) -> "Data": ...


@runtime_checkable
class RewardingWorld(Protocol):
    """World that fills a `reward` field in its produced data dict.

    Required by RL-family Learners (v2).
    """

    @property
    def produces_reward(self) -> bool: ...


@runtime_checkable
class TrajectoryWorld(Protocol):
    """World that produces multi-step trajectory tensors with a leading time axis.

    Required by the v2 multi-step Coordinator. The trajectory data schema is a
    v2 placeholder in `core/types.py`.
    """

    @property
    def trajectory_length(self) -> int: ...


@runtime_checkable
class ClosedFormFixedPoint(Protocol):
    """World with an analytic closed-form RRM fixed point.

    Used in gating tests (GaussianShiftWorld) to verify Learners converge to
    the right point.
    """

    def closed_form_fp(self) -> Tensor: ...
