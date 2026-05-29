"""Environment ABCs and capability traits: the performative map D(theta)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor

from perfsim.core.types import Data, DataSchema

if TYPE_CHECKING:
    from perfsim.core.model import Model


class Environment(ABC):
    """Performative map D(theta): a deployed model produces data to train on."""

    max_meaningful_epoch_size: ClassVar[int | float] = float("inf")

    @property
    @abstractmethod
    def produces_schema(self) -> DataSchema: ...

    @abstractmethod
    def reset(self, seed: int = 0) -> None:
        """Re-initialize RNG and any internal state."""

    @abstractmethod
    def sample(self, model: "Model") -> Data:
        """Peek at D(theta) without mutating state (off-policy evaluation)."""

    @abstractmethod
    def step(self, model: "Model") -> Data:
        """Advance internal state one step under the deployed model."""

    def run(self, model: "Model", n_steps: int) -> Data:
        """Query the model once, evolve n_steps, return the final data dict.

        Default loops `step`; envs whose `step` re-queries the model override
        this to query once and evolve internally.
        """
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive int; got {n_steps!r}")
        final: Data | None = None
        for _ in range(n_steps):
            final = self.step(model)
        assert final is not None
        return final


class Dynamics(Environment):
    """Marker ABC for dynamical-systems environments (stateless or stateful)."""


class StatelessDynamics(Dynamics):
    """Stateless dynamics: D(theta) is history-independent, IID per step."""

    def __init__(self) -> None:
        self._gen: torch.Generator | None = None

    @abstractmethod
    def _sample_batch(self, model: "Model", generator: torch.Generator) -> Data:
        """Produce one batch from D(theta) using the given generator."""

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


class StatefulDynamics(Dynamics):
    """Stateful dynamics: subclasses implement sample, step, reset fully."""


class AgentBased(Environment):
    """Population of stateful agents with per-agent decision rules (stub)."""


@runtime_checkable
class Differentiable(Protocol):
    """Environment whose single-step sample is differentiable wrt theta."""

    def grad_sample(self, model: object) -> "Data": ...


@runtime_checkable
class FullyDifferentiable(Protocol):
    """Environment whose step and population updates are end-to-end autograd-traceable."""

    def grad_sample(self, model: object) -> "Data": ...

    def grad_step(self, model: object) -> "Data": ...


@runtime_checkable
class Rewarding(Protocol):
    """Environment that fills a `reward` field; required by RL learners."""

    @property
    def produces_reward(self) -> bool: ...


@runtime_checkable
class Trajectory(Protocol):
    """Environment that produces multi-step trajectory tensors."""

    @property
    def trajectory_length(self) -> int: ...


@runtime_checkable
class ClosedFormFixedPoint(Protocol):
    """Environment with an analytic closed-form RRM fixed point (gating tests)."""

    def closed_form_fp(self) -> Tensor: ...
