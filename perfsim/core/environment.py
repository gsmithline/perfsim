"""Environment ABCs and capability traits (DESIGN.md §4-5).

`Environment` is the performative map D(theta): given a deployed Predictor
handle (typically the underlying Model), produce a `data` dict the Learner
can train on. Two top-level siblings under Environment:

- `Dynamics`: closed-form / ODE-style updates on a tensor state. Has two
  internal flavors:
  - `StatelessDynamics`: D(theta) is history-independent. Each step samples
    IID from D(theta_t). The base provides a forked-generator pattern so
    `sample` (peek) does not advance the RNG that `step` uses, keeping
    off-policy evaluation hermetic.
  - `StatefulDynamics`: state evolves over the inner N-step loop under
    fixed theta. Subclasses fully implement `sample` (peek) and `step`
    (advance).
- `AgentBased`: population of stateful agent objects with per-agent decision
  rules. v1: ABC stub only; first concrete implementation is a v2
  deliverable.

Capability traits (below) are runtime-checkable Protocols. Learners and
Metrics use them at binding time to declare optional requirements.

Class attribute `max_meaningful_epoch_size` (DESIGN.md §6 #10) declares the
largest `epoch_size` the Simulator may request against this Environment.
Default is unbounded; strategic-classification environments where the inner
step is a one-shot best-response override to 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor

from perfsim.core.types import Data, DataSchema

if TYPE_CHECKING:
    from perfsim.core.model import Model


class Environment(ABC):
    """Base Environment: performative map D(theta).

    Subclasses implement `sample` (peek; no state mutation) and `step`
    (advance state + return data) plus `reset(seed)`.

    The canonical Simulator entry point is `run(model, n_steps)`, which
    encodes the §8 publishing contract: the model is queried once (the
    "deployed handle" produces predictions / initial conditions for all
    agents at the start), and the env then evolves internally for n_steps
    without re-querying the model. This matches Algorithm 1 of
    arxiv 2603.12137 (Wu, Abebe, Mendler-Dünner, 2026).

    The default `run` falls back to looping `step`, which is correct for
    stateless / one-shot envs but inefficient when each `step` re-queries
    the model. Stateful-dynamics envs (FJ, replicator, accumulating shift,
    ...) override `run` to amortize the K-agent query across the inner
    n_steps iterations.
    """

    max_meaningful_epoch_size: ClassVar[int | float] = float("inf")

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
        """One internal update under the deployed model.

        Default callers should use `run(model, n_steps)`. `step` is the
        unit of internal evolution and may re-query the model on each
        call; envs that override `run` typically do not call `step`
        internally from the inner loop.
        """

    def run(self, model: "Model", n_steps: int) -> Data:
        """Run one epoch: query the model once, evolve for n_steps,
        return the final state's data dict.

        Default implementation loops `step(model)` n_steps times. This is
        correct but wasteful for envs whose `step` re-queries the model:
        with theta frozen across the inner loop, the predictions do not
        change, so the redundant queries are pure cost. Envs that benefit
        from amortizing model queries (FJ, replicator, ABM) override `run`
        to query once and evolve internally.

        Subclasses overriding `run` are responsible for state mutation:
        the final state of the n_steps inner loop must be installed before
        returning.
        """
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive int; got {n_steps!r}")
        final: Data | None = None
        for _ in range(n_steps):
            final = self.step(model)
        assert final is not None
        return final


class Dynamics(Environment):
    """Intermediate ABC for dynamical-systems-style environments.

    Marker class; concrete dynamics environments extend `StatelessDynamics`
    or `StatefulDynamics`. Used for isinstance checks that need to
    distinguish dynamics from agent-based.
    """


class StatelessDynamics(Dynamics):
    """Base for stateless dynamics: D(theta) is history-independent.

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


class StatefulDynamics(Dynamics):
    """Base for stateful dynamics environments.

    Subclasses implement `sample`, `step`, `reset` fully.
    """


class AgentBased(Environment):
    """Population of stateful agent objects with per-agent decision rules.

    v1: ABC only; no concrete implementation. The first concrete AgentBased
    environment is a v2 deliverable (DESIGN.md §17). The decision between a
    Mesa-backed implementation and a hand-rolled one is deferred to v2
    kickoff (DESIGN.md §19).

    Subclasses honor the same Environment contract (`sample`, `step`,
    `reset`, `produces_schema`) but internally maintain a list of Agent
    objects with per-agent state and a per-agent `.step()`. Scheduling
    primitives are TODO v2.
    """


# ---- Capability traits (runtime-checkable Protocols) -----------------------


@runtime_checkable
class Differentiable(Protocol):
    """Environment whose single-step sample is differentiable wrt theta."""

    def grad_sample(self, model: object) -> "Data": ...


@runtime_checkable
class FullyDifferentiable(Protocol):
    """Environment whose step and any internal population updates are
    end-to-end autograd-traceable across multiple rounds.

    Stronger than `Differentiable`. Implies differentiable surrogates for
    every primitive in the Environment (Gumbel-softmax for discrete
    sampling, softmax-with-temperature for argmax, reparameterized
    continuous distributions, soft population selection).
    """

    def grad_sample(self, model: object) -> "Data": ...

    def grad_step(self, model: object) -> "Data": ...


@runtime_checkable
class Rewarding(Protocol):
    """Environment that fills a `reward` field in its produced data dict.

    Required by RL-family Learners (v2).
    """

    @property
    def produces_reward(self) -> bool: ...


@runtime_checkable
class Trajectory(Protocol):
    """Environment that produces multi-step trajectory tensors with a
    leading time axis.

    Required by the v2 multi-step Coordinator. The trajectory data schema is
    a v2 placeholder in `core/types.py`.
    """

    @property
    def trajectory_length(self) -> int: ...


@runtime_checkable
class ClosedFormFixedPoint(Protocol):
    """Environment with an analytic closed-form RRM fixed point.

    Used in gating tests (`GaussianShiftWorld`) to verify Learners converge
    to the right point.
    """

    def closed_form_fp(self) -> Tensor: ...
