"""StatefulPopulationWorld: base for per-agent persistent-state worlds.

Subclasses define `_step(model) -> (data, next_state)`; the base handles state
persistence, reset, and the sample (peek) vs step (advance) distinction. State
is a dict of named tensors with a leading per-agent axis. `_step` should be
pure in (state, model); stochastic subclasses manage their own seeded RNG.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import Data
from perfsim.core.environment import StatefulDynamics

State = dict[str, Tensor]


class StatefulPopulationWorld(StatefulDynamics):
    """Base for worlds with per-agent persistent state."""

    def __init__(self, initial_state: State, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if not isinstance(initial_state, dict):
            raise TypeError(
                f"initial_state must be a dict[str, Tensor]; got {type(initial_state).__name__}"
            )
        if not initial_state:
            raise ValueError("initial_state must contain at least one Tensor")
        for k, v in initial_state.items():
            if not isinstance(v, Tensor):
                raise TypeError(
                    f"initial_state[{k!r}] must be a Tensor; got {type(v).__name__}"
                )
        self._dtype = dtype
        self._initial_state: State = {
            k: v.detach().clone() for k, v in initial_state.items()
        }
        self._state: State = {
            k: v.detach().clone() for k, v in initial_state.items()
        }
        # Per-agent index for LM learners to look up profile rows; the leading
        # dim of the first state entry is the per-agent axis by convention.
        first = next(iter(initial_state.values()))
        self._n = int(first.shape[0])
        self._agent_idx = torch.arange(self._n)

    @property
    def state(self) -> State:
        """Defensive snapshot of the current state."""
        return {k: v.detach().clone() for k, v in self._state.items()}

    def reset(self, seed: int = 0) -> None:
        """Restore the initial-state snapshot (override to also reseed RNG)."""
        self._state = {k: v.detach().clone() for k, v in self._initial_state.items()}

    @abstractmethod
    def _step(self, model: Model) -> tuple[Data, State]:
        """One-round transition returning (data, next_state).

        next_state is installed by `step` and ignored by `sample` (peek).
        """

    def sample(self, model: Model) -> Data:
        data, _ = self._step(model)
        return self._with_agent_idx(data)

    def step(self, model: Model) -> Data:
        data, next_state = self._step(model)
        for k, v in next_state.items():
            if not isinstance(v, Tensor):
                raise TypeError(
                    f"next_state[{k!r}] must be a Tensor; got {type(v).__name__}"
                )
        self._state = {k: v for k, v in next_state.items()}
        return self._with_agent_idx(data)

    def _with_agent_idx(self, data: Data) -> Data:
        """Inject `agent_idx` unless the subclass already emitted its own."""
        if "agent_idx" in data:
            return data
        return {**data, "agent_idx": self._agent_idx.clone()}
