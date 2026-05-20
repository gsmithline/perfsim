"""StatefulPopulationWorld: base class for per-agent persistent-state worlds.

Phase 1 of the ABM backbone. Subclasses define a single `_step(model)`
method that returns `(data, next_state)`; the base handles state
persistence, reset, and the `sample` vs `step` (peek vs advance)
distinction.

State is a dict of tensors keyed by name. Convention:

  x:          (N, D)     observable features
  y:          (N, K)     labels
  latent:     (N, L)     hidden agent state (opinions, history, etc.)
  agent_type: (N,)       integer type label for heterogeneous populations

Not every world uses every key; subclasses declare which fields they
maintain. The base does not enforce field presence; it just stores and
restores whatever dict the subclass passes in.

Determinism: `_step` should be a pure function of `(self._state, model)`.
Stochastic subclasses must manage their own seeded RNG; a forked-generator
pattern (as in `StatelessWorld`) is the recommended way to make
`sample(peek) == step(advance)` for the same world state.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import Data
from perfsim.core.world import StatefulWorld

State = dict[str, Tensor]


class StatefulPopulationWorld(StatefulWorld):
    """Base for worlds with per-agent persistent state.

    Subclasses implement `_step(model) -> (data, next_state)`. The base
    handles `reset` (restore initial state), `sample` (peek; computes
    `_step` and discards the next state), and `step` (advance; stores the
    next state).
    """

    def __init__(self, initial_state: State, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if not isinstance(initial_state, dict):
            raise TypeError(
                f"initial_state must be a dict[str, Tensor]; got {type(initial_state).__name__}"
            )
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

    @property
    def state(self) -> State:
        """Snapshot of the current state. Modifying the returned dict does
        not affect the world.
        """
        return {k: v.detach().clone() for k, v in self._state.items()}

    def reset(self, seed: int = 0) -> None:
        """Restore state to the initial-state snapshot.

        Stochastic subclasses override to also reseed their RNG.
        """
        self._state = {k: v.detach().clone() for k, v in self._initial_state.items()}

    @abstractmethod
    def _step(self, model: Model) -> tuple[Data, State]:
        """One-round transition.

        Returns (data, next_state):
          - data: what the predictor trains on this round.
          - next_state: dict to install as `self._state` after `step`;
            ignored by `sample` (peek).

        Subclasses are responsible for state-update determinism. If the
        world is deterministic in `(self._state, model)`, `sample` and
        `step` produce identical data for the same state.
        """

    def sample(self, model: Model) -> Data:
        data, _ = self._step(model)
        return data

    def step(self, model: Model) -> Data:
        data, next_state = self._step(model)
        for k, v in next_state.items():
            if not isinstance(v, Tensor):
                raise TypeError(
                    f"next_state[{k!r}] must be a Tensor; got {type(v).__name__}"
                )
        self._state = {k: v for k, v in next_state.items()}
        return data
