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
pattern (as in `StatelessDynamics`) is the recommended way to make
`sample(peek) == step(advance)` for the same world state.
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
        # Per-agent index used by LM-based Learners to look up profile rows
        # in mask-filtered training batches. Derived from the leading dim of
        # the first state entry, which by convention is the per-agent axis
        # (N for FJ, K for replicator, etc.).
        first = next(iter(initial_state.values()))
        self._n = int(first.shape[0])
        self._agent_idx = torch.arange(self._n)

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
        """Inject `agent_idx` into the data dict if the subclass did not.

        Subclasses that emit their own `agent_idx` (e.g., for a filtered
        view) are respected; the base only fills in when the field is
        absent. This is what lets LM-based Learners look up profile rows
        for the labeled training subset after `Simulator.train_mask`
        filtering.
        """
        if "agent_idx" in data:
            return data
        return {**data, "agent_idx": self._agent_idx.clone()}
