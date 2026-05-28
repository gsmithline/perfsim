"""Lightweight runner for the at_schelling scenario.

Mirrors the shape of `agent_torch.core.Runner` enough that the perfsim
plumbing on top of it (feature_provider / signal_writer / state_extractor)
looks identical to the at_covid wiring. Uses a plain dict for state
(matching AT's `state["agents"]["citizens"][...]` access pattern) and a
list of callable substeps for evolution, instead of AT's
Registry / Observation / Policy / Transition split.

Why not use AT directly:
  AT requires YAML with input/output variable paths, OmegaConf resolvers,
  network adjacency-matrix initializers, and a Registry-registered substep
  per (observation, policy, transition) role per active agent group. For
  a clean Schelling demo we do not need that ceremony; a dict-state +
  list-of-substeps runner gives us the same state-shape contract with far
  less ceremony, and is what the user explicitly authorized as a fallback.

The runner is duck-typed against `AgentTorchEnvironment`'s requirements:
  - `.state` is a nested dict
  - `.step(num_steps=K)` advances state in place
  - `.reset_state_before_episode()` exists (no-op)

That lets `AgentTorchEnvironment` wrap us as a runner_factory.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch


Substep = Callable[[dict, dict], None]
"""Substep signature: `substep(state, config) -> None`. Mutates `state` in place."""


class SchellingRunner:
    """Minimal runner with AT-shaped state and step API."""

    def __init__(
        self,
        config: dict,
        substeps: Sequence[Substep],
        state_initializer: Callable[[dict], dict],
    ) -> None:
        self.config = config
        self._substeps = list(substeps)
        self._state_initializer = state_initializer
        self.state: dict[str, Any] | None = None
        self.state_trajectory: list = []

    def init(self) -> None:
        """Build initial state via the registered initializer."""
        self.state = self._state_initializer(self.config)
        self.state["current_step"] = 0
        self.state["current_substep"] = "0"
        self.state_trajectory = [[self._snapshot()]]

    def reset(self) -> None:
        self.init()

    def reset_state_before_episode(self) -> None:
        """No trajectory clearing needed; we don't deep-copy state per
        substep (would dominate runtime for a 200-agent grid).
        """
        self.state_trajectory = []

    def step(self, num_steps: int | None = None) -> None:
        """Advance the simulation by `num_steps` Schelling rounds.

        Each round runs all substeps in order. Substeps mutate `state`.
        """
        assert self.state is not None, "Call runner.init() before step()"
        if num_steps is None:
            num_steps = self.config["simulation_metadata"]["num_steps_per_episode"]
        for t in range(int(num_steps)):
            self.state["current_step"] = t
            for i, substep in enumerate(self._substeps):
                self.state["current_substep"] = str(i)
                substep(self.state, self.config)

    def _snapshot(self) -> dict[str, Any]:
        """Detached CPU snapshot of agent tensors (cheap copy)."""
        assert self.state is not None
        out: dict[str, Any] = {}
        agents = self.state.get("agents", {})
        for atype, props in agents.items():
            out.setdefault("agents", {})[atype] = {
                k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
                for k, v in props.items()
            }
        return out
