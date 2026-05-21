"""Environment ABCs: canonical names from DESIGN.md plus AgentBased stub.

This module is the public face of the v0 environment layer. The underlying
implementation lives in `perfsim.core.world` for now; a follow-up cleanup
pass moves the contents here and retires the alias module (DESIGN.md §17).

Existing code continues to import from `perfsim.core.world`; new code
should import the canonical names from here.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from perfsim.core.world import (
    ClosedFormFixedPoint,
    DifferentiableWorld,
    FullyDifferentiableWorld,
    RewardingWorld,
    StatefulWorld,
    StatelessWorld,
    TrajectoryWorld,
    World,
)
from perfsim.core.types import Data

if TYPE_CHECKING:
    from perfsim.core.model import Model


# ---- Canonical names (DESIGN.md §5) ----------------------------------------

Environment = World
DynamicsEnvironment = World  # any of the existing concrete envs are Dynamics
StatelessDynamics = StatelessWorld
StatefulDynamics = StatefulWorld

# Capability Protocols rename: drop the "World" suffix per DESIGN.md.
Differentiable = DifferentiableWorld
FullyDifferentiable = FullyDifferentiableWorld
Rewarding = RewardingWorld
Trajectory = TrajectoryWorld


class AgentBasedEnvironment(Environment):
    """Population of stateful agent objects with per-agent decision rules.

    v1: ABC only. No concrete implementation ships in v1; the first concrete
    AgentBased env is a v2 deliverable (DESIGN.md §17). The decision between
    a Mesa-backed implementation and a hand-rolled one is deferred to v2
    kickoff (DESIGN.md §19).

    Subclasses honor the same Environment contract (`sample`, `step`,
    `reset`, `produces_schema`) but internally maintain a list of Agent
    objects with per-agent state and a per-agent `.step()`. Scheduling
    primitives (`RandomActivation`-style) are TODO v2.
    """

    @abstractmethod
    def reset(self, seed: int = 0) -> None:
        """Re-initialize RNG and any per-agent state."""

    @abstractmethod
    def sample(self, model: "Model") -> Data:
        """Peek: sample under the deployed handle without mutating state."""

    @abstractmethod
    def step(self, model: "Model") -> Data:
        """Advance: one round of per-agent decisions under fixed handle."""


__all__ = [
    "AgentBasedEnvironment",
    "ClosedFormFixedPoint",
    "Differentiable",
    "DynamicsEnvironment",
    "Environment",
    "FullyDifferentiable",
    "Rewarding",
    "StatefulDynamics",
    "StatelessDynamics",
    "Trajectory",
]
