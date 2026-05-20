"""AgentSpec: declarative metadata for an agent (A2A AgentCard analog).

Each agent exposes a `spec` property (`AgentSpec`) describing its identity,
role, and available skills. The Executor uses this for type-checking and
(in v2) for serializing to the A2A AgentCard wire format.

`SkillSpec` ties a skill name to its request and response message types.
`Agent` is a runtime-checkable Protocol; any object with a `spec` property
returning an `AgentSpec` satisfies it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Type, runtime_checkable

from perfsim.core.messages import PerfsimMessage
from perfsim.core.types import AgentHandle


@dataclass(frozen=True)
class SkillSpec:
    """Metadata for a single skill."""

    name: str
    request_type: Type[PerfsimMessage]
    response_type: Type[PerfsimMessage]


@dataclass(frozen=True)
class AgentSpec:
    """Declarative metadata for an agent (A2A AgentCard analog)."""

    id: str
    role: str
    skills: tuple[SkillSpec, ...]

    def skill(self, name: str) -> SkillSpec:
        for s in self.skills:
            if s.name == name:
                return s
        raise KeyError(
            f"agent {self.id!r} has no skill {name!r}; "
            f"have {[s.name for s in self.skills]}"
        )

    def has_skill(self, name: str) -> bool:
        return any(s.name == name for s in self.skills)

    @property
    def handle(self) -> AgentHandle:
        return AgentHandle(id=self.id, role=self.role)


@runtime_checkable
class Agent(Protocol):
    """Structural type for agents: a `spec` property plus skill methods.

    A class implements `Agent` by providing:
    - A `spec` property returning an `AgentSpec`.
    - One method per skill named in the spec, taking the request type and
      returning the response type.
    """

    @property
    def spec(self) -> AgentSpec: ...
