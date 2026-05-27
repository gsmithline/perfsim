"""Executor: invocation dispatch for agent skills.

`Executor` is the abstract interface; agents register and are invoked by
handle. Two implementations are planned:

- InProcessExecutor (this file): direct method dispatch on registered
  Agent objects. No serialization. Same Python process. The fast path used
  by ordinary local simulation; never serializes messages.
- A2AExecutor (not yet implemented): wire dispatch via the A2A
  protocol over HTTP / JSON-RPC. Same agent interface, different transport.

Both expose synchronous `invoke` and asynchronous `ainvoke`. The async
variant in `InProcessExecutor` is a thin wrapper around the sync path; v2
A2AExecutor uses real async I/O.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from perfsim.core.agent_spec import Agent
from perfsim.core.messages import PerfsimMessage
from perfsim.core.types import AgentHandle


class Executor(ABC):
    """Dispatch interface. Agents register; invocation goes through a handle."""

    @abstractmethod
    def register(self, agent: Agent) -> AgentHandle:
        """Register an Agent. Returns the handle used to invoke it."""

    @abstractmethod
    def unregister(self, handle: AgentHandle) -> None:
        """Remove an agent from the executor."""

    @abstractmethod
    def invoke(
        self,
        handle: AgentHandle,
        skill: str,
        request: PerfsimMessage,
    ) -> PerfsimMessage:
        """Synchronous skill invocation."""

    @abstractmethod
    async def ainvoke(
        self,
        handle: AgentHandle,
        skill: str,
        request: PerfsimMessage,
    ) -> PerfsimMessage:
        """Asynchronous skill invocation."""


class InProcessExecutor(Executor):
    """Direct method dispatch on registered Agent objects.

    No serialization; the same Python object is invoked. Type-checks the
    request and response against the agent's `SkillSpec` to catch
    mismatches at the boundary rather than deeper in the call stack.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> AgentHandle:
        spec = agent.spec
        if spec.id in self._agents:
            raise ValueError(f"agent id {spec.id!r} already registered")
        self._agents[spec.id] = agent
        return spec.handle

    def unregister(self, handle: AgentHandle) -> None:
        if handle.id not in self._agents:
            raise KeyError(f"no agent registered at handle {handle}")
        del self._agents[handle.id]

    def get(self, handle: AgentHandle) -> Agent:
        try:
            return self._agents[handle.id]
        except KeyError as exc:
            raise KeyError(f"no agent registered at handle {handle}") from exc

    @property
    def registered_ids(self) -> tuple[str, ...]:
        return tuple(self._agents.keys())

    def invoke(
        self,
        handle: AgentHandle,
        skill: str,
        request: PerfsimMessage,
    ) -> PerfsimMessage:
        agent = self.get(handle)
        skill_spec = agent.spec.skill(skill)
        if not isinstance(request, skill_spec.request_type):
            raise TypeError(
                f"skill {skill!r}: expected request of type "
                f"{skill_spec.request_type.__name__}, got {type(request).__name__}"
            )
        method = getattr(agent, skill, None)
        if method is None or not callable(method):
            raise AttributeError(
                f"agent {handle.id!r} declares skill {skill!r} but has no "
                f"callable method named {skill!r}"
            )
        response = method(request)
        if not isinstance(response, skill_spec.response_type):
            raise TypeError(
                f"skill {skill!r}: returned {type(response).__name__}, "
                f"expected {skill_spec.response_type.__name__}"
            )
        return response

    async def ainvoke(
        self,
        handle: AgentHandle,
        skill: str,
        request: PerfsimMessage,
    ) -> PerfsimMessage:
        # In-process: no real async I/O. Wraps the sync path. v2 A2AExecutor
        # will be genuinely async over HTTP.
        return self.invoke(handle, skill, request)
