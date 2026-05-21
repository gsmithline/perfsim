"""perfsim.environments.agent_based: agent-based environments.

v1: stub. No concrete environment ships in v1; the first concrete
AgentBased env is a v2 deliverable (DESIGN.md §17). Re-exports the ABC
from `perfsim.core.environment` so authors can subclass it from here once
v2 work begins.
"""

from perfsim.core.environment import AgentBased

__all__ = ["AgentBased"]
