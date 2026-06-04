"""Concrete Environment implementations.

Sub-packages:
- dynamics/: tensor-state environments (FJ, replicator, strategic, gaussian shift, etc.)
- agent_based/: per-agent stateful environments (AgentTorch adapter)

Modules:
- map_env: MapEnvironment adapter running a DistributionMap in the epoch loop
"""

from perfsim.environments.map_env import MapEnvironment

__all__ = ["MapEnvironment"]
