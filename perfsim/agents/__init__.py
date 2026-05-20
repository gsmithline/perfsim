"""Agent-shell layer: A2A-native wrappers around the numerical core.

`PredictorAgent` and `PopulationAgent` wrap the (Model, Learner, Loss) and
World types from the numerical core, exposing them as Agent instances that
can be invoked through `InProcessExecutor` (v1) or `A2AExecutor` (v2).
"""

from perfsim.agents.population import PopulationAgent
from perfsim.agents.predictor import PredictorAgent

__all__ = ["PopulationAgent", "PredictorAgent"]
