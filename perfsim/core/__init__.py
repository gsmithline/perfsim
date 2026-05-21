"""Core abstractions: types, schemas, and capability traits.

Re-exports the canonical public API (DESIGN.md §4-5):

- `Predictor` facade over (Model, Loss, Learner).
- `Environment` ABC with `Dynamics` and `AgentBased` siblings.
- Capability Protocols: `Differentiable`, `FullyDifferentiable`, `Rewarding`,
  `Trajectory`, `ClosedFormFixedPoint`.

The agent-shell layer (`Executor`, `AgentSpec`, message Pydantic models) is
preserved scaffolding for the eventual A2A wire-up (DESIGN.md §18); it is
not on the active simulation hot path.
"""

from perfsim.core.agent_spec import Agent, AgentSpec, SkillSpec
from perfsim.core.dataset import Dataset, SubsetDataset
from perfsim.core.environment import (
    AgentBased,
    ClosedFormFixedPoint,
    Differentiable,
    Dynamics,
    Environment,
    FullyDifferentiable,
    Rewarding,
    StatefulDynamics,
    StatelessDynamics,
    Trajectory,
)
from perfsim.core.executor import Executor, InProcessExecutor
from perfsim.core.learner import Learner
from perfsim.core.loss import Loss
from perfsim.core.messages import (
    BestRespondRequest,
    BestRespondResponse,
    CloneRequest,
    CloneResponse,
    EvalLossRequest,
    EvalLossResponse,
    GetParamsRequest,
    GetParamsResponse,
    PerfsimMessage,
    PredictRequest,
    PredictResponse,
    SetParamsRequest,
    SetParamsResponse,
    UpdateRequest,
    UpdateResponse,
)
from perfsim.core.model import Model
from perfsim.core.predictor import Predictor
from perfsim.core.types import (
    SUPERVISED_SCHEMA,
    TRAJECTORY_SCHEMA,
    AgentHandle,
    ConfigBase,
    Data,
    DataSchema,
    SchemaError,
)

__all__ = [
    "SUPERVISED_SCHEMA",
    "TRAJECTORY_SCHEMA",
    "Agent",
    "AgentBased",
    "AgentHandle",
    "AgentSpec",
    "BestRespondRequest",
    "BestRespondResponse",
    "ClosedFormFixedPoint",
    "CloneRequest",
    "CloneResponse",
    "ConfigBase",
    "Data",
    "DataSchema",
    "Dataset",
    "Differentiable",
    "Dynamics",
    "Environment",
    "EvalLossRequest",
    "EvalLossResponse",
    "Executor",
    "FullyDifferentiable",
    "GetParamsRequest",
    "GetParamsResponse",
    "InProcessExecutor",
    "Learner",
    "Loss",
    "Model",
    "PerfsimMessage",
    "PredictRequest",
    "PredictResponse",
    "Predictor",
    "Rewarding",
    "SchemaError",
    "SetParamsRequest",
    "SetParamsResponse",
    "SkillSpec",
    "StatefulDynamics",
    "StatelessDynamics",
    "SubsetDataset",
    "Trajectory",
    "UpdateRequest",
    "UpdateResponse",
]
