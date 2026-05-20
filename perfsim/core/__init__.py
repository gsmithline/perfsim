"""Core abstractions: types, schemas, and capability traits.

Concrete ABCs (Model, Loss, Learner, World ABCs) and concrete implementations
land in subsequent tasks. This module re-exports the public types and trait
protocols.
"""

from perfsim.core.agent_spec import Agent, AgentSpec, SkillSpec
from perfsim.core.dataset import Dataset, SubsetDataset
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
from perfsim.core.types import (
    SUPERVISED_SCHEMA,
    TRAJECTORY_SCHEMA,
    AgentHandle,
    ConfigBase,
    Data,
    DataSchema,
    SchemaError,
)
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

__all__ = [
    "SUPERVISED_SCHEMA",
    "TRAJECTORY_SCHEMA",
    "Agent",
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
    "DifferentiableWorld",
    "EvalLossRequest",
    "EvalLossResponse",
    "Executor",
    "FullyDifferentiableWorld",
    "GetParamsRequest",
    "GetParamsResponse",
    "InProcessExecutor",
    "Learner",
    "Loss",
    "Model",
    "PerfsimMessage",
    "PredictRequest",
    "PredictResponse",
    "RewardingWorld",
    "SchemaError",
    "SetParamsRequest",
    "SetParamsResponse",
    "SkillSpec",
    "StatefulWorld",
    "StatelessWorld",
    "SubsetDataset",
    "TrajectoryWorld",
    "UpdateRequest",
    "UpdateResponse",
    "World",
]
