"""Pydantic message schemas for agent skill calls.

Each skill defines a Request and a Response. In v1 (InProcessExecutor),
messages are passed in-memory; the Pydantic schemas serve two purposes:

1. Type-checking at the Executor boundary: Executor verifies the request
   instance matches the SkillSpec's declared request_type at invoke time
   (not at first hot-path call).
2. v2 forward-compat: when the A2AExecutor lands, the same models
   serialize to A2A Message Parts via Pydantic JSON schema generation.

Tensors are kept as `torch.Tensor` (`arbitrary_types_allowed=True`). v2
will add a JSON-safe serializer (likely raw bytes + dtype + shape, akin
to the A2A FilePart pattern).

Models are NOT frozen because some fields (Tensor, AgentHandle) compose
poorly with Pydantic's frozen=True hashing path. Callers should not mutate
message instances; treat them as immutable by convention.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict

from perfsim.core.types import AgentHandle


class PerfsimMessage(BaseModel):
    """Base for all perfsim agent messages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---- Predict ---------------------------------------------------------------


class PredictRequest(PerfsimMessage):
    x: torch.Tensor


class PredictResponse(PerfsimMessage):
    y: torch.Tensor


# ---- Update ----------------------------------------------------------------


class UpdateRequest(PerfsimMessage):
    """Train the learner on one round's data."""

    data: dict[str, torch.Tensor]


class UpdateResponse(PerfsimMessage):
    status: str = "ok"


# ---- Parameters ------------------------------------------------------------


class GetParamsRequest(PerfsimMessage):
    pass


class GetParamsResponse(PerfsimMessage):
    theta: torch.Tensor


class SetParamsRequest(PerfsimMessage):
    theta: torch.Tensor


class SetParamsResponse(PerfsimMessage):
    status: str = "ok"


# ---- Clone -----------------------------------------------------------------


class CloneRequest(PerfsimMessage):
    """Produce an independent copy of the agent (registered with the executor).

    The new_id field optionally sets the cloned agent's id; if None, the
    executor picks one.
    """

    new_id: str | None = None


class CloneResponse(PerfsimMessage):
    """Returns a handle to the cloned agent."""

    handle: AgentHandle


# ---- Evaluate loss ---------------------------------------------------------


class EvalLossRequest(PerfsimMessage):
    """Evaluate the agent's loss on supplied data.

    Used by metrics (PR, DPR) when the predictor lives behind an agent
    boundary instead of being a direct numerical-core handle.
    """

    data: dict[str, torch.Tensor]
    reduction: str = "mean"


class EvalLossResponse(PerfsimMessage):
    loss: torch.Tensor


# ---- Best respond ---------------------------------------------------------


class BestRespondRequest(PerfsimMessage):
    """Ask a population agent to best-respond to a deployed predictor.

    The predictor is identified by handle so the population may call
    `predict` on it via the executor (e.g., when the best-response is
    derivative-based).
    """

    predictor_handle: AgentHandle


class BestRespondResponse(PerfsimMessage):
    """Returns the population's data after best-responding."""

    x: torch.Tensor
    y: torch.Tensor
