"""PredictorAgent: agent-shell wrapper around (Model, Learner, Loss).

Exposes the predict / update / get_params / set_params / eval_loss skills
declared in `perfsim.core.messages`. Stateful: owns the Model and the
Learner's optimizer state across rounds.

Used by `InProcessExecutor` in v1 and by `A2AExecutor` (v2). The hot path
for ordinary local simulation still goes through `Simulator -> Learner ->
Model` directly; the agent boundary exists for the cases where the
predictor lives behind an Executor (multi-process, network, A2A).
"""

from __future__ import annotations

import torch

from perfsim.core.agent_spec import AgentSpec, SkillSpec
from perfsim.core.learner import Learner
from perfsim.core.loss import Loss
from perfsim.core.messages import (
    EvalLossRequest,
    EvalLossResponse,
    GetParamsRequest,
    GetParamsResponse,
    PredictRequest,
    PredictResponse,
    SetParamsRequest,
    SetParamsResponse,
    UpdateRequest,
    UpdateResponse,
)
from perfsim.core.model import Model


class PredictorAgent:
    """Wraps a (Learner, Loss) pair as an A2A-style Agent.

    The Model is read from `learner.model`. We deliberately do NOT take a
    separate `model` argument: the Learner already owns one and `learner.step`
    mutates it. Accepting both invites passing two different Model instances
    by accident, in which case updates would land on `learner.model` while
    predictions and get_params would read from the other instance, with no
    error raised. See the earlier PredictorAgent revision (commit history)
    for the explicit identity-check version of this constructor.
    """

    def __init__(
        self,
        learner: Learner,
        loss: Loss,
        *,
        agent_id: str = "predictor",
    ) -> None:
        self._learner = learner
        self._loss = loss
        self._id = agent_id

    @property
    def model(self) -> Model:
        return self._learner.model

    @property
    def learner(self) -> Learner:
        return self._learner

    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            id=self._id,
            role="predictor",
            skills=(
                SkillSpec(
                    name="predict",
                    request_type=PredictRequest,
                    response_type=PredictResponse,
                ),
                SkillSpec(
                    name="update",
                    request_type=UpdateRequest,
                    response_type=UpdateResponse,
                ),
                SkillSpec(
                    name="get_params",
                    request_type=GetParamsRequest,
                    response_type=GetParamsResponse,
                ),
                SkillSpec(
                    name="set_params",
                    request_type=SetParamsRequest,
                    response_type=SetParamsResponse,
                ),
                SkillSpec(
                    name="eval_loss",
                    request_type=EvalLossRequest,
                    response_type=EvalLossResponse,
                ),
            ),
        )

    def predict(self, req: PredictRequest) -> PredictResponse:
        with torch.no_grad():
            y = self.model(req.x)
        return PredictResponse(y=y)

    def update(self, req: UpdateRequest) -> UpdateResponse:
        self._learner.train(req.data)
        return UpdateResponse(status="ok")

    def get_params(self, req: GetParamsRequest) -> GetParamsResponse:
        return GetParamsResponse(theta=self.model.get_params())

    def set_params(self, req: SetParamsRequest) -> SetParamsResponse:
        self.model.set_params(req.theta)
        return SetParamsResponse(status="ok")

    def eval_loss(self, req: EvalLossRequest) -> EvalLossResponse:
        with torch.no_grad():
            value = self._loss(self.model, req.data, reduction=req.reduction)
        return EvalLossResponse(loss=value)
