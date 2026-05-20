"""PopulationAgent: agent-shell wrapper around a World.

Exposes the `best_respond` skill: given a handle to a PredictorAgent, asks
the underlying World to produce one round of data conditioned on the
deployed predictor's parameters. The agent fetches the predictor's flat
parameter vector via the Executor (`get_params` skill) and writes it into
its own scratch Model before calling `world.step`.

This makes the agent boundary self-sufficient: PopulationAgent never holds
a direct Python reference to the PredictorAgent. The only coupling is the
shared Model architecture (so that the predictor's flat params can be
applied to PopulationAgent's scratch model).
"""

from __future__ import annotations

from perfsim.core.agent_spec import AgentSpec, SkillSpec
from perfsim.core.executor import Executor
from perfsim.core.messages import (
    BestRespondRequest,
    BestRespondResponse,
    GetParamsRequest,
    GetParamsResponse,
)
from perfsim.core.model import Model
from perfsim.core.world import World


class PopulationAgent:
    """Wraps a World as an A2A-style Agent with a `best_respond` skill."""

    def __init__(
        self,
        world: World,
        scratch_model: Model,
        executor: Executor,
        *,
        agent_id: str = "population",
    ) -> None:
        self._world = world
        self._scratch_model = scratch_model
        self._executor = executor
        self._id = agent_id

    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            id=self._id,
            role="population",
            skills=(
                SkillSpec(
                    name="best_respond",
                    request_type=BestRespondRequest,
                    response_type=BestRespondResponse,
                ),
            ),
        )

    @property
    def world(self) -> World:
        return self._world

    def best_respond(self, req: BestRespondRequest) -> BestRespondResponse:
        params_resp = self._executor.invoke(
            req.predictor_handle, "get_params", GetParamsRequest()
        )
        if not isinstance(params_resp, GetParamsResponse):
            raise TypeError(
                f"get_params returned {type(params_resp).__name__}, "
                f"expected GetParamsResponse"
            )
        self._scratch_model.set_params(params_resp.theta)
        data = self._world.step(self._scratch_model)
        return BestRespondResponse(x=data["x"], y=data["y"])
