"""Unit tests for perfsim.core.executor (Executor ABC + InProcessExecutor)."""

from __future__ import annotations

import asyncio

import pytest
import torch

from perfsim.core import (
    AgentSpec,
    Executor,
    InProcessExecutor,
    PerfsimMessage,
    PredictRequest,
    PredictResponse,
    SkillSpec,
    UpdateRequest,
    UpdateResponse,
)


class _EchoRequest(PerfsimMessage):
    message: str


class _EchoResponse(PerfsimMessage):
    message: str


class _MockAgent:
    """Simple agent with predict, update, and an echo skill for tests."""

    def __init__(self, agent_id: str = "p1") -> None:
        self._spec = AgentSpec(
            id=agent_id,
            role="predictor",
            skills=(
                SkillSpec(name="predict", request_type=PredictRequest, response_type=PredictResponse),
                SkillSpec(name="update", request_type=UpdateRequest, response_type=UpdateResponse),
                SkillSpec(name="echo", request_type=_EchoRequest, response_type=_EchoResponse),
            ),
        )
        self.predict_calls: list[PredictRequest] = []
        self.update_calls: list[UpdateRequest] = []

    @property
    def spec(self) -> AgentSpec:
        return self._spec

    def predict(self, req: PredictRequest) -> PredictResponse:
        self.predict_calls.append(req)
        return PredictResponse(y=req.x.sum(dim=-1, keepdim=True))

    def update(self, req: UpdateRequest) -> UpdateResponse:
        self.update_calls.append(req)
        return UpdateResponse(status="updated")

    def echo(self, req: _EchoRequest) -> _EchoResponse:
        return _EchoResponse(message=req.message)


class _BadResponseAgent(_MockAgent):
    def predict(self, req: PredictRequest) -> PredictResponse:  # type: ignore[override]
        return UpdateResponse(status="ok")  # type: ignore[return-value]


class TestExecutorABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Executor()  # type: ignore[abstract]


class TestRegister:
    def test_register_returns_handle(self) -> None:
        ex = InProcessExecutor()
        agent = _MockAgent("p1")
        h = ex.register(agent)
        assert h.id == "p1"
        assert h.role == "predictor"
        assert "p1" in ex.registered_ids

    def test_duplicate_id_rejected(self) -> None:
        ex = InProcessExecutor()
        ex.register(_MockAgent("p1"))
        with pytest.raises(ValueError, match="already registered"):
            ex.register(_MockAgent("p1"))

    def test_unregister(self) -> None:
        ex = InProcessExecutor()
        h = ex.register(_MockAgent("p1"))
        ex.unregister(h)
        assert "p1" not in ex.registered_ids

    def test_unregister_missing_raises(self) -> None:
        ex = InProcessExecutor()
        h = ex.register(_MockAgent("p1"))
        ex.unregister(h)
        with pytest.raises(KeyError, match="no agent registered"):
            ex.unregister(h)


class TestInvokeSync:
    def test_invoke_dispatches_to_skill(self) -> None:
        ex = InProcessExecutor()
        agent = _MockAgent("p1")
        h = ex.register(agent)
        req = PredictRequest(x=torch.tensor([[1.0, 2.0, 3.0]]))
        resp = ex.invoke(h, "predict", req)
        assert isinstance(resp, PredictResponse)
        assert torch.equal(resp.y, torch.tensor([[6.0]]))
        assert agent.predict_calls == [req]

    def test_invoke_unknown_skill_raises(self) -> None:
        ex = InProcessExecutor()
        h = ex.register(_MockAgent("p1"))
        with pytest.raises(KeyError, match="no skill"):
            ex.invoke(h, "missing", PredictRequest(x=torch.zeros(1, 1)))

    def test_invoke_unknown_handle_raises(self) -> None:
        ex = InProcessExecutor()
        from perfsim.core import AgentHandle

        with pytest.raises(KeyError, match="no agent registered"):
            ex.invoke(
                AgentHandle(id="ghost", role="predictor"),
                "predict",
                PredictRequest(x=torch.zeros(1, 1)),
            )

    def test_invoke_wrong_request_type_raises(self) -> None:
        ex = InProcessExecutor()
        h = ex.register(_MockAgent("p1"))
        with pytest.raises(TypeError, match="expected request of type"):
            ex.invoke(h, "predict", UpdateRequest(data={"x": torch.zeros(1)}))

    def test_invoke_wrong_response_type_raises(self) -> None:
        ex = InProcessExecutor()
        h = ex.register(_BadResponseAgent("p1"))
        with pytest.raises(TypeError, match="expected PredictResponse"):
            ex.invoke(h, "predict", PredictRequest(x=torch.zeros(1, 1)))


class TestInvokeAsync:
    def test_ainvoke_returns_response(self) -> None:
        ex = InProcessExecutor()
        h = ex.register(_MockAgent("p1"))
        req = PredictRequest(x=torch.tensor([[1.0, 2.0]]))
        resp = asyncio.run(ex.ainvoke(h, "predict", req))
        assert isinstance(resp, PredictResponse)
        assert torch.equal(resp.y, torch.tensor([[3.0]]))

    def test_ainvoke_matches_invoke(self) -> None:
        ex = InProcessExecutor()
        h = ex.register(_MockAgent("p1"))
        req = PredictRequest(x=torch.randn(5, 3))
        sync = ex.invoke(h, "predict", req)
        async_ = asyncio.run(ex.ainvoke(h, "predict", req))
        assert torch.equal(sync.y, async_.y)


class TestMultiAgent:
    def test_multiple_agents_isolated(self) -> None:
        ex = InProcessExecutor()
        a = _MockAgent("a")
        b = _MockAgent("b")
        ha = ex.register(a)
        hb = ex.register(b)
        ex.invoke(ha, "predict", PredictRequest(x=torch.zeros(1, 1)))
        # Only a saw the call
        assert len(a.predict_calls) == 1
        assert len(b.predict_calls) == 0
