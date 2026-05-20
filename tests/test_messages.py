"""Unit tests for perfsim.core.messages (Pydantic schemas)."""

from __future__ import annotations

import pytest
import torch

from perfsim.core import (
    AgentHandle,
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


class TestPerfsimMessageBase:
    def test_arbitrary_types_allowed(self) -> None:
        # Plain BaseModel would reject torch.Tensor without explicit config.
        # We rely on arbitrary_types_allowed=True at the PerfsimMessage level.
        req = PredictRequest(x=torch.zeros(4, 3))
        assert req.x.shape == (4, 3)


class TestPredict:
    def test_request_response_roundtrip(self) -> None:
        x = torch.randn(8, 3)
        y = torch.randn(8, 1)
        req = PredictRequest(x=x)
        resp = PredictResponse(y=y)
        assert torch.equal(req.x, x)
        assert torch.equal(resp.y, y)


class TestUpdate:
    def test_data_field_is_dict_of_tensors(self) -> None:
        data = {"x": torch.zeros(4, 3), "y": torch.zeros(4)}
        req = UpdateRequest(data=data)
        assert set(req.data.keys()) == {"x", "y"}

    def test_response_default_status_ok(self) -> None:
        resp = UpdateResponse()
        assert resp.status == "ok"


class TestParams:
    def test_get_params_request_empty(self) -> None:
        req = GetParamsRequest()
        assert isinstance(req, GetParamsRequest)

    def test_get_params_response_carries_theta(self) -> None:
        theta = torch.tensor([1.0, 2.0, 3.0])
        resp = GetParamsResponse(theta=theta)
        assert torch.equal(resp.theta, theta)

    def test_set_params_round_trip(self) -> None:
        theta = torch.tensor([0.5, -0.5])
        req = SetParamsRequest(theta=theta)
        resp = SetParamsResponse()
        assert torch.equal(req.theta, theta)
        assert resp.status == "ok"


class TestClone:
    def test_clone_request_optional_new_id(self) -> None:
        a = CloneRequest()
        b = CloneRequest(new_id="agent-copy-1")
        assert a.new_id is None
        assert b.new_id == "agent-copy-1"

    def test_clone_response_carries_handle(self) -> None:
        handle = AgentHandle(id="p1-clone", role="predictor")
        resp = CloneResponse(handle=handle)
        assert resp.handle == handle


class TestEvalLoss:
    def test_request_default_reduction_mean(self) -> None:
        req = EvalLossRequest(data={"x": torch.zeros(4, 3), "y": torch.zeros(4)})
        assert req.reduction == "mean"

    def test_response_loss_scalar(self) -> None:
        resp = EvalLossResponse(loss=torch.tensor(0.42))
        assert resp.loss.item() == pytest.approx(0.42)


class TestBestRespond:
    def test_request_carries_predictor_handle(self) -> None:
        h = AgentHandle(id="p1", role="predictor")
        req = BestRespondRequest(predictor_handle=h)
        assert req.predictor_handle == h

    def test_response_carries_x_y(self) -> None:
        x = torch.randn(10, 4)
        y = torch.zeros(10)
        resp = BestRespondResponse(x=x, y=y)
        assert torch.equal(resp.x, x)
        assert torch.equal(resp.y, y)


class TestValidation:
    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(Exception):
            PredictRequest()  # type: ignore[call-arg]
