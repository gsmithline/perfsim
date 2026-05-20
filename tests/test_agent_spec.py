"""Unit tests for perfsim.core.agent_spec."""

from __future__ import annotations

import pytest

from perfsim.core import (
    Agent,
    AgentHandle,
    AgentSpec,
    PredictRequest,
    PredictResponse,
    SkillSpec,
    UpdateRequest,
    UpdateResponse,
)


def _two_skill_spec(agent_id: str = "p1") -> AgentSpec:
    return AgentSpec(
        id=agent_id,
        role="predictor",
        skills=(
            SkillSpec(name="predict", request_type=PredictRequest, response_type=PredictResponse),
            SkillSpec(name="update", request_type=UpdateRequest, response_type=UpdateResponse),
        ),
    )


class TestSkillSpec:
    def test_hashable_and_equal(self) -> None:
        a = SkillSpec(name="predict", request_type=PredictRequest, response_type=PredictResponse)
        b = SkillSpec(name="predict", request_type=PredictRequest, response_type=PredictResponse)
        assert a == b
        # frozen dataclass -> hashable
        {a, b}


class TestAgentSpec:
    def test_skill_lookup_by_name(self) -> None:
        spec = _two_skill_spec()
        s = spec.skill("predict")
        assert s.name == "predict"
        assert s.request_type is PredictRequest

    def test_unknown_skill_raises(self) -> None:
        spec = _two_skill_spec()
        with pytest.raises(KeyError, match="no skill"):
            spec.skill("missing")

    def test_has_skill(self) -> None:
        spec = _two_skill_spec()
        assert spec.has_skill("predict")
        assert spec.has_skill("update")
        assert not spec.has_skill("missing")

    def test_handle_property(self) -> None:
        spec = _two_skill_spec("agent-x")
        h = spec.handle
        assert h == AgentHandle(id="agent-x", role="predictor")
        assert h.endpoint is None


class TestAgentProtocol:
    def test_recognized_when_spec_attr_present(self) -> None:
        class _Impl:
            @property
            def spec(self) -> AgentSpec:
                return _two_skill_spec()

        assert isinstance(_Impl(), Agent)

    def test_rejected_without_spec(self) -> None:
        class _NoSpec:
            pass

        assert not isinstance(_NoSpec(), Agent)
