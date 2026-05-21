"""Unit tests for core abstractions: schemas, ConfigBase, AgentHandle, traits.

The numbered validation gating tests live in their dedicated files. These are
component-level unit tests for the Task #15 scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from perfsim.core import (
    SUPERVISED_SCHEMA,
    TRAJECTORY_SCHEMA,
    AgentHandle,
    ConfigBase,
    DataSchema,
    Differentiable,
    FullyDifferentiable,
    Rewarding,
    SchemaError,
    Trajectory,
)


class TestDataSchema:
    def test_validate_accepts_required_fields(self) -> None:
        data = {"x": torch.zeros(4, 3), "y": torch.zeros(4)}
        SUPERVISED_SCHEMA.validate(data)

    def test_validate_accepts_extra_fields(self) -> None:
        data = {"x": torch.zeros(4, 3), "y": torch.zeros(4), "extra": torch.zeros(4)}
        SUPERVISED_SCHEMA.validate(data)

    def test_validate_rejects_missing_required(self) -> None:
        data = {"x": torch.zeros(4, 3)}
        with pytest.raises(SchemaError, match="missing required"):
            SUPERVISED_SCHEMA.validate(data)

    def test_covers_self(self) -> None:
        assert SUPERVISED_SCHEMA.covers(SUPERVISED_SCHEMA)

    def test_covers_strict_subset_required(self) -> None:
        subset = DataSchema(name="subset", required=frozenset({"x"}))
        assert SUPERVISED_SCHEMA.covers(subset)

    def test_does_not_cover_extra_required(self) -> None:
        bigger = DataSchema(name="bigger", required=frozenset({"x", "y", "reward"}))
        assert not SUPERVISED_SCHEMA.covers(bigger)

    def test_optional_satisfies_other_required(self) -> None:
        producer = DataSchema(
            name="producer",
            required=frozenset({"x"}),
            optional=frozenset({"y"}),
        )
        consumer = DataSchema(name="consumer", required=frozenset({"x", "y"}))
        assert producer.covers(consumer)


class TestSchemaConstants:
    def test_supervised_schema_fields(self) -> None:
        assert SUPERVISED_SCHEMA.required == frozenset({"x", "y"})
        assert SUPERVISED_SCHEMA.optional == frozenset()

    def test_trajectory_schema_is_empty_placeholder(self) -> None:
        assert TRAJECTORY_SCHEMA.required == frozenset()
        assert TRAJECTORY_SCHEMA.optional == frozenset()


@dataclass(frozen=True)
class _ToyConfig(ConfigBase):
    lr: float = 0.1
    epochs: int = 10
    name: str = "toy"


class TestConfigBase:
    def test_to_dict_round_trip(self) -> None:
        cfg = _ToyConfig(lr=0.01, epochs=5, name="x")
        assert cfg.to_dict() == {"lr": 0.01, "epochs": 5, "name": "x"}

    def test_to_json_is_sorted(self) -> None:
        cfg_a = _ToyConfig(lr=0.01, epochs=5)
        cfg_b = _ToyConfig(epochs=5, lr=0.01)
        assert cfg_a.to_json() == cfg_b.to_json()

    def test_content_hash_stable(self) -> None:
        cfg_a = _ToyConfig(lr=0.01, epochs=5)
        cfg_b = _ToyConfig(lr=0.01, epochs=5)
        assert cfg_a.content_hash() == cfg_b.content_hash()

    def test_content_hash_differs_on_value_change(self) -> None:
        cfg_a = _ToyConfig(lr=0.01)
        cfg_b = _ToyConfig(lr=0.02)
        assert cfg_a.content_hash() != cfg_b.content_hash()


class TestAgentHandle:
    def test_hashable(self) -> None:
        h = AgentHandle(id="p1", role="predictor")
        d = {h: "value"}
        assert d[h] == "value"

    def test_equality(self) -> None:
        a = AgentHandle(id="p1", role="predictor")
        b = AgentHandle(id="p1", role="predictor")
        assert a == b

    def test_endpoint_default_none(self) -> None:
        h = AgentHandle(id="p1", role="predictor")
        assert h.endpoint is None


class TestCapabilityTraits:
    def test_differentiable_world_recognized(self) -> None:
        class _W:
            def grad_sample(self, model):
                return {}

        assert isinstance(_W(), Differentiable)

    def test_non_differentiable_world_rejected(self) -> None:
        class _W:
            def sample(self, model):
                return {}

        assert not isinstance(_W(), Differentiable)

    def test_fully_differentiable_requires_both_methods(self) -> None:
        class _Partial:
            def grad_sample(self, model):
                return {}

        class _Full:
            def grad_sample(self, model):
                return {}

            def grad_step(self, model):
                return {}

        assert not isinstance(_Partial(), FullyDifferentiable)
        assert isinstance(_Full(), FullyDifferentiable)

    def test_rewarding_world_protocol(self) -> None:
        class _W:
            @property
            def produces_reward(self) -> bool:
                return True

        assert isinstance(_W(), Rewarding)

    def test_trajectory_world_protocol(self) -> None:
        class _W:
            @property
            def trajectory_length(self) -> int:
                return 8

        assert isinstance(_W(), Trajectory)
