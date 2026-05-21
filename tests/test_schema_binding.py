"""Validation test 5 (gating): Simulator.bind() rejects mismatched data
schemas at binding time, not at first hot-path call.

The check uses Learner.accepts(world.produces_schema). A Learner with a
custom accepted_schemas (or the v2 trajectory schema, which has empty
required fields) must be rejected when paired with a World whose produced
schema does not satisfy the requirements.
"""

from __future__ import annotations

import pytest
import torch

from perfsim.core import SchemaError
from perfsim.core.types import DataSchema
from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.simulator import Simulator
from perfsim.environments.dynamics import GaussianShiftWorld


def _make_world(d: int = 3) -> GaussianShiftWorld:
    A = 0.5 * torch.eye(d)
    b = torch.tensor([1.0, 0.5, -0.5])[:d].clone()
    return GaussianShiftWorld(A=A, b=b)


def _make_model(d: int = 3) -> LinearModel:
    return LinearModel(in_features=d, out_features=1, bias=False)


def test_supervised_pair_binds() -> None:
    sim = Simulator(
        world=_make_world(),
        learner=ERMLearner(_make_model(), MSELoss()),
        loss=MSELoss(),
    )
    assert sim is not None


def test_rejects_learner_requiring_extra_fields() -> None:
    class _RLLearner(ERMLearner):
        accepted_schemas = (
            DataSchema(
                name="trajectory_v2",
                required=frozenset({"state", "action", "reward", "log_prob"}),
            ),
        )

    learner = _RLLearner(_make_model(), MSELoss())
    with pytest.raises(SchemaError, match="does not accept"):
        Simulator(world=_make_world(), learner=learner, loss=MSELoss())


def test_rejects_learner_requiring_only_reward() -> None:
    class _RewardLearner(ERMLearner):
        accepted_schemas = (
            DataSchema(name="reward_only", required=frozenset({"reward"})),
        )

    learner = _RewardLearner(_make_model(), MSELoss())
    with pytest.raises(SchemaError, match="does not accept"):
        Simulator(world=_make_world(), learner=learner, loss=MSELoss())


def test_learner_with_subset_schema_binds() -> None:
    """A Learner that requires only 'x' (subset of supervised) should bind."""

    class _XOnlyLearner(ERMLearner):
        accepted_schemas = (
            DataSchema(name="x_only", required=frozenset({"x"})),
        )

    learner = _XOnlyLearner(_make_model(), MSELoss())
    Simulator(world=_make_world(), learner=learner, loss=MSELoss())


def test_error_message_lists_accepted_schemas() -> None:
    class _FooLearner(ERMLearner):
        accepted_schemas = (
            DataSchema(name="foo_schema", required=frozenset({"foo"})),
            DataSchema(name="bar_schema", required=frozenset({"bar"})),
        )

    learner = _FooLearner(_make_model(), MSELoss())
    with pytest.raises(SchemaError) as excinfo:
        Simulator(world=_make_world(), learner=learner, loss=MSELoss())
    msg = str(excinfo.value)
    assert "foo_schema" in msg
    assert "bar_schema" in msg
    assert "supervised" in msg
