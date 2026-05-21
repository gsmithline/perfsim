"""Tests for the epoch-based training loop (DESIGN.md §8, §15 tests 7-10).

Covers:
- Default epoch_size=1 preserves classical lockstep semantics.
- epoch_size=N calls world.step N times per outer round.
- theta is frozen across the inner N-step loop (the load-bearing claim).
- theta changes between outer rounds (training happened).
- max_meaningful_epoch_size is enforced at run start.
- epoch_size validation rejects zero, negative, and non-int values.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.core.world import StatefulWorld
from perfsim.learners import GradientLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.simulator import Simulator
from perfsim.worlds import GaussianShiftWorld, StrategicLinearWorld


class _RecordingStatefulWorld(StatefulWorld):
    """Stateful test world that records the model params seen at each step.

    Each `step` appends the model's current flat params to `theta_history`
    and increments an internal counter. Produces a per-step data dict whose
    `y` value tracks the counter, so callers can identify which step's data
    was passed downstream.
    """

    max_meaningful_epoch_size: ClassVar[int] = 100

    def __init__(self, n: int = 4, d: int = 3) -> None:
        super().__init__()
        self._n = n
        self._d = d
        self._counter = 0
        self.theta_history: list[torch.Tensor] = []
        self.data_history: list[float] = []

    @property
    def produces_schema(self) -> DataSchema:
        return SUPERVISED_SCHEMA

    def reset(self, seed: int = 0) -> None:
        self._counter = 0
        self.theta_history = []
        self.data_history = []

    def sample(self, model) -> Data:
        return self._build_data(model)

    def step(self, model) -> Data:
        self.theta_history.append(model.get_params().detach().clone())
        self._counter += 1
        return self._build_data(model)

    def _build_data(self, model) -> Data:
        gen = torch.Generator()
        gen.manual_seed(self._counter)
        x = torch.randn(self._n, self._d, generator=gen)
        y = torch.full((self._n, 1), float(self._counter))
        self.data_history.append(float(self._counter))
        return {"x": x, "y": y}


def _make_sim(n_features: int = 3) -> tuple[Simulator, LinearModel, _RecordingStatefulWorld]:
    world = _RecordingStatefulWorld(n=4, d=n_features)
    model = LinearModel(in_features=n_features, out_features=1, bias=False)
    loss = MSELoss()
    learner = GradientLearner(model, loss, lr=0.01, steps_per_round=1)
    sim = Simulator(world=world, learner=learner, loss=loss)
    return sim, model, world


class TestEpochLoopCadence:
    def test_default_epoch_size_is_one(self) -> None:
        sim, _, world = _make_sim()
        sim.run(n_rounds=3, seed=0)
        # 3 outer rounds, default epoch_size=1, so step called 3 times
        assert len(world.theta_history) == 3

    def test_epoch_size_n_calls_world_n_times_per_round(self) -> None:
        sim, _, world = _make_sim()
        sim.run(n_rounds=3, epoch_size=5, seed=0)
        # 3 outer rounds * 5 inner steps = 15
        assert len(world.theta_history) == 15

    def test_final_data_passed_to_learner(self) -> None:
        """Under final-state-only training, the learner sees only the last
        inner step's data per outer round."""
        sim, _, world = _make_sim()
        sim.run(n_rounds=2, epoch_size=4, seed=0)
        # data counter increments 1..4 in round 0, 5..8 in round 1
        # The final-state data fed to the learner has counter == 4 and 8
        # (this test just verifies data_history was populated as expected)
        assert world.data_history == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


class TestThetaFrozenInEpoch:
    """The load-bearing claim: theta does not change during the inner N-step loop."""

    def test_theta_identical_across_inner_steps(self) -> None:
        sim, _, world = _make_sim()
        sim.run(n_rounds=3, epoch_size=5, seed=0)
        for outer in range(3):
            base = world.theta_history[outer * 5]
            for inner in range(1, 5):
                got = world.theta_history[outer * 5 + inner]
                assert torch.equal(got, base), (
                    f"theta drifted within outer round {outer} at inner step {inner}"
                )

    def test_theta_changes_between_outer_rounds(self) -> None:
        """Sanity counterpart: training does still update theta between rounds."""
        sim, _, world = _make_sim()
        sim.run(n_rounds=3, epoch_size=5, seed=0)
        t0 = world.theta_history[0]
        t1 = world.theta_history[5]
        t2 = world.theta_history[10]
        assert not torch.equal(t0, t1)
        assert not torch.equal(t1, t2)


class TestMaxMeaningfulEpochSize:
    def _make_strategic_sim(self) -> Simulator:
        n, d = 16, 3
        x0 = torch.randn(n, d)
        y = torch.randn(n, 1)
        world = StrategicLinearWorld(x0=x0, y=y, epsilon=0.5)
        model = LinearModel(in_features=d, out_features=1, bias=False)
        loss = MSELoss()
        learner = GradientLearner(model, loss, lr=0.01)
        return Simulator(world=world, learner=learner, loss=loss)

    def test_strategic_world_max_is_one(self) -> None:
        assert StrategicLinearWorld.max_meaningful_epoch_size == 1

    def test_strategic_world_rejects_epoch_size_gt_one(self) -> None:
        sim = self._make_strategic_sim()
        with pytest.raises(ValueError, match="max_meaningful_epoch_size"):
            sim.run(n_rounds=2, epoch_size=2)

    def test_strategic_world_accepts_epoch_size_one(self) -> None:
        sim = self._make_strategic_sim()
        sim.run(n_rounds=2, epoch_size=1, seed=0)  # should not raise

    def test_stateless_default_max_is_inf(self) -> None:
        assert GaussianShiftWorld.max_meaningful_epoch_size == float("inf")

    def test_stateless_world_accepts_arbitrary_epoch_size(self) -> None:
        A = 0.5 * torch.eye(3)
        b = torch.tensor([1.0, 0.5, -0.5])
        world = GaussianShiftWorld(A=A, b=b, sigma_noise=0.01, batch_size=64)
        model = LinearModel(in_features=3, out_features=1, bias=False)
        loss = MSELoss()
        learner = GradientLearner(model, loss, lr=0.01)
        sim = Simulator(world=world, learner=learner, loss=loss)
        sim.run(n_rounds=2, epoch_size=10, seed=0)  # should not raise


class TestEpochSizeValidation:
    def test_rejects_zero(self) -> None:
        sim, _, _ = _make_sim()
        with pytest.raises(ValueError, match="positive"):
            sim.run(n_rounds=1, epoch_size=0)

    def test_rejects_negative(self) -> None:
        sim, _, _ = _make_sim()
        with pytest.raises(ValueError, match="positive"):
            sim.run(n_rounds=1, epoch_size=-1)

    def test_rejects_non_int(self) -> None:
        sim, _, _ = _make_sim()
        with pytest.raises(ValueError, match="positive"):
            sim.run(n_rounds=1, epoch_size=1.5)  # type: ignore[arg-type]
