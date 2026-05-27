"""Unit tests for StatefulPopulationWorld (ABC) and AccumulatingShiftWorld."""

from __future__ import annotations

import pytest
import torch

from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel, MLPModel
from perfsim.environments.dynamics import (
    AccumulatingShiftWorld,
    StatefulPopulationWorld,
    StrategicGradientWorld,
)
from perfsim.simulator import Simulator


# ----- Minimal concrete subclass for ABC unit tests. -----


class _IncrementWorld(StatefulPopulationWorld):
    """Trivial subclass: increments state['x0'] by 1 each step; ignores model."""

    @property
    def produces_schema(self) -> DataSchema:
        return SUPERVISED_SCHEMA

    def _step(self, model):
        x0 = self._state["x0"]
        next_x0 = x0 + 1.0
        return {"x": x0.clone(), "y": torch.zeros(x0.shape[0], 1)}, {"x0": next_x0}


def _fixed_population(n: int, d: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, d, generator=g)
    y = (torch.rand(n, generator=g) > 0.5).float().unsqueeze(-1)
    return x0, y


# ----- ABC tests --------------------------------------------------------------


class TestABC:
    def test_cannot_instantiate_base(self) -> None:
        with pytest.raises(TypeError):
            StatefulPopulationWorld(initial_state={"x0": torch.zeros(2, 3)})  # type: ignore[abstract]

    def test_initial_state_rejects_non_dict(self) -> None:
        with pytest.raises(TypeError, match="initial_state must be a dict"):
            _IncrementWorld(initial_state=[torch.zeros(2, 3)])  # type: ignore[arg-type]

    def test_initial_state_rejects_non_tensor_value(self) -> None:
        with pytest.raises(TypeError, match="must be a Tensor"):
            _IncrementWorld(initial_state={"x0": [1.0, 2.0]})  # type: ignore[arg-type]

    def test_state_snapshot_is_a_copy(self) -> None:
        w = _IncrementWorld(initial_state={"x0": torch.zeros(2, 3)})
        snap = w.state
        snap["x0"].add_(1.0)
        assert w.state["x0"].abs().max().item() == 0.0

    def test_reset_restores_initial_state(self) -> None:
        x0 = torch.zeros(2, 3)
        w = _IncrementWorld(initial_state={"x0": x0})
        w.step(model=None)  # type: ignore[arg-type]
        w.step(model=None)  # type: ignore[arg-type]
        assert w.state["x0"][0, 0].item() == 2.0
        w.reset()
        assert torch.equal(w.state["x0"], x0)

    def test_sample_does_not_mutate_state(self) -> None:
        x0 = torch.zeros(2, 3)
        w = _IncrementWorld(initial_state={"x0": x0})
        before = w.state["x0"].clone()
        w.sample(model=None)  # type: ignore[arg-type]
        assert torch.equal(w.state["x0"], before)

    def test_step_advances_state(self) -> None:
        x0 = torch.zeros(2, 3)
        w = _IncrementWorld(initial_state={"x0": x0})
        w.step(model=None)  # type: ignore[arg-type]
        assert torch.equal(w.state["x0"], x0 + 1.0)
        w.step(model=None)  # type: ignore[arg-type]
        assert torch.equal(w.state["x0"], x0 + 2.0)

    def test_sample_returns_data_for_current_state(self) -> None:
        x0 = torch.full((2, 3), 7.0)
        w = _IncrementWorld(initial_state={"x0": x0})
        # The trivial subclass returns the *current* x0 as data["x"].
        data = w.sample(model=None)  # type: ignore[arg-type]
        assert torch.equal(data["x"], x0)


# ----- AccumulatingShiftWorld tests ------------------------------------------


class TestAccumulatingShiftEquivalence:
    """At eta=0, AccumulatingShiftWorld must match StrategicGradientWorld exactly."""

    def test_full_features_match(self) -> None:
        n, d, eps = 50, 4, 0.5
        x0, y = _fixed_population(n, d)
        w_acc = AccumulatingShiftWorld(x0=x0, y=y, epsilon=eps, eta=0.0)
        w_grad = StrategicGradientWorld(x0=x0, y=y, epsilon=eps)
        model = LinearModel(in_features=d, out_features=1, bias=False)
        model.set_params(torch.tensor([1.0, -0.3, 0.5, 0.2]))
        a = w_acc.sample(model)
        b = w_grad.sample(model)
        assert torch.allclose(a["x"], b["x"], atol=1e-6)
        assert torch.equal(a["y"], b["y"])

    def test_strat_features_subset_match(self) -> None:
        n, d, eps = 30, 5, 1.5
        x0, y = _fixed_population(n, d)
        sf = (0, 2, 4)
        w_acc = AccumulatingShiftWorld(x0=x0, y=y, epsilon=eps, eta=0.0, strat_features=sf)
        w_grad = StrategicGradientWorld(x0=x0, y=y, epsilon=eps, strat_features=sf)
        model = LinearModel(in_features=d, out_features=1, bias=False)
        model.set_params(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
        a = w_acc.sample(model)
        b = w_grad.sample(model)
        assert torch.allclose(a["x"], b["x"], atol=1e-6)

    def test_eta_zero_state_does_not_drift(self) -> None:
        n, d = 20, 3
        x0, y = _fixed_population(n, d)
        w = AccumulatingShiftWorld(x0=x0, y=y, epsilon=1.0, eta=0.0)
        model = LinearModel(in_features=d, out_features=1, bias=False)
        model.set_params(torch.tensor([0.5, -0.5, 1.0]))
        before = w.state["x0"].clone()
        for _ in range(5):
            w.step(model)
        assert torch.allclose(w.state["x0"], before, atol=1e-6)


class TestAccumulatingShiftDrift:
    def test_eta_one_state_equals_strategic_position_after_one_step(self) -> None:
        # eta=1: x_0^{t+1} = x_strategic^t
        n, d = 10, 3
        x0, y = _fixed_population(n, d)
        w = AccumulatingShiftWorld(x0=x0, y=y, epsilon=0.5, eta=1.0)
        model = LinearModel(in_features=d, out_features=1, bias=False)
        model.set_params(torch.tensor([1.0, 0.0, 0.0]))
        data = w.step(model)
        # The new x0 should equal what we just trained on (x_strategic).
        assert torch.allclose(w.state["x0"], data["x"], atol=1e-6)

    def test_partial_eta_drift_lies_between(self) -> None:
        # eta=0.5 -> x_0^{t+1} should be halfway between x_0^t and x_strategic^t
        n, d = 5, 2
        x0, y = _fixed_population(n, d)
        w = AccumulatingShiftWorld(x0=x0, y=y, epsilon=1.0, eta=0.5)
        model = LinearModel(in_features=d, out_features=1, bias=False)
        model.set_params(torch.tensor([1.0, -1.0]))
        x0_t = w.state["x0"].clone()
        data = w.step(model)
        x_strategic = data["x"]
        expected = 0.5 * x0_t + 0.5 * x_strategic
        assert torch.allclose(w.state["x0"], expected, atol=1e-6)

    def test_repeated_steps_strategic_pos_changes(self) -> None:
        # As x0 drifts, the next round's strategic position should differ
        # (for a non-linear predictor where the gradient varies in x).
        n, d = 10, 3
        x0, y = _fixed_population(n, d)
        w = AccumulatingShiftWorld(x0=x0, y=y, epsilon=0.5, eta=1.0)
        mlp = MLPModel(in_features=d, hidden_dims=[8], out_features=1, init_seed=0)
        d_first = w.step(mlp)
        d_second = w.step(mlp)
        # MLP gradients depend on input, so two rounds give different data.
        assert not torch.allclose(d_first["x"], d_second["x"], atol=1e-6)


class TestAccumulatingShiftValidation:
    def test_eta_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="eta must be in"):
            AccumulatingShiftWorld(
                x0=torch.zeros(2, 3), y=torch.zeros(2, 1), eta=1.5
            )

    def test_x0_must_be_2d(self) -> None:
        with pytest.raises(ValueError, match="x0 must be 2-D"):
            AccumulatingShiftWorld(x0=torch.zeros(5), y=torch.zeros(5))

    def test_y_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            AccumulatingShiftWorld(x0=torch.zeros(5, 3), y=torch.zeros(4))


class TestAccumulatingShiftIntegration:
    def test_runs_through_simulator(self) -> None:
        n, d = 30, 3
        x0, y = _fixed_population(n, d)
        world = AccumulatingShiftWorld(
            x0=x0,
            y=y.squeeze(-1).unsqueeze(-1),  # ensure (N, 1)
            epsilon=0.1,
            eta=0.2,
        )
        model = LinearModel(in_features=d, out_features=1, bias=False)
        learner = ERMLearner(model, MSELoss(), max_iter=20)
        sim = Simulator(world=world, learner=learner, loss=MSELoss())
        history = sim.run(n_rounds=5, seed=0)
        # State should have drifted from initial after 5 rounds.
        assert not torch.allclose(world.state["x0"], x0, atol=1e-6)
        assert len(history.records) == 5
