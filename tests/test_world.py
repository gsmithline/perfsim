"""Unit tests for World ABCs, GaussianShiftWorld, StrategicLinearWorld.

Includes validation test 3 (sample non-mutation; section 15) as a property
test against both worlds.
"""

from __future__ import annotations

import pytest
import torch

from perfsim.core import (
    SUPERVISED_SCHEMA,
    ClosedFormFixedPoint,
    Differentiable,
    Environment,
    StatefulDynamics,
    StatelessDynamics,
)
from perfsim.models import LinearModel
from perfsim.environments.dynamics import GaussianShiftWorld, StrategicLinearWorld


def _make_gauss(
    d: int = 3, scale: float = 0.5, seed: int = 0, sigma: float = 0.01, batch: int = 256
) -> tuple[GaussianShiftWorld, torch.Tensor]:
    A = scale * torch.eye(d)
    b = torch.tensor([1.0, 0.5, -0.5])[:d]
    w = GaussianShiftWorld(A=A, b=b, sigma_noise=sigma, batch_size=batch)
    w.reset(seed=seed)
    fp = torch.linalg.solve(torch.eye(d) - A, b)
    return w, fp


def _make_model(d: int = 3) -> LinearModel:
    return LinearModel(in_features=d, out_features=1, bias=False)


class TestWorldABCs:
    def test_cannot_instantiate_Environment(self) -> None:
        with pytest.raises(TypeError):
            Environment()  # type: ignore[abstract]

    def test_cannot_instantiate_StatelessWorld(self) -> None:
        with pytest.raises(TypeError):
            StatelessDynamics()  # type: ignore[abstract]

    def test_cannot_instantiate_StatefulWorld(self) -> None:
        with pytest.raises(TypeError):
            StatefulDynamics()  # type: ignore[abstract]


class TestGaussianShiftWorld:
    def test_validation_A_must_be_square(self) -> None:
        with pytest.raises(ValueError, match="square"):
            GaussianShiftWorld(A=torch.zeros(3, 4), b=torch.zeros(3))

    def test_validation_b_must_match(self) -> None:
        with pytest.raises(ValueError, match="length matching"):
            GaussianShiftWorld(A=torch.zeros(3, 3), b=torch.zeros(2))

    def test_produces_schema_is_supervised(self) -> None:
        w, _ = _make_gauss()
        assert w.produces_schema is SUPERVISED_SCHEMA

    def test_sample_shapes(self) -> None:
        w, _ = _make_gauss(d=3, batch=64)
        m = _make_model(d=3)
        data = w.sample(m)
        assert data["x"].shape == (64, 3)
        assert data["y"].shape == (64, 1)

    def test_closed_form_fp_value(self) -> None:
        d = 3
        A = 0.5 * torch.eye(d)
        b = torch.tensor([1.0, 0.5, -0.5])
        w = GaussianShiftWorld(A=A, b=b)
        # FP = (I - 0.5 I)^-1 b = 2 b = [2.0, 1.0, -1.0]
        assert torch.allclose(w.closed_form_fp(), 2.0 * b)

    def test_implements_DifferentiableWorld(self) -> None:
        w, _ = _make_gauss()
        assert isinstance(w, Differentiable)

    def test_implements_ClosedFormFixedPoint(self) -> None:
        w, _ = _make_gauss()
        assert isinstance(w, ClosedFormFixedPoint)

    def test_step_advances_rng(self) -> None:
        w, _ = _make_gauss()
        m = _make_model()
        d1 = w.step(m)
        d2 = w.step(m)
        assert not torch.allclose(d1["x"], d2["x"])

    def test_grad_sample_traces_through_theta(self) -> None:
        w, _ = _make_gauss(d=3)
        m = _make_model(d=3)
        data = w.grad_sample(m)
        # y depends on theta (model params); ensure grad flows
        loss = data["y"].sum()
        loss.backward()
        assert m.linear.weight.grad is not None
        assert m.linear.weight.grad.abs().sum() > 0

    def test_param_dim_mismatch_raises(self) -> None:
        w, _ = _make_gauss(d=3)
        m = LinearModel(in_features=5, out_features=1, bias=False)
        with pytest.raises(ValueError, match="expects d="):
            w.sample(m)


class TestSampleNonMutation:
    """Validation test 3: World.sample(model) must not advance internal state."""

    def test_sample_does_not_advance_gen(self) -> None:
        w, _ = _make_gauss(seed=42)
        m = _make_model()
        # Snapshot the generator state before sample
        state_before = w._gen.get_state()  # type: ignore[union-attr]
        w.sample(m)
        state_after = w._gen.get_state()  # type: ignore[union-attr]
        assert torch.equal(state_before, state_after)

    def test_repeated_sample_same_output_with_same_state(self) -> None:
        w, _ = _make_gauss(seed=42)
        m = _make_model()
        d1 = w.sample(m)
        d2 = w.sample(m)
        assert torch.allclose(d1["x"], d2["x"])
        assert torch.allclose(d1["y"], d2["y"])

    def test_step_then_sample_gives_next_round_peek(self) -> None:
        # After step (advance), sample (peek) and another step should produce
        # identical data (sample peeks the next draw without advancing).
        w, _ = _make_gauss(seed=42)
        m = _make_model()
        w.step(m)
        peek = w.sample(m)
        step_next = w.step(m)
        assert torch.allclose(peek["x"], step_next["x"])

    def test_reset_returns_to_same_trajectory(self) -> None:
        w, _ = _make_gauss(seed=0)
        m = _make_model()
        first = w.step(m)
        w.reset(seed=0)
        second = w.step(m)
        assert torch.allclose(first["x"], second["x"])


class TestStrategicLinearWorld:
    def _make_world(
        self, n: int = 32, d: int = 4, epsilon: float = 1.0, seed: int = 0
    ) -> tuple[StrategicLinearWorld, LinearModel]:
        g = torch.Generator().manual_seed(seed)
        x0 = torch.randn(n, d, generator=g)
        y = torch.randint(0, 2, (n,), generator=g).float().unsqueeze(-1)
        w = StrategicLinearWorld(x0=x0, y=y, epsilon=epsilon)
        m = LinearModel(in_features=d, out_features=1, bias=True)
        return w, m

    def test_produces_schema_is_supervised(self) -> None:
        w, _ = self._make_world()
        assert w.produces_schema is SUPERVISED_SCHEMA

    def test_validation_x0_must_be_2d(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            StrategicLinearWorld(x0=torch.zeros(10), y=torch.zeros(10))

    def test_validation_y_leading_dim(self) -> None:
        with pytest.raises(ValueError, match="leading dim"):
            StrategicLinearWorld(x0=torch.zeros(10, 4), y=torch.zeros(8))

    def test_no_shift_when_weights_zero(self) -> None:
        w, m = self._make_world()
        data = w.sample(m)
        assert torch.allclose(data["x"], w._x0)

    def test_strategic_shift_uses_weight(self) -> None:
        w, m = self._make_world(d=4, epsilon=2.0)
        m.set_params(torch.tensor([1.0, 2.0, 3.0, 4.0, 0.5]))  # W=[1,2,3,4], b=0.5
        data = w.sample(m)
        expected = w._x0 + 2.0 * torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(data["x"], expected)

    def test_step_equals_sample_for_v0(self) -> None:
        w, m = self._make_world()
        d1 = w.step(m)
        d2 = w.sample(m)
        assert torch.allclose(d1["x"], d2["x"])

    def test_rejects_model_without_linear(self) -> None:
        w, _ = self._make_world(d=4)

        class _Bad:
            def get_params(self):
                return torch.zeros(4)

        with pytest.raises(TypeError, match="`.linear`"):
            w.sample(_Bad())  # type: ignore[arg-type]

    def test_weight_dim_mismatch(self) -> None:
        w, _ = self._make_world(d=4)
        wrong = LinearModel(in_features=3, out_features=1, bias=False)
        with pytest.raises(ValueError, match="population dim"):
            w.sample(wrong)

    def test_strat_features_only_shifts_specified_indices(self) -> None:
        # x0 zeros; weight = [1,2,3,4]; strat_features = [0, 2]
        # Expected shift only at columns 0 and 2.
        x0 = torch.zeros(5, 4)
        y = torch.zeros(5, 1)
        world = StrategicLinearWorld(x0=x0, y=y, epsilon=1.0, strat_features=(0, 2))
        m = LinearModel(in_features=4, out_features=1, bias=False)
        m.linear.weight.data = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        data = world.sample(m)
        expected = torch.zeros(5, 4)
        expected[:, 0] = 1.0
        expected[:, 2] = 3.0
        assert torch.allclose(data["x"], expected)

    def test_strat_features_property(self) -> None:
        x0 = torch.zeros(5, 4)
        y = torch.zeros(5, 1)
        w_all = StrategicLinearWorld(x0=x0, y=y)
        assert w_all.strat_features is None
        w_sub = StrategicLinearWorld(x0=x0, y=y, strat_features=(0, 2))
        assert w_sub.strat_features == (0, 2)

    def test_strat_features_validation(self) -> None:
        x0 = torch.zeros(5, 4)
        y = torch.zeros(5, 1)
        with pytest.raises(ValueError, match="cannot be empty"):
            StrategicLinearWorld(x0=x0, y=y, strat_features=())
        with pytest.raises(ValueError, match="non-negative"):
            StrategicLinearWorld(x0=x0, y=y, strat_features=(-1, 2))
        with pytest.raises(ValueError, match=">= d=4"):
            StrategicLinearWorld(x0=x0, y=y, strat_features=(2, 4))
        with pytest.raises(ValueError, match="must be unique"):
            StrategicLinearWorld(x0=x0, y=y, strat_features=(1, 1))
