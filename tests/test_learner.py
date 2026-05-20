"""Tests for Learner ABC and ERMLearner, GradientLearner."""

from __future__ import annotations

import pytest
import torch

from perfsim.core import SUPERVISED_SCHEMA, Learner
from perfsim.core.types import DataSchema
from perfsim.learners import ERMLearner, GradientLearner
from perfsim.losses import BCEWithLogitsLoss, MSELoss
from perfsim.models import LinearModel, LogisticModel


def _make_linear_regression_data(
    n: int = 256, d: int = 4, noise: float = 0.01, seed: int = 0
) -> tuple[dict, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    true_w = torch.tensor([1.0, -2.0, 0.5, 3.0])
    true_b = torch.tensor(0.7)
    y = X @ true_w + true_b + noise * torch.randn(n, generator=g)
    return {"x": X, "y": y.unsqueeze(-1)}, torch.cat([true_w, true_b.unsqueeze(0)])


class TestLearnerAccepts:
    def test_default_accepts_supervised(self) -> None:
        assert ERMLearner.accepts(SUPERVISED_SCHEMA)
        assert GradientLearner.accepts(SUPERVISED_SCHEMA)

    def test_rejects_unrelated_schema(self) -> None:
        bogus = DataSchema(name="bogus", required=frozenset({"q", "z"}))
        assert not ERMLearner.accepts(bogus)


class TestERMLearner:
    def test_step_updates_model(self) -> None:
        data, _ = _make_linear_regression_data()
        m = LinearModel(in_features=4, out_features=1)
        learner = ERMLearner(m, MSELoss(), max_iter=50)
        before = m.get_params().clone()
        learner.step(data)
        after = m.get_params()
        assert not torch.allclose(before, after)

    def test_recovers_true_weights_on_linear_regression(self) -> None:
        data, true_theta = _make_linear_regression_data(n=512, noise=1e-3, seed=42)
        m = LinearModel(in_features=4, out_features=1)
        learner = ERMLearner(m, MSELoss(), max_iter=200, tolerance_grad=1e-9)
        learner.step(data)
        recovered = m.get_params()
        assert torch.allclose(recovered, true_theta, atol=0.05)

    def test_reset_restores_initial_params(self) -> None:
        data, _ = _make_linear_regression_data()
        m = LinearModel(in_features=4, out_features=1)
        initial = m.get_params().clone()
        learner = ERMLearner(m, MSELoss(), max_iter=50)
        learner.step(data)
        assert not torch.allclose(m.get_params(), initial)
        learner.reset()
        assert torch.allclose(m.get_params(), initial)

    def test_logistic_regression_converges(self) -> None:
        g = torch.Generator().manual_seed(0)
        n, d = 512, 3
        X = torch.randn(n, d, generator=g)
        true_w = torch.tensor([2.0, -1.0, 0.5])
        logits = X @ true_w
        y = (torch.sigmoid(logits) > torch.rand(n, generator=g)).float().unsqueeze(-1)
        data = {"x": X, "y": y}
        m = LinearModel(in_features=d, out_features=1, bias=False)
        learner = ERMLearner(m, BCEWithLogitsLoss(), max_iter=200)
        learner.step(data)
        # loss must be lower after step than at zero init
        loss_at_zero = BCEWithLogitsLoss()(LinearModel(d, 1, bias=False), data)
        loss_after = BCEWithLogitsLoss()(m, data)
        assert loss_after < loss_at_zero


class TestGradientLearner:
    def test_step_updates_model(self) -> None:
        data, _ = _make_linear_regression_data()
        m = LinearModel(in_features=4, out_features=1)
        learner = GradientLearner(m, MSELoss(), lr=0.01, steps_per_round=1)
        before = m.get_params().clone()
        learner.step(data)
        assert not torch.allclose(before, m.get_params())

    def test_loss_decreases_over_steps(self) -> None:
        data, _ = _make_linear_regression_data()
        m = LinearModel(in_features=4, out_features=1)
        loss_fn = MSELoss()
        learner = GradientLearner(m, loss_fn, lr=0.05, steps_per_round=1)
        with torch.no_grad():
            initial = loss_fn(m, data).item()
        for _ in range(50):
            learner.step(data)
        with torch.no_grad():
            final = loss_fn(m, data).item()
        assert final < initial

    def test_converges_to_true_weights_eventually(self) -> None:
        data, true_theta = _make_linear_regression_data(n=512, noise=1e-3, seed=42)
        m = LinearModel(in_features=4, out_features=1)
        learner = GradientLearner(m, MSELoss(), lr=0.05, steps_per_round=1)
        for _ in range(1000):
            learner.step(data)
        recovered = m.get_params()
        assert torch.allclose(recovered, true_theta, atol=0.1)

    def test_unknown_optimizer_raises(self) -> None:
        m = LinearModel(in_features=4, out_features=1)
        with pytest.raises(ValueError, match="unknown optimizer"):
            GradientLearner(m, MSELoss(), optimizer="rmsprop")

    def test_adam_optimizer_works(self) -> None:
        data, _ = _make_linear_regression_data()
        m = LinearModel(in_features=4, out_features=1)
        learner = GradientLearner(m, MSELoss(), lr=0.05, optimizer="adam")
        for _ in range(50):
            learner.step(data)

    def test_reset_restores_initial_and_clears_optimizer(self) -> None:
        data, _ = _make_linear_regression_data()
        m = LinearModel(in_features=4, out_features=1)
        initial = m.get_params().clone()
        learner = GradientLearner(m, MSELoss(), lr=0.01, optimizer="adam")
        for _ in range(20):
            learner.step(data)
        assert not torch.allclose(m.get_params(), initial)
        learner.reset()
        assert torch.allclose(m.get_params(), initial)


class TestLearnerABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            Learner(LinearModel(in_features=4), MSELoss())  # type: ignore[abstract]
