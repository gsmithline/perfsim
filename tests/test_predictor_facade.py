"""Tests for the Predictor facade (DESIGN.md §5, §15 test 10).

Verifies the facade preserves (Model, Loss, Learner) semantics:
- Constructor rejects mismatched model and learner.model.
- predict(x) matches model(x).
- deploy() returns the underlying model handle.
- train(data) mutates the model (delegates to learner.train).
"""

from __future__ import annotations

import pytest
import torch

from perfsim.core.predictor import Predictor
from perfsim.learners import GradientLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel


def _make_components() -> tuple[LinearModel, MSELoss, GradientLearner]:
    model = LinearModel(in_features=3, out_features=1, bias=False)
    loss = MSELoss()
    learner = GradientLearner(model, loss, lr=0.05, steps_per_round=1)
    return model, loss, learner


class TestPredictorFacade:
    def test_constructs_and_exposes_components(self) -> None:
        m, l, lr = _make_components()
        p = Predictor(m, l, lr)
        assert p.model is m
        assert p.loss is l
        assert p.learner is lr

    def test_rejects_mismatched_model(self) -> None:
        m1 = LinearModel(in_features=3, out_features=1, bias=False)
        m2 = LinearModel(in_features=3, out_features=1, bias=False)
        loss = MSELoss()
        learner = GradientLearner(m2, loss, lr=0.01)
        with pytest.raises(ValueError, match="must be the same object"):
            Predictor(m1, loss, learner)

    def test_predict_matches_model_forward(self) -> None:
        m, l, lr = _make_components()
        p = Predictor(m, l, lr)
        x = torch.randn(8, 3)
        torch.testing.assert_close(p.predict(x), m(x))

    def test_deploy_returns_underlying_model(self) -> None:
        m, l, lr = _make_components()
        p = Predictor(m, l, lr)
        assert p.deploy() is m

    def test_train_mutates_model_params(self) -> None:
        m, l, lr = _make_components()
        p = Predictor(m, l, lr)
        initial = m.get_params().clone()
        data = {"x": torch.randn(8, 3), "y": torch.randn(8, 1)}
        p.train(data)
        new = m.get_params()
        assert not torch.equal(initial, new), "train did not update model params"
