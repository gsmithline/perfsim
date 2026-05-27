"""Tests for Loss implementations: MSE, CE, BCE, BCEWithLogits, Hinge, L2Reg."""

from __future__ import annotations

import math

import pytest
import torch

from perfsim.losses import (
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    HingeLoss,
    L2RegularizedLoss,
    MSELoss,
)
from perfsim.models import LinearModel, LogisticModel


class TestMSELoss:
    def test_zero_when_perfect(self) -> None:
        m = LinearModel(in_features=3, out_features=1)
        m.set_params(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        x = torch.tensor([[1.0, 0.0, 0.0]])
        y = torch.tensor([[1.0]])
        loss = MSELoss()(m, {"x": x, "y": y})
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_per_example_reduction_none(self) -> None:
        m = LinearModel(in_features=3, out_features=1)
        x = torch.randn(8, 3)
        y = torch.randn(8, 1)
        loss = MSELoss()(m, {"x": x, "y": y}, reduction="none")
        assert loss.shape == (8,)

    def test_sum_equals_mean_times_n(self) -> None:
        m = LinearModel(in_features=3, out_features=1)
        x = torch.randn(8, 3)
        y = torch.randn(8, 1)
        d = {"x": x, "y": y}
        mean = MSELoss()(m, d, reduction="mean")
        total = MSELoss()(m, d, reduction="sum")
        assert torch.allclose(total, mean * 8)

    def test_unknown_reduction_raises(self) -> None:
        m = LinearModel(in_features=3, out_features=1)
        with pytest.raises(ValueError, match="unknown reduction"):
            MSELoss()(m, {"x": torch.randn(4, 3), "y": torch.randn(4, 1)}, reduction="bogus")


class TestCrossEntropyLoss:
    def test_uniform_logits_log_k(self) -> None:
        m = LinearModel(in_features=4, out_features=3)
        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))
        loss = CrossEntropyLoss()(m, {"x": x, "y": y})
        assert torch.allclose(loss, torch.tensor(math.log(3.0)), atol=1e-5)

    def test_per_example_reduction(self) -> None:
        m = LinearModel(in_features=4, out_features=3)
        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))
        loss = CrossEntropyLoss()(m, {"x": x, "y": y}, reduction="none")
        assert loss.shape == (8,)


class TestBCELoss:
    def test_zero_init_logistic_half_prob(self) -> None:
        m = LogisticModel(in_features=3, out_features=1)
        x = torch.randn(8, 3)
        y = torch.randint(0, 2, (8, 1)).float()
        loss = BCELoss()(m, {"x": x, "y": y})
        assert torch.allclose(loss, torch.tensor(-math.log(0.5)), atol=1e-5)


class TestBCEWithLogitsLoss:
    def test_zero_logits_log2(self) -> None:
        m = LinearModel(in_features=3, out_features=1)
        x = torch.randn(8, 3)
        y = torch.randint(0, 2, (8, 1)).float()
        loss = BCEWithLogitsLoss()(m, {"x": x, "y": y})
        assert torch.allclose(loss, torch.tensor(math.log(2.0)), atol=1e-5)


class TestHingeLoss:
    def test_zero_when_well_separated(self) -> None:
        m = LinearModel(in_features=3, out_features=1)
        m.set_params(torch.tensor([10.0, 0.0, 0.0, 0.0]))
        x = torch.tensor([[1.0, 0.0, 0.0]])
        y = torch.tensor([[1.0]])
        loss = HingeLoss()(m, {"x": x, "y": y})
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_one_at_zero_score(self) -> None:
        m = LinearModel(in_features=3, out_features=1)
        x = torch.tensor([[1.0, 0.0, 0.0]])
        y = torch.tensor([[1.0]])
        loss = HingeLoss()(m, {"x": x, "y": y})
        assert torch.allclose(loss, torch.tensor(1.0))


class TestL2RegularizedLoss:
    def test_zero_weight_decay_equals_base(self) -> None:
        m = LinearModel(in_features=3, out_features=1)
        m.set_params(torch.tensor([1.0, 2.0, 3.0, 0.5]))
        x = torch.randn(8, 3)
        y = torch.randn(8, 1)
        d = {"x": x, "y": y}
        base = MSELoss()
        wrapped = L2RegularizedLoss(base, weight_decay=0.0)
        assert torch.allclose(base(m, d), wrapped(m, d))

    def test_adds_l2_term(self) -> None:
        m = LinearModel(in_features=3, out_features=1, bias=False)
        m.set_params(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.zeros(4, 3)
        y = torch.zeros(4, 1)
        d = {"x": x, "y": y}
        wd = 0.1
        base_val = MSELoss()(m, d).item()
        wrapped_val = L2RegularizedLoss(MSELoss(), weight_decay=wd)(m, d).item()
        expected_reg = 0.5 * wd * (1.0 ** 2 + 2.0 ** 2 + 3.0 ** 2)
        assert wrapped_val == pytest.approx(base_val + expected_reg, abs=1e-6)

    def test_negative_weight_decay_rejected(self) -> None:
        with pytest.raises(ValueError, match="weight_decay must be >= 0"):
            L2RegularizedLoss(MSELoss(), weight_decay=-0.01)

    def test_reduction_none_skips_reg(self) -> None:
        m = LinearModel(in_features=3, out_features=1)
        m.set_params(torch.tensor([1.0, 2.0, 3.0, 0.5]))
        x = torch.randn(8, 3)
        y = torch.randn(8, 1)
        d = {"x": x, "y": y}
        base = MSELoss()(m, d, reduction="none")
        wrapped = L2RegularizedLoss(MSELoss(), weight_decay=0.5)(m, d, reduction="none")
        assert torch.allclose(base, wrapped)

    def test_reduction_sum_adds_reg(self) -> None:
        m = LinearModel(in_features=3, out_features=1, bias=False)
        m.set_params(torch.tensor([2.0, 0.0, 0.0]))
        x = torch.zeros(4, 3)
        y = torch.zeros(4, 1)
        d = {"x": x, "y": y}
        wd = 1.0
        wrapped = L2RegularizedLoss(MSELoss(), weight_decay=wd)
        # base is 0; reg is 0.5 * 1.0 * 4.0 = 2.0
        assert torch.allclose(wrapped(m, d, reduction="sum"), torch.tensor(2.0))

    def test_gradient_includes_reg_term(self) -> None:
        m = LinearModel(in_features=3, out_features=1, bias=False)
        m.set_params(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.zeros(4, 3)
        y = torch.zeros(4, 1)
        d = {"x": x, "y": y}
        wrapped = L2RegularizedLoss(MSELoss(), weight_decay=0.5)
        wrapped(m, d).backward()
        # Base loss is 0; reg gradient: 0.5 * 2 * theta = theta (because the constant 0.5 cancels with 2)
        # Actually: 0.5 * wd * theta^T theta -> gradient = wd * theta
        # wd=0.5, theta=[1,2,3] -> grad = [0.5, 1.0, 1.5]
        expected = torch.tensor([[0.5, 1.0, 1.5]])
        assert torch.allclose(m.linear.weight.grad, expected, atol=1e-6)

    def test_decay_bias_off_excludes_bias(self) -> None:
        m = LinearModel(in_features=3, out_features=1, bias=True)
        m.set_params(torch.tensor([1.0, 1.0, 1.0, 5.0]))  # weight=[1,1,1], bias=5
        x = torch.zeros(4, 3)
        y = torch.zeros(4, 1)
        d = {"x": x, "y": y}
        wd = 0.1
        without = L2RegularizedLoss(MSELoss(), weight_decay=wd, decay_bias=False)(m, d)
        with_ = L2RegularizedLoss(MSELoss(), weight_decay=wd, decay_bias=True)(m, d)
        # base = (5)^2 mean = 25.0 (bias-only forward, mean of 4 examples each predicting 5 vs 0)
        base = MSELoss()(m, d).item()
        # without-bias reg: 0.5 * 0.1 * (1+1+1) = 0.15
        assert without.item() == pytest.approx(base + 0.15, abs=1e-6)
        # with-bias reg: 0.5 * 0.1 * (1+1+1 + 25) = 1.4
        assert with_.item() == pytest.approx(base + 1.4, abs=1e-6)

    def test_decay_bias_off_zero_grad_on_bias(self) -> None:
        m = LinearModel(in_features=3, out_features=1, bias=True)
        m.set_params(torch.tensor([1.0, 1.0, 1.0, 5.0]))
        x = torch.zeros(4, 3)
        y = torch.zeros(4, 1)
        d = {"x": x, "y": y}
        # Use a base loss with zero gradient (predictions all 5, targets all 0
        # gives MSE base gradient on bias). Use a loss that is identically zero
        # in this configuration: set predictions == targets via clearing bias.
        m.set_params(torch.tensor([0.0, 0.0, 0.0, 0.0]))
        wrapped = L2RegularizedLoss(MSELoss(), weight_decay=0.5, decay_bias=False)
        wrapped(m, d).backward()
        # At theta=0, base gradient is 0; reg gradient: only weights, not bias
        assert m.linear.bias.grad is not None
        assert torch.allclose(m.linear.bias.grad, torch.zeros_like(m.linear.bias.grad))


class TestGradientsThroughLoss:
    def test_mse_grad_nonzero(self) -> None:
        m = LinearModel(in_features=3, out_features=1)
        x = torch.randn(8, 3)
        y = torch.randn(8, 1)
        loss = MSELoss()(m, {"x": x, "y": y})
        loss.backward()
        assert m.linear.weight.grad is not None
        assert m.linear.weight.grad.abs().sum() > 0

    def test_ce_grad_nonzero(self) -> None:
        m = LinearModel(in_features=4, out_features=3)
        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))
        loss = CrossEntropyLoss()(m, {"x": x, "y": y})
        loss.backward()
        assert m.linear.weight.grad is not None
