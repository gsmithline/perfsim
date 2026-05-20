"""Tests for Model ABC and LinearModel, LogisticModel."""

from __future__ import annotations

import pytest
import torch

from perfsim.core import Model
from perfsim.models import LinearModel, LogisticModel


class TestLinearModel:
    def test_forward_shape(self) -> None:
        m = LinearModel(in_features=4, out_features=2)
        x = torch.randn(16, 4)
        y = m(x)
        assert y.shape == (16, 2)

    def test_zero_init(self) -> None:
        m = LinearModel(in_features=4, out_features=2)
        assert torch.allclose(m.linear.weight, torch.zeros_like(m.linear.weight))
        assert torch.allclose(m.linear.bias, torch.zeros_like(m.linear.bias))

    def test_no_bias(self) -> None:
        m = LinearModel(in_features=4, out_features=2, bias=False)
        assert m.linear.bias is None
        assert m.num_params == 4 * 2

    def test_num_params_with_bias(self) -> None:
        m = LinearModel(in_features=4, out_features=2, bias=True)
        assert m.num_params == 4 * 2 + 2

    def test_get_params_shape(self) -> None:
        m = LinearModel(in_features=4, out_features=2)
        theta = m.get_params()
        assert theta.shape == (m.num_params,)

    def test_set_params_round_trip(self) -> None:
        m = LinearModel(in_features=4, out_features=2)
        theta = torch.randn(m.num_params)
        m.set_params(theta)
        assert torch.allclose(m.get_params(), theta)

    def test_set_params_wrong_size_raises(self) -> None:
        m = LinearModel(in_features=4, out_features=2)
        with pytest.raises(ValueError, match="elements"):
            m.set_params(torch.randn(m.num_params + 1))

    def test_clone_is_independent(self) -> None:
        m = LinearModel(in_features=4, out_features=2)
        m.set_params(torch.randn(m.num_params))
        c = m.clone()
        assert torch.allclose(m.get_params(), c.get_params())
        c.set_params(torch.zeros(c.num_params))
        # mutating clone does not affect original
        assert not torch.allclose(m.get_params(), torch.zeros(m.num_params))

    def test_device_default_cpu(self) -> None:
        m = LinearModel(in_features=4, out_features=2)
        assert m.device == torch.device("cpu")

    def test_forward_uses_params(self) -> None:
        m = LinearModel(in_features=3, out_features=1)
        m.set_params(torch.tensor([1.0, 2.0, 3.0, 0.5]))  # W=[1,2,3], b=0.5
        x = torch.tensor([[1.0, 1.0, 1.0]])
        y = m(x)
        assert torch.allclose(y, torch.tensor([[6.5]]))

    def test_gradients_flow(self) -> None:
        m = LinearModel(in_features=4, out_features=1)
        x = torch.randn(8, 4)
        y = m(x).sum()
        y.backward()
        assert m.linear.weight.grad is not None
        assert m.linear.bias.grad is not None


class TestLogisticModel:
    def test_outputs_in_unit_interval(self) -> None:
        m = LogisticModel(in_features=4, out_features=1)
        x = torch.randn(32, 4) * 10.0
        y = m(x)
        assert (y >= 0).all() and (y <= 1).all()

    def test_zero_init_gives_half(self) -> None:
        m = LogisticModel(in_features=4, out_features=1)
        x = torch.randn(8, 4)
        y = m(x)
        assert torch.allclose(y, torch.full_like(y, 0.5))

    def test_clone_independence(self) -> None:
        m = LogisticModel(in_features=4, out_features=1)
        m.set_params(torch.randn(m.num_params))
        c = m.clone()
        c.set_params(torch.zeros(c.num_params))
        assert not torch.allclose(m.get_params(), c.get_params())


class TestModelABC:
    def test_model_is_nn_module(self) -> None:
        m = LinearModel(in_features=3)
        assert isinstance(m, torch.nn.Module)

    def test_model_is_perfsim_model(self) -> None:
        m = LinearModel(in_features=3)
        assert isinstance(m, Model)
