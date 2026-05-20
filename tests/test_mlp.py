"""Unit tests for MLPModel."""

from __future__ import annotations

import pytest
import torch

from perfsim.models import MLPModel


class TestConstruction:
    def test_default_relu_one_hidden(self) -> None:
        m = MLPModel(in_features=3, hidden_dims=[8], out_features=1)
        assert m.in_features == 3
        assert m.hidden_dims == (8,)
        assert m.out_features == 1

    def test_multi_hidden(self) -> None:
        m = MLPModel(in_features=4, hidden_dims=[16, 8, 4], out_features=2)
        assert m.hidden_dims == (16, 8, 4)
        assert m.out_features == 2

    def test_no_hidden_layers(self) -> None:
        # Degenerate: same as a LinearModel but via MLPModel.
        m = MLPModel(in_features=3, hidden_dims=[], out_features=1)
        x = torch.randn(5, 3)
        out = m(x)
        assert out.shape == (5, 1)

    def test_unknown_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown activation"):
            MLPModel(in_features=3, hidden_dims=[4], activation="quadratic")

    def test_unknown_final_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown final_activation"):
            MLPModel(in_features=3, hidden_dims=[4], final_activation="softplus")


class TestForward:
    def test_output_shape(self) -> None:
        m = MLPModel(in_features=5, hidden_dims=[8, 4], out_features=2)
        x = torch.randn(16, 5)
        out = m(x)
        assert out.shape == (16, 2)

    def test_final_sigmoid_in_unit_interval(self) -> None:
        m = MLPModel(
            in_features=3, hidden_dims=[4], final_activation="sigmoid"
        )
        x = torch.randn(8, 3)
        y = m(x)
        assert (y >= 0.0).all()
        assert (y <= 1.0).all()

    def test_logit_no_final_activation_unbounded(self) -> None:
        m = MLPModel(in_features=3, hidden_dims=[4])
        x = 1000 * torch.randn(8, 3)
        y = m(x)
        # At large inputs, no final activation -> outputs are large too.
        assert y.abs().max().item() > 1.0


class TestInitialization:
    def test_xavier_init_non_zero(self) -> None:
        m = MLPModel(in_features=3, hidden_dims=[8, 8], out_features=1)
        first_layer = next(
            mod for mod in m.modules() if isinstance(mod, torch.nn.Linear)
        )
        assert first_layer.weight.abs().sum().item() > 0.0
        assert first_layer.bias.abs().sum().item() == 0.0

    def test_init_seed_reproducible(self) -> None:
        m1 = MLPModel(in_features=3, hidden_dims=[4], init_seed=42)
        m2 = MLPModel(in_features=3, hidden_dims=[4], init_seed=42)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_init_different_seeds_differ(self) -> None:
        m1 = MLPModel(in_features=3, hidden_dims=[4], init_seed=1)
        m2 = MLPModel(in_features=3, hidden_dims=[4], init_seed=2)
        any_diff = False
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            if not torch.equal(p1, p2):
                any_diff = True
                break
        assert any_diff


class TestModelInterface:
    def test_get_set_params_round_trip(self) -> None:
        m = MLPModel(in_features=3, hidden_dims=[4], out_features=1, init_seed=0)
        theta = m.get_params()
        scrambled = torch.randn_like(theta)
        m.set_params(scrambled)
        assert torch.allclose(m.get_params(), scrambled)
        m.set_params(theta)
        assert torch.allclose(m.get_params(), theta)

    def test_clone_independent(self) -> None:
        m = MLPModel(in_features=3, hidden_dims=[4], init_seed=0)
        copy = m.clone()
        # Same params now
        assert torch.equal(m.get_params(), copy.get_params())
        # Mutate copy
        copy.set_params(torch.zeros_like(copy.get_params()))
        # Original unchanged
        assert m.get_params().abs().sum().item() > 0.0


class TestBackward:
    def test_gradient_through_inputs(self) -> None:
        m = MLPModel(in_features=3, hidden_dims=[4], init_seed=0)
        x = torch.randn(8, 3, requires_grad=True)
        y = m(x).sum()
        y.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_through_params(self) -> None:
        m = MLPModel(in_features=3, hidden_dims=[4], init_seed=0)
        x = torch.randn(8, 3)
        y = m(x).sum()
        y.backward()
        for p in m.parameters():
            assert p.grad is not None
