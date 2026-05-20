"""Unit tests for StrategicGradientWorld."""

from __future__ import annotations

import pytest
import torch

from perfsim.models import LinearModel, MLPModel
from perfsim.worlds.strategic_gradient import StrategicGradientWorld
from perfsim.worlds.strategic_linear import StrategicLinearWorld


def _fixed_population(n: int, d: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, d, generator=g)
    y = (torch.rand(n, generator=g) > 0.5).float().unsqueeze(-1)
    return x0, y


class TestEquivalenceWithLinear:
    """StrategicGradientWorld with a LinearModel must match StrategicLinearWorld."""

    def test_full_features_match(self) -> None:
        n, d, eps = 50, 4, 0.5
        x0, y = _fixed_population(n, d)
        w_grad = StrategicGradientWorld(x0=x0, y=y, epsilon=eps)
        w_lin = StrategicLinearWorld(x0=x0, y=y, epsilon=eps)

        model = LinearModel(in_features=d, out_features=1, bias=False)
        model.set_params(torch.tensor([1.0, -0.3, 0.5, 0.2]))

        d_grad = w_grad.sample(model)
        d_lin = w_lin.sample(model)

        assert torch.allclose(d_grad["x"], d_lin["x"], atol=1e-6)
        assert torch.equal(d_grad["y"], d_lin["y"])

    def test_strat_features_subset_match(self) -> None:
        n, d, eps = 30, 5, 1.5
        x0, y = _fixed_population(n, d)
        sf = (0, 2, 4)
        w_grad = StrategicGradientWorld(x0=x0, y=y, epsilon=eps, strat_features=sf)
        w_lin = StrategicLinearWorld(x0=x0, y=y, epsilon=eps, strat_features=sf)

        model = LinearModel(in_features=d, out_features=1, bias=False)
        model.set_params(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))

        d_grad = w_grad.sample(model)
        d_lin = w_lin.sample(model)
        assert torch.allclose(d_grad["x"], d_lin["x"], atol=1e-6)

    def test_with_bias_still_matches(self) -> None:
        # Bias drops out of ∂f/∂x, so the strategic shift is the same.
        n, d, eps = 20, 3, 0.8
        x0, y = _fixed_population(n, d)
        w_grad = StrategicGradientWorld(x0=x0, y=y, epsilon=eps)
        w_lin = StrategicLinearWorld(x0=x0, y=y, epsilon=eps)

        model = LinearModel(in_features=d, out_features=1, bias=True)
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[0.5, -0.5, 1.0]]))
            model.linear.bias.fill_(7.0)

        d_grad = w_grad.sample(model)
        d_lin = w_lin.sample(model)
        assert torch.allclose(d_grad["x"], d_lin["x"], atol=1e-6)


class TestMLPNonTrivial:
    def test_shift_depends_on_input(self) -> None:
        # With an MLP, ∂f/∂x varies row-by-row, so the per-agent shift
        # should NOT be constant across rows.
        n, d, eps = 20, 4, 0.5
        x0, y = _fixed_population(n, d)
        world = StrategicGradientWorld(x0=x0, y=y, epsilon=eps)
        mlp = MLPModel(
            in_features=d, hidden_dims=[8], out_features=1, init_seed=0
        )
        data = world.sample(mlp)
        shifts = data["x"] - x0
        # Each row should have a different shift (no constant offset).
        row_diffs = (shifts - shifts[0]).abs().sum(dim=-1)
        assert row_diffs.max().item() > 1e-4

    def test_zero_epsilon_no_shift(self) -> None:
        n, d = 10, 3
        x0, y = _fixed_population(n, d)
        world = StrategicGradientWorld(x0=x0, y=y, epsilon=0.0)
        mlp = MLPModel(in_features=d, hidden_dims=[4], out_features=1, init_seed=0)
        data = world.sample(mlp)
        assert torch.allclose(data["x"], x0)

    def test_strat_features_only_those_move(self) -> None:
        n, d, eps = 10, 5, 1.0
        x0, y = _fixed_population(n, d)
        sf = (1, 3)
        world = StrategicGradientWorld(
            x0=x0, y=y, epsilon=eps, strat_features=sf
        )
        mlp = MLPModel(in_features=d, hidden_dims=[4], init_seed=0)
        data = world.sample(mlp)
        shift = data["x"] - x0
        non_strat = [i for i in range(d) if i not in sf]
        assert shift[:, non_strat].abs().max().item() == 0.0
        assert shift[:, list(sf)].abs().max().item() > 0.0


class TestValidation:
    def test_x0_must_be_2d(self) -> None:
        with pytest.raises(ValueError, match="x0 must be 2-D"):
            StrategicGradientWorld(x0=torch.zeros(5), y=torch.zeros(5))

    def test_y_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            StrategicGradientWorld(x0=torch.zeros(5, 3), y=torch.zeros(4))

    def test_strat_features_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=">= d="):
            StrategicGradientWorld(
                x0=torch.zeros(5, 3), y=torch.zeros(5), strat_features=[0, 5]
            )

    def test_strat_features_duplicates(self) -> None:
        with pytest.raises(ValueError, match="must be unique"):
            StrategicGradientWorld(
                x0=torch.zeros(5, 3), y=torch.zeros(5), strat_features=[0, 1, 1]
            )
