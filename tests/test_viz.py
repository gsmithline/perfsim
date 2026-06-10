"""Tests for perfsim.viz on the closed-form GaussianShiftWorld (theta* = 2)."""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("matplotlib")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perfsim import viz
from perfsim.environments.dynamics import GaussianShiftWorld
from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.simulator import Simulator


def _world(batch: int = 512) -> GaussianShiftWorld:
    # RRM iterates theta -> 0.5 theta + 1; stable = optimal = 2.
    w = GaussianShiftWorld(A=torch.tensor([[0.5]]), b=torch.tensor([1.0]),
                           sigma_noise=0.01, batch_size=batch)
    w.reset(seed=0)
    return w


def _model(theta: float = 0.0) -> LinearModel:
    m = LinearModel(in_features=1, out_features=1, bias=False)
    m.set_params(torch.tensor([theta]))
    return m


def _history(n_rounds: int = 6):
    sim = Simulator(env=_world(), learner=ERMLearner(_model(), MSELoss()), loss=MSELoss())
    return sim.run(n_rounds, seed=0)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


class TestThetaPath:
    def test_from_history(self) -> None:
        path = viz.theta_path(_history(4))
        assert path.shape == (4, 1)

    def test_from_tensor_passthrough(self) -> None:
        t = torch.randn(5, 3)
        assert torch.equal(viz.theta_path(t), t)


class TestRiskSurface:
    def test_finds_fixed_point(self) -> None:
        grid = torch.linspace(0.0, 3.0, 31)
        surf = viz.risk_surface(_world(), _model(), MSELoss(), grid, n_samples=2)
        assert surf.dpr.shape == (31, 31)
        assert torch.allclose(surf.pr, surf.dpr.diagonal())
        # stable and optimal coincide at 2.0 in this world
        assert abs(surf.alpha_stable - 2.0) <= 0.11
        assert abs(surf.alpha_opt - 2.0) <= 0.11
        assert torch.allclose(surf.theta_opt, torch.tensor([surf.alpha_opt]))

    def test_best_response_matches_closed_form(self) -> None:
        # BR(phi) = 0.5 phi + 1 for MSE regression on this map
        grid = torch.linspace(0.0, 3.0, 31)
        surf = viz.risk_surface(_world(), _model(), MSELoss(), grid, n_samples=2)
        expected = 0.5 * grid + 1.0
        assert (surf.best_response - expected).abs().max() <= 0.11

    def test_multiparam_requires_direction(self) -> None:
        m = LinearModel(in_features=2, out_features=1, bias=False)
        with pytest.raises(ValueError, match="direction"):
            viz.risk_surface(_world(), m, MSELoss(), torch.linspace(0, 1, 5))

    def test_project_inverts_slice(self) -> None:
        grid = torch.linspace(0.0, 3.0, 7)
        surf = viz.risk_surface(_world(), _model(), MSELoss(), grid, n_samples=1)
        thetas = surf.base + grid.unsqueeze(1) * surf.direction
        assert torch.allclose(surf.project(thetas), grid)


class TestPlots:
    def test_landscape_with_trajectory(self) -> None:
        grid = torch.linspace(0.0, 3.0, 15)
        surf = viz.risk_surface(_world(), _model(), MSELoss(), grid, n_samples=1)
        ax = viz.plot_landscape(surf, trajectories={"RRM": _history(4)})
        assert ax.get_xlabel() == "evaluated theta"

    def test_landscape_3d(self) -> None:
        grid = torch.linspace(0.0, 3.0, 15)
        surf = viz.risk_surface(_world(), _model(), MSELoss(), grid, n_samples=1)
        ax = viz.plot_landscape_3d(surf, trajectories={"RRM": _history(4)},
                                   cross_sections=[2.0])
        assert ax.name == "3d"

    def test_risk_curve_with_cross_sections(self) -> None:
        grid = torch.linspace(0.0, 3.0, 15)
        surf = viz.risk_surface(_world(), _model(), MSELoss(), grid, n_samples=1)
        ax = viz.plot_risk_curve(surf, cross_sections=[2.0])
        assert ax.get_ylabel() == "risk"

    def test_convergence_to_known_point(self) -> None:
        h = _history(6)
        ax = viz.plot_convergence({"RRM": h}, theta_star=torch.tensor([2.0]))
        y = ax.get_lines()[0].get_ydata()
        assert y[-1] < y[0]  # contraction toward the fixed point

    def test_convergence_stability_gap(self) -> None:
        ax = viz.plot_convergence({"RRM": _history(6)})
        assert "theta_{t-1}" in ax.get_ylabel()


class TestGradientNorms:
    def test_vanishes_at_fixed_point(self) -> None:
        thetas = torch.tensor([[0.0], [2.0]])
        norms = viz.gradient_norms(_world(), _model(), MSELoss(), thetas, n_samples=2)
        for key in ("model", "dist", "total"):
            assert norms[key].shape == (2,)
            assert torch.isfinite(norms[key]).all()
        # both decompositions are far smaller at the stable/optimal point
        assert norms["model"][1] < 0.1 * norms["model"][0]
        assert norms["total"][1] < 0.1 * norms["total"][0]

    def test_plot_runs_on_history(self) -> None:
        ax = viz.plot_gradient_norms(_world(), _model(), MSELoss(), _history(3),
                                     n_samples=1)
        assert ax.get_ylabel() == "gradient norm"
