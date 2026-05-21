"""Unit tests for metrics: PR, DPR, stability_gap, optimality_gap, has_converged."""

from __future__ import annotations

import torch

from perfsim.losses import MSELoss
from perfsim.metrics import (
    decoupled_risk,
    has_converged,
    optimality_gap,
    performative_risk,
    sensitivity_paired,
    sensitivity_sliced,
    stability_gap,
)
from perfsim.models import LinearModel
from perfsim.environments.dynamics import GaussianShiftWorld
from perfsim.environments.dynamics.strategic_linear import StrategicLinearWorld


def _make_world(d: int = 3, sigma: float = 0.01, batch: int = 256, seed: int = 0):
    A = 0.5 * torch.eye(d)
    b = torch.tensor([1.0, 0.5, -0.5])[:d].clone()
    w = GaussianShiftWorld(A=A, b=b, sigma_noise=sigma, batch_size=batch)
    w.reset(seed=seed)
    return w


class TestPerformativeRisk:
    def test_returns_scalar(self) -> None:
        w = _make_world()
        m = LinearModel(in_features=3, out_features=1, bias=False)
        pr = performative_risk(w, m, MSELoss())
        assert pr.ndim == 0

    def test_nonzero_for_random_model(self) -> None:
        w = _make_world()
        m = LinearModel(in_features=3, out_features=1, bias=False)
        m.set_params(torch.tensor([1.0, 0.0, 0.0]))
        pr = performative_risk(w, m, MSELoss())
        assert pr.item() > 0

    def test_does_not_advance_world(self) -> None:
        w = _make_world()
        m = LinearModel(in_features=3, out_features=1, bias=False)
        before = w._gen.get_state()  # type: ignore[union-attr]
        performative_risk(w, m, MSELoss())
        after = w._gen.get_state()  # type: ignore[union-attr]
        assert torch.equal(before, after)


class TestDecoupledRisk:
    def test_dpr_at_same_model_equals_pr(self) -> None:
        # Gating test 2 mechanic, repeated as a unit test
        w = _make_world()
        m = LinearModel(in_features=3, out_features=1, bias=False)
        m.set_params(torch.tensor([0.7, -0.3, 0.2]))
        loss = MSELoss()
        pr = performative_risk(w, m, loss)
        dpr = decoupled_risk(w, m, m, loss)
        assert torch.allclose(pr, dpr, atol=1e-7)

    def test_dpr_with_different_eval_differs(self) -> None:
        w = _make_world()
        m_deploy = LinearModel(in_features=3, out_features=1, bias=False)
        m_eval = LinearModel(in_features=3, out_features=1, bias=False)
        m_eval.set_params(torch.tensor([1.0, 1.0, 1.0]))
        loss = MSELoss()
        pr = performative_risk(w, m_deploy, loss)
        dpr = decoupled_risk(w, m_deploy, m_eval, loss)
        assert not torch.allclose(pr, dpr, atol=1e-2)


class TestStabilityGap:
    def test_zero_for_same_theta(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        assert stability_gap(t, t).item() == 0.0

    def test_norm_of_diff(self) -> None:
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([3.0, 4.0])
        assert torch.allclose(stability_gap(a, b), torch.tensor(5.0))


class TestOptimalityGap:
    def test_zero_when_models_equal(self) -> None:
        w = _make_world()
        m1 = LinearModel(in_features=3, out_features=1, bias=False)
        m2 = LinearModel(in_features=3, out_features=1, bias=False)
        m1.set_params(torch.tensor([0.5, 0.5, 0.5]))
        m2.set_params(torch.tensor([0.5, 0.5, 0.5]))
        gap = optimality_gap(w, m1, m2, MSELoss())
        assert torch.allclose(gap, torch.tensor(0.0), atol=1e-7)

    def test_optimal_model_at_fp_has_lower_pr(self) -> None:
        w = _make_world(sigma=1e-4, batch=2048)
        d = 3
        fp = w.closed_form_fp()
        m_opt = LinearModel(in_features=d, out_features=1, bias=False)
        m_opt.set_params(fp)
        m_bad = LinearModel(in_features=d, out_features=1, bias=False)
        m_bad.set_params(torch.zeros(d))
        gap = optimality_gap(w, m_bad, m_opt, MSELoss())
        assert gap.item() > 0


class TestSensitivityPaired:
    def test_strategic_linear_exact_epsilon(self) -> None:
        # In a StrategicLinearWorld with eps and strat_features=None, the
        # per-agent shift is exactly eps * (w_a - w_b). The paired-coupling
        # mean ||x_a_i - x_b_i|| / ||theta_a - theta_b|| equals |eps| since
        # x_a_i - x_b_i = eps * (w_a - w_b) is identical across i and w = theta
        # for a no-bias LinearModel.
        n, d, eps = 100, 4, 0.5
        x0 = torch.randn(n, d)
        y = torch.zeros(n, 1)
        world = StrategicLinearWorld(x0=x0, y=y, epsilon=eps)
        m_a = LinearModel(in_features=d, out_features=1, bias=False)
        m_b = LinearModel(in_features=d, out_features=1, bias=False)
        m_a.set_params(torch.tensor([1.0, 0.5, -0.3, 0.2]))
        m_b.set_params(torch.tensor([0.2, -0.1, 0.4, 0.7]))
        s = sensitivity_paired(world, m_a, m_b)
        assert torch.allclose(s, torch.tensor(eps), atol=1e-5)

    def test_paired_lipschitz_constant_across_pairs(self) -> None:
        # For a linear shift world the sensitivity should be constant in
        # whichever (theta_a, theta_b) we pick: that's the Lipschitz property.
        n, d, eps = 100, 4, 1.5
        x0 = torch.randn(n, d)
        y = torch.zeros(n, 1)
        world = StrategicLinearWorld(x0=x0, y=y, epsilon=eps)

        def _sens(t_a: list[float], t_b: list[float]) -> float:
            m_a = LinearModel(in_features=d, out_features=1, bias=False)
            m_b = LinearModel(in_features=d, out_features=1, bias=False)
            m_a.set_params(torch.tensor(t_a))
            m_b.set_params(torch.tensor(t_b))
            return sensitivity_paired(world, m_a, m_b).item()

        s1 = _sens([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        s2 = _sens([5.0, -2.0, 1.0, 0.3], [1.0, 1.0, 1.0, 1.0])
        assert abs(s1 - s2) < 1e-5

    def test_raises_on_equal_thetas(self) -> None:
        n, d = 50, 3
        world = StrategicLinearWorld(x0=torch.randn(n, d), y=torch.zeros(n, 1), epsilon=1.0)
        m = LinearModel(in_features=d, out_features=1, bias=False)
        m.set_params(torch.tensor([0.5, 0.5, 0.5]))
        try:
            sensitivity_paired(world, m, m)
        except ValueError:
            return
        raise AssertionError("expected ValueError for equal thetas")


class TestSensitivitySliced:
    def test_strategic_linear_lipschitz_constant(self) -> None:
        # Sliced W1 / ||Δθ|| should be ~constant across (a, b) pairs in a
        # linear shift world. Absolute value is a dimension constant times eps.
        n, d, eps = 200, 6, 1.0
        x0 = torch.randn(n, d)
        y = torch.zeros(n, 1)
        world = StrategicLinearWorld(x0=x0, y=y, epsilon=eps)

        def _sens(scale: float) -> float:
            m_a = LinearModel(in_features=d, out_features=1, bias=False)
            m_b = LinearModel(in_features=d, out_features=1, bias=False)
            m_a.set_params(scale * torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            m_b.set_params(torch.zeros(d))
            return sensitivity_sliced(world, m_a, m_b, n_proj=200, seed=0).item()

        s_small = _sens(0.1)
        s_large = _sens(10.0)
        # Slicing introduces a small finite-projection error but the ratio
        # should still be O(1) across two orders of magnitude in ||Δθ||.
        assert abs(s_small - s_large) / s_large < 0.05

    def test_raises_on_equal_thetas(self) -> None:
        n, d = 50, 3
        world = StrategicLinearWorld(x0=torch.randn(n, d), y=torch.zeros(n, 1), epsilon=1.0)
        m = LinearModel(in_features=d, out_features=1, bias=False)
        m.set_params(torch.tensor([0.5, 0.5, 0.5]))
        try:
            sensitivity_sliced(world, m, m)
        except ValueError:
            return
        raise AssertionError("expected ValueError for equal thetas")

    def test_gaussian_shift_returns_finite(self) -> None:
        d = 3
        A = 0.5 * torch.eye(d)
        b = torch.tensor([1.0, 0.5, -0.5])
        world = GaussianShiftWorld(A=A, b=b, sigma_noise=0.01, batch_size=512)
        world.reset(seed=0)
        m_a = LinearModel(in_features=d, out_features=1, bias=False)
        m_b = LinearModel(in_features=d, out_features=1, bias=False)
        m_a.set_params(torch.tensor([1.0, 0.0, 0.0]))
        m_b.set_params(torch.tensor([0.0, 0.0, 0.0]))
        s = sensitivity_sliced(world, m_a, m_b, n_proj=64, seed=0)
        # GaussianShift's x is theta-independent; the sliced W1 reflects only
        # finite-sample noise, so sensitivity should be small (but finite).
        assert torch.isfinite(s)
        assert s.item() >= 0.0


class TestHasConverged:
    def test_returns_false_with_fewer_than_window(self) -> None:
        thetas = [torch.zeros(3) for _ in range(3)]
        assert not has_converged(thetas, tol=1e-3, window=5)

    def test_converges_when_constant(self) -> None:
        thetas = [torch.ones(3) for _ in range(10)]
        assert has_converged(thetas, tol=1e-3, window=5)

    def test_does_not_converge_when_moving(self) -> None:
        thetas = [torch.zeros(3) + float(i) for i in range(10)]
        assert not has_converged(thetas, tol=1e-3, window=5)
