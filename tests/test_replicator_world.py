"""Tests for ReplicatorWorld (Taylor-Jonker discrete replicator, torch port)."""

from __future__ import annotations

import pytest
import torch

from perfsim.environments.dynamics import ReplicatorWorld
from perfsim.models import LinearModel


def _constant_fitness(f_vec: torch.Tensor):
    """A fitness function that ignores (p, model) and returns f_vec."""
    return lambda p, model: f_vec


class _DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ----- Validation ------------------------------------------------------------


class TestConstruction:
    def test_p0_not_1d(self) -> None:
        with pytest.raises(ValueError, match="p0 must be 1-D"):
            ReplicatorWorld(
                p0=torch.tensor([[0.5, 0.5]]),
                fitness=_constant_fitness(torch.zeros(2)),
            )

    def test_p0_not_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="p0 must sum to 1"):
            ReplicatorWorld(
                p0=torch.tensor([0.3, 0.3]),
                fitness=_constant_fitness(torch.zeros(2)),
            )

    def test_p0_negative_entry(self) -> None:
        with pytest.raises(ValueError, match="p0 must be non-negative"):
            ReplicatorWorld(
                p0=torch.tensor([-0.1, 1.1]),
                fitness=_constant_fitness(torch.zeros(2)),
            )

    def test_n_ticks_below_one(self) -> None:
        with pytest.raises(ValueError, match="n_ticks must be"):
            ReplicatorWorld(
                p0=torch.tensor([0.5, 0.5]),
                fitness=_constant_fitness(torch.zeros(2)),
                n_ticks=0,
            )


# ----- Dynamics: simplex preservation ----------------------------------------


class TestSimplexInvariant:
    def test_step_preserves_simplex(self) -> None:
        p0 = torch.tensor([0.2, 0.3, 0.5])
        f = torch.tensor([1.0, -0.5, 0.0])
        world = ReplicatorWorld(p0=p0, fitness=_constant_fitness(f), n_ticks=1)
        world.step(_DummyModel())
        p = world.state["mixture"]
        assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-6)
        assert (p >= 0).all()

    def test_long_rollout_stays_on_simplex(self) -> None:
        p0 = torch.tensor([0.1, 0.4, 0.5])
        f = torch.tensor([0.5, 0.1, -0.2])
        world = ReplicatorWorld(p0=p0, fitness=_constant_fitness(f), n_ticks=200)
        world.step(_DummyModel())
        p = world.state["mixture"]
        assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-6)
        assert (p >= 0).all()


# ----- Match the source numpy code -------------------------------------------


class TestMatchSource:
    """Manually replay `evoml/dynamics.py::discrete_replicator` in plain torch
    and confirm `ReplicatorWorld` produces identical trajectories."""

    def _source_one_step(self, p: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        # Exact algebra from evoml/dynamics.py:13-14.
        f_shifted = f + 1
        return (p * f_shifted) / (p @ f_shifted)

    def test_one_tick_matches_source(self) -> None:
        p0 = torch.tensor([0.2, 0.3, 0.5])
        f = torch.tensor([1.0, -0.3, 0.5])
        world = ReplicatorWorld(p0=p0, fitness=_constant_fitness(f), n_ticks=1)
        world.step(_DummyModel())
        expected = self._source_one_step(p0, f)
        assert torch.allclose(world.state["mixture"], expected, atol=1e-6)

    def test_ten_ticks_match_source(self) -> None:
        p0 = torch.tensor([0.1, 0.5, 0.4])
        f = torch.tensor([0.4, -0.2, 0.1])
        world = ReplicatorWorld(p0=p0, fitness=_constant_fitness(f), n_ticks=10)
        # Reconstruct expected trajectory via the source algebra.
        expected = p0.clone()
        for _ in range(10):
            expected = self._source_one_step(expected, f)
        world.step(_DummyModel())
        assert torch.allclose(world.state["mixture"], expected, atol=1e-6)


# ----- Dominant-strategy convergence -----------------------------------------


class TestDominantStrategy:
    def test_pure_dominance_converges(self) -> None:
        # If strategy 1 has strictly highest fitness regardless of mixture, the
        # population should concentrate on it over many ticks.
        p0 = torch.tensor([0.4, 0.3, 0.3])
        f = torch.tensor([0.0, 1.0, -0.2])  # strategy 1 dominates
        world = ReplicatorWorld(p0=p0, fitness=_constant_fitness(f), n_ticks=300)
        world.step(_DummyModel())
        p = world.state["mixture"]
        assert p[1].item() > 0.95


# ----- PP coupling (fitness depends on the model) ----------------------------


class TestPPCoupling:
    def test_fitness_can_use_model(self) -> None:
        # Fitness uses the deployed model's parameter norm as a global modulator.
        p0 = torch.tensor([0.5, 0.5])
        captured = []

        def fit(p, model):
            captured.append(model)
            return torch.tensor([0.1, -0.1]) * (model.linear.weight.norm() + 0.5)

        model = LinearModel(in_features=1, out_features=1, bias=False)
        model.set_params(torch.tensor([2.0]))
        world = ReplicatorWorld(p0=p0, fitness=fit, n_ticks=1)
        world.step(model)
        assert len(captured) == 1
        assert captured[0] is model


# ----- Reset and state -------------------------------------------------------


class TestStateLifecycle:
    def test_reset_restores_p0(self) -> None:
        p0 = torch.tensor([0.3, 0.3, 0.4])
        f = torch.tensor([0.5, 0.0, -0.5])
        world = ReplicatorWorld(p0=p0, fitness=_constant_fitness(f), n_ticks=5)
        world.step(_DummyModel())
        world.step(_DummyModel())
        world.reset()
        assert torch.allclose(world.state["mixture"], p0, atol=1e-6)

    def test_sample_does_not_mutate(self) -> None:
        p0 = torch.tensor([0.5, 0.5])
        f = torch.tensor([0.5, -0.5])
        world = ReplicatorWorld(p0=p0, fitness=_constant_fitness(f), n_ticks=5)
        before = world.state["mixture"].clone()
        world.sample(_DummyModel())
        assert torch.allclose(world.state["mixture"], before)
