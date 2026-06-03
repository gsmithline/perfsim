"""Tests for the producer ecosystem (replicating content supply)."""

from __future__ import annotations

import torch

from perfsim.core.environment import Differentiable, FullyDifferentiable
from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.scenarios.recommender import build_producer_env
from perfsim.scenarios.recommender.probes import producer_diversity, user_welfare
from perfsim.simulator import Simulator


def _trained_diversity(boost: float, *, n_rounds: int = 20, seed: int = 0) -> float:
    env = build_producer_env(
        n_items=30, n_users=150, dim=6, alpha=4.0, beta=8.0,
        gamma_supply=1.0, repl_rate=0.1, boost=boost, seed=seed,
    )
    model = LinearModel(6, 1)
    loss = MSELoss()
    sim = Simulator(env=env, learner=ERMLearner(model, loss), loss=loss)
    sim.run(n_rounds=n_rounds, epoch_size=1, seed=seed)
    return producer_diversity(env)


class TestSchema:
    def test_run_emits_supervised_data(self) -> None:
        env = build_producer_env(n_items=12, n_users=40, dim=5, seed=1)
        model = LinearModel(5, 1)
        env.reset(seed=1)
        data = env.run(model, n_steps=4)
        assert data["x"].shape == (12, 5)
        assert data["y"].shape == (12, 1)


class TestAvailabilitySimplex:
    def test_availability_on_simplex_after_run(self) -> None:
        env = build_producer_env(n_items=20, n_users=60, dim=5, seed=0)
        model = LinearModel(5, 1)
        with torch.no_grad():
            model.linear.weight.copy_(env.features[0].reshape(1, -1))
        env.reset(seed=0)
        env.run(model, n_steps=8)
        a = env.last_availability
        assert torch.isclose(a.sum(), torch.tensor(1.0), atol=1e-5)
        assert (a >= 0).all()


class TestRichGetRicher:
    def test_diversity_collapses_under_myopic_ranker(self) -> None:
        env = build_producer_env(
            n_items=30, n_users=150, dim=6, alpha=4.0, beta=8.0,
            gamma_supply=1.0, repl_rate=0.1, boost=0.0, seed=0,
        )
        model = LinearModel(6, 1)
        loss = MSELoss()
        sim = Simulator(env=env, learner=ERMLearner(model, loss), loss=loss)

        divs: list[float] = []
        sim.run(n_rounds=20, epoch_size=1, seed=0,
                on_round=lambda t, r: divs.append(producer_diversity(env)))
        # starts near-uniform (many live topics) and collapses substantially
        assert divs[0] > 10.0
        assert divs[-1] < divs[0] / 2.0


class TestBoostPreservesDiversity:
    def test_boost_keeps_more_producers_alive(self) -> None:
        div_myopic = _trained_diversity(boost=0.0)
        div_boosted = _trained_diversity(boost=2.0)
        assert div_boosted > div_myopic


class TestDifferentiable:
    def test_traits_and_gradient(self) -> None:
        env = build_producer_env(n_items=12, n_users=40, dim=5, seed=2)
        assert isinstance(env, Differentiable)
        assert isinstance(env, FullyDifferentiable)
        model = LinearModel(5, 1)
        env.reset(seed=2)
        data = env.grad_sample(model)
        data["y"].sum().backward()
        g = model.linear.weight.grad
        assert g is not None and float(g.abs().sum()) > 0.0
