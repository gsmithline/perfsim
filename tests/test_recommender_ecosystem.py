"""Tests for the performative recommender ecosystem."""

from __future__ import annotations

import torch

from perfsim.core.environment import Differentiable, FullyDifferentiable
from perfsim.models import LinearModel
from perfsim.scenarios.recommender import (
    RecommenderEcosystemWorld,
    build_recommender_env,
    sample_corpus,
    sample_user_population,
)


def _small_env(alpha: float = 4.0, beta: float = 4.0, seed: int = 0) -> RecommenderEcosystemWorld:
    gen = torch.Generator()
    gen.manual_seed(seed)
    corpus = sample_corpus(10, 4, generator=gen)
    users = sample_user_population(50, 4, n_communities=3, generator=gen)
    return RecommenderEcosystemWorld(corpus, users, alpha=alpha, beta=beta)


class TestSchema:
    def test_run_emits_supervised_data(self) -> None:
        env = _small_env()
        model = LinearModel(4, 1)
        env.reset(seed=0)
        data = env.run(model, n_steps=5)
        assert set(("x", "y")).issubset(data.keys())
        assert data["x"].shape == (10, 4)
        assert data["y"].shape == (10, 1)
        assert data["agent_idx"].shape == (10,)


class TestChoiceSimplex:
    def test_exposure_on_simplex(self) -> None:
        env = _small_env()
        model = LinearModel(4, 1)
        env.reset(seed=0)
        env.run(model, n_steps=1)
        u = env.last_exposure
        assert torch.isclose(u.sum(), torch.tensor(1.0), atol=1e-5)
        assert (u >= 0).all()

    def test_choice_rows_sum_to_one(self) -> None:
        env = _small_env()
        model = LinearModel(4, 1)
        env.reset(seed=0)
        env.run(model, n_steps=1)
        row_sums = env.last_choice.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestExposureBias:
    def test_higher_alpha_boosts_exposed_item(self) -> None:
        # A model that scores item 0 highest (weight = item-0 features).
        def make(alpha: float) -> RecommenderEcosystemWorld:
            env = _small_env(alpha=alpha, beta=5.0, seed=1)
            model = LinearModel(4, 1)
            with torch.no_grad():
                model.linear.weight.copy_(env.features[0].reshape(1, -1))
                model.linear.bias.zero_()
            env.reset(seed=1)
            env.run(model, n_steps=1)
            top = int(env.last_exposure.argmax())
            return env, top

        env0, top0 = make(0.0)
        env5, top5 = make(8.0)
        assert top0 == top5  # same ranking; only the bias strength differs
        assert float(env5.last_engagement[top5]) > float(env0.last_engagement[top0])


class TestPopulationDrift:
    def test_interest_drifts_from_initial(self) -> None:
        env = _small_env()
        model = LinearModel(4, 1)
        with torch.no_grad():
            model.linear.weight.copy_(env.features[0].reshape(1, -1))
        env.reset(seed=0)
        env.run(model, n_steps=10)
        drift = (env.current_interest - env.interest0).norm(dim=1).mean()
        assert float(drift) > 0.0


class TestDifferentiable:
    def test_traits(self) -> None:
        env = _small_env()
        assert isinstance(env, Differentiable)
        assert isinstance(env, FullyDifferentiable)

    def test_gradient_flows_to_theta(self) -> None:
        env = _small_env()
        model = LinearModel(4, 1)
        env.reset(seed=0)
        data = env.grad_sample(model)
        loss = data["y"].sum()
        loss.backward()
        g = model.linear.weight.grad
        assert g is not None
        assert float(g.abs().sum()) > 0.0


class TestBuildHelper:
    def test_build_recommender_env_runs(self) -> None:
        env = build_recommender_env(n_items=12, n_users=40, dim=5, seed=2)
        model = LinearModel(5, 1)
        env.reset(seed=2)
        data = env.run(model, n_steps=4)
        assert data["y"].shape == (12, 1)