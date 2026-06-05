"""Tests for the competing-platforms world and round loop."""

from __future__ import annotations

import torch

from perfsim.competition import run_competition
from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.scenarios.recommender import (
    CompetingRecommendersWorld,
    RecommenderEcosystemWorld,
    build_competition_env,
    sample_corpus,
    sample_user_population,
)


def _entities(seed: int = 0):
    gen = torch.Generator()
    gen.manual_seed(seed)
    corpus = sample_corpus(10, 4, generator=gen)
    users = sample_user_population(50, 4, n_communities=3, generator=gen)
    return corpus, users


def _models(n: int, seed: int = 0) -> list[LinearModel]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    models = [LinearModel(4, 1) for _ in range(n)]
    with torch.no_grad():
        for m in models:
            m.linear.weight.copy_(torch.randn(1, 4, generator=gen))
    return models


class TestRun:
    def test_per_platform_data_shapes(self) -> None:
        env = build_competition_env(n_platforms=2, n_items=10, n_users=50, dim=4)
        env.reset(seed=0)
        out = env.run(_models(2), n_steps=3)
        assert len(out) == 2
        for data in out:
            assert data["x"].shape == (10, 4)
            assert data["y"].shape == (10, 1)
            assert data["agent_idx"].shape == (10,)

    def test_wrong_model_count_raises(self) -> None:
        env = build_competition_env(n_platforms=2, n_items=10, n_users=50, dim=4)
        env.reset(seed=0)
        try:
            env.run(_models(3), n_steps=1)
        except ValueError:
            return
        raise AssertionError("expected ValueError")


class TestShares:
    def test_fixed_when_eta_mob_zero(self) -> None:
        env = build_competition_env(n_platforms=2, n_items=10, n_users=50, dim=4, eta_mob=0.0)
        env.reset(seed=0)
        env.run(_models(2), n_steps=5)
        assert torch.allclose(env.shares, torch.full((50, 2), 0.5))

    def test_mobility_keeps_simplex(self) -> None:
        env = build_competition_env(n_platforms=3, n_items=10, n_users=50, dim=4, eta_mob=0.5)
        env.reset(seed=0)
        env.run(_models(3), n_steps=5)
        sums = env.shares.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        assert not torch.allclose(env.shares, torch.full((50, 3), 1.0 / 3.0))

    def test_mobility_favors_better_platform(self) -> None:
        corpus, users = _entities()
        env = CompetingRecommendersWorld(
            corpus, users, n_platforms=2, eta=0.0, eta_mob=0.5, beta=8.0,
        )
        good = LinearModel(4, 1)
        bad = LinearModel(4, 1)
        with torch.no_grad():
            q = env.hidden_quality()
            w = torch.linalg.lstsq(env.features, q.unsqueeze(1)).solution
            good.linear.weight.copy_(w.t())
            good.linear.bias.zero_()
            bad.linear.weight.copy_(-w.t())
            bad.linear.bias.zero_()
        env.reset(seed=0)
        env.run([good, bad], n_steps=10)
        assert float(env.shares[:, 0].mean()) > 0.5


class TestSinglePlatformConsistency:
    def test_equal_shares_identical_models_match_single_world(self) -> None:
        corpus, users = _entities()
        single = RecommenderEcosystemWorld(corpus, users, alpha=4.0, beta=4.0, eta=0.15)
        multi = CompetingRecommendersWorld(
            corpus, users, n_platforms=2, alpha=4.0, beta=4.0, eta=0.15, eta_mob=0.0,
        )
        model = _models(1)[0]
        single.reset(seed=0)
        multi.reset(seed=0)
        y_single = single.run(model, n_steps=4)["y"]
        out = multi.run([model, model], n_steps=4)
        assert torch.allclose(out[0]["y"], y_single, atol=1e-5)
        assert torch.allclose(out[1]["y"], y_single, atol=1e-5)
        assert torch.allclose(multi.current_interest, single.current_interest, atol=1e-5)


class TestLoop:
    def test_run_competition_trains_both(self) -> None:
        env = build_competition_env(n_platforms=2, n_items=10, n_users=50, dim=4)
        models = _models(2)
        learners = [ERMLearner(m, MSELoss()) for m in models]
        theta0 = [m.get_params().clone() for m in models]
        history = run_competition(env, learners, n_rounds=3, epoch_size=2, seed=0)
        assert len(history) == 3
        for m, t0 in zip(models, theta0):
            assert not torch.allclose(m.get_params(), t0)
