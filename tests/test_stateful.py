"""Tests for stateful combinators and the D_inf / PR_inf helpers."""

from __future__ import annotations

import pytest
import torch

from perfsim.environments import GeometricDecayEnv, MapEnvironment, StaggeredResponseEnv
from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.maps import GaussianShiftMap
from perfsim.metrics import limiting_distribution, long_term_performative_risk
from perfsim.models import LinearModel
from perfsim.simulator import Simulator


def _map() -> GaussianShiftMap:
    return GaussianShiftMap(0.5 * torch.eye(3), torch.ones(3), sigma_noise=0.01)


def _model() -> LinearModel:
    m = LinearModel(3, 1, bias=False)
    m.set_params(torch.tensor([0.5, 0.5, 0.5]))
    return m


class TestGeometricDecay:
    def test_schema_passthrough(self) -> None:
        env = GeometricDecayEnv(_map(), lam=0.5)
        assert env.produces_schema is _map().produces_schema

    def test_sample_is_peek(self) -> None:
        env = GeometricDecayEnv(_map(), lam=0.5, batch_size=32)
        m = _model()
        env.reset(seed=0)
        env.step(m)  # seed the buffer
        a = env.sample(m)
        b = env.sample(m)
        assert torch.equal(a["x"], b["x"])

    def test_step_advances(self) -> None:
        env = GeometricDecayEnv(_map(), lam=0.5, batch_size=32)
        m = _model()
        env.reset(seed=0)
        a = env.step(m)
        b = env.step(m)
        assert not torch.equal(a["x"], b["x"])

    def test_batch_size_preserved(self) -> None:
        env = GeometricDecayEnv(_map(), lam=0.3, batch_size=64)
        m = _model()
        env.reset(seed=0)
        for _ in range(5):
            data = env.step(m)
        assert data["x"].shape[0] == 64

    def test_bad_lam(self) -> None:
        with pytest.raises(ValueError):
            GeometricDecayEnv(_map(), lam=1.5)


class TestStaggeredResponse:
    def test_emits_batch_size(self) -> None:
        env = StaggeredResponseEnv(_map(), k=3, batch_size=48)
        m = _model()
        env.reset(seed=0)
        for _ in range(5):
            data = env.step(m)
        assert data["x"].shape[0] == 48

    def test_bad_k(self) -> None:
        with pytest.raises(ValueError):
            StaggeredResponseEnv(_map(), k=0)


class TestLimitingHelpers:
    def test_pr_inf_finite(self) -> None:
        env = GeometricDecayEnv(_map(), lam=0.7, batch_size=256)
        pr = long_term_performative_risk(env, _model(), MSELoss(), n_iters=50)
        assert torch.isfinite(pr)

    def test_limiting_recovers_map_mean(self) -> None:
        # D_inf of geometric decay equals D(theta); compare label mean to a direct draw.
        gmap = _map()
        m = _model()
        env = GeometricDecayEnv(gmap, lam=0.8, batch_size=4096)
        lim = limiting_distribution(env, m, n_iters=100, seed=0)
        direct = gmap.sample(m, 4096, generator=torch.Generator().manual_seed(1))
        assert torch.allclose(lim["y"].mean(), direct["y"].mean(), atol=0.05)


class TestRRMThroughTransient:
    def test_geometric_decay_still_converges_to_fixed_point(self) -> None:
        gmap = _map()
        env = GeometricDecayEnv(gmap, lam=0.5, batch_size=512)
        model = LinearModel(3, 1, bias=False)
        loss = MSELoss()
        sim = Simulator(env=env, learner=ERMLearner(model, loss), loss=loss)
        sim.run(n_rounds=40, epoch_size=1, seed=0)
        assert torch.allclose(model.get_params(), gmap.closed_form_fp(), atol=0.1)
