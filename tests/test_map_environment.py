"""Tests for the MapEnvironment adapter and the end-to-end RRM loop on maps."""

from __future__ import annotations

import pytest
import torch

from perfsim.core.environment import ClosedFormFixedPoint
from perfsim.environments import MapEnvironment
from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.maps import GaussianShiftMap, LocationScaleMap
from perfsim.models import LinearModel
from perfsim.simulator import Simulator


def _gauss_map() -> GaussianShiftMap:
    return GaussianShiftMap(0.5 * torch.eye(3), torch.ones(3), sigma_noise=0.01)


class TestAdapter:
    def test_schema_passthrough(self) -> None:
        env = MapEnvironment(_gauss_map())
        assert env.produces_schema is _gauss_map().produces_schema

    def test_sample_is_a_peek(self) -> None:
        env = MapEnvironment(_gauss_map(), batch_size=16)
        model = LinearModel(3, 1, bias=False)
        env.reset(seed=0)
        a = env.sample(model)
        b = env.sample(model)
        assert torch.equal(a["x"], b["x"])

    def test_step_advances(self) -> None:
        env = MapEnvironment(_gauss_map(), batch_size=16)
        model = LinearModel(3, 1, bias=False)
        env.reset(seed=0)
        a = env.step(model)
        b = env.step(model)
        assert not torch.equal(a["x"], b["x"])

    def test_fixed_point_delegation(self) -> None:
        with_fp = MapEnvironment(_gauss_map())
        without_fp = MapEnvironment(LocationScaleMap(torch.zeros(4, 2), M=torch.eye(2)))
        assert isinstance(with_fp, ClosedFormFixedPoint)
        assert not isinstance(without_fp, ClosedFormFixedPoint)
        assert torch.allclose(with_fp.closed_form_fp(), _gauss_map().closed_form_fp())

    def test_rejects_non_map(self) -> None:
        with pytest.raises(TypeError):
            MapEnvironment(object())  # type: ignore[arg-type]

    def test_rejects_bad_batch_size(self) -> None:
        with pytest.raises(ValueError):
            MapEnvironment(_gauss_map(), batch_size=0)


class TestRRMOnMap:
    def test_rrm_converges_to_closed_form(self) -> None:
        gmap = _gauss_map()
        env = MapEnvironment(gmap, batch_size=512)
        model = LinearModel(3, 1, bias=False)
        loss = MSELoss()
        sim = Simulator(env=env, learner=ERMLearner(model, loss), loss=loss)
        sim.run(n_rounds=15, epoch_size=1, seed=0)
        theta = model.get_params()
        assert torch.allclose(theta, gmap.closed_form_fp(), atol=0.05)
