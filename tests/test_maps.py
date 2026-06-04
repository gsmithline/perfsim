"""Tests for distribution maps, access channels, and the canonical families."""

from __future__ import annotations

import pytest
import torch

from perfsim.maps import (
    AccessError,
    ModelView,
    TransformationMap,
    access_levels,
)
from perfsim.core.types import FEATURES_SCHEMA, SUPERVISED_SCHEMA, DataSchema
from perfsim.environments.dynamics.gaussian_shift import GaussianShiftWorld
from perfsim.maps import GaussianShiftMap, LocationScaleMap, StrategicLinearMap
from perfsim.models import LinearModel


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class _PeekingMap(TransformationMap):
    """Default predictions channel but transform reads params; must raise."""

    @property
    def produces_schema(self) -> DataSchema:
        return FEATURES_SCHEMA

    def sample_base(self, n, *, generator):
        return {"x": torch.randn(n, 2, generator=generator)}

    def transform(self, z_base, model):
        _ = model.params
        return z_base


class TestModelView:
    def test_predict_always_open(self) -> None:
        model = LinearModel(2, 1, bias=False)
        view = ModelView(model, expose_params=False)
        out = view.predict(torch.randn(4, 2))
        assert out.shape == (4, 1)

    def test_params_closed(self) -> None:
        view = ModelView(LinearModel(2, 1), expose_params=False)
        with pytest.raises(AccessError):
            _ = view.params

    def test_params_open(self) -> None:
        model = LinearModel(2, 1, bias=False)
        view = ModelView(model, expose_params=True)
        assert view.params.numel() == 2

    def test_undeclared_param_read_raises(self) -> None:
        with pytest.raises(AccessError):
            _PeekingMap().sample(LinearModel(2, 1), 4, generator=_gen())

    def test_bad_channel_rejected_at_class_creation(self) -> None:
        with pytest.raises(TypeError):

            class _Bad(TransformationMap):
                model_channel = "weights"


class TestAccessLevels:
    def test_levels(self) -> None:
        gauss = GaussianShiftMap(0.5 * torch.eye(2), torch.ones(2))
        loc = LocationScaleMap(torch.zeros(4, 2), M=torch.eye(2))
        assert access_levels(gauss) == frozenset({"1b", "2a", "2b"})
        assert access_levels(loc) == frozenset({"1b", "2a"})


class TestLocationScaleMap:
    def test_location_shift(self) -> None:
        M = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        lsm = LocationScaleMap(torch.zeros(5, 3), M=M)
        model = LinearModel(2, 1, bias=False)
        model.set_params(torch.tensor([1.0, 2.0]))
        data = lsm.sample(model, 4, generator=_gen())
        expected = M @ torch.tensor([1.0, 2.0])
        assert torch.allclose(data["x"], expected.expand(4, 3))

    def test_scale_shift(self) -> None:
        S = torch.eye(3).unsqueeze(0)  # P=1, scales identity by theta
        x0 = torch.randn(10, 3, generator=_gen(1))
        lsm = LocationScaleMap(x0, S=S)
        model = LinearModel(1, 1, bias=False)
        model.set_params(torch.tensor([0.5]))
        base = lsm.sample_base(6, generator=_gen(2))
        out = lsm.transform(base, lsm.view(model))
        assert torch.allclose(out["x"], 1.5 * base["x"])

    def test_determinism(self) -> None:
        lsm = LocationScaleMap(torch.randn(20, 3, generator=_gen(1)), M=torch.eye(3))
        model = LinearModel(3, 1, bias=False)
        a = lsm.sample(model, 8, generator=_gen(7))
        b = lsm.sample(model, 8, generator=_gen(7))
        assert torch.equal(a["x"], b["x"])

    def test_schema(self) -> None:
        x0 = torch.zeros(4, 2)
        assert LocationScaleMap(x0, M=torch.eye(2)).produces_schema is FEATURES_SCHEMA
        labeled = LocationScaleMap(x0, torch.zeros(4, 1), M=torch.eye(2))
        assert labeled.produces_schema is SUPERVISED_SCHEMA

    def test_wrong_theta_dim(self) -> None:
        lsm = LocationScaleMap(torch.zeros(4, 2), M=torch.eye(2))
        with pytest.raises(ValueError):
            lsm.sample(LinearModel(3, 1, bias=False), 4, generator=_gen())

    def test_requires_M_or_S(self) -> None:
        with pytest.raises(ValueError):
            LocationScaleMap(torch.zeros(4, 2))


class TestStrategicLinearMap:
    def _setup(self, bias: bool = False):
        x0 = torch.randn(8, 4, generator=_gen(3))
        y = torch.randn(8, 1, generator=_gen(4))
        smap = StrategicLinearMap(x0, y, epsilon=0.5, strat_features=[1, 3])
        model = LinearModel(4, 1, bias=bias)
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        return x0, y, smap, model

    def test_only_strat_features_move(self) -> None:
        x0, y, smap, model = self._setup()
        out = smap.transform({"x": x0, "y": y}, smap.view(model))
        assert torch.equal(out["x"][:, 0], x0[:, 0])
        assert torch.equal(out["x"][:, 2], x0[:, 2])
        assert torch.allclose(out["x"][:, 1], x0[:, 1] + 0.5 * 2.0)
        assert torch.allclose(out["x"][:, 3], x0[:, 3] + 0.5 * 4.0)
        assert torch.equal(out["y"], y)

    def test_trailing_bias_dropped(self) -> None:
        x0, y, smap, no_bias = self._setup(bias=False)
        _, _, _, with_bias = self._setup(bias=True)
        a = smap.transform({"x": x0, "y": y}, smap.view(no_bias))
        b = smap.transform({"x": x0, "y": y}, smap.view(with_bias))
        assert torch.allclose(a["x"], b["x"])

    def test_sample_shapes(self) -> None:
        _, _, smap, model = self._setup()
        data = smap.sample(model, 16, generator=_gen())
        assert data["x"].shape == (16, 4)
        assert data["y"].shape == (16, 1)
        assert data["agent_idx"].shape == (16,)


class TestGaussianShiftMap:
    def test_fixed_point_matches_world(self) -> None:
        A = 0.3 * torch.eye(3)
        b = torch.tensor([1.0, -1.0, 2.0])
        world = GaussianShiftWorld(A, b)
        gmap = GaussianShiftMap(A, b)
        assert torch.allclose(world.closed_form_fp(), gmap.closed_form_fp())

    def test_sample_matches_world(self) -> None:
        A = 0.3 * torch.eye(3)
        b = torch.tensor([1.0, -1.0, 2.0])
        world = GaussianShiftWorld(A, b, batch_size=32)
        gmap = GaussianShiftMap(A, b)
        model = LinearModel(3, 1, bias=False)
        model.set_params(torch.tensor([0.5, 0.5, 0.5]))
        world.reset(seed=11)
        from_world = world.step(model)
        from_map = gmap.sample(model, 32, generator=_gen(11))
        assert torch.equal(from_world["x"], from_map["x"])
        assert torch.allclose(from_world["y"], from_map["y"])

    def test_log_prob_prefers_true_theta(self) -> None:
        gmap = GaussianShiftMap(torch.eye(3), torch.zeros(3), sigma_noise=0.5)
        true_model = LinearModel(3, 1, bias=False)
        true_model.set_params(torch.tensor([1.0, -1.0, 2.0]))
        off_model = LinearModel(3, 1, bias=False)
        off_model.set_params(torch.tensor([2.0, 0.0, 3.0]))
        z = gmap.sample(true_model, 2000, generator=_gen(5))
        assert gmap.log_prob(z, true_model).mean() > gmap.log_prob(z, off_model).mean()

    def test_invalid_sigma(self) -> None:
        with pytest.raises(ValueError):
            GaussianShiftMap(torch.eye(2), torch.zeros(2), sigma_noise=0.0)
