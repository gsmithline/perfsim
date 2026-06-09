"""Tests for MapAccess: tier gating, unsupported-level rejection, wiring."""

from __future__ import annotations

import pytest
import torch

from perfsim.environments.map_env import MapEnvironment
from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.maps import AccessError, MapAccess, MixtureShiftMap, StrategicLinearMap
from perfsim.models import LinearModel


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _model(d: int = 3) -> LinearModel:
    m = LinearModel(d, 1, bias=False)
    m.set_params(torch.ones(d))
    return m


def _z(d: int = 3):
    return {"x": torch.zeros(2, d), "y": torch.zeros(2, 1)}


class TestGating:
    def test_samples_only(self) -> None:
        a = MapAccess(MixtureShiftMap(3), "samples", generator=_gen())
        assert a.sample(_model(), 8)["x"].shape == (8, 3)
        with pytest.raises(AccessError):
            a.sample_base(8)
        with pytest.raises(AccessError):
            a.transform({"x": torch.zeros(2, 3)}, _model())
        with pytest.raises(AccessError):
            a.log_prob(_z(), _model())

    def test_mechanism_adds_transform(self) -> None:
        a = MapAccess(MixtureShiftMap(3), "mechanism", generator=_gen())
        assert a.transform(a.sample_base(4), _model())["x"].shape == (4, 3)
        with pytest.raises(AccessError):
            a.log_prob(_z(), _model())

    def test_density_adds_log_prob(self) -> None:
        a = MapAccess(MixtureShiftMap(3), "density", generator=_gen())
        assert a.log_prob(a.sample(_model(), 5), _model()).shape == (5,)


class TestLevelSupport:
    def test_request_unsupported_level_raises(self) -> None:
        smap = StrategicLinearMap(torch.randn(8, 3, generator=_gen()),
                                  torch.randn(8, 1, generator=_gen(1)))
        MapAccess(smap, "mechanism")                  # ok: it's a TransformationMap
        with pytest.raises(AccessError):
            MapAccess(smap, "density")                # not a DensityMap

    def test_bad_level_name(self) -> None:
        with pytest.raises(ValueError):
            MapAccess(MixtureShiftMap(2), "everything")

    def test_survey_codes_accepted_as_aliases(self) -> None:
        a = MapAccess(MixtureShiftMap(3), "2a", generator=_gen())   # 2a == mechanism
        assert a.level == "mechanism"
        a.sample_base(4)
        with pytest.raises(AccessError):
            a.log_prob(_z(), _model())


class TestWiring:
    def test_env_access_handle(self) -> None:
        a = MapEnvironment(MixtureShiftMap(3)).access("density", generator=_gen())
        assert isinstance(a, MapAccess) and a.level == "density"

    def test_default_learner_uses_samples(self) -> None:
        assert ERMLearner(LinearModel(3, 1), MSELoss()).access_level == "samples"

    def test_missing_generator_raises(self) -> None:
        a = MapAccess(MixtureShiftMap(2), "samples")
        with pytest.raises(ValueError):
            a.sample(_model(2), 4)
