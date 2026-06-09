"""Tests for MixtureShiftMap: independent sensitivity / modality knobs, all
three access levels, and a correct closed-form density."""

from __future__ import annotations

import math

import pytest
import torch

from perfsim.core.types import SUPERVISED_SCHEMA
from perfsim.maps import MixtureShiftMap, access_levels
from perfsim.models import LinearModel


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _model(weights, bias=False):
    d = len(weights)
    m = LinearModel(d, 1, bias=bias)
    with torch.no_grad():
        m.linear.weight.copy_(torch.tensor([weights]))
    return m


class TestAccessAndShape:
    def test_exposes_all_levels(self) -> None:
        assert access_levels(MixtureShiftMap(3)) == frozenset({"samples", "mechanism", "density"})

    def test_schema(self) -> None:
        assert MixtureShiftMap(2).produces_schema is SUPERVISED_SCHEMA

    def test_sample_shapes(self) -> None:
        m = MixtureShiftMap(4)
        data = m.sample(_model([1.0, 1.0, 1.0, 1.0]), 16, generator=_gen())
        assert data["x"].shape == (16, 4)
        assert data["y"].shape == (16, 1)

    def test_determinism(self) -> None:
        m = MixtureShiftMap(3, n_modes=2)
        model = _model([1.0, 0.0, 0.0])
        a = m.sample(model, 8, generator=_gen(7))
        b = m.sample(model, 8, generator=_gen(7))
        assert torch.equal(a["x"], b["x"])


class TestSensitivityKnob:
    def test_only_strat_features_move(self) -> None:
        m = MixtureShiftMap(4, epsilon=0.5, strat_features=[1, 3])
        base = m.sample_base(6, generator=_gen(1))
        out = m.transform(base, m.view(_model([1.0, 2.0, 3.0, 4.0])))
        assert torch.equal(out["x"][:, 0], base["x"][:, 0])
        assert torch.equal(out["x"][:, 2], base["x"][:, 2])
        assert torch.allclose(out["x"][:, 1], base["x"][:, 1] + 0.5 * 2.0)
        assert torch.allclose(out["x"][:, 3], base["x"][:, 3] + 0.5 * 4.0)

    def test_epsilon_zero_is_identity(self) -> None:
        m = MixtureShiftMap(3, epsilon=0.0)
        base = m.sample_base(5, generator=_gen(2))
        out = m.transform(base, m.view(_model([1.0, 2.0, 3.0])))
        assert torch.equal(out["x"], base["x"])

    def test_larger_epsilon_shifts_more(self) -> None:
        base = MixtureShiftMap(3).sample_base(5, generator=_gen(3))
        model = _model([1.0, 1.0, 1.0])
        small = MixtureShiftMap(3, epsilon=0.2).transform(base, MixtureShiftMap(3).view(model))
        large = MixtureShiftMap(3, epsilon=2.0).transform(base, MixtureShiftMap(3).view(model))
        assert (large["x"] - base["x"]).abs().sum() > (small["x"] - base["x"]).abs().sum()


class TestModalityKnob:
    def test_unimodal_single_center_at_origin(self) -> None:
        assert torch.allclose(MixtureShiftMap(3, n_modes=1).mode_centers, torch.zeros(1, 3))

    def test_modes_separated(self) -> None:
        c = MixtureShiftMap(2, n_modes=2, separation=4.0).mode_centers
        assert c.shape == (2, 2)
        assert torch.allclose(c[:, 0], torch.tensor([-2.0, 2.0]))

    def test_separation_independent_of_epsilon(self) -> None:
        a = MixtureShiftMap(2, n_modes=3, separation=5.0, epsilon=0.0).mode_centers
        b = MixtureShiftMap(2, n_modes=3, separation=5.0, epsilon=9.0).mode_centers
        assert torch.equal(a, b)


class TestDensity:
    def test_unimodal_matches_closed_form(self) -> None:
        d, sigma, ln = 3, 1.0, 0.1
        m = MixtureShiftMap(d, n_modes=1, sigma=sigma, label_noise=ln, epsilon=0.0)
        z = {"x": torch.zeros(1, d), "y": torch.zeros(1, 1)}
        expected = (-0.5 * d * math.log(2 * math.pi * sigma ** 2)
                    - 0.5 * math.log(2 * math.pi * ln ** 2))
        assert torch.allclose(m.log_prob(z, _model([0.0, 0.0, 0.0])),
                              torch.tensor([expected]), atol=1e-5)

    def test_peaks_at_a_mode(self) -> None:
        m = MixtureShiftMap(2, n_modes=2, separation=6.0, epsilon=0.0)
        model = _model([1.0, 1.0])
        at_mode = {"x": torch.tensor([[3.0, 0.0]]), "y": torch.tensor([[3.0]])}
        valley = {"x": torch.tensor([[0.0, 0.0]]), "y": torch.tensor([[0.0]])}
        assert m.log_prob(at_mode, model).item() > m.log_prob(valley, model).item()

    def test_shift_moves_the_peak(self) -> None:
        m = MixtureShiftMap(2, n_modes=1, epsilon=1.0, strat_features=[0])
        model = _model([2.0, 0.0])           # shift = +2 on dim 0
        shifted = {"x": torch.tensor([[2.0, 0.0]]), "y": torch.tensor([[2.0]])}
        unshifted = {"x": torch.tensor([[0.0, 0.0]]), "y": torch.tensor([[0.0]])}
        assert m.log_prob(shifted, model).item() > m.log_prob(unshifted, model).item()

    def test_log_prob_prefers_true_theta(self) -> None:
        m = MixtureShiftMap(3, n_modes=2, separation=4.0, epsilon=0.8, strat_features=[0, 1])
        true_model = _model([1.0, -1.0, 2.0])
        off_model = _model([3.0, 2.0, -1.0])
        z = m.sample(true_model, 3000, generator=_gen(5))
        assert m.log_prob(z, true_model).mean() > m.log_prob(z, off_model).mean()


class TestValidation:
    def test_bad_dim(self) -> None:
        with pytest.raises(ValueError):
            MixtureShiftMap(0)

    def test_wrong_theta_dim(self) -> None:
        with pytest.raises(ValueError):
            MixtureShiftMap(3).sample(_model([1.0, 1.0]), 4, generator=_gen())
