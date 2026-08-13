"""Unit tests for the centralized AI gate (2026-08-13, sft_icl_reach wave).

Covers: threshold mode is byte-identical to the legacy inline expression
(strict <, boundary cases included), all_open opens every gate bit and is NOT
equivalent to eps_ai=1, eps_ai=0 never opens (the baseline-probe guarantee),
gate evaluation consumes no RNG, nested_presocial_update under the default
mode reproduces the legacy operator bit-for-bit, and unknown modes raise.
"""
import importlib.util
import os

import pytest
import torch

_GP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "experiments", "scripts", "cluster_pipelines",
                   "_gated_pop.py")
_spec = importlib.util.spec_from_file_location("_gated_pop_test", _GP)
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)


def _rand(n=2048, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, generator=g), torch.rand(n, generator=g)


def test_threshold_matches_legacy_expression_bitwise():
    served, x0 = _rand()
    for eps in (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
        legacy = (served - x0).abs() < eps
        assert torch.equal(gp.ai_gate(served, x0, eps), legacy)
        assert torch.equal(gp.ai_gate(served, x0, eps, "threshold"), legacy)


def test_threshold_is_strict_at_the_boundary():
    served = torch.tensor([0.5, 0.5, 0.5])
    x0 = torch.tensor([0.1, 0.3, 0.5])       # |diff| = 0.4, 0.2, 0.0
    got = gp.ai_gate(served, x0, 0.4)
    assert got.tolist() == [False, True, True]


def test_all_open_is_all_true():
    served, x0 = _rand()
    g = gp.ai_gate(served, x0, 0.0, "all_open")
    assert g.dtype == torch.bool and g.shape == served.shape
    assert bool(g.all())


def test_all_open_is_not_eps_ai_1():
    # an agent at distance exactly 1 is REJECTED by the strict threshold
    # at eps_ai=1 but contacted under all_open -- the reason the explicit
    # mode exists
    served = torch.tensor([1.0])
    x0 = torch.tensor([0.0])
    assert not bool(gp.ai_gate(served, x0, 1.0).any())
    assert bool(gp.ai_gate(served, x0, 1.0, "all_open").all())


def test_eps_zero_never_opens():
    served, x0 = _rand()
    assert not bool(gp.ai_gate(served, x0, 0.0).any())
    assert not bool(gp.ai_gate(x0, x0, 0.0).any())   # even at distance 0


def test_gate_draws_no_rng():
    served, x0 = _rand()
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    gp.ai_gate(served, x0, 0.4)
    gp.ai_gate(served, x0, 0.4, "all_open")
    assert torch.equal(before, torch.get_rng_state())


def test_unknown_mode_raises():
    served, x0 = _rand(8)
    with pytest.raises(ValueError, match="AI_GATE_MODE"):
        gp.ai_gate(served, x0, 0.4, "open")


def test_nested_update_threshold_is_bitwise_legacy():
    served, x0 = _rand()
    g = torch.Generator().manual_seed(7)
    innate = torch.rand(x0.shape[0], generator=g)
    w_agent = torch.full_like(x0, 0.5)
    for k, eps in ((0.0, 0.4), (0.2, 0.4), (0.2, 0.05)):
        # legacy operator, byte-for-byte as it stood before gate_mode
        h = k * innate + (1.0 - k) * x0
        gate_l = (served - x0).abs() < eps
        eff_w = torch.where(gate_l, w_agent, torch.zeros_like(w_agent))
        z_l = (1.0 - eff_w) * h + eff_w * served
        z, gate = gp.nested_presocial_update(x0, served, innate, k,
                                             w_agent, eps)
        assert torch.equal(gate, gate_l)
        assert torch.equal(z, z_l)


def test_nested_update_all_open_blends_everyone():
    served, x0 = _rand()
    g = torch.Generator().manual_seed(7)
    innate = torch.rand(x0.shape[0], generator=g)
    w_agent = torch.full_like(x0, 0.5)
    k = 0.2
    z, gate = gp.nested_presocial_update(x0, served, innate, k, w_agent,
                                         0.0, gate_mode="all_open")
    assert bool(gate.all())
    h = k * innate + (1.0 - k) * x0
    assert torch.equal(z, (1.0 - w_agent) * h + w_agent * served)
