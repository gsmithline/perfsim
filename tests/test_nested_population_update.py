"""Round operator for the AB population (population_update=nested_ai_then_social_v1).

    h(t) = k innate + (1-k) x(t)
    z(t) = (1-W) h(t) + W m(t)   if |m(t) - x(t)| < eps_AI   else h(t)
    x(t+1) = D_eps_social(z(t))

The gate is evaluated on the start-of-round opinion x(t), the platform/innate
mixture happens once per round, and peer (Deffuant) dynamics run last.
"""
import importlib.util
import os

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(HERE, os.pardir, "experiments", "scripts",
                     "cluster_pipelines", "_gated_pop.py")
_spec = importlib.util.spec_from_file_location("_gated_pop", _PATH)
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)

EPS_AI = 0.2


def _setup(n=6, w=0.5, k=0.2):
    x0 = torch.tensor([0.10, 0.30, 0.50, 0.70, 0.90, 0.50])
    served = torch.tensor([0.15, 0.90, 0.55, 0.10, 0.95, 0.70])  # 0,2,4 in radius
    innate = torch.tensor([0.60, 0.60, 0.60, 0.60, 0.60, 0.60])
    return x0, served, innate, torch.full((n,), w), k


def _human(innate, x0, k):
    return k * innate + (1.0 - k) * x0


# 1. accepted W=1 gives served before peers, for several k
@pytest.mark.parametrize("k", [0.0, 0.2, 0.5, 1.0])
def test_accepted_w1_is_served_for_any_k(k):
    x0, served, innate, w, _ = _setup(w=1.0)
    z, gate = gp.nested_presocial_update(x0, served, innate, k, w, EPS_AI)
    assert gate.any(), "fixture must contain accepted agents"
    assert torch.allclose(z[gate], served[gate], atol=0), \
        "a gated W=1 agent must land exactly on the served value, independent of k"


# 2. rejected agents get the human component, zero AI contribution
def test_rejected_agents_get_human_only():
    x0, served, innate, w, k = _setup()
    z, gate = gp.nested_presocial_update(x0, served, innate, k, w, EPS_AI)
    h = _human(innate, x0, k)
    assert (~gate).any(), "fixture must contain rejected agents"
    assert torch.allclose(z[~gate], h[~gate], atol=0)


# 3. W=0 gives the human component everywhere
def test_w_zero_is_human_everywhere():
    x0, served, innate, w, k = _setup(w=0.0)
    z, _ = gp.nested_presocial_update(x0, served, innate, k, w, EPS_AI)
    assert torch.allclose(z, _human(innate, x0, k), atol=0)


# 4. k=0 reduces to the ordinary gated platform blend
def test_k_zero_matches_legacy_gated_blend():
    x0, served, innate, w, _ = _setup()
    z, _ = gp.nested_presocial_update(x0, served, innate, 0.0, w, EPS_AI)
    legacy, _ = gp.gated_blend(x0.clone(), served, w, EPS_AI)
    assert torch.allclose(z, legacy, atol=0)


# 5. the gate is strict <: equality at eps_AI is rejected
def test_gate_is_strict_less_than():
    # binary-exact values (0.5, 0.25, 0.75, 0.625): |m - x| is EXACTLY eps for
    # agent 0, so the boundary is really tested rather than float32 rounding
    eps = 0.25
    x0 = torch.tensor([0.50, 0.50])
    served = torch.tensor([0.75, 0.625])          # |dx| = 0.25 (==eps), 0.125
    innate = torch.tensor([0.60, 0.60])
    w, k = torch.full((2,), 1.0), 0.2
    assert float((served[0] - x0[0]).abs()) == eps, "boundary case must be exact"
    z, gate = gp.nested_presocial_update(x0, served, innate, k, w, eps)
    assert not bool(gate[0]), "|m - x| == eps_AI must be REJECTED (strict <)"
    assert bool(gate[1])
    assert torch.allclose(z[0], _human(innate, x0, k)[0], atol=0)


# 6/7. the mixture happens once per round; peers consume z and run after it
def test_mixture_once_and_peers_run_on_z():
    x0, served, innate, w, k = _setup()
    z, _ = gp.nested_presocial_update(x0, served, innate, k, w, EPS_AI)
    z_twice, _ = gp.nested_presocial_update(z, served, innate, k, w, EPS_AI)
    assert not torch.allclose(z, z_twice), \
        "guard: applying the mixture twice must differ, so 'once' is testable"

    n = x0.numel()
    adj = torch.ones(n, n) - torch.eye(n)
    for ab_sweeps in (1, 3):
        state = z.clone()
        gen = torch.Generator().manual_seed(0)
        for _ in range(ab_sweeps):                       # social dynamics only
            gp.ab_sweep(state, adj, 1.0, 0.0, gen=gen)
        assert torch.allclose(state.mean(), z.mean(), atol=1e-5), \
            "Deffuant midpoint moves conserve the mean -- peers started from z"
        # the pre-social state is z regardless of sweep count
        assert torch.allclose(
            gp.nested_presocial_update(x0, served, innate, k, w, EPS_AI)[0], z,
            atol=0)


# 8. the no-AI twin: innate mixing, then the same social dynamics, no gate
def test_twin_is_human_then_social():
    x0, served, innate, _w, k = _setup()
    h = _human(innate, x0, k)
    n = x0.numel()
    adj = torch.ones(n, n) - torch.eye(n)
    twin = h.clone()
    gp.ab_sweep(twin, adj, 1.0, 0.0, gen=torch.Generator().manual_seed(0))
    assert torch.allclose(twin.mean(), h.mean(), atol=1e-5)
    # the twin never sees `served`: its pre-social state is h exactly
    z, _ = gp.nested_presocial_update(x0, served, innate, k, torch.zeros(n), EPS_AI)
    assert torch.allclose(z, h, atol=0)


# 9. eps_social = 0 -> peers inert -> saved opinion equals z exactly
def test_eps_social_zero_leaves_z_untouched():
    x0, served, innate, w, k = _setup()
    z, _ = gp.nested_presocial_update(x0, served, innate, k, w, EPS_AI)
    state = z.clone()
    n = x0.numel()
    adj = torch.ones(n, n) - torch.eye(n)
    for _ in range(3):
        gp.ab_sweep(state, adj, 0.0, 0.0, gen=torch.Generator().manual_seed(0))
    assert torch.equal(state, z), "eps_social=0 accepts no pair; z must survive"


# fixed points quoted in the correction: W=0.5, k=0.2, m=0.25, innate=0.63
def test_nested_fixed_point():
    w, k, m, x_in = torch.tensor([0.5]), 0.2, torch.tensor([0.25]), torch.tensor([0.63])
    x = x_in.clone()
    for _ in range(500):
        x, gate = gp.nested_presocial_update(x, m, x_in, k, w, 10.0)  # gate always open
        assert bool(gate.all())
    assert x.item() == pytest.approx(0.3133, abs=5e-4)
