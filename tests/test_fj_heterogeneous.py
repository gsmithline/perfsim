"""Per-agent (heterogeneous) alpha in FJWorld.run_wu (2026-08-22).

No model is loaded and nothing here touches a dataset: the Wu operator is
linear and fully specified, so every claim is checkable in closed form on
a hand-built graph.

The reference recurrence in `_ref_wu` is written as a per-agent python
loop on purpose. It is a deliberately INDEPENDENT expression of

    x_init_i = (1 - beta_i) innate_i + beta_i m_i
    u_i^(0)  = x_init_i
    u_i^(l+1)= (1 - alpha_i) x_init_i + alpha_i sum_j P_ij u_j^(l)

so a typo shared with the vectorised implementation cannot hide: comparing
the operator against a copy of itself would test nothing.

THE SABOTAGE TEST HERE is the vector complement guard. FJWorld stores
STUBBORNNESS in `peer_sus`, so alpha = 1 - peer_sus. Pokec ships a
per-agent susceptibility vector, and a per-agent swap is HARDER to catch
than the scalar one: the wrong vector still lies in [0, 1], its mean still
prints plausibly, and the resulting dynamics are the near-opposite of the
intended ones while every downstream number stays well-formed.
"""
from __future__ import annotations

import pytest
import torch

from perfsim.environments.dynamics import FJWorld

N = 5


def _graph5():
    """5 agents, connected, no isolated node, deliberately asymmetric in
    degree (3,2,3,2,2) so per-agent neighbourhood averages differ and a
    per-agent alpha has something to bite on."""
    adj = torch.tensor([[0., 1., 1., 0., 1.],
                        [1., 0., 1., 0., 0.],
                        [1., 1., 0., 1., 0.],
                        [0., 0., 1., 0., 1.],
                        [1., 0., 0., 1., 0.]])
    W = adj / adj.sum(dim=1, keepdim=True)
    innate = torch.tensor([0.10, 0.35, 0.55, 0.80, 0.95])
    return innate, W


def _vec(x, n=N):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32)
    return torch.full((n,), float(x), dtype=torch.float32)


def _world(innate, W, beta, alpha, *, peer_sus=None):
    """A world built with the CORRECT complement unless peer_sus is given
    explicitly (which is how the sabotage cases are constructed)."""
    n = innate.shape[0]
    ps = (FJWorld.alpha_to_peer_sus(_vec(alpha, n)) if peer_sus is None
          else _vec(peer_sus, n))
    return FJWorld(innate=innate, graph=W, peer_sus=ps,
                   platform_sus=_vec(beta, n), features=innate)


def _ref_wu(innate, W, beta, alpha, preds, n_inner):
    """Independent per-agent scalar reference. Returns (x_init, u1, uK)."""
    n = int(innate.shape[0])
    b, a = _vec(beta, n), _vec(alpha, n)
    x_init = [(1.0 - float(b[i])) * float(innate[i]) + float(b[i]) * float(preds[i])
              for i in range(n)]
    u = list(x_init)
    u1 = None
    for _ in range(n_inner):
        nxt = []
        for i in range(n):
            mix = sum(float(W[i][j]) * u[j] for j in range(n))
            nxt.append((1.0 - float(a[i])) * x_init[i] + float(a[i]) * mix)
        u = nxt
        if u1 is None:
            u1 = list(u)
    return (torch.tensor(x_init), torch.tensor(u1), torch.tensor(u))


# ------------------------------------------------- the scalar path is intact

@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_scalar_alpha_and_a_constant_vector_are_bit_identical(alpha):
    """The contract's clause (a): the float path must not change. At an
    alpha exactly representable in binary the two routes agree BIT for
    bit, so this is torch.equal, not allclose."""
    innate, W = _graph5()
    preds = torch.tensor([0.2, 0.9, 0.4, 0.15, 0.6])
    out = []
    for a in (alpha, torch.full((N,), float(alpha))):
        w = _world(innate, W, 0.5, alpha)      # world built from the float
        w.run_wu(preds, alpha=a, n_inner=6)
        out.append(w.state["opinion"].clone())
    assert torch.equal(out[0], out[1]), (out[0] - out[1]).abs().max()


def test_production_alpha_agrees_to_float32_complement_rounding():
    """alpha=.9 is NOT representable in binary. The float path forms
    1 - 0.9 in python double and rounds once; the tensor path forms it in
    float32 and rounds differently, so the two disagree by ~2e-7 -- one
    complement ulp, not a behavioural difference. Pinned honestly rather
    than papered over with a loose tolerance elsewhere."""
    innate, W = _graph5()
    preds = torch.tensor([0.2, 0.9, 0.4, 0.15, 0.6])
    out = []
    for a in (0.9, torch.full((N,), 0.9)):
        w = _world(innate, W, 0.5, 0.9)
        w.run_wu(preds, alpha=a, n_inner=100)
        out.append(w.state["opinion"].clone())
    gap = float((out[0] - out[1]).abs().max())
    assert gap < 1e-5, gap


def test_scalar_path_still_matches_the_reference_recurrence():
    """A guard against 'tidying' the float branch into the tensor one."""
    innate, W = _graph5()
    beta, alpha, K = 0.5, 0.9, 4
    preds = torch.tensor([0.2, 0.9, 0.4, 0.15, 0.6])
    w = _world(innate, W, beta, alpha)
    w.run_wu(preds, alpha=alpha, n_inner=K)
    _, _, uK = _ref_wu(innate, W, beta, alpha, preds, K)
    assert torch.allclose(w.state["opinion"], uK.float(), atol=1e-6)


# ----------------------------------------------- the elementwise recurrence

def test_vector_alpha_matches_a_hand_expansion():
    innate, W = _graph5()
    beta = torch.tensor([0.2, 0.5, 0.5, 0.9, 0.0])
    alpha = torch.tensor([0.0, 0.3, 0.6, 0.9, 1.0])
    preds = torch.tensor([0.7, 0.1, 0.9, 0.25, 0.5])
    for K in (1, 2, 3):
        w = _world(innate, W, beta, alpha)
        w.run_wu(preds, alpha=alpha, n_inner=K)
        x_init, u1, uK = _ref_wu(innate, W, beta, alpha, preds, K)
        assert torch.allclose(w.state["opinion"], uK.float(), atol=1e-6), K
        assert torch.allclose(w.last_x_init, x_init.float(), atol=1e-6), K
        assert torch.allclose(w.last_u1, u1.float(), atol=1e-6), K


def test_alpha_is_genuinely_per_agent():
    """An alpha=0 agent stays pinned at its own x_init while an alpha=.9
    NEIGHBOUR of it moves. If the operator collapsed the vector to a mean
    (or to its first entry) both agents would move together."""
    innate, W = _graph5()
    beta = 0.5
    alpha = torch.tensor([0.0, 0.9, 0.0, 0.9, 0.0])
    preds = torch.tensor([0.7, 0.1, 0.9, 0.25, 0.5])
    w = _world(innate, W, beta, alpha)
    w.run_wu(preds, alpha=alpha, n_inner=8)
    x_init = w.last_x_init
    x = w.state["opinion"]

    # non-vacuity: the mobile agents' neighbourhoods must disagree with
    # their own anchors, otherwise "no movement" would prove nothing
    nbr = W @ x_init
    for i in (1, 3):
        assert abs(float(nbr[i] - x_init[i])) > 1e-2, i

    for i in (0, 2, 4):
        assert float((x[i] - x_init[i]).abs()) < 1e-7, i     # alpha = 0
    for i in (1, 3):
        assert float((x[i] - x_init[i]).abs()) > 1e-3, i     # alpha = .9
    # agents 0 and 1 are neighbours, so this is not a disconnected artefact
    assert float(W[0, 1]) > 0 and float(W[1, 0]) > 0


def test_alpha_zero_vector_leaves_the_population_exactly_at_x_init():
    innate, W = _graph5()
    beta = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    preds = torch.tensor([0.7, 0.1, 0.9, 0.25, 0.5])
    w = _world(innate, W, beta, torch.zeros(N))
    w.run_wu(preds, alpha=torch.zeros(N), n_inner=50)
    want = (1.0 - beta) * innate + beta * preds
    assert torch.allclose(w.state["opinion"], want, atol=1e-7)
    assert torch.allclose(w.last_u1, want, atol=1e-7)


def test_alpha_one_vector_is_pure_mixing_with_no_anchor_term():
    innate, W = _graph5()
    beta = 0.5
    preds = torch.tensor([0.7, 0.1, 0.9, 0.25, 0.5])
    K = 3
    w = _world(innate, W, beta, torch.ones(N))
    w.run_wu(preds, alpha=torch.ones(N), n_inner=K)
    x_init = (1.0 - beta) * innate + beta * preds
    want = torch.linalg.matrix_power(W, K) @ x_init
    assert torch.allclose(w.state["opinion"], want, atol=1e-6)
    # non-vacuous: P^K x_init must actually differ from x_init, or
    # "pure mixing" and "pure anchoring" would be indistinguishable here
    assert float((want - x_init).abs().max()) > 1e-2


def test_two_dimensional_innate_broadcasts_alpha_across_features():
    """(N, D) opinions store peer_sus as (N, 1); a per-agent alpha must
    broadcast the same way or the columns silently decouple."""
    innate, W = _graph5()
    innate2 = torch.stack([innate, 1.0 - innate], dim=1)      # (N, 2)
    alpha = torch.tensor([0.0, 0.3, 0.6, 0.9, 1.0])
    beta = torch.tensor([0.2, 0.5, 0.5, 0.9, 0.0])
    preds2 = torch.stack([torch.tensor([0.7, 0.1, 0.9, 0.25, 0.5]),
                          torch.tensor([0.3, 0.8, 0.2, 0.60, 0.4])], dim=1)
    w = FJWorld(innate=innate2, graph=W,
                peer_sus=FJWorld.alpha_to_peer_sus(alpha),
                platform_sus=beta, features=innate2)
    w.run_wu(preds2, alpha=alpha, n_inner=3)
    for d in range(2):
        _, _, uK = _ref_wu(innate2[:, d], W, beta, alpha, preds2[:, d], 3)
        assert torch.allclose(w.state["opinion"][:, d], uK.float(), atol=1e-6), d


# ------------------------------------------------ the complement guard

def test_vector_guard_fires_when_the_world_is_built_with_alpha_itself():
    """THE sabotage case. peer_sus is STUBBORNNESS; a world built with the
    alpha vector raw runs 1 - alpha mixing."""
    innate, W = _graph5()
    alpha = torch.tensor([0.05, 0.30, 0.60, 0.90, 0.99])
    wrong = _world(innate, W, 0.5, alpha, peer_sus=alpha)    # alpha passed raw
    with pytest.raises(ValueError, match="STUBBORNNESS") as e:
        wrong.run_wu(torch.full((N,), 0.5), alpha=alpha, n_inner=100)
    msg = str(e.value)
    assert "alpha" in msg and "peer_sus" in msg               # names BOTH

    # and the guard is not decorative: the mis-built world, driven with the
    # complement so it passes, produces materially different dynamics
    right = _world(innate, W, 0.5, alpha)
    preds = torch.tensor([0.7, 0.1, 0.9, 0.25, 0.5])
    right.run_wu(preds, alpha=alpha, n_inner=100)
    inverted = _world(innate, W, 0.5, 1.0 - alpha, peer_sus=alpha)
    inverted.run_wu(preds, alpha=1.0 - alpha, n_inner=100)
    assert float((right.state["opinion"]
                  - inverted.state["opinion"]).abs().max()) > 1e-2


def test_vector_guard_fires_on_a_single_mismatched_agent():
    """The realistic failure is not a wholesale swap but one stale entry;
    the guard is elementwise, so it must catch a lone agent."""
    innate, W = _graph5()
    alpha = torch.tensor([0.05, 0.30, 0.60, 0.90, 0.99])
    ps = FJWorld.alpha_to_peer_sus(alpha).clone()
    ps[2] += 1e-3
    w = _world(innate, W, 0.5, alpha, peer_sus=ps)
    with pytest.raises(ValueError, match="STUBBORNNESS") as e:
        w.run_wu(torch.full((N,), 0.5), alpha=alpha, n_inner=1)
    assert "1/5" in str(e.value) and "agent 2" in str(e.value)


def test_vector_guard_tolerates_float_noise_at_1e_6():
    """Do not weaken the guard, but do not make it fire on float32 dust
    either: the documented tolerance is atol 1e-6."""
    innate, W = _graph5()
    alpha = torch.tensor([0.05, 0.30, 0.60, 0.90, 0.99])
    ps = FJWorld.alpha_to_peer_sus(alpha) + 5e-8
    w = _world(innate, W, 0.5, alpha, peer_sus=ps)
    w.run_wu(torch.full((N,), 0.5), alpha=alpha, n_inner=1)      # accepted
    assert torch.isfinite(w.state["opinion"]).all()


def test_a_scalar_alpha_is_still_rejected_against_a_heterogeneous_world():
    """A per-agent world silently averaged into one number is the other
    half of the same mistake."""
    innate, W = _graph5()
    alpha = torch.tensor([0.05, 0.30, 0.60, 0.90, 0.99])
    w = _world(innate, W, 0.5, alpha)
    with pytest.raises(ValueError, match="STUBBORNNESS"):
        w.run_wu(torch.full((N,), 0.5), alpha=float(alpha.mean()), n_inner=1)


# ------------------------------------------------------ input validation

@pytest.mark.parametrize("bad", [
    torch.zeros(N + 1),
    torch.zeros(N - 1),
    torch.zeros(N, 1),
    torch.zeros(1, N),
    torch.tensor(0.5),                 # 0-dim: pass a python float instead
])
def test_alpha_shape_rejection(bad):
    innate, W = _graph5()
    w = _world(innate, W, 0.5, 0.5)
    with pytest.raises(ValueError, match="shape"):
        w.run_wu(torch.full((N,), 0.5), alpha=bad, n_inner=1)


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_alpha_range_rejection(bad):
    innate, W = _graph5()
    w = _world(innate, W, 0.5, 0.5)
    a = torch.full((N,), 0.5)
    a[3] = bad
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        w.run_wu(torch.full((N,), 0.5), alpha=a, n_inner=1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_alpha_nonfinite_rejection(bad):
    innate, W = _graph5()
    w = _world(innate, W, 0.5, 0.5)
    a = torch.full((N,), 0.5)
    a[1] = bad
    with pytest.raises(ValueError, match="non-finite"):
        w.run_wu(torch.full((N,), 0.5), alpha=a, n_inner=1)


def test_prediction_and_n_inner_validation_still_applies_under_vector_alpha():
    innate, W = _graph5()
    alpha = torch.tensor([0.05, 0.30, 0.60, 0.90, 0.99])
    w = _world(innate, W, 0.5, alpha)
    with pytest.raises(ValueError):
        w.run_wu(torch.zeros(N + 1), alpha=alpha, n_inner=1)
    with pytest.raises(ValueError):
        w.run_wu(torch.full((N,), float("nan")), alpha=alpha, n_inner=1)
    with pytest.raises(ValueError):
        w.run_wu(torch.zeros(N), alpha=alpha, n_inner=0)


def test_run_wu_does_not_mutate_the_caller_s_alpha_vector():
    innate, W = _graph5()
    alpha = torch.tensor([0.05, 0.30, 0.60, 0.90, 0.99])
    before = alpha.clone()
    w = _world(innate, W, 0.5, alpha)
    w.run_wu(torch.full((N,), 0.5), alpha=alpha, n_inner=3)
    assert torch.equal(alpha, before)


# ------------------------------------------------------ u^(1) as evidence

def test_last_u1_is_the_first_inner_iterate_elementwise():
    innate, W = _graph5()
    beta = torch.tensor([0.2, 0.5, 0.5, 0.9, 0.0])
    alpha = torch.tensor([0.05, 0.30, 0.60, 0.90, 0.99])
    preds = torch.tensor([0.7, 0.1, 0.9, 0.25, 0.5])
    w = _world(innate, W, beta, alpha)
    w.run_wu(preds, alpha=alpha, n_inner=100)
    x_init = (1.0 - beta) * innate + beta * preds
    want = (1.0 - alpha) * x_init + alpha * (W @ x_init)
    assert torch.allclose(w.last_x_init, x_init, atol=1e-6)
    assert torch.allclose(w.last_u1, want, atol=1e-6)


def test_u1_distinguishes_the_initialisation_where_u_k_cannot():
    """The vacuity guard for the test above. At K=100 the inner loop has
    forgotten where it started, so pinning u^(K) proves nothing about
    u^(0) = x_init; u^(1) is affine in u^(0) with per-agent coefficient
    alpha_i, so it still carries it -- but only if the two candidate
    starts are DISTINGUISHABLE on this graph, which is what is asserted
    here rather than assumed."""
    innate, W = _graph5()
    beta = 0.5
    alpha = torch.tensor([0.80, 0.85, 0.90, 0.95, 0.99])
    preds = torch.tensor([0.7, 0.1, 0.9, 0.25, 0.5])
    x_init = (1.0 - beta) * innate + beta * preds
    stale = torch.tensor([0.95, 0.05, 0.05, 0.95, 0.05])   # a previous state
    a_run, b_run = x_init.clone(), stale.clone()
    a_u1 = b_u1 = None
    for _ in range(100):
        a_run = (1.0 - alpha) * x_init + alpha * (W @ a_run)
        b_run = (1.0 - alpha) * x_init + alpha * (W @ b_run)
        if a_u1 is None:
            a_u1, b_u1 = a_run.clone(), b_run.clone()
    assert float((a_u1 - b_u1).abs().max()) > 1e-2      # distinguishable at u^(1)
    assert float((a_run - b_run).abs().max()) < 1e-3    # gone by u^(K)

    # and the operator's own u^(1) is the x_init one, not the stale one
    w = _world(innate, W, beta, alpha)
    w.run_wu(preds, alpha=alpha, n_inner=100)
    assert torch.allclose(w.last_u1, a_u1, atol=1e-6)
    assert not torch.allclose(w.last_u1, b_u1, atol=1e-2)


# ------------------------------------------------------- the convention

def test_alpha_to_peer_sus_handles_tensors_and_floats():
    assert FJWorld.alpha_to_peer_sus(0.0) == 1.0
    assert FJWorld.alpha_to_peer_sus(1.0) == 0.0
    assert isinstance(FJWorld.alpha_to_peer_sus(0.5), float)

    a = torch.tensor([0.05, 0.30, 0.60, 0.90, 0.99])
    ps = FJWorld.alpha_to_peer_sus(a)
    assert isinstance(ps, torch.Tensor)
    assert ps.shape == a.shape and ps.dtype == a.dtype
    assert torch.allclose(ps, 1.0 - a)
    assert torch.allclose(FJWorld.alpha_to_peer_sus(ps), a, atol=1e-7)
    assert torch.equal(a, torch.tensor([0.05, 0.30, 0.60, 0.90, 0.99]))  # no mutation


def test_a_world_built_through_the_helper_is_accepted_by_run_wu():
    """The round trip the runner depends on: dataset alpha -> peer_sus ->
    run_wu, with no hand-written complement anywhere in between."""
    innate, W = _graph5()
    alpha = torch.tensor([0.05, 0.30, 0.60, 0.90, 0.99])
    w = FJWorld(innate=innate, graph=W,
                peer_sus=FJWorld.alpha_to_peer_sus(alpha),
                platform_sus=torch.full((N,), 0.5), features=innate)
    w.run_wu(torch.full((N,), 0.5), alpha=alpha, n_inner=100)
    assert torch.isfinite(w.state["opinion"]).all()
