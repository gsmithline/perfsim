"""Tests for PEER_GATE_MODE (2026-08-20, qwen_wu_limit wave).

The peer (Deffuant) side gets the same opt-in the AI gate got in
2026-08-13: "threshold" (default) is the strict |x_i - x_j| < eps
confidence test, "all_open" accepts every sampled pair.

TWO THINGS MUST HOLD AND BOTH ARE LOAD-BEARING.

1. The default is byte-identical to every archived run. Thousands of
   completed trajectories were produced by the old inline expression; if
   the refactor moved a single bit, every reuse audit in the project
   silently becomes wrong.

2. all_open is NOT eps_social = 1. The test is a STRICT inequality, so a
   pair at (0, 1) sits at distance exactly 1 and eps = 1 REJECTS it --
   precisely the extreme pairs a consensus limit is about. If "open"
   could be spelled as a number, the Wu-boundary experiment would
   quietly exclude its own subject matter.

and one that makes the comparison meaningful:

3. Both modes sample the SAME pairs and consume the SAME RNG. The mode is
   applied after pair selection, so a threshold run and an all-open run
   differ only in which of the identical sampled pairs get accepted.

Run with USE_TF=0.
"""
import importlib.util
import os

import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gp = _load("gp_peer", os.path.join(PIPE, "_gated_pop.py"))


def _ring(n):
    """Connected ring + chords: every agent has neighbours, so the sweep
    never breaks out early off the largest connected component."""
    adj = torch.zeros(n, n)
    for i in range(n):
        adj[i, (i + 1) % n] = 1.0
        adj[(i + 1) % n, i] = 1.0
        adj[i, (i + 7) % n] = 1.0
        adj[(i + 7) % n, i] = 1.0
    return adj


def _state(n, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, generator=g)


# -- 1 the default is byte-identical to the legacy inline expression ------

def test_threshold_mode_is_the_legacy_sweep_bit_for_bit():
    """The pre-2026-08-20 body, inlined here verbatim, against the
    refactored sweep. gamma=0 is the project's standing policy."""
    def legacy(x, adj, eps, gamma, gen):
        n = x.shape[0]
        bsz = min(gp.AB_BATCH, n)
        done, accepted = 0, 0
        while done < n:
            ini = torch.randperm(n, generator=gen)[:bsz]
            ar = torch.arange(ini.shape[0])
            d = (x[ini, None] - x[None, :]).abs().clamp_min(gp.AB_MIN_DIST)
            wts = d.pow(-gamma) * adj[ini]
            wts[ar, ini] = 0.0
            ok_row = wts.sum(1) > 0
            if not bool(ok_row.any()):
                break
            ini, wts = ini[ok_row], wts[ok_row]
            ar = torch.arange(ini.shape[0])
            par = torch.multinomial(wts, 1, generator=gen).squeeze(1)
            pri = torch.rand(ini.shape[0], generator=gen)
            inc = torch.zeros(ini.shape[0], n, dtype=torch.bool)
            inc[ar, ini] = True
            inc[ar, par] = True
            best = torch.where(inc, pri[:, None], torch.tensor(2.0)).amin(0)
            keep = (pri <= best[ini]) & (pri <= best[par])
            idx = keep.nonzero().squeeze(1)[: n - done]
            i1, i2 = ini[idx], par[idx]
            ok = (x[i1] - x[i2]).abs() < eps
            a1, a2 = i1[ok], i2[ok]
            mid = 0.5 * (x[a1] + x[a2])
            x[a1] = mid
            x[a2] = mid
            done += len(idx)
            accepted += int(ok.sum())
        return accepted

    n = 120
    adj = _ring(n)
    for eps in (0.05, 0.2, 1.0):
        xa, xb = _state(n), _state(n)
        acc_a = legacy(xa, adj, eps, 0.0,
                       torch.Generator().manual_seed(424243))
        acc_b = gp.ab_sweep(xb, adj, eps, 0.0,
                            gen=torch.Generator().manual_seed(424243))
        assert torch.equal(xa, xb), f"eps={eps}: state differs"
        assert acc_a == acc_b, f"eps={eps}: {acc_a} != {acc_b}"


def test_default_argument_is_threshold():
    """An omitted gate_mode must mean threshold -- absent == legacy is
    the convention every archived config relies on."""
    n = 80
    adj = _ring(n)
    xa, xb = _state(n), _state(n)
    a = gp.ab_sweep(xa, adj, 0.1, 0.0,
                    gen=torch.Generator().manual_seed(1))
    b = gp.ab_sweep(xb, adj, 0.1, 0.0,
                    gen=torch.Generator().manual_seed(1),
                    gate_mode="threshold")
    assert torch.equal(xa, xb) and a == b


# -- 2 all_open accepts EVERY sampled pair --------------------------------

def test_all_open_accepts_every_selected_pair():
    """ab_sweep fills exactly n disjoint-pair slots per sweep, so under
    all_open the accepted count must equal n exactly -- this is the
    invariant the checker gates the Wu-limit runs on."""
    n = 120
    adj = _ring(n)
    for eps in (0.0001, 0.05, 1.0):
        x = _state(n)
        acc = gp.ab_sweep(x, adj, eps, 0.0,
                          gen=torch.Generator().manual_seed(7),
                          gate_mode="all_open")
        assert acc == n, f"eps={eps}: accepted {acc} != {n} sampled pairs"


def test_peer_gate_returns_all_true_under_all_open():
    d = torch.tensor([0.0, 0.5, 1.0, 2.0])
    assert bool(gp.peer_gate(d, 0.1, "all_open").all())
    assert bool(gp.peer_gate(d, 1.0, "all_open").all())


def test_unknown_mode_raises_rather_than_falling_back():
    """A typo must not silently become the threshold gate."""
    try:
        gp.peer_gate(torch.tensor([0.5]), 1.0, "open")
    except ValueError as exc:
        assert "PEER_GATE_MODE" in str(exc)
    else:
        raise AssertionError("unknown peer gate mode did not raise")


# -- 3 the exact 0-vs-1 pair: why "open" cannot be the number 1 -----------

def test_distance_one_pair_accepted_open_but_rejected_by_threshold_one():
    """THE reason all_open exists. A pair at (0, 1) is at distance
    exactly 1; the strict test 1 < 1 is False, so eps_social=1 REJECTS
    it while a genuinely open channel accepts it."""
    d = torch.tensor([1.0])
    assert not bool(gp.peer_gate(d, 1.0, "threshold").item()), \
        "strict < must reject a pair at distance exactly eps"
    assert bool(gp.peer_gate(d, 1.0, "all_open").item())


def test_distance_one_pair_moves_only_under_all_open_in_a_real_sweep():
    """The same thing end to end: two agents at 0 and 1 on a two-node
    graph. Under eps=1 they never meet; under all_open they go to the
    midpoint."""
    adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    x_thr = torch.tensor([0.0, 1.0])
    gp.ab_sweep(x_thr, adj, 1.0, 0.0,
                gen=torch.Generator().manual_seed(5),
                gate_mode="threshold")
    assert torch.equal(x_thr, torch.tensor([0.0, 1.0])), \
        "eps_social=1 must NOT move a pair at distance exactly 1"
    x_open = torch.tensor([0.0, 1.0])
    gp.ab_sweep(x_open, adj, 1.0, 0.0,
                gen=torch.Generator().manual_seed(5),
                gate_mode="all_open")
    assert torch.allclose(x_open, torch.tensor([0.5, 0.5]))


# -- 4 identical pairs and identical RNG consumption ---------------------

def _spy_sweep(mode, n, adj, eps=0.1, seed=424243):
    """Run one sweep, recording every distance vector handed to
    peer_gate and the generator's end state."""
    seen = []
    real = gp.peer_gate

    def _g(dist, eps_, mode_="threshold"):
        seen.append(dist.clone())
        return real(dist, eps_, mode_)

    try:
        gp.peer_gate = _g
        x = _state(n)
        gen = torch.Generator().manual_seed(seed)
        gp.ab_sweep(x, adj, eps, 0.0, gen=gen, gate_mode=mode)
        return seen, gen.get_state().clone(), x
    finally:
        gp.peer_gate = real


def test_both_modes_consume_identical_rng():
    """THE invariant that makes the two modes comparable. With gamma=0
    (the project's standing policy) the selection weights are the
    adjacency alone, so no draw depends on the opinion state: both modes
    run the same number of batches, draw the same permutations, partners
    and priorities, and leave the generator in the SAME state. Only the
    acceptance decision differs.

    NOTE the gamma=0 condition is load-bearing. With gamma>0 the partner
    weights are d.pow(-gamma), which DOES read x, so the two modes would
    diverge in selection once their states differ."""
    n = 120
    adj = _ring(n)
    seen_t, gen_t, _ = _spy_sweep("threshold", n, adj)
    seen_o, gen_o, _ = _spy_sweep("all_open", n, adj)
    assert len(seen_t) == len(seen_o), \
        "the two modes drew a different number of pair batches"
    assert torch.equal(gen_t, gen_o), \
        "the two modes left the generator in different states"


def test_first_batch_pairs_are_identical_then_states_legitimately_diverge():
    """Selection is mode-blind: the FIRST batch sees the same pairs at
    the same distances in both modes, because nothing has moved yet.

    From the second batch on the recorded distances differ -- and that is
    CORRECT, not a bug: all_open has already sent more pairs to their
    midpoints, so the same pair indices now sit at different distances.
    Pinning batch 1 is what actually isolates 'selection did not
    change'; asserting equality on every batch would be asserting that
    the mode has no effect at all."""
    n = 120
    adj = _ring(n)
    seen_t, _, x_t = _spy_sweep("threshold", n, adj)
    seen_o, _, x_o = _spy_sweep("all_open", n, adj)
    assert len(seen_t) > 1, "test needs a multi-batch sweep to be meaningful"
    assert torch.equal(seen_t[0], seen_o[0]), \
        "the two modes sampled different pairs in the very first batch"
    # and the modes really did do different things
    assert not torch.equal(x_t, x_o)


def test_acceptance_is_the_only_difference_within_one_batch():
    """On a state where every pair is already inside the confidence
    width, threshold and all_open must agree EXACTLY -- same pairs, same
    acceptances, same resulting state. Any difference would mean the
    mode leaked into something other than acceptance."""
    n = 120
    adj = _ring(n)
    x_t = torch.full((n,), 0.5)
    x_o = torch.full((n,), 0.5)
    a = gp.ab_sweep(x_t, adj, 1.0, 0.0,
                    gen=torch.Generator().manual_seed(9),
                    gate_mode="threshold")
    b = gp.ab_sweep(x_o, adj, 1.0, 0.0,
                    gen=torch.Generator().manual_seed(9),
                    gate_mode="all_open")
    assert a == b == n
    assert torch.equal(x_t, x_o)


def test_all_open_still_conserves_the_population_mean():
    """Every accepted pair goes to its midpoint, so the sweep is
    mean-conserving in BOTH modes. The checker's social-branch replay
    depends on this holding under all_open too."""
    n = 120
    adj = _ring(n)
    x = _state(n)
    m0 = float(x.mean())
    gp.ab_sweep(x, adj, 0.2, 0.0, gen=torch.Generator().manual_seed(11),
                gate_mode="all_open")
    assert abs(float(x.mean()) - m0) < 1e-6


# -- the stubborn variant carries the mode identically -------------------

def test_stubborn_sweep_takes_the_mode_and_defaults_to_threshold():
    n = 60
    adj = _ring(n)
    fixed = torch.zeros(n, dtype=torch.bool)
    fixed[:10] = True
    xa, xb = _state(n), _state(n)
    a, _ = gp.ab_sweep_stubborn(xa, adj, 0.1, 0.0,
                                torch.Generator().manual_seed(2), fixed)
    b, _ = gp.ab_sweep_stubborn(xb, adj, 0.1, 0.0,
                                torch.Generator().manual_seed(2), fixed,
                                gate_mode="threshold")
    assert torch.equal(xa, xb) and a == b
    xo = _state(n)
    o, _ = gp.ab_sweep_stubborn(xo, adj, 0.0001, 0.0,
                                torch.Generator().manual_seed(2), fixed,
                                gate_mode="all_open")
    assert o == n, f"all_open stubborn accepted {o} != {n}"
    # fixed agents are still never moved, whatever the gate says
    assert torch.equal(xo[fixed], _state(n)[fixed])
