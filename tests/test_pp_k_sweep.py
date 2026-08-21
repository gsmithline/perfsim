"""Tests for the perfect-prediction k-sweep (2026-08-21). CPU only.

Under perfect prediction the pre-peer map is Friedkin-Johnsen with
beta_eff = 1 - (1-W)k, so k controls carryover of the previous population
state. Two things are easy to get wrong and both are pinned here:

  1. THE DIRECTION. At fixed W, LOWERING k RAISES beta_eff. More
     carryover is a LESS anchored map, not a more anchored one.
  2. THE W=1 CONTROL. There beta_eff = 1 for every k because (1-W)=0
     annihilates the human component, so every k must give a
     bit-identical trajectory at a matched peer seed.

Run with USE_TF=0.
"""
import importlib.util
import json
import os

import pytest
import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")
OUT = os.path.join(REPO, "notes", "pofd", "perfect_prediction_k_sweep")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PP = _load("sim_pp_k", os.path.join(PIPE, "sim_perfect_predictor.py"))
GEN = _load("gen_k", os.path.join(PIPE, "gen_pp_k_sweep.py"))


@pytest.fixture(scope="module")
def setup():
    from pathlib import Path
    return PP.extract_loader()(
        Path(REPO) / "experiments/data/movielens/ml-100k", "Action")


# ------------------------------------------------------------ beta_eff
@pytest.mark.parametrize("k,want", [
    (0.0, 1.0), (0.2, 0.9), (0.4, 0.8), (0.6, 0.7), (0.8, 0.6), (1.0, 0.5)])
def test_beta_eff_at_the_sweep_susceptibility(k, want):
    assert abs(PP.beta_eff(k, 0.5) - want) < 1e-12


def test_lower_k_means_higher_beta_eff():
    """The direction that is easy to assume backwards: k=1 (Jiduan's
    stateless form) is the LEAST anchored-to-state map at W=.5, and k=0
    gives beta_eff=1, an identity pre-peer map."""
    bs = [PP.beta_eff(k, 0.5) for k in GEN.KS]
    assert bs == sorted(bs, reverse=True)
    assert PP.beta_eff(0.0, 0.5) == 1.0 and PP.beta_eff(1.0, 0.5) == 0.5


def test_k_cancels_entirely_at_w_one():
    assert {PP.beta_eff(k, 1.0) for k in GEN.KS} == {1.0}


# ------------------------------------------------------------- the grid
def test_grid_is_the_requested_one():
    assert GEN.KS == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert GEN.SEEDS == [0, 42, 43]
    assert GEN.WS == [0.5, 1.0]
    assert GEN.ROUNDS == 300


def test_gates_are_modes_and_eps_social_is_positive():
    """Open must be a MODE. Both gates are strict inequalities, so eps=1
    still rejects a distance-1 pair; and eps_social=0 is the NO-PEER
    condition, so it cannot double as an open channel."""
    cfg = GEN.cell_cfg(0.4, 0.5, 42, 300)
    assert cfg["ai_gate_mode"] == "all_open"
    assert cfg["peer_gate_mode"] == "all_open"
    assert cfg["eps_social"] > 0
    name = PP.artifact_name(cfg)
    assert "eaopen" in name and "esopen" in name
    assert "_es1_" not in name and "_ea1_" not in name


def test_artifact_name_separates_every_cell():
    seen = {PP.artifact_name(GEN.cell_cfg(k, w, s, 300))
            for k in GEN.KS for w in GEN.WS for s in GEN.SEEDS}
    assert len(seen) == len(GEN.KS) * len(GEN.WS) * len(GEN.SEEDS) == 36


# ------------------------------------------------- structural properties
def test_mean_is_conserved_exactly_for_every_k(setup):
    """(1-b)m0 + b*m0 = m0 for ANY beta_eff, and midpoint peer moves
    conserve the mean, so the population mean never leaves the innate
    mean -- at any k. Short horizon keeps the test fast."""
    m0 = float(setup["innate"].mean())
    for k in (0.0, 0.5, 1.0):
        op, _, _ = PP.simulate(
            setup, innate_k=k, w_plat=0.5, eps_social=0.2, eps_ai=1.0,
            rounds=12, seed=0, ai_gate_mode="all_open",
            peer_gate_mode="all_open")
        assert float((op.mean(dim=1) - m0).abs().max()) < 1e-6, k


def test_w_one_trajectories_are_identical_across_k(setup):
    """THE structural control. At W=1 the (1-W) factor annihilates the
    human component, so z = x for every k and the trajectories must be
    BIT-identical at a matched peer seed."""
    ref = None
    for k in GEN.KS:
        op, _, _ = PP.simulate(
            setup, innate_k=k, w_plat=1.0, eps_social=0.2, eps_ai=1.0,
            rounds=15, seed=0, ai_gate_mode="all_open",
            peer_gate_mode="all_open")
        if ref is None:
            ref = op
        else:
            assert torch.equal(op, ref), f"k={k} differs at W=1"


def test_w_half_trajectories_do_differ_across_k(setup):
    """Negative control on the control: if k did NOT reach the operator,
    the W=1 invariance test above would pass vacuously."""
    a, _, _ = PP.simulate(setup, innate_k=0.0, w_plat=0.5, eps_social=0.2,
                          eps_ai=1.0, rounds=15, seed=0,
                          ai_gate_mode="all_open", peer_gate_mode="all_open")
    b, _, _ = PP.simulate(setup, innate_k=1.0, w_plat=0.5, eps_social=0.2,
                          eps_ai=1.0, rounds=15, seed=0,
                          ai_gate_mode="all_open", peer_gate_mode="all_open")
    assert not torch.equal(a, b)


def test_k_zero_at_w_half_matches_any_k_at_w_one(setup):
    """Both have beta_eff = 1, and both reduce z to exactly x, so they are
    the same dynamical system -- bit-identical, not merely similar."""
    a, _, _ = PP.simulate(setup, innate_k=0.0, w_plat=0.5, eps_social=0.2,
                          eps_ai=1.0, rounds=15, seed=0,
                          ai_gate_mode="all_open", peer_gate_mode="all_open")
    b, _, _ = PP.simulate(setup, innate_k=0.6, w_plat=1.0, eps_social=0.2,
                          eps_ai=1.0, rounds=15, seed=0,
                          ai_gate_mode="all_open", peer_gate_mode="all_open")
    assert torch.equal(a, b)


# --------------------------------------------------------- the artifacts
@pytest.mark.skipif(not os.path.exists(os.path.join(OUT, "manifest.json")),
                    reason="k-sweep not generated in this checkout")
def test_manifest_covers_the_full_grid():
    mf = json.load(open(os.path.join(OUT, "manifest.json")))
    assert mf["n_cells"] == 36
    assert mf["n_reused"] + mf["n_generated"] == 36
    assert mf["rounds"] == 300
    assert mf["ai_gate_mode"] == "all_open"
    assert mf["peer_gate_mode"] == "all_open"
    for c in mf["cells"]:
        assert abs(c["beta_eff"] - PP.beta_eff(c["innate_k"],
                                               c["w_plat"])) < 1e-12


@pytest.mark.skipif(not os.path.exists(os.path.join(OUT, "manifest.json")),
                    reason="k-sweep not generated in this checkout")
def test_analyzer_plots_innate_at_t_zero():
    """The first plotted point must be the innate population, not the
    result of round 1 -- otherwise the x axis is off by one and the
    initial dispersion looks like a post-update state."""
    src = open(os.path.join(PIPE, "analyze_pp_k_sweep.py")).read()
    assert "np.concatenate([innate[None, :], op], axis=0)" in src
    assert '"is_innate"' in src
    import csv as _csv
    with open(os.path.join(OUT, "pp_k_sweep_rounds.csv")) as fh:
        rows = list(_csv.DictReader(fh))
    first = [r for r in rows if r["t"] == "0"]
    assert first and all(r["is_innate"] == "1" for r in first)
    # every k must share that same t=0 point: it is the innate population
    sds = {round(float(r["sd"]), 10) for r in first}
    assert len(sds) == 1, sds
