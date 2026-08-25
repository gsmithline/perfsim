"""Focused tests for the matched Section 3 cross-model equilibrium wave."""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "experiments" / "condor" / "gen_pofd_sweep.py"
ANALYZE = (ROOT / "experiments" / "scripts" / "cluster_pipelines" /
           "analyze_section3_model_equilibria.py")
RUNNER = (ROOT / "experiments" / "scripts" / "cluster_pipelines" /
          "run_pokec_gated_lm.py")
SUBMIT = ROOT / "experiments" / "condor" / "submit_pofd_sweep.sh"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_grid_is_exact_six_models_by_three_seeds_with_no_lambda_zero():
    g = _load(GEN, "_gen_s3m_test")
    rows = g.s3m_rows()
    assert len(rows) == 18
    got = set()
    for row in rows:
        c = [x.strip() for x in row.split(",")]
        assert len(c) == 29
        model = next(m for m in g.S3M_MODELS if f"_{m}_" in c[0])
        seed = int(c[3])
        got.add((model, seed))
        assert c[0] == g.s3m_tag(model, seed)
        assert c[1] == "sft_kl"
        assert float(c[2]) == 2.0
        assert float(c[11]) == 1.0       # beta = W_PLAT
        assert float(c[14]) == 1.0       # gamma = INNATE_LAMBDA (inert)
        assert c[15] == "forward"
        assert int(c[16]) == 100
        assert int(c[23]) == 30
        assert c[24] == g.FAM_MODELS[model]["base_model"]
        assert c[25] == g.FAM_MODELS[model]["chatthink"]
    assert got == {(m, s) for m in g.S3M_MODELS for s in g.S3M_SEEDS}
    assert not g.S3M_REUSED
    assert all("_fwdlam2_" in r.split(",")[0] for r in rows)
    assert not any(", sft, 0," in r for r in rows)


def test_sub_pins_the_scientific_surface_and_routes_submission():
    g = _load(GEN, "_gen_s3m_sub_test")
    sub = g.s3m_sub()
    env = next(line for line in sub.splitlines()
               if line.startswith("environment"))
    for token in (
        "AI_GATE_MODE=all_open",
        "PEER_GATE_MODE=all_open",
        "AI_GATE_REFERENCE=anchor",
        "DEFFUANT_ALPHA=0.5",
        "AB_SWEEPS=$(sweeps)",
        "INNATE_LAMBDA=$(lam)",
        "KL_DIRECTION=$(kldir)",
        "SAVE_RAW_GEN=1",
    ):
        assert token in env
    assert "18 jobs" in sub
    assert "check_section3_model_equilibria.py" in sub
    submit = SUBMIT.read_text()
    assert ("section3_model_equilibria|section3_model_equilibria_smoke) "
            "TARGETS=\"$WHAT\" ;;" in submit)


def test_smoke_is_non_qwen_and_outside_the_production_grid():
    g = _load(GEN, "_gen_s3m_smoke_test")
    row = g.s3m_smoke_rows()[0]
    c = [x.strip() for x in row.split(",")]
    assert g.S3M_SMOKE_MODEL == "olmo3_7b"
    assert int(c[3]) == g.S3M_SMOKE_SEED == 991
    assert int(c[23]) == g.S3M_SMOKE_ROUNDS == 3
    assert c[0] not in {r.split(",")[0] for r in g.s3m_rows()}


def test_runner_records_and_applies_deffuant_alpha():
    source = RUNNER.read_text()
    assert 'deffuant_alpha = _env_float("DEFFUANT_ALPHA", 0.5)' in source
    assert '"deffuant_alpha": deffuant_alpha' in source
    assert source.count("alpha=deffuant_alpha") == 2  # deployed + twin


def test_cell_stats_use_postpeer_population_and_tail_window():
    a = _load(ANALYZE, "_analyze_s3m_test")
    # Twenty post-peer rounds, each a consensus at its round index / 100.
    op = np.repeat((np.arange(20) / 100.0)[:, None], 723, axis=1)
    pred = np.repeat(np.asarray([[.2, .4, .4]]), 20, axis=0)
    stats = a._cell_stats(op, pred, window=10)
    assert np.isclose(stats["equilibrium_mean"], np.mean(np.arange(10, 20)) / 100)
    assert np.isclose(stats["final_mean"], .19)
    assert stats["final_sd"] < 1e-12
    assert np.isclose(stats["drift"], .05)
    assert stats["served_distinct"] == 2
    assert np.isclose(stats["served_max_mode_share"], 2 / 3)


CHECK = (ROOT / "experiments" / "scripts" / "cluster_pipelines" /
         "check_section3_model_equilibria.py")


def _load_checker():
    sys.path.insert(0, str(CHECK.parent))
    try:
        return _load(CHECK, "_check_s3m_test")
    finally:
        sys.path.pop(0)


def _write_raw_log(run_dir, rows):
    import gzip
    import json
    with gzip.open(run_dir / "raw_gen_log.json.gz", "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _good_rows(rounds, n=723, pf=0.0):
    return [{"round": t, "parse_fail_frac": pf, "parsed": [0.5] * n,
             "raw": ["0.50"] * n} for t in range(rounds)]


def test_checker_requires_raw_generation_log_no_nan_fallback(tmp_path):
    c = _load_checker()
    assert not hasattr(c, "check_parse"), \
        "the NaN-fallback helper must not be reachable from this gate"
    errs = []
    c.check_raw_generations(tmp_path, 3, errs)
    assert len(errs) == 1 and "ABSENT" in errs[0]


def test_checker_passes_only_a_complete_zero_failure_log(tmp_path):
    c = _load_checker()
    _write_raw_log(tmp_path, _good_rows(3))
    errs = []
    c.check_raw_generations(tmp_path, 3, errs)
    assert errs == []


def test_checker_fails_parse_failures_missing_rounds_and_short_rows(tmp_path):
    c = _load_checker()
    # a single non-zero parse_fail_frac in one round
    rows = _good_rows(3)
    rows[1]["parse_fail_frac"] = 1 / 723
    _write_raw_log(tmp_path, rows)
    errs = []
    c.check_raw_generations(tmp_path, 3, errs)
    assert any("parse_fail_frac must be exactly 0" in e for e in errs)
    # an absent parse_fail_frac field counts as a failure
    rows = _good_rows(3)
    del rows[2]["parse_fail_frac"]
    _write_raw_log(tmp_path, rows)
    errs = []
    c.check_raw_generations(tmp_path, 3, errs)
    assert any("parse_fail_frac must be exactly 0" in e for e in errs)
    # a missing round is an unverified round
    _write_raw_log(tmp_path, _good_rows(2))
    errs = []
    c.check_raw_generations(tmp_path, 3, errs)
    assert any("exactly rounds 0..2" in e for e in errs)
    # fewer than 723 parsed values in a round
    rows = _good_rows(3)
    rows[0]["parsed"] = rows[0]["parsed"][:700]
    _write_raw_log(tmp_path, rows)
    errs = []
    c.check_raw_generations(tmp_path, 3, errs)
    assert any("parsed fewer than 723" in e for e in errs)


def test_checker_requires_a_live_kl_gradient_in_every_round(tmp_path):
    import json
    c = _load_checker()
    tel = tmp_path / "telemetry.json"
    rows = [{"round": t, "l_init": 1.0, "n_train": 723,
             "grad_norm0": 1.0, "grad_kl_norm0": 0.08} for t in range(3)]
    tel.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    errs = []
    c.check_kl_witness_every_round(tmp_path, 3, errs)
    assert errs == []
    # one round with a vanished anchor gradient fails, even if others bind
    rows[1]["grad_kl_norm0"] = 0.0
    tel.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    errs = []
    c.check_kl_witness_every_round(tmp_path, 3, errs)
    assert len(errs) == 1 and "KL-WITNESS" in errs[0] and "(1, 0.0)" in errs[0]
    # a round with no record at all fails
    tel.write_text("\n".join(json.dumps(r) for r in rows[:2]) + "\n")
    errs = []
    c.check_kl_witness_every_round(tmp_path, 3, errs)
    assert any("(2, None)" in e for e in errs)


def test_tci3_locks_df2_critical_value_and_refuses_other_n():
    import pytest
    a = _load(ANALYZE, "_analyze_s3m_tci_test")
    assert a.T_CRIT_DF2_95 == 4.302652729911275
    mean, sd, half = a.tci3([0.60, 0.62, 0.64])
    assert np.isclose(mean, 0.62)
    assert np.isclose(sd, 0.02)                       # sample SD, ddof=1
    assert np.isclose(half, 4.302652729911275 * 0.02 / np.sqrt(3))
    with pytest.raises(ValueError):
        a.tci3([0.6, 0.62])                           # n != 3 never silently
    with pytest.raises(ValueError):
        a.tci3([0.6, 0.62, 0.64, 0.66])


def test_settled_rejects_a_small_two_cycle_that_half_split_drift_passes():
    a = _load(ANALYZE, "_analyze_s3m_settle_test")
    tol = a.DRIFT_TOL
    # period-2 series of amplitude 0.02 over the 10-round window: the
    # half-split drift is (b-a)/5 = 0.004 <= tol, but the late range is
    # 0.02 > 2*tol, so it must NOT count as settled, and it is cyclic.
    tail = np.tile([0.60, 0.62], 15)                  # 30 rounds
    op = np.repeat(tail[:, None], 723, axis=1)
    pred = np.repeat(np.asarray([[.6, .62, .6]]), 30, axis=0)
    st = a._cell_stats(op, pred, window=10)
    assert abs(st["drift"]) <= tol
    assert not a.settled(st, tol)
    assert a.cyclic(st, tol)
    # a genuinely flat tail is settled and not cyclic
    op_flat = np.full((30, 723), 0.61)
    st = a._cell_stats(op_flat, pred, window=10)
    assert a.settled(st, tol) and not a.cyclic(st, tol)
    # a monotone drift of 0.002/round: drift 0.01 > tol -> unsettled
    op_drift = np.repeat((0.5 + 0.002 * np.arange(30))[:, None], 723, axis=1)
    st = a._cell_stats(op_drift, pred, window=10)
    assert not a.settled(st, tol) and not a.cyclic(st, tol)


def test_analyzer_refuses_smoke_or_stale_gate_verdicts():
    a = _load(ANALYZE, "_analyze_s3m_gate_test")
    g = _load(GEN, "_gen_s3m_gate_test")
    full = [{"tag": g.s3m_tag(m, s), "status": "PASS", "git_sha": "abc"}
            for m in g.S3M_MODELS for s in g.S3M_SEEDS]
    assert a.gate_binds_wave({"ok": True, "n_cells": 18, "cells": full}, g) is None
    smoke = {"ok": True, "n_cells": 1,
             "cells": [{"tag": g.s3m_tag(g.S3M_SMOKE_MODEL, g.S3M_SMOKE_SEED,
                                         rounds=3, smoke=True),
                        "status": "PASS"}]}
    assert "smoke or stale" in a.gate_binds_wave(smoke, g)
    assert a.gate_binds_wave({"ok": False, "n_cells": 18, "cells": full}, g)
    one_bad = [dict(c) for c in full]
    one_bad[0]["status"] = "FAIL"
    assert "not PASS" in a.gate_binds_wave(
        {"ok": True, "n_cells": 18, "cells": one_bad}, g)


def test_checker_requires_runtime_open_gate_evidence_every_round():
    c = _load_checker()
    rows = [{"contact": 1.0, "peer_gate_mode": "all_open",
             "peer_pairs": 72300, "accepted": 72300} for _ in range(3)]
    d = {"trajectory": rows, "config": {"ab_sweeps": 100}}
    errs = []
    c.check_runtime_open_gates(d, 3, errs)
    assert errs == []
    # one rejected pair in one round breaks the all-open invariant
    rows[2] = dict(rows[2], accepted=72299)
    errs = []
    c.check_runtime_open_gates({"trajectory": rows, "config": {"ab_sweeps": 100}},
                               3, errs)
    assert len(errs) == 1 and "GATE-RUNTIME" in errs[0]
    # an AI gate that closed for anyone (contact < 1) fails
    rows[2] = dict(rows[2], accepted=72300, contact=0.998)
    errs = []
    c.check_runtime_open_gates({"trajectory": rows, "config": {"ab_sweeps": 100}},
                               3, errs)
    assert len(errs) == 1
    # legacy rows without the all_open telemetry cannot pass
    errs = []
    c.check_runtime_open_gates({"trajectory": [{"contact": 1.0}] * 3,
                                "config": {"ab_sweeps": 100}}, 3, errs)
    assert len(errs) == 1
