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
