"""Invariants of the section3_frontier_icl wave manifest.

These are the properties that, if broken, would either cost money
silently, leak a key, waste a GPU, or make the frontier cells
incomparable with the local-model Figure-3(a) wave.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "experiments" / "condor" / "gen_frontier_icl.py"


@pytest.fixture(scope="module")
def gen():
    sys.path.insert(0, str(GEN.parent))
    spec = importlib.util.spec_from_file_location("gen_frontier_icl", str(GEN))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_request_counts_match_the_specification(gen):
    """21,690 per model-seed and 65,070 per model for three seeds."""
    assert gen.REQ_PER_MODEL_SEED == 21_690
    assert gen.REQ_PER_MODEL_3SEED == 65_070
    assert gen.N_AGENTS * 30 == 21_690


def test_surface_matches_the_local_icl_wave(gen):
    """Only the SERVER may differ from section3_model_icl, or the
    frontier cells cannot be placed beside the local ones."""
    assert gen.S3I_SWEEPS == 100
    assert gen.S3I_ROUNDS == 30
    assert gen.S3I_ICL_DAYS == 8
    assert float(gen.S3I_BETA) == 1.0      # W_PLAT
    assert float(gen.S3I_GAMMA) == 1.0     # INNATE_LAMBDA
    assert float(gen.S3I_ALPHA) == 0.5
    assert gen.S3_OP_TOKEN == "anch2"
    assert tuple(gen.S3I_SEEDS) == (0, 42, 43)


def test_tags_cannot_collide_with_local_model_runs(gen):
    t = gen.tag("google/gemini-3.7-flash", "Google AI Studio", 0)
    assert t.startswith("pofds3f_")
    for foreign in ("pofds3i_", "pofds3m_", "pofdcac_", "pofdws2_"):
        assert not t.startswith(foreign)
    # the PROVIDER is part of the identity: the same slug served by two
    # providers is two different measurements
    a = gen.tag("google/gemini-3.7-flash", "Google AI Studio", 0)
    b = gen.tag("google/gemini-3.7-flash", "Google Vertex", 0)
    assert a != b


def test_sub_requests_no_gpu_and_carries_no_key(gen):
    sub = gen.SUB.format(
        kind="test", key="k", or_max_tokens=32, concurrency=8, rps=4,
        reasoning_mode="disabled", eps_social="0.2", icl_days=8, wplat="1",
        lam="1", alpha="0.5", sweeps=100, op="anch2", n=723)
    assert "request_gpus" not in sub
    assert "OPENROUTER_API_KEY=" not in sub
    assert "sk-or-" not in sub
    # the key arrives by FILE reference only
    assert "OPENROUTER_API_KEY_FILE=" in sub
    # and the frozen contract is pinned in the environment itself
    for pin in ("TRAINING_STYLE=frozen", "SFT_EPOCHS=0", "USE_LORA=0",
                "KL_BETA=0", "ICL_K=0", "PARSE_MODE=strict",
                "OR_TEMPERATURE=0", "AI_GATE_MODE=all_open",
                "PEER_GATE_MODE=all_open"):
        assert pin in sub, pin


def test_moving_aliases_are_refused(gen):
    import subprocess
    for bad in ("openai/gpt-5-latest@OpenAI", "openrouter/auto@OpenAI"):
        r = subprocess.run(
            [sys.executable, str(GEN), "--models", bad, "--seeds", "0"],
            capture_output=True, text=True)
        assert r.returncode != 0, bad
        assert "alias" in (r.stdout + r.stderr).lower()


def test_cost_estimate_scales_linearly(gen):
    one = gen.estimate_cost(21_690, 0.38, 1.88)
    three = gen.estimate_cost(65_070, 0.38, 1.88)
    assert three["cost_total_usd"] == pytest.approx(
        one["cost_total_usd"] * 3, rel=0.01)
    assert one["requests"] == 21_690
