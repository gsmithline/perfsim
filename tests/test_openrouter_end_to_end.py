"""END-TO-END wiring test for MODEL_BACKEND=openrouter. NO PAID CALLS.

Everything else in this suite tests the client or the model in isolation.
This runs the ACTUAL RUNNER -- env parsing, the backend switch, the
provider-neutral serving prompt, the serving loop, the provenance
artifact and the trajectory -- against a fake transport, because the
integration is exactly where a backend switch goes wrong: the pieces can
each be right while the runner still hands the API a chat-template
string, or serves round t-1's values, or writes no provenance at all.

Two rounds, 723 agents, all mocked: ~1.4k fake requests, no network.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RUNNER = REPO / "experiments" / "scripts" / "cluster_pipelines" / \
    "run_pokec_gated_lm.py"
ML = REPO / "experiments" / "data" / "movielens" / "ml-100k"
pytestmark = pytest.mark.skipif(not ML.exists(), reason="ml-100k not present")


class _Resp:
    """A deterministic fake completion whose value depends on the prompt,
    so a permuted or stale serving vector cannot pass unnoticed."""

    status_code = 200
    headers: dict = {}

    def __init__(self, body):
        content = body["messages"][0]["content"]
        # a stable per-prompt value in [0,1]
        self._v = (abs(hash(content)) % 1000) / 1000.0
        self._model = body["model"]
        self._provider = body["provider"]["order"][0]

    @property
    def text(self):
        return json.dumps(self.json())

    def json(self):
        return {
            "id": f"gen-{self._v}", "model": self._model,
            "provider": self._provider, "system_fingerprint": "fp_mock",
            "choices": [{"message": {"content": f"{self._v:.3f}"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 250, "completion_tokens": 4,
                      "total_tokens": 254, "cost": 0.00001},
        }


class _FakeAsyncClient:
    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None, timeout=None):
        assert headers and "Authorization" in headers
        # the API must NEVER receive a chat-template render
        content = json["messages"][0]["content"]
        for marker in ("<|im_start|>", "[INST]", "<start_of_turn>"):
            assert marker not in content, f"template leaked: {content[:80]}"
        return _Resp(json)


class _FakeHttpx:
    AsyncClient = _FakeAsyncClient


@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_TAG", "pofds3f_mock_e2e")
    monkeypatch.setenv("OUT_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("DATASET", "movielens")
    monkeypatch.setenv("ML_TARGET", "Action")
    monkeypatch.setenv("MODEL_BACKEND", "openrouter")
    monkeypatch.setenv("OR_MODEL", "google/gemini-3.7-flash")
    monkeypatch.setenv("OR_PROVIDER", "Google AI Studio")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-MOCK-not-real")
    monkeypatch.setenv("OR_MAX_REQUESTS", "20000")
    monkeypatch.setenv("OR_MAX_COST", "100")
    monkeypatch.setenv("OR_RPS", "0")
    monkeypatch.setenv("OR_CONCURRENCY", "16")
    monkeypatch.setenv("OR_CACHE", str(tmp_path / "cache.sqlite"))
    monkeypatch.setenv("OR_REQUIRE_PARAMETERS", "1")
    monkeypatch.setenv("OR_ZDR", "1")
    # the frozen ICL surface
    monkeypatch.setenv("TRAINING_STYLE", "frozen")
    monkeypatch.setenv("SFT_EPOCHS", "0")
    monkeypatch.setenv("USE_LORA", "0")
    monkeypatch.setenv("KL_BETA", "0")
    monkeypatch.setenv("LOG_PERPLEXITY", "0")
    monkeypatch.setenv("LOG_ANSWER_DIST", "0")
    monkeypatch.setenv("ANS_SAMPLE_K", "0")
    monkeypatch.setenv("FRESH_EACH_ROUND", "0")
    monkeypatch.setenv("PARSE_MODE", "strict")
    monkeypatch.setenv("SAVE_RAW_GEN", "1")
    monkeypatch.setenv("ICL_K", "0")
    monkeypatch.setenv("ICL_DAYS", "8")
    monkeypatch.setenv("POP_MODEL", "ab")
    monkeypatch.setenv("AI_GATE_MODE", "all_open")
    monkeypatch.setenv("PEER_GATE_MODE", "all_open")
    monkeypatch.setenv("EPS_AI", "1.0")
    monkeypatch.setenv("EPS_SOCIAL", "0.2")
    monkeypatch.setenv("W_PLAT", "1")
    monkeypatch.setenv("INNATE_LAMBDA", "1")
    monkeypatch.setenv("AB_SWEEPS", "2")      # tiny: wiring, not physics
    monkeypatch.setenv("N_ROUNDS", "2")
    monkeypatch.setenv("WITH_TWIN", "1")
    monkeypatch.setenv("SEED", "0")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DISABLED", "true")
    return tmp_path / "run"


def _load_runner():
    import importlib.util
    sys.path.insert(0, str(RUNNER.parent))
    spec = importlib.util.spec_from_file_location("runner_e2e", str(RUNNER))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_openrouter_backend_runs_end_to_end(runner_env, monkeypatch):
    import perfsim.models.openrouter_client as orc
    monkeypatch.setattr(orc, "httpx", _FakeHttpx)
    mod = _load_runner()
    rc = mod.main()
    assert rc in (0, None), f"runner returned {rc}"

    out = runner_env
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_backend"] == "openrouter"
    assert cfg["openrouter"]["model_slug"] == "google/gemini-3.7-flash"
    p = cfg["openrouter"]["provider"]
    assert p["allow_fallbacks"] is False and p["zdr"] is True
    # require_parameters now defaults OFF: measured 2026-08-31, with
    # zdr=true it eliminates every ZDR endpoint for all three frontier
    # models. Its intent survives via preflight (only advertised params are
    # sent) plus the provenance gate. The knob still works, and the env
    # sets it explicitly here so both states stay covered.
    assert p["data_collection"] == "deny"
    assert p["require_parameters"] is True, "OR_REQUIRE_PARAMETERS=1 honoured"
    # the determinism caveat must survive into the artifact
    assert "does not pin" in json.dumps(cfg.get("gen_policy_effective", {}))

    import torch
    traj = torch.load(out / "trajectory.pt", map_location="cpu",
                      weights_only=False)
    assert traj["op_raw"].shape == (2, 723)
    assert traj["pred_raw"].shape == (2, 723)
    assert torch.isfinite(traj["op_raw"]).all()

    # PROVENANCE: one record per agent per round, aligned to pred_raw
    rows = [json.loads(l) for l in
            gzip.open(out / "or_provenance.json.gz", "rt") if l.strip()]
    assert [r["round"] for r in rows] == [0, 1]
    for r in rows:
        # EVERY paid request: 723 agent serves plus the 64-prompt telemetry
        # probe. Recording only the serving call left 8% of spend invisible.
        assert r["n_agents"] == 723
        assert len(r["records"]) == 787, "agent serves + telemetry probe"
        rec = r["records"][0]
        assert rec["resolved_model"] == "google/gemini-3.7-flash"
        assert rec["resolved_provider"] == "Google AI Studio"
        assert rec["finish_reason"] == "stop"
        assert rec["prompt_tokens"] == 250
        assert rec["generation_id"] and rec["request_hash"]
    # the served vector IS what the API returned, in agent order
    got = [float(x["text"]) for x in rows[0]["records"][:723]]
    assert torch.allclose(torch.tensor(got),
                          traj["pred_raw"][0].float(), atol=1e-6)


def test_training_knobs_are_refused_by_the_runner(runner_env, monkeypatch):
    """The refusal must happen in the RUNNER, before any request."""
    monkeypatch.setenv("USE_LORA", "1")
    mod = _load_runner()
    with pytest.raises(ValueError, match="FROZEN-ONLY"):
        mod.main()
