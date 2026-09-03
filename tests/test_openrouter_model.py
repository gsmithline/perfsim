"""OpenRouterModel contract + the Hugging Face no-regression proof.

The byte-identity test is the important one. Task-2's refactor split the
MovieLens prompt into a semantic message plus a chat-template render; if
that split changed ANY character of the HF prompt, every archived run's
comparability would be gone and no test of the new backend would matter.
So this file loads the PRE-REFACTOR builder out of git and compares it
character for character against the current one, on real profiles, with
and without a personal-history context block.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RUNNER = "experiments/scripts/cluster_pipelines/run_pokec_gated_lm.py"
ML = REPO / "experiments" / "data" / "movielens" / "ml-100k"


class MockTokenizer:
    """Stands in for a checkpoint tokenizer: renders messages the way a
    chat template does, so a change in WHAT is rendered shows up."""
    chat_template = "mock"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                       for m in messages)
        return body + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def _load(path, name):
    # the runner imports siblings (_collapse_metrics, _gated_pop) by bare
    # name, so the REAL package dir has to be importable even when the
    # file under exec lives in a tmpdir
    sib = str((REPO / RUNNER).parent)
    if sib not in sys.path:
        sys.path.insert(0, sib)
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def current():
    return _load(REPO / RUNNER, "runner_current")


@pytest.fixture(scope="module")
def previous():
    """The runner as it was BEFORE the provider-neutral split.

    It has to be materialised INSIDE the real package dir: the runner
    resolves its siblings through Path(__file__).parent, so a copy in a
    tmpdir cannot import _collapse_metrics. Removed in the finally.
    """
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{RUNNER}"],
        capture_output=True, text=True, check=True).stdout
    tmp = (REPO / RUNNER).parent / "_runner_prerefactor_tmp.py"
    tmp.write_text(src)
    try:
        yield _load(tmp, "runner_old")
    finally:
        tmp.unlink(missing_ok=True)


@pytest.mark.skipif(not ML.exists(), reason="ml-100k not present")
def test_hf_prompt_is_byte_identical(current, previous):
    """THE no-regression proof: same characters, old code vs new."""
    new = current.load_movielens_setup(ML, "Action")
    old = previous.load_movielens_setup(ML, "Action")
    tok = MockTokenizer()
    ctx = ("This user's own opinion of Action movies over the most recent "
           "days (oldest to newest): 0.50, 0.62, 0.71.")
    n_checked = 0
    for i in (0, 1, 7, 100, 400, 722):
        prof_new = new["profiles"].iloc[i]
        prof_old = old["profiles"].iloc[i]
        for cb in (None, "", ctx):
            a = new["build_prompt"](prof_new, tok, context_block=cb)
            b = old["build_prompt"](prof_old, tok, context_block=cb)
            assert a == b, f"agent {i} ctx={cb!r}:\n NEW {a!r}\n OLD {b!r}"
            n_checked += 1
    assert n_checked == 18


@pytest.mark.skipif(not ML.exists(), reason="ml-100k not present")
def test_messages_carry_the_same_semantic_text_without_the_template(current):
    setup = current.load_movielens_setup(ML, "Action")
    tok = MockTokenizer()
    prof = setup["profiles"].iloc[0]
    msgs = setup["build_messages"](prof)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    body = msgs[0]["content"]
    # the semantic text is exactly what the HF path wraps
    assert setup["build_prompt"](prof, tok) == (
        f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n")
    # ... and carries NO template control tokens of its own
    for marker in ("<|im_start|>", "<|im_end|>", "[INST]", "<start_of_turn>"):
        assert marker not in body


@pytest.mark.skipif(not ML.exists(), reason="ml-100k not present")
def test_personal_history_semantics_are_preserved(current):
    """ICL_K=0 / ICL_DAYS=8: only this agent's own latest eight values, in
    order, and no other agent's number anywhere in the prompt."""
    setup = current.load_movielens_setup(ML, "Action")
    hist = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88]
    ctx = ("This user's own opinion of Action movies over the most recent "
           "days (oldest to newest): " + ", ".join(f"{v:.2f}" for v in hist) + ".")
    body = setup["build_messages"](setup["profiles"].iloc[3],
                                   context_block=ctx)[0]["content"]
    assert ctx in body
    assert body.index("0.11") < body.index("0.88"), "oldest to newest"
    assert "opinions of some other users" not in body


# ---- the frozen contract ------------------------------------------------
def _mk(**kw):
    from perfsim.models.openrouter_lm import OpenRouterModel
    from perfsim.models.openrouter_client import (
        Budget, DecodingPolicy, ProviderPin)
    import pandas as pd
    defaults = dict(
        model_slug="google/gemini-3.7-flash",
        profiles=pd.DataFrame({"age": [30, 40]}),
        message_builder=lambda p, context_block=None: [
            {"role": "user", "content": f"age {int(p['age'])}"}],
        provider=ProviderPin(order=("Google AI Studio",)),
        policy=DecodingPolicy(), budget=Budget(dry_run=True),
        api_key="sk-or-TEST", transport=object())
    defaults.update(kw)
    return OpenRouterModel(**defaults)


@pytest.mark.parametrize("call,name", [
    (lambda m: m.perplexity(["x"]), "perplexity"),
    (lambda m: m.get_params(), "get_params"),
    (lambda m: m.set_params(None), "set_params"),
    (lambda m: m.clone(), "clone"),
    (lambda m: m.num_params(), "num_params"),
    (lambda m: m.answer_distribution_stats(), "answer_distribution_stats"),
    (lambda m: m.answer_sample_stats(), "answer_sample_stats"),
])
def test_training_capabilities_raise(call, name):
    from perfsim.models.openrouter_lm import FrozenBackendError
    with pytest.raises(FrozenBackendError):
        call(_mk())


def test_lenient_parse_mode_is_refused():
    from perfsim.models.openrouter_lm import FrozenBackendError
    with pytest.raises(FrozenBackendError, match="strict"):
        _mk(parse_mode="legacy")


def test_chat_template_render_is_refused_as_a_user_message():
    from perfsim.models.openrouter_client import OpenRouterError
    m = _mk()
    with pytest.raises(OpenRouterError, match="chat-template"):
        m._generate(["<|im_start|>user\nEstimate<|im_end|>\n"])
    with pytest.raises(OpenRouterError, match="chat-template"):
        m._generate(["[INST] Estimate [/INST]"])


def test_contract_surface_exists():
    m = _mk()
    for attr in ("forward", "_generate", "profile_at", "build_prompt",
                 "parse", "parse_ok", "_last_raw", "_last_parse_fail",
                 "effective_generation_policy"):
        assert hasattr(m, attr), attr
    pol = m.effective_generation_policy()
    assert pol["temperature"]["value"] == 0.0
    assert pol["provider"]["value"]["allow_fallbacks"] is False
    assert "determinism_caveat" in pol
    assert "does not pin" in pol["determinism_caveat"]


def test_seed_is_marked_unsupported_when_absent():
    assert _mk().effective_generation_policy()["seed"]["source"] == \
        "unsupported_by_endpoint"
    from perfsim.models.openrouter_client import DecodingPolicy
    m = _mk(policy=DecodingPolicy(seed=7))
    assert m.effective_generation_policy()["seed"]["source"] == "pinned"


def test_unparseable_response_never_serves_the_half_default():
    from perfsim.models.openrouter_lm import ParseFailure
    m = _mk()
    assert m.parse("0.42") == pytest.approx(0.42)
    assert m.parse_ok("0.42") is True
    assert m.parse_ok("I think it is moderate") is False
    with pytest.raises(ParseFailure):
        m.parse("I think it is moderate")


def test_drain_provenance_covers_every_call_site():
    """provenance_records() reports only the last _generate. The runner
    serves agents AND a 64-prompt telemetry probe each round, so capturing
    the last call alone left ~8% of paid requests unrecorded while the gate
    claimed complete cost accounting. drain_provenance cannot miss one."""
    from perfsim.models.openrouter_client import Provenance
    m = _mk()
    fake = [Provenance(requested_model="m", resolved_model="m",
                       requested_provider="p", resolved_provider="p",
                       generation_id=f"g{i}", system_fingerprint=None,
                       finish_reason="stop", prompt_tokens=1,
                       completion_tokens=1, total_tokens=2, cost_usd=0.001,
                       latency_s=0.1, retries=0, cache_status="miss",
                       reasoning_fallback=False, request_hash="h",
                       raw_response=None, text="0.4") for i in range(3)]
    m.client.provenances = list(fake)
    m._last_provenance = fake[-1:]           # what the old path would report
    assert len(m.provenance_records()) == 1
    drained = m.drain_provenance()
    assert len(drained) == 3, "every request in the round must be captured"
    assert m.drain_provenance() == [], "a drain must not double-count"
