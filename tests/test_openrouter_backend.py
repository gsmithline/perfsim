"""Mocked tests for the OpenRouter serving backend. NO PAID CALLS.

Every test here runs against an injected transport. If any of these ever
reaches the network it is a bug in the test, not in the code under test.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from perfsim.models.openrouter_client import (            # noqa: E402
    Budget, BudgetError, DecodingPolicy, MissingKeyError, OpenRouterClient,
    OpenRouterError, ProvenanceError, ProviderPin, ResponseCache,
    RetryExhaustedError, _redact, load_api_key, request_hash,
)

PIN = ProviderPin(order=("Google AI Studio",))
POLICY = DecodingPolicy(temperature=0.0, max_tokens=16)
MSGS = [[{"role": "user", "content": f"agent {i}"}] for i in range(6)]


def _resp(text, *, status=200, model="google/gemini-3.7-flash",
          provider="Google AI Studio", finish="stop", cost=0.0001,
          headers=None):
    class R:
        status_code = status
        def __init__(self):
            self.headers = headers or {}
        @property
        def text(self):
            return json.dumps(self.json())
        def json(self):
            return {
                "id": f"gen-{text}", "model": model, "provider": provider,
                "system_fingerprint": "fp_test",
                "choices": [{"message": {"content": text},
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                          "total_tokens": 13, "cost": cost},
            }
    return R()


class FakeTransport:
    """Async stand-in for httpx.AsyncClient. Scripted per call."""

    def __init__(self, script, *, delay=0.0):
        self.script = list(script)
        self.calls = 0
        self.delay = delay
        self.seen_headers = []

    async def post(self, url, headers=None, json=None, timeout=None):
        self.seen_headers.append(headers or {})
        i = min(self.calls, len(self.script) - 1)
        self.calls += 1
        item = self.script[i]
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(json)
        return item


def _client(transport, budget=None, cache=None, **kw):
    return OpenRouterClient(
        model="google/gemini-3.7-flash", provider=PIN, policy=POLICY,
        budget=budget or Budget(max_requests=50, max_realized_cost_usd=10.0),
        cache=cache, api_key="sk-or-TESTKEY-not-real", transport=transport,
        **kw)


def test_successful_completion():
    c = _client(FakeTransport([_resp("0.42")]))
    out = c.complete_many_sync(MSGS[:1])
    assert out[0].text == "0.42"
    assert out[0].resolved_model == "google/gemini-3.7-flash"
    assert out[0].resolved_provider == "Google AI Studio"
    assert out[0].finish_reason == "stop"
    assert out[0].prompt_tokens == 10 and out[0].completion_tokens == 3
    assert out[0].cost_usd == 0.0001
    assert out[0].system_fingerprint == "fp_test"
    assert out[0].cache_status == "miss"
    assert len(out[0].request_hash) == 64


def test_results_are_in_input_order_under_concurrency():
    """The network answers out of order on purpose; agent i's value must
    still be agent i's. A permutation here would be invisible in any
    aggregate statistic and fatal to the science."""
    def handler(body):
        content = body["messages"][0]["content"]
        return _resp(f"0.{content.split()[-1]}")

    t = FakeTransport([handler], delay=0.01)
    c = _client(t, budget=Budget(max_requests=50, max_realized_cost_usd=10.0,
                                 max_concurrency=6, requests_per_second=0))
    out = c.complete_many_sync(MSGS)
    assert [o.text for o in out] == [f"0.{i}" for i in range(6)]


def test_cache_hit_avoids_second_payment(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    t1 = FakeTransport([_resp("0.42")])
    _client(t1, cache=cache).complete_many_sync(MSGS[:1])
    assert t1.calls == 1

    t2 = FakeTransport([_resp("SHOULD-NOT-BE-CALLED")])
    b2 = Budget(max_requests=50, max_realized_cost_usd=10.0)
    out = _client(t2, budget=b2, cache=cache).complete_many_sync(MSGS[:1])
    assert t2.calls == 0
    assert out[0].text == "0.42"
    assert out[0].cache_status == "hit"
    assert b2.n_requests == 0 and b2.realized_cost_usd == 0.0


def test_cache_key_ignores_authorization_but_tracks_params(tmp_path):
    h1 = request_hash("m", PIN, MSGS[0], POLICY)
    h2 = request_hash("m", PIN, MSGS[0], DecodingPolicy(temperature=0.7))
    assert h1 != h2, "decoding params must be part of the key"
    assert h1 == request_hash("m", PIN, MSGS[0], POLICY), "must be stable"
    cache = ResponseCache(tmp_path / "c.sqlite")
    _client(FakeTransport([_resp("0.42")]), cache=cache).complete_many_sync(MSGS[:1])
    blob = (tmp_path / "c.sqlite").read_bytes()
    assert b"TESTKEY" not in blob and b"Authorization" not in blob


def test_429_is_retried_and_honours_retry_after():
    t = FakeTransport([_resp("", status=429, headers={"Retry-After": "0"}),
                       _resp("0.42")])
    out = _client(t).complete_many_sync(MSGS[:1])
    assert out[0].text == "0.42" and out[0].retries == 1


def test_retryable_5xx_is_retried():
    t = FakeTransport([_resp("", status=503), _resp("", status=502),
                       _resp("0.7")])
    out = _client(t).complete_many_sync(MSGS[:1])
    assert out[0].text == "0.7" and out[0].retries == 2


def test_non_retryable_4xx_fails_immediately():
    t = FakeTransport([_resp("", status=400)])
    with pytest.raises(OpenRouterError):
        _client(t).complete_many_sync(MSGS[:1])
    assert t.calls == 1, "a 400 is our bug; it must not burn the retry budget"


def test_timeout_exhaustion_raises():
    t = FakeTransport([TimeoutError("timed out")])
    b = Budget(max_requests=50, max_realized_cost_usd=10.0, max_retries=2,
               requests_per_second=0)
    with pytest.raises(RetryExhaustedError):
        _client(t, budget=b).complete_many_sync(MSGS[:1])
    assert t.calls == 3


def test_truncated_completion_fails_the_gate():
    t = FakeTransport([_resp("0.4", finish="length")])
    with pytest.raises(ProvenanceError, match="TRUNCATED"):
        _client(t).complete_many_sync(MSGS[:1])


def test_missing_choices_fails():
    class R:
        status_code = 200
        headers = {}
        text = "{}"
        def json(self):
            return {"id": "x", "model": "google/gemini-3.7-flash",
                    "provider": "Google AI Studio", "choices": []}
    with pytest.raises(ProvenanceError):
        _client(FakeTransport([R()])).complete_many_sync(MSGS[:1])


def test_missing_usage_fails_the_gate():
    class R:
        status_code = 200
        headers = {}
        text = "{}"
        def json(self):
            return {"id": "x", "model": "google/gemini-3.7-flash",
                    "provider": "Google AI Studio",
                    "choices": [{"message": {"content": "0.4"},
                                 "finish_reason": "stop"}],
                    "usage": {}}
    with pytest.raises(ProvenanceError, match="token usage"):
        _client(FakeTransport([R()])).complete_many_sync(MSGS[:1])


def test_provider_fallback_is_rejected():
    t = FakeTransport([_resp("0.42", provider="Amazon Bedrock")])
    with pytest.raises(ProvenanceError, match="PROVIDER FALLBACK"):
        _client(t).complete_many_sync(MSGS[:1])


def test_model_substitution_is_rejected():
    t = FakeTransport([_resp("0.42", model="google/gemini-3.6-flash")])
    with pytest.raises(ProvenanceError, match="MODEL SUBSTITUTION"):
        _client(t).complete_many_sync(MSGS[:1])


def test_provider_pin_is_sent_with_fallbacks_off():
    seen = {}

    def handler(body):
        seen.update(body)
        return _resp("0.42")

    _client(FakeTransport([handler])).complete_many_sync(MSGS[:1])
    p = seen["provider"]
    assert p["allow_fallbacks"] is False
    assert p["require_parameters"] is True
    assert p["data_collection"] == "deny"
    assert p["zdr"] is True
    assert p["order"] == ["Google AI Studio"]
    assert seen["temperature"] == 0.0
    assert seen["max_tokens"] == 16
    assert seen["usage"] == {"include": True}


def test_request_cap_raises():
    b = Budget(max_requests=3, max_realized_cost_usd=10.0,
               requests_per_second=0)
    with pytest.raises(BudgetError, match="request cap"):
        _client(FakeTransport([_resp("0.42")]), budget=b).complete_many_sync(MSGS)


def test_realized_cost_cap_raises():
    b = Budget(max_requests=50, max_realized_cost_usd=0.00025,
               requests_per_second=0, max_concurrency=1)
    with pytest.raises(BudgetError, match="realized cost"):
        _client(FakeTransport([_resp("0.42", cost=0.0001)]),
                budget=b).complete_many_sync(MSGS)


def test_dry_run_makes_no_calls():
    t = FakeTransport([_resp("SHOULD-NOT-BE-CALLED")])
    b = Budget(max_requests=50, max_realized_cost_usd=10.0, dry_run=True)
    out = _client(t, budget=b).complete_many_sync(MSGS)
    assert t.calls == 0
    assert all(o.cache_status == "dry_run" for o in out)
    assert b.n_requests == 0


def test_absent_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_FILE", raising=False)
    with pytest.raises(MissingKeyError):
        load_api_key()


def test_key_file_is_read_and_not_echoed(monkeypatch, tmp_path):
    kf = tmp_path / "k"
    kf.write_text("sk-or-filekey-abcdef\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY_FILE", str(kf))
    assert load_api_key() == "sk-or-filekey-abcdef"
    monkeypatch.setenv("OPENROUTER_API_KEY_FILE", str(tmp_path / "nope"))
    with pytest.raises(MissingKeyError) as e:
        load_api_key()
    assert "sk-or" not in str(e.value)


def test_redaction_covers_keys_and_headers():
    assert "sk-or-abcdef123456" not in _redact("leak sk-or-abcdef123456 here")
    assert "REDACTED" in _redact("leak sk-or-abcdef123456 here")
    assert "supersecretvalue" not in _redact(
        '{"Authorization": "Bearer supersecretvalue"}')


def test_error_text_from_provider_is_redacted():
    class R:
        status_code = 403
        headers = {}
        text = 'denied for key sk-or-abcdef123456789'
        def json(self):
            return {}
    with pytest.raises(OpenRouterError) as e:
        _client(FakeTransport([R()])).complete_many_sync(MSGS[:1])
    assert "sk-or-abcdef123456789" not in str(e.value)


def test_headers_carry_bearer_but_body_never_does():
    seen = {}

    def handler(body):
        seen.update(body)
        return _resp("0.42")

    t = FakeTransport([handler])
    _client(t).complete_many_sync(MSGS[:1])
    assert t.seen_headers[0]["Authorization"].startswith("Bearer ")
    assert "TESTKEY" not in json.dumps(seen)


# ---- omitted temperature + cell-coordinate cache keys -------------------
def test_temperature_can_be_omitted_for_endpoints_that_lack_it():
    """The GPT-5.x reasoning family exposes no temperature. With
    require_parameters=true, sending it makes the pinned endpoint
    ineligible, so it must be OMITTED -- not defaulted, not sent as 1.0."""
    seen = {}

    def handler(body):
        seen.update(body)
        return _resp("0.42")

    pol = DecodingPolicy(temperature=None, max_tokens=16, seed=7)
    c = OpenRouterClient(
        model="google/gemini-3.7-flash", provider=PIN, policy=pol,
        budget=Budget(max_requests=5, max_realized_cost_usd=1.0),
        api_key="sk-or-TEST", transport=FakeTransport([handler]))
    c.complete_many_sync(MSGS[:1])
    assert "temperature" not in seen
    assert "top_p" not in seen, "top_p rides with temperature or not at all"
    assert seen["max_tokens"] == 16
    assert seen["seed"] == 7


def test_cache_key_separates_seeds_rounds_and_agents():
    """At round 0 every population seed renders an IDENTICAL prompt. Without
    the coordinates the three cells of one model would share one paid
    response and stop being independent."""
    base = dict(model="m", pin=PIN, messages=MSGS[0], policy=POLICY)
    h = lambda **ctx: request_hash(base["model"], base["pin"],
                                   base["messages"], base["policy"],
                                   context=ctx)
    assert h(seed=0, round=0, agent=1) != h(seed=42, round=0, agent=1)
    assert h(seed=0, round=0, agent=1) != h(seed=0, round=7, agent=1)
    assert h(seed=0, round=0, agent=1) != h(seed=0, round=0, agent=2)
    assert h(seed=0, round=0, agent=1) == h(seed=0, round=0, agent=1)


def test_client_folds_round_and_agent_into_the_key(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    t = FakeTransport([_resp("0.42")])
    c = OpenRouterClient(
        model="google/gemini-3.7-flash", provider=PIN, policy=POLICY,
        budget=Budget(max_requests=50, max_realized_cost_usd=10.0,
                      requests_per_second=0),
        cache=cache, api_key="sk-or-TEST", transport=t,
        cache_context={"seed": 0, "round": 0})
    c.complete_many_sync(MSGS[:2])
    assert t.calls == 2, "two agents, two distinct keys"
    # same agents, later round -> not a hit
    c.cache_context = {"seed": 0, "round": 1}
    c.complete_many_sync(MSGS[:2])
    assert t.calls == 4
    # back to round 0 -> both hits, nothing paid
    c.cache_context = {"seed": 0, "round": 0}
    c.complete_many_sync(MSGS[:2])
    assert t.calls == 4


# ---- canonical-slug drift ----------------------------------------------
def _catalog(model_id, slug):
    return lambda url: {"data": [{"id": model_id, "canonical_slug": slug}]}


def test_canonical_drift_is_a_hard_failure():
    """A provider can re-point a routable id at a new dated build mid-wave.
    The completion response would not show it -- `model` comes back as the
    same routable id -- so the wave would silently become two models."""
    import perfsim.models.openrouter_client as orc
    from perfsim.models.openrouter_client import (
        CanonicalDriftError, assert_canonical)
    orc._CANON_CACHE.clear()
    assert_canonical("openai/gpt-5.6-sol", "openai/gpt-5.6-sol-20260709",
                     fetch=_catalog("openai/gpt-5.6-sol",
                                    "openai/gpt-5.6-sol-20260709"))
    orc._CANON_CACHE.clear()
    with pytest.raises(CanonicalDriftError, match="CANONICAL DRIFT"):
        assert_canonical("openai/gpt-5.6-sol", "openai/gpt-5.6-sol-20260709",
                         fetch=_catalog("openai/gpt-5.6-sol",
                                        "openai/gpt-5.6-sol-20261101"))
    orc._CANON_CACHE.clear()
    with pytest.raises(CanonicalDriftError):
        assert_canonical("openai/gpt-5.6-sol", "openai/gpt-5.6-sol-20260709",
                         fetch=lambda url: {"data": []})


def test_canonical_check_is_cached_so_it_is_cheap_per_round():
    import perfsim.models.openrouter_client as orc
    from perfsim.models.openrouter_client import canonical_slug_of
    orc._CANON_CACHE.clear()
    calls = {"n": 0}

    def fetch(url):
        calls["n"] += 1
        return {"data": [{"id": "m", "canonical_slug": "m-2026"}]}

    for _ in range(30):
        assert canonical_slug_of("m", fetch=fetch) == "m-2026"
    assert calls["n"] == 1, "30 rounds must not mean 30 catalog fetches"


def test_validate_key_never_returns_key_material():
    """OpenRouter's `label` is built from the key's own prefix and suffix
    ('sk-or-v1-b82...f03'), so returning it would leak key material into
    every log line that prints this dict."""
    from perfsim.models.openrouter_client import validate_key

    class R:
        status_code = 200
        def json(self):
            return {"data": {"label": "sk-or-v1-b82...f03",
                             "is_free_tier": True, "usage": 0,
                             "limit": None, "limit_remaining": None,
                             "rate_limit": {"requests": -1}}}

    class T:
        def get(self, url, headers=None, timeout=None):
            assert headers["Authorization"].startswith("Bearer ")
            return R()

    meta = validate_key(api_key="sk-or-TESTKEY", transport=T())
    blob = json.dumps(meta)
    assert "label" not in meta
    assert "sk-or" not in blob and "b82" not in blob and "f03" not in blob
    assert meta["is_free_tier"] is True


def test_require_parameters_defaults_off_but_is_settable():
    """Measured 2026-08-31: with zdr=true, require_parameters=true leaves
    ZERO endpoints for all three frontier models. The default flipped for
    that reason; the knob itself must still work."""
    assert ProviderPin(order=("X",)).to_body()["require_parameters"] is True
    assert ProviderPin(order=("X",), require_parameters=False
                       ).to_body()["require_parameters"] is False


def test_mandatory_reasoning_falls_back_to_minimal_and_records_it():
    """Some endpoints refuse to disable reasoning outright (HTTP 400
    'Reasoning is mandatory for this endpoint and cannot be disabled').
    The spec is 'disable OR MINIMIZE', so the client minimizes -- once, and
    the fallback is recorded on the response rather than hidden."""
    class R400:
        status_code = 400
        headers: dict = {}
        text = ('{"error":{"message":"Reasoning is mandatory for this '
                'endpoint and cannot be disabled.","code":400}}')
        def json(self):
            return json.loads(self.text)

    sent = []

    def handler(body):
        sent.append(body.get("reasoning"))
        return _resp("0.42")

    t = FakeTransport([R400(), handler])
    c = OpenRouterClient(
        model="google/gemini-3.7-flash", provider=PIN,
        policy=DecodingPolicy(temperature=0.0, reasoning_mode="disabled"),
        budget=Budget(max_requests=5, max_realized_cost_usd=1.0,
                      requests_per_second=0),
        api_key="sk-or-TEST", transport=t)
    out = c.complete_many_sync(MSGS[:1])
    assert out[0].text == "0.42"
    assert out[0].reasoning_fallback is True
    assert sent == [{"effort": "minimal", "exclude": True}]


def test_reasoning_modes_render_correctly():
    assert DecodingPolicy(reasoning_mode="disabled").to_body()["reasoning"] \
        == {"enabled": False, "exclude": True}
    assert DecodingPolicy(reasoning_mode="minimal").to_body()["reasoning"] \
        == {"effort": "minimal", "exclude": True}
    assert "reasoning" not in DecodingPolicy(reasoning_mode="none").to_body()
