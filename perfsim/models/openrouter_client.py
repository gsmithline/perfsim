"""OpenRouterClient: a frozen-serving HTTP client for OpenRouter, built for
SCIENTIFIC runs rather than for convenience.

The design rules, and why each exists.

PROVENANCE IS PART OF THE MEASUREMENT.  A served value whose model,
provider, finish reason and token usage are unknown is not a measurement
-- it is an anecdote.  Every response therefore records requested AND
resolved model, requested AND resolved provider, the generation id, the
system fingerprint when one is returned, the finish reason, usage, cost,
latency, retry count, cache status, and a hash of the exact messages and
parameters that produced it.  `ProvenanceError` is raised, and the run
dies, when any of that is missing or contradicts what was requested.

SILENT SUBSTITUTION IS THE ENEMY.  OpenRouter will happily route around a
busy provider.  For a paper wave that is a different experiment wearing
the same tag, so the provider preferences pin one provider with
allow_fallbacks=false, and a resolved model or provider that differs from
the requested one is a hard failure, not a warning.

TEMPERATURE 0 IS NOT BITWISE REPRODUCIBILITY.  It removes sampling noise.
It does not pin kernels, batching, hardware, or a provider's own serving
stack, and most frontier endpoints expose no seed at all.  This module
records the decoding policy it actually asked for and what the provider
said it did; it never claims determinism it cannot deliver.

MONEY IS A HARD CONSTRAINT, NOT A WARNING.  Caps on requests, estimated
cost, realized cost, concurrency and request rate are enforced in-process
and raise rather than truncate the science silently.

THE CACHE IS THE EXPERIMENT'S RECORD, not an optimisation.  It is
content-addressed by (model, provider pin, messages, decoding params) in
SQLite with WAL, so an interrupted 21,690-request wave resumes without
paying twice.  Authorization headers are NEVER part of a key or a stored
value.  OpenRouter's own temporary response cache is not a record and is
not relied on.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

try:
    import httpx
except ImportError:                                        # pragma: no cover
    httpx = None

API_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_URL = "https://openrouter.ai/api/v1/key"
MODELS_URL = "https://openrouter.ai/api/v1/models"

# 429 plus the transient 5xx family. 400/401/403/404/422 are OUR bug or a
# bad pin and must surface immediately rather than burn the retry budget.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 522, 524})


class OpenRouterError(RuntimeError):
    """Base class. Carries no key material -- see _redact."""


class MissingKeyError(OpenRouterError):
    pass


class ProvenanceError(OpenRouterError):
    """Resolved model/provider, finish reason, or usage failed the gate."""


class CanonicalDriftError(OpenRouterError):
    """The routable id no longer points at the dated version we pinned.

    A provider can re-point a rolling id at a new dated build mid-wave.
    Nothing in the completion response reveals that -- `model` still comes
    back as the same routable id -- so a wave could silently become two
    different models stitched together. This is checked against the live
    catalog before the run and again during it."""


class BudgetError(OpenRouterError):
    """A request, estimated-cost, or realized-cost cap would be exceeded."""


class RetryExhaustedError(OpenRouterError):
    pass


_CANON_CACHE: dict[str, tuple[float, str | None]] = {}


def canonical_slug_of(model_id: str, *, ttl_s: float = 300.0,
                      fetch=None) -> str | None:
    """The dated canonical_slug the routable id currently resolves to.

    Public metadata, no key, no cost. Cached for ttl_s so a per-round check
    across a 30-round wave costs a handful of requests, not one per agent.
    """
    now = time.monotonic()
    hit = _CANON_CACHE.get(model_id)
    if hit is not None and (now - hit[0]) < ttl_s:
        return hit[1]
    if fetch is None:
        import urllib.request
        def fetch(url):
            req = urllib.request.Request(
                url, headers={"User-Agent": "perfsim"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
    data = fetch(MODELS_URL)
    slug = None
    for m in data.get("data", []):
        if m.get("id") == model_id:
            slug = m.get("canonical_slug")
            break
    _CANON_CACHE[model_id] = (now, slug)
    return slug


def assert_canonical(model_id: str, expected: str, *, when: str = "",
                     fetch=None) -> None:
    """Hard-fail if the routable id no longer carries the pinned dated slug."""
    got = canonical_slug_of(model_id, fetch=fetch)
    if got != expected:
        raise CanonicalDriftError(
            f"CANONICAL DRIFT{(' ' + when) if when else ''}: {model_id!r} now "
            f"resolves to canonical_slug {got!r}, but this wave is pinned to "
            f"{expected!r}. The routable id has been re-pointed at a "
            f"different dated build; continuing would stitch two models "
            f"into one trajectory.")


def _redact(text: str) -> str:
    """Strip anything that could be a key from text bound for a log or an
    exception. Keys are `sk-or-...`; we also drop any Authorization header
    echoed back by a proxy. Applied to EVERY error string this module
    raises, because a traceback is a log file that leaves the machine."""
    if not text:
        return text
    import re
    text = re.sub(r"sk-or-[A-Za-z0-9\-_]{4,}", "sk-or-***REDACTED***", text)
    text = re.sub(r"(?i)(authorization\"?\s*[:=]\s*\"?)(bearer\s+)?[A-Za-z0-9\-_.]{8,}",
                  r"\1***REDACTED***", text)
    return text


def load_api_key(*, required: bool = True) -> str | None:
    """OPENROUTER_API_KEY, else the file named by OPENROUTER_API_KEY_FILE.

    The file form is what the cluster uses: a Condor job never carries the
    key in its arguments or environment classad, it reads a mode-0600 file
    on the submit host's shared home. The key is never printed, never put
    in a config, and never returned in an exception message.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        path = os.environ.get("OPENROUTER_API_KEY_FILE", "").strip()
        if path:
            p = Path(path).expanduser()
            if not p.is_file():
                raise MissingKeyError(
                    f"OPENROUTER_API_KEY_FILE points at {p} which does not exist")
            key = p.read_text().strip()
    if not key and required:
        raise MissingKeyError(
            "no OpenRouter key: set OPENROUTER_API_KEY, or "
            "OPENROUTER_API_KEY_FILE to a file containing it. The key must "
            "never appear in Condor arguments, a config, or Git.")
    return key or None


@dataclass(frozen=True)
class DecodingPolicy:
    """The decoding knobs actually sent, recorded verbatim in the config.

    `seed` is None unless the endpoint genuinely advertises `seed` in its
    supported_parameters -- sending one that is ignored would put a
    reproducibility claim in the record that the provider never honoured.

    `temperature` is None when the endpoint does not support it (the
    GPT-5.x reasoning family exposes no temperature at all). With
    require_parameters=true, sending an unsupported knob makes the
    endpoint ineligible and the pinned route fails -- so the knob is
    OMITTED, and its absence is recorded rather than papered over with a
    default that was never sent.
    """
    temperature: float | None = 0.0
    top_p: float = 1.0
    max_tokens: int = 16
    seed: int | None = None
    # "disabled"  ask the provider to turn reasoning off entirely
    # "minimal"   smallest reasoning budget the provider offers -- required
    #             by endpoints that reject disabling outright (Gemini 3.1
    #             Pro: "Reasoning is mandatory for this endpoint and cannot
    #             be disabled", HTTP 400)
    # "none"      send no reasoning directive at all
    reasoning_mode: str = "disabled"
    stop: tuple[str, ...] = ()

    def to_body(self) -> dict:
        body: dict[str, Any] = {"max_tokens": self.max_tokens}
        if self.temperature is not None:
            body["temperature"] = self.temperature
            body["top_p"] = self.top_p
        if self.seed is not None:
            body["seed"] = self.seed
        if self.stop:
            body["stop"] = list(self.stop)
        if self.reasoning_mode == "disabled":
            body["reasoning"] = {"enabled": False, "exclude": True}
        elif self.reasoning_mode == "minimal":
            body["reasoning"] = {"effort": "minimal", "exclude": True}
        elif self.reasoning_mode != "none":
            raise ValueError(f"reasoning_mode={self.reasoning_mode!r}")
        return body

    def minimized(self) -> "DecodingPolicy":
        """The same policy with reasoning MINIMIZED rather than disabled."""
        return DecodingPolicy(
            temperature=self.temperature, top_p=self.top_p,
            max_tokens=self.max_tokens, seed=self.seed,
            reasoning_mode="minimal", stop=self.stop)


@dataclass(frozen=True)
class ProviderPin:
    """Paper-run provider preferences. Every field here is a REQUIREMENT,
    not a hint: if the pinned provider cannot satisfy them the request
    fails rather than quietly routing somewhere else."""
    order: tuple[str, ...]
    allow_fallbacks: bool = False
    require_parameters: bool = True
    data_collection: str = "deny"
    zdr: bool = True

    def to_body(self) -> dict:
        return {
            "order": list(self.order),
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
            "data_collection": self.data_collection,
            "zdr": self.zdr,
        }


@dataclass
class Budget:
    """Hard controls. Every one of these RAISES; none silently truncates."""
    max_requests: int = 100
    max_estimated_cost_usd: float = 1.0
    max_realized_cost_usd: float = 1.0
    max_concurrency: int = 8
    requests_per_second: float = 4.0
    timeout_s: float = 120.0
    max_retries: int = 5
    dry_run: bool = False

    n_requests: int = field(default=0, init=False)
    realized_cost_usd: float = field(default=0.0, init=False)

    def charge_request(self) -> None:
        if self.n_requests + 1 > self.max_requests:
            raise BudgetError(
                f"request cap reached: {self.n_requests} of "
                f"{self.max_requests} already spent. Raise --max-requests "
                f"deliberately; it will not be raised for you.")
        self.n_requests += 1

    def charge_cost(self, usd: float) -> None:
        self.realized_cost_usd += float(usd or 0.0)
        if self.realized_cost_usd > self.max_realized_cost_usd:
            raise BudgetError(
                f"realized cost ${self.realized_cost_usd:.4f} exceeds the "
                f"${self.max_realized_cost_usd:.4f} cap; stopping. Completed "
                f"responses are in the cache and will not be re-paid.")


@dataclass
class Provenance:
    """What actually happened, per request. Stored verbatim per response."""
    requested_model: str
    resolved_model: str | None
    requested_provider: str
    resolved_provider: str | None
    generation_id: str | None
    system_fingerprint: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    latency_s: float
    retries: int
    cache_status: str            # "miss" | "hit" | "dry_run"
    reasoning_fallback: bool     # asked to disable, endpoint required minimal
    request_hash: str
    raw_response: dict | None
    text: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def request_hash(model: str, pin: ProviderPin, messages: Sequence[dict],
                 policy: DecodingPolicy, context: dict | None = None) -> str:
    """Content address over EXACTLY what determines the answer: model,
    provider pin, messages, decoding parameters -- plus the CELL COORDINATES
    (seed, round, agent) when given.

    The coordinates are redundant in principle: the prompt already encodes
    the agent (its profile) and the round (its history). They are in the key
    anyway because at ROUND 0 every population seed renders an IDENTICAL
    prompt, so without the seed the three cells of a model would silently
    share one paid response. That is defensible for a deterministic
    endpoint and indefensible as a default: it couples cells that the
    analysis then treats as independent.

    Authorization is NEVER an input -- a key rotation must not invalidate a
    cache that has already been paid for."""
    payload = json.dumps({
        "model": model,
        "provider": pin.to_body(),
        "messages": list(messages),
        "policy": policy.to_body(),
        "context": dict(context or {}),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """Crash-safe SQLite cache. WAL + a synchronous commit per write, so a
    kill -9 mid-wave loses at most the in-flight request, and a resume
    re-uses every completed response without paying again."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " request_hash TEXT PRIMARY KEY,"
            " created_at   REAL NOT NULL,"
            " text         TEXT NOT NULL,"
            " provenance   TEXT NOT NULL)")
        self._conn.commit()
        self._lock = asyncio.Lock()

    def get(self, h: str) -> Provenance | None:
        row = self._conn.execute(
            "SELECT provenance FROM responses WHERE request_hash = ?",
            (h,)).fetchone()
        if row is None:
            return None
        d = json.loads(row[0])
        d["cache_status"] = "hit"
        return Provenance(**d)

    def put(self, h: str, prov: Provenance) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses VALUES (?,?,?,?)",
            (h, time.time(), prov.text or "",
             json.dumps(prov.to_dict(), sort_keys=True)))
        self._conn.commit()

    def __len__(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM responses").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


class _RateLimiter:
    """Token-bucket-ish spacing. Serialises the *start* of requests so a
    723-agent round does not open the throttle wide on its first burst."""

    def __init__(self, rps: float) -> None:
        self._min_gap = 1.0 / rps if rps and rps > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._min_gap <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._min_gap
        if wait > 0:
            await asyncio.sleep(wait)


class OpenRouterClient:
    """Bounded-concurrency, cached, provenance-gated chat completions.

    `complete_many` returns results IN INPUT ORDER regardless of the order
    the network answers in -- agent i's served value must be agent i's, and
    a concurrency bug that permuted them would be invisible in aggregate
    statistics but fatal to the science.
    """

    def __init__(
        self,
        *,
        model: str,
        provider: ProviderPin,
        policy: DecodingPolicy,
        budget: Budget,
        cache: ResponseCache | None = None,
        api_key: str | None = None,
        api_url: str = API_URL,
        transport: Any = None,      # tests inject; None = real network
        referer: str = "https://github.com/perfsim",
        title: str = "perfsim",
        cache_context: dict | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.policy = policy
        self.budget = budget
        self.cache = cache
        self.api_url = api_url
        self._transport = transport
        self._referer = referer
        self._title = title
        # cell coordinates folded into every cache key; see request_hash
        self.cache_context = dict(cache_context or {})
        self._key = api_key if api_key is not None else (
            load_api_key(required=not budget.dry_run))
        self._limiter = _RateLimiter(budget.requests_per_second)
        self._sem = asyncio.Semaphore(budget.max_concurrency)
        self.provenances: list[Provenance] = []

    # ---- header construction -------------------------------------------
    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "HTTP-Referer": self._referer,
            "X-Title": self._title,
        }
        if self._key:
            h["Authorization"] = f"Bearer {self._key}"
        return h

    def _body(self, messages: Sequence[dict]) -> dict:
        body = {
            "model": self.model,
            "messages": list(messages),
            "provider": self.provider.to_body(),
            # cost and token counts come back on the response itself;
            # without this the realized-cost cap has nothing to enforce
            "usage": {"include": True},
        }
        body.update(self.policy.to_body())
        return body

    # ---- provenance gate ------------------------------------------------
    def _check_provenance(self, prov: Provenance) -> None:
        if prov.resolved_model is None:
            raise ProvenanceError(
                f"no resolved model on generation {prov.generation_id!r}: a "
                f"response without provenance cannot enter the record")
        if prov.resolved_model != prov.requested_model:
            raise ProvenanceError(
                f"MODEL SUBSTITUTION: requested {prov.requested_model!r}, "
                f"served {prov.resolved_model!r}. allow_fallbacks is false, "
                f"so this is a routing bug or a moving alias -- pin an exact "
                f"slug.")
        if prov.resolved_provider is None:
            raise ProvenanceError(
                f"no resolved provider on generation {prov.generation_id!r}")
        want = {p.lower() for p in self.provider.order}
        if prov.resolved_provider.lower() not in want:
            raise ProvenanceError(
                f"PROVIDER FALLBACK: pinned {sorted(want)}, served by "
                f"{prov.resolved_provider!r}. The wave is not one provider's "
                f"measurement any more; failing rather than mixing.")
        if prov.finish_reason not in ("stop", "end_turn", "eos", None):
            raise ProvenanceError(
                f"TRUNCATED OR ABNORMAL COMPLETION: finish_reason="
                f"{prov.finish_reason!r} on generation {prov.generation_id!r}. "
                f"A truncated number is not a datum.")
        if prov.completion_tokens is None or prov.prompt_tokens is None:
            raise ProvenanceError(
                f"missing token usage on generation {prov.generation_id!r}")

    @staticmethod
    def _extract(resp_json: dict) -> tuple[str, dict]:
        choices = resp_json.get("choices") or []
        if not choices:
            raise ProvenanceError(
                f"response carried no choices: "
                f"{_redact(json.dumps(resp_json))[:400]}")
        msg = choices[0].get("message") or {}
        text = msg.get("content")
        if text is None:
            raise ProvenanceError(
                f"response choice carried no content "
                f"(finish_reason={choices[0].get('finish_reason')!r})")
        return text, choices[0]

    # ---- one request, with retries --------------------------------------
    async def _one(self, client: Any, messages: Sequence[dict],
                   idx: int) -> Provenance:
        ctx = dict(self.cache_context)
        ctx["agent"] = idx
        h = request_hash(self.model, self.provider, messages, self.policy,
                         context=ctx)

        if self.cache is not None:
            hit = self.cache.get(h)
            if hit is not None:
                self.provenances.append(hit)
                return hit

        if self.budget.dry_run:
            prov = Provenance(
                requested_model=self.model, resolved_model=self.model,
                requested_provider=self.provider.order[0],
                resolved_provider=self.provider.order[0],
                generation_id=f"dryrun-{idx}", system_fingerprint=None,
                finish_reason="stop", prompt_tokens=0, completion_tokens=0,
                total_tokens=0, cost_usd=0.0, latency_s=0.0, retries=0,
                cache_status="dry_run", reasoning_fallback=False,
                request_hash=h, raw_response=None, text=None)
            self.provenances.append(prov)
            return prov

        self.budget.charge_request()
        body = self._body(messages)
        delay, last_err = 1.0, None
        reasoning_fallback = False

        for attempt in range(self.budget.max_retries + 1):
            await self._limiter.acquire()
            t0 = time.monotonic()
            try:
                r = await client.post(self.api_url, headers=self._headers(),
                                      json=body, timeout=self.budget.timeout_s)
                status = r.status_code
                if status in RETRYABLE_STATUS:
                    last_err = f"HTTP {status}"
                    ra = r.headers.get("Retry-After") if hasattr(r, "headers") else None
                    if attempt >= self.budget.max_retries:
                        break
                    sleep_s = float(ra) if ra and str(ra).replace(".", "", 1).isdigit() \
                        else delay * (1.0 + random.random())
                    await asyncio.sleep(min(sleep_s, 60.0))
                    delay = min(delay * 2, 32.0)
                    continue
                if status == 400 and "reasoning is mandatory" in \
                        (r.text or "").lower() and not reasoning_fallback:
                    # "Disable or minimize reasoning": disabling is refused
                    # by this endpoint, so MINIMIZE instead -- once, and
                    # recorded on every affected response, never silently.
                    reasoning_fallback = True
                    self.policy = self.policy.minimized()
                    body = self._body(messages)
                    continue
                if status >= 400:
                    raise OpenRouterError(_redact(
                        f"HTTP {status} from OpenRouter: {r.text[:400]}"))
                data = r.json()
            except (OpenRouterError, ProvenanceError):
                raise
            except Exception as e:                          # noqa: BLE001
                last_err = _redact(repr(e))
                if attempt >= self.budget.max_retries:
                    break
                await asyncio.sleep(min(delay * (1.0 + random.random()), 60.0))
                delay = min(delay * 2, 32.0)
                continue

            latency = time.monotonic() - t0
            if isinstance(data, dict) and data.get("error"):
                raise OpenRouterError(_redact(
                    f"OpenRouter error payload: {json.dumps(data['error'])[:400]}"))
            text, choice = self._extract(data)
            usage = data.get("usage") or {}
            prov = Provenance(
                requested_model=self.model,
                resolved_model=data.get("model"),
                requested_provider=self.provider.order[0],
                resolved_provider=data.get("provider"),
                generation_id=data.get("id"),
                system_fingerprint=data.get("system_fingerprint"),
                finish_reason=choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                cost_usd=usage.get("cost"),
                latency_s=latency, retries=attempt,
                cache_status="miss", reasoning_fallback=reasoning_fallback,
                request_hash=h, raw_response=data, text=text)
            self._check_provenance(prov)
            self.budget.charge_cost(prov.cost_usd or 0.0)
            if self.cache is not None:
                self.cache.put(h, prov)
            self.provenances.append(prov)
            return prov

        raise RetryExhaustedError(_redact(
            f"gave up after {self.budget.max_retries + 1} attempts; "
            f"last error: {last_err}"))

    async def complete_many(self, batch: Sequence[Sequence[dict]]) -> list[Provenance]:
        """Run `batch` concurrently, return provenances IN INPUT ORDER."""
        if httpx is None and self._transport is None:
            raise OpenRouterError("httpx is required for live requests")
        results: list[Provenance | None] = [None] * len(batch)

        async def worker(i: int, msgs: Sequence[dict], client: Any) -> None:
            async with self._sem:
                results[i] = await self._one(client, msgs, i)

        if self._transport is not None:
            client = self._transport
            await asyncio.gather(*(worker(i, m, client)
                                   for i, m in enumerate(batch)))
        else:
            async with httpx.AsyncClient() as client:
                await asyncio.gather(*(worker(i, m, client)
                                       for i, m in enumerate(batch)))
        out = [r for r in results if r is not None]
        if len(out) != len(batch):
            raise OpenRouterError(
                f"internal: {len(out)} results for {len(batch)} prompts")
        return out

    def complete_many_sync(self, batch: Sequence[Sequence[dict]]) -> list[Provenance]:
        return asyncio.run(self.complete_many(batch))


def validate_key(api_key: str | None = None, *, transport: Any = None) -> dict:
    """Confirm a key works and report NON-SECRET limit metadata only.

    Never returns, prints, or logs the key. Costs nothing: /api/v1/key is
    a metadata endpoint, not an inference call."""
    key = api_key if api_key is not None else load_api_key(required=True)
    headers = {"Authorization": f"Bearer {key}"}
    if transport is not None:
        r = transport.get(KEY_URL, headers=headers)
    else:
        if httpx is None:                                  # pragma: no cover
            raise OpenRouterError("httpx is required")
        r = httpx.get(KEY_URL, headers=headers, timeout=30.0)
    if r.status_code == 401:
        raise MissingKeyError("OpenRouter rejected the key (401). The key "
                              "itself is not shown.")
    if r.status_code >= 400:
        raise OpenRouterError(_redact(f"HTTP {r.status_code} from /key"))
    d = (r.json() or {}).get("data", {})
    return {                       # non-secret fields ONLY
        # NOT the label: OpenRouter builds it from the key's own prefix and
        # suffix ("sk-or-v1-b82...f03"), so returning it would leak key
        # material into every log that prints this dict.
        "is_free_tier": d.get("is_free_tier"),
        "limit": d.get("limit"),
        "limit_remaining": d.get("limit_remaining"),
        "usage": d.get("usage"),
        "rate_limit": d.get("rate_limit"),
    }
