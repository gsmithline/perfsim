"""OpenRouterModel: a FROZEN, API-served stand-in for HFCausalLMModel.

It implements exactly the serving surface the runner and _gated_pop use
-- forward, _generate, profile_at, build_prompt, parse, parse_ok,
_last_raw, _last_parse_fail, effective_generation_policy -- and NOTHING
else. Every capability that implies a weight update or a gradient raises
on contact.

WHY THE HARD FAILURES ARE FAILURES AND NOT NO-OPS.  A silent no-op here
would be the worst possible outcome: a run configured for LoRA + SFT
would complete, write a trajectory, and be indistinguishable in its
artifacts from a trained wave -- while the model behind the API never
changed at all. Every such request therefore raises FrozenBackendError,
loudly, before a single request is made.

WHY THE PARSER IS THE EXISTING ONE.  Section 3 already learned that a
lenient parser reads ".64 (" as 1.0 and records no failure. This backend
reuses HFCausalLMModel's strict parser verbatim rather than growing a
second dialect of "what counts as a number", and it NEVER serves the 0.5
fallback: an unparseable frontier-model response is a hard failure,
because unlike a local checkpoint we cannot re-run it deterministically
to see what happened.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.models.openrouter_client import (
    Budget, DecodingPolicy, OpenRouterClient, ProviderPin, Provenance,
    OpenRouterError, assert_canonical,
)

MessageBuilder = Callable[..., list]


class FrozenBackendError(RuntimeError):
    """A training/adaptation capability was requested of a frozen API model."""


class ParseFailure(RuntimeError):
    """An API response could not be parsed as a number in [0, 1]."""


def _strict_parse(text: str) -> tuple[float, bool]:
    """The repo's existing strict parser, imported lazily so this module
    stays importable without transformers installed."""
    from perfsim.models.hf_causal_lm import HFCausalLMModel
    return HFCausalLMModel._parse_strict(text)


class OpenRouterModel(Model):
    """Frozen serving through OpenRouter. Weights are the provider's and
    are never touched; there is no optimizer, no adapter, no gradient."""

    def __init__(
        self,
        *,
        model_slug: str,
        profiles: object,
        message_builder: MessageBuilder,
        provider: ProviderPin,
        policy: DecodingPolicy,
        budget: Budget,
        cache: Any = None,
        api_key: str | None = None,
        transport: Any = None,
        parse_mode: str = "strict",
        run_seed: int | None = None,
        expected_canonical: str | None = None,
    ) -> None:
        super().__init__()
        self._model_slug = model_slug
        self._profiles = profiles
        self._message_builder = message_builder
        self._provider = provider
        self._policy = policy
        self._budget = budget
        self.parse_mode = parse_mode
        if parse_mode != "strict":
            raise FrozenBackendError(
                f"PARSE_MODE={parse_mode!r}: the OpenRouter backend requires "
                f"strict parsing. A frontier response cannot be re-run "
                f"deterministically, so a lenient parse would bake an "
                f"unexplained value into the record.")

        length = getattr(profiles, "__len__", lambda: -1)()
        if length == -1:
            raise TypeError("profiles must have a defined len()")
        self._n = int(length)

        # CELL COORDINATES in every cache key. current_round is advanced by
        # the runner; agent index is added per request by the client. At
        # round 0 every population seed renders an identical prompt, so
        # without run_seed the three cells of one model would share one
        # paid response and stop being independent.
        self.current_round = 0
        self._run_seed = run_seed
        # The DATED build this wave is pinned to. A provider can re-point a
        # routable id at a new dated version mid-wave and the completion
        # response would not show it, so this is verified against the live
        # catalog before the first request and again every round.
        self._expected_canonical = expected_canonical
        if expected_canonical:
            assert_canonical(model_slug, expected_canonical,
                             when="at construction")
        self.client = OpenRouterClient(
            model=model_slug, provider=provider, policy=policy,
            budget=budget, cache=cache, api_key=api_key, transport=transport,
            cache_context={"seed": run_seed, "round": 0})

        # the same telemetry surface HFCausalLMModel exposes
        self._last_raw: list[str] = []
        self._last_parse_fail: float = 0.0
        self._last_provenance: list[Provenance] = []
        # `tokenizer` exists ONLY so code that probes for it sees None and
        # takes no chat-template path. There is no local tokenizer here.
        self.tokenizer = None
        self.inner_model = None

    # ---- the frozen contract: everything below raises -------------------
    def _frozen(self, what: str) -> None:
        raise FrozenBackendError(
            f"{what} is not available on the OpenRouter backend: the model "
            f"is FROZEN behind an API and its weights cannot be read, "
            f"written, or differentiated. Use MODEL_BACKEND=hf for any wave "
            f"that adapts the model.")

    def ensure_loaded(self) -> None:
        return None                      # nothing to load; kept for parity

    def perplexity(self, texts):         # noqa: D102
        self._frozen("perplexity (LOG_PERPLEXITY)")

    def get_params(self) -> Tensor:
        self._frozen("get_params")

    def set_params(self, theta: Tensor) -> None:
        self._frozen("set_params")

    def clone(self):
        self._frozen("clone")

    def num_params(self) -> int:
        self._frozen("num_params")

    def answer_distribution_stats(self) -> dict:
        self._frozen("answer_distribution_stats (LOG_ANSWER_DIST)")

    def answer_sample_stats(self, *a, **kw):
        self._frozen("answer_sample_stats (ANS_SAMPLE_K)")

    # ---- profiles and prompts -------------------------------------------
    def profiles(self) -> object:
        return self._profiles

    def profile_at(self, idx: int) -> object:
        if hasattr(self._profiles, "iloc"):
            return self._profiles.iloc[int(idx)]
        return self._profiles[int(idx)]

    def build_prompt(self, profile: object, tokenizer: Any = None,
                     context_block: str | None = None) -> str:
        """The SEMANTIC user message -- never a chat-template render.

        `tokenizer` is accepted and ignored so the runner can call this
        through the same signature it uses for the HF path; passing one
        here would be a bug, and there is nothing to apply it to.
        """
        msgs = self._message_builder(profile, context_block=context_block)
        if len(msgs) != 1 or msgs[0].get("role") != "user":
            raise OpenRouterError(
                f"message_builder must return exactly one user message; got "
                f"{[m.get('role') for m in msgs]}")
        return msgs[0]["content"]

    def build_messages(self, profile: object,
                       context_block: str | None = None) -> list[dict]:
        return self._message_builder(profile, context_block=context_block)

    # ---- serving ---------------------------------------------------------
    def _generate(self, prompts: Sequence[str]) -> list[str]:
        """Bounded-concurrency completions, returned IN INPUT ORDER.

        Each prompt is a semantic user message and is sent as a standard
        chat message. A prompt that already looks like a chat-template
        render is refused: pasting one provider's wire format into
        another's `user` field is a different prompt, and it would be
        invisible in the served numbers.
        """
        for p in prompts:
            if _looks_like_chat_template(p):
                raise OpenRouterError(
                    "refusing to send what looks like a Hugging Face "
                    "chat-template render as an OpenRouter user message "
                    f"(found a template control token in: {p[:80]!r}). Build "
                    "provider-neutral messages instead.")
        if self._expected_canonical:
            assert_canonical(self._model_slug, self._expected_canonical,
                             when=f"before round {self.current_round}")
        self.client.cache_context = {"seed": self._run_seed,
                                     "round": int(self.current_round)}
        batch = [[{"role": "user", "content": p}] for p in prompts]
        provs = self.client.complete_many_sync(batch)
        self._last_provenance = provs
        return [p.text if p.text is not None else "" for p in provs]

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[0] != self._n:
            raise ValueError(
                f"OpenRouterModel.forward: x leading dim {x.shape[0]} does "
                f"not match profiles N={self._n}")
        prompts = [self.build_prompt(self.profile_at(i))
                   for i in range(self._n)]
        outputs = self._generate(prompts)
        self._last_raw = list(outputs)
        vals = []
        bad = []
        for i, o in enumerate(outputs):
            v, ok = _strict_parse(o)
            if not ok:
                bad.append((i, o[:60]))
            vals.append(v)
        self._last_parse_fail = len(bad) / max(1, len(outputs))
        if bad:
            raise ParseFailure(
                f"{len(bad)} of {len(outputs)} responses were unparseable "
                f"and the 0.5 fallback is NEVER served here. First few: "
                f"{bad[:3]}")
        return torch.tensor(vals, dtype=torch.float32,
                            device=x.device).unsqueeze(-1)

    # ---- parsing ---------------------------------------------------------
    def parse(self, text: str) -> float:
        v, ok = _strict_parse(text)
        if not ok:
            raise ParseFailure(
                f"unparseable response {text[:80]!r}: the OpenRouter backend "
                f"never serves the 0.5 fallback.")
        return v

    def parse_ok(self, text: str) -> bool:
        return _strict_parse(text)[1]

    # ---- provenance ------------------------------------------------------
    def effective_generation_policy(self) -> dict:
        """Everything pinned, plus an explicit statement of what temperature
        0 does and does not buy. Recorded in the run config."""
        return {
            "backend": {"value": "openrouter", "source": "pinned"},
            "model_slug": {"value": self._model_slug, "source": "pinned"},
            "provider": {"value": self._provider.to_body(),
                         "source": "pinned"},
            "temperature": {
                "value": self._policy.temperature,
                "source": ("pinned" if self._policy.temperature is not None
                           else "omitted_unsupported_by_endpoint")},
            "top_p": {"value": (self._policy.top_p
                                if self._policy.temperature is not None
                                else None),
                      "source": ("pinned" if self._policy.temperature is not None
                                 else "omitted_with_temperature")},
            "run_seed": {"value": self._run_seed, "source": "population"},
            "expected_canonical_slug": {"value": self._expected_canonical,
                                        "source": "pinned"},
            "max_tokens": {"value": self._policy.max_tokens,
                           "source": "pinned"},
            "seed": {"value": self._policy.seed,
                     "source": "pinned" if self._policy.seed is not None
                     else "unsupported_by_endpoint"},
            "reasoning_mode": {"value": self._policy.reasoning_mode,
                               "source": "pinned"},
            "determinism_caveat": (
                "temperature=0 removes SAMPLING noise only. It does not pin "
                "the provider's kernels, batching, hardware or serving "
                "stack, and most frontier endpoints expose no seed. Repeat "
                "runs may differ; this is recorded, not claimed away."),
        }

    def provenance_records(self) -> list[dict]:
        return [p.to_dict() for p in self._last_provenance]


# Chat-template renders are recognisable by their control tokens. This is
# a guard against a refactor silently reconnecting the HF prompt path to
# the API backend, which would change the prompt without changing a number
# anyone looks at.
_TEMPLATE_MARKERS = (
    "<|im_start|>", "<|im_end|>", "<|start_header_id|>", "<|end_header_id|>",
    "<|eot_id|>", "<start_of_turn>", "<end_of_turn>", "[INST]", "[/INST]",
    "<|begin_of_text|>", "<think>",
)


def _looks_like_chat_template(text: str) -> bool:
    return any(m in (text or "") for m in _TEMPLATE_MARKERS)
