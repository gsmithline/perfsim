"""Binary token scorer that wraps an HFCausalLMModel.

Why this exists:
  `HFCausalLMModel.forward` does free-form generation and parses a float
  out of the decoded text. For Schelling we need a calibrated probability
  over the two tokens HAPPY / UNHAPPY -- generation + regex parsing
  collapses to the same constant (~0.5 or whatever the model's blanket
  prior is) regardless of the prompt context. Reading logits at the next
  position over those two specific token ids fixes that.

  Concretely, for each prompt we:
    1. Tokenize and run a single forward pass to get logits at the
       last-token position (no generation; the LM is acting as a
       discriminator).
    2. Extract logits at the yes_token_id and no_token_id columns.
    3. Softmax just those two and return p_yes = softmax(...)[0].

  Returns one float in [0, 1] per prompt. The Schelling env reads that
  vector as the per-agent `p_pred`.

Token id resolution:
  Some tokenizers prepend a space to "HAPPY" (' HAPPY') depending on
  context; we resolve the id by tokenizing the exact string the prompt
  expects to be completed with. If the tokenizer splits "HAPPY" into
  multiple subwords, we use the FIRST subword's id with a warning. That
  is the right thing for a one-token discriminator: the LM's
  next-position logit distribution is over the first subword of the
  continuation regardless of how many subwords the full label takes.

Trainability:
  The scorer's `__call__` returns a (N, 1) tensor matching perfsim's
  Model.forward contract, so it can be plugged directly into the
  `AgentTorchEnvironment.run` model slot. For LM training, the user
  passes the underlying HFCausalLMModel to KLSFTLearner with a
  target_formatter that emits "HAPPY" or "UNHAPPY" -- the trainer
  consumes the existing prompt builder + completion target. The scorer
  itself is not directly trained; it's a measurement device wrapping
  the same `inner_model`.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.models.hf_causal_lm import HFCausalLMModel


class BinaryLMScorer(Model):
    """Wraps an HFCausalLMModel to produce per-agent p(yes_token | prompt).

    Args:
        lm:        The wrapped HFCausalLMModel. Must already have a
                   prompt_builder and a profiles object so it can
                   build per-agent prompts.
        yes_token: The token string the LM is supposed to emit for
                   "yes" (e.g. "HAPPY"). Returns its softmax probability.
        no_token:  The complement (e.g. "UNHAPPY").
        batch_size: Forward-pass batch size. Set to ~16 on CPU, higher
                   on GPU.
        space_prefix: If True (default), resolve token ids for
                   ' HAPPY' / ' UNHAPPY' (with leading space). Most BPE
                   tokenizers split "HAPPY" mid-sentence as a
                   space-prefixed token; this flag matches that.
    """

    def __init__(
        self,
        lm: HFCausalLMModel,
        *,
        yes_token: str = "HAPPY",
        no_token: str = "UNHAPPY",
        batch_size: int = 16,
        space_prefix: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(lm, HFCausalLMModel):
            raise TypeError(
                f"BinaryLMScorer expects an HFCausalLMModel; got {type(lm).__name__}"
            )
        self._lm = lm
        self._yes_token = str(yes_token)
        self._no_token = str(no_token)
        self._batch_size = int(batch_size)
        self._space_prefix = bool(space_prefix)
        self._yes_id: int | None = None
        self._no_id: int | None = None

    @property
    def lm(self) -> HFCausalLMModel:
        return self._lm

    def _resolve_token_ids(self) -> tuple[int, int]:
        if self._yes_id is not None and self._no_id is not None:
            return self._yes_id, self._no_id
        self._lm.ensure_loaded()
        tok = self._lm.tokenizer
        assert tok is not None

        def first_id(label: str) -> int:
            s = f" {label}" if self._space_prefix else label
            ids = tok(s, add_special_tokens=False)["input_ids"]
            if len(ids) == 0:
                raise ValueError(f"Tokenizer produced no tokens for {s!r}")
            if len(ids) > 1:
                print(
                    f"[BinaryLMScorer] WARNING: {s!r} tokenized into "
                    f"{len(ids)} subwords {ids}; using first id {ids[0]} "
                    f"({tok.decode([ids[0]])!r})",
                    flush=True,
                )
            return int(ids[0])

        self._yes_id = first_id(self._yes_token)
        self._no_id = first_id(self._no_token)
        return self._yes_id, self._no_id

    def forward(self, x: Tensor) -> Tensor:
        """Return per-agent P(yes_token | prompt) as an (N, 1) tensor.

        `x` is ignored except for shape; per-agent prompts come from the
        wrapped LM's `prompt_builder(profile)`. Profiles are the
        canonical per-agent feature store on HFCausalLMModel.
        """
        n = x.shape[0]
        self._lm.ensure_loaded()
        prompts = [self._lm.build_prompt(self._lm.profile_at(i)) for i in range(n)]
        probs = self.score_binary(prompts)
        return probs.unsqueeze(-1).to(x.device)

    def score_binary(self, prompts: Sequence[str]) -> Tensor:
        """Core API: forward N prompts, return (N,) P(yes_token) floats."""
        self._lm.ensure_loaded()
        yes_id, no_id = self._resolve_token_ids()
        inner = self._lm.inner_model
        tok = self._lm.tokenizer
        assert inner is not None and tok is not None

        # Toggle off gradient checkpointing for the forward; we're not
        # training the scorer here, just reading logits. Mirrors the
        # gen path in HFCausalLMModel._generate.
        was_grad_ckpt = bool(getattr(inner, "is_gradient_checkpointing", False))
        was_use_cache = bool(getattr(inner.config, "use_cache", False))
        if was_grad_ckpt:
            inner.gradient_checkpointing_disable()
        inner.config.use_cache = False

        try:
            out_chunks: list[Tensor] = []
            for i in range(0, len(prompts), self._batch_size):
                batch = list(prompts[i : i + self._batch_size])
                enc = tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self._lm._target_device)  # type: ignore[attr-defined]
                with torch.no_grad():
                    out = inner(**enc)
                # `out.logits` is (B, T, V). We want the logit at the
                # LAST non-pad position for each row. With left-padding
                # (which HFCausalLMModel sets) the last column is the
                # final real token, so logits[:, -1, :] is correct.
                last_logits = out.logits[:, -1, :]
                yes_no = last_logits[:, [yes_id, no_id]]
                p_yes = torch.softmax(yes_no.float(), dim=-1)[:, 0]
                out_chunks.append(p_yes.cpu())
        finally:
            if was_grad_ckpt:
                inner.gradient_checkpointing_enable()
            inner.config.use_cache = was_use_cache

        return torch.cat(out_chunks, dim=0)

    # Pass-through trainable-param interface to the underlying LM so the
    # Simulator's history bookkeeping continues to work.
    def get_params(self) -> Tensor:
        return self._lm.get_params()

    def set_params(self, theta: Tensor) -> None:
        self._lm.set_params(theta)

    @property
    def num_params(self) -> int:
        return self._lm.num_params
