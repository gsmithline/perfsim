"""Binary token scorer for HAPPY / UNHAPPY next-token logits."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.models.hf_causal_lm import HFCausalLMModel


class BinaryLMScorer(Model):
    """Wraps an HFCausalLMModel to produce per-agent p(yes_token | prompt)."""

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
            # Most BPE tokenizers split mid-sentence labels with a leading
            # space; space_prefix=True matches that mid-sentence shape.
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
        """Return per-agent P(yes_token | prompt) as (N, 1)."""
        n = x.shape[0]
        self._lm.ensure_loaded()
        prompts = [self._lm.build_prompt(self._lm.profile_at(i)) for i in range(n)]
        probs = self.score_binary(prompts)
        return probs.unsqueeze(-1).to(x.device)

    def score_binary(self, prompts: Sequence[str]) -> Tensor:
        self._lm.ensure_loaded()
        yes_id, no_id = self._resolve_token_ids()
        inner = self._lm.inner_model
        tok = self._lm.tokenizer
        assert inner is not None and tok is not None

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
                # Left-padding (set by HFCausalLMModel) puts the real final
                # token at the last column, so logits[:, -1, :] is correct.
                last_logits = out.logits[:, -1, :]
                yes_no = last_logits[:, [yes_id, no_id]]
                p_yes = torch.softmax(yes_no.float(), dim=-1)[:, 0]
                out_chunks.append(p_yes.cpu())
        finally:
            if was_grad_ckpt:
                inner.gradient_checkpointing_enable()
            inner.config.use_cache = was_use_cache

        return torch.cat(out_chunks, dim=0)

    def get_params(self) -> Tensor:
        return self._lm.get_params()

    def set_params(self, theta: Tensor) -> None:
        self._lm.set_params(theta)

    @property
    def num_params(self) -> int:
        return self._lm.num_params
