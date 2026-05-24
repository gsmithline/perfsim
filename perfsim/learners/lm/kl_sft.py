"""KLSFTLearner: SFT with a KL anchor against a frozen reference policy.

Extends SFTLearner with the Korbak-Williams form (I) SFT-KL loss:

    L = L_CE(x, y) + beta * KL( pi_theta(.|x) || pi_ref(.|x) )

The KL is computed token-wise over the model's output distribution against
a frozen reference policy `pi_ref`. With `beta = 0` this reduces to plain
SFT; large `beta` anchors the fine-tuned model to the reference.

Mirrors the `KLSFTTrainer.compute_loss` pattern from
`Opinion-dynamics-post-training/llm_predictor.py:366`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch
import torch.nn.functional as F

from perfsim.core.loss import Loss
from perfsim.learners.lm.sft import SFTLearner, _default_target_formatter
from perfsim.models.hf_causal_lm import HFCausalLMModel

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class KLSFTLearner(SFTLearner):
    """SFT with a beta * KL anchor against a frozen reference policy.

    Args:
        model:           See SFTLearner.
        loss:            See SFTLearner.
        ref_model_name:  HF model ID for the reference policy. Loaded
                         frozen (eval mode, requires_grad=False). Typically
                         the same base name as `model._base_model_name` to
                         anchor against the un-fine-tuned starting point.
        kl_beta:         Coefficient on the KL term. 0 reduces to SFT.
        target_formatter, max_steps, learning_rate, per_device_batch_size,
        max_seq_length, output_dir, trainer_kwargs: as in SFTLearner.
    """

    def __init__(
        self,
        model: HFCausalLMModel,
        loss: Loss,
        *,
        ref_model_name: str,
        kl_beta: float = 1.0,
        target_formatter: Callable[[float], str] = _default_target_formatter,
        max_steps: int = 50,
        learning_rate: float = 1e-5,
        per_device_batch_size: int = 4,
        max_seq_length: int = 512,
        output_dir: str | None = None,
        response_template: str | None = None,
        trainer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model,
            loss,
            target_formatter=target_formatter,
            max_steps=max_steps,
            learning_rate=learning_rate,
            per_device_batch_size=per_device_batch_size,
            max_seq_length=max_seq_length,
            output_dir=output_dir,
            response_template=response_template,
            trainer_kwargs=trainer_kwargs,
        )
        self._ref_model_name = ref_model_name
        self._kl_beta = float(kl_beta)
        self._ref_model: "PreTrainedModel | None" = None

    @property
    def kl_beta(self) -> float:
        return self._kl_beta

    def _ensure_ref(self) -> "PreTrainedModel":
        if self._ref_model is None:
            from transformers import AutoModelForCausalLM

            ref = AutoModelForCausalLM.from_pretrained(
                self._ref_model_name,
                torch_dtype=self.model._target_dtype,  # type: ignore[attr-defined]
            ).to(self.model._target_device)  # type: ignore[attr-defined]
            ref.config.pad_token_id = self.model.tokenizer.pad_token_id  # type: ignore[union-attr]
            ref.eval()
            for p in ref.parameters():
                p.requires_grad_(False)
            self._ref_model = ref
        return self._ref_model

    def _build_trainer(self, *, cfg: Any, ds: Any) -> Any:
        import inspect

        from trl import SFTTrainer

        ref_model = self._ensure_ref() if self._kl_beta > 0 else None
        kl_beta = self._kl_beta

        class _KLSFTTrainer(SFTTrainer):
            def compute_loss(
                self,
                model,
                inputs,
                return_outputs: bool = False,
                num_items_in_batch: int | None = None,
            ):
                outputs = model(**inputs)
                ce = outputs.loss
                if ref_model is None or kl_beta == 0.0:
                    return (ce, outputs) if return_outputs else ce
                with torch.no_grad():
                    ref_logits = ref_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                    ).logits
                # Token-wise KL(pi_theta || pi_ref). Masked by attention if present.
                logp = F.log_softmax(outputs.logits, dim=-1)
                logq = F.log_softmax(ref_logits, dim=-1)
                p = logp.exp()
                kl_per_token = (p * (logp - logq)).sum(dim=-1)
                attn = inputs.get("attention_mask")
                if attn is not None:
                    denom = attn.sum().clamp_min(1)
                    kl = (kl_per_token * attn).sum() / denom
                else:
                    kl = kl_per_token.mean()
                total = ce + kl_beta * kl
                return (total, outputs) if return_outputs else total

        _sig = inspect.signature(SFTTrainer.__init__).parameters
        tok_kwarg: dict[str, Any] = {}
        if "processing_class" in _sig:
            tok_kwarg["processing_class"] = self.model.tokenizer
        elif "tokenizer" in _sig:
            tok_kwarg["tokenizer"] = self.model.tokenizer

        collator = self._completion_only_collator()
        if collator is not None and "data_collator" in _sig:
            tok_kwarg["data_collator"] = collator

        return _KLSFTTrainer(
            model=self.model.inner_model,
            args=cfg,
            train_dataset=ds,
            **tok_kwarg,
        )
