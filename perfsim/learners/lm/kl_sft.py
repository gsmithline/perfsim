"""KLSFTLearner: SFT with a KL anchor against a frozen reference policy."""

from __future__ import annotations

import copy
import inspect
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch
import torch.nn.functional as F

from perfsim.core.loss import Loss
from perfsim.learners.lm.sft import SFTLearner, _default_target_formatter
from perfsim.models.hf_causal_lm import HFCausalLMModel

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None  # type: ignore[assignment,misc]

try:
    from trl import SFTTrainer
except ImportError:
    SFTTrainer = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class KLSFTLearner(SFTLearner):
    """SFT with a beta * KL anchor against a frozen reference policy.

    anchor_mode "fixed" keeps the reference at ref_model_name for the whole run
    (re-anchored regime); "chained" re-freezes the reference to the just-trained
    policy after every train() call, so each round anchors to the previous round.
    """

    def __init__(
        self,
        model: HFCausalLMModel,
        loss: Loss,
        *,
        ref_model_name: str,
        kl_beta: float = 1.0,
        anchor_mode: str = "fixed",
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
        if anchor_mode not in ("fixed", "chained"):
            raise ValueError(f"anchor_mode must be 'fixed' or 'chained'; got {anchor_mode!r}")
        self._ref_model_name = ref_model_name
        self._kl_beta = float(kl_beta)
        self._anchor_mode = anchor_mode
        self._ref_model: "PreTrainedModel | None" = None

    @property
    def kl_beta(self) -> float:
        return self._kl_beta

    @property
    def anchor_mode(self) -> str:
        return self._anchor_mode

    def train(self, data: Any) -> None:
        super().train(data)
        if self._anchor_mode == "chained" and self._kl_beta > 0:
            self._chain_ref()

    def _chain_ref(self) -> None:
        """Freeze a copy of the current policy as next round's reference."""
        new_ref = copy.deepcopy(self.model.inner_model)
        new_ref.eval()
        for p in new_ref.parameters():
            p.requires_grad_(False)
        self._ref_model = new_ref

    def _ensure_ref(self) -> "PreTrainedModel":
        if self._ref_model is None:
            if AutoModelForCausalLM is None:
                raise ImportError(
                    "KLSFTLearner requires the 'transformers' package. "
                    "Install with: pip install 'perfsim[lm]'"
                )
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
        if SFTTrainer is None:
            raise ImportError(
                "KLSFTLearner requires the 'trl' package. "
                "Install with: pip install 'perfsim[lm]'"
            )
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
                # KL on the completion tokens only (labels != -100), with the
                # next-token shift, so beta is on the same scale as the
                # opinion-dynamics study (KL anchors the answer, not the prompt).
                labels = inputs.get("labels")
                if labels is not None:
                    mask = (labels != -100).float()
                else:
                    attn = inputs.get("attention_mask")
                    mask = attn.float() if attn is not None else torch.ones(
                        outputs.logits.shape[:2], device=outputs.logits.device
                    )
                logp = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
                logq = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
                mask_shift = mask[:, 1:]
                kl_per_token = (logp.exp() * (logp - logq)).sum(dim=-1)
                kl = (kl_per_token * mask_shift).sum() / mask_shift.sum().clamp_min(1.0)
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
