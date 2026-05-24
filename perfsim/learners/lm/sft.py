"""SFTLearner: supervised fine-tuning via TRL's SFTTrainer.

Consumes data dicts of the form ``{"x": ..., "y": (k, 1), "agent_idx": (k,)}``
produced by `FJWorld` (after `Simulator.train_mask` filters to the labeled
subset). Looks up each training row's profile from the wrapped
`HFCausalLMModel.profiles`, builds the SFT example as
``prompt + formatted_target``, and runs ``SFTTrainer.train()`` once per
``learner.train(data)`` call.

TRL and the HF datasets package are imported lazily so the rest of perfsim
stays usable without the `[lm]` extra installed.

KL-regularized SFT lives in a sibling file (`kl_sft.py`) and subclasses
this one.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch

from perfsim.core.learner import Learner
from perfsim.core.loss import Loss
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.models.hf_causal_lm import HFCausalLMModel

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset


def _default_target_formatter(value: float) -> str:
    """Format an opinion in [0, 1] as a 2-decimal string, e.g. 0.42 -> '0.42'."""
    return f"{float(value):.2f}"


class SFTLearner(Learner):
    """SFT over an HFCausalLMModel via TRL's SFTTrainer.

    Args:
        model:               The HFCausalLMModel being trained.
        loss:                A perfsim Loss instance. Not used by SFTTrainer
                             directly (the LM's own CE loss is the SFT
                             objective), but the field is kept on Learner for
                             API consistency and for metric reuse.
        target_formatter:    Callable mapping a per-agent opinion (float) to
                             the target text string. Default: 2-decimal
                             representation.
        max_steps:           SFTTrainer.max_steps for each train() call.
        learning_rate:       SFTTrainer learning rate.
        per_device_batch_size: TRL batch size knob.
        max_seq_length:      SFTConfig max_seq_length.
        output_dir:          Where TRL writes its checkpoints. Default: a
                             fresh temp dir per learner (kept across calls).
        trainer_kwargs:      Extra kwargs forwarded to SFTConfig.
    """

    accepted_schemas: ClassVar[tuple[DataSchema, ...]] = (SUPERVISED_SCHEMA,)

    def __init__(
        self,
        model: HFCausalLMModel,
        loss: Loss,
        *,
        target_formatter: Callable[[float], str] = _default_target_formatter,
        max_steps: int = 50,
        learning_rate: float = 1e-5,
        per_device_batch_size: int = 4,
        max_seq_length: int = 512,
        output_dir: str | None = None,
        response_template: str | None = None,
        trainer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, HFCausalLMModel):
            raise TypeError(
                f"SFTLearner expects an HFCausalLMModel; got {type(model).__name__}"
            )
        super().__init__(model, loss)
        self._target_formatter = target_formatter
        self._max_steps = int(max_steps)
        self._learning_rate = float(learning_rate)
        self._per_device_batch_size = int(per_device_batch_size)
        self._max_seq_length = int(max_seq_length)
        self._output_dir = output_dir or tempfile.mkdtemp(prefix="perfsim-sft-")
        self._response_template = response_template
        self._trainer_kwargs = trainer_kwargs or {}

    @property
    def model(self) -> HFCausalLMModel:  # type: ignore[override]
        return self._model  # type: ignore[return-value]

    @model.setter
    def model(self, value: HFCausalLMModel) -> None:
        self._model = value

    # ---- Build SFT dataset from one round's filtered data ----------------

    def _build_dataset(self, data: Data) -> "HFDataset":
        from datasets import Dataset as HFDataset

        if "agent_idx" not in data:
            raise KeyError(
                "SFTLearner.train requires data['agent_idx']; the env (e.g. FJWorld) "
                "must emit per-agent indices alongside x and y. Did you wire "
                "the new FJWorld API?"
            )
        y = data["y"]
        if y.ndim > 1:
            y = y.squeeze(-1)
        idx = data["agent_idx"]
        if idx.shape[0] != y.shape[0]:
            raise ValueError(
                f"agent_idx length {idx.shape[0]} does not match y length {y.shape[0]}"
            )

        self.model.ensure_loaded()
        examples: list[dict[str, str]] = []
        # If response_template is set, emit `{prompt, completion}` columns.
        # Modern TRL (>=0.11) auto-masks the prompt in this format so the
        # SFT loss is only on the completion tokens, regardless of whether
        # DataCollatorForCompletionOnlyLM is importable.
        use_prompt_completion = self._response_template is not None
        for i in range(idx.shape[0]):
            profile = self.model.profile_at(int(idx[i].item()))
            prompt = self.model.build_prompt(profile)
            target = self._target_formatter(float(y[i].item()))
            if use_prompt_completion:
                examples.append({"prompt": prompt, "completion": target})
            else:
                examples.append({"text": prompt + target})
        return HFDataset.from_list(examples)

    # ---- Train ----------------------------------------------------------

    def train(self, data: Data) -> None:
        import inspect

        from trl import SFTConfig, SFTTrainer

        ds = self._build_dataset(data)

        cfg_kwargs: dict[str, Any] = dict(
            output_dir=self._output_dir,
            max_steps=self._max_steps,
            per_device_train_batch_size=self._per_device_batch_size,
            learning_rate=self._learning_rate,
            report_to="none",
            save_strategy="no",
            logging_strategy="no",
        )
        # TRL renamed `max_seq_length` -> `max_length` between releases.
        _sig = inspect.signature(SFTConfig.__init__).parameters
        if "max_seq_length" in _sig:
            cfg_kwargs["max_seq_length"] = self._max_seq_length
        elif "max_length" in _sig:
            cfg_kwargs["max_length"] = self._max_seq_length
        # If response_template is set and SFTConfig exposes the flag,
        # explicitly enable completion-only loss. Defense in depth on top
        # of the {prompt, completion} dataset format.
        if self._response_template is not None and "completion_only_loss" in _sig:
            cfg_kwargs.setdefault("completion_only_loss", True)
            print(
                "[SFTLearner] enabled SFTConfig(completion_only_loss=True)",
                flush=True,
            )
        cfg_kwargs.update(self._trainer_kwargs)
        cfg = SFTConfig(**cfg_kwargs)

        trainer = self._build_trainer(cfg=cfg, ds=ds)
        trainer.train()

    def _completion_only_collator(self) -> Any | None:
        """Return a DataCollatorForCompletionOnlyLM if available; else None.

        Belt-and-suspenders backup for prompt masking. The primary mechanism
        is the `{prompt, completion}` dataset format (handled in
        `_build_dataset`). This collator is used in addition when the TRL
        version exposes it, to be defensive about TRL versions whose
        prompt-completion handling is incomplete.

        Tries multiple import paths because newer TRL releases moved the
        class out of the top-level namespace.
        """
        if self._response_template is None:
            return None
        cls = None
        for path in (
            "trl",
            "trl.trainer",
            "trl.trainer.utils",
            "trl.trainer.sft_trainer",
        ):
            try:
                mod = __import__(path, fromlist=["DataCollatorForCompletionOnlyLM"])
                cand = getattr(mod, "DataCollatorForCompletionOnlyLM", None)
                if cand is not None:
                    cls = cand
                    break
            except ImportError:
                continue
        if cls is None:
            print(
                "[SFTLearner] DataCollatorForCompletionOnlyLM not found in TRL; "
                "relying on {prompt, completion} dataset format for masking",
                flush=True,
            )
            return None
        return cls(
            response_template=self._response_template,
            tokenizer=self.model.tokenizer,
        )

    def _build_trainer(self, *, cfg: Any, ds: Any) -> Any:
        """Construct the underlying SFTTrainer. Subclassed by KLSFTLearner."""
        import inspect

        from trl import SFTTrainer

        _sig = inspect.signature(SFTTrainer.__init__).parameters
        tok_kwarg: dict[str, Any] = {}
        if "processing_class" in _sig:
            tok_kwarg["processing_class"] = self.model.tokenizer
        elif "tokenizer" in _sig:
            tok_kwarg["tokenizer"] = self.model.tokenizer

        collator = self._completion_only_collator()
        if collator is not None and "data_collator" in _sig:
            tok_kwarg["data_collator"] = collator

        return SFTTrainer(
            model=self.model.inner_model,
            args=cfg,
            train_dataset=ds,
            **tok_kwarg,
        )

    def reset(self) -> None:
        """No-op: LM training state persists across rounds by design."""
        return None
