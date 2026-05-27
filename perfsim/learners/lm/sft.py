"""SFTLearner: supervised fine-tuning via TRL's SFTTrainer."""

from __future__ import annotations

import inspect
import os
import tempfile
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch

from perfsim.core.learner import Learner
from perfsim.core.loss import Loss
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.models.hf_causal_lm import HFCausalLMModel

try:
    from datasets import Dataset as HFDataset
except ImportError:
    HFDataset = None  # type: ignore[assignment,misc]

try:
    from trl import SFTConfig, SFTTrainer
except ImportError:
    SFTConfig = None  # type: ignore[assignment,misc]
    SFTTrainer = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from datasets import Dataset as _HFDataset


def _default_target_formatter(value: float) -> str:
    """Format an opinion in [0, 1] as a 2-decimal string, e.g. 0.42 -> '0.42'."""
    return f"{float(value):.2f}"


class SFTLearner(Learner):
    """SFT over an HFCausalLMModel via TRL's SFTTrainer."""

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
        self._sanity_printed = False

    @property
    def model(self) -> HFCausalLMModel:  # type: ignore[override]
        return self._model  # type: ignore[return-value]

    @model.setter
    def model(self, value: HFCausalLMModel) -> None:
        self._model = value

    def _build_dataset(self, data: Data) -> "HFDataset":
        if HFDataset is None:
            raise ImportError(
                "SFTLearner requires the 'datasets' package. "
                "Install with: pip install 'perfsim[lm]'"
            )
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

    def train(self, data: Data) -> None:
        if SFTConfig is None or SFTTrainer is None:
            raise ImportError(
                "SFTLearner requires the 'trl' package. "
                "Install with: pip install 'perfsim[lm]'"
            )
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
        _sig = inspect.signature(SFTConfig.__init__).parameters
        if "max_seq_length" in _sig:
            cfg_kwargs["max_seq_length"] = self._max_seq_length
        elif "max_length" in _sig:
            cfg_kwargs["max_length"] = self._max_seq_length
        if self._response_template is not None and "completion_only_loss" in _sig:
            cfg_kwargs.setdefault("completion_only_loss", True)
            print(
                "[SFTLearner] enabled SFTConfig(completion_only_loss=True)",
                flush=True,
            )
        cfg_kwargs.update(self._trainer_kwargs)
        cfg = SFTConfig(**cfg_kwargs)

        trainer = self._build_trainer(cfg=cfg, ds=ds)
        self._maybe_print_sanity(trainer)
        trainer.train()

    def _maybe_print_sanity(self, trainer: Any) -> None:
        """If SFT_SANITY=1, dump one training batch's label-masking summary."""
        if self._sanity_printed:
            return
        if os.environ.get("SFT_SANITY", "0") != "1":
            return
        try:
            batch = next(iter(trainer.get_train_dataloader()))
        except Exception as e:
            print(f"[sanity] dataloader peek failed: {e}", flush=True)
            self._sanity_printed = True
            return
        try:
            lb = batch["labels"][0]
            ids = batch["input_ids"][0]
            n_masked = int((lb == -100).sum().item())
            n_total = int(lb.numel())
            pct = 100.0 * n_masked / max(n_total, 1)
            print(
                f"[sanity] labels shape={tuple(lb.shape)}  "
                f"masked={n_masked}/{n_total} ({pct:.0f}%)",
                flush=True,
            )
            print(f"[sanity] last 12 input_ids: {ids[-12:].tolist()}", flush=True)
            print(f"[sanity] last 12 labels   : {lb[-12:].tolist()}", flush=True)
        except Exception as e:
            print(f"[sanity] label inspection failed: {e}", flush=True)
        finally:
            self._sanity_printed = True

    def _completion_only_collator(self) -> Any | None:
        """Return a DataCollatorForCompletionOnlyLM if available; else None."""
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
