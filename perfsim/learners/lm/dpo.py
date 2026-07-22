"""DPOLearner: closed-loop RLHF via TRL's DPOTrainer.

One round = one iterative-DPO update on ON-POLICY preferences:
  1. For each agent, TEMPERATURE-SAMPLE two candidate opinion predictions from the
     current policy (greedy would give identical candidates -> no preference pair).
  2. Score each candidate by U(y; x_judge) = -|parse(y) - x_judge|, where x_judge is
     the judging population's opinion for that agent. The pipeline supplies x_judge:
       closed arm -> the model's OWN (deployment-shifted) population
       open   arm -> the synchronized no-AI twin
     This is the ONLY thing that differs between the closed and open arms.
  3. Stochastic Bradley-Terry preference label with temperature tau
     (tau -> inf recovers deterministic closer-wins), build {prompt, chosen, rejected}.
  4. DPO update. Reference policy = the LoRA base with the adapter DISABLED
     (ref_model=None on a PEFT model) => a memory-cheap FIXED anchor to the base
     model. (Moving anchor would need a second 7B in memory; left as an ablation.)

Reads DPO knobs from env: DPO_BETA, DPO_TAU, DPO_GEN_TEMP, DPO_MAX_STEPS, DPO_N_PAIRS.
Requires the pipeline to pass data = {"agent_idx": LongTensor, "x_judge": FloatTensor}
and (for the cheap fixed-anchor path) USE_LORA=1 so inner_model is a PEFT model.
"""

from __future__ import annotations

import inspect
import math
import os
import re
import tempfile
from typing import Any, Callable, ClassVar

import torch

from perfsim.core.learner import Learner
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.learners.lm.sft import _default_target_formatter
from perfsim.models.hf_causal_lm import HFCausalLMModel

try:
    from datasets import Dataset as HFDataset
except ImportError:
    HFDataset = None  # type: ignore[assignment,misc]

# trl 1.2.0's DPOTrainer import chain does `from torch.distributed.fsdp import
# FSDPModule`, which does not exist in torch 2.5.1 (the cluster's opdyn/opdyn_gemma
# envs). We use LoRA on a single GPU and never FSDP, so inject a dummy symbol so the
# import succeeds; it is never instantiated (prepare_fsdp is only called under an
# FSDP accelerate plugin, which we do not use).
try:  # pragma: no cover - environment shim
    import torch.distributed.fsdp as _fsdp
    if not hasattr(_fsdp, "FSDPModule"):
        class FSDPModule:  # noqa: N801
            pass
        _fsdp.FSDPModule = FSDPModule
except Exception:
    pass

try:
    from trl import DPOConfig, DPOTrainer
except (ImportError, RuntimeError):
    DPOConfig = None  # type: ignore[assignment,misc]
    DPOTrainer = None  # type: ignore[assignment,misc]


class DPOLearner(Learner):
    """Iterative closed-loop DPO over an HFCausalLMModel via TRL's DPOTrainer."""

    accepted_schemas: ClassVar[tuple[DataSchema, ...]] = (SUPERVISED_SCHEMA,)

    def __init__(
        self,
        model: HFCausalLMModel,
        loss: Any,
        *,
        target_formatter: Callable[[float], str] = _default_target_formatter,
        max_steps: int = 8,
        learning_rate: float = 5e-5,
        per_device_batch_size: int = 4,
        max_seq_length: int = 640,
        output_dir: str | None = None,
        response_template: str | None = None,   # accepted for kwargs-compat; unused
        trainer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, HFCausalLMModel):
            raise TypeError(f"DPOLearner expects an HFCausalLMModel; got {type(model).__name__}")
        super().__init__(model, loss)
        self._target_formatter = target_formatter
        self._learning_rate = float(learning_rate)
        self._per_device_batch_size = int(per_device_batch_size)
        self._max_seq_length = int(max_seq_length)
        self._output_dir = output_dir or tempfile.mkdtemp(prefix="perfsim-dpo-")
        self._trainer_kwargs = trainer_kwargs or {}
        # DPO / preference knobs (env-overridable; logged once)
        self._beta = float(os.environ.get("DPO_BETA", "0.1"))
        self._tau = float(os.environ.get("DPO_TAU", "12.0"))          # BT judge sharpness
        self._gen_temp = float(os.environ.get("DPO_GEN_TEMP", "0.8"))  # candidate sampling temp
        self._max_steps = int(os.environ.get("DPO_MAX_STEPS", str(max_steps)))
        self._n_pairs = int(os.environ.get("DPO_N_PAIRS", "0"))        # 0 = all agents
        self._printed = False

    # --------------------------------------------------------------- candidates
    def _sample_two(self, prompts: list[str]) -> tuple[list[str], list[str]]:
        """Two independent temperature samples per prompt from the current policy."""
        m = self.model
        old_ds, old_t = m._do_sample, m._temperature  # type: ignore[attr-defined]
        m._do_sample, m._temperature = True, self._gen_temp  # type: ignore[attr-defined]
        try:
            a = m._generate(prompts)  # type: ignore[attr-defined]
            b = m._generate(prompts)  # type: ignore[attr-defined]
        finally:
            m._do_sample, m._temperature = old_ds, old_t  # type: ignore[attr-defined]
        return a, b

    def _build_pref_dataset(self, data: Data) -> "HFDataset":
        if HFDataset is None:
            raise ImportError("DPOLearner requires 'datasets'. pip install 'perfsim[lm]'")
        if "agent_idx" not in data or "x_judge" not in data:
            raise KeyError("DPOLearner.train requires data['agent_idx'] and data['x_judge']")
        self.model.ensure_loaded()
        idx = data["agent_idx"].long().view(-1)
        xj = data["x_judge"].float().view(-1)
        if self._n_pairs and self._n_pairs < idx.shape[0]:
            sel = torch.randperm(idx.shape[0])[: self._n_pairs]
            idx, xj = idx[sel], xj[sel]

        prompts = [self.model.build_prompt(self.model.profile_at(int(i))) for i in idx]
        candA, candB = self._sample_two(prompts)

        rows: list[dict[str, str]] = []
        n_tie = n_fail = 0
        for i in range(len(prompts)):
            ta, tb = candA[i].strip(), candB[i].strip()
            if re.search(r"\d", ta) is None or re.search(r"\d", tb) is None:
                n_fail += 1          # non-numeric generation -> no usable candidate
                continue
            ya, yb = self.model._parse(ta), self.model._parse(tb)  # type: ignore[attr-defined]
            if abs(ya - yb) < 0.01:          # same predicted number -> no preference signal
                n_tie += 1
                continue
            xji = float(xj[i])
            ua, ub = -abs(ya - xji), -abs(yb - xji)
            p_a = 1.0 / (1.0 + math.exp(-self._tau * (ua - ub)))   # P(A preferred)
            a_wins = torch.rand(1).item() < p_a
            chosen, rejected = (ta, tb) if a_wins else (tb, ta)
            rows.append({"prompt": prompts[i], "chosen": chosen, "rejected": rejected})

        if not self._printed:
            print(f"[DPOLearner] beta={self._beta} tau={self._tau} gen_temp={self._gen_temp} "
                  f"max_steps={self._max_steps} | pairs={len(rows)} "
                  f"(ties={n_tie}, parse_fail={n_fail} of {len(prompts)})", flush=True)
            self._printed = True
        if not rows:
            raise RuntimeError("DPOLearner: 0 usable preference pairs (all ties/parse-fail). "
                               "Raise DPO_GEN_TEMP or check generation.")
        return HFDataset.from_list(rows)

    # --------------------------------------------------------------- train
    def train(self, data: Data) -> None:
        if DPOConfig is None or DPOTrainer is None:
            raise ImportError("DPOLearner requires 'trl'. pip install 'perfsim[lm]'")
        ds = self._build_pref_dataset(data)

        cfg_kwargs: dict[str, Any] = dict(
            output_dir=self._output_dir,
            beta=self._beta,
            max_steps=self._max_steps,
            per_device_train_batch_size=self._per_device_batch_size,
            learning_rate=self._learning_rate,
            max_length=self._max_seq_length,
            report_to="none",
            save_strategy="no",
            logging_strategy="no",
        )
        cfg_kwargs.update(self._trainer_kwargs)
        # keep only kwargs this trl's DPOConfig accepts (e.g. max_prompt_length is
        # absent in trl 1.2.0); mirrors SFTLearner's signature-guarded config build.
        _cfg_sig = inspect.signature(DPOConfig.__init__).parameters
        cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in _cfg_sig}
        cfg = DPOConfig(**cfg_kwargs)

        _sig = inspect.signature(DPOTrainer.__init__).parameters
        tok_kwarg: dict[str, Any] = {}
        if "processing_class" in _sig:
            tok_kwarg["processing_class"] = self.model.tokenizer
        elif "tokenizer" in _sig:
            tok_kwarg["tokenizer"] = self.model.tokenizer

        # ref_model=None + PEFT model => reference is the adapter-DISABLED base
        # (fixed anchor), no second model in memory.
        trainer = DPOTrainer(
            model=self.model.inner_model,
            ref_model=None,
            args=cfg,
            train_dataset=ds,
            **tok_kwarg,
        )
        trainer.train()

    def reset(self) -> None:
        """No-op: LM training state persists across rounds by design."""
        return None
