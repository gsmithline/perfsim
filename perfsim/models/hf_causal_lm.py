"""HFCausalLMModel: HuggingFace causal LM wrapped as a perfsim Model."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Sequence

import torch
from torch import Tensor

from perfsim.core.model import Model
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizerBase,  AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model



PromptBuilder = Callable[[object, "PreTrainedTokenizerBase"], str]


class HFCausalLMModel(Model):
    """HuggingFace causal LM wrapped as a perfsim Model.

    Args:
        base_model_name: HuggingFace model ID (e.g."Qwen/Qwen2.5-0.5B-Instruct").
        profiles: Row-aligned per-agent metadata, typically a
                         pandas.DataFrame of length N. The i-th row
                         is the profile for agent i.
        prompt_builder: Callable taking profile_row, tokenizer and
                         returning the full prompt string for generation.
        use_lora: Whether to wrap the base model with a LoRA adapter
                         via PEFT. Default True.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_target_modules: Module name patterns LoRA targets.
        device: torch device for the model. CPU by defaulnpass
                         cuda or mps if available.
        dtype: Model dtype. Defaults to float32 on CPU (bf16 on CPU is slow) 
                            use bf16 on GPU.
        max_new_tokens:  Generation budget per query.
        gen_batch_size:  Batched generation chunk size.
        load_now:        If False defer the HF/PEFT imports and model
                         download until ensure_loaded() is called.
                         Lets unit tests construct the wrapper without
                         pulling in transformers.
    """

    def __init__(
        self,
        base_model_name: str,
        profiles: object,
        prompt_builder: PromptBuilder,
        *,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: Sequence[str] = ("q_proj", "v_proj"),
        lora_dropout: float = 0.05,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_new_tokens: int = 8,
        gen_batch_size: int = 16,
        group_prompting: bool = False,
        load_now: bool = True,
    ) -> None:
        super().__init__()
        self._base_model_name = base_model_name
        self._profiles = profiles
        self._prompt_builder = prompt_builder
        self._use_lora = use_lora
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._lora_dropout = lora_dropout
        self._lora_target_modules = tuple(lora_target_modules)
        self._target_device = torch.device(device)
        self._target_dtype = dtype
        self._max_new_tokens = max_new_tokens
        self._gen_batch_size = gen_batch_size
        self._group_prompting = group_prompting

        length = getattr(profiles, "__len__", lambda: -1)()
        if length == -1:
            raise TypeError("profiles must have a defined len()")
        self._n = int(length)

        self.inner_model: "PreTrainedModel | None" = None
        self.tokenizer: "PreTrainedTokenizerBase | None" = None
        if load_now:
            self.ensure_loaded()

 
    def ensure_loaded(self) -> None:
        """Load tokenizer and HF model on demand. Idempotent."""
        if self.inner_model is not None: 
            return

        tok = AutoTokenizer.from_pretrained(self._base_model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left" 
        self.tokenizer = tok

        m = AutoModelForCausalLM.from_pretrained(
            self._base_model_name,
            torch_dtype=self._target_dtype,
        ).to(self._target_device)
        m.config.pad_token_id = tok.pad_token_id

        if self._use_lora:

            lora_cfg = LoraConfig(
                r=self._lora_r,
                lora_alpha=self._lora_alpha,
                target_modules=list(self._lora_target_modules),
                lora_dropout=self._lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            m = get_peft_model(m, lora_cfg)
        self.inner_model = m

    

    @property
    def profiles(self) -> object:
        return self._profiles

    def profile_at(self, idx: int) -> object:
        """Return the profile row for agent index `idx`.

        Works for pandas DataFrames, 
        lists/tuples, or any object exposing __getitem__.
        """
        if hasattr(self._profiles, "iloc"):
            return self._profiles.iloc[int(idx)]
        return self._profiles[int(idx)]

    def build_prompt(self, profile: object) -> str:
        """Convenience: pass a profile row through the prompt builder."""
        if self.tokenizer is None:
            raise RuntimeError("ensure_loaded() must be called before build_prompt")
        return self._prompt_builder(profile, self.tokenizer)

    # ---- Forward (predict for all N agents) ------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Generate per-agent predictions. Returns (N, 1) tensor in [0, 1]."""
        if x.shape[0] != self._n:
            raise ValueError(
                f"HFCausalLMModel.forward: x leading dim {x.shape[0]} does "
                f"not match profiles N={self._n}"
            )
        self.ensure_loaded()
        prompts = [self.build_prompt(self.profile_at(i)) for i in range(self._n)]

        if self._group_prompting:
            unique_prompts, inverse = self._deduplicate_prompts(prompts)
            print(
                f"[group_prompting] {len(prompts)} agents -> "
                f"{len(unique_prompts)} unique prompts",
                flush=True,
            )
            unique_outputs = self._generate(unique_prompts)
            outputs = [unique_outputs[idx] for idx in inverse]
        else:
            outputs = self._generate(prompts)

        values = torch.tensor(
            [self._parse(o) for o in outputs],
            dtype=torch.float32,
            device=x.device,
        ).unsqueeze(-1)
        return values

    @staticmethod
    def _deduplicate_prompts(prompts: list[str]) -> tuple[list[str], list[int]]:
        """Deduplicate prompts by exact string equality.

        Returns (unique_prompts, inverse_indices) where
        inverse_indices[i] is the index into unique_prompts for agent i.
        """
        seen: dict[str, int] = {}
        unique: list[str] = []
        inverse: list[int] = []
        for p in prompts:
            if p not in seen:
                seen[p] = len(unique)
                unique.append(p)
            inverse.append(seen[p])
        return unique, inverse

    def _generate(self, prompts: list[str]) -> list[str]:
        """Batched greedy generation.

        Toggles gradient checkpointing off and re-enables KV cache for the
        duration of generation. HF refuses to populate the KV cache when
        gradient checkpointing is active (the recompute-on-backward logic
        would invalidate cached keys/values), so leaving it on means every
        decode step re-runs the full prefix forward pass. Toggling around
        generation gets the ~5-10x KV-cache speedup back without affecting
        the training path. Original state is restored in `finally` so SFT
        steps after generation continue to use grad checkpointing for the
        memory savings.
        """
        assert self.inner_model is not None
        assert self.tokenizer is not None

        was_grad_ckpt = bool(getattr(self.inner_model, "is_gradient_checkpointing", False))
        was_use_cache = bool(getattr(self.inner_model.config, "use_cache", False))
        if was_grad_ckpt:
            self.inner_model.gradient_checkpointing_disable()
        self.inner_model.config.use_cache = True

        try:
            out: list[str] = []
            for i in range(0, len(prompts), self._gen_batch_size):
                batch = prompts[i : i + self._gen_batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self._target_device)
                with torch.no_grad():
                    gen = self.inner_model.generate(
                        **inputs,
                        max_new_tokens=self._max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                new_tokens = gen[:, inputs["input_ids"].shape[1] :]
                decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                out.extend(decoded)
        finally:
            if was_grad_ckpt:
                self.inner_model.gradient_checkpointing_enable()
            self.inner_model.config.use_cache = was_use_cache

        return out

    @staticmethod
    def _parse(text: str, default: float = 0.5) -> float:
        """Extract the first numeric value from generated text, clipped to [0, 1]."""
        m = re.search(r"\d+\.?\d*", text)
        if m is None:
            return default
        try:
            v = float(m.group())
        except ValueError:
            return default
        return max(0.0, min(1.0, v))

 
    def get_params(self) -> Tensor:
        """Return a 1-element tensor: L2 norm of trainable parameters.

        The flat-tensor view of an LM is not meaningful (billions of
        params, and they live across HF/PEFT/nn.Linear layers in a
        non-canonical order). The Simulator only uses get_params for
        history bookkeeping and stability_gap a scalar diagnostic is the
        honest substitute.
        """
        if self.inner_model is None:
            return torch.zeros(1)
        with torch.no_grad():
            sq = torch.zeros(1, device="cpu")
            for p in self.inner_model.parameters():
                if p.requires_grad:
                    sq = sq + p.detach().pow(2).sum().cpu().reshape(1)
            return sq.sqrt()

    def set_params(self, theta: Tensor) -> None:
        raise NotImplementedError(
            "HFCausalLMModel does not support set_params; use HF checkpointing "
            "(save_pretrained / from_pretrained) for parameter snapshots."
        )

    def clone(self) -> "HFCausalLMModel":
        raise NotImplementedError(
            "HFCausalLMModel does not support clone; instantiate a fresh "
            "HFCausalLMModel from a saved checkpoint instead."
        )

    @property
    def num_params(self) -> int:
        if self.inner_model is None:
            return 0
        return sum(p.numel() for p in self.inner_model.parameters() if p.requires_grad)
