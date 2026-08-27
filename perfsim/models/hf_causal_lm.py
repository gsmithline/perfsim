"""HFCausalLMModel: HuggingFace causal LM wrapped as a perfsim Model.

Needs to be broken up this code is messy, it should just be hte model
"""

from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING, Callable, Sequence

import torch
from torch import Tensor

import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from perfsim.core.model import Model



PromptBuilder = Callable[[object, "PreTrainedTokenizerBase"], str]


class HFCausalLMModel(Model):
    """HuggingFace causal LM wrapped as a perfsim Model.

    profiles is row-aligned per-agent metadata (DataFrame, row i = agent i);
    prompt_builder(profile_row, tokenizer) -> prompt string. use_lora wraps the
    base model with a PEFT LoRA adapter. load_now=False defers the model
    download until ensure_loaded(), so tests can construct without transformers.
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
        do_sample: bool = False,
        temperature: float = 1.0,
        # SAMPLING POLICY. None = INHERIT the checkpoint's
        # generation_config -- the historical behaviour of every archived
        # run, preserved exactly. A value PINS that knob, so a wave can
        # state "T=1 from the model's own distribution" and have it be
        # true: Qwen checkpoints ship top_p/top_k defaults that would
        # otherwise truncate the sampled distribution silently, and
        # several ship repetition_penalty != 1, which perturbs even
        # greedy decoding.
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
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
        self._do_sample = do_sample
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._repetition_penalty = repetition_penalty
        self._group_prompting = group_prompting

        # diagnostics from the most recent forward(): the raw decoded strings and
        # the fraction with no parseable digit. Sampled generation can drift to
        # nonsense (no number), which _parse silently turns into the 0.5 default;
        # these surface that instead of hiding it as a fake collapse-to-0.5.
        self._last_raw: list[str] = []
        self._last_parse_fail: float = 0.0
        # PARSE_MODE (2026-08-25): "legacy" keeps _parse's first-digit-run
        # regex byte-for-byte for every archived run; "strict" (opt-in, set
        # by the runner from the env and recorded in the config) requires a
        # well-formed number in [0, 1] at the start of the generation,
        # accepts a leading-dot form (".64" -> 0.64), and counts anything
        # else as a parse failure instead of clamping it. Found on the
        # Section 3 cross-model wave: Mistral-7B's lambda=2 adapter emitted
        # ".64 (" which legacy parsed as 64 -> 1.0 with NO failure flagged.
        self.parse_mode: str = "legacy"

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

        if "gemma" in self._base_model_name.lower():
            # Gemma3's text model raises when a training forward has no
            # token_type_ids (it uses them to separate text from image
            # tokens). For text-only SFT they are all zeros, and TRL never
            # supplies them. Inject zeros matching each call's input shape,
            # but ONLY in training mode: Gemma3 requires them only when
            # training, and supplying them during generate() routes mask
            # building through or_mask_function, which needs torch>=2.6
            # (eval-mode generation works fine on torch 2.5.1 without them).
            _orig_forward = m.forward

            # functools.wraps keeps the original forward signature visible:
            # generate() inspects it to decide capabilities like
            # logits_to_keep, and an opaque (*args, **kwargs) wrapper makes
            # the model return full-sequence logits and crash in sampling.
            @functools.wraps(_orig_forward)
            def _forward_with_token_type(*args, **kwargs):
                if m.training and kwargs.get("token_type_ids") is None:
                    ref = kwargs.get("input_ids")
                    if ref is None and args and torch.is_tensor(args[0]):
                        ref = args[0]
                    if ref is None:
                        ref = kwargs.get("inputs_embeds")
                    if torch.is_tensor(ref):
                        kwargs["token_type_ids"] = torch.zeros(
                            ref.shape[:2], dtype=torch.long, device=ref.device
                        )
                return _orig_forward(*args, **kwargs)

            m.forward = _forward_with_token_type
        self.inner_model = m

    

    @property
    def profiles(self) -> object:
        return self._profiles

    def profile_at(self, idx: int) -> object:
        """Profile row for agent `idx` (DataFrame, list, or __getitem__ object)."""
        if hasattr(self._profiles, "iloc"):
            return self._profiles.iloc[int(idx)]
        return self._profiles[int(idx)]

    def build_prompt(self, profile: object) -> str:
        """Convenience: pass a profile row through the prompt builder."""
        if self.tokenizer is None:
            raise RuntimeError("ensure_loaded() must be called before build_prompt")
        return self._prompt_builder(profile, self.tokenizer)

    def forward(self, x: Tensor) -> Tensor: #peragent interactins
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

        self._last_raw = list(outputs)
        self._last_parse_fail = sum(
            1 for o in outputs if not self.parse_ok(o)
        ) / max(1, len(outputs))
        values = torch.tensor(
            [self.parse(o) for o in outputs],
            dtype=torch.float32,
            device=x.device,
        ).unsqueeze(-1)
        return values

    @torch.no_grad()
    def perplexity(self, texts: list[str]) -> float:
        """Mean token-level perplexity of `texts` under the current model.

        Standard model-collapse health metric. Scores each text independently
        (no padding) so left-padding for generation does not corrupt the NLL.
        """
        self.ensure_loaded()
        assert self.inner_model is not None and self.tokenizer is not None
        was_use_cache = bool(getattr(self.inner_model.config, "use_cache", False))
        self.inner_model.config.use_cache = False
        total_nll = 0.0
        total_tok = 0
        try:
            for text in texts:
                ids = self.tokenizer(
                    text, return_tensors="pt", truncation=True
                ).input_ids.to(self._target_device)
                if ids.shape[1] < 2:
                    continue
                out = self.inner_model(ids, labels=ids)
                n_tok = ids.shape[1] - 1
                total_nll += float(out.loss) * n_tok
                total_tok += n_tok
        finally:
            self.inner_model.config.use_cache = was_use_cache
        if total_tok == 0:
            return float("nan")
        return float(torch.tensor(total_nll / total_tok).exp())

    @torch.no_grad()
    def answer_distribution_stats(self) -> dict:
        """Entropy and top-1 prob of the answer-position next-token distribution,
        averaged over all agents.

        Model-collapse diagnostic: as the output distribution sharpens, entropy
        falls toward 0 and top-1 rises toward 1. This is the model's own
        distribution, decoupled from the argmax that drives the dynamics, so it
        measures "low-probability tokens disappear" at the model level. One
        forward over all agents (left-padding => logits[:, -1] is the next-token
        distribution after each full prompt).
        """
        self.ensure_loaded()
        assert self.inner_model is not None and self.tokenizer is not None
        prompts = [self.build_prompt(self.profile_at(i)) for i in range(self._n)]
        was_use_cache = bool(getattr(self.inner_model.config, "use_cache", False))
        self.inner_model.config.use_cache = False
        ent_sum = 0.0
        top1_sum = 0.0
        count = 0
        try:
            for i in range(0, len(prompts), self._gen_batch_size):
                batch = prompts[i : i + self._gen_batch_size]
                inputs = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True
                ).to(self._target_device)
                logits = self.inner_model(**inputs).logits[:, -1, :].float()
                logp = torch.log_softmax(logits, dim=-1)
                p = logp.exp()
                ent_sum += float((-(p * logp).sum(dim=-1)).sum())
                top1_sum += float(p.max(dim=-1).values.sum())
                count += logits.shape[0]
        finally:
            self.inner_model.config.use_cache = was_use_cache
        return {"answer_entropy": ent_sum / count, "answer_top1": top1_sum / count}

    @torch.no_grad()
    def answer_sample_stats(
        self, k: int, idx: list[int], temperature: float = 1.0
    ) -> tuple[dict, Tensor]:
        """K sampled draws of the parsed answer VALUE for the agents in `idx`.

        Value-level companion to answer_distribution_stats(), which scores only
        the first answer-token position and is trivially deterministic when
        every answer starts "0.". Sampling end-to-end measures the entropy that
        matters for the finite-sampling loop channel: how much the served
        NUMBER varies under redraws. Entropy is the empirical distribution's
        (natural log, max ln(k)); top1 is the modal draw's frequency.
        Returns (summary means, values[k, len(idx)]).
        """
        self.ensure_loaded()
        prompts = [self.build_prompt(self.profile_at(i)) for i in idx]
        draws = []
        for _ in range(k):
            outs = self._generate(prompts, do_sample=True, temperature=temperature)
            draws.append(torch.tensor([self.parse(o) for o in outs],
                                      dtype=torch.float32))
        vals = torch.stack(draws)                       # [k, n_sub]
        top1, ent = [], []
        for j in range(vals.shape[1]):
            counts = torch.unique(vals[:, j], return_counts=True)[1].float()
            p = counts / float(k)
            top1.append(float(p.max()))
            ent.append(float(-(p * p.log()).sum()))
        summary = {"ans_sample_top1": sum(top1) / len(top1),
                   "ans_sample_entropy": sum(ent) / len(ent),
                   "ans_sample_std": float(vals.std(dim=0).mean())}
        return summary, vals

    @staticmethod
    def _deduplicate_prompts(prompts: list[str]) -> tuple[list[str], list[int]]:
        """Deduplicate by exact equality; returns (unique, inverse_indices)."""
        seen: dict[str, int] = {}
        unique: list[str] = []
        inverse: list[int] = []
        for p in prompts:
            if p not in seen:
                seen[p] = len(unique)
                unique.append(p)
            inverse.append(seen[p])
        return unique, inverse

    def _generate(
        self,
        prompts: list[str],
        do_sample: bool | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        """Batched generation: greedy when do_sample is False, else sampled at temperature.

        do_sample/temperature default to the instance settings; pass explicit
        values to probe a different decoding mode (answer_sample_stats) without
        touching the serving configuration.

        Toggles grad checkpointing off + KV cache on for generation (HF won't
        populate the cache under checkpointing, costing ~5-10x), restoring both
        in `finally` so the training path keeps checkpointing's memory savings.

        EVALUATION MODE (2026-08-20). Generation also forces module.eval()
        and restores the previous train/eval flag in `finally`. Without it
        serving inherits whatever mode the last caller left behind, and
        HF's Trainer.train() leaves the model in TRAINING mode: with the
        default lora_dropout=0.05 that leaves dropout ACTIVE during
        generation, so "greedy" decoding silently becomes stochastic and
        two identical prompts in the same round can decode differently.
        The restore is unconditional -- an exception mid-generation must
        not strand the model in eval and silently disable dropout for the
        next training round.
        """
        assert self.inner_model is not None
        assert self.tokenizer is not None
        if do_sample is None:
            do_sample = self._do_sample
        if temperature is None:
            temperature = self._temperature

        was_grad_ckpt = bool(getattr(self.inner_model, "is_gradient_checkpointing", False))
        was_use_cache = bool(getattr(self.inner_model.config, "use_cache", False))
        was_training = bool(self.inner_model.training)
        if was_grad_ckpt:
            self.inner_model.gradient_checkpointing_disable()
        self.inner_model.config.use_cache = True
        # deterministic serving: no dropout, no batchnorm updates
        self.inner_model.eval()

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
                gen_kwargs = dict(
                    max_new_tokens=self._max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                    # top_p / top_k shape the SAMPLED distribution and are
                    # inert under greedy decoding, so they are sent only
                    # when sampling
                    if self._top_p is not None:
                        gen_kwargs["top_p"] = self._top_p
                    if self._top_k is not None:
                        gen_kwargs["top_k"] = self._top_k
                # repetition_penalty perturbs BOTH decoders
                if self._repetition_penalty is not None:
                    gen_kwargs["repetition_penalty"] = \
                        self._repetition_penalty
                with torch.no_grad():
                    gen = self.inner_model.generate(**inputs, **gen_kwargs)
                new_tokens = gen[:, inputs["input_ids"].shape[1] :]
                decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                out.extend(decoded)
        finally:
            if was_grad_ckpt:
                self.inner_model.gradient_checkpointing_enable()
            self.inner_model.config.use_cache = was_use_cache
            # restore the caller's mode EXACTLY, including on exceptions
            if was_training:
                self.inner_model.train()
            else:
                self.inner_model.eval()

        return out

    def effective_generation_policy(self) -> dict:
        """What generation ACTUALLY used, knob by knob: the pinned value
        when this wrapper pins it, otherwise the value inherited from the
        checkpoint's generation_config. Recorded by the runner so a run
        is auditable without re-loading the checkpoint."""
        gc = getattr(getattr(self, "inner_model", None),
                     "generation_config", None)

        def eff(pinned, name):
            if pinned is not None:
                return {"value": pinned, "source": "pinned"}
            return {"value": (getattr(gc, name, None) if gc is not None
                              else None),
                    "source": "checkpoint_default"}

        return {
            "do_sample": {"value": self._do_sample, "source": "pinned"},
            "temperature": {"value": self._temperature, "source": "pinned"},
            "top_p": eff(self._top_p, "top_p"),
            "top_k": eff(self._top_k, "top_k"),
            "repetition_penalty": eff(self._repetition_penalty,
                                      "repetition_penalty"),
        }

    @staticmethod
    def _parse(text: str, default: float = 0.5) -> float:
        """LEGACY: first digit run in the text, clipped to [0, 1].

        Never change this: every archived run was served through it. Note
        its failure modes -- ".64 (" -> 64 -> 1.0, "58 (58" -> 1.0 -- are
        NOT counted as failures (a failure is only a digit-free string).
        Use parse()/parse_ok() and PARSE_MODE=strict for new waves."""
        m = re.search(r"\d+\.?\d*", text)
        if m is None:
            return default
        try:
            v = float(m.group())
        except ValueError:
            return default
        return max(0.0, min(1.0, v))

    _STRICT_RE = re.compile(r"^\s*(\d*\.\d+|\d+(?:\.\d*)?)")

    @classmethod
    def _parse_strict(cls, text: str, default: float = 0.5):
        """STRICT: (value, ok). The generation must START with a well-formed
        number in [0, 1]; a leading-dot form (".64") is read as 0.64;
        trailing text after the number is ignored. Anything else -- a
        number above 1, a bare integer run like "58 (58", text before the
        number, no number -- is a parse FAILURE: ok=False and the default is
        served (finite, so the population update stays defined; the
        failure is counted in _last_parse_fail and logged per round)."""
        m = cls._STRICT_RE.match(text or "")
        if m is None:
            return default, False
        try:
            v = float(m.group(1))
        except ValueError:
            return default, False
        if not 0.0 <= v <= 1.0:
            return default, False
        return v, True

    # ---- PROSE parsing (2026-08-27) ----------------------------------
    # A number that is NOT part of a longer numeric literal.
    _STANDALONE_NUM = re.compile(r"(?<![\d.])(\d*\.\d+|\d+(?:\.\d*)?)"
                                 r"(?!\d)(?!\.\d)")
    # Scale descriptions echoed back from the prompt ("between 0 and 1",
    # "on a 0-1 scale"). Their 0 and 1 are NOT predictions, and leaving
    # them in would make almost every prose answer look ambiguous.
    _SCALE_PHRASE = re.compile(
        r"(?:between|from)?\s*0(?:\.0+)?\s*(?:and|to|-|–)\s*1(?:\.0+)?"
        r"(?:\s*scale)?", re.I)
    # An explicitly labelled value: "the estimate is 0.72", "answer: 0.8",
    # "predicted rating = 0.65". The LAST such match wins -- prose states
    # its conclusion at the end.
    _LABELLED = re.compile(
        r"(?:answer|estimate[ds]?|estimation|prediction|predicted|rating|"
        r"score|value|opinion)\b[^0-9\n]{0,24}?"
        r"(?<![\d.])(\d*\.\d+|\d+(?:\.\d*)?)(?!\d)(?!\.\d)", re.I)

    @classmethod
    def _parse_prose(cls, text, default: float = 0.5):
        """PROSE: (value, ok). Accepts a prediction stated inside prose,
        but ONLY when it is unambiguous.

        Order of decision:
          1. a well-formed number at the START -- identical to strict, so
             every already-well-formed generation parses to the same
             value it did before;
          2. else an explicitly LABELLED final value ("the estimate is
             0.72") -- the last such match;
          3. else, among standalone numbers in [0, 1] with scale
             descriptions removed, EXACTLY ONE distinct value;
          4. else FAILURE.
        Zero candidates and multiple distinct candidates are both
        failures: the default is never a prediction. Never takes "the
        first number anywhere" -- explanatory prose can mention unrelated
        numbers, and two different plausible values mean the generation
        did not state one answer."""
        t = text or ""
        m = cls._STRICT_RE.match(t)
        if m is not None:
            try:
                v = float(m.group(1))
            except ValueError:
                v = None
            if v is not None and 0.0 <= v <= 1.0:
                return v, True
            # a leading number OUT of range is a failure, as in strict:
            # falling through to prose recovery would let "58 (58" become
            # something else entirely
            return default, False
        lab = cls._LABELLED.findall(t)
        for cand in reversed(lab):
            try:
                v = float(cand)
            except ValueError:
                continue
            if 0.0 <= v <= 1.0:
                return v, True
        stripped = cls._SCALE_PHRASE.sub(" ", t)
        vals = []
        for cand in cls._STANDALONE_NUM.findall(stripped):
            try:
                v = float(cand)
            except ValueError:
                continue
            if 0.0 <= v <= 1.0:
                vals.append(v)
        uniq = sorted(set(vals))
        if len(uniq) == 1:
            return uniq[0], True
        return default, False

    def parse(self, text: str) -> float:
        if self.parse_mode == "prose":
            return self._parse_prose(text)[0]
        if self.parse_mode == "strict":
            return self._parse_strict(text)[0]
        return self._parse(text)

    def parse_ok(self, text: str) -> bool:
        if self.parse_mode == "prose":
            return self._parse_prose(text)[1]
        if self.parse_mode == "strict":
            return self._parse_strict(text)[1]
        return re.search(r"\d", text or "") is not None

 
    def get_params(self) -> Tensor:
        """L2 norm of trainable params as a 1-element tensor.

        A flat-tensor view of an LM is not meaningful; the Simulator only uses
        this for history / stability_gap, so a scalar norm is the honest stand-in.
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
