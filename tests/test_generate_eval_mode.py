"""Regression test: serving must happen in EVALUATION mode.

THE BUG (found 2026-08-20 while gating the Qwen Wu-limit wave).
HFCausalLMModel._generate() saved and restored grad checkpointing and the
KV-cache flag, but never touched the module's train/eval state -- it just
inherited whatever the last caller left behind. And the last caller is
the SFT trainer: HF's Trainer.train() puts the model in TRAINING mode and
never restores eval. With the wrapper's default lora_dropout=0.05 (the
runner does not override it), that left LoRA dropout ACTIVE during
generation, so:

  * "greedy" decoding was not deterministic -- the dropout masks are
    resampled per forward pass, so the same prompt could decode
    differently twice in the same round;
  * every served prediction in every LoRA arm carried dropout noise it
    was never supposed to have.

The fix forces .eval() for the duration of generation and restores the
caller's exact previous mode in `finally`, so an exception mid-generation
cannot strand the model in eval and silently disable dropout for the next
training round either.

These tests use a tiny pure-torch stub, not a real LM -- no download, no
GPU, no transformers. The stub is deliberately mode-SENSITIVE and the
first test proves that, because a stub that ignored train/eval would make
every assertion below pass vacuously.

Run with USE_TF=0.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pd = pytest.importorskip("pandas")

from perfsim.models.hf_causal_lm import HFCausalLMModel   # noqa: E402


class _Batch(dict):
    """Mapping that survives `**inputs` and answers `.to(device)`."""

    def to(self, _device):
        return self


class _StubTokenizer:
    pad_token_id = 0

    def __call__(self, batch, return_tensors=None, padding=None,
                 truncation=None):
        n = len(batch)
        return _Batch(input_ids=torch.ones(n, 3, dtype=torch.long),
                      attention_mask=torch.ones(n, 3, dtype=torch.long))

    @staticmethod
    def batch_decode(tokens, skip_special_tokens=True):
        return [" ".join(str(int(t)) for t in row) for row in tokens]


class _StubConfig:
    def __init__(self):
        self.pad_token_id = 0
        self.use_cache = False


class _DropoutLM(nn.Module):
    """Stands in for a LoRA-adapted causal LM whose output depends on
    dropout. p=0.5 over a wide vector makes a train-mode collision
    astronomically unlikely, so a difference between the two modes is
    real signal rather than a coin flip."""

    def __init__(self, width: int = 4096):
        super().__init__()
        self.drop = nn.Dropout(p=0.5)
        self.register_buffer("w", torch.arange(1, width + 1,
                                               dtype=torch.float32))
        self.config = _StubConfig()

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        n = input_ids.shape[0]
        # dropout is IDENTITY in eval and random in train -> the emitted
        # token is deterministic in eval and effectively random in train
        vals = self.drop(self.w.expand(n, -1)).sum(dim=1)
        tok = (vals.long() % 977).unsqueeze(1)
        return torch.cat([input_ids, tok], dim=1)


def _model():
    m = HFCausalLMModel(
        base_model_name="stub/never-downloaded",
        profiles=pd.DataFrame({"age": [20, 30], "gender": [0, 1]}),
        prompt_builder=lambda p, tok: "Answer: ",
        load_now=False,
        do_sample=False,
    )
    m.inner_model = _DropoutLM()
    m.tokenizer = _StubTokenizer()
    m._target_device = "cpu"
    return m


PROMPTS = ["Answer: "] * 4


# -- the stub really is mode-sensitive (guards against a vacuous test) ---

def test_stub_is_genuinely_mode_sensitive():
    """If this fails, every assertion below is meaningless: it would mean
    the stub decodes identically in both modes regardless of the fix."""
    lm = _DropoutLM()
    ids = torch.ones(4, 3, dtype=torch.long)
    lm.train()
    a = lm.generate(input_ids=ids)[:, -1]
    b = lm.generate(input_ids=ids)[:, -1]
    assert not torch.equal(a, b), \
        "stub is not dropout-sensitive in training mode"
    lm.eval()
    c = lm.generate(input_ids=ids)[:, -1]
    d = lm.generate(input_ids=ids)[:, -1]
    assert torch.equal(c, d), "stub is not deterministic in eval mode"


# -- the actual regression ----------------------------------------------

def test_greedy_generation_is_identical_when_left_in_training_mode():
    """THE regression. Put the model in training mode -- exactly the state
    HF's Trainer.train() leaves behind -- and two back-to-back greedy
    calls must still agree."""
    m = _model()
    m.inner_model.train()
    assert m.inner_model.training
    first = m._generate(PROMPTS)
    second = m._generate(PROMPTS)
    assert first == second, (
        f"greedy generation is not deterministic from training mode: "
        f"{first} != {second} -- dropout is still active during serving")


def test_training_mode_is_restored_after_generation():
    m = _model()
    m.inner_model.train()
    m._generate(PROMPTS)
    assert m.inner_model.training, (
        "generation left the model in eval mode; the next training round "
        "would silently run without dropout")


def test_eval_mode_is_preserved_for_an_already_eval_model():
    m = _model()
    m.inner_model.eval()
    m._generate(PROMPTS)
    assert not m.inner_model.training


def test_mode_is_restored_even_when_generation_raises():
    """An exception mid-generation must not strand the model in eval --
    that would disable dropout for every later training round, which is
    the same class of silent corruption in the opposite direction."""
    m = _model()

    def boom(**kwargs):
        raise RuntimeError("CUDA OOM")

    m.inner_model.generate = boom
    m.inner_model.train()
    with pytest.raises(RuntimeError, match="CUDA OOM"):
        m._generate(PROMPTS)
    assert m.inner_model.training, \
        "an exception during generation left the model in eval mode"

    m.inner_model.eval()
    with pytest.raises(RuntimeError, match="CUDA OOM"):
        m._generate(PROMPTS)
    assert not m.inner_model.training


def test_generation_actually_ran_in_eval_mode():
    """Directly observe the mode INSIDE generate(), rather than inferring
    it from the outputs."""
    m = _model()
    seen = []
    real = m.inner_model.generate

    def spy(**kwargs):
        seen.append(m.inner_model.training)
        return real(**kwargs)

    m.inner_model.generate = spy
    m.inner_model.train()
    m._generate(PROMPTS)
    assert seen and not any(seen), \
        f"generate() saw training={seen}; serving must run in eval mode"


def test_cache_and_checkpoint_restore_still_work():
    """The pre-existing finally-block behaviour must be unchanged."""
    m = _model()
    m.inner_model.config.use_cache = False
    m.inner_model.train()
    m._generate(PROMPTS)
    assert m.inner_model.config.use_cache is False
    assert m.inner_model.training
