"""Training witness (TRAIN_WITNESS=1, Figure-4 anchor-tradeoff wave).

Pure-torch tests of the witness helpers in _gated_pop.py plus string pins
on the runner / learner plumbing. No HF model is ever built: tiny fake
causal LMs with a peft-like lora_A / lora_B ModuleDict layout stand in.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "experiments" / "scripts" / "cluster_pipelines"
RUNNER = PIPE / "run_pokec_gated_lm.py"
GP = PIPE / "_gated_pop.py"
KL_SFT = ROOT / "perfsim" / "learners" / "lm" / "kl_sft.py"

_spec = importlib.util.spec_from_file_location("_gated_pop_witness", str(GP))
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)

VOCAB = 11
CONTRACT_FIELDS = [
    "witness_steps", "witness_steps_requested", "witness_n_rows",
    "witness_lora_b_norm", "witness_lora_ab_norm",
    "witness_data_loss_last", "witness_kl_last",
    "witness_probe_kl_fwd", "witness_probe_kl_rev",
    "witness_probe_argmax_agree", "witness_probe_n", "witness_probe_sha",
]


class FakeLoraLinear(nn.Module):
    """peft-like LoRA wrapper: lora_A / lora_B are ModuleDicts keyed by
    adapter name, B starts at zero exactly as a fresh adapter does."""

    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.base_layer = nn.Linear(d_in, d_out, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        nn.init.zeros_(self.lora_B["default"].weight)

    def forward(self, x):
        return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))


class FakeCausalLM(nn.Module):
    """Embedding -> LoRA-wrapped linear -> vocab head; returns .logits."""

    def __init__(self, seed=0, d=8, r=3, n_lora=2):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(VOCAB, d)
        self.layers = nn.ModuleList([FakeLoraLinear(d, d, r) for _ in range(n_lora)])
        self.head = nn.Linear(d, VOCAB)
        with torch.no_grad():
            for p in self.parameters():
                if p.requires_grad and p.abs().sum() > 0:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.7)
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, input_ids=None, labels=None, **kw):
        h = self.emb(input_ids)
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return SimpleNamespace(logits=self.head(h))


def _examples(n=5, L=9, n_prompt=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        ids = torch.randint(0, VOCAB, (1, L), generator=g)
        labels = ids.clone()
        labels[:, :n_prompt] = -100
        out.append((ids, labels))
    return out


# --- (a) LoRA norms ---------------------------------------------------------

def test_lora_norms_zero_for_fresh_adapter_then_positive():
    m = FakeCausalLM(seed=0)
    b, ab = gp.lora_norms(m)
    assert b == 0.0 and ab == 0.0
    with torch.no_grad():
        m.layers[0].lora_B["default"].weight.fill_(0.5)
    b, ab = gp.lora_norms(m)
    assert b > 0 and ab > 0
    # exact against the materialized product on the one nonzero pair
    A = m.layers[0].lora_A["default"].weight.detach()
    B = m.layers[0].lora_B["default"].weight.detach()
    assert b == pytest.approx(float(B.norm()), rel=1e-6)
    assert ab == pytest.approx(float((B @ A).norm()), rel=1e-5)


def test_lora_norms_sum_over_modules_and_ignore_non_lora():
    m = FakeCausalLM(seed=0)
    with torch.no_grad():
        m.layers[0].lora_B["default"].weight.fill_(1.0)
        m.layers[1].lora_B["default"].weight.fill_(-2.0)
    pairs = [(n, A.detach(), B.detach()) for n, A, B in gp.lora_pairs(m)]
    assert len(pairs) == 2
    b, ab = gp.lora_norms(m)
    want_b = math.sqrt(sum(float(B.pow(2).sum()) for _, _, B in pairs))
    want_ab = math.sqrt(sum(float((B @ A).pow(2).sum()) for _, A, B in pairs))
    assert b == pytest.approx(want_b, rel=1e-6)
    assert ab == pytest.approx(want_ab, rel=1e-5)
    assert gp.lora_norms(nn.Linear(3, 3)) == (0.0, 0.0)


# --- (b) probe divergence ---------------------------------------------------

def test_probe_divergence_zero_for_identical_models():
    a, b = FakeCausalLM(seed=3), FakeCausalLM(seed=3)
    d = gp.probe_divergence(a, b, _examples())
    assert d["kl_fwd"] == pytest.approx(0.0, abs=1e-6)
    assert d["kl_rev"] == pytest.approx(0.0, abs=1e-6)
    assert d["argmax_agree"] == pytest.approx(1.0)
    assert d["n_tokens"] == 5 * 5   # 9 tokens, 4 prompt, shift drops none of the 5


def test_probe_divergence_positive_for_different_models():
    a, b = FakeCausalLM(seed=3), FakeCausalLM(seed=4)
    d = gp.probe_divergence(a, b, _examples())
    assert d["kl_fwd"] > 1e-4 and math.isfinite(d["kl_fwd"])
    assert d["kl_rev"] >= 0 and d["kl_rev"] > 1e-4
    assert 0.0 <= d["argmax_agree"] <= 1.0
    # direction convention: swapping the models swaps fwd/rev
    e = gp.probe_divergence(b, a, _examples())
    assert e["kl_fwd"] == pytest.approx(d["kl_rev"], rel=1e-5)
    assert e["kl_rev"] == pytest.approx(d["kl_fwd"], rel=1e-5)


def test_probe_divergence_restores_mode_and_cache_and_masks_prompt():
    a, b = FakeCausalLM(seed=3), FakeCausalLM(seed=4)
    a.train(); b.eval()
    a.config.use_cache = True
    gp.probe_divergence(a, b, _examples())
    assert a.training is True and b.training is False
    assert a.config.use_cache is True
    # prompt-only labels (all -100) score zero tokens and return finite zeros
    ex = [(ids, torch.full_like(ids, -100)) for ids, _ in _examples(n=2)]
    d = gp.probe_divergence(a, b, ex)
    assert d == {"kl_fwd": 0.0, "kl_rev": 0.0, "argmax_agree": 0.0, "n_tokens": 0}


# --- (c) requested steps ----------------------------------------------------

def test_witness_steps_requested_formula():
    assert gp.witness_steps_requested(723, 4, 1, 1) == 181
    assert gp.witness_steps_requested(723, 4, 10, 1) == 19
    assert gp.witness_steps_requested(723, 4, 1, 2) == 362
    assert gp.witness_steps_requested(724, 4, 1, 1) == 181
    assert gp.witness_steps_requested(0, 4, 1, 1) == 0
    assert gp.witness_steps_requested(723, 4, 1, 0) == 0


# --- (d) runner + learner plumbing ------------------------------------------

def test_runner_plumbs_train_witness_opt_in_and_records_it():
    src = RUNNER.read_text()
    assert 'train_witness = _env_int("TRAIN_WITNESS", 0) == 1' in src
    assert 'if train_witness:' in src
    assert 'config["train_witness"] = True' in src
    assert 'config["witness_probe_n"] = n_probe' in src
    assert 'learner.record_last_terms = True' in src
    assert 'gp.train_witness_block(' in src
    assert 'loss_block.update(_wit)' in src
    # the refusal is up front, before any round runs
    assert 'TRAIN_WITNESS=1 requires TRAINING_STYLE=sft_kl' in src
    # the witness sits in the sft/sft_kl branch right after the learner call
    i_train = src.index("                    learner.train(train_data)\n")
    i_wit = src.index("gp.train_witness_block(")
    i_dep = src.index("            cur_dep += 1\n")
    assert i_train < i_wit < i_dep


def test_runner_writes_peer_gate_evidence_in_every_mode_under_witness():
    """Trajectory rows carry peer_gate_mode / peer_pairs only under all_open
    (legacy rows byte-identical). F4A runs threshold, so under the witness
    the same two keys are written in every mode; the all_open branch stays
    untouched and the witness arm is an `elif` on it."""
    src = RUNNER.read_text()
    k_mode = 'row["peer_gate_mode"] = peer_gate_mode'
    k_pairs = 'row["peer_pairs"] = int(n) * int(ab_sweeps)'
    assert src.count(k_mode) == 2 and src.count(k_pairs) == 2
    i_open = src.index('            if peer_gate_mode == "all_open":')
    i_elif = src.index('            elif train_witness:', i_open)
    i_next = src.index('            if ab_sweeps > 1:', i_elif)
    # legacy branch: the first pair of writes, before the elif
    first = src.index(k_mode)
    assert i_open < first < src.index(k_pairs) < i_elif
    # witness arm: the second pair of writes, between the elif and the
    # next statement, and nothing else is written there
    seg = src[i_elif:i_next]
    assert seg.count(k_mode) == 1 and seg.count(k_pairs) == 1
    assert seg.count('row[') == 2
    # off-witness rows in non-all_open modes are untouched: no other
    # writer of either key anywhere in the file
    assert src.count('row["peer_pairs"]') == 2
    assert src.count('row["peer_gate_mode"]') == 2


def test_learner_exposes_last_step_terms_only_when_asked():
    src = KL_SFT.read_text()
    assert "self.record_last_terms: bool = False" in src
    assert "self.last_loss_terms: dict[str, float] = {}" in src
    assert "learner_self._last_terms = (ce.detach(), kl.detach())" in src
    assert "learner_self._last_terms = (ce.detach(), None)" in src
    assert '"data_loss_last": float(ce.detach().float().item())' in src
    # the stash is guarded, and the returned loss is unchanged
    assert src.count("if record:") == 2
    assert "total = ce + kl_beta * kl\n" in src
    assert "return (total, outputs) if return_outputs else total" in src


# --- (e) off -> {} ; on -> the exact contract fields ------------------------

def test_witness_block_is_empty_when_off():
    assert gp.train_witness_block(False) == {}
    assert gp.train_witness_block(0, lm=object(), learner=object()) == {}


class _FakeTok:
    """char-level tokenizer with the HF call signature _example_ids uses."""

    def __call__(self, text, return_tensors=None, truncation=False):
        ids = torch.tensor([[(ord(c) % (VOCAB - 1)) + 1 for c in text]])
        return SimpleNamespace(input_ids=ids)


class _FakeLM:
    def __init__(self, module):
        self.inner_model = module
        self.tokenizer = _FakeTok()
        self._target_device = torch.device("cpu")

    def profile_at(self, i):
        return {"i": int(i)}

    def build_prompt(self, profile):
        return f"agent {profile['i']} likes:"


class _FakeLearner:
    def __init__(self, ref, terms):
        self._ref = ref
        self.last_loss_terms = dict(terms)

    def _ensure_ref(self):
        return self._ref


def _fmt(y):
    return f"{float(y):.2f}"


def test_witness_block_on_writes_exactly_the_contract_fields():
    adapter = FakeCausalLM(seed=5)
    ref = FakeCausalLM(seed=5)
    lm = _FakeLM(adapter)
    probe_idx = torch.tensor([7, 3, 11, 0])
    probe_y = torch.tensor([0.1, 0.5, 0.9, 0.25])
    learner = _FakeLearner(ref, {"data_loss_last": 0.31, "kl_last": 0.02})
    stats = {"global_step": 181, "n_rows": 723, "trainer_seed": 42}
    row = gp.train_witness_block(True, lm=lm, learner=learner, train_stats=stats,
                                 n_train=723, probe_idx=probe_idx, probe_y=probe_y,
                                 fmt=_fmt, steps_requested=181)
    assert list(row) == CONTRACT_FIELDS
    assert row["witness_steps"] == 181 and row["witness_steps_requested"] == 181
    assert row["witness_n_rows"] == 723
    assert row["witness_lora_b_norm"] == 0.0 and row["witness_lora_ab_norm"] == 0.0
    assert row["witness_data_loss_last"] == pytest.approx(0.31)
    assert row["witness_kl_last"] == pytest.approx(0.02)
    assert row["witness_probe_kl_fwd"] == pytest.approx(0.0, abs=1e-6)
    assert row["witness_probe_kl_rev"] == pytest.approx(0.0, abs=1e-6)
    assert row["witness_probe_argmax_agree"] == pytest.approx(1.0)
    assert row["witness_probe_n"] == 4
    assert row["witness_probe_sha"] == gp.probe_sha(probe_idx)
    assert len(row["witness_probe_sha"]) == 64
    for k, v in row.items():
        if isinstance(v, float):
            assert math.isfinite(v)
        else:
            assert isinstance(v, (int, str))

    # after the adapter moves: B > 0, the probe divergence turns positive,
    # the probe identity is unchanged
    with torch.no_grad():
        adapter.layers[1].lora_B["default"].weight.fill_(0.8)
    row2 = gp.train_witness_block(True, lm=lm, learner=learner, train_stats=stats,
                                  n_train=723, probe_idx=probe_idx, probe_y=probe_y,
                                  fmt=_fmt, steps_requested=181)
    assert row2["witness_lora_b_norm"] > 0 and row2["witness_lora_ab_norm"] > 0
    assert row2["witness_probe_kl_fwd"] > 1e-9 and row2["witness_probe_kl_rev"] >= 0
    assert row2["witness_probe_sha"] == row["witness_probe_sha"]


def test_probe_sha_is_order_sensitive_and_stable():
    a = gp.probe_sha(torch.tensor([1, 2, 3]))
    assert a == gp.probe_sha([1, 2, 3]) == gp.probe_sha(torch.tensor([1, 2, 3]))
    assert a != gp.probe_sha([3, 2, 1])
