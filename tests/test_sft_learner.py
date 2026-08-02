"""Structural tests for SFTLearner and the KLSFTLearner anchor math.

These tests do not invoke TRL's SFTTrainer (which would need a real model
+ training step). They verify the perfsim-side glue: that
`_build_dataset` filters / formats correctly given an `agent_idx`-bearing
data dict, that missing `agent_idx` raises a clear error, and that the
per-token anchor divergence (reverse/forward KL, JS) is mathematically
correct on pure tensors.
"""

from __future__ import annotations

import math

import pytest
import torch

pd = pytest.importorskip("pandas")  # type: ignore[assignment]
pytest.importorskip("datasets")

from perfsim.core.types import SUPERVISED_SCHEMA
from perfsim.learners.lm.kl_sft import KLSFTLearner, _anchor_divergence_per_token
from perfsim.learners.lm.sft import SFTLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.models.hf_causal_lm import HFCausalLMModel


def _stub_prompt_builder(profile, tokenizer):
    return f"Q(age={int(profile['age'])}, sex={int(profile['gender'])}) A: "


def _make_learner_unloaded() -> tuple[SFTLearner, HFCausalLMModel]:
    profiles = pd.DataFrame(
        {
            "age": [17, 18, 19, 20, 21],
            "gender": [0, 1, 0, 1, 0],
        }
    )
    model = HFCausalLMModel(
        base_model_name="stub/never-downloaded",
        profiles=profiles,
        prompt_builder=_stub_prompt_builder,
        load_now=False,
    )
    # _build_dataset calls ensure_loaded(); bypass the real download by
    # setting sentinel tokenizer and inner_model attributes. The stub
    # prompt_builder doesn't actually use the tokenizer.
    model.tokenizer = object()  # type: ignore[assignment]
    model.inner_model = object()  # type: ignore[assignment]
    learner = SFTLearner(model, MSELoss(), max_steps=1, output_dir="/tmp/perfsim-sft-test")
    return learner, model


class TestBuildDataset:
    def test_builds_one_example_per_agent_idx(self) -> None:
        learner, _ = _make_learner_unloaded()
        # Three labeled agents: indices 0, 2, 4. Targets: 0.12, 0.55, 0.83.
        data = {
            "x": torch.zeros(3, 4),
            "y": torch.tensor([[0.12], [0.55], [0.83]]),
            "agent_idx": torch.tensor([0, 2, 4]),
        }
        ds = learner._build_dataset(data)
        assert len(ds) == 3
        assert ds[0]["text"] == "Q(age=17, sex=0) A: 0.12"
        assert ds[1]["text"] == "Q(age=19, sex=0) A: 0.55"
        assert ds[2]["text"] == "Q(age=21, sex=0) A: 0.83"

    def test_handles_y_without_trailing_singleton(self) -> None:
        learner, _ = _make_learner_unloaded()
        data = {
            "x": torch.zeros(2, 4),
            "y": torch.tensor([0.30, 0.70]),  # shape (2,) not (2, 1)
            "agent_idx": torch.tensor([1, 3]),
        }
        ds = learner._build_dataset(data)
        assert len(ds) == 2
        assert ds[0]["text"].endswith("0.30")
        assert ds[1]["text"].endswith("0.70")

    def test_missing_agent_idx_raises(self) -> None:
        learner, _ = _make_learner_unloaded()
        data = {"x": torch.zeros(2, 4), "y": torch.tensor([[0.5], [0.5]])}
        with pytest.raises(KeyError, match="agent_idx"):
            learner._build_dataset(data)

    def test_idx_length_mismatch_raises(self) -> None:
        learner, _ = _make_learner_unloaded()
        data = {
            "x": torch.zeros(3, 4),
            "y": torch.tensor([[0.1], [0.2], [0.3]]),
            "agent_idx": torch.tensor([0, 1]),  # only 2 indices for 3 targets
        }
        with pytest.raises(ValueError, match="does not match"):
            learner._build_dataset(data)


class TestSchema:
    def test_accepts_supervised_schema(self) -> None:
        assert SFTLearner.accepts(SUPERVISED_SCHEMA)


class TestConstructorChecks:
    def test_rejects_non_lm_model(self) -> None:
        with pytest.raises(TypeError, match="HFCausalLMModel"):
            SFTLearner(LinearModel(in_features=3), MSELoss())

    def test_kl_direction_validated(self) -> None:
        _, model = _make_learner_unloaded()
        with pytest.raises(ValueError, match="kl_direction"):
            KLSFTLearner(model, MSELoss(), ref_model_name="stub/never-downloaded",
                         kl_direction="sideways")
        learner = KLSFTLearner(model, MSELoss(),
                               ref_model_name="stub/never-downloaded",
                               kl_direction="js")
        assert learner.kl_direction == "js"


def _rand_logps(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(2, 3, 5, generator=g), dim=-1)


class TestAnchorDivergence:
    """Pure-tensor checks on _anchor_divergence_per_token (no model, no TRL)."""

    def test_reverse_and_forward_match_manual_kl(self) -> None:
        logp, logq = _rand_logps(0), _rand_logps(1)
        rev = _anchor_divergence_per_token(logp, logq, "reverse")
        fwd = _anchor_divergence_per_token(logp, logq, "forward")
        assert torch.allclose(rev, (logp.exp() * (logp - logq)).sum(-1))
        assert torch.allclose(fwd, (logq.exp() * (logq - logp)).sum(-1))

    def test_js_matches_definition(self) -> None:
        logp, logq = _rand_logps(0), _rand_logps(1)
        logm = (0.5 * (logp.exp() + logq.exp())).log()
        manual = 0.5 * ((logp.exp() * (logp - logm)).sum(-1)
                        + (logq.exp() * (logq - logm)).sum(-1))
        js = _anchor_divergence_per_token(logp, logq, "js")
        assert torch.allclose(js, manual, atol=1e-6)

    def test_js_symmetric_bounded_zero_at_equality(self) -> None:
        logp, logq = _rand_logps(0), _rand_logps(1)
        js_pq = _anchor_divergence_per_token(logp, logq, "js")
        js_qp = _anchor_divergence_per_token(logq, logp, "js")
        assert torch.allclose(js_pq, js_qp, atol=1e-6)
        assert (js_pq >= -1e-7).all()
        assert (js_pq <= math.log(2.0) + 1e-6).all()
        at_eq = _anchor_divergence_per_token(logp, logp, "js")
        assert torch.allclose(at_eq, torch.zeros_like(at_eq), atol=1e-6)

    def test_gradient_flows_through_policy_all_directions(self) -> None:
        logq = _rand_logps(1)
        for direction in ("reverse", "forward", "js"):
            logits = torch.randn(2, 3, 5, requires_grad=True)
            logp = torch.log_softmax(logits, dim=-1)
            _anchor_divergence_per_token(logp, logq, direction).sum().backward()
            assert logits.grad is not None
            assert torch.isfinite(logits.grad).all()
            assert logits.grad.abs().sum() > 0
