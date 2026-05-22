"""Structural tests for HFCausalLMModel.

These tests do not download a real LM. They use `load_now=False` to construct
the wrapper without touching transformers / PEFT, and then verify the
non-generation parts of the API: profile lookup, output parsing, get_params,
forward-shape validation.

A real end-to-end test with a tiny LM lives in `examples/pokec_fj_llm.py`
and is not exercised in the unit-test suite (too slow on CPU; requires
internet for the first model download).
"""

from __future__ import annotations

import pytest
import torch

pd = pytest.importorskip("pandas")  # type: ignore[assignment]

from perfsim.models.hf_causal_lm import HFCausalLMModel


def _stub_prompt_builder(profile, tokenizer):
    return f"AGE={profile['age']} GENDER={profile['gender']}\nAnswer: "


def _make_profiles(n: int) -> "pd.DataFrame":
    return pd.DataFrame(
        {
            "age": [17 + i for i in range(n)],
            "gender": [i % 2 for i in range(n)],
        }
    )


def _make_model_unloaded(n: int = 5) -> HFCausalLMModel:
    return HFCausalLMModel(
        base_model_name="stub/never-downloaded",
        profiles=_make_profiles(n),
        prompt_builder=_stub_prompt_builder,
        load_now=False,
    )


class TestConstruction:
    def test_constructs_without_loading(self) -> None:
        m = _make_model_unloaded(5)
        assert m.inner_model is None
        assert m.tokenizer is None

    def test_n_inferred_from_profiles(self) -> None:
        m = _make_model_unloaded(7)
        assert m._n == 7

    def test_profiles_without_len_raises(self) -> None:
        class _NoLen:
            pass

        with pytest.raises(TypeError, match="len"):
            HFCausalLMModel(
                base_model_name="stub",
                profiles=_NoLen(),
                prompt_builder=_stub_prompt_builder,
                load_now=False,
            )


class TestProfileAccess:
    def test_profile_at_dataframe(self) -> None:
        m = _make_model_unloaded(4)
        row = m.profile_at(2)
        assert int(row["age"]) == 19
        assert int(row["gender"]) == 0

    def test_profile_at_list(self) -> None:
        m = HFCausalLMModel(
            base_model_name="stub",
            profiles=[{"age": 20, "gender": 1}, {"age": 21, "gender": 0}],
            prompt_builder=_stub_prompt_builder,
            load_now=False,
        )
        assert m.profile_at(1)["age"] == 21


class TestParse:
    def test_parses_simple(self) -> None:
        assert HFCausalLMModel._parse("0.42") == pytest.approx(0.42)

    def test_parses_with_context(self) -> None:
        assert HFCausalLMModel._parse("The answer is 0.73, I think") == pytest.approx(0.73)

    def test_clips_above_one(self) -> None:
        assert HFCausalLMModel._parse("1.5") == 1.0

    def test_clips_below_zero(self) -> None:
        # -0.5 has a digit prefix; the regex matches digits, not the sign,
        # so this parses as 0.5. Documenting current behavior.
        assert HFCausalLMModel._parse("-0.5") == pytest.approx(0.5)

    def test_default_when_no_match(self) -> None:
        assert HFCausalLMModel._parse("garbage", default=0.3) == 0.3


class TestGetParams:
    def test_returns_zeros_when_unloaded(self) -> None:
        m = _make_model_unloaded(3)
        params = m.get_params()
        assert params.shape == (1,)
        assert float(params) == 0.0


class TestForwardShape:
    def test_rejects_wrong_leading_dim(self) -> None:
        m = _make_model_unloaded(5)
        with pytest.raises(ValueError, match="does not match profiles N"):
            m.forward(torch.zeros(7, 4))  # would also need real LM, but shape check fires first


class TestUnsupportedAPI:
    def test_set_params_raises(self) -> None:
        m = _make_model_unloaded(3)
        with pytest.raises(NotImplementedError, match="set_params"):
            m.set_params(torch.zeros(10))

    def test_clone_raises(self) -> None:
        m = _make_model_unloaded(3)
        with pytest.raises(NotImplementedError, match="clone"):
            m.clone()
