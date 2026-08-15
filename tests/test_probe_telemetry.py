"""probe_predictions must mirror forward()'s generation telemetry.

The in-context serving path calls gp.probe_predictions rather than
HFCausalLMModel.forward, so before 2026-08-15 it never populated
_last_raw / _last_parse_fail. DEBUG_GEN=1 then printed the stale
__init__ values -- raw=[] and parse_fail_frac=0.0 -- for EVERY ICL run,
which reads as "no parse failures" when nothing was measured. That
false signal sent the mistral7b K=32 diagnosis down the wrong path.

These tests pin the mirror, and pin that it is telemetry-only: the
served values must remain exactly lm._parse over the same generations.
"""
import importlib.util
import os
import re

import pytest

_GP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "experiments", "scripts", "cluster_pipelines",
                   "_gated_pop.py")
_spec = importlib.util.spec_from_file_location("_gated_pop_probe", _GP)
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)


class FakeLM:
    """Stands in for HFCausalLMModel with the same parse/telemetry
    contract: _parse takes the first number and clips to [0,1],
    defaulting to 0.5 when the text holds no digit."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self._last_raw = []
        self._last_parse_fail = 0.0

    def _generate(self, prompts):
        assert len(prompts) == len(self._outputs)
        return list(self._outputs)

    @staticmethod
    def _parse(text, default=0.5):
        m = re.search(r"\d+\.?\d*", text)
        if m is None:
            return default
        try:
            v = float(m.group())
        except ValueError:
            return default
        return max(0.0, min(1.0, v))


def test_raw_generations_are_recorded():
    lm = FakeLM(["0.42", "0.9", "0.13"])
    gp.probe_predictions(lm, ["p"] * 3)
    assert lm._last_raw == ["0.42", "0.9", "0.13"]


def test_parse_fail_fraction_counts_digitless_outputs():
    # two of four hold no digit at all -> they take the 0.5 default
    lm = FakeLM(["0.42", "I cannot say", "0.7", "unknown"])
    vals = gp.probe_predictions(lm, ["p"] * 4)
    assert lm._last_parse_fail == pytest.approx(0.5)
    assert vals == [0.42, 0.5, 0.7, 0.5]


def test_total_parse_failure_is_visible_not_silent():
    """The mistral-K=32 shape: every prediction 0.5. The telemetry must
    now distinguish 'no digit anywhere' from a genuine constant."""
    digitless = FakeLM(["Based on the profile, I would estimate"] * 8)
    vals_d = gp.probe_predictions(digitless, ["p"] * 8)
    assert vals_d == [0.5] * 8
    assert digitless._last_parse_fail == pytest.approx(1.0)

    genuine = FakeLM(["0.5"] * 8)
    vals_g = gp.probe_predictions(genuine, ["p"] * 8)
    assert vals_g == [0.5] * 8
    assert genuine._last_parse_fail == pytest.approx(0.0)
    # identical served values, opposite diagnoses -- exactly the
    # ambiguity the trajectory alone cannot resolve
    assert vals_d == vals_g


def test_served_values_unchanged_by_the_mirror():
    """Telemetry-only: the returned values must equal _parse over the
    same generations, so no archived run can shift."""
    outs = ["0.42", "no number", "1.7", "0", "0.815"]
    lm = FakeLM(outs)
    assert gp.probe_predictions(lm, ["p"] * len(outs)) == \
        [FakeLM._parse(o) for o in outs]


def test_empty_prompt_list_does_not_divide_by_zero():
    lm = FakeLM([])
    assert gp.probe_predictions(lm, []) == []
    assert lm._last_parse_fail == 0.0
