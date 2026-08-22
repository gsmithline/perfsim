"""Tests for the THREE Wu in-context mechanisms (2026-08-22).

They are three different experiments and the whole risk is that they get
called "context" and quietly become one:

  observed_context     STRICT. K demonstrations from the OBSERVED set O,
                       each showing that agent's CURRENT opinion x_j(t).
                       Inside Wu's information set: the platform really
                       does observe O.
  prediction_history   STRICT. The held-out agent's OWN past PLATFORM
                       PREDICTIONS, depth D. The platform remembering
                       what IT said -- a function of its own outputs.
  expressed_history    EXTENSION, NOT Wu. The held-out agent's OWN past
                       POST-FJ OPINIONS, depth D. Wu's platform never
                       observes a held-out opinion; that is what "held
                       out" means. Every log line and config row this
                       mode produces must be flagged.

Collapsing (2) into (3) would be invisible in every downstream number
while turning a faithful replication into a claim Wu's paper does not
support. So the tests below check the three history_source strings are
distinct, that (2) reads served_raw while (3) reads op_raw, and that the
two produce DIFFERENT text on the same run.

Also pinned here:
  * pokec_build_prompt is BYTE-IDENTICAL with no context, against a
    frozen copy of the pre-2026-08-22 implementation. Every archived
    Pokec wave's prompts depend on this;
  * K=8 is a PREFIX of K=32, so a dose sweep varies how much evidence,
    never which evidence;
  * D=0 / K=0 render the empty string, so the frozen no-memory baseline
    IS the zero-shot prompt rather than a fourth condition;
  * SABOTAGE: a strict context carrying a held-out id, and one carrying
    a held-out value smuggled into the prose, are both detected.

Run with USE_TF=0.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
PIPE = REPO / "experiments" / "scripts" / "cluster_pipelines"
POKEC = REPO / "examples" / "pokec"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUN = _load("run_wu_icl", PIPE / "run_pokec_gated_lm.py")
WUC = _load("wu_ctx", PIPE / "wu_context.py")

N = 2163
N_OBS = 1730
N_HELD = 433
OBS_IDS = np.arange(N_OBS, dtype=np.int64)
HELD_IDS = np.arange(N_OBS, N, dtype=np.int64)


# ===================================================================
#   the prompt builder, byte-identical without context
# ===================================================================

class _StubTokenizer:
    chat_template = "stub"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        return "<|u|>" + messages[0]["content"] + "<|a|>"


def _frozen_pokec_build_prompt(profile, tokenizer):
    """VERBATIM copy of pokec_build_prompt as it stood before the
    context slot was added (2026-08-22). Kept here, not imported, so it
    cannot drift with the thing it is supposed to pin."""
    profile_lines = []
    for col in RUN.PROMPT_COLS:
        val = profile.get(col, "")
        if pd.isna(val) or val == "" or str(val) == "nan":
            continue
        if col == "age":
            if float(val) == 0.0:
                continue
            val = int(val)
        elif col == "gender":
            val = {0.0: "female", 1.0: "male"}.get(float(val), "unknown")
        elif col == "relation_to_alcohol":
            val = RUN.translate_alcohol(val)
        profile_lines.append(f"- {col}: {val}")
    profile_str = ("\n".join(profile_lines) if profile_lines
                   else "- (no profile info)")
    user_msg = (
        "Estimate this user's attitude toward smoking based on their profile.\n"
        "Profile:\n"
        f"{profile_str}\n\n"
        "Output a single number in [0, 1] (1 = very positive, 0 = very negative). "
        "Respond with only the number, e.g. 0.42."
    )
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


@pytest.fixture(scope="module")
def setup():
    return RUN.load_pokec_setup(POKEC)


def test_prompt_bytes_unchanged_without_context(setup):
    """Over a wide sample of REAL Pokec profiles, including the ragged
    ones (missing age, missing alcohol, untranslatable strings)."""
    tok = _StubTokenizer()
    prof = setup["profiles"]
    sample = list(range(0, N, 7))
    assert len(sample) > 300
    for i in sample:
        row = prof.iloc[i]
        old = _frozen_pokec_build_prompt(row, tok)
        for ctx in (None, ""):
            new = RUN.pokec_build_prompt(row, tok, context_block=ctx)
            assert new == old, f"agent {i}: prompt bytes moved (ctx={ctx!r})"
    # the default argument path too (how HFCausalLMModel calls it)
    assert RUN.pokec_build_prompt(prof.iloc[0], tok) == \
        _frozen_pokec_build_prompt(prof.iloc[0], tok)


def test_profiles_with_no_usable_columns_still_match(setup):
    """The '- (no profile info)' branch is the one a context slot is
    most likely to break, and it is rare enough to miss by sampling."""
    tok = _StubTokenizer()
    empty = pd.Series({"age": 0.0, "gender": float("nan"),
                       "relation_to_alcohol": ""})
    assert RUN.pokec_build_prompt(empty, tok) == \
        _frozen_pokec_build_prompt(empty, tok)
    assert "(no profile info)" in RUN.pokec_build_prompt(empty, tok)


def test_context_block_is_actually_inserted(setup):
    tok = _StubTokenizer()
    row = setup["profiles"].iloc[0]
    base = RUN.pokec_build_prompt(row, tok)
    withc = RUN.pokec_build_prompt(row, tok, context_block="HELLO CONTEXT")
    assert withc != base
    assert "HELLO CONTEXT" in withc
    # inserted between the profile and the answer instruction, exactly
    # where load_movielens_setup's builder puts it
    assert withc.index("HELLO CONTEXT") < withc.index("Output a single")


# ===================================================================
#          nested selection, and what it may never contain
# ===================================================================

@pytest.mark.parametrize("agent", [1730, 1900, 2162])
def test_k8_is_a_prefix_of_k32(agent):
    for t in range(3):
        a = WUC.select_observed_demos(agent, OBS_IDS, 8, seed=0, round_t=t)
        b = WUC.select_observed_demos(agent, OBS_IDS, 32, seed=0, round_t=t)
        assert len(a) == 8 and len(b) == 32
        assert list(a) == list(b[:8]), (
            "K=8 must be a PREFIX of K=32 -- otherwise a K sweep varies "
            "WHICH people, not HOW MANY")
        assert WUC.nested_prefix_ok(agent, OBS_IDS, 8, 32, seed=0, round_t=t)


def test_k0_selects_nothing():
    assert WUC.select_observed_demos(1800, OBS_IDS, 0,
                                     seed=0, round_t=0).size == 0


def test_selection_is_deterministic_and_agent_specific():
    a = WUC.select_observed_demos(1800, OBS_IDS, 8, seed=0, round_t=1)
    again = WUC.select_observed_demos(1800, OBS_IDS, 8, seed=0, round_t=1)
    other = WUC.select_observed_demos(1801, OBS_IDS, 8, seed=0, round_t=1)
    later = WUC.select_observed_demos(1800, OBS_IDS, 8, seed=0, round_t=2)
    assert list(a) == list(again)
    assert list(a) != list(other)
    assert list(a) != list(later)


def test_selection_never_leaves_the_observed_set():
    for agent in range(N_OBS, N, 37):
        ids = WUC.select_observed_demos(agent, OBS_IDS, 32,
                                        seed=0, round_t=0)
        assert ids.size == 32
        assert ids.max() < N_OBS, "a HELD-OUT agent was used as evidence"
        assert len(set(ids.tolist())) == 32


def test_selection_never_includes_the_target():
    """Only reachable when the target is itself observed, which is why
    the guard exists rather than being argued away."""
    for agent in (0, 5, 999, 1729):
        ids = WUC.select_observed_demos(agent, OBS_IDS, 32,
                                        seed=0, round_t=0)
        assert agent not in set(ids.tolist())


def test_assert_selection_safe_rejects_a_heldout_id():
    with pytest.raises(ValueError, match="NOT in the observed set"):
        WUC.assert_selection_safe(np.array([5, 2000]), OBS_IDS, 1900)


def test_assert_selection_safe_rejects_self_demonstration():
    with pytest.raises(ValueError, match="its own demonstrations"):
        WUC.assert_selection_safe(np.array([5, 7]), OBS_IDS, 7)


# ===================================================================
#              the three mechanisms stay three
# ===================================================================

def test_the_three_history_sources_are_distinct_strings():
    srcs = [WUC.HISTORY_SOURCE[m] for m in
            ("observed_context", "prediction_history", "expressed_history")]
    assert srcs == ["observed_peer", "platform_prediction",
                    "post_fj_opinion"]
    assert len(set(srcs)) == 3
    assert set(srcs) == set(WUC.HISTORY_SOURCES)


def test_only_expressed_history_is_an_extension():
    assert not WUC.is_extension("none")
    assert not WUC.is_extension("observed_context")
    assert not WUC.is_extension("prediction_history")
    assert WUC.is_extension("expressed_history")
    assert WUC.EXTENSION_MODES == ("expressed_history",)


def test_unknown_mode_is_rejected():
    for bad in ("history", "icl", "context", "observed", ""):
        with pytest.raises(ValueError, match="WU_ICL_MODE"):
            WUC.validate_mode(bad)


def _profile_fn(setup):
    return lambda a: WUC.pokec_profile_bits(
        setup["profiles"].iloc[int(a)],
        alcohol_translator=RUN.translate_alcohol)


def test_prediction_and_expressed_history_read_different_series(setup):
    """The two personal-memory mechanisms are handed DIFFERENT lists and
    must render different text. If a refactor ever passes op_raw to both,
    this fails."""
    served = [torch.full((N,), 0.11), torch.full((N,), 0.22)]
    op = [torch.full((N,), 0.77), torch.full((N,), 0.88)]
    tp, ep = WUC.build_context(
        "prediction_history", 2000, d=2,
        pred_history=served, expr_history=op)
    te, ee = WUC.build_context(
        "expressed_history", 2000, d=2,
        pred_history=served, expr_history=op)
    assert ep["values"] == pytest.approx([0.11, 0.22], abs=1e-6)
    assert ee["values"] == pytest.approx([0.77, 0.88], abs=1e-6)
    assert tp != te
    assert "platform previously showed" in tp
    assert "own recorded attitudes" in te
    assert ep["history_source"] == "platform_prediction"
    assert ee["history_source"] == "post_fj_opinion"
    assert ep["extension"] is False and ee["extension"] is True


def test_d_zero_renders_the_empty_string():
    served = [torch.full((N,), 0.11), torch.full((N,), 0.22)]
    for mode in ("prediction_history", "expressed_history"):
        text, entry = WUC.build_context(mode, 2000, d=0,
                                        pred_history=served,
                                        expr_history=served)
        assert text == ""
        assert entry["ids"] == [] and entry["values"] == []
        # and the flag is still carried, so "no context" runs are still
        # attributable to their mode
        assert entry["history_source"] == WUC.HISTORY_SOURCE[mode]


def test_mode_none_renders_the_empty_string_and_no_source():
    text, entry = WUC.build_context("none", 2000)
    assert text == ""
    assert entry["history_source"] is None
    assert entry["extension"] is False


def test_empty_context_is_the_zero_shot_prompt(setup):
    """The required K=0 / D=0 baseline is not a fourth condition: it must
    be the plain frozen prompt, to the byte."""
    tok = _StubTokenizer()
    row = setup["profiles"].iloc[0]
    text, _ = WUC.build_context("prediction_history", 2000, d=0,
                                pred_history=[], expr_history=[])
    assert RUN.pokec_build_prompt(row, tok, context_block=text or None) == \
        _frozen_pokec_build_prompt(row, tok)


def test_history_depth_takes_the_most_recent_d_oldest_to_newest():
    served = [torch.full((N,), float(v)) for v in
              (0.10, 0.20, 0.30, 0.40, 0.50)]
    _, e = WUC.build_context("prediction_history", 2000, d=3,
                             pred_history=served)
    assert e["values"] == pytest.approx([0.30, 0.40, 0.50], abs=1e-6)
    assert e["ids"] == [2000, 2000, 2000]


# ===================================================================
#           strict contexts show nothing held out
# ===================================================================

def test_observed_context_shows_current_observed_opinions(setup):
    op = torch.rand(N)
    text, entry = WUC.build_context(
        "observed_context", 1900, observed_ids=OBS_IDS, opinion=op, k=8,
        profile_fn=_profile_fn(setup), seed=0, round_t=0)
    assert len(entry["ids"]) == 8
    assert all(a < N_OBS for a in entry["ids"])
    for a, v in zip(entry["ids"], entry["values"]):
        assert v == pytest.approx(float(op[a]))
    assert WUC.audit_entry(entry, observed_ids=OBS_IDS, opinion=op) == []


def test_rendered_text_shows_exactly_the_logged_values(setup):
    """The link that makes a held-out label detectable: the prose can
    contain no number the log does not account for."""
    op = torch.rand(N)
    _, entry = WUC.build_context(
        "observed_context", 1900, observed_ids=OBS_IDS, opinion=op, k=32,
        profile_fn=_profile_fn(setup), seed=0, round_t=0)
    shown = WUC.text_values(entry["text"])
    assert len(shown) == 32
    assert shown == [round(v, 2) for v in entry["values"]]


def test_no_heldout_agent_appears_in_a_strict_context(setup):
    """The direct statement of the guarantee, over every held-out agent:
    no held-out id is ever used as evidence, at any round."""
    op = torch.rand(N)
    held = set(HELD_IDS.tolist())
    pf = _profile_fn(setup)
    for agent in range(N_OBS, N, 29):
        for t in range(3):
            _, e = WUC.build_context(
                "observed_context", agent, observed_ids=OBS_IDS,
                opinion=op, k=32, profile_fn=pf, seed=0, round_t=t)
            assert not (set(e["ids"]) & held)
            assert agent not in set(e["ids"])
            assert WUC.audit_entry(e, observed_ids=OBS_IDS, opinion=op) == []


def test_personal_memory_never_carries_another_agents_id():
    served = [torch.rand(N) for _ in range(4)]
    for mode in ("prediction_history", "expressed_history"):
        _, e = WUC.build_context(mode, 2100, d=4, pred_history=served,
                                 expr_history=served)
        assert set(e["ids"]) == {2100}
        assert WUC.audit_entry(e) == []


# ===================================================================
#                          SABOTAGE
# ===================================================================

def test_sabotage_heldout_id_in_a_strict_context(setup):
    """A strict context that used a held-out agent as a demonstration.
    The values are real opinions and the text is well-formed prose --
    only the id membership check catches it."""
    op = torch.rand(N)
    _, e = WUC.build_context(
        "observed_context", 1900, observed_ids=OBS_IDS, opinion=op, k=8,
        profile_fn=_profile_fn(setup), seed=0, round_t=0)
    victim = 2100                                   # a HELD-OUT agent
    bad = copy.deepcopy(e)
    bad["ids"][3] = victim
    bad["values"][3] = float(op[victim])
    bad["text"] = "\n".join(
        line if k != 4 else
        f"- fabricated -> attitude {float(op[victim]):.2f}"
        for k, line in enumerate(bad["text"].split("\n")))
    viol = WUC.audit_entry(bad, observed_ids=OBS_IDS, opinion=op)
    assert any("HELD-OUT ids" in v for v in viol), viol


def test_sabotage_heldout_label_smuggled_into_the_prose(setup):
    """An extra demonstration line appended to the text but not to the
    log. Caught because the text's numbers must reconcile with the
    logged values exactly -- scanning for "a held-out agent's value"
    instead would be vacuous: at two decimals almost every value
    collides with somebody."""
    op = torch.rand(N)
    _, e = WUC.build_context(
        "observed_context", 1900, observed_ids=OBS_IDS, opinion=op, k=8,
        profile_fn=_profile_fn(setup), seed=0, round_t=0)
    bad = copy.deepcopy(e)
    bad["text"] += f"\n- age 30, male -> attitude {float(op[2100]):.2f}"
    viol = WUC.audit_entry(bad, observed_ids=OBS_IDS, opinion=op)
    assert any("rendered text shows" in v for v in viol), viol
    # sanity: the naive scan really is vacuous, which is why we do not
    # rely on it -- the CLEAN entry also "contains a held-out value"
    clean_hits = sum(
        1 for h in HELD_IDS
        if f"{float(op[h]):.2f}" in e["text"])
    assert clean_hits > 0


def test_sabotage_a_swapped_demonstration_label(setup):
    """Right ids, wrong numbers: somebody substituted held-out opinions
    behind correct-looking identities."""
    op = torch.rand(N)
    _, e = WUC.build_context(
        "observed_context", 1900, observed_ids=OBS_IDS, opinion=op, k=8,
        profile_fn=_profile_fn(setup), seed=0, round_t=0)
    bad = copy.deepcopy(e)
    bad["values"][2] = float(op[2100])
    bad["text"] = "\n".join(
        [bad["text"].split("\n")[0]] +
        [f"- x -> attitude {v:.2f}" for v in bad["values"]])
    viol = WUC.audit_entry(bad, observed_ids=OBS_IDS, opinion=op)
    assert any("is not that agent's opinion" in v for v in viol), viol


def test_sabotage_extension_masquerading_as_strict():
    """expressed_history relabelled as a strict mode. The flag and the
    history_source must both stop agreeing with the mode."""
    _, e = WUC.build_context("expressed_history", 2000, d=2,
                             expr_history=[torch.rand(N), torch.rand(N)])
    bad = copy.deepcopy(e)
    bad["mode"] = "prediction_history"
    viol = WUC.audit_entry(bad)
    assert any("history_source" in v for v in viol), viol
    bad2 = copy.deepcopy(e)
    bad2["extension"] = False
    assert any("extension flag" in v for v in WUC.audit_entry(bad2))


# ===================================================================
#                        end to end
# ===================================================================

def _stub_value(prompt):
    h = int(hashlib.sha1(prompt.encode()).hexdigest()[:8], 16)
    return round(0.02 + 0.96 * (h % 9973) / 9973.0, 4)


class _StubLM:
    def __init__(self, *a, profiles=None, prompt_builder=None, **kw):
        self._profiles = profiles
        self._prompt_builder = prompt_builder
        self.tokenizer = _StubTokenizer()
        self.inner_model = torch.nn.Linear(1, 1)
        self.batches = []
        self._last_raw = []
        self._last_parse_fail = 0.0

    def profile_at(self, idx):
        return self._profiles.iloc[int(idx)]

    def build_prompt(self, profile):
        return self._prompt_builder(profile, self.tokenizer)

    def _generate(self, prompts):
        self.batches.append(list(prompts))
        return [f"{_stub_value(p):.4f}" for p in prompts]

    @staticmethod
    def _parse(text, default=0.5):
        try:
            return float(text)
        except ValueError:
            return default

    def perplexity(self, texts):
        return 1.0

    def answer_distribution_stats(self):
        return {}


BASE_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "HOME": os.environ.get("HOME", ""),
    "USE_TF": "0",
    "DATASET": "pokec", "POKEC_DIR": str(POKEC),
    "POP_MODEL": "fj", "FJ_UPDATE_VERSION": "wu1", "RUN_MODE": "loop",
    "TRAINING_STYLE": "frozen", "DATA_REGIME": "replace",
    "USE_LORA": "0", "DEVICE": "cpu", "SEED": "0",
    "N_ROUNDS": "3", "DEPLOY_EVERY": "1", "N_LABELED": str(N_OBS),
    "FJ_INNER_STEPS": "4", "FJ_ALPHA": "0.9", "W_PLAT": "0.3",
    "N_PROBE": "8", "LOG_PERPLEXITY": "0", "LOG_ANSWER_DIST": "0",
    "LOG_PPL_DIST": "0", "GRAD_NORM_N": "0", "TEL_EVAL_CAP": "0",
    "FJ_OBSERVED_PASSTHROUGH": "1",
}


def run_pipeline(**overrides):
    env = dict(BASE_ENV)
    env["RUN_TAG"] = "wu_icl_test"
    env.update({k: str(v) for k, v in overrides.items()})
    tmp = tempfile.mkdtemp(prefix="wu_icl_")
    env["OUT_DIR"] = tmp
    holder = {}

    def _mk(*a, **kw):
        holder["lm"] = _StubLM(*a, **kw)
        return holder["lm"]

    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(RUN, "HFCausalLMModel", _mk), \
            mock.patch.object(RUN.gp, "sft_batch_loss", lambda *a, **k: 0.0), \
            mock.patch.object(RUN.gp, "sft_grad_norm", lambda *a, **k: 0.0):
        assert RUN.main() == 0
    traj = torch.load(Path(tmp) / "trajectory.pt", map_location="cpu",
                      weights_only=False)
    cfg = json.loads((Path(tmp) / "config.json").read_text())
    log = []
    p = Path(tmp) / "wu_ctx_log.json.gz"
    if p.exists():
        with gzip.open(p, "rt") as fh:
            log = [json.loads(ln) for ln in fh if ln.strip()]
    return traj, cfg, holder["lm"], log


@pytest.fixture(scope="module")
def oc_run():
    return run_pipeline(WU_ICL_MODE="observed_context", WU_ICL_K=8)


def test_observed_context_run_logs_every_heldout_agent(oc_run):
    _, cfg, _, log = oc_run
    assert cfg["wu_icl_mode"] == "observed_context"
    assert cfg["wu_icl_k"] == 8 and cfg["wu_icl_d"] == 0
    assert cfg["wu_icl_extension"] is False
    assert cfg["wu_icl_history_source"] == "observed_peer"
    assert len(log) == 3
    for line in log:
        assert line["mode"] == "observed_context"
        assert line["wu_icl_extension"] is False
        assert len(line["agents"]) == N_HELD
        assert {e["agent"] for e in line["agents"]} == set(HELD_IDS.tolist())


def test_every_logged_context_audits_clean_against_the_population(oc_run):
    """The full strict guarantee, re-derived per round from the artifact
    alone: ids in O, values == x_O(t), text numbers == logged values."""
    traj, _, _, log = oc_run
    om = traj["observed_mask"]
    obs_ids = torch.arange(N)[om].numpy()
    for line in log:
        t = line["round"]
        x_cur = traj["innate"] if t == 0 else traj["op_raw"][t - 1]
        for e in line["agents"]:
            viol = WUC.audit_entry(e, observed_ids=obs_ids, opinion=x_cur,
                                   tol=1e-4)
            assert viol == [], (t, e["agent"], viol)


def test_the_demonstrations_track_the_moving_population(oc_run):
    """The displayed values are x_j(t), not innate: a run that froze the
    demonstrations at round 0 would pass every membership check."""
    traj, _, _, log = oc_run
    a0 = {e["agent"]: e for e in log[0]["agents"]}
    a2 = {e["agent"]: e for e in log[2]["agents"]}
    agent = int(HELD_IDS[0])
    assert a0[agent]["values"] != a2[agent]["values"]
    x2 = traj["op_raw"][1]
    for a, v in zip(a2[agent]["ids"], a2[agent]["values"]):
        assert v == pytest.approx(float(x2[a]), abs=1e-4)


def test_context_reaches_the_prompt(oc_run, setup):
    """The prompts the model actually saw carry the demonstration block
    -- the context is not built and then dropped."""
    _, _, lm, log = oc_run
    serving = [b for b in lm.batches if len(b) == N_HELD]
    assert len(serving) == 3
    first = serving[0]
    entry = log[0]["agents"][0]
    assert entry["text"] in first[0]
    assert "attitudes toward smoking of some other users" in first[0]


def test_extension_mode_is_flagged_everywhere_it_appears():
    traj, cfg, _, log = run_pipeline(WU_ICL_MODE="expressed_history",
                                     WU_ICL_D=8, N_ROUNDS=3)
    assert cfg["wu_icl_mode"] == "expressed_history"
    assert cfg["wu_icl_extension"] is True
    assert cfg["wu_icl_history_source"] == "post_fj_opinion"
    for line in log:
        assert line["wu_icl_extension"] is True
        assert line["history_source"] == "post_fj_opinion"
        for e in line["agents"]:
            assert e["extension"] is True
            assert e["history_source"] == "post_fj_opinion"
    # round 0 has no history yet -> empty; round 2 shows op_raw[0:2]
    assert all(e["values"] == [] for e in log[0]["agents"])
    agent = int(HELD_IDS[5])
    e2 = next(e for e in log[2]["agents"] if e["agent"] == agent)
    assert e2["values"] == pytest.approx(
        [float(traj["op_raw"][0][agent]), float(traj["op_raw"][1][agent])],
        abs=1e-4)


def test_prediction_history_shows_what_the_platform_served():
    """The strict memory mechanism reads served_raw, and for a held-out
    agent that is the model's own past output -- NOT the agent's
    post-FJ opinion, which is the extension."""
    traj, cfg, _, log = run_pipeline(WU_ICL_MODE="prediction_history",
                                     WU_ICL_D=8, N_ROUNDS=3)
    assert cfg["wu_icl_extension"] is False
    assert cfg["wu_icl_history_source"] == "platform_prediction"
    agent = int(HELD_IDS[5])
    e2 = next(e for e in log[2]["agents"] if e["agent"] == agent)
    assert e2["values"] == pytest.approx(
        [float(traj["served_raw"][0][agent]),
         float(traj["served_raw"][1][agent])], abs=1e-4)
    # and it is NOT the post-FJ opinion: the two mechanisms are not the
    # same numbers wearing different labels
    assert e2["values"] != pytest.approx(
        [float(traj["op_raw"][0][agent]), float(traj["op_raw"][1][agent])],
        abs=1e-4)


def test_k0_baseline_serves_the_zero_shot_prompt(setup):
    """The required frozen no-memory baseline: mode none must produce
    exactly the prompts a plain passthrough run produces."""
    _, cfg, lm_ctx, log = run_pipeline(WU_ICL_MODE="observed_context",
                                       WU_ICL_K=0, N_ROUNDS=1)
    tok = _StubTokenizer()
    want = [_frozen_pokec_build_prompt(setup["profiles"].iloc[i], tok)
            for i in HELD_IDS]
    serving = [b for b in lm_ctx.batches if len(b) == N_HELD]
    assert serving[0] == want
    assert all(e["text"] == "" for e in log[0]["agents"])


# ===================================================================
#                     validation is loud
# ===================================================================

@pytest.mark.parametrize("over,msg", [
    ({"WU_ICL_MODE": "expressed_history", "WU_ICL_D": "8",
      "TRAINING_STYLE": "sft"}, "frozen"),
    ({"WU_ICL_MODE": "observed_context", "WU_ICL_K": "8",
      "FJ_OBSERVED_PASSTHROUGH": "0"}, "FJ_OBSERVED_PASSTHROUGH"),
    ({"WU_ICL_MODE": "observed_context", "WU_ICL_K": "8",
      "WU_ICL_D": "8"}, "WU_ICL_D=0"),
    ({"WU_ICL_MODE": "prediction_history", "WU_ICL_D": "8",
      "WU_ICL_K": "8"}, "WU_ICL_K=0"),
    ({"WU_ICL_MODE": "banana"}, "WU_ICL_MODE"),
])
def test_bad_icl_configurations_are_refused(over, msg):
    env = dict(BASE_ENV)
    env.update({"RUN_TAG": "bad",
                "OUT_DIR": tempfile.mkdtemp(prefix="wu_icl_bad_")})
    env.update(over)
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(RUN, "HFCausalLMModel", _StubLM):
        with pytest.raises(ValueError) as e:
            RUN.main()
    assert msg in str(e.value)


def test_legacy_icl_knobs_are_exclusive_with_the_wu_modes():
    env = dict(BASE_ENV)
    env.update({"RUN_TAG": "bad", "WU_ICL_MODE": "prediction_history",
                "WU_ICL_D": "8", "ICL_DAYS": "4",
                "OUT_DIR": tempfile.mkdtemp(prefix="wu_icl_bad_")})
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(RUN, "HFCausalLMModel", _StubLM):
        with pytest.raises(ValueError) as e:
            RUN.main()
    assert "ICL_K" in str(e.value) or "movielens" in str(e.value)
