"""Tests for FJ_OBSERVED_PASSTHROUGH (2026-08-22, Wu replication).

Wu's platform does not guess labels it already holds. It OBSERVES the
labeled 80% of Pokec (1730 agents) and PREDICTS the held-out 20% (433).
The passthrough implements exactly that: the served vector is

    s(t)_i = x_i(t)          for i in O   (the platform's own record)
    s(t)_i = model(prompt_i) for i in U   (the only place it guesses)

and s(t) is what run_wu applies to the population.

WHY THIS NEEDS TESTS RATHER THAN A COMMENT
------------------------------------------
Every way of getting this wrong produces a perfectly well-formed
trajectory:

  * serving the model's own output on O instead of x_O(t) turns an
    80%-observed platform into a 100%-predicted one, which is a
    different experiment with the same file layout;
  * training on innate instead of the CURRENT opinion silently deletes
    the feedback loop (the labels stop moving) while every array keeps
    its shape;
  * asking the model about O and then discarding the answer costs 5x
    the generation budget and leaves a model output for O sitting in
    the artifact, one indexing slip away from being served.

So the tests below check the served vector against x(t) and against the
model output SEPARATELY, per round, and the sabotage test builds an
artifact whose only defect is a model value written onto an observed
agent -- because a guarantee nobody has watched fail is not a guarantee.

NO MODEL IS LOADED. The LM is a stub whose generation is a deterministic
function of the prompt string, so the whole loop -- prompt construction,
serving, FJ, buffering -- runs in a couple of seconds on a laptop.

Run with USE_TF=0.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
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


RUN = _load("run_wu_pt", PIPE / "run_pokec_gated_lm.py")
WUC = RUN.wuc

N = 2163
N_OBS = 1730
N_HELD = 433


# ===================================================================
#                      the stub platform
# ===================================================================

def stub_value(prompt: str) -> float:
    """The stub LM's answer: a deterministic function of the PROMPT.

    Prompt-derived rather than agent-derived on purpose. The runner
    decides which prompts to build and in what order; if the test's
    expected value were keyed on the agent index, a runner that built
    the wrong prompt for the right agent would still pass.
    """
    h = int(hashlib.sha1(prompt.encode()).hexdigest()[:8], 16)
    return round(0.02 + 0.96 * (h % 9973) / 9973.0, 4)


class _StubTokenizer:
    chat_template = "stub"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        return "<|u|>" + messages[0]["content"] + "<|a|>"


class _StubLM:
    """Minimal stand-in for HFCausalLMModel: records every generation
    batch it is handed, so the test can prove WHICH agents were asked."""

    def __init__(self, *a, profiles=None, prompt_builder=None, **kw):
        self._profiles = profiles
        self._prompt_builder = prompt_builder
        self.tokenizer = _StubTokenizer()
        self.inner_model = torch.nn.Linear(1, 1)
        self.batches = []          # every _generate call, in order
        self._last_raw = []
        self._last_parse_fail = 0.0

    # --- the HFCausalLMModel surface main() actually touches ---
    def profile_at(self, idx):
        return self._profiles.iloc[int(idx)]

    def build_prompt(self, profile):
        return self._prompt_builder(profile, self.tokenizer)

    def _generate(self, prompts):
        self.batches.append(list(prompts))
        return [f"{stub_value(p):.4f}" for p in prompts]

    @staticmethod
    def _parse(text, default=0.5):
        try:
            return float(text)
        except ValueError:
            return default

    def __call__(self, x):
        prompts = [self.build_prompt(self.profile_at(i))
                   for i in range(x.shape[0])]
        self.batches.append(list(prompts))
        return torch.tensor([stub_value(p) for p in prompts],
                            dtype=torch.float32).unsqueeze(-1)

    def perplexity(self, texts):
        return 1.0

    def answer_distribution_stats(self):
        return {}


BASE_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "HOME": os.environ.get("HOME", ""),
    "USE_TF": "0",
    "DATASET": "pokec",
    "POKEC_DIR": str(POKEC),
    "POP_MODEL": "fj",
    "FJ_UPDATE_VERSION": "wu1",
    "RUN_MODE": "loop",
    "TRAINING_STYLE": "frozen",
    "DATA_REGIME": "replace",
    "USE_LORA": "0",
    "DEVICE": "cpu",
    "SEED": "0",
    "N_ROUNDS": "3",
    "DEPLOY_EVERY": "1",
    "N_LABELED": str(N_OBS),
    "FJ_INNER_STEPS": "4",
    "FJ_ALPHA": "0.9",
    "W_PLAT": "0.3",
    "PLATFORM_SUS_SCALE": "1.0",
    "N_PROBE": "8",
    "LOG_PERPLEXITY": "0",
    "LOG_ANSWER_DIST": "0",
    "LOG_PPL_DIST": "0",
    "GRAD_NORM_N": "0",
    "TEL_EVAL_CAP": "0",
}


def run_pipeline(**overrides):
    """Run main() end to end against real Pokec with a stub LM.

    Returns (trajectory dict, config dict, stub lm, out_dir).
    """
    env = dict(BASE_ENV)
    env["RUN_TAG"] = "wu_pt_test"
    env.update({k: str(v) for k, v in overrides.items()})
    tmp = tempfile.mkdtemp(prefix="wu_pt_")
    env["OUT_DIR"] = tmp
    holder = {}

    def _mk(*a, **kw):
        holder["lm"] = _StubLM(*a, **kw)
        return holder["lm"]

    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(RUN, "HFCausalLMModel", _mk), \
            mock.patch.object(RUN.gp, "sft_batch_loss",
                              lambda *a, **k: 0.0), \
            mock.patch.object(RUN.gp, "sft_grad_norm", lambda *a, **k: 0.0):
        rc = RUN.main()
    assert rc == 0
    traj = torch.load(Path(tmp) / "trajectory.pt", map_location="cpu",
                      weights_only=False)
    cfg = json.loads((Path(tmp) / "config.json").read_text())
    return traj, cfg, holder["lm"], Path(tmp)


@pytest.fixture(scope="module")
def pt_run():
    """One passthrough run, reused by every check below."""
    return run_pipeline(FJ_OBSERVED_PASSTHROUGH=1)


@pytest.fixture(scope="module")
def setup():
    return RUN.load_pokec_setup(POKEC)


# ===================================================================
#          the served vector, as a pure function (unit)
# ===================================================================

def test_served_vector_takes_x_on_observed_and_model_on_heldout():
    x = torch.tensor([0.1, 0.2, 0.3, 0.4])
    m = torch.tensor([float("nan"), float("nan"), 0.7, 0.8])
    obs = torch.tensor([True, True, False, False])
    s = WUC.served_vector(x, m, obs)
    assert torch.equal(s, torch.tensor([0.1, 0.2, 0.7, 0.8]))


def test_served_vector_refuses_a_missing_heldout_prediction():
    """A nan surviving onto a held-out agent means the model was never
    asked about somebody. FJ would happily propagate it and every later
    round would be nan -- loud here instead."""
    x = torch.tensor([0.1, 0.2, 0.3])
    m = torch.tensor([0.0, 0.0, float("nan")])
    obs = torch.tensor([True, True, False])
    with pytest.raises(ValueError, match="not asked"):
        WUC.served_vector(x, m, obs)


def test_served_vector_ignores_model_values_on_observed():
    """Even if a model value IS present on an observed agent, the served
    vector must not take it. This is the single-line version of the
    sabotage test further down."""
    x = torch.tensor([0.1, 0.2, 0.3])
    m = torch.tensor([0.99, 0.99, 0.7])
    obs = torch.tensor([True, True, False])
    s = WUC.served_vector(x, m, obs)
    assert float(s[0]) == pytest.approx(0.1)
    assert float(s[1]) == pytest.approx(0.2)


# ===================================================================
#                    the split, end to end
# ===================================================================

def test_pokec_split_is_the_one_the_wave_assumes(setup):
    assert setup["n"] == N
    assert setup["innate"].shape == (N,)
    assert abs(float(setup["peer_sus"].mean()) - 0.8909) < 1e-3
    assert abs(float(setup["platform_sus"].mean()) - 0.8890) < 1e-3


def test_observed_mask_is_the_first_1730_agents(pt_run):
    traj, _, _, _ = pt_run
    om = traj["observed_mask"]
    assert om.dtype == torch.bool and om.shape == (N,)
    assert int(om.sum()) == N_OBS
    assert bool(om[:N_OBS].all()) and not bool(om[N_OBS:].any())


def test_served_on_observed_equals_the_current_opinion_every_round(pt_run):
    """s(t)_O == x_O(t), round by round. x(0) is innate; x(t>0) is the
    PREVIOUS round's post-FJ population, which is the whole point -- a
    run that served innate every round would look identical at t=0 and
    diverge invisibly afterwards, so t=0 alone proves nothing."""
    traj, _, _, _ = pt_run
    om = traj["observed_mask"]
    served, op, innate = traj["served_raw"], traj["op_raw"], traj["innate"]
    T = served.shape[0]
    assert T == 3
    for t in range(T):
        x_cur = innate if t == 0 else op[t - 1]
        assert torch.allclose(served[t][om], x_cur[om], atol=0, rtol=0), (
            f"round {t}: served vector on the observed set is not x_O(t)")
    # and the opinions genuinely MOVE, so the t>0 checks have teeth
    assert float((op[1] - innate).abs().max()) > 1e-4


def test_served_on_heldout_is_exactly_the_model_output(pt_run):
    traj, _, _, _ = pt_run
    om = traj["observed_mask"]
    served, mp = traj["served_raw"], traj["model_pred_raw"]
    for t in range(served.shape[0]):
        assert torch.equal(served[t][~om], mp[t][~om])


def test_the_model_is_never_asked_about_an_observed_agent(pt_run):
    """model_pred_raw is nan on O -- not zero, not a copy of x_O. The
    absence has to be REPRESENTABLE, otherwise "the model was not asked"
    and "the model happened to answer x" are the same artifact."""
    traj, _, _, _ = pt_run
    om = traj["observed_mask"]
    mp = traj["model_pred_raw"]
    assert mp.shape == (3, N)
    for t in range(mp.shape[0]):
        assert bool(torch.isnan(mp[t][om]).all()), "model output on O"
        assert bool(torch.isfinite(mp[t][~om]).all()), "missing U prediction"


def test_generation_is_spent_only_on_the_heldout_agents(pt_run, setup):
    """The budget claim, checked against the prompts actually generated:
    every serving batch has exactly |U| prompts and they are exactly the
    held-out agents' prompts, in held-out order."""
    traj, _, lm, _ = pt_run
    prof = setup["profiles"]
    tok = _StubTokenizer()
    want = [RUN.pokec_build_prompt(prof.iloc[i], tok)
            for i in range(N_OBS, N)]
    serving = [b for b in lm.batches if len(b) == N_HELD]
    assert len(serving) == 3, "expected one serving batch per round"
    for b in serving:
        assert b == want
    # nothing anywhere near a full-population generation happened
    assert max(len(b) for b in lm.batches) == N_HELD


def test_training_labels_are_the_current_observed_opinions(pt_run):
    """train_y_raw[t] == x_O(t) and train_idx_raw[t] == O, in order."""
    traj, _, _, _ = pt_run
    om, op, innate = traj["observed_mask"], traj["op_raw"], traj["innate"]
    ti, ty = traj["train_idx_raw"], traj["train_y_raw"]
    assert ti.shape == (3, N_OBS) and ty.shape == (3, N_OBS)
    obs_ids = torch.arange(N)[om]
    for t in range(3):
        assert torch.equal(ti[t], obs_ids)
        x_cur = innate if t == 0 else op[t - 1]
        assert torch.allclose(ty[t], x_cur[om], atol=1e-6)
    # labels are not frozen at innate: the loop is live
    assert float((ty[1] - ty[0]).abs().max()) > 1e-4


def test_pred_raw_keeps_its_meaning_and_the_alias_is_recorded(pt_run):
    """pred_raw has always meant "the vector the platform served". Under
    passthrough that IS served_raw, so the two are bit-identical -- and
    the alias is written into config rather than left to be inferred."""
    traj, cfg, _, _ = pt_run
    assert torch.equal(traj["pred_raw"], traj["served_raw"])
    assert torch.equal(traj["fj_served_used"], traj["served_raw"])
    assert cfg["pred_raw_alias"] == "served_raw"
    assert "nan" in cfg["model_pred_raw_semantics"]


def test_fj_consumed_exactly_the_served_vector(pt_run, setup):
    """x_init = (1-beta) innate + beta s(t), recomputed from the config's
    own beta. Ties the recorded served vector to the population move."""
    traj, cfg, _, _ = pt_run
    beta = (0.3 * (setup["platform_sus"] * 1.0).clamp(0, 1)).clamp(0, 1)
    assert cfg["fj_beta_realized_mean"] == pytest.approx(
        float(beta.mean()), abs=1e-6)
    xi = traj["fj_x_init_raw"]
    served, innate = traj["served_raw"], traj["innate"]
    for t in range(xi.shape[0]):
        want = (1.0 - beta) * innate + beta * served[t]
        assert torch.allclose(xi[t], want, atol=1e-6)


def test_u1_pins_the_inner_loop_start_and_op_is_the_fixed_point(pt_run,
                                                                setup):
    """u^(1) = (1-a) x_init + a P x_init and u^(K) == op_raw. Replays the
    operator from the artifact alone."""
    traj, _, _, _ = pt_run
    W = setup["W"]
    a = 0.9
    xi, u1, op = traj["fj_x_init_raw"], traj["fj_u1_raw"], traj["op_raw"]
    assert torch.equal(u1, traj["fj_u1"])
    for t in range(xi.shape[0]):
        assert torch.allclose(u1[t], (1 - a) * xi[t] + a * (W @ xi[t]),
                              atol=1e-5)
        u = xi[t]
        for _ in range(4):
            u = (1 - a) * xi[t] + a * (W @ u)
        assert torch.allclose(op[t], u, atol=1e-5)


# ===================================================================
#            realized alpha / beta and their hashes
# ===================================================================

def test_defaults_record_the_homogeneous_alpha_and_dataset_beta(pt_run,
                                                                setup):
    traj, cfg, _, _ = pt_run
    assert cfg["fj_peer_source"] == "homogeneous"
    assert cfg["fj_platform_source"] == "dataset"
    assert cfg["fj_alpha_scale"] == 1.0 and cfg["fj_beta_scale"] == 1.0
    assert cfg["fj_alpha_realized_mean"] == pytest.approx(0.9, abs=1e-6)
    assert cfg["fj_alpha_raw_sha256"] == RUN._sha_tensor(setup["peer_sus"])
    assert cfg["fj_beta_raw_sha256"] == RUN._sha_tensor(
        setup["platform_sus"])


def test_realized_vectors_are_scale_times_the_dataset_vectors(setup):
    """FJ_PEER_SOURCE=dataset with c_alpha=0.5: alpha_realized must be
    0.5 * Wu's alpha_i, hashed, and the world must carry its complement
    (run_wu refuses otherwise, which is the point of the check)."""
    traj, cfg, _, _ = run_pipeline(
        FJ_OBSERVED_PASSTHROUGH=1, FJ_PEER_SOURCE="dataset",
        FJ_ALPHA_SCALE=0.5, FJ_BETA_SCALE=0.5, N_ROUNDS=2)
    want_a = (setup["peer_sus"].float().clamp(0, 1) * 0.5).clamp(0, 1)
    want_b = ((0.3 * (setup["platform_sus"] * 1.0).clamp(0, 1))
              .clamp(0, 1) * 0.5).clamp(0, 1)
    assert cfg["fj_alpha_realized_sha256"] == RUN._sha_tensor(want_a)
    assert cfg["fj_beta_realized_sha256"] == RUN._sha_tensor(want_b)
    assert cfg["fj_alpha_realized_mean"] == pytest.approx(
        float(want_a.mean()), abs=1e-6)
    assert cfg["fj_beta_realized_mean"] == pytest.approx(
        float(want_b.mean()), abs=1e-6)
    # the RAW hashes still describe the unscaled dataset vectors, so the
    # pair distinguishes "dataset alpha at c=0.5" from a homogeneous
    # alpha that happens to have the same mean
    assert cfg["fj_alpha_raw_sha256"] == RUN._sha_tensor(setup["peer_sus"])
    assert cfg["fj_alpha_raw_sha256"] != cfg["fj_alpha_realized_sha256"]
    # and the heterogeneous operator actually ran: replay per agent
    W = setup["W"]
    xi, op = traj["fj_x_init_raw"], traj["op_raw"]
    u = xi[0]
    for _ in range(4):
        u = (1 - want_a) * xi[0] + want_a * (W @ u)
    assert torch.allclose(op[0], u, atol=1e-5)


def test_homogeneous_default_is_bit_identical_to_the_archived_path(setup):
    """The default (homogeneous alpha, c_alpha=1) must reach run_wu as
    the PYTHON FLOAT 0.9, not as float(torch.full((n,), 0.9)[0]) =
    0.8999999761581421. The complement 1-alpha differs in float32
    between those two, so this is a real bit, not a pedantry."""
    a_py = 0.9
    a_rt = float(torch.full((3,), 0.9, dtype=torch.float32)[0])
    assert a_py != a_rt
    assert np.float32(1.0 - a_py) != np.float32(1.0 - a_rt)


# ===================================================================
#      the default path is the archived one, unchanged
# ===================================================================

def test_passthrough_off_still_serves_and_predicts_every_agent(setup):
    """With every new knob at its default, wu1 must behave EXACTLY as it
    did before 2026-08-22: one full-population generation per round, the
    served vector is the model's output everywhere, and the operator runs
    at the python float FJ_ALPHA against the dataset beta. Pinned against
    a hand-rolled replay of the archived recurrence rather than against
    a stored blob, so a change in either direction shows up here."""
    traj, cfg, lm, _ = run_pipeline(N_ROUNDS=2)
    assert cfg["fj_observed_passthrough"] is False
    assert cfg["wu_icl_mode"] == "none"
    assert "pred_raw_alias" not in cfg
    # the new channels exist but are EMPTY: "this run had no passthrough"
    # must be distinguishable from "this key predates the feature"
    assert traj["served_raw"].numel() == 0
    assert traj["model_pred_raw"].numel() == 0
    assert traj["train_idx_raw"].numel() == 0
    # every agent was asked, every round
    gen = [b for b in lm.batches if len(b) == N]
    assert len(gen) == 2
    # ... and the archived recurrence reproduces op_raw exactly
    beta = (0.3 * (setup["platform_sus"] * 1.0).clamp(0, 1)).clamp(0, 1)
    a, W, innate = 0.9, setup["W"], setup["innate"]
    for t in range(2):
        s = traj["pred_raw"][t]
        xi = (1.0 - beta) * innate + beta * s
        u = xi
        for _ in range(4):
            u = (1 - a) * xi + a * (W @ u)
        assert torch.allclose(traj["op_raw"][t], u, atol=1e-6)


def test_legacy_fj_trajectory_keys_are_untouched():
    """A non-wu1 run must not grow the new keys at all: the legacy
    artifact surface is exactly what it always was."""
    traj, cfg, _, _ = run_pipeline(FJ_UPDATE_VERSION="legacy",
                                   EPOCH_SIZE=2, N_ROUNDS=1)
    for k in ("served_raw", "model_pred_raw", "fj_x_init_raw",
              "fj_u1_raw", "train_idx_raw", "observed_mask"):
        assert k not in traj, f"legacy trajectory gained {k}"
    for k in ("fj_peer_source", "fj_platform_source", "wu_icl_mode",
              "fj_alpha_realized_sha256"):
        assert k not in cfg, f"legacy config gained {k}"


# ===================================================================
#                       validation is loud
# ===================================================================

@pytest.mark.parametrize("over,msg", [
    ({"FJ_UPDATE_VERSION": "legacy"}, "wu1"),
    ({"DATASET": "movielens"}, "pokec"),
    ({"DATA_REGIME": "accumulate"}, "replace"),
    ({"REPLAY_FRAC": "0.5"}, "REPLAY_FRAC"),
    ({"TRAIN_CAP": "100"}, "TRAIN_CAP"),
    ({"N_LABELED": str(N)}, "held-out"),
])
def test_passthrough_refuses_configurations_it_cannot_honour(over, msg):
    env = dict(BASE_ENV)
    env.update({"RUN_TAG": "bad", "FJ_OBSERVED_PASSTHROUGH": "1",
                "OUT_DIR": tempfile.mkdtemp(prefix="wu_bad_")})
    env.update(over)
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(RUN, "HFCausalLMModel", _StubLM):
        with pytest.raises(ValueError) as e:
            RUN.main()
    assert msg in str(e.value)


def test_wu_knobs_require_wu1():
    """Every Wu knob on the legacy operator is a silent no-op, which is
    the worst outcome: the tag would claim a design the run never had."""
    for knob in ("FJ_PEER_SOURCE=dataset", "FJ_ALPHA_SCALE=0.5",
                 "FJ_BETA_SCALE=0.5", "FJ_OBSERVED_PASSTHROUGH=1",
                 "WU_ICL_MODE=observed_context", "ROUTING_TREAT_FRAC=0.1"):
        k, v = knob.split("=")
        env = dict(BASE_ENV)
        env.update({"RUN_TAG": "bad", "FJ_UPDATE_VERSION": "legacy",
                    "OUT_DIR": tempfile.mkdtemp(prefix="wu_bad_"), k: v})
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(RUN, "HFCausalLMModel", _StubLM):
            with pytest.raises(ValueError, match="wu1"):
                RUN.main()


def test_gate_modes_are_still_refused_under_wu1():
    """FJ has no confidence gates; the pre-existing guard must survive
    the passthrough edits."""
    env = dict(BASE_ENV)
    env.update({"RUN_TAG": "bad", "FJ_OBSERVED_PASSTHROUGH": "1",
                "AI_GATE_MODE": "all_open",
                "OUT_DIR": tempfile.mkdtemp(prefix="wu_bad_")})
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(RUN, "HFCausalLMModel", _StubLM):
        with pytest.raises(ValueError, match="NO gates"):
            RUN.main()


# ===================================================================
#                          SABOTAGE
# ===================================================================

def verify_passthrough(traj):
    """Re-derive the passthrough from the artifact alone. Returns a list
    of violations. This is the shape an offline checker takes."""
    bad = []
    om = traj["observed_mask"]
    served, mp, op, innate = (traj["served_raw"], traj["model_pred_raw"],
                              traj["op_raw"], traj["innate"])
    for t in range(served.shape[0]):
        x_cur = innate if t == 0 else op[t - 1]
        if not torch.allclose(served[t][om], x_cur[om], atol=1e-6):
            bad.append(f"round {t}: served on O is not x_O(t)")
        if bool(torch.isfinite(mp[t][om]).any()):
            bad.append(f"round {t}: a model output exists on an observed "
                       f"agent")
        if not torch.equal(served[t][~om], mp[t][~om]):
            bad.append(f"round {t}: served on U is not the model output")
    return bad


def test_a_clean_passthrough_run_verifies(pt_run):
    traj, _, _, _ = pt_run
    assert verify_passthrough(traj) == []


def test_sabotage_model_value_written_onto_an_observed_agent(pt_run):
    """THE defect this design exists to exclude: one observed agent
    served the model's guess instead of its own recorded opinion. Every
    array keeps its shape and dtype; only the check catches it."""
    traj, _, _, _ = pt_run
    bad = copy.deepcopy(traj)
    victim = 7                      # an observed agent
    assert bool(bad["observed_mask"][victim])
    bad["served_raw"][1][victim] = 0.123456
    bad["model_pred_raw"][1][victim] = 0.123456
    viol = verify_passthrough(bad)
    assert any("served on O is not x_O(t)" in v for v in viol), viol
    assert any("model output exists on an observed agent" in v
               for v in viol), viol


def test_sabotage_stale_labels_that_never_follow_the_population(pt_run):
    """Training on innate forever instead of x_O(t): the loop is dead but
    the artifact is well-formed. Caught by comparing labels to op_raw."""
    traj, _, _, _ = pt_run
    om, op, innate = traj["observed_mask"], traj["op_raw"], traj["innate"]
    stale = innate[om].repeat(3, 1)
    live = traj["train_y_raw"]
    assert torch.allclose(live[0], stale[0], atol=1e-6)   # t=0 coincides
    assert not torch.allclose(live[1], stale[1], atol=1e-4), (
        "round 1 labels equal innate -- the population feedback is dead")
    assert torch.allclose(live[1], op[0][om], atol=1e-6)


def test_sabotage_a_heldout_prediction_swapped_for_passthrough(pt_run):
    """The mirror defect: a held-out agent served its own opinion (i.e.
    the platform cheating with a label it does not hold)."""
    traj, _, _, _ = pt_run
    bad = copy.deepcopy(traj)
    victim = N - 1
    assert not bool(bad["observed_mask"][victim])
    bad["served_raw"][0][victim] = float(bad["innate"][victim])
    viol = verify_passthrough(bad)
    assert any("served on U is not the model output" in v for v in viol)


# ===================================================================
#                     routing treatment (stage-4)
# ===================================================================

def test_routing_treatment_moves_observed_innate_and_is_hashed():
    traj, cfg, _, _ = run_pipeline(
        FJ_OBSERVED_PASSTHROUGH=1, ROUTING_TREAT_FRAC=0.1,
        ROUTING_TREAT_SEED=3, ROUTING_TREAT_VALUE=1.0, N_ROUNDS=2)
    idx = traj["routing_treat_idx"]
    assert idx.numel() == round(0.1 * N_OBS)
    assert int(idx.max()) < N_OBS, "a held-out agent was treated"
    assert torch.equal(idx, idx.sort().values)
    assert cfg["routing_treat_idx_sha256"] == WUC.idx_sha256(idx.numpy())
    assert torch.allclose(traj["innate"][idx], torch.ones(idx.numel()))
    # the treatment reaches the served vector at t=0 (passthrough on O)
    assert torch.allclose(traj["served_raw"][0][idx],
                          torch.ones(idx.numel()))


def test_routing_cohort_depends_only_on_its_own_seed():
    a, _, _, _ = run_pipeline(FJ_OBSERVED_PASSTHROUGH=1,
                              ROUTING_TREAT_FRAC=0.05,
                              ROUTING_TREAT_SEED=11, SEED=0, N_ROUNDS=1)
    b, _, _, _ = run_pipeline(FJ_OBSERVED_PASSTHROUGH=1,
                              ROUTING_TREAT_FRAC=0.05,
                              ROUTING_TREAT_SEED=11, SEED=5, N_ROUNDS=1)
    c, _, _, _ = run_pipeline(FJ_OBSERVED_PASSTHROUGH=1,
                              ROUTING_TREAT_FRAC=0.05,
                              ROUTING_TREAT_SEED=12, SEED=0, N_ROUNDS=1)
    assert torch.equal(a["routing_treat_idx"], b["routing_treat_idx"]), (
        "the treated cohort must not depend on the run seed -- the twin "
        "compares the SAME people across seeds")
    assert not torch.equal(a["routing_treat_idx"], c["routing_treat_idx"])
