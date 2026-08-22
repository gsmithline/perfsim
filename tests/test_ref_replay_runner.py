"""REF_REPLAY_*: reference-replay training labels (2026-08-22).

THE DESIGN
----------
Every deploy round the learner is handed a FULL n-row training set

    y_i^(t) = x_i(t)   if i in S_t   (live: this round's real opinion)
              b_i      otherwise      (frozen reference prediction)

with |S_t| = round(q n) fixed, S_t redrawn EVERY round (round 0
included), and b pinned to another run's pred_raw[0].

Every arm holds all n rows, so the optimizer-step count and the compute
are identical at every q and the only thing q varies is the LABEL
SOURCE. q is a dose of live population feedback, not a dose of data.

WHY EACH PROPERTY NEEDS A TEST
------------------------------
Every way of getting this wrong produces a well-formed trajectory:

  * ACCUMULATION. Substituting into last round's label vector instead of
    rebuilding from b turns the reference arm into a slowly-drifting
    self-training arm. The row counts, the shapes and the value ranges
    are all still right, and the q=0.1 arm quietly becomes a q=1 arm run
    at a delay.
  * REORDERING. Sorting the batch by live/non-live puts agent 40's label
    on agent 3's prompt. The label VECTOR is still exactly right as a
    multiset, so any check that compares sets passes.
  * A NON-NESTED live set. If the sample depended on q (or on a
    generator the loop advances), the arms would differ in WHO is live as
    well as HOW MANY, and q would no longer be a dose along one axis.
  * SHRINKING the batch to the live rows. That is the obvious
    implementation, and it changes the optimizer-step count with q, which
    is exactly the confound the full-batch design exists to remove.

So each is asserted directly AND has a sabotage twin that builds the
defective artifact and shows the check fires.

NO MODEL IS LOADED. The LM is a stub whose generation is a deterministic
function of the prompt string, and the SFT learner is a stub that records
the batch it is handed.

Run with USE_TF=0.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
PIPE = REPO / "experiments" / "scripts" / "cluster_pipelines"
ML_DIR = REPO / "experiments" / "data" / "movielens" / "ml-100k"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gp = _load("_gated_pop_refreplay", PIPE / "_gated_pop.py")
RUN = _load("run_ref_replay", PIPE / "run_pokec_gated_lm.py")

# the pilot surface: movielens / Action, 723 agents, batch 4, one epoch
N = 723
BATCH = 4
STEPS_PER_ROUND = math.ceil(N / BATCH)          # 181


# ===================================================================
#                        the stubs
# ===================================================================

def stub_value(prompt: str) -> float:
    h = int(hashlib.sha1(prompt.encode()).hexdigest()[:8], 16)
    return round(0.02 + 0.96 * (h % 9973) / 9973.0, 4)


class _StubTokenizer:
    chat_template = "stub"
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        return "<|u|>" + messages[0]["content"] + "<|a|>"


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

    def build_prompt(self, profile, context_block=None):
        if context_block is None:
            return self._prompt_builder(profile, self.tokenizer)
        return self._prompt_builder(profile, self.tokenizer,
                                    context_block=context_block)

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


class _StubSFTLearner:
    """Records the EXACT (ids, labels) of every batch handed to train().

    No optimizer runs -- the point is the batch, since with a fixed batch
    size and epoch count the row count is what determines the step
    count."""

    seen: list = []          # class-level: survives the runner's scope

    def __init__(self, model=None, loss=None, per_device_batch_size=4, **kw):
        self.model = model
        self._batch = int(per_device_batch_size)
        self.last_train_stats = {}

    def train(self, data):
        y = data["y"]
        y = (y.squeeze(-1) if y.ndim > 1 else y)
        rows = int(y.shape[0])
        _StubSFTLearner.seen.append({
            "idx": data["agent_idx"].detach().cpu().long().clone(),
            "y": y.detach().cpu().float().clone(),
            "rows": rows,
            "steps": math.ceil(rows / self._batch),
        })
        self.last_train_stats = {"n_rows": rows, "global_step": 0,
                                 "per_device_batch_size": self._batch}

    def reset(self):
        pass


BASE_ENV = {
    "PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""),
    "USE_TF": "0",
    "DATASET": "movielens", "ML_DIR": str(ML_DIR), "ML_TARGET": "Action",
    "POP_MODEL": "ab", "RUN_MODE": "loop", "TRAINING_STYLE": "frozen",
    "DATA_REGIME": "replace", "USE_LORA": "0", "DEVICE": "cpu", "SEED": "0",
    "N_ROUNDS": "4", "DEPLOY_EVERY": "1", "N_LABELED": str(N),
    # the pilot surface: W=1, k=1, BOTH gates all_open (the QWU corner)
    "W_PLAT": "1", "INNATE_LAMBDA": "1",
    "AI_GATE_MODE": "all_open", "PEER_GATE_MODE": "all_open",
    "EPS": "0.2", "EPS_AI": "0.2", "GAMMA_BIAS": "0",
    "SFT_EPOCHS": "1", "SFT_BATCH_SIZE": str(BATCH),
    "N_PROBE": "8", "LOG_PERPLEXITY": "0", "LOG_ANSWER_DIST": "0",
    "LOG_PPL_DIST": "0", "GRAD_NORM_N": "0", "TEL_EVAL_CAP": "0",
}


def run_pipeline(**overrides):
    env = dict(BASE_ENV)
    env["RUN_TAG"] = "ref_replay_test"
    env.update({k: str(v) for k, v in overrides.items()})
    tmp = tempfile.mkdtemp(prefix="ref_replay_")
    env["OUT_DIR"] = tmp
    holder = {}

    def _mk(*a, **kw):
        holder["lm"] = _StubLM(*a, **kw)
        return holder["lm"]

    _StubSFTLearner.seen = []
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(RUN, "HFCausalLMModel", _mk), \
            mock.patch.object(RUN, "SFTLearner", _StubSFTLearner), \
            mock.patch.object(RUN.gp, "sft_batch_loss", lambda *a, **k: 0.0), \
            mock.patch.object(RUN.gp, "sft_grad_norm", lambda *a, **k: 0.0):
        rc = RUN.main()
        seen = list(_StubSFTLearner.seen)
    assert rc == 0
    traj = torch.load(Path(tmp) / "trajectory.pt", map_location="cpu",
                      weights_only=False)
    cfg = json.loads((Path(tmp) / "config.json").read_text())
    return traj, cfg, seen, Path(tmp)


@pytest.fixture(scope="module")
def ref_run():
    """The run that SUPPLIES b. Plain, no ref-replay: b is its
    pred_raw[0], i.e. the frozen model's round-0 predictions."""
    traj, cfg, _, out = run_pipeline(RUN_TAG="ref_donor", N_ROUNDS=1)
    return traj, out


@pytest.fixture(scope="module")
def b_vec(ref_run):
    return ref_run[0]["pred_raw"][0].float().clone()


@pytest.fixture(scope="module")
def q10(ref_run):
    return run_pipeline(REF_REPLAY_Q=0.10, REF_REPLAY_SEED=7,
                        REF_REPLAY_REF_RUN=str(ref_run[1]))


@pytest.fixture(scope="module")
def q20(ref_run):
    return run_pipeline(REF_REPLAY_Q=0.20, REF_REPLAY_SEED=7,
                        REF_REPLAY_REF_RUN=str(ref_run[1]))


def x_of_round(traj, t):
    """x(t): the labels the ordinary pipeline would have used at round t
    -- innate at t=0, the PRECEDING population afterwards."""
    return traj["innate"] if t == 0 else traj["op_raw"][t - 1]


# ===================================================================
#  1. the sampler (unit): stateless, nested, refreshed
# ===================================================================

def test_live_count_is_round_q_n_and_independent_of_the_round():
    """The pilot ladder, pinned: {.10, .20, .50, .75, 1} at N=723 must
    give exactly {72, 145, 362, 542, 723}. The offline checker refuses
    off-ladder q rather than rounding it, so the two roundings have to
    agree exactly."""
    assert gp.ref_replay_n_live(N, 0.10) == 72
    assert gp.ref_replay_n_live(N, 0.20) == 145
    assert gp.ref_replay_n_live(N, 0.50) == 362
    assert gp.ref_replay_n_live(N, 0.75) == 542
    assert gp.ref_replay_n_live(N, 1.0) == 723 == N
    for t in (0, 1, 17, 99):
        assert gp.ref_replay_live(N, 0.2, 3, t).numel() == 145


def test_the_permutation_is_stateless_in_seed_and_round():
    """Called twice, in any order, interleaved with other RNG use: the
    same (seed, round) must give the same permutation. This is what makes
    the arms comparable at all."""
    a = gp.ref_replay_perm(N, 7, 5)
    torch.manual_seed(999)
    torch.rand(1000)
    b = gp.ref_replay_perm(N, 7, 5)
    assert torch.equal(a, b)
    assert not torch.equal(a, gp.ref_replay_perm(N, 7, 6))
    assert not torch.equal(a, gp.ref_replay_perm(N, 8, 5))


def test_the_permutation_is_the_projects_standard_selection_stream():
    """torch.Generator().manual_seed(seed + t) -> randperm(N): the same
    one-liner SFT_SAMPLE_N uses and check_pofd_sanity already replays.
    Pinned here because an offline checker reconstructs the live set
    with exactly this expression -- a bespoke hash on either side would
    make every arm unverifiable."""
    for seed, t in ((7, 0), (7, 3), (0, 11)):
        g = torch.Generator().manual_seed(seed + t)
        assert torch.equal(gp.ref_replay_perm(N, seed, t),
                           torch.randperm(N, generator=g))


def test_live_sets_are_nested_across_q_at_a_fixed_round():
    """q=.10 is a literal PREFIX of q=.20 of q=.50 of q=1, at EVERY
    round. Nesting is what makes 'more feedback' a dose along one axis
    rather than a different experiment."""
    for t in (0, 1, 2, 9):
        prev = None
        for q in (0.05, 0.10, 0.20, 0.50, 1.0):
            cur = gp.ref_replay_live(N, q, 7, t)
            if prev is not None:
                assert torch.equal(prev, cur[:prev.numel()]), (
                    f"round {t}: q={q} is not a superset-prefix of the "
                    f"previous q")
                assert set(prev.tolist()) <= set(cur.tolist())
            prev = cur


def test_nesting_holds_across_arms_because_q_never_enters_the_draw():
    """The permutation is a function of (seed, round) ALONE. Two
    independently launched arms therefore share the same live agents at
    the same round -- not merely the same COUNT."""
    import inspect
    src = inspect.getsource(gp.ref_replay_perm)
    assert " q" not in src.split('"""')[-1], (
        "the permutation must not read q at all")
    for t in range(4):
        small = gp.ref_replay_live(N, 0.10, 7, t)
        big = gp.ref_replay_live(N, 0.50, 7, t)
        assert torch.equal(small, big[:small.numel()])


def test_live_set_refreshes_every_round_including_round_zero():
    sets = [gp.ref_replay_live(N, 0.2, 7, t) for t in range(5)]
    for t in range(4):
        assert not torch.equal(sets[t], sets[t + 1]), f"round {t} == {t+1}"
    # round 0 is a draw like any other, NOT the full set
    assert sets[0].numel() == 145 < N


def test_q_one_is_every_agent():
    live = gp.ref_replay_live(N, 1.0, 7, 3)
    assert live.numel() == N
    assert set(live.tolist()) == set(range(N))


def test_degenerate_q_is_an_error_not_a_silent_all_reference_run():
    with pytest.raises(ValueError, match="REF_REPLAY_Q"):
        gp.ref_replay_n_live(N, 0.0)
    with pytest.raises(ValueError, match="REF_REPLAY_Q"):
        gp.ref_replay_n_live(N, 1.5)
    with pytest.raises(ValueError, match="degenerate"):
        gp.ref_replay_n_live(10, 0.001)


# ===================================================================
#  2. the label builder (unit): rebuilt from b, canonical order
# ===================================================================

def test_labels_are_x_on_live_and_b_elsewhere():
    x = torch.linspace(0.0, 1.0, 20)
    b = torch.full((20,), 0.375)
    live = torch.tensor([3, 11, 0])
    y = gp.ref_replay_labels(x, b, live)
    assert torch.equal(y[live], x[live])
    other = torch.tensor([i for i in range(20) if i not in (0, 3, 11)])
    assert torch.equal(y[other], b[other])


def test_labels_are_rebuilt_from_b_and_cannot_accumulate():
    """Feed the PREVIOUS round's labels back in as if a careless
    implementation had: the output is still b on the non-live rows,
    because the base is always the original b."""
    x = torch.linspace(0.0, 1.0, 20)
    b = torch.full((20,), 0.375)
    y1 = gp.ref_replay_labels(x, b, torch.tensor([1, 2, 3]))
    y2 = gp.ref_replay_labels(x, b, torch.tensor([4, 5]))
    assert float(y2[1]) == float(y2[2]) == float(y2[3]) == 0.375
    assert torch.equal(y2, gp.ref_replay_labels(x, b, torch.tensor([4, 5])))
    # and b itself is never mutated in place
    assert torch.equal(b, torch.full((20,), 0.375))
    assert torch.equal(y1[torch.tensor([1, 2, 3])],
                       x[torch.tensor([1, 2, 3])])


def test_labels_keep_canonical_agent_order():
    """Row i is agent i, whatever order the live indices arrive in."""
    x = torch.linspace(0.0, 1.0, 20)
    b = torch.zeros(20)
    shuffled = torch.tensor([17, 2, 9, 0])
    y = gp.ref_replay_labels(x, b, shuffled)
    for i in shuffled.tolist():
        assert float(y[i]) == pytest.approx(float(x[i]))
    assert y.shape == (20,)


def test_reference_vector_validation_is_loud():
    good = torch.rand(N)
    gp.validate_ref_replay_vec(good, n=N)
    with pytest.raises(ValueError, match="non-finite"):
        bad = good.clone(); bad[5] = float("nan")
        gp.validate_ref_replay_vec(bad, n=N)
    with pytest.raises(ValueError, match=r"out of \[0, 1\]"):
        bad = good.clone(); bad[5] = 1.5
        gp.validate_ref_replay_vec(bad, n=N)
    with pytest.raises(ValueError, match="agents, population has"):
        gp.validate_ref_replay_vec(good[:10], n=N)
    with pytest.raises(ValueError, match="1-D"):
        gp.validate_ref_replay_vec(good.unsqueeze(-1), n=N)


# ===================================================================
#  3. end to end: config, artifacts, and the labels actually trained on
# ===================================================================

def test_config_records_the_contract_fields(q10, b_vec, ref_run):
    _, cfg, _, _ = q10
    assert cfg["ref_replay_q"] == 0.10
    assert cfg["ref_replay_seed"] == 7
    assert cfg["ref_replay_n_live"] == 72
    assert cfg["ref_replay_ref_run"] == str(ref_run[1])
    assert cfg["ref_replay_ref_sha256"] == gp.ref_replay_hash(b_vec)
    assert len(cfg["ref_replay_ref_sha256"]) == 64


def test_a_plain_run_carries_none_of_it(ref_run):
    traj, cfg, _, _ = run_pipeline(N_ROUNDS=1)
    for key in ("ref_replay_q", "ref_replay_seed", "ref_replay_n_live",
                "ref_replay_ref_run", "ref_replay_ref_sha256"):
        assert key not in cfg, f"plain config gained {key}"
    for key in ("ref_replay_live_idx", "ref_replay_labels",
                "ref_replay_ref_vec"):
        assert key not in traj, f"plain trajectory gained {key}"


def test_artifact_shapes_are_the_contract(q10, b_vec):
    traj, cfg, _, _ = q10
    T = traj["op_raw"].shape[0]
    assert tuple(traj["ref_replay_live_idx"].shape) == (T, 72)
    assert tuple(traj["ref_replay_labels"].shape) == (T, N)
    assert tuple(traj["ref_replay_ref_vec"].shape) == (N,)
    assert torch.equal(traj["ref_replay_ref_vec"], b_vec)


def test_recorded_live_sets_are_the_stateless_reconstruction(q10):
    traj, cfg, _, _ = q10
    for t in range(traj["ref_replay_live_idx"].shape[0]):
        want = gp.ref_replay_live(N, cfg["ref_replay_q"],
                                  cfg["ref_replay_seed"], t)
        assert torch.equal(traj["ref_replay_live_idx"][t].long(), want), t


def test_labels_equal_x_on_live_and_b_on_the_rest_every_round(q10, b_vec):
    """The central claim, checked round by round against the population
    the run actually produced."""
    traj, _, _, _ = q10
    live_all = traj["ref_replay_live_idx"].long()
    lab = traj["ref_replay_labels"]
    for t in range(lab.shape[0]):
        x = x_of_round(traj, t)
        live = live_all[t]
        nonlive = torch.tensor([i for i in range(N)
                                if i not in set(live.tolist())])
        assert torch.equal(lab[t][live], x[live].float()), f"round {t} live"
        assert torch.equal(lab[t][nonlive], b_vec[nonlive]), \
            f"round {t} non-live"


def test_non_live_labels_never_accumulate_over_many_rounds(ref_run, b_vec):
    """The defect this design's ONE-LINE implementation choice exists to
    exclude, checked where it would actually show: an agent that was LIVE
    in an early round and is not live later must carry b again, not the
    opinion it contributed back then.

    Eight rounds so that at q=.2 nearly every agent has been live at
    least once -- a 2-round run cannot distinguish 'rebuilt from b' from
    'nothing has drifted yet'."""
    traj, _, _, _ = run_pipeline(N_ROUNDS=8, REF_REPLAY_Q=0.2,
                                 REF_REPLAY_SEED=5,
                                 REF_REPLAY_REF_RUN=str(ref_run[1]))
    lab, live_all = traj["ref_replay_labels"], \
        traj["ref_replay_live_idx"].long()
    ever_live = set()
    n_checked = 0
    for t in range(lab.shape[0]):
        live = set(live_all[t].tolist())
        # agents that WERE live earlier and are not live now
        lapsed = sorted(ever_live - live)
        for i in lapsed:
            assert float(lab[t][i]) == float(b_vec[i]), (
                f"round {t}: agent {i} was live earlier and now carries "
                f"{float(lab[t][i]):.6f} instead of b={float(b_vec[i]):.6f} "
                f"-- the substitution accumulated")
        n_checked += len(lapsed)
        ever_live |= live
    assert n_checked > 500, (f"only {n_checked} lapsed-agent checks -- the "
                             f"fixture does not exercise accumulation")
    # the population really did move, so 'label == b' is a real constraint
    assert float((traj["op_raw"][-1] - traj["innate"]).abs().max()) > 1e-4


def test_the_labels_on_record_are_the_labels_the_learner_got(ref_run):
    """ref_replay_labels must be the vector actually handed to train(),
    not a parallel reconstruction that could drift from it."""
    traj, cfg, seen, _ = run_pipeline(
        TRAINING_STYLE="sft", REF_REPLAY_Q=0.2, REF_REPLAY_SEED=7,
        REF_REPLAY_REF_RUN=str(ref_run[1]))
    assert len(seen) == traj["ref_replay_labels"].shape[0] > 0
    for t, batch in enumerate(seen):
        assert torch.equal(batch["y"], traj["ref_replay_labels"][t])
        assert torch.equal(batch["idx"], torch.arange(N)), \
            f"round {t}: the batch is not in canonical agent order"


# ===================================================================
#  4. compute is matched across q
# ===================================================================

@pytest.mark.parametrize("q", [0.1, 0.5, 1.0])
def test_every_arm_trains_on_all_723_rows_with_a_fixed_step_count(q, ref_run):
    """The whole point of the full-batch design. A shrink-to-live-rows
    implementation would give 72 / 362 / 723 rows and 18 / 91 / 181
    steps, and the q effect would be confounded with an update-dose
    effect."""
    traj, cfg, seen, _ = run_pipeline(
        TRAINING_STYLE="sft", REF_REPLAY_Q=q, REF_REPLAY_SEED=7,
        REF_REPLAY_REF_RUN=str(ref_run[1]))
    assert len(seen) == 4
    for t, batch in enumerate(seen):
        assert batch["rows"] == N, f"q={q} round {t}: {batch['rows']} rows"
        assert batch["steps"] == STEPS_PER_ROUND == 181
    assert traj["ref_replay_labels"].shape == (4, N)


def test_q_is_a_label_dose_not_a_data_dose(ref_run):
    """Stated as a comparison: two arms, same rows, same steps, different
    labels."""
    kw = dict(TRAINING_STYLE="sft", REF_REPLAY_SEED=7,
              REF_REPLAY_REF_RUN=str(ref_run[1]))
    _, _, lo, _ = run_pipeline(REF_REPLAY_Q=0.1, **kw)
    _, _, hi, _ = run_pipeline(REF_REPLAY_Q=0.9, **kw)
    assert [b["rows"] for b in lo] == [b["rows"] for b in hi] == [N] * 4
    assert [b["steps"] for b in lo] == [b["steps"] for b in hi]
    assert not torch.equal(lo[1]["y"], hi[1]["y"])


# ===================================================================
#  5. q = 1 is EXACTLY ordinary SFT
# ===================================================================

def test_q_one_labels_are_exactly_the_ordinary_labels(ref_run):
    traj, _, _, _ = run_pipeline(REF_REPLAY_Q=1.0, REF_REPLAY_SEED=7,
                                 REF_REPLAY_REF_RUN=str(ref_run[1]))
    lab = traj["ref_replay_labels"]
    for t in range(lab.shape[0]):
        assert torch.equal(lab[t], x_of_round(traj, t).float()), f"round {t}"


def test_q_one_reproduces_the_plain_run_bit_for_bit(ref_run):
    """So the existing q=1 trajectory is REUSABLE rather than merely
    comparable: no extra RNG is drawn, no label is perturbed, and the
    whole population path is identical to a run with the feature off."""
    plain, cfg_p, _, _ = run_pipeline()
    q1, cfg_q, _, _ = run_pipeline(REF_REPLAY_Q=1.0, REF_REPLAY_SEED=7,
                                   REF_REPLAY_REF_RUN=str(ref_run[1]))
    for key in ("op_raw", "pred_raw", "gate_raw", "twin_raw", "innate"):
        assert torch.equal(plain[key], q1[key]), key
    added = {k for k in cfg_q if k not in cfg_p}
    assert added == {"ref_replay_q", "ref_replay_seed", "ref_replay_n_live",
                     "ref_replay_ref_run", "ref_replay_ref_sha256"}, added
    assert {k for k in cfg_p if cfg_p[k] != cfg_q.get(k)} == set()


def test_a_partial_q_really_substitutes_labels(ref_run, b_vec):
    """Guard on the test above: if ref-replay never changed anything, the
    q=1 identity would be trivially true. At q=.1 exactly 651 of the 723
    rows must carry b, and most of them must therefore differ from the
    live opinion.

    ("most", not "all": b can coincide with x_i(t) for an agent the peer
    step happened to leave on it, and a strict 651 would be asserting the
    population never lands on its served value.)"""
    part, _, _, _ = run_pipeline(TRAINING_STYLE="sft", REF_REPLAY_Q=0.1,
                                 REF_REPLAY_SEED=7,
                                 REF_REPLAY_REF_RUN=str(ref_run[1]))
    x1 = x_of_round(part, 1).float()
    lab1 = part["ref_replay_labels"][1]
    live = set(part["ref_replay_live_idx"][1].long().tolist())
    nonlive = torch.tensor([i for i in range(N) if i not in live])
    assert nonlive.numel() == N - 72
    assert torch.equal(lab1[nonlive], b_vec[nonlive])
    assert int((lab1[nonlive] != x1[nonlive]).sum()) > (N - 72) // 2


# ===================================================================
#  6. every exclusivity guard fires
# ===================================================================

@pytest.mark.parametrize("over,msg", [
    ({"REPLAY_FRAC": "0.5"}, "REPLAY_FRAC"),
    ({"DATA_REGIME": "accumulate", "PRISTINE_FRAC": "0.5"}, "PRISTINE_FRAC"),
    ({"SFT_SAMPLE_N": "100", "TRAINING_STYLE": "sft"}, "SFT_SAMPLE_N"),
    ({"TRAIN_CAP": "100"}, "TRAIN_CAP"),
    ({"DATA_REGIME": "accumulate"}, "DATA_REGIME"),
])
def test_mutually_exclusive_knobs_raise(over, msg, ref_run):
    with pytest.raises(ValueError, match=msg):
        run_pipeline(REF_REPLAY_Q=0.2, REF_REPLAY_SEED=7,
                     REF_REPLAY_REF_RUN=str(ref_run[1]), **over)


def test_train_cap_at_or_above_the_pool_is_allowed(ref_run):
    """The predicate is `0 < TRAIN_CAP < N_LABELED`, matching the rest of
    the codebase: subsample_train_data returns the pool unchanged when
    n <= cap, so TRAIN_CAP=723 is a provable no-op and rejecting it would
    be a false positive on generated subs that set it to document intent."""
    traj, _, _, _ = run_pipeline(TRAIN_CAP=N, REF_REPLAY_Q=0.2,
                                 REF_REPLAY_SEED=7,
                                 REF_REPLAY_REF_RUN=str(ref_run[1]),
                                 N_ROUNDS=2)
    assert traj["ref_replay_labels"].shape == (2, N)


@pytest.mark.parametrize("over,msg", [
    ({"REF_REPLAY_Q": "0.2"}, "REF_REPLAY_REF_RUN"),
    ({"REF_REPLAY_Q": "1.5"}, "REF_REPLAY_Q"),
    ({"REF_REPLAY_Q": "-0.1"}, "REF_REPLAY_Q"),
])
def test_bad_ref_replay_settings_raise(over, msg):
    with pytest.raises(ValueError, match=msg):
        run_pipeline(**over)


def test_a_set_but_inert_knob_is_refused(ref_run):
    """REF_REPLAY_REF_RUN without REF_REPLAY_Q would be a silent no-op --
    the run tag would claim a design the run never had."""
    with pytest.raises(ValueError, match="silent no-op"):
        run_pipeline(REF_REPLAY_REF_RUN=str(ref_run[1]))


def test_dpo_is_refused_because_the_labels_are_the_judge_there(ref_run):
    with pytest.raises(ValueError, match="dpo"):
        run_pipeline(TRAINING_STYLE="dpo", REF_REPLAY_Q=0.2,
                     REF_REPLAY_SEED=7,
                     REF_REPLAY_REF_RUN=str(ref_run[1]))


def test_a_missing_reference_run_is_refused():
    with pytest.raises(ValueError, match="no trajectory.pt"):
        run_pipeline(REF_REPLAY_Q=0.2, REF_REPLAY_SEED=7,
                     REF_REPLAY_REF_RUN="/nonexistent/run/dir")


def test_a_reference_vector_of_the_wrong_length_is_refused(ref_run, tmp_path):
    """Caught against the REALIZED population, not against a hardcoded
    723: a b from another dataset must not become 723 labels by luck."""
    d = torch.load(ref_run[1] / "trajectory.pt", map_location="cpu",
                   weights_only=False)
    d["pred_raw"] = d["pred_raw"][:, :100].clone()
    bad = tmp_path / "shortref"
    bad.mkdir()
    torch.save(d, bad / "trajectory.pt")
    with pytest.raises(ValueError, match="agents, population has"):
        run_pipeline(REF_REPLAY_Q=0.2, REF_REPLAY_SEED=7,
                     REF_REPLAY_REF_RUN=str(bad))


@pytest.mark.parametrize("mutate,msg", [
    (lambda v: v.index_put_((torch.tensor([3]),),
                            torch.tensor([float("nan")])), "non-finite"),
    (lambda v: v.index_put_((torch.tensor([3]),), torch.tensor([2.0])),
     r"out of \[0, 1\]"),
])
def test_a_malformed_reference_vector_is_refused(mutate, msg, ref_run,
                                                 tmp_path):
    d = torch.load(ref_run[1] / "trajectory.pt", map_location="cpu",
                   weights_only=False)
    d["pred_raw"] = d["pred_raw"].clone()
    mutate(d["pred_raw"][0])
    bad = tmp_path / "badref"
    bad.mkdir()
    torch.save(d, bad / "trajectory.pt")
    with pytest.raises(ValueError, match=msg):
        run_pipeline(REF_REPLAY_Q=0.2, REF_REPLAY_SEED=7,
                     REF_REPLAY_REF_RUN=str(bad))


# ===================================================================
#  7. SABOTAGE
# ===================================================================

def verify_ref_replay(traj, cfg):
    """Re-derive the whole scheme from the artifact alone. Returns a list
    of violations -- the shape an offline checker takes."""
    bad = []
    q, seed = cfg["ref_replay_q"], cfg["ref_replay_seed"]
    lab, live_all = traj["ref_replay_labels"], \
        traj["ref_replay_live_idx"].long()
    b, innate, op = traj["ref_replay_ref_vec"], traj["innate"], traj["op_raw"]
    n = int(b.shape[0])
    if gp.ref_replay_hash(b) != cfg["ref_replay_ref_sha256"]:
        bad.append("ref vector does not match its recorded sha256")
    if lab.shape[1] != n:
        # positional comparison is meaningless once the row set is wrong
        bad.append(f"labels carry {lab.shape[1]} rows, not all {n} agents")
        return bad
    prev = None
    for t in range(lab.shape[0]):
        want_live = gp.ref_replay_live(n, q, seed, t)
        got_live = live_all[t]
        if not torch.equal(got_live, want_live):
            bad.append(f"round {t}: live set is not the (seed, round) "
                       f"reconstruction")
        if prev is not None and torch.equal(got_live, prev):
            bad.append(f"round {t}: live set did not refresh")
        prev = got_live
        x = innate if t == 0 else op[t - 1]
        want = gp.ref_replay_labels(x.float(), b, want_live)
        if not torch.equal(lab[t], want):
            n_off = int((lab[t] != want).sum())
            live_set = set(want_live.tolist())
            off = [int(i) for i in (lab[t] != want).nonzero().squeeze(-1)]
            where = ("live" if all(i in live_set for i in off)
                     else "non-live" if not any(i in live_set for i in off)
                     else "mixed")
            bad.append(f"round {t}: {n_off} labels differ ({where} rows)")
    return bad


def test_a_clean_ref_replay_run_verifies(q10):
    traj, cfg, _, _ = q10
    assert verify_ref_replay(traj, cfg) == []


def test_nesting_verifies_across_two_independently_launched_arms(q10, q20):
    """q=.10 and q=.20 are SEPARATE runs, launched independently, and
    their live sets still nest at every round -- because the draw reads
    only (seed, round). This is the cross-arm form of the property, and
    it is the one the wave actually depends on.

    (The stub model never learns, so both arms' populations coincide here
    by construction; what differs, and what is asserted, is the labels.)"""
    a, _, _, _ = q10
    b, _, _, _ = q20
    small, big = a["ref_replay_live_idx"].long(), b["ref_replay_live_idx"].long()
    assert small.shape[1] == 72 and big.shape[1] == 145
    for t in range(small.shape[0]):
        assert torch.equal(small[t], big[t][:72]), f"round {t} not nested"
    # the arms really are two different label regimes
    assert not torch.equal(a["ref_replay_labels"][1],
                           b["ref_replay_labels"][1])


def test_sabotage_a_run_whose_substitutions_accumulate(q10, b_vec):
    """The defect: round t's labels built by editing round t-1's label
    vector instead of rebuilding from b. Shapes, dtypes, row order and
    value range are all still correct."""
    traj, cfg, _, _ = q10
    bad = {k: (v.clone() if torch.is_tensor(v) else v)
           for k, v in traj.items()}
    lab, live = bad["ref_replay_labels"], bad["ref_replay_live_idx"].long()
    carry = b_vec.clone()
    for t in range(lab.shape[0]):
        x = x_of_round(traj, t).float()
        carry[live[t]] = x[live[t]]          # <- never reset to b
        lab[t] = carry.clone()
    viol = verify_ref_replay(bad, cfg)
    assert viol, "an accumulating run went undetected"
    assert any("non-live rows" in v for v in viol), viol


def test_sabotage_a_run_that_reorders_rows(q10):
    """Live rows first, then the rest. The label MULTISET is unchanged,
    so only a positional check catches it -- and getting this wrong puts
    agent 40's opinion on agent 3's prompt."""
    traj, cfg, _, _ = q10
    bad = {k: (v.clone() if torch.is_tensor(v) else v)
           for k, v in traj.items()}
    lab, live = bad["ref_replay_labels"], bad["ref_replay_live_idx"].long()
    for t in range(lab.shape[0]):
        live_set = live[t].tolist()
        order = torch.tensor(live_set + [i for i in range(N)
                                         if i not in set(live_set)])
        lab[t] = lab[t][order]
    viol = verify_ref_replay(bad, cfg)
    assert viol, "a reordered batch went undetected"
    assert sorted(lab[1].tolist()) == sorted(
        traj["ref_replay_labels"][1].tolist()), (
        "the sabotage must leave the label multiset intact, or it is "
        "catching the wrong thing")


def test_sabotage_a_live_set_that_is_not_nested(q10):
    """A sample drawn per-arm (or from a generator the loop advances)
    would give the right COUNT and the wrong PEOPLE."""
    traj, cfg, _, _ = q10
    bad = {k: (v.clone() if torch.is_tensor(v) else v)
           for k, v in traj.items()}
    n_live = bad["ref_replay_live_idx"].shape[1]
    for t in range(bad["ref_replay_live_idx"].shape[0]):
        g = torch.Generator().manual_seed(1000 + t)
        bad["ref_replay_live_idx"][t] = torch.randperm(
            N, generator=g)[:n_live]
    viol = verify_ref_replay(bad, cfg)
    assert viol
    assert any("not the (seed, round) reconstruction" in v for v in viol), viol


def test_sabotage_a_live_set_frozen_across_rounds(q10):
    """The other half of 'refreshes every round': a run that draws once
    and reuses it trains the SAME 72 people on live labels forever."""
    traj, cfg, _, _ = q10
    bad = {k: (v.clone() if torch.is_tensor(v) else v)
           for k, v in traj.items()}
    for t in range(1, bad["ref_replay_live_idx"].shape[0]):
        bad["ref_replay_live_idx"][t] = bad["ref_replay_live_idx"][0]
    viol = verify_ref_replay(bad, cfg)
    assert any("did not refresh" in v for v in viol), viol


def test_sabotage_a_shrunken_batch(q10):
    """The obvious implementation -- train on the live rows only -- shows
    up as a row count that is not n."""
    traj, cfg, _, _ = q10
    bad = {k: (v.clone() if torch.is_tensor(v) else v)
           for k, v in traj.items()}
    live = bad["ref_replay_live_idx"].long()
    bad["ref_replay_labels"] = torch.stack(
        [bad["ref_replay_labels"][t][live[t]] for t in range(live.shape[0])])
    viol = verify_ref_replay(bad, cfg)
    assert any("not all 723 agents" in v for v in viol), viol


def test_sabotage_a_swapped_reference_vector(q10):
    """b replaced by something else: the recorded sha256 is what makes
    that detectable without holding the reference run."""
    traj, cfg, _, _ = q10
    bad = {k: (v.clone() if torch.is_tensor(v) else v)
           for k, v in traj.items()}
    bad["ref_replay_ref_vec"] = torch.full((N,), 0.5)
    viol = verify_ref_replay(bad, cfg)
    assert any("recorded sha256" in v for v in viol), viol
