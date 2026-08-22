"""AI_GATE_REFERENCE: the gate measures against the ANCHOR (2026-08-22).

THE CORRECTION
--------------
The intended acceptance rule is

    |m_i(t) - x'_i(t)| < eps_AI ,   x'_i(t) = k innate_i + (1-k) x_i(t)

-- the agent judges the served value against the opinion it is actually
holding, which is the ANCHORED one. `k` here is INNATE_LAMBDA (gamma in
the write-up), and x' is the same vector the mixture then blends into,
already computed in nested_presocial_update as `h`. The code gated on
the raw start-of-round x_i(t) instead, and its docstring ARGUED for that
("the platform serves before the population moves"), so this is not a
slip to be quietly patched: it is a semantic correction, and both
semantics stay reachable and named.

    gate_on="anchor"  (NEW DEFAULT)  |m - h| < eps_AI
    gate_on="x0"      (archived)     |m - x| < eps_AI

WHAT THESE TESTS ARE FOR
------------------------
1. The archived path is EXACT, not approximate. Thousands of completed
   trajectories were produced by the old expression; "x0" has to
   reproduce it bit-for-bit or every reuse audit in the project is
   quietly wrong.
2. The BLAST RADIUS is a measured fact, not an assumption. The claim is
   that only k>0 AND AI_GATE_MODE=threshold can change: at k=0 the two
   references are the SAME VECTOR (h == x0), and under all_open the
   distance is never read. Both are proved end to end, including on the
   QWU surface (k=1, both gates all_open) the pilot runs on.
3. EVERY gate site moved together. A dry/counterfactual gate computed
   under one rule while the deployed update runs another is worse than
   either rule alone: the A/B arm would PICK a served vector under one
   gate and APPLY it under a different one, and the no_feedback arm's
   recorded counterfactual would not be the counterfactual of anything.

NO MODEL IS LOADED. The LM is a stub whose generation is a deterministic
function of the prompt string.

Run with USE_TF=0.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
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


gp = _load("_gated_pop_anchor", PIPE / "_gated_pop.py")
RUN = _load("run_gate_anchor", PIPE / "run_pokec_gated_lm.py")

N = 723          # movielens / Action, the pilot surface


# ===================================================================
#                        the stub platform
# ===================================================================

def stub_value(prompt: str) -> float:
    """Deterministic function of the PROMPT, not of the agent index: a
    runner that built the wrong prompt for the right agent still shows
    up as a changed served value."""
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


BASE_ENV = {
    "PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""),
    "USE_TF": "0",
    "DATASET": "movielens", "ML_DIR": str(ML_DIR), "ML_TARGET": "Action",
    "POP_MODEL": "ab", "RUN_MODE": "loop", "TRAINING_STYLE": "frozen",
    "DATA_REGIME": "replace", "USE_LORA": "0", "DEVICE": "cpu", "SEED": "0",
    "N_ROUNDS": "4", "DEPLOY_EVERY": "1", "N_LABELED": str(N),
    "W_PLAT": "1", "INNATE_LAMBDA": "0.5", "EPS": "0.2", "EPS_AI": "0.2",
    # gamma=0 is the project's standing policy (no homophily selection bias)
    "GAMMA_BIAS": "0", "N_PROBE": "8",
    "LOG_PERPLEXITY": "0", "LOG_ANSWER_DIST": "0", "LOG_PPL_DIST": "0",
    "GRAD_NORM_N": "0", "TEL_EVAL_CAP": "0",
}


def run_pipeline(**overrides):
    """main() end to end on real MovieLens with a stub LM."""
    env = dict(BASE_ENV)
    env["RUN_TAG"] = "gate_anchor_test"
    env.update({k: str(v) for k, v in overrides.items()})
    tmp = tempfile.mkdtemp(prefix="gate_anchor_")
    env["OUT_DIR"] = tmp
    holder = {}

    def _mk(*a, **kw):
        holder["lm"] = _StubLM(*a, **kw)
        return holder["lm"]

    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(RUN, "HFCausalLMModel", _mk), \
            mock.patch.object(RUN.gp, "sft_batch_loss", lambda *a, **k: 0.0), \
            mock.patch.object(RUN.gp, "sft_grad_norm", lambda *a, **k: 0.0):
        rc = RUN.main()
    assert rc == 0
    traj = torch.load(Path(tmp) / "trajectory.pt", map_location="cpu",
                      weights_only=False)
    cfg = json.loads((Path(tmp) / "config.json").read_text())
    return traj, cfg, holder["lm"], Path(tmp)


def _rand(n=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, generator=g), torch.rand(n, generator=g),
            torch.rand(n, generator=g))


# ===================================================================
#  1. the reference itself (unit)
# ===================================================================

def test_anchor_reference_is_the_human_component_h():
    """x' = k innate + (1-k) x0, bit-for-bit the same expression the
    operator uses for h -- one vector, not two that happen to agree."""
    served, x0, innate = _rand()
    for k in (0.0, 0.2, 0.5, 1.0):
        h = k * innate + (1.0 - k) * x0
        assert torch.equal(gp.gate_reference(x0, innate, k, "anchor"), h)


def test_x0_reference_is_the_raw_start_of_round_opinion():
    served, x0, innate = _rand()
    for k in (0.0, 0.2, 1.0):
        got = gp.gate_reference(x0, innate, k, "x0")
        assert torch.equal(got, x0)
        assert got is x0, "the archived path must not even copy x0"


def test_default_reference_is_anchor():
    """The corrected rule is what you get without asking for anything."""
    served, x0, innate = _rand(64)
    k = 0.4
    assert torch.equal(gp.gate_reference(x0, innate, k),
                       gp.gate_reference(x0, innate, k, "anchor"))
    assert not torch.equal(gp.gate_reference(x0, innate, k),
                           gp.gate_reference(x0, innate, k, "x0"))
    import inspect
    sig = inspect.signature(gp.nested_presocial_update)
    assert sig.parameters["gate_on"].default == "anchor"


def test_unknown_reference_raises_rather_than_falling_back():
    """A typo must not silently become one of the two semantics."""
    served, x0, innate = _rand(8)
    with pytest.raises(ValueError, match="AI_GATE_REFERENCE"):
        gp.gate_reference(x0, innate, 0.5, "h")
    with pytest.raises(ValueError, match="AI_GATE_REFERENCE"):
        gp.nested_presocial_update(x0, served, innate, 0.5,
                                   torch.full_like(x0, 0.5), 0.2,
                                   gate_on="anchor2")


# ===================================================================
#  2. "x0" is the archived operator, bit-for-bit
# ===================================================================

def test_x0_mode_is_the_pre_correction_operator_bitwise():
    """The body as it stood before 2026-08-22, inlined verbatim."""
    served, x0, innate = _rand()
    w_agent = torch.full_like(x0, 0.5)
    for k, eps in ((0.0, 0.4), (0.2, 0.4), (0.2, 0.05), (1.0, 0.3)):
        h = k * innate + (1.0 - k) * x0
        gate_l = (served - x0).abs() < eps
        eff_w = torch.where(gate_l, w_agent, torch.zeros_like(w_agent))
        z_l = (1.0 - eff_w) * h + eff_w * served
        z, gate = gp.nested_presocial_update(x0, served, innate, k, w_agent,
                                             eps, gate_on="x0")
        assert torch.equal(gate, gate_l), f"k={k} eps={eps}"
        assert torch.equal(z, z_l), f"k={k} eps={eps}"


def test_anchor_mode_is_exactly_the_corrected_rule():
    served, x0, innate = _rand()
    w_agent = torch.full_like(x0, 0.5)
    for k, eps in ((0.2, 0.4), (0.5, 0.2), (1.0, 0.3)):
        h = k * innate + (1.0 - k) * x0
        gate_w = (served - h).abs() < eps
        eff_w = torch.where(gate_w, w_agent, torch.zeros_like(w_agent))
        z_w = (1.0 - eff_w) * h + eff_w * served
        z, gate = gp.nested_presocial_update(x0, served, innate, k, w_agent,
                                             eps)
        assert torch.equal(gate, gate_w), f"k={k} eps={eps}"
        assert torch.equal(z, z_w), f"k={k} eps={eps}"


def test_the_two_references_really_do_differ_at_k_above_zero():
    """Guard: if this ever stops differing, every 'unchanged' claim below
    becomes vacuous."""
    served, x0, innate = _rand()
    w = torch.full_like(x0, 0.5)
    ga = gp.nested_presocial_update(x0, served, innate, 0.5, w, 0.2)[1]
    gx = gp.nested_presocial_update(x0, served, innate, 0.5, w, 0.2,
                                    gate_on="x0")[1]
    assert int((ga != gx).sum()) > 0


# ===================================================================
#  3. BLAST RADIUS -- the three invariance claims
# ===================================================================

def test_unchanged_at_k_zero_because_the_references_are_the_same_vector():
    """k = 0 makes h == x0 IDENTICALLY, so the correction cannot reach
    the whole k=0 archive. Bitwise, at every eps including the extremes."""
    served, x0, innate = _rand()
    w = torch.full_like(x0, 0.5)
    assert torch.equal(gp.gate_reference(x0, innate, 0.0, "anchor"), x0)
    for eps in (0.0, 0.05, 0.2, 0.5, 1.0):
        za, ga = gp.nested_presocial_update(x0, served, innate, 0.0, w, eps)
        zx, gx = gp.nested_presocial_update(x0, served, innate, 0.0, w, eps,
                                            gate_on="x0")
        assert torch.equal(ga, gx) and torch.equal(za, zx), f"eps={eps}"


def test_unchanged_under_all_open_because_no_distance_is_read():
    """all_open ignores the reference entirely -- at EVERY k, including
    k=1 where the two references are maximally far apart."""
    served, x0, innate = _rand()
    w = torch.full_like(x0, 0.5)
    for k in (0.0, 0.2, 0.5, 1.0):
        za, ga = gp.nested_presocial_update(x0, served, innate, k, w, 0.0,
                                            gate_mode="all_open")
        zx, gx = gp.nested_presocial_update(x0, served, innate, k, w, 0.0,
                                            gate_mode="all_open",
                                            gate_on="x0")
        assert bool(ga.all()) and torch.equal(ga, gx), f"k={k}"
        assert torch.equal(za, zx), f"k={k}"


def test_only_k_positive_and_threshold_can_change_anything():
    """The claim stated as a truth table over the whole grid."""
    served, x0, innate = _rand()
    w = torch.full_like(x0, 0.5)
    for k in (0.0, 0.3, 1.0):
        for mode in ("threshold", "all_open"):
            za = gp.nested_presocial_update(x0, served, innate, k, w, 0.2,
                                            gate_mode=mode)[0]
            zx = gp.nested_presocial_update(x0, served, innate, k, w, 0.2,
                                            gate_mode=mode,
                                            gate_on="x0")[0]
            same = torch.equal(za, zx)
            expect_same = (k == 0.0) or (mode == "all_open")
            assert same == expect_same, (
                f"k={k} mode={mode}: same={same}, expected {expect_same}")


def test_gate_reference_draws_no_rng():
    served, x0, innate = _rand()
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    gp.gate_reference(x0, innate, 0.5, "anchor")
    gp.nested_presocial_update(x0, served, innate, 0.5,
                               torch.full_like(x0, 0.5), 0.2)
    assert torch.equal(before, torch.get_rng_state())


# ===================================================================
#  4. end to end: the runner, the marker, and the same three claims
# ===================================================================

@pytest.fixture(scope="module")
def anchor_run():
    return run_pipeline()


@pytest.fixture(scope="module")
def x0_run():
    return run_pipeline(AI_GATE_REFERENCE="x0")


def test_runner_default_is_anchor_and_records_the_v2_marker(anchor_run):
    """The marker strings are a CONTRACT -- an offline checker dispatches
    its replay on them, so they are pinned literally here."""
    _, cfg, _, _ = anchor_run
    assert cfg["ai_gate_reference"] == "anchor"
    assert cfg["population_update"] == "nested_ai_anchored_then_social_v2"


def test_runner_x0_keeps_the_archived_marker(x0_run):
    _, cfg, _, _ = x0_run
    assert cfg["ai_gate_reference"] == "x0"
    assert cfg["population_update"] == "nested_ai_then_social_v1"


def test_runner_rejects_an_unknown_reference_loudly():
    with pytest.raises(ValueError, match="AI_GATE_REFERENCE"):
        run_pipeline(AI_GATE_REFERENCE="anchored")
    with pytest.raises(ValueError, match="AI_GATE_REFERENCE"):
        run_pipeline(AI_GATE_REFERENCE="h")


def test_deployed_gate_reproduces_from_the_artifact_under_anchor(anchor_run):
    """gate_raw must be replayable offline through the SHARED definition,
    from (pred_raw, op_raw, innate, k) alone -- and must NOT match the x0
    replay, or the artifact would not distinguish the two semantics."""
    traj, cfg, _, _ = anchor_run
    innate, op, pred = traj["innate"], traj["op_raw"], traj["pred_raw"]
    k = float(cfg["innate_lambda"])
    eps_ai = float(cfg["eps_ai"])
    n_wrong = 0
    for t in range(op.shape[0]):
        x0 = innate if t == 0 else op[t - 1]
        want = gp.ai_gate(pred[t].clamp(0.0, 1.0),
                          gp.gate_reference(x0, innate, k, "anchor"),
                          eps_ai, cfg["ai_gate_mode"])
        assert torch.equal(traj["gate_raw"][t], want), f"round {t}"
        wrong = gp.ai_gate(pred[t].clamp(0.0, 1.0),
                           gp.gate_reference(x0, innate, k, "x0"),
                           eps_ai, cfg["ai_gate_mode"])
        n_wrong += int((wrong != want).sum())
    assert n_wrong > 0, ("the x0 replay agrees everywhere -- this fixture "
                         "cannot tell the two semantics apart")


def test_deployed_gate_reproduces_from_the_artifact_under_x0(x0_run):
    traj, cfg, _, _ = x0_run
    innate, op, pred = traj["innate"], traj["op_raw"], traj["pred_raw"]
    k, eps_ai = float(cfg["innate_lambda"]), float(cfg["eps_ai"])
    for t in range(op.shape[0]):
        x0 = innate if t == 0 else op[t - 1]
        want = gp.ai_gate(pred[t].clamp(0.0, 1.0), x0, eps_ai,
                          cfg["ai_gate_mode"])
        assert torch.equal(traj["gate_raw"][t], want), f"round {t}"


def test_round_zero_agrees_by_construction_then_the_paths_diverge(
        anchor_run, x0_run):
    """x(0) IS innate, so h(0) == x(0) whatever k is: round 0 must be
    bit-identical under both references. Pinning only round 0 would
    therefore prove nothing, which is why the divergence from round 1 is
    asserted in the same test."""
    a, _, _, _ = anchor_run
    b, _, _, _ = x0_run
    assert torch.equal(a["gate_raw"][0], b["gate_raw"][0])
    assert torch.equal(a["op_raw"][0], b["op_raw"][0])
    assert not torch.equal(a["gate_raw"][1:], b["gate_raw"][1:]), (
        "k=0.5 with a threshold gate must eventually differ, or the "
        "end-to-end blast-radius tests below are vacuous")


def test_end_to_end_identical_at_k_zero():
    """The whole k=0 archive is untouched: identical opinions, identical
    predictions, identical gates, identical config apart from the two
    fields that NAME the semantics."""
    a, ca, _, _ = run_pipeline(INNATE_LAMBDA=0)
    b, cb, _, _ = run_pipeline(INNATE_LAMBDA=0, AI_GATE_REFERENCE="x0")
    for key in ("op_raw", "pred_raw", "gate_raw", "twin_raw"):
        assert torch.equal(a[key], b[key]), key
    diff = {k for k in ca if ca[k] != cb.get(k)}
    assert diff == {"ai_gate_reference", "population_update"}, diff


def test_end_to_end_identical_under_all_open_even_at_k_one():
    a, _, _, _ = run_pipeline(INNATE_LAMBDA=1, AI_GATE_MODE="all_open")
    b, _, _, _ = run_pipeline(INNATE_LAMBDA=1, AI_GATE_MODE="all_open",
                              AI_GATE_REFERENCE="x0")
    for key in ("op_raw", "pred_raw", "gate_raw"):
        assert torch.equal(a[key], b[key]), key
    assert bool(a["gate_raw"].all())


def test_the_qwu_surface_is_invariant():
    """THE surface the ref-replay pilot runs on: 723 agents, W_PLAT=1,
    INNATE_LAMBDA=1, BOTH gates all_open. Every number in that wave has
    to be unaffected by the correction, or the pilot's comparability to
    the existing q=1 trajectory is gone."""
    kw = dict(N_LABELED=N, W_PLAT=1, INNATE_LAMBDA=1,
              AI_GATE_MODE="all_open", PEER_GATE_MODE="all_open")
    a, ca, _, _ = run_pipeline(**kw)
    b, cb, _, _ = run_pipeline(AI_GATE_REFERENCE="x0", **kw)
    assert a["op_raw"].shape == (4, N)
    for key in ("op_raw", "pred_raw", "gate_raw", "twin_raw"):
        assert torch.equal(a[key], b[key]), f"QWU surface moved in {key}"
    assert {k for k in ca if ca[k] != cb.get(k)} == {
        "ai_gate_reference", "population_update"}


def test_end_to_end_differs_only_where_it_should():
    """The one cell that DOES change: k>0 and a threshold gate."""
    a, _, _, _ = run_pipeline()
    b, _, _, _ = run_pipeline(AI_GATE_REFERENCE="x0")
    assert not torch.equal(a["op_raw"], b["op_raw"])
    assert not torch.equal(a["gate_raw"], b["gate_raw"])


# ===================================================================
#  5. every gate site moved together
# ===================================================================

class _GateSpy:
    """Records every (served, reference) pair handed to gp.ai_gate while
    the runner is executing. nested_presocial_update resolves `ai_gate`
    as a module global, so patching the module attribute catches the
    DEPLOYED call as well as the dry and counterfactual ones."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        self._real = gp.ai_gate
        self._real_run = RUN.gp.ai_gate

        def _spy(served, ref, eps_ai, mode="threshold"):
            self.calls.append((served.detach().cpu().clone(),
                               ref.detach().cpu().clone()))
            return self._real(served, ref, eps_ai, mode)

        gp.ai_gate = _spy
        RUN.gp.ai_gate = _spy
        return self

    def __exit__(self, *exc):
        gp.ai_gate = self._real
        RUN.gp.ai_gate = self._real_run
        return False


def test_the_ab_dry_gate_uses_the_same_reference_as_the_deployed_gate():
    """AB_RETAIN scores a candidate and the retained winner on the same
    start-of-round state and DEPLOYS the more engaging one. Those dry
    scores go through gp.ai_gate too, so if they gated on x0 while the
    update gated on the anchor, the round would choose under one rule and
    apply another. Every reference vector seen inside a round must
    therefore be the SAME vector, and it must be the anchor."""
    with _GateSpy() as spy:
        traj, cfg, _, _ = run_pipeline(ICRH=1, AB_RETAIN=1,
                                       REWARD_KIND="engagement",
                                       AB_SWEEPS=1, N_ROUNDS=3)
    innate, op = traj["innate"], traj["op_raw"]
    k = float(cfg["innate_lambda"])
    assert cfg["ab_retain"] is True
    # three gate calls per round: candidate (dry), winner (dry), deployed
    assert len(spy.calls) >= 3 * op.shape[0], len(spy.calls)
    per_round = len(spy.calls) // op.shape[0]
    assert per_round >= 3, per_round
    seen_anchor_differs = False
    for t in range(op.shape[0]):
        x0 = innate if t == 0 else op[t - 1]
        want = gp.gate_reference(x0, innate, k, "anchor")
        refs = [r for _, r in spy.calls[t * per_round:(t + 1) * per_round]]
        for j, ref in enumerate(refs):
            assert torch.allclose(ref, want, atol=0), (
                f"round {t} gate call {j} used a reference that is not the "
                f"anchor -- a dry gate diverging from the deployed one")
        if not torch.equal(want, x0):
            seen_anchor_differs = True
    assert seen_anchor_differs, ("every round had h == x0, so this run "
                                 "cannot distinguish the two references")


def test_the_no_feedback_counterfactual_gate_is_the_corrected_one():
    """RUN_MODE=no_feedback adopts nothing; gate_raw there is the gate the
    LOOP arm WOULD have applied. A counterfactual measured under a
    different rule than the deployed one is not a counterfactual."""
    traj, cfg, _, _ = run_pipeline(RUN_MODE="no_feedback")
    innate, op, pred = traj["innate"], traj["op_raw"], traj["pred_raw"]
    k, eps_ai = float(cfg["innate_lambda"]), float(cfg["eps_ai"])
    n_diff = 0
    for t in range(op.shape[0]):
        x0 = innate if t == 0 else op[t - 1]
        want = gp.ai_gate(pred[t].clamp(0.0, 1.0),
                          gp.gate_reference(x0, innate, k, "anchor"),
                          eps_ai, cfg["ai_gate_mode"])
        assert torch.equal(traj["gate_raw"][t], want), f"round {t}"
        # ... and the scalar telemetry agrees with the mask
        assert abs(traj["trajectory"][t]["contact"]
                   - float(want.float().mean())) < 1e-6
        old = gp.ai_gate(pred[t].clamp(0.0, 1.0), x0, eps_ai,
                         cfg["ai_gate_mode"])
        n_diff += int((old != want).sum())
    assert n_diff > 0, "the no_feedback fixture cannot tell the rules apart"


def test_no_gate_distance_is_computed_inline_anywhere_in_the_runner():
    """Structural: every acceptance test must go through the shared
    gp.ai_gate / gp.gate_reference, or a future edit can reintroduce the
    divergence this file exists to exclude."""
    src = (PIPE / "run_pokec_gated_lm.py").read_text()
    for bad in ("- x0).abs() < eps_ai", "- x).abs() < eps_ai",
                "- ab_x).abs() < eps_ai"):
        assert bad not in src, f"inline gate distance found: {bad!r}"
    # and every ai_gate call in the runner passes a gate_reference(...)
    # result rather than a bare opinion vector
    calls = src.count("gp.ai_gate(")
    refs = src.count("gp.gate_reference(")
    assert calls == 2 and refs == 2, (
        f"expected the two dry/counterfactual gp.ai_gate sites to each pass "
        f"gp.gate_reference; found {calls} ai_gate and {refs} "
        f"gate_reference calls")


# ===================================================================
#  6. SABOTAGE -- a run that gated on the wrong reference
# ===================================================================

def verify_gate(traj, cfg):
    """Re-derive gate_raw from the artifact alone, dispatching on the
    population_update marker. This is the shape an offline checker takes;
    the marker, not the env, is what says which operator ran."""
    marker = cfg.get("population_update")
    gate_on = {"nested_ai_anchored_then_social_v2": "anchor",
               "nested_ai_then_social_v1": "x0"}.get(marker)
    if gate_on is None:
        return [f"unknown population_update marker {marker!r}"]
    innate, op, pred = traj["innate"], traj["op_raw"], traj["pred_raw"]
    k, eps_ai = float(cfg["innate_lambda"]), float(cfg["eps_ai"])
    mode = cfg.get("ai_gate_mode") or "threshold"
    bad = []
    for t in range(op.shape[0]):
        x0 = innate if t == 0 else op[t - 1]
        want = gp.ai_gate(pred[t].clamp(0.0, 1.0),
                          gp.gate_reference(x0, innate, k, gate_on),
                          eps_ai, mode)
        if not torch.equal(traj["gate_raw"][t], want):
            bad.append(f"round {t}: gate_raw is not the {gate_on} gate "
                       f"({int((traj['gate_raw'][t] != want).sum())} agents)")
    return bad


def test_a_clean_run_verifies_under_both_markers(anchor_run, x0_run):
    assert verify_gate(*anchor_run[:2]) == []
    assert verify_gate(*x0_run[:2]) == []


def test_sabotage_a_run_that_gated_on_the_wrong_reference(anchor_run):
    """THE defect this file exists to exclude: a trajectory that CLAIMS
    the corrected semantics (marker v2) but whose gates were computed
    against x0. Every array keeps its shape, dtype and value range; only
    the marker-driven replay catches it."""
    traj, cfg, _, _ = anchor_run
    bad = {k: (v.clone() if torch.is_tensor(v) else v)
           for k, v in traj.items()}
    innate, op, pred = bad["innate"], bad["op_raw"], bad["pred_raw"]
    eps_ai, k = float(cfg["eps_ai"]), float(cfg["innate_lambda"])
    for t in range(op.shape[0]):
        x0 = innate if t == 0 else op[t - 1]
        bad["gate_raw"][t] = gp.ai_gate(pred[t].clamp(0.0, 1.0), x0, eps_ai,
                                        cfg["ai_gate_mode"])
    viol = verify_gate(bad, cfg)
    assert viol, "gating on x0 under the v2 marker went undetected"
    assert all("not the anchor gate" in v for v in viol), viol


def test_sabotage_the_mirror_defect_anchor_gates_under_the_v1_marker(x0_run):
    traj, cfg, _, _ = x0_run
    bad = {k: (v.clone() if torch.is_tensor(v) else v)
           for k, v in traj.items()}
    innate, op, pred = bad["innate"], bad["op_raw"], bad["pred_raw"]
    eps_ai, k = float(cfg["eps_ai"]), float(cfg["innate_lambda"])
    for t in range(op.shape[0]):
        x0 = innate if t == 0 else op[t - 1]
        bad["gate_raw"][t] = gp.ai_gate(
            pred[t].clamp(0.0, 1.0),
            gp.gate_reference(x0, innate, k, "anchor"), eps_ai,
            cfg["ai_gate_mode"])
    viol = verify_gate(bad, cfg)
    assert viol and all("not the x0 gate" in v for v in viol), viol


def test_sabotage_a_marker_that_does_not_name_a_semantics(anchor_run):
    """A checker must refuse to guess. An artifact carrying an unknown or
    abbreviated marker is unreplayable, not 'probably the default'."""
    traj, cfg, _, _ = anchor_run
    for marker in ("nested_ai_anchored_then_social", "v2", None):
        viol = verify_gate(traj, dict(cfg, population_update=marker))
        assert viol and "unknown population_update marker" in viol[0]
