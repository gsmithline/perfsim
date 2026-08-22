"""Tests for REFERENCE REPLAY at the Wu boundary (2026-08-22).

ref_replay[_smoke]: the completed Wu-boundary b0 cell with ONE change.
Every round the SFT set is rebuilt FULL SIZE (723 rows, one epoch, batch
4 -> 181 optimizer steps), but only a fraction q of the rows carries the
LIVE population value; the rest carry a PINNED frozen-Qwen prediction
b_i for the same agent. Data volume and compute are therefore held
fixed and the ladder varies FEEDBACK alone -- the confound the subsample
wave could not remove, since a smaller sample also took fewer steps.

What these pin, and why each one is here:
  * the two things that are REUSED rather than re-run -- q=1 (which is
    ordinary SFT exactly) and the frozen reference vector -- are named,
    and the names are checked against the real generated runs, not
    against a similar-looking string;
  * n(q) is recomputed half-up, because Python's round() is banker's
    rounding and round(361.5) == 362 only because 362 is even;
  * the surface is asserted field by field against the QWU cells, since
    "same surface" is the whole claim this pilot rests on;
  * the brace regression in the submit usage strings, which silently
    truncated BID and WHAT for EVERY key project-wide.

Run with USE_TF=0.
"""
import importlib.util
import os
import re
import subprocess

import pytest

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
CONDOR = os.path.join(REPO, "experiments", "condor")
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")
SUBMIT = os.path.join(CONDOR, "submit_pofd_sweep.sh")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load("gen_rr", os.path.join(CONDOR, "gen_pofd_sweep.py"))

# queue column indices, straight off ROW_RR / the sub's queue line
C_TAG, C_STYLE, C_BETA, C_SEED = 0, 1, 2, 3
C_EPS, C_GAMMA, C_WPLAT = 9, 10, 11
C_LAM, C_REFQ, C_ICLK, C_SNAP = 14, 15, 16, 17
C_USELORA, C_FRESH, C_ROUNDS, C_MODEL = 18, 19, 22, 23
N_COLS = 28


def _cols(row):
    return [c.strip() for c in row.split(",")]


def _cfg(key):
    with open(os.path.join(CONDOR, f"configs_pofd_{key}.txt")) as fh:
        return [ln for ln in fh.read().splitlines() if ln.strip()]


def _sub(key):
    with open(os.path.join(CONDOR, f"at_pofd_{key}.sub")) as fh:
        return fh.read()


def _env_line(sub_text):
    return next(ln for ln in sub_text.splitlines()
                if ln.startswith("environment"))


# --------------------------------------------------------- job counts
def test_job_counts_are_exactly_four_and_one():
    assert len(GEN.rr_rows()) == 4
    assert len(GEN.rr_smoke_rows()) == 1
    assert len(GEN.RR_QS) == 4
    assert len(_cfg(GEN.RR_KEY)) == 4
    assert len(_cfg(GEN.RR_SMOKE_KEY)) == 1


def test_keys_are_the_requested_ones():
    assert GEN.RR_KEY == "ref_replay"
    assert GEN.RR_SMOKE_KEY == "ref_replay_smoke"


def test_on_disk_configs_and_subs_match_the_generator():
    """The generator writes .sub files as well as .txt; a regenerated sub
    left behind is a known way to break --verify later."""
    assert _cfg(GEN.RR_KEY) == GEN.rr_rows()
    assert _cfg(GEN.RR_SMOKE_KEY) == GEN.rr_smoke_rows()
    assert _sub(GEN.RR_KEY) == GEN.rr_sub()
    assert _sub(GEN.RR_SMOKE_KEY) == GEN.rr_sub(smoke=True)


def test_generator_verify_is_clean_project_wide():
    r = subprocess.run(["python3", os.path.join(CONDOR, "gen_pofd_sweep.py"),
                        "--verify"], capture_output=True, text=True,
                       cwd=REPO)
    assert "MISMATCH" not in r.stdout, [
        ln for ln in r.stdout.splitlines() if "MISMATCH" in ln]
    assert r.returncode == 0, r.stdout[-2000:]


# ------------------------------------------------------------ the grid
def test_q_grid_is_the_requested_ladder():
    assert GEN.RR_QS == [0.10, 0.20, 0.50, 0.75]
    assert GEN.RR_Q_FULL == 1.0
    assert sorted(GEN.RR_N) == [0.10, 0.20, 0.50, 0.75, 1.0]
    assert {float(_cols(r)[C_REFQ]) for r in _cfg(GEN.RR_KEY)} == \
        set(GEN.RR_QS)


def test_n_per_q_is_the_pinned_arithmetic_not_a_rounding_accident():
    """n = floor(q*723 + .5): 72 / 145 / 362 / 542 / 723. Pinned AND
    recomputed. Python's round() is banker's rounding, so round(361.5)
    giving 362 is luck (362 is even), not a rule -- the generator must
    not depend on it."""
    assert GEN.RR_N_AGENTS == 723
    assert [GEN.RR_N[q] for q in sorted(GEN.RR_N)] == \
        [72, 145, 362, 542, 723]
    for q, n in GEN.RR_N.items():
        assert GEN.rr_n(q) == n == int(q * 723 + 0.5), (q, n)
    # q = 1 keeps every row live; every queued arm replaces at least one
    # row and leaves at least one live
    assert GEN.RR_N[1.0] == 723
    for q in GEN.RR_QS:
        assert 0 < GEN.RR_N[q] < 723
    # the ladder is monotone in q, so "more feedback" is well ordered
    ns = [GEN.RR_N[q] for q in sorted(GEN.RR_N)]
    assert ns == sorted(ns) and len(set(ns)) == len(ns)


def test_replaced_row_count_is_the_complement():
    """The set is rebuilt FULL SIZE: live + replayed == 723 in every
    arm, which is what holds compute fixed across the ladder."""
    for q in GEN.RR_QS + [GEN.RR_Q_FULL]:
        assert GEN.RR_N[q] + (723 - GEN.RR_N[q]) == 723
    assert 723 - GEN.RR_N[0.10] == 651      # the smoke's corner


# ------------------------------------------------------------ the tags
def test_tag_grammar():
    tags = [_cols(r)[C_TAG] for r in _cfg(GEN.RR_KEY)]
    assert len(tags) == len(set(tags)) == 4
    for t in tags:
        assert t.startswith("pofdrr_qwen7b_q"), t
        # both gates are GENUINELY open here, so the established MODE
        # tokens -- never the numeric 1, which is a strict-< threshold
        # that still rejects a distance-1 pair
        assert "_eaopen_" in t and "_esopen_" in t, t
        assert "_ea1_" not in t and "_es1_" not in t, t
        assert "_w1_" in t and "_l1_" in t, t
        assert "_ss0_" in t, t                    # selection seed
        assert t.endswith("_s0_r100"), t          # run seed + horizon
        assert "smoke" not in t, t
    assert set(tags) == {
        "pofdrr_qwen7b_q0p1_ss0_eaopen_w1_l1_esopen_s0_r100",
        "pofdrr_qwen7b_q0p2_ss0_eaopen_w1_l1_esopen_s0_r100",
        "pofdrr_qwen7b_q0p5_ss0_eaopen_w1_l1_esopen_s0_r100",
        "pofdrr_qwen7b_q0p75_ss0_eaopen_w1_l1_esopen_s0_r100"}


def test_q_token_uses_the_project_num_grammar():
    """_num() is how every other family spells a fraction (_w0p5_,
    _es0p05_, _l0p2_). A bespoke two-decimal token would be the one
    place in the project where 0.1 is not 0p1."""
    for q in GEN.RR_QS + [GEN.RR_Q_FULL]:
        assert GEN.rr_q_tok(q) == f"q{GEN._num(q)}"
    assert [GEN.rr_q_tok(q) for q in GEN.RR_QS] == \
        ["q0p1", "q0p2", "q0p5", "q0p75"]
    assert GEN.rr_q_tok(GEN.RR_Q_FULL) == "q1"


def test_the_two_seeds_have_distinct_tokens():
    """A selection seed and a run seed are different objects; one _s0_
    could not say which is which."""
    t = GEN.rr_tag(0.5)
    assert f"_ss{GEN.RR_SEL_SEED}_" in t and t.endswith(f"_s{GEN.RR_SEED}_r100")
    assert GEN.rr_tag(0.5, ) != GEN.rr_tag(0.5, rounds=3)
    # the horizon is in the tag: a 3-round and a 100-round cell of one
    # condition are different runs, not the same run twice
    assert GEN.rr_tag(0.5).endswith("_r100")


def test_tags_collide_with_nothing_anywhere_in_the_project():
    ours = {_cols(r)[C_TAG] for r in _cfg(GEN.RR_KEY) + _cfg(GEN.RR_SMOKE_KEY)}
    assert len(ours) == 5
    others = set()
    for fn in os.listdir(CONDOR):
        if not (fn.startswith("configs_pofd_") and fn.endswith(".txt")):
            continue
        if fn in (f"configs_pofd_{GEN.RR_KEY}.txt",
                  f"configs_pofd_{GEN.RR_SMOKE_KEY}.txt"):
            continue
        with open(os.path.join(CONDOR, fn)) as fh:
            for ln in fh:
                if ln.strip():
                    others.add(ln.split(",")[0].strip())
    assert not (ours & others), ours & others
    # and the family itself is new -- both prefixes
    assert not any(t.startswith("pofdrr") for t in others)


# ---------------------------------------------------------- the surface
def test_every_row_carries_the_qwu_surface():
    for r in _cfg(GEN.RR_KEY) + _cfg(GEN.RR_SMOKE_KEY):
        c = _cols(r)
        assert len(c) == N_COLS, r
        assert c[C_STYLE] == "sft" and c[C_BETA] == "0", r  # ordinary SFT
        assert c[C_SEED] == "0", r                          # run seed 0
        assert c[C_EPS] == "0.2", r          # inert under all_open
        # W_PLAT = 1 -- beta_eff = 1 - (1-W)k = 1, the boundary itself
        assert c[C_WPLAT] == "1", r
        # INNATE_LAMBDA = 1: the innate anchor, the paper's gamma
        assert c[C_LAM] == "1", r
        # the HOMOPHILY gamma column is a different knob and stays 0
        assert c[C_GAMMA] == "0.0", r
        assert c[C_ICLK] == "0" and c[C_SNAP] == "-1", r    # no context
        assert c[C_USELORA] == "1" and c[C_FRESH] == "1", r
        assert c[C_MODEL] == "Qwen/Qwen2.5-7B-Instruct", r


def test_production_rows_are_one_hundred_rounds_and_the_smoke_is_three():
    for r in _cfg(GEN.RR_KEY):
        assert _cols(r)[C_ROUNDS] == "100", r
    assert _cols(_cfg(GEN.RR_SMOKE_KEY)[0])[C_ROUNDS] == "3"
    assert GEN.RR_ROUNDS == GEN.QWU_ROUNDS == 100
    assert GEN.RR_SMOKE_ROUNDS == GEN.QWU_SMOKE_ROUNDS == 3


def test_the_smoke_is_one_three_round_cell_at_q_ten_percent():
    rows = _cfg(GEN.RR_SMOKE_KEY)
    assert len(rows) == 1
    c = _cols(rows[0])
    assert float(c[C_REFQ]) == GEN.RR_SMOKE_Q == 0.10
    assert c[C_ROUNDS] == "3"
    assert c[C_TAG] == \
        "pofdrrsmk_qwen7b_q0p1_ss0_eaopen_w1_l1_esopen_s0_r3"
    # it must never shadow a production cell
    assert c[C_TAG] not in {_cols(r)[C_TAG] for r in _cfg(GEN.RR_KEY)}


def test_the_smoke_wears_its_own_prefix_for_the_horizon_gate():
    """check_ref_replay enforces the declared 100-round horizon for
    pofdrr_ runs and exempts pofdrrsmk_ ones, so a 3-round cell under
    the production prefix would be gated as a TRUNCATED run. Both
    prefixes still start with pofdrr, which is how that checker claims
    the run at all."""
    smoke = _cols(_cfg(GEN.RR_SMOKE_KEY)[0])[C_TAG]
    prod = {_cols(r)[C_TAG] for r in _cfg(GEN.RR_KEY)}
    assert smoke.startswith("pofdrrsmk_")
    assert smoke.startswith("pofdrr")
    assert not any(t.startswith("pofdrrsmk") for t in prod)
    assert all(t.startswith("pofdrr_") for t in prod)
    chk = os.path.join(PIPE, "check_ref_replay.py")
    if not os.path.exists(chk):
        pytest.skip("checker not present in this checkout")
    src = open(chk).read()
    assert 'startswith("pofdrrsmk")' in src, (
        "the checker's smoke rule moved -- the smoke prefix must follow it")
    assert 'startswith("pofdrr")' in src


@pytest.mark.parametrize("key", ["ref_replay", "ref_replay_smoke"])
def test_sub_env_carries_the_surface_and_the_new_knobs(key):
    sub = _sub(key)
    env = _env_line(sub)
    # the three REF_REPLAY_* knobs, exactly as the runner names them
    assert "REF_REPLAY_Q=$(refq) " in env       # q rides the queue
    assert f"REF_REPLAY_SEED={GEN.RR_SEL_SEED} " in env
    assert f"REF_REPLAY_REF_RUN={GEN.RR_REF_RUN} " in env
    # both gates GENUINELY open, as modes
    assert "AI_GATE_MODE=all_open" in env
    assert "PEER_GATE_MODE=all_open" in env
    # the training surface, identical to the QWU cells
    assert "LORA_R=512" in env and "USE_LORA=$(uselora)" in env
    assert "SFT_EPOCHS=1" in env and "SFT_BATCH_SIZE=4" in env
    assert "SFT_LR=5e-5" in env
    assert "FRESH_EACH_ROUND=$(fresh)" in env
    assert "N_ROUNDS=$(nrounds)" in env
    assert "INNATE_LAMBDA=$(lam)" in env
    assert "TRAIN_CAP=723" in env and "N_LABELED=723" in env
    assert "DATASET=movielens" in env and "ML_TARGET=Action" in env
    assert "BASE_MODEL=$(basemodel)" in env
    assert "KL_DIRECTION=forward" in env
    assert "WITH_TWIN=1" in env and "SAVE_RAW_GEN=1" in env
    assert "POP_RESET" not in env
    # FULL-SIZE rebuild: the subsample / step-cap knobs must be absent,
    # or two waves would be cutting the same batch at once
    assert "SFT_SAMPLE_N" not in env and "SFT_SAMPLE_REPEAT_TO" not in env
    assert "SFT_MAX_STEPS" not in env
    # the exact H100 SKU, not the family: the pool also reports a bare
    # "NVIDIA H100" for different silicon
    assert GEN.RR_H100 == GEN.QMECH_H100 == "NVIDIA H100 80GB HBM3"
    assert f'CUDADeviceName == "{GEN.RR_H100}"' in sub
    assert "CUDAGlobalMemoryMb >= 80000" in sub
    assert GEN.BAD_NODE_REQ in sub
    # the idempotent executable, so a resubmit no-ops a finished cell
    exe = next(ln for ln in sub.splitlines() if ln.startswith("executable"))
    assert exe.endswith("run_one_pokec_gated_idempotent.sh"), exe


@pytest.mark.parametrize("key", ["ref_replay", "ref_replay_smoke"])
def test_sub_queue_line_matches_the_row_shape(key):
    sub = _sub(key)
    q = next(ln for ln in sub.splitlines() if ln.startswith("queue "))
    cols = q.split(" from ")[0][len("queue "):].split(",")
    assert len(cols) == N_COLS, cols
    assert [c.strip() for c in cols][C_REFQ] == "refq"
    assert f"configs_pofd_{key}.txt" in q
    for r in _cfg(key):
        assert len(_cols(r)) == len(cols)


def test_surface_matches_the_qwu_b0_cell_field_for_field():
    """The claim is 'the existing clean QWU surface, one knob added'.
    Check it rather than assert it in prose: every queue column of a
    ref-replay row must equal the reused QWU b0 W=1 row, except the tag,
    the horizon-independent refq column, and nothing else."""
    qwu = next(_cols(r) for r in _cfg("qwen_wu_limit")
               if _cols(r)[C_TAG] == GEN.RR_REUSED_Q1_TAG)
    for r in _cfg(GEN.RR_KEY):
        c = _cols(r)
        # QWU rows have no refq column: drop ours and the two must be
        # identical from style through the end
        mine = c[:C_REFQ] + c[C_REFQ + 1:]
        assert len(mine) == len(qwu), (mine, qwu)
        assert mine[1:] == qwu[1:], (mine, qwu)


# ------------------------------------------------- what is REUSED, not run
def test_q_equals_one_is_not_queued_and_names_the_real_qwu_cell():
    """q=1 means every row is live, which IS ordinary SFT -- so the arm
    already exists. Naming it is the point: an absent fifth job that is
    not named reads as an oversight."""
    assert GEN.RR_Q_FULL not in GEN.RR_QS
    assert GEN.RR_REUSED_Q1_TAG == \
        "pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100"
    # it is the generator's own QWU b0 / W=1 cell, not a lookalike
    assert GEN.RR_REUSED_Q1_TAG == GEN.qwu_tag(GEN.RR_ARM, GEN.RR_W)
    # ... and the very cell the subsample wave reuses for its 100% arm
    assert GEN.RR_REUSED_Q1_TAG == GEN.QSS_REUSED_TAG
    # it is a REAL generated run, present in the QWU config on disk
    assert GEN.RR_REUSED_Q1_TAG in {_cols(r)[C_TAG]
                                    for r in _cfg("qwen_wu_limit")}
    # and it is NOT queued here, under any spelling
    rr = {_cols(r)[C_TAG] for r in _cfg(GEN.RR_KEY) + _cfg(GEN.RR_SMOKE_KEY)}
    assert GEN.RR_REUSED_Q1_TAG not in rr
    assert not any("_q1_" in t for t in rr), rr
    assert not any(float(_cols(r)[C_REFQ]) == 1.0
                   for r in _cfg(GEN.RR_KEY)), "q=1 must never queue"


def test_frozen_reference_run_is_real_and_carries_the_canonical_vector():
    """b is REUSED, not re-extracted. The canonical frozen K=D=0 vector
    already exists and is pinned in four places; REF_REPLAY_REF_RUN must
    name a run that actually supplies it."""
    assert GEN.RR_REF_SHA == GEN.QMECH_CANONICAL_PRED_SHA == (
        "1674ee5f8d833f46de672791d933e1d3bdeefb07484c2d110dec84ce71da30bb")
    # the checker pins the same constant
    chk = open(os.path.join(PIPE, "check_pofd_sanity.py")).read()
    assert GEN.RR_REF_SHA in chk
    # the named run is generated by this same script: it is a queued
    # cell of the mechanism-frozen key, not a hand-typed string
    assert GEN.RR_REF_RUN in {_cols(r)[C_TAG]
                              for r in _cfg("qwen_mechanism_frozen")}
    # and it is the FIRST entry of the established FROZEN_SOURCES list
    # in both analyzers -- the project's own primary frozen source
    for fn in ("analyze_sft_dose.py", "analyze_qwen_subsample.py"):
        src = open(os.path.join(PIPE, fn)).read()
        head = src.split("FROZEN_SOURCES")[1][:300]
        assert GEN.RR_REF_RUN in head, fn
        assert GEN.RR_REF_RUN_TWIN in head, fn
        assert head.index(GEN.RR_REF_RUN) < head.index(GEN.RR_REF_RUN_TWIN)


def test_frozen_reference_twin_is_an_audited_complete_archive():
    """The fallback is not a guess either: the mechanism manifest hashed
    it, and the hash is the canonical one."""
    import json
    mf = json.load(open(os.path.join(CONDOR,
                                     "manifest_qwen_mechanism.json")))
    assert mf["canonical_frozen_pred_sha256"] == GEN.RR_REF_SHA
    twin = next(c for c in mf["cells"]
                if c.get("run_tag") == GEN.RR_REF_RUN_TWIN)
    assert twin["pred_sha256"] == GEN.RR_REF_SHA
    assert twin["verdict"] == "PASS" and twin["note"] == "complete"
    assert twin["arm"] == "k0"                     # frozen, K = D = 0
    assert twin["gpu_name"] == GEN.RR_H100         # same silicon


@pytest.mark.skipif(
    not os.path.exists(os.path.join(
        REPO, "notes", "pofd", "cluster", GEN.RR_REF_RUN, "trajectory.pt")),
    reason="reference run not pulled locally")
def test_pulled_reference_run_hashes_to_the_canonical_vector():
    """When the pull is on this machine, prove the pin end to end: the
    served vector must be CONSTANT across rounds (a frozen K=D=0 model
    never sees the population) and hash to the canonical sha."""
    import hashlib
    torch = pytest.importorskip("torch")
    p = os.path.join(REPO, "notes", "pofd", "cluster", GEN.RR_REF_RUN,
                     "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    pred = d["pred_raw"]
    assert pred.shape[1] == GEN.RR_N_AGENTS == 723
    assert bool((pred == pred[0]).all()), "frozen predictions must be constant"
    sha = hashlib.sha256(pred[0].contiguous().numpy().tobytes()).hexdigest()
    assert sha == GEN.RR_REF_SHA


# ----------------------------------------------------- the submit script
def test_submit_script_knows_both_keys():
    src = open(SUBMIT).read()
    assert "ref_replay|ref_replay_smoke)" in src
    for key in (GEN.RR_KEY, GEN.RR_SMOKE_KEY):
        assert os.path.exists(os.path.join(CONDOR, f"at_pofd_{key}.sub"))
        assert f"submit_pofd_sweep.sh <BID> {key}" in _sub(key)


def test_usage_strings_contain_no_braces():
    """REGRESSION (2026-08-21, re-pinned here because this wave edits
    those lines). The usage text lives inside BID="${1:?usage: ...}" and
    WHAT="${2:?usage: ...}". Bash ends a ${x:?word} expansion at the
    FIRST unescaped '}', so a brace anywhere in that word truncates the
    expansion and corrupts BID and WHAT for EVERY key project-wide --
    not just the new ones. Only [ ] and | are safe."""
    checked = 0
    for ln in open(SUBMIT).read().split("\n"):
        m = re.match(r'^(BID|WHAT)="\$\{[12]:\?(.*)\}"$', ln)
        if not m:
            continue
        checked += 1
        body = m.group(2)
        assert "{" not in body and "}" not in body, (
            f"{m.group(1)} usage text contains a brace, which truncates "
            f"the parameter expansion and corrupts the variable")
        assert "ref_replay[_smoke]" in body, "new keys missing from usage"
    assert checked == 2, f"expected BID and WHAT lines, found {checked}"


def test_the_new_keys_parse_to_themselves():
    """The end-to-end consequence the brace bug had: a corrupted WHAT
    matches no case arm and every key falls through to the catch-all."""
    head = subprocess.run(["sed", "-n", "17,18p", SUBMIT],
                          capture_output=True, text=True).stdout
    for key in ("ref_replay", "ref_replay_smoke", "qwen_wu_limit",
                "qwen_subsample", "smoke"):
        r = subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + head
             + '\nprintf "%s|%s" "$BID" "$WHAT"', "_", "50", key],
            capture_output=True, text=True)
        assert r.stdout == f"50|{key}", (key, r.stdout[:80], r.stderr[:80])


def test_case_dispatch_resolves_each_key_to_its_own_target():
    """No earlier arm may swallow the new keys, and neither key may
    expand to the other: they are separate submissions on purpose."""
    src = open(SUBMIT).read().split("\n")
    start = next(i for i, ln in enumerate(src) if ln == 'case "$WHAT" in')
    end = next(i for i, ln in enumerate(src) if ln == "esac")
    block = "\n".join(src[start:end + 1])
    for key in ("ref_replay", "ref_replay_smoke"):
        r = subprocess.run(
            ["bash", "-c", f'BID=50\nWHAT="{key}"\n{block}\n'
                           'printf "%s" "$TARGETS"'],
            capture_output=True, text=True)
        assert r.stdout == key, (key, r.stdout[:200], r.stderr[:200])
