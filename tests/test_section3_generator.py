"""ADVERSARIAL tests for the Section-3 retention wave generator (pofds3_).

Every test here encodes a way the wave could be WRONG in a way that no
downstream artifact would reveal. The generator is the only place these
mistakes are still cheap: once 46 H100 jobs have run against a sub whose
`environment` reads a column the `queue` line never declared, the runs
LOOK fine -- config.json records a direction, the trajectory is complete,
the checker's per-run fields all agree -- and the headline is inverted.

Design notes for whoever maintains this file:

  * We do not import the S3 helper functions by name. We run
    gen_pofd_sweep.main() with HERE redirected into a throwaway
    directory and read exactly the bytes that would be written to
    experiments/condor/. That makes the tests independent of how the S3
    block is factored and binds them to the artifact that is actually
    submitted.
  * Column positions are never hard-coded. They are read off the sub's
    own `queue ... from ...` line, which is the only definition Condor
    itself uses. A row/queue arity mismatch is therefore its own test:
    it would shift every column silently.

Contract (pinned 2026-08-22, Section-3 retention wave):
  key            section3_retention / section3_retention_smoke
  tag            pofds3_{model}_{arm}_eaopen_w{beta}_k{k}_esopen_anch2_s0_r100
  models         qwen7b -> Qwen/Qwen2.5-7B-Instruct
                 qwen3_8b -> Qwen/Qwen3-8B (CHAT_THINKING=0)
  envs           (beta, k) in {(0.5, 1), (1, 1), (0.5, 0.2)}   -- (1, 0.2)
                 MUST NOT exist: k drops out of the FJ update at beta=1
  arms           sft (lambda 0, direction-NEUTRAL, no direction token),
                 forward lambda {0.1, 0.5, 1, 2, 4, 8},
                 reverse lambda {1, 8} in env(0.5,1) and env(1,1) only
  counts         2 x 3 x 7 = 42 forward+sft, 2 x 2 x 2 = 8 reverse,
                 50 conceptual, 4 satisfied by archived QWU cells,
                 46 NEW production rows + 1 smoke
"""
from __future__ import annotations

import contextlib
import glob
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDOR = os.path.join(ROOT, "experiments", "condor")
GEN = os.path.join(CONDOR, "gen_pofd_sweep.py")
KL_SFT = os.path.join(ROOT, "perfsim", "learners", "lm", "kl_sft.py")
RUNNER = os.path.join(ROOT, "experiments", "scripts", "cluster_pipelines",
                      "run_pokec_gated_lm.py")

# ------------------------------------------------------------------ contract
S3_KEY = "section3_retention"
S3_SMOKE_KEY = "section3_retention_smoke"
PROD_PREFIX = "pofds3_"
SMOKE_PREFIX = "pofds3smk_"

CFG = f"configs_pofd_{S3_KEY}.txt"
SUB = f"at_pofd_{S3_KEY}.sub"
CFG_SMOKE = f"configs_pofd_{S3_SMOKE_KEY}.txt"
SUB_SMOKE = f"at_pofd_{S3_SMOKE_KEY}.sub"

MODELS = {"qwen7b": "Qwen/Qwen2.5-7B-Instruct",
          "qwen3_8b": "Qwen/Qwen3-8B"}
FWD_LAMS = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0)
REV_LAMS = (1.0, 8.0)
ENVS = ((0.5, 1.0), (1.0, 1.0), (0.5, 0.2))     # (beta = W_PLAT, k = INNATE)
REV_ENVS = ((0.5, 1.0), (1.0, 1.0))
FORBIDDEN_ENV = (1.0, 0.2)
N_PRODUCTION = 46
N_CONCEPTUAL = 50
ROUNDS = 100
SMOKE_ROUNDS = 3
SEED = 0
H100 = "NVIDIA H100 80GB HBM3"
SMOKE_TAG = ("pofds3smk_qwen3_8b_revlam1_eaopen_w1_k1"
             "_esopen_anch2_s0_r3")

# The four archived Qwen2.5 QWU cells that satisfy 4 of the 50 conceptual
# arms. Queueing any of them would burn a GPU job AND create a second,
# differently-provenanced answer to a question that already has one.
REUSE_TAGS = (
    "pofdqwu_qwen7b_b0_eaopen_w0p5_l1_esopen_s0_r100",
    "pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100",
    "pofdqwu_qwen7b_b1_eaopen_w0p5_l1_esopen_s0_r100",
    "pofdqwu_qwen7b_b1_eaopen_w1_l1_esopen_s0_r100",
)
# (model, arm, env) triples the reuse covers: qwen7b sft and forward
# lambda=1 in the two k=1 environments.
REUSED_CELLS = {
    ("qwen7b", "sft", (0.5, 1.0)),
    ("qwen7b", "sft", (1.0, 1.0)),
    ("qwen7b", "fwdlam1", (0.5, 1.0)),
    ("qwen7b", "fwdlam1", (1.0, 1.0)),
}

W_OF_TOK = {"0p5": 0.5, "1": 1.0}
K_OF_TOK = {"1": 1.0, "0p2": 0.2}

TAG_RE = re.compile(
    r"^pofds3_(?P<model>qwen7b|qwen3_8b)"
    r"_(?P<arm>sft|fwdlam0p1|fwdlam0p5|fwdlam1|fwdlam2|fwdlam4|fwdlam8"
    r"|revlam1|revlam8)"
    r"_eaopen_w(?P<w>0p5|1)_k(?P<k>1|0p2)_esopen_anch2_s0_r100$")

LAM_OF_ARM = {"sft": 0.0, "fwdlam0p1": 0.1, "fwdlam0p5": 0.5,
              "fwdlam1": 1.0, "fwdlam2": 2.0, "fwdlam4": 4.0,
              "fwdlam8": 8.0, "revlam1": 1.0, "revlam8": 8.0}


def _conceptual_cells():
    cells = set()
    for m in MODELS:
        for env in ENVS:
            cells.add((m, "sft", env))
            for lam in FWD_LAMS:
                cells.add((m, f"fwdlam{_num(lam)}", env))
        for env in REV_ENVS:
            for lam in REV_LAMS:
                cells.add((m, f"revlam{_num(lam)}", env))
    return cells


def _num(v):
    """Project tag grammar: 0.5 -> '0p5', 1.0 -> '1', 0.2 -> '0p2'."""
    return f"{v:g}".replace(".", "p")


# ------------------------------------------------------------------ fixtures
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """{basename: text} for EVERY file gen_pofd_sweep would write.

    HERE is redirected into a throwaway dir so the repo is never touched.
    Manifests are copied in first: several blocks resolve manifest paths
    at import time (absolute, unaffected by the patch) but a future block
    could resolve one at call time, and a missing manifest would look
    like an S3 bug.
    """
    tmp = str(tmp_path_factory.mktemp("gen_s3"))
    for j in glob.glob(os.path.join(CONDOR, "*.json")):
        shutil.copy(j, tmp)
    mod = _load(GEN, "_gen_pofd_s3")
    mod.HERE = tmp
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.main()
    out = {}
    for f in sorted(os.listdir(tmp)):
        if f.endswith(".json"):
            continue
        with open(os.path.join(tmp, f)) as fh:
            out[f] = fh.read()
    return out


def _require_s3(generated):
    if CFG not in generated:
        stray = [f for f in generated
                 if "section3" in f or "pofds3" in f or "_s3" in f]
        if stray:
            pytest.fail(
                f"S3 wave present under NON-CONTRACT filenames {stray}; "
                f"the contract pins S3_KEY={S3_KEY!r} -> {CFG} / {SUB}")
        pytest.skip("the Section-3 block has not landed in "
                    "gen_pofd_sweep.py yet")


@pytest.fixture(scope="module")
def s3(generated):
    """(rows, sub, smoke_rows, smoke_sub) as parsed text."""
    _require_s3(generated)
    rows = [r for r in generated[CFG].splitlines() if r.strip()]
    smoke = [r for r in generated.get(CFG_SMOKE, "").splitlines() if r.strip()]
    return rows, generated.get(SUB, ""), smoke, generated.get(SUB_SMOKE, "")


def _queue_cols(sub):
    """Column names EXACTLY as Condor reads them off the queue line."""
    q = [l for l in sub.splitlines() if l.startswith("queue ")]
    assert len(q) == 1, f"expected exactly one queue line, got {len(q)}"
    head = q[0].split(" from ")[0]
    assert head != q[0], "queue line does not read from a config file"
    return [c.strip() for c in head[len("queue "):].split(",") if c.strip()]


def _env_line(sub):
    ls = [l for l in sub.splitlines() if l.startswith("environment")]
    assert len(ls) == 1, f"expected one environment line, got {len(ls)}"
    return ls[0]


def _dicts(rows, cols):
    out = []
    for r in rows:
        vals = [x.strip() for x in r.split(",")]
        assert len(vals) == len(cols), (
            f"row declares {len(vals)} fields but the queue line declares "
            f"{len(cols)} columns -- every column after the first "
            f"disagreement is silently shifted.\n  cols={cols}\n  row={r}")
        out.append(dict(zip(cols, vals)))
    return out


def _tags(rows):
    return [r.split(",")[0].strip() for r in rows]


def _parsed(tags):
    out = []
    for t in tags:
        m = TAG_RE.match(t)
        assert m, f"tag violates the pinned S3 grammar: {t!r}"
        out.append((m["model"], m["arm"],
                    (W_OF_TOK[m["w"]], K_OF_TOK[m["k"]])))
    return out


def _col(d, *names):
    for n in names:
        if n in d:
            return d[n]
    raise AssertionError(f"none of {names} in queue columns {sorted(d)}")


# ---------------------------------------------------------------- file names
def test_contract_filenames_exist(generated):
    _require_s3(generated)
    for f in (CFG, SUB, CFG_SMOKE, SUB_SMOKE):
        assert f in generated, f"generator never writes {f}"


# ------------------------------------------------------------------- counts
def test_exactly_46_production_rows_and_1_smoke_row(s3):
    rows, _, smoke, _ = s3
    assert len(rows) == N_PRODUCTION, (
        f"{len(rows)} production rows, contract says {N_PRODUCTION} "
        f"(50 conceptual - 4 archived QWU cells)")
    assert len(smoke) == 1, f"{len(smoke)} smoke rows, contract says 1"


def test_queued_plus_reused_is_exactly_the_50_cell_conceptual_grid(s3):
    """The count 46 is only meaningful if it is the RIGHT 46. A wave that
    dropped two forward cells and queued the two reused ones would also
    have 46 rows."""
    rows, _, _, _ = s3
    queued = set(_parsed(_tags(rows)))
    conceptual = _conceptual_cells()
    assert len(conceptual) == N_CONCEPTUAL, len(conceptual)
    assert queued <= conceptual, (
        f"queued cells outside the conceptual grid: {queued - conceptual}")
    assert queued == conceptual - REUSED_CELLS, (
        f"missing: {sorted(conceptual - REUSED_CELLS - queued)}\n"
        f"extra:   {sorted(queued - (conceptual - REUSED_CELLS))}")


def test_no_duplicate_tags_within_the_wave(s3):
    rows, _, smoke, _ = s3
    tags = _tags(rows) + _tags(smoke)
    dupes = {t for t in tags if tags.count(t) > 1}
    assert not dupes, dupes


# ------------------------------------------------------------- models / envs
def test_both_models_present_and_balanced(s3):
    rows, _, _, _ = s3
    cells = _parsed(_tags(rows))
    per_model = {m: sum(1 for c in cells if c[0] == m) for m in MODELS}
    assert set(per_model) == set(MODELS), per_model
    # qwen3_8b queues its full 25; qwen7b queues 25 - 4 reused = 21.
    assert per_model["qwen3_8b"] == 25, per_model
    assert per_model["qwen7b"] == 21, per_model


def test_exactly_three_environments(s3):
    rows, _, _, _ = s3
    envs = {c[2] for c in _parsed(_tags(rows))}
    assert envs == set(ENVS), envs


def test_beta1_k0p2_environment_does_not_exist(s3):
    """k multiplies the (1 - beta) innate term in the FJ update, so at
    beta = 1 it drops out entirely: (1, 0.2) is the SAME dynamical system
    as (1, 1). Queueing it would spend 14 H100 jobs producing a duplicate
    of env2 and would put a fourth column in the figure that is a
    relabelled copy of an existing one."""
    rows, _, _, _ = s3
    tags = _tags(rows)
    bad = [t for t, c in zip(tags, _parsed(tags)) if c[2] == FORBIDDEN_ENV]
    assert not bad, bad
    assert "_w1_k0p2_" not in "\n".join(rows)


# --------------------------------------------------------------------- arms
def test_forward_ladder_is_exactly_the_six_doses_per_model_and_env(s3):
    rows, _, _, _ = s3
    cells = _parsed(_tags(rows))
    want = {f"fwdlam{_num(l)}" for l in FWD_LAMS}
    for m in MODELS:
        for env in ENVS:
            got = {a for mm, a, e in cells
                   if mm == m and e == env and a.startswith("fwdlam")}
            reused = {a for mm, a, e in REUSED_CELLS
                      if mm == m and e == env and a.startswith("fwdlam")}
            assert got | reused == want, (
                f"{m} {env}: forward ladder {sorted(got | reused)} != "
                f"{sorted(want)}")
            assert not (got & reused), (
                f"{m} {env}: queued a reused forward cell {got & reused}")


def test_reverse_only_in_the_two_k1_environments_and_exactly_eight(s3):
    """Reverse KL is the RLHF-practice comparison, not the canon. It is
    defined only where the retention question is sharp (k = 1). A reverse
    cell in env3 would be read as part of the ladder."""
    rows, _, _, _ = s3
    cells = _parsed(_tags(rows))
    rev = [c for c in cells if c[1].startswith("revlam")]
    assert len(rev) == 8, [f"{m}/{a}/{e}" for m, a, e in rev]
    assert {e for _, _, e in rev} == set(REV_ENVS), {e for _, _, e in rev}
    assert {a for _, a, _ in rev} == {f"revlam{_num(l)}" for l in REV_LAMS}
    for m in MODELS:
        for env in REV_ENVS:
            got = {a for mm, a, e in cells
                   if mm == m and e == env and a.startswith("revlam")}
            assert got == {"revlam1", "revlam8"}, (m, env, got)


def test_no_reverse_cell_in_env3(s3):
    rows, _, _, _ = s3
    tags = _tags(rows)
    bad = [t for t, c in zip(tags, _parsed(tags))
           if c[1].startswith("revlam") and c[2] == (0.5, 0.2)]
    assert not bad, bad


def test_sft_rows_are_lambda_zero_and_direction_neutral(s3):
    """The sft arm is lambda = 0. It has no direction because there is no
    anchor term to take a direction. A tag carrying 'fwd' or 'rev' over
    kl_beta = 0 would be sorted into the forward or reverse ladder by
    every downstream grouping."""
    rows, sub, _, _ = s3
    cols = _queue_cols(sub)
    for d in _dicts(rows, cols):
        tag = d["tag"]
        if TAG_RE.match(tag)["arm"] != "sft":
            continue
        assert "fwd" not in tag and "rev" not in tag, tag
        assert _col(d, "style") == "sft", d
        assert float(_col(d, "beta")) == 0.0, d


def test_non_sft_rows_are_sft_kl_with_the_lambda_the_tag_advertises(s3):
    rows, sub, _, _ = s3
    cols = _queue_cols(sub)
    for d in _dicts(rows, cols):
        arm = TAG_RE.match(d["tag"])["arm"]
        if arm == "sft":
            continue
        assert _col(d, "style") == "sft_kl", d["tag"]
        assert float(_col(d, "beta")) == LAM_OF_ARM[arm], (
            f"{d['tag']}: kl_beta column {d.get('beta')} != "
            f"{LAM_OF_ARM[arm]} -- lambda == the config field kl_beta")


def test_direction_token_agrees_with_the_kldir_column_in_every_row(s3):
    """The single failure that inverts the headline and is invisible
    everywhere downstream: a tag saying fwd over a run that trained
    reverse."""
    rows, sub, _, _ = s3
    cols = _queue_cols(sub)
    assert "kldir" in cols, cols
    for d in _dicts(rows, cols):
        arm = TAG_RE.match(d["tag"])["arm"]
        kldir = d["kldir"]
        assert kldir != "", (
            f"{d['tag']}: empty kldir column. The sub pins "
            f"KL_DIRECTION=$(kldir); an empty value reaches the runner "
            f"as KL_DIRECTION='' which is not one of "
            f"reverse/forward/js.")
        if arm.startswith("fwdlam"):
            assert kldir == "forward", d["tag"]
        elif arm.startswith("revlam"):
            assert kldir == "reverse", d["tag"]


# ----------------------------------------------------------- THE BIG ONE
def test_kldir_is_referenced_in_env_AND_declared_by_the_queue(s3):
    """BOTH HALVES OR NOTHING.

    `environment = "... KL_DIRECTION=$(kldir) ..."` without `kldir` on
    the queue line makes Condor expand the macro to nothing. The job then
    starts with KL_DIRECTION set to the empty string, and every FORWARD
    cell in this wave is decided by whatever the runner does with that
    value -- not by the tag, and not by anything visible in the figure.

    (Verified against run_pokec_gated_lm.py: kl_direction =
    _env_or("KL_DIRECTION", "reverse") -> os.environ.get returns "" for
    an empty assignment, so today that raises. The converse hole is the
    quiet one: if the env line omits KL_DIRECTION altogether, the
    default 'reverse' applies to the forward ladder in silence. Both are
    ruled out below.)
    """
    _, sub, _, smoke_sub = s3
    for s, what in ((sub, "production"), (smoke_sub, "smoke")):
        if not s:
            continue
        env = _env_line(s)
        assert "KL_DIRECTION=$(kldir)" in env, (
            f"{what} sub does not put KL_DIRECTION on the queue")
        assert "KL_DIRECTION=forward" not in env, what
        assert "KL_DIRECTION=reverse" not in env, what
        cols = _queue_cols(s)
        assert "kldir" in cols, (
            f"{what} sub reads $(kldir) but the queue line declares "
            f"{cols} -- Condor expands the macro to the empty string")


def test_every_macro_the_env_and_arguments_reference_is_declared(s3):
    """Generalisation of the above: no $(macro) anywhere in the sub may
    reference a column the queue line does not supply. One undeclared
    macro is one silently-defaulted knob."""
    _, sub, _, smoke_sub = s3
    for s, what in ((sub, "production"), (smoke_sub, "smoke")):
        if not s:
            continue
        cols = set(_queue_cols(s))
        used = set(re.findall(r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)", s))
        # Condor's own automatic macros are always available.
        auto = {"Cluster", "Process", "ClusterId", "ProcId", "Item",
                "ItemIndex", "Step", "Row", "ENV"}
        undeclared = {u for u in used if u not in cols and u not in auto}
        assert not undeclared, (
            f"{what} sub references undeclared macros {sorted(undeclared)}; "
            f"queue declares {sorted(cols)}")


# ------------------------------------------------------------------- reuse
def test_the_four_archived_qwu_cells_are_never_queued(s3, generated):
    rows, _, smoke, _ = s3
    s3_tags = set(_tags(rows)) | set(_tags(smoke))
    for t in REUSE_TAGS:
        assert t not in s3_tags, f"reuse tag queued by the S3 wave: {t}"
    # ... and no S3 config file may contain the string at all (a reuse
    # tag pasted into a comment column would also be queued).
    for name in (CFG, CFG_SMOKE):
        body = generated.get(name, "")
        for t in REUSE_TAGS:
            assert t not in body, f"{name} contains reuse tag {t}"


def test_no_ten_round_scout_tag_is_referenced_or_reused(s3, generated):
    """The pofdkd_ cells are 10-round SCOUTS. A 10-round trajectory
    cannot satisfy a 100-round retention arm: the whole claim lives in
    the late window (rounds 81-100)."""
    for name in (CFG, CFG_SMOKE):
        body = generated.get(name, "")
        assert "pofdkd_" not in body, f"{name} references a 10-round scout"
        assert "pofdkdsmk_" not in body, name
    rows, _, _, _ = s3
    for t in _tags(rows):
        assert t.endswith("_r100"), t


# ------------------------------------------------------------ model plumbing
def test_qwen3_disables_thinking_and_qwen25_does_not(s3):
    """Qwen3-8B is a hybrid-reasoning checkpoint. With thinking ON it
    emits a <think> block; the parser then reads a reasoning trace as an
    opinion, and the failure presents as a clean constant, not as an
    error. Qwen2.5 has no thinking mode, so pinning CHAT_THINKING=0 on it
    would be a silent template change on the model that anchors the
    entire archived comparison."""
    rows, sub, smoke, smoke_sub = s3
    for s in (sub, smoke_sub):
        if s:
            assert "CHAT_THINKING=$(chatthink)" in _env_line(s), \
                "CHAT_THINKING must ride the queue: one sub, two models"
    cols = _queue_cols(sub)
    for d in _dicts(rows, cols):
        model = TAG_RE.match(d["tag"])["model"]
        ct = _col(d, "chatthink")
        if model == "qwen3_8b":
            assert ct == "0", f"{d['tag']}: CHAT_THINKING={ct!r}, want '0'"
        else:
            assert ct != "0", (
                f"{d['tag']}: Qwen2.5 must not disable thinking (got "
                f"{ct!r})")


def test_base_model_column_matches_the_model_token(s3):
    rows, sub, _, _ = s3
    cols = _queue_cols(sub)
    for d in _dicts(rows, cols):
        model = TAG_RE.match(d["tag"])["model"]
        assert _col(d, "basemodel") == MODELS[model], d["tag"]


# ----------------------------------------------------------------- surface
def test_environment_and_k_columns_match_the_tag(s3):
    """A tag that says w0p5_k1 over a row that runs W=1 is an
    environment relabelling. Nothing downstream reads the config back
    against the tag."""
    rows, sub, _, _ = s3
    cols = _queue_cols(sub)
    for d in _dicts(rows, cols):
        m = TAG_RE.match(d["tag"])
        w, k = W_OF_TOK[m["w"]], K_OF_TOK[m["k"]]
        assert float(_col(d, "wplat")) == w, d["tag"]
        assert float(_col(d, "lam", "innate", "k")) == k, d["tag"]


def test_seed_zero_and_hundred_rounds_everywhere(s3):
    rows, sub, _, _ = s3
    cols = _queue_cols(sub)
    for d in _dicts(rows, cols):
        assert int(_col(d, "seed")) == SEED, d["tag"]
        assert int(_col(d, "nrounds")) == ROUNDS, d["tag"]


def test_homophily_gamma_stays_zero(s3):
    """Project-wide policy: no homophily selection bias in any pofd sim.
    It also keeps RNG-matched baselines possible."""
    rows, sub, _, _ = s3
    cols = _queue_cols(sub)
    if "gamma" not in cols:
        pytest.skip("no gamma column in this row template")
    for d in _dicts(rows, cols):
        assert float(d["gamma"]) == 0.0, d["tag"]


def test_h100_train_cap_lora_rank_and_horizon_are_pinned(s3):
    rows, sub, _, _ = s3
    req = [l for l in sub.splitlines() if l.startswith("requirements")]
    assert req and H100 in req[0], req
    env = _env_line(sub)
    assert "TRAIN_CAP=723" in env, env
    assert "LORA_R=512" in env, env
    assert "USE_LORA=" in env, env
    if "N_ROUNDS=$(nrounds)" in env:
        cols = _queue_cols(sub)
        assert all(int(_col(d, "nrounds")) == ROUNDS
                   for d in _dicts(rows, cols))
    else:
        assert f"N_ROUNDS={ROUNDS}" in env, env
    assert "AI_GATE_MODE=all_open" in env, env
    assert "PEER_GATE_MODE=all_open" in env, env
    assert "SAVE_RAW_GEN=1" in env, (
        "raw_gen_log.json.gz is where parse_fail_frac lives; without it "
        "the checker's parse gate has nothing to read")


def test_sub_requests_a_gpu(s3):
    _, sub, _, _ = s3
    assert any(l.startswith("request_gpus") and l.strip().endswith("1")
               for l in sub.splitlines()), sub[:400]


# ------------------------------------------------------------------- smoke
def test_smoke_is_the_pinned_cell(s3):
    rows, _, smoke, smoke_sub = s3
    assert len(smoke) == 1
    tag = _tags(smoke)[0]
    assert tag == SMOKE_TAG, f"{tag!r} != {SMOKE_TAG!r}"
    if smoke_sub:
        cols = _queue_cols(smoke_sub)
        d = _dicts(smoke, cols)[0]
        assert d["kldir"] == "reverse", d
        assert _col(d, "style") == "sft_kl", d
        assert float(_col(d, "beta")) == 1.0, d
        assert float(_col(d, "wplat")) == 1.0, d
        assert float(_col(d, "lam", "innate", "k")) == 1.0, d
        assert int(_col(d, "nrounds")) == SMOKE_ROUNDS, d
        assert _col(d, "basemodel") == MODELS["qwen3_8b"], d
        assert _col(d, "chatthink") == "0", d


def test_smoke_tag_cannot_shadow_or_be_shadowed_by_production(s3):
    """`runs/pokec_gated_lm/pofds3_*` is how every downstream tool
    collects the wave. A smoke tag that prefix-matches production would
    put a 3-round run into a 100-round figure; a production tag that
    prefix-matches the smoke would put a 100-round run through the smoke
    gate's 3-round horizon check."""
    rows, _, smoke, _ = s3
    prod = _tags(rows)
    for st in _tags(smoke):
        assert st.startswith(SMOKE_PREFIX), st
        assert not st.startswith(PROD_PREFIX), (
            f"{st}: the smoke prefix must not be a prefix of the "
            f"production prefix")
        for pt in prod:
            assert st != pt
            assert not st.startswith(pt), (st, pt)
            assert not pt.startswith(st), (pt, st)
    for pt in prod:
        assert pt.startswith(PROD_PREFIX) and not pt.startswith(SMOKE_PREFIX)


# ------------------------------------------------- cross-wave tag collisions
def test_s3_tags_collide_with_nothing_else_in_the_generator(generated):
    """565 KB of generator, ~350 config files, one namespace of run
    directories. A tag reused across waves overwrites an archived run
    dir on the cluster."""
    _require_s3(generated)
    s3_files = {CFG, CFG_SMOKE}
    mine, theirs = set(), {}
    for name, body in generated.items():
        if not name.startswith("configs_pofd_"):
            continue
        tags = {l.split(",")[0].strip() for l in body.splitlines()
                if l.strip()}
        if name in s3_files:
            mine |= tags
        else:
            for t in tags:
                theirs[t] = name
    clash = {t: theirs[t] for t in mine if t in theirs}
    assert not clash, f"S3 tags already queued elsewhere: {clash}"
    assert mine, "no S3 tags found"


# ------------------------------------------------ archived waves are untouched
def test_verify_leaves_every_preexisting_generated_file_unchanged():
    """An edit inside gen_pofd_sweep.py that perturbs a SHARED constant
    (a row template, a model table, a tag helper) silently rewrites
    archived waves' config files. --verify diffs every generated file
    against disk; nothing outside the S3 wave may move."""
    r = subprocess.run([sys.executable, GEN, "--verify"],
                       capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "USE_TF": "0"})
    lines = [l for l in r.stdout.splitlines() if l.startswith("[verify]")]
    assert lines, r.stdout[-2000:] + r.stderr[-2000:]
    bad = [l for l in lines if "MISMATCH" in l and "section3" not in l]
    assert not bad, "archived generated files moved:\n" + "\n".join(bad)


def test_s3_files_are_written_to_disk_and_match_the_generator(generated):
    """Separate from the test above: once the block lands, its config and
    sub must actually be on disk, or `submit_pofd_sweep.sh <BID>
    section3_retention` has nothing to queue."""
    _require_s3(generated)
    r = subprocess.run([sys.executable, GEN, "--verify"],
                       capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "USE_TF": "0"})
    lines = [l for l in r.stdout.splitlines()
             if l.startswith("[verify]") and "section3" in l]
    assert lines, "--verify never mentions a section3 file"
    bad = [l for l in lines if "MISMATCH" in l]
    assert not bad, ("S3 files not written (run gen_pofd_sweep.py):\n"
                     + "\n".join(bad))


# ------------------------------------------------ semantics the wave rests on
def test_forward_is_kl_ref_policy_and_reverse_is_kl_policy_ref():
    """Section 3's entire reading of the ladder depends on this mapping.
    In _anchor_divergence_per_token, logp is the POLICY (grad flows) and
    logq the frozen REFERENCE. 'forward' must therefore compute
    sum q (log q - log p) = KL(ref || policy). If this ever flips, every
    forward panel becomes a reverse panel with no other symptom."""
    import torch
    spec = importlib.util.spec_from_file_location("_kl_sft_s3", KL_SFT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod._anchor_divergence_per_token
    torch.manual_seed(0)
    logp = torch.log_softmax(torch.randn(3, 5), dim=-1)
    logq = torch.log_softmax(torch.randn(3, 5), dim=-1)
    fwd = fn(logp, logq, "forward")
    rev = fn(logp, logq, "reverse")
    want_fwd = (logq.exp() * (logq - logp)).sum(-1)      # KL(ref || pi)
    want_rev = (logp.exp() * (logp - logq)).sum(-1)      # KL(pi || ref)
    assert torch.allclose(fwd, want_fwd, atol=1e-6)
    assert torch.allclose(rev, want_rev, atol=1e-6)
    assert not torch.allclose(fwd, rev)


def test_runner_default_direction_is_still_reverse():
    """Pinned because the 'undeclared kldir column' failure mode is only
    dangerous while the default is a REAL direction. If someone changes
    the default to a hard error, the BIG ONE test above becomes belt and
    braces -- but until then an omitted KL_DIRECTION silently trains the
    forward ladder in reverse."""
    with open(RUNNER) as fh:
        src = fh.read()
    assert '_env_or("KL_DIRECTION", "reverse")' in src, (
        "runner default changed; re-read "
        "test_kldir_is_referenced_in_env_AND_declared_by_the_queue")


# --------------------------------------------------- REF_REPLAY_ALL_FIXED
def test_all_fixed_guard_is_conditioned_on_ref_replay_on():
    """q = 0 sets BOTH q == 0 and a reference run. An earlier version
    rejected exactly that: the branch handling q > 0 had been narrowed to
    exclude all-fixed, so q = 0 fell into the elif that rejects a
    reference run without q > 0, and every q = 0 job held on the cluster
    with 'REF_REPLAY_REF_RUN / REF_REPLAY_SEED without REF_REPLAY_Q>0'.
    The guard must key off ref_replay_on, not off falling through."""
    src = open(RUNNER).read()
    assert "elif not ref_replay_on and (ref_replay_ref_run" in src, (
        "the REF_REPLAY inert-knob guard must be conditioned on "
        "`not ref_replay_on`; otherwise ALL_FIXED (q=0 + a reference run) "
        "lands in it and every q=0 job is held")
    assert "ref_replay_on = ref_replay_q > 0 or ref_replay_all_fixed" in src
    assert ('ref_replay_all_fixed = _env_int("REF_REPLAY_ALL_FIXED", 0) == 1'
            in src)


def test_generator_marks_only_q0_as_all_fixed(generated):
    """The allfixed column must be 1 for q = 0 and 0 everywhere else. A
    q > 0 row carrying the flag would silently discard its live set."""
    rows = [r for r in
            generated["configs_pofd_section3_label_mix.txt"].splitlines()
            if r.strip()]
    assert len(rows) == 4, rows
    for r in rows:
        c = [x.strip() for x in r.split(",")]
        assert (c[18] == "1") == (float(c[17]) == 0.0), r
    assert sum(1 for r in rows
               if [x.strip() for x in r.split(",")][18] == "1") == 1, rows
