"""ADVERSARIAL tests for the redesigned-Figure-3 wave generator (pofdf3_),
UNIFIED 108-CELL GRID (2026-08-24 v2).

Figure 3 is a 108-cell grid of which only 91 can be GPU runs, 28 of those
are already on disk, and the remaining 17 cells are free for three
DIFFERENT structural reasons. Every one of those reasons is a way to get
the figure quietly wrong:

  * queue a beta = 0 cell and you burn a GPU job recomputing twin_raw,
    which every run already contains;
  * queue beta = 1 at more than one gamma and you produce "different"
    conditions that are algebraically the same run;
  * queue lambda = inf and you have not run the frozen condition at all
    -- you have run a trained one and mislabelled it;
  * re-queue one of the 28 archived cells and you overwrite a result the
    paper already cites.
None of those show up in a trajectory. This file is where they are cheap.

Harness mirrored from tests/test_section3_generator.py: gen_pofd_sweep
.main() runs with HERE redirected into a throwaway dir and we read
exactly the bytes that would be written, so the tests bind to the
artifact Condor submits and not to how the F3 block is factored. Column
positions are never hard-coded -- they come off each sub's own
`queue ... from` line.

Contract (pinned 2026-08-24 v2):
  keys     fig3_full_loop / fig3_full_loop_smoke / fig3_full_loop_ext
  tag      pofdf3_qwen3_8b_{arm}_sw100_eaopen_w{beta}_k{gamma}_esopen
           _anch2_s0_r{30|60|100}          (smoke: pofdf3smk_..._r3)
  names    beta = W_PLAT (wplat), gamma = INNATE_LAMBDA (lam),
           lambda = kl_beta (beta). They collide; the queue is the
           authority.
  grid     beta {0,.25,.5,.75,1} x gamma {0,.2,.5,1} x lambda
           {0,.25,.5,1,2,4,8,inf}, deduplicated:
           beta=0 -> one twin per gamma; beta=1 -> one cell per lambda;
           lambda=inf -> frozen CPU replay
  counts   108 unique = 91 gpu (28 reused + 63 new) + 13 frozen
           (4 reused + 9 CPU replays) + 4 twins; 1 smoke;
           new = 56 at beta {.25,.75} + 7 at (beta=.5, gamma=0)
"""
from __future__ import annotations

import contextlib
import glob
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDOR = os.path.join(ROOT, "experiments", "condor")
GEN = os.path.join(CONDOR, "gen_pofd_sweep.py")

KEY = "fig3_full_loop"
SMOKE_KEY = "fig3_full_loop_smoke"
EXT_KEY = "fig3_full_loop_ext"
CFG = f"configs_pofd_{KEY}.txt"
SUB = f"at_pofd_{KEY}.sub"
CFG_SMOKE = f"configs_pofd_{SMOKE_KEY}.txt"
SUB_SMOKE = f"at_pofd_{SMOKE_KEY}.sub"
CFG_EXT = f"configs_pofd_{EXT_KEY}.txt"

N_CELLS = 108
N_GPU = 91
N_NEW_GPU = 63
N_REUSED_GPU = 28
N_FROZEN = 13
N_NEW_FROZEN = 9
N_TWIN = 4
ROUNDS = 30
SMOKE_ROUNDS = 3
SWEEPS = 100
MODEL_ID = "Qwen/Qwen3-8B"

BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
GAMMAS = [0.0, 0.2, 0.5, 1.0]
FINITE_LAMS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

TAG_RE = re.compile(
    r"^pofdf3_qwen3_8b_(?P<arm>sft|fwdlam[0-9p]+)_sw100_eaopen"
    r"_w(?P<w>[0-9p]+)_k(?P<k>[0-9p]+)_esopen_anch2_s0_r30$")
EXT_TAG_RE = re.compile(
    r"^pofdf3_qwen3_8b_(?P<arm>sft|fwdlam[0-9p]+)_sw100_eaopen"
    r"_w(?P<w>[0-9p]+)_k(?P<k>[0-9p]+)_esopen_anch2_s0_r(?P<r>60|100)$")
SMOKE_TAG_RE = re.compile(
    r"^pofdf3smk_qwen3_8b_fwdlam8_sw100_eaopen_w0p25_k0_esopen"
    r"_anch2_s0_r3$")


def _tok(v):
    return f"{v:g}".replace(".", "p")


def _untok(s):
    return float(s.replace("p", "."))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    """The generator module itself -- for the GRID definition, which the
    checker and analyzer also read, so all three cannot disagree."""
    return _load(GEN, "_gen_f3_test")


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("gen_f3"))
    for j in glob.glob(os.path.join(CONDOR, "*.json")):
        shutil.copy(j, tmp)
    mod = _load(GEN, "_gen_pofd_f3")
    mod.HERE = tmp
    # the extension-request path is resolved at import time (absolute);
    # re-point it into the sandbox copy so the test reads what main() read
    mod.F3X_REQUEST_PATH = os.path.join(tmp, "fig3_extension_request.json")
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


def _rows(generated, name):
    assert name in generated, f"{name} was not generated"
    return [r for r in generated[name].splitlines() if r.strip()]


def _queue_cols(sub):
    q = [l for l in sub.splitlines() if l.startswith("queue ")]
    assert len(q) == 1, f"expected one queue line, got {len(q)}"
    head = q[0].split(" from ")[0]
    assert head != q[0], "queue line does not read from a config file"
    return [c.strip() for c in head[len("queue "):].split(",") if c.strip()]


def _env(sub):
    ls = [l for l in sub.splitlines() if l.startswith("environment")]
    assert len(ls) == 1
    return ls[0]


def _dicts(rows, cols):
    out = []
    for r in rows:
        vals = [x.strip() for x in r.split(",")]
        assert len(vals) == len(cols), (
            f"row declares {len(vals)} fields, queue line declares "
            f"{len(cols)} -- every column after the first disagreement is "
            f"silently shifted.\n  cols={cols}\n  row={r}")
        out.append(dict(zip(cols, vals)))
    return out


# ================================================================== grid
def test_grid_is_108_unique_cells_of_three_kinds(gen):
    cells = gen.f3_cells()
    assert len(cells) == N_CELLS, len(cells)
    assert len(set(cells)) == N_CELLS, "duplicate cells"
    kinds = {}
    for *_r, k in cells:
        kinds[k] = kinds.get(k, 0) + 1
    assert kinds == {"gpu": N_GPU, "frozen": N_FROZEN,
                     "twin": N_TWIN}, kinds


def test_the_grid_is_one_unified_ladder_not_two_panels(gen):
    """Every beta in (0, 1) carries the COMPLETE 8-dose ladder at every
    gamma -- the old panel-A (4 doses) / panel-B (beta .5 only) split is
    gone."""
    cells = gen.f3_cells()
    for b in (0.25, 0.5, 0.75):
        for g in GAMMAS:
            lams = {l for (b2, g2, l, _k) in cells if b2 == b and g2 == g}
            assert lams == set(FINITE_LAMS) | {gen.F3_INF}, (b, g, lams)


def test_beta_zero_collapses_lambda_and_is_never_a_job(gen):
    zero = [c for c in gen.f3_cells() if c[0] == 0.0]
    assert len(zero) == len(GAMMAS), zero
    assert all(k == "twin" for *_r, k in zero)
    assert {c[1] for c in zero} == set(GAMMAS)
    assert all(c[2] is None for c in zero), "lambda must not survive at beta=0"
    assert not any(b == 0.0 for (b, _g, _l) in gen.f3_missing_gpu())


def test_beta_one_deduplicates_gamma_over_the_full_ladder(gen):
    one = [c for c in gen.f3_cells() if c[0] == 1.0]
    assert {c[1] for c in one} == {None}, "beta=1 must not carry a gamma"
    assert len(one) == 8, one            # one per lambda, 7 finite + inf
    assert {c[2] for c in one} == set(FINITE_LAMS) | {gen.F3_INF}


def test_lambda_inf_is_never_a_gpu_row(gen):
    assert all(not math.isinf(l) for (_b, _g, l) in gen.f3_missing_gpu())
    froz = [c for c in gen.f3_cells() if c[3] == "frozen"]
    assert len(froz) == N_FROZEN
    assert all(math.isinf(c[2]) for c in froz)
    with pytest.raises(AssertionError):
        gen.f3_row(0.5, 0.0, gen.F3_INF)
    with pytest.raises(AssertionError):
        gen.f3_row(0.0, 0.0, 1.0)


def test_reuse_covers_exactly_the_28_audited_cells(gen):
    """The complete beta=.5 ladder at gamma {.2,.5,1} (21) plus the
    complete beta=1 ladder (7) -- including the beta=1 lambda
    {.25,.5,2,8} cells the first draft omitted."""
    want = {(0.5, g, l) for g in (0.2, 0.5, 1.0) for l in FINITE_LAMS}
    want |= {(1.0, None, l) for l in FINITE_LAMS}
    assert set(gen.F3_REUSED) == want, set(gen.F3_REUSED) ^ want
    assert len(gen.F3_REUSED) == N_REUSED_GPU
    # the four previously-omitted beta=1 dose cells, by exact tag
    assert gen.F3_REUSED[(1.0, None, 0.25)].startswith(
        "pofdlam_qwen3_8b_fwdlam0p25_sw100_eaopen_w1_k1")
    assert gen.F3_REUSED[(1.0, None, 0.5)].startswith(
        "pofdlam_qwen3_8b_fwdlam0p5_sw100_eaopen_w1_k1")
    assert gen.F3_REUSED[(1.0, None, 2.0)].startswith(
        "pofdlam_qwen3_8b_fwdlam2_sw100_eaopen_w1_k1")
    assert gen.F3_REUSED[(1.0, None, 8.0)].startswith(
        "pofdps_qwen3_8b_fwdlam8_sw100_eaopen_w1_k1")


def test_the_63_missing_cells_are_exactly_the_specified_split(gen):
    want = {(b, g, l) for b in (0.25, 0.75)
            for g in GAMMAS for l in FINITE_LAMS}          # 56
    want |= {(0.5, 0.0, l) for l in FINITE_LAMS}           # + 7
    assert set(gen.f3_missing_gpu()) == want
    assert len(want) == N_NEW_GPU == 63
    assert len(gen.f3_missing_frozen()) == N_NEW_FROZEN == 9


# ================================================================== rows
def test_exactly_63_unique_new_gpu_rows(generated):
    rows = _rows(generated, CFG)
    assert len(rows) == N_NEW_GPU, len(rows)
    tags = [r.split(",")[0].strip() for r in rows]
    assert len(set(tags)) == N_NEW_GPU


def test_every_row_matches_its_own_tag_column_by_column(generated):
    cols = _queue_cols(generated[SUB])
    for d in _dicts(_rows(generated, CFG), cols):
        m = TAG_RE.match(d["tag"])
        assert m, f"tag off contract: {d['tag']}"
        beta, gamma = _untok(m.group("w")), _untok(m.group("k"))
        lam = 0.0 if m.group("arm") == "sft" else _untok(
            m.group("arm")[len("fwdlam"):])
        assert float(d["wplat"]) == beta, d          # beta  = W_PLAT
        assert float(d["lam"]) == gamma, d           # gamma = INNATE_LAMBDA
        assert float(d["beta"]) == lam, d            # lambda = kl_beta
        assert d["style"] == ("sft" if lam == 0.0 else "sft_kl"), d
        assert 0.0 < beta < 1.0, ("beta=0 is the twin; the complete "
                                  "beta=1 ladder reuses", d)
        assert gamma in GAMMAS and lam in FINITE_LAMS, d
        assert int(d["sweeps"]) == SWEEPS, d
        assert d["nrounds"] == str(ROUNDS), d
        assert d["kldir"] == "forward", d
        assert d["gamma"] == "0.0", ("homophily gamma is a DIFFERENT gamma "
                                     "and stays 0", d)
        assert d["seed"] == "0", d
        assert int(d["iclk"]) == 0, d
        assert d["basemodel"] == MODEL_ID, d
        assert d["chatthink"] == "0", ("Qwen3 thinking must be pinned OFF", d)


def test_new_rows_are_exactly_the_audited_missing_cells(gen, generated):
    want = set(gen.f3_missing_gpu())
    got = set()
    cols = _queue_cols(generated[SUB])
    for d in _dicts(_rows(generated, CFG), cols):
        got.add((float(d["wplat"]), float(d["lam"]), float(d["beta"])))
    assert got == want, want ^ got


def test_smoke_is_one_three_round_cell_on_both_new_dials(generated):
    rows = _rows(generated, CFG_SMOKE)
    assert len(rows) == 1, len(rows)
    cols = _queue_cols(generated[SUB_SMOKE])
    d = _dicts(rows, cols)[0]
    assert SMOKE_TAG_RE.match(d["tag"]), d["tag"]
    assert d["nrounds"] == str(SMOKE_ROUNDS), d
    assert float(d["wplat"]) == 0.25, "beta=.25 has never been run"
    assert float(d["lam"]) == 0.0, "gamma=0 is the new no-re-anchor regime"
    assert float(d["beta"]) == 8.0, "strongest anchor is the demanding path"
    assert int(d["sweeps"]) == SWEEPS


def test_smoke_sub_names_the_gate_command_with_the_smoke_flag(generated):
    assert "check_fig3_full_loop.py --smoke" in generated[SUB_SMOKE]
    # and the production sub must NOT tell you to gate the grid with it
    main_gate = [l for l in generated[SUB].splitlines()
                 if "check_fig3_full_loop.py" in l]
    assert main_gate and all("--smoke" not in l for l in main_gate)


# ============================================================ extensions
def test_extension_key_emits_the_committed_request(gen, generated):
    """The seeded request: (beta=.5, gamma=.2, lambda=1) at 60 rounds --
    the one completed cell verified unsettled (drift +0.0107 > 0.005)."""
    if CFG_EXT not in generated:
        pytest.skip("no fig3_extension_request.json committed")
    rows = _rows(generated, CFG_EXT)
    cols = _queue_cols(generated[f"at_pofd_{EXT_KEY}.sub"])
    seen = set()
    for d in _dicts(rows, cols):
        m = EXT_TAG_RE.match(d["tag"])
        assert m, f"ext tag off contract: {d['tag']}"
        assert d["nrounds"] == m.group("r"), d
        seen.add((float(d["wplat"]), float(d["lam"]),
                  float(d["beta"]), int(d["nrounds"])))
    assert (0.5, 0.2, 1.0, 60) in seen, seen
    # ext tags never collide with base rows or archived cells
    base = {r.split(",")[0].strip() for r in _rows(generated, CFG)}
    ext = {r.split(",")[0].strip() for r in rows}
    assert not (ext & base)
    assert not (ext & set(gen.F3_REUSED.values()))


def test_extension_requests_are_validated_against_the_grid(gen, tmp_path):
    """A malformed request must fail the GENERATOR, not a wave."""
    import json as _json
    orig = gen.F3X_REQUEST_PATH
    try:
        bad = tmp_path / "req.json"
        gen.F3X_REQUEST_PATH = str(bad)
        # outside the grid
        bad.write_text(_json.dumps({"cells": [
            {"beta": 0.4, "gamma": 0.2, "lam": 1.0, "rounds": 60}]}))
        with pytest.raises(AssertionError):
            gen.f3x_requests()
        # a 60-round request for a cell whose reused artifact has 60
        bad.write_text(_json.dumps({"cells": [
            {"beta": 1.0, "gamma": None, "lam": 8.0, "rounds": 60}]}))
        with pytest.raises(AssertionError):
            gen.f3x_requests()
        # ...but 100 rounds for the same cell is legitimate
        bad.write_text(_json.dumps({"cells": [
            {"beta": 1.0, "gamma": None, "lam": 8.0, "rounds": 100}]}))
        assert gen.f3x_requests() == [(1.0, None, 8.0, 100)]
        # bad horizon
        bad.write_text(_json.dumps({"cells": [
            {"beta": 0.5, "gamma": 0.2, "lam": 1.0, "rounds": 45}]}))
        with pytest.raises(AssertionError):
            gen.f3x_requests()
    finally:
        gen.F3X_REQUEST_PATH = orig


# =================================================================== sub
def test_sub_pins_the_corrected_operator_and_the_open_gates(generated):
    subs = [SUB, SUB_SMOKE]
    if f"at_pofd_{EXT_KEY}.sub" in generated:
        subs.append(f"at_pofd_{EXT_KEY}.sub")
    for key in subs:
        env = _env(generated[key])
        assert "AI_GATE_REFERENCE=anchor" in env, key
        assert "AI_GATE_MODE=all_open" in env, key
        assert "PEER_GATE_MODE=all_open" in env, key
        assert "AB_SWEEPS=$(sweeps)" in env, key
        assert "INNATE_LAMBDA=$(lam)" in env, key
        assert "WITH_TWIN=1" in env, ("the beta=0 column IS twin_raw", key)
        assert "SAVE_RAW_GEN=1" in env, key
        assert "FRESH_EACH_ROUND=$(fresh)" in env, key
        assert "REF_REPLAY" not in env and "INNATE_CLAMP" not in env, key


def test_sub_env_reads_only_declared_queue_columns(generated):
    subs = [SUB, SUB_SMOKE]
    if f"at_pofd_{EXT_KEY}.sub" in generated:
        subs.append(f"at_pofd_{EXT_KEY}.sub")
    for key in subs:
        sub = generated[key]
        cols = set(_queue_cols(sub))
        for line in sub.splitlines():
            if line.startswith(("environment", "request_", "arguments")):
                for macro in re.findall(r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)",
                                        line):
                    assert macro in cols, (key, macro, sorted(cols))


def test_every_new_sub_uses_the_idempotent_executable(generated):
    subs = [SUB, SUB_SMOKE]
    if f"at_pofd_{EXT_KEY}.sub" in generated:
        subs.append(f"at_pofd_{EXT_KEY}.sub")
    for key in subs:
        ex = [l for l in generated[key].splitlines()
              if l.startswith("executable")]
        assert len(ex) == 1
        assert ex[0].rstrip().endswith("run_one_pokec_gated_idempotent.sh")


# ================================================================= reuse
def test_no_archived_cell_is_ever_requeued(gen, generated):
    new = {r.split(",")[0].strip() for r in _rows(generated, CFG)}
    reused = set(gen.F3_REUSED.values())
    assert not (new & reused), new & reused
    # and each reuse tag must be a tag some OTHER key really produces, so
    # a rename in the mem/lam/ps blocks cannot silently orphan a cell
    everywhere = set()
    for name, text in generated.items():
        if name.startswith("configs_pofd_"):
            everywhere |= {r.split(",")[0].strip()
                           for r in text.splitlines() if r.strip()}
    missing = sorted(t for t in reused if t not in everywhere)
    assert not missing, f"declared reuse not generated anywhere: {missing}"


def test_new_tags_collide_with_nothing(generated):
    counts = {}
    for name, text in generated.items():
        if not name.startswith("configs_pofd_"):
            continue
        for r in text.splitlines():
            if r.strip():
                counts.setdefault(r.split(",")[0].strip(), []).append(name)
    mine = {r.split(",")[0].strip() for r in _rows(generated, CFG)}
    if CFG_EXT in generated:
        mine |= {r.split(",")[0].strip() for r in _rows(generated, CFG_EXT)}
    for tag in mine:
        assert len(counts[tag]) == 1, (
            f"{tag} is queued by {counts[tag]} -- a double-queue into one "
            f"run dir is a WRITE RACE, not an idempotent no-op")


def test_reused_figure3_configs_are_byte_unchanged(generated):
    """The 28 archived cells this figure reuses must not move by a byte."""
    for key in ("section3_peer_sweeps", "section3_memory",
                "section3_lambda_fill"):
        name = f"configs_pofd_{key}.txt"
        on_disk = os.path.join(CONDOR, name)
        if not os.path.exists(on_disk):
            pytest.skip(f"{name} not on disk")
        with open(on_disk) as fh:
            assert generated[name] == fh.read(), name


def test_submit_script_registers_all_three_keys():
    with open(os.path.join(CONDOR, "submit_pofd_sweep.sh")) as fh:
        s = fh.read()
    assert ('fig3_full_loop|fig3_full_loop_smoke|fig3_full_loop_ext) '
            'TARGETS="$WHAT"') in s
    assert s.count("|fig3_full_loop[_smoke|_ext]") == 3
