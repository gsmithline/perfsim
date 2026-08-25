"""ADVERSARIAL tests for the two 2026-08-24 waves' GENERATOR output:

  section4_gate_anch2[_smoke]{_fixed,_evo}   72 production + 4 smoke jobs
  fig4_family_prior_repl30                   18 production jobs

Both waves exist to CORRECT a provenance/gate detail on already-published
surfaces, so the failure mode that matters is not "the run crashed" -- it
is "the run looks perfect and answers a different question". A sub whose
`environment` reads a column the `queue` line never declares, a tag that
collides with an archived Section-4 cell, an eps that silently shifts by
one column: none of those show up in a trajectory. This file is the last
place they are cheap.

Design notes (mirrored from tests/test_section3_generator.py):
  * we do NOT import the S4G/F4R helpers by name. gen_pofd_sweep.main()
    runs with HERE redirected into a throwaway dir and we read exactly
    the bytes that would be written to experiments/condor/. The tests are
    therefore independent of how either block is factored and bound to
    the artifact Condor actually submits.
  * column positions are never hard-coded. They come off each sub's own
    `queue ... from ...` line, the only definition Condor itself reads,
    so a row/queue arity mismatch is its own test.
  * the Figure-4 field-by-field claim is checked against the ARTIFACT --
    the six pofdfam_ Figure-4 rows this same generator writes -- not
    against a second call of the builder that produced the F4R rows.

Contract (pinned 2026-08-24):
  S4G  mistral7b, movielens Action, 30 rounds, W_PLAT 0.5, INNATE_LAMBDA
       0.2, homophily gamma 0, ea {0.2, 1} x es {0, 0.2, 1} x arms
       {b0, d8} x conds {fixed, evolving} x seeds {0, 42, 43} = 72.
       AI_GATE_REFERENCE=anchor pinned in BOTH subs -> config
       population_update = nested_ai_anchored_then_social_v2.
       tag pofds4g_mistral7b_{arm}_{fixb20|evoall}_anch2_ea{ea}
           _w0p5_l0p2_es{es}_s{seed}
  F4R  the displayed Figure-4 condition (forward-KL lambda=1, ea1,
       es0p05, W 0.5, lam 0.2) on six checkpoints x seeds {0, 42, 43} at
       the MATCHED 30-round horizon = 18. Because the horizon now agrees
       with the displayed cells, the tag and the seed are the only
       columns an F4R row may move off its Figure-4 counterpart.
       tag pofdf4r_{slug}_b1_anch2_ea1_w0p5_l0p2_es0p05_r30_s{seed}
"""
from __future__ import annotations

import contextlib
import glob
import importlib.util
import io
import os
import re
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDOR = os.path.join(ROOT, "experiments", "condor")
GEN = os.path.join(CONDOR, "gen_pofd_sweep.py")

# ------------------------------------------------------------------ contract
S4G_FIXED_KEY = "section4_gate_anch2_fixed"
S4G_EVO_KEY = "section4_gate_anch2_evo"
S4G_SMOKE_FIXED_KEY = "section4_gate_anch2_smoke_fixed"
S4G_SMOKE_EVO_KEY = "section4_gate_anch2_smoke_evo"
F4R_KEY = "fig4_family_prior_repl30"
FAM_KEY = "fig2_family_prior_scout"

S4G_ARMS = ("b0", "d8")
S4G_GATES = (0.2, 1.0)
S4G_ESS = (0.0, 0.2, 1.0)
S4G_SEEDS = (0, 42, 43)
S4G_COND_TOK = {"fixed": "fixb20", "evolving": "evoall"}
S4G_ROUNDS = 30
S4G_SMOKE_ROUNDS = 3
S4G_N_PER_COND = 36
S4G_N_TOTAL = 72

F4R_MODELS = ("qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b",
              "mistral7b", "ministral8b")
F4R_SEEDS = (0, 42, 43)
F4R_ROUNDS = 30
F4R_N = 18
# the CURRENTLY DISPLAYED Figure-4 cells (plot_sft_family_prior_one_row.py)
FIG4_TAG = "pofdfam_{slug}_b1_ea1_w0p5_l0p2_es0p05_s0"

# the ONLY columns an F4R row may move off its Figure-4 cell
# (at the matched 30-round horizon nrounds agrees too)
F4R_ALLOWED_DIFF = {"tag", "seed"}

# archived Section-4 families -- a new tag may never land in one
OLD_S4_PREFIXES = ("pofdclamp_", "pofdevo_", "pofdreach_", "pofdpeer2_",
                   "pofdgate2d_", "pofdws2f_")

S4G_TAG_RE = re.compile(
    r"^pofds4g_mistral7b_(?P<arm>b0|d8)_(?P<cond>fixb20|evoall)"
    r"_anch2_ea(?P<ea>0p2|1)_w0p5_l0p2_es(?P<es>0|0p2|1)"
    r"_s(?P<seed>0|42|43)$")
S4G_SMOKE_TAG_RE = re.compile(
    r"^pofds4gsmk_mistral7b_(?P<arm>b0|d8)_(?P<cond>fixb20|evoall)"
    r"_anch2_ea1_w0p5_l0p2_es0p2_s0$")
F4R_TAG_RE = re.compile(
    r"^pofdf4r_(?P<slug>qwen7b|qwen3_8b|olmo7b|olmo3_7b|mistral7b"
    r"|ministral8b)_b1_anch2_ea1_w0p5_l0p2_es0p05_r30"
    r"_s(?P<seed>0|42|43)$")


def _tok(v):
    """Project tag grammar: 0.2 -> '0p2', 1.0 -> '1', 0.0 -> '0'."""
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
    Manifests are copied in first: some blocks resolve manifest paths at
    import time (absolute, unaffected by the patch) but a future block
    could resolve one at call time, and a missing manifest would read as
    a bug in these waves.
    """
    tmp = str(tmp_path_factory.mktemp("gen_s4g_f4r"))
    for j in glob.glob(os.path.join(CONDOR, "*.json")):
        shutil.copy(j, tmp)
    mod = _load(GEN, "_gen_pofd_s4g")
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


def _cfg(generated, key):
    name = f"configs_pofd_{key}.txt"
    if name not in generated:
        stray = [f for f in generated
                 if key.split("_")[0] in f and "configs_" in f]
        pytest.fail(f"{name} was not generated (contract key {key!r}); "
                    f"related files present: {stray}")
    return [r for r in generated[name].splitlines() if r.strip()]


def _sub(generated, key):
    name = f"at_pofd_{key}.sub"
    assert name in generated, f"{name} was not generated"
    return generated[name]


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


def _all_tags(generated):
    tags = {}
    for name, text in generated.items():
        if not name.startswith("configs_pofd_"):
            continue
        for r in text.splitlines():
            if r.strip():
                tags.setdefault(r.split(",")[0].strip(), []).append(name)
    return tags


# =============================================================== S4G: counts
def test_s4g_production_is_exactly_72_across_two_schemas(generated):
    fixed = _cfg(generated, S4G_FIXED_KEY)
    evo = _cfg(generated, S4G_EVO_KEY)
    assert len(fixed) == S4G_N_PER_COND, len(fixed)
    assert len(evo) == S4G_N_PER_COND, len(evo)
    assert len(fixed) + len(evo) == S4G_N_TOTAL


def test_s4g_smoke_is_three_rounds_and_covers_both_schemas(generated):
    for key in (S4G_SMOKE_FIXED_KEY, S4G_SMOKE_EVO_KEY):
        rows = _cfg(generated, key)
        assert len(rows) == len(S4G_ARMS), (key, len(rows))
        cols = _queue_cols(_sub(generated, key))
        for d in _dicts(rows, cols):
            assert S4G_SMOKE_TAG_RE.match(d["tag"]), d["tag"]
            assert d["nrounds"] == str(S4G_SMOKE_ROUNDS), d
            assert d["seed"] == "0", d


def test_s4g_grid_is_complete_and_has_no_extra_cells(generated):
    want = {(a, _tok(g), _tok(e), s)
            for a in S4G_ARMS for g in S4G_GATES
            for e in S4G_ESS for s in S4G_SEEDS}
    assert len(want) == S4G_N_PER_COND
    for cond, key in (("fixed", S4G_FIXED_KEY), ("evolving", S4G_EVO_KEY)):
        rows = _cfg(generated, key)
        got = set()
        for r in rows:
            m = S4G_TAG_RE.match(r.split(",")[0].strip())
            assert m, f"{cond}: tag off contract: {r.split(',')[0]}"
            assert m.group("cond") == S4G_COND_TOK[cond], (cond, m.group(0))
            got.add((m.group("arm"), m.group("ea"), m.group("es"),
                     int(m.group("seed"))))
        assert got == want, (cond, want ^ got)


# ============================================================ S4G: provenance
def test_every_new_tag_carries_the_anch2_token(generated):
    for key in (S4G_FIXED_KEY, S4G_EVO_KEY, S4G_SMOKE_FIXED_KEY,
                S4G_SMOKE_EVO_KEY, F4R_KEY):
        for r in _cfg(generated, key):
            tag = r.split(",")[0].strip()
            assert "_anch2_" in tag, (key, tag)


def test_both_s4g_subs_pin_the_corrected_gate_reference(generated):
    """The whole wave IS the gate reference. Inheriting the runner default
    would let a future default flip silently rewrite the experiment."""
    for key in (S4G_FIXED_KEY, S4G_EVO_KEY, S4G_SMOKE_FIXED_KEY,
                S4G_SMOKE_EVO_KEY, F4R_KEY):
        env = _env_line(_sub(generated, key))
        assert "AI_GATE_REFERENCE=anchor" in env, key
        assert "AI_GATE_REFERENCE=x0" not in env, key


def test_fixed_sub_carries_the_clamp_env_and_evo_sub_carries_none(generated):
    fenv = _env_line(_sub(generated, S4G_FIXED_KEY))
    assert "INNATE_CLAMP_MODE=$(cmode)" in fenv
    assert "INNATE_CLAMP_FRAC=0.2" in fenv
    assert "INNATE_CLAMP_SEED=$(seed)" in fenv
    assert "INNATE_CLAMP_PEER_MODE=stubborn" in fenv
    assert "SFT_EXCLUDE_CLAMPED=$(sftexcl)" in fenv
    eenv = _env_line(_sub(generated, S4G_EVO_KEY))
    assert "INNATE_CLAMP" not in eenv, (
        "the evolving sub must carry NO clamp env: a clamp key there "
        "changes the config surface the fixed/evolving contrast rests on")
    assert "SFT_EXCLUDE_CLAMPED" not in eenv


def test_s4g_never_queues_the_b0xa_source_exclusion(generated):
    """b0xa is a different experiment; SFT_EXCLUDE_CLAMPED must stay 0."""
    rows = _cfg(generated, S4G_FIXED_KEY)
    cols = _queue_cols(_sub(generated, S4G_FIXED_KEY))
    for d in _dicts(rows, cols):
        assert d["sftexcl"] == "0", d
        assert "b0xa" not in d["tag"], d


# ============================================================= S4G: env grid
def test_s4g_rows_match_their_own_tags_column_by_column(generated):
    arm_cols = {
        "b0": {"style": "sft", "beta": "0", "iclk": "0", "uselora": "1",
               "fresh": "1", "icldays": "0"},
        "d8": {"style": "frozen", "beta": "0", "iclk": "0", "uselora": "0",
               "fresh": "0", "icldays": "8"},
    }
    for cond, key in (("fixed", S4G_FIXED_KEY), ("evolving", S4G_EVO_KEY)):
        sub = _sub(generated, key)
        cols = _queue_cols(sub)
        for d in _dicts(_cfg(generated, key), cols):
            m = S4G_TAG_RE.match(d["tag"])
            assert m, d["tag"]
            assert d["eps_ai"] == f"{float(m.group('ea').replace('p', '.')):g}"
            assert d["eps"] == f"{float(m.group('es').replace('p', '.')):g}"
            assert d["seed"] == m.group("seed")
            assert d["gamma"] == "0.0", ("homophily gamma must stay 0", d)
            assert d["wplat"] == "0.5", d
            assert d["gatemode"] == "threshold", d
            assert d["nrounds"] == str(S4G_ROUNDS), d
            assert d["pop"] == "ab" and d["mode"] == "loop", d
            for col, val in arm_cols[m.group("arm")].items():
                assert d[col] == val, (cond, col, d)
            if cond == "fixed":
                assert d["cmode"] == "bottom", d
            else:
                assert "cmode" not in cols, (
                    "the evolving schema must not carry a clamp column")


def test_s4g_env_reads_only_columns_the_queue_line_declares(generated):
    """A $(macro) the queue line never declares expands to nothing and
    the run silently takes a DEFAULT -- invisible in every artifact."""
    for key in (S4G_FIXED_KEY, S4G_EVO_KEY, S4G_SMOKE_FIXED_KEY,
                S4G_SMOKE_EVO_KEY, F4R_KEY):
        sub = _sub(generated, key)
        cols = set(_queue_cols(sub))
        for line in sub.splitlines():
            if line.startswith(("environment", "request_", "arguments")):
                for macro in re.findall(r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)",
                                        line):
                    if macro.startswith("MY."):
                        continue
                    assert macro in cols, (key, macro, sorted(cols))


def test_s4g_env_pins_the_shared_section4_settings(generated):
    for key in (S4G_FIXED_KEY, S4G_EVO_KEY):
        env = _env_line(_sub(generated, key))
        assert "INNATE_LAMBDA=0.2" in env, key      # the paper's gamma / k
        assert "WITH_TWIN=1" in env, key
        assert "KL_DIRECTION=forward" in env, key
        assert "TRAIN_CAP=723" in env and "N_LABELED=723" in env, key
        assert "LORA_R=512" in env and "SFT_LR=5e-5" in env, key
        assert "ML_TARGET=Action" in env and "DATASET=movielens" in env, key
        assert "mistralai/Mistral-7B-Instruct-v0.3" in env, key
        assert "REF_REPLAY" not in env, key
        assert "AB_SWEEPS" not in env, key


# ============================================================ F4R: the 18
def test_f4r_is_exactly_18_jobs_on_six_models_x_three_seeds(generated):
    rows = _cfg(generated, F4R_KEY)
    assert len(rows) == F4R_N, len(rows)
    seen = set()
    for r in rows:
        m = F4R_TAG_RE.match(r.split(",")[0].strip())
        assert m, f"tag off contract: {r.split(',')[0]}"
        seen.add((m.group("slug"), int(m.group("seed"))))
    assert seen == {(s, sd) for s in F4R_MODELS for sd in F4R_SEEDS}, seen


def test_f4r_runs_one_hundred_rounds(generated):
    cols = _queue_cols(_sub(generated, F4R_KEY))
    for d in _dicts(_cfg(generated, F4R_KEY), cols):
        assert d["nrounds"] == str(F4R_ROUNDS), d
        assert f"_r{F4R_ROUNDS}_" in d["tag"], d


def test_f4r_matches_the_displayed_figure4_cell_field_by_field(generated):
    """The claim the wave rests on: apart from seed, round count and the
    provenance marker, an F4R row IS the Figure-4 row. Checked against
    the six pofdfam_ ARTIFACT rows this generator writes, not against a
    second call of the builder that produced the F4R rows."""
    f4r_cols = _queue_cols(_sub(generated, F4R_KEY))
    fam_cols = _queue_cols(_sub(generated, FAM_KEY))
    assert f4r_cols == fam_cols, (
        "F4R reuses the family-prior queue schema; a divergence here "
        "makes the field-by-field comparison meaningless")
    fam = {d["tag"]: d for d in
           _dicts(_cfg(generated, FAM_KEY), fam_cols)}
    f4r = {d["tag"]: d for d in
           _dicts(_cfg(generated, F4R_KEY), f4r_cols)}
    checked = 0
    for slug in F4R_MODELS:
        ref = fam.get(FIG4_TAG.format(slug=slug))
        assert ref is not None, (
            f"the displayed Figure-4 cell for {slug} is not in "
            f"configs_pofd_{FAM_KEY}.txt -- re-point FIG4_TAG before "
            f"trusting this test")
        for seed in F4R_SEEDS:
            new = next(d for t, d in f4r.items()
                       if F4R_TAG_RE.match(t).group("slug") == slug
                       and F4R_TAG_RE.match(t).group("seed") == str(seed))
            for col in f4r_cols:
                if col in F4R_ALLOWED_DIFF:
                    continue
                assert new[col] == ref[col], (slug, seed, col,
                                              new[col], ref[col])
            assert new["seed"] == str(seed)
            assert ref["nrounds"] == new["nrounds"] == "30"
            checked += 1
    assert checked == F4R_N, checked


def test_f4r_is_the_forward_kl_lambda1_arm_at_ea1_es0p05(generated):
    cols = _queue_cols(_sub(generated, F4R_KEY))
    for d in _dicts(_cfg(generated, F4R_KEY), cols):
        assert d["style"] == "sft_kl", d          # forward-KL SFT
        assert d["beta"] == "1", d                # lambda = 1
        assert d["eps_ai"] == "1", d
        assert d["eps"] == "0.05", d
        assert d["wplat"] == "0.5", d             # the paper's beta
        assert d["gamma"] == "0.0", d             # homophily stays 0
        assert d["gatemode"] == "threshold", d
        assert d["iclk"] == "0" and d["uselora"] == "1", d
        assert d["fresh"] == "1", d
    env = _env_line(_sub(generated, F4R_KEY))
    assert "KL_DIRECTION=forward" in env
    assert "INNATE_LAMBDA=0.2" in env             # the paper's gamma / k
    assert "SAVE_RAW_GEN=1" in env and "WITH_TWIN=1" in env
    assert "ICL_DAYS=0" in env
    assert "INNATE_CLAMP" not in env and "REF_REPLAY" not in env


def test_f4r_declares_the_six_exact_checkpoints(generated):
    want = {
        "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen3_8b": "Qwen/Qwen3-8B",
        "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
        "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
        "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "ministral8b": "mistralai/Ministral-8B-Instruct-2410",
    }
    cols = _queue_cols(_sub(generated, F4R_KEY))
    seen = {}
    for d in _dicts(_cfg(generated, F4R_KEY), cols):
        slug = F4R_TAG_RE.match(d["tag"]).group("slug")
        seen.setdefault(slug, set()).add(d["basemodel"])
        # Qwen3's hybrid-reasoning template must be pinned OFF, or
        # completion-only SFT masking lands after the wrong prefix
        assert d["chatthink"] == ("0" if slug == "qwen3_8b" else "default"), d
    assert set(seen) == set(want), set(seen) ^ set(want)
    for slug, ids in seen.items():
        assert ids == {want[slug]}, (slug, ids)


def test_f4r_env_uses_the_offline_cache(generated):
    env = _env_line(_sub(generated, F4R_KEY))
    assert "HF_HUB_OFFLINE=1" in env
    assert "HF_HOME=/lustre/fast/fast/gsmithline/hf_cache" in env


# ====================================================== collisions and reuse
# The Figure-6 grid (2026-08-25) deliberately names the SAME tags as
# this wave for the 24 cells both grids contain (ea=1, es in {0,1}): same
# grammar, same cell. The generator asserts those rows byte-identical and
# the submit script forbids co-submitting the two keys, so whichever runs
# first makes the other's copy an idempotent no-op.
FIG6_KEYS = {"configs_pofd_section4_gate_anch2_fig6_fixed.txt",
             "configs_pofd_section4_gate_anch2_fig6_evo.txt"}


def test_no_new_tag_collides_with_any_other_generated_tag(generated):
    tags = _all_tags(generated)
    new_keys = {f"configs_pofd_{k}.txt" for k in
                (S4G_FIXED_KEY, S4G_EVO_KEY, S4G_SMOKE_FIXED_KEY,
                 S4G_SMOKE_EVO_KEY, F4R_KEY)}
    shared = 0
    for tag, where in tags.items():
        mine = [w for w in where if w in new_keys]
        if not mine:
            continue
        others = [w for w in where if w not in new_keys]
        if others and all(w in FIG6_KEYS for w in others):
            # the documented fig6 overlap: allowed ONLY for ea=1 cells at
            # es in {0, 1}, and the rows must be byte-identical
            assert "_ea1_" in tag and ("_es0_" in tag or "_es1_" in tag), tag
            rows = {generated[w] for w in where}
            mine_row = [r for r in generated[mine[0]].splitlines()
                        if r.startswith(tag + ",")]
            for w in others:
                assert mine_row[0] in generated[w].splitlines(), tag
            shared += 1
            continue
        assert len(where) == 1, (
            f"{tag} is queued by {where}: a double-queue into one run dir "
            f"is a WRITE RACE, not an idempotent no-op")
    assert shared == 24, shared


def test_new_tags_never_land_in_an_archived_family(generated):
    for key in (S4G_FIXED_KEY, S4G_EVO_KEY, S4G_SMOKE_FIXED_KEY,
                S4G_SMOKE_EVO_KEY, F4R_KEY):
        for r in _cfg(generated, key):
            tag = r.split(",")[0].strip()
            assert not tag.startswith(OLD_S4_PREFIXES), (key, tag)
            assert not tag.startswith("pofdfam_"), (key, tag)
            assert tag.startswith(("pofds4g_", "pofds4gsmk_",
                                   "pofdf4r_")), (key, tag)


def test_old_section4_and_figure4_configs_are_byte_unchanged(generated):
    """Old artifacts stay untouched: the two new waves may not perturb a
    single byte of the keys their results will be compared against."""
    for key in ("mistral_bottom20_source_impact",
                "mistral_bottom20_evolving",
                "mistral_bottom20_section4_repl_fixed",
                "mistral_bottom20_section4_repl_evo",
                FAM_KEY):
        name = f"configs_pofd_{key}.txt"
        on_disk = os.path.join(CONDOR, name)
        if not os.path.exists(on_disk):
            pytest.skip(f"{name} not on disk")
        with open(on_disk) as fh:
            assert generated[name] == fh.read(), name


# =========================================================== idempotent exec
def test_every_new_sub_uses_the_idempotent_executable(generated):
    """Resubmission must be a no-op on completed runs, not a restart."""
    for key in (S4G_FIXED_KEY, S4G_EVO_KEY, S4G_SMOKE_FIXED_KEY,
                S4G_SMOKE_EVO_KEY, F4R_KEY):
        sub = _sub(generated, key)
        ex = [l for l in sub.splitlines() if l.startswith("executable")]
        assert len(ex) == 1, key
        assert ex[0].rstrip().endswith(
            "run_one_pokec_gated_idempotent.sh"), (key, ex[0])
        assert "on_exit_hold      = (ExitCode =!= 0)" in sub, key
        assert "periodic_release" in sub and "periodic_remove" in sub, key


def test_submit_script_registers_both_umbrella_keys():
    with open(os.path.join(CONDOR, "submit_pofd_sweep.sh")) as fh:
        s = fh.read()
    assert ('section4_gate_anch2)       TARGETS="section4_gate_anch2_fixed '
            'section4_gate_anch2_evo"') in s
    assert ('section4_gate_anch2_smoke) TARGETS='
            '"section4_gate_anch2_smoke_fixed '
            'section4_gate_anch2_smoke_evo"') in s
    assert 'fig4_family_prior_repl30) TARGETS="$WHAT"' in s
    # the usage strings must advertise them, or the `*)` arm rejects the key
    assert s.count("section4_gate_anch2[_smoke][_fixed|_evo]") == 3, \
        s.count("section4_gate_anch2[_smoke][_fixed|_evo]")
    assert s.count("|fig4_family_prior_repl30") == 3
