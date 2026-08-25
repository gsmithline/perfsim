"""ADVERSARIAL tests for the Figure-6 grid generator (section4_gate_anch2_fig6,
2026-08-25) -- the corrected-gate Section 4 on the matched 4 x 4 gate grid.

Contract:
  cells    2 arms x 2 conds x ea {0,.1,.3,1} x es {0,.1,.3,1} x 3 seeds = 192
  kinds    144 gpu (ea > 0) + 4 witness (ea = 0, both arms x both
           conds, es=.3, seed 0) + 44 twin-derived (ea = 0, 0 jobs)
  reused   0 -- the 2026-08-25 audit found NO corrected-gate cell on disk
  jobs     148 = 74 fixed + 74 evolving; smoke 4 (ea=.1, es=.3, seed 0, 3r)
  overlap  exactly the 24 (ea=1, es in {0,1}) tags are shared with the
           unrun section4_gate_anch2 key, byte-identical rows
  ea = 0   gp.ai_gate is strict-<, so the gate is closed everywhere: the
           population IS twin_raw; only the four witnesses are queued
  raw gen  SAVE_RAW_GEN=1 in every Section-4 sub: the parser falls back to
           a FINITE 0.5 on failure, so parse_fail_frac (raw_gen_log only)
           is the sole witness of zero parse failures
"""
from __future__ import annotations

import contextlib
import glob
import importlib.util
import io
import json
import os
import re
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDOR = os.path.join(ROOT, "experiments", "condor")
GEN = os.path.join(CONDOR, "gen_pofd_sweep.py")

KEY = "section4_gate_anch2_fig6"
CFG_FIXED = f"configs_pofd_{KEY}_fixed.txt"
CFG_EVO = f"configs_pofd_{KEY}_evo.txt"
SUB_FIXED = f"at_pofd_{KEY}_fixed.sub"
SUB_EVO = f"at_pofd_{KEY}_evo.sub"
CFG_SMK_FIXED = f"configs_pofd_{KEY}_smoke_fixed.txt"
CFG_SMK_EVO = f"configs_pofd_{KEY}_smoke_evo.txt"
OLD_FIXED = "configs_pofd_section4_gate_anch2_fixed.txt"
OLD_EVO = "configs_pofd_section4_gate_anch2_evo.txt"

GATES = [0.0, 0.1, 0.3, 1.0]
TAG_RE = re.compile(
    r"^pofds4g_mistral7b_(?P<arm>b0|d8)_(?P<cond>fixb20|evoall)_anch2"
    r"_ea(?P<ea>0|0p1|0p3|1)_w0p5_l0p2_es(?P<es>0|0p1|0p3|1)"
    r"_s(?P<seed>0|42|43)$")


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
    return _load(GEN, "_gen_fig6_test")


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("gen_fig6"))
    for j in glob.glob(os.path.join(CONDOR, "*.json")):
        shutil.copy(j, tmp)
    mod = _load(GEN, "_gen_pofd_fig6")
    mod.HERE = tmp
    mod.F3X_REQUEST_PATH = os.path.join(tmp, "fig3_extension_request.json")
    mod.S4G2_EXT_REQUEST_PATH = os.path.join(
        tmp, "section4_fig6_extension_request.json")
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


def _cols(sub):
    q = [l for l in sub.splitlines() if l.startswith("queue ")][0]
    return [c.strip() for c in q.split(" from ")[0][len("queue "):].split(",")]


def _dicts(rows, cols):
    out = []
    for r in rows:
        v = [x.strip() for x in r.split(",")]
        assert len(v) == len(cols), (len(v), len(cols), r)
        out.append(dict(zip(cols, v)))
    return out


# ================================================================= grid
def test_grid_is_192_cells_of_three_kinds(gen):
    cells = gen.s4g2_cells()
    assert len(cells) == 192 and len(set(cells)) == 192
    kinds = {}
    for *_c, k in cells:
        kinds[k] = kinds.get(k, 0) + 1
    assert kinds == {"gpu": 144, "witness": 4, "twin": 44}, kinds
    assert {(a, c, ea, es, s) for (a, c, ea, es, s, _k) in cells} == {
        (a, c, ea, es, s) for a in ("b0", "d8") for c in ("fixed", "evolving")
        for ea in GATES for es in GATES for s in (0, 42, 43)}


def test_ea_zero_is_twin_derived_except_the_four_witnesses(gen):
    cells = gen.s4g2_cells()
    ea0 = [c for c in cells if c[2] == 0.0]
    assert len(ea0) == 48
    assert all(k in ("twin", "witness") for *_c, k in ea0)
    wit = [(a, c, es, s) for (a, c, _ea, es, s, k) in ea0 if k == "witness"]
    assert wit == [("b0", "fixed", 0.3, 0), ("b0", "evolving", 0.3, 0),
                   ("d8", "fixed", 0.3, 0), ("d8", "evolving", 0.3, 0)]
    assert all(ea > 0.0 for (_a, _c, ea, _es, _s, k) in cells if k == "gpu")


def test_reuse_audit_is_zero(gen):
    assert gen.S4G2_N_REUSED == 0


# ================================================================= rows
def test_148_jobs_split_74_fixed_74_evolving(generated):
    fixed = _rows(generated, CFG_FIXED)
    evo = _rows(generated, CFG_EVO)
    assert len(fixed) == 74 and len(evo) == 74
    tags = [r.split(",")[0].strip() for r in fixed + evo]
    assert len(set(tags)) == 148


def test_every_row_matches_its_tag_and_the_grid(generated):
    for cfg, sub, cond in ((CFG_FIXED, SUB_FIXED, "fixb20"),
                           (CFG_EVO, SUB_EVO, "evoall")):
        cols = _cols(generated[sub])
        for d in _dicts(_rows(generated, cfg), cols):
            m = TAG_RE.match(d["tag"])
            assert m, d["tag"]
            assert m.group("cond") == cond
            assert float(d["eps_ai"]) == _untok(m.group("ea"))
            assert float(d["eps"]) == _untok(m.group("es"))
            assert d["seed"] == m.group("seed")
            assert d["gatemode"] == "threshold" and d["wplat"] == "0.5"
            assert d["gamma"] == "0.0" and d["nrounds"] == "30"
            arm = m.group("arm")
            assert d["style"] == ("sft" if arm == "b0" else "frozen")
            assert d["icldays"] == ("8" if arm == "d8" else "0")
            if cond == "fixb20":
                assert d["cmode"] == "bottom" and d["sftexcl"] == "0"
            else:
                assert "cmode" not in cols


def test_only_the_four_witnesses_run_at_ea_zero(generated):
    evo = [r for r in _rows(generated, CFG_EVO) if "_ea0_" in r.split(",")[0]]
    fixed = [r for r in _rows(generated, CFG_FIXED)
             if "_ea0_" in r.split(",")[0]]
    assert sorted(r.split(",")[0].strip() for r in fixed) == [
        "pofds4g_mistral7b_b0_fixb20_anch2_ea0_w0p5_l0p2_es0p3_s0",
        "pofds4g_mistral7b_d8_fixb20_anch2_ea0_w0p5_l0p2_es0p3_s0"]
    assert sorted(r.split(",")[0].strip() for r in evo) == [
        "pofds4g_mistral7b_b0_evoall_anch2_ea0_w0p5_l0p2_es0p3_s0",
        "pofds4g_mistral7b_d8_evoall_anch2_ea0_w0p5_l0p2_es0p3_s0"]


def test_smoke_is_four_jobs_at_a_genuinely_new_cell(generated):
    for cfg in (CFG_SMK_FIXED, CFG_SMK_EVO):
        rows = _rows(generated, cfg)
        assert len(rows) == 2
        for r in rows:
            c = [x.strip() for x in r.split(",")]
            assert c[0].startswith("pofds4gsmk_") and "_ea0p1_" in c[0] \
                and "_es0p3_" in c[0] and c[0].endswith("_s0")
            assert c[22] == "3"
    # neither gate value exists in ANY archived family
    for name, text in generated.items():
        if name.startswith("configs_pofd_") and "fig6" not in name:
            for r in text.splitlines():
                t = r.split(",")[0]
                assert not (t.startswith("pofds4g") and "_ea0p1_" in t), t


def test_exactly_the_24_expected_tags_are_shared_with_the_unrun_key(generated):
    mine = {r.split(",")[0].strip(): r for r in
            _rows(generated, CFG_FIXED) + _rows(generated, CFG_EVO)}
    old = {r.split(",")[0].strip(): r for r in
           _rows(generated, OLD_FIXED) + _rows(generated, OLD_EVO)}
    shared = set(mine) & set(old)
    assert len(shared) == 24
    for t in shared:
        assert "_ea1_" in t and ("_es0_" in t or "_es1_" in t), t
        assert mine[t] == old[t], ("shared tag rows must be byte-identical", t)
    # and nothing else in this key exists anywhere else
    everywhere = {}
    for name, text in generated.items():
        if name.startswith("configs_pofd_"):
            for r in text.splitlines():
                if r.strip():
                    everywhere.setdefault(r.split(",")[0].strip(), set()).add(name)
    # (the seed-staged keys _s0 / _s42_43 re-queue the SAME rows by design;
    # they are the only other files allowed to name them)
    staged = {CFG_FIXED.replace(f"{KEY}_", f"{KEY}_s0_"),
              CFG_FIXED.replace(f"{KEY}_", f"{KEY}_s42_43_"),
              CFG_EVO.replace(f"{KEY}_", f"{KEY}_s0_"),
              CFG_EVO.replace(f"{KEY}_", f"{KEY}_s42_43_")}
    for t in set(mine) - shared:
        where = everywhere[t] - staged
        assert where == {CFG_FIXED} or where == {CFG_EVO}, (t, everywhere[t])


def test_subs_pin_the_corrected_gate_and_the_clamp_split(generated):
    for sub, cond in ((SUB_FIXED, "fixed"), (SUB_EVO, "evolving")):
        env = [l for l in generated[sub].splitlines()
               if l.startswith("environment")][0]
        assert "AI_GATE_REFERENCE=anchor" in env
        assert "SAVE_RAW_GEN=1" in env, ("the parser defaults failures to a "
                                         "finite 0.5; only raw_gen_log "
                                         "carries parse_fail_frac", sub)
        assert "WITH_TWIN=1" in env and "INNATE_LAMBDA=0.2" in env
        assert ("INNATE_CLAMP_PEER_MODE=stubborn" in env) == (cond == "fixed")
        assert "ea {0, .1, .3, 1} x es {0, .1, .3, 1}" in generated[sub]
        cols = set(_cols(generated[sub]))
        for macro in re.findall(r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)", env):
            assert macro in cols, (sub, macro)


# ============================================================ extensions
def test_extension_requests_must_be_matched_pairs_on_grid(gen, tmp_path):
    orig = gen.S4G2_EXT_REQUEST_PATH
    try:
        req = tmp_path / "req.json"
        gen.S4G2_EXT_REQUEST_PATH = str(req)
        both = [{"arm": "b0", "cond": c, "eps_ai": 0.3, "eps_social": 0.1,
                 "seed": 42, "rounds": 60} for c in ("fixed", "evolving")]
        req.write_text(json.dumps({"cells": both}))
        got = gen.s4g2_ext_requests()
        assert sorted(got) == sorted([("b0", c, 0.3, 0.1, 42, 60)
                                      for c in ("fixed", "evolving")])
        rows = gen.s4g2_ext_rows("fixed") + gen.s4g2_ext_rows("evolving")
        assert len(rows) == 2
        assert all(r.split(",")[0].strip().endswith("_r60") for r in rows)
        assert all([x.strip() for x in r.split(",")][22] == "60" for r in rows)
        # unpaired -> refused
        req.write_text(json.dumps({"cells": both[:1]}))
        with pytest.raises(AssertionError):
            gen.s4g2_ext_requests()
        # a twin-derived ea=0 cell has no run to extend -> refused
        req.write_text(json.dumps({"cells": [
            dict(e, eps_ai=0.0) for e in both]}))
        with pytest.raises(AssertionError):
            gen.s4g2_ext_requests()
        # bad horizon -> refused
        req.write_text(json.dumps({"cells": [dict(e, rounds=45) for e in both]}))
        with pytest.raises(AssertionError):
            gen.s4g2_ext_requests()
    finally:
        gen.S4G2_EXT_REQUEST_PATH = orig


def test_submit_script_routes_all_fig6_keys():
    with open(os.path.join(CONDOR, "submit_pofd_sweep.sh")) as fh:
        s = fh.read()
    for k in ("section4_gate_anch2_fig6)", "section4_gate_anch2_fig6_smoke)",
              "section4_gate_anch2_fig6_ext)"):
        assert k in s, k
    assert s.count("section4_gate_anch2_fig6[_smoke|_ext|_s0|_s42_43][_fixed|_evo]") == 3
    assert "section4_gate_anch2_fig6_s0)" in s and "section4_gate_anch2_fig6_s42_43)" in s


def test_old_section4_configs_are_byte_unchanged(generated):
    """Generalizing the helpers must not move the original wave by a byte."""
    for name in (OLD_FIXED, OLD_EVO,
                 "configs_pofd_section4_gate_anch2_smoke_fixed.txt",
                 "configs_pofd_section4_gate_anch2_smoke_evo.txt"):
        with open(os.path.join(CONDOR, name)) as fh:
            assert generated[name] == fh.read(), name


def test_seed_staged_keys_partition_the_148_rows_byte_identically(generated, gen):
    """section4_gate_anch2_fig6_s0 (seed 0: 26 per condition = 24 GPU +
    2 witnesses) and _s42_43 (48 per condition) are an exact partition of
    the full key's rows, with identical tags and an identical sub env."""
    for cond, cfg in (("fixed", CFG_FIXED), ("evolving", CFG_EVO)):
        full = _rows(generated, cfg)
        s0 = _rows(generated, cfg.replace(f"{KEY}_", f"{KEY}_s0_"))
        rest = _rows(generated, cfg.replace(f"{KEY}_", f"{KEY}_s42_43_"))
        assert len(s0) == 26 and len(rest) == 48 and len(full) == 74
        assert sorted(s0 + rest) == sorted(full)
        assert all(r.split(",")[0].strip().endswith("_s0") for r in s0)
        assert all(r.split(",")[0].strip().rsplit("_s", 1)[1] in ("42", "43")
                   for r in rest)
        env = lambda k: next(l for l in gen.s4g2_sub(cond, k).splitlines()
                             if l.startswith("environment"))
        assert env("s0") == env("main") == env("rest")
    submit = open(os.path.join(CONDOR, "submit_pofd_sweep.sh")).read()
    assert 'section4_gate_anch2_fig6_s0)     TARGETS="section4_gate_anch2_fig6_s0_fixed section4_gate_anch2_fig6_s0_evo" ;;' in submit
    assert 'section4_gate_anch2_fig6_s42_43) TARGETS="section4_gate_anch2_fig6_s42_43_fixed section4_gate_anch2_fig6_s42_43_evo" ;;' in submit
