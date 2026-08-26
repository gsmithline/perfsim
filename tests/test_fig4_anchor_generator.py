"""ADVERSARIAL tests for the Figure-4 anchor trade-off generator block
(fig4_anchor_tradeoff[_smoke|_ext|_zsprior], 2026-08-25).

Contract:
  cells    2 models x es {.05,.2} x beta {0,.25,.5,.75,1} x gamma
           {1,.5,.2,0} = 80 nominal
  kinds    60 gpu + 20 dup: beta=1 -> gamma drops out (12 dups, source
           (model, es, 1, 1)); beta=0 -> population == twin, trained for
           qwen3_8b only (8 qwen7b dups, source (qwen3_8b, es, 0, gamma))
  rows     60 (ROW_PS, 29 cols): sft_kl, kl_beta 2, forward, seed 0, one
           sweep, 30 rounds, ICL_K 0, fresh r512 LoRA, homophily gamma 0
  smoke    2 rows (pofdf4asmk_, 3 rounds) + 2 zsprior rows (both models,
           NEW _a100 tags) on their own sub
  hardware ONE class for the whole wave (2026-08-26 H100 outage):
           F4A_GPU_NAME = "NVIDIA A100-SXM4-80GB" pinned in every F4A sub
           (main / smoke / ext / zsprior); no "H100" string in any of them
  env      all_open AI gate (strict-< form documented as "<"), threshold
           peer gate, AB_SWEEPS=$(sweeps), DEFFUANT_ALPHA=0.5,
           PARSE_MODE=strict SAVE_RAW_GEN=1 WITH_TWIN=1 TRAIN_WITNESS=1
  ext      manifest-driven, trained cells only, rounds {60, 100}
  files    the three fig4_anchor_tradeoff files only; every other
           generated file byte-identical; pofdf4a_ is a NEW tag family
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
SUBMIT = os.path.join(CONDOR, "submit_pofd_sweep.sh")

KEY = "fig4_anchor_tradeoff"
CFG = f"configs_pofd_{KEY}.txt"
CFG_SMOKE = f"configs_pofd_{KEY}_smoke.txt"
CFG_ZS = f"configs_pofd_{KEY}_zsprior.txt"
CFG_EXT = f"configs_pofd_{KEY}_ext.txt"
SUB = f"at_pofd_{KEY}.sub"
SUB_SMOKE = f"at_pofd_{KEY}_smoke.sub"
SUB_ZS = f"at_pofd_{KEY}_zsprior.sub"
SUB_EXT = f"at_pofd_{KEY}_ext.sub"

TAG_RE = re.compile(
    r"^pofdf4a_(?P<model>qwen7b|qwen3_8b)_fwdlam2_sw100_eaopen"
    r"_w(?P<w>0|0p25|0p5|0p75|1)_k(?P<k>1|0p5|0p2|0)_es(?P<es>0p05|0p2)"
    r"_anch2_s0_r(?P<r>30|60|100)$")
SMOKE_TAG_RE = re.compile(
    r"^pofdf4asmk_(?P<model>qwen7b|qwen3_8b)_fwdlam2_sw100_eaopen"
    r"_w0p5_k0p5_es0p2_anch2_s0_r3$")

# files whose bytes the new block must not move
FROZEN_FILES = (
    "configs_pofd_section3_model_equilibria.txt",
    "configs_pofd_section3_model_equilibria_smoke.txt",
    "configs_pofd_section4_gate_anch2_fig6_fixed.txt",
    "configs_pofd_section4_gate_anch2_fig6_evo.txt",
    "configs_pofd_fig3_full_loop.txt",
    "configs_pofd_sft_update_dose_loop.txt",
    "configs_pofd_zsprior_screen.txt",
    "at_pofd_section3_model_equilibria.sub",
    "at_pofd_zsprior_screen.sub",
    "at_pofd_fig3_full_loop.sub",
)


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
    return _load(GEN, "_gen_f4a_test")


def _generate(tmp, manifest=None):
    """Run gen_pofd_sweep.main() with HERE redirected into `tmp`; returns
    {basename: text} for every file it would write."""
    for j in glob.glob(os.path.join(CONDOR, "*.json")):
        shutil.copy(j, tmp)
    mod = _load(GEN, f"_gen_pofd_f4a_{os.path.basename(tmp)}")
    mod.HERE = tmp
    mod.F3X_REQUEST_PATH = os.path.join(tmp, "fig3_extension_request.json")
    mod.S4G2_EXT_REQUEST_PATH = os.path.join(
        tmp, "section4_fig6_extension_request.json")
    mod.F4A_EXT_REQUEST_PATH = os.path.join(
        tmp, "fig4_anchor_extension_request.json")
    if manifest is not None:
        with open(mod.F4A_EXT_REQUEST_PATH, "w") as fh:
            json.dump(manifest, fh)
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


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    return _generate(str(tmp_path_factory.mktemp("gen_f4a")))


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
def test_cells_are_80_with_the_exact_dedup_algebra(gen):
    cells = gen.f4a_cells()
    assert len(cells) == 80 and len(set(cells)) == 80
    kinds = {}
    for c in cells:
        kinds[c[4]] = kinds.get(c[4], 0) + 1
    assert kinds == {"gpu": 60, "dup": 20}
    assert {c[:4] for c in cells} == {
        (m, e, b, g) for m in ("qwen7b", "qwen3_8b") for e in (0.05, 0.2)
        for b in (0.0, 0.25, 0.5, 0.75, 1.0) for g in (1.0, 0.5, 0.2, 0.0)}
    gpu = {c[:4] for c in cells if c[4] == "gpu"}
    dup1 = [c for c in cells if c[4] == "dup" and c[2] == 1.0]
    dup0 = [c for c in cells if c[4] == "dup" and c[2] == 0.0]
    assert len(dup1) == 12 and len(dup0) == 8
    for (m, e, b, g, k, s) in cells:
        assert s in gpu
        assert s == gen.f4a_source(m, e, b, g)
        if k == "gpu":
            assert s == (m, e, b, g)
        elif b == 1.0:
            assert g != 1.0 and s == (m, e, 1.0, 1.0)
        else:
            assert b == 0.0 and m == "qwen7b" and s == ("qwen3_8b", e, 0.0, g)
    for e in (0.05, 0.2):
        assert sum(1 for c in cells if c[4] == "gpu" and c[1] == e) == 30
    assert gen.F4A_BETA0_MODEL == "qwen3_8b"
    assert gen.F4A_REUSED == {}
    assert (gen.F4A_MODELS, gen.F4A_ES, gen.F4A_BETAS, gen.F4A_GAMMAS) == (
        ("qwen7b", "qwen3_8b"), (0.05, 0.2), (0.0, 0.25, 0.5, 0.75, 1.0),
        (1.0, 0.5, 0.2, 0.0))
    assert (gen.F4A_SEED, gen.F4A_ROUNDS, gen.F4A_SMOKE_ROUNDS, gen.F4A_SWEEPS,
            gen.F4A_LAMBDA, gen.F4A_ALPHA) == (0, 30, 3, 100, 2.0, 0.5)


def test_tag_grammar_matches_the_contract(gen):
    assert gen.f4a_tag("qwen3_8b", 0.05, 0.25, 0.2) == \
        "pofdf4a_qwen3_8b_fwdlam2_sw100_eaopen_w0p25_k0p2_es0p05_anch2_s0_r30"
    assert gen.f4a_tag("qwen7b", 0.2, 1.0, 1.0) == \
        "pofdf4a_qwen7b_fwdlam2_sw100_eaopen_w1_k1_es0p2_anch2_s0_r30"
    assert gen.f4a_tag("qwen7b", 0.2, 0.5, 0.5, rounds=3, smoke=True) == \
        "pofdf4asmk_qwen7b_fwdlam2_sw100_eaopen_w0p5_k0p5_es0p2_anch2_s0_r3"
    assert gen.f4a_tag("qwen3_8b", 0.2, 0.0, 0.0, rounds=60).endswith("_r60")


# ================================================================= rows
def test_60_rows_whose_columns_match_their_tags(generated, gen):
    rows = _rows(generated, CFG)
    assert len(rows) == 60
    cols = _cols(generated[SUB])
    assert len(cols) == 29
    seen = set()
    for d in _dicts(rows, cols):
        m = TAG_RE.match(d["tag"])
        assert m, d["tag"]
        model = m.group("model")
        assert float(d["wplat"]) == _untok(m.group("w"))
        assert float(d["lam"]) == _untok(m.group("k"))
        assert float(d["eps"]) == _untok(m.group("es"))
        assert m.group("r") == "30" and d["nrounds"] == "30"
        assert d["style"] == "sft_kl" and d["beta"] == "2"
        assert d["seed"] == "0" and d["kldir"] == "forward"
        assert d["sweeps"] == "100" and d["iclk"] == "0" and d["snap"] == "-1"
        assert d["uselora"] == "1" and d["fresh"] == "1"
        assert d["gamma"] == "0.0"                      # homophily gamma
        assert d["deploy_every"] == "1" and d["regime"] == "replace"
        assert d["pop"] == "ab" and d["mode"] == "loop"
        assert d["basemodel"] == gen.FAM_MODELS[model]["base_model"]
        assert d["chatthink"] == ("0" if model == "qwen3_8b" else "default")
        assert float(d["eps"]) in (0.05, 0.2)
        key = (model, float(d["eps"]), float(d["wplat"]), float(d["lam"]))
        assert gen.f4a_source(*key) == key, ("a dup must never queue", key)
        seen.add(key)
    assert seen == {c[:4] for c in gen.f4a_cells() if c[4] == "gpu"}
    assert generated[CFG] == "\n".join(gen.f4a_rows()) + "\n"


def test_on_disk_files_equal_the_generator(gen):
    with open(os.path.join(CONDOR, CFG)) as fh:
        assert fh.read() == "\n".join(gen.f4a_rows()) + "\n"
    with open(os.path.join(CONDOR, CFG_SMOKE)) as fh:
        assert fh.read() == "\n".join(gen.f4a_smoke_rows()) + "\n"
    with open(os.path.join(CONDOR, CFG_ZS)) as fh:
        assert fh.read() == "\n".join(gen.f4a_zsprior_rows()) + "\n"
    with open(os.path.join(CONDOR, SUB)) as fh:
        assert fh.read() == gen.f4a_sub("main")
    with open(os.path.join(CONDOR, SUB_SMOKE)) as fh:
        assert fh.read() == gen.f4a_sub("smoke")
    with open(os.path.join(CONDOR, SUB_ZS)) as fh:
        assert fh.read() == gen.f4a_zsprior_sub()


# ============================================================ collisions
def test_pofdf4a_is_a_new_family_and_nothing_else_moved(generated):
    mine = {r.split(",")[0].strip() for r in _rows(generated, CFG)}
    everywhere = {}
    for name, text in generated.items():
        if name.startswith("configs_pofd_"):
            for r in text.splitlines():
                if r.strip():
                    everywhere.setdefault(r.split(",")[0].strip(), set()).add(name)
    for t in mine:
        assert everywhere[t] == {CFG}, (t, everywhere[t])
    for t, where in everywhere.items():
        if t.startswith("pofdf4a"):
            assert where <= {CFG, CFG_SMOKE, CFG_EXT, CFG.replace(f"{KEY}.", f"{KEY}_sw1."), CFG_SMOKE.replace(f"{KEY}_", f"{KEY}_sw1_"), CFG_EXT.replace(f"{KEY}_", f"{KEY}_sw1_")}, (t, where)
    assert not any(t.startswith("pofdf4asmk") for t in mine)
    # the archived files are byte-identical
    for name in FROZEN_FILES:
        with open(os.path.join(CONDOR, name)) as fh:
            assert generated[name] == fh.read(), name
    # the zsprior_screen file did NOT gain the qwen7b row
    zs = _rows(generated, "configs_pofd_zsprior_screen.txt")
    assert len(zs) == 4
    assert not any("_qwen7b_" in r for r in zs)
    # the _a100 prior tags are NEW to every registered config, and the
    # old H100-pinned qwen7b tag (it never produced a result) is nowhere
    for t in ("pofdzsprior_qwen7b_w0p5_l0p2_es0_a100_s0",
              "pofdzsprior_qwen3_8b_w0p5_l0p2_es0_a100_s0"):
        assert everywhere[t] == {CFG_ZS}, (t, everywhere[t])
    assert "pofdzsprior_qwen7b_w0p5_l0p2_es0_s0" not in everywhere
    assert everywhere["pofdzsprior_qwen3_8b_w0p5_l0p2_es0_s0"] == \
        {"configs_pofd_zsprior_screen.txt"}
    assert CFG_EXT not in generated and SUB_EXT not in generated


# ================================================================== sub
def test_sub_pins_the_surface_and_documents_a_strict_gate(generated):
    sub = generated[SUB]
    env = [l for l in sub.splitlines() if l.startswith("environment")][0]
    for tok in ("DATASET=movielens", "ML_TARGET=Action", "AI_GATE_MODE=all_open",
                "PEER_GATE_MODE=threshold", "AI_GATE_REFERENCE=anchor",
                "EPS_AI=1", "INNATE_LAMBDA=$(lam)", "AB_SWEEPS=$(sweeps)",
                "DEFFUANT_ALPHA=0.5", "KL_DIRECTION=$(kldir)", "WITH_TWIN=1",
                "SAVE_RAW_GEN=1", "PARSE_MODE=strict", "TRAIN_WITNESS=1",
                "CHAT_THINKING=$(chatthink)", "BASE_MODEL=$(basemodel)",
                "LORA_R=512", "SFT_LR=5e-5", "SFT_EPOCHS=1", "SFT_BATCH_SIZE=4",
                "TRAIN_CAP=723", "N_LABELED=723", "N_ROUNDS=$(nrounds)",
                "USE_LORA=$(uselora)", "FRESH_EACH_ROUND=$(fresh)",
                "ICL_K=$(iclk)", "HF_HUB_OFFLINE=1",
                "WANDB_RUN_SUFFIX=_fig4_anchor_tradeoff"):
        assert f" {tok} " in env.replace('"', " "), tok
    for bad in ("SFT_GRAD_ACCUM", "REF_REPLAY", "INNATE_CLAMP",
                "PEER_GATE_MODE=all_open", "AI_GATE_MODE=threshold"):
        assert bad not in env, bad
    assert "<=" not in sub and "< eps_AI" in sub
    assert "60 jobs" in sub
    assert "check_fig4_anchor.py" in sub
    req = next(l for l in sub.splitlines() if l.startswith("requirements"))
    assert '(TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB")' in req
    assert "(TARGET.CUDAGlobalMemoryMb >= 80000)" in req
    assert "H100" not in sub
    assert 'g106.internal' in sub and 'i104.internal' in sub
    cols = set(_cols(sub))
    for macro in re.findall(r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)", sub):
        assert macro in cols, macro
    assert "request_memory    = $(mem)" in sub
    # the smoke and ext subs share the env byte-for-byte
    env_smoke = [l for l in generated[SUB_SMOKE].splitlines()
                 if l.startswith("environment")][0]
    assert env_smoke == env
    assert "check_fig4_anchor.py --smoke" in generated[SUB_SMOKE]
    assert "<=" not in generated[SUB_SMOKE]
    req_smoke = next(l for l in generated[SUB_SMOKE].splitlines()
                     if l.startswith("requirements"))
    assert req_smoke == req and "H100" not in generated[SUB_SMOKE]


# ================================================================ smoke
def test_smoke_is_two_3_round_cells_of_a_trained_cell(generated, gen):
    rows = _rows(generated, CFG_SMOKE)
    assert len(rows) == 2
    cols = _cols(generated[SUB_SMOKE])
    models = set()
    prod = {r.split(",")[0].strip() for r in _rows(generated, CFG)}
    for d in _dicts(rows, cols):
        m = SMOKE_TAG_RE.match(d["tag"])
        assert m, d["tag"]
        models.add(m.group("model"))
        assert d["nrounds"] == "3" and d["wplat"] == "0.5" and d["lam"] == "0.5"
        assert d["eps"] == "0.2" and d["style"] == "sft_kl" and d["beta"] == "2"
        assert d["tag"] not in prod
        # identical to the production row of the same cell except tag/horizon
        p = [x.strip() for x in gen.f4a_row(m.group("model"), 0.2, 0.5, 0.5).split(",")]
        c = [x.strip() for x in ",".join(d[k] for k in cols).split(",")]
        assert c[1:23] == p[1:23] and c[24:] == p[24:]
    assert models == {"qwen7b", "qwen3_8b"}
    assert gen.F4A_SMOKE_CELLS == (("qwen7b", 0.2, 0.5, 0.5),
                                   ("qwen3_8b", 0.2, 0.5, 0.5))
    assert all(c[:4] in {x[:4] for x in gen.f4a_cells() if x[4] == "gpu"}
               for c in [(m, 0.2, 0.5, 0.5) for m in models])


# ============================================================== zsprior
def test_both_zero_shot_priors_mirror_the_archived_qwen3_row_on_their_own_sub(generated, gen):
    rows = _rows(generated, CFG_ZS)
    assert len(rows) == 2
    assert gen.F4A_ZSPRIOR == {
        "qwen7b": "pofdzsprior_qwen7b_w0p5_l0p2_es0_a100_s0",
        "qwen3_8b": "pofdzsprior_qwen3_8b_w0p5_l0p2_es0_a100_s0"}
    assert all(gen.f4a_zsprior_tag(m) == t for m, t in gen.F4A_ZSPRIOR.items())
    q3 = [x.strip() for x in
          next(r for r in _rows(generated, "configs_pofd_zsprior_screen.txt")
               if "_qwen3_8b_" in r).split(",")]
    assert q3[0] == "pofdzsprior_qwen3_8b_w0p5_l0p2_es0_s0"     # archived (H100)
    assert q3[23] == "Qwen/Qwen3-8B" and q3[24] == "0"
    want = {"qwen7b": ("Qwen/Qwen2.5-7B-Instruct", "default"),
            "qwen3_8b": ("Qwen/Qwen3-8B", "0")}
    seen = {}
    for r in rows:
        z = [x.strip() for x in r.split(",")]
        assert len(z) == 25
        model = next(m for m in gen.F4A_MODELS if z[0] == gen.F4A_ZSPRIOR[m])
        assert z[0].endswith("_es0_a100_s0") and z[0] != q3[0]
        assert z[1:23] == q3[1:23], "mirror the archived qwen3_8b row column for column"
        assert z[1] == "frozen" and z[9] == "0" and z[22] == "1" and z[18] == "0"
        assert (z[23], z[24]) == want[model]
        seen[model] = z
    assert set(seen) == {"qwen7b", "qwen3_8b"}
    # the qwen3_8b prior is a NEW artifact: the archived row with only the
    # tag changed
    assert seen["qwen3_8b"][1:] == q3[1:]
    cols = _cols(generated[SUB_ZS])
    assert len(cols) == 25 and cols == _cols(generated["at_pofd_zsprior_screen.sub"])
    body = lambda s, k: [l for l in s.replace(f"configs_pofd_{k}.txt", "X")
                         .splitlines() if not l.startswith("#")]
    # env + queue identical to the archived zsprior template; the ONLY
    # difference is the requirements line, which pins the wave's class
    def _lines(text, key):
        # body() already returns a list of non-comment lines
        return [l for l in body(text, key)
                if not l.startswith("requirements") and not l.startswith("#")]
    assert _lines(generated[SUB_ZS], f"{KEY}_zsprior") == \
        _lines(generated["at_pofd_zsprior_screen.sub"], "zsprior_screen")
    req = next(l for l in generated[SUB_ZS].splitlines()
               if l.startswith("requirements"))
    assert '(TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB")' in req
    assert "(TARGET.CUDAGlobalMemoryMb >= 80000)" in req
    assert "(TARGET.Machine =!= MY.LastRemoteHost)" in req
    assert "H100" not in generated[SUB_ZS]
    assert "2 job" in generated[SUB_ZS]
    assert "<=" not in generated[SUB_ZS]
    # sha pins: BOTH unpinned until the A100 serves run (the coordinator
    # pins them afterwards); the archived H100-served vectors are WARN
    # references only, never pins
    assert set(gen.F4A_ZSPRIOR_SHA) == {"qwen7b", "qwen3_8b"}
    assert all(v is None or re.fullmatch(r"[0-9a-f]{64}", v)
               for v in gen.F4A_ZSPRIOR_SHA.values())   # pinned 2026-08-26
    assert gen.F4A_ZSPRIOR_WARN_SHA == {
        "qwen7b": "1674ee5f8d833f46de672791d933e1d3bdeefb07484c2d110dec84ce71da30bb",
        "qwen3_8b": "fdfdeab7466345159cd7ae16ee487d4982d686cfdb93287780ae4d109ccba3f7"}
    assert gen.F4A_ZSPRIOR_WARN_SHA["qwen3_8b"] == gen.F3_FROZEN_SHA


# ============================================================= hardware
def test_hardware_class_is_exported_and_pinned_on_every_sub(generated, gen):
    assert gen.F4A_GPU_NAME == "NVIDIA A100-SXM4-80GB"
    assert not hasattr(gen, "F4A_H100")
    pin = '(TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB")'
    reqs = {}
    for name in (SUB, SUB_SMOKE, SUB_ZS):
        req = next(l for l in generated[name].splitlines()
                   if l.startswith("requirements"))
        assert pin in req and "(TARGET.CUDAGlobalMemoryMb >= 80000)" in req, name
        assert "H100" not in generated[name], name
        assert gen.F4A_GPU_NAME in generated[name], name
        reqs[name] = req
    assert reqs[SUB] == reqs[SUB_SMOKE]
    assert 'i101.internal' in reqs[SUB] and 'i101.internal' not in reqs[SUB_ZS]
    assert "MY.LastRemoteHost" in reqs[SUB_ZS]
    ext = gen.f4a_sub("ext")
    assert next(l for l in ext.splitlines()
                if l.startswith("requirements")) == reqs[SUB]
    assert "H100" not in ext
    with open(GEN) as fh:
        src = fh.read()
    assert "HARDWARE CLASS (2026-08-26)" in src
    assert "comparability with the archived H100-served waves is NOT claimed" in src


def test_reuse_audit_verdict_is_recorded(gen):
    assert gen.F4A_REUSED == {}
    with open(GEN) as fh:
        src = fh.read()
    assert "F4A_REUSED == {} (0/80 exact" in src
    assert "every archived anch2 run is esopen" in src


# =========================================================== extensions
def test_extension_requests_accept_only_trained_cells_at_60_or_100(gen, tmp_path):
    req = tmp_path / "req.json"
    assert gen.f4a_ext_requests(str(req)) == []          # absent -> none
    req.write_text("[]")
    assert gen.f4a_ext_requests(str(req)) == []          # empty -> none
    good = [{"model": "qwen7b", "es": 0.05, "beta": 0.25, "gamma": 0.2,
             "rounds": 60},
            {"model": "qwen3_8b", "es": 0.2, "beta": 0.0, "gamma": 1.0,
             "rounds": 100}]
    req.write_text(json.dumps(good))
    got = gen.f4a_ext_requests(str(req))
    assert sorted(got) == sorted([("qwen7b", 0.05, 0.25, 0.2, 60),
                                  ("qwen3_8b", 0.2, 0.0, 1.0, 100)])
    req.write_text(json.dumps({"cells": good}))          # wrapper tolerated
    assert sorted(gen.f4a_ext_requests(str(req))) == sorted(got)
    # a dup cell (qwen7b beta=0; any beta=1 gamma!=1) has no run to extend
    req.write_text(json.dumps([dict(good[0], beta=0.0)]))
    with pytest.raises(AssertionError):
        gen.f4a_ext_requests(str(req))
    req.write_text(json.dumps([dict(good[0], beta=1.0, gamma=0.5)]))
    with pytest.raises(AssertionError):
        gen.f4a_ext_requests(str(req))
    # bad horizon / off-grid / duplicate -> refused
    req.write_text(json.dumps([dict(good[0], rounds=45)]))
    with pytest.raises(AssertionError):
        gen.f4a_ext_requests(str(req))
    req.write_text(json.dumps([dict(good[0], es=0.1)]))
    with pytest.raises(AssertionError):
        gen.f4a_ext_requests(str(req))
    req.write_text(json.dumps([good[0], good[0]]))
    with pytest.raises(AssertionError):
        gen.f4a_ext_requests(str(req))


def test_extension_files_appear_only_with_a_manifest(tmp_path_factory, gen):
    tmp = str(tmp_path_factory.mktemp("gen_f4a_ext"))
    manifest = [{"model": "qwen7b", "es": 0.05, "beta": 0.25, "gamma": 0.2,
                 "rounds": 60},
                {"model": "qwen3_8b", "es": 0.2, "beta": 0.0, "gamma": 1.0,
                 "rounds": 100}]
    out = _generate(tmp, manifest=manifest)
    rows = [r for r in out[CFG_EXT].splitlines() if r.strip()]
    assert len(rows) == 2
    cols = _cols(out[SUB_EXT])
    tags = set()
    for d in _dicts(rows, cols):
        m = TAG_RE.match(d["tag"])
        assert m and m.group("r") == d["nrounds"] and d["nrounds"] in ("60", "100")
        tags.add(d["tag"])
    assert tags == {gen.f4a_tag("qwen7b", 0.05, 0.25, 0.2, rounds=60),
                    gen.f4a_tag("qwen3_8b", 0.2, 0.0, 1.0, rounds=100)}
    env_ext = [l for l in out[SUB_EXT].splitlines() if l.startswith("environment")][0]
    env = [l for l in out[SUB].splitlines() if l.startswith("environment")][0]
    assert env_ext == env
    assert "2 jobs" in out[SUB_EXT]
    req_ext = next(l for l in out[SUB_EXT].splitlines()
                   if l.startswith("requirements"))
    assert req_ext == next(l for l in out[SUB].splitlines()
                           if l.startswith("requirements"))
    assert '"NVIDIA A100-SXM4-80GB"' in req_ext and "H100" not in out[SUB_EXT]
    # the base files are unchanged by the manifest
    assert out[CFG] == "\n".join(gen.f4a_rows()) + "\n"


# =============================================================== frozen
def test_frozen_replay_names_are_60_with_shared_beta0(gen):
    names = {gen.f4a_frozen_name(*c[5]) for c in gen.f4a_cells()}
    assert len(names) == 60
    assert all(n.startswith("frozen_f4a_") and n.endswith("_sw100_r30.pt")
               for n in names)
    assert sum(1 for n in names if "_shared_" in n) == 8
    assert gen.f4a_frozen_name("qwen7b", 0.05, 0.0, 0.2) == \
        gen.f4a_frozen_name("qwen3_8b", 0.05, 0.0, 0.2) == \
        "frozen_f4a_shared_w0_k0p2_es0p05_sw100_r30.pt"
    assert gen.f4a_frozen_name("qwen3_8b", 0.2, 0.75, 1.0) == \
        "frozen_f4a_qwen3_8b_w0p75_k1_es0p2_sw100_r30.pt"
    assert gen.f4a_frozen_name("qwen7b", 0.2, 1.0, 1.0, rounds=60).endswith("_r60.pt")
    per_es = {}
    for n in names:
        per_es[n.split("_es")[1].split("_")[0]] = per_es.get(
            n.split("_es")[1].split("_")[0], 0) + 1
    assert per_es == {"0p05": 30, "0p2": 30}


# =============================================================== submit
def test_submit_script_routes_every_fig4_anchor_key():
    with open(SUBMIT) as fh:
        s = fh.read()
    assert ('fig4_anchor_tradeoff|fig4_anchor_tradeoff_ext|'
            'fig4_anchor_tradeoff_zsprior) TARGETS="$WHAT" ;;' in s)
    assert ('fig4_anchor_tradeoff_smoke) TARGETS="fig4_anchor_tradeoff_smoke '
            'fig4_anchor_tradeoff_zsprior" ;;' in s)
    assert s.count("fig4_anchor_tradeoff[_sw1][_smoke|_ext|_zsprior|_node[_a|_b]|_node_smoke]") == 3
    assert 'fig4_anchor_tradeoff_node)       TARGETS="fig4_anchor_tradeoff_node_a fig4_anchor_tradeoff_node_b" ;;' in s
    assert 'fig4_anchor_tradeoff_node_smoke) TARGETS="fig4_anchor_tradeoff_node_smoke_a" ;;' in s
    assert "< eps_AI" in s.split("fig4_anchor_tradeoff|")[0][-2000:]
