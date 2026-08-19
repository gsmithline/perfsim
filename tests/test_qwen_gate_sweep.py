"""Tests for the QWEN GATE SWEEP (2026-08-19, qwen_gate_sweep).

Grid: 2 Qwen checkpoints x eps_AI {.05,.1,.2,.4,1} x eps_social
{0,.05,.1,.2,.4,1} x seed 0 = 60 conceptual cells of regularized SFT
at lambda=1 (kl_beta=1 forward). The audited manifest reports 30
reused / 30 new; only the 30 missing cells queue, in the new
pofdqgs_ family.

Checker: the new is_qgs branch pins the exact checkpoint per slug and
Qwen3's thinking template OFF, on top of the generic environment
tokens and the _b token gate. Exercised against REAL trajectories --
a healthy qwen3 cell passes; a qwen3 cell whose chat_thinking key is
absent (i.e. the thinking template ran), a qwen7b cell carrying a
thinking directive, and a slug/checkpoint mismatch all fail.

Analyzer: hard-requires 60, reads nothing from the twin, reports
equilibrium mean / SD / W1-from-initial over rounds 25-29, and draws
both models on shared scales.

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac).
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
CONDOR = os.path.join(REPO, "experiments", "condor")
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")
CHECKER = os.path.join(PIPE, "check_pofd_sanity.py")
RUNS = os.path.join(REPO, "notes", "pofd", "cluster")
MANIFEST = os.path.join(CONDOR, "manifest_qwen_gate_sweep.json")
MODELS = ["qwen7b", "qwen3_8b"]
GATES = [0.05, 0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
QWEN3_SRC = "pofdfam_qwen3_8b_b1_ea1_w0p5_l0p2_es0p2_s0"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load("gen_qgs", os.path.join(CONDOR, "gen_pofd_sweep.py"))
AS = _load("analyze_qgs",
           os.path.join(PIPE, "analyze_qwen_gate_sweep.py"))


def manifest():
    with open(MANIFEST) as fh:
        return json.load(fh)


def build_qgs(parent, tag, src=QWEN3_SRC, mutate=None):
    """A real trajectory re-tagged into the pofdqgs_ family. The
    checker validates the config EMBEDDED in trajectory.pt, so the
    mutation has to land there (and config.json is kept in sync)."""
    d = os.path.join(str(parent), tag)
    os.makedirs(d, exist_ok=True)
    t = torch.load(os.path.join(RUNS, src, "trajectory.pt"),
                   map_location="cpu", weights_only=False)
    cfg = t["config"]
    cfg["run_tag"] = tag
    if mutate:
        mutate(cfg)
    torch.save(t, os.path.join(d, "trajectory.pt"))
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    return d


def run_check(run_dir):
    p = subprocess.run([sys.executable, CHECKER, str(run_dir)],
                       capture_output=True, text=True,
                       env={**os.environ, "USE_TF": "0"})
    return p.returncode, p.stdout + p.stderr


def assert_verdict(run_dir, want_pass, want_str=None):
    rc, out = run_check(run_dir)
    if want_pass:
        assert rc == 0, f"expected PASS, got {rc}:\n{out[-2000:]}"
    else:
        assert rc != 0, f"expected FAIL, checker passed:\n{out[-2000:]}"
        if want_str is not None:
            assert want_str in out, \
                f"expected {want_str!r} in:\n{out[-2000:]}"


# -- manifest ------------------------------------------------------------

def test_manifest_covers_the_60_cell_grid():
    mf = manifest()
    assert mf["key"] == "qwen_gate_sweep"
    assert mf["n_cells"] == 60 and len(mf["cells"]) == 60
    assert {(c["model"], c["gate"], c["es"]) for c in mf["cells"]} == \
        {(m, g, e) for m in MODELS for g in GATES for e in ESS}
    assert all(c["seed"] == 0 for c in mf["cells"])


def test_manifest_split_and_reuse_sources():
    mf = manifest()
    reused = [c for c in mf["cells"] if c["status"] == "reused"]
    new = [c for c in mf["cells"] if c["status"] == "new"]
    assert mf["n_reused"] == len(reused) == 30
    assert mf["n_new"] == len(new) == 30
    assert all(c.get("verdict") == "PASS" for c in reused)
    assert len({c["run_tag"] for c in reused}) == 30
    # reuse comes from completed waves on the identical surface
    fams = {c["run_tag"].split("_")[0] for c in reused}
    assert fams <= {"pofdfam", "pofdesf", "pofdw2f", "pofdws2f"}
    # per-model split
    for model, want_new in (("qwen7b", 12), ("qwen3_8b", 18)):
        assert sum(1 for c in new if c["model"] == model) == want_new


# -- generator -----------------------------------------------------------

def test_generator_queues_only_the_missing_cells():
    rows = GEN.qgs_rows()
    assert len(rows) == 30
    tags = {r.split(",")[0] for r in rows}
    assert len(tags) == 30
    assert all(t.startswith("pofdqgs_") and "_b1_" in t
               and t.endswith("_s0") for t in tags)
    mf = manifest()
    assert tags == {c["new_tag"] for c in mf["cells"]
                    if c["status"] == "new"}
    # no reused (archived) tag is ever re-queued
    assert not (tags & {c["run_tag"] for c in mf["cells"]
                        if c["status"] == "reused"})


def test_generator_tags_stay_inside_the_declared_grid():
    tags = {r.split(",")[0] for r in GEN.qgs_rows()}
    grid = {GEN.qgs_tag(m, g, e) for m in GEN.QGS_MODELS
            for g in GEN.QGS_GATES for e in GEN.QGS_ESS}
    assert tags <= grid
    assert GEN.QGS_GATES == GATES and GEN.QGS_ESS == ESS


def test_generator_row_surface_and_thinking_flag():
    for r in GEN.qgs_rows():
        cols = [c.strip() for c in r.split(",")]
        assert cols[1] == "sft_kl" and cols[2] == "1", r   # lambda=1
        assert cols[3] == "0", r                            # seed 0
        assert cols[15] == "threshold", r                   # numeric gate
        assert cols[22] == "30", r                          # 30 rounds
        if "_qwen3_8b_" in cols[0]:
            assert cols[23] == "Qwen/Qwen3-8B", r
            assert cols[24] == "0", r          # CHAT_THINKING=0
        else:
            assert cols[23] == "Qwen/Qwen2.5-7B-Instruct", r
            assert cols[24] == "default", r


def test_generator_no_collision_with_other_keys():
    tags = {r.split(",")[0] for r in GEN.qgs_rows()}
    for other in (GEN.fam_rows(), GEN.famg_rows(), GEN.evo_rows(),
                  GEN.b20_rows()):
        assert not (tags & {r.split(",")[0] for r in other})


def test_qgs_sub_surface():
    sub = GEN.qgs_sub()
    assert "CHAT_THINKING=$(chatthink)" in sub
    assert "BASE_MODEL=$(basemodel)" in sub
    assert "KL_DIRECTION=forward" in sub
    assert "WITH_TWIN=1" in sub
    assert "INNATE_LAMBDA=0.2" in sub
    assert "LORA_R=512" in sub and "FRESH_EACH_ROUND=$(fresh)" in sub
    # no raw-generation dump: it is not part of the audited surface
    assert "SAVE_RAW_GEN" not in sub
    assert sub.rstrip().endswith(
        "from experiments/condor/configs_pofd_qwen_gate_sweep.txt")


def test_submit_key_registered():
    with open(os.path.join(CONDOR, "submit_pofd_sweep.sh")) as fh:
        sh = fh.read()
    assert 'qwen_gate_sweep) TARGETS="$WHAT" ;;' in sh
    # present in all THREE usage strings (BID prompt, WHAT prompt and
    # the *) fallback echo), but don't pin the neighbouring key --
    # later waves insert next to it
    assert sh.count("|qwen_gate_sweep|") == 3


# -- checker (real trajectories, pofdqgs_ branch) ------------------------

def test_healthy_qgs_qwen3_passes(tmp_path):
    rd = build_qgs(tmp_path,
                   "pofdqgs_qwen3_8b_b1_ea1_w0p5_l0p2_es0p2_s0")
    assert_verdict(rd, True)


def test_qwen3_without_thinking_directive_fails(tmp_path):
    # chat_thinking absent == the hybrid-reasoning template actually
    # ran; that cell is not comparable and must never gate clean
    rd = build_qgs(tmp_path,
                   "pofdqgs_qwen3_8b_b1_ea1_w0p5_l0p2_es0p2_s0",
                   mutate=lambda c: c.pop("chat_thinking", None))
    assert_verdict(rd, False, "CHAT_THINKING=0 is mandatory")


def test_qwen7b_carrying_a_thinking_directive_fails(tmp_path):
    rd = build_qgs(
        tmp_path, "pofdqgs_qwen7b_b1_ea1_w0p5_l0p2_es0p2_s0",
        mutate=lambda c: c.update(
            base_model="Qwen/Qwen2.5-7B-Instruct"))
    assert_verdict(rd, False, "must use the default chat template")


def test_slug_checkpoint_mismatch_fails(tmp_path):
    rd = build_qgs(
        tmp_path, "pofdqgs_qwen3_8b_b1_ea1_w0p5_l0p2_es0p2_s0",
        mutate=lambda c: c.update(
            base_model="Qwen/Qwen2.5-7B-Instruct"))
    assert_verdict(rd, False, "base_model=")


def test_nonzero_seed_is_rejected(tmp_path):
    rd = build_qgs(
        tmp_path, "pofdqgs_qwen3_8b_b1_ea1_w0p5_l0p2_es0p2_s42",
        mutate=lambda c: c.update(seed=42))
    assert_verdict(rd, False, "seed-0 only")


# -- analyzer ------------------------------------------------------------

def test_analyzer_grid_and_window():
    assert AS.MODELS == MODELS
    assert AS.GATES == GATES and AS.ESS == ESS
    assert AS.LATE == list(range(25, 30))
    assert AS.DENSITY_GATES == [0.05, 0.4, 1.0]
    assert AS.DENSITY_ESS == [0.0, 0.2, 1.0]


def test_w1_is_the_exact_equal_n_wasserstein():
    a = np.array([0.1, 0.9, 0.5])
    b = np.array([0.2, 0.4, 0.8])
    # sorted: [.1,.5,.9] vs [.2,.4,.8] -> mean(|.1|+|.1|+|.1|) = .1
    assert abs(AS.w1(a, b) - 0.1) < 1e-12
    assert AS.w1(a, a) == 0.0


def test_density_integrates_to_one():
    rng = np.random.default_rng(0)
    dens = AS.density(rng.uniform(0, 1, 5000))
    width = AS.BINS[1] - AS.BINS[0]
    assert abs(float(dens.sum() * width) - 1.0) < 1e-9


def test_cell_stats_match_hand_computation():
    op = torch.rand(30, 723)
    innate = torch.rand(723)
    stats = AS.cell_stats({"op_raw": op, "innate": innate})
    late = op.numpy()[25:30]
    assert abs(stats["eq_mean"] - float(np.mean(late.mean(axis=1)))) < 1e-9
    assert abs(stats["eq_sd"]
               - float(np.mean([r.std(ddof=0) for r in late]))) < 1e-9
    assert abs(stats["w1_init"]
               - float(np.mean([AS.w1(r, innate.numpy())
                                for r in late]))) < 1e-9


def test_analyzer_reads_nothing_from_the_twin():
    with open(os.path.join(PIPE,
                           "analyze_qwen_gate_sweep.py")) as fh:
        src = fh.read()
    # reused es=0 cells predate WITH_TWIN, so any twin dependence
    # would silently drop half the grid
    assert "twin_raw" not in src


def test_analyzer_surface():
    with open(os.path.join(PIPE,
                           "analyze_qwen_gate_sweep.py")) as fh:
        src = fh.read()
    assert "qwen_gate_sweep_analysis" in src
    assert "HARD FAIL" in src and "sys.exit(1)" in src
    for out in ("qgs_per_cell.csv", "qgs_density_points.csv",
                "qgs_heatmaps_", "qgs_density_"):
        assert out in src, out
    # PNG and PDF for every figure
    assert '("png", "pdf")' in src
    # both models share the colour scale and the density y-limit
    assert "SHARED colour scale" in src
    assert "dmax = max(" in src
