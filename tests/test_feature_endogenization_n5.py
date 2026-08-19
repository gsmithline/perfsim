"""Tests for the FIVE-SEED EXTENSION of the main feature-endogenization
figure (2026-08-19, feature_endogenization_n5).

Manifest: 12 target cells (6 established Qwen conditions x seeds 44/45),
audited 0 reused / 12 new, with the want surface self-verified against all
18 established cells first.

Generator: 12 rows across the four established queue schemas (natural 6 /
frozen 2 / gender-removed 2 / gender-permuted 2), byte-identical row
grammar and environment to the seeds-{0,42,43} subs apart from the seed
and the A100 pin; no established tag re-queues; no collisions anywhere.

Analyzer: panels (a)/(b) hard-require seeds {0,42,43,44,45}, plot the
five-seed mean with a 95% Student-t ribbon (df=4), export per-seed and
per-round summary CSVs, and display the coefficient as lambda. Panel (c)
and the three-seed beta-sweep figure are untouched.

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac).
"""
import importlib.util
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
CONDOR = os.path.join(REPO, "experiments", "condor")
LLM = os.path.join(REPO, "experiments", "llm")
MANIFEST = os.path.join(CONDOR, "manifest_feature_endogenization_n5.json")
CONDITIONS = ["nat_l0", "nat_l0p5", "nat_l1", "frozen", "removed",
              "permuted"]
NEW_SEEDS = [44, 45]
ARMS = {"nat": 6, "frozen": 2, "gd": 2, "gp": 2}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load("gen_fe5", os.path.join(CONDOR, "gen_pofd_sweep.py"))


def _llm(name):
    if LLM not in sys.path:
        sys.path.insert(0, LLM)
    return importlib.import_module(name)


def manifest():
    with open(MANIFEST) as fh:
        return json.load(fh)


def cfg_rows(key):
    with open(os.path.join(CONDOR, f"configs_pofd_{key}.txt")) as fh:
        return [ln for ln in fh.read().splitlines() if ln.strip()]


def norm(line):
    """Row with the seed normalised, so established and new rows are
    directly comparable."""
    line = re.sub(r"_s4[2345]", "_sNN", line)
    return re.sub(r", 4[2345], ", ", NN, ", line)


# -- manifest ------------------------------------------------------------

def test_manifest_surface_and_split():
    mf = manifest()
    assert mf["key"] == "feature_endogenization_n5"
    assert mf["n_cells"] == 12 and len(mf["cells"]) == 12
    assert {(c["cond"], c["seed"]) for c in mf["cells"]} == \
        {(cd, s) for cd in CONDITIONS for s in NEW_SEEDS}
    # audited split: nothing completed exists at seeds 44/45
    assert mf["n_reused"] == 0 and mf["n_new"] == 12
    assert all(c["status"] == "new" for c in mf["cells"])
    assert len({c["new_tag"] for c in mf["cells"]}) == 12


def test_manifest_baseline_self_verified():
    mf = manifest()
    base = mf["baseline"]
    # the want surface was checked against every established cell
    assert len(base) == 18
    assert {(b["cond"], b["seed"]) for b in base} == \
        {(cd, s) for cd in CONDITIONS for s in (0, 42, 43)}
    assert all(b["verdict"] == "PASS" for b in base)
    # the seed-0 lambda=0 anchor legitimately lives in the reverse-era
    # pofdws2_ family; everything else is pofdws2f_/pofdicls2_/pofdfe*
    s0_nat0 = [b for b in base
               if b["cond"] == "nat_l0" and b["seed"] == 0][0]
    assert s0_nat0["run_tag"].startswith("pofdws2_qwen7b_b0_")


# -- generator -----------------------------------------------------------

def test_generator_emits_exactly_12_rows():
    total, tags = 0, set()
    for arm, want_n in ARMS.items():
        rows = GEN.fe5_rows(arm)
        assert len(rows) == want_n, arm
        arm_tags = {r.split(",")[0] for r in rows}
        assert len(arm_tags) == want_n, arm
        tags |= arm_tags
        total += len(rows)
    assert total == 12 and len(tags) == 12


def test_generator_never_requeues_established_seeds():
    tags = {r.split(",")[0] for arm in ARMS
            for r in GEN.fe5_rows(arm)}
    assert all(t.endswith("_s44") or t.endswith("_s45")
               or t.endswith("_s44_fresh_data")
               or t.endswith("_s45_fresh_data") for t in tags)
    for seed in (0, 42, 43):
        assert not any(f"_s{seed}_" in t or t.endswith(f"_s{seed}")
                       for t in tags), seed
    established = {GEN.fe5_tag(c, s) for c in CONDITIONS
                   for s in (0, 42, 43)}
    assert not (tags & established)


def test_generator_one_family_per_condition():
    tags = {r.split(",")[0] for arm in ARMS
            for r in GEN.fe5_rows(arm)}
    for prefix in ("pofdws2f_qwen7b_b0_", "pofdws2f_qwen7b_b0p5_",
                   "pofdws2f_qwen7b_b1_", "pofdicls2_qwen7b_",
                   "pofdfegd_qwen7b_b1_", "pofdfegp_qwen7b_b1_"):
        assert sum(1 for t in tags if t.startswith(prefix)) == 2, prefix


def test_generated_rows_match_the_established_grammar():
    # every new row must be byte-identical to its established
    # counterpart once the seed is normalised
    for new_key, old_key in (
        ("feature_endogenization_n5_nat", "qwen7b_fes"),
        ("feature_endogenization_n5_frozen", "qwen7b_fef"),
        ("feature_endogenization_n5_gd", "qwen7b_fegd"),
        ("feature_endogenization_n5_gp", "qwen7b_fegp"),
    ):
        new = sorted(norm(r) for r in cfg_rows(new_key))
        old = sorted(norm(r) for r in cfg_rows(old_key)
                     if re.search(r"_s4[23]", r))
        assert new == old, (new_key, new, old)


def test_sub_templates_pin_the_established_gpu():
    for arm in ARMS:
        sub = GEN.fe5_sub(arm)
        assert 'CUDADeviceName == "NVIDIA A100-SXM4-80GB"' in sub, arm
        assert "CUDAGlobalMemoryMb >= 80000" in sub, arm
        assert "WITH_TWIN" not in sub          # eps>0 forces the twin
        assert "N_ROUNDS=30" in sub and "TRAIN_CAP=723" in sub
        assert sub.rstrip().endswith(
            f"from experiments/condor/"
            f"configs_pofd_feature_endogenization_n5_{arm}.txt"), arm


def test_sub_feature_knobs_only_where_they_belong():
    assert "PROFILE_DROP_COLS=gender" in GEN.fe5_sub("gd")
    assert "PROFILE_PERMUTE_COLS=gender" in GEN.fe5_sub("gp")
    for arm in ("nat", "frozen"):
        sub = GEN.fe5_sub(arm)
        assert "PROFILE_DROP_COLS" not in sub, arm
        assert "PROFILE_PERMUTE_COLS" not in sub, arm
    # forward KL on every trained arm; the frozen arm never trains
    for arm in ("nat", "gd", "gp"):
        assert "KL_DIRECTION=forward" in GEN.fe5_sub(arm), arm
    assert "USE_LORA=0" in GEN.fe5_sub("frozen")
    assert "FRESH_EACH_ROUND=0" in GEN.fe5_sub("frozen")


def test_sub_environments_match_the_established_subs():
    # token-for-token equality with the seeds-{0,42,43} environment,
    # ignoring only the wandb run-name suffix
    for arm, old in (("nat", "fes"), ("frozen", "fef"),
                     ("gd", "fegd"), ("gp", "fegp")):
        with open(os.path.join(CONDOR,
                               f"at_pofd_qwen7b_{old}.sub")) as fh:
            old_env = [ln for ln in fh.read().splitlines()
                       if ln.startswith("environment")][0]
        new_env = [ln for ln in GEN.fe5_sub(arm).splitlines()
                   if ln.startswith("environment")][0]

        def toks(line):
            return sorted(t for t in line.split()
                          if not t.startswith("WANDB_RUN_SUFFIX"))

        assert toks(new_env) == toks(old_env), arm


def test_submit_key_is_registered():
    with open(os.path.join(CONDOR, "submit_pofd_sweep.sh")) as fh:
        sh = fh.read()
    assert ('feature_endogenization_n5) TARGETS='
            '"feature_endogenization_n5_nat '
            'feature_endogenization_n5_frozen '
            'feature_endogenization_n5_gd '
            'feature_endogenization_n5_gp"') in sh
    for arm in ARMS:
        assert f"feature_endogenization_n5_{arm}" in sh, arm


# -- analyzer ------------------------------------------------------------

def test_panels_ab_use_five_seeds():
    m = _llm("plot_feature_endogenization_main")
    assert m.PANEL_SEEDS == (0, 42, 43, 44, 45)


def test_shared_three_seed_constant_is_untouched():
    # the beta-sweep and environment-dose figures still run at three
    # seeds -- extending them was NOT requested
    b = _llm("plot_feature_endogenization_beta_final")
    assert b.SEEDS == (0, 42, 43)


def test_student_t_critical_value_for_five_seeds():
    t = _llm("plot_teacher_intervention_appendix")
    # two-sided 95%, df = n - 1 = 4
    assert abs(t.T_CRIT_95[5] - 2.7764451051977987) < 1e-12
    assert abs(t.T_CRIT_95[3] - 4.302652729911275) < 1e-12


def test_mean_ci_uses_the_five_seed_critical_value():
    import numpy as np
    t = _llm("plot_teacher_intervention_appendix")
    values = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    mean, interval = t.mean_ci(values)
    assert abs(float(mean[0]) - 3.0) < 1e-12
    se = np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5)
    assert abs(float(interval[0]) - 2.7764451051977987 * se) < 1e-12


def test_displayed_labels_use_lambda_not_beta():
    with open(os.path.join(
            LLM, "plot_feature_endogenization_main.py")) as fh:
        src = fh.read()
    assert r"$\lambda=1$" in src
    assert r"$\lambda=.5$" in src
    assert r"$\lambda=0$" in src
    assert r"\beta" not in src
    # the exported condition labels carry lambda too
    for key in ("lambda_1", "lambda_0.5", "lambda_0"):
        assert f'"{key}"' in src, key
    assert '"beta_1"' not in src and '"beta_0"' not in src


def test_panel_ab_plot_ribbons_and_export_summary():
    with open(os.path.join(
            LLM, "plot_feature_endogenization_main.py")) as fh:
        src = fh.read()
    assert "fill_between" in src
    assert "mean_ci(values)" in src
    assert "feature_endogenization_main_summary.csv" in src
    for column in ("ci_half_width", "ci_lo", "ci_hi", "n_seeds"):
        assert f'"{column}"' in src, column
    # per-seed values are still exported
    assert "population_incremental_r2" in src


def test_panel_c_is_unchanged():
    with open(os.path.join(
            LLM, "plot_feature_endogenization_main.py")) as fh:
        src = fh.read()
    # panel (c) keeps its own five-seed teacher loader and constants
    assert "load_teacher_runs()" in src
    assert "(c) Reference source check" in src
    assert "teacher_constant(teacher)" in src
    # and its plotting block is NOT driven by PANEL_SEEDS (the panel
    # ends at the figure save; the reporting section below it belongs
    # to panels a/b)
    c_block = src.split("source_axis = axes[2]")[1].split(
        "for extension in")[0]
    assert "PANEL_SEEDS" not in c_block
    assert "panel_a" not in c_block and "panel_b" not in c_block


def test_peak_is_read_off_the_five_seed_mean_curve():
    with open(os.path.join(
            LLM, "plot_feature_endogenization_main.py")) as fh:
        src = fh.read()
    tail = src.split("summary_path.open")[1]
    # the mean curve is computed first, the peak located on it, and the
    # whole trajectory printed -- no window/seed re-selection
    assert "mean, interval = mean_ci(values)" in tail
    assert "peak = int(mean.argmax())" in tail
    assert "full five-seed mean trajectory" in tail


def test_missing_seed_runs_are_a_hard_error(tmp_path, monkeypatch):
    m = _llm("plot_feature_endogenization_main")
    # point the loader at an empty root: every tag is missing
    monkeypatch.setattr(m, "RUN_ROOT", tmp_path)
    try:
        m.require_runs((0, 42, 43, 44, 45))
    except SystemExit as exc:
        text = str(exc)
        assert "hard-require" in text
        assert "30 required runs are missing" in text
        assert "pofdws2f_qwen7b_b1" in text
    else:
        raise AssertionError("missing runs must raise SystemExit")
