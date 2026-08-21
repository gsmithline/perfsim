"""Tests for the SFT TRAINING-DOSE scouts (2026-08-21).

Three one-round static families ask whether a WEAKER SFT fit leaves the
served vector closer to the entering Qwen model: optimizer updates,
learning rate, LoRA rank.

The design lessons these pin come from the 100-round subsample wave:
  * ONE round, because the closed loop drives the population to an
    absorbing constant after which a == 1 by construction;
  * exactly ONE dial moves per family;
  * NO expected ordering is encoded anywhere -- monotonicity is a result
    to be measured, not an assumption to be baked in.

Run with USE_TF=0.
"""
import importlib.util
import os

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
CONDOR = os.path.join(REPO, "experiments", "condor")
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")
RUNNER = os.path.join(PIPE, "run_pokec_gated_lm.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load("gen_sftd", os.path.join(CONDOR, "gen_pofd_sweep.py"))
AN = _load("an_sftd", os.path.join(PIPE, "analyze_sft_dose.py"))


# ------------------------------------------------------------ the grids
def test_grids_are_the_requested_ones():
    assert GEN.SFTD_UPDATES == [1, 5, 20, 50, 100, 181]
    assert GEN.SFTD_LRS == ["1e-6", "3e-6", "1e-5", "3e-5"]
    assert GEN.SFTD_RANKS == [1, 4, 8, 32, 128]
    # the analyzer carries the FULL grids including the shared endpoint
    assert AN.UPDATES == [1, 5, 20, 50, 100, 181]
    assert AN.LRS == [1e-6, 3e-6, 1e-5, 3e-5, 5e-5]
    assert AN.RANKS == [1, 4, 8, 32, 128, 512]


def test_job_counts():
    assert len(GEN.sftd_update_rows()) == 6
    assert len(GEN.sftd_lr_rows()) == 4
    assert len(GEN.sftd_rank_rows()) == 5
    assert len(GEN.sftd_smoke_rows()) == 1


def test_shared_endpoint_is_queued_exactly_once():
    """U=181 / LR=5e-5 / rank=512 is the full-dose end of ALL THREE
    families. Queuing it in more than one key would mean two jobs writing
    the same run directory -- the race that hit the Wu-limit wave."""
    shared = GEN.sftd_tag(GEN.SFTD_STD_U, GEN.SFTD_STD_LR,
                          GEN.SFTD_STD_RANK)
    allt = ([r.split(",")[0] for r in GEN.sftd_update_rows()]
            + [r.split(",")[0] for r in GEN.sftd_lr_rows()]
            + [r.split(",")[0] for r in GEN.sftd_rank_rows()])
    assert allt.count(shared) == 1
    assert shared in [r.split(",")[0] for r in GEN.sftd_update_rows()]


def test_zero_dose_arms_have_no_job():
    """U=0, LR=0 and rank=0 all mean 'no adaptation happened', which IS
    the entering model -- so none of them may appear as a GPU cell."""
    allt = ([r.split(",")[0] for r in GEN.sftd_update_rows()]
            + [r.split(",")[0] for r in GEN.sftd_lr_rows()]
            + [r.split(",")[0] for r in GEN.sftd_rank_rows()])
    for t in allt:
        assert "_u0_" not in t and "_lr0_" not in t and "_rank0_" not in t


def test_exactly_one_dial_moves_per_family():
    for rows, idx, others in ((GEN.sftd_update_rows(), 15, [16, 17]),
                              (GEN.sftd_lr_rows(), 16, [15, 17]),
                              (GEN.sftd_rank_rows(), 17, [15, 16])):
        cols = [[c.strip() for c in r.split(",")] for r in rows]
        assert len({c[idx] for c in cols}) == len(rows)      # the dial
        for o in others:
            assert len({c[o] for c in cols}) == 1, o          # held fixed


def test_every_cell_is_one_round_ordinary_sft():
    for r in (GEN.sftd_update_rows() + GEN.sftd_lr_rows()
              + GEN.sftd_rank_rows() + GEN.sftd_smoke_rows()):
        c = [x.strip() for x in r.split(",")]
        assert c[1] == "sft" and c[2] == "0", r      # lambda_KL = 0
        assert c[11] == "1" and c[14] == "1", r      # W = 1, k = 1
        assert c[24] == "1", r                        # ONE round


def test_tag_grammar_does_not_reuse_l1_for_kl_strength():
    """_l1_ has meant the innate anchor k=1 in every tag in this project.
    The dose dials get their own unambiguous tokens, and the LoRA rank
    token is spelled 'rank' because a bare _r<N>_ would collide with the
    trailing round count _r1."""
    t = GEN.sftd_tag(20, "3e-6", 8)
    assert "_l1_" in t and "_u20_" in t and "_lr3em6_" in t
    assert "_rank8_" in t and t.endswith("_r1")
    assert "_r8_" not in t


def test_sub_uses_the_existing_step_cap_not_a_new_knob():
    """SFT_EPOCHS=0 + SFT_MAX_STEPS is the path the pofdbud_ budget wave
    already used. Inventing a second step-cap env var would mean two
    controls that can disagree."""
    for key in (GEN.SFTD_UPDATE_KEY, GEN.SFTD_LR_KEY, GEN.SFTD_RANK_KEY):
        env = next(ln for ln in GEN.sftd_sub(key).splitlines()
                   if ln.startswith("environment"))
        assert "SFT_EPOCHS=0 SFT_MAX_STEPS=$(steps)" in env
        assert "SFT_MAX_UPDATES" not in env
        assert "SFT_LR=$(lr)" in env and "LORA_R=$(rank)" in env
        assert "SAVE_SFT_ORDER=1" in env


def test_lora_alpha_is_two_r_so_scaling_is_constant():
    """alpha/r constant across ranks is what makes the rank sweep a
    capacity dial rather than also a scaling dial."""
    src = open(RUNNER).read()
    assert "lora_alpha=2 * lora_r" in src


def test_submit_registers_all_four_keys():
    src = open(os.path.join(CONDOR, "submit_pofd_sweep.sh")).read()
    assert ("qwen_sft_update_dose|qwen_sft_lr_dose|qwen_sft_rank_dose|"
            "qwen_sft_dose_smoke) TARGETS=") in src


# --------------------------------------------------- runner provenance
def test_save_sft_order_is_opt_in_and_absent_by_default():
    src = open(RUNNER).read()
    assert 'save_sft_order = _env_int("SAVE_SFT_ORDER", 0) == 1' in src
    assert 'config["save_sft_order"] = True' in src


def test_serve_eval_mode_is_recorded():
    """The 2026-08-20 eval fix left NO trace in the artifact, so runs
    before and after it are indistinguishable from trajectory.pt alone.
    Every later run must self-certify."""
    src = open(RUNNER).read()
    assert '"serve_eval_mode": True' in src
    assert '"git_sha": _git_sha()' in src


def test_learner_records_actual_optimizer_steps():
    """Requesting max_steps=U is not the same as taking U steps."""
    src = open(os.path.join(REPO, "perfsim/learners/lm/sft.py")).read()
    assert "last_train_stats" in src
    assert '"global_step": int(getattr(trainer.state, "global_step", -1))' \
        in src
    assert '"trainer_seed"' in src


# ------------------------------------------------------------ analyzer
def test_a_recovers_known_mixtures_and_is_unclipped():
    rng = np.random.default_rng(0)
    base, star = rng.random(723), rng.random(723)
    for t in (0.0, 0.3, 1.0):
        a, res = AN.fit_a(t * star + (1 - t) * base, star, base)
        assert abs(a - t) < 1e-9 and res < 1e-9
    hi, _ = AN.fit_a(1.4 * star - 0.4 * base, star, base)
    assert hi > 1.0


def test_residual_flags_outputs_neither_reference_explains():
    rng = np.random.default_rng(1)
    base, star, junk = rng.random(723), rng.random(723), rng.random(723)
    _, res = AN.fit_a(junk, star, base)
    assert res > 0.1


def test_collapse_diagnostics_detect_a_constant_output():
    """A small adapter may learn only a global scalar shift; SD alone
    would not distinguish that from a faithful narrow fit, so the CSV
    carries unique-value count and max mode share too."""
    const = np.full(723, 0.62)
    uniq, counts = np.unique(np.round(const, 6), return_counts=True)
    assert uniq.size == 1 and counts.max() / counts.sum() == 1.0
    src = open(os.path.join(PIPE, "analyze_sft_dose.py")).read()
    assert '"n_unique"' in src and '"max_mode_share"' in src


def test_no_expected_ordering_is_encoded():
    src = open(os.path.join(PIPE, "analyze_sft_dose.py")).read()
    assert "No expected ordering is encoded" in src
    assert "no monotonicity assumed" in src


def test_analyzer_hard_fails_on_missing_cells(tmp_path):
    with pytest.raises(SystemExit) as ei:
        AN.analyse([str(tmp_path)], tmp_path / "out")
    assert "HARD FAIL" in str(ei.value)


def test_analyzer_reads_only_the_first_served_vector():
    """These are one-round cells; touching the population trajectory
    would reintroduce the feedback confound the design exists to avoid."""
    src = open(os.path.join(PIPE, "analyze_sft_dose.py")).read()
    assert 'pred = np_(d["pred_raw"])[0]' in src
    assert "op_raw" not in src.split("def cell_metrics")[1][:1500]


def test_structure_discriminator_separates_shift_from_collapse():
    """THE distinction the rank sweep exists to make. Low rank limits the
    SHAPE of the change, not its magnitude -- even rank 1 can learn a
    large global shift and serve one value to everyone. RMSE to Qwen
    cannot tell those apart; centered RMSE and cross-agent correlation
    can."""
    rng = np.random.default_rng(0)
    base = rng.random(723)
    shifted = base + 0.2                       # Qwen + global offset
    const = np.full(723, float(base.mean()))   # collapsed at Qwen's mean

    # preserved structure: raw RMSE large, but shape identical
    assert AN.rmse(shifted, base) > 0.1
    assert AN.rmse_centered(shifted, base) < 1e-9
    assert AN.corr(shifted, base) > 0.999

    # collapse: no structure at all
    assert AN.rmse_centered(const, base) > 0.1
    assert np.isnan(AN.corr(const, base)), (
        "a constant output must give nan, not 0 -- 0 would read as "
        "'uncorrelated' instead of 'no structure at all'")


def test_structure_metrics_are_in_the_csv_and_figure():
    src = open(os.path.join(PIPE, "analyze_sft_dose.py")).read()
    for field in ("rmse_to_base_centered", "corr_to_base",
                  "mean_shift_from_base", "corr_to_target"):
        assert f'"{field}"' in src, field
    assert "structure vs collapse" in src
    assert "IMPLICIT ANCHOR" in src and "MODE COLLAPSE" in src
