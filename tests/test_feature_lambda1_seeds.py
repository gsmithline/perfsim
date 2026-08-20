"""Tests for the lambda=1 REPLICATION wave (2026-08-20,
feature_lambda1_seeds).

10 jobs: the natural lambda=1 feature-endogenization cell at seeds
46-55, byte-identical to the existing cells apart from the seed. The
point is the LOCK-IN RATE: across seeds 0/42/43/44/45 the condition is
bimodal (four transient spikes that decay to the noise floor, one
persistent lock-in), and 1-in-5 is consistent with anything from ~5%
to ~50%.

The analyzer's lock-in criterion is PRE-SPECIFIED and must stay that
way -- these tests pin the threshold and check the classification is
insensitive to it across the plausible range, so it cannot be quietly
tuned to whatever the new seeds do.

Run with USE_TF=0.
"""
import importlib.util
import os
import sys

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
CONDOR = os.path.join(REPO, "experiments", "condor")
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")
EXISTING = [0, 42, 43, 44, 45]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load("gen_fl1", os.path.join(CONDOR, "gen_pofd_sweep.py"))
AL = _load("analyze_fl1",
           os.path.join(PIPE, "analyze_feature_lambda1_seeds.py"))


# -- generator -----------------------------------------------------------

def test_ten_new_seeds_and_none_of_the_existing_five():
    rows = GEN.fl1_rows()
    assert len(rows) == 10
    tags = {r.split(",")[0] for r in rows}
    assert len(tags) == 10
    seeds = {int(t.split("_s")[-1].split("_")[0]) for t in tags}
    assert seeds == set(range(46, 56))
    # the five completed seeds are load-bearing for the published
    # figure -- re-queueing one would overwrite it
    assert not (seeds & set(EXISTING))
    for s in EXISTING:
        assert GEN.fl1_tag(s) not in tags


def test_rows_match_the_existing_natural_cell_grammar():
    """Same family and same environment as the seeds already run --
    a replication has to differ ONLY in the seed."""
    for r in GEN.fl1_rows():
        cols = [c.strip() for c in r.split(",")]
        assert cols[0].startswith(
            "pofdws2f_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2_")
        assert cols[0].endswith("_fresh_data")
        assert cols[1] == "sft_kl" and cols[2] == "1"   # lambda_KL=1
        assert cols[9] == "0.2"                          # eps_social
        assert cols[14] == "0.4"                         # eps_AI


def test_anchor_stays_at_k_0p2_here():
    """This wave replicates the ORIGINAL feature cell, so the FJ
    anchor stays 0.2 -- it must not inherit the k=1 grid's anchor."""
    env = next(ln for ln in GEN.fl1_sub().splitlines()
               if ln.startswith("environment"))
    assert "INNATE_LAMBDA=0.2" in env
    assert "INNATE_LAMBDA=1 " not in env
    assert "KL_DIRECTION=forward" in env
    assert f'CUDADeviceName == "{GEN.FE5_A100}"' in GEN.fl1_sub()


def test_no_collision_with_any_other_key():
    tags = {r.split(",")[0] for r in GEN.fl1_rows()}
    for other in (GEN.qk1_rows(), GEN.qgs_rows(), GEN.fam_rows(),
                  GEN.fe5_rows("nat"), GEN.fe5_rows("gd")):
        assert not (tags & {r.split(",")[0] for r in other})


# -- analyzer ------------------------------------------------------------

def test_lockin_threshold_is_prespecified():
    assert AL.LOCKIN_THRESHOLD == 0.02
    assert AL.LATE == list(range(25, 30))
    src = open(os.path.join(
        PIPE, "analyze_feature_lambda1_seeds.py")).read()
    assert "PRE-SPECIFIED" in src
    # the rationale and the sensitivity sweep must ship with it
    assert "SENSITIVITY" in src
    assert AL.SENSITIVITY[0] < AL.LOCKIN_THRESHOLD < AL.SENSITIVITY[-1]


def test_classification_is_insensitive_across_the_plausible_range():
    """The five known late-window values are 0.0063, 0.0081, 0.0042,
    0.0001, 0.1025. Any cut in the gap gives the same answer -- that
    is what makes the criterion a description of bimodality rather
    than a tuned knob."""
    late = [0.0063, 0.0081, 0.0042, 0.0001, 0.1025]
    counts = {sum(1 for v in late if v > th)
              for th in AL.SENSITIVITY}
    assert counts == {1}, counts


def test_clopper_pearson_matches_known_values():
    # 1/5: exact 95% interval is about [0.005, 0.716]
    lo, hi = AL.clopper_pearson(1, 5)
    assert abs(lo - 0.00505) < 2e-3, lo
    assert abs(hi - 0.71642) < 2e-3, hi
    # degenerate ends are closed, not NaN
    lo0, hi0 = AL.clopper_pearson(0, 5)
    assert lo0 == 0.0 and hi0 > 0.0
    lon, hin = AL.clopper_pearson(5, 5)
    assert hin == 1.0 and lon < 1.0


def test_rate_interval_narrows_with_more_seeds():
    """The whole point of the wave: 1/5 is uninformative, and the
    same rate at 3/15 must be materially tighter."""
    lo5, hi5 = AL.clopper_pearson(1, 5)
    lo15, hi15 = AL.clopper_pearson(3, 15)
    assert (hi15 - lo15) < (hi5 - lo5) * 0.75


# -- reproducibility control (FL1R) --------------------------------------

def test_repro_wave_is_three_replicates_plus_four_seeds():
    rows = GEN.fl1r_rows()
    assert len(rows) == 7
    tags = [r.split(",")[0] for r in rows]
    assert len(set(tags)) == 7
    reps = [t for t in tags if "_rep" in t]
    new = [t for t in tags if "_rep" not in t]
    assert len(reps) == 3 and len(new) == 4
    assert {int(t.split("_s")[-1].split("_")[0]) for t in new} == {1, 2, 3, 4}


def test_replicates_carry_seed_0_in_tag_and_queue():
    """A replicate is the SAME seed with a different tag -- if the
    queue column drifted it would be an ordinary new seed and the
    control would prove nothing."""
    for r in GEN.fl1r_rows():
        cols = [c.strip() for c in r.split(",")]
        if "_rep" in cols[0]:
            assert "_s0_rep" in cols[0], r
            assert cols[3] == "0", r


def test_repro_wave_never_requeues_a_completed_run():
    tags = {r.split(",")[0] for r in GEN.fl1r_rows()}
    done = {GEN.fl1_tag(s) for s in EXISTING + list(range(46, 56))}
    assert not (tags & done)
    # seed 0's plain tag is the PUBLISHED cell -- must never appear
    assert GEN.fl1_tag(0) not in tags
    assert "pofdws2f_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2_s0_fresh_data" \
        not in tags


def test_repro_wave_is_gpu_pinned():
    """Seed 0 originally ran on an A100; pinning means a replicate
    that diverges cannot be blamed on GPU architecture."""
    assert f'CUDADeviceName == "{GEN.FE5_A100}"' in GEN.fl1r_sub()


def test_analyzer_detects_replicates_and_reports_a_verdict():
    src = open(os.path.join(
        PIPE, "analyze_feature_lambda1_seeds.py")).read()
    assert "_rep{r}_" in src or '_rep' in src
    assert "REPRODUCIBILITY" in src
    assert "across-seed spread" in src
    # the verdict must key off replicate spread vs across-seed spread
    assert "spread > 0.5 * across" in src
