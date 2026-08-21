"""Tests for the OBSERVATION-RATE SUBSAMPLING wave (2026-08-21).

Celestine's hypothesis: ordinary SFT leans MORE on the pretrained model
when it observes less of the population. The wave varies ONE dial on the
completed Wu-boundary b0 cell -- how many of the 723 agents' labels reach
the optimizer -- and the three things that make it interpretable are all
mechanical, so all three are tested:

  1. serving is untouched (all 723 served in every arm);
  2. subsets are NESTED across arms, so "different people" cannot explain
     a between-arm difference;
  3. round 0 is subsampled too, which is why TRAIN_CAP could not be
     reused.

Run with USE_TF=0.
"""
import copy
import importlib.util
import json
import os
import tempfile

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
CONDOR = os.path.join(REPO, "experiments", "condor")
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load("gen_qss", os.path.join(CONDOR, "gen_pofd_sweep.py"))
CHK = _load("chk_qss", os.path.join(PIPE, "check_pofd_sanity.py"))
AN = _load("an_qss", os.path.join(PIPE, "analyze_qwen_subsample.py"))
MF = json.load(open(os.path.join(CONDOR, "manifest_qwen_subsample.json")))

N = 723
SEED0 = 0 + 777331


# ================================================================ sampling
def _perm(t, seed=SEED0):
    g = torch.Generator().manual_seed(seed + t)
    return torch.randperm(N, generator=g)


def test_observation_grid_is_the_intended_percentages():
    """14/36/72/181/362/542 of 723 = 2/5/10/25/50/75%. 542 joined on
    2026-08-21, before any production job was submitted, which is why it
    could go into the existing key instead of needing its own."""
    want = {14: 0.02, 36: 0.05, 72: 0.10, 181: 0.25, 362: 0.50, 542: 0.75}
    assert GEN.QSS_COUNTS == sorted(want)
    for c, frac in want.items():
        assert abs(c / N - frac) < 0.006, (c, frac, c / N)
        assert round(frac * N) == c or abs(round(frac * N) - c) <= 1, c


def test_audit_and_generator_agree_on_the_grid():
    """Two modules carry the count list -- the audit runs standalone on
    the cluster, so it cannot import the generator. A silent divergence
    would queue one grid and audit another."""
    AUD = _load("aud_qss", os.path.join(PIPE, "audit_qwen_subsample.py"))
    assert AUD.COUNTS == GEN.QSS_COUNTS
    assert AUD.FULL == GEN.QSS_FULL
    assert (AUD.CM_N, AUD.CM_REPEAT) == (GEN.QSS_CM_N, GEN.QSS_CM_REPEAT)
    assert AUD.REUSED_TAG == GEN.QSS_REUSED_TAG
    assert MF["counts"] == GEN.QSS_COUNTS


def test_subsets_are_nested_across_every_arm_and_round():
    """THE property that makes the arms comparable: at a given round the
    14 sit inside the 36 sit inside the 72... If the draw depended on the
    sample size these would be independent subsets and any difference
    between arms could just be different people."""
    counts = GEN.QSS_COUNTS + [GEN.QSS_FULL]
    for t in (0, 1, 37, 99):
        p = _perm(t)
        for a, b in zip(counts, counts[1:]):
            assert set(p[:a].tolist()) <= set(p[:b].tolist()), (t, a, b)


def test_round_zero_is_subsampled_too():
    """TRAIN_CAP is applied only for t>0, so round 0 would train on all
    723 and the first adapter would not be subsampled at all. That is
    exactly why this wave needed a new knob."""
    src = open(os.path.join(
        REPO, "experiments/scripts/cluster_pipelines/"
              "run_pokec_gated_lm.py")).read()
    # the sampling block is NOT inside the t>0 branch
    i_block = src.index("OBSERVATION-RATE SUBSAMPLING (2026-08-21)")
    i_t0 = src.index("if t == 0:\n                train_data = initial_data")
    assert i_block > i_t0
    assert "sft_sample_n > 0 and train_data is not None" in src
    # and round 0 really does draw a different subset from round 1
    assert not torch.equal(_perm(0)[:72], _perm(1)[:72])


def test_draw_is_deterministic_and_reconstructible():
    for t in (0, 5, 99):
        g = torch.Generator().manual_seed(SEED0 + t)
        assert torch.equal(torch.randperm(N, generator=g)[:36],
                           _perm(t)[:36])


def test_subset_is_unique_within_a_round():
    for t in (0, 50, 99):
        for c in GEN.QSS_COUNTS:
            ids = _perm(t)[:c]
            assert int(torch.unique(ids).numel()) == c


def test_sampling_stream_is_separate_from_the_population_stream():
    """A dedicated offset: if sampling shared the peer stream, drawing a
    subset would shift every subsequent peer pairing."""
    src = open(os.path.join(
        REPO, "experiments/scripts/cluster_pipelines/"
              "run_pokec_gated_lm.py")).read()
    assert "777331" in src and "424243" in src
    assert SEED0 != 0 + 424243


def test_compute_matched_tiling_matches_full_data_steps():
    tile = torch.arange(GEN.QSS_CM_REPEAT) % GEN.QSS_CM_N
    assert tile.numel() == 723
    ids = _perm(0)[:GEN.QSS_CM_N][tile]
    assert int(torch.unique(ids).numel()) == GEN.QSS_CM_N == 72
    steps = -(-GEN.QSS_CM_REPEAT // 4)
    assert steps == -(-723 // 4) == 181


# =============================================================== generator
def test_seven_new_jobs_and_the_full_arm_is_reused():
    rows = GEN.qss_rows()
    assert len(rows) == 7
    assert MF["n_reused"] == 1 and MF["n_new"] == 7
    assert MF["n_conceptual_cells"] == 8
    tags = [r.split(",")[0] for r in rows]
    assert len(set(tags)) == 7
    # the 100% arm is the completed QWU cell, NOT a queued job
    assert MF["reused_tag"] == GEN.QSS_REUSED_TAG
    assert GEN.QSS_REUSED_TAG not in tags
    assert not any("_n723_" in t for t in tags)


def test_each_observation_count_appears_once():
    tags = [r.split(",")[0] for r in GEN.qss_rows()]
    for c in GEN.QSS_COUNTS:
        assert sum(1 for t in tags if f"_n{c}_" in t) == 1, c


def test_compute_matched_tag_is_distinct_from_the_plain_arm():
    cm = GEN.qss_tag(GEN.QSS_CM_N, GEN.QSS_CM_REPEAT)
    plain = GEN.qss_tag(GEN.QSS_CM_N)
    assert cm != plain and "rep723" in cm and "rep" not in plain
    assert cm in [r.split(",")[0] for r in GEN.qss_rows()]


def test_queue_surface_is_ordinary_sft_at_the_boundary():
    for r in GEN.qss_rows():
        c = [x.strip() for x in r.split(",")]
        assert c[1] == "sft" and c[2] == "0", r      # lambda = 0
        assert c[3] == "0", r                         # seed 0
        assert c[11] == "1" and c[14] == "1", r       # W = 1, k = 1
        assert c[23] == "100", r


def test_sub_passes_both_sampling_knobs_and_opens_both_gates():
    env = next(ln for ln in GEN.qss_sub().splitlines()
               if ln.startswith("environment"))
    assert "SFT_SAMPLE_N=$(samplen)" in env
    assert "SFT_SAMPLE_REPEAT_TO=$(repeatto)" in env
    assert "AI_GATE_MODE=all_open" in env
    assert "PEER_GATE_MODE=all_open" in env
    assert f'CUDADeviceName == "{GEN.QSS_H100}"' in GEN.qss_sub()
    for host in GEN.BAD_NODES:
        assert host in GEN.qss_sub()


def test_smoke_is_one_short_cell_on_the_new_path():
    rows = GEN.qss_smoke_rows()
    assert len(rows) == 1
    t = rows[0].split(",")[0]
    assert t.endswith("_s0_r3smoke")
    assert t not in [r.split(",")[0] for r in GEN.qss_rows()]


def test_no_collision_with_any_other_key():
    mine = {r.split(",")[0] for r in GEN.qss_rows()} | \
        {r.split(",")[0] for r in GEN.qss_smoke_rows()}
    for other in (GEN.qwu_rows(), GEN.qwu_icl_rows(), GEN.qwu_smoke_rows(),
                  GEN.qmech_rows(), GEN.qk1_rows()):
        assert not (mine & {r.split(",")[0] for r in other})


def test_submit_registers_both_keys():
    src = open(os.path.join(CONDOR, "submit_pofd_sweep.sh")).read()
    assert "qwen_subsample|qwen_subsample_smoke) TARGETS=" in src


# ================================================================= analyzer
def test_a_recovers_known_mixtures():
    rng = np.random.default_rng(0)
    base = rng.random(N)
    star = rng.random(N)
    for true_a in (0.0, 0.25, 0.5, 1.0):
        m = true_a * star + (1 - true_a) * base
        a, res = AN.fit_a(m, star, base)
        assert abs(a - true_a) < 1e-9, (true_a, a)
        assert res < 1e-9


def test_a_is_not_clipped_to_the_unit_interval():
    """A served vector outside the segment between prior and data is a
    real statement; clipping would hide it."""
    rng = np.random.default_rng(1)
    base, star = rng.random(N), rng.random(N)
    a_hi, _ = AN.fit_a(1.5 * star - 0.5 * base, star, base)
    a_lo, _ = AN.fit_a(-0.3 * star + 1.3 * base, star, base)
    assert a_hi > 1.0 and a_lo < 0.0


def test_residual_separates_bad_fit_from_low_a():
    """A small a with a LARGE residual means neither reference explains
    the served vector -- a different finding from 'it followed the
    prior', so the two must not be conflated."""
    rng = np.random.default_rng(2)
    base, star = rng.random(N), rng.random(N)
    junk = rng.random(N)
    a, res = AN.fit_a(junk, star, base)
    assert res > 0.1


def test_degenerate_references_give_nan_not_noise():
    v = np.linspace(0, 1, N)
    a, res = AN.fit_a(v, v, v)
    assert np.isnan(a) and res == pytest.approx(0.0, abs=1e-12)


def test_analyzer_uses_the_runs_own_labels_not_the_oracle():
    src = open(os.path.join(PIPE, "analyze_qwen_subsample.py")).read()
    assert "m_star = innate if t == 0 else op[t - 1]" in src
    assert "Never the oracle" in src
    assert "perfect-prediction" in src


def test_analyzer_pins_the_canonical_base_hash():
    assert AN.CANON_SHA == CHK.QMECH_CANONICAL_PRED_SHA


def test_analyzer_does_not_assume_monotonicity():
    src = open(os.path.join(PIPE, "analyze_qwen_subsample.py")).read()
    assert "NO MONOTONICITY IS ASSUMED OR ENFORCED" in src
    assert all(c["monotonicity"] == "NOT assumed or enforced"
               for c in [{"monotonicity": "NOT assumed or enforced"}])


def test_analyzer_hard_fails_without_the_cells(tmp_path):
    with pytest.raises(SystemExit) as ei:
        AN.resolve_cells([str(tmp_path)])
    assert "HARD FAIL" in str(ei.value)


# ====================================================== checker sabotage
def _build_qss(setup_pp, tmp, count=72, repeat_to=0, rounds=100,
               corrupt_ids_round=None, corrupt_label_round=None,
               wrong_seed=False, drop_arrays=False, bad_n_train=None):
    """A genuine sampled run: real population operators, and the SFT
    subsets built exactly the way the runner builds them (one permutation
    of all 723 per round from SFT_SAMPLE_SEED + round, prefix taken)."""
    PP = _load("sim_pp_qss", os.path.join(PIPE, "sim_perfect_predictor.py"))
    gp = _load("gp_qss", os.path.join(PIPE, "_gated_pop.py"))
    n = int(setup_pp["n"])
    innate = setup_pp["innate"]
    served = (innate * 0.5 + 0.25).clone()
    acc = []
    op, tw, pred = PP.simulate(
        setup_pp, innate_k=1.0, w_plat=1.0, eps_social=0.2, eps_ai=1.0,
        rounds=rounds, seed=0, ai_gate_mode="all_open",
        peer_gate_mode="all_open", accepted_out=acc,
        served_fn=lambda x, t: served, require_open_gate=False)
    s_seed = 777331
    n_rows = repeat_to or count
    tile = (torch.arange(repeat_to) % count) if repeat_to else None
    idx_rows, y_rows, gates, traj = [], [], [], []
    for t in range(rounds):
        g = torch.Generator().manual_seed(s_seed + t)
        ids = torch.randperm(n, generator=g)[:count]
        if repeat_to:
            ids = ids[tile]
        if corrupt_ids_round == t:
            ids = ids.clone()
            ids[3] = (int(ids[3]) + 1) % n
        src = innate if t == 0 else op[t - 1]
        y = src[ids.long()].clone()
        if corrupt_label_round == t:
            y = y.clone()
            y[2] += 0.25
        idx_rows.append(ids.long())
        y_rows.append(y.float())
        gates.append(gp.ai_gate(pred[t], innate if t == 0 else op[t - 1],
                                1.0, "all_open"))
        traj.append({"round": t, "deployment": 0, "is_deploy": 1,
                     "n_train": (bad_n_train if (bad_n_train and t == 4)
                                 else n_rows),
                     "contact": 1.0, "accepted": acc[t], "s_tag": 0.0,
                     "peer_gate_mode": "all_open", "peer_pairs": n,
                     "twin_mean": float(tw[t].mean()),
                     "twin_std": float(tw[t].std()), "twin_bias": 0.0,
                     "op_twin_l1": 0.0, "op_twin_w1": 0.0})
    tok = f"n{count}" + (f"rep{repeat_to}" if repeat_to else "")
    cfg = {
        "run_tag": f"pofdqss_qwen7b_b0_eaopen_w1_l1_esopen_{tok}_s0"
                   f"_r{rounds}",
        "kl_beta": 0.0, "kl_direction": "forward", "kl_ref_adapter": None,
        "training_style": "sft", "rlhf_feedback": False,
        "base_model": "Qwen/Qwen2.5-7B-Instruct", "n_rounds": rounds,
        "epoch_size": 100, "deploy_every": 1, "data_regime": "replace",
        "seed": 0, "n_labeled": 723, "max_steps": 0, "sft_epochs": 1,
        "sft_batch_size": 4, "lora_r": 512, "use_lora": True,
        "sft_lr": 5e-5, "hist_bins": 50, "seed_base_data": True,
        "train_cap": 723, "platform_sus_scale": 1.0, "anchor_mode": "fixed",
        "pop_model": "ab", "eps": 0.2, "eps_ai": 1.0, "gamma_bias": 0.0,
        "ai_gate_mode": "all_open", "peer_gate_mode": "all_open",
        "w_plat": 1.0, "innate_lambda": 1.0,
        "population_update": "nested_ai_then_social_v1", "run_mode": "loop",
        "canary_delta": 0.0, "grad_decomp": 1, "save_adapter_rounds": [],
        "icl_k": 0, "icl_days": 0, "icl_select": "random",
        "icl_ctx_source": "live", "icl_snapshot_round": -1,
        "icl_ctx_donor": None, "icl_ctx_donor_tag": None,
        "icl_ctx_donor_round": None, "icl_ctx_donor_hash": None,
        "feedback_mode": "none", "icrh": False, "reward_kind": "accuracy",
        "ab_retain": False, "n_probe": 64, "tel_eval_cap": 64,
        "grad_norm_n": 8, "fresh_each_round": True, "pristine_frac": 0.0,
        "replay_frac": 0.0, "pop_reset": False, "ab_sweeps": 1,
        "pop_order": "peer_first", "profile_shuffle_p": 0.0,
        "profile_sort_q": 0.0, "profile_drop_cols": [],
        "profile_permute_cols": [], "teacher_label_delta": 0.0,
        "teacher_label_col": None, "teacher_label_fav": None,
        "teacher_group_seed": 0, "log_gender_gaps": False,
        "dataset": "movielens", "ml_target": "Action", "log_ppl_dist": True,
        "ppl_dist_cap": 0, "do_sample": False, "gen_temperature": 1.0,
        "ans_sample_k": 16, "ans_sample_n": 64, "ans_sample_t": 1.0,
        "host": "gpu-node", "save_raw_gen": True,
        "sft_sample_n": count,
        "sft_sample_seed": (s_seed + 1 if wrong_seed else s_seed),
        "hardware": {"hostname": "g001",
                     "gpu_name": "NVIDIA H100 80GB HBM3", "gpu_cc": "9.0",
                     "cuda_version": "12.4", "torch_version": "2.5.0",
                     "transformers_version": "4.46.0"},
    }
    if repeat_to:
        cfg["sft_sample_repeat_to"] = repeat_to
    payload = {
        "trajectory": traj, "config": cfg, "op_raw": op, "pred_raw": pred,
        "twin_raw": tw, "gate_raw": torch.stack(gates),
        "ppl_raw": torch.empty(0), "ans_raw": torch.empty(0),
        "ans_idx": torch.tensor([], dtype=torch.long),
        "replay_raw": torch.empty(0), "train_y_raw": torch.empty(0),
        "icl_idx_raw": torch.empty(0), "icl_val_raw": torch.empty(0),
        "icl_donor_vec": torch.empty(0), "innate": innate, "profiles": {},
        "probe_idx": torch.tensor([], dtype=torch.long),
        "canary": torch.zeros(n), "gender_true": None, "gender_disp": None,
        "teacher_pred": torch.empty(0),
    }
    if not drop_arrays:
        payload["sft_sample_idx_raw"] = torch.stack(idx_rows)
        payload["sft_sample_y_raw"] = torch.stack(y_rows)
    d = os.path.join(tmp, cfg["run_tag"])
    os.makedirs(d, exist_ok=True)
    torch.save(payload, os.path.join(d, "trajectory.pt"))
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    return d


@pytest.fixture(scope="module")
def setup_pp():
    from pathlib import Path
    PP = _load("sim_pp_setup", os.path.join(PIPE, "sim_perfect_predictor.py"))
    return PP.extract_loader()(
        Path(REPO) / "experiments/data/movielens/ml-100k", "Action")


def test_clean_sampled_fixture_passes_the_real_checker(setup_pp, tmp_path):
    """If this fails every sabotage below is meaningless."""
    d = _build_qss(setup_pp, str(tmp_path))
    errs = CHK.check_run(d)
    assert errs == [], errs


def test_clean_compute_matched_fixture_passes(setup_pp, tmp_path):
    d = _build_qss(setup_pp, str(tmp_path), count=72, repeat_to=723)
    errs = CHK.check_run(d)
    assert errs == [], errs


def test_sabotage_swapped_agent_id(setup_pp, tmp_path):
    """One id off the reconstruction must be caught -- the subset claim
    is the whole experiment."""
    d = _build_qss(setup_pp, str(tmp_path), corrupt_ids_round=7)
    errs = CHK.check_run(d)
    assert any("recorded SFT ids differ" in e for e in errs), errs


def test_sabotage_tampered_label(setup_pp, tmp_path):
    d = _build_qss(setup_pp, str(tmp_path), corrupt_label_round=11)
    errs = CHK.check_run(d)
    assert any("labels differ from the population state" in e
               for e in errs), errs


def test_sabotage_wrong_sampling_seed(setup_pp, tmp_path):
    """A run whose recorded seed does not generate its own subsets is not
    reconstructible, and nothing else would notice."""
    d = _build_qss(setup_pp, str(tmp_path), wrong_seed=True)
    errs = CHK.check_run(d)
    assert any("recorded SFT ids differ" in e for e in errs), errs


def test_sabotage_missing_provenance_arrays(setup_pp, tmp_path):
    d = _build_qss(setup_pp, str(tmp_path), drop_arrays=True)
    errs = CHK.check_run(d)
    assert any("unprovable" in e for e in errs), errs


def test_sabotage_n_train_not_the_requested_count(setup_pp, tmp_path):
    d = _build_qss(setup_pp, str(tmp_path), bad_n_train=723)
    errs = CHK.check_run(d)
    assert any("n_train !=" in e for e in errs), errs


def test_sabotage_full_arm_rerun_with_sampling(setup_pp, tmp_path):
    """The 100% arm is the REUSED QWU cell. A 723-sample cell wearing a
    qss tag would silently substitute a new code path for the reference."""
    d = _build_qss(setup_pp, str(tmp_path), count=723, rounds=100)
    errs = CHK.check_run(d)
    assert any("FULL-DATA arm" in e for e in errs), errs


def test_sabotage_tiling_hidden_from_the_tag(setup_pp, tmp_path):
    """A compute-matched run whose tag omits the rep token would be read
    as the plain 72-agent arm."""
    d = _build_qss(setup_pp, str(tmp_path), count=72, repeat_to=723)
    import shutil
    d2 = d.replace("_n72rep723_", "_n72_")
    shutil.move(d, d2)
    errs = CHK.check_run(d2)
    assert any("must be visible in the tag" in e
               or "sft_sample_idx_raw shape" in e for e in errs), errs


# =================================================== startup-order safety
def _main_fn():
    """main()'s AST from the runner, parsed without importing it (the
    runner's top-level transformers import hangs on some machines)."""
    import ast
    src = open(os.path.join(
        REPO, "experiments/scripts/cluster_pipelines/"
              "run_pokec_gated_lm.py")).read()
    tree = ast.parse(src)
    return next(nd for nd in tree.body
                if isinstance(nd, ast.FunctionDef) and nd.name == "main")


def test_sampling_guard_reads_no_name_before_it_is_bound():
    """REGRESSION (2026-08-21). The SFT_SAMPLE_N validation block was
    placed next to its env read, but it inspects sft_exclude_clamped,
    which is bound several hundred lines later -- so every sampled run
    died at startup with UnboundLocalError. The smoke caught it; nothing
    in the unit tests did, because they build fixtures directly and never
    execute main().

    This walks main()'s AST and asserts that every local the guard reads
    is assigned STRICTLY EARLIER in the function. It is deliberately
    about the guard's dependencies rather than the whole function, so it
    stays cheap and does not fire on unrelated forward references."""
    import ast
    fn = _main_fn()
    # first assignment line for every local name in main()
    first_assign = {}
    for nd in ast.walk(fn):
        if isinstance(nd, ast.Assign):
            for tgt in nd.targets:
                for sub in ast.walk(tgt):
                    if isinstance(sub, ast.Name):
                        first_assign.setdefault(sub.id, nd.lineno)
        elif isinstance(nd, (ast.For, ast.comprehension)):
            pass
    # locate the guard: the raise carrying its message
    guard_line = None
    for nd in ast.walk(fn):
        if isinstance(nd, ast.Raise):
            txt = ast.dump(nd)
            if "SFT_SAMPLE_N is incompatible with" in txt:
                guard_line = nd.lineno
    assert guard_line is not None, "guard not found -- did it move?"
    DEPS = ["sft_exclude_clamped", "replay_frac", "pristine_frac",
            "data_regime", "train_cap", "n_labeled", "sft_sample_n",
            "training_style"]
    late = [(dep, first_assign.get(dep)) for dep in DEPS
            if first_assign.get(dep) is None
            or first_assign[dep] >= guard_line]
    assert not late, (
        f"the SFT_SAMPLE_N guard at line {guard_line} reads name(s) bound "
        f"at or after it: {late} -- this is the UnboundLocalError the "
        f"smoke hit on 2026-08-21")


def test_sampling_block_runs_for_every_round_not_just_after_zero():
    """The second half of the same lesson: the guard must not be the only
    thing that moved. The sampling BLOCK itself has to stay outside the
    t>0 branch, or round 0 is unsubsampled again."""
    src = open(os.path.join(
        REPO, "experiments/scripts/cluster_pipelines/"
              "run_pokec_gated_lm.py")).read()
    blk = src.index("if sft_sample_n > 0 and train_data is not None:")
    # it sits after the whole if/elif/else that selects train_data, so it
    # applies to the t==0 branch too
    assert src.index("train_data = initial_data") < blk
    assert src.index("train_data = select_train_data(") < blk
