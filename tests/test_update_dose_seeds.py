"""Tests for the UPDATE-DOSE SEED REPLICATION (2026-08-28, user):
seeds 42 and 43 at U in {1, 5, 181}, ten rounds, everything else
byte-identical to the completed seed-0 cells -- exactly six new jobs.

The generator tests read the EMITTED artifacts on disk (the config file
and .sub Condor actually consumes) and restate the grid independently:
a test that reads the grid from the thing under test cannot catch a
grid bug.  The checker tests synthesize a physically consistent
pofdud_ cell and assert the --seeds gate PASSES it and FAILS each
sabotage.

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac):
  USE_TF=0 python -m pytest tests/test_update_dose_seeds.py -q
"""
import gzip
import json
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
CONDOR = os.path.join(REPO, "experiments", "condor")
CHECKER = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines",
                       "check_update_dose.py")
SUBMIT = os.path.join(CONDOR, "submit_pofd_sweep.sh")

# ---------------------------------------------- contract (restated)
N = 723
ROUNDS = 10
SEEDS = (42, 43)
US = {1: 181, 5: 37, 181: 1}          # realized steps -> grad accum
KEY = "sft_update_dose_loop_seeds"
CFG = os.path.join(CONDOR, f"configs_pofd_{KEY}.txt")
SUB = os.path.join(CONDOR, f"at_pofd_{KEY}.sub")
SEED0_CFG = os.path.join(CONDOR, "configs_pofd_sft_update_dose_loop.txt")
SEED0_SUB = os.path.join(CONDOR, "at_pofd_sft_update_dose_loop.sub")


def rows(path):
    with open(path) as fh:
        return [l.strip() for l in fh if l.strip()]


def cols(row):
    return [c.strip() for c in row.split(",")]


# ------------------------------------------------------- generator
def test_exactly_six_jobs():
    assert len(rows(CFG)) == 6


def test_grid_is_exactly_three_doses_by_two_seeds():
    seen = set()
    for r in rows(CFG):
        c = cols(r)
        tag, seed, accum, nrounds = c[0], int(c[3]), int(c[17]), c[24]
        u = int(tag.split("_u")[1].split("_")[0])
        assert (u, accum) in US.items()
        assert -(-181 // accum) == u, "tag must promise the realized steps"
        assert nrounds == str(ROUNDS)
        assert seed in SEEDS
        assert tag.endswith(f"_s{seed}_r{ROUNDS}")
        seen.add((u, seed))
    assert seen == {(u, s) for u in US for s in SEEDS}


def test_u19_is_not_replicated():
    """The reviewer named U in {1,5,181}; widening the ladder would turn
    a replication into a new grid."""
    assert not any("_u19_" in cols(r)[0] for r in rows(CFG))


def test_tags_cannot_collide_with_the_completed_seed0_wave():
    new = {cols(r)[0] for r in rows(CFG)}
    old = {cols(r)[0] for r in rows(SEED0_CFG)}
    assert not (new & old)
    assert all("_s0_" not in t for t in new)


def test_seed_column_matches_the_tag():
    for r in rows(CFG):
        c = cols(r)
        assert f"_s{int(c[3])}_r" in c[0]


def test_settings_other_than_seed_and_accum_match_seed0():
    """Every column except the tag, the seed and the accum dial must be
    byte-identical to the seed-0 rows -- that is what 'replication'
    means here."""
    ref = {int(cols(r)[0].split("_u")[1].split("_")[0]): cols(r)
           for r in rows(SEED0_CFG)}
    for r in rows(CFG):
        c = cols(r)
        u = int(c[0].split("_u")[1].split("_")[0])
        if u not in ref:            # u181 has no seed-0 pofdud_ twin
            continue
        for i, (a, b) in enumerate(zip(c, ref[u])):
            if i in (0, 3, 17):     # tag, seed, accum
                continue
            assert a == b, f"column {i} differs at u{u}: {a!r} vs {b!r}"


def test_sub_reuses_the_seed0_environment_verbatim():
    def env(p):
        return next(l for l in open(p) if l.startswith("environment"))
    assert env(SUB) == env(SEED0_SUB)


def test_sub_never_sets_the_step_cap():
    """SFT_MAX_STEPS cuts data exposure; this wave holds it fixed."""
    e = next(l for l in open(SUB) if l.startswith("environment"))
    assert "SFT_MAX_STEPS" not in e
    assert "SFT_GRAD_ACCUM=$(accum)" in e
    assert "SAVE_SFT_ORDER=1" in e and "SFT_EPOCHS=1" in e
    assert "AI_GATE_REFERENCE=anchor" in e


def test_sub_queues_the_right_file_and_gate():
    s = open(SUB).read()
    assert f"configs_pofd_{KEY}.txt" in s
    assert "check_update_dose.py --seeds 42,43" in s
    assert f"submit_pofd_sweep.sh <BID> {KEY}" in s


def test_sub_does_not_claim_u181_is_unqueued():
    assert "u181 (accum 1) IS queued here" in open(SUB).read()
    assert "u181 (accum 1) is NOT queued" in open(SEED0_SUB).read()


# --------------------------------------------------------- submit
def test_submit_usage_lines_have_no_stray_brace():
    """A literal } inside ${1:?...} closes the expansion early and
    silently mangles EVERY key (the 2026-08-27 bug)."""
    lines = open(SUBMIT).read().splitlines()
    for i in (16, 17):                       # 0-indexed lines 17, 18
        assert lines[i].count("}") == 1, lines[i][:80]


@pytest.mark.parametrize("key", ["sft_update_dose_loop",
                                 "sft_update_dose_loop_smoke",
                                 "sft_update_dose_loop_seeds"])
def test_submit_case_resolves_every_ud_key(key):
    case = "\n".join(l for l in open(SUBMIT).read().splitlines()
                     if "sft_update_dose_loop" in l and "TARGETS=" in l)
    out = subprocess.run(
        ["bash", "-c", f'WHAT={key}; case "$WHAT" in\n{case}\n'
                       ' *) exit 1;; esac; printf %s "$TARGETS"'],
        capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout == key


# --------------------------------------------------------- checker
def _cell(root, u, accum, seed, rounds=ROUNDS, **over):
    """A physically consistent pofdud_ cell: 723 live labels consumed
    once per round in a recorded permutation, u optimizer steps."""
    tag = (f"pofdud_qwen3_8b_sft_u{u}_sw100_eaopen_w1_k1_esopen_anch2"
           f"_s{seed}_r{rounds}")
    d = os.path.join(root, tag)
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(seed + 7)
    innate = torch.rand(N, generator=g)
    op = torch.stack([(innate + 0.001 * (t + 1)).clamp(0, 1)
                      for t in range(rounds)])
    pred = torch.stack([(innate * 0.9 + 0.02 * t).clamp(0, 1)
                        for t in range(rounds)])
    idx = torch.stack([torch.randperm(N, generator=g)
                       for _ in range(rounds)])
    y = torch.stack([(innate if t == 0 else op[t - 1])[idx[t]]
                     for t in range(rounds)])
    cfg = {"w_plat": 1.0, "innate_lambda": 1.0, "kl_beta": 0.0,
           "ab_sweeps": 100, "n_rounds": rounds, "seed": seed,
           "training_style": "sft", "run_mode": "loop",
           "data_regime": "replace", "train_cap": 723,
           "save_sft_order": True, "save_raw_gen": True,
           "chat_thinking": False, "do_sample": False,
           "ai_gate_reference": "anchor", "ai_gate_mode": "all_open",
           "peer_gate_mode": "all_open", "sft_lr": 5e-5, "lora_r": 512,
           "sft_batch_size": 4, "sft_epochs": 1,
           "base_model": "Qwen/Qwen3-8B", "sft_grad_accum": accum,
           "population_update": "nested_ai_anchored_then_social_v2"}
    cfg.update(over.pop("cfg", {}))
    dose = [{"round": t, "global_step": over.pop(f"step{t}", u),
             "n_rows": N, "trainer_seed": 42} for t in range(rounds)]
    torch.save({"config": cfg, "innate": innate, "op_raw": op,
                "pred_raw": pred, "sft_order_idx_raw": idx,
                "sft_order_y_raw": y, "sft_dose": dose,
                **over}, os.path.join(d, "trajectory.pt"))
    parsed = over.get("parsed_override")
    with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt") as fh:
        for t in range(rounds):
            vals = (parsed[t] if parsed is not None
                    else pred[t].tolist())
            fh.write(json.dumps({
                "round": t, "parse_fail_frac": over.get("pff", 0.0),
                "raw": [f"{v:.2f}" for v in vals],
                "parsed": list(vals)}) + "\n")
    return d


def _run(root, seeds="42,43"):
    return subprocess.run(
        [sys.executable, CHECKER, "--run-root", root, "--seeds", seeds],
        capture_output=True, text=True,
        env={**os.environ, "USE_TF": "0"})


@pytest.fixture
def healthy(tmp_path):
    root = str(tmp_path)
    for s in SEEDS:
        for u, acc in US.items():
            _cell(root, u, acc, s)
    return root


def test_healthy_replication_passes(healthy):
    r = _run(healthy)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("PASS") >= 6


def test_absent_cell_fails(healthy):
    import shutil
    shutil.rmtree(os.path.join(
        healthy, "pofdud_qwen3_8b_sft_u181_sw100_eaopen_w1_k1_esopen"
                 "_anch2_s43_r10"))
    assert _run(healthy).returncode == 1


def test_wrong_step_count_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 1, 181, 42, step3=7)
    assert "global_step" in _run(root, "42").stdout


def test_accum_one_is_gated_not_skipped(tmp_path):
    """u181 runs at accum 1; the old checker skipped the config gate
    whenever accum == 1, so a mislabelled cell passed silently."""
    root = str(tmp_path)
    _cell(root, 181, 1, 42, cfg={"sft_grad_accum": 4})
    assert "sft_grad_accum" in _run(root, "42").stdout


def test_absent_accum_key_means_one(tmp_path):
    """run_pokec_gated_lm writes sft_grad_accum only when it exceeds 1,
    so at accum 1 the key is legitimately ABSENT and must read as 1.
    The real seed-42/43 u181 cells have no such key."""
    root = str(tmp_path)
    d = _cell(root, 181, 1, 42)
    p = os.path.join(d, "trajectory.pt")
    t = torch.load(p, weights_only=False)
    t["config"].pop("sft_grad_accum")
    torch.save(t, p)
    r = _run(root, "42")
    assert "sft_grad_accum" not in r.stdout, r.stdout


def test_absent_accum_key_still_fails_when_more_than_one_expected(tmp_path):
    """Absence means 1, so a u5 cell (accum 37) with no key must FAIL --
    the default must not become a blanket escape hatch."""
    root = str(tmp_path)
    d = _cell(root, 5, 37, 42)
    p = os.path.join(d, "trajectory.pt")
    t = torch.load(p, weights_only=False)
    t["config"].pop("sft_grad_accum")
    torch.save(t, p)
    assert "sft_grad_accum" in _run(root, "42").stdout


def test_seed_gate_uses_the_cell_seed(tmp_path):
    root = str(tmp_path)
    _cell(root, 1, 181, 42, cfg={"seed": 0})
    assert "seed" in _run(root, "42").stdout


def test_uncorrected_operator_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 1, 181, 42,
          cfg={"population_update": "nested_ai_then_social_v1"})
    assert "population_update" in _run(root, "42").stdout


def test_replayed_labels_fail(tmp_path):
    """Labels must be the LIVE previous post-peer opinions."""
    root = str(tmp_path)
    d = _cell(root, 1, 181, 42)
    p = os.path.join(d, "trajectory.pt")
    t = torch.load(p, weights_only=False)
    t["sft_order_y_raw"][5] = t["innate"][t["sft_order_idx_raw"][5]]
    torch.save(t, p)
    assert "ORDER" in _run(root, "42").stdout


def test_short_exposure_fails(tmp_path):
    root = str(tmp_path)
    d = _cell(root, 1, 181, 42)
    p = os.path.join(d, "trajectory.pt")
    t = torch.load(p, weights_only=False)
    t["sft_dose"][2]["n_rows"] = 700
    torch.save(t, p)
    assert "n_rows" in _run(root, "42").stdout


def test_missing_raw_log_fails(tmp_path):
    """Absence is a FAILURE: the parser stores a finite 0.5, so an
    absent log means the malformed rate is unknowable."""
    root = str(tmp_path)
    d = _cell(root, 1, 181, 42)
    os.remove(os.path.join(d, "raw_gen_log.json.gz"))
    assert "ABSENT" in _run(root, "42").stdout


def test_nonzero_parse_fail_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 1, 181, 42, pff=0.01)
    assert "parse_fail_frac" in _run(root, "42").stdout


def test_parsed_must_equal_pred_raw(tmp_path):
    """A well-formed log that describes a DIFFERENT trajectory must not
    pass: parsed[t] has to equal pred_raw[t] agent by agent."""
    root = str(tmp_path)
    d = _cell(root, 1, 181, 42)
    p = os.path.join(d, "trajectory.pt")
    t = torch.load(p, weights_only=False)
    bad = [t["pred_raw"][i].tolist() for i in range(ROUNDS)]
    bad[4][17] = float(min(1.0, bad[4][17] + 0.3))
    torch.save(t, p)
    with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt") as fh:
        for i in range(ROUNDS):
            fh.write(json.dumps({"round": i, "parse_fail_frac": 0.0,
                                 "raw": [f"{v:.2f}" for v in bad[i]],
                                 "parsed": bad[i]}) + "\n")
    assert "PARSE-VS-SERVED" in _run(root, "42").stdout


def test_split_sampler_order_within_a_seed_fails(tmp_path):
    root = str(tmp_path)
    for u, acc in US.items():
        d = _cell(root, u, acc, 42)
        if u == 5:
            p = os.path.join(d, "trajectory.pt")
            t = torch.load(p, weights_only=False)
            for rec in t["sft_dose"]:
                rec["trainer_seed"] = 999
            torch.save(t, p)
    assert "cross-arm sampler order" in _run(root, "42").stdout


def test_seed0_wave_still_passes_unchanged():
    """The tightened config gates must not break the completed wave."""
    root = os.path.join(REPO, "notes", "pofd", "cluster")
    if not os.path.isdir(os.path.join(
            root, "pofdud_qwen3_8b_sft_u1_sw100_eaopen_w1_k1_esopen"
                  "_anch2_s0_r10")):
        pytest.skip("seed-0 update-dose runs not cached locally")
    r = subprocess.run([sys.executable, CHECKER, "--run-root", root],
                       capture_output=True, text=True,
                       env={**os.environ, "USE_TF": "0"})
    assert r.returncode == 0, r.stdout + r.stderr
