"""Tests for the forward-vs-reverse KL wave (pofdkd_, 2026-08-22).

The generator tests pin the things that would silently invert the
comparison: a direction that does not ride the queue, a tag whose
direction token disagrees with the column, or the reused forward
lambda=1 cells being queued a second time.

The checker tests pin the one failure the artifacts cannot reveal on
their own -- a run tagged "rev" that trained forward.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(HERE, "experiments", "condor", "gen_pofd_sweep.py")
CHECK = os.path.join(HERE, "experiments", "scripts", "cluster_pipelines",
                     "check_kl_direction.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load(GEN, "_gen_pofd_kd")


# --------------------------------------------------------------- generator
def test_ten_production_cells_four_forward_six_reverse(gen):
    rows = gen.kd_rows()
    assert len(rows) == 10
    tags = [r.split(",")[0] for r in rows]
    assert sum("_fwd" in t for t in tags) == 4
    assert sum("_rev" in t for t in tags) == 6


def test_forward_lambda1_is_reused_never_queued(gen):
    """forward lambda=1 at both W is the archived QWU b1 pair. Queueing it
    again would burn two GPU jobs AND create a second, differently-seeded
    answer to a question that already has one."""
    tags = [r.split(",")[0] for r in gen.kd_rows()]
    assert not any("_fwdlam1_" in t for t in tags), tags
    assert set(gen.KD_REUSED) == {
        ("sft0", 0.5), ("sft0", 1.0), ("fwdlam1", 0.5), ("fwdlam1", 1.0)}


def test_direction_column_matches_tag_token_in_every_row(gen):
    for r in gen.kd_rows():
        c = [x.strip() for x in r.split(",")]
        tag, kldir = c[0], c[15]
        assert kldir in ("forward", "reverse"), r
        assert ("_fwd" in tag) == (kldir == "forward"), r
        assert ("_rev" in tag) == (kldir == "reverse"), r


def test_direction_rides_the_queue_and_is_declared(gen):
    """The env may only reference $(kldir) if the queue line actually
    supplies it. A sub that reads $(kldir) without queueing the column
    expands it to the empty string, and the runner then falls back to its
    'reverse' default -- on the FORWARD cells too, which would look like
    a real result."""
    sub = gen.kd_sub()
    env = next(l for l in sub.splitlines() if l.startswith("environment"))
    assert "KL_DIRECTION=$(kldir)" in env
    assert "KL_DIRECTION=forward" not in env
    assert "KL_DIRECTION=reverse" not in env
    q = next(l for l in sub.splitlines() if l.startswith("queue "))
    assert ", lam, kldir, iclk," in q


def test_lambda_token_cannot_be_read_as_the_w_plat_arm_token(gen):
    """This project spells forward-KL arms b0/b1/b2/b8. If the KL weight
    were spelled the same way, _b1_ in a pofdkd_ tag would be ambiguous
    against the w_plat token. It is spelled lam<x>."""
    for r in gen.kd_rows():
        tag = r.split(",")[0]
        assert "lam" in tag
        assert "_b0_" not in tag and "_b1_" not in tag


def test_homophily_gamma_stays_zero_and_k_is_one(gen):
    """Celestine's gamma is the INNATE anchor k, not the homophily gamma.
    The homophily column stays 0 as it does in every pofd sim."""
    for r in gen.kd_rows():
        c = [x.strip() for x in r.split(",")]
        assert c[10] == "0.0", r      # homophily gamma
        assert c[14] == "1", r        # k = INNATE_LAMBDA


def test_both_w_columns_present(gen):
    ws = {c.split(",")[11].strip() for c in gen.kd_rows()}
    assert ws == {"0.5", "1"}


def test_smoke_is_reverse_lambda1_w1_and_cannot_shadow_production(gen):
    rows = gen.kd_smoke_rows()
    assert len(rows) == 1
    c = [x.strip() for x in rows[0].split(",")]
    assert c[15] == "reverse" and c[2] == "1" and c[11] == "1"
    assert c[22] == str(gen.KD_SMOKE_ROUNDS)
    assert c[0].startswith("pofdkdsmk_")
    prod = {r.split(",")[0] for r in gen.kd_rows()}
    assert c[0] not in prod


def test_generator_writes_both_keys():
    r = subprocess.run([sys.executable, GEN], capture_output=True, text=True,
                       env={**os.environ, "USE_TF": "0"})
    assert r.returncode == 0, r.stderr[-2000:]
    d = os.path.dirname(GEN)
    for f in ("configs_pofd_qwen_kl_direction.txt",
              "at_pofd_qwen_kl_direction.sub",
              "configs_pofd_qwen_kl_direction_smoke.txt",
              "at_pofd_qwen_kl_direction_smoke.sub"):
        assert os.path.exists(os.path.join(d, f)), f


# ----------------------------------------------------------------- checker
def _mk_run(tmp, tag, *, kl_direction, rounds=10, n=723, lam=1.0,
            parse_fail=0.0, const_agents=False, const_rounds=False):
    d = os.path.join(tmp, tag)
    os.makedirs(d, exist_ok=True)
    cfg = {
        "kl_direction": kl_direction, "training_style": "sft_kl",
        "kl_beta": lam, "kl_ref_adapter": "", "anchor_mode": "fixed",
        "base_model": "Qwen/Qwen2.5-7B-Instruct", "seed": 0,
        "dataset": "movielens", "ml_target": "Action", "n_labeled": n,
        "ai_gate_mode": "all_open", "peer_gate_mode": "all_open",
        "ab_sweeps": 1, "pop_model": "ab", "icl_k": 0, "icl_days": 0,
        "lora_r": 512, "sft_epochs": 1, "epoch_size": 100, "train_cap": n,
        "eps": 0.2, "innate_lambda": 1.0, "w_plat": 1.0, "sft_lr": 5e-05,
        "use_lora": True, "fresh_each_round": True, "n_rounds": rounds,
        "serve_eval_mode": True,
        "hardware": {"gpu_name": "NVIDIA H100 80GB HBM3"},
    }
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    g = torch.Generator().manual_seed(0)
    if const_agents:
        pr = torch.full((rounds, n), 0.5)
    elif const_rounds:
        row = torch.rand(n, generator=g)
        pr = row.unsqueeze(0).repeat(rounds, 1)
    else:
        pr = torch.rand(rounds, n, generator=g)
    torch.save({
        "pred_raw": pr, "op_raw": torch.rand(rounds, n, generator=g),
        "innate": torch.rand(n, generator=g),
        "trajectory": [{"round": i} for i in range(rounds)],
    }, os.path.join(d, "trajectory.pt"))
    # telemetry.json is JSONL: l_init and the anchor-gradient norm live
    # here, not in trajectory.pt. grad_kl_norm0 is ~0 at round 0 by
    # construction (a fresh LoRA IS the reference) and nonzero after.
    with open(os.path.join(d, "telemetry.json"), "w") as fh:
        for i in range(rounds):
            fh.write(json.dumps({
                "round": i, "l_init": 2.0 / (i + 1), "grad_norm0": 3.0,
                "grad_kl_norm0": (0.02 if i == 0 else 1.5),
            }) + "\n")
    # parse_fail_frac lives in the gzipped raw-generation log
    import gzip
    with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt") as fh:
        for i in range(rounds):
            fh.write(json.dumps({
                "round": i, "parse_fail_frac": parse_fail,
                "raw": ["0.25"] * n, "parsed": [0.25] * n,
            }) + "\n")
    return d


def _run_check(dirs, smoke=False):
    cmd = [sys.executable, CHECK] + (["--smoke"] if smoke else []) + dirs
    return subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, "USE_TF": "0"})


def test_checker_catches_direction_mismatch(tmp_path):
    """The headline failure: a cell tagged reverse that trained forward.
    Nothing downstream would reveal it."""
    d = _mk_run(str(tmp_path),
                "pofdkd_qwen7b_revlam1_eaopen_w1_l1_esopen_s0_r10",
                kl_direction="forward")
    r = _run_check([d])
    assert r.returncode == 1
    assert "DIRECTION MISMATCH" in r.stdout


def test_checker_rejects_a_half_wave_with_one_direction(tmp_path):
    ds = [_mk_run(str(tmp_path),
                  f"pofdkd_qwen7b_revlam{t}_eaopen_w{w}_l1_esopen_s0_r10",
                  kl_direction="reverse")
          for t, w in (("0p1", "1"), ("1", "1"))]
    r = _run_check(ds)
    assert r.returncode == 1
    assert "directions present" in r.stdout


def test_checker_rejects_parse_failures(tmp_path):
    d = _mk_run(str(tmp_path),
                "pofdkd_qwen7b_revlam1_eaopen_w1_l1_esopen_s0_r10",
                kl_direction="reverse", parse_fail=0.004)
    r = _run_check([d])
    assert r.returncode == 1
    assert "parse failures" in r.stdout


def test_checker_rejects_a_collapsed_served_map(tmp_path):
    d = _mk_run(str(tmp_path),
                "pofdkd_qwen7b_revlam10_eaopen_w1_l1_esopen_s0_r10",
                kl_direction="reverse", lam=10.0, const_agents=True)
    r = _run_check([d])
    assert r.returncode == 1
    assert "CONSTANT across agents" in r.stdout


def test_checker_rejects_a_trained_arm_that_never_moved(tmp_path):
    """lambda so large the optimizer no-ops looks like retention but is
    the frozen signature."""
    d = _mk_run(str(tmp_path),
                "pofdkd_qwen7b_revlam10_eaopen_w1_l1_esopen_s0_r10",
                kl_direction="reverse", lam=10.0, const_rounds=True)
    r = _run_check([d])
    assert r.returncode == 1
    assert "bit-identical in EVERY round" in r.stdout


def test_checker_rejects_an_arm_whose_anchor_never_contributed(tmp_path):
    """A run can be tagged reverse, record kl_direction=reverse, and still
    have contributed no anchor gradient -- that arm is ordinary SFT
    wearing a lambda, and its "closeness to frozen" would be read as
    retention. Round 0 is exempt: a fresh LoRA IS the reference there."""
    d = _mk_run(str(tmp_path),
                "pofdkd_qwen7b_revlam1_eaopen_w1_l1_esopen_s0_r10",
                kl_direction="reverse")
    with open(os.path.join(d, "telemetry.json"), "w") as fh:
        for i in range(10):
            fh.write(json.dumps({"round": i, "l_init": 1.0,
                                 "grad_norm0": 3.0,
                                 "grad_kl_norm0": 0.0}) + "\n")
    r = _run_check([d])
    assert r.returncode == 1
    assert "contributed no gradient" in r.stdout


def test_checker_accepts_zero_kl_gradient_at_round_zero_only(tmp_path):
    d = _mk_run(str(tmp_path),
                "pofdkdsmk_qwen7b_revlam1_eaopen_w1_l1_esopen_s0_r3",
                kl_direction="reverse", rounds=3)
    r = _run_check([d], smoke=True)
    assert r.returncode == 0, r.stdout


def test_checker_rejects_smoke_tag_as_production(tmp_path):
    d = _mk_run(str(tmp_path),
                "pofdkdsmk_qwen7b_revlam1_eaopen_w1_l1_esopen_s0_r3",
                kl_direction="reverse", rounds=3)
    r = _run_check([d])
    assert r.returncode == 1


def test_checker_passes_a_well_formed_smoke(tmp_path):
    d = _mk_run(str(tmp_path),
                "pofdkdsmk_qwen7b_revlam1_eaopen_w1_l1_esopen_s0_r3",
                kl_direction="reverse", rounds=3)
    r = _run_check([d], smoke=True)
    assert r.returncode == 0, r.stdout
    assert "PASS" in r.stdout
