"""Tests for the CORRECTED continual-adaptation control (2026-08-29):
fresh vs carried LoRA weights under anch2, lambda in {0,2} x 3 seeds,
on the NUMERIC-gate fec surface -- 12 new jobs + 1 smoke.

The load-bearing test is test_gate_is_never_all_open: under all_open the
AI gate returns all-ones before the gate reference is read, so anch2 and
the legacy operator are numerically identical and the whole control
would be a tautology.

  USE_TF=0 python -m pytest tests/test_continual_anch2.py -q
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
                       "check_continual_anch2.py")
SUBMIT = os.path.join(CONDOR, "submit_pofd_sweep.sh")
sys.path.insert(0, CONDOR)

N = 723
ROUNDS = 30
SMOKE_ROUNDS = 3
LAMS = (0.0, 2.0)
SEEDS = (0, 42, 43)
ARMS = {"adfresh": 1, "adcont": 0}
CFG = os.path.join(CONDOR, "configs_pofd_continual_anch2.txt")
SMOKE_CFG = os.path.join(CONDOR, "configs_pofd_continual_anch2_smoke.txt")
SUB = os.path.join(CONDOR, "at_pofd_continual_anch2.sub")
SMOKE_SUB = os.path.join(CONDOR, "at_pofd_continual_anch2_smoke.sub")


def rows(p):
    with open(p) as fh:
        return [l.strip() for l in fh if l.strip()]


def cols(r):
    return [c.strip() for c in r.split(",")]


# ------------------------------------------------------- generator
def test_exactly_twelve_jobs_and_one_smoke():
    assert len(rows(CFG)) == 12
    assert len(rows(SMOKE_CFG)) == 1


def test_grid_is_two_lambdas_x_two_arms_x_three_seeds():
    seen = set()
    for r in rows(CFG):
        c = cols(r)
        lam, seed, fresh = float(c[2]), int(c[3]), c[20]
        arm = "adfresh" if fresh == "1" else "adcont"
        assert lam in LAMS and seed in SEEDS
        assert c[1] == ("sft" if lam == 0.0 else "sft_kl")
        seen.add((lam, arm, seed))
    assert seen == {(l, a, s) for l in LAMS for a in ARMS for s in SEEDS}


def test_arm_token_matches_the_fresh_column():
    """If these disagree a continual cell is filed as a fresh one and the
    contrast is silently backwards."""
    for r in rows(CFG):
        c = cols(r)
        arm = "adfresh" if c[20] == "1" else "adcont"
        assert f"_{arm}_" in c[0], (arm, c[0])


def test_surface_is_the_fec_one():
    for r in rows(CFG):
        c = cols(r)
        assert c[9] == "0.2"     # eps_social
        assert c[10] == "0.0"    # homophily gamma
        assert c[11] == "0.5"    # W
        assert c[14] == "0.2"    # k
        assert c[15] == "forward"
        assert int(c[16]) == 1   # S = 1: dispersion is preserved
        assert int(c[17]) == 0   # ICL_K
        assert c[19] == "1"      # LoRA on
        assert c[23] == str(ROUNDS)


def test_gate_is_never_all_open():
    """THE load-bearing check. _gated_pop.ai_gate returns an all-ones
    mask under all_open BEFORE reading the gate reference, so anch2 and
    the legacy operator are numerically identical there and a
    'corrected' control would prove nothing."""
    for p in (SUB, SMOKE_SUB):
        env = next(l for l in open(p) if l.startswith("environment"))
        assert "AI_GATE_MODE=threshold" in env
        assert "all_open" not in env
        assert "AI_GATE_REFERENCE=anchor" in env
        assert "EPS_AI=0.4" in env


def test_all_open_really_short_circuits():
    """Guards the premise above against a change in _gated_pop."""
    sys.path.insert(0, os.path.join(REPO, "experiments", "scripts",
                                    "cluster_pipelines"))
    import _gated_pop as gp
    served = torch.rand(16)
    x0 = torch.rand(16)
    assert bool(gp.ai_gate(served, x0, 0.4, mode="all_open").all())
    assert not bool(gp.ai_gate(served, x0, 0.0, mode="threshold").any())


def test_both_arms_ride_one_sub_via_a_queue_column():
    env = next(l for l in open(SUB) if l.startswith("environment"))
    assert "FRESH_EACH_ROUND=$(fresh)" in env


def test_raw_generations_are_saved():
    """The archived fec wave omitted SAVE_RAW_GEN, which is why its
    malformed rate could only be bounded, not counted."""
    for p in (SUB, SMOKE_SUB):
        assert "SAVE_RAW_GEN=1" in open(p).read()


def test_smoke_is_the_continual_kl_cell():
    c = cols(rows(SMOKE_CFG)[0])
    assert c[0].startswith("pofdcacsmk_") and "_adcont_" in c[0]
    assert float(c[2]) == 2.0 and c[20] == "0"
    assert c[23] == str(SMOKE_ROUNDS)


def test_tags_cannot_collide_with_the_archived_fec_wave():
    new = {cols(r)[0] for r in rows(CFG) + rows(SMOKE_CFG)}
    for f in ("configs_pofd_qwen7b_fesc.txt",
              "configs_pofd_qwen7b_fec_smoke.txt"):
        old = {cols(r)[0] for r in rows(os.path.join(CONDOR, f))}
        assert not (new & old)
    assert all(t.startswith("pofdcac") for t in new)


@pytest.mark.parametrize("key", ["continual_anch2", "continual_anch2_smoke"])
def test_submit_case_resolves(key):
    case = "\n".join(l for l in open(SUBMIT).read().splitlines()
                     if l.strip().startswith("continual_anch2")
                     and "TARGETS=" in l)
    out = subprocess.run(
        ["bash", "-c", f'WHAT={key}; case "$WHAT" in\n{case}\n'
                       ' *) exit 1;; esac; printf %s "$TARGETS"'],
        capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout == key


def test_submit_usage_lines_have_no_stray_brace():
    lines = open(SUBMIT).read().splitlines()
    for i in (16, 17):
        assert lines[i].count("}") == 1


# --------------------------------------------------------- checker
def _cell(root, lam, arm, seed, rounds=ROUNDS, smoke=False, **over):
    import gen_pofd_sweep as g
    tag = g.cac_tag(lam, arm, seed, rounds, smoke)
    d = os.path.join(root, tag)
    os.makedirs(d, exist_ok=True)
    gen = torch.Generator().manual_seed(seed + 5)
    innate = torch.rand(N, generator=gen)
    op = torch.stack([(innate + .001 * (t + 1)).clamp(0, 1)
                      for t in range(rounds)])
    pred = torch.stack([torch.full((N,), 0.50 + .01 * t)
                        for t in range(rounds)])
    cfg = {"kl_beta": lam, "seed": seed, "n_rounds": rounds,
           "w_plat": 0.5, "innate_lambda": 0.2, "eps": 0.2, "eps_ai": 0.4,
           "ab_sweeps": 1, "lora_r": 512, "train_cap": N, "n_labeled": N,
           "data_regime": "replace", "dataset": "movielens",
           "ml_target": "Action", "pristine_frac": 0.0,
           "training_style": "sft" if lam == 0 else "sft_kl",
           "kl_direction": "forward", "gamma_bias": 0.0,
           "fresh_each_round": bool(ARMS[arm]),
           "ai_gate_mode": "threshold", "ai_gate_reference": "anchor",
           "base_model": "Qwen/Qwen2.5-7B-Instruct", "sft_lr": 5e-5,
           "sft_epochs": 1, "sft_batch_size": 4, "icl_k": 0,
           "icl_days": 0, "do_sample": False, "seed_base_data": True,
           "population_update": "nested_ai_anchored_then_social_v2"}
    cfg.update(over.pop("cfg", {}))
    torch.save({"config": cfg, "innate": innate, "op_raw": op,
                "pred_raw": pred, "twin_raw": op.clone()},
               os.path.join(d, "trajectory.pt"))
    tel = over.pop("tel", None)
    with open(os.path.join(d, "telemetry.json"), "w") as fh:
        for t in range(rounds):
            row = {"round": t, "l_init": 1.0, "n_train": N, "w_norm": 100.0,
                   "b_norm": 1.5, "ba_norm": 0.9,
                   "grad_kl_norm0": (0.0 if lam == 0 else 0.8)}
            if tel:
                row.update(tel(t) or {})
            fh.write(json.dumps(row) + "\n")
    with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt") as fh:
        for t in range(rounds):
            vals = pred[t].tolist()
            fh.write(json.dumps({
                "round": t, "parse_fail_frac": over.get("pff", 0.0),
                "raw": over.get("raw") or [f"{v:.2f}" for v in vals],
                "parsed": vals}) + "\n")
    return d


def _run(root, smoke=False):
    cmd = [sys.executable, CHECKER, "--run-root", root]
    if smoke:
        cmd.append("--smoke")
    return subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, "USE_TF": "0"})


@pytest.fixture
def healthy(tmp_path):
    root = str(tmp_path)
    for lam in LAMS:
        for arm in ARMS:
            for s in SEEDS:
                _cell(root, lam, arm, s)
    return root


def test_healthy_wave_passes(healthy):
    r = _run(healthy)
    assert r.returncode == 0, r.stdout


def test_all_open_cell_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 2.0, "adcont", 0, rounds=SMOKE_ROUNDS, smoke=True,
          cfg={"ai_gate_mode": "all_open"})
    assert "all_open" in _run(root, smoke=True).stdout


def test_legacy_operator_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 2.0, "adcont", 0, rounds=SMOKE_ROUNDS, smoke=True,
          cfg={"population_update": "nested_ai_then_social_v1"})
    assert "population_update" in _run(root, smoke=True).stdout


def test_swapped_arm_flag_fails(tmp_path):
    """A continual cell recording fresh_each_round=True would invert the
    contrast."""
    root = str(tmp_path)
    _cell(root, 2.0, "adcont", 0, rounds=SMOKE_ROUNDS, smoke=True,
          cfg={"fresh_each_round": True})
    assert "arms would be swapped" in _run(root, smoke=True).stdout


def test_pairing_catches_a_second_differing_dial(healthy):
    import gen_pofd_sweep as g
    p = os.path.join(healthy, g.cac_tag(2.0, "adcont", 42, ROUNDS),
                     "trajectory.pt")
    d = torch.load(p, weights_only=False)
    d["config"]["sft_lr"] = 1e-4
    torch.save(d, p)
    out = _run(healthy).stdout
    assert "differ on" in out and "sft_lr" in out


def test_kl_leak_at_lambda0_fails(tmp_path):
    root = str(tmp_path)
    # full-length: the non-smoke checker only looks for _r30 tags
    for lam in LAMS:
        for arm in ARMS:
            for sd in SEEDS:
                _cell(root, lam, arm, sd)
    import gen_pofd_sweep as g
    d = os.path.join(root, g.cac_tag(0.0, "adcont", 0, ROUNDS))
    lines = [json.loads(l) for l in open(os.path.join(d, "telemetry.json"))]
    for r in lines:
        r["grad_kl_norm0"] = 0.4
    open(os.path.join(d, "telemetry.json"), "w").write(
        "\n".join(json.dumps(r) for r in lines) + "\n")
    r = subprocess.run([sys.executable, CHECKER, "--run-root", root],
                       capture_output=True, text=True,
                       env={**os.environ, "USE_TF": "0"})
    assert "carries no KL" in r.stdout


def test_dead_adapter_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 2.0, "adcont", 0, rounds=SMOKE_ROUNDS, smoke=True,
          tel=lambda t: {"b_norm": 0.0, "ba_norm": 0.0})
    assert "identically zero" in _run(root, smoke=True).stdout


def test_missing_raw_log_fails(tmp_path):
    root = str(tmp_path)
    d = _cell(root, 2.0, "adcont", 0, rounds=SMOKE_ROUNDS, smoke=True)
    os.remove(os.path.join(d, "raw_gen_log.json.gz"))
    assert "ABSENT" in _run(root, smoke=True).stdout


def test_malformed_generation_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 2.0, "adcont", 0, rounds=SMOKE_ROUNDS, smoke=True,
          raw=["0.5 (approx"] * N)
    assert "malformed" in _run(root, smoke=True).stdout


def test_short_training_set_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 2.0, "adcont", 0, rounds=SMOKE_ROUNDS, smoke=True,
          tel=lambda t: {"n_train": 700} if t == 1 else None)
    assert "live labels" in _run(root, smoke=True).stdout
