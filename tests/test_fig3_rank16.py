"""Tests for the Figure-3 LoRA-RANK ROBUSTNESS wave (2026-08-28):
r=16 at lambda in {0, 2}, beta=gamma=1, everything else identical to the
figure -- 2 new jobs plus one 3-round smoke.

The generator tests read the EMITTED artifacts and restate the contract
independently; the critical one is that the 108-cell figure's own config
and sub come out BYTE-IDENTICAL, since the rank wave generalizes that
machinery rather than forking it.

Run with USE_TF=0:
  USE_TF=0 python -m pytest tests/test_fig3_rank16.py -q
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
                       "check_fig3_full_loop.py")
SUBMIT = os.path.join(CONDOR, "submit_pofd_sweep.sh")
sys.path.insert(0, os.path.join(REPO, "experiments", "scripts",
                                "cluster_pipelines"))

N = 723
ROUNDS = 30
SMOKE_ROUNDS = 3
RANK = 16
LAMS = (0.0, 2.0)
CFG = os.path.join(CONDOR, "configs_pofd_fig3_rank16.txt")
SMOKE_CFG = os.path.join(CONDOR, "configs_pofd_fig3_rank16_smoke.txt")
SUB = os.path.join(CONDOR, "at_pofd_fig3_rank16.sub")
SMOKE_SUB = os.path.join(CONDOR, "at_pofd_fig3_rank16_smoke.sub")
F3_SUB = os.path.join(CONDOR, "at_pofd_fig3_full_loop.sub")


def rows(path):
    with open(path) as fh:
        return [l.strip() for l in fh if l.strip()]


def cols(r):
    return [c.strip() for c in r.split(",")]


# --------------------------------------------------------- generator
def test_exactly_two_new_jobs():
    assert len(rows(CFG)) == 2
    assert len(rows(SMOKE_CFG)) == 1


def test_grid_is_r16_x_two_lambdas_at_beta_gamma_one():
    seen = set()
    for r in rows(CFG):
        c = cols(r)
        assert f"_rank{RANK}_" in c[0]
        assert c[11] == "1" and c[14] == "1"        # beta = gamma = 1
        assert c[10] == "0.0"                       # homophily gamma
        assert c[15] == "forward" and int(c[16]) == 100
        assert int(c[17]) == 0                      # ICL_K = 0
        assert c[19] == "1" and c[20] == "1"        # LoRA on, fresh
        assert c[23] == str(ROUNDS) and c[25] == "0"
        lam = float(c[2])
        assert c[1] == ("sft" if lam == 0.0 else "sft_kl")
        seen.add(lam)
    assert seen == set(LAMS)


def test_rank_token_is_not_the_rounds_token():
    """`_r16` already means 16 ROUNDS in this repo; the rank must be
    spelled `_rank16` or a tag becomes ambiguous."""
    for r in rows(CFG) + rows(SMOKE_CFG):
        t = cols(r)[0]
        assert "_rank16_" in t
        assert not t.endswith("_r16")


def test_smoke_is_the_kl_arm_at_three_rounds():
    c = cols(rows(SMOKE_CFG)[0])
    assert c[0].startswith("pofdf3smk_") and c[0].endswith("_r3")
    assert float(c[2]) == 2.0 and c[1] == "sft_kl"
    assert c[23] == str(SMOKE_ROUNDS)


def test_new_tags_cannot_collide_with_the_figure():
    new = {cols(r)[0] for r in rows(CFG)} | {cols(r)[0] for r in rows(SMOKE_CFG)}
    old = {cols(r)[0] for r in rows(
        os.path.join(CONDOR, "configs_pofd_fig3_full_loop.txt"))}
    old |= {cols(r)[0] for r in rows(
        os.path.join(CONDOR, "configs_pofd_fig3_full_loop_smoke.txt"))}
    assert not (new & old)


def test_rank_rows_request_more_memory_than_the_default_tier():
    """The first r=16 lambda=2 smoke was held at 128G having used
    146485 MB -- environment drift, not rank. The rank rows carry their
    own memory request; the figure's rows must be untouched."""
    for r in rows(CFG) + rows(SMOKE_CFG):
        assert cols(r)[26] == "200G", cols(r)[26]
    for r in rows(os.path.join(CONDOR, "configs_pofd_fig3_full_loop.txt")):
        assert cols(r)[26] == "128G"


def test_sub_differs_from_the_figure_only_by_lora_r():
    def env(p):
        return next(l for l in open(p) if l.startswith("environment"))
    assert env(SUB) == env(F3_SUB).replace("LORA_R=512", "LORA_R=16")
    assert "LORA_R=16 " in env(SUB)


def test_lora_alpha_is_never_a_dial():
    """alpha = 2r is set in the runner, so the scaling ratio is constant
    across ranks by construction; an env override would break that."""
    for p in (SUB, SMOKE_SUB, F3_SUB):
        assert "LORA_ALPHA" not in open(p).read()
    src = open(os.path.join(REPO, "experiments", "scripts",
                            "cluster_pipelines",
                            "run_pokec_gated_lm.py")).read()
    assert "lora_alpha=2 * lora_r" in src


def test_figure3_wave_is_untouched_by_this_generalization():
    """The rank wave generalizes the F3 machinery; the figure's own
    emitted artifacts must be byte-identical."""
    out = subprocess.run(["git", "diff", "--stat", "--",
                          "experiments/condor/configs_pofd_fig3_full_loop.txt",
                          "experiments/condor/at_pofd_fig3_full_loop.sub",
                          "experiments/condor/configs_pofd_fig3_full_loop_smoke.txt"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.stdout.strip() == "", out.stdout


def test_gate_command_names_the_rank_wave():
    assert "check_fig3_full_loop.py --wave rank16" in open(SUB).read()
    assert "check_fig3_full_loop.py --wave rank16 --smoke" in \
        open(SMOKE_SUB).read()


@pytest.mark.parametrize("key", ["fig3_rank16", "fig3_rank16_smoke",
                                 "fig3_full_loop"])
def test_submit_case_resolves(key):
    case = "\n".join(l for l in open(SUBMIT).read().splitlines()
                     if l.strip().startswith("fig3_") and "TARGETS=" in l)
    out = subprocess.run(
        ["bash", "-c", f'WHAT={key}; case "$WHAT" in\n{case}\n'
                       ' *) exit 1;; esac; printf %s "$TARGETS"'],
        capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout == key


def test_submit_usage_lines_have_no_stray_brace():
    lines = open(SUBMIT).read().splitlines()
    for i in (16, 17):
        assert lines[i].count("}") == 1


# ------------------------------------------------- the B/BA witness
def test_lora_ab_norms_separates_a_from_b():
    """w_norm cannot witness training: A is random-initialised, so it is
    nonzero before any update. ||B|| is zero until the optimizer moves."""
    import _gated_pop as gp
    A = torch.randn(4, 8)
    pre = {"m.lora_A.default.weight": A,
           "m.lora_B.default.weight": torch.zeros(8, 4)}
    post = {"m.lora_A.default.weight": A,
            "m.lora_B.default.weight": torch.randn(8, 4) * 0.1}
    b0, ba0 = gp.lora_ab_norms(pre)
    b1, ba1 = gp.lora_ab_norms(post)
    assert b0 == 0.0 and ba0 == 0.0
    assert b1 > 0.0 and ba1 > 0.0
    assert gp.adapter_step(pre) > 0.0        # w_norm is blind to this


# ------------------------------------------------------------ checker
def _cell(root, lam, rounds=ROUNDS, smoke=False, lora_r=RANK, **over):
    sys.path.insert(0, CONDOR)
    import gen_pofd_sweep as g
    tag = g.f3_tag(1.0, 1.0, lam, rounds, smoke, lora_r)
    d = os.path.join(root, tag)
    os.makedirs(d, exist_ok=True)
    gen = torch.Generator().manual_seed(3)
    innate = torch.rand(N, generator=gen)
    op = torch.stack([(innate + .001 * (t + 1)).clamp(0, 1)
                      for t in range(rounds)])
    pred = torch.stack([torch.full((N,), 0.50 + .01 * t)
                        for t in range(rounds)])
    cfg = {"w_plat": 1.0, "innate_lambda": 1.0, "kl_beta": lam,
           "ab_sweeps": 100, "n_rounds": rounds, "seed": 0,
           "dataset": "movielens", "ml_target": "Action",
           "base_model": "Qwen/Qwen3-8B", "kl_direction": "forward",
           "ai_gate_mode": "all_open", "peer_gate_mode": "all_open",
           "icl_k": 0, "train_cap": N, "n_labeled": N,
           "lora_r": lora_r, "lora_alpha": 2 * lora_r,
           "gamma_bias": 0.0, "use_lora": True, "fresh_each_round": True,
           "training_style": "sft" if lam == 0 else "sft_kl",
           "ai_gate_reference": "anchor",
           "population_update": "nested_ai_anchored_then_social_v2"}
    cfg.update(over.pop("cfg", {}))
    torch.save({"config": cfg, "innate": innate, "op_raw": op,
                "pred_raw": pred, "twin_raw": op.clone(),
                "trajectory": [{"t": t} for t in range(rounds)],
                "sft_dose": over.pop("dose", None) or
                [{"round": t, "global_step": 181, "n_rows": N}
                 for t in range(rounds)]},
               os.path.join(d, "trajectory.pt"))
    tel = over.pop("tel", None)
    with open(os.path.join(d, "telemetry.json"), "w") as fh:
        for t in range(rounds):
            row = {"round": t, "l_init": 1.0, "n_train": N,
                   "w_norm": 6.3, "b_norm": 0.5, "ba_norm": 1.2,
                   "grad_kl_norm0": (0.0 if lam == 0 else 0.7)}
            if tel:
                row.update(tel(t) or {})
            fh.write(json.dumps(row) + "\n")
    with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt") as fh:
        for t in range(rounds):
            vals = pred[t].tolist()
            fh.write(json.dumps({
                "round": t, "parse_fail_frac": 0.0,
                "raw": over.get("raw", [f"{v:.2f}" for v in vals]),
                "parsed": vals}) + "\n")
    return d


def _run(root, smoke=False):
    cmd = [sys.executable, CHECKER, "--run-root", root, "--wave", "rank16"]
    if smoke:
        cmd.append("--smoke")
    return subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, "USE_TF": "0"})


def test_healthy_smoke_passes(tmp_path):
    root = str(tmp_path)
    _cell(root, 2.0, rounds=SMOKE_ROUNDS, smoke=True)
    r = _run(root, smoke=True)
    assert "PASS" in r.stdout, r.stdout


def test_wrong_rank_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 2.0, rounds=SMOKE_ROUNDS, smoke=True,
          cfg={"lora_r": 512, "lora_alpha": 1024})
    assert "lora_r" in _run(root, smoke=True).stdout


def test_alpha_not_twice_rank_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 2.0, rounds=SMOKE_ROUNDS, smoke=True,
          cfg={"lora_alpha": 64})
    assert "lora_alpha" in _run(root, smoke=True).stdout


def test_zero_b_norm_fails(tmp_path):
    """The adapter never moved -- exactly the undercapacity failure this
    wave has to be able to see."""
    root = str(tmp_path)
    _cell(root, 2.0, rounds=SMOKE_ROUNDS, smoke=True,
          tel=lambda t: {"b_norm": 0.0, "ba_norm": 0.0})
    assert "B or BA is identically zero" in _run(root, smoke=True).stdout


def test_missing_b_norm_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 2.0, rounds=SMOKE_ROUNDS, smoke=True,
          tel=lambda t: {"b_norm": None})
    d = os.path.join(root, os.listdir(root)[0], "telemetry.json")
    lines = [json.loads(l) for l in open(d)]
    for r in lines:
        r.pop("b_norm", None)
    open(d, "w").write("\n".join(json.dumps(r) for r in lines) + "\n")
    assert "no b_norm telemetry" in _run(root, smoke=True).stdout


def test_short_training_set_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 2.0, rounds=SMOKE_ROUNDS, smoke=True,
          tel=lambda t: {"n_train": 700} if t == 1 else None)
    assert "did not train all" in _run(root, smoke=True).stdout


def test_missing_optimizer_steps_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 2.0, rounds=SMOKE_ROUNDS, smoke=True,
          dose=[{"round": t, "global_step": 181 if t else 90, "n_rows": N}
                for t in range(SMOKE_ROUNDS)])
    assert "global_step" in _run(root, smoke=True).stdout


def test_kl_gradient_must_be_nonzero_at_lambda2(tmp_path):
    root = str(tmp_path)
    _cell(root, 2.0, rounds=SMOKE_ROUNDS, smoke=True,
          tel=lambda t: {"grad_kl_norm0": 0.0})
    assert "KL" in _run(root, smoke=True).stdout


def test_uncorrected_operator_fails(tmp_path):
    root = str(tmp_path)
    _cell(root, 2.0, rounds=SMOKE_ROUNDS, smoke=True,
          cfg={"population_update": "nested_ai_then_social_v1"})
    assert "OPERATOR" in _run(root, smoke=True).stdout


def test_malformed_generation_fails(tmp_path):
    root = str(tmp_path)
    d = _cell(root, 2.0, rounds=SMOKE_ROUNDS, smoke=True)
    p = os.path.join(d, "raw_gen_log.json.gz")
    rows_ = [json.loads(l) for l in gzip.open(p, "rt")]
    rows_[1]["raw"][3] = "0.5 (about"
    with gzip.open(p, "wt") as fh:
        for r in rows_:
            fh.write(json.dumps(r) + "\n")
    assert "malformed" in _run(root, smoke=True).stdout


def test_parsed_must_equal_pred_raw(tmp_path):
    root = str(tmp_path)
    d = _cell(root, 2.0, rounds=SMOKE_ROUNDS, smoke=True)
    p = os.path.join(d, "raw_gen_log.json.gz")
    rows_ = [json.loads(l) for l in gzip.open(p, "rt")]
    rows_[2]["parsed"][11] = 0.99
    rows_[2]["raw"] = [f"{v:.2f}" for v in rows_[2]["parsed"]]
    with gzip.open(p, "wt") as fh:
        for r in rows_:
            fh.write(json.dumps(r) + "\n")
    assert "does not describe this trajectory" in _run(root, smoke=True).stdout
