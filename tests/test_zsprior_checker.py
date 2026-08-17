"""Tests for the zero-shot prior screen (2026-08-17, zsprior_screen).

CHAT_THINKING plumbing (run_pokec_gated_lm._chat_template_kwargs):
unset/"default" -> {} (archived prompts byte-identical), "0"/"1" ->
the explicit enable_thinking directive, anything else -> loud
ValueError.

Checker ZSPRIOR branch, via synthetic pofdzsprior_ fixtures: all four
candidate checkpoints PASS when healthy; digit-free responses, a
parsed/served mismatch, a missing raw log, a Qwen3 probe without
chat_thinking=False, a thinking directive on a non-Qwen3 probe, moved
opinions, an open gate, and out-of-range predictions all FAIL; a
constant prior PASSES with a WARN.

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac).
"""
import gzip
import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")
CHECKER = os.path.join(PIPE, "check_pofd_sanity.py")

os.environ.setdefault("USE_TF", "0")
_spec_rm = importlib.util.spec_from_file_location(
    "runner_for_zsprior_test", os.path.join(PIPE,
                                            "run_pokec_gated_lm.py"))
RM = importlib.util.module_from_spec(_spec_rm)
_spec_rm.loader.exec_module(RM)

N = 60
_G0 = torch.Generator().manual_seed(20260818)
INNATE = torch.rand(N, generator=_G0)
BASE_OF = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
    "ministral8b": "mistralai/Ministral-8B-Instruct-2410",
    "mistralnemo": "mistralai/Mistral-Nemo-Instruct-2407",
}


# -- CHAT_THINKING plumbing ----------------------------------------------

def test_chat_kwargs_default_is_empty(monkeypatch):
    monkeypatch.delenv("CHAT_THINKING", raising=False)
    assert RM._chat_template_kwargs() == {}
    monkeypatch.setenv("CHAT_THINKING", "")
    assert RM._chat_template_kwargs() == {}
    monkeypatch.setenv("CHAT_THINKING", "default")
    assert RM._chat_template_kwargs() == {}


def test_chat_kwargs_directives(monkeypatch):
    monkeypatch.setenv("CHAT_THINKING", "0")
    assert RM._chat_template_kwargs() == {"enable_thinking": False}
    monkeypatch.setenv("CHAT_THINKING", "1")
    assert RM._chat_template_kwargs() == {"enable_thinking": True}


def test_chat_kwargs_rejects_garbage(monkeypatch):
    monkeypatch.setenv("CHAT_THINKING", "off")
    with pytest.raises(ValueError):
        RM._chat_template_kwargs()


# -- checker fixtures ----------------------------------------------------

def cfg_for(tag, slug):
    c = {"run_tag": tag, "base_model": BASE_OF[slug], "n_rounds": 1,
         "seed": 0, "eps": 0.0, "gamma_bias": 0.0, "w_plat": 0.5,
         "innate_lambda": 0.2,
         "population_update": "nested_ai_then_social_v1",
         "data_regime": "replace", "deploy_every": 1, "pop_model": "ab",
         "run_mode": "loop", "canary_delta": 0.0, "pop_reset": False,
         "ab_sweeps": 1, "platform_sus_scale": 1.0,
         "anchor_mode": "fixed", "dataset": "movielens",
         "ml_target": "Action", "do_sample": False, "n_labeled": 723,
         "seed_base_data": True, "pristine_frac": 0.0,
         "replay_frac": 0.0, "teacher_label_delta": 0.0,
         "kl_ref_adapter": "", "icrh": False, "feedback_mode": "none",
         "profile_shuffle_p": 0.0, "profile_sort_q": 0.0,
         "profile_drop_cols": [], "profile_permute_cols": [],
         "eps_ai": 0.0, "ai_gate_mode": "threshold", "host": "g204",
         "training_style": "frozen", "kl_beta": 0.0, "use_lora": 0,
         "fresh_each_round": False, "icl_k": 0, "icl_days": 0,
         "save_raw_gen": True,
         "hardware": {"hostname": "g204", "gpu_name": "A100",
                      "gpu_cc": "8.0", "cuda_version": "12.4",
                      "torch_version": "2.5.1",
                      "transformers_version": "4.56.0"}}
    if slug == "qwen3_8b":
        c["chat_thinking"] = False
    return c


def build(parent, slug, tag=None, cfg_mut=None, post=None,
          raw_log=True, constant=None):
    if tag is None:
        tag = f"pofdzsprior_{slug}_w0p5_l0p2_es0_s0"
    cfg = cfg_for(tag, slug)
    if cfg_mut:
        cfg_mut(cfg)
    g = torch.Generator().manual_seed(5000)
    if constant is not None:
        raw = [f"{constant:.2f}" for _ in range(N)]
    else:
        raw = [f"{int(torch.randint(0, 101, (1,), generator=g)) / 100:.2f}"
               for _ in range(N)]
    pred = torch.tensor([float(s) for s in raw], dtype=torch.float32)
    lam = 0.2
    op = lam * INNATE + (1.0 - lam) * INNATE
    gate = torch.zeros(N, dtype=torch.bool)
    rows = [{"round": 0, "deployment": 0, "is_deploy": 1, "accepted": 0,
             "contact": 0.0, "perplexity": 7.77}]
    d = {"trajectory": rows, "config": cfg,
         "op_raw": op.unsqueeze(0), "pred_raw": pred.unsqueeze(0),
         "gate_raw": gate.unsqueeze(0),
         "twin_raw": torch.empty(0), "innate": INNATE.clone()}
    if post:
        post(d)
    rd = parent / d["config"]["run_tag"]
    rd.mkdir(parents=True)
    (rd / "config.json").write_text(json.dumps(d["config"]))
    torch.save(d, rd / "trajectory.pt")
    if raw_log:
        with gzip.open(rd / "raw_gen_log.json.gz", "wt") as fh:
            fh.write(json.dumps({
                "round": 0, "parse_fail_frac": 0.0, "raw": raw,
                "parsed": [float(v)
                           for v in d["pred_raw"][0].tolist()]}) + "\n")
    return rd


def run_check(rd):
    p = subprocess.run([sys.executable, CHECKER, str(rd)],
                       capture_output=True, text=True,
                       env={**os.environ, "USE_TF": "0"})
    return p.returncode, p.stdout + p.stderr


def assert_verdict(rd, want_pass, want_str=None):
    rc, out = run_check(rd)
    if want_pass:
        assert rc == 0, f"expected PASS, got exit {rc}:\n{out[-2500:]}"
    else:
        assert rc != 0, f"expected FAIL, checker passed:\n{out[-2500:]}"
        if want_str is not None:
            assert want_str in out, \
                f"expected {want_str!r} in output:\n{out[-2500:]}"


# -- healthy -------------------------------------------------------------

@pytest.mark.parametrize("slug", list(BASE_OF))
def test_healthy_probe(tmp_path, slug):
    assert_verdict(build(tmp_path, slug), True)


def test_constant_prior_warns_but_passes(tmp_path):
    rd = build(tmp_path, "mistralnemo", constant=0.5)
    rc, out = run_check(rd)
    assert rc == 0, out[-2500:]
    assert "WARN" in out and "constant zero-shot prior" in out


# -- sabotage ------------------------------------------------------------

def test_digit_free_responses_fail(tmp_path):
    rd = build(tmp_path, "olmo3_7b", raw_log=False)
    with gzip.open(rd / "raw_gen_log.json.gz", "wt") as fh:
        d = torch.load(rd / "trajectory.pt", weights_only=False)
        raw = [f"{v:.2f}" for v in d["pred_raw"][0].tolist()]
        raw[7] = "I think this user would enjoy"
        fh.write(json.dumps({"round": 0, "parse_fail_frac": 1 / N,
                             "raw": raw,
                             "parsed": [float(v) for v
                                        in d["pred_raw"][0].tolist()]})
                 + "\n")
    assert_verdict(rd, False, "numeric parsing FAILED")


def test_parsed_served_mismatch_fails(tmp_path):
    rd = build(tmp_path, "ministral8b", raw_log=False)
    d = torch.load(rd / "trajectory.pt", weights_only=False)
    parsed = [float(v) for v in d["pred_raw"][0].tolist()]
    parsed[3] = 1.0 - parsed[3] if parsed[3] != 0.5 else 0.9
    with gzip.open(rd / "raw_gen_log.json.gz", "wt") as fh:
        fh.write(json.dumps({
            "round": 0, "parse_fail_frac": 0.0,
            "raw": [f"{v:.2f}" for v in parsed],
            "parsed": parsed}) + "\n")
    assert_verdict(rd, False, "provenance broken")


def test_missing_raw_log_fails(tmp_path):
    rd = build(tmp_path, "qwen3_8b", raw_log=False)
    assert_verdict(rd, False, "raw_gen_log.json.gz missing")


def test_qwen3_without_thinking_directive_fails(tmp_path):
    rd = build(tmp_path, "qwen3_8b",
               cfg_mut=lambda c: c.pop("chat_thinking"))
    assert_verdict(rd, False, "chat_thinking")


def test_thinking_directive_on_other_checkpoint_fails(tmp_path):
    rd = build(tmp_path, "mistralnemo",
               cfg_mut=lambda c: c.update(chat_thinking=False))
    assert_verdict(rd, False, "only the Qwen3 probe")


def test_moved_opinions_fail(tmp_path):
    rd = build(tmp_path, "olmo3_7b")
    d = torch.load(rd / "trajectory.pt", weights_only=False)
    d["op_raw"][0][5] += 1e-3
    torch.save(d, rd / "trajectory.pt")
    assert_verdict(rd, False, "cannot update opinions")


def test_open_gate_fails(tmp_path):
    rd = build(tmp_path, "ministral8b")
    d = torch.load(rd / "trajectory.pt", weights_only=False)
    d["gate_raw"][0][0] = True
    torch.save(d, rd / "trajectory.pt")
    assert_verdict(rd, False, "can never open")


def test_out_of_range_prediction_fails(tmp_path):
    def post(d):
        d["pred_raw"][0][0] = 1.5
    rd = build(tmp_path, "mistralnemo", post=post, raw_log=False)
    d = torch.load(rd / "trajectory.pt", weights_only=False)
    with gzip.open(rd / "raw_gen_log.json.gz", "wt") as fh:
        fh.write(json.dumps({
            "round": 0, "parse_fail_frac": 0.0,
            "raw": [f"{v:.2f}" for v in d["pred_raw"][0].tolist()],
            "parsed": [float(v) for v in d["pred_raw"][0].tolist()]})
            + "\n")
    assert_verdict(rd, False, "outside [0,1]")
