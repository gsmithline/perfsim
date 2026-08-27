"""Tests for the SECTION-3 PERSONAL-HISTORY ICL wave (section3_model_icl,
tags pofds3i_) and its gate.

The wave exists to be PAIRED with section3_model_equilibria: same
surface, one thing changed (reference-regularized SFT -> frozen
personal-history ICL). So the failure that matters is not "the run
crashed" but "the run looks perfect and is not comparable" -- a drifted
environment field, weights that quietly trained, a history log that does
not replay, a cross-agent value in a prompt, or a malformed generation
silently served as a default. Every one of those is asserted here.

The grid and the arm are restated INDEPENDENTLY of the generator: a test
that reads its expectations from the thing under test cannot catch a
grid bug.

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac):
  USE_TF=0 python -m pytest tests/test_section3_model_icl.py -q
"""
from __future__ import annotations

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
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")
CHECKER = os.path.join(PIPE, "check_section3_model_icl.py")

# ------------------------------------------------- contract (restated)
ARMS = ("greedy", "sample_t1")
DECODE = {"greedy": {"do_sample": "0", "temp": None},
          "sample_t1": {"do_sample": "1", "temp": "1"}}
KEY = "section3_model_icl_greedy"
SMOKE_KEY = "section3_model_icl_greedy_smoke"


def key_for(arm, smoke=False):
    return f"section3_model_icl_{arm}" + ("_smoke" if smoke else "")
PREFIX, SMOKE_PREFIX = "pofds3i", "pofds3ismk"
MODELS = ("qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b",
          "mistral7b", "ministral8b")
BASE = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "ministral8b": "mistralai/Ministral-8B-Instruct-2410",
}
SEEDS = (0, 42, 43)
ROUNDS, SMOKE_ROUNDS, SMOKE_SEED = 30, 3, 991
SMOKE_MODEL = "mistral7b"
SWEEPS, N, ICL_DAYS = 100, 723, 8
TARGET = "Action"

_G = torch.Generator().manual_seed(20260827)
INNATE = torch.rand(N, generator=_G)


def tag_for(model, seed, arm="greedy", rounds=ROUNDS, smoke=False):
    pre = SMOKE_PREFIX if smoke else PREFIX
    return (f"{pre}_{model}_d8_{arm}_sw{SWEEPS}_eaopen_w1_k1_esopen_anch2"
            f"_s{seed}_r{rounds}")


def _rows(key):
    return [l for l in open(os.path.join(
        CONDOR, f"configs_pofd_{key}.txt")).read().splitlines() if l.strip()]


def _cols(key):
    import re
    sub = open(os.path.join(CONDOR, f"at_pofd_{key}.sub")).read()
    q = next(l for l in sub.splitlines() if l.startswith("queue"))
    m = re.match(r"queue\s+(.*?)\s+from\s+(\S+)", q)
    return [c.strip() for c in m.group(1).split(",")], sub


# ------------------------------------------------ emitted artifacts
@pytest.mark.parametrize("arm", ARMS)
def test_production_grid_is_six_models_by_three_seeds(arm):
    cols, sub = _cols(key_for(arm))
    rows = _rows(key_for(arm))
    assert len(rows) == 18
    got = set()
    for r in rows:
        d = dict(zip(cols, [x.strip() for x in r.split(",")]))
        # THE ARM: frozen, no optimizer, no cross-user exemplars
        assert d["style"] == "frozen" and d["beta"] == "0"
        assert d["uselora"] == "0" and d["fresh"] == "0"
        assert d["iclk"] == "0", "ICL_K must be 0"
        # THE SURFACE: byte-matched to the SFT wave
        assert d["wplat"] == "1" and d["lam"] == "1"
        assert d["sweeps"] == str(SWEEPS) and d["gamma"] == "0.0"
        assert d["nrounds"] == str(ROUNDS)
        assert d["pop"] == "ab" and d["mode"] == "loop"
        model = next(m for m in MODELS if d["basemodel"] == BASE[m])
        assert d["tag"] == tag_for(model, int(d["seed"]), arm)
        got.add((model, int(d["seed"])))
    assert got == {(m, s) for m in MODELS for s in SEEDS}


@pytest.mark.parametrize("arm", ARMS)
def test_environment_is_the_icl_arm(arm):
    _, sub = _cols(key_for(arm))
    env = next(l for l in sub.splitlines() if l.startswith("environment"))
    for tok in ("ICL_DAYS=8 ", "SFT_EPOCHS=0 ", "PARSE_MODE=strict",
                "AI_GATE_MODE=all_open", "PEER_GATE_MODE=all_open",
                "AI_GATE_REFERENCE=anchor", "DEFFUANT_ALPHA=0.5",
                "AB_SWEEPS=$(sweeps)", "ICL_K=$(iclk)",
                "USE_LORA=$(uselora)", "SAVE_RAW_GEN=1", "WITH_TWIN=1",
                "ML_TARGET=Action", "TRAIN_CAP=723"):
        assert tok in env, tok
    # the SFT wave's optimizer settings must NOT survive into this one
    assert "SFT_EPOCHS=1" not in env
    assert "ICL_DAYS=0" not in env
    # THE DECODING ARM, pinned explicitly rather than left to a default
    assert f"DO_SAMPLE={DECODE[arm]['do_sample']}" in env
    assert ("GEN_TEMPERATURE=1" in env) == (arm == "sample_t1")
    assert f"check_section3_model_icl.py --arm {arm}" in sub \
        or f"--smoke --arm {arm}" in sub


@pytest.mark.parametrize("arm", ARMS)
def test_smoke_is_one_three_round_mistral_cell(arm):
    rows = _rows(key_for(arm, True))
    assert len(rows) == 1
    cols, _ = _cols(key_for(arm, True))
    d = dict(zip(cols, [x.strip() for x in rows[0].split(",")]))
    assert d["tag"] == tag_for(SMOKE_MODEL, SMOKE_SEED, arm, SMOKE_ROUNDS,
                               smoke=True)
    assert d["nrounds"] == str(SMOKE_ROUNDS)
    assert d["basemodel"] == BASE[SMOKE_MODEL]
    assert d["style"] == "frozen" and d["iclk"] == "0"


def test_tags_never_collide_with_the_sft_wave():
    mine = set()
    for arm in ARMS:
        for sm in (False, True):
            mine |= {r.split(",")[0].strip()
                     for r in _rows(key_for(arm, sm))}
    assert len(mine) == 38, "18 x 2 production + 2 smokes"
    greedy = {t for t in mine if "_greedy_" in t}
    samp = {t for t in mine if "_sample_t1_" in t}
    assert len(greedy) == 19 and len(samp) == 19 and not (greedy & samp)
    others = set()
    for f in os.listdir(CONDOR):
        if f.startswith("configs_pofd_") and "model_icl" not in f:
            others |= {l.split(",")[0].strip()
                       for l in open(os.path.join(CONDOR, f)) if l.strip()}
    assert others and not (mine & others)
    # and specifically the S3M family, which this wave is paired with
    s3m = {t for t in others if t.startswith(("pofds3m_", "pofds3msmk_"))}
    assert s3m and not (mine & s3m)
    assert all(t.startswith((PREFIX + "_", SMOKE_PREFIX + "_"))
               for t in mine)


def test_submit_script_knows_both_arms():
    src = open(os.path.join(CONDOR, "submit_pofd_sweep.sh")).read()
    for arm in ARMS:
        assert (f"section3_model_icl_{arm}|section3_model_icl_{arm}"
                f"_smoke) TARGETS=") in src


# -------------------------------------------------- synthetic runs
def render_days(vals):
    return (f"This user's own opinion of {TARGET} movies over the most "
            f"recent days (oldest to newest): "
            + ", ".join(f"{v:.2f}" for v in vals) + ".")


def build_run(root, model, seed, arm="greedy", rounds=ROUNDS, smoke=False,
              cfg_mut=None, days_mut=None, raw_mut=None, extra_files=(),
              traj_mut=None, pred_jitter=0.0):
    tag = tag_for(model, seed, arm, rounds, smoke)
    d = os.path.join(str(root), tag)
    os.makedirs(d, exist_ok=True)
    op, pred = [], []
    x = INNATE.clone()
    for t in range(rounds):
        x = x + 0.05 * ((0.5 + 0.0001 * seed) - x)
        op.append(x.clone())
        # a sampled arm's served vector must depend on the seed; a
        # greedy arm's round-0 vector must not
        pred.append(torch.full((N,), 0.55)
                    + (pred_jitter * seed * (t + 1)))
    op_t, pred_t = torch.stack(op), torch.stack(pred)
    cfg = {
        "run_tag": tag, "base_model": BASE[model], "dataset": "movielens",
        "ml_target": TARGET, "n_rounds": rounds, "seed": seed,
        "w_plat": 1.0, "innate_lambda": 1.0, "deffuant_alpha": 0.5,
        "ab_sweeps": SWEEPS, "gamma_bias": 0.0,
        "ai_gate_mode": "all_open", "peer_gate_mode": "all_open",
        "ai_gate_reference": "anchor",
        "population_update": "nested_ai_anchored_then_social_v2",
        "pop_model": "ab", "train_cap": N, "n_labeled": N,
        "seed_base_data": True, "save_raw_gen": True,
        "serve_eval_mode": True,
        "do_sample": arm == "sample_t1",
        "training_style": "frozen", "kl_beta": 0.0, "use_lora": False,
        "fresh_each_round": False, "sft_epochs": 0,
        "icl_k": 0, "icl_days": ICL_DAYS, "parse_mode": "strict",
        "git_sha": "0123456789abcdef0123456789abcdef01234567",
        **({"gen_temperature": 1.0} if arm == "sample_t1" else {}),
        "chat_thinking": False if model == "qwen3_8b" else None,
    }
    if cfg_mut:
        cfg_mut(cfg)
    traj = [{"round": t, "contact": 1.0, "peer_gate_mode": "all_open",
             "peer_pairs": N * SWEEPS, "accepted": N * SWEEPS}
            for t in range(rounds)]
    if traj_mut:
        traj_mut(traj)
    payload = {"config": cfg, "trajectory": traj, "op_raw": op_t,
               "pred_raw": pred_t, "innate": INNATE.clone(),
               "twin_raw": op_t.clone()}
    torch.save(payload, os.path.join(d, "trajectory.pt"))
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh, default=str)

    raws = [{"round": t, "parse_fail_frac": 0.0,
             "raw": ["0.55"] * N, "parsed": [0.55] * N}
            for t in range(rounds)]
    if raw_mut:
        raw_mut(raws)
    with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt",
                   compresslevel=1) as fh:
        for r in raws:
            fh.write(json.dumps(r) + "\n")

    hist = [INNATE.tolist()]
    rows = []
    for t in range(rounds):
        win = hist[-ICL_DAYS:]
        rows.append({"round": t,
                     "ctx": [render_days([h[i] for h in win])
                             for i in range(N)]})
        hist.append(op_t[t].tolist())
    if days_mut:
        days_mut(rows)
    with gzip.open(os.path.join(d, "icl_days_log.json.gz"), "wt",
                   compresslevel=1) as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    for name in extra_files:
        p = os.path.join(d, name)
        if name.endswith("/"):
            os.makedirs(p.rstrip("/"), exist_ok=True)
        else:
            open(p, "w").close()
    return d


def build_wave(root, arm="greedy", **kw):
    # the sampled arm's served vectors must differ across seeds
    kw.setdefault("pred_jitter", 1e-4 if arm == "sample_t1" else 0.0)
    for m in MODELS:
        for s in SEEDS:
            build_run(root, m, s, arm, **kw)
    return root


def run_checker(root, extra=(), arm="greedy"):
    env = dict(os.environ, USE_TF="0")
    return subprocess.run(
        [sys.executable, CHECKER, "--run-root", str(root), "--arm", arm]
        + list(extra), capture_output=True, text=True, env=env, cwd=REPO)


# ------------------------------------------------------------ healthy
@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_full_wave_passes(tmp_path, arm):
    root = build_wave(tmp_path / "runs", arm)
    r = run_checker(root, arm=arm)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "18/18 cells PASS" in r.stdout
    assert "byte-exact" in r.stdout


@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_smoke_passes(tmp_path, arm):
    root = tmp_path / "smk"
    build_run(root, SMOKE_MODEL, SMOKE_SEED, arm, SMOKE_ROUNDS, smoke=True)
    r = run_checker(root, extra=("--smoke",), arm=arm)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1/1 cells PASS" in r.stdout


# ----------------------------------------------------------- sabotage
@pytest.mark.slow
@pytest.mark.parametrize("field,value,needle", [
    ("training_style", "sft_kl", "training_style"),
    ("use_lora", True, "use_lora"),
    ("kl_beta", 2.0, "kl_beta"),
    ("sft_epochs", 1, "sft_epochs"),
    ("icl_days", 0, "icl_days"),
    ("icl_k", 8, "icl_k"),
    ("ab_sweeps", 1, "ab_sweeps"),
    ("w_plat", 0.5, "w_plat"),
    ("innate_lambda", 0.2, "innate_lambda"),
    ("peer_gate_mode", "threshold", "peer_gate_mode"),
    ("parse_mode", "legacy", "parse_mode"),
])
def test_drifted_config_field_fails(tmp_path, field, value, needle):
    """Every pinned field is load-bearing: the wave is only comparable to
    the SFT one if all of them hold."""
    def mut(c):
        c[field] = value
    root = tmp_path / "runs"
    build_run(root, "qwen7b", 0, cfg_mut=mut)
    r = run_checker(root)
    assert r.returncode == 1
    assert needle in r.stdout


@pytest.mark.slow
def test_adapter_on_a_frozen_arm_fails(tmp_path):
    """A config field is a claim; an adapter directory is evidence."""
    root = tmp_path / "runs"
    build_run(root, "qwen7b", 0, extra_files=("round0_adapter/",))
    r = run_checker(root)
    assert r.returncode == 1
    assert "round0_adapter" in r.stdout and "frozen" in r.stdout.lower()


@pytest.mark.slow
def test_optimizer_witness_on_a_frozen_arm_fails(tmp_path):
    def mut(traj):
        traj[2]["grad_kl_norm0"] = 0.31
    root = tmp_path / "runs"
    build_run(root, "qwen7b", 0, traj_mut=mut)
    r = run_checker(root)
    assert r.returncode == 1
    assert "grad_kl_norm0" in r.stdout


@pytest.mark.slow
def test_cross_agent_context_fails(tmp_path):
    """A foreign value in one prompt breaks the locality claim, and the
    byte-exact replay is what catches it."""
    def mut(rows):
        rows[2]["ctx"][7] = rows[2]["ctx"][8]      # agent 8's sentence
    root = tmp_path / "runs"
    build_run(root, "qwen7b", 0, days_mut=mut)
    r = run_checker(root)
    assert r.returncode == 1
    assert "HISTORY" in r.stdout


@pytest.mark.slow
def test_cross_agent_context_log_fails(tmp_path):
    root = tmp_path / "runs"
    build_run(root, "qwen7b", 0, extra_files=("icl_ctx_log.json.gz",))
    r = run_checker(root)
    assert r.returncode == 1
    assert "LOCALITY" in r.stdout


@pytest.mark.slow
def test_missing_history_log_fails(tmp_path):
    root = tmp_path / "runs"
    d = build_run(root, "qwen7b", 0)
    os.remove(os.path.join(d, "icl_days_log.json.gz"))
    r = run_checker(root)
    assert r.returncode == 1
    assert "icl_days_log.json.gz ABSENT" in r.stdout


@pytest.mark.slow
def test_short_history_log_fails(tmp_path):
    def mut(rows):
        del rows[5]
    root = tmp_path / "runs"
    build_run(root, "qwen7b", 0, days_mut=mut)
    r = run_checker(root)
    assert r.returncode == 1
    assert "HISTORY" in r.stdout


@pytest.mark.slow
def test_mistral_gets_no_parser_exemption(tmp_path):
    """The S3M wave carried a Mistral-only strict-parse exemption. This
    wave has none: a malformed Mistral generation is a hard failure, and
    the served value may not be a silent default."""
    def mut(raws):
        raws[1]["raw"] = [".64 ("] + ["0.55"] * (N - 1)
        raws[1]["parsed"] = [0.5] + [0.55] * (N - 1)
        raws[1]["parse_fail_frac"] = 1.0 / N
    root = tmp_path / "runs"
    build_run(root, "mistral7b", 0, raw_mut=mut)
    r = run_checker(root)
    assert r.returncode == 1
    assert "PARSE" in r.stdout
    # the real property, asserted against the CODE (the module docstring
    # is allowed to explain WHY the S3M wave needed a rerun): this gate
    # has no exemption machinery and branches on no model name except
    # the Qwen3 thinking pin.
    import ast
    tree = ast.parse(open(CHECKER).read())
    # every IDENTIFIER the gate defines or reads -- prose in docstrings
    # and error messages is deliberately not searched, because the gate
    # SHOULD explain why the SFT wave needed an exemption and this one
    # does not.
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
    flat = {x.upper().replace("_", "") for x in names}
    for forbidden in ("S3MRERUN", "RERUN", "STRICT", "SHAEXEMPT",
                      "EXEMPT", "INSPECTARCHIVED", "ALLOWUNGATED"):
        assert not any(forbidden in f for f in flat), forbidden
    # the only per-model branch is the Qwen3 thinking pin. Counted on
    # the AST, not on quoting: ast.unparse normalizes string quotes.
    per_model = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Compare)
                 and isinstance(n.left, ast.Name) and n.left.id == "model"]
    assert len(per_model) == 1, [ast.unparse(n) for n in per_model]
    assert ast.unparse(per_model[0]) in ("model == 'qwen3_8b'",
                                         'model == "qwen3_8b"')


@pytest.mark.slow
def test_closed_gate_in_the_runtime_evidence_fails(tmp_path):
    def mut(traj):
        traj[4]["contact"] = 0.4
    root = tmp_path / "runs"
    build_run(root, "qwen7b", 0, traj_mut=mut)
    r = run_checker(root)
    assert r.returncode == 1
    assert "GATE-RUNTIME" in r.stdout


@pytest.mark.slow
def test_two_git_shas_across_the_wave_fails(tmp_path):
    """This wave claims ONE code provenance and, unlike S3M, has no
    exemption to record."""
    root = tmp_path / "runs"
    for m in MODELS:
        for s in SEEDS:
            def mut(c, _m=m):
                if _m == "olmo7b":
                    c["git_sha"] = "f" * 40
            build_run(root, m, s, cfg_mut=mut)
    r = run_checker(root)
    assert r.returncode == 1
    assert "distinct git_sha" in r.stdout


# --------------------------------------------- the decoding evidence
@pytest.mark.slow
def test_sampled_seeds_sharing_a_served_vector_fails(tmp_path):
    """If the seed never reached the sampler, the three 'replicates' are
    one observation. pred_jitter=0 makes every seed serve the same
    vector, which is exactly that failure."""
    root = build_wave(tmp_path / "runs", "sample_t1", pred_jitter=0.0)
    r = run_checker(root, arm="sample_t1")
    assert r.returncode == 1
    assert "DECODE" in r.stdout and "did not reach the sampler" in r.stdout


@pytest.mark.slow
def test_greedy_seeds_disagreeing_at_round_zero_fails(tmp_path):
    """Greedy decoding on frozen weights with innate-only round-0
    prompts must be seed-invariant at round 0."""
    root = build_wave(tmp_path / "runs", "greedy", pred_jitter=1e-3)
    r = run_checker(root, arm="greedy")
    assert r.returncode == 1
    assert "disagree at ROUND 0" in r.stdout


@pytest.mark.slow
def test_wrong_decoder_in_the_config_fails(tmp_path):
    """A greedy cell whose config says it sampled -- and the reverse --
    is not the arm its tag claims."""
    def mut(c):
        c["do_sample"] = True
    root = tmp_path / "a"
    build_run(root, "qwen7b", 0, "greedy", cfg_mut=mut)
    r = run_checker(root, arm="greedy")
    assert r.returncode == 1 and "do_sample" in r.stdout

    def mut2(c):
        c["do_sample"] = False
    root2 = tmp_path / "b"
    build_run(root2, "qwen7b", 0, "sample_t1", cfg_mut=mut2)
    r = run_checker(root2, arm="sample_t1")
    assert r.returncode == 1 and "do_sample" in r.stdout


@pytest.mark.slow
def test_wrong_sampling_temperature_fails(tmp_path):
    def mut(c):
        c["gen_temperature"] = 0.7
    root = tmp_path / "runs"
    build_run(root, "qwen7b", 0, "sample_t1", cfg_mut=mut)
    r = run_checker(root, arm="sample_t1")
    assert r.returncode == 1
    assert "gen_temperature" in r.stdout


@pytest.mark.slow
def test_arm_tags_are_not_interchangeable(tmp_path):
    """A greedy run dir cannot satisfy a sampled cell: the tags differ,
    so the sampled gate reports its own cells absent."""
    root = build_wave(tmp_path / "runs", "greedy")
    r = run_checker(root, arm="sample_t1")
    assert r.returncode == 1
    assert "absent" in r.stdout.lower()
