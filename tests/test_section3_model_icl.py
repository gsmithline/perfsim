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
DECODE = {"greedy": {"do_sample": "0", "temp": None,
                     "parse_mode": "strict", "max_new_tokens": None},
          "sample_t1": {"do_sample": "1", "temp": "1",
                        "parse_mode": "prose", "max_new_tokens": 32}}
SMOKE_REV = {"sample_t1": "_p2"}      # the revised sampled smoke
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
    rev = SMOKE_REV.get(arm, "") if smoke else ""
    return (f"{pre}_{model}_d8_{arm}{rev}_sw{SWEEPS}_eaopen_w1_k1"
            f"_esopen_anch2_s{seed}_r{rounds}")


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
    for tok in ("ICL_DAYS=8 ", "SFT_EPOCHS=0 ",
                f"PARSE_MODE={DECODE[arm]['parse_mode']}",
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
    mnt = DECODE[arm]["max_new_tokens"]
    assert (f"MAX_NEW_TOKENS={mnt}" in env) == (mnt is not None)
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
        # varies BY ROUND (so a round-shifted raw log is detectable),
        # and by SEED only on the sampled arm -- the greedy arm's round-0
        # vector must be seed-invariant
        pred.append(torch.full((N,), 0.55 + 0.001 * t)
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
        **({"gen_top_p": 1.0, "gen_top_k": 0,
            "gen_repetition_penalty": 1.0,
            "gen_policy_effective": {
                "do_sample": {"value": True, "source": "pinned"},
                "temperature": {"value": 1.0, "source": "pinned"},
                "top_p": {"value": 1.0, "source": "pinned"},
                "top_k": {"value": 0, "source": "pinned"},
                "repetition_penalty": {"value": 1.0, "source": "pinned"}}}
           if arm == "sample_t1" else {}),
        "training_style": "frozen", "kl_beta": 0.0, "use_lora": False,
        "fresh_each_round": False, "sft_epochs": 0,
        "icl_k": 0, "icl_days": ICL_DAYS,
        "parse_mode": DECODE[arm]["parse_mode"],
        **({"max_new_tokens": DECODE[arm]["max_new_tokens"]}
           if DECODE[arm]["max_new_tokens"] is not None else {}),
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

    raws = []
    for t in range(rounds):
        served = [round(float(v), 6) for v in pred_t[t].tolist()]
        raws.append({"round": t, "parse_fail_frac": 0.0,
                     "raw": [f"{v:.6f}" for v in served],
                     "parsed": served})
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


def test_submit_usage_strings_contain_no_brace():
    """REGRESSION (2026-08-27). The usage text lives inside
    ``BID="${1:?usage: ...}"`` and ``WHAT="${2:?usage: ...}"``. A literal
    '}' anywhere in it CLOSES the parameter expansion early, so BID and
    WHAT silently absorb the rest of the string -- BID became
    '25[_smoke]|section4...' and every key fell through to the usage
    branch. Writing a key as '..._{greedy,sample_t1}' caused exactly
    that. The usage grammar uses [] and | only; braces are forbidden."""
    src = open(os.path.join(CONDOR, "submit_pofd_sweep.sh")).read().splitlines()
    for ln in src:
        st = ln.strip()
        if st.startswith(("BID=", "WHAT=")) or st.startswith('*) echo "usage:'):
            body = st.split("usage:", 1)[1] if "usage:" in st else ""
            # the ONLY closing brace allowed is the one that terminates
            # the expansion at the very end of the assignment
            assert "{" not in body, f"brace in usage text: {st[:80]}"


def test_submit_resolves_both_arms_end_to_end(tmp_path):
    """The case must actually resolve TARGETS for all four keys -- the
    check that would have caught the brace bug."""
    sh = os.path.join(CONDOR, "submit_pofd_sweep.sh")
    body = open(sh).read()
    start = body.index('case "$WHAT" in')
    end = body.index("\nesac", start) + len("\nesac")
    case_block = body[start:end]
    for key in ("section3_model_icl_greedy",
                "section3_model_icl_greedy_smoke",
                "section3_model_icl_sample_t1",
                "section3_model_icl_sample_t1_smoke"):
        script = f'WHAT="{key}"\nTARGETS=""\n{case_block}\necho "$TARGETS"'
        r = subprocess.run(["bash", "-c", script], capture_output=True,
                           text=True)
        assert r.returncode == 0, (key, r.stderr[:200])
        assert r.stdout.strip() == key, (key, r.stdout.strip())


# ============ RELEASE BLOCKERS (2026-08-27): regression tests ==========
def test_sampled_arm_pins_the_decoding_policy_in_the_sub():
    """BLOCKER 1. generate() sends only max_new_tokens/do_sample/
    pad_token_id/temperature, so top_p, top_k and repetition_penalty come
    from EACH CHECKPOINT's generation_config unless pinned. 'One draw at
    T=1 from the model's own distribution' is false when a checkpoint
    truncates, so the sampled arm must pin them."""
    _, sub = _cols(key_for("sample_t1"))
    env = next(l for l in sub.splitlines() if l.startswith("environment"))
    for tok in ("DO_SAMPLE=1", "GEN_TEMPERATURE=1", "GEN_TOP_P=1",
                "GEN_TOP_K=0", "GEN_REPETITION_PENALTY=1"):
        assert tok in env, tok
    # greedy deliberately INHERITS: argmax ignores top_p/top_k and the
    # repetition penalty must match the SFT wave it is paired against
    _, gsub = _cols(key_for("greedy"))
    genv = next(l for l in gsub.splitlines() if l.startswith("environment"))
    assert "GEN_TOP_P" not in genv and "GEN_TOP_K" not in genv


def test_model_wrapper_defaults_preserve_archived_behaviour():
    """The new knobs must default to None = inherit, or every archived
    run's decoding would change meaning retroactively."""
    import ast
    src = open(os.path.join(REPO, "perfsim", "models",
                            "hf_causal_lm.py")).read()
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    init = next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    pos = init.args.posonlyargs + init.args.args
    defaults = {}
    if init.args.defaults:
        for a, d in zip(pos[-len(init.args.defaults):], init.args.defaults):
            defaults[a.arg] = d
    for a, d in zip(init.args.kwonlyargs, init.args.kw_defaults):
        defaults[a.arg] = d
    args = pos + init.args.kwonlyargs
    for knob in ("top_p", "top_k", "repetition_penalty"):
        assert knob in [a.arg for a in args], f"{knob} not a parameter"
        d = defaults.get(knob)
        assert isinstance(d, ast.Constant) and d.value is None, \
            f"{knob} must default to None (inherit)"
    # and they must only be SENT when pinned
    assert "if self._top_p is not None:" in src
    assert "if self._repetition_penalty is not None:" in src


@pytest.mark.slow
def test_sampled_arm_inheriting_a_checkpoint_default_fails(tmp_path):
    """BLOCKER 1, gate side: a sampled cell whose recorded effective
    policy says a knob came from the checkpoint must FAIL."""
    def mut(c):
        c["gen_policy_effective"]["top_p"] = {"value": 0.8,
                                              "source": "checkpoint_default"}
    root = tmp_path / "runs"
    build_run(root, "qwen7b", 0, "sample_t1", cfg_mut=mut)
    r = run_checker(root, arm="sample_t1")
    assert r.returncode == 1
    assert "checkpoint_default" in r.stdout or "not 'pinned'" in r.stdout


@pytest.mark.slow
def test_sampled_arm_missing_the_effective_policy_fails(tmp_path):
    def mut(c):
        c.pop("gen_policy_effective", None)
    root = tmp_path / "runs"
    build_run(root, "qwen7b", 0, "sample_t1", cfg_mut=mut)
    r = run_checker(root, arm="sample_t1")
    assert r.returncode == 1
    assert "gen_policy_effective" in r.stdout


@pytest.mark.slow
def test_parsed_vector_must_equal_pred_raw(tmp_path):
    """BLOCKER 3. The strict raw-generation gate proves the log is
    internally consistent; it does NOT prove the log describes THIS run.
    A well-formed log whose parsed vector is not pred_raw must fail."""
    def mut(raws):
        raws[4]["parsed"] = [0.42] * N
        raws[4]["raw"] = ["0.420000"] * N          # still well-formed
    root = tmp_path / "runs"
    build_run(root, "qwen7b", 0, "greedy", raw_mut=mut)
    r = run_checker(root, arm="greedy")
    assert r.returncode == 1
    assert "PARSE-VS-SERVED" in r.stdout


@pytest.mark.slow
def test_parsed_vector_shifted_by_one_round_fails(tmp_path):
    """The sharpest form of the same failure: a log that is correct but
    belongs to the wrong rounds."""
    def mut(raws):
        vals = [r["parsed"] for r in raws]
        for i, r in enumerate(raws):
            r["parsed"] = vals[(i + 1) % len(vals)]
            r["raw"] = [f"{v:.6f}" for v in r["parsed"]]
    root = tmp_path / "runs"
    build_run(root, "qwen7b", 0, "greedy", raw_mut=mut)
    r = run_checker(root, arm="greedy")
    assert r.returncode == 1
    assert "PARSE-VS-SERVED" in r.stdout


def test_analyzer_aggregates_sampled_stats_at_model_level():
    """BLOCKER 2. Per-cell sampled columns are not reportable; the model
    row must carry the across-seed mean AND across-seed uncertainty of
    each, and must keep the three spreads distinct."""
    src = open(os.path.join(
        PIPE, "analyze_section3_model_equilibria.py")).read()
    for key in ("late_mean", "round30_mean", "temporal_sd", "late_drift",
                "pop_sd_final", "pop_sd_late"):
        assert f'("{key}", "{key}")' in src, key
    for suffix in ("_seed_sd", "_ci95_low", "_ci95_high"):
        assert f'f"{{out_prefix}}{suffix}"' in src, suffix
    assert '"headline_column"' in src
    assert '"stochastic_arm"' in src


def test_d8_history_wording_starts_at_innate():
    """The history is the last <=8 of [innate, op_raw[0], ...]: it starts
    with the INNATE opinion at t=0, then post-peer states."""
    gen = open(os.path.join(CONDOR, "gen_pofd_sweep.py")).read()
    i = gen.index("S3I_KEY = ")
    block = gen[max(0, i - 6000):i]
    assert "INNATE opinion" in block and "post-peer states" in block
    chk = open(CHECKER).read()
    assert "STARTS with the" in chk and "innate opinion" in chk.lower()


# ===== PROSE PARSER + GENERATION BUDGET (2026-08-27 audit) ============
def _prose_parser():
    """Load _parse_prose without importing the torch-heavy module."""
    import ast, re as _re
    src = open(os.path.join(REPO, "perfsim", "models",
                            "hf_causal_lm.py")).read()
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    ns = {"re": _re}
    for n in cls.body:
        if isinstance(n, ast.Assign) and getattr(
                n.targets[0], "id", "").startswith("_"):
            exec(compile(ast.Module([n], []), "<x>", "exec"), ns)
        if isinstance(n, ast.FunctionDef) and n.name == "_parse_prose":
            exec(compile(ast.Module([n], []), "<x>", "exec"), ns)

    class C:
        pass
    for k in ("_STRICT_RE", "_STANDALONE_NUM", "_SCALE_PHRASE",
              "_LABELLED"):
        setattr(C, k, ns[k])
    C._parse_prose = classmethod(ns["_parse_prose"])
    return C


@pytest.mark.parametrize("text,value,ok,why", [
    ("0.58 (", 0.58, True, "leading number: identical to strict"),
    ("0.7", 0.7, True, "bare number"),
    ("Based on the provided data, the estimated rating is 0.72",
     0.72, True, "explicitly labelled final value"),
    ("Based on the data, I would predict 0.65 for this user",
     0.65, True, "exactly one standalone value"),
    ("On a scale between 0 and 1, the answer is 0.4",
     0.4, True, "scale phrase is not a candidate"),
    ("The user rated 3 movies; the estimate is 0.8",
     0.8, True, "out-of-range int ignored, label wins"),
    ("The estimate is 0.6, though it could be 0.9",
     0.6, True, "the LABELLED value wins over an unlabelled hedge"),
    ("The estimate is 0.6; the final answer is 0.9",
     0.9, True, "with two labels, the FINAL one wins"),
    # --- rejections: the default is never a prediction ---
    ("It could be 0.3 or 0.7", None, False, "two values: AMBIGUOUS"),
    ("Based on the provided data,", None, False, "no number at all"),
    ("To estimate the user's", None, False, "truncated preamble"),
    ("58 (58", None, False, "leading number out of range"),
    ("The scale is 0 to 1", None, False, "only a scale description"),
    ("", None, False, "empty"),
])
def test_prose_parser_decision_rules(text, value, ok, why):
    C = _prose_parser()
    got_v, got_ok = C._parse_prose(text)
    assert got_ok is ok, f"{why}: {text!r} -> ok={got_ok}"
    if ok:
        assert abs(got_v - value) < 1e-9, why
    else:
        # a rejection must NOT be presented as a prediction; the caller
        # counts it as a parse failure
        assert got_v == 0.5


def test_prose_never_takes_the_first_number_anywhere():
    """The explicit warning: explanatory prose can carry unrelated
    numbers, so 'first number found' is wrong."""
    C = _prose_parser()
    # 0.2 appears first but is not the stated answer
    v, ok = C._parse_prose("Similar users scored 0.2; the estimate is 0.9")
    assert ok and abs(v - 0.9) < 1e-9
    # two bare values with no label must be REJECTED, not first-wins
    v, ok = C._parse_prose("Maybe 0.2, maybe 0.9")
    assert not ok


def test_prose_mode_is_accepted_by_the_runner():
    src = open(os.path.join(PIPE, "run_pokec_gated_lm.py")).read()
    assert '("legacy", "strict", "prose")' in src
    # and the generation budget is now RECORDED (it was not before)
    assert '"max_new_tokens": max_new_tokens,' in src


def test_sampled_arm_pins_budget_and_prose_parser():
    _, sub = _cols(key_for("sample_t1"))
    env = next(l for l in sub.splitlines() if l.startswith("environment"))
    assert "MAX_NEW_TOKENS=32" in env and "PARSE_MODE=prose" in env
    _, gsub = _cols(key_for("greedy"))
    genv = next(l for l in gsub.splitlines() if l.startswith("environment"))
    # greedy keeps strict and the runner default budget: it emits the
    # number first (max 2 words) and must stay comparable to the SFT wave
    assert "PARSE_MODE=strict" in genv and "MAX_NEW_TOKENS" not in genv


@pytest.mark.slow
def test_wrong_budget_or_parse_mode_fails(tmp_path):
    for field, value, needle in (("max_new_tokens", 6, "max_new_tokens"),
                                 ("parse_mode", "strict", "parse_mode")):
        def mut(c, _f=field, _v=value):
            c[_f] = _v
        root = tmp_path / f"runs_{field}"
        build_run(root, "qwen7b", 0, "sample_t1", cfg_mut=mut)
        r = run_checker(root, arm="sample_t1")
        assert r.returncode == 1, field
        assert needle in r.stdout, field


def test_revised_sampled_smoke_cannot_reuse_the_contaminated_run():
    """The first sampled smoke served 0.5 to 210 agents under the old
    policy. The idempotent wrapper skips a tag whose run dir exists, so
    the revised smoke must be a DIFFERENT tag."""
    old = ("pofds3ismk_mistral7b_d8_sample_t1_sw100_eaopen_w1_k1"
           "_esopen_anch2_s991_r3")
    new = tag_for(SMOKE_MODEL, SMOKE_SEED, "sample_t1", SMOKE_ROUNDS,
                  smoke=True)
    assert new != old and "_p2_" in new
    emitted = {r.split(",")[0].strip()
               for r in _rows(key_for("sample_t1", True))}
    assert emitted == {new}
    # greedy is UNrevised: its smoke passed under the policy it still runs
    g = tag_for(SMOKE_MODEL, SMOKE_SEED, "greedy", SMOKE_ROUNDS, smoke=True)
    assert "_p2_" not in g


# =============== 100-ROUND HORIZON EXTENSION (2026-08-27) =============
EXT = {"key": "section3_model_icl_greedy_r100", "rounds": 100,
       "models": ("qwen3_8b", "olmo7b", "olmo3_7b"), "n": 9}


def test_extension_grid_is_the_unsettled_models_only():
    rows = _rows(EXT["key"])
    assert len(rows) == EXT["n"]
    cols, sub = _cols(EXT["key"])
    got = set()
    for r in rows:
        d = dict(zip(cols, [x.strip() for x in r.split(",")]))
        # CONFIGURATION UNCHANGED except the horizon
        assert d["sweeps"] == str(SWEEPS), "100 Deffuant sweeps unchanged"
        assert d["wplat"] == "1" and d["lam"] == "1"
        assert d["style"] == "frozen" and d["iclk"] == "0"
        assert d["uselora"] == "0" and d["gamma"] == "0.0"
        assert d["nrounds"] == str(EXT["rounds"])
        m = next(k for k in MODELS if d["basemodel"] == BASE[k])
        assert d["tag"] == tag_for(m, int(d["seed"]), "greedy",
                                   EXT["rounds"])
        assert d["tag"].endswith("_r100")
        got.add((m, int(d["seed"])))
    assert got == {(m, s) for m in EXT["models"] for s in SEEDS}
    assert "--arm greedy" in sub


def test_extension_never_overwrites_the_30_round_runs():
    ext = {r.split(",")[0].strip() for r in _rows(EXT["key"])}
    base = {r.split(",")[0].strip() for r in _rows(key_for("greedy"))}
    assert len(ext) == 9 and not (ext & base)
    # every extension cell has a completed 30-round twin, differing ONLY
    # in the horizon token
    for t in ext:
        assert t.endswith("_r100")
        assert t[:-5] + "_r30" in base


def test_extension_environment_matches_the_wave_it_extends():
    _, ext_sub = _cols(EXT["key"])
    _, main_sub = _cols(key_for("greedy"))
    env = lambda s: next(l for l in s.splitlines()
                         if l.startswith("environment"))
    assert env(ext_sub) == env(main_sub)


def test_submit_script_knows_the_extension():
    src = open(os.path.join(CONDOR, "submit_pofd_sweep.sh")).read()
    assert 'section3_model_icl_greedy_r100) TARGETS="$WHAT" ;;' in src


@pytest.mark.slow
def test_extension_prefix_must_match_the_30_round_run(tmp_path):
    """The extension is run from round 0, so rounds 0..29 must reproduce
    the completed run BIT-FOR-BIT. A drifted prefix means it is a
    different process and cannot replace the short cell."""
    root = tmp_path / "runs"
    for m in EXT["models"]:
        for s_ in SEEDS:
            build_run(root, m, s_, "greedy")                    # r30
            build_run(root, m, s_, "greedy", rounds=EXT["rounds"])  # r100
    r = run_checker(root, extra=("--ext",), arm="greedy")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "9/9 cells PASS" in r.stdout

    # now perturb ONE extension run's prefix
    tag = tag_for("olmo7b", 42, "greedy", EXT["rounds"])
    p = os.path.join(str(root), tag, "trajectory.pt")
    d = torch.load(p, weights_only=False)
    d["op_raw"][5] = d["op_raw"][5] + 1e-4
    torch.save(d, p)
    r = run_checker(root, extra=("--ext",), arm="greedy")
    assert r.returncode == 1
    assert "PREFIX" in r.stdout and "bit-identical" in r.stdout


@pytest.mark.slow
def test_extension_without_its_30_round_twin_fails(tmp_path):
    root = tmp_path / "runs"
    for m in EXT["models"]:
        for s_ in SEEDS:
            build_run(root, m, s_, "greedy", rounds=EXT["rounds"])
    r = run_checker(root, extra=("--ext",), arm="greedy")
    assert r.returncode == 1
    assert "PREFIX" in r.stdout and "absent" in r.stdout
