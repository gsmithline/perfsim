"""ADVERSARIAL tests for the FIGURE-4 REPLICATION wave tooling
(check_fig4_repl.py + analyze_fig4_repl.py).

Every sabotage below is a run that LOOKS finished: config.json parses,
trajectory.pt is complete, the figure would render. The gate is the only
place these are still catchable.

THE FILE-PROVENANCE TRAP. parse_fail_frac is NOT in trajectory.pt -- it
lives in raw_gen_log.json.gz, a GZIPPED JSONL file, and l_init/grad_*
live in telemetry.json. A checker that read them off the per-round dicts
inside trajectory.pt would get None for every gate and pass vacuously.
test_parse_fail_is_read_from_the_gz_not_the_trajectory plants a
CONTRADICTORY clean value inside trajectory.pt and asserts the gate
follows the real file.

THE INNATE PIN. The gate pins the canonical movielens-Action 723-agent
innate vector by sha256. A synthetic fixture cannot match it, and if the
pin were left on here every "must FAIL" assertion would pass VACUOUSLY
-- the run would be rejected for the fixture's sake, not for the
sabotage's. So the per-cell tests run with --no-innate-pin, and the pin
itself gets its own two tests (match and mismatch) against a sha
computed from the fixture.

Contract pins are duplicated here on purpose: a test that imports its
expectations from the thing it is testing tests nothing. The only things
imported from the modules are the pure helpers whose ARITHMETIC is under
test (seed_agg, convergence, t_crit, w1).

Run:  USE_TF=0 python -m pytest tests/test_fig4_repl.py -q
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "experiments", "scripts", "cluster_pipelines")
CHECK = os.path.join(PIPE, "check_fig4_repl.py")
ANALYZE = os.path.join(PIPE, "analyze_fig4_repl.py")

# ---- the contract, restated (never imported) ------------------------
N_AGENTS = 723
ROUNDS = 30
SEEDS = (0, 42, 43)
# the tail window: the last 10 rounds of the 30-round horizon, split
# 21-25 vs 26-30 by the convergence test
LATE_LO, LATE_HI = 21, 30
WIN_LEN = LATE_HI - LATE_LO + 1
HALF = WIN_LEN // 2
POP_V1 = "nested_ai_then_social_v1"
POP_V2 = "nested_ai_anchored_then_social_v2"
GATE_REF = "anchor"
TAG = "pofdf4r_{slug}_b1_anch2_ea1_w0p5_l0p2_es0p05_r30_s{seed}"
FIG4_B1 = "pofdfam_{slug}_b1_ea1_w0p5_l0p2_es0p05_s0"
FIG4_K0 = "pofdfam_{slug}_k0_ea1_w0p5_l0p2_es0p05_s0"
HF = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "ministral8b": "mistralai/Ministral-8B-Instruct-2410",
}
MODEL_ORDER = ("qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b", "mistral7b",
               "ministral8b")
# 97.5% Student-t quantile at df = 2, restated (the wave's n = 3)
T2 = 4.302652729911275

needs_files = pytest.mark.skipif(
    not (os.path.exists(CHECK) and os.path.exists(ANALYZE)),
    reason="check_fig4_repl.py / analyze_fig4_repl.py have not landed yet")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- fixtures
def _innate(n=N_AGENTS, seed=1234):
    """The world. Bit-identical across checkpoints at a given seed --
    on movielens it is a deterministic function of the dataset, so the
    six cells of a seed must carry the same vector or they are not
    comparable at all."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, generator=g)


def _op_from_means(means, spread=0.05, n=N_AGENTS, seed=7):
    """A [T, n] population whose per-round MEAN is exactly means[t] and
    whose per-round SD is exactly the SD of a fixed zero-mean offset --
    so the aggregation arithmetic under test has a closed form."""
    g = torch.Generator().manual_seed(seed)
    off = torch.rand(n, generator=g) - 0.5
    off = off - off.mean()
    off = off / off.std() * spread
    return torch.stack([torch.full((n,), float(m)) + off for m in means])


def _sha_t(t):
    a = torch.as_tensor(t).detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def _mk_run(root, tag, *, slug="qwen7b", base_model=None, seed=0,
            rounds=ROUNDS, n_rounds=None, n=N_AGENTS, innate=None,
            op=None, pred=None, twin=None, pop_update=POP_V2,
            gate_ref=GATE_REF, parse_fail=0.0, chat_thinking="auto",
            grad_norm0=3.0, grad_kl0=None, skip_files=(), extra_cfg=None,
            traj_parse_fail=0.0):
    """One synthetic run dir matching the VERIFIED artifact schema.

    config.json           flat dict
    trajectory.pt         {"trajectory", "config", "op_raw", "pred_raw",
                           "twin_raw", "innate", "profiles"}
    telemetry.json        JSONL: round, l_init, grad_norm0, grad_kl_norm0
    raw_gen_log.json.gz   GZIPPED JSONL: round, parse_fail_frac, parsed

    `traj_parse_fail` plants a value for parse_fail_frac INSIDE
    trajectory.pt, where it does not belong -- the provenance trap.
    """
    d = os.path.join(root, tag)
    os.makedirs(d, exist_ok=True)
    if base_model is None:
        base_model = HF[slug]
    if chat_thinking == "auto":
        chat_thinking = False if slug == "qwen3_8b" else None
    cfg = {
        "run_tag": tag,
        "base_model": base_model, "seed": seed,
        "n_rounds": rounds if n_rounds is None else n_rounds,
        "dataset": "movielens", "ml_target": "Action", "n_labeled": n,
        "train_cap": n, "training_style": "sft_kl", "kl_beta": 1.0,
        "kl_direction": "forward", "kl_ref_adapter": "",
        "anchor_mode": "fixed", "w_plat": 0.5, "innate_lambda": 0.2,
        "gamma_bias": 0.0, "eps": 0.05, "eps_ai": 1.0,
        "ai_gate_mode": "threshold", "peer_gate_mode": "threshold",
        "ai_gate_reference": gate_ref, "population_update": pop_update,
        "pop_model": "ab", "ab_sweeps": 1, "use_lora": True,
        "fresh_each_round": True, "icl_k": 0, "icl_days": 0,
        "lora_r": 512, "sft_lr": 5e-05, "sft_epochs": 1,
        "sft_batch_size": 4, "epoch_size": 100, "max_steps": 1,
        "seed_base_data": True, "save_raw_gen": True, "deploy_every": 1,
        "data_regime": "replace", "feedback_mode": "none", "icrh": False,
        "do_sample": False, "pristine_frac": 0.0, "replay_frac": 0.0,
        "teacher_label_delta": 0.0, "serve_eval_mode": True,
        "git_sha": "cafe123", "fj_update_version": "legacy",
        "host": "g191",
        "hardware": {"hostname": "g191", "gpu_name": "NVIDIA H100",
                     "gpu_cc": "9.0", "cuda_version": "12.1",
                     "torch_version": "2.5.1",
                     "transformers_version": "4.46.0"},
    }
    if chat_thinking is not None:
        cfg["chat_thinking"] = chat_thinking
    if extra_cfg:
        cfg.update(extra_cfg)
    if "config.json" not in skip_files:
        with open(os.path.join(d, "config.json"), "w") as fh:
            json.dump(cfg, fh)

    g = torch.Generator().manual_seed(11 + seed)
    if innate is None:
        innate = _innate(n)
    if op is None:
        op = 0.3 + 0.3 * torch.rand(rounds, n, generator=g)
    if pred is None:
        pred = 0.25 + 0.4 * torch.rand(rounds, n, generator=g)
    if twin is None:
        twin = 0.2 + 0.3 * torch.rand(rounds, n, generator=g)
    rows = [{"round": t, "op_mean": float(op[t].mean()),
             "op_std": float(op[t].std()), "contact": 0.5,
             "accepted": n, "parse_fail_frac": traj_parse_fail}
            for t in range(op.shape[0])]
    if "trajectory.pt" not in skip_files:
        torch.save({"trajectory": rows, "config": cfg, "op_raw": op,
                    "pred_raw": pred, "twin_raw": twin, "innate": innate,
                    "profiles": [{"i": i} for i in range(n)]},
                   os.path.join(d, "trajectory.pt"))

    if grad_kl0 is None:
        # round 0 is EXEMPT: a fresh LoRA at round 0 IS the reference
        grad_kl0 = [0.0 if t == 0 else 1.5 for t in range(rounds)]
    if isinstance(grad_norm0, (int, float)):
        grad_norm0 = [float(grad_norm0)] * rounds
    if "telemetry.json" not in skip_files:
        with open(os.path.join(d, "telemetry.json"), "w") as fh:
            for t in range(rounds):
                fh.write(json.dumps({
                    "round": t, "l_init": 2.0 / (t + 1),
                    "grad_norm0": grad_norm0[t],
                    "grad_kl_norm0": grad_kl0[t], "n_train": 100}) + "\n")

    if isinstance(parse_fail, (int, float)):
        parse_fail = [float(parse_fail)] * rounds
    if "raw_gen_log.json.gz" not in skip_files:
        with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt") as fh:
            for t in range(rounds):
                vals = [round(float(v), 3) for v in pred[t].tolist()]
                fh.write(json.dumps({"round": t,
                                     "parse_fail_frac": parse_fail[t],
                                     "parsed": vals}) + "\n")
    return d


def _mk_archived(root, tag, *, slug="qwen7b", extra=None, drop=()):
    """The ARCHIVED Figure-4 cell, as it really is on disk: the OLD v1
    operator, a 30-round horizon, and none of the fields the runner
    gained after 2026-08-17 (ai_gate_reference, peer_gate_mode,
    serve_eval_mode, git_sha, fj_*). --against-fig4 reads only its
    config.json."""
    d = os.path.join(root, tag)
    os.makedirs(d, exist_ok=True)
    cfg = {
        # n_rounds 30: the replication now runs the ARCHIVED horizon, so
        # --against-fig4 asserts equality here instead of exempting it
        "run_tag": tag, "base_model": HF[slug], "seed": 0, "n_rounds": 30,
        "dataset": "movielens", "ml_target": "Action", "n_labeled": 723,
        "train_cap": 723, "training_style": "sft_kl", "kl_beta": 1.0,
        "kl_direction": "forward", "kl_ref_adapter": "",
        "anchor_mode": "fixed", "w_plat": 0.5, "innate_lambda": 0.2,
        "gamma_bias": 0.0, "eps": 0.05, "eps_ai": 1.0,
        "ai_gate_mode": "threshold", "population_update": POP_V1,
        "pop_model": "ab", "ab_sweeps": 1, "use_lora": True,
        "fresh_each_round": True, "icl_k": 0, "icl_days": 0,
        "lora_r": 512, "sft_lr": 5e-05, "sft_epochs": 1,
        "sft_batch_size": 4, "epoch_size": 100, "max_steps": 1,
        "seed_base_data": True, "save_raw_gen": True, "deploy_every": 1,
        "data_regime": "replace", "feedback_mode": "none", "icrh": False,
        "do_sample": False, "pristine_frac": 0.0, "replay_frac": 0.0,
        "teacher_label_delta": 0.0, "host": "g077",
        "hardware": {"hostname": "g077", "gpu_name": "NVIDIA H100",
                     "gpu_cc": "9.0", "cuda_version": "12.1",
                     "torch_version": "2.5.1",
                     "transformers_version": "4.46.0"},
    }
    if slug == "qwen3_8b":
        cfg["chat_thinking"] = False
    for k in drop:
        cfg.pop(k, None)
    if extra:
        cfg.update(extra)
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    return d


def _run_check(*flags, dirs=(), pin=False):
    """Drive the gate. --allow-partial and --no-innate-pin by default:
    the per-cell tests interrogate ONE cell at a time (coverage has its
    own test) and a synthetic innate cannot match the canonical sha."""
    cmd = [sys.executable, CHECK, "--allow-partial"]
    if not pin:
        cmd.append("--no-innate-pin")
    cmd += list(flags) + list(dirs)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "USE_TF": "0",
                               "OMP_NUM_THREADS": "1"})


def _out(r):
    return (r.stdout or "") + (r.stderr or "")


def _assert_fail(r, why):
    assert r.returncode != 0, (
        f"gate PASSED a run it must reject ({why}).\n{_out(r)[-3000:]}")


def _assert_pass(r, why):
    assert r.returncode == 0, (
        f"gate REJECTED a run it must accept ({why}).\n{_out(r)[-3000:]}")


# ================================================================ 1. grammar
@needs_files
def test_tag_parser_accepts_the_full_18_cell_product():
    CF = _load("check_fig4_repl", CHECK)
    grid = CF.full_grid()
    assert len(grid) == 18
    assert set(grid) == {(m, s) for m in MODEL_ORDER for s in SEEDS}
    for slug, seed in grid:
        tag = TAG.format(slug=slug, seed=seed)
        assert CF.expected_tag(slug, seed) == tag
        slot, errs = CF.parse_tag(tag)
        assert errs == [], (tag, errs)
        assert slot["slug"] == slug and slot["seed"] == seed
        assert slot["rounds"] == ROUNDS
        assert slot["base_model"] == HF[slug]


@needs_files
@pytest.mark.parametrize("tag", [
    # a foreign wave
    "pofdfam_qwen7b_b1_ea1_w0p5_l0p2_es0p05_s0",
    # the unregularized arm wearing this wave's prefix
    "pofdf4r_qwen7b_b0_anch2_ea1_w0p5_l0p2_es0p05_r30_s0",
    # the OLD operator token
    "pofdf4r_qwen7b_b1_x0_ea1_w0p5_l0p2_es0p05_r30_s0",
    # the ABANDONED 100-round draft of this very wave -- the horizon is
    # part of the identity, and a 100-round cell is not a 30-round one
    "pofdf4r_qwen7b_b1_anch2_ea1_w0p5_l0p2_es0p05_r100_s0",
    # a truncated run claiming a production slot
    "pofdf4r_qwen7b_b1_anch2_ea1_w0p5_l0p2_es0p05_r10_s0",
    # a seed outside the wave
    "pofdf4r_qwen7b_b1_anch2_ea1_w0p5_l0p2_es0p05_r30_s44",
    # a checkpoint that is not in the family
    "pofdf4r_llama8b_b1_anch2_ea1_w0p5_l0p2_es0p05_r30_s0",
    # a different environment wearing the grammar
    "pofdf4r_qwen7b_b1_anch2_ea1_w1_l0p2_es0p05_r30_s0",
    "pofdf4r_qwen7b_b1_anch2_ea0p4_w0p5_l0p2_es0p05_r30_s0",
])
def test_tag_parser_rejects_everything_that_is_not_this_condition(tag):
    CF = _load("check_fig4_repl", CHECK)
    slot, errs = CF.parse_tag(tag)
    assert slot is None and errs, f"parser accepted {tag!r}"


# =============================================================== 2. baseline
@needs_files
def test_a_well_formed_cell_passes(tmp_path):
    """If this fails, every FAIL assertion below is meaningless."""
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=0))
    _assert_pass(_run_check(dirs=[d]), "well-formed production cell")


# =============================================================== 3. operator
@needs_files
def test_v1_operator_run_fails_loudly_naming_the_old_operator(tmp_path):
    """The archived pofdfam_ cells carry v1. Under a NUMERIC AI gate v1
    and v2 are different arithmetic, so a v1 run here is the old
    experiment wearing the new tag."""
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=0),
                pop_update=POP_V1, gate_ref="x0")
    r = _run_check(dirs=[d])
    _assert_fail(r, "v1 round operator")
    assert POP_V1 in _out(r), "the failure must NAME the old operator"


@needs_files
def test_v2_marker_with_an_x0_gate_reference_fails(tmp_path):
    """Half-corrected provenance: the marker says v2 but the run gated
    on x0. Both fields have to agree or the artifact is self-inconsistent."""
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=0),
                pop_update=POP_V2, gate_ref="x0")
    _assert_fail(_run_check(dirs=[d]), "ai_gate_reference=x0 under a v2 marker")


@needs_files
def test_a_config_with_no_ai_gate_reference_fails(tmp_path):
    """Absent means 'written before 2026-08-22', i.e. not this tree."""
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=0))
    p = os.path.join(d, "config.json")
    cfg = json.load(open(p))
    cfg.pop("ai_gate_reference")
    json.dump(cfg, open(p, "w"))
    _assert_fail(_run_check(dirs=[d]), "no ai_gate_reference recorded")


# ================================================================= 4. models
@needs_files
def test_wrong_model_id_fails(tmp_path):
    """The slug says OLMo 3; the run loaded OLMo 2. Nothing downstream
    would reveal it -- the trajectory has the same shape either way."""
    d = _mk_run(str(tmp_path), TAG.format(slug="olmo3_7b", seed=0),
                slug="olmo3_7b", base_model=HF["olmo7b"])
    r = _run_check(dirs=[d])
    _assert_fail(r, "base_model does not match the slug")
    assert "base_model" in _out(r)


@needs_files
def test_every_slug_must_carry_its_own_exact_hf_id(tmp_path):
    for slug in MODEL_ORDER:
        root = str(tmp_path / slug)
        os.makedirs(root, exist_ok=True)
        ok = _mk_run(root, TAG.format(slug=slug, seed=0), slug=slug)
        _assert_pass(_run_check(dirs=[ok]), f"correct id for {slug}")
        bad_root = str(tmp_path / f"{slug}_bad")
        os.makedirs(bad_root, exist_ok=True)
        bad = _mk_run(bad_root, TAG.format(slug=slug, seed=0), slug=slug,
                      base_model="mistralai/Mistral-7B-Instruct-v0.2")
        _assert_fail(_run_check(dirs=[bad]), f"wrong id for {slug}")


@needs_files
def test_qwen3_must_record_thinking_off(tmp_path):
    on = _mk_run(str(tmp_path / "on"), TAG.format(slug="qwen3_8b", seed=0),
                 slug="qwen3_8b", chat_thinking=True)
    _assert_fail(_run_check(dirs=[on]), "Qwen3 with thinking ON")
    absent = _mk_run(str(tmp_path / "absent"),
                     TAG.format(slug="qwen3_8b", seed=0), slug="qwen3_8b",
                     chat_thinking=None)
    _assert_fail(_run_check(dirs=[absent]),
                 "Qwen3 with no chat_thinking recorded")
    off = _mk_run(str(tmp_path / "off"), TAG.format(slug="qwen3_8b", seed=0),
                  slug="qwen3_8b", chat_thinking=False)
    _assert_pass(_run_check(dirs=[off]), "Qwen3 with thinking OFF")


# ================================================================== 5. seeds
@needs_files
def test_config_seed_must_match_the_tag_seed(tmp_path):
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=42), seed=0)
    _assert_fail(_run_check(dirs=[d]), "tag s42 over a seed-0 config")


# =========================================== 6. shared population and graph
@needs_files
def test_shared_innate_across_models_fails_when_innate_differs(tmp_path):
    """SEED_BASE_DATA=1 builds the SAME population and graph for every
    checkpoint, so at one seed the six innate vectors must be
    bit-identical. Two worlds means the six panels are not comparable."""
    root = str(tmp_path)
    shared = _innate()
    good = [_mk_run(root, TAG.format(slug=s, seed=0), slug=s,
                    innate=shared.clone(),
                    op=_op_from_means([0.5 + 0.001 * i for i in range(ROUNDS)],
                                      seed=hash(s) % 1000))
            for s in ("qwen7b", "qwen3_8b")]
    _assert_pass(_run_check(dirs=good), "two checkpoints on one world")

    other = str(tmp_path / "split")
    os.makedirs(other, exist_ok=True)
    a = _mk_run(other, TAG.format(slug="qwen7b", seed=0), slug="qwen7b",
                innate=shared.clone())
    b = _mk_run(other, TAG.format(slug="olmo7b", seed=0), slug="olmo7b",
                innate=_innate(seed=999))
    r = _run_check(dirs=[a, b])
    _assert_fail(r, "two different innate vectors at one seed")
    assert "bit-identical" in _out(r)


@needs_files
def test_two_seeds_with_a_bit_identical_trajectory_fail(tmp_path):
    """The seed feeds the peer and training RNG. Two seeds producing the
    SAME 30x723 op_raw means the seed never reached the runner and the
    three-seed claim is one run counted three times."""
    root = str(tmp_path)
    shared = _innate()
    op = _op_from_means([0.5] * ROUNDS)
    a = _mk_run(root, TAG.format(slug="qwen7b", seed=0), seed=0,
                innate=shared.clone(), op=op.clone())
    b = _mk_run(root, TAG.format(slug="qwen7b", seed=42), seed=42,
                innate=shared.clone(), op=op.clone())
    r = _run_check(dirs=[a, b])
    _assert_fail(r, "bit-identical trajectories at two seeds")
    assert "BIT-IDENTICAL" in _out(r)


@needs_files
def test_the_innate_sha_pin_accepts_the_named_vector_and_rejects_another(
        tmp_path):
    inn = _innate()
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=0),
                innate=inn.clone())
    good = _sha_t(inn)
    _assert_pass(_run_check("--innate-sha", good, dirs=[d], pin=True),
                 "innate matching the named sha")
    wrong = _sha_t(_innate(seed=4321))
    _assert_fail(_run_check("--innate-sha", wrong, dirs=[d], pin=True),
                 "innate not matching the named sha")


# ================================================================ 7. horizon
@needs_files
def test_a_non_30_round_count_fails(tmp_path):
    """The horizon is 30. A LONGER run is just as wrong as a shorter one
    -- the wave replicates the published cell at its own horizon, and a
    100-round cell is the abandoned draft of this wave."""
    long_run = _mk_run(str(tmp_path / "a"), TAG.format(slug="qwen7b", seed=0),
                       rounds=100, n_rounds=100)
    r = _run_check(dirs=[long_run])
    _assert_fail(r, "a 100-round run in a 30-round slot")
    assert "n_rounds" in _out(r) or "op_raw shape" in _out(r)

    short = _mk_run(str(tmp_path / "c"), TAG.format(slug="qwen7b", seed=0),
                    rounds=10, n_rounds=10)
    _assert_fail(_run_check(dirs=[short]), "a truncated 10-round run")

    lying = _mk_run(str(tmp_path / "b"), TAG.format(slug="qwen7b", seed=0),
                    rounds=10, n_rounds=30)
    _assert_fail(_run_check(dirs=[lying]),
                 "config says 30 rounds, tensors carry 10")


@needs_files
def test_an_empty_twin_fails(tmp_path):
    """WITH_TWIN is NOT a config field -- a non-empty twin_raw is the
    only evidence the matched no-platform twin was simulated, and the
    analyzer measures W1 to it."""
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=0),
                twin=torch.empty(0))
    r = _run_check(dirs=[d])
    _assert_fail(r, "empty twin_raw")
    assert "twin_raw" in _out(r)


# ============================================================== 8. the parse
@needs_files
def test_a_nonzero_parse_fail_frac_fails(tmp_path):
    pf = [0.0] * ROUNDS
    pf[17] = 0.0013
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=0),
                parse_fail=pf)
    r = _run_check(dirs=[d])
    _assert_fail(r, "one round with a nonzero parse_fail_frac")
    assert "round 17" in _out(r)


@needs_files
def test_parse_fail_is_read_from_the_gz_not_the_trajectory(tmp_path):
    """THE PROVENANCE TRAP. trajectory.pt carries a CLEAN 0.0 on every
    row; raw_gen_log.json.gz carries the real, dirty rate. A gate that
    read the trajectory would pass this run."""
    pf = [0.0] * ROUNDS
    pf[3] = 0.5
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=0),
                parse_fail=pf, traj_parse_fail=0.0)
    _assert_fail(_run_check(dirs=[d]),
                 "dirty parse rate in the gz, clean one in trajectory.pt")


@needs_files
def test_a_missing_raw_gen_log_fails(tmp_path):
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=0),
                skip_files=("raw_gen_log.json.gz",))
    _assert_fail(_run_check(dirs=[d]),
                 "no raw_gen_log.json.gz -- the parse rate is not "
                 "establishable")


@needs_files
def test_incomplete_serving_fails(tmp_path):
    """parse_fail_frac 0 but only 700 of 723 agents parsed."""
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=0))
    gz = os.path.join(d, "raw_gen_log.json.gz")
    rows = [json.loads(x) for x in gzip.open(gz, "rt")]
    rows[9]["parsed"] = rows[9]["parsed"][:700]
    with gzip.open(gz, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    _assert_fail(_run_check(dirs=[d]), "700 of 723 agents parsed")


# =========================================================== 9. grid fields
@needs_files
@pytest.mark.parametrize("field,value", [
    ("eps_ai", 0.4),                 # a different AI gate
    ("ai_gate_mode", "all_open"),    # not the numeric gate the figure ran
    ("eps", 0.2),                    # a different peer gate
    ("w_plat", 1.0),                 # a different beta
    ("innate_lambda", 1.0),          # a different anchor k
    ("gamma_bias", 0.5),             # homophily selection bias, pinned 0
    ("training_style", "sft"),       # the unregularized arm
    ("kl_beta", 8.0),                # a different lambda
    ("kl_direction", "reverse"),     # the RLHF-practice direction
    ("icl_k", 32),                   # an in-context arm
    ("icl_days", 5),
    ("use_lora", False),
    ("fresh_each_round", False),
    ("lora_r", 64),
    ("dataset", "pokec"),
    ("ml_target", "Romance"),
    ("n_labeled", 500),
    ("train_cap", 500),
    ("save_raw_gen", False),
    ("peer_gate_mode", "all_open"),
])
def test_every_grid_dial_is_pinned(tmp_path, field, value):
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=0),
                extra_cfg={field: value})
    r = _run_check(dirs=[d])
    _assert_fail(r, f"{field}={value!r}")
    assert field in _out(r)


@needs_files
def test_a_learner_that_never_trained_fails(tmp_path):
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=0),
                grad_norm0=0.0)
    _assert_fail(_run_check(dirs=[d]), "grad_norm0 zero in every round")


@needs_files
def test_a_dead_anchor_gradient_fails(tmp_path):
    """lambda = 1 recorded but the forward-KL term never contributed."""
    d = _mk_run(str(tmp_path), TAG.format(slug="qwen7b", seed=0),
                grad_kl0=[0.0] * ROUNDS)
    _assert_fail(_run_check(dirs=[d]), "anchor gradient dead after round 0")


# ========================================================= 10. --against-fig4
@needs_files
def test_against_fig4_tolerates_exactly_the_allowed_differences(tmp_path):
    """seed (same here), the operator fields, and the run_tag that is a
    pure function of them. Plus the fields the runner GAINED after
    2026-08-17, each registered. n_rounds is NOT on that list any more:
    the horizons match, so 30 == 30 and the comparison asserts it."""
    root = str(tmp_path)
    _mk_archived(root, FIG4_B1.format(slug="qwen7b"))
    new = _mk_run(root, TAG.format(slug="qwen7b", seed=0))
    r = _run_check("--against-fig4", "--run-root", root, dirs=[new])
    _assert_pass(r, "only the allowed differences")
    out = _out(r)
    assert "EXPECTED" in out and "population_update" in out
    assert "serve_eval_mode" in out, (
        "the one NON-neutral code delta must be printed, not buried")


@needs_files
def test_against_fig4_hard_fails_a_moving_horizon(tmp_path):
    """n_rounds used to be an EXPECTED difference (100 vs 30). The user
    cut the wave to the archived horizon, so equality is now the point
    of the comparison and a difference is a hard failure with its own
    message."""
    root = str(tmp_path)
    _mk_archived(root, FIG4_B1.format(slug="qwen7b"), extra={"n_rounds": 100})
    new = _mk_run(root, TAG.format(slug="qwen7b", seed=0))
    r = _run_check("--against-fig4", "--run-root", root, dirs=[new])
    _assert_fail(r, "archived 100 rounds vs replicated 30")
    out = _out(r)
    assert "HORIZON MISMATCH" in out
    assert "n_rounds" in out and "old=100" in out and "new=30" in out


@needs_files
def test_against_fig4_expected_bucket_no_longer_holds_n_rounds():
    CF = _load("check_fig4_repl", CHECK)
    assert "n_rounds" not in CF.AGAINST_EXPECTED_DIFF, (
        "the horizons match now; n_rounds must be asserted equal, not "
        "exempted")
    assert set(CF.AGAINST_EXPECTED_DIFF) == {
        "seed", "population_update", "ai_gate_reference", "run_tag"}
    assert CF.ROUNDS == ROUNDS
    assert (CF.LATE_LO, CF.LATE_HI) == (LATE_LO, LATE_HI)


@needs_files
def test_against_fig4_catches_an_unexpected_changed_field(tmp_path):
    """A field present in BOTH configs with a different value, and not on
    the allowed list. Here the SFT batch size -- invisible in every
    downstream artifact."""
    root = str(tmp_path)
    _mk_archived(root, FIG4_B1.format(slug="qwen7b"),
                 extra={"sft_batch_size": 4})
    new = _mk_run(root, TAG.format(slug="qwen7b", seed=0),
                  extra_cfg={"sft_batch_size": 16})
    r = _run_check("--against-fig4", "--run-root", root, dirs=[new])
    _assert_fail(r, "sft_batch_size changed silently")
    out = _out(r)
    assert "sft_batch_size" in out and "old=4" in out and "new=16" in out


@needs_files
def test_against_fig4_catches_an_unregistered_one_sided_field(tmp_path):
    """A field the new config carries and the archived one does not, with
    nothing in ONE_SIDED_REGISTRY saying why. Waiving it silently is how
    a surface change ships unnoticed."""
    root = str(tmp_path)
    _mk_archived(root, FIG4_B1.format(slug="qwen7b"))
    new = _mk_run(root, TAG.format(slug="qwen7b", seed=0),
                  extra_cfg={"sft_sample_n": 200})
    r = _run_check("--against-fig4", "--run-root", root, dirs=[new])
    _assert_fail(r, "unregistered one-sided field sft_sample_n")
    assert "sft_sample_n" in _out(r)


@needs_files
def test_against_fig4_marks_an_absent_archived_cell_skipped(tmp_path):
    """Never a silent pass: the checkpoint is named SKIPPED and the run
    still says the published condition was not compared for it."""
    root = str(tmp_path)
    new = _mk_run(root, TAG.format(slug="qwen7b", seed=0))
    r = _run_check("--against-fig4", "--run-root", root, dirs=[new])
    out = _out(r)
    assert "SKIPPED" in out
    assert FIG4_B1.format(slug="qwen7b") in out


@needs_files
def test_compare_configs_buckets_every_difference(tmp_path):
    CF = _load("check_fig4_repl", CHECK)
    old = {"seed": 0, "n_rounds": 30, "population_update": POP_V1,
           "run_tag": "old", "host": "g077", "w_plat": 0.5}
    new = {"seed": 0, "n_rounds": 30, "population_update": POP_V2,
           "ai_gate_reference": "anchor", "run_tag": "new", "host": "g191",
           "serve_eval_mode": True, "git_sha": "abc", "w_plat": 0.5}
    hard, exp, env, delta = CF.compare_configs(old, new, "qwen7b")
    assert hard == [], hard
    assert {f for f, *_ in exp} == {"population_update", "run_tag"}
    # git_sha is environment, not a code delta: it is one-sided here only
    # because the archived config predates the field, and either way it
    # names the tree, not the experiment.
    assert {f for f, *_ in env} == {"host", "git_sha"}
    assert {f for f, *_ in delta} == {"ai_gate_reference",
                                      "serve_eval_mode"}
    # w_plat and n_rounds are identical and must appear in no bucket
    for field in ("w_plat", "n_rounds"):
        assert all(field not in {f for f, *_ in b}
                   for b in (hard, exp, env, delta)), field

    # and the horizon clause fires in BOTH directions
    hard2, *_ = CF.compare_configs({**old, "n_rounds": 100}, new, "qwen7b")
    assert [f for f, *_ in hard2] == ["n_rounds"], hard2
    hard3, *_ = CF.compare_configs({**old, "n_rounds": 100},
                                   {**new, "n_rounds": 100}, "qwen7b")
    assert [f for f, *_ in hard3] == ["n_rounds"], (
        "both agreeing on the wrong horizon is still wrong")


# ======================================================== 11. grid coverage
@needs_files
def test_partial_coverage_is_a_hard_failure_by_default(tmp_path):
    """Never silently pass on partial coverage: the missing cells are
    named by expected tag and the run exits non-zero without
    --allow-partial."""
    root = str(tmp_path)
    shared = _innate()
    # written to disk and then found by the ROOT SCAN, not passed as
    # explicit dirs -- coverage is exactly what a root scan has to report
    for s in SEEDS:
        _mk_run(root, TAG.format(slug="qwen7b", seed=s), seed=s,
                innate=shared.clone(),
                op=_op_from_means([0.5 + 0.0001 * s * t
                                   for t in range(ROUNDS)]))
    cmd = [sys.executable, CHECK, "--no-innate-pin", "--run-root", root]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "USE_TF": "0"})
    assert r.returncode != 0, "3 of 18 cells must not pass by default"
    out = _out(r)
    assert "15 MISSING" in out
    assert TAG.format(slug="olmo3_7b", seed=43) in out, (
        "every missing cell must be named by its expected tag")

    r2 = subprocess.run(cmd + ["--allow-partial"], capture_output=True,
                        text=True, cwd=ROOT,
                        env={**os.environ, "USE_TF": "0"})
    assert r2.returncode == 0
    assert "PARTIAL GRID" in _out(r2)
    assert "3/18" in _out(r2)


@needs_files
def test_a_complete_18_cell_grid_passes(tmp_path):
    root = str(tmp_path)
    shared = _innate()
    for i, slug in enumerate(MODEL_ORDER):
        for j, seed in enumerate(SEEDS):
            _mk_run(root, TAG.format(slug=slug, seed=seed), slug=slug,
                    seed=seed, innate=shared.clone(),
                    op=_op_from_means(
                        [0.4 + 0.01 * i + 0.002 * j + 0.0001 * t
                         for t in range(ROUNDS)], seed=100 * i + j))
    cmd = [sys.executable, CHECK, "--no-innate-pin", "--run-root", root]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "USE_TF": "0"})
    _assert_pass(r, "the complete 18-cell grid")
    assert "all 18 cells present" in _out(r)


# ================================================ 12. analyzer arithmetic
@needs_files
def test_t_crit_and_seed_agg_arithmetic():
    AN = _load("analyze_fig4_repl", ANALYZE)
    assert abs(AN.t_crit(2) - T2) < 1e-9, AN.t_crit(2)
    assert AN.T_CRIT_SOURCE, "the t source must be recorded, not implied"

    vals = [0.60, 0.62, 0.64]
    n, mean, sd, se, tc, lo, hi = AN.seed_agg(vals)
    assert n == 3
    assert abs(mean - 0.62) < 1e-12
    want_sd = (((0.60 - 0.62) ** 2 + 0.0 + (0.64 - 0.62) ** 2) / 2) ** 0.5
    assert abs(sd - want_sd) < 1e-12
    want_se = want_sd / 3 ** 0.5
    assert abs(se - want_se) < 1e-12
    assert abs(tc - T2) < 1e-9
    assert abs(lo - (0.62 - T2 * want_se)) < 1e-12
    assert abs(hi - (0.62 + T2 * want_se)) < 1e-12

    # n = 1 gets a mean and NO interval -- a zero-width one would be a lie
    n1, m1, sd1, se1, t1, lo1, hi1 = AN.seed_agg([0.5])
    assert n1 == 1 and m1 == 0.5
    assert sd1 is None and lo1 is None and hi1 is None
    # n = 2 falls back to df = 1, not to the n = 3 constant
    n2, _, _, _, t2, _, _ = AN.seed_agg([0.5, 0.6])
    assert n2 == 2 and abs(t2 - 12.706204736432095) < 1e-6


@needs_files
def test_convergence_splits_the_ten_round_window_into_fives():
    """The real window is 10 rounds (21-30), so the halves are FIVE
    rounds each -- 21-25 vs 26-30. Every case below is at that length,
    because that is what actually runs."""
    AN = _load("analyze_fig4_repl", ANALYZE)
    tol = 0.002

    flat = [0.5] * WIN_LEN
    c = AN.convergence(flat, tol)
    assert c["settled"] and c["n"] == WIN_LEN
    assert abs(c["first_half"] - 0.5) < 1e-12
    assert abs(c["second_half"] - 0.5) < 1e-12

    # a two-round excursion INSIDE one half cancels out of D
    wig = [0.5] * WIN_LEN
    wig[6], wig[7] = 0.5 + 0.008, 0.5 - 0.008
    c = AN.convergence(wig, tol)
    assert c["settled"], c
    assert abs(c["drift"]) < 1e-12
    assert c["range"] > 0.0, "the window is not flat; only D cancels"
    # 3 of 9 consecutive steps are non-zero, so the MEDIAN step is 0 --
    # the noise floor is a reported margin, never a gate
    assert c["noise_floor"] == 0.0

    # THE HONEST WEAKENING at this horizon: the SAME excursion straddling
    # the halves moves D by 2/5 of its size and now fires. At a 100-round
    # window it would have moved D by 2/10 and stayed inside tol.
    straddle = [0.5] * WIN_LEN
    straddle[HALF - 1], straddle[HALF] = 0.5 + 0.008, 0.5 - 0.008
    c = AN.convergence(straddle, tol)
    assert not c["settled"], c
    assert abs(c["drift"] - 2 * 0.008 / HALF) < 1e-12

    # a real trend, caught by BOTH tests
    trend = [0.5 + 0.001 * t for t in range(WIN_LEN)]
    c = AN.convergence(trend, tol)
    assert not c["settled"]
    assert abs(c["drift"] - 0.001 * HALF) < 1e-9
    assert abs(c["trend"] - 0.001 * WIN_LEN) < 1e-9

    # a slope small enough that D passes but T does not: for a straight
    # line D = slope*HALF and T = slope*WIN_LEN, so this is the band the
    # fitted trend exists to cover
    slow = [0.5 + 0.00035 * t for t in range(WIN_LEN)]
    c = AN.convergence(slow, tol)
    assert c["drift"] <= tol, c
    assert c["trend"] > tol, c
    assert not c["settled"], "T alone must be able to fail a cell"


@needs_files
def test_w1_is_the_sorted_mean_absolute_difference():
    AN = _load("analyze_fig4_repl", ANALYZE)
    assert abs(AN.w1([0.0, 1.0], [0.0, 1.0])) < 1e-12
    assert abs(AN.w1([0.0, 0.0], [1.0, 1.0]) - 1.0) < 1e-12
    assert abs(AN.w1([1.0, 0.0], [0.0, 1.0])) < 1e-12   # order-free


# ================================================= 13. analyzer end to end
@needs_files
def test_analyzer_aggregates_the_tail_window_over_three_seeds(tmp_path):
    """The rounds-21-30 aggregation, end to end, against a fixture whose
    window mean is known in closed form."""
    root = tmp_path / "runs"
    root.mkdir()
    shared = _innate()
    # per seed, a window mean chosen so the three-seed aggregate is exact
    want = {0: 0.60, 42: 0.62, 43: 0.64}
    for seed in SEEDS:
        means = [0.30] * (LATE_LO - 1) + [want[seed]] * WIN_LEN
        _mk_run(str(root), TAG.format(slug="qwen7b", seed=seed),
                slug="qwen7b", seed=seed, innate=shared.clone(),
                op=_op_from_means(means, spread=0.05, seed=seed + 1))
    out_dir = tmp_path / "analysis"
    r = subprocess.run(
        [sys.executable, ANALYZE, "--run-root", str(root),
         "--out-dir", str(out_dir), "--no-figs"],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "USE_TF": "0", "OMP_NUM_THREADS": "1"})
    assert r.returncode == 0, _out(r)[-3000:]

    import csv as _csv
    rows = list(_csv.DictReader(
        open(out_dir / "fig4_repl_equilibrium.csv")))
    by_model = {x["model"]: x for x in rows}
    q = by_model["qwen7b"]
    assert int(q["n_seeds"]) == 3
    assert abs(float(q["pop_mean_mean"]) - 0.62) < 1e-6, q["pop_mean_mean"]
    want_sd = (((0.60 - 0.62) ** 2 + (0.64 - 0.62) ** 2) / 2) ** 0.5
    want_se = want_sd / 3 ** 0.5
    assert abs(float(q["pop_mean_ci_lo"]) - (0.62 - T2 * want_se)) < 1e-6
    assert abs(float(q["pop_mean_ci_hi"]) - (0.62 + T2 * want_se)) < 1e-6
    assert abs(float(q["pop_mean_t"]) - T2) < 1e-6
    # per-round SD is the fixture's fixed spread, flat across the window
    assert abs(float(q["pop_sd_mean"]) - 0.05) < 1e-3
    # the window is flat by construction => settled on both quantities
    assert int(q["n_settled_both"]) == 3
    assert q["state_label"].startswith("equilibrium")
    assert q["late_lo"] == str(LATE_LO) and q["late_hi"] == str(LATE_HI)

    # the horizon caveat and the two-differences sentence must be in the
    # verdict text even though every cell settled
    out = _out(r)
    assert "HORIZON CAVEAT" in out
    assert "NOT ESTABLISHED BY THAT" in out, (
        "an all-settled grid must not be reported as 'the horizon was "
        "enough'")
    assert "serve_eval_mode" in out and "LoRA dropout" in out

    # per-round CSV: exactly the specified columns, 100 rows per cell
    with open(out_dir / "fig4_repl_per_round.csv") as fh:
        rd = _csv.DictReader(fh)
        assert rd.fieldnames == ["model", "seed", "round", "mean", "sd",
                                 "w1_to_twin", "w1_to_innate",
                                 "served_mean"]
        pr = list(rd)
    assert len(pr) == 3 * ROUNDS
    assert {int(x["round"]) for x in pr} == set(range(1, ROUNDS + 1))
    last = [x for x in pr
            if int(x["round"]) == ROUNDS and x["seed"] == "0"][0]
    assert abs(float(last["mean"]) - 0.60) < 1e-6
    assert float(last["w1_to_twin"]) > 0.0


@needs_files
def test_analyzer_reports_partial_coverage_and_never_averages_silently(
        tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    shared = _innate()
    for seed in (0, 42):                       # seed 43 deliberately absent
        _mk_run(str(root), TAG.format(slug="qwen7b", seed=seed),
                slug="qwen7b", seed=seed, innate=shared.clone(),
                op=_op_from_means([0.5 + 0.001 * seed] * ROUNDS,
                                  seed=seed + 3))
    out_dir = tmp_path / "analysis"
    r = subprocess.run(
        [sys.executable, ANALYZE, "--run-root", str(root),
         "--out-dir", str(out_dir), "--no-figs"],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "USE_TF": "0", "OMP_NUM_THREADS": "1"})
    assert r.returncode == 0, _out(r)[-3000:]
    out = _out(r)
    assert "PARTIAL" in out
    assert TAG.format(slug="qwen7b", seed=43) in out, (
        "the missing cell must be named by expected tag")
    assert TAG.format(slug="olmo3_7b", seed=0) in out

    import csv as _csv
    rows = {x["model"]: x for x in _csv.DictReader(
        open(out_dir / "fig4_repl_equilibrium.csv"))}
    q = rows["qwen7b"]
    assert int(q["n_seeds"]) == 2 and q["complete"] == "0"
    # df = 1 at n = 2, NOT the n = 3 constant
    assert abs(float(q["pop_mean_t"]) - 12.706204736432095) < 1e-3
    assert rows["olmo7b"]["state_label"] == "ABSENT"


@needs_files
def test_analyzer_writes_titleless_figures_and_prints_captions(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    shared = _innate()
    for seed in SEEDS:
        _mk_run(str(root), TAG.format(slug="qwen7b", seed=seed),
                slug="qwen7b", seed=seed, innate=shared.clone(),
                op=_op_from_means([0.5 + 0.0005 * seed] * ROUNDS,
                                  seed=seed + 5))
    # the frozen control the figure draws as "entering model"
    _mk_run(str(root), FIG4_K0.format(slug="qwen7b"), slug="qwen7b",
            rounds=30, n_rounds=30, innate=shared.clone())
    out_dir = tmp_path / "analysis"
    r = subprocess.run(
        [sys.executable, ANALYZE, "--run-root", str(root),
         "--out-dir", str(out_dir)],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "USE_TF": "0", "OMP_NUM_THREADS": "1"})
    assert r.returncode == 0, _out(r)[-3000:]
    for stem in ("fig4_repl_distributions", "fig4_repl_convergence"):
        for ext in (".pdf", ".png"):
            assert (out_dir / f"{stem}{ext}").exists(), stem
    out = _out(r)
    assert out.count("CAPTION --") == 2, "one caption block per figure"
    assert "carries NO title text" in out
    # BOTH captions must carry the two-differences sentence
    # once in EACH caption block (it also appears in the convergence
    # report above them, which is why this splits rather than counts)
    blocks = out.split("CAPTION --")[1:]
    assert len(blocks) == 2
    for b in blocks:
        assert "no shift between the two may be attributed" in b
    assert "MUCH WEAKER" in out, (
        "the caption must say a 30-round tail is a weaker equilibrium "
        "claim than a 100-round one")
    assert (out_dir / "fig4_repl_convergence.txt").exists()

    # the W1-to-frozen column must be populated when the control is there
    import csv as _csv
    rows = {x["model"]: x for x in _csv.DictReader(
        open(out_dir / "fig4_repl_equilibrium.csv"))}
    assert rows["qwen7b"]["frozen_status"] == "OK"
    assert float(rows["qwen7b"]["w1_frozen_mean"]) >= 0.0
    # and SKIPPED, by name, where it is not
    assert rows["olmo7b"]["frozen_status"] == "SKIPPED"


@needs_files
def test_the_tail_window_is_a_flag_and_the_default_is_the_last_ten(
        tmp_path):
    """The window is a CHOICE, so it must be visible and movable. The
    default is the last 10 rounds; --window-start/--window-end move it
    and every CSV row records what was actually used."""
    root = tmp_path / "runs"
    root.mkdir()
    shared = _innate()
    # first 15 rounds at 0.30, last 15 at 0.70: the default 21-30 window
    # sees only 0.70, while 11-20 straddles the step
    means = [0.30] * 15 + [0.70] * 15
    for seed in SEEDS:
        _mk_run(str(root), TAG.format(slug="qwen7b", seed=seed),
                slug="qwen7b", seed=seed, innate=shared.clone(),
                op=_op_from_means(means, seed=seed + 21))

    import csv as _csv

    def _run_analyzer(out_name, *flags):
        out_dir = tmp_path / out_name
        r = subprocess.run(
            [sys.executable, ANALYZE, "--run-root", str(root),
             "--out-dir", str(out_dir), "--no-figs", *flags],
            capture_output=True, text=True, cwd=ROOT,
            env={**os.environ, "USE_TF": "0"})
        assert r.returncode == 0, _out(r)[-2500:]
        rows = {x["model"]: x for x in _csv.DictReader(
            open(out_dir / "fig4_repl_equilibrium.csv"))}
        return r, rows["qwen7b"]

    r, q = _run_analyzer("default")
    assert q["late_lo"] == str(LATE_LO) and q["late_hi"] == str(LATE_HI)
    assert abs(float(q["pop_mean_mean"]) - 0.70) < 1e-6
    assert f"rounds {LATE_LO}-{LATE_HI}" in _out(r)

    # a window that straddles the step: the mean moves and it stops
    # being settled -- proof the flag actually reaches the arithmetic
    r2, q2 = _run_analyzer("moved", "--window-start", "11",
                           "--window-end", "20")
    assert q2["late_lo"] == "11" and q2["late_hi"] == "20"
    assert abs(float(q2["pop_mean_mean"]) - 0.50) < 1e-6
    assert int(q2["n_settled_both"]) == 0
    assert q2["state_label"].startswith("late-round state (rounds 11-20)")

    # and the guard rails
    bad = subprocess.run(
        [sys.executable, ANALYZE, "--run-root", str(root),
         "--out-dir", str(tmp_path / "bad"), "--no-figs",
         "--window-start", "25", "--window-end", "40"],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "USE_TF": "0"})
    assert bad.returncode == 2 and "window-start" in _out(bad)
    tiny = subprocess.run(
        [sys.executable, ANALYZE, "--run-root", str(root),
         "--out-dir", str(tmp_path / "tiny"), "--no-figs",
         "--window-start", "28", "--window-end", "30"],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "USE_TF": "0"})
    assert tiny.returncode == 2, "a 3-round window is arithmetic, not evidence"


@needs_files
def test_the_json_verdict_carries_both_displayed_vs_replicated_deltas(
        tmp_path):
    """The gate reference AND the serving path. The archived Figure-4
    configs record no serve_eval_mode at all, so the published figure
    served with LoRA dropout live; nobody may read a shift as evidence
    about the gate alone."""
    root = tmp_path / "runs"
    root.mkdir()
    shared = _innate()
    for seed in SEEDS:
        _mk_run(str(root), TAG.format(slug="qwen7b", seed=seed),
                slug="qwen7b", seed=seed, innate=shared.clone(),
                op=_op_from_means([0.5] * ROUNDS, seed=seed + 31))
    out_json = tmp_path / "summary.json"
    r = subprocess.run(
        [sys.executable, ANALYZE, "--run-root", str(root),
         "--out-dir", str(tmp_path / "out"), "--no-figs",
         "--json", str(out_json)],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "USE_TF": "0"})
    assert r.returncode == 0, _out(r)[-2500:]
    v = json.loads(out_json.read_text())
    dvr = v["displayed_vs_replicated"]
    for needle in ("serve_eval_mode", "LoRA dropout", "gate reference",
                   "eval()"):
        assert needle in dvr, (needle, dvr)
    assert v["horizon_caveat"]
    assert "not 100" in v["horizon_caveat"]
    assert v["late_window"] == [LATE_LO, LATE_HI]
    assert v["late_window_halves"] == [f"{LATE_LO}-{LATE_LO + HALF - 1}",
                                       f"{LATE_LO + HALF}-{LATE_HI}"]
    assert v["rounds"] == ROUNDS
    # and it survives into the on-disk convergence report
    txt = (tmp_path / "out" / "fig4_repl_convergence.txt").read_text()
    assert "DISPLAYED vs REPLICATED" in txt
    assert "HORIZON CAVEAT" in txt


@needs_files
def test_no_source_file_sets_a_figure_title():
    """PAPER FIGURES CARRY NO TITLE TEXT. The analyzer must never reach
    for set_title / suptitle, whatever the panel needs to say."""
    src = open(ANALYZE).read()
    for banned in ("set_title(", "suptitle("):
        assert banned not in src, f"analyze_fig4_repl.py calls {banned}"


@needs_files
def test_the_analyzer_refuses_to_write_under_paper(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    _mk_run(str(root), TAG.format(slug="qwen7b", seed=0))
    r = subprocess.run(
        [sys.executable, ANALYZE, "--run-root", str(root),
         "--out-dir", str(tmp_path / "paper" / "figures"), "--no-figs"],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "USE_TF": "0"})
    assert r.returncode == 2, _out(r)[-2000:]
    assert "paper/" in _out(r)
