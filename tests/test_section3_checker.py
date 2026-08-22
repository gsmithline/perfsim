"""ADVERSARIAL tests for check_section3.py -- the Section-3 retention gate.

Every sabotage below is a run that LOOKS finished: config.json parses,
trajectory.pt is complete, the figure would render. The gate is the only
place these are still catchable.

THE FILE-PROVENANCE TRAP (the reason this file exists at all).
A previous checker in this repo read `l_init` / `grad_*` off the
per-round dicts inside trajectory.pt and `parse_fail_frac` off the same
place. Those keys are NOT there: the training telemetry lives in
telemetry.json (JSONL, one object per round) and the parse rate lives in
raw_gen_log.json.gz (gzipped JSONL). Reading trajectory.pt therefore
yielded `None` for every gate, every gate vacuously passed, and a broken
wave shipped. test_checker_reads_telemetry_and_rawgen_not_trajectory
plants CONTRADICTORY values in trajectory.pt and asserts the checker
follows the real files.

THE COLLAPSE POLICY (the other half).
A served map that is constant or coarse is a legitimate SCIENTIFIC
OUTCOME of this wave -- the whole question is how much prior survives,
and "none of it" is an answer. So collapse must NOT hard-fail the gate
the way it does in check_kl_direction.py, where the wave's premise was
heterogeneity. It must be REPORTED: distinct values, largest mode share,
top-3 mode share, effective modes, prediction SD.

Contract pins are duplicated here on purpose: a test that imports its
expectations from the thing it is testing tests nothing.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "experiments", "scripts", "cluster_pipelines")

N_AGENTS = 723
ROUNDS = 100
SMOKE_ROUNDS = 3
H100 = "NVIDIA H100 80GB HBM3"
QWEN25 = "Qwen/Qwen2.5-7B-Instruct"
QWEN3 = "Qwen/Qwen3-8B"
# What the runner actually writes for these cells. The tag's "anch2" token
# is a provenance marker with NO matching runner string -- see the S3
# block comment in gen_pofd_sweep.py. A gate that demands the config
# marker agree with the tag token fails all 46 production runs.
POP_UPDATE = "nested_ai_anchored_then_social_v2"

PROD = "pofds3_{m}_{arm}_eaopen_w{w}_k{k}_esopen_anch2_s0_r100"
SMOKE = "pofds3smk_qwen3_8b_revlam1_eaopen_w1_k1_esopen_anch2_s0_r3"


def _find_checker():
    for c in (os.path.join(PIPE, "check_section3.py"),
              os.path.join(ROOT, "experiments", "condor", "check_section3.py"),
              os.path.join(ROOT, "check_section3.py")):
        if os.path.exists(c):
            return c
    return None


CHECK = _find_checker()

# The gate pins the CANONICAL movielens-Action innate vector by sha256.
# A synthetic random vector fails that pin, which would make every
# "must FAIL" assertion below pass VACUOUSLY -- the run would be
# rejected for the fixture's sake, not for the sabotage's. So the real
# vector is borrowed from any locally-present CPU perfect-prediction
# artifact, and if none is present the whole file skips rather than
# reporting a false green. (No production data is REQUIRED: absence
# degrades to a skip, never to an error.)
CANONICAL_INNATE_SHA = (
    "be34f284f929e2198996a37b080c03eef5750e1917d90269cd3fde81a7b31b19")


def _load_canonical_innate():
    import glob
    import hashlib
    for p in sorted(glob.glob(os.path.join(
            ROOT, "notes", "pofd", "perfect_prediction", "*.pt"))):
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:                                    # noqa: BLE001
            continue
        t = d.get("innate") if isinstance(d, dict) else None
        if not torch.is_tensor(t) or t.numel() != N_AGENTS:
            continue
        a = t.detach().cpu().float().contiguous().numpy()
        if hashlib.sha256(a.tobytes()).hexdigest() == CANONICAL_INNATE_SHA:
            return t.detach().clone().float()
    return None


CANON_INNATE = _load_canonical_innate() if CHECK else None

needs_checker = pytest.mark.skipif(
    CHECK is None or CANON_INNATE is None,
    reason=("check_section3.py has not landed yet" if CHECK is None else
            "no local artifact carries the canonical 723-agent innate "
            "vector; without it every sabotage would be rejected for the "
            "fixture's sake and the assertions would be vacuous"))


# --------------------------------------------------------------- fixtures
def _innate(n=N_AGENTS, seed=1234):
    """The environment. IDENTICAL across every cell of the wave: the
    arms differ only in the learner, so a differing innate vector means
    the arms are not comparable at all.

    seed=1234 returns the CANONICAL vector; any other seed returns a
    deliberately different one, for the cross-cell mismatch test."""
    if seed == 1234 and CANON_INNATE is not None and n == N_AGENTS:
        return CANON_INNATE.clone()
    return torch.rand(n, generator=torch.Generator().manual_seed(seed))


def _mk_run(root, tag, *, model="qwen7b", style="sft_kl", kl_beta=1.0,
            kl_direction="forward", w_plat=0.5, innate_lambda=1.0,
            rounds=ROUNDS, n=N_AGENTS, seed=0, gpu=H100,
            kl_ref_adapter="", chat_thinking=None, n_rounds=None,
            parse_fail=0.0, pred=None, op=None, innate=None,
            grad_norm0=3.0, grad_kl0=None, l_init=None,
            pop_update=POP_UPDATE, traj_rows=None,
            anchor_mode="fixed", serve_eval_mode=True,
            skip_files=(), extra_cfg=None):
    """One synthetic run dir matching the VERIFIED artifact schema.

    config.json      flat dict
    trajectory.pt    {"trajectory", "config", "op_raw", "pred_raw",
                      "innate", "profiles", "twin_raw", "probe_idx"}
    telemetry.json   JSONL: round, l_init, grad_norm0, grad_kl_norm0,
                     grad_cos0, grad_ratio0, n_train
    raw_gen_log.json.gz  gzipped JSONL: round, parse_fail_frac, raw,
                     parsed
    """
    base = QWEN3 if model == "qwen3_8b" else QWEN25
    if chat_thinking is None:
        chat_thinking = 0 if model == "qwen3_8b" else "default"
    d = os.path.join(root, tag)
    os.makedirs(d, exist_ok=True)
    cfg = {
        "base_model": base, "seed": seed, "seed_base_data": 1,
        "dataset": "movielens", "ml_target": "Action", "n_labeled": n,
        "training_style": style, "kl_beta": kl_beta,
        "kl_direction": kl_direction, "kl_ref_adapter": kl_ref_adapter,
        "anchor_mode": anchor_mode, "w_plat": w_plat,
        "innate_lambda": innate_lambda, "ai_gate_mode": "all_open",
        "peer_gate_mode": "all_open", "eps": 0.2, "eps_ai": 1.0,
        "ab_sweeps": 1, "pop_model": "ab", "fresh_each_round": True,
        "use_lora": True, "lora_r": 512, "sft_lr": 5e-05,
        "sft_epochs": 1, "epoch_size": 100, "max_steps": None,
        "train_cap": 723, "icl_k": 0, "icl_days": 0,
        "n_rounds": rounds if n_rounds is None else n_rounds,
        "serve_eval_mode": serve_eval_mode,
        "population_update": pop_update, "chat_thinking": chat_thinking,
        "hardware": {"gpu_name": gpu},
    }
    if extra_cfg:
        cfg.update(extra_cfg)
    if "config.json" not in skip_files:
        with open(os.path.join(d, "config.json"), "w") as fh:
            json.dump(cfg, fh)

    g = torch.Generator().manual_seed(7)
    if innate is None:
        innate = _innate(n)
    if pred is None:
        pred = 0.2 + 0.6 * torch.rand(rounds, n, generator=g)
    if op is None:
        op = 0.2 + 0.6 * torch.rand(rounds, n, generator=g)
    if traj_rows is None:
        # peer_pairs / accepted are the cross-arm world fingerprint the
        # runner writes (run_pokec_gated_lm.py ~3413/3425). Under
        # all_open with one AB sweep every pair is accepted, so
        # accepted == peer_pairs == n.
        traj_rows = [{"round": t, "mean_op": float(op[t].mean()),
                      "peer_pairs": n, "accepted": n}
                     for t in range(op.shape[0])]
    if "trajectory.pt" not in skip_files:
        torch.save({"trajectory": traj_rows, "config": cfg,
                    "op_raw": op, "pred_raw": pred, "innate": innate,
                    "profiles": [{"i": i} for i in range(n)],
                    "twin_raw": op.clone(),
                    "probe_idx": list(range(min(32, n)))},
                   os.path.join(d, "trajectory.pt"))

    # telemetry.json -- the ONLY home of l_init and the gradient norms.
    # grad_kl_norm0 is ~0 at round 0 by construction: a fresh LoRA IS
    # the reference there, so the anchor term has nothing to pull on.
    if grad_kl0 is None:
        grad_kl0 = [0.0 if t == 0 else 1.5 for t in range(rounds)]
    if l_init is None:
        l_init = [2.0 / (t + 1) for t in range(rounds)]
    if isinstance(grad_norm0, (int, float)):
        grad_norm0 = [float(grad_norm0)] * rounds
    if "telemetry.json" not in skip_files:
        with open(os.path.join(d, "telemetry.json"), "w") as fh:
            for t in range(rounds):
                fh.write(json.dumps({
                    "round": t, "l_init": l_init[t],
                    "grad_norm0": grad_norm0[t],
                    "grad_kl_norm0": grad_kl0[t],
                    "grad_cos0": 0.1, "grad_ratio0": 0.3,
                    "n_train": 100}) + "\n")

    # raw_gen_log.json.gz -- the ONLY home of parse_fail_frac.
    if isinstance(parse_fail, (int, float)):
        parse_fail = [float(parse_fail)] * rounds
    if "raw_gen_log.json.gz" not in skip_files:
        with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt") as fh:
            for t in range(rounds):
                vals = [round(float(v), 3) for v in pred[t].tolist()]
                fh.write(json.dumps({
                    "round": t, "parse_fail_frac": parse_fail[t],
                    "raw": [f"{v:.3f}" for v in vals],
                    "parsed": vals}) + "\n")
    return d


def _run(dirs, *flags, partial=True, manifest=None):
    """Drive the gate over a handful of synthetic run dirs.

    --allow-partial by default: these tests interrogate ONE cell at a
    time, and grid completeness is a separate concern with its own test
    (test_partial_grid_is_a_hard_failure_by_default). Without the flag
    every per-cell test would fail for the wrong reason -- 45 absent
    cells -- and would keep failing even if the sabotage went undetected.

    --reuse-manifest is pointed at a path that does not exist unless a
    test supplies one, so no test silently depends on whatever manifest
    happens to be checked into notes/pofd/section3/.
    """
    if partial and "--allow-partial" not in flags:
        flags = ("--allow-partial",) + tuple(flags)
    if manifest is None:
        manifest = os.path.join(ROOT, "tests", "_no_such_reuse_manifest.json")
    flags = tuple(flags) + ("--reuse-manifest", manifest)
    cmd = [sys.executable, CHECK, *flags] + list(dirs)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "USE_TF": "0"})


def _out(r):
    return (r.stdout or "") + (r.stderr or "")


def _assert_fail(r, why):
    assert r.returncode != 0, (
        f"checker PASSED a run it must reject ({why}).\n{_out(r)[-3000:]}")


def _assert_pass(r, why):
    assert r.returncode == 0, (
        f"checker REJECTED a run it must accept ({why}).\n{_out(r)[-3000:]}")


def _good(root, **kw):
    """A well-formed qwen7b forward-lambda-1 env1 production cell."""
    tag = kw.pop("tag", PROD.format(m="qwen7b", arm="fwdlam1", w="0p5",
                                    k="1"))
    return _mk_run(root, tag, **kw)


# ------------------------------------------------------------- the baseline
@needs_checker
def test_a_well_formed_cell_passes(tmp_path):
    """If this fails, every FAIL assertion below is meaningless."""
    r = _run([_good(str(tmp_path))])
    _assert_pass(r, "well-formed production cell")


@needs_checker
def test_a_well_formed_smoke_passes(tmp_path):
    d = _mk_run(str(tmp_path), SMOKE, model="qwen3_8b", kl_direction="reverse",
                kl_beta=1.0, w_plat=1.0, innate_lambda=1.0,
                rounds=SMOKE_ROUNDS)
    r = _run([d], "--smoke")
    _assert_pass(r, "well-formed 3-round smoke")


# ------------------------------------------------------ direction integrity
@needs_checker
def test_tag_says_forward_but_config_says_reverse(tmp_path):
    """The headline inverter. Nothing downstream would reveal it: the
    trajectory, the served map and the figure are all identical in
    shape whichever direction trained."""
    d = _mk_run(str(tmp_path),
                PROD.format(m="qwen7b", arm="fwdlam2", w="0p5", k="1"),
                kl_beta=2.0, kl_direction="reverse")
    _assert_fail(_run([d]), "tag fwd / config reverse")


@needs_checker
def test_tag_says_reverse_but_config_says_forward(tmp_path):
    d = _mk_run(str(tmp_path),
                PROD.format(m="qwen3_8b", arm="revlam8", w="1", k="1"),
                model="qwen3_8b", kl_beta=8.0, kl_direction="forward",
                w_plat=1.0)
    _assert_fail(_run([d]), "tag rev / config forward")


@needs_checker
def test_lambda_in_tag_must_equal_kl_beta_in_config(tmp_path):
    """lambda IS the config field kl_beta. A cell tagged fwdlam8 that
    trained at kl_beta=1 lands on the wrong rung of the dose ladder --
    the single curve the section is about."""
    d = _mk_run(str(tmp_path),
                PROD.format(m="qwen7b", arm="fwdlam8", w="0p5", k="1"),
                kl_beta=1.0, kl_direction="forward")
    _assert_fail(_run([d]), "tag fwdlam8 / kl_beta 1")


@needs_checker
def test_sft_cell_with_an_inert_direction_placeholder_passes(tmp_path):
    """The generator writes kldir='forward' into the sft rows as an INERT
    placeholder (empty would reach the runner as KL_DIRECTION=''), so
    every archived sft config.json records a direction it never used.
    The sft TAG carries no direction token, so there is no claim to
    contradict -- the gate must not read one in."""
    for placeholder in ("forward", "reverse"):
        d = _mk_run(str(tmp_path / placeholder),
                    PROD.format(m="qwen7b", arm="sft", w="0p5", k="0p2"),
                    style="sft", kl_beta=0.0, kl_direction=placeholder,
                    w_plat=0.5, innate_lambda=0.2,
                    grad_kl0=[0.0] * ROUNDS)
        _assert_pass(_run([d]), f"sft cell with kldir={placeholder!r}")


@needs_checker
def test_sft_cell_that_actually_trained_with_kl_is_rejected(tmp_path):
    """The converse: an "sft" arm whose config says sft_kl with a live
    lambda is a forward cell hiding in the lambda=0 slot."""
    d = _mk_run(str(tmp_path),
                PROD.format(m="qwen7b", arm="sft", w="0p5", k="1"),
                style="sft_kl", kl_beta=1.0, kl_direction="forward")
    _assert_fail(_run([d]), "sft tag over a live KL anchor")


# ---------------------------------------------------------- the KL reference
@needs_checker
def test_nonempty_kl_ref_adapter_is_rejected(tmp_path):
    """The anchor must be the RAW BASE CHECKPOINT. If one arm anchored
    to a trained adapter, its lambda is not on the same axis as the
    others and the whole ladder is incommensurable -- with no visible
    symptom."""
    d = _good(str(tmp_path), kl_ref_adapter="runs/.../adapter_r40")
    _assert_fail(_run([d]), "kl_ref_adapter nonempty")


@needs_checker
def test_chained_anchor_mode_is_rejected(tmp_path):
    """anchor_mode 'chained' re-freezes the reference to last round's
    policy: that measures round-to-round drift, not retention of the
    PRETRAINED prior."""
    d = _good(str(tmp_path), anchor_mode="chained")
    _assert_fail(_run([d]), "anchor_mode chained")


# ------------------------------------------------------------- Qwen3 template
@needs_checker
def test_qwen3_cell_without_thinking_disabled_is_rejected(tmp_path):
    """With the hybrid-reasoning template ON, Qwen3 emits a <think>
    block. The parser reads the reasoning trace, not the answer, and the
    result presents as a confident CONSTANT -- exactly the shape a
    'total collapse' finding would take."""
    for bad in ("default", 1, True, "1"):
        d = _mk_run(str(tmp_path / f"ct_{bad}"),
                    PROD.format(m="qwen3_8b", arm="fwdlam1", w="0p5", k="1"),
                    model="qwen3_8b", chat_thinking=bad)
        _assert_fail(_run([d]), f"qwen3 with chat_thinking={bad!r}")


@needs_checker
def test_qwen25_cell_is_not_required_to_disable_thinking(tmp_path):
    """Qwen2.5 has no thinking mode. Demanding CHAT_THINKING=0 on it
    would reject the four archived reuse cells and every new Qwen2.5
    row."""
    d = _good(str(tmp_path), model="qwen7b", chat_thinking="default")
    _assert_pass(_run([d]), "qwen2.5 with chat_thinking='default'")


@needs_checker
def test_base_model_must_match_the_model_token(tmp_path):
    d = _mk_run(str(tmp_path),
                PROD.format(m="qwen3_8b", arm="fwdlam1", w="0p5", k="1"),
                model="qwen3_8b", chat_thinking=0)
    with open(os.path.join(d, "config.json")) as fh:
        c = json.load(fh)
    c["base_model"] = QWEN25
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(c, fh)
    _assert_fail(_run([d]), "qwen3_8b tag over a Qwen2.5 checkpoint")


# ---------------------------------------------------------------- parsing
@needs_checker
def test_any_nonzero_parse_failure_in_any_round_is_rejected(tmp_path):
    """A parse failure is recorded as a confident value, not as a gap,
    so it enters the distribution as signal. One bad round in a hundred
    is still a hundredth of the late window."""
    pf = [0.0] * ROUNDS
    pf[93] = 0.0014                      # inside the late window
    d = _good(str(tmp_path), parse_fail=pf)
    _assert_fail(_run([d]), "parse_fail_frac 0.0014 in round 93")


@needs_checker
def test_a_single_early_parse_failure_is_also_rejected(tmp_path):
    pf = [0.0] * ROUNDS
    pf[0] = 0.5
    d = _good(str(tmp_path / "early"), parse_fail=pf)
    _assert_fail(_run([d]), "parse_fail_frac 0.5 in round 0")


@needs_checker
def test_missing_raw_gen_log_is_rejected_not_silently_passed(tmp_path):
    """No log is not the same as a clean log. SAVE_RAW_GEN=1 is in the
    sub; its absence means the run did not do what the sub says."""
    d = _good(str(tmp_path), skip_files=("raw_gen_log.json.gz",))
    _assert_fail(_run([d]), "raw_gen_log.json.gz missing")


# ------------------------------------------------------------------ tensors
@needs_checker
def test_short_trajectory_is_rejected(tmp_path):
    """A run held at 87 rounds still writes a valid trajectory.pt. Its
    late window (81-100) would be computed from 7 rounds."""
    d = _mk_run(str(tmp_path),
                PROD.format(m="qwen7b", arm="fwdlam1", w="0p5", k="1"),
                rounds=87, n_rounds=ROUNDS)
    _assert_fail(_run([d]), "pred_raw is [87, 723]")


@needs_checker
def test_wrong_agent_count_is_rejected(tmp_path):
    d = _mk_run(str(tmp_path),
                PROD.format(m="qwen7b", arm="fwdlam1", w="0p5", k="1"),
                n=700)
    _assert_fail(_run([d]), "pred_raw is [100, 700]")


@needs_checker
def test_op_raw_shape_is_checked_too(tmp_path):
    """op_raw is the END-OF-ROUND POST-PEER population state -- the
    thing every population figure is drawn from. pred_raw being right
    does not make op_raw right."""
    g = torch.Generator().manual_seed(3)
    d = _good(str(tmp_path),
              op=0.2 + 0.6 * torch.rand(ROUNDS, 700, generator=g))
    _assert_fail(_run([d]), "op_raw is [100, 700]")


@needs_checker
def test_nonfinite_prediction_is_rejected(tmp_path):
    """A single NaN poisons a mean and silently drops an agent from
    every quantile."""
    g = torch.Generator().manual_seed(5)
    pred = 0.2 + 0.6 * torch.rand(ROUNDS, N_AGENTS, generator=g)
    pred[88, 17] = float("nan")
    _assert_fail(_run([_good(str(tmp_path / "nan"), pred=pred)]), "NaN")
    pred2 = 0.2 + 0.6 * torch.rand(ROUNDS, N_AGENTS, generator=g)
    pred2[3, 400] = float("inf")
    _assert_fail(_run([_good(str(tmp_path / "inf"), pred=pred2)]), "inf")


# --------------------------------------------------------- training happened
@needs_checker
def test_a_run_that_never_trained_is_rejected(tmp_path):
    """grad_norm0 == 0 in every round is an optimizer that no-opped.
    The served map still moves (sampling noise) and the figure still
    renders."""
    d = _good(str(tmp_path), grad_norm0=0.0)
    _assert_fail(_run([d]), "grad_norm0 zero in every round")


@needs_checker
def test_kl_arm_whose_anchor_never_contributed_is_rejected(tmp_path):
    """A cell can record kl_beta=1, kl_direction=forward, and still have
    contributed NO anchor gradient in any round -- that arm is ordinary
    SFT wearing a lambda, and its closeness to the prior would be read
    as retention. This is the entire dose ladder's failure mode."""
    d = _good(str(tmp_path), grad_kl0=[0.0] * ROUNDS)
    _assert_fail(_run([d]), "grad_kl_norm0 zero in every round")


@needs_checker
def test_zero_kl_gradient_at_round_zero_only_is_accepted(tmp_path):
    """A FRESH LoRA IS the reference at round 0, so the anchor term is
    identically zero there by construction. A gate that rejects it
    rejects every correctly-run cell in the wave."""
    d = _good(str(tmp_path),
              grad_kl0=[0.0] + [1.5] * (ROUNDS - 1))
    _assert_pass(_run([d]), "grad_kl_norm0 zero at round 0 only")


@needs_checker
def test_sft_arm_with_no_kl_gradient_is_accepted(tmp_path):
    """kl_beta = 0 means there is no anchor term to produce a gradient.
    Demanding one would reject all six sft cells -- the ladder's
    lambda=0 endpoint."""
    d = _mk_run(str(tmp_path),
                PROD.format(m="qwen3_8b", arm="sft", w="0p5", k="1"),
                model="qwen3_8b", style="sft", kl_beta=0.0,
                kl_direction="forward", grad_kl0=[0.0] * ROUNDS)
    _assert_pass(_run([d]), "sft arm with no KL gradient")


# ------------------------------------------------------------- surface pins
@needs_checker
def test_wrong_gpu_is_rejected(tmp_path):
    """Every archived cell this wave is compared against ran on an H100.
    A different SKU changes numerics; at es=0.05 this project has
    already seen 1 ulp amplify to 1e-1."""
    d = _good(str(tmp_path), gpu="NVIDIA A100-SXM4-80GB")
    _assert_fail(_run([d]), "wrong GPU SKU")


@needs_checker
def test_wrong_seed_is_rejected(tmp_path):
    d = _good(str(tmp_path), seed=42)
    _assert_fail(_run([d]), "seed 42")


@needs_checker
def test_w_plat_must_match_the_tag(tmp_path):
    """Tag says env1 (W=0.5); the run executed env2 (W=1). Both are real
    environments in this wave, so the run is perfectly healthy -- it is
    just filed in the wrong column of the figure."""
    d = _mk_run(str(tmp_path),
                PROD.format(m="qwen7b", arm="fwdlam1", w="0p5", k="1"),
                w_plat=1.0, innate_lambda=1.0)
    _assert_fail(_run([d]), "tag w0p5 / config w_plat 1.0")


@needs_checker
def test_innate_lambda_must_match_the_tag(tmp_path):
    """Tag says k=0.2 (the weak-anchor 'memory' environment); the run
    executed k=1."""
    d = _mk_run(str(tmp_path),
                PROD.format(m="qwen7b", arm="fwdlam1", w="0p5", k="0p2"),
                w_plat=0.5, innate_lambda=1.0)
    _assert_fail(_run([d]), "tag k0p2 / config innate_lambda 1.0")


@needs_checker
def test_n_rounds_other_than_100_is_rejected(tmp_path):
    d = _good(str(tmp_path), n_rounds=30)
    _assert_fail(_run([d]), "config n_rounds 30")


@needs_checker
def test_gates_must_be_genuinely_open(tmp_path):
    """_eaopen_/_esopen_ in the tag mean AI_GATE_MODE and PEER_GATE_MODE
    are all_open. A numeric threshold is a strict inequality and would
    silently drop the extreme pairs this table is about."""
    for field in ("ai_gate_mode", "peer_gate_mode"):
        d = _good(str(tmp_path / field),
                  extra_cfg={field: "threshold"})
        _assert_fail(_run([d]), f"{field}=threshold under an _open_ tag")


@needs_checker
def test_lora_rank_and_freshness_are_pinned(tmp_path):
    for bad in ({"lora_r": 64}, {"fresh_each_round": False},
                {"use_lora": False}, {"train_cap": 200},
                {"n_labeled": 500}):
        d = _good(str(tmp_path / "_".join(map(str, bad))), extra_cfg=bad)
        _assert_fail(_run([d]), f"config override {bad}")


@needs_checker
def test_serve_eval_mode_must_be_on(tmp_path):
    """serve_eval_mode=False decodes with dropout live: the served
    vector is then not the model's greedy map, and the retention number
    is measured against a stochastic surface."""
    d = _good(str(tmp_path), serve_eval_mode=False)
    _assert_fail(_run([d]), "serve_eval_mode False")


@needs_checker
def test_runner_population_update_marker_is_accepted(tmp_path):
    """The tag's 'anch2' token has NO matching runner string: the runner
    writes 'nested_ai_anchored_then_social_v2'. A gate that requires the
    tag token and the config marker to agree fails ALL 46 production
    runs. (The distinction is inert here anyway: with all_open the gate
    returns before it consults the anchor.)"""
    d = _good(str(tmp_path), pop_update=POP_UPDATE)
    _assert_pass(_run([d]), f"population_update={POP_UPDATE!r}")


# ------------------------------------------------- the environment is shared
@needs_checker
def test_innate_vector_must_be_identical_across_cells(tmp_path):
    """The arms differ only in the LEARNER. If two cells drew different
    innate vectors they are not two arms of one experiment, they are two
    experiments -- and every cross-arm delta in the section is then a
    mixture of learner effect and environment effect."""
    a = _mk_run(str(tmp_path),
                PROD.format(m="qwen7b", arm="fwdlam1", w="0p5", k="1"),
                innate=_innate(seed=1234))
    b = _mk_run(str(tmp_path),
                PROD.format(m="qwen7b", arm="fwdlam2", w="0p5", k="1"),
                kl_beta=2.0, innate=_innate(seed=9999))
    _assert_fail(_run([a, b]), "two cells with different innate vectors")


@needs_checker
def test_matching_innate_vectors_across_cells_pass(tmp_path):
    a = _mk_run(str(tmp_path),
                PROD.format(m="qwen7b", arm="fwdlam1", w="0p5", k="1"))
    b = _mk_run(str(tmp_path),
                PROD.format(m="qwen7b", arm="fwdlam2", w="0p5", k="1"),
                kl_beta=2.0)
    _assert_pass(_run([a, b]), "two cells sharing one innate vector")


# ------------------------------------------------------------- scout hygiene
@needs_checker
def test_a_ten_round_scout_offered_as_production_is_rejected(tmp_path):
    """pofdkd_* are 10-round SCOUTS. The Section-3 claim lives entirely
    in the late window (rounds 81-100), which a 10-round run does not
    have. Its trajectory is nonetheless complete and self-consistent."""
    d = _mk_run(str(tmp_path),
                "pofdkd_qwen7b_fwdlam1_eaopen_w0p5_l1_esopen_s0_r10",
                rounds=10, n_rounds=10)
    _assert_fail(_run([d]), "a 10-round pofdkd_ scout")


@needs_checker
def test_smoke_tag_is_rejected_under_the_production_gate(tmp_path):
    d = _mk_run(str(tmp_path), SMOKE, model="qwen3_8b",
                kl_direction="reverse", w_plat=1.0, rounds=SMOKE_ROUNDS)
    _assert_fail(_run([d]), "3-round smoke offered as production")


@needs_checker
def test_production_tag_is_rejected_under_the_smoke_gate(tmp_path):
    d = _good(str(tmp_path))
    _assert_fail(_run([d], "--smoke"), "100-round cell offered as a smoke")


# --------------------------------------------------- COLLAPSE IS AN OUTCOME
def _collapse_metrics_reported(text):
    """Which of the five required collapse readouts appear.

    Abbreviated table headers count -- the requirement is that the
    number reaches the reader, not that it is spelled out in prose."""
    t = text.lower()
    return {
        "distinct": any(s in t for s in ("distinct", "n_uniq", "nuniq",
                                         "ndist", "n_dist", "uniq")),
        "mode_share": any(s in t for s in ("mode share", "mode_share",
                                           "modeshare", "  mode ",
                                           "mode_mass")),
        "top3": any(s in t for s in ("top-3", "top3", "top_3")),
        "eff_modes": any(s in t for s in ("eff_mode", "effmod",
                                          "effective mode", "eff_support",
                                          "eff modes")),
        "sd": any(s in t for s in ("pred_sd", "predsd", "pred sd", " sd ",
                                   "std", "stdev", "sigma")),
    }


@needs_checker
def test_a_constant_served_map_is_not_a_hard_failure(tmp_path):
    """CRITICAL POLICY. Total collapse is a legitimate ANSWER to this
    wave's question, not a broken run. check_kl_direction.py hard-fails
    a constant served map because its premise was heterogeneity; that
    rule must NOT be copied here or the most interesting result in the
    section becomes ungateable. Provenance is what is gated: parsing
    clean, training real, direction honest."""
    pred = torch.full((ROUNDS, N_AGENTS), 0.25)
    d = _good(str(tmp_path), pred=pred)
    r = _run([d])
    _assert_pass(r, "constant served map with clean provenance")


@needs_checker
def test_a_constant_served_map_is_reported_in_detail(tmp_path):
    """Passing silently is the other failure: a collapsed cell that
    looks like any other row in the table. The gate must SAY so."""
    pred = torch.full((ROUNDS, N_AGENTS), 0.25)
    d = _good(str(tmp_path), pred=pred)
    got = _collapse_metrics_reported(_out(_run([d])))
    missing = sorted(k for k, v in got.items() if not v)
    assert not missing, (
        f"collapsed cell passed but the gate never reported {missing}; "
        f"required: distinct values, largest mode share, top-3 mode "
        f"share, effective modes, prediction SD")


@needs_checker
def test_a_coarse_binary_served_map_is_not_a_hard_failure(tmp_path):
    """The real Qwen2.5 shape: the frozen model serves only 0.25/0.65 to
    98.9% of agents, so served-space distances are argmax-quantized.
    That is the measurement, not a defect."""
    g = torch.Generator().manual_seed(11)
    pick = (torch.rand(ROUNDS, N_AGENTS, generator=g) < 0.989).float()
    pred = torch.where(pick.bool(),
                       torch.where(torch.rand(ROUNDS, N_AGENTS,
                                              generator=g) < 0.5,
                                   torch.tensor(0.25), torch.tensor(0.65)),
                       0.05 + 0.9 * torch.rand(ROUNDS, N_AGENTS,
                                               generator=g))
    d = _good(str(tmp_path), pred=pred)
    r = _run([d])
    _assert_pass(r, "coarse binary served map")
    got = _collapse_metrics_reported(_out(r))
    assert all(got.values()), sorted(k for k, v in got.items() if not v)


@needs_checker
def test_a_frozen_served_map_across_rounds_is_reported_not_ignored(tmp_path):
    """Bit-identical predictions in every round means the learner had no
    effect on serving. Under this wave's policy that is still an
    OUTCOME, but it must be visible: the SD across rounds is 0."""
    g = torch.Generator().manual_seed(13)
    row = 0.2 + 0.6 * torch.rand(N_AGENTS, generator=g)
    pred = row.unsqueeze(0).repeat(ROUNDS, 1)
    r = _run([_good(str(tmp_path), pred=pred)])
    assert "sd" in _out(r).lower() or "std" in _out(r).lower(), _out(r)[-2000:]


# -------------------------------------------- FILE PROVENANCE (the big one)
@needs_checker
def test_checker_reads_telemetry_and_rawgen_not_trajectory(tmp_path):
    """THE BUG A PREVIOUS CHECKER SHIPPED.

    l_init and grad_* live in telemetry.json. parse_fail_frac lives in
    raw_gen_log.json.gz. NEITHER is in trajectory.pt. A checker that
    reads them off trajectory.pt's per-round dicts gets None for every
    gate and passes everything.

    This fixture makes the two sources CONTRADICT:
      trajectory.pt rows say  parse_fail_frac 0.0, grad_kl_norm0 9.9
                              (perfect -- the vacuous-pass answer)
      the real files say      parse_fail_frac 0.37 in round 40
                              (broken -- the truthful answer)
    A checker reading the right files FAILS. One reading trajectory.pt
    PASSES, and that pass is the bug.
    """
    pf = [0.0] * ROUNDS
    pf[40] = 0.37
    liar = [{"round": t, "parse_fail_frac": 0.0, "grad_norm0": 5.0,
             "grad_kl_norm0": 9.9, "l_init": 1.0} for t in range(ROUNDS)]
    d = _good(str(tmp_path), parse_fail=pf, traj_rows=liar)
    _assert_fail(
        _run([d]),
        "parse_fail_frac 0.37 in raw_gen_log.json.gz while trajectory.pt "
        "claims 0.0 -- the checker read trajectory.pt")


@needs_checker
def test_checker_reads_grad_from_telemetry_not_trajectory(tmp_path):
    """Mirror image: telemetry.json says the anchor never contributed;
    trajectory.pt's rows claim a healthy 9.9 every round."""
    liar = [{"round": t, "parse_fail_frac": 0.0, "grad_norm0": 5.0,
             "grad_kl_norm0": 9.9, "l_init": 1.0} for t in range(ROUNDS)]
    d = _good(str(tmp_path), grad_kl0=[0.0] * ROUNDS, traj_rows=liar)
    _assert_fail(
        _run([d]),
        "grad_kl_norm0 all-zero in telemetry.json while trajectory.pt "
        "claims 9.9 -- the checker read trajectory.pt")


@needs_checker
def test_missing_telemetry_is_rejected_not_silently_passed(tmp_path):
    """The vacuous pass in its purest form: no telemetry file at all.
    'I could not check' must never render as 'it checked out'."""
    d = _good(str(tmp_path), skip_files=("telemetry.json",))
    _assert_fail(_run([d]), "telemetry.json missing")


@needs_checker
def test_missing_config_or_trajectory_is_rejected(tmp_path):
    a = _good(str(tmp_path / "nocfg"), skip_files=("config.json",))
    _assert_fail(_run([a]), "config.json missing")
    b = _good(str(tmp_path / "notraj"), skip_files=("trajectory.pt",))
    _assert_fail(_run([b]), "trajectory.pt missing -- run did not finish")


@needs_checker
def test_checker_exits_nonzero_when_given_nothing(tmp_path):
    """An empty glob is the commonest way a gate reports PASS on a wave
    that does not exist."""
    r = _run([os.path.join(str(tmp_path), "no_such_run_dir")])
    _assert_fail(r, "a run dir that does not exist")


# =====================================================================
# PER-SOURCE OPERATOR-MARKER VALIDATION (user contract, 2026-08-22)
# =====================================================================
# The analyzer MAY reuse v1 artifacts, but only because the two
# operators are NUMERICALLY IDENTICAL under all_open: _gated_pop.ai_gate
# returns an all-ones mask (line ~205) before nested_presocial_update
# ever reads the gate reference (~line 247). That is an argument about
# ONE configuration, not a general licence to treat v1 and v2 as
# interchangeable -- so the marker must be validated PER SOURCE:
#
#   new pofds3_ cell   + v1  -> HARD FAIL (did not come from this tree)
#   new pofds3_ cell   + v2  -> PASS
#   archived QWU reuse + v1  -> ACCEPTED (that IS the audited artifact)
#   archived QWU reuse + v2  -> HARD FAIL (not what the manifest audited)
#   ANY source + v1 + NUMERIC ai_gate_mode -> HARD FAIL (the equivalence
#                                             argument does not hold)
POP_V1 = "nested_ai_then_social_v1"
POP_V2 = "nested_ai_anchored_then_social_v2"

# The four archived cells, and the Section-3 slot each one satisfies.
QWU_REUSE = {
    "pofdqwu_qwen7b_b0_eaopen_w0p5_l1_esopen_s0_r100":
        ("qwen7b", "sft", 0.5, 1.0, "sft", 0.0),
    "pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100":
        ("qwen7b", "sft", 1.0, 1.0, "sft", 0.0),
    "pofdqwu_qwen7b_b1_eaopen_w0p5_l1_esopen_s0_r100":
        ("qwen7b", "fwdlam1", 0.5, 1.0, "sft_kl", 1.0),
    "pofdqwu_qwen7b_b1_eaopen_w1_l1_esopen_s0_r100":
        ("qwen7b", "fwdlam1", 1.0, 1.0, "sft_kl", 1.0),
}


def _sha_t(t):
    """The gate's own recipe: sha256 over float32 bytes."""
    import hashlib
    a = torch.as_tensor(t).detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def _mk_reuse_cell(root, tag, *, pop_update=POP_V1, ai_gate_mode="all_open",
                   **kw):
    """An archived QWU run dir plus the reuse-manifest entry that
    attests it. Both are needed: the gate admits a non-grammar tag only
    on an explicit REUSE verdict backed by an artifact hash."""
    model, arm, beta, k, style, lam = QWU_REUSE[tag]
    extra = dict(kw.pop("extra_cfg", None) or {})
    if ai_gate_mode != "all_open":
        extra["ai_gate_mode"] = ai_gate_mode
    d = _mk_run(root, tag, model=model, style=style, kl_beta=lam,
                kl_direction="forward", w_plat=beta, innate_lambda=k,
                pop_update=pop_update, extra_cfg=extra or None, **kw)
    blob = torch.load(os.path.join(d, "trajectory.pt"), map_location="cpu",
                      weights_only=False)
    cell = {"model": model, "arm": arm, "beta": beta, "k": k,
            "status": "reused", "run_tag": tag, "run_dir": d,
            "pred_raw_sha256": _sha_t(blob["pred_raw"]),
            "op_raw_sha256": _sha_t(blob["op_raw"])}
    mf = os.path.join(root, f"manifest_{os.path.basename(tag)}.json")
    with open(mf, "w") as fh:
        json.dump({"key": "section3", "cells": [cell]}, fh)
    return d, mf


@needs_checker
def test_new_cell_carrying_the_v1_marker_is_a_hard_failure(tmp_path):
    """A pofds3_ cell is NEW: it can only have been produced by the
    current tree, which writes v2. A v1 marker on it means the job ran
    against an older checkout -- so its round operator is not the one
    the other 45 cells used, and no amount of all_open equivalence makes
    it the same EXPERIMENT."""
    d = _good(str(tmp_path), pop_update=POP_V1)
    _assert_fail(_run([d]), f"pofds3_ cell carrying {POP_V1!r}")


@needs_checker
def test_new_cell_carrying_the_v2_marker_passes(tmp_path):
    d = _good(str(tmp_path), pop_update=POP_V2)
    _assert_pass(_run([d]), f"pofds3_ cell carrying {POP_V2!r}")


@needs_checker
def test_archived_reuse_cell_carrying_v1_is_accepted(tmp_path):
    """v1 IS the archived artifact. The reuse audit inspected a run that
    records v1; demanding v2 of it would reject the very cells the reuse
    exists to admit."""
    d, mf = _mk_reuse_cell(
        str(tmp_path), "pofdqwu_qwen7b_b1_eaopen_w0p5_l1_esopen_s0_r100",
        pop_update=POP_V1)
    _assert_pass(_run([d], manifest=mf), f"archived reuse cell with {POP_V1!r}")


@needs_checker
def test_archived_reuse_cell_carrying_v2_is_a_hard_failure(tmp_path):
    """The mirror image, and the subtler one. An archived tag that
    records v2 is NOT the artifact the manifest audited -- something
    re-ran or overwrote it. Accepting it silently substitutes an
    unaudited run for an audited one under the audited run's name."""
    d, mf = _mk_reuse_cell(
        str(tmp_path), "pofdqwu_qwen7b_b1_eaopen_w0p5_l1_esopen_s0_r100",
        pop_update=POP_V2)
    _assert_fail(_run([d], manifest=mf),
                 f"archived reuse cell recording {POP_V2!r}")


@needs_checker
def test_v1_under_a_numeric_ai_gate_is_a_hard_failure_for_any_source(tmp_path):
    """The whole v1/v2 equivalence rests on AI_GATE_MODE=all_open, where
    ai_gate returns an all-ones mask before the reference is read. Under
    a NUMERIC threshold the reference is the decision, so v1 and v2 are
    different operators and a v1 artifact is not admissible at all."""
    d, mf = _mk_reuse_cell(
        str(tmp_path / "reuse"),
        "pofdqwu_qwen7b_b1_eaopen_w0p5_l1_esopen_s0_r100",
        pop_update=POP_V1, ai_gate_mode="threshold")
    _assert_fail(_run([d], manifest=mf),
                 "archived v1 cell under a NUMERIC ai_gate_mode")
    n = _good(str(tmp_path / "new"), pop_update=POP_V1,
              extra_cfg={"ai_gate_mode": "threshold"})
    _assert_fail(_run([n]), "new v1 cell under a NUMERIC ai_gate_mode")


@needs_checker
def test_the_marker_found_is_emitted_for_every_source(tmp_path):
    """Provenance that is checked but never emitted cannot be audited by
    a reader. The marker each source ACTUALLY carries must reach the
    output -- stdout or the machine-readable --json verdict."""
    def _emitted(run_dirs, marker, manifest=None):
        jp = str(tmp_path / f"v_{marker[-2:]}_{len(run_dirs)}.json")
        r = _run(run_dirs, "--json", jp, manifest=manifest)
        blob = _out(r)
        if os.path.exists(jp):
            with open(jp) as fh:
                blob += fh.read()
        return marker in blob, r

    new = _good(str(tmp_path / "new"), pop_update=POP_V2)
    ok, r = _emitted([new], POP_V2)
    assert ok, (f"the gate never emits the operator marker the NEW cell "
                f"carries.\n{_out(r)[-2000:]}")
    d, mf = _mk_reuse_cell(
        str(tmp_path / "reuse"),
        "pofdqwu_qwen7b_b0_eaopen_w0p5_l1_esopen_s0_r100",
        pop_update=POP_V1)
    ok2, r2 = _emitted([d], POP_V1, manifest=mf)
    assert ok2, (f"the gate never emits the operator marker the ARCHIVED "
                 f"cell carries.\n{_out(r2)[-2000:]}")


@needs_checker
def test_an_unknown_marker_is_never_guessed_at(tmp_path):
    d = _good(str(tmp_path / "unknown"),
              pop_update="nested_ai_then_social_hgate_v2")
    _assert_fail(_run([d]), "a marker string that exists nowhere in the repo")
    e = _good(str(tmp_path / "absent"), extra_cfg={"population_update": None})
    _assert_fail(_run([e]), "population_update absent")


# --------------------------------------------- the equivalence itself
def test_v1_and_v2_operators_are_numerically_identical_under_all_open():
    """The load-bearing premise of the reuse policy, tested directly and
    locally. If this ever stops holding, every v1 reuse in this wave
    becomes a different experiment -- so it is pinned here rather than
    trusted as a comment."""
    import importlib.util
    gp_path = os.path.join(PIPE, "_gated_pop.py")
    spec = importlib.util.spec_from_file_location("_gp_s3", gp_path)
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    g = torch.Generator().manual_seed(0)
    n = 512
    x0 = torch.rand(n, generator=g)
    served = torch.rand(n, generator=g)
    innate = torch.rand(n, generator=g)
    w = torch.full((n,), 0.5)
    for k in (0.2, 1.0):
        a = gp.nested_presocial_update(x0, served, innate, k, w, 0.2,
                                       gate_mode="all_open", gate_on="anchor")
        b = gp.nested_presocial_update(x0, served, innate, k, w, 0.2,
                                       gate_mode="all_open", gate_on="x0")
        xa = a[0] if isinstance(a, tuple) else a
        xb = b[0] if isinstance(b, tuple) else b
        assert torch.equal(xa, xb), (
            f"v1 and v2 differ under all_open at k={k} -- the reuse "
            f"policy's premise is false")
    # ... and they DO differ under a numeric gate, which is why the
    # equivalence may not be generalised.
    c = gp.nested_presocial_update(x0, served, innate, 1.0, w, 0.2,
                                   gate_mode="threshold", gate_on="anchor")
    d = gp.nested_presocial_update(x0, served, innate, 1.0, w, 0.2,
                                   gate_mode="threshold", gate_on="x0")
    xc = c[0] if isinstance(c, tuple) else c
    xd = d[0] if isinstance(d, tuple) else d
    assert not torch.equal(xc, xd), (
        "v1 and v2 agree under a NUMERIC gate too -- then the per-source "
        "marker rule below is unnecessary; re-derive it before relaxing")


# --------------------------------- sim_perfect_predictor self-labelling
PP = os.path.join(PIPE, "sim_perfect_predictor.py")


@pytest.mark.skipif(not os.path.exists(PP),
                    reason="sim_perfect_predictor.py not present")
def test_perfect_predictor_records_the_operator_it_actually_executed():
    """It used to hard-code the v1 string in build_config while
    simulate() called nested_presocial_update with NO gate_on=, whose
    default flipped to "anchor" on 2026-08-22 -- so the artifact
    mislabelled its own round operator. The label and the call must
    agree, and the call must be EXPLICIT: a marker that is correct only
    because a default happens to point the right way is one edit away
    from lying again."""
    import re
    src = open(PP).read()
    calls = re.findall(r"nested_presocial_update\((?:[^()]|\([^()]*\))*\)",
                       src, re.S)
    calls = [c for c in calls if "def " not in c]
    assert calls, "no nested_presocial_update call found"
    gates = set()
    for c in calls:
        m = re.search(r'gate_on\s*=\s*["\'](\w+)["\']', c)
        assert m, (
            "nested_presocial_update is called without an EXPLICIT "
            f"gate_on=; the artifact's marker then depends on a library "
            f"default that has already flipped once:\n{c[:300]}")
        gates.add(m.group(1))
    assert len(gates) == 1, f"mixed gate references in one simulator: {gates}"
    gate_on = gates.pop()
    markers = set(re.findall(r'"(nested_ai_[a-z0-9_]*)"', src))
    markers |= set(re.findall(r"'(nested_ai_[a-z0-9_]*)'", src))
    assert markers, "the simulator records no population_update marker"
    want = POP_V2 if gate_on == "anchor" else POP_V1
    assert markers == {want}, (
        f"simulate() executes gate_on={gate_on!r} but the artifact "
        f"records {sorted(markers)}; it must record {want!r}")


# ------------------------------- the audit's manifest must be readable
AUDIT_MF = os.path.join(ROOT, "notes", "pofd", "section3",
                        "reuse_manifest.json")


@needs_checker
@pytest.mark.skipif(not os.path.exists(AUDIT_MF),
                    reason="the reuse audit has not written its manifest yet")
def test_the_gate_can_actually_read_the_audit_manifest():
    """CROSS-SIBLING INTEGRATION. The reuse audit writes the manifest and
    the gate reads it. If the two disagree about the schema, the gate
    hard-fails on the real tree before it inspects a single run -- and
    that failure looks like a data problem, not a tooling problem."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_chk_s3", CHECK)
    chk = importlib.util.module_from_spec(spec)
    sys.modules["_chk_s3"] = chk
    spec.loader.exec_module(chk)
    out = []
    by_tag, by_slot, ok = chk.load_manifest(AUDIT_MF, out)
    assert ok, ("check_section3.load_manifest rejects the manifest that "
                "audit_section3_reuse.py writes:\n" + "\n".join(out))
    assert len(by_tag) >= 4, (
        f"the gate resolves {len(by_tag)} manifest entries by tag; the "
        f"wave declares 4 archived reuse cells.\n{out}")
    assert len(by_slot) >= 4, (
        f"the gate resolves {len(by_slot)} manifest entries by "
        f"(model, arm, beta, k) slot; without a slot key an archived tag "
        f"cannot be bound to the Section-3 cell it satisfies.\n{out}")
    for tag, cell in by_tag.items():
        if not tag.startswith("pofdqwu_"):
            continue
        assert chk._manifest_says_reuse(cell), (
            f"{tag}: the gate does not read the audit's verdict as REUSE "
            f"(status/verdict={cell.get('verdict', cell.get('status'))!r})")
        present = [k for k in chk._MF_HASHES if k in cell and cell[k]]
        assert present, (
            f"{tag}: the audit records "
            f"{[k for k in cell if 'sha' in k]} but the gate only "
            f"recognises {sorted(chk._MF_HASHES)} -- so it will reject "
            f"this cell for carrying NO artifact hash")


# --------------------------------------------- grid completeness default
@needs_checker
def test_partial_grid_is_a_hard_failure_by_default(tmp_path):
    """Every other test in this file passes --allow-partial. This one
    proves that flag is doing work: without it, a one-cell wave must not
    look like a complete result."""
    d = _good(str(tmp_path))
    r = _run([d], partial=False)
    _assert_fail(r, "a 1-of-50 grid with no --allow-partial")
    assert "ABSENT" in _out(r) or "COMPLETENESS" in _out(r).upper()


# ---------------------------------------- the twin is not a new simulator
@needs_checker
def test_the_matched_no_platform_process_is_twin_raw_not_a_reimplementation():
    """The matched no-platform counterfactual is `twin_raw`, recorded
    INSIDE each artifact by the same RNG stream as the treated run. A
    reimplementation would be a different process with different draws,
    and the matched comparison would silently stop being matched."""
    import glob
    strays = []
    for p in glob.glob(os.path.join(PIPE, "*section3*.py")):
        src = open(p).read()
        if "twin_raw" not in src:
            continue
        for marker in ("def simulate_twin", "def _simulate_twin",
                       "def no_platform", "def _no_platform",
                       "def simulate_no_platform"):
            if marker in src:
                strays.append(f"{os.path.basename(p)}: {marker}")
    assert not strays, (
        "the matched no-platform process is twin_raw inside the existing "
        f"artifacts, not a new simulator: {strays}")
