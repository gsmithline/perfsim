"""Fixture tests for check_section4_gate.py (Section-4 corrected-gate wave).

Synthesizes physically-consistent pofds4g_ runs in tmp_path -- no model,
no cluster, no HuggingFace: pure torch tensors plus the two artifacts the
gate reads (trajectory.pt and raw_gen_log.json.gz), and icl_days_log
.json.gz on the d8 personal-history arm.

Healthy (must PASS): the complete 72-cell production grid, and the
4-cell 3-round smoke.

Sabotage (must FAIL): a run recording the OLD "nested_ai_then_social_v1"
operator, a cell whose config eps_ai contradicts its tag, a fixed cell
whose stored clamp mask is not the 145 lowest-innate agents, a
fixed/evolving pair sitting on different innate vectors, a missing cell,
and an evolving cell carrying a clamp key.

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac):
  USE_TF=0 python -m pytest tests/test_section4_gate_checker.py -q
"""
import gzip
import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
CHECKER = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines",
                       "check_section4_gate.py")

N = 723
CLAMP_N = 145
BASE = "mistralai/Mistral-7B-Instruct-v0.3"
PROD_ROUNDS = 30
SMOKE_ROUNDS = 3
ARMS = ("b0", "d8")
CONDS = ("fixed", "evolving")
EAS = (0.2, 1.0)
ESS = (0.0, 0.2, 1.0)
SEEDS = (0, 42, 43)
COND_TOK = {"fixed": "fixb20", "evolving": "evoall"}

_G0 = torch.Generator().manual_seed(20260824)
INNATE = torch.rand(N, generator=_G0)

ARM_CFG = {
    "b0": {"training_style": "sft", "kl_beta": 0.0, "icl_k": 0,
           "icl_days": 0, "use_lora": True, "fresh_each_round": True},
    "d8": {"training_style": "frozen", "kl_beta": 0.0, "icl_k": 0,
           "icl_days": 8, "use_lora": False, "fresh_each_round": False},
}


def _num(v):
    return f"{float(v):g}".replace(".", "p")


def tag_for(arm, cond, ea, es, seed, smoke=False):
    pre = "pofds4gsmk" if smoke else "pofds4g"
    return (f"{pre}_mistral7b_{arm}_{COND_TOK[cond]}_anch2_ea{_num(ea)}"
            f"_w0p5_l0p2_es{_num(es)}_s{seed}")


def bottom_mask(innate, k):
    """The 145 lowest-innate agents, agent id as tie-break -- the same
    deterministic ranking _gated_pop.innate_clamp_mask('bottom') uses."""
    n = int(innate.numel())
    order = sorted(range(n), key=lambda i: (float(innate[i]), i))
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.tensor(sorted(order[:k]), dtype=torch.long)] = True
    return m


def clamp_hash(mask):
    return hashlib.sha256(
        mask.detach().cpu().to(torch.uint8).numpy().tobytes()).hexdigest()


def make_cfg(tag, arm, cond, ea, es, seed, nrounds):
    c = {
        "run_tag": tag, "base_model": BASE, "dataset": "movielens",
        "ml_target": "Action", "n_rounds": nrounds, "seed": seed,
        "eps": es, "eps_ai": ea,
        "ai_gate_mode": "threshold", "peer_gate_mode": "threshold",
        "ai_gate_reference": "anchor",
        "population_update": "nested_ai_anchored_then_social_v2",
        "gamma_bias": 0.0, "w_plat": 0.5, "innate_lambda": 0.2,
        "n_labeled": N, "train_cap": N,
        "kl_direction": "forward", "kl_ref_adapter": "",
        "pop_model": "ab", "run_mode": "loop", "anchor_mode": "fixed",
        "data_regime": "replace", "deploy_every": 1,
        "platform_sus_scale": 1.0, "canary_delta": 0.0, "ab_sweeps": 1,
        "epoch_size": 100, "sft_epochs": 1, "sft_batch_size": 4,
        "lora_r": 512, "sft_lr": 5e-5,
        "pristine_frac": 0.0, "replay_frac": 0.0,
        "teacher_label_delta": 0.0, "feedback_mode": "none",
        "icrh": False, "do_sample": False, "seed_base_data": True,
        "serve_eval_mode": True, "fj_update_version": "legacy",
        "icl_select": "random", "icl_ctx_source": "live",
        "icl_snapshot_round": -1,
        "host": "g204",
        "hardware": {"hostname": "g204", "gpu_name": "NVIDIA H100 80GB HBM3"},
    }
    c.update(ARM_CFG[arm])
    if cond == "fixed":
        c.update({"innate_clamp_mode": "bottom", "innate_clamp_frac": 0.2,
                  "innate_clamp_seed": seed,
                  "innate_clamp_peer_mode": "stubborn"})
    return c


def build_run(root, arm, cond, ea, es, seed, nrounds, smoke=False,
              innate=None, cfg_mut=None, payload_mut=None, write_raw=True,
              traj_seed=None):
    """One synthetic run dir. The population moves toward a SEED-DEPENDENT
    attractor and the twin toward another; the fixed cohort is written
    back BIT-EXACTLY from innate in BOTH, which is the invariant the gate
    replays.

    `traj_seed` overrides the seed the DYNAMICS use while leaving the tag
    and config seed alone -- that is exactly the failure the
    seed-distinctness check exists for: a seed that never reached the
    training/serving stream.
    """
    innate = INNATE if innate is None else innate
    ts = seed if traj_seed is None else traj_seed
    tag = tag_for(arm, cond, ea, es, seed, smoke)
    d = os.path.join(str(root), tag)
    os.makedirs(d, exist_ok=True)
    mask = bottom_mask(innate, CLAMP_N) if cond == "fixed" else None

    op, tw, pr = [], [], []
    x, y = innate.clone(), innate.clone()
    for t in range(nrounds):
        x = x + 0.02 * ((0.5 + 0.0007 * ts) - x)
        y = y + 0.01 * ((0.35 + 0.0005 * ts) - y)
        if mask is not None:
            x[mask] = innate[mask]
            y[mask] = innate[mask]
        op.append(x.clone())
        tw.append(y.clone())
        pr.append(torch.full((N,), 0.55 + 0.005 * t))

    cfg = make_cfg(tag, arm, cond, ea, es, seed, nrounds)
    if cfg_mut:
        cfg_mut(cfg)
    payload = {
        "config": cfg,
        "trajectory": [{"t": t} for t in range(nrounds)],
        "op_raw": torch.stack(op),
        "twin_raw": torch.stack(tw),
        "pred_raw": torch.stack(pr),
        "innate": innate.clone(),
    }
    if mask is not None:
        touch = torch.zeros(nrounds, N, dtype=torch.bool)
        touch[:, ~mask] = True
        payload.update({
            "innate_clamp_mask": mask.clone(),
            "innate_clamp_count": int(mask.sum()),
            "innate_clamp_mode": "bottom",
            "innate_clamp_frac": 0.2,
            "innate_clamp_seed": seed,
            "innate_clamp_hash": clamp_hash(mask),
            "innate_clamp_peer_mode": "stubborn",
            "clamp_fr_touch_raw": touch,
        })
    if payload_mut:
        payload_mut(payload)
    torch.save(payload, os.path.join(d, "trajectory.pt"))
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(payload["config"], fh, default=str)

    if write_raw:
        with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt",
                       compresslevel=1) as fh:
            for t in range(nrounds):
                fh.write(json.dumps({
                    "round": t, "parse_fail_frac": 0.0, "raw": [],
                    "parsed": [0.55] * N}) + "\n")
    if int(cfg.get("icl_days") or 0) > 0:
        with gzip.open(os.path.join(d, "icl_days_log.json.gz"), "wt",
                       compresslevel=1) as fh:
            for t in range(nrounds):
                fh.write(json.dumps({"round": t, "ctx": []}) + "\n")
    return d


def cells(smoke):
    if smoke:
        return [(arm, cond, 1.0, 0.2, 0) for arm in ARMS for cond in CONDS]
    return [(arm, cond, ea, es, seed)
            for seed in SEEDS for arm in ARMS for cond in CONDS
            for ea in EAS for es in ESS]


def build_wave(root, smoke, skip=(), per_cell=None):
    """The whole grid. `per_cell(cell) -> dict of build_run kwargs` lets a
    single cell be sabotaged without touching the other 71."""
    nrounds = SMOKE_ROUNDS if smoke else PROD_ROUNDS
    for cell in cells(smoke):
        if cell in skip:
            continue
        kw = dict(per_cell(cell) or {}) if per_cell else {}
        build_run(root, *cell, nrounds=nrounds, smoke=smoke, **kw)
    return str(root)


def run_checker(root, smoke=False, extra=()):
    env = dict(os.environ)
    env["USE_TF"] = "0"
    cmd = [sys.executable, CHECKER, "--run-root", str(root)]
    if smoke:
        cmd.append("--smoke")
    cmd += list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=REPO)


# ---------------------------------------------------------------- healthy
@pytest.mark.slow
def test_full_production_set_passes(tmp_path):
    """(e) a correct full synthetic set passes -- all 72 cells."""
    root = build_wave(tmp_path / "runs", smoke=False)
    p = run_checker(root)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "GRID COMPLETENESS: all 72 cells present" in p.stdout
    assert "[check_s4g] PASS" in p.stdout


def test_smoke_set_passes_and_honours_the_run_root(tmp_path):
    """--smoke gates the 3-round cells IN THE DIRECTORY IT IS PASSED
    (check_section3.py --smoke ignores its run dir; that bug is not
    copied here)."""
    root = build_wave(tmp_path / "smokeruns", smoke=True)
    # a second root holding a BROKEN smoke wave must not influence the
    # verdict for the first
    bad_root = build_wave(
        tmp_path / "otherruns", smoke=True,
        per_cell=lambda c: ({"cfg_mut": lambda cfg: cfg.update(
            {"population_update": "nested_ai_then_social_v1"})}
            if c == ("b0", "fixed", 1.0, 0.2, 0) else {}))
    p = run_checker(root, smoke=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "GRID COMPLETENESS: all 4 cells present" in p.stdout
    q = run_checker(bad_root, smoke=True)
    assert q.returncode == 1, q.stdout


def test_production_run_of_a_smoke_root_is_a_usage_error(tmp_path):
    root = build_wave(tmp_path / "runs", smoke=True)
    p = run_checker(root, smoke=False)
    assert p.returncode == 2, p.stdout + p.stderr


# --------------------------------------------------------------- sabotage
def test_v1_operator_run_fails(tmp_path):
    """(a) the archived gate reference is a HARD failure, named."""
    victim = ("d8", "evolving", 1.0, 0.2, 0)

    def per_cell(c):
        if c != victim:
            return {}
        return {"cfg_mut": lambda cfg: cfg.update(
            {"population_update": "nested_ai_then_social_v1",
             "ai_gate_reference": "x0"})}

    root = build_wave(tmp_path / "runs", smoke=True, per_cell=per_cell)
    p = run_checker(root, smoke=True)
    assert p.returncode == 1, p.stdout
    assert "nested_ai_then_social_v1" in p.stdout
    assert "ai_gate_reference='x0'" in p.stdout
    assert tag_for(*victim, smoke=True) in p.stdout


def test_wrong_grid_cell_fails(tmp_path):
    """(b) a config field that contradicts its own tag."""
    victim = ("b0", "evolving", 1.0, 0.2, 0)

    def per_cell(c):
        if c != victim:
            return {}
        # tag says ea1 / es0p2 / s0; the config says otherwise
        return {"cfg_mut": lambda cfg: cfg.update(
            {"eps_ai": 0.5, "n_rounds": SMOKE_ROUNDS, "kl_direction":
             "reverse"})}

    root = build_wave(tmp_path / "runs", smoke=True, per_cell=per_cell)
    p = run_checker(root, smoke=True)
    assert p.returncode == 1, p.stdout
    assert "eps_ai=0.5" in p.stdout
    assert "kl_direction='reverse'" in p.stdout


def test_broken_clamp_mask_fails(tmp_path):
    """(c) a stored mask that is internally consistent (count + hash) but
    is NOT the 145 lowest-innate agents. The gate RECONSTRUCTS, so a
    self-consistent lie does not survive."""
    victim = ("b0", "fixed", 1.0, 0.2, 0)
    true_mask = bottom_mask(INNATE, CLAMP_N)
    rolled = torch.roll(true_mask, shifts=7)
    assert int(rolled.sum()) == CLAMP_N
    assert not torch.equal(rolled, true_mask)

    def sabotage(payload):
        payload["innate_clamp_mask"] = rolled.clone()
        payload["innate_clamp_count"] = int(rolled.sum())
        payload["innate_clamp_hash"] = clamp_hash(rolled)
        touch = torch.zeros(SMOKE_ROUNDS, N, dtype=torch.bool)
        touch[:, ~rolled] = True
        payload["clamp_fr_touch_raw"] = touch
        # keep the population honest to the STORED mask, so only the
        # reconstruction can catch it
        op = payload["op_raw"].clone()
        tw = payload["twin_raw"].clone()
        op[:, rolled] = INNATE[rolled].unsqueeze(0)
        tw[:, rolled] = INNATE[rolled].unsqueeze(0)
        payload["op_raw"] = op
        payload["twin_raw"] = tw

    def per_cell(c):
        return {"payload_mut": sabotage} if c == victim else {}

    root = build_wave(tmp_path / "runs", smoke=True, per_cell=per_cell)
    p = run_checker(root, smoke=True)
    assert p.returncode == 1, p.stdout
    assert "lowest-innate agents" in p.stdout
    assert tag_for(*victim, smoke=True) in p.stdout


def test_mismatched_fixed_evolving_innate_pair_fails(tmp_path):
    """(d) the two members of a pair must sit on ONE innate vector."""
    victim = ("d8", "evolving", 1.0, 0.2, 0)
    other = INNATE.clone()
    other[3] = float(other[3]) + 0.25

    def per_cell(c):
        return {"innate": other} if c == victim else {}

    root = build_wave(tmp_path / "runs", smoke=True, per_cell=per_cell)
    p = run_checker(root, smoke=True)
    assert p.returncode == 1, p.stdout
    assert "FAIL pair d8/ea1/es0p2/s0" in p.stdout
    assert "DIFFERENT innate vectors" in p.stdout


def test_missing_cell_is_a_hard_failure(tmp_path):
    root = build_wave(tmp_path / "runs", smoke=True,
                      skip=[("d8", "fixed", 1.0, 0.2, 0)])
    p = run_checker(root, smoke=True)
    assert p.returncode == 1, p.stdout
    assert "3 of 4 cells present -- 1 ABSENT" in p.stdout
    assert tag_for("d8", "fixed", 1.0, 0.2, 0, smoke=True) in p.stdout


def test_evolving_cell_carrying_a_clamp_key_fails(tmp_path):
    victim = ("b0", "evolving", 1.0, 0.2, 0)

    def per_cell(c):
        if c != victim:
            return {}
        return {"cfg_mut": lambda cfg: cfg.update(
            {"innate_clamp_mode": "bottom"})}

    root = build_wave(tmp_path / "runs", smoke=True, per_cell=per_cell)
    p = run_checker(root, smoke=True)
    assert p.returncode == 1, p.stdout
    assert "innate_clamp_mode" in p.stdout


def test_missing_anch2_token_fails(tmp_path):
    root = tmp_path / "runs"
    build_wave(root, smoke=True)
    tag = tag_for("b0", "fixed", 1.0, 0.2, 0, smoke=True)
    # rename the BASENAME only: tmp_path itself carries this test's name,
    # which also contains "anch2"
    os.rename(os.path.join(str(root), tag),
              os.path.join(str(root), tag.replace("_anch2_", "_")))
    p = run_checker(root, smoke=True)
    assert p.returncode == 1, p.stdout
    assert "_anch2_" in p.stdout


def test_parse_failure_in_the_raw_gen_log_fails(tmp_path):
    root = tmp_path / "runs"
    build_wave(root, smoke=True)
    d = os.path.join(str(root), tag_for("b0", "fixed", 1.0, 0.2, 0,
                                        smoke=True))
    with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt") as fh:
        for t in range(SMOKE_ROUNDS):
            fh.write(json.dumps({
                "round": t, "parse_fail_frac": 0.0 if t else 0.01,
                "raw": [], "parsed": [0.55] * N}) + "\n")
    p = run_checker(root, smoke=True)
    assert p.returncode == 1, p.stdout
    assert "parse failures in 1 round(s)" in p.stdout


def test_missing_raw_gen_log_falls_back_to_pred_raw_nan(tmp_path):
    """This wave does not set SAVE_RAW_GEN, so raw_gen_log.json.gz does
    not exist. The gate then reads the SAME event out of pred_raw: an
    unparsable generation is stored as NaN."""
    root = build_wave(tmp_path / "runs", smoke=True,
                      per_cell=lambda c: {"write_raw": False})
    p = run_checker(root, smoke=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "pred_raw_nan" in p.stdout
    assert "PARSE-RATE EVIDENCE" in p.stdout
    # ... and --require-raw-gen makes the absence itself fatal
    q = run_checker(root, smoke=True, extra=["--require-raw-gen"])
    assert q.returncode == 1, q.stdout
    assert "raw_gen_log.json.gz missing" in q.stdout

    # a NaN served value under the fallback is a hard failure
    root2 = build_wave(
        tmp_path / "runs2", smoke=True,
        per_cell=lambda c: {
            "write_raw": False,
            **({"payload_mut": _nan_pred}
               if c == ("b0", "fixed", 1.0, 0.2, 0) else {})})
    r = run_checker(root2, smoke=True)
    assert r.returncode == 1, r.stdout
    assert "non-finite entries in pred_raw" in r.stdout


def _nan_pred(payload):
    pr = payload["pred_raw"].clone()
    pr[1, 5] = float("nan")
    payload["pred_raw"] = pr


def test_json_verdict_is_written(tmp_path):
    root = build_wave(tmp_path / "runs", smoke=True)
    out = tmp_path / "verdict.json"
    p = run_checker(root, smoke=True, extra=["--json", str(out)])
    assert p.returncode == 0, p.stdout + p.stderr
    v = json.loads(out.read_text())
    assert v["pass"] is True
    assert v["wave"] == "section4_gate_anch2"
    assert v["n_cells_present"] == 4 and v["n_cells_total"] == 4
    assert v["operator_required"] == "nested_ai_anchored_then_social_v2"


# --------------------------------------------------- seed-distinctness
def _seed_group_root(tmp_path, name, collide):
    """Three seeds of ONE (arm, cond, ea, es) cell, gated via --tags-file.
    `collide` makes seed 42 replay seed 0's dynamics while keeping its own
    tag and config seed -- the signature of a seed that never reached the
    training/serving stream."""
    root = tmp_path / name
    group = ("b0", "evolving", 0.2, 0.0)
    tags = []
    for s in SEEDS:
        build_run(root, *group, s, nrounds=PROD_ROUNDS, smoke=False,
                  traj_seed=0 if (collide and s == 42) else None)
        tags.append(tag_for(*group, s))
    tf = tmp_path / f"{name}_tags.txt"
    tf.write_text("\n".join(tags) + "\n")
    return root, tf


def test_three_distinct_seeds_pass_seed_distinctness(tmp_path):
    root, tf = _seed_group_root(tmp_path, "runs_ok", collide=False)
    p = run_checker(root, extra=["--tags-file", str(tf)])
    assert p.returncode == 0, p.stdout + p.stderr
    assert "seed-distinctness" in p.stdout       # named in the PASS line
    assert "FAIL seed-distinctness" not in p.stdout


def test_two_seeds_sharing_op_raw_fail_seed_distinctness(tmp_path):
    root, tf = _seed_group_root(tmp_path, "runs_bad", collide=True)
    p = run_checker(root, extra=["--tags-file", str(tf)])
    assert p.returncode == 1, p.stdout
    assert "FAIL seed-distinctness b0/evolving/ea0p2/es0" in p.stdout
    assert "seeds [0, 42] produced a BIT-IDENTICAL op_raw" in p.stdout
    # seed 43 is genuinely distinct, so only ONE collision is reported
    assert "1 seed-distinctness collision(s)" in p.stdout


def test_a_missing_seed_is_coverage_not_a_seed_distinctness_pass(tmp_path):
    """A short seed set must fail as COVERAGE; the distinctness check
    itself must stay silent rather than pass on one observation."""
    root = tmp_path / "runs_short"
    group = ("b0", "evolving", 0.2, 0.0)
    build_run(root, *group, 0, nrounds=PROD_ROUNDS, smoke=False)
    tf = tmp_path / "short_tags.txt"
    tf.write_text("\n".join(tag_for(*group, s) for s in SEEDS) + "\n")
    p = run_checker(root, extra=["--tags-file", str(tf)])
    assert p.returncode == 1, p.stdout
    assert "1 of 3 cells present -- 2 ABSENT" in p.stdout
    assert "FAIL seed-distinctness" not in p.stdout
