"""Fixture tests for check_section4_gate.py (Section-4 corrected-gate
waves: the original 72-cell section4_gate_anch2 and the 192-cell
Figure-6 grid section4_gate_anch2_fig6).

Synthesizes physically-consistent pofds4g_ runs in tmp_path -- no model,
no cluster, no HuggingFace: pure torch tensors plus the artifacts the
gate reads (trajectory.pt, raw_gen_log.json.gz, telemetry.json, and
icl_days_log.json.gz on the d8 personal-history arm, rendered from
(innate, op_raw) exactly as the runner renders it).

Healthy (must PASS): the complete 72-cell production grid, the 4-cell
3-round smoke; the complete 192-cell Figure-6 grid (146 runs + 46
twin-derived cells + 2 ea=0 witnesses, with PENDING extensions), the
fig6 4-cell smoke, a matched extension pair.

Sabotage (must FAIL): a run recording the OLD "nested_ai_then_social_v1"
operator, a cell whose config eps_ai contradicts its tag, a fixed cell
whose stored clamp mask is not the 145 lowest-innate agents, a
fixed/evolving pair sitting on different innate vectors, a missing cell,
an evolving cell carrying a clamp key; and in fig6 mode a witness whose
op_raw != twin_raw, a witness with nonzero contact, a twin disagreement
at one (cond, es, seed), a d8 context carrying another agent's value,
and a present extension without its partner.

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

# the Figure-6 grid, restated INDEPENDENTLY of the generator on purpose
# (a test that reads the grid from the thing under test cannot catch a
# grid bug)
FIG6_GATES = (0.0, 0.1, 0.3, 1.0)
FIG6_ESS = (0.0, 0.1, 0.3, 1.0)
FIG6_WITNESS = {("b0", "evolving", 0.3, 0), ("d8", "evolving", 0.3, 0)}
FIG6_SMOKE_EA, FIG6_SMOKE_ES = 0.1, 0.3
ICL_DAYS = 8
TARGET = "Action"

_G0 = torch.Generator().manual_seed(20260824)
INNATE = torch.rand(N, generator=_G0)

ARM_CFG = {
    "b0": {"training_style": "sft", "kl_beta": 0.0, "icl_k": 0,
           "icl_days": 0, "use_lora": True, "fresh_each_round": True},
    "d8": {"training_style": "frozen", "kl_beta": 0.0, "icl_k": 0,
           "icl_days": ICL_DAYS, "use_lora": False, "fresh_each_round": False},
}


def _num(v):
    return f"{float(v):g}".replace(".", "p")


def tag_for(arm, cond, ea, es, seed, smoke=False, horizon=None):
    pre = "pofds4gsmk" if smoke else "pofds4g"
    return (f"{pre}_mistral7b_{arm}_{COND_TOK[cond]}_anch2_ea{_num(ea)}"
            f"_w0p5_l0p2_es{_num(es)}_s{seed}"
            + ("" if horizon is None else f"_r{horizon}"))


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


def render_days(vals):
    """The runner's personal-history sentence, byte for byte."""
    return (f"This user's own opinion of {TARGET} movies over the most "
            f"recent days (oldest to newest): "
            + ", ".join(f"{v:.2f}" for v in vals) + ".")


def days_rows(innate, op_rows):
    """icl_days_log rows from (innate, op_raw): agent i's round-t sentence
    carries the last ICL_DAYS of [innate_i, op[0,i], ..., op[t-1,i]]."""
    hist = [innate.tolist()]
    rows = []
    for t, op_t in enumerate(op_rows):
        win = hist[-ICL_DAYS:]
        rows.append({"round": t,
                     "ctx": [render_days([h[i] for h in win])
                             for i in range(len(hist[0]))]})
        hist.append(op_t.tolist())
    return rows


def make_cfg(tag, arm, cond, ea, es, seed, nrounds):
    c = {
        "run_tag": tag, "base_model": BASE, "dataset": "movielens",
        "ml_target": TARGET, "n_rounds": nrounds, "seed": seed,
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
              traj_seed=None, horizon=None, tel_mut=None, days_mut=None):
    """One synthetic run dir. The population moves toward a SEED-DEPENDENT
    attractor and the twin toward another; the fixed cohort is written
    back BIT-EXACTLY from innate in BOTH, which is the invariant the gate
    replays. The twin depends on (cond, seed) only -- never on the arm or
    eps_AI -- which is the twin-agreement invariant; at eps_AI = 0 the
    population IS the twin (the strict-< gate never opens).

    `traj_seed` overrides the seed the DYNAMICS use while leaving the tag
    and config seed alone -- that is exactly the failure the
    seed-distinctness check exists for: a seed that never reached the
    training/serving stream.

    `horizon` appends _r<horizon> to the tag (a Figure-6 extension run);
    `tel_mut(rows)` / `days_mut(rows)` mutate telemetry.json /
    icl_days_log.json.gz rows before they are written.
    """
    innate = INNATE if innate is None else innate
    ts = seed if traj_seed is None else traj_seed
    tag = tag_for(arm, cond, ea, es, seed, smoke, horizon)
    d = os.path.join(str(root), tag)
    os.makedirs(d, exist_ok=True)
    mask = bottom_mask(innate, CLAMP_N) if cond == "fixed" else None

    op, tw, pr = [], [], []
    x, y = innate.clone(), innate.clone()
    for t in range(nrounds):
        y = y + 0.01 * ((0.35 + 0.0005 * ts) - y)
        if float(ea) == 0.0:
            x = y.clone()                 # eps_AI = 0 IS the twin
        else:
            x = x + 0.02 * ((0.5 + 0.0007 * ts) - x)
        if mask is not None:
            x[mask] = innate[mask]
            y[mask] = innate[mask]
        op.append(x.clone())
        tw.append(y.clone())
        pr.append(torch.full((N,), 0.55 + 0.005 * t))

    cfg = make_cfg(tag, arm, cond, ea, es, seed, nrounds)
    if cfg_mut:
        cfg_mut(cfg)
    contact = 0.0 if float(ea) == 0.0 else 0.4
    payload = {
        "config": cfg,
        "trajectory": [{"round": t, "contact": contact}
                       for t in range(nrounds)],
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

    tel = [{"round": t, "deployment": t, "is_deploy": 1, "l_init": 2.3,
            "probe_pred": [], "contact": contact} for t in range(nrounds)]
    if tel_mut:
        tel_mut(tel)
    with open(os.path.join(d, "telemetry.json"), "w") as fh:
        for r in tel:
            fh.write(json.dumps(r) + "\n")

    if write_raw:
        with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt",
                       compresslevel=1) as fh:
            for t in range(nrounds):
                fh.write(json.dumps({
                    "round": t, "parse_fail_frac": 0.0, "raw": [],
                    "parsed": [0.55] * N}) + "\n")
    if int(cfg.get("icl_days") or 0) > 0:
        rows = days_rows(payload["innate"], list(payload["op_raw"]))
        if days_mut:
            days_mut(rows)
        with gzip.open(os.path.join(d, "icl_days_log.json.gz"), "wt",
                       compresslevel=1) as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
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


def fig6_cells():
    """(arm, cond, ea, es, seed, kind) -- 192 cells: 144 gpu + 2 witness
    + 46 twin-derived."""
    out = []
    for arm in ARMS:
        for cond in CONDS:
            for ea in FIG6_GATES:
                for es in FIG6_ESS:
                    for seed in SEEDS:
                        if ea == 0.0:
                            kind = ("witness" if (arm, cond, es, seed)
                                    in FIG6_WITNESS else "twin")
                        else:
                            kind = "gpu"
                        out.append((arm, cond, ea, es, seed, kind))
    return out


def fig6_smoke_cells():
    return [(arm, cond, FIG6_SMOKE_EA, FIG6_SMOKE_ES, 0)
            for arm in ARMS for cond in CONDS]


def build_fig6_wave(root, skip=(), per_cell=None):
    """Runs for every gpu + witness cell (146); twin cells get none."""
    for (arm, cond, ea, es, seed, kind) in fig6_cells():
        if kind == "twin":
            continue
        cell = (arm, cond, ea, es, seed)
        if cell in skip:
            continue
        kw = dict(per_cell(cell) or {}) if per_cell else {}
        build_run(root, *cell, nrounds=PROD_ROUNDS, **kw)
    return str(root)


def build_fig6_smoke(root, per_cell=None):
    for cell in fig6_smoke_cells():
        kw = dict(per_cell(cell) or {}) if per_cell else {}
        build_run(root, *cell, nrounds=SMOKE_ROUNDS, smoke=True, **kw)
    return str(root)


def write_manifest(path, entries):
    """entries: [(arm, cond, ea, es, seed, rounds)] -> the committed
    extension-request schema s4g2_ext_requests() reads."""
    path.write_text(json.dumps({"cells": [
        {"arm": a, "cond": c, "eps_ai": ea, "eps_social": es, "seed": sd,
         "rounds": r} for (a, c, ea, es, sd, r) in entries]}, indent=1))
    return str(path)


def run_checker(root, smoke=False, extra=(), wave=None):
    env = dict(os.environ)
    env["USE_TF"] = "0"
    cmd = [sys.executable, CHECKER, "--run-root", str(root)]
    if wave:
        cmd += ["--wave", wave]
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


# ================================================================ FIG6
def _tags_file(tmp_path, name, tags):
    tf = tmp_path / f"{name}.txt"
    tf.write_text("\n".join(tags) + "\n")
    return str(tf)


@pytest.mark.slow
def test_fig6_full_grid_passes(tmp_path):
    """The complete Figure-6 grid: 146 run dirs (144 gpu + 2 witnesses),
    46 twin-derived cells drawn from their (cond, es, seed) neighbours,
    and a two-entry extension manifest with nothing run yet
    (PENDING-EXT, non-failing)."""
    root = build_fig6_wave(tmp_path / "runs")
    man = write_manifest(tmp_path / "ext.json",
                         [("b0", "fixed", 0.1, 0.3, 0, 60),
                          ("b0", "evolving", 0.1, 0.3, 0, 60)])
    out = tmp_path / "verdict.json"
    p = run_checker(root, wave="fig6",
                    extra=["--ext-manifest", man, "--json", str(out)])
    assert p.returncode == 0, p.stdout + p.stderr
    assert ("GRID COMPLETENESS: all 192 cells present (146 run + 46 "
            "twin-derived)") in p.stdout
    assert "WITNESSES: 2/2" in p.stdout
    assert "EXTENSIONS: 0 present, 2 PENDING-EXT, 0 unpaired" in p.stdout
    assert p.stdout.count("PENDING-EXT (requested at 60 rounds") == 2
    twin_rows = [ln for ln in p.stdout.splitlines()
                 if ln.endswith("  twin-derived")]
    assert len(twin_rows) == 46
    assert all("   PASS " in ln for ln in twin_rows)
    assert "[check_s4g] PASS -- wave section4_gate_anch2_fig6" in p.stdout
    assert "46 twin-derived" in p.stdout and "2/2 witness" in p.stdout
    # d8 at es=0 is the structural null: the distinctness check must
    # SKIP it (with the reason), never fail it, never silently pass it
    assert p.stdout.count("NOTE seed-distinctness d8/") == 6
    assert "FAIL seed-distinctness" not in p.stdout
    v = json.loads(out.read_text())
    assert v["pass"] is True
    assert v["wave"] == "section4_gate_anch2_fig6"
    assert v["n_runs"] == 146
    assert v["n_cells_total"] == 192 and v["n_cells_present"] == 192
    assert v["n_twin_cells_total"] == 46 and v["n_twin_cells_ok"] == 46
    assert v["n_witness_cells_total"] == 2 and v["n_witness_cells_ok"] == 2
    assert v["n_ext_pending"] == 2 and v["n_ext_present"] == 0
    assert all(e["status"] == "PENDING-EXT" for e in v["extensions"])
    assert len(v["seed_distinctness_skipped_structural_null"]) == 6
    wit = {(c["arm"], c["cond"], c["eps_ai"], c["eps_social"], c["seed"])
           for c in v["witness_cells"]}
    assert wit == {("b0", "evolving", 0.0, 0.3, 0),
                   ("d8", "evolving", 0.0, 0.3, 0)}
    assert sum(1 for c in v["cells"] if c["d8_replay"] == "byte-exact") == 73


def test_fig6_witness_with_op_raw_off_the_twin_fails(tmp_path):
    """ea=0 IS the twin: a witness whose op_raw departs from twin_raw
    breaks the identity every twin-derived cell rests on."""
    victim = ("b0", "evolving", 0.0, 0.3, 0)

    def sabotage(payload):
        op = payload["op_raw"].clone()
        op[4, 10] = float(op[4, 10]) + 0.01
        payload["op_raw"] = op

    root = tmp_path / "runs"
    build_run(root, *victim, nrounds=PROD_ROUNDS, payload_mut=sabotage)
    build_run(root, "d8", "evolving", 0.0, 0.3, 0, nrounds=PROD_ROUNDS)
    tf = _tags_file(tmp_path, "tags", [
        tag_for(*victim), tag_for("d8", "evolving", 0.0, 0.3, 0)])
    p = run_checker(root, wave="fig6", extra=["--tags-file", tf])
    assert p.returncode == 1, p.stdout
    assert f"FAIL {tag_for(*victim)}: ea0-witness: op_raw != twin_raw" \
        in p.stdout
    assert "1 agent(s) differ over 1 round(s), first round 4" in p.stdout
    # the healthy d8 witness passes on its own
    assert f"FAIL {tag_for('d8', 'evolving', 0.0, 0.3, 0)}" not in p.stdout


def test_fig6_witness_with_nonzero_contact_fails(tmp_path):
    """The second half of the ea=0 proof: the recorded AI-gate open
    fraction must be exactly 0 in every round."""
    victim = ("d8", "evolving", 0.0, 0.3, 0)

    def tel_mut(rows):
        rows[1]["contact"] = 0.01

    root = tmp_path / "runs"
    build_run(root, *victim, nrounds=PROD_ROUNDS, tel_mut=tel_mut)
    tf = _tags_file(tmp_path, "tags", [tag_for(*victim)])
    p = run_checker(root, wave="fig6", extra=["--tags-file", tf])
    assert p.returncode == 1, p.stdout
    assert "ea0-witness: telemetry contact (AI-gate open fraction) is not " \
           "exactly 0 in 1 round(s), e.g. round 1 contact=0.01" in p.stdout
    # op_raw == twin_raw still holds, so that half is not reported
    assert "op_raw != twin_raw" not in p.stdout

    # a missing telemetry.json is equally fatal for a witness
    root2 = tmp_path / "runs2"
    d = build_run(root2, *victim, nrounds=PROD_ROUNDS)
    os.remove(os.path.join(d, "telemetry.json"))
    q = run_checker(root2, wave="fig6", extra=["--tags-file", tf])
    assert q.returncode == 1, q.stdout
    assert "ea0-witness: telemetry.json missing" in q.stdout


@pytest.mark.slow
def test_fig6_twin_disagreement_and_undrawable_twin_cells_fail(tmp_path):
    """Two ways a twin-derived cell fails: the runs at its (cond, es,
    seed) disagree on twin_raw, or no run exists there at all."""
    liar = ("d8", "evolving", 0.1, 0.3, 42)
    gone = [(arm, "fixed", ea, 1.0, 43) for arm in ARMS
            for ea in (0.1, 0.3, 1.0)]

    def sabotage(payload):
        tw = payload["twin_raw"].clone()
        tw[:, 7] = tw[:, 7] + 1e-3
        payload["twin_raw"] = tw

    root = build_fig6_wave(
        tmp_path / "runs", skip=gone,
        per_cell=lambda c: {"payload_mut": sabotage} if c == liar else {})
    p = run_checker(root, wave="fig6")
    assert p.returncode == 1, p.stdout
    assert ("FAIL twin evolving/es0p3/s42: 2 distinct twin_raw (over the "
            "first 30 rounds)") in p.stdout
    assert tag_for(*liar) in p.stdout
    for arm in ARMS:
        assert (f"FAIL twin-derived {tag_for(arm, 'evolving', 0.0, 0.3, 42)}: "
                f"the runs at this (cond, es, seed) disagree") in p.stdout
        assert (f"FAIL twin-derived {tag_for(arm, 'fixed', 0.0, 1.0, 43)}: "
                f"no run exists at this (cond, es, seed)") in p.stdout
    assert "6 cell(s) absent" in p.stdout
    assert "1 twin disagreement(s), 4 undrawable twin-derived cell(s)" \
        in p.stdout
    assert "182 of 192 cells present -- 10 ABSENT" in p.stdout


def test_fig6_d8_context_locality_violations_fail(tmp_path):
    """The byte-exact replay of icl_days_log.json.gz catches, and names,
    another agent's sentence, a foreign value, and too many values."""
    swap = ("d8", "evolving", 0.1, 0.3, 0)     # agent 0 gets agent 1's ctx
    foreign = ("d8", "fixed", 0.1, 0.3, 0)     # a value nobody ever held
    toomany = ("d8", "evolving", 0.3, 0.3, 0)  # nine values rendered

    def mut_swap(rows):
        rows[5]["ctx"][0], rows[5]["ctx"][1] = \
            rows[5]["ctx"][1], rows[5]["ctx"][0]

    def mut_foreign(rows):
        s = rows[9]["ctx"][200]
        head, tail = s.rsplit(": ", 1)
        vals = tail.rstrip(".").split(", ")
        vals[-1] = "9.99"
        rows[9]["ctx"][200] = head + ": " + ", ".join(vals) + "."

    def mut_toomany(rows):
        s = rows[12]["ctx"][3]
        rows[12]["ctx"][3] = s.rstrip(".") + ", 0.50."

    root = tmp_path / "runs"
    build_run(root, *swap, nrounds=PROD_ROUNDS, days_mut=mut_swap)
    build_run(root, *foreign, nrounds=PROD_ROUNDS, days_mut=mut_foreign)
    build_run(root, *toomany, nrounds=PROD_ROUNDS, days_mut=mut_toomany)
    build_run(root, "d8", "fixed", 0.3, 0.3, 0, nrounds=PROD_ROUNDS)
    tf = _tags_file(tmp_path, "tags", [
        tag_for(*swap), tag_for(*foreign), tag_for(*toomany),
        tag_for("d8", "fixed", 0.3, 0.3, 0)])
    p = run_checker(root, wave="fig6", extra=["--tags-file", tf])
    assert p.returncode == 1, p.stdout
    assert (f"FAIL {tag_for(*swap)}: d8 locality: personal-history context "
            f"is OFF the byte-exact (innate, op_raw) replay at round 5 "
            f"agent 0:") in p.stdout
    assert "the sentence is agent 1's context (ANOTHER agent's history)" \
        in p.stdout
    assert (f"FAIL {tag_for(*foreign)}: d8 locality: personal-history "
            f"context is OFF the byte-exact (innate, op_raw) replay at "
            f"round 9 agent 200: value(s) ['9.99'] are NOT among agent "
            f"200's own previous opinions") in p.stdout
    assert (f"FAIL {tag_for(*toomany)}: d8 locality: personal-history "
            f"context is OFF the byte-exact (innate, op_raw) replay at "
            f"round 12 agent 3: 9 values rendered > icl_days 8") in p.stdout
    # the untouched d8 run replays byte-exactly
    assert f"FAIL {tag_for('d8', 'fixed', 0.3, 0.3, 0)}" not in p.stdout


def test_fig6_extension_without_its_partner_fails(tmp_path):
    """Extensions are matched fixed/evolving pairs: a present _r60 run
    whose partner is absent fails; the pair present passes at 60 rounds;
    an _r60 run the manifest never requested is EXTRA; and the horizon
    token is rejected outright by the v1 grammar."""
    man = write_manifest(tmp_path / "ext.json",
                         [("b0", "fixed", 0.1, 0.3, 0, 60),
                          ("b0", "evolving", 0.1, 0.3, 0, 60)])
    fx = ("b0", "fixed", 0.1, 0.3, 0)
    ev = ("b0", "evolving", 0.1, 0.3, 0)
    root = tmp_path / "runs"
    build_run(root, *fx, nrounds=60, horizon=60)
    tf1 = _tags_file(tmp_path, "one", [tag_for(*fx, horizon=60)])
    p = run_checker(root, wave="fig6",
                    extra=["--ext-manifest", man, "--tags-file", tf1])
    assert p.returncode == 1, p.stdout
    assert (f"FAIL ext {tag_for(*fx, horizon=60)}: its evolving partner "
            f"{tag_for(*ev, horizon=60)} is absent") in p.stdout
    assert "1 unpaired extension(s)" in p.stdout

    build_run(root, *ev, nrounds=60, horizon=60)
    tf2 = _tags_file(tmp_path, "two", [tag_for(*fx, horizon=60),
                                       tag_for(*ev, horizon=60)])
    q = run_checker(root, wave="fig6",
                    extra=["--ext-manifest", man, "--tags-file", tf2])
    assert q.returncode == 0, q.stdout + q.stderr
    assert "0 unpaired" not in q.stdout or "FAIL" not in q.stdout
    assert f"{tag_for(*fx, horizon=60)}" in q.stdout
    assert "    60 " in q.stdout          # gated at the 60-round horizon

    # an extension with the wrong horizon in its artifact fails on
    # n_rounds, not silently
    root3 = tmp_path / "runs3"
    build_run(root3, *fx, nrounds=60, horizon=60,
              cfg_mut=lambda c: c.update({"n_rounds": 30}))
    build_run(root3, *ev, nrounds=60, horizon=60)
    r = run_checker(root3, wave="fig6",
                    extra=["--ext-manifest", man, "--tags-file", tf2])
    assert r.returncode == 1, r.stdout
    assert "n_rounds=30, expected 60 (the _r60 horizon" in r.stdout

    # an _r60 run nobody requested is EXTRA; under the v1 grammar the
    # horizon token itself is rejected
    root4 = tmp_path / "runs4"
    build_run(root4, "d8", "fixed", 0.3, 0.1, 42, nrounds=60, horizon=60)
    build_run(root4, "d8", "evolving", 0.3, 0.1, 42, nrounds=60, horizon=60)
    tf4 = _tags_file(tmp_path, "four", [
        tag_for("d8", "fixed", 0.3, 0.1, 42, horizon=60),
        tag_for("d8", "evolving", 0.3, 0.1, 42, horizon=60)])
    s = run_checker(root4, wave="fig6",
                    extra=["--ext-manifest", man, "--tags-file", tf4])
    assert s.returncode == 1, s.stdout
    assert "an extension the committed manifest does not request" in s.stdout
    t = run_checker(root4, wave="v1", extra=["--tags-file", tf4])
    assert t.returncode == 1, t.stdout
    assert ("horizon token _r60, which is not part of the "
            "section4_gate_anch2 grammar") in t.stdout


def test_fig6_smoke_checks_exactly_the_4_cells(tmp_path):
    """--wave fig6 --smoke gates the 4 cells of s4g2_smoke_rows (both
    arms x both conditions at ea=0.1, es=0.3, seed 0, 3 rounds) under
    the run root it is passed -- and nothing else."""
    root = build_fig6_smoke(tmp_path / "smokeruns")
    out = tmp_path / "verdict.json"
    p = run_checker(root, wave="fig6", smoke=True,
                    extra=["--json", str(out)])
    assert p.returncode == 0, p.stdout + p.stderr
    assert "GRID COMPLETENESS: all 4 cells present" in p.stdout
    assert "PER-CELL REPORT -- wave section4_gate_anch2_fig6, SMOKE grid, " \
           "4/4 cells present" in p.stdout
    v = json.loads(out.read_text())
    assert v["wave"] == "section4_gate_anch2_fig6" and v["smoke"] is True
    assert v["n_cells_total"] == 4 and v["n_runs"] == 4
    assert {c["tag"] for c in v["cells"]} == {
        tag_for(*c, smoke=True) for c in fig6_smoke_cells()}
    assert v["n_twin_cells_total"] == 0 and v["n_ext_pending"] == 0

    # the v1 smoke cells (ea=1, es=0.2) are NOT fig6 smoke cells, and the
    # reverse: the two waves' smokes never stand in for each other
    v1root = build_wave(tmp_path / "v1smoke", smoke=True)
    q = run_checker(v1root, wave="fig6", smoke=True)
    assert q.returncode == 1, q.stdout
    assert "smoke cell must be ea0p1 es0p3 s0; got ea1 es0p2 s0" in q.stdout
    r = run_checker(root, wave="v1", smoke=True)
    assert r.returncode == 1, r.stdout
    assert "smoke cell must be ea1 es0p2 s0; got ea0p1 es0p3 s0" in r.stdout

    # honours the run root: a broken fig6 smoke elsewhere is invisible
    bad = build_fig6_smoke(
        tmp_path / "otherruns",
        per_cell=lambda c: ({"cfg_mut": lambda cfg: cfg.update(
            {"population_update": "nested_ai_then_social_v1"})}
            if c == ("b0", "fixed", 0.1, 0.3, 0) else {}))
    s = run_checker(bad, wave="fig6", smoke=True)
    assert s.returncode == 1, s.stdout
    p2 = run_checker(root, wave="fig6", smoke=True)
    assert p2.returncode == 0, p2.stdout

    # a missing fig6 smoke cell is a hard failure
    os.rename(os.path.join(root, tag_for("d8", "evolving", 0.1, 0.3, 0,
                                         smoke=True)),
              os.path.join(str(tmp_path), "parked"))
    m = run_checker(root, wave="fig6", smoke=True)
    assert m.returncode == 1, m.stdout
    assert "3 of 4 cells present -- 1 ABSENT" in m.stdout


def test_wave_aliases_select_the_same_grids(tmp_path):
    """v1 <-> section4_gate_anch2, fig6 <-> section4_gate_anch2_fig6, and
    the default is the original wave."""
    root = build_wave(tmp_path / "v1smoke", smoke=True)
    for w in (None, "v1", "section4_gate_anch2"):
        out = tmp_path / f"v_{w}.json"
        p = run_checker(root, wave=w, smoke=True, extra=["--json", str(out)])
        assert p.returncode == 0, p.stdout + p.stderr
        assert json.loads(out.read_text())["wave"] == "section4_gate_anch2"
    root6 = build_fig6_smoke(tmp_path / "fig6smoke")
    for w in ("fig6", "section4_gate_anch2_fig6"):
        out = tmp_path / f"v_{w}.json"
        p = run_checker(root6, wave=w, smoke=True,
                        extra=["--json", str(out)])
        assert p.returncode == 0, p.stdout + p.stderr
        assert json.loads(out.read_text())["wave"] == \
            "section4_gate_anch2_fig6"
