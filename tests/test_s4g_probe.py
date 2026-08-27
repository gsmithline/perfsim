"""Tests for the beta=0.75 SECTION-4 family:
  section4_gate_anch2_probe (2026-08-26): 8 jobs, 5 rounds, es {0, 1}
  section4_gate_anch2_scout (2026-08-27): 20 jobs, 30 rounds,
                                          es {0, .1, .2, .3, 1}
both seed 0, ea 0.7, arms {b0, d8} x conds {fixed, evolving}, and for
check_section4_gate.py --wave probe|scout.

The generator tests read the EMITTED artifacts on disk (the configs and
subs Condor actually consumes), with the grid restated independently --
a test that reads the grid from the thing under test cannot catch a
grid bug. The checker tests synthesize physically-consistent pofds4gp_
runs (the test_section4_gate_checker.py pattern) and assert the probe
wave PASSES a healthy set, FAILS the sabotage cases, and NEVER sees the
S4G/fig6 waves' dirs (the prefix-scoping lesson from the F4A sibling
waves).

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac):
  USE_TF=0 python -m pytest tests/test_s4g_probe.py -q
"""
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
CONDOR = os.path.join(REPO, "experiments", "condor")
CHECKER = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines",
                       "check_section4_gate.py")

# ------------------------------------------------- contract (restated)
N = 723
CLAMP_N = 145
ROUNDS = 5
SEED = 0
EA = 0.7
ESS = (0.0, 1.0)
W_PLAT = 0.75
ARMS = ("b0", "d8")
CONDS = ("fixed", "evolving")
COND_TOK = {"fixed": "fixb20", "evolving": "evoall"}
ICL_DAYS = 8
TARGET = "Action"
BASE = "mistralai/Mistral-7B-Instruct-v0.3"
KEYS = {"fixed": "section4_gate_anch2_probe_fixed",
        "evolving": "section4_gate_anch2_probe_evo"}
# the two variants, restated independently of the generator
QWEN3 = "Qwen/Qwen3-8B"
VARIANTS = {
    "probe": dict(prefix="pofds4gp", ess=(0.0, 1.0), rounds=5, n=8,
                  slug="mistral7b", base_model=BASE, thinking=None,
                  keys={"fixed": "section4_gate_anch2_probe_fixed",
                        "evolving": "section4_gate_anch2_probe_evo"}),
    "scout": dict(prefix="pofds4gs", ess=(0.0, 0.1, 0.2, 0.3, 1.0),
                  rounds=30, n=20, slug="mistral7b", base_model=BASE,
                  thinking=None,
                  keys={"fixed": "section4_gate_anch2_scout_fixed",
                        "evolving": "section4_gate_anch2_scout_evo"}),
    # the same scout on Qwen3-8B, thinking OFF (CHAT_THINKING=0 ->
    # config chat_thinking False), tag slug qwen3_8b
    "scout_qwen3": dict(prefix="pofds4gq", ess=(0.0, 0.1, 0.2, 0.3, 1.0),
                        rounds=30, n=20, slug="qwen3_8b", base_model=QWEN3,
                        thinking=False,
                        keys={"fixed": "section4_gate_anch2_scout_qwen3_fixed",
                              "evolving": "section4_gate_anch2_scout_qwen3_evo"}),
}
# the eps_AI sweep: the SAME wave's second arm -- es pinned at 0.3, ea
# varying, the shared (0.7, 0.3) corner EXCLUDED (already run under the
# es-sweep key). Restated independently of the generator.
EA_SWEEP = {
    "prefix": "pofds4gq", "slug": "qwen3_8b", "rounds": 15,
    "es": 0.3, "eas": (0.0, 0.3, 0.5, 0.7, 1.0), "n": 20,
    "keys": {"fixed": "section4_gate_anch2_scout_qwen3_ea_fixed",
             "evolving": "section4_gate_anch2_scout_qwen3_ea_evo"},
}

_G0 = torch.Generator().manual_seed(20260826)
INNATE = torch.rand(N, generator=_G0)

ARM_CFG = {
    "b0": {"training_style": "sft", "kl_beta": 0.0, "icl_k": 0,
           "icl_days": 0, "use_lora": True, "fresh_each_round": True},
    "d8": {"training_style": "frozen", "kl_beta": 0.0, "icl_k": 0,
           "icl_days": ICL_DAYS, "use_lora": False,
           "fresh_each_round": False},
}


def _num(v):
    return f"{float(v):g}".replace(".", "p")


def tag_for(arm, cond, es, prefix="pofds4gp", slug="mistral7b", ea=0.7,
            rounds=None):
    return (f"{prefix}_{slug}_{arm}_{COND_TOK[cond]}_anch2_ea{_num(ea)}"
            f"_w0p75_l0p2_es{_num(es)}_s{SEED}"
            + ("" if rounds is None else f"_r{rounds}"))


# ------------------------------------------------ emitted-artifact tests
def _sub_and_rows(cond, variant="probe"):
    key = VARIANTS[variant]["keys"][cond]
    sub = open(os.path.join(CONDOR, f"at_pofd_{key}.sub")).read()
    rows = [ln for ln in open(os.path.join(
        CONDOR, f"configs_pofd_{key}.txt")).read().splitlines()
        if ln.strip()]
    return sub, rows


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_emitted_grid_and_columns(variant):
    v = VARIANTS[variant]
    all_tags = set()
    for cond in CONDS:
        sub, rows = _sub_and_rows(cond, variant)
        assert len(rows) == v["n"] // 2
        # column positions come off the sub's own queue line -- the only
        # definition Condor reads
        q = next(l for l in sub.splitlines() if l.startswith("queue"))
        m = re.match(r"queue\s+(.*?)\s+from\s+(\S+)", q)
        cols = [c.strip() for c in m.group(1).split(",")]
        assert m.group(2).endswith(f"configs_pofd_{v['keys'][cond]}.txt")
        got = set()
        for r in rows:
            c = [x.strip() for x in r.split(",")]
            assert len(c) == len(cols), (cond, len(c), len(cols))
            row = dict(zip(cols, c))
            assert row["wplat"] == "0.75" and row["eps_ai"] == "0.7"
            assert row["gamma"] == "0.0" and row["mode"] == "loop"
            assert row["seed"] == "0" and row["nrounds"] == str(v["rounds"])
            assert row["gatemode"] == "threshold"
            assert float(row["eps"]) in v["ess"]
            arm = "d8" if row["style"] == "frozen" else "b0"
            assert row["icldays"] == ("8" if arm == "d8" else "0")
            assert row["uselora"] == ("0" if arm == "d8" else "1")
            if cond == "fixed":
                assert row["cmode"] == "bottom" and row["sftexcl"] == "0"
            else:
                assert "cmode" not in row and "sftexcl" not in row
            assert row["tag"] == tag_for(arm, cond, float(row["eps"]),
                                         v["prefix"], v["slug"])
            got.add((arm, float(row["eps"])))
            all_tags.add(row["tag"])
        assert got == {(a, e) for a in ARMS for e in v["ess"]}
        env = next(l for l in sub.splitlines()
                   if l.startswith("environment"))
        for tok in ("AI_GATE_REFERENCE=anchor", "SAVE_RAW_GEN=1",
                    "PARSE_MODE=strict", "DEFFUANT_ALPHA=0.5",
                    "WITH_TWIN=1", "INNATE_LAMBDA=0.2",
                    "KL_DIRECTION=forward",
                    f"BASE_MODEL={v['base_model']} "):
            assert tok in env, (cond, tok)
        assert ("CHAT_THINKING=0" in env) == (v["thinking"] is False)
        assert f"WANDB_RUN_SUFFIX=_{v['slug']}_pofds4g_" in env
        assert ("INNATE_CLAMP_PEER_MODE=stubborn" in env) == \
            (cond == "fixed")
        assert f"submit_pofd_sweep.sh <BID> {v['keys'][cond]}" in sub
        assert f"--wave {variant}" in sub
    assert len(all_tags) == v["n"]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_family_tags_disjoint_from_every_other_s4g_key(variant):
    mine = set()
    for cond in CONDS:
        _, rows = _sub_and_rows(cond, variant)
        mine |= {r.split(",")[0].strip() for r in rows}
    others = set()
    for f in os.listdir(CONDOR):
        if f.startswith("configs_pofd_section4_gate_anch2") and \
                variant not in f:
            others |= {ln.split(",")[0].strip()
                       for ln in open(os.path.join(CONDOR, f))
                       if ln.strip()}
    assert others and not (mine & others)
    # and the prefix itself can never be swallowed by the S4G scans
    pre = VARIANTS[variant]["prefix"] + "_"
    assert all(t.startswith(pre) and not t.startswith("pofds4g_")
               for t in mine)


def test_submit_script_knows_the_probe_keys():
    src = open(os.path.join(CONDOR, "submit_pofd_sweep.sh")).read()
    for k in ("section4_gate_anch2_probe_fixed",
              "section4_gate_anch2_probe_evo"):
        assert f'{k}) TARGETS="$WHAT" ;;' in src.replace(
            "section4_gate_anch2_probe_fixed|section4_gate_anch2_probe_evo)",
            "section4_gate_anch2_probe_evo)") or k in src
    assert ('section4_gate_anch2_probe) TARGETS='
            '"section4_gate_anch2_probe_fixed '
            'section4_gate_anch2_probe_evo" ;;') in src
    assert ('section4_gate_anch2_scout) TARGETS='
            '"section4_gate_anch2_scout_fixed '
            'section4_gate_anch2_scout_evo" ;;') in src
    assert ('section4_gate_anch2_scout_qwen3) TARGETS='
            '"section4_gate_anch2_scout_qwen3_fixed '
            'section4_gate_anch2_scout_qwen3_evo" ;;') in src


# --------------------------------------------------- synthetic runs
def make_cfg(tag, arm, cond, es, rounds=ROUNDS, base_model=BASE,
             thinking=None, ea=EA):
    c = {
        "run_tag": tag, "base_model": base_model, "dataset": "movielens",
        "ml_target": TARGET, "n_rounds": rounds, "seed": SEED,
        "eps": es, "eps_ai": ea,
        "ai_gate_mode": "threshold", "peer_gate_mode": "threshold",
        "ai_gate_reference": "anchor",
        "population_update": "nested_ai_anchored_then_social_v2",
        "gamma_bias": 0.0, "w_plat": W_PLAT, "innate_lambda": 0.2,
        "parse_mode": "strict", "deffuant_alpha": 0.5,
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
        "hardware": {"hostname": "g204",
                     "gpu_name": "NVIDIA H100 80GB HBM3"},
    }
    c.update(ARM_CFG[arm])
    if thinking is not None:
        c["chat_thinking"] = thinking      # recorded only when CHAT_THINKING is set
    if cond == "fixed":
        c.update({"innate_clamp_mode": "bottom", "innate_clamp_frac": 0.2,
                  "innate_clamp_seed": SEED,
                  "innate_clamp_peer_mode": "stubborn"})
    return c


def bottom_mask(innate, k):
    n = int(innate.numel())
    order = sorted(range(n), key=lambda i: (float(innate[i]), i))
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.tensor(sorted(order[:k]), dtype=torch.long)] = True
    return m


def clamp_hash(mask):
    return hashlib.sha256(
        mask.detach().cpu().to(torch.uint8).numpy().tobytes()).hexdigest()


def render_days(vals):
    return (f"This user's own opinion of {TARGET} movies over the most "
            f"recent days (oldest to newest): "
            + ", ".join(f"{v:.2f}" for v in vals) + ".")


def build_run(root, arm, cond, es, cfg_mut=None, raw_mut=None,
              prefix="pofds4gp", rounds=ROUNDS, slug="mistral7b",
              base_model=BASE, thinking=None, ea=EA, tag_rounds=None):
    tag = tag_for(arm, cond, es, prefix, slug, ea, tag_rounds)
    d = os.path.join(str(root), tag)
    os.makedirs(d, exist_ok=True)
    mask = bottom_mask(INNATE, CLAMP_N) if cond == "fixed" else None

    op, tw, pr = [], [], []
    x, y = INNATE.clone(), INNATE.clone()
    for t in range(rounds):
        # the twin depends on (cond, es, seed) only -- NEVER the arm --
        # which is the twin-agreement invariant the gate replays
        y = y + 0.01 * ((0.35 + 0.02 * es) - y)
        if float(ea) == 0.0:
            # the strict-< AI gate NEVER opens at eps_AI = 0: the served
            # vector cannot enter, so the population IS the twin
            x = y.clone()
        else:
            x = x + 0.02 * ((0.5 + 0.01 * es) - x)
        if mask is not None:
            x[mask] = INNATE[mask]
            y[mask] = INNATE[mask]
        op.append(x.clone())
        tw.append(y.clone())
        pr.append(torch.full((N,), 0.55 + 0.005 * t))

    cfg = make_cfg(tag, arm, cond, es, rounds, base_model, thinking, ea)
    if cfg_mut:
        cfg_mut(cfg)
    contact = 0.0 if float(ea) == 0.0 else 0.4
    payload = {
        "config": cfg,
        "trajectory": [{"round": t, "contact": contact}
                       for t in range(rounds)],
        "op_raw": torch.stack(op),
        "twin_raw": torch.stack(tw),
        "pred_raw": torch.stack(pr),
        "innate": INNATE.clone(),
    }
    if mask is not None:
        touch = torch.zeros(rounds, N, dtype=torch.bool)
        touch[:, ~mask] = True
        payload.update({
            "innate_clamp_mask": mask.clone(),
            "innate_clamp_count": int(mask.sum()),
            "innate_clamp_mode": "bottom",
            "innate_clamp_frac": 0.2,
            "innate_clamp_seed": SEED,
            "innate_clamp_hash": clamp_hash(mask),
            "innate_clamp_peer_mode": "stubborn",
            "clamp_fr_touch_raw": touch,
        })
    torch.save(payload, os.path.join(d, "trajectory.pt"))
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh, default=str)
    with open(os.path.join(d, "telemetry.json"), "w") as fh:
        for t in range(rounds):
            fh.write(json.dumps({"round": t, "deployment": t,
                                 "is_deploy": 1,
                                 "contact": contact}) + "\n")
    raws = [{"round": t, "parse_fail_frac": 0.0,
             "raw": ["0.55"] * N, "parsed": [0.55] * N}
            for t in range(rounds)]
    if raw_mut:
        raw_mut(raws)
    with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt",
                   compresslevel=1) as fh:
        for r in raws:
            fh.write(json.dumps(r) + "\n")
    if arm == "d8":
        hist = [INNATE.tolist()]
        with gzip.open(os.path.join(d, "icl_days_log.json.gz"), "wt",
                       compresslevel=1) as fh:
            for t in range(rounds):
                win = hist[-ICL_DAYS:]
                fh.write(json.dumps({
                    "round": t,
                    "ctx": [render_days([h[i] for h in win])
                            for i in range(N)]}) + "\n")
                hist.append(payload["op_raw"][t].tolist())
    return d


def variant_points(variant):
    """[(ea, es, rounds, tag_rounds)] -- the variant's cells, restated
    independently of the generator. scout_qwen3 is a CROSS: its es sweep
    at ea=0.7 runs 30 rounds (bare tags) and its ea sweep at es=0.3 runs
    15 (tags carry _r15)."""
    v = VARIANTS[variant]
    pts = [(0.7, es, v["rounds"], None) for es in v["ess"]]
    if variant == "scout_qwen3":
        e = EA_SWEEP
        pts += [(ea, e["es"], e["rounds"], e["rounds"])
                for ea in e["eas"]]
    return pts


def build_probe(root, skip=(), cfg_mut_on=None, cfg_mut=None,
                raw_mut_on=None, raw_mut=None, variant="probe"):
    v = VARIANTS[variant]
    for cond in CONDS:
        for arm in ARMS:
            for (ea, es, nr, tag_r) in variant_points(variant):
                cell = (arm, cond, es) if tag_r is None else \
                    (arm, cond, es, ea)
                if cell in skip or (arm, cond, es) in skip:
                    continue
                build_run(root, arm, cond, es,
                          cfg_mut=cfg_mut if cell == cfg_mut_on else None,
                          raw_mut=raw_mut if cell == raw_mut_on else None,
                          prefix=v["prefix"], rounds=nr,
                          slug=v["slug"], base_model=v["base_model"],
                          thinking=v["thinking"], ea=ea, tag_rounds=tag_r)
    return root


def run_checker(root, extra=(), wave="probe"):
    env = dict(os.environ)
    env["USE_TF"] = "0"
    cmd = [sys.executable, CHECKER, "--run-root", str(root),
           "--wave", wave] + list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=REPO)


# ---------------------------------------------------------- healthy
@pytest.mark.slow
def test_probe_full_set_passes_and_ignores_foreign_dirs(tmp_path):
    root = build_probe(tmp_path / "runs")
    # foreign S4G/fig6/smoke dirs beside the probe must be INVISIBLE to
    # the probe scan (the F4A sibling-wave lesson), not EXTRA failures
    for foreign in ("pofds4g_mistral7b_b0_fixb20_anch2_ea1_w0p5_l0p2"
                    "_es1_s0",
                    "pofds4gsmk_mistral7b_b0_fixb20_anch2_ea1_w0p5"
                    "_l0p2_es0p2_s0"):
        os.makedirs(root / foreign, exist_ok=True)
    r = run_checker(root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout and "8/8 cells present" in r.stdout


@pytest.mark.slow
def test_v1_wave_never_sees_probe_dirs(tmp_path):
    root = build_probe(tmp_path / "runs")
    env = dict(os.environ)
    env["USE_TF"] = "0"
    r = subprocess.run([sys.executable, CHECKER, "--run-root", str(root),
                        "--wave", "v1"], capture_output=True, text=True,
                       env=env, cwd=REPO)
    # nothing to gate (the probe dirs are not pofds4g_*) is a USAGE
    # error, never a pass and never an EXTRA report
    assert r.returncode == 2, r.stdout + r.stderr
    assert "no pofds4g_*" in r.stderr


def test_smoke_flag_is_a_usage_error(tmp_path):
    (tmp_path / "runs").mkdir()
    r = run_checker(tmp_path / "runs", extra=("--smoke",))
    assert r.returncode == 2
    assert "no smoke" in r.stderr


# ---------------------------------------------------------- sabotage
@pytest.mark.slow
def test_wrong_wplat_in_config_fails(tmp_path):
    def mut(c):
        c["w_plat"] = 0.5
    root = build_probe(tmp_path / "runs", cfg_mut_on=("b0", "fixed", 1.0),
                       cfg_mut=mut)
    r = run_checker(root)
    assert r.returncode == 1
    assert "w_plat=0.5" in r.stdout


@pytest.mark.slow
def test_legacy_parse_mode_fails(tmp_path):
    def mut(c):
        del c["parse_mode"]
    root = build_probe(tmp_path / "runs", cfg_mut_on=("d8", "evolving", 0.0),
                       cfg_mut=mut)
    r = run_checker(root)
    assert r.returncode == 1
    assert "parse_mode" in r.stdout


@pytest.mark.slow
def test_parse_failure_fails(tmp_path):
    def mut(raws):
        raws[2]["parse_fail_frac"] = 2.0 / N
        raws[2]["raw"] = ["#.69"] * 2 + ["0.55"] * (N - 2)
        raws[2]["parsed"] = [0.5] * 2 + [0.55] * (N - 2)
    root = build_probe(tmp_path / "runs", raw_mut_on=("b0", "evolving", 1.0),
                       raw_mut=mut)
    r = run_checker(root)
    assert r.returncode == 1
    assert "parse failures" in r.stdout


@pytest.mark.slow
def test_missing_cell_is_a_hard_failure(tmp_path):
    root = build_probe(tmp_path / "runs", skip=(("d8", "fixed", 1.0),))
    r = run_checker(root)
    assert r.returncode == 1
    assert "ABSENT" in r.stdout


# ---------------------------------------------------------------- scout
@pytest.mark.slow
def test_scout_full_set_passes_and_ignores_probe_dirs(tmp_path):
    """The 20-cell 30-round scout passes; the probe's finished run dirs
    (and S4G/smoke dirs) beside it are invisible to the scout scan."""
    root = build_probe(tmp_path / "runs", variant="scout")
    build_probe(root, variant="probe")            # the real situation
    os.makedirs(root / ("pofds4g_mistral7b_b0_fixb20_anch2_ea1_w0p5_l0p2"
                        "_es1_s0"), exist_ok=True)
    r = run_checker(root, wave="scout")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout and "20/20 cells present" in r.stdout
    assert "n_rounds" not in r.stdout          # every cell at 30 rounds
    # and the probe gate, on the same root, still sees exactly its 8
    r = run_checker(root, wave="probe")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "8/8 cells present" in r.stdout and "EXTRA" not in r.stdout


@pytest.mark.slow
def test_scout_cell_at_the_probe_horizon_fails(tmp_path):
    """A 5-round artifact wearing a scout tag is a short run, never a
    scout cell."""
    root = build_probe(tmp_path / "runs", variant="scout",
                       skip=(("d8", "fixed", 0.2),))
    build_run(root, "d8", "fixed", 0.2, prefix="pofds4gs", rounds=5)
    r = run_checker(root, wave="scout")
    assert r.returncode == 1
    assert "n_rounds=5, expected 30" in r.stdout


@pytest.mark.slow
def test_scout_missing_social_gate_is_absent(tmp_path):
    root = build_probe(tmp_path / "runs", variant="scout",
                       skip=(("b0", "evolving", 0.3),))
    r = run_checker(root, wave="scout")
    assert r.returncode == 1
    assert "ABSENT" in r.stdout and "es0p3" in r.stdout


# ---------------------------------------------------------- scout_qwen3
@pytest.mark.slow
def test_scout_qwen3_full_set_passes_beside_the_mistral_scout(tmp_path):
    root = build_probe(tmp_path / "runs", variant="scout_qwen3")
    build_probe(root, variant="scout")            # the Mistral cells too
    r = run_checker(root, wave="scout_qwen3")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout and "40/40 cells present" in r.stdout
    assert "EXTRA" not in r.stdout
    # and the Mistral scout gate still sees exactly its own 20
    r = run_checker(root, wave="scout")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "20/20 cells present" in r.stdout and "EXTRA" not in r.stdout


@pytest.mark.slow
def test_scout_qwen3_run_on_the_wrong_model_fails(tmp_path):
    """A Mistral config under a qwen3_8b tag, or thinking left on, is
    not a Qwen3 scout cell."""
    def wrong_model(c):
        c["base_model"] = BASE
    root = build_probe(tmp_path / "runs", variant="scout_qwen3",
                       cfg_mut_on=("b0", "fixed", 0.3), cfg_mut=wrong_model)
    r = run_checker(root, wave="scout_qwen3")
    assert r.returncode == 1
    assert "base_model='mistralai/Mistral-7B-Instruct-v0.3'" in r.stdout

    def thinking_on(c):
        c["chat_thinking"] = True
    root = build_probe(tmp_path / "runs2", variant="scout_qwen3",
                       cfg_mut_on=("d8", "evolving", 1.0),
                       cfg_mut=thinking_on)
    r = run_checker(root, wave="scout_qwen3")
    assert r.returncode == 1
    assert "chat_thinking=True" in r.stdout


def test_scout_qwen3_tag_slug_parses_with_its_underscore():
    """qwen3_8b carries an underscore; the checker's grammar must still
    split it from the arm token (and never mis-parse a Mistral tag)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("chk", CHECKER)
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    m = chk.TAG_RE.match(tag_for("d8", "evolving", 0.3, "pofds4gq",
                                 "qwen3_8b"))
    assert m and m.group("slug") == "qwen3_8b" and m.group("arm") == "d8"
    m = chk.TAG_RE.match(tag_for("b0", "fixed", 1.0, "pofds4gs"))
    assert m and m.group("slug") == "mistral7b" and m.group("arm") == "b0"


# ------------------------------------------------------- the ea sweep
def test_ea_sweep_emitted_grid():
    """es pinned, ea varying, the shared corner never re-queued."""
    v = EA_SWEEP
    all_tags = set()
    for cond in CONDS:
        key = v["keys"][cond]
        sub = open(os.path.join(CONDOR, f"at_pofd_{key}.sub")).read()
        rows = [ln for ln in open(os.path.join(
            CONDOR, f"configs_pofd_{key}.txt")).read().splitlines()
            if ln.strip()]
        assert len(rows) == v["n"] // 2
        q = next(l for l in sub.splitlines() if l.startswith("queue"))
        cols = [c.strip() for c in
                re.match(r"queue\s+(.*?)\s+from\s+(\S+)", q).group(1).split(",")]
        got = set()
        for r in rows:
            row = dict(zip(cols, [x.strip() for x in r.split(",")]))
            assert float(row["eps"]) == v["es"], "the ea sweep holds es"
            assert float(row["eps_ai"]) in v["eas"]
            assert row["wplat"] == "0.75"
            assert row["nrounds"] == str(v["rounds"]), "15-round sweep"
            assert row["seed"] == "0" and row["gamma"] == "0.0"
            arm = "d8" if row["style"] == "frozen" else "b0"
            # the SHORTER horizon must be declared in the tag, or the
            # idempotent wrapper could no-op it against a 30-round dir
            assert row["tag"].endswith(f"_r{v['rounds']}")
            assert row["tag"] == tag_for(arm, cond, v["es"], v["prefix"],
                                         v["slug"], float(row["eps_ai"]),
                                         rounds=v["rounds"])
            got.add((arm, float(row["eps_ai"])))
            all_tags.add(row["tag"])
        assert got == {(a, e) for a in ARMS for e in v["eas"]}
        # the ea sweep must run the SAME environment as the es sweep
        main = open(os.path.join(
            CONDOR, f"at_pofd_{VARIANTS['scout_qwen3']['keys'][cond]}.sub")).read()
        env_of = lambda s: next(l for l in s.splitlines()
                                if l.startswith("environment"))
        assert env_of(sub) == env_of(main)
    assert len(all_tags) == v["n"]


def test_ea_sweep_tags_disjoint_from_the_es_sweep():
    ea, es = set(), set()
    for cond in CONDS:
        ea |= {ln.split(",")[0].strip() for ln in open(os.path.join(
            CONDOR, f"configs_pofd_{EA_SWEEP['keys'][cond]}.txt"))
            if ln.strip()}
        es |= {ln.split(",")[0].strip() for ln in open(os.path.join(
            CONDOR,
            f"configs_pofd_{VARIANTS['scout_qwen3']['keys'][cond]}.txt"))
            if ln.strip()}
    assert len(ea) == 20 and len(es) == 20
    assert not (ea & es), "a finished cell would be re-queued"
    # 10 (point, horizon) cells x 2 arms x 2 conds = 40
    assert len(ea | es) == 40
    # the shared corner appears at both horizons, differing ONLY by _r15
    c30 = tag_for("b0", "fixed", 0.3, "pofds4gq", "qwen3_8b", ea=0.7)
    c15 = tag_for("b0", "fixed", 0.3, "pofds4gq", "qwen3_8b", ea=0.7,
                  rounds=15)
    assert c30 in es and c15 in ea and c15 == f"{c30}_r15"


def test_submit_script_knows_the_ea_sweep():
    src = open(os.path.join(CONDOR, "submit_pofd_sweep.sh")).read()
    assert ('section4_gate_anch2_scout_qwen3_ea) TARGETS='
            '"section4_gate_anch2_scout_qwen3_ea_fixed '
            'section4_gate_anch2_scout_qwen3_ea_evo" ;;') in src


def test_checker_rejects_an_off_cross_cell():
    """The wave is a CROSS: (ea=1, es=1) uses values that each appear on
    one arm but is not a cell, and must not be silently admitted."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("chk", CHECKER)
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    gen = chk.load_generator(os.path.join(CONDOR, "gen_pofd_sweep.py"))
    w = chk.Wave("scout_qwen3", gen)
    # 9 DISTINCT (ea, es) points; 10 (point, horizon) cells, because the
    # shared corner runs at both horizons
    assert len(w.points) == 9 and len(w.pt_keys) == 10
    assert w.horizons_ok == (15,) and w.min_rounds == 15
    off = tag_for("b0", "fixed", 1.0, "pofds4gq", "qwen3_8b", ea=1.0,
                  rounds=15)
    info, errs = chk.parse_tag(off, False, w)
    assert any("is not a point" in e for e in errs), errs
    for on in (tag_for("b0", "fixed", 0.3, "pofds4gq", "qwen3_8b", ea=1.0,
                       rounds=15),
               tag_for("d8", "evolving", 1.0, "pofds4gq", "qwen3_8b", ea=0.7),
               tag_for("d8", "evolving", 0.3, "pofds4gq", "qwen3_8b", ea=0.0,
                       rounds=15)):
        info, errs = chk.parse_tag(on, False, w)
        assert not errs, (on, errs)


def test_checker_enforces_the_per_point_horizon():
    """An ea-sweep cell without _r15 is a 30-round cell; an es-sweep cell
    with _r15 is not a cell at all. Both must FAIL, or the idempotent
    wrapper could pair a 15-round job with a 30-round run dir."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("chk", CHECKER)
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    gen = chk.load_generator(os.path.join(CONDOR, "gen_pofd_sweep.py"))
    w = chk.Wave("scout_qwen3", gen)
    bare = tag_for("b0", "fixed", 0.3, "pofds4gq", "qwen3_8b", ea=0.3)
    assert any("wrong horizon" in e
               for e in chk.parse_tag(bare, False, w)[1])
    tagged = tag_for("d8", "evolving", 1.0, "pofds4gq", "qwen3_8b",
                     ea=0.7, rounds=15)
    assert any("wrong horizon" in e
               for e in chk.parse_tag(tagged, False, w)[1])


@pytest.mark.slow
def test_ea0_cell_that_served_anything_fails(tmp_path):
    """The eps_AI=0 cells are WITNESSES: the strict-< gate never opens,
    so op_raw must BE the twin. A cell where the served vector reached
    the population must fail even though every other field is perfect."""
    root = build_probe(tmp_path / "runs", variant="scout_qwen3")
    tag = tag_for("b0", "fixed", 0.3, "pofds4gq", "qwen3_8b", ea=0.0,
                  rounds=15)
    p = os.path.join(str(root), tag, "trajectory.pt")
    d = torch.load(p, weights_only=False)
    d["op_raw"][3] = d["op_raw"][3] + 0.02      # the platform got in
    torch.save(d, p)
    r = run_checker(root, wave="scout_qwen3")
    assert r.returncode == 1
    assert "op_raw != twin_raw" in r.stdout
