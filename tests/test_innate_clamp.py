"""Focused tests for the no-peer innate-clamp intervention (2026-08-17).

Cohort helper (_gated_pop.innate_clamp_mask): determinism, global-RNG
independence, bottom selection with the agent-id tie-break, stratified
quintile quotas at 723/0.20 -> exactly 145 (29 per quintile), and
paired-mask equality across arms/gates within a seed.

Checker (check_pofd_sanity CLAMP branch), via synthetic pofdclamp_
fixtures: healthy b0/dyn/smoke runs PASS; corrupted masks, frozen-agent
drift (deployed and twin), clamping beyond the mask, a peer-dose tag,
an all-open gate, a mode-token mismatch, and live-ICL contexts that
exclude frozen agents all FAIL.

Legacy mode-off non-regression: an untouched pofdctxgrid_ es=0 fixture
(no clamp keys anywhere) still passes the extended checker -- the CLAMP
branch must never fire outside its own family. The runner-side guard
(clamp + nonzero EPS_SOCIAL hard-fails before any model load) is
exercised in-process by test_runner_guard_source below; byte-identity
of off-mode runs is enforced by construction (every clamp write is
gated on mode != "off" and the mask generator never touches the global
RNG -- see test_mask_global_rng_independence).

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

_spec_gp = importlib.util.spec_from_file_location(
    "gp_clamp_test", os.path.join(PIPE, "_gated_pop.py"))
gp = importlib.util.module_from_spec(_spec_gp)
_spec_gp.loader.exec_module(gp)

_spec_cg = importlib.util.spec_from_file_location(
    "ctxgrid_fixtures", os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_ctxgrid_checker.py"))
cgfix = importlib.util.module_from_spec(_spec_cg)
_spec_cg.loader.exec_module(cgfix)

N = 60
FRAC = 0.2
N_FROZEN = round(FRAC * N)          # 12 of 60 (145 of 723 in production)
_G0 = torch.Generator().manual_seed(20260817)
INNATE = torch.rand(N, generator=_G0)
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
MODE_OF_TOK = {"strat": "stratified_random", "bottom": "bottom"}


def _num(v):
    return f"{v:g}".replace(".", "p")


# -- cohort helper -------------------------------------------------------

def test_mask_determinism():
    m1 = gp.innate_clamp_mask(INNATE, "stratified_random", FRAC, 42)
    m2 = gp.innate_clamp_mask(INNATE, "stratified_random", FRAC, 42)
    assert torch.equal(m1, m2)
    assert int(m1.sum()) == N_FROZEN
    assert not torch.equal(
        m1, gp.innate_clamp_mask(INNATE, "stratified_random", FRAC, 43))


def test_mask_global_rng_independence():
    torch.manual_seed(1)
    m1 = gp.innate_clamp_mask(INNATE, "stratified_random", FRAC, 42)
    torch.manual_seed(999999)
    m2 = gp.innate_clamp_mask(INNATE, "stratified_random", FRAC, 42)
    assert torch.equal(m1, m2)
    # and building a mask must not advance the global stream
    torch.manual_seed(7)
    a = torch.rand(4)
    torch.manual_seed(7)
    gp.innate_clamp_mask(INNATE, "stratified_random", FRAC, 42)
    b = torch.rand(4)
    assert torch.equal(a, b)


def test_bottom_selection_lowest_with_id_tiebreak():
    # ties at the cutoff: agents 5..14 share one value; the id tie-break
    # must admit the LOWEST ids among them, deterministically
    innate = torch.linspace(0.3, 0.9, N)
    innate[5:15] = 0.05
    m = gp.innate_clamp_mask(innate, "bottom", FRAC, 0)
    order = sorted(range(N), key=lambda i: (float(innate[i]), i))
    assert set(m.nonzero().flatten().tolist()) == set(order[:N_FROZEN])
    assert m[5] and m[6] and not m[15]
    # seed cannot matter for bottom
    assert torch.equal(m, gp.innate_clamp_mask(innate, "bottom", FRAC, 77))


def test_stratified_counts_production_size():
    g = torch.Generator().manual_seed(3)
    innate = torch.rand(723, generator=g)
    m = gp.innate_clamp_mask(innate, "stratified_random", 0.2, 42)
    assert int(m.sum()) == 145
    order = sorted(range(723), key=lambda i: (float(innate[i]), i))
    cuts = gp._largest_remainder([1] * 5, 723)
    assert cuts == [145, 145, 145, 144, 144]
    lo = 0
    for c in cuts:
        b = torch.tensor(order[lo:lo + c])
        lo += c
        assert int(m[b].sum()) == 29   # 145 split 29 per quintile


def test_paired_mask_equality_within_seed():
    # the mask depends on (innate, mode, frac, seed) ONLY -- SFT vs ICL
    # arms and every AI gate at one seed must share the cohort
    ref = gp.innate_clamp_mask(INNATE, "stratified_random", FRAC, 42)
    for _arm in ("b0", "dyn"):
        for _gate in (0.05, 0.1, 0.2, 0.4, 1.0):
            assert torch.equal(ref, gp.innate_clamp_mask(
                INNATE, "stratified_random", FRAC, 42))


def test_mask_rejects_bad_modes_and_fracs():
    with pytest.raises(ValueError):
        gp.innate_clamp_mask(INNATE, "off", FRAC, 0)
    with pytest.raises(ValueError):
        gp.innate_clamp_mask(INNATE, "top", FRAC, 0)
    with pytest.raises(ValueError):
        gp.innate_clamp_mask(INNATE, "bottom", 0.0, 0)
    with pytest.raises(ValueError):
        gp.innate_clamp_mask(INNATE, "bottom", 1.0, 0)


def test_hash_changes_on_bit_flip():
    m = gp.innate_clamp_mask(INNATE, "bottom", FRAC, 0)
    h1 = gp.innate_clamp_hash(m)
    m2 = m.clone()
    m2[0] = ~m2[0]
    assert gp.innate_clamp_hash(m2) != h1


# -- checker fixtures ----------------------------------------------------

def cfg_for(tag, arm, mode_tok, gate, seed, nrounds):
    c = {"run_tag": tag, "base_model": MISTRAL, "n_rounds": nrounds,
         "seed": seed, "eps": 0.0, "gamma_bias": 0.0, "w_plat": 0.5,
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
         "eps_ai": gate, "ai_gate_mode": "threshold", "host": "g204",
         "innate_clamp_mode": MODE_OF_TOK[mode_tok],
         "innate_clamp_frac": FRAC, "innate_clamp_seed": seed,
         "hardware": {"hostname": "g204", "gpu_name": "A100",
                      "gpu_cc": "8.0", "cuda_version": "12.4",
                      "torch_version": "2.5.1",
                      "transformers_version": "4.46.0"}}
    if arm == "b0":
        c.update({"training_style": "sft", "kl_beta": 0.0,
                  "use_lora": True, "lora_r": 512, "sft_lr": 5e-5,
                  "sft_epochs": 1, "sft_batch_size": 4,
                  "fresh_each_round": True, "train_cap": 723,
                  "icl_k": 0, "icl_days": 0, "icl_select": "random",
                  "icl_ctx_source": "live"})
    else:
        c.update({"training_style": "frozen", "kl_beta": 0.0,
                  "use_lora": 0, "fresh_each_round": False,
                  "train_cap": 723, "icl_k": 8, "icl_days": 0,
                  "icl_select": "random", "icl_ctx_source": "live",
                  "icl_snapshot_round": -1})
    return c


def build(parent, arm, mode_tok, gate, seed=0, nrounds=30,
          prefix="pofdclamp", tag=None, cfg_mut=None, post=None,
          clamp_applied=True, exemplars_avoid_frozen=False):
    if tag is None:
        tag = (f"{prefix}_mistral7b_{arm}_{mode_tok}_ea{_num(gate)}"
               f"_w0p5_l0p2_es0_s{seed}")
    cfg = cfg_for(tag, arm, mode_tok, gate, seed, nrounds)
    if cfg_mut:
        cfg_mut(cfg)
    mask = gp.innate_clamp_mask(INNATE, MODE_OF_TOK[mode_tok],
                                FRAC, seed)
    g = torch.Generator().manual_seed(4000 + seed)
    x = INNATE.clone()
    tw = INNATE.clone()
    lam, w = 0.2, 0.5
    eps_ai = cfg["eps_ai"]
    rows, op_raw, pred_raw, gate_raw, twin_raw = [], [], [], [], []
    icl_idx, icl_val, ctx_rows = [], [], []
    resp_ids = (~mask).nonzero().flatten()
    for t in range(nrounds):
        pred = torch.rand(N, generator=g)
        served = pred.clamp(0.0, 1.0)
        gate_t = (served - x).abs() < eps_ai
        h = lam * INNATE + (1.0 - lam) * x
        x = torch.where(gate_t, (1.0 - w) * h + w * served, h)
        tw = lam * INNATE + (1.0 - lam) * tw
        if clamp_applied:
            x[mask] = INNATE[mask]
            tw[mask] = INNATE[mask]
        row = {"round": t, "deployment": t, "is_deploy": 1,
               "accepted": 0,
               "contact": float(gate_t.float().mean()),
               "twin_mean": float(tw.mean()), "twin_std": float(tw.std()),
               "twin_bias": float(tw.mean() - INNATE.mean())}
        if arm == "b0":
            row["n_train"] = 723
        else:
            row["perplexity"] = 7.77
        if arm == "dyn":
            if exemplars_avoid_frozen:
                # SABOTAGE: draw exemplars from responsive agents only
                sel = resp_ids[torch.randint(0, resp_ids.numel(), (N, 8),
                                             generator=g)]
                own = torch.arange(N).unsqueeze(1)
                sel = torch.where(sel == own, resp_ids[0].expand(N, 8),
                                  sel)
                sel = torch.where(sel == own, resp_ids[1].expand(N, 8),
                                  sel)
                ii = sel
            else:
                base = torch.stack([torch.tensor(
                    [(i + 1 + ((j + t) % (N - 1))) % N
                     for j in range(8)]) for i in range(N)])
                ii = base
            vv = torch.rand(N, 8, generator=g)
            icl_idx.append(ii.clone())
            icl_val.append(vv.clone())
            ctx_rows.append({"round": t,
                             "ctx": [f"ctx-clamp-{t}-{i}"
                                     for i in range(N)]})
        rows.append(row)
        op_raw.append(x.clone())
        pred_raw.append(pred.clone())
        gate_raw.append(gate_t.clone())
        twin_raw.append(tw.clone())
    d = {"trajectory": rows, "config": cfg,
         "op_raw": torch.stack(op_raw), "pred_raw": torch.stack(pred_raw),
         "gate_raw": torch.stack(gate_raw),
         "twin_raw": torch.stack(twin_raw), "innate": INNATE.clone(),
         "innate_clamp_mask": mask.clone(),
         "innate_clamp_count": int(mask.sum()),
         "innate_clamp_mode": cfg["innate_clamp_mode"],
         "innate_clamp_frac": FRAC,
         "innate_clamp_seed": cfg["innate_clamp_seed"],
         "innate_clamp_hash": gp.innate_clamp_hash(mask)}
    if icl_idx:
        d["icl_idx_raw"] = torch.stack(icl_idx)
        d["icl_val_raw"] = torch.stack(icl_val)
    if post:
        post(d)
    rd = parent / d["config"]["run_tag"]
    rd.mkdir(parents=True)
    (rd / "config.json").write_text(json.dumps(d["config"]))
    torch.save(d, rd / "trajectory.pt")
    if ctx_rows:
        with gzip.open(rd / "icl_ctx_log.json.gz", "wt") as fh:
            for r in ctx_rows:
                fh.write(json.dumps(r) + "\n")
    return rd


def run_check(rd):
    p = subprocess.run([sys.executable, CHECKER, str(rd)],
                       capture_output=True, text=True,
                       env={**os.environ, "USE_TF": "0"})
    return p.returncode, p.stdout + p.stderr


def edit(rd, fn):
    t = torch.load(rd / "trajectory.pt", weights_only=False)
    fn(t)
    torch.save(t, rd / "trajectory.pt")
    (rd / "config.json").write_text(json.dumps(t["config"]))


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

def test_healthy_b0_strat(tmp_path):
    assert_verdict(build(tmp_path, "b0", "strat", 0.4, seed=42), True)


def test_healthy_dyn_bottom(tmp_path):
    assert_verdict(build(tmp_path, "dyn", "bottom", 0.2), True)


def test_healthy_b0_full_open_numeric_gate(tmp_path):
    assert_verdict(build(tmp_path, "b0", "bottom", 1.0), True)


def test_healthy_smoke(tmp_path):
    assert_verdict(build(tmp_path, "dyn", "strat", 0.2, seed=991,
                         nrounds=3, prefix="pofdclampsmk"), True)


def test_paired_fixture_masks_identical(tmp_path):
    rd_b0 = build(tmp_path, "b0", "strat", 0.1, seed=42)
    rd_dyn = build(tmp_path, "dyn", "strat", 0.4, seed=42)
    m_b0 = torch.load(rd_b0 / "trajectory.pt",
                      weights_only=False)["innate_clamp_mask"]
    m_dyn = torch.load(rd_dyn / "trajectory.pt",
                       weights_only=False)["innate_clamp_mask"]
    assert torch.equal(m_b0, m_dyn)


# -- sabotage ------------------------------------------------------------

def test_corrupted_mask_detection(tmp_path):
    rd = build(tmp_path, "b0", "strat", 0.4)

    def fn(t):
        t["innate_clamp_mask"][3] = ~t["innate_clamp_mask"][3]
    edit(rd, fn)
    assert_verdict(rd, False, "CLAMP")


def test_corrupted_mask_with_consistent_hash_still_fails(tmp_path):
    # an attacker fixing count AND hash still trips reconstruction +
    # the mode-specific cohort property
    rd = build(tmp_path, "b0", "bottom", 0.4)

    def fn(t):
        m = t["innate_clamp_mask"]
        frozen = m.nonzero().flatten()
        free = (~m).nonzero().flatten()
        m[frozen[0]] = False
        m[free[-1]] = True
        t["innate_clamp_count"] = int(m.sum())
        t["innate_clamp_hash"] = gp.innate_clamp_hash(m)
    edit(rd, fn)
    assert_verdict(rd, False, "reconstruct")


def test_frozen_agent_drift_deployed(tmp_path):
    rd = build(tmp_path, "dyn", "bottom", 0.4)

    def fn(t):
        fro = t["innate_clamp_mask"].nonzero().flatten()
        t["op_raw"][10][fro[2]] += 1e-4
    edit(rd, fn)
    assert_verdict(rd, False, "drift off")


def test_frozen_agent_drift_twin(tmp_path):
    rd = build(tmp_path, "b0", "strat", 0.4)

    def fn(t):
        fro = t["innate_clamp_mask"].nonzero().flatten()
        t["twin_raw"][7][fro[0]] += 1e-5
    edit(rd, fn)
    assert_verdict(rd, False, "twin_raw")


def test_clamp_applied_beyond_mask(tmp_path):
    rd = build(tmp_path, "b0", "strat", 0.4)

    def fn(t):
        # freeze EVERYONE: responsive agents pinned to innate despite
        # gated contact with distinct served values
        t["op_raw"][:] = t["innate"].unsqueeze(0)
    edit(rd, fn)
    assert_verdict(rd, False, "beyond its mask")


def test_peer_dose_tag_rejected(tmp_path):
    rd = build(
        tmp_path, "b0", "strat", 0.4,
        tag="pofdclamp_mistral7b_b0_strat_ea0p4_w0p5_l0p2_es0p2_s0",
        cfg_mut=lambda c: c.update(eps=0.2))
    assert_verdict(rd, False, "reset-after-peer")


def test_all_open_gate_rejected(tmp_path):
    rd = build(
        tmp_path, "b0", "strat", 1.0,
        tag="pofdclamp_mistral7b_b0_strat_eaopen_w0p5_l0p2_es0_s0",
        cfg_mut=lambda c: c.update(ai_gate_mode="all_open"))
    assert_verdict(rd, False, "all_open")


def test_mode_token_mismatch(tmp_path):
    # tag says bottom, config + mask say stratified_random
    rd = build(
        tmp_path, "b0", "strat", 0.4,
        tag="pofdclamp_mistral7b_b0_bottom_ea0p4_w0p5_l0p2_es0_s0")
    assert_verdict(rd, False, "innate_clamp_mode")


def test_dyn_exemplars_must_include_frozen(tmp_path):
    rd = build(tmp_path, "dyn", "bottom", 0.4,
               exemplars_avoid_frozen=True)
    assert_verdict(rd, False, "remain eligible")


def test_missing_mask_fails(tmp_path):
    rd = build(tmp_path, "b0", "strat", 0.4)

    def fn(t):
        del t["innate_clamp_mask"]
    edit(rd, fn)
    assert_verdict(rd, False, "innate_clamp_mask missing")


# -- legacy mode-off non-regression --------------------------------------

def test_legacy_ctxgrid_fixture_still_passes(tmp_path):
    # an untouched pofdctxgrid_ es=0 run (no clamp keys anywhere) must
    # sail through the extended checker: the CLAMP branch is family-
    # scoped and off-mode runs carry no clamp surface at all
    rd = cgfix.build(tmp_path, "qwen7b", "b0", 0.4, 0.0)
    rc, out = run_check(rd)
    assert rc == 0, out[-2500:]
    assert "CLAMP" not in out


def test_clamp_prefix_captures_no_legacy_family():
    for legacy in ("pofdctxgrid_mistral7b_b0_ea0p4_w0p5_l0p2_es0_s0",
                   "pofdctf_qwen7b_pri_ea0p4_w0p5_l0p2_es0p2_s0",
                   "pofdreach_mistral7b_b0_ea1_w0p5_l0p2_es0_s0"):
        assert not legacy.startswith("pofdclamp")
    assert "pofdclampsmk_x".startswith("pofdclamp")


# -- runner guard (source-level) -----------------------------------------

def test_runner_guard_source():
    # the guard must hard-fail clamp + nonzero EPS_SOCIAL and reject
    # unknown modes BEFORE any model/dataset work; asserted on source to
    # avoid importing transformers in this suite
    src = open(os.path.join(PIPE, "run_pokec_gated_lm.py")).read()
    assert "INNATE_CLAMP_MODE" in src
    assert "reset-after-peer approximation" in src
    guard = src[src.index("innate_clamp_mode = "):]
    assert guard.index("reset-after-peer") < guard.index(
        "out_dir.mkdir"), \
        "clamp/peer guard must fire before any output is created"
    # every clamp write is gated on mode != 'off' (byte-identity of
    # off-mode runs): config keys and trajectory keys both conditional
    assert 'if innate_clamp_mode != "off":' in src
    assert src.count("innate_clamp_mask") >= 1
    assert "if clamp_mask is not None:" in src
