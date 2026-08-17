"""Focused tests for the Figure-2 FAMILY-PRIOR SCOUT (2026-08-17,
fig2_family_prior_scout): six checkpoints x b0/b0p5/b1/k0 x ea1
threshold x es {0.05, 0.2}, seed 0, canonical Action loop.

Generator: rows come from the field-level manifest (consistency
asserted, reuse split never forced); qwen3 rows pin CHAT_THINKING=0;
b0p5 is the b1 envelope at kl_beta=0.5; 3 b1 smokes for the three NEW
checkpoints. Runner: the qwen3 masking marker runs THROUGH the empty
think block and trained styles hard-require CHAT_THINKING=0. Audit:
clamp runs excluded by field, qwen3 requires chat_thinking=False,
new tags match the generator. Checker, via REAL-population es>0
fixtures (723 agents, artifact edges, real peer sweep): healthy cells
PASS per arm; wrong base_model, missing/stray chat_thinking, wrong
kl_beta, reverse KL, an off-grid es token and a non-b1 smoke FAIL.

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac).
"""
import importlib.util
import json
import os

import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")

_spec_tg = importlib.util.spec_from_file_location(
    "clamp_graph_fixtures_fam", os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_innate_clamp_graph.py"))
tg = importlib.util.module_from_spec(_spec_tg)
_spec_tg.loader.exec_module(tg)
GEN = tg.GEN
gp = tg.gp

_spec_au = importlib.util.spec_from_file_location(
    "audit_fam", os.path.join(PIPE, "audit_fig2_family_prior_reuse.py"))
AU = importlib.util.module_from_spec(_spec_au)
_spec_au.loader.exec_module(AU)

FAM_BASE = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "ministral8b": "mistralai/Ministral-8B-Instruct-2410",
}


def _num(v):
    return f"{v:g}".replace(".", "p")


# -- generator -----------------------------------------------------------

def test_generator_grid_and_manifest_consistency():
    mf = GEN._fam_manifest()
    cells = mf["cells"]
    assert len(cells) == 48
    assert mf["counts"]["reused"] + mf["counts"]["new"] == 48
    rows = GEN.fam_rows()
    assert len(rows) == mf["counts"]["new"]
    tags = [r.split(",")[0] for r in rows]
    assert len(set(tags)) == len(tags)
    assert all(t.startswith("pofdfam_") and "_ea1_" in t
               and t.endswith("_s0") for t in tags)
    # reused cells keep their ORIGINAL tags and never queue here
    reused_tags = {GEN.fam_tag(c["model"], c["arm"], c["eps_social"])
                   for c in cells if c["status"] == "reused"}
    assert not (set(tags) & reused_tags)
    # rows + reused = the full conceptual grid
    new_keys = {(c["model"], c["arm"], c["eps_social"])
                for c in cells if c["status"] == "new"}
    assert {t for t in tags} == {GEN.fam_tag(m, a, e)
                                 for m, a, e in new_keys}


def test_generator_row_payloads():
    rows = GEN.fam_rows()
    for r in rows:
        cols = [c.strip() for c in r.split(",")]
        tag = cols[0]
        # chatthink col: "0" iff qwen3_8b
        assert cols[24] == ("0" if "_qwen3_8b_" in tag else "default"), r
        # exact checkpoint id per slug
        slug = next(s for s in FAM_BASE if tag.startswith(f"pofdfam_{s}_"))
        assert cols[23] == FAM_BASE[slug], r
        if "_b0p5_" in tag:
            assert cols[1] == "sft_kl" and cols[2] == "0.5", r
        elif "_b1_" in tag:
            assert cols[1] == "sft_kl" and cols[2] == "1", r
        elif "_k0_" in tag:
            assert cols[1] == "frozen" and cols[18] == "0", r
        else:
            assert cols[1] == "sft" and cols[2] == "0", r
        # ea1 numeric threshold on every row
        assert cols[14] == "1" and cols[15] == "threshold", r


def test_generator_smokes():
    smk = GEN.fam_smoke_rows()
    assert len(smk) == 3
    tags = [r.split(",")[0] for r in smk]
    assert {t.split("_b1_")[0].replace("pofdfamsmk_", "")
            for t in tags} == {"qwen3_8b", "olmo3_7b", "ministral8b"}
    assert all(t.startswith("pofdfamsmk_") and "_b1_ea1_" in t
               and "_es0p2_s0" in t for t in tags)
    for r in smk:
        cols = [c.strip() for c in r.split(",")]
        assert cols[22] == "3", r    # 3-round smokes


def test_fam_sub_template_surface():
    sub = GEN.fam_sub("main")
    assert "CHAT_THINKING=$(chatthink)" in sub
    assert "BASE_MODEL=$(basemodel)" in sub
    assert "HF_HUB_OFFLINE=1" in sub
    assert "SAVE_RAW_GEN=1" in sub
    assert "KL_DIRECTION=forward" in sub
    assert "WITH_TWIN=1" in sub
    assert "request_memory    = $(mem)" in sub
    assert sub.rstrip().endswith(
        "mem, disk, pplbatch from experiments/condor/"
        "configs_pofd_fig2_family_prior_scout.txt")


# -- runner: qwen3 masking marker ----------------------------------------

def test_runner_qwen3_marker_and_guard():
    src = open(os.path.join(PIPE, "run_pokec_gated_lm.py")).read()
    assert 'resp_marker = "</think>\\n\\n"' in src
    assert "Qwen3 training requires CHAT_THINKING=0" in src
    # the guard covers every trained style
    assert 'training_style in ("sft", "sft_kl", "dpo")' in src


# -- audit ---------------------------------------------------------------

def test_audit_want_surfaces():
    w = AU.cell_want("qwen3_8b", "b0p5", 0.05)
    assert w["chat_thinking"] == (False,)
    assert w["kl_beta"] == (0.5,)
    assert w["kl_direction"] == ("forward",)
    assert w["eps"] == (0.05,) and w["eps_ai"] == (1.0,)
    assert w["innate_clamp_mode"] == (AU.AR.ABSENT,)
    w2 = AU.cell_want("mistral7b", "k0", 0.2)
    assert w2["chat_thinking"] == (AU.AR.ABSENT,)
    assert w2["training_style"] == ("frozen",) and w2["icl_k"] == (0,)


def test_audit_new_tags_match_generator():
    for model in FAM_BASE:
        for arm in ("b0", "b0p5", "b1", "k0"):
            for es in (0.05, 0.2):
                assert AU.new_tag(model, arm, es) == \
                    GEN.fam_tag(model, arm, es)


# -- checker fixtures (REAL population, live peer step) ------------------

N = tg.N
INNATE = tg.INNATE
ADJ = tg.ADJ


def fam_cfg(tag, model, arm, es, nrounds):
    c = {"run_tag": tag, "base_model": FAM_BASE[model],
         "n_rounds": nrounds, "seed": 0, "eps": es, "gamma_bias": 0.0,
         "w_plat": 0.5, "innate_lambda": 0.2,
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
         "eps_ai": 1.0, "ai_gate_mode": "threshold", "train_cap": 723,
         "save_raw_gen": True, "host": "g204",
         "hardware": {"hostname": "g204", "gpu_name": "A100",
                      "gpu_cc": "8.0", "cuda_version": "12.4",
                      "torch_version": "2.5.1",
                      "transformers_version": "4.46.0"}}
    if model == "qwen3_8b":
        c["chat_thinking"] = False
    if arm == "b0":
        c.update({"training_style": "sft", "kl_beta": 0.0,
                  "use_lora": 1, "lora_r": 512, "sft_lr": 5e-5,
                  "sft_epochs": 1, "sft_batch_size": 4,
                  "fresh_each_round": True, "icl_k": 0, "icl_days": 0})
    elif arm in ("b0p5", "b1"):
        c.update({"training_style": "sft_kl",
                  "kl_beta": 0.5 if arm == "b0p5" else 1.0,
                  "kl_direction": "forward", "use_lora": 1,
                  "lora_r": 512, "sft_lr": 5e-5, "sft_epochs": 1,
                  "sft_batch_size": 4, "fresh_each_round": True,
                  "icl_k": 0, "icl_days": 0})
    else:
        c.update({"training_style": "frozen", "kl_beta": 0.0,
                  "use_lora": 0, "fresh_each_round": False,
                  "icl_k": 0, "icl_days": 0})
    return c


def build_fam(parent, model, arm, es, nrounds=30, prefix="pofdfam",
              tag=None, cfg_mut=None):
    if tag is None:
        tag = (f"{prefix}_{model}_{arm}_ea1_w0p5_l0p2"
               f"_es{_num(es)}_s0")
    cfg = fam_cfg(tag, model, arm, es, nrounds)
    if cfg_mut:
        cfg_mut(cfg)
    g = torch.Generator().manual_seed(9000)
    g_peer = torch.Generator().manual_seed(424244)
    g_peer_cf = torch.Generator().manual_seed(424244)
    lam, w, eps_ai = 0.2, 0.5, 1.0
    x, tw = INNATE.clone(), INNATE.clone()
    rows, op_raw, pred_raw, gate_raw, twin_raw = [], [], [], [], []
    for t in range(nrounds):
        pred = torch.rand(N, generator=g)
        served = pred.clamp(0.0, 1.0)
        gate_t = (served - x).abs() < eps_ai
        h = lam * INNATE + (1.0 - lam) * x
        x = torch.where(gate_t, (1.0 - w) * h + w * served, h)
        acc = gp.ab_sweep(x, ADJ, es, 0.0, gen=g_peer)
        tw = lam * INNATE + (1.0 - lam) * tw
        gp.ab_sweep(tw, ADJ, es, 0.0, gen=g_peer_cf)
        row = {"round": t, "deployment": t, "is_deploy": 1,
               "accepted": acc,
               "contact": float(gate_t.float().mean()),
               "twin_mean": float(tw.mean()),
               "twin_std": float(tw.std()),
               "twin_bias": float(tw.mean() - INNATE.mean())}
        if cfg.get("training_style") in ("sft", "sft_kl"):
            row["n_train"] = 723
        else:
            row["perplexity"] = 7.77
        rows.append(row)
        op_raw.append(x.clone())
        pred_raw.append(pred.clone())
        gate_raw.append(gate_t.clone())
        twin_raw.append(tw.clone())
    d = {"trajectory": rows, "config": cfg,
         "op_raw": torch.stack(op_raw), "pred_raw": torch.stack(pred_raw),
         "gate_raw": torch.stack(gate_raw),
         "twin_raw": torch.stack(twin_raw), "innate": INNATE.clone()}
    rd = parent / cfg["run_tag"]
    rd.mkdir(parents=True)
    (rd / "config.json").write_text(json.dumps(cfg))
    torch.save(d, rd / "trajectory.pt")
    return rd


def test_healthy_qwen3_b1_smoke(tmp_path):
    rd = build_fam(tmp_path, "qwen3_8b", "b1", 0.2, nrounds=3,
                   prefix="pofdfamsmk")
    tg.assert_verdict(rd, True)


def test_healthy_mistral_k0(tmp_path):
    rd = build_fam(tmp_path, "mistral7b", "k0", 0.2)
    tg.assert_verdict(rd, True)


def test_healthy_qwen7b_b0p5(tmp_path):
    rd = build_fam(tmp_path, "qwen7b", "b0p5", 0.05)
    tg.assert_verdict(rd, True)


def test_healthy_ministral_b0(tmp_path):
    rd = build_fam(tmp_path, "ministral8b", "b0", 0.05)
    tg.assert_verdict(rd, True)


def test_wrong_base_model_fails(tmp_path):
    def mut(c):
        c["base_model"] = "Qwen/Qwen2.5-7B-Instruct"
    rd = build_fam(tmp_path, "olmo3_7b", "b1", 0.2, cfg_mut=mut)
    tg.assert_verdict(rd, False, "base_model")


def test_qwen3_missing_thinking_directive_fails(tmp_path):
    def mut(c):
        c.pop("chat_thinking", None)
    rd = build_fam(tmp_path, "qwen3_8b", "b1", 0.2, cfg_mut=mut)
    tg.assert_verdict(rd, False, "chat_thinking")


def test_stray_thinking_directive_fails(tmp_path):
    def mut(c):
        c["chat_thinking"] = False
    rd = build_fam(tmp_path, "olmo7b", "b0", 0.2, cfg_mut=mut)
    tg.assert_verdict(rd, False, "only Qwen3 carries")


def test_b0p5_wrong_beta_fails(tmp_path):
    def mut(c):
        c["kl_beta"] = 1.0
    rd = build_fam(tmp_path, "qwen7b", "b0p5", 0.05, cfg_mut=mut)
    tg.assert_verdict(rd, False, "kl_beta")


def test_reverse_kl_fails(tmp_path):
    def mut(c):
        c["kl_direction"] = "reverse"
    rd = build_fam(tmp_path, "mistral7b", "b1", 0.2, cfg_mut=mut)
    tg.assert_verdict(rd, False, "kl_direction")


def test_off_grid_es_fails(tmp_path):
    rd = build_fam(tmp_path, "mistral7b", "b0", 0.4)
    tg.assert_verdict(rd, False, "_es0p05_ or _es0p2_")


def test_non_b1_smoke_fails(tmp_path):
    rd = build_fam(tmp_path, "qwen3_8b", "b0", 0.2, nrounds=3,
                   prefix="pofdfamsmk")
    tg.assert_verdict(rd, False, "smokes are b1-only")
