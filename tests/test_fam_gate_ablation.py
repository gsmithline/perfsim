"""Tests for the SECTION-3 FAMILY-GATE ABLATION (2026-08-18,
fam_gate_ablation): the six family-prior checkpoints x SFT arms
{b0, b1} x ea {0.1, 0.2, 0.4, 1} at the fixed es=0.05 canonical
Action surface, seed 0, on the EXACT completed fam-scout code path
(same pofdfam_ family and sub surface -- eps_ai already rides the
queue).

Generator: exactly 36 new rows from the audited manifest (the 12
completed ea=1 scout cells reuse; hard-asserted, never forced),
qwen3 thinking OFF on its rows, zero collisions, NO smoke.

Checker: the fam branch now accepts ea in {0.1, 0.2, 0.4, 1} with
eps_ai pinned from the token; the ablation gates (ea != 1) are
b0/b1-only, es0p05-only and seed-0-only; healthy ablation fixtures
PASS, the restricted combinations FAIL, and the completed ea1 scout
fixtures still pass unchanged.

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac).
"""
import importlib.util
import json
import os

import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")
TESTS = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tfam = _load("famg_fam_fixtures",
             os.path.join(TESTS, "test_fig2_family_prior.py"))
tg = tfam.tg
GEN = tfam.GEN
MANIFEST = json.load(open(os.path.join(
    REPO, "experiments", "condor",
    "manifest_fam_gate_ablation.json")))
MODELS = ["qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b", "mistral7b",
          "ministral8b"]
GATES = [0.1, 0.2, 0.4, 1.0]


def _num(v):
    return f"{v:g}".replace(".", "p")


def famg_tag(model, arm, gate, es=0.05, seed=0):
    return (f"pofdfam_{model}_{arm}_ea{_num(gate)}_w0p5_l0p2"
            f"_es{_num(es)}_s{seed}")


# -- generator + manifest ------------------------------------------------

def test_generator_famg_is_exactly_36():
    rows = GEN.famg_rows()
    assert len(rows) == 36
    tags = [r.split(",")[0] for r in rows]
    assert len(set(tags)) == 36
    assert all(t.startswith("pofdfam_") and "_es0p05_" in t
               and t.endswith("_s0") for t in tags)
    assert not any("_ea1_" in t for t in tags)
    assert sum(1 for t in tags if "_b0_" in t) == 18
    assert sum(1 for t in tags if "_b1_" in t) == 18
    for g in (0.1, 0.2, 0.4):
        assert sum(1 for t in tags if f"_ea{_num(g)}_" in t) == 12, g
    for m in MODELS:
        assert sum(1 for t in tags if f"_{m}_" in t) == 6, m
    # eps_ai rides col 14 and matches the ea token; chatthink col 24
    for r in rows:
        cols = [c.strip() for c in r.split(",")]
        ea_tok = cols[0].split("_ea")[1].split("_")[0]
        assert cols[14] == ea_tok.replace("p", "."), r
        assert cols[9] == "0.05", r
        assert cols[24] == ("0" if "_qwen3_8b_" in cols[0]
                            else "default"), r
    # no collision with the scout / beta / confirm keys
    for other in (GEN.fam_rows(), GEN.fam_beta_rows(),
                  GEN.fam_confirm_rows("b8")):
        assert not (set(tags) & {r.split(",")[0] for r in other})
    assert not hasattr(GEN, "FAMG_SMOKE_KEY")


def test_manifest_expected_split_and_tags():
    assert MANIFEST["n_cells"] == 48
    assert MANIFEST["n_reused"] == 12 and MANIFEST["n_new"] == 36
    reused = [c for c in MANIFEST["cells"] if c["status"] == "reused"]
    assert all(c["gate"] == 1.0 and c["verdict"] == "PASS"
               and "_ea1_" in c["run_tag"] for c in reused)
    assert {(c["model"], c["arm"]) for c in reused} == \
        {(m, a) for m in MODELS for a in ("b0", "b1")}
    for c in MANIFEST["cells"]:
        assert c["new_tag"] == famg_tag(c["model"], c["arm"],
                                        c["gate"])


def test_famg_sub_surface():
    sub = GEN.famg_sub()
    assert "SAVE_RAW_GEN=1" in sub
    assert "CHAT_THINKING=$(chatthink)" in sub
    assert sub.rstrip().endswith(
        "pplbatch from experiments/condor/"
        "configs_pofd_fam_gate_ablation.txt")


# -- audit / analyzer surface --------------------------------------------

def test_audit_surface():
    AB = _load("audit_famg", os.path.join(
        PIPE, "audit_fam_gate_reuse.py"))
    assert AB.cell_tag("mistral7b", "b0", 0.2) == \
        famg_tag("mistral7b", "b0", 0.2)
    w = AB.cell_want("mistral7b", "b0", 0.2)
    assert w["eps_ai"] == (0.2,)
    # the pofdevo_ wave shares the mistral b0 surface at these gates;
    # save_raw_gen is the matched discriminator
    assert w["save_raw_gen"] == (True,)


def test_analyzer_surface():
    src = open(os.path.join(
        PIPE, "analyze_fam_gate_ablation.py")).read()
    assert "fam_gate_analysis" in src
    assert "HARD FAIL" in src and "sys.exit(1)" in src
    for out in ("fam_gate_pairs.csv", "fam_gate_pairs.tex",
                "fam_gate_panels", "fam_gate_grid"):
        assert out in src, out
    assert "itertools.combinations" in src
    assert "median" in src and r"\toprule" in src


# -- fixtures ------------------------------------------------------------

def build_famg(parent, model, arm, gate, es=0.05, nrounds=30,
               tag=None, cfg_mut=None):
    tag = tag or famg_tag(model, arm, gate, es)
    cfg = tfam.fam_cfg(tag, model, arm, es, nrounds)
    cfg["eps_ai"] = gate
    if cfg_mut:
        cfg_mut(cfg)
    N = tfam.N
    INNATE = tfam.INNATE
    g = torch.Generator().manual_seed(9100)
    g_peer = torch.Generator().manual_seed(424244)
    g_peer_cf = torch.Generator().manual_seed(424244)
    lam, w = 0.2, 0.5
    x, tw = INNATE.clone(), INNATE.clone()
    rows, op_raw, pred_raw, gate_raw, twin_raw = [], [], [], [], []
    for t in range(nrounds):
        pred = torch.rand(N, generator=g)
        served = pred.clamp(0.0, 1.0)
        gate_t = (served - x).abs() < cfg["eps_ai"]
        h = lam * INNATE + (1.0 - lam) * x
        x = torch.where(gate_t, (1.0 - w) * h + w * served, h)
        acc = tfam.gp.ab_sweep(x, tfam.ADJ, es, 0.0, gen=g_peer)
        tw = lam * INNATE + (1.0 - lam) * tw
        tfam.gp.ab_sweep(tw, tfam.ADJ, es, 0.0, gen=g_peer_cf)
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
         "op_raw": torch.stack(op_raw),
         "pred_raw": torch.stack(pred_raw),
         "gate_raw": torch.stack(gate_raw),
         "twin_raw": torch.stack(twin_raw),
         "innate": INNATE.clone()}
    rd = parent / cfg["run_tag"]
    rd.mkdir(parents=True)
    (rd / "config.json").write_text(json.dumps(cfg))
    torch.save(d, rd / "trajectory.pt")
    return rd


# -- checker: healthy ----------------------------------------------------

def test_healthy_mistral_b0_ea0p2(tmp_path):
    tg.assert_verdict(build_famg(tmp_path, "mistral7b", "b0", 0.2),
                      True)


def test_healthy_qwen3_b1_ea0p4(tmp_path):
    tg.assert_verdict(build_famg(tmp_path, "qwen3_8b", "b1", 0.4),
                      True)


# -- checker: sabotage ---------------------------------------------------

def test_b0p5_at_ablation_gate_fails(tmp_path):
    rd = build_famg(tmp_path, "olmo7b", "b0p5", 0.2)
    tg.assert_verdict(rd, False, "b0/b1 arms")


def test_k0_at_ablation_gate_fails(tmp_path):
    rd = build_famg(tmp_path, "ministral8b", "k0", 0.1)
    tg.assert_verdict(rd, False, "b0/b1 arms")


def test_ablation_gate_seed42_fails(tmp_path):
    def mut(c):
        c["seed"] = 42
    rd = build_famg(tmp_path, "mistral7b", "b0", 0.2,
                    tag=famg_tag("mistral7b", "b0", 0.2, seed=42),
                    cfg_mut=mut)
    tg.assert_verdict(rd, False, "seed-0 only")


def test_ablation_gate_es0p1_fails(tmp_path):
    # 0.1 is not in the allowed social-gate set {0, 0.05, 0.2}
    rd = build_famg(tmp_path, "mistral7b", "b0", 0.2, es=0.1)
    tg.assert_verdict(rd, False, "_es0p05_ or _es0p2_")


# -- social-gate extension (2026-08-18) ----------------------------------

def test_generator_famgs_is_exactly_84():
    rows = GEN.famgs_rows()
    assert len(rows) == 84
    tags = [r.split(",")[0] for r in rows]
    assert len(set(tags)) == 84
    assert all(t.startswith("pofdfam_") and t.endswith("_s0")
               for t in tags)
    assert not any("_es0p05_" in t for t in tags)
    assert sum(1 for t in tags if "_es0_" in t) == 48
    assert sum(1 for t in tags if "_es0p2_" in t) == 36
    assert not any("_ea1_" in t and "_es0p2_" in t for t in tags)
    assert sum(1 for t in tags if "_b0_" in t) == 42
    assert sum(1 for t in tags if "_b1_" in t) == 42
    for m in MODELS:
        assert sum(1 for t in tags if f"_{m}_" in t) == 14, m
    # no collision with the ablation / scout keys
    for other in (GEN.famg_rows(), GEN.fam_rows()):
        assert not (set(tags) & {r.split(",")[0] for r in other})


def test_social_manifest_expected_split():
    m = json.load(open(os.path.join(
        REPO, "experiments", "condor",
        "manifest_fam_gate_social.json")))
    assert m["n_cells"] == 144 and m["n_new"] == 84
    new = [c for c in m["cells"] if c["status"] == "new"]
    assert sum(1 for c in new if c["es"] == 0.0) == 48
    assert sum(1 for c in new if c["es"] == 0.2) == 36
    assert not any(c["status"] == "new" for c in m["cells"]
                   if c["es"] == 0.05)
    cov = [c for c in m["cells"] if c["status"] == "covered_running"]
    assert len(cov) == 36 and all(c["es"] == 0.05 for c in cov)
    re = [c for c in m["cells"] if c["status"] == "reused"]
    assert all(c["verdict"] == "PASS" for c in re)
    inh = [c for c in re if c.get("inherited")]
    assert [(c["model"], c["arm"], c["es"]) for c in inh] == \
        [("mistral7b", "b0", 0.2)]


def test_famgs_sub_surface():
    sub = GEN.famgs_sub()
    assert "SAVE_RAW_GEN=1" in sub
    assert sub.rstrip().endswith(
        "pplbatch from experiments/condor/"
        "configs_pofd_fam_gate_social.txt")


def test_healthy_b0_ea0p2_es0p2(tmp_path):
    tg.assert_verdict(build_famg(tmp_path, "mistral7b", "b0", 0.2,
                                 es=0.2), True)


def test_healthy_b1_ea1_es0(tmp_path):
    tg.assert_verdict(build_famg(tmp_path, "qwen7b", "b1", 1.0,
                                 es=0.0), True)


def test_k0_at_es0_fails(tmp_path):
    rd = build_famg(tmp_path, "ministral8b", "k0", 1.0, es=0.0)
    tg.assert_verdict(rd, False, "b0/b1 arms")


def test_es0_seed42_fails(tmp_path):
    def mut(c):
        c["seed"] = 42
    rd = build_famg(tmp_path, "mistral7b", "b0", 1.0, es=0.0,
                    tag=famg_tag("mistral7b", "b0", 1.0, es=0.0,
                                 seed=42),
                    cfg_mut=mut)
    tg.assert_verdict(rd, False, "seed-0 only")


def test_gate_token_config_mismatch_fails(tmp_path):
    def mut(c):
        c["eps_ai"] = 1.0
    rd = build_famg(tmp_path, "mistral7b", "b0", 0.2, cfg_mut=mut)
    tg.assert_verdict(rd, False, "eps_ai")


# -- legacy non-regression -----------------------------------------------

def test_completed_ea1_scout_fixture_still_passes(tmp_path):
    rd = tfam.build_fam(tmp_path, "mistral7b", "b0", 0.05)
    tg.assert_verdict(rd, True)
