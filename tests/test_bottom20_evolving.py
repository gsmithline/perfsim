"""Tests for the FULLY-EVOLVING comparison wave (2026-08-18,
mistral_bottom20_evolving): the no-clamp b0/d8 companion grid to the
completed bottom-20%-fixed wave. All 723 agents evolve normally
(symmetric peers, matched no-platform twin, serving path matched to
the fixed runs); cohort A exists only in the ANALYSIS.

Generator: exactly 48 seed-0 rows in the NEW pofdevo_ family (24 b0 +
24 d8 x ea {0.1,0.2,0.4,1} x es {0,0.05,0.1,0.2,0.4,1}), icldays on
queue col 24, zero collisions, no clamp env in the sub, NO smoke.

Checker (EVO section), via REAL-population fixtures (723 agents,
artifact adjacency, symmetric gp.ab_sweep on population AND twin):
healthy b0/d8 runs PASS at es=0 and es>0; a perturbed personal-
history value, a stray cross-user context log, a missing days log, a
clamp key in the config, a clamp-mask artifact and a missing twin
all FAIL; the completed clamp fixtures still pass unchanged.

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


tg = _load("evo_graph_fixtures",
           os.path.join(TESTS, "test_innate_clamp_graph.py"))
td8 = _load("evo_d8_fixtures",
            os.path.join(TESTS, "test_innate_clamp_d8.py"))

GEN = tg.GEN
ncfix = tg.ncfix
GATES = [0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]


def _num(v):
    return f"{v:g}".replace(".", "p")


def evo_tag(arm, gate, es, seed=0):
    return (f"pofdevo_mistral7b_{arm}_ea{_num(gate)}"
            f"_w0p5_l0p2_es{_num(es)}_s{seed}")


# -- generator -----------------------------------------------------------

def test_generator_evo_is_exactly_48():
    rows = GEN.evo_rows()
    assert len(rows) == 48
    tags = [r.split(",")[0] for r in rows]
    assert len(set(tags)) == 48
    assert all(t.startswith("pofdevo_mistral7b_")
               and t.endswith("_s0") for t in tags)
    assert sum(1 for t in tags if "_b0_" in t) == 24
    assert sum(1 for t in tags if "_d8_" in t) == 24
    assert not any("bottom" in t or "_stub_" in t for t in tags)
    for g in GATES:
        assert sum(1 for t in tags if f"_ea{_num(g)}_" in t) == 12, g
    for e in ESS:
        assert sum(1 for t in tags if f"_es{_num(e)}_" in t) == 8, e
    for r in rows:
        want = ", 8" if "_d8_" in r.split(",")[0] else ", 0"
        assert r.rstrip().endswith(want), r
        cols = [c.strip() for c in r.split(",")]
        assert cols[16] == "0", r          # ICL_K=0 everywhere
        if "_b0_" in cols[0]:
            assert cols[1] == "sft", r
        else:
            assert cols[1] == "frozen", r
    # brand-new family: collides with nothing anywhere
    for other in (GEN.clamp_rows(), GEN.clamp_graph_rows(),
                  GEN.clamp_graph_d8_rows(), GEN.clamp_xa_rows(),
                  GEN.b20_rows()):
        assert not (set(tags) & {r.split(",")[0] for r in other})
    assert not hasattr(GEN, "EVO_SMOKE_KEY")


def test_evo_sub_template_surface():
    sub = GEN.evo_sub()
    assert "ICL_DAYS=$(icldays)" in sub
    assert "INNATE_CLAMP" not in sub
    assert "SFT_EXCLUDE_CLAMPED" not in sub
    assert "WITH_TWIN=1" in sub
    assert sub.rstrip().endswith(
        "nrounds, icldays from experiments/condor/"
        "configs_pofd_mistral_bottom20_evolving.txt")


def test_analyzer_surface():
    src = open(os.path.join(
        PIPE, "analyze_bottom20_vs_evolving.py")).read()
    assert "bottom20_evolving_analysis" in src
    assert "HARD FAIL" in src and "sys.exit(1)" in src
    for out in ("evolving_per_cell.csv", "evolving_contrast.csv",
                "evolving_heatmaps"):
        assert out in src, out
    assert "did_b_sd" in src and "b_sd0_shared" in src
    # one shared denominator (initial B SD); twin SDs never divide a
    # cross-condition comparison
    assert "sd0 = float(innate[~mask].std())" in src
    assert "twin_std_ratio" not in src


# -- fixtures ------------------------------------------------------------

def evo_cfg(tag, arm, gate, es, nrounds=30):
    c = ncfix.cfg_for(tag, "b0" if arm == "b0" else arm, "strat",
                      gate, 0, nrounds)
    for k in ("innate_clamp_mode", "innate_clamp_frac",
              "innate_clamp_seed"):
        c.pop(k, None)
    c["eps"] = es
    if arm == "d8":
        c["icl_k"] = 0
        c["icl_days"] = 8
    return c


def build_evo(parent, arm, gate, es, nrounds=30, tag=None,
              cfg_mut=None, post=None):
    tag = tag or evo_tag(arm, gate, es)
    cfg = evo_cfg(tag, arm, gate, es, nrounds)
    if cfg_mut:
        cfg_mut(cfg)
    N = tg.N
    innate = tg.INNATE
    g = torch.Generator().manual_seed(7100)
    g_peer = torch.Generator().manual_seed(424244)
    g_peer_cf = torch.Generator().manual_seed(424244)
    lam, w = 0.2, 0.5
    x, tw = innate.clone(), innate.clone()
    rows, op_raw, pred_raw, gate_raw, twin_raw = [], [], [], [], []
    for t in range(nrounds):
        pred = torch.rand(N, generator=g)
        served = pred.clamp(0.0, 1.0)
        gate_t = (served - x).abs() < gate
        h = lam * innate + (1.0 - lam) * x
        x = torch.where(gate_t, (1.0 - w) * h + w * served, h)
        acc = tg.gp.ab_sweep(x, tg.ADJ, es, 0.0, gen=g_peer)
        tw = lam * innate + (1.0 - lam) * tw
        tg.gp.ab_sweep(tw, tg.ADJ, es, 0.0, gen=g_peer_cf)
        row = {"round": t, "deployment": t, "is_deploy": 1,
               "accepted": acc,
               "contact": float(gate_t.float().mean()),
               "twin_mean": float(tw.mean()),
               "twin_std": float(tw.std()),
               "twin_bias": float(tw.mean() - innate.mean())}
        if arm == "b0":
            row["n_train"] = N
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
         "innate": innate.clone()}
    if post:
        post(d)
    rd = parent / cfg["run_tag"]
    rd.mkdir(parents=True)
    (rd / "config.json").write_text(json.dumps(cfg))
    torch.save(d, rd / "trajectory.pt")
    return rd


def build_evo_d8(parent, gate=0.4, es=0.4, **kw):
    rd = build_evo(parent, "d8", gate, es, **kw)
    td8.write_days_log(rd)
    return rd


# -- checker: healthy ----------------------------------------------------

def test_healthy_evo_b0_nopeer(tmp_path):
    tg.assert_verdict(build_evo(tmp_path, "b0", 0.4, 0.0), True)


def test_healthy_evo_b0_peer(tmp_path):
    tg.assert_verdict(build_evo(tmp_path, "b0", 1.0, 0.2), True)


def test_healthy_evo_d8_peer(tmp_path):
    tg.assert_verdict(build_evo_d8(tmp_path, 0.2, 0.4), True)


def test_healthy_evo_d8_nopeer(tmp_path):
    tg.assert_verdict(build_evo_d8(tmp_path, 0.1, 0.0), True)


# -- checker: sabotage ---------------------------------------------------

def test_evo_d8_perturbed_day_value_fails(tmp_path):
    rd = build_evo_d8(tmp_path)

    def mutate(rows):
        s = rows[2]["ctx"][5]
        head, vals = s.rsplit(": ", 1)
        v = vals.rstrip(".").split(", ")
        v[-1] = "0.99" if v[-1] != "0.99" else "0.01"
        rows[2]["ctx"][5] = head + ": " + ", ".join(v) + "."
    td8.write_days_log(rd, mutate=mutate)
    tg.assert_verdict(rd, False, "off the (innate, op_raw) replay")


def test_evo_d8_missing_days_log_fails(tmp_path):
    rd = build_evo_d8(tmp_path)
    os.remove(rd / "icl_days_log.json.gz")
    tg.assert_verdict(rd, False, "icl_days_log.json.gz missing")


def test_evo_d8_stray_ctx_log_fails(tmp_path):
    import gzip
    rd = build_evo_d8(tmp_path)
    with gzip.open(rd / "icl_ctx_log.json.gz", "wt") as fh:
        fh.write(json.dumps({"round": 0, "ctx": ["x"] * tg.N}) + "\n")
    tg.assert_verdict(rd, False, "icl_ctx_log.json.gz present")


def test_evo_clamp_key_in_config_fails(tmp_path):
    rd = build_evo(tmp_path, "b0", 0.4, 0.2)

    def fn(t):
        t["config"]["innate_clamp_mode"] = "bottom"
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "fully-evolving evo tag")


def test_evo_clamp_mask_artifact_fails(tmp_path):
    rd = build_evo(tmp_path, "b0", 0.4, 0.2)

    def fn(t):
        m = torch.zeros(tg.N, dtype=torch.bool)
        m[:145] = True
        t["innate_clamp_mask"] = m
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "must not carry clamp artifacts")


def test_evo_missing_twin_fails(tmp_path):
    rd = build_evo(tmp_path, "b0", 0.4, 0.0)

    def fn(t):
        t["twin_raw"] = torch.empty(0)
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "twin_raw missing")


# -- legacy non-regression -----------------------------------------------

def test_completed_bottom_fixture_still_passes(tmp_path):
    rc, out = tg.run_check(ncfix.build(tmp_path, "b0", "bottom", 0.4,
                                       seed=42))
    assert rc == 0, out[-2500:]
