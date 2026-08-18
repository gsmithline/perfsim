"""Tests for the BOTTOM-20% SOURCE-IMPACT wave (2026-08-18 FULL-GRID
revision, mistral_bottom20_source_impact): cohort A = the 145
LOWEST-innate agents (deterministic innate-then-id ranking,
INNATE_CLAMP_MODE=bottom), pinned bit-exact in population and twin;
B = 578; full grid ea {0.1,0.2,0.4,1} x es {0,0.05,0.1,0.2,0.4,1},
SEED 0 ONLY; arms b0 (all-723-label SFT) / b0xa (A excluded,
volume-matched 723 rows) / d8 (frozen personal-history ICL) = 72
conceptual cells. Every NEW cell carries the _stub_ token and runs
the one-sided stubborn operator (inert at es=0, the graph-wave
precedent); only the 4 reused b0 no-peer cells stay tokenless.

Generator: exactly 68 new rows from the audited manifest (b0 es>0 20
+ b0xa 24 + d8 24; the 4 completed seed-0 b0 bottom no-peer cells
reuse, hard-asserted -- never forced), icldays/sftexcl on queue cols
25/26, INNATE_CLAMP_PEER_MODE=stubborn pinned in the env, zero
collisions, NO smoke key.

Checker: bottom + _stub_ now legal at es>=0 for every clamp arm
(strat still requires a nonzero dose); healthy bottom-stub fixtures
(REAL 723 population, artifact adjacency, the real stubborn
operator) PASS across arms including the inert es=0 baseline; a
fixed id inside a training batch, missing provenance, a perturbed
personal-history value, a touch mark on a fixed agent, zeroed F-pair
telemetry and a strat-cohort b0xa tag all FAIL; the completed
tokenless b0/dyn bottom fixtures still pass unchanged.

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


tg = _load("b20_graph_fixtures",
           os.path.join(TESTS, "test_innate_clamp_graph.py"))
txa = _load("b20_xa_fixtures",
            os.path.join(TESTS, "test_clamp_exclude_a.py"))
td8 = _load("b20_d8_fixtures",
            os.path.join(TESTS, "test_innate_clamp_d8.py"))

GEN = tg.GEN
ncfix = tg.ncfix
MANIFEST = json.load(open(os.path.join(
    REPO, "experiments", "condor",
    "manifest_bottom20_source_impact.json")))
GATES = [0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]


def _num(v):
    return f"{v:g}".replace(".", "p")


def b20_tag(arm, gate, es, seed=0):
    stub = "" if (arm == "b0" and es == 0.0) else "_stub"
    return (f"pofdclamp_mistral7b_{arm}_bottom{stub}_ea{_num(gate)}"
            f"_w0p5_l0p2_es{_num(es)}_s{seed}")


# -- generator + manifest ------------------------------------------------

def test_generator_b20_is_exactly_68_new():
    rows = GEN.b20_rows()
    assert len(rows) == 68
    tags = [r.split(",")[0] for r in rows]
    assert len(set(tags)) == 68
    assert all("_bottom_stub_" in t and t.endswith("_s0")
               and t.startswith("pofdclamp_mistral7b_") for t in tags)
    assert sum(1 for t in tags if "_b0_" in t) == 20
    assert sum(1 for t in tags if "_b0xa_" in t) == 24
    assert sum(1 for t in tags if "_d8_" in t) == 24
    assert not any("_dyn_" in t for t in tags)
    for g in GATES:
        assert sum(1 for t in tags if f"_ea{_num(g)}_" in t) == 17, g
    for e in ESS:
        want = 8 if e == 0.0 else 12
        assert sum(1 for t in tags
                   if f"_es{_num(e)}_" in t) == want, e
    # the four reused tokenless b0 es0 tags never queue here
    assert not any(b20_tag("b0", g, 0.0) in tags for g in GATES)
    # queue tails: (icldays, sftexcl)
    for r in rows:
        t = r.split(",")[0]
        want = (", 8, 0" if "_d8_" in t
                else ", 0, 1" if "_b0xa_" in t else ", 0, 0")
        assert r.rstrip().endswith(want), r
    # arm queue surfaces incl. the es column (col 9)
    for r in rows:
        cols = [c.strip() for c in r.split(",")]
        es_tok = cols[0].rsplit("_es", 1)[1].rsplit("_s", 1)[0]
        assert cols[9] == es_tok.replace("p", "."), r
        if "_b0xa_" in cols[0] or "_b0_" in cols[0]:
            assert cols[1] == "sft", r
        else:
            assert cols[1] == "frozen", r
        assert cols[16] == "0", r          # ICL_K=0 everywhere
    # zero collisions with every other clamp key
    for other in (GEN.clamp_rows(), GEN.clamp_graph_rows(),
                  GEN.clamp_graph_d8_rows(), GEN.clamp_xa_rows()):
        assert not (set(tags) & {r.split(",")[0] for r in other})
    assert not hasattr(GEN, "B20_SMOKE_KEY")


def test_manifest_expected_split_and_tags():
    assert MANIFEST["n_cells"] == 72
    assert MANIFEST["n_reused"] == 4 and MANIFEST["n_new"] == 68
    assert MANIFEST["seeds"] == [0]
    reused = [c for c in MANIFEST["cells"] if c["status"] == "reused"]
    assert all(c["arm"] == "b0" and c["es"] == 0.0
               and c["verdict"] == "PASS"
               and "_b0_bottom_ea" in c["run_tag"] for c in reused)
    assert {c["gate"] for c in reused} == set(GATES)
    for c in MANIFEST["cells"]:
        assert c["new_tag"] == b20_tag(c["arm"], c["gate"], c["es"])


def test_b20_sub_template_surface():
    sub = GEN.b20_sub()
    assert "ICL_DAYS=$(icldays)" in sub
    assert "SFT_EXCLUDE_CLAMPED=$(sftexcl)" in sub
    assert "INNATE_CLAMP_PEER_MODE=stubborn" in sub
    assert sub.rstrip().endswith(
        "nrounds, cmode, icldays, sftexcl from experiments/condor/"
        "configs_pofd_mistral_bottom20_source_impact.txt")


# -- audit / analyzer surface --------------------------------------------

def test_audit_script_surface():
    src = open(os.path.join(PIPE, "audit_bottom20_reuse.py")).read()
    assert '"innate_clamp_mode"] = ("bottom",)' in src
    assert "HARD FAIL" in src and "sys.exit(1)" in src
    AB = _load("audit_b20", os.path.join(PIPE,
                                         "audit_bottom20_reuse.py"))
    assert AB.cell_tag("b0xa", 0.4, 0.2) == b20_tag("b0xa", 0.4, 0.2)
    assert AB.cell_tag("b0", 0.4, 0.0) == b20_tag("b0", 0.4, 0.0)
    assert "_stub_" not in AB.cell_tag("b0", 0.4, 0.0)


def test_analyzer_surface():
    src = open(os.path.join(
        PIPE, "analyze_bottom20_source_impact.py")).read()
    assert "bottom20_impact_analysis" in src
    assert "HARD FAIL" in src and "sys.exit(1)" in src
    for out in ("bottom20_per_cell.csv", "bottom20_contrast.csv",
                "bottom20_grid_heatmaps"):
        assert out in src, out
    assert "split_heatmap" in src and "sd_ratio_late" in src
    assert "t_move" in src and "t_b0_minus_b0xa" in src


# -- fixtures ------------------------------------------------------------
# REAL-population bottom-stub builder modeled on the graph fixture:
# artifact 723 innate + adjacency, the real one-sided stubborn
# operator on population AND twin, bottom-145 mask, peer telemetry.

def build_bottom(parent, arm, gate, es, seed=0, nrounds=30, tag=None,
                 cfg_mut=None, post=None):
    import json as _json
    tag = tag or b20_tag(arm, gate, es, seed)
    cfg = ncfix.cfg_for(tag, "b0" if arm in ("b0", "b0xa") else arm,
                        "bottom", gate, seed, nrounds)
    cfg["eps"] = es
    cfg["innate_clamp_peer_mode"] = "stubborn"
    if cfg_mut:
        cfg_mut(cfg)
    N = tg.N
    innate = tg.INNATE
    mask = tg.gp.innate_clamp_mask(innate, "bottom", 0.2, seed)
    resp = ~mask
    g = torch.Generator().manual_seed(6000 + seed)
    g_peer = torch.Generator().manual_seed(seed + 424243)
    g_peer_cf = torch.Generator().manual_seed(seed + 424243)
    x, tw = innate.clone(), innate.clone()
    lam, w = 0.2, 0.5
    rows, op_raw, pred_raw, gate_raw, twin_raw, touch_raw = \
        [], [], [], [], [], []
    touch_cum = torch.zeros(N, dtype=torch.bool)
    for t in range(nrounds):
        pred = torch.rand(N, generator=g)
        served = pred.clamp(0.0, 1.0)
        gate_t = (served - x).abs() < gate
        h = lam * innate + (1.0 - lam) * x
        x = torch.where(gate_t, (1.0 - w) * h + w * served, h)
        x[mask] = innate[mask]
        acc, st = tg.gp.ab_sweep_stubborn(x, tg.ADJ, es, 0.0,
                                          g_peer, mask)
        x[mask] = innate[mask]
        tw = lam * innate + (1.0 - lam) * tw
        tw[mask] = innate[mask]
        tg.gp.ab_sweep_stubborn(tw, tg.ADJ, es, 0.0, g_peer_cf, mask)
        tw[mask] = innate[mask]
        touch_cum |= st["touched"]
        touch_raw.append(st["touched"].clone())
        s_twr = float(tw[resp].std())
        row = {"round": t, "deployment": t, "is_deploy": 1,
               "accepted": acc,
               "contact": float(gate_t.float().mean()),
               "twin_mean": float(tw.mean()),
               "twin_std": float(tw.std()),
               "twin_bias": float(tw.mean() - innate.mean()),
               "clamp_fr_sampled": st["fr_sampled"],
               "clamp_fr_accepted": st["fr_accepted"],
               "clamp_fr_reach": float(touch_cum[resp].float().mean()),
               "clamp_resp_std_ratio": (float(x[resp].std()) / s_twr
                                        if s_twr > 0 else None),
               "clamp_resp_disp": float((x[resp] - tw[resp])
                                        .abs().mean()),
               "clamp_gap_mean": float(x[resp].mean()
                                       - x[mask].mean()),
               "clamp_gap_w1": tg.gp.quantile_w1(x[resp], x[mask])}
        g0m = float(innate[resp].mean() - innate[mask].mean())
        row["clamp_gap_closure"] = (1.0 - row["clamp_gap_mean"] / g0m
                                    if abs(g0m) > 1e-9 else None)
        if arm in ("b0", "b0xa"):
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
         "twin_raw": torch.stack(twin_raw), "innate": innate.clone(),
         "innate_clamp_mask": mask.clone(),
         "innate_clamp_count": int(mask.sum()),
         "innate_clamp_mode": "bottom",
         "innate_clamp_frac": 0.2,
         "innate_clamp_seed": seed,
         "innate_clamp_hash": tg.gp.innate_clamp_hash(mask),
         "innate_clamp_peer_mode": "stubborn",
         "clamp_fr_touch_raw": torch.stack(touch_raw)}
    if post:
        post(d)
    rd = parent / cfg["run_tag"]
    rd.mkdir(parents=True)
    (rd / "config.json").write_text(_json.dumps(cfg))
    torch.save(d, rd / "trajectory.pt")
    return rd


def build_b0xa_bottom(parent, gate=0.4, es=0.2, seed=0, **kw):
    return build_bottom(parent, "b0xa", gate, es, seed=seed,
                        cfg_mut=txa.xa_cfg, post=txa.xa_post, **kw)


def build_d8_bottom(parent, gate=0.4, es=0.4, seed=0, **kw):
    rd = build_bottom(parent, "d8", gate, es, seed=seed,
                      cfg_mut=td8.d8_cfg, **kw)
    td8.write_days_log(rd)
    return rd


# -- checker: healthy ----------------------------------------------------

def test_healthy_bottom_b0_peer(tmp_path):
    tg.assert_verdict(build_bottom(tmp_path, "b0", 0.4, 0.05), True)


def test_healthy_bottom_b0xa_peer(tmp_path):
    tg.assert_verdict(build_b0xa_bottom(tmp_path, 0.4, 0.2), True)


def test_healthy_bottom_b0xa_es0_inert_baseline(tmp_path):
    tg.assert_verdict(build_b0xa_bottom(tmp_path, 1.0, 0.0), True)


def test_healthy_bottom_d8_peer(tmp_path):
    tg.assert_verdict(build_d8_bottom(tmp_path, 0.2, 0.4), True)


# -- checker: sabotage ---------------------------------------------------

def test_bottom_b0xa_fixed_id_in_training_fails(tmp_path):
    rd = build_b0xa_bottom(tmp_path)

    def fn(t):
        fro = t["innate_clamp_mask"].nonzero().flatten()
        t["sft_idx_raw"][2][0] = fro[0]
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "cohort A must never enter SFT")


def test_bottom_b0xa_missing_provenance_fails(tmp_path):
    rd = build_b0xa_bottom(tmp_path)

    def fn(t):
        del t["sft_idx_raw"], t["sft_y_raw"]
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "provenance is mandatory")


def test_bottom_d8_perturbed_day_value_fails(tmp_path):
    rd = build_d8_bottom(tmp_path)

    def mutate(rows):
        s = rows[3]["ctx"][7]
        head, vals = s.rsplit(": ", 1)
        v = vals.rstrip(".").split(", ")
        v[0] = "0.97" if v[0] != "0.97" else "0.03"
        rows[3]["ctx"][7] = head + ": " + ", ".join(v) + "."
    td8.write_days_log(rd, mutate=mutate)
    tg.assert_verdict(rd, False, "off the (innate, op_raw) replay")


def test_bottom_touch_on_fixed_agent_fails(tmp_path):
    rd = build_bottom(tmp_path, "b0", 0.4, 0.2)

    def fn(t):
        fro = t["innate_clamp_mask"].nonzero().flatten()
        t["clamp_fr_touch_raw"][5][fro[0]] = True
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "touch marks a FIXED agent")


def test_bottom_zeroed_fr_telemetry_fails(tmp_path):
    rd = build_bottom(tmp_path, "b0", 0.4, 0.2)

    def fn(t):
        for r in t["trajectory"]:
            r["clamp_fr_sampled"] = 0
            r["clamp_fr_accepted"] = 0
            r["clamp_fr_reach"] = 0.0
        t["clamp_fr_touch_raw"][:] = False
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "no isolated condition exists")


def test_b0xa_strat_cohort_still_rejected(tmp_path):
    rd = build_b0xa_bottom(
        tmp_path, tag="pofdclamp_mistral7b_b0xa_strat_stub_ea0p4"
                      "_w0p5_l0p2_es0p2_s0")
    tg.assert_verdict(rd, False, "only in the graph-placement wave")


# -- legacy non-regression -----------------------------------------------

def test_completed_bottom_b0_fixture_still_passes(tmp_path):
    rc, out = tg.run_check(ncfix.build(tmp_path, "b0", "bottom", 0.4,
                                       seed=42))
    assert rc == 0, out[-2500:]


def test_completed_bottom_dyn_fixture_still_passes(tmp_path):
    rc, out = tg.run_check(ncfix.build(tmp_path, "dyn", "bottom", 0.2,
                                       seed=0))
    assert rc == 0, out[-2500:]
