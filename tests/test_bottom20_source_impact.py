"""Tests for the BOTTOM-20% SOURCE-IMPACT wave (2026-08-18,
mistral_bottom20_source_impact): cohort A = the 145 LOWEST-innate
agents (deterministic innate-then-id ranking, INNATE_CLAMP_MODE=
bottom), pinned bit-exact in population and twin; es=0 (primary
no-peer test); arms b0 (reused: SFT on all 723 labels) vs b0xa (NEW:
A excluded, volume-matched 723 rows via the run-seeded 145-duplicate
procedure) vs d8 (NEW: frozen weights, personal-history ICL) x
ea {0.1,0.2,0.4,1} x seeds {0,42,43} = 36 conceptual cells.

Generator: exactly 24 new rows from the audited manifest (12 b0xa +
12 d8; the 12 completed b0 bottom cells reuse, hard-asserted -- never
forced), icldays/sftexcl riding queue cols 25/26, zero collisions, NO
smoke key.

Checker: bottom cohort now legal for the b0xa and d8 arms (strat
still rejected); healthy bottom-b0xa (full provenance) and bottom-d8
(personal-history log) fixtures PASS; a fixed id inside a training
batch, missing provenance, a perturbed day value and a strat-cohort
b0xa tag all FAIL; the completed no-peer b0/dyn bottom fixtures still
pass unchanged.

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


def _num(v):
    return f"{v:g}".replace(".", "p")


def b20_tag(arm, gate, seed):
    return (f"pofdclamp_mistral7b_{arm}_bottom_ea{_num(gate)}"
            f"_w0p5_l0p2_es0_s{seed}")


# -- generator + manifest ------------------------------------------------

def test_generator_b20_is_exactly_24_new():
    rows = GEN.b20_rows()
    assert len(rows) == 24
    tags = [r.split(",")[0] for r in rows]
    assert len(set(tags)) == 24
    assert all("_bottom_" in t and "_es0_" in t and "_stub_" not in t
               and t.startswith("pofdclamp_mistral7b_") for t in tags)
    assert sum(1 for t in tags if "_b0xa_" in t) == 12
    assert sum(1 for t in tags if "_d8_" in t) == 12
    assert not any("_b0_" in t or "_dyn_" in t for t in tags)
    for g in (0.1, 0.2, 0.4, 1.0):
        assert sum(1 for t in tags if f"_ea{_num(g)}_" in t) == 6, g
    for s in (0, 42, 43):
        assert sum(1 for t in tags if t.endswith(f"_s{s}")) == 8, s
    # queue tails: (icldays, sftexcl) = (8, 0) on d8, (0, 1) on b0xa
    for r in rows:
        want = ", 8, 0" if "_d8_" in r.split(",")[0] else ", 0, 1"
        assert r.rstrip().endswith(want), r
    # arm queue surfaces
    for r in rows:
        cols = [c.strip() for c in r.split(",")]
        if "_b0xa_" in cols[0]:
            assert cols[1] == "sft" and cols[18] == "1", r
        else:
            assert cols[1] == "frozen" and cols[18] == "0", r
        assert cols[9] == "0" and cols[16] == "0", r   # es0, ICL_K=0
    # b0/d8/b0xa bottom tags collide with NOTHING already queued
    for other in (GEN.clamp_rows(), GEN.clamp_graph_rows(),
                  GEN.clamp_graph_d8_rows(), GEN.clamp_xa_rows()):
        assert not (set(tags) & {r.split(",")[0] for r in other})
    # no smoke key exists for this wave by design
    assert not hasattr(GEN, "B20_SMOKE_KEY")


def test_manifest_expected_split_and_tags():
    assert MANIFEST["n_cells"] == 36
    assert MANIFEST["n_reused"] == 12 and MANIFEST["n_new"] == 24
    reused = [c for c in MANIFEST["cells"] if c["status"] == "reused"]
    assert all(c["arm"] == "b0" and c["verdict"] == "PASS"
               and "_b0_bottom_" in c["run_tag"] for c in reused)
    assert {(c["gate"], c["seed"]) for c in reused} == \
        {(g, s) for g in (0.1, 0.2, 0.4, 1.0) for s in (0, 42, 43)}
    for c in MANIFEST["cells"]:
        assert c["new_tag"] == b20_tag(c["arm"], c["gate"], c["seed"])


def test_b20_sub_template_surface():
    sub = GEN.b20_sub()
    assert "ICL_DAYS=$(icldays)" in sub
    assert "SFT_EXCLUDE_CLAMPED=$(sftexcl)" in sub
    assert "INNATE_CLAMP_PEER_MODE" not in sub   # no-peer wave
    assert sub.rstrip().endswith(
        "nrounds, cmode, icldays, sftexcl from experiments/condor/"
        "configs_pofd_mistral_bottom20_source_impact.txt")


# -- audit / analyzer surface --------------------------------------------

def test_audit_script_surface():
    src = open(os.path.join(PIPE, "audit_bottom20_reuse.py")).read()
    assert '"innate_clamp_mode"] = ("bottom",)' in src
    assert '"innate_clamp_peer_mode"] = (ABSENT,)' in src
    assert "HARD FAIL" in src and "sys.exit(1)" in src
    # audit and generator agree on the cell tags
    AB = _load("audit_b20", os.path.join(PIPE,
                                         "audit_bottom20_reuse.py"))
    assert AB.cell_tag("b0xa", 0.4, 42) == b20_tag("b0xa", 0.4, 42)


def test_analyzer_surface():
    src = open(os.path.join(
        PIPE, "analyze_bottom20_source_impact.py")).read()
    assert "bottom20_impact_analysis" in src
    assert "HARD FAIL" in src and "sys.exit(1)" in src
    for out in ("bottom20_per_cell.csv", "bottom20_summary.csv",
                "bottom20_contrast.csv", "bottom20_impact_panels"):
        assert out in src, out
    assert "t_move" in src and "4.302652729911275" in src


# -- fixtures ------------------------------------------------------------
# REAL-population bottom builder: the ncfix no-peer fixtures run a
# SYNTHETIC 60-agent world (with n_train hard-coded to 723), which the
# b0xa provenance replay would rightly reject -- the volume-matched
# batch must be internally consistent (n_cl rows). So bottom-wave
# fixtures use the artifact's real 723 innate (tg.INNATE) with the
# deterministic bottom-145 mask, no peer step, lam-anchored twin.

def build_bottom(parent, arm, gate, seed=0, nrounds=30, tag=None,
                 cfg_mut=None, post=None):
    import json as _json
    tag = tag or b20_tag(arm, gate, seed)
    cfg = ncfix.cfg_for(tag, "b0" if arm == "b0xa" else arm,
                        "bottom", gate, seed, nrounds)
    if cfg_mut:
        cfg_mut(cfg)
    N = tg.N
    innate = tg.INNATE
    mask = tg.gp.innate_clamp_mask(innate, "bottom", 0.2, seed)
    g = torch.Generator().manual_seed(5000 + seed)
    x, tw = innate.clone(), innate.clone()
    lam, w = 0.2, 0.5
    rows, op_raw, pred_raw, gate_raw, twin_raw = [], [], [], [], []
    for t in range(nrounds):
        pred = torch.rand(N, generator=g)
        served = pred.clamp(0.0, 1.0)
        gate_t = (served - x).abs() < gate
        h = lam * innate + (1.0 - lam) * x
        x = torch.where(gate_t, (1.0 - w) * h + w * served, h)
        tw = lam * innate + (1.0 - lam) * tw
        x[mask] = innate[mask]
        tw[mask] = innate[mask]
        row = {"round": t, "deployment": t, "is_deploy": 1,
               "accepted": 0,
               "contact": float(gate_t.float().mean()),
               "twin_mean": float(tw.mean()),
               "twin_std": float(tw.std()),
               "twin_bias": float(tw.mean() - innate.mean())}
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
         "innate_clamp_hash": tg.gp.innate_clamp_hash(mask)}
    if post:
        post(d)
    rd = parent / cfg["run_tag"]
    rd.mkdir(parents=True)
    (rd / "config.json").write_text(_json.dumps(cfg))
    torch.save(d, rd / "trajectory.pt")
    return rd


def build_b0xa_bottom(parent, gate=0.4, seed=0, **kw):
    return build_bottom(parent, "b0xa", gate, seed=seed,
                        cfg_mut=txa.xa_cfg, post=txa.xa_post, **kw)


def build_d8_bottom(parent, gate=0.4, seed=0, **kw):
    rd = build_bottom(parent, "d8", gate, seed=seed,
                      cfg_mut=td8.d8_cfg, **kw)
    td8.write_days_log(rd)
    return rd


# -- checker: healthy ----------------------------------------------------

def test_healthy_bottom_b0xa(tmp_path):
    tg.assert_verdict(build_b0xa_bottom(tmp_path, 0.4, 42), True)


def test_healthy_bottom_d8(tmp_path):
    tg.assert_verdict(build_d8_bottom(tmp_path, 1.0, 43), True)


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


def test_b0xa_strat_cohort_still_rejected(tmp_path):
    rd = build_b0xa_bottom(
        tmp_path, tag="pofdclamp_mistral7b_b0xa_strat_ea0p4"
                      "_w0p5_l0p2_es0_s0")
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
