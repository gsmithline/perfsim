"""Tests for the SECTION-4 THREE-SEED REPLICATION (2026-08-19,
mistral_bottom20_section4_repl): the completed seed-0 fixed-vs-
evolving bottom-20% surface extended to seeds 42/43.

Manifest: 192 audited target cells (2 seeds x 2 conditions x b0/d8 x
4 gates x 6 social doses), 40 reused / 152 new -- the audited split
is asserted for consistency, never forced. Reuse composition: the 8
tokenless fixed-SFT no-peer originals (b0 es0, both seeds) + 32
archived evolving b0 cells (pofdreach / pofdpeer2 / pofdgate2d /
pofdws2f field-level matches).

Generator: 88 _fixed rows (b20 schema, always _stub_) + 64 _evo rows
(pofdevo schema, no clamp anywhere) under one umbrella key; reused
tags never re-queue; zero collisions with every earlier key.

Checker: already seed-generic -- healthy fixed/evolving fixtures at
seeds 42/43 PASS under the exact existing contracts; tag/config seed
mismatches and clamp keys on evolving tags FAIL.

Analyzer: three-seed machinery (Student-t df=2 interval, exclusion
markers, manifest-driven tag resolution, seed-0 tag scheme identical
to the completed per-seed analyzers).

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac).
"""
import importlib.util
import json
import os

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")
TESTS = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tg = _load("b20r_graph_fixtures",
           os.path.join(TESTS, "test_innate_clamp_graph.py"))
tb = _load("b20r_bottom_fixtures",
           os.path.join(TESTS, "test_bottom20_source_impact.py"))
te = _load("b20r_evo_fixtures",
           os.path.join(TESTS, "test_bottom20_evolving.py"))
AS = _load("analyze_b20r",
           os.path.join(PIPE, "analyze_bottom20_section4_3seed.py"))
AE = _load("analyze_b20_evo",
           os.path.join(PIPE, "analyze_bottom20_vs_evolving.py"))

GEN = tg.GEN
GATES = [0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
SEEDS_REPL = [42, 43]
MANIFEST = os.path.join(REPO, "experiments", "condor",
                        "manifest_bottom20_section4_repl.json")


def _num(v):
    return f"{v:g}".replace(".", "p")


def manifest():
    with open(MANIFEST) as fh:
        return json.load(fh)


# -- manifest ------------------------------------------------------------

def test_manifest_covers_the_192_cell_surface():
    mf = manifest()
    assert mf["key"] == "mistral_bottom20_section4_repl"
    assert mf["n_cells"] == 192 and len(mf["cells"]) == 192
    assert {(c["cond"], c["arm"], c["gate"], c["es"], c["seed"])
            for c in mf["cells"]} == \
        {(cd, a, g, e, s) for cd in ("fixed", "evolving")
         for a in ("b0", "d8") for g in GATES for e in ESS
         for s in SEEDS_REPL}


def test_manifest_audited_split_and_reuse_composition():
    mf = manifest()
    reused = [c for c in mf["cells"] if c["status"] == "reused"]
    new = [c for c in mf["cells"] if c["status"] == "new"]
    assert mf["n_reused"] == len(reused) == 40
    assert mf["n_new"] == len(new) == 152
    # only b0 cells reuse, every reuse carries a PASS verdict and a
    # unique archived run dir
    assert all(c["arm"] == "b0" for c in reused)
    assert all(c.get("verdict") == "PASS" for c in reused)
    tags = [c["run_tag"] for c in reused]
    assert len(set(tags)) == 40
    fixed_r = [c for c in reused if c["cond"] == "fixed"]
    evo_r = [c for c in reused if c["cond"] == "evolving"]
    assert len(fixed_r) == 8 and len(evo_r) == 32
    # fixed reuse = exactly the tokenless no-peer originals at es0
    assert all(c["es"] == 0.0 for c in fixed_r)
    assert all("_bottom_" in c["run_tag"]
               and "_stub_" not in c["run_tag"] for c in fixed_r)
    # evolving reuse comes from the four archived no-clamp families
    # at es {0, 0.2, 0.4, 1}; es 0.05/0.1 have no archived cells
    assert all(c["es"] in (0.0, 0.2, 0.4, 1.0) for c in evo_r)
    fams = {c["run_tag"].split("_mistral7b_")[0] for c in evo_r}
    assert fams == {"pofdreach", "pofdpeer2", "pofdgate2d",
                    "pofdws2f"}


def test_manifest_new_tags_are_collision_safe_and_well_formed():
    mf = manifest()
    new = [c for c in mf["cells"] if c["status"] == "new"]
    tags = [c["new_tag"] for c in new]
    assert len(set(tags)) == 152
    for c in new:
        t = c["new_tag"]
        assert t.endswith(f"_s{c['seed']}")
        assert f"_ea{_num(c['gate'])}_" in t
        assert f"_es{_num(c['es'])}_" in t
        if c["cond"] == "fixed":
            assert f"pofdclamp_mistral7b_{c['arm']}_bottom_stub_" \
                in t
        else:
            assert t.startswith(f"pofdevo_mistral7b_{c['arm']}_")
            assert "bottom" not in t and "_stub_" not in t
    reused_tags = {c["run_tag"] for c in mf["cells"]
                   if c["status"] == "reused"}
    assert not (set(tags) & reused_tags)


# -- generator -----------------------------------------------------------

def test_generator_b20r_is_exactly_88_plus_64():
    rows_f, rows_e = GEN.b20r_rows()
    assert len(rows_f) == 88 and len(rows_e) == 64
    tags_f = [r.split(",")[0] for r in rows_f]
    tags_e = [r.split(",")[0] for r in rows_e]
    assert len(set(tags_f)) == 88 and len(set(tags_e)) == 64
    assert all("_bottom_stub_" in t for t in tags_f)
    assert all(t.startswith("pofdevo_mistral7b_") for t in tags_e)
    for sd, nf, ne in ((42, 44, 32), (43, 44, 32)):
        assert sum(1 for t in tags_f if t.endswith(f"_s{sd}")) == nf
        assert sum(1 for t in tags_e if t.endswith(f"_s{sd}")) == ne
    assert sum(1 for t in tags_f if "_b0_" in t) == 40
    assert sum(1 for t in tags_f if "_d8_" in t) == 48
    assert sum(1 for t in tags_e if "_b0_" in t) == 16
    assert sum(1 for t in tags_e if "_d8_" in t) == 48
    assert not any("_b0xa_" in t or "_dyn_" in t
                   for t in tags_f + tags_e)
    # reused cells never re-queue: no fixed b0 at es0, evolving b0
    # only at the archived-missing es 0.05/0.1
    assert not any("_b0_" in t and "_es0_" in t for t in tags_f)
    assert all("_es0p05_" in t or "_es0p1_" in t
               for t in tags_e if "_b0_" in t)


def test_generator_b20r_row_schemas():
    rows_f, rows_e = GEN.b20r_rows()
    for r in rows_f:
        cols = [c.strip() for c in r.split(",")]
        t = cols[0]
        assert cols[16] == "0", r          # ICL_K=0 everywhere
        assert cols[23] == "bottom", r     # cmode rides the queue
        assert cols[3] in ("42", "43"), r
        want = (", 8, 0" if "_d8_" in t else ", 0, 0")
        assert r.rstrip().endswith(want), r
        assert cols[1] == ("frozen" if "_d8_" in t else "sft"), r
    for r in rows_e:
        cols = [c.strip() for c in r.split(",")]
        t = cols[0]
        assert cols[16] == "0", r
        assert cols[3] in ("42", "43"), r
        want = ", 8" if "_d8_" in t else ", 0"
        assert r.rstrip().endswith(want), r
        assert cols[1] == ("frozen" if "_d8_" in t else "sft"), r


def test_generator_b20r_no_collisions_with_earlier_keys():
    rows_f, rows_e = GEN.b20r_rows()
    tags = {r.split(",")[0] for r in rows_f + rows_e}
    for other in (GEN.b20_rows(), GEN.evo_rows(), GEN.clamp_rows(),
                  GEN.clamp_graph_rows(), GEN.clamp_graph_d8_rows(),
                  GEN.clamp_xa_rows()):
        assert not (tags & {r.split(",")[0] for r in other})
    reused_tags = {c["run_tag"] for c in manifest()["cells"]
                   if c["status"] == "reused"}
    assert not (tags & reused_tags)


def test_b20r_sub_templates():
    sub_f = GEN.b20r_fixed_sub()
    assert "INNATE_CLAMP_MODE=$(cmode)" in sub_f
    assert "INNATE_CLAMP_PEER_MODE=stubborn" in sub_f
    assert "SFT_EXCLUDE_CLAMPED=$(sftexcl)" in sub_f
    assert "INNATE_CLAMP_SEED=$(seed)" in sub_f
    assert "WITH_TWIN=1" in sub_f
    assert sub_f.rstrip().endswith(
        "cmode, icldays, sftexcl from experiments/condor/"
        "configs_pofd_mistral_bottom20_section4_repl_fixed.txt")
    sub_e = GEN.b20r_evo_sub()
    assert "INNATE_CLAMP" not in sub_e
    assert "SFT_EXCLUDE_CLAMPED" not in sub_e
    assert "WITH_TWIN=1" in sub_e
    assert sub_e.rstrip().endswith(
        "nrounds, icldays from experiments/condor/"
        "configs_pofd_mistral_bottom20_section4_repl_evo.txt")
    # the umbrella and both sub-keys ride the submit script
    sh = open(os.path.join(REPO, "experiments", "condor",
                           "submit_pofd_sweep.sh")).read()
    assert ('mistral_bottom20_section4_repl) TARGETS='
            '"mistral_bottom20_section4_repl_fixed '
            'mistral_bottom20_section4_repl_evo"') in sh


# -- checker at seeds 42/43 (exact existing contracts) -------------------

def test_healthy_fixed_b0_stub_seed42(tmp_path):
    tg.assert_verdict(
        tb.build_bottom(tmp_path, "b0", 0.4, 0.2, seed=42), True)


def test_healthy_fixed_d8_stub_es0_seed43(tmp_path):
    tg.assert_verdict(
        tb.build_d8_bottom(tmp_path, 0.4, 0.0, seed=43), True)


def test_healthy_evo_b0_seed42(tmp_path):
    tg.assert_verdict(
        te.build_evo(tmp_path, "b0", 0.4, 0.05,
                     tag=te.evo_tag("b0", 0.4, 0.05, seed=42),
                     cfg_mut=lambda c: c.update(seed=42)), True)


def test_healthy_evo_d8_seed43(tmp_path):
    tg.assert_verdict(
        te.build_evo_d8(tmp_path, 0.4, 0.1,
                        tag=te.evo_tag("d8", 0.4, 0.1, seed=43),
                        cfg_mut=lambda c: c.update(seed=43)), True)


def test_fixed_seed_token_mismatch_fails(tmp_path):
    rd = tb.build_bottom(tmp_path, "b0", 0.4, 0.2, seed=42,
                         cfg_mut=lambda c: c.update(seed=0))
    tg.assert_verdict(rd, False, "tag says 42")


def test_fixed_cohort_seed_mismatch_fails(tmp_path):
    rd = tb.build_bottom(
        tmp_path, "b0", 0.4, 0.2, seed=43,
        cfg_mut=lambda c: c.update(innate_clamp_seed=0))
    tg.assert_verdict(rd, False, "innate_clamp_seed")


def test_evo_seed42_with_clamp_key_fails(tmp_path):
    rd = te.build_evo(
        tmp_path, "b0", 0.4, 0.05,
        tag=te.evo_tag("b0", 0.4, 0.05, seed=42),
        cfg_mut=lambda c: c.update(seed=42,
                                   innate_clamp_mode="bottom"))
    tg.assert_verdict(rd, False, "recorded on a fully-evolving")


# -- analyzer ------------------------------------------------------------

def test_tci3_matches_the_df2_student_t():
    m, sd, lo, hi = AS.tci3([1.0, 2.0, 3.0])
    assert abs(m - 2.0) < 1e-12 and abs(sd - 1.0) < 1e-12
    half = 4.302652729911275 / 3 ** 0.5
    assert abs((hi - lo) / 2 - half) < 1e-12
    assert abs((hi + lo) / 2 - 2.0) < 1e-12
    # degenerate spread: the interval collapses onto the mean
    m, sd, lo, hi = AS.tci3([0.5, 0.5, 0.5])
    assert sd == 0.0 and lo == hi == 0.5


def test_excludes_reference_marker():
    assert AS.excludes(0.1, 0.5, 0.0)
    assert AS.excludes(-0.5, -0.1, 0.0)
    assert not AS.excludes(-0.1, 0.5, 0.0)
    assert AS.excludes(1.02, 1.30, 1.0)
    assert not AS.excludes(0.95, 1.05, 1.0)


def test_seed0_tags_match_the_completed_per_seed_analyzers():
    for cond in ("fixed", "evolving"):
        for arm in ("b0", "d8"):
            for g in GATES:
                for e in ESS:
                    assert AS.seed0_tag(cond, arm, g, e) == \
                        AE.cell_tag(cond, arm, g, e)


def test_repl_tags_resolve_every_seed42_43_cell():
    rt = AS.repl_tags(manifest())
    assert len(rt) == 192
    for (cond, arm, gate, es, seed), tag in rt.items():
        assert seed in SEEDS_REPL
        assert f"_es{_num(es)}_" in tag
    # reused cells keep their archived tags, new cells the queue tag
    mf = manifest()
    for c in mf["cells"]:
        k = (c["cond"], c["arm"], c["gate"], c["es"], c["seed"])
        want = (c["run_tag"] if c["status"] == "reused"
                else c["new_tag"])
        assert rt[k] == want


def test_analyzer_surface():
    src = open(os.path.join(
        PIPE, "analyze_bottom20_section4_3seed.py")).read()
    # NEW output directory: the seed-0 analyses are never touched
    assert "bottom20_section4_3seed_analysis" in src
    assert "bottom20_impact_analysis" not in src
    assert "bottom20_evolving_analysis" not in src
    assert "HARD FAIL" in src and "sys.exit(1)" in src
    for out in ("section4_per_seed_cells.csv",
                "section4_source_effect.csv",
                "section4_dispersion.csv", "section4_contrast.csv",
                "section4_null_floor.csv",
                "section4_source_effect.png",
                "section4_sd_ratio.png"):
        assert out.split(".")[0] in src, out
    # equilibrium window, hard 288-cell requirement, structural null
    assert AS.LATE == range(25, 30)
    assert AS.SEEDS == [0, 42, 43]
    assert "structural null" in src.lower()
    assert AS.T_CRIT == 4.302652729911275


def test_structural_null_tolerance_is_hardware_aware():
    # bit-exactness is required only where greedy generation IS
    # reproducible (one GPU architecture); across architectures the
    # residual is generation nondeterminism, not an A->B pathway
    assert AS.NULL_TOL == 1e-9
    assert AS.NULL_TOL_XHW == 5e-3
    # the cross-architecture allowance must stay far below the
    # effects it guards (seed-0 source effects are order 1e-1)
    assert AS.NULL_TOL_XHW < 0.01
    src = open(os.path.join(
        PIPE, "analyze_bottom20_section4_3seed.py")).read()
    assert "hardware_matched" in src and "gpu_arch" in src


def test_gpu_arch_classifies_and_degrades_safely(tmp_path):
    for name, want in (("NVIDIA H100 80GB HBM3", "H100"),
                       ("NVIDIA H100", "H100"),
                       ("NVIDIA A100-SXM4-80GB", "A100"),
                       ("NVIDIA RTX A6000", "A6000")):
        rd = tmp_path / name.replace(" ", "_")
        rd.mkdir()
        (rd / "config.json").write_text(
            json.dumps({"hardware": {"gpu_name": name}}))
        assert AS.gpu_arch(str(rd)) == want
    # missing metadata / missing file never crash and never claim a
    # match (an "unknown" pair is treated as cross-architecture)
    rd = tmp_path / "nohw"
    rd.mkdir()
    (rd / "config.json").write_text(json.dumps({"hardware": {}}))
    assert AS.gpu_arch(str(rd)) == "unknown"
    assert AS.gpu_arch(str(tmp_path / "does_not_exist")) == "unknown"
