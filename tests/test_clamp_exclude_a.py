"""Tests for the CAUSAL SOURCE-EXCLUSION wave (2026-08-18,
mistral_clamp_exclude_a): the completed graph-clamp _b0_ SFT runs
include the 145 fixed agents' innate labels in every shared weight
update; the _b0xa_ arm keeps cohort A fully present in the
environment (served, gated, pinned, stubborn peer pairing, matched
twin) but drops its rows from EVERY SFT batch (round 0 included) via
SFT_EXCLUDE_CLAMPED=1, so each round trains on all and only the 578
responsive agents with their current live opinions.

Generator: exactly 48 rows (2 masks x 4 AI gates x 6 social gates,
seed 0, the exact b0 queue surface) + 2 smokes (seed 991, ea0p4
es0p2); the sub pins SFT_EXCLUDE_CLAMPED=1 in the env; zero tag
collisions with the completed b0/d8 families (both are REUSED).

Runner: persists sft_idx_raw/sft_y_raw -- the ordered training ids +
labels per deploy round -- whenever the flag is on.

Checker, via REAL-population fixtures (723 agents, artifact masks +
edges, the real stubborn operator, reusing test_innate_clamp_graph's
builder): healthy b0xa production/es0-baseline/smoke runs PASS; a
fixed id inside a training batch, a missing responsive id, a
duplicate responsive id, a label off the live-opinion replay, a
tampered mask, a b0xa tag without the flag, the flag on a b0 tag and
an n_train row off the 578 complement all FAIL; legacy b0 fixtures
still pass unchanged.

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac).
"""
import importlib.util
import os

import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")

_spec_tg = importlib.util.spec_from_file_location(
    "clamp_graph_fixtures_xa", os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_innate_clamp_graph.py"))
tg = importlib.util.module_from_spec(_spec_tg)
_spec_tg.loader.exec_module(tg)

GEN = tg.GEN


def _num(v):
    return f"{v:g}".replace(".", "p")


# -- generator -----------------------------------------------------------

def test_generator_xa_is_exactly_48():
    rows = GEN.clamp_xa_rows()
    assert len(rows) == 48
    tags = [r.split(",")[0] for r in rows]
    assert len(set(tags)) == 48
    assert all("_b0xa_" in t and "_stub_" in t and t.endswith("_s0")
               for t in tags)
    for tok, want in (("_gclump_", 24), ("_gscat_", 24)):
        assert sum(1 for t in tags if tok in t) == want, tok
    for g in (0.1, 0.2, 0.4, 1.0):
        assert sum(1 for t in tags if f"_ea{_num(g)}_" in t) == 12, g
    for e in (0.0, 0.05, 0.1, 0.2, 0.4, 1.0):
        assert sum(1 for t in tags if f"_es{_num(e)}_" in t) == 8, e
    # the completed b0/d8 families are REUSED: zero shared tags
    assert not (set(tags)
                & {r.split(",")[0] for r in GEN.clamp_graph_rows()})
    assert not (set(tags)
                & {r.split(",")[0] for r in GEN.clamp_graph_d8_rows()})
    # the exact b0 queue surface: ordinary SFT, beta 0, zero exemplars
    for r in rows:
        cols = [c.strip() for c in r.split(",")]
        assert cols[1] == "sft" and cols[2] == "0" \
            and cols[16] == "0", r


def test_generator_xa_smokes():
    smk = GEN.clamp_xa_smoke_rows()
    assert len(smk) == 2
    tags = [r.split(",")[0] for r in smk]
    assert all("_b0xa_" in t and "_stub_ea0p4_" in t
               and "_es0p2_s991" in t for t in tags)
    assert sum(1 for t in tags if "_gclump_" in t) == 1


def test_xa_sub_template_pins_the_exclusion_flag():
    for kind in ("main", "smoke"):
        sub = GEN.clamp_xa_sub(kind)
        assert "SFT_EXCLUDE_CLAMPED=1" in sub
        assert "INNATE_CLAMP_PEER_MODE=stubborn" in sub
    assert GEN.clamp_xa_sub("main").rstrip().endswith(
        "nrounds, cmode from experiments/condor/"
        "configs_pofd_mistral_clamp_exclude_a.txt")


# -- runner / analyzer surface -------------------------------------------

def test_runner_exclusion_surface():
    src = open(os.path.join(PIPE, "run_pokec_gated_lm.py")).read()
    assert 'sft_exclude_clamped = _env_int("SFT_EXCLUDE_CLAMPED", 0)' \
        in src
    assert "_keep_xa = ~clamp_mask[train_data" in src
    assert "sft_idx_raw.append(" in src and "sft_y_raw.append(" in src
    assert 'config["sft_exclude_clamped"] = True' in src


def test_analyzer_exclude_a_surface():
    src = open(os.path.join(
        PIPE, "analyze_clamp_graph_exclude_a.py")).read()
    assert "clamp_graph_exclude_a_analysis" in src
    assert "HARD FAIL" in src and "sys.exit(1)" in src
    assert "exclude_a_per_cell.csv" in src
    assert "exclude_a_pairwise.csv" in src
    assert "exclude_a_arm_contrast.csv" in src


# -- fixtures ------------------------------------------------------------

def xa_post(d):
    """Attach the exclusion provenance the runner persists: per round
    the ORDERED responsive ids and each agent's live opinion label
    (innate on round 0, op_raw[t-1] after), plus the 578 n_train."""
    resp_ids = (~d["innate_clamp_mask"]).nonzero().flatten().long()
    idx_rows, y_rows = [], []
    for t in range(d["op_raw"].shape[0]):
        y = d["innate"] if t == 0 else d["op_raw"][t - 1]
        idx_rows.append(resp_ids.clone())
        y_rows.append(y[resp_ids].float().clone())
    d["sft_idx_raw"] = torch.stack(idx_rows)
    d["sft_y_raw"] = torch.stack(y_rows)
    for r in d["trajectory"]:
        r["n_train"] = int(resp_ids.numel())


def xa_cfg(c):
    c["sft_exclude_clamped"] = True


def build_xa(parent, mask_name, gate=0.4, es=0.2, seed=0, nrounds=30,
             prefix="pofdclamp", **kw):
    gtok = tg.GTOK_OF[mask_name]
    tag = kw.pop("tag", None) or (
        f"{prefix}_mistral7b_b0xa_{gtok}_stub"
        f"_ea{_num(gate)}_w0p5_l0p2_es{_num(es)}_s{seed}")
    return tg.build(parent, "b0", mask_name, gate, es, seed=seed,
                    nrounds=nrounds, prefix=prefix, tag=tag,
                    cfg_mut=xa_cfg, post=xa_post, **kw)


# -- checker: healthy ----------------------------------------------------

def test_healthy_xa_production_clumped(tmp_path):
    tg.assert_verdict(build_xa(tmp_path, "graph_clumped", 0.4, 0.2),
                      True)


def test_healthy_xa_scattered_es0_baseline(tmp_path):
    tg.assert_verdict(build_xa(tmp_path, "graph_scattered", 1.0, 0.0),
                      True)


def test_healthy_xa_smoke(tmp_path):
    tg.assert_verdict(build_xa(tmp_path, "graph_clumped", 0.4, 0.2,
                               seed=991, nrounds=3,
                               prefix="pofdclampsmk"), True)


# -- checker: sabotage ---------------------------------------------------

def test_fixed_id_in_training_fails(tmp_path):
    rd = build_xa(tmp_path, "graph_clumped")

    def fn(t):
        fro = t["innate_clamp_mask"].nonzero().flatten()
        t["sft_idx_raw"][3][0] = fro[0]
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "cohort A must never enter SFT")


def test_missing_responsive_id_fails(tmp_path):
    rd = build_xa(tmp_path, "graph_clumped")

    def fn(t):
        t["sft_idx_raw"] = t["sft_idx_raw"][:, :-1]
        t["sft_y_raw"] = t["sft_y_raw"][:, :-1]
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "training count off")


def test_duplicate_responsive_id_fails(tmp_path):
    rd = build_xa(tmp_path, "graph_clumped")

    def fn(t):
        t["sft_idx_raw"][4][10] = t["sft_idx_raw"][4][11]
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "duplicate training id")


def test_wrong_label_fails(tmp_path):
    rd = build_xa(tmp_path, "graph_clumped")

    def fn(t):
        t["sft_y_raw"][2][7] += 0.05
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "off the live-opinion replay")


def test_wrong_mask_fails(tmp_path):
    rd = build_xa(tmp_path, "graph_clumped")

    def fn(t):
        m = t["innate_clamp_mask"]
        frozen = m.nonzero().flatten()
        free = (~m).nonzero().flatten()
        m[frozen[0]] = False
        m[free[-1]] = True
        t["innate_clamp_count"] = int(m.sum())
        t["innate_clamp_hash"] = tg.gp.innate_clamp_hash(m)
        t["op_raw"][:, m] = t["innate"][m].unsqueeze(0)
        t["twin_raw"][:, m] = t["innate"][m].unsqueeze(0)
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "differ")


def test_xa_tag_without_flag_fails(tmp_path):
    rd = build_xa(tmp_path, "graph_clumped")

    def fn(t):
        del t["config"]["sft_exclude_clamped"]
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "sft_exclude_clamped")


def test_flag_on_b0_tag_fails(tmp_path):
    rd = tg.build(tmp_path, "b0", "graph_clumped", 0.4, 0.2)

    def fn(t):
        t["config"]["sft_exclude_clamped"] = True
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "non-b0xa clamp tag")


def test_n_train_off_complement_fails(tmp_path):
    rd = build_xa(tmp_path, "graph_clumped")

    def fn(t):
        t["trajectory"][5]["n_train"] = 723
    tg.edit(rd, fn)
    tg.assert_verdict(rd, False, "n_train")


def test_xa_outside_graph_wave_fails(tmp_path):
    rd = build_xa(
        tmp_path, "graph_clumped",
        tag="pofdclamp_mistral7b_b0xa_strat_stub_ea0p4"
            "_w0p5_l0p2_es0p2_s0")
    tg.assert_verdict(rd, False, "only in the graph-placement wave")


# -- legacy non-regression -----------------------------------------------

def test_included_b0_fixture_still_passes(tmp_path):
    tg.assert_verdict(
        tg.build(tmp_path, "b0", "graph_clumped", 0.4, 0.2), True)
