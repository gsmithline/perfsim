"""Tests for the CORRECTED graph-clamp wave (2026-08-17,
mistral_innate_clamp_graph_d8): the cross-user live K=8 arm (_dyn_) is
replaced by PERSONAL-HISTORY ICL (_d8_: frozen weights, ICL_K=0,
ICL_DAYS=8 -- each agent's prompt carries only its OWN latest eight
recorded opinions, oldest to newest, innate-first on early rounds;
fixed agents therefore see only repetitions of their own innate).

Generator: exactly 51 rows (48 new _d8_ cells + the 3 _b0_ cells
missing at the 2026-08-17 audit, tags shared with the main graph key
by design) + 2 d8 smokes; icldays rides the last queue col (8 on d8,
0 on b0); the retired i104-retry key is gone.

Runner: writes icl_days_log.json.gz (the FULL context_block per agent
per round) whenever ICL_DAYS > 0.

Checker, via REAL-population fixtures (723 agents, artifact masks +
edges, the real stubborn operator, reusing test_innate_clamp_graph's
builder): healthy d8 runs PASS; a missing days log, a perturbed day
value, a reversed (newest-first) sequence, a cross-user exemplar
block, a stray icl_ctx_log, icl_k>0 in the config, a too-long window
and a fixed agent whose history is not pure innate repetition all
FAIL; the d8 arm is rejected outside the graph-placement wave.

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac).
"""
import gzip
import importlib.util
import json
import os

import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")

_spec_tg = importlib.util.spec_from_file_location(
    "clamp_graph_fixtures", os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_innate_clamp_graph.py"))
tg = importlib.util.module_from_spec(_spec_tg)
_spec_tg.loader.exec_module(tg)

GEN = tg.GEN
DAYS_SENTENCE = ("This user's own opinion of Action movies over the "
                 "most recent days (oldest to newest): {days}.")


def _num(v):
    return f"{v:g}".replace(".", "p")


# -- generator -----------------------------------------------------------

def test_generator_d8_is_exactly_51():
    rows = GEN.clamp_graph_d8_rows()
    assert len(rows) == 51
    tags = [r.split(",")[0] for r in rows]
    assert len(set(tags)) == 51
    d8 = [t for t in tags if "_d8_" in t]
    b0 = [t for t in tags if "_b0_" in t]
    assert len(d8) == 48 and len(b0) == 3
    assert not any("_dyn_" in t for t in tags)
    assert all("_stub_" in t and t.endswith("_s0") for t in tags)
    for tok, want in (("_gclump_", 24), ("_gscat_", 24)):
        assert sum(1 for t in d8 if tok in t) == want, tok
    for g in (0.1, 0.2, 0.4, 1.0):
        assert sum(1 for t in d8 if f"_ea{_num(g)}_" in t) == 12, g
    for e in (0.0, 0.05, 0.1, 0.2, 0.4, 1.0):
        assert sum(1 for t in d8 if f"_es{_num(e)}_" in t) == 8, e
    # the 3 b0 backfill tags are shared with the main 96 by design
    main_tags = {r.split(",")[0] for r in GEN.clamp_graph_rows()}
    assert set(b0) <= main_tags
    assert not (set(d8) & main_tags)
    # icldays is the last queue col: 8 on d8 rows, 0 on b0 rows
    for r in rows:
        want_tail = ", 8" if "_d8_" in r.split(",")[0] else ", 0"
        assert r.rstrip().endswith(want_tail), r
    # d8 rows are frozen-weight, zero-exemplar
    for r in rows:
        cols = [c.strip() for c in r.split(",")]
        if "_d8_" in cols[0]:
            assert cols[1] == "frozen" and cols[16] == "0", r


def test_generator_d8_smokes_and_retired_retry():
    smk = GEN.clamp_graph_d8_smoke_rows()
    assert len(smk) == 2
    tags = [r.split(",")[0] for r in smk]
    assert all("_d8_" in t and "_stub_ea0p4_" in t
               and "_es0p4_s991" in t for t in tags)
    assert sum(1 for t in tags if "_gclump_" in t) == 1
    assert all(r.rstrip().endswith(", 8") for r in smk)
    # the i104-retry key is retired (its dyn cells left the design;
    # its b0 cells moved into the d8 key)
    assert not hasattr(GEN, "CLAMP_GRAPH_RETRY_KEY")
    assert not os.path.exists(os.path.join(
        REPO, "experiments", "condor",
        "configs_pofd_mistral_innate_clamp_graph_retry.txt"))


def test_d8_sub_template_puts_icldays_on_the_queue():
    sub = GEN.clamp_graph_d8_sub("main")
    assert "ICL_DAYS=$(icldays)" in sub
    assert "INNATE_CLAMP_PEER_MODE=stubborn" in sub
    assert sub.rstrip().endswith(
        "nrounds, cmode, icldays from experiments/condor/"
        "configs_pofd_mistral_innate_clamp_graph_d8.txt")


# -- runner / analyzer surface -------------------------------------------

def test_runner_writes_days_log():
    src = open(os.path.join(PIPE, "run_pokec_gated_lm.py")).read()
    assert "icl_days_log.json.gz" in src
    assert "days_txt_t.append(ctx_i)" in src
    assert "icl_days_texts.append((t, days_txt_t))" in src


def test_analyzer_d8_mode_surface():
    src = open(os.path.join(PIPE, "analyze_innate_clamp_graph.py")).read()
    assert '"--icl-arm"' in src and 'default="d8"' in src
    assert "clamp_graph_d8_analysis" in src
    assert "HARD FAIL" in src and "sys.exit(1)" in src
    assert "clamp_graph_arm_contrast.csv" in src


# -- personal-history fixtures -------------------------------------------

def write_days_log(rd, icl_days=8, mutate=None):
    """Render the exact personal-history log from the fixture's own
    (innate, op_raw) -- the runner's formula. mutate(rows) may edit the
    JSON rows before writing."""
    t = torch.load(rd / "trajectory.pt", weights_only=False)
    innate, op = t["innate"], t["op_raw"]
    hist = [innate]
    rows = []
    for tt in range(op.shape[0]):
        win = hist[-icl_days:]
        ctxs = [DAYS_SENTENCE.format(
            days=", ".join(f"{float(h[i]):.2f}" for h in win))
            for i in range(op.shape[1])]
        rows.append({"round": tt, "ctx": ctxs})
        hist.append(op[tt])
    if mutate:
        mutate(rows)
    with gzip.open(rd / "icl_days_log.json.gz", "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rd


def d8_cfg(c):
    c["icl_k"] = 0
    c["icl_days"] = 8


def build_d8(parent, mask_name, gate=0.4, es=0.4, nrounds=3,
             prefix="pofdclampsmk", seed=991, **kw):
    rd = tg.build(parent, "d8", mask_name, gate, es, seed=seed,
                  nrounds=nrounds, prefix=prefix, cfg_mut=d8_cfg, **kw)
    return write_days_log(rd)


def test_days_window_semantics(tmp_path):
    # innate-first early rounds, then a sliding 8-window: length
    # min(8, t+1), round 0 shows exactly innate
    rd = tg.build(tmp_path, "d8", "graph_clumped", 0.4, 0.4, seed=0,
                  nrounds=30, prefix="pofdclamp", cfg_mut=d8_cfg)
    write_days_log(rd)
    t = torch.load(rd / "trajectory.pt", weights_only=False)
    with gzip.open(rd / "icl_days_log.json.gz", "rt") as fh:
        rows = [json.loads(line) for line in fh]
    for tt in (0, 3, 7, 8, 29):
        vals = rows[tt]["ctx"][0].rsplit(": ", 1)[1].rstrip(".") \
            .split(", ")
        assert len(vals) == min(8, tt + 1), tt
    assert rows[0]["ctx"][0].rsplit(": ", 1)[1].rstrip(".") == \
        f"{float(t['innate'][0]):.2f}"
    # oldest to newest: round 8's first value is round-0's recorded one
    v_r8 = rows[8]["ctx"][0].rsplit(": ", 1)[1].rstrip(".").split(", ")
    assert v_r8[0] == f"{float(t['op_raw'][0][0]):.2f}"
    assert v_r8[-1] == f"{float(t['op_raw'][7][0]):.2f}"


# -- checker: healthy ----------------------------------------------------

def test_healthy_d8_production_clumped(tmp_path):
    rd = tg.build(tmp_path, "d8", "graph_clumped", 0.4, 0.2, seed=0,
                  nrounds=30, prefix="pofdclamp", cfg_mut=d8_cfg)
    write_days_log(rd)
    tg.assert_verdict(rd, True)


def test_healthy_d8_scattered_es0_baseline(tmp_path):
    rd = tg.build(tmp_path, "d8", "graph_scattered", 1.0, 0.0, seed=0,
                  nrounds=30, prefix="pofdclamp", cfg_mut=d8_cfg)
    write_days_log(rd)
    tg.assert_verdict(rd, True)


def test_healthy_d8_smoke(tmp_path):
    rd = build_d8(tmp_path, "graph_scattered")
    tg.assert_verdict(rd, True)


# -- checker: sabotage ---------------------------------------------------

def test_days_log_missing_fails(tmp_path):
    rd = build_d8(tmp_path, "graph_clumped")
    os.remove(rd / "icl_days_log.json.gz")
    tg.assert_verdict(rd, False, "icl_days_log.json.gz missing")


def test_day_value_perturbed_fails(tmp_path):
    rd = build_d8(tmp_path, "graph_clumped")

    def mutate(rows):
        s = rows[2]["ctx"][5]
        head, vals = s.rsplit(": ", 1)
        v = vals.rstrip(".").split(", ")
        v[-1] = "0.99" if v[-1] != "0.99" else "0.01"
        rows[2]["ctx"][5] = head + ": " + ", ".join(v) + "."
    write_days_log(rd, mutate=mutate)
    tg.assert_verdict(rd, False, "off the (innate, op_raw) replay")


def test_days_order_reversed_fails(tmp_path):
    rd = build_d8(tmp_path, "graph_clumped")

    def mutate(rows):
        # a RESPONSIVE agent's round-2 window has 3 distinct-position
        # values; newest-first ordering must be caught
        t = torch.load(rd / "trajectory.pt", weights_only=False)
        resp = (~t["innate_clamp_mask"]).nonzero().flatten()
        i = int(resp[0])
        s = rows[2]["ctx"][i]
        head, vals = s.rsplit(": ", 1)
        v = vals.rstrip(".").split(", ")
        if v == v[::-1]:   # degenerate constant history: force distinct
            v[0] = "0.11"
        rows[2]["ctx"][i] = head + ": " + ", ".join(v[::-1]) + "."
    write_days_log(rd, mutate=mutate)
    tg.assert_verdict(rd, False, "off the (innate, op_raw) replay")


def test_cross_user_exemplar_block_fails(tmp_path):
    rd = build_d8(tmp_path, "graph_clumped")

    def mutate(rows):
        rows[1]["ctx"][0] += ("\n\nHere are the current opinions of "
                              "some other users:\n- age 31, male; "
                              "ratings: Action 4.0 -> opinion 0.80")
    write_days_log(rd, mutate=mutate)
    tg.assert_verdict(rd, False, "off the (innate, op_raw) replay")


def test_stray_icl_ctx_log_fails(tmp_path):
    rd = build_d8(tmp_path, "graph_clumped")
    with gzip.open(rd / "icl_ctx_log.json.gz", "wt") as fh:
        fh.write(json.dumps({"round": 0, "ctx": ["x"] * tg.N}) + "\n")
    tg.assert_verdict(rd, False, "icl_ctx_log.json.gz present")


def test_cfg_icl_k_nonzero_fails(tmp_path):
    def bad_cfg(c):
        d8_cfg(c)
        c["icl_k"] = 8
    rd = tg.build(tmp_path, "d8", "graph_clumped", 0.4, 0.4, seed=991,
                  nrounds=3, prefix="pofdclampsmk", cfg_mut=bad_cfg)
    write_days_log(rd)
    tg.assert_verdict(rd, False, "icl_k")


def test_window_too_long_fails(tmp_path):
    rd = tg.build(tmp_path, "d8", "graph_clumped", 0.4, 0.2, seed=0,
                  nrounds=30, prefix="pofdclamp", cfg_mut=d8_cfg)
    write_days_log(rd, icl_days=9)   # 9-value windows on late rounds
    tg.assert_verdict(rd, False, "off the (innate, op_raw) replay")


def test_fixed_history_not_innate_repetition_fails(tmp_path):
    # drift a FROZEN agent in op_raw and render the log from the
    # drifted state: the byte replay then passes, so the explicit
    # frozen-repetition check must catch it (alongside the bit-exact
    # op_raw error)
    rd = build_d8(tmp_path, "graph_clumped")
    t = torch.load(rd / "trajectory.pt", weights_only=False)
    i = int(t["innate_clamp_mask"].nonzero().flatten()[0])
    t["op_raw"][1:, i] = 0.123
    torch.save(t, rd / "trajectory.pt")
    write_days_log(rd)
    tg.assert_verdict(rd, False, "not pure innate repetition")


def test_d8_outside_graph_wave_fails(tmp_path):
    rd = tg.build(tmp_path, "d8", "graph_clumped", 0.4, 0.4, seed=991,
                  nrounds=3, prefix="pofdclampsmk", cfg_mut=d8_cfg,
                  tag=("pofdclampsmk_mistral7b_d8_strat_stub_ea0p4"
                       "_w0p5_l0p2_es0p4_s991"))
    write_days_log(rd)
    tg.assert_verdict(rd, False,
                      "only in the graph-placement wave")
