"""Tests for the QWEN2.5 FULL-ANCHOR (k=1) Section-3 grid
(2026-08-19, qwen_k1_grid).

24 jobs: b0/b1 x eps_AI {.1,.2,.4,1} x eps_social {0,.05,.2}, seed 0,
Qwen2.5-7B-Instruct. The wave is the completed k=0.2 Section-3 grid
with INNATE_LAMBDA 0.2 -> 1 and NOTHING else -- asserted here by
diffing the generated environment against the shipped k=0.2 sub, so
an accidental second change cannot pass.

Tag grammar: the anchor rides the established _l1_ token, never a
bare _k1_ token (that grammar is ICL-K: _k0_, _k8live_, _k32noai_,
and "k0" is in the fam arm regex). The pofdfamk1_ prefix keeps the
checker's FAM branch while the generic _w/_l/_es gate pins
innate_lambda=1, so no checker change is needed -- verified against
a real trajectory re-tagged into the family.

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac).
"""
import importlib.util
import json
import os
import subprocess
import sys

import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
CONDOR = os.path.join(REPO, "experiments", "condor")
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")
CHECKER = os.path.join(PIPE, "check_pofd_sanity.py")
RUNS = os.path.join(REPO, "notes", "pofd", "cluster")
ARMS = ["b0", "b1"]
GATES = [0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.2]
# a completed k=0.2 Section-3 cell, used as the fixture source
SRC = "pofdfam_qwen7b_b1_ea1_w0p5_l0p2_es0p2_s0"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load("gen_qk1", os.path.join(CONDOR, "gen_pofd_sweep.py"))
AK = _load("analyze_qk1",
           os.path.join(PIPE, "analyze_qwen_k1_grid.py"))


def env_line(text):
    return next(ln for ln in text.splitlines()
                if ln.startswith("environment"))


def env_tokens(text):
    return sorted(t for t in env_line(text).split()
                  if not t.startswith("WANDB_RUN_SUFFIX"))


# -- generator -----------------------------------------------------------

def test_grid_is_exactly_24_cells():
    rows = GEN.qk1_rows()
    assert len(rows) == 24
    tags = [r.split(",")[0] for r in rows]
    assert len(set(tags)) == 24
    assert {(a, g, e) for a in ARMS for g in GATES for e in ESS} == {
        (a, g, e) for a in ARMS for g in GATES for e in ESS}
    for arm in ARMS:
        assert sum(1 for t in tags if f"_{arm}_" in t) == 12, arm
    for g in GATES:
        assert sum(1 for t in tags
                   if f"_ea{GEN._num(g)}_" in t) == 6, g
    for e in ESS:
        assert sum(1 for t in tags
                   if f"_es{GEN._num(e)}_" in t) == 8, e


def test_anchor_token_is_l1_never_a_bare_k1():
    tags = [r.split(",")[0] for r in GEN.qk1_rows()]
    assert all("_w0p5_l1_" in t for t in tags)
    # _k<N>_ is the ICL-K grammar; a bare _k1_ token would be
    # ambiguous with it (and with the fam arm regex's k0)
    assert not any("_k1_" in t for t in tags)
    # and the k=0.2 anchor must never appear -- these must not shadow
    # the completed Section-3 cells
    assert not any("_l0p2_" in t for t in tags)
    assert all(t.startswith("pofdfamk1_qwen7b_") for t in tags)
    # still routes to the checker's FAM branch, not the smoke branch
    assert all(t.startswith("pofdfam") for t in tags)
    assert not any(t.startswith("pofdfamsmk") for t in tags)


def test_no_collision_with_the_k0p2_grid_or_any_other_key():
    tags = {r.split(",")[0] for r in GEN.qk1_rows()}
    k02 = {f"pofdfam_qwen7b_{a}_ea{GEN._num(g)}_w0p5_l0p2"
           f"_es{GEN._num(e)}_s0"
           for a in ARMS for g in GATES for e in ESS}
    assert not (tags & k02)
    for other in (GEN.fam_rows(), GEN.famg_rows(), GEN.qgs_rows(),
                  GEN.evo_rows()):
        assert not (tags & {r.split(",")[0] for r in other})


def test_row_surface_matches_the_section3_grammar():
    for r in GEN.qk1_rows():
        cols = [c.strip() for c in r.split(",")]
        assert cols[1] == ("sft" if "_b0_" in cols[0] else "sft_kl"), r
        assert cols[3] == "0", r                      # seed 0
        assert cols[15] == "threshold", r             # numeric gate
        assert cols[16] == "0", r                     # ICL off
        assert cols[22] == "30", r                    # 30 rounds
        assert cols[23] == "Qwen/Qwen2.5-7B-Instruct", r
        assert cols[24] == "default", r               # Qwen2.5 template


def test_environment_differs_from_k0p2_by_exactly_the_anchor():
    """The whole point of the wave: one dial, nothing else."""
    with open(os.path.join(CONDOR,
                           "at_pofd_fam_gate_social.sub")) as fh:
        k02 = fh.read()
    k1 = GEN.qk1_sub()
    a, b = env_tokens(k02), env_tokens(k1)
    only_k02 = [t for t in a if t not in b]
    only_k1 = [t for t in b if t not in a]
    assert only_k02 == ["INNATE_LAMBDA=0.2"], only_k02
    assert only_k1 == ["INNATE_LAMBDA=1"], only_k1


def test_pop_reset_stays_off():
    # the comment block names POP_RESET to record its absence, so the
    # assertion has to look at the environment line only
    assert "POP_RESET" not in env_line(GEN.qk1_sub())


def test_submit_key_registered():
    with open(os.path.join(CONDOR, "submit_pofd_sweep.sh")) as fh:
        sh = fh.read()
    assert 'qwen_k1_grid) TARGETS="$WHAT" ;;' in sh
    assert "qwen_gate_sweep|qwen_k1_grid" in sh


# -- checker (real trajectory re-tagged into the k1 family) --------------

tg = _load("k1_graph_fixtures",
           os.path.join(REPO, "tests", "test_innate_clamp_graph.py"))


def build_k1(parent, tag, lam=1.0, es=0.2, gate=1.0, nrounds=30,
             cfg_mut=None):
    """A REAL k=1 trajectory: the canonical no-clamp social loop with
    the FJ anchor at `lam`, on the artifact 723-agent population.

    Relabelling a k=0.2 run does NOT work -- the checker replays the
    nested AI-then-peer update using the config's anchor and compares
    against op_raw, so the dynamics have to genuinely be k=1.
    """
    src = json.load(open(os.path.join(RUNS, SRC, "config.json")))
    cfg = dict(src)
    cfg.update({"run_tag": tag, "innate_lambda": lam, "eps": es,
                "eps_ai": gate, "n_rounds": nrounds})
    if cfg_mut:
        cfg_mut(cfg)
    N, innate = tg.N, tg.INNATE
    g = torch.Generator().manual_seed(9100)
    g_peer = torch.Generator().manual_seed(424245)
    g_peer_cf = torch.Generator().manual_seed(424245)
    w = float(cfg["w_plat"])
    x, tw = innate.clone(), innate.clone()
    rows, op_raw, pred_raw, gate_raw, twin_raw = [], [], [], [], []
    for t in range(nrounds):
        pred = torch.rand(N, generator=g)
        served = pred.clamp(0.0, 1.0)
        gate_t = (served - x).abs() < gate
        h = lam * innate + (1.0 - lam) * x      # k=1 -> h == innate
        x = torch.where(gate_t, (1.0 - w) * h + w * served, h)
        acc = tg.gp.ab_sweep(x, tg.ADJ, es, 0.0, gen=g_peer)
        tw = lam * innate + (1.0 - lam) * tw
        tg.gp.ab_sweep(tw, tg.ADJ, es, 0.0, gen=g_peer_cf)
        rows.append({"round": t, "deployment": t, "is_deploy": 1,
                     "accepted": acc,
                     "contact": float(gate_t.float().mean()),
                     "twin_mean": float(tw.mean()),
                     "twin_std": float(tw.std()),
                     "twin_bias": float(tw.mean() - innate.mean()),
                     "n_train": N})
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
    rd = parent / tag
    rd.mkdir(parents=True)
    (rd / "config.json").write_text(json.dumps(cfg))
    torch.save(d, rd / "trajectory.pt")
    return rd


def run_check(rd):
    p = subprocess.run([sys.executable, CHECKER, str(rd)],
                       capture_output=True, text=True,
                       env={**os.environ, "USE_TF": "0"})
    return p.returncode, p.stdout + p.stderr


def test_k1_tag_gates_clean_with_no_checker_change(tmp_path):
    """A pofdfamk1_ run with GENUINE k=1 dynamics must PASS: the FAM
    branch applies (prefix) and the generic _l gate pins the anchor."""
    rd = build_k1(tmp_path,
                  "pofdfamk1_qwen7b_b1_ea1_w0p5_l1_es0p2_s0")
    rc, out = run_check(rd)
    assert rc == 0, out[-3000:]


def test_checker_replays_the_anchor_not_just_the_config_field(tmp_path):
    """k=0.2 dynamics under a _l1_ tag and an innate_lambda=1 config
    must FAIL -- otherwise a mislabelled rerun of the existing grid
    would gate clean and silently duplicate it."""
    rd = build_k1(tmp_path,
                  "pofdfamk1_qwen7b_b1_ea1_w0p5_l1_es0p2_s0",
                  lam=0.2, cfg_mut=lambda c: c.update(innate_lambda=1.0))
    rc, out = run_check(rd)
    assert rc != 0, out[-3000:]
    assert "lam=1" in out or "innate_lambda" in out, out[-3000:]


def test_k1_tag_with_the_wrong_anchor_fails(tmp_path):
    """Tag says _l1_ but the config kept k=0.2 -- the generic token
    gate must catch it, otherwise the wave could silently be a
    duplicate of the k=0.2 grid."""
    tag = "pofdfamk1_qwen7b_b1_ea1_w0p5_l1_es0p2_s0"
    d = tmp_path / tag
    d.mkdir()
    t = torch.load(os.path.join(RUNS, SRC, "trajectory.pt"),
                   map_location="cpu", weights_only=False)
    cfg = t["config"]
    cfg["run_tag"] = tag                    # innate_lambda stays 0.2
    torch.save(t, d / "trajectory.pt")
    (d / "config.json").write_text(json.dumps(cfg))
    p = subprocess.run([sys.executable, CHECKER, str(d)],
                       capture_output=True, text=True,
                       env={**os.environ, "USE_TF": "0"})
    out = p.stdout + p.stderr
    assert p.returncode != 0, out[-2500:]
    assert "innate_lambda" in out, out[-2500:]


# -- analyzer ------------------------------------------------------------

def test_analyzer_pairs_the_two_anchors():
    assert AK.ARMS == ARMS and AK.GATES == GATES and AK.ESS == ESS
    assert AK.LATE == list(range(25, 30))
    assert AK.cell_tag("k0p2", "b1", 1.0, 0.2) == \
        "pofdfam_qwen7b_b1_ea1_w0p5_l0p2_es0p2_s0"
    assert AK.cell_tag("k1", "b1", 1.0, 0.2) == \
        "pofdfamk1_qwen7b_b1_ea1_w0p5_l1_es0p2_s0"
    # 24 pairs = 48 trajectories, all hard-required
    src = open(os.path.join(PIPE, "analyze_qwen_k1_grid.py")).read()
    assert "HARD FAIL" in src and "sys.exit(1)" in src
    assert "k1_per_cell.csv" in src and "k1_contrast.csv" in src


def test_analyzer_verifies_the_anchor_actually_differs():
    src = open(os.path.join(PIPE, "analyze_qwen_k1_grid.py")).read()
    # a k=1 run whose config says 0.2 would make the contrast a
    # comparison of a wave with itself
    assert 'want_lam = 1.0 if k[0] == "k1" else 0.2' in src
    assert "innate_lambda" in src


def test_analyzer_metrics():
    import numpy as np
    a = np.array([0.1, 0.9, 0.5])
    b = np.array([0.2, 0.4, 0.8])
    assert abs(AK.w1(a, b) - 0.1) < 1e-12
    op = torch.rand(30, 723)
    innate = torch.rand(723)
    s = AK.cell_stats({"op_raw": op, "innate": innate})
    late = op.numpy()[25:30]
    assert abs(s["mean"] - float(np.mean(late.mean(axis=1)))) < 1e-9
    assert abs(s["sd"]
               - float(np.mean([r.std(ddof=0) for r in late]))) < 1e-9
