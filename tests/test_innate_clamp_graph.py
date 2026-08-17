"""Tests for the innate-clamp GRAPH-PLACEMENT wave (2026-08-17,
mistral_innate_clamp_graph_s0). STUBBORN-ONLY design: fixed agents
participate fully in peer pairing; there is NO isolated condition.

Operator units (_gated_pop.ab_sweep_stubborn): one-sided F-R midpoints
with the closed-form double-halving on the n=2 graph -- which also
DISCRIMINATES against a move-then-reset implementation (resetting the
fixed agent only after the sweep would leave the responsive agent at
0.5, not 0.35); F-F immobility; no-fixed calls reproducing legacy
ab_sweep bit-for-bit (R-R midpoint correctness); fixed bit-exactness,
mandatory F-pair participation, and responsive-mean conservation on
F-R-free sweeps.

Mask artifact (clamp_graph_masks.json, committed): 145 ids per mask,
identical joint-stratum quotas, hashes matching the ids, graph stats
recomputing from the artifact's own edge list, the innate match
(d mean/SD < 0.01, W1 < 0.01) and the placement criteria (scattered
exposure >= clumped + 20pp, cut >= 1.3x) all holding.

Generator: exactly 96 production rows + 4 smokes, every tag carrying
_gclump_/_gscat_ + _stub_, none _iso_, full 4x6 gate coverage, zero
collisions with any other wave.

Checker, via REAL-population fixtures (723 agents, artifact masks +
artifact edges, the real stubborn operator): healthy clumped/scattered
/es0-baseline/smoke runs PASS; fixed drift (deployed AND twin), mask
ids off the artifact, zeroed F-pair telemetry (accidental isolated
behavior), a tokenless graph tag, a touch mark on a fixed agent and a
reach/touch-log mismatch all FAIL; the completed no-peer wave's
fixture still passes.

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
ARTIFACT = os.path.join(REPO, "experiments", "condor",
                        "clamp_graph_masks.json")

_spec_gp = importlib.util.spec_from_file_location(
    "gp_clamp_graph_test", os.path.join(PIPE, "_gated_pop.py"))
gp = importlib.util.module_from_spec(_spec_gp)
_spec_gp.loader.exec_module(gp)

_spec_nc = importlib.util.spec_from_file_location(
    "clamp_nopeer_fixtures", os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_innate_clamp.py"))
ncfix = importlib.util.module_from_spec(_spec_nc)
_spec_nc.loader.exec_module(ncfix)

_spec_gen = importlib.util.spec_from_file_location(
    "gen_pofd_for_graph_test", os.path.join(
        REPO, "experiments", "condor", "gen_pofd_sweep.py"))
GEN = importlib.util.module_from_spec(_spec_gen)
_spec_gen.loader.exec_module(GEN)

ART = json.load(open(ARTIFACT))
N = ART["n"]
INNATE = torch.tensor(ART["innate"], dtype=torch.float32)
ADJ = torch.zeros(N, N)
for _a, _b in ART["edges"]:
    ADJ[_a][_b] = 1.0
    ADJ[_b][_a] = 1.0
MASK_OF = {}
for _name in ("graph_clumped", "graph_scattered"):
    _m = torch.zeros(N, dtype=torch.bool)
    _m[torch.tensor(ART["masks"][_name]["ids"])] = True
    MASK_OF[_name] = _m
GTOK_OF = {"graph_clumped": "gclump", "graph_scattered": "gscat"}


def _num(v):
    return f"{v:g}".replace(".", "p")


# -- operator units ------------------------------------------------------

def test_stubborn_one_sided_halving_n2_no_reset_artifact():
    x = torch.tensor([0.2, 0.8])
    adj2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    fixed = torch.tensor([True, False])
    g = torch.Generator().manual_seed(1)
    acc, st = gp.ab_sweep_stubborn(x, adj2, 1.0, 0.0, g, fixed)
    # two sequential F-R selections at eps=1: the responsive agent
    # halves its distance to the STANDING fixed value twice
    # (0.8 -> 0.5 -> 0.35). A move-then-reset implementation would land
    # at 0.5 (the second pair would see the transiently moved fixed
    # agent). One number separates the implementations.
    assert torch.equal(x[0], torch.tensor(0.2))   # bit-exact fixed
    assert abs(float(x[1]) - 0.35) < 1e-7
    assert acc == 2 and st["fr_sampled"] == 2 == st["fr_accepted"]
    assert st["touched"].tolist() == [False, True]


def test_stubborn_ff_pair_immobile():
    x = torch.tensor([0.3, 0.4])
    x0 = x.clone()
    adj2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    g = torch.Generator().manual_seed(1)
    acc, st = gp.ab_sweep_stubborn(x, adj2, 1.0, 0.0, g,
                                   torch.tensor([True, True]))
    assert torch.equal(x, x0)
    assert acc == 2 and st["fr_sampled"] == 0 and not st["touched"].any()


def test_stubborn_rr_pairs_match_legacy_midpoint():
    g1 = torch.Generator().manual_seed(4)
    xa = torch.rand(N, generator=g1)
    xb = xa.clone()
    ga, gb = torch.Generator().manual_seed(11), \
        torch.Generator().manual_seed(11)
    acc_a, st = gp.ab_sweep_stubborn(
        xa, ADJ, 0.4, 0.0, ga, torch.zeros(N, dtype=torch.bool))
    acc_b = gp.ab_sweep(xb, ADJ, 0.4, 0.0, gen=gb)
    assert torch.equal(xa, xb) and acc_a == acc_b
    assert st["fr_sampled"] == 0 and not st["touched"].any()


@pytest.mark.parametrize("mask_name", list(MASK_OF))
def test_stubborn_fixed_bitexact_and_participating(mask_name):
    fixed = MASK_OF[mask_name]
    g1 = torch.Generator().manual_seed(30)
    x0 = torch.rand(N, generator=g1)
    x = x0.clone()
    acc, st = gp.ab_sweep_stubborn(
        x, ADJ, 0.9, 0.0, torch.Generator().manual_seed(6), fixed)
    assert torch.equal(x[fixed], x0[fixed])
    assert st["fr_accepted"] <= st["fr_sampled"]
    assert not (st["touched"] & fixed).any()
    assert st["fr_sampled"] > 0     # anti-isolated on the REAL graph


def test_stubborn_conserves_responsive_mean_without_fr_accepts():
    fixed = MASK_OF["graph_clumped"]
    x0 = torch.where(fixed, torch.full((N,), 0.95),
                     torch.full((N,), 0.05))
    x = x0.clone()
    _, st = gp.ab_sweep_stubborn(
        x, ADJ, 0.03, 0.0, torch.Generator().manual_seed(3), fixed)
    assert st["fr_accepted"] == 0   # 0.9 gap >> eps: F-R never accepts
    assert abs(float(x[~fixed].mean() - x0[~fixed].mean())) < 1e-6
    assert torch.equal(x[fixed], x0[fixed])


def test_runner_mode_surface():
    src = open(os.path.join(PIPE, "run_pokec_gated_lm.py")).read()
    assert 'innate_clamp_peer_mode not in ("", "stubborn")' in src
    assert "no isolated condition" in src
    assert "graph_clumped" in src and "graph_scattered" in src


# -- mask artifact -------------------------------------------------------

def test_artifact_counts_hashes_and_quotas():
    assert ART["n"] == 723 and ART["n_fixed"] == 145
    strat = torch.tensor(ART["stratum"])
    for name, m in MASK_OF.items():
        assert int(m.sum()) == 145
        assert gp.innate_clamp_hash(m) == ART["masks"][name]["hash"]
        cnt = [int((strat[m] == s).sum())
               for s in range(len(ART["quotas"]))]
        assert cnt == ART["quotas"], name
    assert all(ART["criteria"].values())


def test_artifact_graph_stats_recompute():
    A = ADJ > 0
    for name, m in MASK_OF.items():
        st = ART["masks"][name]["stats"]
        r = ~m
        assert int(A[m][:, r].sum()) == st["cut_edges"], name
        assert int(A[m][:, m].sum()) // 2 == st["ff_edges"], name
        exposed = int((A[r][:, m].sum(1) > 0).sum())
        assert exposed == st["exposed_responsive"], name


def test_artifact_innate_match_and_placement_criteria():
    a = torch.sort(INNATE[MASK_OF["graph_clumped"]]).values
    b = torch.sort(INNATE[MASK_OF["graph_scattered"]]).values
    assert abs(float(a.mean() - b.mean())) < 0.01
    assert abs(float(a.std() - b.std())) < 0.01
    assert float((a - b).abs().mean()) < 0.01
    sc = ART["masks"]["graph_clumped"]["stats"]
    ss = ART["masks"]["graph_scattered"]["stats"]
    assert ss["exposure"] >= sc["exposure"] + 0.20
    assert ss["cut_edges"] >= 1.3 * sc["cut_edges"]


# -- generator counts ----------------------------------------------------

def test_generator_grid_is_exactly_96():
    rows = GEN.clamp_graph_rows()
    assert len(rows) == 96
    tags = [r.split(",")[0] for r in rows]
    assert len(set(tags)) == 96
    assert all("_stub_" in t for t in tags)
    assert not any("_iso_" in t for t in tags)
    assert all(t.endswith("_s0") for t in tags)
    for tok, want in (("_gclump_", 48), ("_gscat_", 48),
                      ("_b0_", 48), ("_dyn_", 48)):
        assert sum(1 for t in tags if tok in t) == want, tok
    for g in (0.1, 0.2, 0.4, 1.0):
        assert sum(1 for t in tags if f"_ea{_num(g)}_" in t) == 24, g
    for e in (0.0, 0.05, 0.1, 0.2, 0.4, 1.0):
        assert sum(1 for t in tags if f"_es{_num(e)}_" in t) == 16, e
    smk = GEN.clamp_graph_smoke_rows()
    assert len(smk) == 4
    assert all("_stub_ea0p4_" in r.split(",")[0]
               and "_es0p4_s991" in r.split(",")[0] for r in smk)
    assert sum(1 for r in smk if "_gclump_" in r) == 2
    assert not (set(tags)
                & {r.split(",")[0] for r in GEN.clamp_rows()})


# -- checker fixtures (REAL population + artifact masks/edges) -----------

def build(parent, arm, mask_name, gate, es, seed=0, nrounds=30,
          prefix="pofdclamp", tag=None, cfg_mut=None, post=None):
    gtok = GTOK_OF[mask_name]
    if tag is None:
        tag = (f"{prefix}_mistral7b_{arm}_{gtok}_stub"
               f"_ea{_num(gate)}_w0p5_l0p2_es{_num(es)}_s{seed}")
    cfg = ncfix.cfg_for(tag, arm, "strat", gate, seed, nrounds)
    cfg["eps"] = es
    cfg["innate_clamp_mode"] = mask_name
    cfg["innate_clamp_peer_mode"] = "stubborn"
    if cfg_mut:
        cfg_mut(cfg)
    mask = MASK_OF[mask_name].clone()
    resp = ~mask
    g = torch.Generator().manual_seed(7000 + seed)
    g_peer = torch.Generator().manual_seed(seed + 424243)
    g_peer_cf = torch.Generator().manual_seed(seed + 424243)
    x = INNATE.clone()
    tw = INNATE.clone()
    lam, w = 0.2, 0.5
    eps_ai = cfg["eps_ai"]
    rows, op_raw, pred_raw, gate_raw, twin_raw, touch_raw = \
        [], [], [], [], [], []
    icl_idx, icl_val, ctx_rows = [], [], []
    touch_cum = torch.zeros(N, dtype=torch.bool)
    for t in range(nrounds):
        pred = torch.rand(N, generator=g)
        served = pred.clamp(0.0, 1.0)
        gate_t = (served - x).abs() < eps_ai
        h = lam * INNATE + (1.0 - lam) * x
        x = torch.where(gate_t, (1.0 - w) * h + w * served, h)
        x[mask] = INNATE[mask]
        acc, st = gp.ab_sweep_stubborn(x, ADJ, es, 0.0, g_peer, mask)
        x[mask] = INNATE[mask]
        tw = lam * INNATE + (1.0 - lam) * tw
        tw[mask] = INNATE[mask]
        gp.ab_sweep_stubborn(tw, ADJ, es, 0.0, g_peer_cf, mask)
        tw[mask] = INNATE[mask]
        touch_cum |= st["touched"]
        touch_raw.append(st["touched"].clone())
        s_twr = float(tw[resp].std())
        row = {"round": t, "deployment": t, "is_deploy": 1,
               "accepted": acc,
               "contact": float(gate_t.float().mean()),
               "twin_mean": float(tw.mean()), "twin_std": float(tw.std()),
               "twin_bias": float(tw.mean() - INNATE.mean()),
               "clamp_fr_sampled": st["fr_sampled"],
               "clamp_fr_accepted": st["fr_accepted"],
               "clamp_fr_reach": float(touch_cum[resp].float().mean()),
               "clamp_resp_std_ratio": (float(x[resp].std()) / s_twr
                                        if s_twr > 0 else None),
               "clamp_resp_disp": float((x[resp] - tw[resp])
                                        .abs().mean()),
               "clamp_gap_mean": float(x[resp].mean()
                                       - x[mask].mean()),
               "clamp_gap_w1": gp.quantile_w1(x[resp], x[mask])}
        g0m = float(INNATE[resp].mean() - INNATE[mask].mean())
        row["clamp_gap_closure"] = (1.0 - row["clamp_gap_mean"] / g0m
                                    if abs(g0m) > 1e-9 else None)
        if arm == "b0":
            row["n_train"] = 723
        else:
            row["perplexity"] = 7.77
        if arm == "dyn":
            ii = (torch.arange(N).unsqueeze(1) + 1
                  + (torch.arange(8).unsqueeze(0) + t) % (N - 1)) % N
            icl_idx.append(ii.clone())
            icl_val.append(torch.rand(N, 8, generator=g))
            ctx_rows.append({"round": t,
                             "ctx": [f"ctx-graph-{t}-{i}"
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
         "innate_clamp_frac": 0.2,
         "innate_clamp_seed": cfg["innate_clamp_seed"],
         "innate_clamp_hash": gp.innate_clamp_hash(mask),
         "innate_clamp_peer_mode": "stubborn",
         "clamp_fr_touch_raw": torch.stack(touch_raw)}
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

def test_healthy_clumped_b0(tmp_path):
    assert_verdict(build(tmp_path, "b0", "graph_clumped", 0.4, 0.2),
                   True)


def test_healthy_scattered_dyn(tmp_path):
    assert_verdict(build(tmp_path, "dyn", "graph_scattered", 0.2, 0.4),
                   True)


def test_healthy_es0_baseline(tmp_path):
    assert_verdict(build(tmp_path, "b0", "graph_clumped", 0.4, 0.0),
                   True)


def test_healthy_smoke(tmp_path):
    assert_verdict(build(tmp_path, "dyn", "graph_scattered", 0.4, 0.4,
                         seed=991, nrounds=3, prefix="pofdclampsmk"),
                   True)


# -- sabotage ------------------------------------------------------------

def test_fixed_drift_deployed_fails(tmp_path):
    rd = build(tmp_path, "b0", "graph_clumped", 0.4, 0.2)

    def fn(t):
        fro = t["innate_clamp_mask"].nonzero().flatten()
        t["op_raw"][12][fro[1]] += 1e-4
    edit(rd, fn)
    assert_verdict(rd, False, "drift off")


def test_fixed_drift_twin_fails(tmp_path):
    rd = build(tmp_path, "dyn", "graph_scattered", 0.2, 0.4)

    def fn(t):
        fro = t["innate_clamp_mask"].nonzero().flatten()
        t["twin_raw"][9][fro[0]] += 1e-5
    edit(rd, fn)
    assert_verdict(rd, False, "twin_raw")


def test_mask_off_artifact_fails(tmp_path):
    rd = build(tmp_path, "b0", "graph_clumped", 0.4, 0.2)

    def fn(t):
        m = t["innate_clamp_mask"]
        frozen = m.nonzero().flatten()
        free = (~m).nonzero().flatten()
        m[frozen[0]] = False
        m[free[-1]] = True
        t["innate_clamp_count"] = int(m.sum())
        t["innate_clamp_hash"] = gp.innate_clamp_hash(m)
        # keep op/twin consistent with the tampered mask so ONLY the
        # artifact comparison can catch it
        t["op_raw"][:, m] = t["innate"][m].unsqueeze(0)
        t["twin_raw"][:, m] = t["innate"][m].unsqueeze(0)
    edit(rd, fn)
    assert_verdict(rd, False, "differ")


def test_zeroed_fr_telemetry_isolated_behavior_fails(tmp_path):
    rd = build(tmp_path, "b0", "graph_clumped", 0.4, 0.2)

    def fn(t):
        for r in t["trajectory"]:
            r["clamp_fr_sampled"] = 0
            r["clamp_fr_accepted"] = 0
            r["clamp_fr_reach"] = 0.0
        t["clamp_fr_touch_raw"][:] = False
    edit(rd, fn)
    assert_verdict(rd, False, "no isolated condition exists")


def test_tokenless_graph_tag_fails(tmp_path):
    rd = build(
        tmp_path, "b0", "graph_clumped", 0.4, 0.2,
        tag="pofdclamp_mistral7b_b0_gclump_ea0p4_w0p5_l0p2_es0p2_s0")
    assert_verdict(rd, False, "_stub_ token")


def test_touch_on_fixed_agent_fails(tmp_path):
    rd = build(tmp_path, "dyn", "graph_scattered", 0.2, 0.4)

    def fn(t):
        fro = t["innate_clamp_mask"].nonzero().flatten()
        t["clamp_fr_touch_raw"][5][fro[0]] = True
    edit(rd, fn)
    assert_verdict(rd, False, "touch marks a FIXED agent")


def test_reach_touchlog_mismatch_fails(tmp_path):
    rd = build(tmp_path, "dyn", "graph_scattered", 0.2, 0.4)

    def fn(t):
        t["trajectory"][10]["clamp_fr_reach"] = 0.999
    edit(rd, fn)
    assert_verdict(rd, False, "touch-log cumulative")


# -- legacy non-regression -----------------------------------------------

def test_nopeer_clamp_fixture_still_passes(tmp_path):
    rd = ncfix.build(tmp_path, "b0", "strat", 0.4, seed=42)
    rc, out = run_check(rd)
    assert rc == 0, out[-2500:]
