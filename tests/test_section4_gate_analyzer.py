"""ADVERSARIAL tests for analyze_section4_gate.py (the Section-4
CORRECTED-GATE wave, key section4_gate_anch2, 72 GPU cells).

Everything here is SYNTHETIC and LOCAL: trajectory.pt fixtures written
into tmp_path with pure torch tensors.  No model is loaded, no cluster is
touched, no artifact from runs/ is read.  Run with

    USE_TF=0 python -m pytest tests/test_section4_gate_analyzer.py -q

WHAT THIS FILE IS DEFENDING AGAINST.  The analyzer is where a correct wave
turns into a wrong claim, and every failure mode it has renders perfectly:

  * a tag parsed loosely, so an eps_AI=0.4 leftover from another wave gets
    swept into a 72-cell grid that then looks complete;
  * a late window off by one -- or silently different from the published
    Section-4 window, which would make the corrected-gate numbers
    non-comparable to the numbers already in the paper;
  * cohort A reconstructed differently in the two conditions, so the
    "fixed minus evolving" contrast masks two different sets of agents;
  * a sign flip between "fixed minus evolving" and the prior analyzer's
    "evolving minus fixed";
  * a two-round wiggle inside the late window read as a trend;
  * a missing seed averaged into a row that still prints a three-seed
    confidence interval.

CONTRACT UNDER TEST
  tag grammar  pofds4g_mistral7b_{arm}_{fixb20|evoall}_anch2
               _ea{EA}_w0p5_l0p2_es{ES}_s{SEED}
  late window  op_raw indices 25..29 -- IDENTICAL to
               analyze_bottom20_section4_3seed.LATE (cross-checked against
               that module, not against a copied literal)
  interval     three-seed mean +/- 4.302652729911275 * sd / sqrt(3)
  cohort A     the round(0.20 n) lowest-innate agents, (innate, id)
               ranking, reconstructed from `innate` alone so the evolving
               condition is masked exactly like its fixed partner
  partial      a short seed set is NEVER averaged
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")
ANALYZER = os.path.join(PIPE, "analyze_section4_gate.py")
PRIOR = os.path.join(PIPE, "analyze_bottom20_section4_3seed.py")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/perfsim-s4gate-test-mpl")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AN = _load("_analyze_s4gate", ANALYZER)
gp = _load("_s4gate_gated_pop", os.path.join(PIPE, "_gated_pop.py"))

# a small synthetic population: the analyzer derives n and the cohort size
# from `innate`, so the contract can be exercised without 723 agents
N = 60
T = 30
GPU = "NVIDIA H100 80GB HBM3"
INNATE = 0.1 + 0.8 * torch.rand(N, generator=torch.Generator()
                                .manual_seed(20260824))
MASK_A = AN.cohort_a_mask(INNATE)
MASK_B = ~MASK_A
MU0_B = float(INNATE[MASK_B].mean())
SD0_B = float(INNATE[MASK_B].std())


# ---------------------------------------------------------------- fixtures
def _off(arm, cond, ea, es, seed):
    """Late-window cohort-B offset for one cell.  Constant in t (a SETTLED
    cell), and identical across conditions for d8/eps_social=0 so the
    structural null holds bit-exactly there, as it must."""
    if arm == "d8" and es == 0.0:
        return 0.02
    base = 0.010 * (2.0 if arm == "b0" else 1.0) * (1.0 + es)
    if cond == "fixed":
        base += 0.004 * (1.0 if ea == 1.0 else 0.5)
    sign = 1.0 if cond == "fixed" else -1.0
    return base + 0.0005 * AN.SEEDS.index(seed) * sign


def _scale(arm, cond, ea, es, seed):
    """Cohort-B SD multiplier for one cell (SD_B = scale * SD_B(innate))."""
    if arm == "d8" and es == 0.0:
        return 0.9
    s = 0.9 + 0.002 * AN.SEEDS.index(seed)
    if cond == "fixed":
        s += 0.05 * (1.0 + es)
    return s


def build_cell(root, arm, cond, ea, es, seed, off=None, scale=None,
               rounds=T, tag=None, gpu=GPU, cfg_mut=None, post=None,
               innate=None, drift=0.0, cycle_amp=0.0):
    """Write ONE synthetic trajectory.pt with the real artifact schema.

    op_raw[t] is treated as the END-OF-ROUND POST-PEER state, exactly as
    the analyzer documents.  Cohort B is an affine image of innate, so
    every late-window statistic is known in closed form:
        mean_B(t) = MU0_B + off + drift*(t - 25) + cycle_amp*(-1)**t
        SD_B(t)   = scale * SD0_B
    Cohort A is pinned bit-exactly at innate in the FIXED condition and
    displaced in the EVOLVING one.
    """
    innate = INNATE.clone() if innate is None else innate.clone()
    mask = AN.cohort_a_mask(innate)
    b = ~mask
    off = _off(arm, cond, ea, es, seed) if off is None else off
    scale = _scale(arm, cond, ea, es, seed) if scale is None else scale
    mu0 = innate[b].mean()
    tag = tag or AN.cell_tag(arm, cond, ea, es, seed)

    op, twin, pred, rows = [], [], [], []
    for t in range(rounds):
        x = innate.clone()
        shift = off + drift * (t - AN.LATE_IDX[0]) + cycle_amp * (-1) ** t
        x[b] = mu0 + scale * (innate[b] - mu0) + shift
        if cond == "evolving":
            x[mask] = innate[mask] + 0.03
        op.append(x)
        twin.append(innate.clone())
        pred.append(torch.full((N,), 0.6))
        rows.append({"round": t, "accepted": 0.5,
                     "op_std": float(x.std())})

    cfg = {"run_tag": tag, "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
           "n_rounds": rounds, "seed": seed, "eps": es, "eps_ai": ea,
           "gamma_bias": 0.0, "ai_gate_mode": "threshold",
           "peer_gate_mode": "threshold", "w_plat": AN.W_PLAT,
           "innate_lambda": AN.INNATE_LAMBDA,
           "population_update": AN.POP_UPDATE_V2,
           "icl_days": 8 if arm == "d8" else 0,
           "hardware": {"gpu_name": gpu}}
    if cond == "fixed":
        cfg.update({"innate_clamp_mode": "bottom",
                    "innate_clamp_frac": AN.CLAMP_FRAC,
                    "innate_clamp_peer_mode": "stubborn"})
    if cfg_mut:
        cfg_mut(cfg)

    d = {"config": cfg, "trajectory": rows,
         "op_raw": torch.stack(op), "twin_raw": torch.stack(twin),
         "pred_raw": torch.stack(pred), "innate": innate}
    if cond == "fixed":
        d.update({"innate_clamp_mask": mask.clone(),
                  "innate_clamp_count": int(mask.sum()),
                  "innate_clamp_mode": "bottom",
                  "innate_clamp_frac": AN.CLAMP_FRAC,
                  "innate_clamp_seed": seed,
                  "innate_clamp_peer_mode": "stubborn"})
    if post:
        post(d)

    rd = os.path.join(str(root), tag)
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    torch.save(d, os.path.join(rd, "trajectory.pt"))
    return rd


def build_grid(root, **kw):
    """All 72 declared cells."""
    for arm in AN.ARMS:
        for cond in AN.CONDS:
            for ea in AN.EAS:
                for es in AN.ESS:
                    for seed in AN.SEEDS:
                        build_cell(root, arm, cond, ea, es, seed, **kw)
    return root


@pytest.fixture(scope="module")
def grid_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs_s4gate")
    build_grid(root)
    return str(root)


@pytest.fixture(scope="module")
def full_run(grid_root, tmp_path_factory):
    """The happy path, run once: returns (rc, out_dir, summary)."""
    out = tmp_path_factory.mktemp("out_s4gate")
    js = os.path.join(str(out), "summary.json")
    rc = AN.main(["--run-root", grid_root, "--out-dir", str(out),
                  "--json", js])
    with open(js) as fh:
        summary = json.load(fh)
    return rc, str(out), summary


def read_csv(path):
    import csv
    with open(path) as fh:
        return list(csv.DictReader(fh))


def copy_grid(grid_root, tmp_path):
    dst = os.path.join(str(tmp_path), "runs")
    shutil.copytree(grid_root, dst)
    return dst


# ================================================================ tag parser
def test_tag_grammar_is_the_pinned_one():
    assert AN.cell_tag("b0", "fixed", 0.2, 0.0, 42) == (
        "pofds4g_mistral7b_b0_fixb20_anch2_ea0p2_w0p5_l0p2_es0_s42")
    assert AN.cell_tag("d8", "evolving", 1.0, 0.2, 0) == (
        "pofds4g_mistral7b_d8_evoall_anch2_ea1_w0p5_l0p2_es0p2_s0")
    assert AN.cell_tag("d8", "fixed", 1.0, 1.0, 43) == (
        "pofds4g_mistral7b_d8_fixb20_anch2_ea1_w0p5_l0p2_es1_s43")


def test_tag_parser_round_trips_the_whole_grid():
    seen = set()
    for arm in AN.ARMS:
        for cond in AN.CONDS:
            for ea in AN.EAS:
                for es in AN.ESS:
                    for seed in AN.SEEDS:
                        tag = AN.cell_tag(arm, cond, ea, es, seed)
                        assert tag not in seen, f"tag collision: {tag}"
                        seen.add(tag)
                        p = AN.parse_tag(tag)
                        assert p is not None, tag
                        assert p["in_grid"]
                        assert (p["arm"], p["cond"], p["eps_ai"],
                                p["eps_social"], p["seed"]) == (
                            arm, cond, ea, es, seed)
                        assert AN.cell_tag(p["arm"], p["cond"], p["eps_ai"],
                                           p["eps_social"],
                                           p["seed"]) == tag
    assert len(seen) == AN.N_CELLS == 72


@pytest.mark.parametrize("tag", [
    "pofdclamp_mistral7b_b0_bottom_ea0p2_w0p5_l0p2_es0_s0",   # old wave
    "pofdevo_mistral7b_b0_ea0p2_w0p5_l0p2_es0_s0",            # old wave
    "pofds4g_mistral7b_b0_fixb20_ea0p2_w0p5_l0p2_es0_s0",     # no anch2
    "pofds4g_mistral7b_b0_bottom_anch2_ea0p2_w0p5_l0p2_es0_s0",
    "pofds4g_mistral7b_b0_fixb20_anch2_ea0p2_w0p5_l0p2_es0",  # no seed
    "pofds4g_mistral7b_xx_fixb20_anch2_ea0p2_w0p5_l0p2_es0_s0",
])
def test_tag_parser_rejects_foreign_tags(tag):
    assert AN.parse_tag(tag) is None, tag


@pytest.mark.parametrize("tag", [
    # parseable, but NOT one of the 72 declared cells
    "pofds4g_mistral7b_b0_fixb20_anch2_ea0p4_w0p5_l0p2_es0_s0",
    "pofds4g_mistral7b_b0_fixb20_anch2_ea0p2_w0p5_l0p2_es0p05_s0",
    "pofds4g_mistral7b_b0_fixb20_anch2_ea0p2_w1_l0p2_es0_s0",
    "pofds4g_mistral7b_b0_fixb20_anch2_ea0p2_w0p5_l1_es0_s0",
    "pofds4g_qwen3_8b_b0_fixb20_anch2_ea0p2_w0p5_l0p2_es0_s0",
    "pofds4g_mistral7b_b0_fixb20_anch2_ea0p2_w0p5_l0p2_es0_s7",
])
def test_tag_parser_flags_out_of_grid_cells(tag):
    p = AN.parse_tag(tag)
    assert p is not None and p["in_grid"] is False, tag


def test_scan_reports_tags_that_are_not_in_the_grid(tmp_path):
    build_cell(tmp_path, "b0", "fixed", 0.2, 0.0, 0)
    stray = "pofds4g_mistral7b_b0_fixb20_anch2_ea0p4_w0p5_l0p2_es0_s0"
    build_cell(tmp_path, "b0", "fixed", 0.2, 0.0, 0, tag=stray)
    found = {f["tag"]: f for f in AN.scan_run_root(str(tmp_path))}
    assert found[AN.cell_tag("b0", "fixed", 0.2, 0.0, 0)]["in_grid"]
    assert found[stray]["in_grid"] is False


# ============================================================== late window
def test_late_window_matches_the_published_section4_analyzer():
    """The corrected-gate numbers must be comparable to the published
    v1-gate ones, so the window is cross-checked against the prior module
    itself, never against a copied literal."""
    prior = _load("_prior_s4_3seed", PRIOR)
    assert AN.LATE_IDX == list(prior.LATE) == [25, 26, 27, 28, 29]
    assert AN.T_CRIT_DF2 == prior.T_CRIT
    assert AN.NULL_TOL == prior.NULL_TOL
    assert AN.NULL_TOL_XHW == prior.NULL_TOL_XHW


def test_half_window_split_is_two_then_three():
    h1, h2 = AN.half_split(AN.LATE_IDX)
    assert h1 == [25, 26] and h2 == [27, 28, 29]
    assert (AN.LATE_H1, AN.LATE_H2) == (h1, h2)


# ======================================================= cohort reconstruction
def test_cohort_a_matches_the_shared_clamp_rule():
    """The analysis-side reconstruction must be the SAME cohort the runner
    clamped -- _gated_pop.innate_clamp_mask(mode='bottom')."""
    for seed in AN.SEEDS:
        want = gp.innate_clamp_mask(INNATE, "bottom", AN.CLAMP_FRAC, seed)
        assert torch.equal(AN.cohort_a_mask(INNATE), want.bool())


def test_cohort_a_is_the_lowest_innate_with_id_tie_break():
    innate = torch.tensor([0.5, 0.1, 0.1, 0.9, 0.1, 0.7, 0.3, 0.2, 0.4, 0.6])
    m = AN.cohort_a_mask(innate, frac=0.3)          # round(0.3*10) = 3
    assert int(m.sum()) == 3
    # the three 0.1s tie; the id tie-break takes ids 1, 2, 4
    assert [i for i in range(10) if m[i]] == [1, 2, 4]


def test_cohort_a_size_is_the_bottom_20_percent():
    assert int(AN.cohort_a_mask(INNATE).sum()) == round(0.20 * N)
    big = torch.rand(723, generator=torch.Generator().manual_seed(1))
    assert int(AN.cohort_a_mask(big).sum()) == AN.EXPECTED_N_CLAMP == 145


def test_fixed_and_evolving_are_masked_identically(tmp_path):
    """Cohort A in the evolving condition is an ANALYSIS MASK: the pair
    must be masked with the same vector and share a bit-identical innate."""
    f = build_cell(tmp_path, "b0", "fixed", 0.2, 0.2, 0)
    e = build_cell(tmp_path, "b0", "evolving", 0.2, 0.2, 0)
    df, de = AN.load(f), AN.load(e)
    assert torch.equal(df["innate"], de["innate"])
    assert AN.innate_sha(df["innate"]) == AN.innate_sha(de["innate"])
    mask = AN.cohort_a_mask(df["innate"])
    assert torch.equal(df["innate_clamp_mask"].bool(), mask)
    assert de.get("innate_clamp_mask") is None
    _, lf = AN.reduce_cell(df, mask)
    _, le = AN.reduce_cell(de, mask)
    assert lf["n_a"] == le["n_a"] and lf["n_b"] == le["n_b"]
    assert lf["innate_b_mean"] == le["innate_b_mean"]


# ======================================================== late aggregation
def test_reduce_cell_late_window_reads_exactly_rounds_25_to_29(tmp_path):
    """A cell whose cohort-B mean is a known constant on 25..29 and a very
    different constant everywhere else: an off-by-one window would show."""
    rd = build_cell(tmp_path, "b0", "evolving", 1.0, 0.2, 0,
                    off=0.0, scale=1.0)
    d = AN.load(rd)
    op = d["op_raw"].clone()
    b = ~MASK_A
    op[:, b] += 1.0                       # poison every round ...
    for t in AN.LATE_IDX:
        op[t, b] -= 1.0                   # ... except the late window
    d["op_raw"] = op
    _, late = AN.reduce_cell(d, MASK_A)
    assert late["mu_b_eq"] == pytest.approx(MU0_B, abs=1e-6)
    assert late["mu_pop_eq"] != pytest.approx(MU0_B, abs=1e-3)


def test_reduce_cell_scalars_are_the_closed_form(tmp_path):
    off, scale = 0.037, 1.25
    rd = build_cell(tmp_path, "b0", "evolving", 1.0, 1.0, 0,
                    off=off, scale=scale)
    d = AN.load(rd)
    rounds, late = AN.reduce_cell(d, MASK_A)
    assert len(rounds) == T
    assert late["n_rounds"] == T
    assert late["twin_source"] == "twin_raw"
    assert late["mu_b_eq"] == pytest.approx(MU0_B + off, abs=1e-6)
    assert late["sd_b_late"] == pytest.approx(scale * SD0_B, rel=1e-5)
    # twin_raw is innate every round, so SD(B)/SD(B twin) == scale
    assert late["sd_ratio_late"] == pytest.approx(scale, rel=1e-5)
    assert late["served_mean_late"] == pytest.approx(0.6, abs=1e-6)
    assert late["mu_b_h1"] == pytest.approx(late["mu_b_h2"], abs=1e-6)


def test_per_round_rows_carry_both_indexings_and_post_peer_labels(tmp_path,
                                                                  ):
    rd = build_cell(tmp_path, "b0", "fixed", 0.2, 0.2, 0)
    rounds, _ = AN.reduce_cell(AN.load(rd), MASK_A)
    assert [r["round"] for r in rounds] == list(range(T))
    assert [r["round_1based"] for r in rounds] == list(range(1, T + 1))
    for key in ("pop_mean", "pop_sd", "a_mean", "a_sd", "b_mean", "b_sd",
                "w1_twin_pop", "served_mean"):
        assert key in rounds[0], key
    src = open(ANALYZER).read()
    assert "END-OF-ROUND POST-PEER" in src
    assert "post-peer" in src


def test_w1_is_the_house_equal_size_definition():
    a = torch.tensor([0.0, 1.0, 2.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    assert AN.w1(a, b) == pytest.approx(1.0)
    assert AN.w1(a, a.flip(0)) == pytest.approx(0.0)


def test_twin_falls_back_to_innate_when_the_run_has_none(tmp_path):
    """At eps_social=0 the runner may not write a twin; innate IS the
    no-platform process there (k>0 has innate as a fixed point)."""
    rd = build_cell(tmp_path, "d8", "evolving", 0.2, 0.0, 0,
                    post=lambda d: d.pop("twin_raw"))
    _, late = AN.reduce_cell(AN.load(rd), MASK_A)
    assert late["twin_source"] == "innate_es0"
    assert math.isfinite(late["sd_ratio_late"])


# ========================================================= t-interval math
def test_tci3_matches_the_literal_and_scipy():
    vals = [0.10, 0.14, 0.12]
    m, sd, lo, hi = AN.tci3(vals)
    assert m == pytest.approx(0.12)
    assert sd == pytest.approx(0.02)
    half = 4.302652729911275 * 0.02 / math.sqrt(3)
    assert lo == pytest.approx(0.12 - half)
    assert hi == pytest.approx(0.12 + half)
    scipy_stats = pytest.importorskip("scipy.stats")
    assert AN.T_CRIT_DF2 == pytest.approx(
        float(scipy_stats.t.ppf(0.975, 2)), rel=1e-12)


def test_tci3_refuses_a_short_seed_set():
    with pytest.raises(ValueError):
        AN.tci3([0.1, 0.2])
    with pytest.raises(ValueError):
        AN.tci3([0.1, 0.2, 0.3, 0.4])


def test_excludes_reference():
    assert AN.excludes(0.1, 0.2, 0.0) is True
    assert AN.excludes(-0.2, -0.1, 0.0) is True
    assert AN.excludes(-0.1, 0.2, 0.0) is False
    assert AN.excludes(1.1, 1.3, 1.0) is True
    assert AN.excludes(0.9, 1.3, 1.0) is False


def test_agg_block_never_averages_a_short_seed_set():
    full = {s: 0.1 + 0.01 * i for i, s in enumerate(AN.SEEDS)}
    row = AN.agg_block("x", full)
    assert row["x_mean"] == pytest.approx(0.11)
    assert row["x_ci_excludes_zero"] is True
    short = {AN.SEEDS[0]: 0.1, AN.SEEDS[1]: 0.2}
    row = AN.agg_block("x", short)
    assert row["x_s0"] == 0.1 and row[f"x_s{AN.SEEDS[2]}"] is None
    assert row["x_mean"] is None and row["x_ci_lo"] is None
    assert row["x_ci_excludes_zero"] is None
    nan = dict(full)
    nan[AN.SEEDS[1]] = float("nan")
    assert AN.agg_block("x", nan)["x_mean"] is None


# ========================================== source / dispersion arithmetic
def _synth_cells(fixed_b, evolving_b, fixed_sd=0.10, evolving_sd=0.10,
                 arm="b0", ea=0.2, es=0.2, h1=None, h2=None):
    """A hand-built `cells` map (scalars only) for the aggregation layer."""
    cells = {}
    for cond, mus, sd in (("fixed", fixed_b, fixed_sd),
                          ("evolving", evolving_b, evolving_sd)):
        for i, s in enumerate(AN.SEEDS):
            if mus[i] is None:
                continue
            cells[(arm, cond, ea, es, s)] = {
                "mu_b_eq": mus[i], "mu_pop_eq": mus[i] + 0.01,
                "sd_b_late": sd, "sd_pop_late": sd + 0.02,
                "sd_ratio_late": sd / 0.10,
                "mu_b_h1": (h1[cond][i] if h1 else mus[i]),
                "mu_b_h2": (h2[cond][i] if h2 else mus[i]),
                "sd_b_h1": sd, "sd_b_h2": sd,
                "gpu_arch": "H100", "innate_sha256": "deadbeef"}
    return cells


def test_source_effect_is_fixed_minus_evolving_with_the_prior_sign_too():
    cells = _synth_cells([0.50, 0.52, 0.54], [0.40, 0.41, 0.42])
    row = [r for r in AN.build_source_rows(cells, 0.002)
           if r["arm"] == "b0" and r["eps_ai"] == 0.2
           and r["eps_social"] == 0.2][0]
    assert row["status"] == "complete"
    assert row["delta_mu_b_s0"] == pytest.approx(0.10)
    assert row["delta_mu_b_s42"] == pytest.approx(0.11)
    assert row["delta_mu_b_s43"] == pytest.approx(0.12)
    m, sd, lo, hi = AN.tci3([0.10, 0.11, 0.12])
    assert row["delta_mu_b_mean"] == pytest.approx(m)
    assert row["delta_mu_b_ci_lo"] == pytest.approx(lo)
    assert row["delta_mu_b_ci_hi"] == pytest.approx(hi)
    assert row["delta_mu_b_ci_excludes_zero"] is True
    # the published analyzer's sign, and its interval, both flipped
    assert row["t_a_evolving_minus_fixed_mean"] == pytest.approx(-m)
    assert row["t_a_evolving_minus_fixed_ci_lo"] == pytest.approx(-hi)
    assert row["t_a_evolving_minus_fixed_ci_hi"] == pytest.approx(-lo)
    assert (row["t_a_evolving_minus_fixed_ci_lo"]
            <= row["t_a_evolving_minus_fixed_ci_hi"])


def test_source_effect_incomplete_seed_set_is_not_averaged():
    cells = _synth_cells([0.50, 0.52, None], [0.40, 0.41, 0.42])
    row = [r for r in AN.build_source_rows(cells, 0.002)
           if r["eps_ai"] == 0.2 and r["eps_social"] == 0.2
           and r["arm"] == "b0"][0]
    assert row["status"] == "incomplete"
    assert row["n_seeds_paired"] == 2
    assert row["delta_mu_b_mean"] is None
    assert row["delta_mu_b_ci_lo"] is None
    assert row["delta_mu_b_ci_excludes_zero"] is None
    assert row["delta_mu_b_s0"] == pytest.approx(0.10)


def test_dispersion_ratio_is_paired_per_seed():
    cells = _synth_cells([0.5] * 3, [0.5] * 3, fixed_sd=0.12,
                         evolving_sd=0.10)
    row = [r for r in AN.build_dispersion_rows(cells, 0.002)
           if r["arm"] == "b0" and r["eps_ai"] == 0.2
           and r["eps_social"] == 0.2][0]
    assert row["sd_b_fixed_mean"] == pytest.approx(0.12)
    assert row["sd_b_evolving_mean"] == pytest.approx(0.10)
    assert row["delta_sd_b_mean"] == pytest.approx(0.02)
    assert row["sd_ratio_b_mean"] == pytest.approx(1.2)
    assert row["sd_ratio_b_ci_excludes_one"] is True
    assert row["sd_ratio_b_sd"] == pytest.approx(0.0, abs=1e-12)


def test_drift_flag_separates_a_settled_series_from_a_drifting_one():
    settled = _synth_cells([0.50, 0.52, 0.54], [0.40, 0.41, 0.42])
    row = [r for r in AN.build_source_rows(settled, 0.002)
           if r["arm"] == "b0" and r["eps_ai"] == 0.2
           and r["eps_social"] == 0.2][0]
    assert row["delta_mu_b_drift_mean"] == pytest.approx(0.0)
    assert row["delta_mu_b_drift_flag"] is False

    # the same late-window MEAN, but the second half sits 0.02 above the
    # first: a trend that a window mean alone would hide
    h1 = {"fixed": [0.49, 0.51, 0.53], "evolving": [0.40, 0.41, 0.42]}
    h2 = {"fixed": [0.51, 0.53, 0.55], "evolving": [0.40, 0.41, 0.42]}
    drifting = _synth_cells([0.50, 0.52, 0.54], [0.40, 0.41, 0.42],
                            h1=h1, h2=h2)
    row = [r for r in AN.build_source_rows(drifting, 0.002)
           if r["arm"] == "b0" and r["eps_ai"] == 0.2
           and r["eps_social"] == 0.2][0]
    assert row["delta_mu_b_mean"] == pytest.approx(0.11)
    assert row["delta_mu_b_drift_mean"] == pytest.approx(0.02)
    assert row["delta_mu_b_drift_flag"] is True
    # 0.02 is above the 0.002 tolerance but still inside the three-seed
    # interval half-width (4.3027 * 0.01 / sqrt(3) = 0.0248), so the second,
    # stricter flag stays down -- the two tests are not redundant
    assert row["delta_mu_b_drift_exceeds_ci"] is False

    h1 = {"fixed": [0.48, 0.50, 0.52], "evolving": [0.40, 0.41, 0.42]}
    h2 = {"fixed": [0.52, 0.54, 0.56], "evolving": [0.40, 0.41, 0.42]}
    hard = _synth_cells([0.50, 0.52, 0.54], [0.40, 0.41, 0.42],
                        h1=h1, h2=h2)
    row = [r for r in AN.build_source_rows(hard, 0.002)
           if r["arm"] == "b0" and r["eps_ai"] == 0.2
           and r["eps_social"] == 0.2][0]
    assert row["delta_mu_b_drift_mean"] == pytest.approx(0.04)
    assert row["delta_mu_b_drift_flag"] is True
    assert row["delta_mu_b_drift_exceeds_ci"] is True


def test_drift_flag_fires_end_to_end_on_a_drifting_trajectory(tmp_path):
    root = os.path.join(str(tmp_path), "runs")
    for seed in AN.SEEDS:
        build_cell(root, "b0", "fixed", 0.2, 0.2, seed, off=0.05,
                   scale=1.0, drift=0.01)
        build_cell(root, "b0", "evolving", 0.2, 0.2, seed, off=0.0,
                   scale=1.0)
    cells = {}
    for cond in AN.CONDS:
        for seed in AN.SEEDS:
            rd = os.path.join(root, AN.cell_tag("b0", cond, 0.2, 0.2, seed))
            _, late = AN.reduce_cell(AN.load(rd), MASK_A)
            late["gpu_arch"] = "H100"
            cells[("b0", cond, 0.2, 0.2, seed)] = late
    row = [r for r in AN.build_source_rows(cells, 0.002)
           if r["arm"] == "b0" and r["eps_ai"] == 0.2
           and r["eps_social"] == 0.2][0]
    # drift 0.01/round: the halves' round centres are 25.5 and 28, so the
    # half-to-half difference is 0.025 -- an off-by-one split would not
    # land on this number
    assert row["delta_mu_b_drift_mean"] == pytest.approx(0.025, abs=1e-5)
    assert row["delta_mu_b_drift_flag"] is True


# ============================================================== end to end
def test_full_grid_runs_clean(full_run):
    rc, out, summary = full_run
    assert rc == 0, "a complete, structurally sound grid must exit 0"
    assert summary["n_cells_located"] == 72 == summary["n_cells_expected"]
    assert summary["missing_tags"] == []
    assert summary["partial"] is False
    assert summary["unexpected_tags"] == []
    assert summary["late_window_op_raw_rounds"] == [25, 29]
    assert summary["null_probe_failures"] == 0
    assert summary["n_series_complete"] == summary["n_series"] == 12


def test_full_grid_writes_every_artifact(full_run):
    _, out, _ = full_run
    for name in ("section4_gate_per_round.csv", "section4_gate_cells.csv",
                 "section4_gate_source_effect.csv",
                 "section4_gate_dispersion.csv",
                 "section4_gate_null_probe.csv",
                 "section4_gate_coverage.csv",
                 "section4_gate_captions.txt",
                 "section4_gate_source_effect.pdf",
                 "section4_gate_source_effect.png",
                 "section4_gate_dispersion.pdf",
                 "section4_gate_dispersion.png"):
        p = os.path.join(out, name)
        assert os.path.exists(p) and os.path.getsize(p) > 0, name
    assert not os.path.exists(os.path.join(out,
                                           "SUSPECT_NULL_VIOLATION.txt"))


def test_per_round_csv_is_one_row_per_cell_round(full_run):
    _, out, _ = full_run
    rows = read_csv(os.path.join(out, "section4_gate_per_round.csv"))
    assert len(rows) == 72 * T
    keys = {(r["arm"], r["cond"], r["eps_ai"], r["eps_social"], r["seed"],
             r["round"]) for r in rows}
    assert len(keys) == len(rows)
    for col in ("pop_mean", "pop_sd", "a_mean", "a_sd", "b_mean", "b_sd",
                "w1_twin_pop", "served_mean", "run_tag", "round_1based"):
        assert col in rows[0], col


def test_source_effect_csv_matches_the_fixture_by_construction(full_run):
    _, out, _ = full_run
    rows = read_csv(os.path.join(out, "section4_gate_source_effect.csv"))
    assert len(rows) == 12
    for r in rows:
        arm, ea, es = r["arm"], float(r["eps_ai"]), float(r["eps_social"])
        want = [_off(arm, "fixed", ea, es, s) - _off(arm, "evolving", ea,
                                                     es, s)
                for s in AN.SEEDS]
        m, sd, lo, hi = AN.tci3(want)
        assert float(r["delta_mu_b_mean"]) == pytest.approx(m, abs=1e-6)
        assert float(r["delta_mu_b_ci_lo"]) == pytest.approx(lo, abs=1e-6)
        assert float(r["delta_mu_b_ci_hi"]) == pytest.approx(hi, abs=1e-6)
        assert r["status"] == "complete"
        assert r["delta_mu_b_drift_flag"] == "False"
        assert r["innate_pair_bit_identical"] == "True"
        assert r["n_seeds_hardware_matched"] == "3"


def test_dispersion_csv_matches_the_fixture_by_construction(full_run):
    _, out, _ = full_run
    rows = read_csv(os.path.join(out, "section4_gate_dispersion.csv"))
    assert len(rows) == 12
    for r in rows:
        arm, ea, es = r["arm"], float(r["eps_ai"]), float(r["eps_social"])
        ratios = [_scale(arm, "fixed", ea, es, s)
                  / _scale(arm, "evolving", ea, es, s) for s in AN.SEEDS]
        m, sd, lo, hi = AN.tci3(ratios)
        assert float(r["sd_ratio_b_mean"]) == pytest.approx(m, rel=1e-5)
        assert float(r["sd_ratio_b_ci_lo"]) == pytest.approx(lo, rel=1e-4)
        # the inherited twin-referenced ratio, per condition
        assert float(r["sd_ratio_twin_b_fixed_mean"]) == pytest.approx(
            sum(_scale(arm, "fixed", ea, es, s) for s in AN.SEEDS) / 3,
            rel=1e-5)


def test_structural_null_probe_passes_on_a_clean_wave(full_run):
    _, out, _ = full_run
    rows = read_csv(os.path.join(out, "section4_gate_null_probe.csv"))
    assert len(rows) == len(AN.EAS) * len(AN.SEEDS) == 6
    for r in rows:
        assert r["verdict"] == "PASS", r
        assert r["hardware_matched"] == "True"
        assert float(r["tol"]) == AN.NULL_TOL      # bit-exact demanded
        assert abs(float(r["delta_mu_b"])) <= AN.NULL_TOL


def test_captions_are_printed_and_carry_the_key_facts(grid_root, tmp_path,
                                                      capsys):
    out = os.path.join(str(tmp_path), "out")
    rc = AN.main(["--run-root", grid_root, "--out-dir", out, "--no-figs"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert printed.count("CAPTION --") == 2
    for phrase in ("post-peer", "95% Student-t", "anchored",
                   "ANALYSIS MASK", "Exploratory"):
        assert phrase in printed, phrase
    text = open(os.path.join(out, "section4_gate_captions.txt")).read()
    assert "section4_gate_source_effect" in text
    assert "section4_gate_dispersion" in text
    # --no-figs really means no figures
    assert not os.path.exists(
        os.path.join(out, "section4_gate_source_effect.pdf"))


# ========================================================= partial coverage
def test_partial_coverage_is_labelled_and_never_averaged(grid_root,
                                                         tmp_path, capsys):
    root = copy_grid(grid_root, tmp_path)
    gone = [AN.cell_tag("b0", "fixed", 0.2, 1.0, 43),
            AN.cell_tag("d8", "evolving", 1.0, 0.2, 0)]
    for tag in gone:
        shutil.rmtree(os.path.join(root, tag))
    out = os.path.join(str(tmp_path), "out")
    js = os.path.join(str(tmp_path), "s.json")
    rc = AN.main(["--run-root", root, "--out-dir", out, "--no-figs",
                  "--json", js])
    assert rc == 2, "partial coverage must be signalled, not swallowed"
    printed = capsys.readouterr().out
    for tag in gone:
        assert f"MISSING {tag}" in printed
    assert "PARTIAL COVERAGE" in printed

    summary = json.load(open(js))
    assert summary["partial"] is True
    assert sorted(summary["missing_tags"]) == sorted(gone)
    assert summary["n_cells_located"] == 70

    cov = read_csv(os.path.join(out, "section4_gate_coverage.csv"))
    assert len(cov) == 72
    assert {r["run_tag"] for r in cov if r["present"] == "False"} == set(gone)

    src = read_csv(os.path.join(out, "section4_gate_source_effect.csv"))
    hurt = {("b0", 0.2, 1.0), ("d8", 1.0, 0.2)}
    for r in src:
        key = (r["arm"], float(r["eps_ai"]), float(r["eps_social"]))
        if key in hurt:
            assert r["status"] == "incomplete"
            assert r["n_seeds_paired"] == "2"
            assert r["delta_mu_b_mean"] == "NA"
            assert r["delta_mu_b_ci_lo"] == "NA"
            assert r["delta_mu_b_ci_excludes_zero"] == "NA"
        else:
            assert r["status"] == "complete"
            assert r["delta_mu_b_mean"] != "NA"
    disp = read_csv(os.path.join(out, "section4_gate_dispersion.csv"))
    for r in disp:
        key = (r["arm"], float(r["eps_ai"]), float(r["eps_social"]))
        if key in hurt:
            assert r["sd_ratio_b_mean"] == "NA"


def test_partial_series_are_not_plotted(grid_root, tmp_path):
    root = copy_grid(grid_root, tmp_path)
    shutil.rmtree(os.path.join(root, AN.cell_tag("b0", "fixed", 0.2, 1.0,
                                                 43)))
    out = os.path.join(str(tmp_path), "out")
    assert AN.main(["--run-root", root, "--out-dir", out]) == 2
    rows = read_csv(os.path.join(out, "section4_gate_source_effect.csv"))
    hurt = [r for r in rows if r["arm"] == "b0"
            and float(r["eps_ai"]) == 0.2 and float(r["eps_social"]) == 1.0]
    xs, ys, lo, hi, fl = AN._series(
        [{"arm": r["arm"], "eps_ai": float(r["eps_ai"]),
          "eps_social": float(r["eps_social"]), "status": r["status"],
          "delta_mu_b_mean": (None if r["delta_mu_b_mean"] == "NA"
                              else float(r["delta_mu_b_mean"]))}
         for r in hurt], "b0", 0.2, "delta_mu_b")
    assert xs == [] and ys == []
    assert os.path.exists(os.path.join(out,
                                       "section4_gate_source_effect.pdf"))


def test_empty_run_root_is_a_hard_fail(tmp_path):
    out = os.path.join(str(tmp_path), "out")
    assert AN.main(["--run-root", str(tmp_path / "nothing"),
                    "--out-dir", out, "--no-figs"]) == 1
    assert not os.path.exists(os.path.join(
        out, "section4_gate_source_effect.csv"))


# ===================================================== structural failures
def _expect_fatal(root, tmp_path, name):
    out = os.path.join(str(tmp_path), "out_" + name)
    rc = AN.main(["--run-root", root, "--out-dir", out, "--no-figs"])
    assert rc == 1, f"{name} must be a hard failure"
    assert not os.path.exists(os.path.join(
        out, "section4_gate_source_effect.csv")), \
        f"{name} must write NO output"


def test_a_perturbed_innate_vector_is_fatal(grid_root, tmp_path):
    root = copy_grid(grid_root, tmp_path)
    tag = AN.cell_tag("b0", "evolving", 1.0, 0.2, 42)
    p = os.path.join(root, tag, "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    d["innate"] = d["innate"].clone()
    d["innate"][3] += 1e-6
    torch.save(d, p)
    _expect_fatal(root, tmp_path, "innate")


def test_a_wrong_clamp_mask_is_fatal(grid_root, tmp_path):
    root = copy_grid(grid_root, tmp_path)
    tag = AN.cell_tag("d8", "fixed", 0.2, 1.0, 0)
    p = os.path.join(root, tag, "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    m = d["innate_clamp_mask"].clone()
    idx = int(torch.nonzero(m)[0])
    m[idx] = False
    m[int(torch.nonzero(~m)[-1])] = True
    d["innate_clamp_mask"] = m
    torch.save(d, p)
    _expect_fatal(root, tmp_path, "mask")


def test_an_evolving_run_carrying_a_clamp_mask_is_fatal(grid_root,
                                                        tmp_path):
    root = copy_grid(grid_root, tmp_path)
    tag = AN.cell_tag("b0", "evolving", 0.2, 0.0, 0)
    p = os.path.join(root, tag, "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    d["innate_clamp_mask"] = AN.cohort_a_mask(d["innate"])
    torch.save(d, p)
    _expect_fatal(root, tmp_path, "evomask")


def test_a_moving_cohort_a_in_a_fixed_run_is_fatal(grid_root, tmp_path):
    root = copy_grid(grid_root, tmp_path)
    tag = AN.cell_tag("b0", "fixed", 1.0, 1.0, 43)
    p = os.path.join(root, tag, "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    op = d["op_raw"].clone()
    a = int(torch.nonzero(d["innate_clamp_mask"])[0])
    op[7, a] += 0.01
    d["op_raw"] = op
    torch.save(d, p)
    _expect_fatal(root, tmp_path, "amoved")


def test_the_v1_gate_marker_is_fatal(grid_root, tmp_path):
    """This wave IS the corrected gate; a v1 run in the grid would make the
    comparison meaningless while still producing beautiful numbers."""
    root = copy_grid(grid_root, tmp_path)
    tag = AN.cell_tag("d8", "evolving", 1.0, 1.0, 42)
    p = os.path.join(root, tag, "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    d["config"]["population_update"] = "nested_ai_then_social_v1"
    torch.save(d, p)
    _expect_fatal(root, tmp_path, "v1gate")


def test_a_config_that_disagrees_with_the_tag_is_fatal(grid_root, tmp_path):
    root = copy_grid(grid_root, tmp_path)
    tag = AN.cell_tag("b0", "fixed", 0.2, 0.2, 0)
    p = os.path.join(root, tag, "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    d["config"]["eps_ai"] = 1.0                 # tag says 0.2
    torch.save(d, p)
    _expect_fatal(root, tmp_path, "cfgtag")


def test_a_short_run_is_fatal(grid_root, tmp_path):
    root = copy_grid(grid_root, tmp_path)
    tag = AN.cell_tag("b0", "evolving", 0.2, 0.2, 42)
    p = os.path.join(root, tag, "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    for k in ("op_raw", "twin_raw", "pred_raw"):
        d[k] = d[k][:20]
    torch.save(d, p)
    _expect_fatal(root, tmp_path, "short")


def test_nonzero_homophily_is_fatal(grid_root, tmp_path):
    root = copy_grid(grid_root, tmp_path)
    tag = AN.cell_tag("d8", "fixed", 1.0, 0.2, 43)
    p = os.path.join(root, tag, "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    d["config"]["gamma_bias"] = 0.3
    torch.save(d, p)
    _expect_fatal(root, tmp_path, "gamma")


def test_structural_null_violation_marks_results_suspect(grid_root,
                                                         tmp_path):
    """d8 at eps_social=0: frozen weights, own-history prompts, no peer
    step -- no cohort-A opinion can reach a cohort-B prompt, so a nonzero
    cohort-B source effect there means the wave is not what it says."""
    root = copy_grid(grid_root, tmp_path)
    tag = AN.cell_tag("d8", "fixed", 0.2, 0.0, 0)
    p = os.path.join(root, tag, "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    op = d["op_raw"].clone()
    op[:, ~AN.cohort_a_mask(d["innate"])] += 0.05
    d["op_raw"] = op
    torch.save(d, p)
    out = os.path.join(str(tmp_path), "out")
    rc = AN.main(["--run-root", root, "--out-dir", out, "--no-figs"])
    assert rc == 3
    assert os.path.exists(os.path.join(out, "SUSPECT_NULL_VIOLATION.txt"))
    rows = read_csv(os.path.join(out, "section4_gate_null_probe.csv"))
    fails = [r for r in rows if r["verdict"] == "FAIL"]
    assert len(fails) == 1
    assert float(fails[0]["eps_ai"]) == 0.2 and fails[0]["seed"] == "0"


# ============================================================== guardrails
def test_out_dir_under_paper_is_refused(grid_root, tmp_path):
    bad = os.path.join(str(tmp_path), "paper", "figures", "s4")
    with pytest.raises(SystemExit) as e:
        AN.main(["--run-root", grid_root, "--out-dir", bad, "--no-figs"])
    assert e.value.code == 1
    assert not os.path.exists(bad)


def test_default_out_dir_is_runs_adjacent_and_not_under_paper():
    d = AN.default_out_dir("/home/gsmithline/perfsim/runs/pokec_gated_lm")
    assert d == "/home/gsmithline/perfsim/runs/analysis/section4_gate_anch2"
    assert "paper" not in d.split(os.sep)


def test_figures_carry_no_title_text():
    """PROJECT CONVENTION: paper figures carry NO title text; the narrative
    lives in the printed caption block."""
    src = open(ANALYZER).read()
    assert "set_title(" not in src
    assert "suptitle(" not in src
    assert "no set_title" in src        # and the convention is stated
    assert "caption_source" in src and "caption_dispersion" in src


def test_threads_are_pinned_before_torch_is_imported():
    src = open(ANALYZER).read()
    i_omp = src.index('os.environ["OMP_NUM_THREADS"]')
    i_mkl = src.index('os.environ["MKL_NUM_THREADS"]')
    i_torch = src.index("\nimport torch")
    assert i_omp < i_torch and i_mkl < i_torch
    assert "matplotlib.use(\"Agg\")" in src
    assert "torch.set_num_threads(1)" in src


def test_header_states_what_was_inherited():
    doc = AN.__doc__ or ""
    assert "analyze_bottom20_section4_3seed.py" in doc
    assert "INHERITED" in doc
    assert "range(25, 30)" in doc
    assert "4.302652729911275" in doc
    assert AN.POP_UPDATE_V2 in doc


# ======================================================================
# FIGURE-6 MODE (--wave section4_gate_anch2_fig6, alias fig6)
# ======================================================================
# A synthetic 192-cell grid: runs for the 144 gpu + 2 witness cells,
# NO run for the 46 twin cells (they are derived from twin_raw), one
# drifting pair, served maps with 1-3 distinct values, and a passing gate
# verdict.  The whole thing is pure torch in tmp_path.
FIG6 = AN.GRID_FIG6
PLOT = _load("_plot_s4fig6", os.path.join(PIPE, "plot_section4_fig6.py"))


def _twin_off(cond, es, seed):
    """Cohort-B offset of the matched no-AI twin at (cond, es, seed): a
    function of (cond, eps_social, seed) ONLY -- the mirrored-RNG twin is
    the same for every arm and every eps_AI -- and 0 at es=0 (no peer
    step: the twin sits at innate)."""
    if es == 0.0:
        return 0.0
    return (0.006 * (1.0 + es) * (1.0 if cond == "evolving" else 0.6)
            + 0.0004 * AN.SEEDS.index(seed))


def _served_values(arm, ea):
    """Late-window served map: d8 serves ONE value, b0 at ea=.1 THREE
    (equal shares), b0 elsewhere TWO (0.65 holds two thirds)."""
    if arm == "d8":
        return [0.6]
    if ea == 0.1:
        return [0.25, 0.45, 0.65]
    return [0.25, 0.65, 0.65]


# the series with a KNOWN cohort-B gap: evolving sits 0.030/0.031/0.032
# above fixed at seeds 0/42/43, so T_a = +gap by construction
GAP_SERIES = ("b0", 0.3, 0.1)
GAP = {0: 0.030, 42: 0.031, 43: 0.032}
# ... and d8 at the same (ea, es) gets a smaller known gap, so the paired
# method gap G = T_a(b0) - T_a(d8) is +0.020/+0.021/+0.022 by construction
ICL_GAP = {0: 0.010, 42: 0.010, 43: 0.010}
G_KNOWN = {s: GAP[s] - ICL_GAP[s] for s in GAP}
# the drifting pair (fixed member drifts 0.01/round in the late window)
DRIFT_PAIR = ("b0", 1.0, 0.3, 42)


def build_fig6_cell(root, arm, cond, ea, es, seed, kind, rounds=T,
                    tag_rounds=None, drift=0.0, off=None, scale=None,
                    cycle_amp=0.0):
    """One fig6 run: a build_cell whose twin_raw is the (cond, es, seed)
    twin (identical across arms and eps_AI), whose pred_raw carries the
    cell's served map, and -- for a witness -- whose op_raw IS twin_raw."""
    innate = INNATE
    mask = AN.cohort_a_mask(innate)
    b = ~mask
    toff = _twin_off(cond, es, seed)
    vals = _served_values(arm, ea)
    n = int(innate.numel())

    def post(d):
        tw = []
        for t in range(rounds):
            x = innate.clone()
            x[b] = innate[b] + toff
            if cond == "evolving":
                x[mask] = innate[mask] + 0.01     # the responsive twin moves
            tw.append(x)
        d["twin_raw"] = torch.stack(tw)
        pred = torch.stack([torch.tensor([vals[(i + t) % len(vals)]
                                          for i in range(n)])
                            for t in range(rounds)])
        d["pred_raw"] = pred
        if kind == "witness":
            d["op_raw"] = d["twin_raw"].clone()
        d["config"]["n_rounds"] = rounds

    tag = AN.cell_tag(arm, cond, ea, es, seed, rounds=tag_rounds)
    return build_cell(root, arm, cond, ea, es, seed, rounds=rounds, tag=tag,
                      post=post, drift=drift, off=off, scale=scale,
                      cycle_amp=cycle_amp)


def build_fig6_grid(root):
    for (arm, cond, ea, es, seed, kind) in FIG6.cells:
        if kind == "twin":
            continue
        kw = {}
        if (ea, es) == GAP_SERIES[1:]:
            gap = GAP if arm == "b0" else ICL_GAP
            kw["scale"] = 1.0
            kw["off"] = 0.10 if cond == "fixed" else 0.10 + gap[seed]
        if (arm, ea, es, seed) == DRIFT_PAIR and cond == "fixed":
            kw["drift"] = 0.01
        build_fig6_cell(root, arm, cond, ea, es, seed, kind, **kw)
    return root


def _gate_json(path, ok=True):
    with open(path, "w") as fh:
        json.dump({"wave": "section4_gate_anch2", "pass": ok,
                   "n_cells_present": 146, "n_cells_total": 146,
                   "n_cells_failed": 0 if ok else 3}, fh)
    return path


@pytest.fixture(scope="module")
def fig6_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs_s4fig6")
    build_fig6_grid(root)
    return str(root)


@pytest.fixture(scope="module")
def fig6_gate(tmp_path_factory):
    return _gate_json(str(tmp_path_factory.mktemp("gate") / "gate.json"))


@pytest.fixture(scope="module")
def fig6_run(fig6_root, fig6_gate, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("out_s4fig6"))
    rc = AN.main(["--wave", "fig6", "--run-root", fig6_root,
                  "--out-dir", out, "--gate-json", fig6_gate])
    with open(os.path.join(out, "section4_fig6_summary.json")) as fh:
        summary = json.load(fh)
    return rc, out, summary


def _fig6_main(root, out, gate, extra=()):
    return AN.main(["--wave", "fig6", "--run-root", root, "--out-dir", out,
                    "--no-figs"] + (["--gate-json", gate] if gate else [])
                   + list(extra))


def _expect_fig6_fatal(root, tmp_path, name, gate, extra=()):
    out = os.path.join(str(tmp_path), "out_" + name)
    rc = _fig6_main(root, out, gate, extra)
    assert rc == 1, f"{name} must be a hard failure"
    assert not os.path.exists(out) or not os.listdir(out), \
        f"{name} must write NOTHING (partial output is not allowed)"


def _row(rows, arm, ea, es):
    return [r for r in rows if r["arm"] == arm
            and float(r["eps_ai"]) == ea and float(r["eps_social"]) == es][0]


# --------------------------------------------------------------- the grid
def test_fig6_grid_is_read_from_the_generator():
    gen = _load("_gen_for_tests", os.path.join(REPO, "experiments", "condor",
                                               "gen_pofd_sweep.py"))
    assert FIG6.cells == gen.s4g2_cells()
    assert FIG6.n_cells == 192
    assert FIG6.n_kind == {"gpu": 144, "witness": 4, "twin": 44}
    assert gen.S4G2_EA0_WITNESS == [("b0", "fixed", 0.3, 0),
                                    ("b0", "evolving", 0.3, 0),
                                    ("d8", "fixed", 0.3, 0),
                                    ("d8", "evolving", 0.3, 0)]
    assert FIG6.gates == gen.S4G2_GATES == [0.0, 0.1, 0.3, 1.0]
    assert FIG6.ess == gen.S4G2_ESS == [0.0, 0.1, 0.3, 1.0]
    assert FIG6.seeds == gen.S4G2_SEEDS == AN.SEEDS
    assert FIG6.ext_rounds_ok == (60, 100)
    # the original wave's grid is the generator's too, and unchanged
    assert AN.GRID_V1.gates == gen.S4G_GATES == [0.2, 1.0]
    assert AN.GRID_V1.ess == gen.S4G_ESS == [0.0, 0.2, 1.0]
    assert AN.EAS is AN.GRID_V1.gates and AN.ESS is AN.GRID_V1.ess
    assert AN.N_CELLS == 72
    assert AN.WAVE_ALIASES["v1"] == AN.KEY == "section4_gate_anch2"
    assert AN.WAVE_ALIASES["fig6"] == AN.KEY_FIG6 == \
        "section4_gate_anch2_fig6"
    # tags come from the generator's s4g_tag, including the horizon suffix
    assert AN.cell_tag("d8", "fixed", 0.1, 0.3, 42, rounds=100) == \
        gen.s4g_tag("d8", "fixed", 0.1, 0.3, 42, rounds=100)
    assert AN.cell_tag("d8", "fixed", 0.1, 0.3, 42, rounds=100).endswith(
        "_es0p3_s42_r100")


def test_late_window_is_rounds_26_to_30_and_the_final_five_of_an_extension():
    assert AN.late_window(30) == AN.LATE_IDX == list(AN.LATE) == \
        [25, 26, 27, 28, 29]
    assert [t + 1 for t in AN.late_window(30)] == [26, 27, 28, 29, 30]
    assert AN.late_window(60) == [55, 56, 57, 58, 59]
    assert AN.late_window(100) == [95, 96, 97, 98, 99]
    assert AN.half_split(AN.late_window(60)) == ([55, 56], [57, 58, 59])
    with pytest.raises(ValueError):
        AN.late_window(4)


def test_extension_tags_parse_and_are_in_grid_only_for_fig6():
    tag = AN.cell_tag("b0", "evolving", 0.1, 0.3, 0, rounds=60)
    p = AN.parse_tag(tag)                       # default: original wave
    assert p is not None and p["rounds"] == 60 and p["in_grid"] is False
    p6 = AN.parse_tag(tag, FIG6)
    assert p6["in_grid"] is True and p6["kind"] == "gpu"
    # a horizon the generator does not allow is never in grid
    bad = AN.cell_tag("b0", "evolving", 0.1, 0.3, 0, rounds=45)
    assert AN.parse_tag(bad, FIG6)["in_grid"] is False
    # an ea=0.2 cell is the ORIGINAL wave's, not fig6's
    v1 = AN.cell_tag("b0", "evolving", 0.2, 0.2, 0)
    assert AN.parse_tag(v1)["in_grid"] is True
    assert AN.parse_tag(v1, FIG6)["in_grid"] is False
    assert AN.parse_tag(AN.cell_tag("b0", "evolving", 0.0, 0.3, 0),
                        FIG6)["kind"] == "witness"


def test_fig6_default_out_dir_is_the_fig6_key():
    d = AN.default_out_dir("/home/gsmithline/perfsim/runs/pokec_gated_lm",
                           FIG6)
    assert d == ("/home/gsmithline/perfsim/runs/analysis/"
                 "section4_gate_anch2_fig6")


# ------------------------------------------------------------- end to end
def test_fig6_full_grid_runs_and_writes_everything(fig6_run):
    rc, out, summary = fig6_run
    # one pair drifts, so the honest verdict is "written, but unsettled"
    assert rc == 2
    for name in ("section4_fig6_per_round.csv", "section4_fig6_cells.csv",
                 "section4_fig6_source_effect.csv",
                 "section4_fig6_dispersion.csv",
                 "section4_fig6_null_probe.csv",
                 "section4_fig6_coverage.csv",
                 "section4_fig6_captions.txt",
                 "section4_fig6_summary.json",
                 "section4_fig6_extension_request.json",
                 "section4_fig6_source_effect.pdf",
                 "section4_fig6_source_effect.png",
                 "section4_fig6_dispersion.pdf"):
        p = os.path.join(out, name)
        assert os.path.exists(p) and os.path.getsize(p) > 0, name
    assert not os.path.exists(os.path.join(out, "SUSPECT_NULL_VIOLATION.txt"))
    assert summary["mode"] == "fig6"
    assert summary["key"] == "section4_gate_anch2_fig6"
    assert summary["n_cells_expected"] == 192 == summary["n_cells_located"]
    assert summary["n_cells_from_run"] == 148
    assert summary["n_cells_twin_derived"] == 44
    assert summary["missing_tags"] == [] and summary["partial"] is False
    assert summary["null_probe_failures"] == 0
    assert summary["n_series"] == 32
    assert summary["n_series_complete"] == 32
    assert summary["n_series_settled"] == 31
    assert summary["n_pairs_unsettled"] == 1
    assert summary["primary_column"] == "t_a_evolving_minus_fixed"
    assert "EVOLVING MINUS FIXED" in summary["t_a_sign"]
    assert "26-30" in summary["late_window_rule"]
    assert summary["gate_ok"] is True
    cov = read_csv(os.path.join(out, "section4_fig6_coverage.csv"))
    assert len(cov) == 192 and all(r["present"] == "True" for r in cov)
    assert sum(1 for r in cov if r["analysed_from"] == "twin_raw") == 44
    cells = read_csv(os.path.join(out, "section4_fig6_cells.csv"))
    assert len(cells) == 192
    assert {r["kind"] for r in cells} == {"gpu", "witness", "twin"}
    assert all(r["late_rounds_1based"] == "26-30" for r in cells)
    per_round = read_csv(os.path.join(out, "section4_fig6_per_round.csv"))
    assert len(per_round) == 192 * T


def test_fig6_t_a_sign_is_evolving_minus_fixed(fig6_run):
    """Fixed and evolving built with a KNOWN cohort-B gap: evolving sits
    +gap above fixed, so t_a_evolving_minus_fixed must be +gap, per seed
    and in the mean; delta_mu_b is the same number with the other sign."""
    _, out, summary = fig6_run
    rows = read_csv(os.path.join(out, "section4_fig6_source_effect.csv"))
    r = _row(rows, *GAP_SERIES)
    assert r["status"] == "complete" and r["settled"] == "True"
    for s, gap in GAP.items():
        assert float(r[f"t_a_evolving_minus_fixed_s{s}"]) == pytest.approx(
            +gap, abs=1e-6)
        assert float(r[f"delta_mu_b_s{s}"]) == pytest.approx(-gap, abs=1e-6)
    m, sd, lo, hi = AN.tci3([GAP[s] for s in AN.SEEDS])
    assert float(r["t_a_evolving_minus_fixed_mean"]) == pytest.approx(
        m, abs=1e-6)
    assert m > 0
    assert float(r["delta_mu_b_mean"]) == pytest.approx(-m, abs=1e-6)
    # T_a is the PRIMARY column in fig6 mode: it precedes delta_mu_b
    with open(os.path.join(out, "section4_fig6_source_effect.csv")) as fh:
        header = fh.readline().strip().split(",")
    assert header.index("t_a_evolving_minus_fixed_mean") < \
        header.index("delta_mu_b_mean")
    assert header.index("settled") < header.index(
        "t_a_evolving_minus_fixed_mean")
    # the JSON carries the same primary numbers
    js = [x for x in summary["source_effect"]
          if (x["arm"], x["eps_ai"], x["eps_social"]) == GAP_SERIES][0]
    assert js["t_a_evolving_minus_fixed_mean"] == pytest.approx(m, abs=1e-6)


def test_fig6_paired_ci_is_the_df2_t_interval_over_per_seed_differences(
        fig6_run):
    _, out, _ = fig6_run
    rows = read_csv(os.path.join(out, "section4_fig6_source_effect.csv"))
    n_checked = 0
    for r in rows:
        if r["status"] != "complete":
            continue
        per_seed = [float(r[f"t_a_evolving_minus_fixed_s{s}"])
                    for s in AN.SEEDS]
        m, sd, lo, hi = AN.tci3(per_seed)
        half = AN.T_CRIT_DF2 * sd / math.sqrt(3)
        assert float(r["t_a_evolving_minus_fixed_mean"]) == pytest.approx(m)
        assert float(r["t_a_evolving_minus_fixed_sd"]) == pytest.approx(
            sd, abs=1e-9)
        assert float(r["t_a_evolving_minus_fixed_ci_lo"]) == pytest.approx(
            m - half, abs=1e-9)
        assert float(r["t_a_evolving_minus_fixed_ci_hi"]) == pytest.approx(
            m + half, abs=1e-9)
        assert r["t_a_evolving_minus_fixed_ci_excludes_zero"] == str(
            AN.excludes(m - half, m + half, 0.0))
        # and the closed form of the fixture for the gpu cells
        arm, ea, es = r["arm"], float(r["eps_ai"]), float(r["eps_social"])
        if ea > 0 and (ea, es) != GAP_SERIES[1:] \
                and (arm, ea, es) != DRIFT_PAIR[:3]:
            want = [_off(arm, "evolving", ea, es, s)
                    - _off(arm, "fixed", ea, es, s) for s in AN.SEEDS]
            assert per_seed == pytest.approx(want, abs=1e-6)
        n_checked += 1
    assert n_checked == 32


def test_fig6_ea0_cells_are_twin_derived_and_the_method_collapses(fig6_run):
    """At eps_AI = 0 the gate is closed: the population IS twin_raw of
    the runs at the same (cond, es, seed), so b0 and d8 share one value
    and T_a(ea=0) is the pure peer-transmission effect."""
    _, out, _ = fig6_run
    cells = read_csv(os.path.join(out, "section4_fig6_cells.csv"))
    ea0 = [r for r in cells if float(r["eps_ai"]) == 0.0]
    assert len(ea0) == 48
    twins = [r for r in ea0 if r["kind"] == "twin"]
    wits = [r for r in ea0 if r["kind"] == "witness"]
    assert len(twins) == 44 and len(wits) == 4
    for r in twins:
        assert r["analysed_from"] == "twin_raw"
        assert r["derived_from"].startswith("pofds4g_")
        assert r["served_distinct"] == "n/a (gate closed)"
        assert r["served_top_share"] == "n/a (gate closed)"
        assert r["served_mean_late"] == "NA"
        want = MU0_B + _twin_off(r["cond"], float(r["eps_social"]),
                                 int(r["seed"]))
        assert float(r["mu_b_eq"]) == pytest.approx(want, abs=1e-6)
    assert sorted((r["arm"], r["cond"]) for r in wits) == [
        ("b0", "evolving"), ("b0", "fixed"), ("d8", "evolving"),
        ("d8", "fixed")]
    for r in wits:
        assert r["analysed_from"] == "op_raw"
        assert float(r["eps_social"]) == 0.3 and r["seed"] == "0"
        assert r["served_distinct"] != "n/a (gate closed)"
        # a FIXED witness went through the clamp checks and equals the
        # fixed twin (cohort A pinned in both)
        assert float(r["mu_b_eq"]) == pytest.approx(
            MU0_B + _twin_off(r["cond"], 0.3, 0), abs=1e-6)
        assert float(r["mu_a_eq"]) == pytest.approx(
            float(INNATE[MASK_A].mean()) + (0.0 if r["cond"] == "fixed"
                                            else 0.01), abs=1e-6)
    # method collapse: b0 and d8 identical at every (cond, es, seed)
    by = {}
    for r in ea0:
        by.setdefault((r["cond"], r["eps_social"], r["seed"]), {})[
            r["arm"]] = float(r["mu_b_eq"])
    assert len(by) == 24
    for k, v in by.items():
        assert v["b0"] == v["d8"], k
    # the witness pair (b0/d8, evolving, es=.3, seed 0) equals the twin
    assert by[("evolving", "0.3", "0")]["b0"] == pytest.approx(
        MU0_B + _twin_off("evolving", 0.3, 0), abs=1e-6)
    rows = read_csv(os.path.join(out, "section4_fig6_source_effect.csv"))
    for es in FIG6.ess:
        b0, d8 = _row(rows, "b0", 0.0, es), _row(rows, "d8", 0.0, es)
        assert b0["t_a_evolving_minus_fixed_mean"] == \
            d8["t_a_evolving_minus_fixed_mean"]
        for s in AN.SEEDS:
            want = _twin_off("evolving", es, s) - _twin_off("fixed", es, s)
            assert float(b0[f"t_a_evolving_minus_fixed_s{s}"]) == \
                pytest.approx(want, abs=1e-6)
        if es == 0.3:
            assert b0["kind_fixed"] == "witness|twin|twin"
            assert b0["kind_evolving"] == "witness|twin|twin"
        else:
            assert b0["kind_fixed"] == "twin|twin|twin"
    # the twin sha is one value per (cond, es, seed)
    _, _, summary = fig6_run
    assert all(len(v) == 1 for v in
               summary["twin_sha256_by_cond_es_seed"].values())
    assert len(summary["twin_sha256_by_cond_es_seed"]) == 24


def test_fig6_served_cardinality_columns(fig6_run, capsys):
    _, out, _ = fig6_run
    cells = read_csv(os.path.join(out, "section4_fig6_cells.csv"))
    n_pool = N * len(AN.LATE_IDX)                       # 300 pooled values
    for r in cells:
        if r["kind"] == "twin":
            continue
        arm, ea = r["arm"], float(r["eps_ai"])
        vals = _served_values(arm, ea)
        assert int(r["served_distinct"]) == len(set(vals)), r["run_tag"]
        assert int(r["served_n_finite"]) == n_pool
        top = max(vals.count(v) for v in set(vals)) / len(vals)
        assert float(r["served_top_share"]) == pytest.approx(top, abs=1e-9)
        if arm == "d8":
            assert int(r["served_distinct"]) == 1
            assert float(r["served_top_share"]) == 1.0
            assert float(r["served_top_value"]) == pytest.approx(0.6)
    rows = read_csv(os.path.join(out, "section4_fig6_source_effect.csv"))
    r = _row(rows, "d8", 1.0, 1.0)
    assert r["served_distinct_fixed"] == "1|1|1"
    assert r["served_distinct_evolving"] == "1|1|1"
    assert r["served_distinct_fixed_min"] == "1"
    assert r["served_top_share_fixed"] == "1.000|1.000|1.000"
    r = _row(rows, "b0", 0.1, 0.0)
    assert r["served_distinct_fixed"] == "3|3|3"
    assert float(r["served_top_share_evolving_max"]) == pytest.approx(1 / 3)
    r = _row(rows, "b0", 0.0, 0.1)
    assert r["served_distinct_fixed"] == "n/a (gate closed)|" * 2 + \
        "n/a (gate closed)"
    assert r["served_distinct_fixed_min"] == "NA"


def test_fig6_printed_report_puts_cardinality_next_to_t_a(fig6_root,
                                                          fig6_gate,
                                                          tmp_path, capsys):
    out = os.path.join(str(tmp_path), "out")
    rc = _fig6_main(fig6_root, out, fig6_gate)
    assert rc == 2
    printed = capsys.readouterr().out
    assert "EVOLVING MINUS FIXED" in printed
    assert "rounds 26-30" in printed
    assert "t_a_evolving_minus_fixed" in printed
    assert "FIG6 detail" in printed
    assert "distinct f|e" in printed
    assert "SERVED MAP QUANTIZED" in printed
    assert "UNSETTLED" in printed
    assert "extend_to_60" in printed
    assert "paired method gap G = T_a(SFT) - T_a(ICL)" in printed
    assert "positive G = SFT's source effect exceeds ICL's" in printed
    assert printed.count("CAPTION --") == 2
    assert "twin-derived" in printed.lower() or "twin" in printed
    # the caption names the unsettled series and the sign convention
    text = open(os.path.join(out, "section4_fig6_captions.txt")).read()
    assert "EVOLVING MINUS FIXED" in text
    assert "UNSETTLED series (1): b0 ea=1 es=0.3 [extend_to_60]" in text
    assert "SERVED-VALUE QUANTIZATION" in text
    assert "IDENTICAL for both methods" in text


# ------------------------------------------------------- settled / extension
def test_fig6_unsettled_pair_is_never_an_equilibrium_and_is_requested(
        fig6_run, monkeypatch):
    rc, out, summary = fig6_run
    rows = read_csv(os.path.join(out, "section4_fig6_source_effect.csv"))
    arm, ea, es, seed = DRIFT_PAIR
    r = _row(rows, arm, ea, es)
    assert r["settled"] == "False"
    assert r["outcome"] == "extend_to_60"
    assert r["n_pairs_settled"] == "2"
    assert r[f"pair_outcome_s{seed}"] == "extend_to_60"
    assert r[f"pair_settled_s{seed}"] == "False"
    other = [s for s in AN.SEEDS if s != seed]
    for s in other:
        assert r[f"pair_outcome_s{s}"] == "equilibrium"
        assert r[f"pair_settled_s{s}"] == "True"
    # drift 0.01/round: half centres 25.5 and 28 -> 0.025 on the fixed
    # member, ~0 on the evolving one
    assert float(r[f"mu_b_drift_fixed_s{seed}"]) == pytest.approx(0.025,
                                                                   abs=1e-5)
    assert abs(float(r[f"mu_b_drift_evolving_s{seed}"])) < 1e-6
    # the T_a is still reported (never hidden), just not as an equilibrium
    assert r["t_a_evolving_minus_fixed_mean"] != "NA"
    # every other series is settled
    assert sum(1 for x in rows if x["outcome"] == "equilibrium") == 31
    assert all(x["horizon"] == "30" for x in rows)

    # the manifest: BOTH members, matched, rounds 60, with a reason
    req = json.load(open(os.path.join(
        out, "section4_fig6_extension_request.json")))
    assert req["key"] == "section4_gate_anch2_fig6"
    assert req["n_cells"] == 2 == len(req["cells"])
    assert {c["cond"] for c in req["cells"]} == {"fixed", "evolving"}
    for c in req["cells"]:
        assert (c["arm"], c["eps_ai"], c["eps_social"], c["seed"],
                c["rounds"]) == (arm, ea, es, seed, 60)
        assert "fixed FAILED (a)(b)(c)" in c["reason"]
        assert "evolving passed (a)(b)(c)" in c["reason"]
        assert "extend_to_60" in c["reason"]
        assert f"tol {AN.DEFAULT_DRIFT_TOL:g}" in c["reason"]
        assert not c["reason"].startswith("cyclic")
        assert set(c) >= {"arm", "cond", "eps_ai", "eps_social", "seed",
                          "rounds", "reason"}
    assert req["twin_derived_unsettled"] == []
    assert req["not_extendable"] == []
    assert summary["extension_request"]["n_cells"] == 2
    assert summary["n_pairs_cyclic"] == 0
    # ... and the GENERATOR accepts it as a matched pair
    monkeypatch.setattr(AN.GEN, "S4G2_EXT_REQUEST_PATH", os.path.join(
        out, "section4_fig6_extension_request.json"))
    got = AN.GEN.s4g2_ext_requests()
    assert sorted(got) == sorted([(arm, "fixed", ea, es, seed, 60),
                                  (arm, "evolving", ea, es, seed, 60)])


def test_fig6_extension_artifacts_are_preferred_and_settle_the_pair(
        fig6_root, fig6_gate, tmp_path):
    root = copy_grid(fig6_root, tmp_path)
    arm, ea, es, seed = DRIFT_PAIR
    for cond in AN.CONDS:
        build_fig6_cell(root, arm, cond, ea, es, seed, "gpu", rounds=60,
                        tag_rounds=60)
    out = os.path.join(str(tmp_path), "out")
    rc = _fig6_main(root, out, fig6_gate)
    assert rc == 0, "the extended pair settles, so the whole grid is settled"
    rows = read_csv(os.path.join(out, "section4_fig6_source_effect.csv"))
    r = _row(rows, arm, ea, es)
    assert r["outcome"] == "equilibrium" and r["settled"] == "True"
    assert r[f"pair_horizon_s{seed}"] == "60"
    assert r["horizon"] == "30|60"
    cells = read_csv(os.path.join(out, "section4_fig6_cells.csv"))
    ext = [c for c in cells if c["run_tag"].endswith("_r60")]
    assert len(ext) == 2
    for c in ext:
        assert c["horizon"] == "60" and c["n_rounds"] == "60"
        assert c["late_rounds_op_raw"] == "55-59"
        assert c["late_rounds_1based"] == "56-60"
    cov = read_csv(os.path.join(out, "section4_fig6_coverage.csv"))
    assert sum(1 for c in cov if c["horizon"] == "60") == 2
    req = json.load(open(os.path.join(
        out, "section4_fig6_extension_request.json")))
    assert req["cells"] == [] and req["n_cells"] == 0
    # the base artifacts still stand behind the twin checks: 24 twin keys
    summary = json.load(open(os.path.join(out,
                                          "section4_fig6_summary.json")))
    assert summary["n_cells_extended_horizon"] == 2
    assert summary["n_pairs_unsettled"] == 0


def test_fig6_a_drifting_r60_pair_asks_for_100(fig6_root, fig6_gate,
                                                tmp_path):
    root = copy_grid(fig6_root, tmp_path)
    arm, ea, es, seed = DRIFT_PAIR
    for cond in AN.CONDS:
        build_fig6_cell(root, arm, cond, ea, es, seed, "gpu", rounds=60,
                        tag_rounds=60, drift=0.01 if cond == "fixed" else 0)
    out = os.path.join(str(tmp_path), "out")
    assert _fig6_main(root, out, fig6_gate) == 2
    rows = read_csv(os.path.join(out, "section4_fig6_source_effect.csv"))
    assert _row(rows, arm, ea, es)["outcome"] == "extend_to_100"
    req = json.load(open(os.path.join(
        out, "section4_fig6_extension_request.json")))
    assert [c["rounds"] for c in req["cells"]] == [100, 100]


def test_fig6_an_unpaired_horizon_is_fatal(fig6_root, fig6_gate, tmp_path):
    """Only the fixed member has a _r60: the pair would compare a
    60-round equilibrium with a 30-round one."""
    root = copy_grid(fig6_root, tmp_path)
    arm, ea, es, seed = DRIFT_PAIR
    build_fig6_cell(root, arm, "fixed", ea, es, seed, "gpu", rounds=60,
                    tag_rounds=60)
    _expect_fig6_fatal(root, tmp_path, "unpaired_horizon", fig6_gate)


def test_fig6_a_lying_extension_tag_is_fatal(fig6_root, fig6_gate, tmp_path):
    """A _r60 tag whose artifact holds 30 rounds."""
    root = copy_grid(fig6_root, tmp_path)
    arm, ea, es, seed = DRIFT_PAIR
    for cond in AN.CONDS:
        build_fig6_cell(root, arm, cond, ea, es, seed, "gpu", rounds=30,
                        tag_rounds=60)
    _expect_fig6_fatal(root, tmp_path, "lying_r60", fig6_gate)


# -------------------------------------------------------------- hard fails
def test_fig6_a_missing_gpu_cell_is_fatal_with_no_output(fig6_root,
                                                         fig6_gate,
                                                         tmp_path, capsys):
    root = copy_grid(fig6_root, tmp_path)
    tag = AN.cell_tag("d8", "fixed", 0.3, 1.0, 43)
    shutil.rmtree(os.path.join(root, tag))
    _expect_fig6_fatal(root, tmp_path, "missing", fig6_gate)
    err = capsys.readouterr()
    assert f"MISSING {tag}" in err.out
    assert "NO partial output" in err.err


def test_fig6_a_twin_cell_with_no_base_run_is_fatal(fig6_root, fig6_gate,
                                                    tmp_path, capsys):
    """Every base run at (fixed, es=1, seed 43) gone: the eps_AI=0 twin
    cells there cannot be derived."""
    root = copy_grid(fig6_root, tmp_path)
    for arm in AN.ARMS:
        for ea in (0.1, 0.3, 1.0):
            shutil.rmtree(os.path.join(root, AN.cell_tag(arm, "fixed", ea,
                                                         1.0, 43)))
    _expect_fig6_fatal(root, tmp_path, "notwin", fig6_gate)
    out = capsys.readouterr().out
    assert "twin-derived: no base run" in out


def test_fig6_requires_a_passing_gate(fig6_root, tmp_path):
    good = _gate_json(os.path.join(str(tmp_path), "good.json"), ok=True)
    bad = _gate_json(os.path.join(str(tmp_path), "bad.json"), ok=False)
    _expect_fig6_fatal(fig6_root, tmp_path, "nogate", None)
    _expect_fig6_fatal(fig6_root, tmp_path, "failgate", bad)
    _expect_fig6_fatal(fig6_root, tmp_path, "absentgate",
                       os.path.join(str(tmp_path), "nope.json"))
    _expect_fig6_fatal(fig6_root, tmp_path, "ungated", good,
                       extra=["--allow-ungated"])
    _expect_fig6_fatal(fig6_root, tmp_path, "ungated_bad", bad,
                       extra=["--allow-ungated"])


def test_fig6_witness_identity_is_enforced(fig6_root, fig6_gate, tmp_path,
                                           capsys):
    root = copy_grid(fig6_root, tmp_path)
    tag = AN.cell_tag("d8", "evolving", 0.0, 0.3, 0)
    p = os.path.join(root, tag, "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    op = d["op_raw"].clone()
    op[27, int(torch.nonzero(~AN.cohort_a_mask(d["innate"]))[0])] += 1e-4
    d["op_raw"] = op
    torch.save(d, p)
    _expect_fig6_fatal(root, tmp_path, "witness", fig6_gate)
    assert "WITNESS but op_raw != twin_raw" in capsys.readouterr().err


def test_fig6_twin_disagreement_is_fatal(fig6_root, fig6_gate, tmp_path,
                                         capsys):
    root = copy_grid(fig6_root, tmp_path)
    tag = AN.cell_tag("b0", "evolving", 0.3, 0.1, 42)
    p = os.path.join(root, tag, "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    tw = d["twin_raw"].clone()
    tw[3, 5] += 1e-6
    d["twin_raw"] = tw
    torch.save(d, p)
    _expect_fig6_fatal(root, tmp_path, "twin", fig6_gate)
    assert "twin_raw DISAGREES" in capsys.readouterr().err


def test_fig6_a_run_without_a_twin_is_fatal(fig6_root, fig6_gate, tmp_path):
    root = copy_grid(fig6_root, tmp_path)
    tag = AN.cell_tag("b0", "fixed", 1.0, 0.0, 0)
    p = os.path.join(root, tag, "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    d.pop("twin_raw")
    torch.save(d, p)
    _expect_fig6_fatal(root, tmp_path, "notwinraw", fig6_gate)


def test_fig6_inherited_structural_checks_still_apply(fig6_root, fig6_gate,
                                                      tmp_path):
    root = copy_grid(fig6_root, tmp_path)
    tag = AN.cell_tag("d8", "fixed", 0.1, 1.0, 42)
    p = os.path.join(root, tag, "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    d["config"]["population_update"] = "nested_ai_then_social_v1"
    torch.save(d, p)
    _expect_fig6_fatal(root, tmp_path, "v1gate6", fig6_gate)


def test_fig6_a_twin_cell_with_a_run_is_verified_as_a_witness(
        fig6_root, fig6_gate, tmp_path, capsys):
    """S4G2_RUN_ALL_EA0 flipped, or a stray ea=0 run: it must satisfy the
    identity and it changes no number."""
    root = copy_grid(fig6_root, tmp_path)
    build_fig6_cell(root, "b0", "fixed", 0.0, 0.1, 42, "witness")
    out = os.path.join(str(tmp_path), "out")
    assert _fig6_main(root, out, fig6_gate) == 2
    printed = capsys.readouterr().out
    assert "twin cell WITH a run" in printed
    cells = read_csv(os.path.join(out, "section4_fig6_cells.csv"))
    r = [c for c in cells if c["arm"] == "b0" and c["cond"] == "fixed"
         and float(c["eps_ai"]) == 0.0 and float(c["eps_social"]) == 0.1
         and c["seed"] == "42"][0]
    assert r["analysed_from"] == "op_raw" and r["kind"] == "twin"
    assert float(r["mu_b_eq"]) == pytest.approx(
        MU0_B + _twin_off("fixed", 0.1, 42), abs=1e-6)
    # a broken one is fatal
    root2 = copy_grid(fig6_root, tmp_path / "second")
    build_fig6_cell(root2, "b0", "fixed", 0.0, 0.1, 42, "gpu")   # op != twin
    _expect_fig6_fatal(root2, tmp_path, "badextra", fig6_gate)


# ----------------------------------------------------- the original wave
def test_original_wave_is_the_default_and_unchanged(grid_root, tmp_path):
    out = os.path.join(str(tmp_path), "out")
    rc = AN.main(["--run-root", grid_root, "--out-dir", out, "--no-figs"])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "section4_gate_source_effect.csv"))
    assert not os.path.exists(os.path.join(out, "section4_fig6_summary.json"))
    rows = read_csv(os.path.join(out, "section4_gate_source_effect.csv"))
    assert len(rows) == 12
    assert "settled" not in rows[0] and "outcome" not in rows[0]
    with open(os.path.join(out, "section4_gate_source_effect.csv")) as fh:
        header = fh.readline().strip().split(",")
    assert header.index("delta_mu_b_mean") < header.index(
        "t_a_evolving_minus_fixed_mean")
    # --wave v1 is the same thing; an optional gate JSON is honoured
    out2 = os.path.join(str(tmp_path), "out2")
    bad = _gate_json(os.path.join(str(tmp_path), "bad.json"), ok=False)
    assert AN.main(["--wave", "v1", "--run-root", grid_root, "--out-dir",
                    out2, "--no-figs", "--gate-json", bad]) == 1
    assert AN.main(["--wave", "v1", "--run-root", grid_root, "--out-dir",
                    out2, "--no-figs", "--gate-json", bad,
                    "--allow-ungated"]) == 0


def test_fig6_header_states_the_semantics():
    doc = AN.__doc__ or ""
    for phrase in ("FIGURE-6 MODE", "EVOLVING MINUS FIXED", "twin_raw",
                   "witness", "extend_to_60", "served_distinct",
                   "served_top_share", "range(25, 30)", "26-30",
                   "s4g2_cells", "analyze_bottom20_section4_3seed.py"):
        assert phrase in doc, phrase


def test_fig6_csv_feeds_the_plot_script(fig6_run):
    """The plot reads ONLY the analyzer's CSV/JSON: its reader must accept
    the real fig6 output as written."""
    _, out, _ = fig6_run
    rows = PLOT.read_rows(out)
    assert len(rows) == 32
    drawable, absent = PLOT.classify(rows)
    assert len(drawable) == 32 and absent == []
    assert sum(1 for r in drawable if not r["settled"]) == 1
    assert all(r["served_min"] is None for r in rows
               if r["eps_ai"] == 0.0) is False   # witness rows carry one
    assert PLOT.read_summary(out)["primary_column"] == \
        "t_a_evolving_minus_fixed"
    gap = PLOT.read_gap_rows(out)
    assert len(gap) == 16
    gd, ga = PLOT.classify(gap)
    assert len(gd) == 16 and ga == []
    assert [(r["eps_ai"], r["eps_social"]) for r in gd
            if not r["settled"]] == [DRIFT_PAIR[1:3]]


# ================================================= paired method gap G
def test_fig6_method_gap_sign_is_sft_minus_icl(fig6_run):
    """SFT and ICL built with a KNOWN T_a difference at one (ea, es):
    g_sft_minus_icl must be +diff, per seed and in the mean."""
    _, out, summary = fig6_run
    p = os.path.join(out, "section4_fig6_method_gap.csv")
    assert os.path.exists(p)
    rows = read_csv(p)
    assert len(rows) == 16
    ea, es = GAP_SERIES[1:]
    r = [x for x in rows if float(x["eps_ai"]) == ea
         and float(x["eps_social"]) == es][0]
    assert r["status"] == "complete" and r["settled"] == "True"
    for s, g in G_KNOWN.items():
        assert float(r[f"g_sft_minus_icl_s{s}"]) == pytest.approx(g, abs=1e-6)
    m, sd, lo, hi = AN.tci3([G_KNOWN[s] for s in AN.SEEDS])
    assert m > 0
    assert float(r["g_sft_minus_icl_mean"]) == pytest.approx(m, abs=1e-6)
    assert float(r["g_sft_minus_icl_ci_lo"]) == pytest.approx(lo, abs=1e-6)
    assert float(r["g_sft_minus_icl_ci_hi"]) == pytest.approx(hi, abs=1e-6)
    assert r["g_sft_minus_icl_excludes_zero"] == "True"
    assert float(r["t_a_sft_mean"]) - float(r["t_a_icl_mean"]) == \
        pytest.approx(m, abs=1e-6)
    assert "positive" in r["sign"] and "SFT" in r["sign"]
    # the JSON block and the sign statement
    js = [x for x in summary["method_gap"]
          if x["eps_ai"] == ea and x["eps_social"] == es][0]
    assert js["g_sft_minus_icl_mean"] == pytest.approx(m, abs=1e-6)
    assert "positive G = SFT's source effect exceeds ICL's" in summary["g_sign"]
    assert "positive G = SFT's source effect exceeds ICL's" in AN.__doc__


def test_fig6_method_gap_paired_ci_and_drift(fig6_run):
    _, out, _ = fig6_run
    rows = read_csv(os.path.join(out, "section4_fig6_method_gap.csv"))
    src_rows = read_csv(os.path.join(out, "section4_fig6_source_effect.csv"))
    for r in rows:
        assert r["status"] == "complete"
        ea, es = float(r["eps_ai"]), float(r["eps_social"])
        b0, d8 = _row(src_rows, "b0", ea, es), _row(src_rows, "d8", ea, es)
        per_seed = [float(r[f"g_sft_minus_icl_s{s}"]) for s in AN.SEEDS]
        want = [float(b0[f"t_a_evolving_minus_fixed_s{s}"])
                - float(d8[f"t_a_evolving_minus_fixed_s{s}"])
                for s in AN.SEEDS]
        assert per_seed == pytest.approx(want, abs=1e-12)
        m, sd, lo, hi = AN.tci3(per_seed)
        half = AN.T_CRIT_DF2 * sd / math.sqrt(3)
        assert float(r["g_sft_minus_icl_mean"]) == pytest.approx(m, abs=1e-9)
        assert float(r["g_sft_minus_icl_sd"]) == pytest.approx(sd, abs=1e-9)
        assert float(r["g_sft_minus_icl_ci_lo"]) == pytest.approx(m - half,
                                                                  abs=1e-9)
        assert float(r["g_sft_minus_icl_ci_hi"]) == pytest.approx(m + half,
                                                                  abs=1e-9)
        assert r["g_sft_minus_icl_excludes_zero"] == str(
            AN.excludes(m - half, m + half, 0.0))
        assert "g_sft_minus_icl_ci_excludes_zero" not in r
        assert "g_sft_minus_icl_drift_mean" in r
        # settled iff both arms' pairs are settled
        want_settled = b0["settled"] == "True" and d8["settled"] == "True"
        assert r["settled"] == str(want_settled)
    unsettled = [r for r in rows if r["settled"] == "False"]
    assert [(float(r["eps_ai"]), float(r["eps_social"])) for r in unsettled] \
        == [DRIFT_PAIR[1:3]]
    assert unsettled[0]["outcome"] == "extend_to_60"
    # the drifting SFT pair makes G's own drift visible at that (ea, es)
    assert abs(float(unsettled[0]["g_sft_minus_icl_drift_mean"])) > 0.002
    assert unsettled[0]["g_sft_minus_icl_drift_flag"] == "True"


def test_fig6_method_gap_is_identically_zero_at_eps_ai_zero(fig6_run):
    _, out, _ = fig6_run
    rows = read_csv(os.path.join(out, "section4_fig6_method_gap.csv"))
    z = [r for r in rows if float(r["eps_ai"]) == 0.0]
    assert len(z) == 4
    for r in z:
        for s in AN.SEEDS:
            assert float(r[f"g_sft_minus_icl_s{s}"]) == 0.0
        assert float(r["g_sft_minus_icl_mean"]) == 0.0
        assert float(r["g_sft_minus_icl_sd"]) == 0.0
        assert float(r["g_sft_minus_icl_ci_lo"]) == 0.0
        assert float(r["g_sft_minus_icl_ci_hi"]) == 0.0
        assert r["g_sft_minus_icl_excludes_zero"] == "False"
        assert r["settled"] == "True" and r["outcome"] == "equilibrium"


# =========================================== stronger convergence a/b/c
def test_settle_verdict_needs_all_three_tests():
    tol = 0.002
    ok = {"mu_b_drift": 0.001, "late10_drift": 0.001, "late5_range": 0.003,
          "cycle_alternation": 0.0}
    v = AN.settle_verdict(ok, tol)
    assert v["settled"] and v["settled_a"] and v["settled_b"] \
        and v["settled_c"] and not v["cyclic"]
    assert v["range_tol"] == 0.004
    for k, bad in (("mu_b_drift", 0.0025), ("late10_drift", 0.0025),
                   ("late5_range", 0.0045)):
        d = dict(ok)
        d[k] = bad
        v = AN.settle_verdict(d, tol)
        assert v["settled"] is False
        assert v["settled_" + {"mu_b_drift": "a", "late10_drift": "b",
                               "late5_range": "c"}[k]] is False
        assert not v["cyclic"]
    cyc = dict(ok)
    cyc["late5_range"] = 0.009
    cyc["cycle_alternation"] = 0.78
    assert AN.settle_verdict(cyc, tol)["cyclic"] is True
    cyc["cycle_alternation"] = 0.6
    assert AN.settle_verdict(cyc, tol)["cyclic"] is False
    # a SETTLED cell is never cyclic, whatever it alternation
    ok2 = dict(ok)
    ok2["cycle_alternation"] = 1.0
    assert AN.settle_verdict(ok2, tol)["cyclic"] is False
    missing = dict(ok)
    missing["late10_drift"] = None
    assert AN.settle_verdict(missing, tol)["settled_b"] is False


def test_reduce_cell_reports_late10_drift_range_and_alternation(tmp_path):
    rd = build_cell(tmp_path, "b0", "evolving", 1.0, 0.2, 0, off=0.0,
                    scale=1.0, cycle_amp=0.0045)
    _, late = AN.reduce_cell(AN.load(rd), MASK_A)
    A = 0.0045
    assert late["late10_rounds_1based"] == "21-25 vs 26-30"
    # rounds 25..29: -A,+A,-A,+A,-A -> h1 = 0, h2 = -A/3
    assert late["mu_b_drift"] == pytest.approx(-A / 3, abs=1e-6)
    # rounds 20..24: +A,-A,+A,-A,+A -> +A/5 ; final five -> -A/5
    assert late["late10_drift"] == pytest.approx(-2 * A / 5, abs=1e-6)
    assert late["late5_range"] == pytest.approx(2 * A, abs=1e-6)
    assert late["cycle_alternation"] == 1.0
    rd = build_cell(tmp_path, "b0", "evolving", 1.0, 0.2, 42, off=0.0,
                    scale=1.0, drift=0.0006)
    _, late = AN.reduce_cell(AN.load(rd), MASK_A)
    assert late["mu_b_drift"] == pytest.approx(0.0015, abs=1e-6)
    assert late["late10_drift"] == pytest.approx(0.003, abs=1e-6)
    assert late["late5_range"] == pytest.approx(0.0024, abs=1e-6)
    assert late["cycle_alternation"] == 0.0


def test_fig6_a_period_two_cycle_is_cyclic_not_an_equilibrium(
        fig6_root, fig6_gate, tmp_path, capsys):
    """Amplitude 0.0045: the final-5 half-split drift is A/3 = 0.0015 and
    the final-10 drift 2A/5 = 0.0018, BOTH inside tol -- the old rule
    would have called this settled.  The range 2A = 0.009 > 2*tol fails
    (c), the sign alternation is 100%, so the pair is CYCLIC and in the
    manifest with a 'cyclic:' reason."""
    root = copy_grid(fig6_root, tmp_path)
    arm, cond, ea, es, seed = "d8", "fixed", 0.1, 1.0, 43
    build_fig6_cell(root, arm, cond, ea, es, seed, "gpu", cycle_amp=0.0045)
    out = os.path.join(str(tmp_path), "out")
    assert _fig6_main(root, out, fig6_gate) == 2
    cells = read_csv(os.path.join(out, "section4_fig6_cells.csv"))
    c = [x for x in cells if x["arm"] == arm and x["cond"] == cond
         and float(x["eps_ai"]) == ea and float(x["eps_social"]) == es
         and x["seed"] == str(seed)][0]
    assert c["settled_a"] == "True" and c["settled_b"] == "True"
    assert c["settled_c"] == "False" and c["settled"] == "False"
    assert c["cyclic"] == "True"
    assert float(c["cycle_alternation"]) == 1.0
    assert float(c["late5_range"]) == pytest.approx(0.009, abs=1e-6)
    rows = read_csv(os.path.join(out, "section4_fig6_source_effect.csv"))
    r = _row(rows, arm, ea, es)
    assert r["outcome"] == "cyclic" and r["settled"] == "False"
    assert r[f"pair_outcome_s{seed}"] == "cyclic"
    assert r[f"cyclic_fixed_s{seed}"] == "True"
    assert r[f"settled_flags_fixed_s{seed}"] == "a:T b:T c:F"
    assert r[f"settled_flags_evolving_s{seed}"] == "a:T b:T c:T"
    req = json.load(open(os.path.join(
        out, "section4_fig6_extension_request.json")))
    cyc = [x for x in req["cells"] if x["arm"] == arm
           and x["eps_social"] == es and x["seed"] == seed]
    assert len(cyc) == 2 and {x["cond"] for x in cyc} == {"fixed", "evolving"}
    for x in cyc:
        assert x["rounds"] == 60
        assert x["reason"].startswith("cyclic: ")
        assert "fixed FAILED (c)" in x["reason"]
        assert "CYCLIC" in x["reason"]
        assert "evolving passed (a)(b)(c)" in x["reason"]
    assert req["n_cells"] == 4                 # the drifting pair + this one
    summary = json.load(open(os.path.join(out,
                                          "section4_fig6_summary.json")))
    assert summary["n_pairs_cyclic"] == 1 and summary["n_pairs_unsettled"] == 2
    printed = capsys.readouterr().out
    assert "1 cyclic" in printed
    gap = read_csv(os.path.join(out, "section4_fig6_method_gap.csv"))
    g = [x for x in gap if float(x["eps_ai"]) == ea
         and float(x["eps_social"]) == es][0]
    assert g["settled"] == "False" and g["outcome"] == "cyclic"


def test_fig6_final10_drift_catches_a_slow_trend_the_final5_misses(
        fig6_root, fig6_gate, tmp_path):
    """0.0006 / round: final-5 half-split 0.0015 (passes a), range 0.0024
    (passes c), but the final-10 half-split is 0.003 > tol -> (b) fails,
    outcome extend_to_60, not cyclic."""
    root = copy_grid(fig6_root, tmp_path)
    arm, cond, ea, es, seed = "b0", "evolving", 0.3, 1.0, 0
    build_fig6_cell(root, arm, cond, ea, es, seed, "gpu", drift=0.0006)
    out = os.path.join(str(tmp_path), "out")
    assert _fig6_main(root, out, fig6_gate) == 2
    cells = read_csv(os.path.join(out, "section4_fig6_cells.csv"))
    c = [x for x in cells if x["arm"] == arm and x["cond"] == cond
         and float(x["eps_ai"]) == ea and float(x["eps_social"]) == es
         and x["seed"] == str(seed)][0]
    assert c["settled_a"] == "True" and c["settled_c"] == "True"
    assert c["settled_b"] == "False" and c["settled"] == "False"
    assert c["cyclic"] == "False"
    assert float(c["late10_drift"]) == pytest.approx(0.003, abs=1e-6)
    rows = read_csv(os.path.join(out, "section4_fig6_source_effect.csv"))
    r = _row(rows, arm, ea, es)
    assert r["outcome"] == "extend_to_60"
    assert r[f"settled_flags_evolving_s{seed}"] == "a:T b:F c:T"
    req = json.load(open(os.path.join(
        out, "section4_fig6_extension_request.json")))
    mine = [x for x in req["cells"] if x["arm"] == arm and x["eps_ai"] == ea
            and x["eps_social"] == es and x["seed"] == seed]
    assert len(mine) == 2
    assert "evolving FAILED (b)" in mine[0]["reason"]
    assert "fixed passed (a)(b)(c)" in mine[0]["reason"]
    assert not mine[0]["reason"].startswith("cyclic")


def test_gate_is_written_as_strict():
    src = open(ANALYZER).read()
    assert "|m - x'| <= eps_AI" not in src
    assert "|m - x'| < eps_AI" in src
