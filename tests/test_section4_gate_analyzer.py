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
               innate=None, drift=0.0):
    """Write ONE synthetic trajectory.pt with the real artifact schema.

    op_raw[t] is treated as the END-OF-ROUND POST-PEER state, exactly as
    the analyzer documents.  Cohort B is an affine image of innate, so
    every late-window statistic is known in closed form:
        mean_B(t) = MU0_B + off + drift*(t - 25)
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
        shift = off + drift * (t - AN.LATE_IDX[0])
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
