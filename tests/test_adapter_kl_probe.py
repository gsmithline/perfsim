"""Tests for the adapter KL / soft-decode probe (2026-08-21).

No model is ever loaded here: every quantity the probe reports is
computed by a pure function taking arrays, and those are driven directly
with synthetic distributions. The GPU-only parts are covered
structurally instead -- an AST check that the teacher-forced forward
passes position_ids (a left-padded forward silently uses the wrong RoPE
phase without them, which would corrupt every number in the probe while
looking perfectly well-formed).

The checker tests are SABOTAGE tests: each builds a probe directory that
is valid except for one defect and asserts the checker rejects it. A
gate nobody has watched fail is not a gate.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
PIPE = REPO / "experiments" / "scripts" / "cluster_pipelines"
CONDOR = REPO / "experiments" / "condor"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AKL = _load("probe_akl_t", PIPE / "probe_adapter_kl.py")
ANA = _load("analyze_akl_t", PIPE / "analyze_adapter_kl_probe.py")
CHK = _load("check_akl_t", PIPE / "check_adapter_kl_probe.py")
GEN = _load("gen_pofd_t", CONDOR / "gen_pofd_sweep.py")


# ---------------------------------------------------------------- support

def test_numeric_support_picks_digit_leading_tokens():
    vocab = ["0", " 1", "25", "hello", ".", "  7x", "", "a1", "\n3"]
    got = AKL.numeric_support_ids(vocab)
    assert got == [0, 1, 2, 5, 8]
    assert 3 not in got and 4 not in got and 7 not in got


def test_numeric_support_ignores_non_strings():
    assert AKL.numeric_support_ids(["1", None, 5, "2"]) == [0, 3]


# ------------------------------------------------------------- value maps

def _toy_decode(seqs):
    """token id -> character, joined. ids 0-9 are the digits."""
    tbl = {i: str(i) for i in range(10)}
    tbl.update({10: "0", 11: ".", 12: "x"})
    return ["".join(tbl.get(t, "?") for t in s) for s in seqs]


def _parse(text, default=0.5):
    import re
    m = re.search(r"\d+\.?\d*", text)
    return default if m is None else max(0.0, min(1.0, float(m.group())))


def test_value_map_substitutes_and_keeps_the_tail():
    # answer "0.25" as ids [10, 11, 2, 5]; substitute at position 2
    ans = [10, 11, 2, 5]
    vals = AKL.value_map_for(ans, 2, [2, 6, 9], _toy_decode, _parse)
    assert vals == pytest.approx([0.25, 0.65, 0.95])


def test_value_map_at_a_position_with_no_leverage():
    # the LAST digit of "0.25" moves the value only in the hundredths
    ans = [10, 11, 2, 5]
    vals = AKL.value_map_for(ans, 3, [0, 5, 9], _toy_decode, _parse)
    assert vals == pytest.approx([0.20, 0.25, 0.29])
    lo = AKL.value_map_for(ans, 2, [0, 5, 9], _toy_decode, _parse)
    assert (lo.max() - lo.min()) > (vals.max() - vals.min())


def test_value_map_does_not_mutate_the_answer():
    ans = [10, 11, 2, 5]
    AKL.value_map_for(ans, 2, [6], _toy_decode, _parse)
    assert ans == [10, 11, 2, 5]


# ------------------------------------------------------ soft value + tail

def test_soft_value_renormalizes_and_reports_the_tail():
    probs = np.zeros(10)
    probs[2] = 0.5
    probs[6] = 0.3
    probs[9] = 0.2          # 9 is OUTSIDE the support below
    sv, tail = AKL.soft_value(probs, [2, 6], [0.25, 0.65])
    assert tail == pytest.approx(0.2)
    # renormalized over {2, 6}: .5/.8 * .25 + .3/.8 * .65
    assert sv == pytest.approx((0.5 * 0.25 + 0.3 * 0.65) / 0.8)


def test_soft_value_equals_the_greedy_value_for_a_point_mass():
    probs = np.zeros(10)
    probs[6] = 1.0
    sv, tail = AKL.soft_value(probs, [2, 6], [0.25, 0.65])
    assert sv == pytest.approx(0.65)
    assert tail == pytest.approx(0.0)


def test_soft_value_moves_continuously_where_greedy_jumps():
    """The probe's whole reason to exist: as the base's near-tie tips, the
    ARGMAX value jumps 0.25 -> 0.65 while the soft value slides."""
    softs, greedys = [], []
    for p2 in (0.51, 0.50, 0.49):
        pr = np.zeros(10)
        pr[2], pr[6] = p2, 1.0 - p2
        softs.append(AKL.soft_value(pr, [2, 6], [0.25, 0.65])[0])
        greedys.append(0.25 if pr[2] > pr[6] else 0.65)
    assert greedys == [0.25, 0.65, 0.65]                  # discontinuous
    assert max(np.diff(softs)) < 0.02                      # continuous
    assert softs[0] < softs[1] < softs[2]


# ------------------------------------------------------------- leverage

def test_leverage_zero_when_every_candidate_parses_the_same():
    pr = np.array([0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0])
    assert AKL.leverage(pr, [2, 6], [0.4, 0.4]) == pytest.approx(0.0)


def test_leverage_is_the_value_sd_under_the_model():
    pr = np.zeros(10)
    pr[2], pr[6] = 0.5, 0.5
    got = AKL.leverage(pr, [2, 6], [0.25, 0.65])
    assert got == pytest.approx(0.2)                       # sd of a fair coin


def test_pick_tstar_takes_the_max_and_the_earliest_tie():
    assert AKL.pick_tstar([0.1, 0.9, 0.3]) == 1
    assert AKL.pick_tstar([0.5, 0.5, 0.1]) == 0
    with pytest.raises(ValueError):
        AKL.pick_tstar([])


# ------------------------------------------------------------------- KL

def test_kl_is_zero_against_itself_and_positive_otherwise():
    lp = np.log(np.array([[0.5, 0.3, 0.2]]))
    lq = np.log(np.array([[0.2, 0.3, 0.5]]))
    assert AKL.kl_rows(lp, lp)[0] == pytest.approx(0.0, abs=1e-12)
    assert AKL.kl_rows(lp, lq)[0] > 0.0


def test_kl_matches_a_hand_computation():
    # 2-D rows, as the probe always calls it: one row per answer position
    p = np.array([[0.5, 0.5]])
    q = np.array([[0.25, 0.75]])
    want = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert AKL.kl_rows(np.log(p), np.log(q))[0] == pytest.approx(want)


def test_kl_is_asymmetric():
    lp, lq = np.log([[0.9, 0.1]]), np.log([[0.5, 0.5]])
    assert AKL.kl_rows(lp, lq)[0] != pytest.approx(AKL.kl_rows(lq, lp)[0])


def test_kl_never_returns_negative_from_float_noise():
    lp = np.log(np.array([[0.3333333333333333, 0.6666666666666666]]))
    assert AKL.kl_rows(lp, lp)[0] >= 0.0


# ------------------------------------------------------------ small utils

def test_top2_margin():
    top1, marg = AKL.top2_margin([0.1, 0.55, 0.35])
    assert top1 == pytest.approx(0.55)
    assert marg == pytest.approx(0.20)


def test_strip_tail_cuts_at_the_first_stop_token():
    assert AKL.strip_tail([5, 6, 7, 99, 8], {99}) == [5, 6, 7]
    assert AKL.strip_tail([5, 6], {99}) == [5, 6]
    assert AKL.strip_tail([99, 5], {99}) == []


def test_position_ids_match_the_left_padded_generation_convention():
    attn = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    got = AKL._position_ids(attn)
    assert got[0].tolist() == [0, 0, 0, 1, 2]     # pads pinned at 0
    assert got[1].tolist() == [0, 1, 2, 3, 4]


# --------------------------------------------- tags come from ONE place

def test_dose_tags_read_from_the_on_disk_configs():
    tags = AKL.read_dose_tags(CONDOR)
    assert len(tags) == 15
    assert len(set(tags)) == 15
    assert all(t.startswith("pofdsftdose_qwen7b_u") for t in tags)


def test_analyzer_parses_every_generator_built_tag():
    """The 5em05/5em5 class of bug: the analyzer must read back exactly
    what the generator wrote, for every cell, with no second grammar."""
    cases = ([(u, GEN.SFTD_STD_LR, GEN.SFTD_STD_RANK) for u in GEN.SFTD_UPDATES]
             + [(GEN.SFTD_STD_U, lr, GEN.SFTD_STD_RANK) for lr in GEN.SFTD_LRS]
             + [(GEN.SFTD_STD_U, GEN.SFTD_STD_LR, r) for r in GEN.SFTD_RANKS])
    for u, lr, rank in cases:
        tag = GEN.sftd_tag(u, lr, rank)
        gu, glr, grank, fam = ANA.parse_tag(tag)
        assert (gu, glr, grank) == (u, lr, rank), tag
        assert fam in ("update", "lr", "rank", "endpoint")


def test_untok_lr_inverts_the_generator_token():
    for lr in ("1e-6", "3e-6", "1e-5", "3e-5", "5e-5", "1.25e-5"):
        assert ANA.untok_lr(GEN._lrtok(lr)) == lr


def test_family_assignment_over_the_real_tag_set():
    fams = {}
    for t in AKL.read_dose_tags(CONDOR):
        fams.setdefault(ANA.parse_tag(t)[3], []).append(t)
    assert len(fams["update"]) == 5      # 6 minus the shared endpoint
    assert len(fams["lr"]) == 4
    assert len(fams["rank"]) == 5
    assert len(fams["endpoint"]) == 1    # queued once, shared by all three


def test_parse_tag_rejects_a_foreign_tag():
    with pytest.raises(SystemExit):
        ANA.parse_tag("pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100")


# ------------------------------------- structural guards on the GPU path

def _probe_ast():
    return ast.parse((PIPE / "probe_adapter_kl.py").read_text())


def test_teacher_forced_forward_passes_position_ids():
    """Without position_ids a left-padded forward puts the RoPE phase in
    the wrong place for every padded row -- silently, and the probe's
    numbers would all be wrong while staying finite and ordered."""
    fn = next(n for n in ast.walk(_probe_ast())
              if isinstance(n, ast.FunctionDef) and n.name == "_span_logp")
    kwargs = [k.arg for n in ast.walk(fn) if isinstance(n, ast.Call)
              for k in n.keywords if k.arg]
    assert "position_ids" in kwargs
    assert "attention_mask" in kwargs


def test_base_reference_comes_from_disable_adapter_on_the_same_batch():
    """The base and adapter distributions must be scored on ONE set of
    input tensors; re-tokenising the base separately would let the
    padding width differ between the two sides of the KL."""
    fn = next(n for n in ast.walk(_probe_ast())
              if isinstance(n, ast.FunctionDef)
              and n.name == "dual_span_logprobs")
    src = ast.dump(fn)
    assert "disable_adapter" in src
    # _batch_tensors is called exactly once: both passes share its output
    calls = [n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert calls.count("_batch_tensors") == 1
    assert calls.count("_span_logp") == 2


def test_probe_compiles():
    for p in ("probe_adapter_kl.py", "check_adapter_kl_probe.py",
              "analyze_adapter_kl_probe.py"):
        compile((PIPE / p).read_text(), str(PIPE / p), "exec")


# ---------------------------------------------------- checker sabotage

N = AKL.N_AGENTS
SUPPORT = [2, 6]


def _write_probe(tmp, *, served=None, kl_scale=None, recheck_drift=0.0,
                 tail=0.0, n_lev_positions=1, topm_out=0.0,
                 mismatch_n=100, mismatch_margin=1e-3):
    """A structurally valid probe directory, defect injected by kwargs."""
    tags = AKL.read_dose_tags(CONDOR)
    rng = np.random.default_rng(0)
    soft_base = rng.uniform(0.2, 0.8, N)
    if served is None:
        served = rng.uniform(0.2, 0.8, N).astype(np.float32)
    # the checker reads max(leverage)/sum(leverage). n equal positions
    # therefore give a share of 1/n: n=1 is a clean probe (all the value
    # uncertainty at t*), larger n is the defect where a one-position
    # soft value hides structure.
    base = {
        "served": served,
        "soft_base": soft_base,
        "support": SUPPORT,
        "values_at_tstar": [np.array([0.25, 0.65])] * N,
        "tstar": np.zeros(N, dtype=np.int64),
        "leverage": [[1.0] * n_lev_positions] * N,
        "tail_base": np.full(N, tail),
        "topm_outside_support": np.full(N, topm_out),
        "base_top1": np.full(N, 0.5),
        "base_margin": np.full(N, 0.02),
        "ans_ids": [[10, 11, 2, 5]] * N,
    }
    tmp.mkdir(parents=True, exist_ok=True)
    torch.save(base, tmp / "base_probe.pt")
    for i, tag in enumerate(tags):
        scale = (i + 1.0) if kl_scale is None else kl_scale
        torch.save({
            "kl_fwd_sum": np.full(N, 0.01 * scale),
            "kl_rev_sum": np.full(N, 0.01 * scale),
            "kl_fwd_tstar": np.full(N, 0.005 * scale),
            "kl_rev_tstar": np.full(N, 0.005 * scale),
            "soft_base_recheck": soft_base + recheck_drift,
            "soft_adapter": soft_base + 0.01 * scale,
            "tail_adapter": np.full(N, tail),
            "greedy_tf": np.full(N, 0.65),
            "first_div": np.full(N, -1.0),
            "n_tok": np.full(N, 4.0),
        }, tmp / f"adapter_{tag}.pt")
    json.dump({"n_agents": N, "hash_gate": "enforced", "tags": tags,
               "base_top1_mean": 0.5, "base_margin_mean": 0.02,
               "tf_mismatch": {"n": mismatch_n, "frac_of_agents": 0.1,
                               "max_margin": mismatch_margin,
                               "median_margin": 1e-4, "positions": [2],
                               "threshold": AKL.MARGIN_STRUCTURAL}},
              open(tmp / "probe_manifest.json", "w"))
    return tags


@pytest.fixture
def pinned_hash(tmp_path, monkeypatch):
    """Pin CANON_SHA to whatever the fixture's served vector hashes to, so
    the clean case can pass without fabricating a real Qwen vector.
    Patched on CHK.AKL -- the module object the checker actually reads,
    not a second import of the same file."""
    served = np.random.default_rng(7).uniform(0.2, 0.8, N).astype(np.float32)
    monkeypatch.setattr(CHK.AKL, "CANON_SHA", AKL.sha_vec(served))
    return served


def test_clean_probe_passes(tmp_path, pinned_hash):
    d = tmp_path / "probe"
    _write_probe(d, served=pinned_hash)
    errs, notes = CHK.check(d, tmp_path / "no_runs")
    assert errs == [], errs
    assert any("leverage share" in n for n in notes)


def test_checker_rejects_a_non_canonical_base(tmp_path, pinned_hash):
    d = tmp_path / "probe"
    _write_probe(d, served=(pinned_hash + 0.001).astype(np.float32))
    errs, _ = CHK.check(d, tmp_path / "no_runs")
    assert any("canonical" in e for e in errs), errs


def test_checker_rejects_identical_kl_vectors(tmp_path, pinned_hash):
    """The failure that looks most like a real result: an adapter that
    never got applied scores exactly like its neighbours."""
    d = tmp_path / "probe"
    _write_probe(d, served=pinned_hash, kl_scale=1.0)
    errs, _ = CHK.check(d, tmp_path / "no_runs")
    assert any("identical KL vectors" in e for e in errs), errs


def test_checker_rejects_a_drifting_base_reference(tmp_path, pinned_hash):
    d = tmp_path / "probe"
    _write_probe(d, served=pinned_hash, recheck_drift=1e-3)
    errs, _ = CHK.check(d, tmp_path / "no_runs")
    assert any("leaking into its own reference" in e for e in errs), errs


def test_checker_rejects_mass_outside_the_support(tmp_path, pinned_hash):
    d = tmp_path / "probe"
    _write_probe(d, served=pinned_hash, tail=0.2)
    errs, _ = CHK.check(d, tmp_path / "no_runs")
    assert any("tail mass" in e for e in errs), errs


def test_checker_rejects_leverage_spread_across_positions(tmp_path,
                                                          pinned_hash):
    """If t* does not carry the value uncertainty, a one-position soft
    value is hiding structure rather than summarising it."""
    d = tmp_path / "probe"
    _write_probe(d, served=pinned_hash, n_lev_positions=4)
    errs, _ = CHK.check(d, tmp_path / "no_runs")
    assert any("leverage" in e for e in errs), errs


def test_checker_rejects_a_missing_adapter(tmp_path, pinned_hash):
    d = tmp_path / "probe"
    tags = _write_probe(d, served=pinned_hash)
    (d / f"adapter_{tags[3]}.pt").unlink()
    errs, _ = CHK.check(d, tmp_path / "no_runs")
    assert any("missing" in e for e in errs), errs


def test_checker_rejects_a_truncated_smoke_as_a_gateable_run(tmp_path,
                                                             pinned_hash):
    d = tmp_path / "probe"
    _write_probe(d, served=pinned_hash)
    mf = json.load(open(d / "probe_manifest.json"))
    mf["n_agents"] = 32
    json.dump(mf, open(d / "probe_manifest.json", "w"))
    errs, _ = CHK.check(d, tmp_path / "no_runs")
    assert any("truncated" in e for e in errs), errs


def test_checker_rejects_negative_kl(tmp_path, pinned_hash):
    d = tmp_path / "probe"
    tags = _write_probe(d, served=pinned_hash)
    r = torch.load(d / f"adapter_{tags[0]}.pt", weights_only=False)
    r["kl_fwd_sum"] = np.full(N, -0.01)
    torch.save(r, d / f"adapter_{tags[0]}.pt")
    errs, _ = CHK.check(d, tmp_path / "no_runs")
    assert any("negative" in e for e in errs), errs


# ------------------------------------------------------- condor wiring

def test_akl_sub_uses_its_own_executable_and_an_h100():
    sub = GEN.akl_sub(GEN.AKL_KEY)
    exe = next(ln for ln in sub.splitlines() if ln.startswith("executable"))
    assert exe.endswith("run_one_adapter_kl_probe.sh")
    assert f'CUDADeviceName == "{GEN.AKL_H100}"' in sub
    assert "queue mode from" in sub


def test_akl_configs_on_disk_match_the_generator():
    for key, smoke in ((GEN.AKL_KEY, False), (GEN.AKL_SMOKE_KEY, True)):
        p = CONDOR / f"configs_pofd_{key}.txt"
        assert p.read_text() == "\n".join(GEN.akl_rows(smoke=smoke)) + "\n"
        s = CONDOR / f"at_pofd_{key}.sub"
        assert s.read_text() == GEN.akl_sub(key, smoke=smoke)


def test_akl_modes_are_exactly_what_the_wrapper_accepts():
    """Same class as the submit-script brace bug: the mode string is
    produced in one file and consumed in another."""
    wrapper = (CONDOR / "run_one_adapter_kl_probe.sh").read_text()
    for mode in ("full", "smoke"):
        assert f'"{mode}"' in wrapper, mode
    assert GEN.akl_rows(smoke=False) == ["full"]
    assert GEN.akl_rows(smoke=True) == ["smoke"]


def test_submit_script_knows_the_akl_keys():
    s = (CONDOR / "submit_pofd_sweep.sh").read_text()
    assert "qwen_adapter_kl_probe|qwen_adapter_kl_probe_smoke)" in s
    assert "run_one_adapter_kl_probe.sh" in s


def test_submit_usage_strings_have_no_brace():
    """The bug that corrupted BID and WHAT for every key project-wide: a
    '}' inside ${1:?...} terminates the expansion early."""
    for ln in (CONDOR / "submit_pofd_sweep.sh").read_text().splitlines():
        if ln.startswith(("BID=", "WHAT=")):
            body = ln.split("usage:", 1)[1]
            assert "{" not in body[:-2] and body.count("}") == 1, ln[:120]


def test_checker_accepts_sub_threshold_teacher_forced_mismatches(tmp_path,
                                                                 pinned_hash):
    """generate() is KV-cached and the probe's pass is one full forward;
    in bf16 they can pick different argmaxes on a near-tie. On this task
    that is the phenomenon under study, not a fault."""
    d = tmp_path / "probe"
    _write_probe(d, served=pinned_hash, mismatch_n=102,
                 mismatch_margin=AKL.MARGIN_STRUCTURAL / 10)
    errs, notes = CHK.check(d, tmp_path / "no_runs")
    assert errs == [], errs
    assert any("teacher-forced argmax mismatches" in n for n in notes)


def test_checker_rejects_a_confident_teacher_forced_mismatch(tmp_path,
                                                             pinned_hash):
    """A CONFIDENT position cannot flip from float noise -- that is a
    misaligned span, and it must still abort."""
    d = tmp_path / "probe"
    _write_probe(d, served=pinned_hash, mismatch_n=5,
                 mismatch_margin=AKL.MARGIN_STRUCTURAL * 2)
    errs, _ = CHK.check(d, tmp_path / "no_runs")
    assert any("misaligned" in e for e in errs), errs


def test_checker_rejects_a_probe_with_no_alignment_record(tmp_path,
                                                          pinned_hash):
    d = tmp_path / "probe"
    _write_probe(d, served=pinned_hash)
    mf = json.load(open(d / "probe_manifest.json"))
    del mf["tf_mismatch"]
    json.dump(mf, open(d / "probe_manifest.json", "w"))
    errs, _ = CHK.check(d, tmp_path / "no_runs")
    assert any("no tf_mismatch record" in e for e in errs), errs


def test_flip_rate_is_measured_against_the_teacher_forced_base():
    """Folding the cached-vs-full numerical difference into every
    adapter's flip rate would inflate it by the base's own tie-break
    rate, identically for all 15 cells."""
    fn = next(n for n in ast.walk(_probe_ast())
              if isinstance(n, ast.FunctionDef) and n.name == "adapter_stage")
    src = ast.unparse(fn)
    assert "out['flip_tstar'][i] = float(am != int(base['tf_argmax'][i]))" in src
    assert "flip_vs_generated" in src


def test_teacher_forced_batch_matches_the_generation_batch():
    """Different padding widths change the bf16 reduction order and add
    argmax disagreement that has nothing to do with the model."""
    src = (PIPE / "probe_adapter_kl.py").read_text()
    assert '"--tf-batch", type=int, default=GEN_BATCH' in src
