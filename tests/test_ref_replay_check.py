"""Sabotage tests for the reference-replay pilot gates.

TWO SUITES LIVE HERE.

PART 1 -- DUAL AI-GATE SEMANTICS in check_pofd_sanity. The intended AI
gate measures |m - x'| with x' = gamma*innate + (1-gamma)*x, and in this
codebase gamma IS k, so x' is the vector already computed as
h = k*innate + (1-k)*x0. The archive was produced gating on x0 instead.
Both operators therefore have to stay replayable, dispatched on the run's
own population_update marker:

    nested_ai_then_social_v1           -> gate on x0      (archived)
    nested_ai_anchored_then_social_v2  -> gate on h       (corrected)
    marker absent                      -> gate on x0      (what those ran)

The tests below prove each artifact passes under ITS OWN semantics and
FAILS under the other, and that the dispatch is not vacuous: at k = 0
(h == x0) and under an all_open gate (the reference is unused) the two
coincide exactly, so a checker that always answered "x0" would look
correct on those runs alone.

PART 2 -- check_ref_replay, the pofdrr_ pilot gate. Small synthetic
artifacts are built at a toy population size (the production defaults
stay 723 agents / 181 optimizer steps) and then sabotaged one failure
mode at a time.

The pilot mechanism is EXPLICIT DATA-SPACE REFERENCE REPLAY: named
training rows are overwritten with a pinned frozen vector b. Nothing in
these tests encodes an expected direction for the population.
"""
import importlib.util
import json
import os

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_CP = os.path.join(HERE, os.pardir, "experiments", "scripts",
                   "cluster_pipelines")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_CP, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chk = _load("check_pofd_sanity_rr", "check_pofd_sanity.py")
rr = _load("check_ref_replay", "check_ref_replay.py")


# ===========================================================================
# PART 1 -- dual AI-gate semantics
# ===========================================================================

# 300 agents x 6 rounds is deliberate. At t = 0 x(0) == innate, so
# h(0) == x(0) and the two references AGREE no matter what -- the
# operators can only separate from round 1, and only on agents whose
# |m - x| sits within k*|innate - x| of the threshold. A 24-agent,
# 4-round fixture produced ZERO such agents and every dual-semantics
# test passed vacuously.
NP, PROUNDS, EPS_AI, W = 300, 6, 0.2, 0.5
V1 = "nested_ai_then_social_v1"
V2 = "nested_ai_anchored_then_social_v2"


def _roll(ref_kind, lam, gate_mode="threshold", eps_ai=EPS_AI, seed=0,
          operator="nested"):
    """A no-peer trajectory (eps_social=0, so the saved opinion IS z).

    operator="nested"  the v1/v2 order: AI mixture on the anchored human
                       component, peers last.
    operator="legacy"  the marker-absent order: gated blend on x, then
                       the innate re-anchor over everyone.
    ref_kind picks which vector the GATE is measured against, which is
    orthogonal to the operator.
    """
    g = torch.Generator().manual_seed(seed)
    innate = torch.rand(NP, generator=g)
    x = innate.clone()
    ops, preds, traj = [], [], []
    for t in range(PROUNDS):
        served = torch.rand(NP, generator=g)
        h = lam * innate + (1.0 - lam) * x
        ref = h if ref_kind == "anchor" else x
        if gate_mode == "all_open":
            gate = torch.ones(NP, dtype=torch.bool)
        else:
            gate = (served - ref).abs() < eps_ai
        if operator == "nested":
            x = torch.where(gate, (1.0 - W) * h + W * served, h)
        else:
            mid = torch.where(gate, (1.0 - W) * x + W * served, x)
            x = (1.0 - lam) * mid + lam * innate
        ops.append(x.clone())
        preds.append(served.clone())
        traj.append({"round": t, "accepted": 0, "is_deploy": True,
                     "n_train": 723,
                     "contact": float(gate.float().mean())})
    return innate, torch.stack(ops), torch.stack(preds), traj


def _write_pofd(tmp_path, marker, ref_kind, lam=0.2,
                gate_mode="threshold", eps_ai=EPS_AI,
                ai_gate_reference=None, operator=None):
    if operator is None:
        operator = "legacy" if marker is None else "nested"
    innate, op, pred, traj = _roll(ref_kind, lam, gate_mode, eps_ai,
                                   operator=operator)
    lam_tok = {0.2: "l0p2", 0.0: "l0"}[lam]
    ea_tok = "eaopen" if gate_mode == "all_open" else "ea0p2"
    cfg = {"eps": 0.0, "w_plat": W, "innate_lambda": lam,
           "eps_ai": eps_ai, "canary_delta": 0.0,
           "data_regime": "replace", "pop_model": "ab",
           "run_mode": "loop", "ab_sweeps": 1, "pop_reset": False,
           "platform_sus_scale": 1.0, "dataset": "movielens",
           "pristine_frac": 0.0, "fresh_each_round": True,
           "train_cap": 723, "seed": 0,
           "base_model": "Qwen/Qwen2.5-7B-Instruct"}
    if gate_mode != "threshold":
        cfg["ai_gate_mode"] = gate_mode
    if marker is not None:
        cfg["population_update"] = marker
    if ai_gate_reference is not None:
        cfg["ai_gate_reference"] = ai_gate_reference
    d = os.path.join(str(tmp_path),
                     f"pofdw_qwen7b_b1_{ea_tok}_w0p5_{lam_tok}_s0_fresh_data")
    os.makedirs(d, exist_ok=True)
    torch.save({"config": cfg, "trajectory": traj, "op_raw": op,
                "pred_raw": pred, "innate": innate},
               os.path.join(d, "trajectory.pt"))
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    return d


@pytest.mark.parametrize("marker,ref_kind", [
    (V1, "x0"),        # archived runs
    (None, "x0"),      # marker absent -> what those runs actually did
    (V2, "anchor"),    # corrected runs
])
def test_marker_matching_its_own_semantics_passes(tmp_path, marker,
                                                  ref_kind):
    errs = chk.check_run(_write_pofd(tmp_path, marker, ref_kind))
    assert errs == [], f"expected a clean replay, got {errs}"


@pytest.mark.parametrize("marker,ref_kind", [
    (V1, "anchor"),    # corrected dynamics wearing the archived marker
    (V2, "x0"),        # archived dynamics wearing the corrected marker
    (None, "anchor"),  # corrected dynamics with no marker at all
])
def test_marker_against_the_other_semantics_fails(tmp_path, marker,
                                                  ref_kind):
    errs = chk.check_run(_write_pofd(tmp_path, marker, ref_kind))
    assert any("EXACT-COPY" in e for e in errs), \
        f"a run replayed under the WRONG gate reference must fail: {errs}"


def _remarked(run_dir, marker):
    """Re-stamp an existing artifact with a different marker, in place."""
    p = os.path.join(run_dir, "trajectory.pt")
    blob = torch.load(p, map_location="cpu", weights_only=False)
    if marker is None:
        blob["config"].pop("population_update", None)
    else:
        blob["config"]["population_update"] = marker
    torch.save(blob, p)
    return run_dir


def test_k0_is_invariant_to_the_gate_reference(tmp_path):
    """k = 0 makes h == x0 identically, so the two references coincide and
    ONE artifact must verify under BOTH markers -- and, since the nested
    and legacy blends also coincide at k = 0, with no marker at all."""
    d = _write_pofd(tmp_path, V1, "x0", lam=0.0)
    for marker in (V1, V2, None):
        errs = chk.check_run(_remarked(d, marker))
        assert errs == [], f"k=0 under marker {marker!r}: {errs}"


def test_all_open_is_invariant_to_the_gate_reference(tmp_path):
    """An all_open gate ignores its reference entirely, so ONE all-open
    artifact verifies under either nested marker."""
    d = _write_pofd(tmp_path, V1, "x0", gate_mode="all_open")
    for marker in (V1, V2):
        errs = chk.check_run(_remarked(d, marker))
        assert errs == [], f"all_open under marker {marker!r}: {errs}"


def test_k0_and_all_open_checks_are_not_vacuous():
    """The guard on the two guards. At k > 0 under a THRESHOLD gate the
    reference genuinely changes the acceptance set on this fixture, so
    the two invariance tests above are about a real degeneracy and not
    about a checker that never looks at the reference at all.

    Round 0 is exempt by construction: x(0) == innate, so h(0) == x(0)
    and no gate can separate there.
    """
    innate, op, pred, _ = _roll("x0", 0.2)
    x0 = torch.cat([innate.unsqueeze(0), op[:-1]], dim=0)
    h = 0.2 * innate + 0.8 * x0
    g_x0 = (pred - x0).abs() < EPS_AI
    g_h = (pred - h).abs() < EPS_AI
    assert torch.equal(g_x0[0], g_h[0]), \
        "round 0 must coincide: x(0) == innate == h(0)"
    flips = int((g_x0[1:] ^ g_h[1:]).sum())
    assert flips > 0, ("fixture never separates the two references -- "
                       "widen NP/PROUNDS or move EPS_AI")


def test_unknown_marker_is_refused(tmp_path):
    d = _write_pofd(tmp_path, V1, "x0")
    p = os.path.join(d, "trajectory.pt")
    blob = torch.load(p, map_location="cpu", weights_only=False)
    blob["config"]["population_update"] = "nested_ai_then_social_v3"
    torch.save(blob, p)
    errs = chk.check_run(d)
    assert any("AI-GATE-REF unknown population_update" in e for e in errs), \
        f"an unknown marker must never fall back silently: {errs}"


def test_recorded_gate_reference_must_match_the_marker(tmp_path):
    errs = chk.check_run(_write_pofd(tmp_path, V1, "x0",
                                     ai_gate_reference="anchor"))
    assert any("AI-GATE-REF" in e and "disagree" in e for e in errs), errs


def test_recorded_gate_reference_agreeing_is_clean(tmp_path):
    assert chk.check_run(_write_pofd(tmp_path, V2, "anchor",
                                     ai_gate_reference="anchor")) == []
    assert chk.check_run(_write_pofd(tmp_path, V1, "x0",
                                     ai_gate_reference="x0")) == []


def test_gate_mask_replay_dispatches_on_the_marker(tmp_path):
    """gate_raw is bit-replayed too, and against the run's OWN reference:
    a v2 mask stored on a v1 artifact must not verify."""
    d = _write_pofd(tmp_path, V2, "anchor")
    p = os.path.join(d, "trajectory.pt")
    blob = torch.load(p, map_location="cpu", weights_only=False)
    innate, op, pred = blob["innate"], blob["op_raw"], blob["pred_raw"]
    x0 = torch.cat([innate.unsqueeze(0), op[:-1]], dim=0)
    h = 0.2 * innate + 0.8 * x0
    blob["gate_raw"] = (pred - h).abs() < EPS_AI
    torch.save(blob, p)
    assert chk.check_run(d) == []
    # the same mask under the archived marker: gate_raw no longer matches
    blob["config"]["population_update"] = V1
    torch.save(blob, p)
    errs = chk.check_run(d)
    assert any(e.startswith("GATE round") for e in errs), errs


# ===========================================================================
# PART 2 -- check_ref_replay sabotage suite
# ===========================================================================

N = 40                    # toy population; production stays 723
ROUNDS = 5
STEPS = 11                # toy fixed compute; production stays 181
SEED = 7
Q = 0.50
N_LIVE = 20               # round(0.50 * 40)


def _build(tmp_path, q=Q, n=N, rounds=ROUNDS, steps=STEPS, seed=SEED,
           name=None, n_rounds_cfg=None):
    """A clean synthetic pofdrr_ artifact. Returns (run_dir, b_sha)."""
    g = torch.Generator().manual_seed(1234)
    innate = torch.rand(n, generator=g)
    b = torch.rand(n, generator=g)
    n_live = rr.n_live_for(q, n)
    op = torch.zeros(rounds, n)
    pred = torch.zeros(rounds, n)
    labels = torch.zeros(rounds, n)
    live = torch.zeros(rounds, n_live, dtype=torch.long)
    x = innate.clone()
    traj, dose = [], []
    for t in range(rounds):
        idx = rr.live_perm(seed, t, n)[:n_live]
        live[t] = idx
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx] = True
        labels[t] = torch.where(mask, x, b)
        # the served vector is a stand-in: this checker never replays the
        # population operator, only the training labels and provenance
        pred[t] = (0.5 * labels[t] + 0.5 * b).clamp(0, 1)
        x = (0.5 * x + 0.5 * pred[t]).clamp(0, 1)
        op[t] = x.clone()
        traj.append({"round": t, "is_deploy": True, "n_train": n,
                     "parse_fail_frac": 0.0})
        dose.append({"round": t, "global_step": steps, "n_rows": n})
    cfg = {
        "ref_replay_q": q, "ref_replay_seed": seed,
        "ref_replay_n_live": n_live,
        "ref_replay_ref_run": "pofdqmech_qwen7b_k0_ea1_w0p5_l1_es0p05_s0",
        "ref_replay_ref_sha256": rr.sha_vec(b),
        "n_rounds": n_rounds_cfg if n_rounds_cfg is not None else rounds,
        # ORDINARY SFT. The surface's "beta = gamma = 1" is W_PLAT and
        # INNATE_LAMBDA, not the KL weight -- and the reused q=1 arm is
        # the completed QWU b0 (ordinary-SFT) cell, so every rung must
        # be the same learner.
        "kl_beta": 0.0, "training_style": "sft",
        "w_plat": 1.0, "innate_lambda": 1.0,
        "ai_gate_mode": "all_open", "peer_gate_mode": "all_open",
        "eps": 0.05, "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "fresh_each_round": True, "use_lora": 1, "seed": 0,
        "hardware": {"hostname": "h", "gpu_name": "NVIDIA H100 80GB HBM3",
                     "gpu_cc": "9.0", "torch": "2.4"},
    }
    d = os.path.join(str(tmp_path),
                     name or f"pofdrr_qwen7b_q{int(round(q * 100))}"
                             f"_s{seed}")
    os.makedirs(d, exist_ok=True)
    torch.save({"config": cfg, "trajectory": traj, "op_raw": op,
                "pred_raw": pred, "innate": innate,
                "ref_replay_live_idx": live,
                "ref_replay_labels": labels,
                "ref_replay_ref_vec": b,
                "sft_dose": dose}, os.path.join(d, "trajectory.pt"))
    return d, rr.sha_vec(b)


def _check(d, sha, **kw):
    kw.setdefault("n_agents", N)
    kw.setdefault("opt_steps", STEPS)
    kw.setdefault("expect_rounds", ROUNDS)
    kw.setdefault("canon_sha", sha)
    return rr.check_ref_replay(d, **kw)[0]


def _mutate(d, fn):
    p = os.path.join(d, "trajectory.pt")
    blob = torch.load(p, map_location="cpu", weights_only=False)
    fn(blob)
    torch.save(blob, p)
    return d


# -- the clean artifact must pass, or every sabotage below proves nothing --

@pytest.mark.parametrize("q", [0.10, 0.20, 0.50, 0.75, 1.0])
def test_clean_artifact_passes_on_every_rung(tmp_path, q):
    d, sha = _build(tmp_path / f"q{q}", q=q)
    assert _check(d, sha) == []


def test_production_defaults_are_the_paper_numbers():
    assert rr.N_AGENTS == 723 and rr.OPT_STEPS == 181
    assert rr.N_ROUNDS == 100
    assert rr.Q_LADDER_723 == {0.10: 72, 0.20: 145, 0.50: 362,
                               0.75: 542, 1.0: 723}
    assert rr.CANON_REF_SHA == (
        "1674ee5f8d833f46de672791d933e1d3bdeefb07484c2d110dec84ce71da30bb")


def test_canonical_sha_agrees_with_check_pofd_sanity():
    ok, msg = rr.canonical_sha_agrees()
    assert ok, msg


# -- CONTRACT: every missing field is a NAMED failure, never a crash -------

@pytest.mark.parametrize("key", ["ref_replay_live_idx", "ref_replay_labels",
                                 "ref_replay_ref_vec", "op_raw",
                                 "pred_raw", "innate"])
def test_missing_tensor_is_named_not_a_crash(tmp_path, key):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b.pop(key))
    errs = _check(d, sha)
    assert any(key in e and "CONTRACT" in e for e in errs), errs


@pytest.mark.parametrize("key", ["ref_replay_q", "ref_replay_seed",
                                 "ref_replay_n_live", "ref_replay_ref_run",
                                 "ref_replay_ref_sha256"])
def test_missing_config_key_is_named_not_a_crash(tmp_path, key):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b["config"].pop(key))
    errs = _check(d, sha)
    assert any(f"CONTRACT config.{key}" in e for e in errs), errs


def test_missing_trajectory_file_is_named(tmp_path):
    empty = tmp_path / "pofdrr_qwen7b_q50_s7"
    empty.mkdir()
    errs, _ = rr.check_ref_replay(str(empty), n_agents=N)
    assert errs and errs[0].startswith("MISSING"), errs


# -- LABELS ----------------------------------------------------------------

def test_live_label_not_the_agents_own_opinion_fails(tmp_path):
    d, sha = _build(tmp_path)

    def bad(b):
        live = b["ref_replay_live_idx"][2]
        b["ref_replay_labels"][2, live[0]] += 0.3
    _mutate(d, bad)
    errs = _check(d, sha)
    assert any(e.startswith("LABELS round 2") and "live rows" in e
               for e in errs), errs


def test_non_live_label_off_the_reference_fails(tmp_path):
    """A CONSTANT offset on the replayed rows in every round: it does not
    drift, so this must be caught as a label mismatch on its own."""
    d, sha = _build(tmp_path)

    def bad(b):
        n = b["ref_replay_labels"].shape[1]
        for t in range(b["ref_replay_labels"].shape[0]):
            mask = torch.ones(n, dtype=torch.bool)
            mask[b["ref_replay_live_idx"][t]] = False
            b["ref_replay_labels"][t][mask] = \
                b["ref_replay_ref_vec"][mask] + 0.05
    _mutate(d, bad)
    errs = _check(d, sha)
    assert any(e.startswith("LABELS") and "non-live" in e for e in errs), errs
    assert not any(e.startswith("DRIFT") for e in errs), \
        f"a constant offset is a label error, not accumulation: {errs}"


def test_accumulating_reference_is_caught_as_drift(tmp_path):
    """The replayed rows tracking the loop's own previous output instead
    of the pinned vector -- shapes all still check out."""
    d, sha = _build(tmp_path)

    def bad(b):
        op = b["op_raw"]
        for t in range(1, b["ref_replay_labels"].shape[0]):
            n = b["ref_replay_labels"].shape[1]
            mask = torch.ones(n, dtype=torch.bool)
            mask[b["ref_replay_live_idx"][t]] = False
            b["ref_replay_labels"][t][mask] = op[t - 1][mask]
    _mutate(d, bad)
    errs = _check(d, sha)
    assert any(e.startswith("DRIFT") for e in errs), errs


# -- LIVE SET / NESTING / ORDER -------------------------------------------

def test_live_set_off_the_deterministic_permutation_fails(tmp_path):
    d, sha = _build(tmp_path)

    def bad(b):
        # a DIFFERENT slice of the same permutation: still unique, still
        # in range, and no longer a prefix
        perm = rr.live_perm(SEED, 3, N)
        idx = perm[N_LIVE:2 * N_LIVE]
        b["ref_replay_live_idx"][3] = idx
        mask = torch.zeros(N, dtype=torch.bool)
        mask[idx] = True
        b["ref_replay_labels"][3] = torch.where(
            mask, b["op_raw"][2], b["ref_replay_ref_vec"])
    _mutate(d, bad)
    errs = _check(d, sha)
    assert any(e.startswith("LIVE round 3") for e in errs), errs
    assert any(e.startswith("NESTING round 3") for e in errs), \
        f"a non-prefix slice also breaks nesting: {errs}"


def test_reordered_live_row_breaks_order_not_nesting(tmp_path):
    """The elementwise-IN-ORDER claim is real: the same SET in a shuffled
    order is still nested but is no longer the recorded draw."""
    d, sha = _build(tmp_path)

    def bad(b):
        row = b["ref_replay_live_idx"][1]
        b["ref_replay_live_idx"][1] = row.flip(0)
    _mutate(d, bad)
    errs = _check(d, sha)
    assert any(e.startswith("LIVE round 1") for e in errs), errs
    assert not any(e.startswith("NESTING") for e in errs), errs


def test_live_index_out_of_range_is_named(tmp_path):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b["ref_replay_live_idx"].__setitem__((0, 0), N + 5))
    errs = _check(d, sha)
    assert any(e.startswith("ORDER round 0") and "range" in e
               for e in errs), errs


def test_duplicate_live_index_is_named(tmp_path):
    d, sha = _build(tmp_path)

    def bad(b):
        b["ref_replay_live_idx"][0, 1] = b["ref_replay_live_idx"][0, 0]
    _mutate(d, bad)
    errs = _check(d, sha)
    assert any(e.startswith("ORDER round 0") and "repeats" in e
               for e in errs), errs


def test_non_canonical_row_order_is_named(tmp_path):
    d, sha = _build(tmp_path)

    def bad(b):
        order = torch.arange(N).repeat(ROUNDS, 1)
        order[2] = order[2].flip(0)
        b["ref_replay_row_idx"] = order
    _mutate(d, bad)
    errs = _check(d, sha)
    assert any(e.startswith("ORDER recorded row order") for e in errs), errs


def test_permuted_label_rows_break_the_label_identity(tmp_path):
    """Row j must be agent j. A permuted label matrix fails immediately
    even at q = 1, which is what actually pins the row order."""
    d, sha = _build(tmp_path, q=1.0)
    _mutate(d, lambda b: b.__setitem__(
        "ref_replay_labels", b["ref_replay_labels"].flip(1)))
    errs = _check(d, sha)
    assert any(e.startswith("LABELS") or e.startswith("SFT-IDENTITY")
               for e in errs), errs


def test_cross_arm_nesting_violation_is_caught(tmp_path):
    small, sha_s = _build(tmp_path / "a", q=0.20,
                          name="pofdrr_qwen7b_q20_s7")
    big, sha_b = _build(tmp_path / "b", q=0.50,
                        name="pofdrr_qwen7b_q50_s7")
    _, i_s = rr.check_ref_replay(small, n_agents=N, opt_steps=STEPS,
                                 expect_rounds=ROUNDS, canon_sha=sha_s)
    _, i_b = rr.check_ref_replay(big, n_agents=N, opt_steps=STEPS,
                                 expect_rounds=ROUNDS, canon_sha=sha_b)
    assert rr.check_arms([i_s, i_b], n_agents=N) == []
    # move the small arm off the shared permutation prefix
    perm = rr.live_perm(SEED, 0, N)
    i_s["live_idx"] = i_s["live_idx"].clone()
    i_s["live_idx"][0] = perm[N - rr.n_live_for(0.20, N):]
    errs = rr.check_arms([i_s, i_b], n_agents=N)
    assert any(e.startswith("NESTING-ARMS") for e in errs), errs


def test_arms_on_different_seeds_are_refused(tmp_path):
    small, sha_s = _build(tmp_path / "a", q=0.20,
                          name="pofdrr_qwen7b_q20_s7")
    big, sha_b = _build(tmp_path / "b", q=0.50, seed=SEED + 1,
                        name="pofdrr_qwen7b_q50_s8")
    _, i_s = rr.check_ref_replay(small, n_agents=N, opt_steps=STEPS,
                                 expect_rounds=ROUNDS, canon_sha=sha_s)
    _, i_b = rr.check_ref_replay(big, n_agents=N, opt_steps=STEPS,
                                 expect_rounds=ROUNDS, canon_sha=sha_b)
    errs = rr.check_arms([i_s, i_b], n_agents=N)
    assert any("must share one live-set stream" in e for e in errs), errs


# -- LADDER ----------------------------------------------------------------

def test_wrong_n_live_for_the_declared_q_fails(tmp_path):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b["config"].__setitem__("ref_replay_n_live",
                                                 N_LIVE + 1))
    errs = _check(d, sha)
    assert any(e.startswith("LADDER") for e in errs), errs


def test_off_ladder_q_is_refused(tmp_path):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b["config"].__setitem__("ref_replay_q", 0.33))
    errs = _check(d, sha)
    assert any("off the pilot ladder" in e for e in errs), errs


def test_production_ladder_sizes():
    for q, n in ((0.10, 72), (0.20, 145), (0.50, 362), (0.75, 542),
                 (1.0, 723)):
        assert rr.n_live_for(q, 723) == n
    assert rr.n_live_for(0.33, 723) is None


def test_tag_q_token_must_match_the_config(tmp_path):
    d, sha = _build(tmp_path, q=0.50, name="pofdrr_qwen7b_q20_s7")
    errs = _check(d, sha)
    assert any("tag says q=" in e for e in errs), errs


# -- SHAPE / STEPS ---------------------------------------------------------

def test_wrong_row_count_fails(tmp_path):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b.__setitem__(
        "ref_replay_labels", b["ref_replay_labels"][:, :N - 1]))
    errs = _check(d, sha)
    assert any(e.startswith("SHAPE ref_replay_labels") for e in errs), errs


def test_wrong_prediction_length_fails(tmp_path):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b.__setitem__("pred_raw",
                                       b["pred_raw"][:, :N - 2]))
    errs = _check(d, sha)
    assert any(e.startswith("SHAPE pred_raw") for e in errs), errs


def test_optimizer_steps_must_be_fixed(tmp_path):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b["sft_dose"][3].__setitem__("global_step",
                                                      STEPS - 1))
    errs = _check(d, sha)
    assert any(e.startswith("STEPS optimizer steps") for e in errs), errs


def test_missing_step_provenance_is_named(tmp_path):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b.pop("sft_dose"))
    errs = _check(d, sha)
    assert any("no per-round optimizer-step provenance" in e
               for e in errs), errs


# -- REFERENCE VECTOR ------------------------------------------------------

def test_reference_sha_must_match_the_config(tmp_path):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b["config"].__setitem__("ref_replay_ref_sha256",
                                                 "0" * 64))
    errs = _check(d, sha)
    assert any("!= config ref_replay_ref_sha256" in e for e in errs), errs


def test_reference_must_be_the_canonical_frozen_vector(tmp_path):
    d, sha = _build(tmp_path)
    errs = _check(d, sha, canon_sha="f" * 64)
    assert any("canonical frozen-Qwen vector" in e for e in errs), errs


def test_reference_out_of_range_fails(tmp_path):
    d, sha = _build(tmp_path)

    def bad(b):
        v = b["ref_replay_ref_vec"].clone()
        v[3] = 1.7
        b["ref_replay_ref_vec"] = v
        b["config"]["ref_replay_ref_sha256"] = rr.sha_vec(v)
    _mutate(d, bad)
    errs = _check(d, sha, require_canon_sha=False)
    assert any("out of [0,1]" in e for e in errs), errs


def test_reference_non_finite_fails(tmp_path):
    d, sha = _build(tmp_path)

    def bad(b):
        v = b["ref_replay_ref_vec"].clone()
        v[5] = float("nan")
        b["ref_replay_ref_vec"] = v
        b["config"]["ref_replay_ref_sha256"] = rr.sha_vec(v)
    _mutate(d, bad)
    errs = _check(d, sha, require_canon_sha=False)
    assert any("non-finite" in e for e in errs), errs


def test_reference_wrong_length_fails(tmp_path):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b.__setitem__("ref_replay_ref_vec",
                                       b["ref_replay_ref_vec"][:N - 3]))
    errs = _check(d, sha)
    assert any(e.startswith("SHAPE ref_replay_ref_vec") for e in errs), errs


# -- q = 1 IDENTITY --------------------------------------------------------

def test_q1_must_reduce_exactly_to_ordinary_sft(tmp_path):
    d, sha = _build(tmp_path, q=1.0)
    assert _check(d, sha) == []

    def bad(b):
        # one replayed row sneaking into the "ordinary SFT" arm
        b["ref_replay_labels"][2, 0] = b["ref_replay_ref_vec"][0] + 0.25
    _mutate(d, bad)
    errs = _check(d, sha)
    assert any(e.startswith("SFT-IDENTITY") or e.startswith("LABELS")
               for e in errs), errs


def test_q1_with_a_short_live_set_fails(tmp_path):
    d, sha = _build(tmp_path, q=1.0)

    def bad(b):
        b["ref_replay_live_idx"] = b["ref_replay_live_idx"][:, :N - 1]
        b["config"]["ref_replay_n_live"] = N - 1
    _mutate(d, bad)
    errs = _check(d, sha)
    assert any(e.startswith("LADDER") or e.startswith("SFT-IDENTITY")
               for e in errs), errs


# -- SURFACE ---------------------------------------------------------------

@pytest.mark.parametrize("key,value", [
    ("kl_beta", 1.0),            # a KL arm is NOT this surface
    ("training_style", "sft_kl"),
    ("w_plat", 0.5),             # platform beta must be 1
    ("innate_lambda", 0.2),
    ("ai_gate_mode", "threshold"),
    ("peer_gate_mode", "threshold"),
    ("eps", 0.0),
    ("base_model", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("fresh_each_round", False),
])
def test_wrong_surface_fails(tmp_path, key, value):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b["config"].__setitem__(key, value))
    errs = _check(d, sha)
    assert any(e.startswith("SURFACE") and key in e for e in errs), errs


def test_wrong_gpu_sku_fails(tmp_path):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b["config"].__setitem__(
        "hardware", {"gpu_name": "NVIDIA A100-SXM4-80GB"}))
    errs = _check(d, sha)
    assert any("not an H100" in e for e in errs), errs


def test_wrong_horizon_fails(tmp_path):
    d, sha = _build(tmp_path)
    errs = _check(d, sha, expect_rounds=ROUNDS + 1)
    assert any(e.startswith("HORIZON") for e in errs), errs


def test_truncated_run_fails(tmp_path):
    d, sha = _build(tmp_path, n_rounds_cfg=ROUNDS + 3)
    errs = _check(d, sha)
    assert any(e.startswith("HORIZON") and "truncated" in e
               for e in errs), errs


def test_smoke_tag_is_exempt_from_the_fixed_horizon(tmp_path):
    d, sha = _build(tmp_path, name="pofdrrsmk_qwen7b_q50_s7")
    assert _check(d, sha, expect_rounds=100) == []


def test_non_pofdrr_tag_is_refused(tmp_path):
    d, sha = _build(tmp_path, name="pofdqwu_qwen7b_b1_eaopen_w1_l1")
    errs = _check(d, sha)
    assert any(e.startswith("TAG") for e in errs), errs


# -- SERVING ---------------------------------------------------------------

def test_non_finite_predictions_fail(tmp_path):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b["pred_raw"].__setitem__((1, 4), float("nan")))
    errs = _check(d, sha)
    assert any(e.startswith("SERVING non-finite") for e in errs), errs


def test_out_of_range_predictions_fail(tmp_path):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b["pred_raw"].__setitem__((1, 4), 1.9))
    errs = _check(d, sha)
    assert any("predictions out of [0,1]" in e for e in errs), errs


def test_parse_failures_fail(tmp_path):
    d, sha = _build(tmp_path)
    _mutate(d, lambda b: b["trajectory"][2].__setitem__("parse_fail_frac",
                                                        0.004))
    errs = _check(d, sha)
    assert any("parse failures" in e for e in errs), errs


def test_missing_parse_provenance_is_named(tmp_path):
    d, sha = _build(tmp_path)

    def bad(b):
        for r in b["trajectory"]:
            r.pop("parse_fail_frac", None)
    _mutate(d, bad)
    errs = _check(d, sha)
    assert any("no parse provenance" in e for e in errs), errs


def test_digit_free_generation_fails(tmp_path):
    import gzip
    d, sha = _build(tmp_path)
    with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt") as fh:
        for t in range(ROUNDS):
            raw = ["0.5"] * N
            if t == 3:
                raw[0] = "I am not sure about this one."
            fh.write(json.dumps({"round": t, "raw": raw}) + "\n")
    errs = _check(d, sha)
    assert any("contain no digit" in e for e in errs), errs


# ===========================================================================
# PART 3 -- analyze_ref_replay
# ===========================================================================

ana = _load("analyze_ref_replay", "analyze_ref_replay.py")


def _frozen_run(root, n=N, rounds=ROUNDS):
    """A frozen-Qwen stand-in: a CONSTANT served vector every round."""
    g = torch.Generator().manual_seed(99)
    b = torch.rand(n, generator=g)
    pred = b.unsqueeze(0).repeat(rounds, 1)
    op = torch.rand(rounds, n, generator=g)
    d = root / "pofdqmech_toy_frozen"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"config": {}, "trajectory": [], "op_raw": op,
                "pred_raw": pred, "innate": b}, d / "trajectory.pt")
    return d, ana.sha_vec(b.numpy())


def _arms_root(tmp_path):
    root = tmp_path / "runs"
    root.mkdir(parents=True, exist_ok=True)
    for q in (0.50, 1.0):
        _build(root, q=q)
    return root


def test_analyzer_runs_and_writes_its_outputs(tmp_path, monkeypatch):
    root = _arms_root(tmp_path)
    frozen, fsha = _frozen_run(root)
    monkeypatch.setattr(ana, "CANON_REF_SHA", fsha)
    monkeypatch.setattr(ana, "FROZEN_CANDIDATES", ("pofdqmech_toy_frozen",))
    out = tmp_path / "out"
    rounds, per_arm = ana.analyse([root], out, want_q=(0.50, 1.0),
                                  window=2)
    assert (out / "ref_replay_rounds.csv").exists()
    assert (out / "ref_replay_per_arm.csv").exists()
    assert (out / "ref_replay_population_mean_sd.png").exists()
    assert {a["q"] for a in per_arm} == {0.5, 1.0}
    for a in per_arm:
        for key in ("w1_to_frozen_equilibrium", "w1_to_sft_equilibrium",
                    "late_served_mae_to_b", "late_served_w1_to_b",
                    "late_served_unique", "late_served_max_mode_share",
                    "late_parse_fail_frac", "late_mean_drift",
                    "late_mean_slope_per_round", "outer_stationarity"):
            assert key in a, key
        assert a["replicates"].startswith("none")
    # the q=1 arm is its own ordinary-SFT endpoint, so that distance is 0
    sft = [a for a in per_arm if a["q"] == 1.0][0]
    assert sft["w1_to_sft_equilibrium"] == pytest.approx(0.0, abs=1e-9)
    # rounds are reported 1-based and only end-of-round states appear
    assert min(r["completed_round"] for r in rounds) == 1
    assert max(r["completed_round"] for r in rounds) == ROUNDS


def test_analyzer_hard_fails_on_a_missing_arm(tmp_path, monkeypatch):
    root = _arms_root(tmp_path)
    frozen, fsha = _frozen_run(root)
    monkeypatch.setattr(ana, "CANON_REF_SHA", fsha)
    monkeypatch.setattr(ana, "FROZEN_CANDIDATES", ("pofdqmech_toy_frozen",))
    with pytest.raises(SystemExit) as e:
        ana.analyse([root], tmp_path / "out", want_q=(0.10, 0.50, 1.0),
                    window=2)
    assert "arm(s) absent" in str(e.value)


def test_analyzer_hard_fails_without_the_canonical_frozen_reference(
        tmp_path, monkeypatch):
    root = _arms_root(tmp_path)
    _frozen_run(root)
    monkeypatch.setattr(ana, "CANON_REF_SHA", "d" * 64)
    monkeypatch.setattr(ana, "FROZEN_CANDIDATES", ("pofdqmech_toy_frozen",))
    with pytest.raises(SystemExit) as e:
        ana.analyse([root], tmp_path / "out", want_q=(0.50, 1.0), window=2)
    assert "canonical frozen-Qwen" in str(e.value)


def test_analyzer_hard_fails_without_the_q1_arm(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir(parents=True, exist_ok=True)
    _build(root, q=0.50)
    frozen, fsha = _frozen_run(root)
    monkeypatch.setattr(ana, "CANON_REF_SHA", fsha)
    monkeypatch.setattr(ana, "FROZEN_CANDIDATES", ("pofdqmech_toy_frozen",))
    with pytest.raises(SystemExit) as e:
        ana.analyse([root], tmp_path / "out", want_q=(0.50,), window=2)
    assert "ordinary SFT" in str(e.value)


def test_headline_figure_carries_no_title():
    """Project convention: paper figures carry NO title text -- the
    narrative goes in the caption block."""
    src = open(os.path.join(_CP, "analyze_ref_replay.py")).read()
    assert "set_title(" not in src and "suptitle(" not in src


# -- neither tool encodes a direction --------------------------------------

@pytest.mark.parametrize("fname", ["check_ref_replay.py",
                                   "analyze_ref_replay.py"])
def test_tools_are_direction_neutral(fname):
    low = " ".join(open(os.path.join(_CP, fname)).read().lower().split())
    assert "explicit data-space reference replay" in low, \
        "the mechanism must be named EXPLICIT DATA-SPACE REFERENCE REPLAY"
    # "implicit anchoring" may appear only as something the file DISOWNS
    i = 0
    while True:
        i = low.find("implicit anchor", i)
        if i < 0:
            break
        window = low[max(0, i - 60):i]
        assert "not" in window or "never" in window, \
            f"{fname} describes the mechanism as implicit anchoring"
        i += 1
    for banned in ("must decrease", "must increase", "should move toward",
                   "expected to converge", "should converge to"):
        assert banned not in low, f"directional assumption {banned!r}"


def test_missing_step_telemetry_falls_back_to_derived_arithmetic(tmp_path):
    """The REUSED q=1 arm is the completed QWU b0 cell, which predates
    sft_dose telemetry (added 2026-08-21). Requiring MEASURED steps would
    reject the one arm this design gets for free, so the checker derives
    ceil(rows/batch)*epochs from config -- and records that it did, so a
    reader can tell an arithmetic claim from a measured one."""
    d, sha = _build(tmp_path / "derived", q=1.0)

    def strip(blob):
        blob.pop("sft_dose", None)
        for r in (blob.get("trajectory") or []):
            r.pop("opt_steps", None); r.pop("global_step", None)
        blob["config"].update({"n_labeled": N, "sft_batch_size": 4,
                               "sft_epochs": 1})
    _mutate(d, strip)
    errs, info = rr.check_ref_replay(
        d, n_agents=N, opt_steps=-(-N // 4), expect_rounds=ROUNDS,
        canon_sha=sha)
    assert not [e for e in errs if e.startswith("STEPS")], errs
    assert info.get("steps_source") == "derived"
    assert "ARITHMETIC, not measurement" in info.get("steps_note", "")


def test_derived_steps_still_reject_a_wrong_compute_budget(tmp_path):
    """The fallback must not become a bypass."""
    d, sha = _build(tmp_path / "derived_bad", q=1.0)

    def strip(blob):
        blob.pop("sft_dose", None)
        for r in (blob.get("trajectory") or []):
            r.pop("opt_steps", None); r.pop("global_step", None)
        blob["config"].update({"n_labeled": N, "sft_batch_size": 8,
                               "sft_epochs": 1})
    _mutate(d, strip)
    errs, _ = rr.check_ref_replay(
        d, n_agents=N, opt_steps=-(-N // 4), expect_rounds=ROUNDS,
        canon_sha=sha)
    assert any(e.startswith("STEPS derived") for e in errs), errs


def test_no_step_provenance_and_no_config_is_still_a_hard_failure(tmp_path):
    d, sha = _build(tmp_path / "derived_none", q=1.0)

    def strip(blob):
        blob.pop("sft_dose", None)
        for r in (blob.get("trajectory") or []):
            r.pop("opt_steps", None); r.pop("global_step", None)
        for k in ("n_labeled", "sft_batch_size", "sft_epochs"):
            blob["config"].pop(k, None)
    _mutate(d, strip)
    errs, _ = rr.check_ref_replay(
        d, n_agents=N, opt_steps=STEPS, expect_rounds=ROUNDS, canon_sha=sha)
    assert any("cannot be assumed" in e for e in errs), errs


def test_the_two_betas_are_not_confused(tmp_path):
    """beta is overloaded: W_PLAT (platform) and kl_beta (KL weight).
    The pilot surface is platform beta=1 with KL weight 0. Reading the
    surface's 'beta=1' as the KL weight rejected the first smoke, so
    both are pinned explicitly and in opposite directions."""
    d, sha = _build(tmp_path / "betas", q=0.5)
    assert _check(d, sha) == []
    _mutate(d, lambda b: b["config"].update({"kl_beta": 1.0,
                                             "training_style": "sft_kl"}))
    errs = _check(d, sha)
    assert any("ORDINARY SFT" in e for e in errs), errs
    d2, sha2 = _build(tmp_path / "betas2", q=0.5)
    _mutate(d2, lambda b: b["config"].update({"w_plat": 0.5}))
    assert any("platform beta = 1" in e for e in _check(d2, sha2))
