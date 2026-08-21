"""Tests for the Friedkin-Johnsen robustness wave (2026-08-21).

No model is loaded. The FJ operator is linear and fully specified, so
every claim the wave makes about it is checkable in closed form against
small hand-built graphs.

The checker tests are SABOTAGE tests: each builds a run that is valid
except for one defect and asserts the checker rejects it. The defects
are the four blockers the archived FJ path actually had -- wrong beta,
inverted alpha convention, wrong inner count, stale previous-state
initialisation, and a served vector that is not the recorded one --
plus arm swaps, wrong hardware, truncated horizons and corrupted
trajectories. A gate nobody has watched fail is not a gate.
"""
from __future__ import annotations

import importlib.util
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


from perfsim.environments.dynamics import FJWorld, normalize_adjacency  # noqa: E402

CHK = _load("chk_fjr", PIPE / "check_pofd_sanity.py")
CTL = _load("fjctl", PIPE / "fj_controls.py")
ANA = _load("ana_fjr", PIPE / "analyze_fj_robustness.py")
GEN = _load("gen_fjr", CONDOR / "gen_pofd_sweep.py")

N = 4


def _toy():
    """A 4-agent ring: row-stochastic, every agent has neighbours, so
    alpha genuinely mixes and nothing is trivially isolated."""
    adj = torch.tensor([[0., 1., 0., 1.],
                        [1., 0., 1., 0.],
                        [0., 1., 0., 1.],
                        [1., 0., 1., 0.]])
    W = adj / adj.sum(dim=1, keepdim=True)
    innate = torch.tensor([0.1, 0.4, 0.7, 0.9])
    return innate, W


def _world(innate, W, beta, alpha):
    return FJWorld(
        innate=innate, graph=W,
        peer_sus=torch.full((N,), FJWorld.alpha_to_peer_sus(alpha)),
        platform_sus=torch.full((N,), float(beta)),
        features=innate,
    )


# ------------------------------------------------------- the convention

def test_alpha_maps_to_the_stored_anchor_weight():
    """peer_sus is the ANCHOR weight. Calling it alpha is the mistake
    that makes MovieLens (peer_sus=1) look like full mixing when it is
    actually full anchoring."""
    assert FJWorld.alpha_to_peer_sus(0.0) == 1.0    # no mixing = anchored
    assert FJWorld.alpha_to_peer_sus(1.0) == 0.0
    assert FJWorld.alpha_to_peer_sus(0.5) == 0.5


def test_movielens_peer_sus_of_one_would_mean_no_mixing():
    """The archived blocker, pinned: peer_sus=1 is FULLY ANCHORED."""
    innate, W = _toy()
    w = FJWorld(innate=innate, graph=W, peer_sus=torch.ones(N),
                platform_sus=torch.zeros(N), features=innate)
    out = w.run_wu(torch.zeros(N), alpha=0.0, n_inner=1)
    assert torch.allclose(w.state["opinion"], innate)


# --------------------------------------------------- the recurrence

def test_run_wu_matches_the_declared_recurrence():
    innate, W = _toy()
    beta, alpha = 0.5, 0.5
    preds = torch.tensor([0.2, 0.2, 0.8, 0.8])
    w = _world(innate, W, beta, alpha)
    w.run_wu(preds, alpha=alpha, n_inner=1)
    x0 = (1 - beta) * innate + beta * preds
    want = (1 - alpha) * x0 + alpha * (W @ x0)
    assert torch.allclose(w.state["opinion"], want, atol=1e-6)


def test_inner_loop_starts_at_the_new_anchor_not_the_previous_state():
    """Blocker 4. Run two rounds; round 2 must depend on round 2's anchor
    ALONE. If the inner loop were seeded from round 1's state, the result
    would differ -- and the test asserts those two are distinguishable so
    it cannot pass vacuously."""
    innate, W = _toy()
    beta, alpha, n_inner = 0.5, 0.5, 3
    w = _world(innate, W, beta, alpha)
    w.run_wu(torch.tensor([0.9, 0.9, 0.1, 0.1]), alpha=alpha, n_inner=n_inner)
    prev = w.state["opinion"].clone()
    p2 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    w.run_wu(p2, alpha=alpha, n_inner=n_inner)

    x0 = (1 - beta) * innate + beta * p2
    fresh = x0.clone()
    for _ in range(n_inner):
        fresh = (1 - alpha) * x0 + alpha * (W @ fresh)
    stale = prev.clone()
    for _ in range(n_inner):
        stale = (1 - alpha) * x0 + alpha * (W @ stale)

    assert not torch.allclose(fresh, stale, atol=1e-6), \
        "test is vacuous: the two initialisations agree"
    assert torch.allclose(w.state["opinion"], fresh, atol=1e-6)


def test_run_wu_never_queries_a_model():
    """Blocker 3: generation must happen exactly once per round, in the
    runner, and the SERVED vector is passed in."""
    innate, W = _toy()
    w = _world(innate, W, 0.5, 0.5)

    class Explode:
        def __call__(self, *a, **k):
            raise AssertionError("run_wu queried the model")

    w.run_wu(torch.full((N,), 0.5), alpha=0.5, n_inner=1)   # no model arg
    import inspect
    assert "model" not in inspect.signature(w.run_wu).parameters


def test_archived_run_is_untouched():
    """The opt-in requirement: legacy behaviour must be bit-identical."""
    innate, W = _toy()
    w1 = FJWorld(innate=innate, graph=W, peer_sus=torch.full((N,), 0.5),
                 platform_sus=torch.full((N,), 0.5), features=innate)
    w2 = FJWorld(innate=innate, graph=W, peer_sus=torch.full((N,), 0.5),
                 platform_sus=torch.full((N,), 0.5), features=innate)

    class Const:
        def __call__(self, x):
            return torch.full((N, 1), 0.3)

    w1.run(Const(), n_steps=4)
    # legacy: inner loop from the PREVIOUS state, model queried inside
    x0 = 0.5 * innate + 0.5 * torch.full((N,), 0.3)
    x = w2.state["opinion"]
    for _ in range(4):
        x = 0.5 * x0 + 0.5 * (W @ x)
    assert torch.allclose(w1.state["opinion"], x, atol=1e-6)


# ------------------------------------------------------ limiting cases

def test_beta_zero_means_predictions_have_no_effect():
    innate, W = _toy()
    w_a = _world(innate, W, 0.0, 0.5)
    w_b = _world(innate, W, 0.0, 0.5)
    w_a.run_wu(torch.zeros(N), alpha=0.5, n_inner=2)
    w_b.run_wu(torch.ones(N), alpha=0.5, n_inner=2)
    assert torch.allclose(w_a.state["opinion"], w_b.state["opinion"])


def test_alpha_zero_leaves_the_population_at_the_anchor():
    innate, W = _toy()
    beta = 0.5
    preds = torch.tensor([0.2, 0.4, 0.6, 0.8])
    w = _world(innate, W, beta, 0.0)
    w.run_wu(preds, alpha=0.0, n_inner=5)
    assert torch.allclose(w.state["opinion"],
                          (1 - beta) * innate + beta * preds, atol=1e-6)


def test_perfect_prediction_replay_matches_the_recurrence():
    innate, W = _toy()
    beta, alpha = 0.5, 0.5
    traj = CTL.run_control("perfect", innate, W, beta=beta, alpha=alpha,
                           n_inner=1, rounds=6)
    x = innate.clone()
    for t in range(6):
        x0 = (1 - beta) * innate + beta * x
        x = (1 - alpha) * x0 + alpha * (W @ x0)
        assert torch.allclose(traj[t], x, atol=1e-6)


def test_perfect_prediction_contracts_at_beta():
    """Why 30 rounds is a horizon and not an anecdote: the map is linear
    with contraction factor beta, so the step shrinks geometrically."""
    innate, W = _toy()
    traj = CTL.run_control("perfect", innate, W, beta=0.5, alpha=0.5,
                           n_inner=1, rounds=12)
    steps = [float((traj[t] - traj[t - 1]).abs().max())
             for t in range(1, 12)]
    ratios = [steps[i + 1] / steps[i] for i in range(len(steps) - 1)
              if steps[i] > 1e-12]
    assert all(r < 0.75 for r in ratios), ratios


def test_a_constant_predictor_gives_a_constant_population():
    """Why the frozen control needs no GPU loop: with a stateless human
    component the only channel between rounds is the model."""
    innate, W = _toy()
    fz = torch.tensor([0.3, 0.3, 0.6, 0.6])
    traj = CTL.run_control("frozen", innate, W, n_inner=1, rounds=5,
                           frozen_vec=fz)
    for t in range(1, 5):
        assert torch.allclose(traj[t], traj[0], atol=1e-7)


def test_no_platform_control_is_beta_zero():
    innate, W = _toy()
    traj = CTL.run_control("no_platform", innate, W, n_inner=1, rounds=3)
    want = CTL.fj_step(innate, torch.zeros(N), 0.0, CTL.ALPHA, 1, W)
    assert torch.allclose(traj[0], want, atol=1e-7)


def test_run_wu_rejects_bad_arguments():
    innate, W = _toy()
    w = _world(innate, W, 0.5, 0.5)
    with pytest.raises(ValueError):
        w.run_wu(torch.zeros(N), alpha=1.5, n_inner=1)
    with pytest.raises(ValueError):
        w.run_wu(torch.zeros(N), alpha=0.5, n_inner=0)
    with pytest.raises(ValueError):
        w.run_wu(torch.zeros(N + 1), alpha=0.5, n_inner=1)
    with pytest.raises(ValueError):
        w.run_wu(torch.full((N,), float("nan")), alpha=0.5, n_inner=1)


# --------------------------------------------------- checker sabotage

def _mk_run(tmp, *, beta=0.5, alpha=0.5, n_inner=1, rounds=4, arm="b0",
            model="qwen7b", gpu="NVIDIA H100 80GB HBM3", stale=False,
            served_mismatch=False, style=None, kl_beta=None, corrupt=False):
    """A structurally valid FJ run, defect injected by kwargs."""
    innate, W = _toy()
    rng = np.random.default_rng(0)
    preds, ops, trainy = [], [], []
    x_prev = None
    for t in range(rounds):
        p = torch.tensor(rng.uniform(0.1, 0.9, N), dtype=torch.float32)
        x0 = (1 - beta) * innate + beta * p
        seed = (x_prev if (stale and x_prev is not None) else x0).clone()
        x = seed
        for _ in range(n_inner):
            x = (1 - alpha) * x0 + alpha * (W @ x)
        preds.append(p)
        ops.append(x)
        trainy.append(innate.clone() if t == 0 else ops[t - 1].clone())
        x_prev = x
    op_raw = torch.stack(ops)
    if corrupt:
        op_raw = op_raw + 0.05
    used = torch.stack(preds)
    if served_mismatch:
        used = used.clone()
        used[1, 0] += 0.1
    st, kb = (style or ("sft" if arm == "b0" else "sft_kl")), (
        kl_beta if kl_beta is not None else (0.0 if arm == "b0" else 1.0))
    cfg = {
        "training_style": st, "kl_beta": kb, "kl_direction": "forward",
        "kl_ref_adapter": "", "w_plat": beta, "fj_alpha": alpha,
        "fj_inner_steps": n_inner, "fj_update_version": "wu1",
        "fj_beta_mean": beta, "fj_human_component": "stateless_innate_k1",
        "fj_graph_sha256": CHK._sha_t(W),
        "hardware": {"gpu_name": gpu}, "serve_eval_mode": True,
        "pop_model": "fj",
    }
    d = {"config": cfg, "op_raw": op_raw, "pred_raw": torch.stack(preds),
         "innate": innate, "fj_served_used": used,
         "train_y_raw": torch.stack(trainy),
         "trajectory": [{"t": t, "parse_fail": 0.0} for t in range(rounds)]}
    tag = (f"pofdfj_{model}_{arm}_beta{GEN._num(beta)}"
           f"_alpha{GEN._num(alpha)}_in{n_inner}_k1_s0_r{rounds}")
    return tag, d, W


def _check(tmp, **kw):
    rounds = kw.get("rounds", 4)
    tag, d, W = _mk_run(tmp, **kw)
    CHK._FJR_GRAPH_CACHE["W"] = W
    return CHK.check_fjr(tag, d["config"], d, d["op_raw"].float(),
                         d["pred_raw"].float(), d["innate"].float(),
                         expect_rounds=rounds, expect_agents=N)


def test_clean_fj_run_passes(tmp_path):
    errs = _check(tmp_path)
    assert errs == [], errs


def test_checker_rejects_wrong_beta(tmp_path):
    """The tag says beta=.5; the operator ran at something else. This is
    blocker 1 -- W_PLAT never reaching FJ -- in its observable form."""
    tag, d, W = _mk_run(tmp_path, beta=0.5)
    CHK._FJR_GRAPH_CACHE["W"] = W
    d["config"]["w_plat"] = 0.5
    d["config"]["fj_beta_mean"] = 1.0        # what actually ran
    errs = CHK.check_fjr(tag, d["config"], d, d["op_raw"].float(),
                         d["pred_raw"].float(), d["innate"].float(),
                         expect_rounds=4, expect_agents=N)
    assert any("replay the declared recurrence" in e for e in errs), errs


def test_checker_rejects_inverted_alpha_convention(tmp_path):
    """peer_sus vs alpha swapped: the run mixed with 1-alpha."""
    tag, d, W = _mk_run(tmp_path, alpha=0.25)
    CHK._FJR_GRAPH_CACHE["W"] = W
    d["config"]["fj_alpha"] = 0.75           # 1 - 0.25
    errs = CHK.check_fjr(tag, d["config"], d, d["op_raw"].float(),
                         d["pred_raw"].float(), d["innate"].float(),
                         expect_rounds=4, expect_agents=N)
    assert any("replay the declared recurrence" in e for e in errs), errs


def test_checker_rejects_wrong_inner_step_count(tmp_path):
    tag, d, W = _mk_run(tmp_path, n_inner=3)
    CHK._FJR_GRAPH_CACHE["W"] = W
    d["config"]["fj_inner_steps"] = 1
    errs = CHK.check_fjr(tag, d["config"], d, d["op_raw"].float(),
                         d["pred_raw"].float(), d["innate"].float(),
                         expect_rounds=4, expect_agents=N)
    assert any("replay the declared recurrence" in e for e in errs), errs


def test_checker_rejects_stale_previous_state_initialisation(tmp_path):
    """Blocker 4 in its observable form."""
    errs = _check(tmp_path, stale=True, n_inner=3)
    assert errs, "a stale-state run must not pass"
    assert any("replay" in e or "PREVIOUS-STATE" in e for e in errs), errs


def test_checker_rejects_a_served_vector_that_is_not_the_recorded_one(tmp_path):
    """Blocker 3: pred_raw recorded one vector, the operator consumed
    another."""
    errs = _check(tmp_path, served_mismatch=True)
    assert any("differs from the vector recorded" in e for e in errs), errs


def test_checker_rejects_a_missing_served_record(tmp_path):
    tag, d, W = _mk_run(tmp_path)
    CHK._FJR_GRAPH_CACHE["W"] = W
    d.pop("fj_served_used")
    errs = CHK.check_fjr(tag, d["config"], d, d["op_raw"].float(),
                         d["pred_raw"].float(), d["innate"].float(),
                         expect_rounds=4, expect_agents=N)
    assert any("no fj_served_used" in e for e in errs), errs


def test_checker_rejects_arm_label_swap(tmp_path):
    """b0 tag carrying forward-KL semantics, and vice versa."""
    errs = _check(tmp_path, arm="b0", style="sft_kl", kl_beta=1.0)
    assert any("training_style" in e or "kl_beta" in e for e in errs), errs
    errs = _check(tmp_path, arm="b1", style="sft", kl_beta=0.0)
    assert any("training_style" in e or "kl_beta" in e for e in errs), errs


def test_checker_rejects_a_reference_adapter_on_b1(tmp_path):
    tag, d, W = _mk_run(tmp_path, arm="b1")
    CHK._FJR_GRAPH_CACHE["W"] = W
    d["config"]["kl_ref_adapter"] = "/some/adapter"
    errs = CHK.check_fjr(tag, d["config"], d, d["op_raw"].float(),
                         d["pred_raw"].float(), d["innate"].float(),
                         expect_rounds=4, expect_agents=N)
    assert any("reference adapter" in e for e in errs), errs


def test_checker_rejects_reverse_kl_on_b1(tmp_path):
    tag, d, W = _mk_run(tmp_path, arm="b1")
    CHK._FJR_GRAPH_CACHE["W"] = W
    d["config"]["kl_direction"] = "reverse"
    errs = CHK.check_fjr(tag, d["config"], d, d["op_raw"].float(),
                         d["pred_raw"].float(), d["innate"].float(),
                         expect_rounds=4, expect_agents=N)
    assert any("FORWARD" in e for e in errs), errs


def test_checker_rejects_wrong_hardware(tmp_path):
    errs = _check(tmp_path, gpu="NVIDIA A100-SXM4-80GB")
    assert any("gpu" in e for e in errs), errs


def test_checker_rejects_a_truncated_horizon(tmp_path):
    tag, d, W = _mk_run(tmp_path, rounds=4)
    CHK._FJR_GRAPH_CACHE["W"] = W
    errs = CHK.check_fjr(tag, d["config"], d, d["op_raw"].float(),
                         d["pred_raw"].float(), d["innate"].float(),
                         expect_rounds=30, expect_agents=N)
    assert any("!= (30," in e for e in errs), errs


def test_checker_rejects_a_corrupted_trajectory(tmp_path):
    errs = _check(tmp_path, corrupt=True)
    assert any("replay the declared recurrence" in e for e in errs), errs


def test_checker_rejects_non_eval_mode_serving(tmp_path):
    tag, d, W = _mk_run(tmp_path)
    CHK._FJR_GRAPH_CACHE["W"] = W
    d["config"]["serve_eval_mode"] = False
    errs = CHK.check_fjr(tag, d["config"], d, d["op_raw"].float(),
                         d["pred_raw"].float(), d["innate"].float(),
                         expect_rounds=4, expect_agents=N)
    assert any("eval mode" in e for e in errs), errs


def test_checker_rejects_parse_failures(tmp_path):
    tag, d, W = _mk_run(tmp_path)
    CHK._FJR_GRAPH_CACHE["W"] = W
    d["trajectory"][2]["parse_fail"] = 0.01
    errs = CHK.check_fjr(tag, d["config"], d, d["op_raw"].float(),
                         d["pred_raw"].float(), d["innate"].float(),
                         expect_rounds=4, expect_agents=N)
    assert any("parse_fail" in e for e in errs), errs


def test_checker_rejects_wrong_round_zero_labels(tmp_path):
    tag, d, W = _mk_run(tmp_path)
    CHK._FJR_GRAPH_CACHE["W"] = W
    d["train_y_raw"][0] += 0.1
    errs = CHK.check_fjr(tag, d["config"], d, d["op_raw"].float(),
                         d["pred_raw"].float(), d["innate"].float(),
                         expect_rounds=4, expect_agents=N)
    assert any("round-0 SFT labels" in e for e in errs), errs


def test_checker_rejects_labels_that_are_not_the_previous_population(tmp_path):
    tag, d, W = _mk_run(tmp_path)
    CHK._FJR_GRAPH_CACHE["W"] = W
    d["train_y_raw"][2] += 0.1
    errs = CHK.check_fjr(tag, d["config"], d, d["op_raw"].float(),
                         d["pred_raw"].float(), d["innate"].float(),
                         expect_rounds=4, expect_agents=N)
    assert any("post-FJ population" in e for e in errs), errs


# ---------------------------------------------------- generator wiring

def test_grid_is_twelve_cells_six_models_two_arms():
    rows = GEN.fjr_rows()
    assert len(rows) == 12
    tags = [r.split(",")[0] for r in rows]
    assert len(set(tags)) == 12
    assert len({ANA.model_of(t) for t in tags}) == 6
    assert sorted({ANA.arm_of(t) for t in tags}) == ["b0", "b1"]


def test_tags_carry_no_gate_tokens():
    """FJ applies no AI or bounded-confidence gate, so _ea/_es would name
    something that never ran."""
    for r in GEN.fjr_rows() + GEN.fjr_smoke_rows() + GEN.fjr_frozen_rows():
        t = r.split(",")[0]
        assert "_ea" not in t and "_es" not in t, t


def test_tags_encode_the_full_configuration():
    t = GEN.fjr_tag("qwen7b", "b0")
    for token in ("qwen7b", "_b0_", "beta0p5", "alpha0p5", "_in1_",
                  "_k1_", "_s0_", "_r30"):
        assert token in t, (token, t)


def test_beta1_key_is_disjoint_from_the_core_wave():
    """Same models and arms at a different beta: the tags MUST differ or
    submitting both double-queues one set of run dirs."""
    core = {r.split(",")[0] for r in GEN.fjr_rows()}
    b1 = {r.split(",")[0] for r in GEN.fjr_rows(beta=1.0)}
    assert core & b1 == set()
    assert len(b1) == 12


def test_smoke_is_one_short_forward_kl_qwen_cell():
    rows = GEN.fjr_smoke_rows()
    assert len(rows) == 1
    t = rows[0].split(",")[0]
    assert "qwen7b" in t and "_b1_" in t and t.endswith("r3smoke")
    assert rows[0].split(",")[1].strip() == "sft_kl"


def test_frozen_extraction_covers_exactly_the_unreusable_model():
    rows = GEN.fjr_frozen_rows()
    assert len(rows) == 1
    assert "mistral7b" in rows[0].split(",")[0]
    assert "Mistral-7B-Instruct-v0.3" in rows[0]
    # and the five reused ones are named, not silently assumed
    assert set(GEN.FJR_FROZEN_REUSE) | set(GEN.FJR_FROZEN_NEW) == set(GEN.FJR_MODELS)


def test_frozen_tag_carries_no_fj_parameters():
    """It is an extraction, not an FJ run; the replay is local."""
    t = GEN.fjr_frozen_tag("mistral7b")
    for token in ("beta", "alpha", "_in", "_k1"):
        assert token not in t, (token, t)


def test_sub_pins_the_exact_h100_sku_and_the_wu1_operator():
    sub = GEN.fjr_sub(GEN.FJR_KEY, GEN.fjr_rows())
    assert f'CUDADeviceName == "{GEN.FJR_H100}"' in sub
    env = next(ln for ln in sub.splitlines() if ln.startswith("environment"))
    assert "POP_MODEL=fj" in env and "FJ_UPDATE_VERSION=wu1" in env
    assert "AI_GATE_MODE" not in env and "PEER_GATE_MODE" not in env
    assert "KL_REF_ADAPTER" not in env


def test_on_disk_configs_match_the_generator():
    for key, rows in ((GEN.FJR_KEY, GEN.fjr_rows()),
                      (GEN.FJR_SMOKE_KEY, GEN.fjr_smoke_rows()),
                      (GEN.FJR_FROZEN_KEY, GEN.fjr_frozen_rows()),
                      (GEN.FJR_BETA1_KEY, GEN.fjr_rows(beta=1.0))):
        p = CONDOR / f"configs_pofd_{key}.txt"
        assert p.read_text() == "\n".join(rows) + "\n", key


def test_submit_script_knows_the_fj_keys():
    s = (CONDOR / "submit_pofd_sweep.sh").read_text()
    assert "fj_robustness|fj_robustness_smoke|fj_robustness_frozen|" \
           "fj_robustness_beta1)" in s
    # the optional beta=1 key must not ride any umbrella
    for ln in s.splitlines():
        if "TARGETS=" in ln and "fj_robustness_beta1" in ln:
            assert 'TARGETS="$WHAT"' in ln, ln


# ---------------------------------------------------- analyzer gating

def test_analyzer_hard_fails_on_an_incomplete_grid(tmp_path):
    with pytest.raises(SystemExit) as e:
        ANA.analyse([tmp_path], tmp_path / "out")
    assert "HARD FAIL" in str(e.value)


def test_model_and_arm_parsing_handles_underscored_slugs():
    """qwen3_8b and olmo3_7b contain underscores; a positional split
    returns '8b'."""
    t = GEN.fjr_tag("qwen3_8b", "b1")
    assert ANA.model_of(t) == "qwen3_8b"
    assert ANA.arm_of(t) == "b1"


def test_runner_refuses_gates_under_wu1():
    """FJ has no gates; accepting one would let a tag claim something the
    operator never applies."""
    src = (PIPE / "run_pokec_gated_lm.py").read_text()
    assert 'FJ_UPDATE_VERSION=wu1 has NO gates' in src
    assert 'os.environ.get("FJ_UPDATE_VERSION", "legacy")' in src


def test_runner_applies_w_plat_to_the_fj_beta():
    """Blocker 1: the archived path drops W_PLAT entirely."""
    src = (PIPE / "run_pokec_gated_lm.py").read_text()
    assert "fj_beta = (w_plat * plat_sus_eff).clamp(0.0, 1.0)" in src


def test_runner_feeds_the_recorded_vector_to_the_operator():
    """Blocker 3: one generation per round, and the operator consumes the
    vector that pred_raw records."""
    src = (PIPE / "run_pokec_gated_lm.py").read_text()
    assert "world.run_wu(last_preds" in src
    assert "fj_served_used.append(last_preds" in src


def test_production_agent_count_is_still_gated():
    """The toy tests pass expect_agents=N; production must stay at 723."""
    import inspect
    sig = inspect.signature(CHK.check_fjr)
    assert sig.parameters["expect_agents"].default == 723
    assert sig.parameters["expect_rounds"].default == 30
