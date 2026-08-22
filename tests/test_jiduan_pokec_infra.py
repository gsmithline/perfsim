"""Tests for the Jiduan Wu / Pokec replication infrastructure (2026-08-22).

NO MODEL IS EVER LOADED. The FJ operator is linear and fully specified,
so every claim this wave makes about it is checkable in closed form on a
small hand-built graph, and the checker's failure modes are checkable by
building a run that is valid except for one defect.

THE CHECKER TESTS ARE SABOTAGE TESTS. Each builds a structurally valid
artifact, injects exactly one defect, and asserts the checker names it.
A gate nobody has watched fail is not a gate -- and the defects here are
not hypothetical: the dataset file is called peer_sus and holds alpha,
the observed set is a prefix so an off-by-one mask still has the right
size, and a converged inner loop makes a stale-state start invisible in
the final state.

THE TOY SIZE IS PARAMETERIZED ON PURPOSE. The tests run 12 agents (9
observed, 3 held out) so a ring graph and a 3-step inner loop stay
readable; production stays at 2163 / 1730 / 433, and
test_production_sizes_are_still_gated pins the defaults so the toy can
never quietly become the contract.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import math
import os
import subprocess
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


CHK = _load("chk_wu", PIPE / "check_jiduan_pokec.py")
WUC = _load("wuc_t", PIPE / "wu_context.py")
ANA = _load("ana_wu", PIPE / "analyze_jiduan_pokec.py")
GEN = _load("gen_wu", CONDOR / "gen_pofd_sweep.py")

N = 12
N_OBS = 9
N_HELD = 3
INNER = 3
ROUNDS = 4
GPU = "NVIDIA H100 80GB HBM3"
ROUTE_FRAC = 1.0 / 3.0     # 3 of the 9 observed agents
ROUTE_SEED = 7
ROUTE_VALUE = 0.5


# --------------------------------------------------------- the toy world

def _toy_env(alpha=None, beta=None):
    """A 12-agent ring: row-stochastic, everybody has neighbours, so
    alpha genuinely mixes and nothing is trivially isolated. alpha and
    beta are per agent and all distinct -- a checker that only looks at
    means would pass a scalar, and this makes that visible."""
    adj = torch.zeros(N, N)
    for i in range(N):
        adj[i, (i - 1) % N] = 1.0
        adj[i, (i + 1) % N] = 1.0
    W = adj / adj.sum(dim=1, keepdim=True)
    innate = torch.linspace(0.08, 0.92, N)
    a = torch.tensor([0.90, 0.85, 0.80, 0.95, 0.88, 0.92,
                      0.83, 0.87, 0.91, 0.79, 0.94, 0.86])
    b = torch.tensor([0.70, 0.62, 0.81, 0.75, 0.66, 0.88,
                      0.72, 0.69, 0.90, 0.64, 0.77, 0.83])
    return {"innate": innate, "alpha_raw": a if alpha is None else alpha,
            "beta_raw": b if beta is None else beta, "W": W,
            "n_nodes": N, "n_edges": N, "n_observed": N_OBS,
            "n_heldout": N_HELD}


def _sha(t):
    return CHK._sha_t(t)


def _ctx_log(mode, k, d_depth, hist, x_entry, rounds, agents):
    """A wu_ctx_log payload in wu_context's OWN schema (one
    round_log_line per round), built through wu_context so the tests and
    the runner cannot drift apart on what a context record looks like."""
    out = []
    for t in range(rounds):
        entries = []
        for i in agents:
            ids, vals = [], []
            if mode == "observed_context":
                ids = list(range(k))
                vals = [float(x_entry[t, j]) for j in ids]
                text = " ".join(WUC.VALUE_FMT.format(v) for v in vals)
            elif mode in ("prediction_history", "expressed_history"):
                vals = [float(hist[s_, i])
                        for s_ in range(max(0, t - d_depth), t)]
                ids = [int(i)] * len(vals)
                text = " ".join(WUC.VALUE_FMT.format(v) for v in vals)
            else:
                text = ""
            entries.append({"agent": int(i), "ids": ids, "values": vals,
                            "text": text,
                            "history_source": WUC.HISTORY_SOURCE[mode],
                            "mode": mode,
                            "extension": bool(WUC.is_extension(mode))})
        out.append(WUC.round_log_line(t, mode, entries, k=k, d=d_depth))
    return out


def _mk_run(*, arm="b0", ca=1.0, cb=1.0, rounds=ROUNDS, inner=INNER,
            seed=0, route=None, env=None, model="qwen7b", stale=False,
            invert_alpha=False, passthrough="live", pred_on_observed=False,
            train_heldout=False, ctx_agents=(9, 10, 11),
            nan_pred_on_observed=False):
    """A structurally valid Wu/Pokec run; defects arrive by kwarg."""
    env = env or _toy_env()
    innate = env["innate"].clone()
    W = env["W"]
    alpha = env["alpha_raw"] * ca
    beta = env["beta_raw"] * cb
    rng = np.random.default_rng(11)
    O = torch.zeros(N, dtype=torch.bool)
    O[:N_OBS] = True
    # ROUTING is a SOURCE injection at OBSERVED agents: the runner
    # rewrites their innate before anything reads it, and the control
    # twin simply runs at frac 0.
    route_frac = ROUTE_FRAC if route == "T" else 0.0
    cohort = CHK.routing_cohort(route_frac, ROUTE_SEED, N_OBS, N) \
        if route == "T" else torch.zeros(0, dtype=torch.long)
    if route == "T":
        innate = innate.clone()
        innate[cohort] = ROUTE_VALUE

    preds, serveds, ops, x0s, u1s, ys, idxs = [], [], [], [], [], [], []
    prev = None
    for t in range(rounds):
        p = torch.tensor(rng.uniform(0.1, 0.9, N), dtype=torch.float32)
        x_entry = innate if prev is None else prev
        sv = p.clone()
        sv[O] = (x_entry[O] if passthrough == "live" else innate[O])
        if pred_on_observed:
            sv[0] = p[0]                              # the classic defect
        x0 = (1.0 - beta) * innate + beta * sv
        u = (prev.clone() if (stale and prev is not None) else x0.clone())
        u1 = None
        for _ in range(inner):
            if invert_alpha:
                u = alpha * x0 + (1.0 - alpha) * (W @ u)
            else:
                u = (1.0 - alpha) * x0 + alpha * (W @ u)
            if u1 is None:
                u1 = u.clone()
        preds.append(p)
        serveds.append(sv)
        x0s.append(x0)
        u1s.append(u1)
        ops.append(u)
        idx = torch.arange(N_OBS)
        if train_heldout:
            idx = torch.cat([torch.arange(N_OBS - 1),
                             torch.tensor([N - 1])])
        idxs.append(idx)
        ys.append(x_entry[idx].clone())
        prev = u
    op = torch.stack(ops)
    pred = torch.stack(preds)
    served = torch.stack(serveds)
    if nan_pred_on_observed:
        # the runner's "the model was never asked" marker: absence has to
        # be representable, or "not asked" and "answered x_O" are the
        # same artifact
        pred = pred.clone()
        pred[:, O] = float("nan")

    style, kl, mode, k_ctx, d_ctx = CHK.WU_ARM_SEMANTICS[arm]
    cfg = {
        "dataset": "pokec", "pop_model": "fj", "fj_update_version": "wu1",
        "fj_inner_steps": inner, "n_rounds": rounds, "seed": seed,
        "fj_peer_source": "dataset", "fj_platform_source": "dataset",
        "fj_alpha_scale": ca, "fj_beta_scale": cb,
        "fj_observed_passthrough": True,
        "fj_alpha_raw_sha256": _sha(env["alpha_raw"]),
        "fj_beta_raw_sha256": _sha(env["beta_raw"]),
        "fj_alpha_realized_sha256": _sha(alpha),
        "fj_beta_realized_sha256": _sha(beta),
        "fj_alpha_realized_mean": float(alpha.mean()),
        "fj_beta_realized_mean": float(beta.mean()),
        "fj_peer_sus_sha256": _sha(1.0 - alpha),
        "fj_peer_sus_mean": float((1.0 - alpha).mean()),
        "fj_graph_sha256": _sha(W),
        "wu_icl_mode": mode, "wu_icl_k": k_ctx, "wu_icl_d": d_ctx,
        "n_labeled": N_OBS, "training_style": style, "kl_beta": kl,
        "kl_direction": "forward", "kl_ref_adapter": "",
        "use_lora": 1 if arm in CHK.WU_TRAINED_ARMS else 0,
        "fresh_each_round": arm in CHK.WU_TRAINED_ARMS,
        "serve_eval_mode": True, "hardware": {"gpu_name": GPU},
        "data_regime": "replace", "run_mode": "loop", "anchor_mode": "fixed",
        "eps": 0.0, "eps_ai": 0.0, "gamma_bias": 0.0, "canary_delta": 0.0,
        "pristine_frac": 0.0, "replay_frac": 0.0, "ab_sweeps": 1,
        "pop_reset": False,
    }
    d = {
        "config": cfg, "op_raw": op, "model_pred_raw": pred,
        "served_raw": served, "fj_x_init_raw": torch.stack(x0s),
        "fj_u1_raw": torch.stack(u1s), "innate": innate,
        "observed_mask": O, "train_idx_raw": torch.stack(idxs),
        "train_y_raw": torch.stack(ys),
        "trajectory": [{"t": t, "parse_fail": 0.0} for t in range(rounds)],
    }
    if route is not None:
        cfg["routing_treat_frac"] = route_frac
        cfg["routing_treat_seed"] = ROUTE_SEED
        cfg["routing_treat_value"] = ROUTE_VALUE
        if route == "T":
            cfg["routing_treat_idx_sha256"] = WUC.idx_sha256(cohort.numpy())
            cfg["routing_treat_n"] = int(cohort.numel())
    x_entry_all = torch.cat([innate.unsqueeze(0), op[:-1]], dim=0)
    if mode != "none":
        hist = {"prediction_history": served,
                "expressed_history": op}.get(mode)
        d["wu_ctx_log"] = _ctx_log(mode, k_ctx, d_ctx, hist, x_entry_all,
                                   rounds, ctx_agents)
    tag = GEN.wu_tag(model, arm, ca=ca, cb=cb, seed=seed, rounds=rounds,
                     inner=inner, route=route)
    return tag, d, env


def _check(tag, d, env, **kw):
    kw.setdefault("expect_rounds", ROUNDS)
    kw.setdefault("expect_n", N)
    kw.setdefault("expect_observed", N_OBS)
    kw.setdefault("expect_heldout", N_HELD)
    kw.setdefault("expect_inner", INNER)
    return CHK.check_jiduan_run(tag, d, env=env, **kw)


def _run(**kw):
    tag, d, env = _mk_run(**kw)
    return _check(tag, d, env)


# ------------------------------------------------------- the clean paths

def test_clean_sft_run_passes():
    assert _run() == []


def test_clean_forward_kl_run_passes():
    assert _run(arm="b1") == []


def test_clean_frozen_run_passes():
    assert _run(arm="frz") == []


def test_clean_observed_context_run_passes():
    assert _run(arm="octx8") == []


def test_clean_prediction_history_run_passes():
    assert _run(arm="phist8") == []


def test_clean_routing_twins_pass():
    assert _run(route="T") == []
    assert _run(route="C") == []


def test_clean_run_at_a_zero_scale_passes():
    """c_alpha = 0 leaves the population at its anchor and c_beta = 0
    removes the platform; both are real cells of the environment grid and
    must not trip a gate written for the c = 1 case."""
    assert _run(ca=0.0) == []
    assert _run(cb=0.0) == []


# ------------------------------------ dataset, node order, vector hashes

def test_rejects_a_run_whose_innate_is_not_the_dataset():
    tag, d, env = _mk_run()
    d["innate"] = d["innate"] + 0.01
    errs = _check(tag, d, env)
    assert any("not the Pokec innate vector" in e for e in errs), errs


def test_rejects_a_wrong_alpha_vector_hash():
    tag, d, env = _mk_run()
    d["config"]["fj_alpha_raw_sha256"] = "0" * 64
    errs = _check(tag, d, env)
    assert any("fj_alpha_raw_sha256" in e for e in errs), errs


def test_rejects_a_realized_hash_that_is_not_raw_times_the_scale():
    """The scale in the config must be the scale that was applied -- a
    tag can say c_alpha=0.5 while the operator ran 1."""
    tag, d, env = _mk_run(ca=0.5)
    d["config"]["fj_alpha_realized_sha256"] = _sha(env["alpha_raw"])
    errs = _check(tag, d, env)
    assert any("fj_alpha_realized_sha256" in e for e in errs), errs


def test_rejects_a_realized_mean_that_disagrees():
    tag, d, env = _mk_run()
    d["config"]["fj_beta_realized_mean"] = 0.5
    errs = _check(tag, d, env)
    assert any("fj_beta_realized_mean" in e for e in errs), errs


def test_rejects_a_non_pokec_dataset():
    tag, d, env = _mk_run()
    d["config"]["dataset"] = "movielens"
    errs = _check(tag, d, env)
    assert any("expected 'pokec'" in e for e in errs), errs


def test_production_hashes_are_pinned_and_checked():
    """At production size the canonical Pokec hashes apply, and a
    tampered environment is refused by name."""
    env = _toy_env()
    tag, d, _ = _mk_run(env=env)
    errs = _check(tag, d, env, pin_pokec=True)
    assert any("canonical Pokec" in e for e in errs), errs


# ---------------------------------------------- N, |O|, |U| and the split

def test_rejects_the_wrong_population_size():
    tag, d, env = _mk_run()
    errs = _check(tag, d, env, expect_n=N + 1, expect_observed=N_OBS,
                  expect_heldout=N + 1 - N_OBS)
    assert any("agents, expected" in e for e in errs), errs


def test_rejects_the_wrong_observed_count():
    tag, d, env = _mk_run()
    m = torch.zeros(N, dtype=torch.bool)
    m[:N_OBS - 1] = True
    d["observed_mask"] = m
    errs = _check(tag, d, env)
    assert any("|O| =" in e for e in errs), errs


def test_rejects_an_observed_set_that_is_not_the_dataset_prefix():
    """Right size, wrong set. The observed set is a PREFIX by
    construction (innate = y_label ++ y_unlabel), so a scattered mask of
    the same size is a different experiment."""
    tag, d, env = _mk_run()
    m = torch.zeros(N, dtype=torch.bool)
    m[torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 11])] = True
    d["observed_mask"] = m
    errs = _check(tag, d, env)
    assert any("not the first" in e for e in errs), errs


def test_rejects_an_inconsistent_split():
    tag, d, env = _mk_run()
    errs = _check(tag, d, env, expect_observed=N_OBS, expect_heldout=1)
    assert any("!= N" in e for e in errs), errs


# ------------------------------------------- operator, K and the horizon

def test_rejects_the_legacy_fj_operator():
    tag, d, env = _mk_run()
    d["config"]["fj_update_version"] = "legacy"
    errs = _check(tag, d, env)
    assert any("LEGACY FJ operator" in e for e in errs), errs


def test_rejects_the_wrong_inner_step_count():
    tag, d, env = _mk_run()
    d["config"]["fj_inner_steps"] = 1
    errs = _check(tag, d, env)
    assert any("expected K" in e for e in errs), errs
    assert any("replay the declared recurrence" in e for e in errs), errs


def test_rejects_the_wrong_outer_horizon():
    tag, d, env = _mk_run(rounds=ROUNDS)
    errs = _check(tag, d, env, expect_rounds=ROUNDS + 1)
    assert any("wrong prediction/state lengths" in e for e in errs), errs


def test_rejects_a_config_horizon_that_disagrees_with_the_artifact():
    tag, d, env = _mk_run()
    d["config"]["n_rounds"] = ROUNDS + 7
    errs = _check(tag, d, env)
    assert any("horizon under test" in e for e in errs), errs


# ------------------------------------- scalar parameters and the complement

def test_rejects_a_scalar_alpha_in_the_exact_heterogeneous_key():
    env = _toy_env(alpha=torch.full((N,), 0.89))
    tag, d, _ = _mk_run(env=env)
    errs = _check(tag, d, env)
    assert any("realized alpha is CONSTANT" in e for e in errs), errs


def test_rejects_a_scalar_beta_in_the_exact_heterogeneous_key():
    env = _toy_env(beta=torch.full((N,), 0.889))
    tag, d, _ = _mk_run(env=env)
    errs = _check(tag, d, env)
    assert any("realized beta is CONSTANT" in e for e in errs), errs


def test_rejects_a_scalar_alpha_recorded_next_to_a_dataset_source():
    tag, d, env = _mk_run()
    d["config"]["fj_peer_alpha"] = 0.9
    errs = _check(tag, d, env)
    assert any("two sources for one parameter" in e for e in errs), errs


def test_rejects_a_stubbornness_that_is_not_one_minus_alpha():
    """THE trap this dataset sets. hetero_peer_sus2163.pkl is named
    peer_sus and holds alpha; FJWorld.peer_sus is 1 - alpha."""
    tag, d, env = _mk_run()
    d["config"]["fj_peer_sus_sha256"] = _sha(env["alpha_raw"])
    errs = _check(tag, d, env)
    assert any("not 1 - alpha elementwise" in e for e in errs), errs


def test_rejects_a_run_that_actually_ran_the_inverted_convention():
    """The behavioural version: the config is impeccable and the
    TRAJECTORY is the one you get from passing alpha in as the anchor
    weight. Every number is finite, ordered and wrong."""
    tag, d, env = _mk_run(invert_alpha=True)
    errs = _check(tag, d, env)
    assert any("INVERTED alpha convention" in e for e in errs), errs


def test_the_two_alpha_conventions_are_actually_distinguishable():
    """Guard against a vacuous test: at this alpha the two conventions
    must give visibly different trajectories, or the check above proves
    nothing."""
    env = _toy_env()
    innate, W, a = env["innate"], env["W"], env["alpha_raw"]
    x0 = 0.5 * innate + 0.5 * torch.full((N,), 0.3)
    ok, inv = x0.clone(), x0.clone()
    for _ in range(INNER):
        ok = (1 - a) * x0 + a * (W @ ok)
        inv = a * x0 + (1 - a) * (W @ inv)
    assert float((ok - inv).abs().max()) > 1e-2


# -------------------------------------------------- observed passthrough

def test_rejects_a_model_prediction_replacing_an_observed_value():
    tag, d, env = _mk_run(pred_on_observed=True)
    errs = _check(tag, d, env)
    hit = [e for e in errs if "observed passthrough violated" in e]
    assert hit, errs
    assert "model prediction REPLACED an observed opinion" in hit[0]


def test_rejects_the_wrong_passthrough_reading_and_names_the_other():
    """A run built on the static reading fails the live one -- and the
    message reports the distance to BOTH so the reader is told which
    reading it is closest to instead of guessing."""
    tag, d, env = _mk_run(passthrough="innate")
    errs = _check(tag, d, env, passthrough="live")
    hit = [e for e in errs if "observed passthrough violated" in e]
    assert hit, errs
    assert "'innate' reading" in hit[0]
    assert _check(tag, d, env, passthrough="innate") == []


def test_rejects_passthrough_switched_off_in_the_config():
    tag, d, env = _mk_run()
    d["config"]["fj_observed_passthrough"] = 0
    errs = _check(tag, d, env)
    assert any("fj_observed_passthrough" in e for e in errs), errs


# ---------------------------------------------------- training on O only

def test_rejects_training_on_a_held_out_agent():
    tag, d, env = _mk_run(train_heldout=True)
    errs = _check(tag, d, env)
    assert any("HELD-OUT agent" in e for e in errs), errs


def test_rejects_n_labeled_that_is_not_the_observed_set():
    tag, d, env = _mk_run()
    d["config"]["n_labeled"] = N
    errs = _check(tag, d, env)
    assert any("n_labeled" in e for e in errs), errs


def test_rejects_labels_that_are_not_the_entering_population():
    tag, d, env = _mk_run()
    d["train_y_raw"][2] += 0.1
    errs = _check(tag, d, env)
    assert any("SFT labels are not the opinions" in e for e in errs), errs


def test_rejects_round_zero_labels_that_are_not_innate():
    tag, d, env = _mk_run()
    d["train_y_raw"][0] += 0.1
    errs = _check(tag, d, env)
    assert any("round-0 SFT labels" in e for e in errs), errs


# ------------------------------------------- Deffuant / AI-gate leftovers

@pytest.mark.parametrize("key,bad", [
    ("ai_gate_mode", "all_open"), ("peer_gate_mode", "all_open"),
    ("eps", 0.2), ("eps_ai", 0.4), ("gamma_bias", 1.5),
    ("canary_delta", 0.1), ("ab_sweeps", 4), ("pop_reset", True),
    ("data_regime", "accumulate"), ("pristine_frac", 0.5),
    ("replay_frac", 0.25), ("anchor_mode", "moving"),
])
def test_rejects_hidden_deffuant_or_gate_settings(key, bad):
    tag, d, env = _mk_run()
    d["config"][key] = bad
    errs = _check(tag, d, env)
    assert any(key in e for e in errs), (key, errs)


# --------------------------------------- FJ initialisation, u^(1), u^(K)

def test_rejects_a_stale_previous_state_initialisation():
    tag, d, env = _mk_run(stale=True)
    errs = _check(tag, d, env)
    assert any("u^(1)" in e for e in errs), errs
    assert any("PREVIOUS-POPULATION start" in e for e in errs), errs


def test_rejects_a_missing_u1_with_the_convergence_explanation():
    tag, d, env = _mk_run()
    d.pop("fj_u1_raw")
    errs = _check(tag, d, env)
    hit = [e for e in errs if "no fj_u1_raw" in e]
    assert hit, errs
    assert "converges" in hit[0].lower() and "u^(1)" in hit[0]


def test_rejects_a_stored_u1_that_does_not_match_the_replay():
    tag, d, env = _mk_run()
    d["fj_u1_raw"][1] += 0.05
    errs = _check(tag, d, env)
    assert any("u^(1)" in e for e in errs), errs


def test_rejects_a_final_state_that_does_not_match_the_replay():
    tag, d, env = _mk_run()
    d["op_raw"][2] += 0.05
    errs = _check(tag, d, env)
    assert any("replay the declared recurrence" in e for e in errs), errs


def test_rejects_an_anchor_that_is_not_the_declared_blend():
    tag, d, env = _mk_run()
    d["fj_x_init_raw"][0] += 0.05
    errs = _check(tag, d, env)
    assert any("fj_x_init_raw is not" in e for e in errs), errs


def test_u1_still_pins_the_start_where_the_final_state_cannot():
    """At production alpha and K the inner loop contracts by ~.89^100, so
    two different starts land on the same u^(K) and only u^(1) tells them
    apart. This is why fj_u1_raw is mandatory."""
    env = _toy_env()
    innate, W, a = env["innate"], env["W"], env["alpha_raw"]
    x0 = 0.4 * innate + 0.6 * torch.linspace(0.2, 0.8, N)
    other = torch.linspace(0.9, 0.1, N)
    fa, fb, ua, ub = x0.clone(), other.clone(), None, None
    for _ in range(100):
        fa = (1 - a) * x0 + a * (W @ fa)
        fb = (1 - a) * x0 + a * (W @ fb)
        if ua is None:
            ua, ub = fa.clone(), fb.clone()
    assert float((fa - fb).abs().max()) < 1e-4
    assert float((ua - ub).abs().max()) > 1e-2


# --------------------------------------- hygiene: finite, parsed, shaped

def test_rejects_non_finite_predictions_on_the_held_out_set():
    tag, d, env = _mk_run()
    d["model_pred_raw"][1, N - 1] = float("nan")
    errs = _check(tag, d, env)
    assert any("non-finite entries on the HELD-OUT set" in e
               for e in errs), errs


def test_rejects_non_finite_population_states():
    tag, d, env = _mk_run()
    d["op_raw"][1, 3] = float("nan")
    errs = _check(tag, d, env)
    assert any("op_raw has non-finite" in e for e in errs), errs


def test_accepts_nan_predictions_on_the_observed_set():
    """The runner marks 'the model was never asked' with NaN on O, and it
    has to be representable: otherwise 'not asked' and 'answered exactly
    x_O' are the same artifact. NaN is legal THERE and nowhere else."""
    assert _run(nan_pred_on_observed=True) == []


def test_rejects_an_infinity_on_the_observed_set():
    """NaN means 'unasked'. An infinity means the serving path broke, and
    the two must not be confused."""
    tag, d, env = _mk_run(nan_pred_on_observed=True)
    d["model_pred_raw"][0, 0] = float("inf")
    errs = _check(tag, d, env)
    assert any("nor NaN" in e for e in errs), errs


def test_rejects_parse_failures():
    tag, d, env = _mk_run()
    d["trajectory"][2]["parse_fail"] = 0.01
    errs = _check(tag, d, env)
    assert any("parse_fail" in e for e in errs), errs


def test_rejects_wrong_prediction_lengths():
    tag, d, env = _mk_run()
    d["model_pred_raw"] = d["model_pred_raw"][:, :N - 1]
    errs = _check(tag, d, env)
    assert any("wrong prediction/state lengths" in e for e in errs), errs


def test_rejects_predictions_outside_the_unit_interval():
    tag, d, env = _mk_run()
    d["model_pred_raw"][0, 10] = 1.4
    errs = _check(tag, d, env)
    assert any("leaves [0, 1]" in e for e in errs), errs


def test_rejects_serving_outside_eval_mode():
    tag, d, env = _mk_run()
    d["config"]["serve_eval_mode"] = False
    errs = _check(tag, d, env)
    assert any("eval mode" in e for e in errs), errs


# ------------------------------------------------- missing fields, named

def test_a_missing_config_field_is_named_not_a_crash():
    tag, d, env = _mk_run()
    d["config"].pop("fj_alpha_scale")
    errs = _check(tag, d, env)
    assert any("missing required field(s): fj_alpha_scale" in e
               for e in errs), errs


def test_a_missing_tensor_is_named_not_a_crash():
    tag, d, env = _mk_run()
    d.pop("served_raw")
    errs = _check(tag, d, env)
    assert any("missing required tensor(s): served_raw" in e
               for e in errs), errs


def test_every_required_field_is_individually_load_bearing():
    """Drop each required field in turn: none may pass silently, and none
    may raise. Agent-to-agent interface drift is exactly how a checker
    quietly stops checking."""
    for k in CHK.WU_REQUIRED_CFG:
        tag, d, env = _mk_run()
        d["config"].pop(k)
        errs = _check(tag, d, env)
        assert any(k in e for e in errs), k
    for k in CHK.WU_REQUIRED_ART:
        tag, d, env = _mk_run()
        d.pop(k)
        errs = _check(tag, d, env)
        assert any(k in e for e in errs), k


# ---------------------------------------------------------- arm semantics

def test_rejects_an_arm_label_swap():
    tag, d, env = _mk_run(arm="b0")
    d["config"]["training_style"] = "sft_kl"
    d["config"]["kl_beta"] = 1.0
    errs = _check(tag, d, env)
    assert any("training_style" in e or "kl_beta" in e for e in errs), errs


def test_rejects_reverse_kl_on_a_regularized_arm():
    tag, d, env = _mk_run(arm="b1")
    d["config"]["kl_direction"] = "reverse"
    errs = _check(tag, d, env)
    assert any("FORWARD KL" in e for e in errs), errs


def test_rejects_an_icl_mode_that_disagrees_with_the_arm_token():
    tag, d, env = _mk_run(arm="phist8")
    d["config"]["wu_icl_mode"] = "observed_context"
    errs = _check(tag, d, env)
    assert any("but the run recorded" in e for e in errs), errs


def test_rejects_a_carried_adapter_in_a_primary_sft_arm():
    """FRESH-ADAPTER SEMANTICS. A carried adapter turns 50 rounds of
    retraining into one long fine-tune, and the trajectory afterwards
    cannot tell you which one happened."""
    for arm in CHK.WU_TRAINED_ARMS:
        tag, d, env = _mk_run(arm=arm)
        d["config"]["fresh_each_round"] = False
        errs = _check(tag, d, env)
        assert any("rebuilt every round" in e for e in errs), (arm, errs)


def test_rejects_lora_in_a_frozen_arm():
    tag, d, env = _mk_run(arm="frz")
    d["config"]["use_lora"] = 1
    errs = _check(tag, d, env)
    assert any("FROZEN arm" in e for e in errs), errs


def test_checker_and_generator_agree_on_what_an_arm_means():
    """The two tables are deliberately independent witnesses; this is
    what catches them drifting apart."""
    assert set(CHK.WU_ARM_SEMANTICS) == set(GEN.WU_ARM_COLS)
    for arm, (style, kl, mode, k, dep) in CHK.WU_ARM_SEMANTICS.items():
        col = GEN.WU_ARM_COLS[arm]
        assert col["style"] == style, arm
        assert float(col["beta"]) == kl, arm
        assert (col["iclmode"], col["iclk"], col["icld"]) == (mode, k, dep), arm
    assert set(CHK.WU_TRAINED_ARMS) == set(GEN.WU_TRAINED_ARMS)
    assert set(CHK.WU_STRICT_ICL_MODES) == {
        GEN.WU_ARM_COLS[a]["iclmode"] for a in GEN.WU_STRICT_ARMS}
    assert set(CHK.WU_EXTENSION_ICL_MODES) == {
        GEN.WU_ARM_COLS[a]["iclmode"] for a in GEN.WU_EXTENSION_ARMS}
    assert set(ANA.STRICT_ARMS) == set(GEN.WU_STRICT_ARMS)
    assert set(ANA.EXTENSION_ARMS) == set(GEN.WU_EXTENSION_ARMS)


# ------------------------------------------------- the in-context arms

def test_rejects_held_out_truth_in_a_strict_observed_context():
    """THE leak. An exemplar drawn from U tells the model the answer for
    an agent it is about to be scored on."""
    tag, d, env = _mk_run(arm="octx8")
    rec = d["wu_ctx_log"][1]["agents"][0]
    rec["ids"][0] = N - 1
    errs = _check(tag, d, env)
    assert any("HELD-OUT TRUTH LEAK" in e for e in errs), errs


def test_rejects_a_context_value_that_is_not_the_observed_opinion():
    tag, d, env = _mk_run(arm="octx8")
    rec = d["wu_ctx_log"][1]["agents"][0]
    rec["values"][2] = 0.123456
    rec["text"] = " ".join(WUC.VALUE_FMT.format(v) for v in rec["values"])
    errs = _check(tag, d, env)
    assert any("is not that agent's opinion" in e for e in errs), errs


def test_rejects_personal_memory_belonging_to_another_agent():
    tag, d, env = _mk_run(arm="phist8")
    rec = d["wu_ctx_log"][2]["agents"][0]
    rec["ids"][0] = 0
    errs = _check(tag, d, env)
    assert any("OTHER agents' ids" in e for e in errs), errs


def test_rejects_a_seeded_personal_history_at_round_zero():
    """At round 0 the platform has produced nothing, so a non-empty
    history came from outside the run -- which on the held-out set means
    from the truth."""
    tag, d, env = _mk_run(arm="phist8")
    rec = d["wu_ctx_log"][0]["agents"][0]
    v = float(d["innate"][rec["agent"]])
    rec["ids"] = [rec["agent"]]
    rec["values"] = [v]
    rec["text"] = WUC.VALUE_FMT.format(v)
    errs = _check(tag, d, env)
    assert any("nothing to remember" in e for e in errs), errs


def test_rejects_a_history_that_is_not_this_runs_own_record():
    for arm in ("phist8", "ehist8"):
        tag, d, env = _mk_run(arm=arm)
        rec = d["wu_ctx_log"][3]["agents"][0]
        rec["values"][0] = 0.99
        rec["text"] = " ".join(WUC.VALUE_FMT.format(v)
                               for v in rec["values"])
        errs = _check(tag, d, env)
        assert any("is not this run's own last" in e for e in errs), (arm,
                                                                     errs)


def test_prediction_history_and_expressed_history_are_different_quantities():
    """prediction_history shows what the platform SERVED, expressed_history
    what the population then EXPRESSED. Relabelling one as the other must
    not pass -- and the two must be distinguishable, or the test is
    vacuous."""
    tag, d, env = _mk_run(arm="phist8")
    served, op = d["served_raw"], d["op_raw"]
    assert float((served - op).abs().max()) > 1e-3, "vacuous: identical"
    d["config"]["wu_icl_mode"] = "expressed_history"
    for r in d["wu_ctx_log"]:
        r["mode"] = "expressed_history"
        r["history_source"] = WUC.HISTORY_SOURCE["expressed_history"]
        r["wu_icl_extension"] = True
        for e in r["agents"]:
            e["mode"] = "expressed_history"
            e["history_source"] = WUC.HISTORY_SOURCE["expressed_history"]
            e["extension"] = True
    errs = _check(tag, d, env)
    assert errs, "a swapped history source must not pass"


def test_rejects_text_that_disagrees_with_the_recorded_values():
    tag, d, env = _mk_run(arm="octx8")
    d["wu_ctx_log"][1]["agents"][0]["text"] = "no numbers here"
    errs = _check(tag, d, env)
    assert any("rendered text shows" in e for e in errs), errs


def test_rejects_a_history_source_that_does_not_match_the_mode():
    tag, d, env = _mk_run(arm="phist8")
    d["wu_ctx_log"][1]["history_source"] = "observed_peer"
    errs = _check(tag, d, env)
    assert any("history_source" in e for e in errs), errs


def test_rejects_a_mislabelled_extension_flag():
    """wu_context is the one place that says which mechanism is the
    observation-semantic extension; an artifact claiming otherwise would
    put an extension cell in a strict panel."""
    tag, d, env = _mk_run(arm="octx8")
    d["wu_ctx_log"][1]["wu_icl_extension"] = True
    errs = _check(tag, d, env)
    assert any("extension" in e for e in errs), errs


def test_rejects_a_missing_context_log_on_an_in_context_arm():
    tag, d, env = _mk_run(arm="octx8")
    d.pop("wu_ctx_log")
    errs = _check(tag, d, env)
    assert any("no wu_ctx_log.json.gz" in e for e in errs), errs


def test_rejects_a_context_log_missing_a_schema_key():
    tag, d, env = _mk_run(arm="octx8")
    d["wu_ctx_log"][0].pop("wu_icl_k")
    errs = _check(tag, d, env)
    assert any("missing the required key 'wu_icl_k'" in e for e in errs), errs


def test_rejects_a_context_log_with_the_wrong_round_count():
    tag, d, env = _mk_run(arm="octx8")
    d["wu_ctx_log"] = d["wu_ctx_log"][:-1]
    errs = _check(tag, d, env)
    assert any("context log has" in e for e in errs), errs


def test_rejects_exemplars_on_a_no_context_arm():
    tag, d, env = _mk_run(arm="frz")
    d["wu_ctx_log"] = [WUC.round_log_line(0, "none", [
        {"agent": 0, "ids": [1], "values": [0.5], "text": "0.50",
         "history_source": None, "mode": "none", "extension": False}])]
    errs = _check(tag, d, env)
    assert any("carries exemplars" in e for e in errs), errs


def test_a_gzipped_jsonl_context_log_on_disk_is_read(tmp_path):
    tag, d, env = _mk_run(arm="octx8")
    log = d.pop("wu_ctx_log")
    p = tmp_path / "wu_ctx_log.json.gz"
    with gzip.open(p, "wt") as fh:
        for r in log:
            fh.write(json.dumps(r) + "\n")
    d["_wu_ctx_log_path"] = str(p)
    assert _check(tag, d, env) == []


def test_an_unreadable_context_log_is_a_failure_not_a_skip(tmp_path):
    tag, d, env = _mk_run(arm="octx8")
    d.pop("wu_ctx_log")
    p = tmp_path / "wu_ctx_log.json.gz"
    p.write_bytes(b"not gzip")
    d["_wu_ctx_log_path"] = str(p)
    errs = _check(tag, d, env)
    assert any("will not parse" in e for e in errs), errs


# ------------------------------------------------------------- routing
# The routing treatment is a SOURCE INJECTION at OBSERVED agents: the
# runner rewrites their innate opinion before anything reads it. The
# cohort is a function of (cohort seed, frac, |O|) and NOT of the run
# seed, so the checker rebuilds it -- which is what lets the CONTROL
# twin run at frac 0, carry no cohort at all, and still be comparable.

def test_the_recomputed_cohort_is_the_runners_cohort():
    """If this drifted from the runner's draw, every routing check below
    would be testing a set nobody ran."""
    c = CHK.routing_cohort(ROUTE_FRAC, ROUTE_SEED, N_OBS, N)
    pool = torch.arange(N_OBS)
    g = torch.Generator().manual_seed(ROUTE_SEED + 611_000)
    want = pool[torch.randperm(pool.numel(), generator=g)[:3]].sort().values
    assert torch.equal(c, want.long())
    assert int(c.numel()) == 3
    assert int(c.max()) < N_OBS      # the cohort lives inside O


def test_rejects_a_cohort_hash_that_is_not_the_recomputed_one():
    tag, d, env = _mk_run(route="T")
    d["config"]["routing_treat_idx_sha256"] = "0" * 64
    errs = _check(tag, d, env)
    assert any("does not match" in e and "routing_treat_idx_sha256" in e
               for e in errs), errs


def test_rejects_a_treatment_with_no_cohort_provenance():
    tag, d, env = _mk_run(route="T")
    d["config"].pop("routing_treat_idx_sha256")
    errs = _check(tag, d, env)
    assert any("records no routing_treat_idx_sha256" in e for e in errs), errs


def test_rejects_a_treatment_that_did_not_inject_the_declared_value():
    tag, d, env = _mk_run(route="T")
    cohort = CHK.routing_cohort(ROUTE_FRAC, ROUTE_SEED, N_OBS, N)
    d["innate"] = d["innate"].clone()
    d["innate"][cohort[0]] = 0.9
    errs = _check(tag, d, env)
    assert any("did not set the cohort's innate" in e for e in errs), errs


def test_rejects_a_treatment_that_leaked_outside_its_cohort():
    tag, d, env = _mk_run(route="T")
    d["innate"] = d["innate"].clone()
    d["innate"][N - 1] += 0.2                # a held-out agent, off cohort
    errs = _check(tag, d, env)
    assert any("OUTSIDE its routed cohort" in e for e in errs), errs


def test_rejects_a_control_twin_that_injected():
    tag, d, env = _mk_run(route="C")
    d["innate"] = d["innate"].clone()
    d["innate"][0] = 0.5
    errs = _check(tag, d, env)
    assert any("CONTROL twin's innate differs" in e for e in errs), errs


def test_rejects_a_control_twin_running_at_a_nonzero_fraction():
    tag, d, env = _mk_run(route="C")
    d["config"]["routing_treat_frac"] = ROUTE_FRAC
    errs = _check(tag, d, env)
    assert any("a control that injects is not a control" in e
               for e in errs), errs


def test_rejects_an_injected_value_the_runner_would_refuse():
    tag, d, env = _mk_run(route="T")
    d["config"]["routing_treat_value"] = -1.0
    errs = _check(tag, d, env)
    assert any("outside [0, 1]" in e for e in errs), errs


def test_rejects_routing_config_on_a_run_with_no_routing_side():
    tag, d, env = _mk_run()
    d["config"]["routing_treat_frac"] = 0.25
    errs = _check(tag, d, env)
    assert any("tag and the intervention disagree" in e for e in errs), errs


def test_rejects_an_undeclared_innate_edit():
    """innate that is not the dataset's, with no routing declared, is an
    intervention nobody named."""
    tag, d, env = _mk_run()
    d["innate"] = d["innate"].clone()
    d["innate"][3] += 0.1
    errs = _check(tag, d, env)
    assert any("not the Pokec innate vector" in e for e in errs), errs


def test_paired_twins_with_the_same_cohort_pass():
    t_t, d_t, _ = _mk_run(route="T")
    t_c, d_c, _ = _mk_run(route="C")
    assert CHK.check_routing_pair(t_t, d_t, t_c, d_c) == []


def test_rejects_twins_whose_difference_is_not_the_declared_cohort():
    """A control that moved different people is not a control: the gap
    between the twins is then a difference of cohorts."""
    t_t, d_t, _ = _mk_run(route="T")
    t_c, d_c, _ = _mk_run(route="C")
    d_c["innate"] = d_c["innate"].clone()
    d_c["innate"][N - 1] += 0.3
    errs = CHK.check_routing_pair(t_t, d_t, t_c, d_c)
    assert any("SOURCE MASKS differ" in e for e in errs), errs


def test_rejects_twins_whose_cohort_seed_is_not_matched():
    t_t, d_t, _ = _mk_run(route="T")
    t_c, d_c, _ = _mk_run(route="C")
    d_c["config"]["routing_treat_seed"] = 8
    errs = CHK.check_routing_pair(t_t, d_t, t_c, d_c)
    assert any("routing_treat_seed differs" in e for e in errs), errs


def test_rejects_twins_whose_run_seed_is_not_matched():
    t_t, d_t, _ = _mk_run(route="T")
    t_c, d_c, _ = _mk_run(route="C")
    d_c["config"]["seed"] = 42
    errs = CHK.check_routing_pair(t_t, d_t, t_c, d_c)
    assert any("seed differs" in e for e in errs), errs


def test_twin_stem_pairs_the_two_sides():
    t_t, _, _ = _mk_run(route="T")
    t_c, _, _ = _mk_run(route="C")
    assert t_t != t_c
    assert CHK.twin_stem(t_t) == CHK.twin_stem(t_c)

# ------------------------------------------------- production is pinned

def test_production_sizes_are_still_gated():
    """The toy passes 12/9/3; production must stay 2163/1730/433 with
    K=100 and T=50."""
    import inspect
    sig = inspect.signature(CHK.check_jiduan_run)
    assert sig.parameters["expect_n"].default == 2163
    assert sig.parameters["expect_observed"].default == 1730
    assert sig.parameters["expect_heldout"].default == 433
    assert sig.parameters["expect_inner"].default == 100
    assert sig.parameters["expect_rounds"].default == 50
    assert CHK.WU_GRAPH_NODES == 2163 and CHK.WU_GRAPH_EDGES == 2346
    assert abs(CHK.WU_ALPHA_RAW_MEAN - 0.8909) < 1e-3
    assert abs(CHK.WU_BETA_RAW_MEAN - 0.8890) < 1e-3


@pytest.mark.skipif(not (REPO / "examples" / "pokec" / "parametric_params"
                         / "y_label2163.pk").exists(),
                    reason="pokec dataset not present")
def test_the_pinned_hashes_are_the_dataset_on_disk():
    """The canonical hashes are only worth anything if they are the real
    file's. Rebuild the environment and compare."""
    CHK._WU_ENV_CACHE.pop("env", None)
    env = CHK.wu_env()
    CHK._WU_ENV_CACHE.pop("env", None)
    assert env is not None
    assert env["innate"].shape[0] == 2163
    assert env["n_observed"] == 1730 and env["n_heldout"] == 433
    assert (env["n_nodes"], env["n_edges"]) == (2163, 2346)
    assert CHK._sha_t(env["innate"]) == CHK.WU_INNATE_SHA
    assert CHK._sha_t(env["alpha_raw"]) == CHK.WU_ALPHA_RAW_SHA
    assert CHK._sha_t(env["beta_raw"]) == CHK.WU_BETA_RAW_SHA
    assert CHK._sha_t(env["W"]) == CHK.WU_GRAPH_SHA
    # alpha is SUSCEPTIBILITY: the shipped mean is .89, not .11
    assert abs(float(env["alpha_raw"].mean()) - CHK.WU_ALPHA_RAW_MEAN) < 1e-6
    assert abs(float(env["beta_raw"].mean()) - CHK.WU_BETA_RAW_MEAN) < 1e-6


# =================================================== generator wiring

WU_KEY_JOBS = {
    "jiduan_pokec_smoke": 1,
    "jiduan_pokec_prior": 12,
    "jiduan_pokec_prior_seeds": 8,
    "jiduan_pokec_lambda_ladder": 3,
    "jiduan_pokec_icl": 8,
    "jiduan_pokec_environment": 10,
    "jiduan_pokec_routing_smoke": 16,
    "jiduan_pokec_routing_seeds": 32,
    "jiduan_pokec_frozen": 6,
}


def _cfg_tags(key):
    p = CONDOR / f"configs_pofd_{key}.txt"
    assert p.exists(), key
    return [ln.split(",")[0].strip()
            for ln in p.read_text().splitlines() if ln.strip()]


def test_every_key_has_its_exact_job_count():
    for key, n in WU_KEY_JOBS.items():
        tags = _cfg_tags(key)
        assert len(tags) == n, (key, len(tags), n)
        assert len(set(tags)) == n, key


def test_the_controls_key_generates_nothing():
    """A documented ZERO-JOB key: the Wu controls are linear maps of
    vectors that already exist, so a GPU job would only re-derive
    arithmetic. It must have no config and no sub, or someone will
    submit an empty queue."""
    assert not (CONDOR / "configs_pofd_jiduan_pokec_controls.txt").exists()
    assert not (CONDOR / "at_pofd_jiduan_pokec_controls.sub").exists()
    sh = (CONDOR / "submit_pofd_sweep.sh").read_text()
    assert "jiduan_pokec_controls)" in sh
    assert "queues ZERO jobs" in sh


def test_lambda_ladder_reuses_lambda_zero_and_one():
    """5 conceptual cells, 2 reused, 3 queued -- and the reused tags must
    be BYTE-IDENTICAL to the prior key's, or the ladder has lost its
    anchors and nobody would notice."""
    prior = set(_cfg_tags("jiduan_pokec_prior"))
    reused = set(GEN.wu_ladder_reused())
    queued = set(_cfg_tags("jiduan_pokec_lambda_ladder"))
    assert len(reused) == 2 and reused <= prior
    assert len(queued) == 3
    assert reused & queued == set()
    assert len(reused | queued) == len(GEN.WU_LADDER) == 5
    lams = sorted(ANA.LAMBDA_OF[ANA.parse_tag(t)["arm"]]
                  for t in reused | queued)
    assert lams == [0.0, 0.1, 0.5, 1.0, 10.0]


def test_the_frozen_endpoint_is_not_on_the_lambda_ladder():
    """A frozen run never trains, so it is not 'SFT at an infinite KL
    weight'. Putting it on the dose axis would place a different code
    path on a continuum."""
    for t in _cfg_tags("jiduan_pokec_lambda_ladder"):
        assert ANA.parse_tag(t)["arm"] in ("b0p1", "b0p5", "b10")
    assert "frz" not in {a for _, a in GEN.WU_LADDER}


def test_icl_key_reuses_the_prior_b0_and_b1_cells():
    prior = set(_cfg_tags("jiduan_pokec_prior"))
    reused = set(GEN.wu_icl_reused())
    queued = set(_cfg_tags("jiduan_pokec_icl"))
    assert len(reused) == 4 and reused <= prior
    assert len(queued) == 8 and reused & queued == set()
    assert len(reused | queued) == 12 == len(GEN.WU_ICL_MODELS) * len(GEN.WU_ICL_ARMS)
    # BOTH history mechanisms are queued and must stay distinguishable:
    # phist8 = the platform's own past predictions (strict Wu);
    # ehist8 = realized post-peer opinions (the Section 4 extension).
    assert sum(1 for t in queued if "_phist8_" in t) == 2
    assert sum(1 for t in queued if "_ehist8_" in t) == 2


def test_environment_grid_shares_the_centre_and_collapses_c_beta_zero():
    """15 arm-by-pair combinations become 13 cells (c_beta=0 makes the
    model irrelevant, so three arms are one trajectory), 3 of which are
    the shared centre and are reused, leaving 10."""
    pairs = GEN.wu_env_pairs()
    assert len(pairs) == 5 and pairs.count((1.0, 1.0)) == 1
    cells = GEN.wu_env_cells()
    assert len(cells) == 13
    assert [a for a, ca, cb in cells if cb == 0.0] == ["b0"]
    reused = set(GEN.wu_env_reused())
    queued = set(_cfg_tags("jiduan_pokec_environment"))
    assert len(reused) == 3 and reused & queued == set()
    assert len(queued) == 10 == len(cells) - len(reused)
    prior = set(_cfg_tags("jiduan_pokec_prior"))
    icl = set(_cfg_tags("jiduan_pokec_icl"))
    assert len(reused & prior) == 2 and len(reused & icl) == 1


def test_routing_keys_are_paired_twins_and_reuse_nothing():
    prior = set(_cfg_tags("jiduan_pokec_prior"))
    for key, n, seeds in (("jiduan_pokec_routing_smoke", 16, {0}),
                          ("jiduan_pokec_routing_seeds", 32, {42, 43})):
        tags = _cfg_tags(key)
        assert len(tags) == n
        assert set(tags) & prior == set()
        parsed = [ANA.parse_tag(t) for t in tags]
        assert {p["seed"] for p in parsed} == seeds
        stems = {}
        for t in tags:
            stems.setdefault(CHK.twin_stem(t), set()).add(
                CHK.route_side_of(t))
        assert len(stems) == n // 2
        assert all(v == {"T", "C"} for v in stems.values())


def test_seed_replicates_change_nothing_but_the_seed():
    base = {}
    for r in (CONDOR / "configs_pofd_jiduan_pokec_prior.txt").read_text().splitlines():
        c = [x.strip() for x in r.split(",")]
        base["_".join(c[0].split("_")[:-2])] = c[1:3] + c[14:29]
    for r in (CONDOR / "configs_pofd_jiduan_pokec_prior_seeds.txt").read_text().splitlines():
        c = [x.strip() for x in r.split(",")]
        assert c[1:3] + c[14:29] == base["_".join(c[0].split("_")[:-2])], r


def test_tags_carry_no_gate_tokens():
    """FJ applies no AI gate and no bounded-confidence gate, so _ea/_es
    would name something the operator never ran."""
    for key in WU_KEY_JOBS:
        for t in _cfg_tags(key):
            assert "_ea" not in t and "_es" not in t, (key, t)


def test_tags_encode_the_full_configuration():
    t = GEN.wu_tag("qwen7b", "b1", ca=0.5, cb=1.0, seed=42, rounds=50)
    for token in ("qwen7b", "_b1_", "pad0p5", "pbd1", "_in100_", "_s42_",
                  "_r50"):
        assert token in t, (token, t)
    r = GEN.wu_tag("qwen7b", "frz", route="T")
    assert "_rtT_" in r and "_frz_" in r


def test_tag_parsing_survives_underscored_model_slugs_and_b1_vs_b10():
    """qwen3_8b contains an underscore, and 'b1' is a prefix of 'b10' --
    both break a positional or prefix-based split."""
    p = ANA.parse_tag(GEN.wu_tag("qwen3_8b", "b1"))
    assert p["model"] == "qwen3_8b" and p["arm"] == "b1"
    p10 = ANA.parse_tag(GEN.wu_tag("qwen3_8b", "b10"))
    assert p10["model"] == "qwen3_8b" and p10["arm"] == "b10"
    assert CHK.arm_of(GEN.wu_tag("olmo3_7b", "b10")) == "b10"
    assert CHK.arm_of(GEN.wu_tag("olmo3_7b", "b1")) == "b1"


def test_tags_round_trip_through_the_analyzer_parser():
    for key in WU_KEY_JOBS:
        if key == "jiduan_pokec_frozen":
            continue
        for t in _cfg_tags(key):
            p = ANA.parse_tag(t)
            assert p is not None, t
            assert GEN.wu_tag(p["model"], p["arm"], ca=p["ca"], cb=p["cb"],
                              seed=p["seed"], rounds=p["rounds"],
                              inner=p["inner"], route=p["route"],
                              smoke=p["smoke"]) == t


def test_frozen_extraction_covers_every_model_and_claims_no_reuse():
    """No frozen Pokec vector was found for ANY checkpoint, so all six
    extract. Claiming reuse we cannot verify would silently score five
    models against a vector from another dataset or another SKU."""
    tags = _cfg_tags("jiduan_pokec_frozen")
    assert len(tags) == 6
    assert {ANA.frozen_model_of(t) for t in tags} == set(GEN.WU_MODELS)
    assert GEN.WU_FROZEN_MODELS == GEN.WU_MODELS
    for t in tags:
        for tok in ("_pa", "_pb", "_in", "_rt"):
            assert tok not in t, (tok, t)
    body = (CONDOR / "configs_pofd_jiduan_pokec_frozen.txt").read_text()
    assert body.count("frozen") == 6
    src = (CONDOR / "gen_pofd_sweep.py").read_text()
    assert "NO REUSE IS CLAIMED" in src
    assert "NO REUSE WAS FOUND" in src


def test_smoke_is_one_short_forward_kl_qwen_cell():
    rows = GEN.wu_smoke_rows()
    assert len(rows) == 1
    t = rows[0].split(",")[0]
    assert "qwen7b" in t and "_b1_" in t and t.endswith("_r3smoke")
    assert rows[0].split(",")[1].strip() == "sft_kl"


def test_no_scalar_alpha_or_beta_in_any_generated_sub():
    """A scalar next to a per-agent source is the ambiguity this family
    exists to remove -- whichever one ran, the artifact would look fine."""
    for key in WU_KEY_JOBS:
        sub = (CONDOR / f"at_pofd_{key}.sub").read_text()
        env = next(ln for ln in sub.splitlines()
                   if ln.startswith("environment"))
        assert "FJ_ALPHA=" not in env, key
        assert "FJ_BETA=" not in env, key
        assert "FJ_PEER_ALPHA" not in env, key


def test_subs_pin_pokec_the_wu1_operator_and_training_on_o_only():
    for key in WU_KEY_JOBS:
        if key == "jiduan_pokec_frozen":
            continue
        sub = (CONDOR / f"at_pofd_{key}.sub").read_text()
        env = next(ln for ln in sub.splitlines()
                   if ln.startswith("environment"))
        assert "DATASET=pokec" in env, key
        assert "POP_MODEL=fj" in env and "FJ_UPDATE_VERSION=wu1" in env, key
        assert "FJ_OBSERVED_PASSTHROUGH=1" in env, key
        assert "FJ_ALPHA_SCALE=$(ascale)" in env, key
        assert "FJ_BETA_SCALE=$(bscale)" in env, key
        # train on the OBSERVED prefix only: a held-out opinion cannot
        # reach the optimizer if the batch is drawn from the first 1730
        assert "N_LABELED=1730" in env and "TRAIN_CAP=1730" in env, key
        assert "AI_GATE_MODE" not in env and "PEER_GATE_MODE" not in env, key
        assert f'CUDADeviceName == "{GPU}"' in sub, key


def test_the_frozen_sub_applies_no_fj_parameter():
    sub = (CONDOR / "at_pofd_jiduan_pokec_frozen.sub").read_text()
    env = next(ln for ln in sub.splitlines()
               if ln.startswith("environment"))
    assert "FJ_" not in env and "POP_MODEL" not in env
    assert "DATASET=pokec" in env
    assert f'CUDADeviceName == "{GPU}"' in sub


def test_w_plat_is_pinned_so_beta_is_never_scaled_twice():
    for key in WU_KEY_JOBS:
        if key == "jiduan_pokec_frozen":
            continue
        for r in (CONDOR / f"configs_pofd_{key}.txt").read_text().splitlines():
            assert [x.strip() for x in r.split(",")][11] == "1.0", (key, r)


def test_on_disk_configs_match_the_generator():
    for key, rows in (
            ("jiduan_pokec_smoke", GEN.wu_smoke_rows()),
            ("jiduan_pokec_prior", GEN.wu_prior_rows()),
            ("jiduan_pokec_prior_seeds", GEN.wu_prior_seed_rows()),
            ("jiduan_pokec_lambda_ladder", GEN.wu_ladder_rows()),
            ("jiduan_pokec_icl", GEN.wu_icl_rows()),
            ("jiduan_pokec_environment", GEN.wu_env_rows()),
            ("jiduan_pokec_routing_smoke", GEN.wu_route_rows()),
            ("jiduan_pokec_routing_seeds",
             GEN.wu_route_rows(GEN.WU_SEEDS)),
            ("jiduan_pokec_frozen", GEN.wu_frozen_rows())):
        p = CONDOR / f"configs_pofd_{key}.txt"
        assert p.read_text() == "\n".join(rows) + "\n", key


def test_no_wu_tag_collides_with_any_other_wave():
    ours = set()
    for key in WU_KEY_JOBS:
        ours |= set(_cfg_tags(key))
    others = set()
    for p in CONDOR.glob("configs_pofd_*.txt"):
        if "jiduan_pokec" in p.name:
            continue
        for ln in p.read_text().splitlines():
            if ln.strip():
                others.add(ln.split(",")[0].strip())
    assert ours & others == set()


def test_production_constants_are_wus():
    assert GEN.WU_INNER == 100
    assert GEN.WU_ROUNDS == 50
    assert GEN.WU_N == 2163
    assert GEN.WU_N_OBSERVED == 1730
    assert GEN.WU_N_HELDOUT == 433
    assert GEN.WU_SEEDS == [42, 43]
    assert len(GEN.WU_MODELS) == 6


# ------------------------------------------------------- submit wiring

def test_submit_script_registers_every_generated_key():
    sh = (CONDOR / "submit_pofd_sweep.sh").read_text()
    case = next(ln for ln in sh.splitlines()
                if ln.strip().startswith("jiduan_pokec_smoke|"))
    for k in WU_KEY_JOBS:
        assert f"{k}|" in case or f"{k})" in case, k
    # present in all THREE usage strings (BID, WHAT and the *) fallback)
    for k in ("jiduan_pokec_smoke", "jiduan_pokec_icl",
              "jiduan_pokec_environment", "jiduan_pokec_frozen"):
        assert sh.count(f"|{k}|") >= 3, k


def test_usage_strings_still_contain_no_braces():
    """REGRESSION, and a real one. The usage text lives inside
    BID="${1:?usage: ...}"; bash ends a ${x:?word} expansion at the FIRST
    unescaped '}', so one brace truncates BID and WHAT for EVERY key in
    the project, not just the new ones. Braces are banned outright."""
    import re
    checked = 0
    for ln in (CONDOR / "submit_pofd_sweep.sh").read_text().split("\n"):
        m = re.match(r'^(BID|WHAT)="\$\{[12]:\?(.*)\}"$', ln)
        if not m:
            continue
        checked += 1
        assert "{" not in m.group(2) and "}" not in m.group(2), m.group(1)
    assert checked == 2


def test_every_new_key_parses_to_itself_through_the_expansions():
    head = subprocess.run(
        ["sed", "-n", "17,18p", str(CONDOR / "submit_pofd_sweep.sh")],
        capture_output=True, text=True).stdout
    for key in list(WU_KEY_JOBS) + ["jiduan_pokec_controls", "smoke",
                                    "fj_robustness"]:
        r = subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + head
             + '\nprintf "%s|%s" "$BID" "$WHAT"', "_", "50", key],
            capture_output=True, text=True)
        assert r.stdout == f"50|{key}", (key, r.stdout[:80])


def test_submit_script_is_valid_bash():
    r = subprocess.run(["bash", "-n", str(CONDOR / "submit_pofd_sweep.sh")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ==================================================== analyzer behaviour

def test_analyzer_hard_fails_on_an_incomplete_grid(tmp_path):
    with pytest.raises(SystemExit) as e:
        ANA.analyse([tmp_path], tmp_path / "out", grid="prior")
    assert "HARD FAIL" in str(e.value)


def test_analyzer_hard_fails_when_a_cell_was_never_generated(tmp_path):
    with pytest.raises(SystemExit) as e:
        ANA.resolve(ANA.grid_specs("prior"), [])
    assert "incomplete" in str(e.value)


def test_analyzer_resolves_every_grid_against_the_on_disk_configs():
    tags = ANA.read_config_tags()
    for grid in ("prior", "prior_seeds", "ladder", "icl", "environment",
                 "routing"):
        got = ANA.resolve(ANA.grid_specs(grid), tags)
        assert len(got) == len(ANA.grid_specs(grid)), grid
        assert len({g["tag"] for g in got}) == len(got), grid


def test_environment_grid_in_the_analyzer_matches_the_generator():
    ana = {(s["arm"], s["ca"], s["cb"]) for s in ANA.grid_specs("environment")}
    gen = set(GEN.wu_env_cells())
    assert ana == gen


def test_analyzer_hard_fails_without_a_frozen_reference(tmp_path):
    """The primary held-out estimand is distance to the frozen map, so a
    missing frozen extraction is not a degraded run, it is no run."""
    root = tmp_path / "runs"
    for s in ANA.resolve(ANA.grid_specs("prior"), ANA.read_config_tags()):
        _write_fake_cell(root, s["tag"])
    with pytest.raises(SystemExit) as e:
        ANA.analyse([root], tmp_path / "out", grid="prior", controls=False)
    assert "frozen prediction map" in str(e.value)


def _write_fake_cell(root, tag, rounds=50, n=N, n_obs=N_OBS, innate=None,
                     alpha_sha="a", beta_sha="b", graph_sha="g"):
    d_env = _toy_env()
    innate = d_env["innate"] if innate is None else innate
    g = torch.Generator().manual_seed(abs(hash(tag)) % (2 ** 31))
    op = torch.rand(rounds, n, generator=g) * 0.5 + 0.25
    pred = torch.rand(rounds, n, generator=g)
    p = Path(root) / tag
    p.mkdir(parents=True, exist_ok=True)
    torch.save({"config": {"fj_inner_steps": 100,
                           "fj_alpha_realized_mean": 0.8909,
                           "fj_graph_sha256": graph_sha,
                           "fj_alpha_raw_sha256": alpha_sha,
                           "fj_beta_raw_sha256": beta_sha,
                           "n_labeled": n_obs},
                "op_raw": op, "model_pred_raw": pred, "served_raw": pred,
                "innate": innate}, p / "trajectory.pt")
    return p


def _write_frozen(root, model, n=N):
    p = Path(root) / f"pofdwuzs_{model}_s0_r1"
    p.mkdir(parents=True, exist_ok=True)
    torch.save({"config": {}, "pred_raw": torch.linspace(0.2, 0.8, n
                                                         ).unsqueeze(0)},
               p / "trajectory.pt")


def test_analyzer_runs_end_to_end_and_writes_the_named_outputs(tmp_path):
    root = tmp_path / "runs"
    specs = ANA.resolve(ANA.grid_specs("prior"), ANA.read_config_tags())
    for s in specs:
        _write_fake_cell(root, s["tag"])
    for m in {s["model"] for s in specs}:
        _write_frozen(root, m)
    out = tmp_path / "out"
    cells = ANA.analyse([root], out, grid="prior", n_observed=N_OBS,
                        controls=False)
    assert (out / "jiduan_pokec_rounds.csv").exists()
    assert (out / "jiduan_pokec_cells.csv").exists()
    for stem in ("jiduan_pokec_prior_retention",):
        assert (out / f"{stem}.png").exists()
        assert (out / f"{stem}.pdf").exists()
    # O, U and full are reported SEPARATELY, and U is flagged primary
    assert {c["subset"] for c in cells} == {"U", "O", "full"}
    prim = [c for c in cells if c["subset"] == "U"]
    assert all("PRIMARY" in c["subset_role"] for c in prim)
    assert all("secondary" in c["subset_role"]
               for c in cells if c["subset"] != "U")


def test_analyzer_refuses_a_grid_built_on_two_environments(tmp_path):
    root = tmp_path / "runs"
    specs = ANA.resolve(ANA.grid_specs("prior"), ANA.read_config_tags())
    for i, s in enumerate(specs):
        _write_fake_cell(root, s["tag"],
                         graph_sha="g" if i else "OTHER-GRAPH")
    for m in {s["model"] for s in specs}:
        _write_frozen(root, m)
    with pytest.raises(SystemExit) as e:
        ANA.analyse([root], tmp_path / "out", grid="prior",
                    n_observed=N_OBS, controls=False)
    assert "not one environment" in str(e.value)


def test_round_rows_start_at_innate_and_only_show_complete_rounds():
    """Headline figures plot innate at t=0 and the post-FJ state after
    each complete outer round -- never a within-round state."""
    d_env = _toy_env()
    op = torch.rand(5, N)
    d = {"config": {"fj_inner_steps": 100, "fj_alpha_realized_mean": 0.89},
         "op_raw": op, "model_pred_raw": torch.rand(5, N),
         "served_raw": torch.rand(5, N), "innate": d_env["innate"]}
    spec = {"tag": "t", "model": "m", "arm": "b0", "ca": 1.0, "cb": 1.0,
            "seed": 0, "route": None}
    rows = ANA.round_rows(spec, d, None, N_OBS)
    ts = sorted({r["t"] for r in rows})
    assert ts == list(range(0, 6))
    zero = [r for r in rows if r["t"] == 0]
    assert all(r["state"] == "innate" for r in zero)
    assert all(math.isnan(r["served_mean"]) for r in zero)
    assert all(r["state"] == "post_fj" for r in rows if r["t"] > 0)


def test_inner_convergence_is_separated_from_outer_stationarity():
    """They are different questions. The inner loop converges BY
    CONSTRUCTION at alpha ~ .89, K = 100; whether the outer loop has
    settled is empirical, and a still-drifting outer loop must not be
    called converged just because the inner one is."""
    d_env = _toy_env()
    drifting = torch.stack([torch.full((N,), 0.2 + 0.02 * t)
                            for t in range(50)])
    d = {"config": {"fj_inner_steps": 100,
                    "fj_alpha_realized_mean": 0.8909},
         "op_raw": drifting, "model_pred_raw": torch.rand(50, N),
         "served_raw": torch.rand(50, N), "innate": d_env["innate"]}
    spec = {"tag": "t", "model": "m", "arm": "b0", "ca": 1.0, "cb": 1.0,
            "seed": 0, "route": None}
    cells = ANA.cell_rows(spec, ANA.round_rows(spec, d, None, N_OBS), d)
    c = [x for x in cells if x["subset"] == "U"][0]
    assert c["inner_converged_by_construction"] is True
    assert c["inner_contraction_bound"] < 1e-3
    assert c["outer_stationary"] is False
    assert c["outer_late_mean_drift"] > ANA.EQ_DRIFT


def test_a_settled_run_with_a_noise_floor_is_called_stationary():
    """The point of using MEAN DRIFT rather than a vanishing step: a
    fresh LoRA every round leaves a per-round floor that never goes to
    zero, and demanding a tiny step would mislabel this as unconverged."""
    d_env = _toy_env()
    g = torch.Generator().manual_seed(3)
    noisy = 0.5 + 0.002 * torch.rand(50, N, generator=g)
    d = {"config": {"fj_inner_steps": 100,
                    "fj_alpha_realized_mean": 0.8909},
         "op_raw": noisy, "model_pred_raw": torch.rand(50, N),
         "served_raw": torch.rand(50, N), "innate": d_env["innate"]}
    spec = {"tag": "t", "model": "m", "arm": "b0", "ca": 1.0, "cb": 1.0,
            "seed": 0, "route": None}
    cells = ANA.cell_rows(spec, ANA.round_rows(spec, d, None, N_OBS), d)
    c = [x for x in cells if x["subset"] == "U"][0]
    assert c["outer_stationary"] is True
    assert c["outer_noise_floor"] > 0.0        # the step does NOT vanish


def test_seed_spread_is_nan_at_one_seed_not_a_fake_zero():
    rows = [{"model": "m", "arm": "b0", "ca": 1.0, "cb": 1.0, "route": "",
             "subset": "U", "seed": 0, "late_mean": 0.5, "late_sd": 0.06,
             "late_pop_w1_from_innate": 0.1, "late_served_mean": 0.5,
             "late_served_sd": 0.05, "late_w1_pred_to_frozen": 0.1,
             "late_rmse_pred_to_frozen": 0.1,
             "late_corr_pred_to_frozen": 0.5,
             "late_w1_pred_to_pop": 0.1, "outer_late_mean_drift": 0.0,
             "outer_stationary": True}]
    agg = ANA.across_seeds(rows)
    assert agg[0]["n_seeds"] == 1
    assert math.isnan(agg[0]["late_sd_seed_sd"])


def test_seed_spread_is_across_seeds_only():
    def mk(seed, sd):
        return {"model": "m", "arm": "b0", "ca": 1.0, "cb": 1.0,
                "route": "", "subset": "U", "seed": seed, "late_mean": 0.5,
                "late_sd": sd, "late_pop_w1_from_innate": 0.1,
                "late_served_mean": 0.5, "late_served_sd": 0.05,
                "late_w1_pred_to_frozen": 0.1,
                "late_rmse_pred_to_frozen": 0.1,
                "late_corr_pred_to_frozen": 0.5,
                "late_w1_pred_to_pop": 0.1, "outer_late_mean_drift": 0.0,
                "outer_stationary": True}
    agg = ANA.across_seeds([mk(0, 0.06), mk(42, 0.07), mk(43, 0.08)])
    assert len(agg) == 1
    assert agg[0]["seeds"] == "0,42,43" and agg[0]["n_seeds"] == 3
    assert agg[0]["late_sd"] == pytest.approx(0.07)
    assert agg[0]["late_sd_seed_sd"] == pytest.approx(0.01)


def test_pair_gap_is_direction_neutral():
    """Swapping the two arms must flip the sign and change nothing else.
    A verdict that survived the swap would be encoding an expectation."""
    def mk(arm, v, sd):
        return {"model": "m", "arm": arm, "ca": 1.0, "cb": 1.0, "route": "",
                "subset": "U", "n_seeds": 3, "late_w1_pred_to_frozen": v,
                "late_w1_pred_to_frozen_seed_sd": sd}
    agg = [mk("b0", 0.10, 0.002), mk("b1", 0.16, 0.002)]
    fwd = ANA.pair_gap(agg, "b0", "b1")[0]
    rev = ANA.pair_gap(agg, "b1", "b0")[0]
    assert fwd["gap_b_minus_a"] == pytest.approx(-rev["gap_b_minus_a"])
    assert fwd["separated"] is True and rev["separated"] is True
    tight = ANA.pair_gap([mk("b0", 0.10, 0.05), mk("b1", 0.11, 0.05)],
                         "b0", "b1")[0]
    assert tight["separated"] is False


def test_no_expected_ordering_is_encoded_anywhere():
    """DIRECTION NEUTRALITY, checked as text as well as behaviour: a
    replication that only recognises one outcome is not a replication."""
    src = (PIPE / "analyze_jiduan_pokec.py").read_text().lower()
    for phrase in ("expected ordering", "should be higher",
                   "should be lower", "as expected", "we expect",
                   "better arm", "arm wins", "outperform", "preferred arm"):
        assert phrase not in src, phrase


def test_figures_carry_no_titles():
    """Project convention: paper figures carry NO title text; the
    narrative goes in the caption block."""
    src = (PIPE / "analyze_jiduan_pokec.py").read_text()
    assert "set_title" not in src
    assert "suptitle" not in src


def test_strict_and_extension_channels_are_labelled_and_never_pooled():
    src = (PIPE / "analyze_jiduan_pokec.py").read_text()
    assert "strict_wu" in src and "observation_extension" in src
    # phist8 is the extension precisely because Wu's platform has no
    # memory of its own outputs
    # wu_context.is_extension() is the ONE home for the classification:
    # a mechanism is an extension when it shows the model something Wu's
    # platform cannot observe. The platform HAS its own past outputs, so
    # prediction_history is strict; it does NOT observe held-out agents'
    # realised opinions, so expressed_history is the extension.
    assert ANA.EXTENSION_ARMS == ("ehist8",)
    assert "octx8" in ANA.STRICT_ARMS and "phist8" in ANA.STRICT_ARMS
    for arm in ANA.ARMS:
        mode = CHK.WU_ARM_SEMANTICS[arm][2]
        assert (arm in ANA.EXTENSION_ARMS) == WUC.is_extension(mode), arm


def test_controls_are_model_independent_and_reproduce_the_recurrence():
    env = _toy_env()
    traj = ANA.run_control("perfect", env, rounds=4, n_inner=INNER,
                           n_observed=N_OBS)
    innate = np.asarray(env["innate"], dtype=float)
    a = np.asarray(env["alpha_raw"], dtype=float)
    b = np.asarray(env["beta_raw"], dtype=float)
    Wn = np.asarray(env["W"], dtype=float)
    x = innate.copy()
    for t in range(4):
        x = ANA.fj_apply(innate, x.copy(), a, b, Wn, INNER)
        assert np.abs(traj[t] - x).max() < 1e-9


def test_no_platform_control_is_constant_from_round_one():
    env = _toy_env()
    traj = ANA.run_control("no_platform", env, cb=0.0, rounds=5,
                           n_inner=INNER, n_observed=N_OBS)
    for t in range(1, 5):
        assert np.abs(traj[t] - traj[0]).max() < 1e-9


def test_the_frozen_control_is_not_constant_under_observed_passthrough():
    """A real difference from the MovieLens FJ wave, and worth stating:
    the passthrough feeds the evolving observed population back into the
    anchor every round even when the model never changes."""
    env = _toy_env()
    fz = np.linspace(0.05, 0.95, N)
    traj = ANA.run_control("frozen", env, rounds=5, n_inner=INNER,
                           frozen=fz, n_observed=N_OBS)
    assert np.abs(traj[2] - traj[0]).max() > 1e-6


def test_rounds_and_agents_are_never_treated_as_replicates():
    """No standard error is ever taken over rounds or agents; the only
    spread in the file is across SEEDS."""
    src = (PIPE / "analyze_jiduan_pokec.py").read_text()
    assert "_seed_sd" in src
    for bad in ("sem(", "stats.ttest", "scipy.stats", "p_value", "pvalue"):
        assert bad not in src, bad
    assert "ACROSS SEEDS" in src


def test_routing_carries_both_history_mechanisms_unpooled():
    """The Section 4 claim needs ehist8 (realized post-peer opinions);
    phist8 is the strict Wu comparison. Both must be present, and a
    reader must be able to tell them apart from the tag alone -- pooling
    them would destroy the contrast the stage exists to draw."""
    for key in ("jiduan_pokec_routing_smoke", "jiduan_pokec_routing_seeds"):
        tags = _cfg_tags(key)
        n_p = sum(1 for t in tags if "_phist8_" in t)
        n_e = sum(1 for t in tags if "_ehist8_" in t)
        assert n_p == n_e == len(tags) // 4, (key, n_p, n_e, len(tags))
        assert not any("_phist8_" in t and "_ehist8_" in t for t in tags)


def test_only_expressed_history_is_flagged_an_extension():
    assert "ehist8" in GEN.WU_EXTENSION_ARMS
    assert "phist8" not in GEN.WU_EXTENSION_ARMS
    assert "octx8" not in GEN.WU_EXTENSION_ARMS
