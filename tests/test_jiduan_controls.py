"""Tests for the model-free Jiduan/Wu Pokec controls (2026-08-22).

No language model is loaded. Everything except one slow-marked test runs
on a hand-built 6-agent graph, so the arithmetic is checkable in closed
form and the suite stays fast.

The reference recurrence in `_ref_round` is written as a per-agent python
loop on purpose -- an INDEPENDENT expression of the operator, not a copy
of it. Checking wu_round against a re-import of itself would test nothing.

DIRECTION-NEUTRAL: nothing here asserts which control ends up higher,
lower, wider or narrower. What is asserted is structural -- that the
observed set passes through, that the held-out set is predicted from
observed labels only, that beta=0 severs the platform channel, and that
the loop runs Wu's K and T.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
PIPE = REPO / "experiments" / "scripts" / "cluster_pipelines"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


JC = _load("jiduan_ctl", PIPE / "jiduan_controls.py")

N = 6
N_OBS = 4          # O = agents 0..3, U = agents 4..5


def _graph6():
    """6 agents, connected, no isolated node, degrees (3,2,3,3,2,3) so the
    per-agent neighbourhood averages genuinely differ."""
    adj = torch.tensor([[0., 1., 1., 0., 0., 1.],
                        [1., 0., 1., 0., 0., 0.],
                        [1., 1., 0., 1., 0., 0.],
                        [0., 0., 1., 0., 1., 1.],
                        [0., 0., 0., 1., 0., 1.],
                        [1., 0., 0., 1., 1., 0.]])
    W = adj / adj.sum(dim=1, keepdim=True)
    innate = torch.tensor([0.10, 0.30, 0.45, 0.60, 0.85, 0.95])
    return innate, W


def _env(*, alpha=None, beta=None, X=None, colnames=None, n_observed=N_OBS):
    innate, W = _graph6()
    alpha = torch.tensor([0.85, 0.90, 0.95, 0.80, 0.99, 0.88]) if alpha is None else alpha
    beta = torch.tensor([0.90, 0.85, 0.95, 0.80, 0.75, 0.99]) if beta is None else beta
    return JC.make_env(innate, W, alpha, beta, n_observed=n_observed,
                       X=X, colnames=colnames)


def _ref_round(innate, served, beta, alpha, W, n_inner):
    """Independent per-agent scalar reference for one outer Wu round."""
    n = int(innate.shape[0])
    x_init = [(1.0 - float(beta[i])) * float(innate[i])
              + float(beta[i]) * float(served[i]) for i in range(n)]
    u = list(x_init)
    for _ in range(n_inner):
        u = [(1.0 - float(alpha[i])) * x_init[i]
             + float(alpha[i]) * sum(float(W[i][j]) * u[j] for j in range(n))
             for i in range(n)]
    return torch.tensor(u, dtype=torch.float32)


# ------------------------------------------------------- the split

def test_masks_are_disjoint_and_cover_n():
    obs, unobs = JC.observed_unobserved_masks(N, N_OBS)
    assert not bool((obs & unobs).any())            # disjoint
    assert bool((obs | unobs).all())                # cover
    assert int(obs.sum()) + int(unobs.sum()) == N
    assert int(obs.sum()) == N_OBS


def test_pokec_split_sizes_are_the_documented_ones():
    """O = the first 1730 rows (y_label), U = the last 433 (y_unlabel);
    load_pokec_setup builds innate as concat(y_label, y_unlabel), so the
    prefix IS the labeled set and a suffix split would silently relabel
    1297 observed agents as held out."""
    obs, unobs = JC.observed_unobserved_masks(JC.N_AGENTS, JC.N_OBSERVED)
    assert int(obs.sum()) == 1730 and int(unobs.sum()) == 433
    assert bool(obs[:1730].all()) and not bool(obs[1730:].any())


@pytest.mark.parametrize("bad", [0, -1, JC.N_AGENTS + 1])
def test_mask_bounds_are_validated(bad):
    with pytest.raises(ValueError):
        JC.observed_unobserved_masks(JC.N_AGENTS, bad)


# -------------------------------------------------- Wu's configuration

def test_the_controls_loop_runs_k100_and_t50_by_default():
    assert JC.K_FJ == 100
    assert JC.T_ROUNDS == 50
    sig = inspect.signature(JC.run_control)
    assert sig.parameters["n_inner"].default == JC.K_FJ == 100
    assert sig.parameters["rounds"].default == JC.T_ROUNDS == 50
    a = inspect.signature(JC.analyse)
    assert a.parameters["n_inner"].default == 100
    assert a.parameters["rounds"].default == 50
    assert inspect.signature(JC.wu_round).parameters["n_inner"].default == 100


def test_the_control_set_is_the_declared_one():
    assert JC.CONTROLS == ("no_platform", "perfect_prediction", "mean",
                           "ols", "lightgbm")


def test_run_control_rejects_an_unknown_control():
    with pytest.raises(ValueError):
        JC.run_control("nonesuch", _env(), rounds=1, n_inner=1)


# ------------------------------------------------------- the operator

def test_wu_round_matches_the_reference_recurrence():
    env = _env()
    served = torch.tensor([0.2, 0.9, 0.4, 0.15, 0.6, 0.55])
    for K in (1, 3, 10):
        got = JC.wu_round(env["innate"], served, env["beta"], env["alpha"],
                          env["W"], K)
        want = _ref_round(env["innate"], served, env["beta"], env["alpha"],
                          env["W"], K)
        assert torch.allclose(got, want, atol=1e-6), K


def test_wu_round_agrees_with_fjworld_run_wu_on_the_synthetic_env():
    """The tie between this file's arithmetic and the operator the LM runs
    use. Per-agent alpha, so it exercises the heterogeneous path."""
    env = _env()
    gap = JC.crosscheck_against_fjworld(env)
    assert gap <= 1e-5


def test_observed_passthrough_serves_current_opinions_on_o_only():
    env = _env()
    x = torch.tensor([0.11, 0.22, 0.33, 0.44, 0.55, 0.66])
    preds = torch.tensor([0.01, 0.02])
    s = JC.served_vector(x, preds, env["obs"], env["unobs"])
    assert torch.equal(s[env["obs"]], x[env["obs"]])
    assert torch.equal(s[env["unobs"]], preds)
    assert not torch.equal(s, x)          # non-vacuous: U really changed


# ---------------------------------------------- perfect prediction

def test_perfect_prediction_replay_matches_the_recurrence():
    """m_U(t) = x_U(t); with the observed passthrough that is s(t) = x(t)
    exactly, so the whole trajectory is a pure replay of the recurrence."""
    env = _env()
    K, T = 7, 6
    traj, served = JC.run_control("perfect_prediction", env, rounds=T,
                                  n_inner=K)
    x = env["innate"].clone()
    for t in range(T):
        assert torch.allclose(served[t], x[env["unobs"]], atol=1e-7), t
        x = _ref_round(env["innate"], x, env["beta"], env["alpha"],
                       env["W"], K)
        assert torch.allclose(traj[t], x, atol=1e-6), t
    # non-vacuous: the population actually moved off innate
    assert float((traj[0] - env["innate"]).abs().max()) > 1e-3


def test_perfect_prediction_serves_the_pre_round_held_out_opinion():
    """It is the CURRENT opinion, not the post-round one: serving x(t+1)
    would be a one-round lookahead, a strictly stronger predictor than
    'perfect'."""
    env = _env()
    traj, served = JC.run_control("perfect_prediction", env, rounds=4,
                                  n_inner=5)
    assert torch.allclose(served[0], env["innate"][env["unobs"]], atol=1e-7)
    for t in range(1, 4):
        assert torch.allclose(served[t], traj[t - 1][env["unobs"]], atol=1e-7)
        assert not torch.allclose(served[t], traj[t][env["unobs"]], atol=1e-6)


# ----------------------------------------------------------- beta = 0

def test_beta_zero_makes_predictions_irrelevant():
    env = _env()
    zero = torch.zeros_like(env["beta"])
    a, b = torch.zeros(N), torch.ones(N)
    assert not torch.allclose(a, b)                 # non-vacuous
    xa = JC.wu_round(env["innate"], a, zero, env["alpha"], env["W"], 20)
    xb = JC.wu_round(env["innate"], b, zero, env["alpha"], env["W"], 20)
    assert torch.equal(xa, xb)
    # and with beta > 0 the two served vectors DO separate, so the test
    # above is about beta and not about an inert operator
    ya = JC.wu_round(env["innate"], a, env["beta"], env["alpha"], env["W"], 20)
    yb = JC.wu_round(env["innate"], b, env["beta"], env["alpha"], env["W"], 20)
    assert float((ya - yb).abs().max()) > 1e-2


def test_no_platform_is_constant_from_round_one():
    """With a stateless human component and no platform channel there is
    nothing to carry information between rounds."""
    env = _env()
    traj, _ = JC.run_control("no_platform", env, rounds=5, n_inner=20)
    for t in range(1, 5):
        assert torch.allclose(traj[t], traj[0], atol=1e-7), t
    want = JC.wu_round(env["innate"], torch.zeros(N),
                       torch.zeros_like(env["beta"]), env["alpha"],
                       env["W"], 20)
    assert torch.allclose(traj[0], want, atol=1e-7)


def test_no_platform_changes_only_beta():
    """Same graph, same alpha, same innate: if it also moved alpha the
    control would no longer isolate the platform channel.

    ROUND 1 IS DELIBERATELY EQUAL. x(0) = innate, so a perfect predictor
    serves innate back and x_init = (1-beta) innate + beta innate =
    innate, exactly the beta=0 anchor. The two controls can only separate
    once the population has moved, i.e. from round 2 -- worth pinning,
    because "the platform control matches the no-platform control at
    round 1" is a correct result that reads like a bug.
    """
    env = _env()
    traj_np, _ = JC.run_control("no_platform", env, rounds=3, n_inner=10)
    traj_pp, _ = JC.run_control("perfect_prediction", env, rounds=3,
                                n_inner=10)
    assert torch.allclose(traj_np[0], traj_pp[0], atol=1e-7)
    assert float((traj_np[1] - traj_pp[1]).abs().max()) > 1e-3
    assert torch.equal(env["beta"], _env()["beta"])   # env not mutated


# ---------------------------------------------------- mean predictor

def test_mean_predictor_is_constant_on_u_and_equals_the_observed_mean():
    env = _env()
    x = torch.tensor([0.10, 0.20, 0.30, 0.40, 0.99, 0.01])
    m = JC.predict_u("mean", x_now=x, obs=env["obs"], unobs=env["unobs"],
                     X=env["X"])
    assert m.shape == (N - N_OBS,)
    assert float(m.std()) == 0.0
    assert float(m[0]) == pytest.approx(float(x[env["obs"]].mean()), abs=1e-7)
    # the held-out values are extreme here, so a leak would show
    assert not float(m[0]) == pytest.approx(float(x.mean()), abs=1e-4)


def test_mean_predictor_stays_flat_on_u_over_the_whole_loop():
    env = _env()
    _, served = JC.run_control("mean", env, rounds=8, n_inner=10)
    for t in range(8):
        assert float(served[t].std()) < 1e-7, t


# ------------------------------------------------- fitted predictors

@pytest.mark.parametrize("kind", ["mean", "ols"])
def test_fitted_predictors_never_see_held_out_labels(kind):
    """THE leakage test. x_U(t) is moved to something extreme; a predictor
    fit only on O must return the identical vector."""
    innate, W = _graph6()
    X = np.stack([np.ones(N), innate.numpy(),
                  np.arange(N, dtype=float)], axis=1)
    env = _env(X=X, colnames=["intercept", "innate", "idx"])
    x = torch.tensor([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    x_tampered = x.clone()
    x_tampered[env["unobs"]] = torch.tensor([0.99, 0.01])
    a = JC.predict_u(kind, x_now=x, obs=env["obs"], unobs=env["unobs"],
                     X=env["X"])
    b = JC.predict_u(kind, x_now=x_tampered, obs=env["obs"],
                     unobs=env["unobs"], X=env["X"])
    assert torch.equal(a, b)
    # and the observed labels DO move it, so the predictor is not inert
    x_obs_moved = x.clone()
    x_obs_moved[env["obs"]] += 0.2
    c = JC.predict_u(kind, x_now=x_obs_moved, obs=env["obs"],
                     unobs=env["unobs"], X=env["X"])
    assert float((c - a).abs().max()) > 1e-3


def test_ols_recovers_an_exactly_linear_signal_on_u():
    """A closed-form case: y = 0.2 + 0.5 * f on O, so the fit must return
    0.2 + 0.5 * f_U on the held-out rows."""
    f = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    X = np.stack([np.ones(N), f], axis=1)
    env = _env(X=X, colnames=["intercept", "f"])
    y = 0.2 + 0.5 * f
    x = torch.tensor(y, dtype=torch.float32)
    x[env["unobs"]] = 0.0                  # never seen by the fit
    m = JC.predict_u("ols", x_now=x, obs=env["obs"], unobs=env["unobs"],
                     X=env["X"])
    assert torch.allclose(m, torch.tensor([0.6, 0.7]), atol=1e-5)


def test_ols_differs_from_the_mean_when_the_features_carry_signal():
    """Otherwise 'ols' would be the mean predictor under another name."""
    f = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    env = _env(X=np.stack([np.ones(N), f], axis=1),
               colnames=["intercept", "f"])
    x = torch.tensor(0.2 + 0.5 * f, dtype=torch.float32)
    ols = JC.predict_u("ols", x_now=x, obs=env["obs"], unobs=env["unobs"],
                       X=env["X"])
    mean = JC.predict_u("mean", x_now=x, obs=env["obs"], unobs=env["unobs"],
                        X=env["X"])
    assert float((ols - mean).abs().max()) > 1e-2


def test_ols_predictions_are_clipped_to_the_opinion_scale():
    """Opinions live in [0, 1]; an unclipped extrapolation would push the
    anchor outside the scale every downstream metric assumes."""
    f = np.array([0.0, 0.2, 0.4, 0.6, 5.0, -5.0])
    env = _env(X=np.stack([np.ones(N), f], axis=1),
               colnames=["intercept", "f"])
    x = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.0, 0.0], dtype=torch.float32)
    m = JC.predict_u("ols", x_now=x, obs=env["obs"], unobs=env["unobs"],
                     X=env["X"])
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0
    assert torch.allclose(m, torch.tensor([1.0, 0.0]), atol=1e-5)


def test_lightgbm_is_optional_and_never_a_hard_dependency(tmp_path, capsys,
                                                          monkeypatch):
    monkeypatch.setattr(JC, "_lightgbm", lambda: None)
    env = _env()
    rows, conv = JC.analyse(tmp_path, rounds=3, n_inner=5, env=env,
                            controls=("mean", "lightgbm"))
    assert {c["control"] for c in conv} == {"mean"}
    out = capsys.readouterr().out
    assert "lightgbm" in out and "SKIPPED" in out
    assert (tmp_path / "jiduan_pokec_controls.csv").exists()
    assert len(rows) == 3


def test_lightgbm_predictor_refuses_to_pretend_when_unavailable():
    env = _env()
    with pytest.raises(RuntimeError):
        JC.predict_u("lightgbm", x_now=env["innate"], obs=env["obs"],
                     unobs=env["unobs"], X=env["X"], lgb=None)


# ------------------------------------------------------- the reporting

def test_every_metric_is_reported_for_o_u_and_full_separately(tmp_path):
    env = _env()
    rows, conv = JC.analyse(tmp_path, rounds=4, n_inner=5, env=env,
                            controls=("perfect_prediction", "mean"))
    for r in rows:
        for stat in ("mean", "sd", "w1_from_innate"):
            for sub in ("O", "U", "all"):
                assert f"{stat}_{sub}" in r, (stat, sub)
        assert {"late_mean_drift_O", "late_mean_drift_U",
                "late_mean_drift_all"} <= set(r)
    for c in conv:
        for sub in ("O", "U", "all"):
            assert f"late_mean_{sub}" in c and f"late_sd_{sub}" in c
            assert f"late_w1_from_innate_{sub}" in c

    # the three subsets are genuinely different numbers on this env, so
    # the split is not decorative
    r = rows[0]
    assert r["mean_O"] != r["mean_U"]
    assert r["n_O"] == N_OBS and r["n_U"] == N - N_OBS and r["n_all"] == N


def test_late_window_drift_is_zero_for_a_constant_trajectory():
    env = _env()
    x = JC.wu_round(env["innate"], torch.zeros(N),
                    torch.zeros_like(env["beta"]), env["alpha"], env["W"], 5)
    traj = torch.stack([x.clone() for _ in range(10)])
    d = JC._drifts(traj, env, 10)
    assert set(d) == {"O", "U", "all"}
    assert all(abs(v) < 1e-9 for v in d.values())


def test_late_window_drift_reads_a_moving_mean():
    """Non-vacuity for the test above: a trajectory whose mean is still
    climbing must produce a non-zero drift."""
    env = _env()
    traj = torch.stack([torch.full((N,), 0.1 * t) for t in range(10)])
    d = JC._drifts(traj, env, 10)
    lo, hi = JC.late_window(10)
    assert d["all"] == pytest.approx(0.1 * (hi - lo), abs=1e-5)
    assert abs(d["all"]) > JC.EQ_DRIFT


def test_the_late_window_is_a_suffix_not_a_fixed_pair():
    """A window clamped to (min(46, T), min(50, T)) collapses to a single
    round for any T < 46, which makes the drift trivially 0 and reports
    every short run as stationary. The window is the FINAL LATE_LEN
    rounds, and at the production T=50 that is exactly LATE."""
    assert JC.late_window(JC.T_ROUNDS) == JC.LATE
    assert JC.late_window(10) == (6, 10)
    assert JC.late_window(3) == (1, 3)
    assert JC.late_window(1) == (1, 1)

    env = _env()
    traj = torch.stack([torch.full((N,), 0.1 * t) for t in range(8)])
    assert JC._drifts(traj, env, 8)["all"] == pytest.approx(0.4, abs=1e-5)
    # a single-round window reports unknown, never a fake zero
    one = torch.stack([torch.full((N,), 0.3)])
    assert np.isnan(JC._drifts(one, env, 1)["all"])
    rows, conv = JC.summarize("mean", one, torch.zeros(1, N - N_OBS), env,
                              n_inner=5, rounds=1)
    assert rows[0]["stationary_all"] is False
    assert conv["stationary_U"] is False


def test_no_platform_reports_no_served_vector():
    """beta=0 serves nothing; a placeholder RMSE printed there would read
    as a predictor result."""
    env = _env()
    traj, served = JC.run_control("no_platform", env, rounds=3, n_inner=5)
    r, c = JC.summarize("no_platform", traj, served, env, n_inner=5, rounds=3)
    assert all(np.isnan(x["served_rmse_to_pre_U"]) for x in r)
    assert np.isnan(c["late_served_rmse_U"])


def test_w1_is_the_equal_n_wasserstein():
    a = np.array([0.0, 0.5, 1.0])
    assert JC.w1(a, a) == 0.0
    assert JC.w1(a, a + 0.25) == pytest.approx(0.25)
    assert JC.w1(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0  # sorted


# --------------------------------------------------------- real Pokec

@pytest.mark.slow
def test_the_real_pokec_environment_is_the_documented_one():
    """The only test that touches the dataset. It pins the facts every
    other number in this pipeline rests on."""
    env = JC.load_env()
    assert env["n"] == 2163 == JC.N_AGENTS
    assert int(env["obs"].sum()) == 1730 and int(env["unobs"].sum()) == 433
    assert float(env["alpha"].mean()) == pytest.approx(0.8909, abs=5e-4)
    assert float(env["beta"].mean()) == pytest.approx(0.8890, abs=5e-4)
    assert int((env["W"] > 0).sum().item()) > 0
    rowsum = env["W"].sum(dim=1)
    assert torch.allclose(rowsum, torch.ones(env["n"]), atol=1e-5)
    assert int((rowsum == 0).sum()) == 0             # no isolated node
    # the heterogeneous operator agrees with FJWorld on the real vectors
    assert JC.crosscheck_against_fjworld(env) <= 1e-5
    # and a short loop runs end to end
    traj, served = JC.run_control("perfect_prediction", env, rounds=2,
                                  n_inner=5)
    assert traj.shape == (2, 2163) and served.shape == (2, 433)
    assert torch.isfinite(traj).all()
