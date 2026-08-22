#!/usr/bin/env python3
"""MODEL-FREE CONTROLS FOR THE JIDUAN/WU POKEC REPLICATION (2026-08-22).
CPU only -- no language model is ever loaded, nothing is downloaded.

WHAT THIS IS. The Wu outer loop run on the EXACT Pokec environment the LM
runs use (same 2163-agent LCC, same normalized graph, same innate vector,
same per-agent alpha and beta), with the platform replaced by predictors
whose behaviour is known in closed form. These are the reference points a
language-model arm has to be read against: without them a trajectory is a
number with nothing to compare it to.

THE SPLIT IS THE POINT. Pokec ships 1730 agents with observed opinions
(O = the first 1730 rows, y_label) and 433 held out (U = the last 433,
y_unlabel). The platform SEES O and has to PREDICT U:

    s_i(t) = x_i(t)      i in O        (observed passthrough)
             m_i(t)      i in U        (the predictor's output)

    x_init_i(t) = (1 - beta_i) innate_i + beta_i s_i(t)
    u^(0)       = x_init(t)
    u^(l+1)_i   = (1 - alpha_i) x_init_i(t) + alpha_i (P u^(l))_i
    x(t+1)      = u^(K)

with K = 100 inner steps and T = 50 outer rounds. alpha is the dataset's
PEER SUSCEPTIBILITY vector (hetero_peer_sus2163.pkl, mean .8909) and beta
the platform-trust vector (hetero_platform_sus2163.pkl, mean .8890).
NOTE THE CONVENTION: FJWorld stores 1 - alpha internally (stubbornness),
see FJWorld.alpha_to_peer_sus; this module works in alpha throughout and
cross-checks one round against FJWorld.run_wu so the two agree.

EVERY QUANTITY IS REPORTED THREE TIMES -- on O, on U, and on the full
population. The headline estimands of this project live on HELD-OUT U:
an effect that only shows up on O is a passthrough of opinions the
platform was handed, not a prediction.

CONTROLS
  no_platform         beta forced to 0. The predictions cannot reach the
      population at all, so the trajectory is the innate-anchored FJ
      equilibrium and is CONSTANT from round 1. Any arm landing here had
      no effect.
  perfect_prediction  m_U(t) = the true current x_U(t). Combined with the
      observed passthrough this makes s(t) = x(t) exactly: the upper
      bound on what any predictor could do, and the fastest the feedback
      loop can possibly move.
  mean                m_U(t) = mean of x_O(t). The intercept-only
      predictor: the least a fitted model can be said to have learned.
  ols                 least squares on the profile features (age, gender,
      relation_to_alcohol), refit on O each round against x_O(t).
  lightgbm            same features and protocol, gradient boosting.
      OPTIONAL: if lightgbm is not importable this control is skipped
      with a printed note and the rest still run. It is not a dependency.

Outputs (notes/pofd/jiduan_pokec_replication/):
  jiduan_pokec_controls.csv       per control, per round: mean / sd /
      W1-from-innate on O, U and all; the served-prediction summary on U;
      the per-round step; and the late-window mean drift (outer
      stationarity) carried on every row so the file is self-contained.
  jiduan_pokec_controls_conv.csv  one row per control: convergence and
      late-window summary, again split O / U / all.

DIRECTION-NEUTRAL. Nothing here asserts which control ends up higher,
lower, wider or narrower; the script computes and reports.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
REPO = HERE.parent.parent.parent
OUT_DIR = REPO / "notes" / "pofd" / "jiduan_pokec_replication"
POKEC_DIR = REPO / "examples" / "pokec"

# ---- the production-matched Wu configuration -------------------------
K_FJ = 100               # inner FJ steps per outer round
T_ROUNDS = 50            # outer rounds
N_OBSERVED = 1730        # |O|; the remaining rows are U (433 on Pokec)
N_AGENTS = 2163          # Pokec LCC size, asserted not assumed

LATE_LEN = 5             # the late window is the FINAL LATE_LEN rounds
LATE = (46, 50)          # inclusive, 1-indexed: that window at T = 50
EQ_DRIFT = 0.005         # |late-window mean drift| below this = stationary
CONV_TOL = 1e-6          # these controls are noiseless, so a step tolerance
                         # is meaningful here in a way it is not for LM arms

CONTROLS = ("no_platform", "perfect_prediction", "mean", "ols", "lightgbm")


# ======================================================================
# environment
# ======================================================================

def _load_runner():
    """The Pokec loader lives in the runner; import it rather than
    re-deriving the environment, so these controls cannot drift from the
    environment the LM arms actually ran on."""
    spec = importlib.util.spec_from_file_location(
        "run_gated_jiduan", str(HERE / "run_pokec_gated_lm.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def observed_unobserved_masks(n: int, n_observed: int = N_OBSERVED):
    """O = the first n_observed rows, U = the rest. Returns bool masks.

    The order is not cosmetic: load_pokec_setup builds innate as
    concat(y_label, y_unlabel), so the prefix IS the labeled set.
    """
    if not (0 < n_observed <= n):
        raise ValueError(f"n_observed must be in (0, {n}]; got {n_observed}")
    obs = torch.zeros(n, dtype=torch.bool)
    obs[:n_observed] = True
    return obs, ~obs


def make_env(innate, W, alpha, beta, *, n_observed=N_OBSERVED, X=None,
             colnames=None, profiles=None, runner=None):
    """Assemble the environment dict every function here consumes.

    One definition of the contract, so a synthetic environment used in a
    test is structurally the same object as the real Pokec one.
    """
    innate = torch.as_tensor(innate).float()
    n = int(innate.shape[0])
    W = torch.as_tensor(W).float()
    if W.shape != (n, n):
        raise ValueError(f"W must be ({n}, {n}); got {tuple(W.shape)}")
    alpha = torch.as_tensor(alpha).float().expand(n).clone()
    beta = torch.as_tensor(beta).float().expand(n).clone()
    obs, unobs = observed_unobserved_masks(n, n_observed)
    if X is None:
        # intercept only: the fitted predictors then reduce to the mean,
        # which is a legitimate (and clearly labelled) degenerate design
        X = np.ones((n, 1), dtype=np.float64)
        colnames = ["intercept"]
    return {
        "n": n, "innate": innate, "W": W, "alpha": alpha, "beta": beta,
        "obs": obs, "unobs": unobs, "profiles": profiles,
        "X": np.asarray(X, dtype=np.float64),
        "colnames": list(colnames or []), "runner": runner,
    }


def load_env(pokec_dir: Path = POKEC_DIR, n_observed: int = N_OBSERVED):
    """The exact Pokec environment, plus the O/U split and the design
    matrix the fitted predictors use."""
    mod = _load_runner()
    setup = mod.load_pokec_setup(Path(pokec_dir))
    n = int(setup["n"])
    obs, _ = observed_unobserved_masks(n, n_observed)
    X, colnames = build_design(setup["profiles"], mod.translate_alcohol, obs)
    # THE DATASET VECTORS. hetero_peer_sus is Wu's alpha (peer
    # susceptibility, mean .8909 -- which is why the homogeneous waves ran
    # alpha=.9); hetero_platform_sus is beta.
    return make_env(setup["innate"], setup["W"], setup["peer_sus"],
                    setup["platform_sus"], n_observed=n_observed, X=X,
                    colnames=colnames, profiles=setup["profiles"], runner=mod)


def build_design(profiles, translate_alcohol, obs_mask):
    """Design matrix for the fitted predictors, from the SAME three
    profile fields the LM prompt shows (PROMPT_COLS).

    ENCODING, stated explicitly because it is a modelling choice:
      intercept       a column of ones.
      age_z           age standardized by the mean/sd of the NON-MISSING
                      ages among the OBSERVED agents only (no held-out
                      label ever enters, and the features are static, so
                      this is a fixed transform computed once).
      age_missing     Pokec encodes a missing age as 0.0 (the prompt
                      builder drops the line); 662/2163 rows. Those rows
                      get age_z = 0 and this indicator set to 1, so the
                      linear model can give the missing group its own
                      intercept instead of treating it as "aged zero".
      gender_male     1 for gender == 1.0, else 0. Pokec has no missing
                      gender in this LCC, so no indicator is needed; if
                      one ever appears it lands in the female group and
                      gender_missing carries it.
      gender_missing  1 where gender is neither 0.0 nor 1.0.
      alc_*           one-hot over translate_alcohol(raw text) -- the
                      runner's own mapping, i.e. exactly the string the LM
                      prompt displays -- with the MODAL category dropped
                      as the reference level. Free-text Slovak collapses
                      to a handful of buckets this way; rare buckets can
                      be absent from O, which leaves a rank-deficient
                      column, and lstsq's minimum-norm solution handles it
                      (documented rather than silently pruned).
    """
    import pandas as pd

    n = len(profiles)
    age = pd.to_numeric(profiles["age"], errors="coerce").to_numpy(dtype=float)
    age_missing = ~np.isfinite(age) | (age <= 0.0)
    age_f = np.where(age_missing, 0.0, age)
    o = np.asarray(obs_mask.numpy(), dtype=bool)
    ref = age_f[o & ~age_missing]
    mu = float(ref.mean()) if ref.size else 0.0
    sd = float(ref.std(ddof=1)) if ref.size > 1 else 1.0
    sd = sd if sd > 1e-12 else 1.0
    age_z = np.where(age_missing, 0.0, (age_f - mu) / sd)

    gender = pd.to_numeric(profiles["gender"], errors="coerce").to_numpy(dtype=float)
    gender_male = (gender == 1.0).astype(float)
    gender_missing = (~np.isin(gender, (0.0, 1.0))).astype(float)

    cats = profiles["relation_to_alcohol"].map(translate_alcohol).astype(str)
    counts = cats.value_counts()
    reference = str(counts.index[0])            # modal bucket = reference
    levels = [c for c in counts.index if str(c) != reference]

    cols = [np.ones(n), age_z, age_missing.astype(float),
            gender_male, gender_missing]
    names = ["intercept", "age_z", "age_missing", "gender_male",
             "gender_missing"]
    for lv in levels:
        cols.append((cats == lv).to_numpy(dtype=float))
        names.append(f"alc[{lv}]")
    X = np.stack(cols, axis=1).astype(np.float64)
    return X, names


# ======================================================================
# the operator
# ======================================================================

def wu_round(innate, served, beta, alpha, W, n_inner=K_FJ):
    """ONE outer round of the Wu recurrence, per-agent alpha and beta.

    Deliberately a SEPARATE implementation from FJWorld.run_wu (the same
    stance fj_controls.py takes): two independent expressions of one
    recurrence catch a typo that a shared helper would hide. They are
    checked against each other by crosscheck_against_fjworld().

        x_init = (1 - beta) innate + beta served
        u^(0)  = x_init
        u^(l+1)= (1 - alpha) x_init + alpha (W u^(l))
    """
    if not isinstance(n_inner, int) or n_inner < 1:
        raise ValueError(f"n_inner must be a positive int; got {n_inner!r}")
    x_init = (1.0 - beta) * innate + beta * served
    u = x_init.clone()
    for _ in range(n_inner):
        u = (1.0 - alpha) * x_init + alpha * (W @ u)
    return u


def served_vector(x_now, preds_u, obs, unobs):
    """s(t): the observed agents pass their CURRENT opinion through, the
    held-out agents get the predictor's output. The platform never sees
    x_U(t) unless the control is perfect_prediction, which is exactly
    what makes that control an upper bound."""
    s = x_now.clone()
    s[unobs] = preds_u.to(dtype=s.dtype)
    return s


def crosscheck_against_fjworld(env, atol=1e-5):
    """Assert the local operator and FJWorld.run_wu (per-agent alpha)
    agree on one real round. Cheap, and it is the only thing tying this
    file's numbers to the operator the LM runs use."""
    from perfsim.environments.dynamics import FJWorld

    innate, W = env["innate"], env["W"]
    alpha, beta = env["alpha"], env["beta"]
    n = env["n"]
    g = torch.Generator().manual_seed(0)
    preds = torch.rand(n, generator=g)
    world = FJWorld(innate=innate, graph=W,
                    peer_sus=FJWorld.alpha_to_peer_sus(alpha),
                    platform_sus=beta, features=innate)
    world.run_wu(preds, alpha=alpha, n_inner=K_FJ)
    mine = wu_round(innate, preds, beta, alpha, W, K_FJ)
    gap = float((world.state["opinion"] - mine).abs().max())
    if not gap <= atol:
        raise SystemExit(
            f"[jiduan] operator mismatch: local wu_round and "
            f"FJWorld.run_wu disagree by {gap:.3e} (> {atol})")
    return gap


# ======================================================================
# predictors -- each is fit on O with labels x_O(t) and predicts U only
# ======================================================================

def _lightgbm():
    try:
        import lightgbm  # noqa: F401
        return lightgbm
    except Exception:
        return None


def predict_u(kind, *, x_now, obs, unobs, X, lgb=None, seed=0):
    """m_U(t). Returns a tensor of length |U|.

    Every fitted predictor sees ONLY x_O(t) as labels; none of them is
    ever shown x_U(t). perfect_prediction is the deliberate exception and
    is labelled as such.
    """
    y_o = x_now[obs].numpy().astype(np.float64)
    n_u = int(unobs.sum())
    if kind == "perfect_prediction":
        return x_now[unobs].clone()
    if kind == "no_platform":
        # beta is 0, so nothing served can reach the population. Zeros
        # make that inertness visible rather than hiding a real vector.
        return torch.zeros(n_u, dtype=x_now.dtype)
    if kind == "mean":
        return torch.full((n_u,), float(y_o.mean()), dtype=x_now.dtype)
    if kind == "ols":
        Xo, Xu = X[obs.numpy()], X[unobs.numpy()]
        coef, *_ = np.linalg.lstsq(Xo, y_o, rcond=None)
        return torch.tensor(np.clip(Xu @ coef, 0.0, 1.0), dtype=x_now.dtype)
    if kind == "lightgbm":
        if lgb is None:
            raise RuntimeError("lightgbm predictor requested but unavailable")
        Xo, Xu = X[obs.numpy()], X[unobs.numpy()]
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                  num_leaves=15, min_child_samples=20,
                                  subsample=1.0, random_state=seed,
                                  verbose=-1)
        model.fit(Xo, y_o)
        return torch.tensor(np.clip(model.predict(Xu), 0.0, 1.0),
                            dtype=x_now.dtype)
    raise ValueError(f"unknown control {kind!r}")


# ======================================================================
# the loop
# ======================================================================

def run_control(kind, env, *, rounds=T_ROUNDS, n_inner=K_FJ, lgb=None,
                seed=0):
    """Returns (traj [rounds, n], served_u [rounds, |U|]).

    traj[t] is x(t+1): the population AFTER t+1 outer rounds, so row 0 is
    round 1. x(0) = innate.
    """
    if kind not in CONTROLS:
        raise ValueError(f"unknown control {kind!r}; expected one of {CONTROLS}")
    innate, W = env["innate"], env["W"]
    alpha = env["alpha"]
    # THE ONLY THING no_platform CHANGES is beta -> 0. Same operator, same
    # graph, same alpha, so any difference is attributable to the platform
    # channel and to nothing else.
    beta = torch.zeros_like(env["beta"]) if kind == "no_platform" else env["beta"]
    obs, unobs = env["obs"], env["unobs"]
    x = innate.clone()
    traj, served_u = [], []
    for _ in range(rounds):
        m_u = predict_u(kind, x_now=x, obs=obs, unobs=unobs, X=env["X"],
                        lgb=lgb, seed=seed)
        s = served_vector(x, m_u, obs, unobs)
        x = wu_round(innate, s, beta, alpha, W, n_inner)
        traj.append(x.clone())
        served_u.append(m_u.clone())
    return torch.stack(traj), torch.stack(served_u)


# ======================================================================
# reporting
# ======================================================================

def w1(a, b):
    """Equal-n 1-Wasserstein: mean |sort(a) - sort(b)|. Same definition
    as fj_controls.py / analyze_fj_robustness.py."""
    return float(np.abs(np.sort(np.asarray(a)) - np.sort(np.asarray(b))).mean())


def _subsets(env):
    return (("O", env["obs"]), ("U", env["unobs"]),
            ("all", torch.ones(env["n"], dtype=torch.bool)))


def late_window(rounds):
    """The final LATE_LEN rounds, 1-indexed inclusive.

    Defined as a SUFFIX rather than the fixed (46, 50) pair so a shorter
    run does not collapse the window to a single round -- which would
    make the drift trivially 0 and report every short run as stationary.
    At the production T = 50 it is exactly LATE.
    """
    rounds = int(rounds)
    return max(1, rounds - LATE_LEN + 1), rounds


def _drifts(traj, env, rounds):
    """Late-window mean drift per subset: mean at the last round of the
    window minus mean at the first. This is the OUTER stationarity test
    the FJ analyzer uses (a vanishing per-round step is the inner loop's
    property and says nothing about the model-population loop).

    NaN when the window holds a single round, so a run too short to have
    a late window reports 'unknown' instead of a fake zero.
    """
    lo, hi = late_window(rounds)
    out = {}
    for name, m in _subsets(env):
        if hi <= lo:
            out[name] = float("nan")
            continue
        a = float(traj[lo - 1][m].mean())
        b = float(traj[hi - 1][m].mean())
        out[name] = b - a
    return out


def summarize(kind, traj, served_u, env, *, n_inner, rounds):
    innate = env["innate"]
    unobs = env["unobs"]
    drift = _drifts(traj, env, rounds)
    beta_used = 0.0 if kind == "no_platform" else float(env["beta"].mean())
    rows = []
    prev = None
    first_conv = -1
    for t in range(traj.shape[0]):
        x = traj[t]
        step = float((x - prev).abs().max()) if prev is not None else float("nan")
        step_u = (float((x[unobs] - prev[unobs]).abs().max())
                  if prev is not None else float("nan"))
        if prev is not None and step <= CONV_TOL and first_conv < 0:
            first_conv = t + 1
        row = {
            "control": kind, "round": t + 1, "K_inner": n_inner,
            "T": rounds, "alpha_mean": float(env["alpha"].mean()),
            "beta_mean": beta_used,
            "n_O": int(env["obs"].sum()), "n_U": int(unobs.sum()),
            "n_all": env["n"],
        }
        for name, m in _subsets(env):
            v = x[m].numpy()
            row[f"mean_{name}"] = float(v.mean())
            row[f"sd_{name}"] = float(v.std(ddof=1))
            row[f"w1_from_innate_{name}"] = w1(v, innate[m].numpy())
        # What the platform actually served on the held-out set, and how
        # far it was from the truth it never saw. The comparison is
        # against x_U(t), the START-of-round opinion the predictor is
        # nominally estimating -- not against x_U(t+1), which the round
        # itself just moved.
        m_u = served_u[t].numpy()
        pre_u = (innate if t == 0 else traj[t - 1])[unobs].numpy()
        if kind == "no_platform":
            # beta = 0, so no vector is served at all; reporting the
            # placeholder's distance to anything would invite reading it
            # as a predictor result.
            row["served_mean_U"] = float("nan")
            row["served_sd_U"] = float("nan")
            row["served_rmse_to_pre_U"] = float("nan")
        else:
            row["served_mean_U"] = float(m_u.mean())
            row["served_sd_U"] = (float(m_u.std(ddof=1)) if m_u.size > 1
                                  else float("nan"))
            row["served_rmse_to_pre_U"] = float(
                np.sqrt(((m_u - pre_u) ** 2).mean()))
        row["step_max_all"] = step
        row["step_max_U"] = step_u
        for name in ("O", "U", "all"):
            row[f"late_mean_drift_{name}"] = drift[name]
        row["stationary_all"] = bool(abs(drift["all"]) <= EQ_DRIFT)
        row["stationary_U"] = bool(abs(drift["U"]) <= EQ_DRIFT)
        rows.append(row)
        prev = x

    lo, hi = late_window(rounds)
    late = traj[lo - 1:hi]
    conv = {
        "control": kind, "K_inner": n_inner, "T": rounds,
        "beta_mean": beta_used, "alpha_mean": float(env["alpha"].mean()),
        "first_round_within_tol": first_conv,
        "converged_step": bool(first_conv > 0),
        "final_step": (float((traj[-1] - traj[-2]).abs().max())
                       if traj.shape[0] > 1 else float("nan")),
        "late_window": f"{lo}-{hi}",
        # predictor accuracy on the HELD-OUT set, averaged over the late
        # window: RMSE(m_U(t), x_U(t)). NaN for no_platform, which serves
        # nothing.
        "late_served_rmse_U": float(np.mean(
            [r["served_rmse_to_pre_U"] for r in rows[lo - 1:hi]])),
    }
    for name, m in _subsets(env):
        sub = late[:, m]
        conv[f"late_mean_{name}"] = float(sub.mean())
        conv[f"late_sd_{name}"] = float(np.mean(
            [x.numpy().std(ddof=1) for x in sub]))
        conv[f"late_w1_from_innate_{name}"] = float(np.mean(
            [w1(x.numpy(), innate[m].numpy()) for x in sub]))
        conv[f"late_mean_drift_{name}"] = drift[name]
        conv[f"stationary_{name}"] = bool(abs(drift[name]) <= EQ_DRIFT)
    conv["note"] = {
        "no_platform": "beta=0: the platform channel is severed, so the "
                       "trajectory is the innate-anchored FJ equilibrium "
                       "and is constant from round 1 by construction",
        "perfect_prediction": "m_U(t) = x_U(t): with the observed "
                              "passthrough this is s(t)=x(t), the upper "
                              "bound on predictor quality",
        "mean": "intercept-only predictor on U",
        "ols": "least squares on the profile features, refit on O each round",
        "lightgbm": "gradient boosting on the profile features, refit on O "
                    "each round",
    }[kind]
    return rows, conv


def _csv(path, rows):
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[jiduan] wrote {path} ({len(rows)} rows)")


def _report(conv_rows, skipped):
    if not conv_rows:
        print("\n[jiduan] no control produced a trajectory")
        for s in skipped:
            print(f"[jiduan] SKIPPED {s}")
        return
    print(f"\n[jiduan] late window {conv_rows[0]['late_window']} of "
          f"T={conv_rows[0]['T']}, K={conv_rows[0]['K_inner']} inner steps")
    print(f"[jiduan] {'control':<19} {'set':>4} {'late mean':>10} "
          f"{'late sd':>9} {'W1 innate':>10} {'drift':>9} {'stat':>5} "
          f"{'pred RMSE':>10}")
    for c in conv_rows:
        for name in ("O", "U", "all"):
            rmse = (f"{c['late_served_rmse_U']:>10.4f}"
                    if name == "U" and c["late_served_rmse_U"] == c["late_served_rmse_U"]
                    else f"{'':>10}")
            print(f"[jiduan] {c['control'] if name == 'O' else '':<19} "
                  f"{name:>4} {c[f'late_mean_{name}']:>10.4f} "
                  f"{c[f'late_sd_{name}']:>9.4f} "
                  f"{c[f'late_w1_from_innate_{name}']:>10.4f} "
                  f"{c[f'late_mean_drift_{name}']:>+9.5f} "
                  f"{str(c[f'stationary_{name}']):>5} {rmse}")
    print(f"[jiduan] stat = |late-window mean drift| <= {EQ_DRIFT} (OUTER "
          f"stationarity; the inner loop's own contraction is separate)")
    print("[jiduan] U is the HELD-OUT set the platform must predict; O is "
          "passed through, so O moving is not evidence of prediction.")
    for s in skipped:
        print(f"[jiduan] SKIPPED {s}")


# ======================================================================

def analyse(out_dir=OUT_DIR, *, pokec_dir=POKEC_DIR, rounds=T_ROUNDS,
            n_inner=K_FJ, controls=CONTROLS, seed=0, strict_n=True, env=None):
    if env is None:
        env = load_env(pokec_dir)
        if strict_n and env["n"] != N_AGENTS:
            raise SystemExit(
                f"[jiduan] expected N={N_AGENTS} agents, got {env['n']}")
    print(f"[jiduan] Pokec: N={env['n']}, |O|={int(env['obs'].sum())}, "
          f"|U|={int(env['unobs'].sum())}, alpha mean "
          f"{float(env['alpha'].mean()):.4f}, beta mean "
          f"{float(env['beta'].mean()):.4f}", flush=True)
    gap = crosscheck_against_fjworld(env)
    print(f"[jiduan] operator cross-check vs FJWorld.run_wu (per-agent "
          f"alpha): max |diff| = {gap:.3e}", flush=True)
    print(f"[jiduan] design matrix: {env['X'].shape[1]} columns "
          f"{env['colnames']}", flush=True)

    lgb = _lightgbm()
    skipped = []
    rows, conv_rows = [], []
    for kind in controls:
        if kind == "lightgbm" and lgb is None:
            skipped.append("lightgbm: not importable in this environment "
                           "(pip install lightgbm to add it). Every other "
                           "control still ran; lightgbm is NOT a dependency "
                           "of this pipeline.")
            print(f"[jiduan] {skipped[-1]}", flush=True)
            continue
        traj, served = run_control(kind, env, rounds=rounds, n_inner=n_inner,
                                   lgb=lgb, seed=seed)
        r, c = summarize(kind, traj, served, env, n_inner=n_inner,
                         rounds=rounds)
        rows.extend(r)
        conv_rows.append(c)
        print(f"[jiduan] {kind}: done ({rounds} rounds)", flush=True)

    out_dir = Path(out_dir)
    _csv(out_dir / "jiduan_pokec_controls.csv", rows)
    _csv(out_dir / "jiduan_pokec_controls_conv.csv", conv_rows)
    _report(conv_rows, skipped)
    return rows, conv_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--pokec-dir", default=str(POKEC_DIR))
    ap.add_argument("--rounds", type=int, default=T_ROUNDS)
    ap.add_argument("--k-inner", type=int, default=K_FJ)
    ap.add_argument("--controls", default=",".join(CONTROLS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    kinds = tuple(k.strip() for k in a.controls.split(",") if k.strip())
    torch.manual_seed(a.seed)
    analyse(Path(a.out), pokec_dir=Path(a.pokec_dir), rounds=a.rounds,
            n_inner=a.k_inner, controls=kinds, seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
