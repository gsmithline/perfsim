#!/usr/bin/env python3
"""MODEL-INDEPENDENT FJ CONTROLS for the robustness wave (2026-08-21).
CPU only -- no model is ever loaded.

Three controls, none of which needs a GPU because none of them depends on
what any language model would say:

  PERFECT PREDICTION   m(t) = x(t-1). The upper bound on what adaptation
      can achieve. Under the wu1 recurrence this is a linear map with
      contraction factor beta, so it converges geometrically -- and the
      rate is what tells you whether 30 rounds is a horizon or an
      anecdote. Computed, not assumed.

  NO PLATFORM          beta = 0. The predictions cannot reach the
      population at all, so x(t) = M x_innate for every t: a CONSTANT
      from round 1. Any trained arm that lands here has had no effect.

  FROZEN (lambda -> infinity)   m(t) = a fixed zero-shot vector. Also
      constant from round 1, for the same reason: with a stateless human
      component the only channel between rounds is the model, and a
      frozen model is the same every round. This is why the six frozen
      controls need no GPU loop -- one FJ application of a static vector
      IS the whole trajectory. The vectors themselves come from archived
      H100 runs (five reused after a field-by-field provenance check) or
      from the one extraction job.

K=100 IS THE PRODUCTION-MATCHED WU CONFIGURATION. At alpha=.9 the inner
loop contracts by ~0.9^K, so K=100 is effectively the FJ equilibrium of
the anchor -- which is what Wu's model specifies. The smaller K values in
the sweep are DIAGNOSTIC only: they show what truncating the inner loop
would have done, and they are not the wave's configuration.

Outputs (notes/pofd/fj_controls/):
  fj_controls.csv        per control, n_inner, round: mean/sd/W1
  fj_controls_conv.csv   convergence: per-round max |x(t) - x(t-1)|
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
OUT_DIR = REPO / "notes" / "pofd" / "fj_controls"
BETA = 0.5
BETA_BOUNDARY = 1.0
ALPHA = 0.9                 # PEER SUSCEPTIBILITY (Wu); stubbornness is .1
PRODUCTION_INNER = 100      # K, the production-matched Wu configuration
# K=100 is the production setting; the smaller values are retained as a
# DIAGNOSTIC only, to show how much the result depends on running the
# inner loop to convergence rather than truncating it
INNER_SWEEP = (1, 2, 5, 20, 100)
ROUNDS = 30
CONV_TOL = 1e-6


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_gated_ctl", str(HERE / "run_pokec_gated_lm.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fj_step(innate, preds, beta, alpha, n_inner, W):
    """One outer round of the wu1 recurrence. Deliberately a separate
    implementation from FJWorld.run_wu: the checker replays against this
    shape, and two independent expressions of one recurrence catch a
    typo that a shared helper would hide."""
    x0 = (1.0 - beta) * innate + beta * preds
    x = x0.clone()
    for _ in range(n_inner):
        x = (1.0 - alpha) * x0 + alpha * (W @ x)
    return x


def w1(a, b):
    """Equal-n 1-Wasserstein: mean |sort(a) - sort(b)|."""
    return float(np.abs(np.sort(np.asarray(a)) - np.sort(np.asarray(b))).mean())


def run_control(kind, innate, W, *, beta=BETA, alpha=ALPHA, n_inner=1,
                rounds=ROUNDS, frozen_vec=None):
    """Returns the trajectory [rounds, n] for one control."""
    x = innate.clone()
    out = []
    for _ in range(rounds):
        if kind == "perfect":
            preds = x.clone()                 # m(t) = x(t-1)
        elif kind == "no_platform":
            preds = torch.zeros_like(x)       # inert: beta is 0 below
        elif kind == "frozen":
            preds = frozen_vec
        else:
            raise ValueError(f"unknown control {kind!r}")
        b = 0.0 if kind == "no_platform" else beta
        x = fj_step(innate, preds, b, alpha, n_inner, W)
        out.append(x.clone())
    return torch.stack(out)


def analyse(out_dir, frozen_vecs=None):
    mod = _load_runner()
    setup = mod.load_movielens_setup(
        REPO / "experiments/data/movielens/ml-100k", "Action")
    innate = setup["innate"].float()
    W = setup["W"].float()
    rows, conv = [], []
    kinds = [("perfect", None), ("no_platform", None)]
    for name, vec in (frozen_vecs or {}).items():
        kinds.append((f"frozen:{name}", vec.float()))

    for kind_full, vec in kinds:
        kind = kind_full.split(":")[0]
        for n_inner in INNER_SWEEP:
            traj = run_control(kind, innate, W, n_inner=n_inner,
                               frozen_vec=vec)
            is_prod = (n_inner == PRODUCTION_INNER)
            prev = None
            first_conv = -1
            for t in range(traj.shape[0]):
                v = traj[t].numpy()
                step = (float((traj[t] - prev).abs().max())
                        if prev is not None else float("nan"))
                if prev is not None and step <= CONV_TOL and first_conv < 0:
                    first_conv = t
                rows.append({
                    "control": kind_full, "beta": (0.0 if kind == "no_platform"
                                                   else BETA),
                    "alpha": ALPHA, "n_inner": n_inner,
                    "production_matched": is_prod, "t": t,
                    "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "w1_from_innate": w1(v, innate.numpy()),
                    "step_from_prev": step,
                })
                prev = traj[t]
            conv.append({
                "control": kind_full, "n_inner": n_inner,
                "production_matched": is_prod,
                "first_round_within_tol": first_conv,
                "final_step": float((traj[-1] - traj[-2]).abs().max()),
                "converged": bool(first_conv >= 0),
                "note": ("constant from round 1 by construction: with a "
                         "stateless human component a constant predictor "
                         "gives a constant population"
                         if kind in ("frozen", "no_platform") else
                         "linear contraction with factor beta"),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    _csv(out_dir / "fj_controls.csv", rows)
    _csv(out_dir / "fj_controls_conv.csv", conv)
    _report(conv)
    return rows, conv


def _csv(path, rows):
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[fjctl] wrote {path} ({len(rows)} rows)")


def _report(conv):
    print(f"\n[fjctl] {'control':<22} {'K':>5} {'prod':>5} {'converged':>10} "
          f"{'first round':>12} {'final step':>12}")
    for c in conv:
        print(f"[fjctl] {c['control']:<22} {c['n_inner']:>5} "
              f"{'*' if c['production_matched'] else '':>5} "
              f"{str(c['converged']):>10} {c['first_round_within_tol']:>12} "
              f"{c['final_step']:>12.3e}")
    print(f"[fjctl] * = production-matched Wu configuration "
          f"(alpha={ALPHA}, K={PRODUCTION_INNER}); smaller K is DIAGNOSTIC "
          f"only, not the wave's setting.")
    print("[fjctl] converged here is the OUTER loop. The INNER loop also "
          "converges at alpha=.9, K=100 (~0.9^100), but that is the FJ "
          "fixed point of one round's anchor and says nothing about the "
          "outer model-population loop.")


def build_frozen(npz_path, out_dir, beta=BETA, alpha=ALPHA,
                 n_inner=PRODUCTION_INNER):
    """Replay each model's static zero-shot vector through ONE FJ update
    and write the frozen (lambda -> infinity) control the analyzer reads.

    K inner steps, ONE outer round: that is the whole trajectory. With a
    stateless human component
    the only channel between rounds is the model, so a constant predictor
    gives a constant population from round 1 -- which is exactly why these
    controls need no GPU loop, only the static vector. The vectors come
    from archived H100 runs (reused only after a field-by-field
    provenance check) or from the extraction job; their sha256 is carried
    into the artifact so the analyzer's reference is traceable.
    """
    mod = _load_runner()
    setup = mod.load_movielens_setup(
        REPO / "experiments/data/movielens/ml-100k", "Action")
    innate = setup["innate"].float()
    W = setup["W"].float()
    z = np.load(npz_path, allow_pickle=True)
    src = dict(z["sources"].item()) if "sources" in z else {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for model in [k for k in z.files if k != "sources"]:
        vec = torch.tensor(np.asarray(z[model], dtype=np.float32))
        if vec.shape != innate.shape:
            raise SystemExit(f"[fjctl] {model}: vector {tuple(vec.shape)} "
                             f"!= innate {tuple(innate.shape)}")
        if not torch.isfinite(vec).all():
            raise SystemExit(f"[fjctl] {model}: non-finite zero-shot vector")
        post = fj_step(innate, vec, beta, alpha, n_inner, W)
        # the constancy claim, asserted rather than assumed
        again = fj_step(innate, vec, beta, alpha, n_inner, W)
        assert torch.equal(post, again)
        import hashlib
        torch.save({"post_fj": post, "zero_shot": vec,
                    "source_tag": src.get(model, "UNRECORDED"),
                    "zero_shot_sha256": hashlib.sha256(
                        vec.contiguous().numpy().tobytes()).hexdigest(),
                    "beta": beta, "alpha": alpha, "n_inner": n_inner,
                    "note": ("constant from round 1: a stateless human "
                             "component plus a constant predictor")},
                   out_dir / f"frozen_{model}.pt")
        written.append(model)
    print(f"[fjctl] wrote {len(written)} frozen controls to {out_dir}: "
          f"{', '.join(written)}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--frozen-npz", type=Path, default=None,
                    help="npz of model -> zero-shot vector; builds the "
                         "frozen (lambda -> infinity) FJ controls")
    ap.add_argument("--frozen-out", type=Path, default=None)
    # the frozen control is beta-SPECIFIC: x_init = (1-beta) innate +
    # beta m, so the beta=.5 reference is simply wrong for the beta=1
    # wave. Built per beta, and the artifact records which.
    ap.add_argument("--frozen-beta", type=float, default=BETA)
    args = ap.parse_args()
    if args.frozen_npz:
        build_frozen(args.frozen_npz,
                     args.frozen_out or (OUT_DIR.parent / "fj_robustness"
                                         / "frozen"),
                     beta=args.frozen_beta)
        return 0
    analyse(args.out_dir)
    print(f"[fjctl] outputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
