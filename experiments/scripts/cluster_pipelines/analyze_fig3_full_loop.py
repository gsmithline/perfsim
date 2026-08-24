#!/usr/bin/env python3
"""ANALYSIS for the redesigned Figure 3 (reference_retention_memory_equilibrium),
UNIFIED 108-CELL GRID (2026-08-24 v2).

Reports, per conceptual cell: the post-peer outcome value, the
final-window drift, and an OUTCOME classification.  Writes a tidy
per-cell CSV and a per-round CSV that plot_fig3_previews.py consumes.

EVERY POPULATION STATISTIC IS POST-PEER.  op_raw[t] is the END-OF-ROUND
state: the peer sweeps run last, after the AI mixture.

OUTCOMES -- a value is only called an equilibrium when it earned it:
    equilibrium        settled by the inherited drift test
    extend_to_60/100   NOT settled at the current horizon; the cell needs
                       a targeted horizon extension (fig3_full_loop_ext)
    cyclic_long_run    not settled at >= 100 rounds with sign-alternating
                       tail drift: a persistent cycle, reported as a
                       long-run outcome, never as a fixed equilibrium
    drifting_long_run  not settled at >= 100 rounds, monotone drift
--paper mode HARD-FAILS (exit 4) while any cell still needs extension:
the paper plot must never print a number the dynamics have not finished
producing.  Long-run/cyclic cells pass --paper but carry their label.

HORIZONS.  Each GPU cell is analysed at the LONGEST horizon available:
a pofdf3_ _r100 extension beats _r60 beats the base artifact (reused
pofdps_ cells are already 60).  The horizon used is recorded per cell.

INHERITED VERBATIM from plot_section3_lambda_s100.py (the script that
draws the figure today) so the numbers stay comparable:
    settled(op): means = op.mean(1)[-10:]
                 shift  = means[-5:].mean() - means[:5].mean()
                 settled iff |shift| <= 0.005
Window and tolerance are flags but DEFAULT to those values; the choice
is printed and written into every CSV row.

THE THREE NAMES, once:  beta = W_PLAT, gamma = INNATE_LAMBDA,
lambda = kl_beta.  The homophily gamma is a different gamma and is 0.

HARD FAILURES (this tool refuses to produce numbers rather than partial
ones): any required cell absent (exit 3); the grid ungated -- --gate-json
must point at a check_fig3_full_loop verdict with ok=true, or
--allow-ungated must be passed and the output is stamped UNGATED (exit 2);
--paper with unsettled cells (exit 4).

When cells need extension this writes fig3_extension_request.json into
the out-dir; copy it to experiments/condor/ and commit, and the
fig3_full_loop_ext key runs exactly those cells at 60/100 rounds.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
GEN = REPO / "experiments" / "condor" / "gen_pofd_sweep.py"
DEFAULT_RUN_ROOTS = (
    Path("/home/gsmithline/perfsim/runs/pokec_gated_lm"),
    REPO / "runs" / "pokec_gated_lm",
    REPO / "notes" / "pofd" / "cluster",
)
DEFAULT_FROZEN_DIR = REPO / "notes" / "pofd" / "frozen_replay"
DEFAULT_OUT = REPO / "notes" / "pofd" / "fig3_full_loop"

BASE_ROUNDS = 30       # plot_section3_lambda_s100.ROUNDS, verbatim
WINDOW = 10            # settled() looks at the last 10 rounds
DRIFT_TOL = 0.005      # settled() tolerance, verbatim
MAX_HORIZON = 100      # beyond this an unsettled cell is a long-run outcome


def _load_gen():
    spec = importlib.util.spec_from_file_location("_gen_f3a", str(GEN))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gen_f3a"] = mod
    spec.loader.exec_module(mod)
    return mod


def _num(v):
    return f"{v:g}".replace(".", "p")


def _find(tag, roots):
    for r in roots:
        p = Path(r) / tag / "trajectory.pt"
        if p.exists():
            return p
    return None


def _frozen_path(beta, gamma, sweeps, frozen_dir):
    for rounds in (BASE_ROUNDS, 60):
        p = Path(frozen_dir) / (f"frz_k{_num(gamma)}_w{_num(beta)}"
                                f"_eaopen_esopen_sw{sweeps}_s0_r{rounds}.pt")
        if p.exists():
            return p
    return None


def _classify(op, horizon, window, tol):
    """(final mean, final sd, drift, outcome) on the POST-PEER population
    at the given horizon, using the inherited half-vs-half convention."""
    op = np.asarray(op, dtype=float)[:horizon]
    means = op.mean(axis=1)
    tail = means[-window:]
    h = window // 2
    drift = float(tail[-h:].mean() - tail[:h].mean())
    if abs(drift) <= tol:
        outcome = "equilibrium"
    elif horizon < 60:
        outcome = "extend_to_60"
    elif horizon < MAX_HORIZON:
        outcome = "extend_to_100"
    else:
        # persistent non-convergence at the max horizon: cycle vs drift.
        # A cycle alternates the sign of consecutive tail deltas; a drift
        # keeps it.
        deltas = np.diff(tail)
        nz = deltas[np.abs(deltas) > 1e-9]
        if nz.size >= 4 and \
                float(np.mean(np.sign(nz[1:]) != np.sign(nz[:-1]))) >= 0.7:
            outcome = "cyclic_long_run"
        else:
            outcome = "drifting_long_run"
    return (float(means[horizon - 1]), float(op[horizon - 1].std()),
            drift, outcome)


def load_gpu_cell(beta, gamma, lam, g, roots):
    """Longest-horizon artifact for a finite-lambda cell:
    pofdf3 _r100 > pofdf3 _r60 > the base (reused or pofdf3 r30).
    -> (op, innate, source_tag, horizon) or (None, None, reason, 0)."""
    gam = 1.0 if gamma is None else gamma
    base = g.F3_REUSED.get((beta, gamma, lam)) or g.f3_tag(beta, gam, lam)
    candidates = [(g.f3_tag(beta, gam, lam, rounds=100), 100),
                  (g.f3_tag(beta, gam, lam, rounds=60), 60),
                  (base, 60 if base.endswith("_r60") else BASE_ROUNDS)]
    for tag, horizon in candidates:
        p = _find(tag, roots)
        if p is not None:
            d = torch.load(p, map_location="cpu", weights_only=False)
            return (d["op_raw"].float().numpy(),
                    d["innate"].float().numpy(), tag, horizon)
    return None, None, f"ABSENT {base}", 0


def load_cell(kind, beta, gamma, lam, g, roots, frozen_dir):
    gam = 1.0 if gamma is None else gamma
    if kind == "gpu":
        return load_gpu_cell(beta, gamma, lam, g, roots)
    if kind == "frozen":
        p = _frozen_path(beta, gam, g.F3_SWEEPS, frozen_dir)
        if p is None:
            return None, None, (f"ABSENT frozen replay "
                                f"k{_num(gam)}_w{_num(beta)}"), 0
        d = torch.load(p, map_location="cpu", weights_only=False)
        op = d["op_raw"].float().numpy()
        return (op, d["innate"].float().numpy(), p.name,
                min(op.shape[0], BASE_ROUNDS))
    # twin: beta = 0 is the matched no-platform control, read off ANY
    # cell at this gamma -- the served vector cannot reach it, so every
    # lambda shares one twin
    for (b2, g2, l2, k2) in g.f3_cells():
        if k2 != "gpu" or (1.0 if g2 is None else g2) != gam:
            continue
        tag = g.F3_REUSED.get((b2, g2, l2)) or g.f3_tag(b2, g2, l2)
        p = _find(tag, roots)
        if p is not None:
            d = torch.load(p, map_location="cpu", weights_only=False)
            tw = d["twin_raw"].float().numpy()
            return (tw, d["innate"].float().numpy(),
                    f"{tag}:twin_raw", min(tw.shape[0], BASE_ROUNDS))
    return None, None, f"ABSENT twin at gamma={gam:g}", 0


def main():
    ap = argparse.ArgumentParser(
        description="post-peer outcome, final-window drift and convergence "
                    "classification for every cell of the redesigned "
                    "Figure 3")
    ap.add_argument("--run-root", action="append", default=None)
    ap.add_argument("--frozen-dir", default=str(DEFAULT_FROZEN_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--gate-json", default=None,
                    help="check_fig3_full_loop.py --json verdict; required "
                         "unless --allow-ungated")
    ap.add_argument("--allow-ungated", action="store_true")
    ap.add_argument("--paper", action="store_true",
                    help="HARD-FAIL (exit 4) while any cell still needs a "
                         "horizon extension -- the paper plot must never "
                         "print an unearned equilibrium")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--drift-tol", type=float, default=DRIFT_TOL)
    args = ap.parse_args()

    if args.window < 4:
        ap.error("--window under 4 rounds makes 'split in half and compare' "
                 "arithmetic rather than evidence")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roots = [Path(r) for r in (args.run_root or DEFAULT_RUN_ROOTS)]
    g = _load_gen()

    # ---- gate ---------------------------------------------------------
    gated = False
    if args.gate_json:
        v = json.loads(Path(args.gate_json).read_text())
        if not v.get("ok"):
            print(f"[analyze_f3] REFUSING: the gate verdict at "
                  f"{args.gate_json} reports {v.get('n_failing')} failing "
                  f"cell(s). Fix the grid before analysing it.",
                  file=sys.stderr)
            return 1
        gated = True
    elif not args.allow_ungated:
        print("[analyze_f3] REFUSING: no --gate-json. Run "
              "check_fig3_full_loop.py first, or pass --allow-ungated and "
              "accept that the output is stamped UNGATED.", file=sys.stderr)
        return 2

    rows, per_round, missing, extensions = [], [], [], []
    innate_mean = None
    for (beta, gamma, lam, kind) in g.f3_cells():
        op, innate, src, horizon = load_cell(kind, beta, gamma, lam, g,
                                             roots, args.frozen_dir)
        if op is None:
            missing.append((beta, gamma, lam, kind, src))
            continue
        if op.shape[0] < horizon or horizon == 0:
            missing.append((beta, gamma, lam, kind,
                            f"only {op.shape[0]} rounds"))
            continue
        if innate_mean is None:
            innate_mean = float(np.asarray(innate).mean())
        mean, sd, drift, outcome = _classify(op, horizon, args.window,
                                             args.drift_tol)
        lam_s = ("inf" if (lam is not None and math.isinf(lam))
                 else ("-" if lam is None else f"{lam:g}"))
        gam_s = "dedup" if gamma is None else f"{gamma:g}"
        rows.append({
            "kind": kind,
            "beta_w_plat": f"{beta:g}",
            "gamma_innate_lambda": gam_s,
            "lambda_kl_beta": lam_s,
            "source": src,
            "horizon_rounds": horizon,
            "mean_postpeer_final": f"{mean:.6f}",
            "sd_postpeer_final": f"{sd:.6f}",
            "final_window_drift": f"{drift:.6f}",
            "outcome": outcome,
            "converged": "yes" if outcome == "equilibrium" else "NO",
            "window_rounds": f"{horizon - args.window + 1}-{horizon}",
            "drift_tol": f"{args.drift_tol:g}",
        })
        if outcome.startswith("extend_to_") and kind == "gpu":
            extensions.append({
                "beta": beta, "gamma": gamma, "lam": lam,
                "rounds": int(outcome.rsplit("_", 1)[1]),
                "reason": (f"drift {drift:+.4f} > {args.drift_tol:g} over "
                           f"rounds {horizon - args.window + 1}-{horizon} "
                           f"of {src}")})
        m = op[:horizon].mean(axis=1)
        s = op[:horizon].std(axis=1)
        for t in range(horizon):
            per_round.append({
                "kind": kind, "beta_w_plat": f"{beta:g}",
                "gamma_innate_lambda": gam_s,
                "lambda_kl_beta": lam_s, "round": t + 1,
                "mean_postpeer": f"{m[t]:.6f}", "sd_postpeer": f"{s[t]:.6f}"})

    if missing:
        print(f"[analyze_f3] REFUSING: {len(missing)} required cell(s) are "
              f"absent or short. Figure 3 cannot be drawn from a partial "
              f"grid.", file=sys.stderr)
        for m in missing:
            print(f"    MISSING beta={m[0]} gamma={m[1]} lambda={m[2]} "
                  f"kind={m[3]}: {m[4]}", file=sys.stderr)
        return 3

    cells_csv = out / "fig3_cells.csv"
    with cells_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pr_csv = out / "fig3_per_round.csv"
    with pr_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_round[0].keys()))
        w.writeheader()
        w.writerows(per_round)

    if extensions:
        req = {"_what": ("Targeted horizon extensions for "
                         "fig3_full_loop_ext -- copy to experiments/condor/"
                         "fig3_extension_request.json and commit; the "
                         "generator validates every entry against the grid."),
               "_convention": ("beta = W_PLAT, gamma = INNATE_LAMBDA (null "
                               "where beta=1 dedups it), lam = the "
                               "forward-KL coefficient; rounds in "
                               "{60, 100}."),
               "cells": extensions}
        (out / "fig3_extension_request.json").write_text(
            json.dumps(req, indent=2))

    n_ext = sum(1 for r in rows if r["outcome"].startswith("extend_to_"))
    n_long = sum(1 for r in rows if r["outcome"].endswith("_long_run"))
    summary = {
        "gated": gated, "n_cells": len(rows),
        "n_equilibrium": sum(1 for r in rows
                             if r["outcome"] == "equilibrium"),
        "n_extend": n_ext, "n_long_run": n_long,
        "innate_mean": innate_mean,
        "window": args.window, "drift_tol": args.drift_tol,
        "postpeer": True,
        "cells_csv": str(cells_csv), "per_round_csv": str(pr_csv),
    }
    (out / "fig3_summary.json").write_text(json.dumps(summary, indent=2))

    stamp = "" if gated else "  [UNGATED -- NOT FOR THE PAPER]"
    print("=" * 108)
    print(f"REDESIGNED FIGURE 3 -- {len(rows)} cells, POST-PEER, window "
          f"last {args.window} of each cell's horizon, tol "
          f"{args.drift_tol:g}{stamp}")
    print("=" * 108)
    print(f"{'beta':>6}{'gamma':>8}{'lambda':>8}{'kind':>8}{'hor':>5}"
          f"{'final mean':>12}{'final sd':>10}{'drift':>10}  outcome")
    print("-" * 108)
    for r in sorted(rows, key=lambda r: (r["beta_w_plat"],
                                         r["gamma_innate_lambda"],
                                         r["lambda_kl_beta"])):
        print(f"{r['beta_w_plat']:>6}{r['gamma_innate_lambda']:>8}"
              f"{r['lambda_kl_beta']:>8}{r['kind']:>8}"
              f"{r['horizon_rounds']:>5}{r['mean_postpeer_final']:>12}"
              f"{r['sd_postpeer_final']:>10}{r['final_window_drift']:>10}"
              f"  {r['outcome']}")
    print("=" * 108)
    print(f"[analyze_f3] {summary['n_equilibrium']} equilibria, {n_ext} "
          f"needing extension, {n_long} long-run/cyclic.")
    print(f"[analyze_f3] wrote {cells_csv}")
    print(f"[analyze_f3] wrote {pr_csv}")
    if extensions:
        print(f"[analyze_f3] wrote {out / 'fig3_extension_request.json'} -- "
              f"copy to experiments/condor/fig3_extension_request.json, "
              f"commit, and submit fig3_full_loop_ext.")
    if n_long:
        print(f"[analyze_f3] NOTE {n_long} cell(s) are long-run/cyclic at "
              f">= {MAX_HORIZON} rounds: report them as long-run outcomes, "
              f"NEVER as fixed equilibria.")
    if args.paper and n_ext:
        print(f"[analyze_f3] PAPER GATE FAIL: {n_ext} cell(s) have not "
              f"finished converging and still need a horizon extension. "
              f"The paper figure must not print these values.",
              file=sys.stderr)
        for r in rows:
            if r["outcome"].startswith("extend_to_"):
                print(f"    EXTEND beta={r['beta_w_plat']} "
                      f"gamma={r['gamma_innate_lambda']} "
                      f"lambda={r['lambda_kl_beta']} -> "
                      f"{r['outcome']}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
