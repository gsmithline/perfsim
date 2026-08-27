#!/usr/bin/env python3
"""Analyze the three-seed open-gate cross-model equilibrium comparison.

Each model-seed trajectory contributes one estimate: the mean post-peer
population opinion averaged over its final ten rounds (rounds 21-30 of 30).
Models are then aggregated across seeds with a three-seed 95% t interval
(df = 2); agents and rounds are never treated as independent replicates.
Paper mode refuses missing, ungated, drifting, cyclic, or non-consensus
cells.

Why the innate mean is the perfect-prediction reference here: with
W_PLAT = 1 and an all-open AI gate every agent lands exactly on the served
value (z = m), and alpha = 0.5 all-open Deffuant sweeps preserve the
population mean pair by pair, so mean(x_{t+1}) = mean(m_t).  A predictor
that reproduces its label vector (innate in round 0, the post-peer
population afterwards) therefore keeps mean(x_t) = mean(innate) for every
t while the sweeps drive consensus.  Labels are formatted .2f, so even a
perfect predictor is quantised to the 0.01 grid (a shift of at most 0.005
per agent per round); differences below 0.01 are not interpretable.

The gate verdict must be THIS wave's full 18-cell verdict: the smoke
verdict or a stale PASS from another wave is refused.
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
DEFAULT_OUT = REPO / "notes" / "pofd" / "section3_model_equilibria"
WINDOW = 10
DRIFT_TOL = 0.005
CONSENSUS_SD_TOL = 0.005
T_CRIT_DF2_95 = 4.302652729911275

DISPLAY = {
    "qwen7b": "Qwen 2.5",
    "qwen3_8b": "Qwen 3",
    "olmo7b": "OLMo 2",
    "olmo3_7b": "OLMo 3",
    "mistral7b": "Mistral",
    "ministral8b": "Ministral",
}


def _load_gen():
    spec = importlib.util.spec_from_file_location("_gen_s3m_analysis", str(GEN))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gen_s3m_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


def _find(tag, roots):
    for root in roots:
        path = Path(root) / tag / "trajectory.pt"
        if path.exists():
            return path
    return None


def _cell_stats(op, pred, window):
    op = np.asarray(op, dtype=float)
    pred = np.asarray(pred, dtype=float)
    means = op.mean(axis=1)
    tail = means[-window:]
    half = window // 2
    drift = float(tail[-half:].mean() - tail[:half].mean())
    # Half-split drift alone lets a short cycle through (a period-2
    # series a,b,a,b,... has drift (b-a)/5 over ten rounds), so the
    # settled verdict also bounds the RANGE of the late means and the
    # analyzer flags sign-alternating consecutive differences.
    late = tail[-half:]
    diffs = np.diff(tail)
    signs = np.sign(diffs[np.abs(diffs) > 0])
    alternating = (float(np.mean(signs[1:] * signs[:-1] < 0))
                   if len(signs) >= 2 else 0.0)
    values, counts = np.unique(np.round(pred[-1], 8), return_counts=True)
    return {
        "equilibrium_mean": float(tail.mean()),
        "final_mean": float(means[-1]),
        "final_sd": float(op[-1].std()),
        "drift": drift,
        "late_range": float(late.max() - late.min()),
        "window_range": float(tail.max() - tail.min()),
        "alternating_frac": alternating,
        "served_distinct": int(len(values)),
        "served_max_mode_share": float(counts.max() / counts.sum()),
    }


def settled(stats, drift_tol):
    """Settled iff |half-split drift| <= tol AND the last half-window's
    round means span <= 2*tol.  A 2-cycle of amplitude > 2*tol fails the
    range test even when its drift is small."""
    return (abs(stats["drift"]) <= drift_tol
            and stats["late_range"] <= 2.0 * drift_tol)


def cyclic(stats, drift_tol):
    """Diagnostic: >= 70% of the non-zero consecutive differences of the
    window means alternate in sign while the window is not flat."""
    return (stats["alternating_frac"] >= 0.7
            and stats["window_range"] > drift_tol)


def tci3(vals):
    """Three-seed mean and 95% t half-width (df = 2; the ONLY replicate
    unit is the seed).  Refuses any other n so the df=2 critical value can
    never be applied to a different sample size."""
    vals = np.asarray(vals, dtype=float)
    if vals.shape != (3,):
        raise ValueError(f"tci3 needs exactly 3 seed values, got "
                         f"{vals.shape}")
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1))
    half = T_CRIT_DF2_95 * sd / math.sqrt(3.0)
    return mean, sd, half


class WaveCfg:
    """Which Section-3 wave to analyse. The SFT wave (reference-
    regularized, the published Figure 3(a)) and the personal-history ICL
    wave share this entire analysis -- window, drift, consensus and the
    three-seed interval are defined once -- and differ only in which
    tags/constants they resolve."""

    def __init__(self, name, g):
        self.name = name
        if name == "icl":
            self.models, self.seeds = g.S3I_MODELS, g.S3I_SEEDS
            self.rounds = g.S3I_ROUNDS
            self.cell_tag = g.s3i_cell_tag
            self.key = g.S3I_KEY
            self.arm_label = "personal-history ICL"
        else:
            self.models, self.seeds = g.S3M_MODELS, g.S3M_SEEDS
            self.rounds = g.S3M_ROUNDS
            self.cell_tag = g.s3m_cell_tag
            self.key = g.S3M_KEY
            self.arm_label = "reference-regularized SFT"


def gate_binds_wave(verdict, g, w=None):
    """The gate JSON must be this wave's full production verdict: PASS,
    18 cells, and every production (cell) tag present with status PASS.

    w defaults to the SFT wave, so the original two-argument call site
    (and its tests) keeps working unchanged."""
    if w is None:
        w = WaveCfg("sft", g)
    if not verdict.get("ok"):
        return "gate is not PASS"
    want = {w.cell_tag(m, s) for m in w.models for s in w.seeds}
    cells = verdict.get("cells") or []
    got = {c.get("tag"): c.get("status") for c in cells}
    if verdict.get("n_cells") != len(want):
        return (f"gate verdict covers {verdict.get('n_cells')} cell(s), "
                f"not the {len(want)} production cells (smoke or stale?)")
    missing = sorted(t for t in want if got.get(t) != "PASS")
    if missing:
        return f"{len(missing)} production tag(s) not PASS in the gate"
    return None


def main():
    ap = argparse.ArgumentParser(
        description="analyze the Section 3 model-specific equilibrium wave")
    ap.add_argument("--wave", default="sft", choices=("sft", "icl"),
                    help="sft = the reference-regularized wave (the "
                         "published Figure 3(a); DEFAULT); icl = the "
                         "frozen personal-history analogue")
    ap.add_argument("--run-root", action="append", default=None)
    ap.add_argument("--gate-json", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--drift-tol", type=float, default=DRIFT_TOL)
    ap.add_argument("--consensus-sd-tol", type=float,
                    default=CONSENSUS_SD_TOL)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--accept-limit-cycle", action="append", default=[],
                    metavar="MODEL",
                    help="treat this model's bounded oscillation as its "
                         "equilibrium (recorded in summary.json; the "
                         "window mean +- t-CI is still the estimate)")
    args = ap.parse_args()
    for m in args.accept_limit_cycle:
        if m not in DISPLAY:
            ap.error(f"--accept-limit-cycle {m!r}: unknown model")

    gate_path = Path(args.gate_json)
    if not gate_path.exists():
        print(f"[analyze_s3m] REFUSING: gate JSON {gate_path} absent",
              file=sys.stderr)
        return 1
    verdict = json.loads(gate_path.read_text())
    if args.window < 4 or args.window % 2:
        ap.error("--window must be an even integer >= 4")

    roots = [Path(r) for r in (args.run_root or DEFAULT_RUN_ROOTS)]
    g = _load_gen()
    w = WaveCfg(args.wave, g)
    out = Path(args.out_dir) if args.out_dir else (
        DEFAULT_OUT if args.wave == "sft"
        else REPO / "notes" / "pofd" / "section3_model_icl")
    out.mkdir(parents=True, exist_ok=True)
    why = gate_binds_wave(verdict, g, w)
    if why:
        print(f"[analyze_s3m] REFUSING: {why}", file=sys.stderr)
        return 1
    gate_shas = sorted({str(c.get("git_sha")) for c in verdict["cells"]
                        if c.get("git_sha")})

    rows, missing, innate_means = [], [], []
    for model in w.models:
        for seed in w.seeds:
            tag = w.cell_tag(model, seed)
            path = _find(tag, roots)
            if path is None:
                missing.append(tag)
                continue
            d = torch.load(path, map_location="cpu", weights_only=False)
            op = torch.as_tensor(d["op_raw"]).float().numpy()
            pred = torch.as_tensor(d["pred_raw"]).float().numpy()
            if op.shape[0] != w.rounds or pred.shape[0] != w.rounds:
                # the window is defined on the wave's fixed 30-round
                # horizon; a longer artifact would otherwise be silently
                # truncated and a shorter one silently accepted
                missing.append(f"{tag} (horizon {op.shape[0]}/"
                               f"{pred.shape[0]} != {w.rounds})")
                continue
            stats = _cell_stats(op, pred, args.window)
            innate = torch.as_tensor(d["innate"]).float().numpy()
            innate_means.append(float(innate.mean()))
            rows.append({
                "model": model,
                "model_label": DISPLAY[model],
                "seed": seed,
                "tag": tag,
                "equilibrium_mean": f"{stats['equilibrium_mean']:.8f}",
                "final_mean": f"{stats['final_mean']:.8f}",
                "final_postpeer_sd": f"{stats['final_sd']:.8f}",
                "final_window_drift": f"{stats['drift']:.8f}",
                "late_mean_range": f"{stats['late_range']:.8f}",
                "window_mean_range": f"{stats['window_range']:.8f}",
                "alternating_frac": f"{stats['alternating_frac']:.4f}",
                "converged": settled(stats, args.drift_tol),
                "cyclic": cyclic(stats, args.drift_tol),
                "accepted_limit_cycle": model in args.accept_limit_cycle,
                "parse_mode": (d.get("config", {}) or {}).get("parse_mode",
                                                              "legacy"),
                "consensus": stats["final_sd"] <= args.consensus_sd_tol,
                "git_sha": (d.get("config", {}) or {}).get("git_sha"),
                "path": str(path),
                "served_distinct": stats["served_distinct"],
                "served_max_mode_share":
                    f"{stats['served_max_mode_share']:.8f}",
            })

    expected = len(w.models) * len(w.seeds)
    if missing or len(rows) != expected:
        print(f"[analyze_s3m] REFUSING: {len(missing)} required cell(s) "
              "missing or short", file=sys.stderr)
        for tag in missing:
            print(f"    {tag}", file=sys.stderr)
        return 2
    if max(innate_means) - min(innate_means) > 1e-7:
        print("[analyze_s3m] REFUSING: innate means differ across cells",
              file=sys.stderr)
        return 3
    perfect_mean = float(np.mean(innate_means))

    model_rows = []
    for model in w.models:
        selected = [r for r in rows if r["model"] == model]
        vals = [float(r["equilibrium_mean"]) for r in selected]
        mean, sd, half = tci3(vals)
        model_rows.append({
            "model": model,
            "model_label": DISPLAY[model],
            "n_seeds": len(vals),
            "equilibrium_mean": f"{mean:.8f}",
            "seed_sd": f"{sd:.8f}",
            "ci95_low": f"{mean - half:.8f}",
            "ci95_high": f"{mean + half:.8f}",
            "shift_from_perfect": f"{mean - perfect_mean:.8f}",
            "all_converged": all(r["converged"] for r in selected),
            "any_cyclic": any(r["cyclic"] for r in selected),
            "accepted_limit_cycle": model in args.accept_limit_cycle,
            "all_consensus": all(r["consensus"] for r in selected),
        })

    cells_csv = out / "model_equilibrium_cells.csv"
    with cells_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    models_csv = out / "model_equilibria.csv"
    with models_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)

    unsettled = [r for r in rows if not r["converged"]
                 and not r["accepted_limit_cycle"]]
    cyclic_rows = [r for r in rows if r["cyclic"]
                   and not r["accepted_limit_cycle"]]
    nonconsensus = [r for r in rows if not r["consensus"]]
    summary = {
        "gated": True,
        "gate_json": str(gate_path),
        "gate_n_cells": verdict.get("n_cells"),
        "git_sha": gate_shas,
        "provenance": verdict.get("provenance"),
        "accepted_limit_cycle": sorted(set(args.accept_limit_cycle)),
        "accepted_limit_cycle_note": (
            "cells of these models are reported at their window mean +- "
            "3-seed t-CI although they fail the settling test; the user "
            "accepted the bounded oscillation as the equilibrium"
            if args.accept_limit_cycle else None),
        "n_cells": len(rows),
        "n_models": len(model_rows),
        "wave": w.name, "wave_key": w.key, "arm": w.arm_label,
        "seeds": list(w.seeds),
        "perfect_prediction_mean": perfect_mean,
        "perfect_prediction_reference": "mean(innate); see module docstring",
        "postpeer": True,
        "tensor": "op_raw (end-of-round, post-peer)",
        "window": args.window,
        "window_rounds": [w.rounds - args.window + 1, w.rounds],
        "replicate_unit": "seed",
        "ci": "95% t, df=2",
        "t_crit_df2_95": T_CRIT_DF2_95,
        "drift_tol": args.drift_tol,
        "late_range_tol": 2.0 * args.drift_tol,
        "consensus_sd_tol": args.consensus_sd_tol,
        "served_quantization": 0.01,
        "n_unsettled": len(unsettled),
        "n_cyclic": len(cyclic_rows),
        "n_nonconsensus": len(nonconsensus),
        "unsettled_tags": [r["tag"] for r in unsettled],
        "cyclic_tags": [r["tag"] for r in cyclic_rows],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 106)
    print("SECTION-3 MODEL-SPECIFIC OPEN-GATE EQUILIBRIA")
    print("=" * 106)
    print(f"Perfect prediction / innate mean: {perfect_mean:.6f}")
    print(f"{'model':<14}{'mean':>10}{'95% CI':>25}{'shift':>11}  status")
    for r in model_rows:
        status = ("equilibrium/consensus" if r["all_converged"] and
                  r["all_consensus"] and not r["any_cyclic"] else
                  ("limit-cycle (accepted)" if r["accepted_limit_cycle"]
                   and r["all_consensus"] else "CHECK"))
        ci = f"[{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}]"
        print(f"{r['model_label']:<14}{float(r['equilibrium_mean']):>10.4f}"
              f"{ci:>25}{float(r['shift_from_perfect']):>+11.4f}  {status}")
    print(f"[analyze_s3m] wrote {cells_csv}")
    print(f"[analyze_s3m] wrote {models_csv}")

    if args.paper and (unsettled or cyclic_rows or nonconsensus):
        print(f"[analyze_s3m] PAPER GATE FAIL: {len(unsettled)} unsettled, "
              f"{len(cyclic_rows)} cyclic and {len(nonconsensus)} "
              f"non-consensus cell(s)", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
