#!/usr/bin/env python3
"""Analyze the three-seed open-gate cross-model equilibrium comparison.

Each model-seed trajectory contributes one estimate: the mean post-peer
population opinion averaged over its final ten rounds.  Models are then
aggregated across seeds; agents and rounds are never treated as independent
replicates.  Paper mode refuses missing, ungated, drifting, or non-consensus
cells.
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
    values, counts = np.unique(np.round(pred[-1], 8), return_counts=True)
    return {
        "equilibrium_mean": float(tail.mean()),
        "final_mean": float(means[-1]),
        "final_sd": float(op[-1].std()),
        "drift": drift,
        "served_distinct": int(len(values)),
        "served_max_mode_share": float(counts.max() / counts.sum()),
    }


def main():
    ap = argparse.ArgumentParser(
        description="analyze the Section 3 model-specific equilibrium wave")
    ap.add_argument("--run-root", action="append", default=None)
    ap.add_argument("--gate-json", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--drift-tol", type=float, default=DRIFT_TOL)
    ap.add_argument("--consensus-sd-tol", type=float,
                    default=CONSENSUS_SD_TOL)
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()

    verdict = json.loads(Path(args.gate_json).read_text())
    if not verdict.get("ok"):
        print("[analyze_s3m] REFUSING: gate is not PASS", file=sys.stderr)
        return 1
    if args.window < 4 or args.window % 2:
        ap.error("--window must be an even integer >= 4")

    roots = [Path(r) for r in (args.run_root or DEFAULT_RUN_ROOTS)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = _load_gen()

    rows, missing, innate_means = [], [], []
    for model in g.S3M_MODELS:
        for seed in g.S3M_SEEDS:
            tag = g.s3m_tag(model, seed)
            path = _find(tag, roots)
            if path is None:
                missing.append(tag)
                continue
            d = torch.load(path, map_location="cpu", weights_only=False)
            op = torch.as_tensor(d["op_raw"]).float().numpy()
            pred = torch.as_tensor(d["pred_raw"]).float().numpy()
            if op.shape[0] < g.S3M_ROUNDS or pred.shape[0] < g.S3M_ROUNDS:
                missing.append(f"{tag} (short horizon)")
                continue
            stats = _cell_stats(op[:g.S3M_ROUNDS], pred[:g.S3M_ROUNDS],
                                args.window)
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
                "converged": abs(stats["drift"]) <= args.drift_tol,
                "consensus": stats["final_sd"] <= args.consensus_sd_tol,
                "served_distinct": stats["served_distinct"],
                "served_max_mode_share":
                    f"{stats['served_max_mode_share']:.8f}",
            })

    expected = len(g.S3M_MODELS) * len(g.S3M_SEEDS)
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
    for model in g.S3M_MODELS:
        selected = [r for r in rows if r["model"] == model]
        vals = np.asarray([float(r["equilibrium_mean"]) for r in selected])
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1))
        sem = sd / math.sqrt(len(vals))
        half = T_CRIT_DF2_95 * sem
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

    unsettled = [r for r in rows if not r["converged"]]
    nonconsensus = [r for r in rows if not r["consensus"]]
    summary = {
        "gated": True,
        "n_cells": len(rows),
        "n_models": len(model_rows),
        "seeds": list(g.S3M_SEEDS),
        "perfect_prediction_mean": perfect_mean,
        "postpeer": True,
        "window": args.window,
        "drift_tol": args.drift_tol,
        "consensus_sd_tol": args.consensus_sd_tol,
        "n_unsettled": len(unsettled),
        "n_nonconsensus": len(nonconsensus),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 106)
    print("SECTION-3 MODEL-SPECIFIC OPEN-GATE EQUILIBRIA")
    print("=" * 106)
    print(f"Perfect prediction / innate mean: {perfect_mean:.6f}")
    print(f"{'model':<14}{'mean':>10}{'95% CI':>25}{'shift':>11}  status")
    for r in model_rows:
        status = ("equilibrium/consensus" if r["all_converged"] and
                  r["all_consensus"] else "CHECK")
        ci = f"[{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}]"
        print(f"{r['model_label']:<14}{float(r['equilibrium_mean']):>10.4f}"
              f"{ci:>25}{float(r['shift_from_perfect']):>+11.4f}  {status}")
    print(f"[analyze_s3m] wrote {cells_csv}")
    print(f"[analyze_s3m] wrote {models_csv}")

    if args.paper and (unsettled or nonconsensus):
        print(f"[analyze_s3m] PAPER GATE FAIL: {len(unsettled)} drifting and "
              f"{len(nonconsensus)} non-consensus cell(s)", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
