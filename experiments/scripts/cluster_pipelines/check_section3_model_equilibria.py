#!/usr/bin/env python3
"""Gate the matched open-gate, cross-model Section 3 equilibrium wave.

The required surface is read from ``gen_pofd_sweep.py``: six models x
three seeds, beta=1, gamma=1, forward-KL lambda=2, Deffuant alpha=.5,
100 complete peer sweeps, both gates open, and 30 recursive rounds.
Every production cell is fresh under one provenance; no lambda=0 cell is
part of this figure.  Perfect prediction is the shared innate-mean reference
and therefore needs no model job.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import torch

from check_fig3_full_loop import (
    _load_gen,
    _resolve,
    _sha_t,
    check_arrays,
    check_full_loop,
    check_parse,
)

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
DEFAULT_RUN_ROOTS = (
    Path("/home/gsmithline/perfsim/runs/pokec_gated_lm"),
    REPO / "runs" / "pokec_gated_lm",
    REPO / "notes" / "pofd" / "cluster",
)
N_AGENTS = 723
TOL = 1e-9


def _eq(cfg, key, want, errs):
    got = cfg.get(key, "<absent>")
    if isinstance(want, float):
        ok = isinstance(got, (int, float)) and abs(float(got) - want) <= TOL
    else:
        ok = got == want
    if not ok:
        errs.append(f"CONFIG {key}={got!r} (want {want!r})")


def check_cell(tag, model, seed, rounds, g, roots):
    errs, notes = [], []
    path = _resolve(tag, roots)
    if path is None:
        return {"tag": tag, "model": model, "seed": seed,
                "status": "ABSENT", "errors": ["trajectory.pt absent"],
                "notes": []}

    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = d.get("config", {}) or {}
    expected = {
        "w_plat": g.S3M_BETA,
        "innate_lambda": g.S3M_GAMMA,
        "kl_beta": g.S3M_LAMBDA,
        "deffuant_alpha": g.S3M_ALPHA,
        "ab_sweeps": g.S3M_SWEEPS,
        "n_rounds": rounds,
        "seed": seed,
        "dataset": "movielens",
        "ml_target": "Action",
        "base_model": g.FAM_MODELS[model]["base_model"],
        "training_style": "sft_kl",
        "kl_direction": "forward",
        "kl_ref_adapter": "",
        "ai_gate_mode": "all_open",
        "peer_gate_mode": "all_open",
        "ai_gate_reference": "anchor",
        "population_update": "nested_ai_anchored_then_social_v2",
        "pop_model": "ab",
        "icl_k": 0,
        "train_cap": N_AGENTS,
        "n_labeled": N_AGENTS,
        "seed_base_data": True,
        "lora_r": 512,
        "sft_epochs": 1,
        "sft_batch_size": 4,
        "sft_lr": 5e-5,
        "use_lora": True,
        "fresh_each_round": True,
        "save_raw_gen": True,
        "serve_eval_mode": True,
        "do_sample": False,
    }
    for key, want in expected.items():
        _eq(cfg, key, want, errs)
    if not isinstance(cfg.get("git_sha"), str) or not cfg["git_sha"].strip():
        errs.append("CONFIG git_sha absent or empty")
    hg = cfg.get("homophily_gamma", cfg.get("gamma_bias", 0.0))
    if hg not in (0, 0.0):
        errs.append(f"CONFIG homophily gamma={hg!r} (want 0.0)")
    if model == "qwen3_8b" and cfg.get("chat_thinking") is not False:
        errs.append(f"CONFIG Qwen3 chat_thinking={cfg.get('chat_thinking')!r} "
                    "(want False)")

    check_arrays(d, rounds, errs)
    if all(not e.startswith("ARTIFACT") for e in errs):
        check_full_loop(d, path.parent, g.S3M_LAMBDA, rounds, errs, notes)
        check_parse(path.parent, torch.as_tensor(d["pred_raw"]).float(),
                    rounds, errs)

    op = torch.as_tensor(d.get("op_raw", torch.empty(0))).float()
    rec = {
        "tag": tag,
        "model": model,
        "seed": seed,
        "status": "PASS" if not errs else "FAIL",
        "errors": errs,
        "notes": notes,
        "git_sha": cfg.get("git_sha"),
        "innate_sha": (_sha_t(torch.as_tensor(d["innate"]).float())
                       if "innate" in d else None),
        "twin_sha": (_sha_t(torch.as_tensor(d["twin_raw"]).float()[:rounds])
                     if "twin_raw" in d else None),
    }
    if op.ndim == 2 and op.shape[0] >= rounds:
        rec["mean"] = float(op[rounds - 1].mean())
        rec["sd"] = float(op[rounds - 1].std())
    return rec


def main():
    ap = argparse.ArgumentParser(
        description="gate the Section 3 open-gate cross-model equilibrium wave")
    ap.add_argument("--run-root", action="append", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    roots = [Path(r) for r in (args.run_root or DEFAULT_RUN_ROOTS)]
    g = _load_gen()
    if args.smoke:
        cells = [(g.S3M_SMOKE_MODEL, g.S3M_SMOKE_SEED,
                  g.s3m_tag(g.S3M_SMOKE_MODEL, g.S3M_SMOKE_SEED,
                            rounds=g.S3M_SMOKE_ROUNDS, smoke=True),
                  g.S3M_SMOKE_ROUNDS)]
    else:
        cells = [(model, seed, g.s3m_tag(model, seed), g.S3M_ROUNDS)
                 for model in g.S3M_MODELS for seed in g.S3M_SEEDS]

    records = [check_cell(tag, model, seed, rounds, g, roots)
               for model, seed, tag, rounds in cells]

    # All production cells must share the same population and code version.
    if not args.smoke:
        innate = {r.get("innate_sha") for r in records if r.get("innate_sha")}
        if len(innate) != 1:
            records.append({"tag": "<cross-cell innate>", "model": "-",
                            "seed": "-", "status": "FAIL", "notes": [],
                            "errors": [f"{len(innate)} distinct innate hashes; "
                                       "all model cells must share one population"]})
        shas = {r.get("git_sha") for r in records if r.get("git_sha")}
        if len(shas) != 1:
            records.append({"tag": "<cross-cell provenance>", "model": "-",
                            "seed": "-", "status": "FAIL", "notes": [],
                            "errors": [f"{len(shas)} distinct git SHAs; this "
                                       "wave requires one common provenance"]})
        for seed in g.S3M_SEEDS:
            twins = {r.get("twin_sha") for r in records
                     if r.get("seed") == seed and r.get("twin_sha")}
            if len(twins) != 1:
                records.append({"tag": f"<seed {seed} twins>", "model": "-",
                                "seed": seed, "status": "FAIL", "notes": [],
                                "errors": [f"{len(twins)} distinct no-platform "
                                           "twins across models"]})

    ok = all(r["status"] == "PASS" for r in records)
    print("=" * 112)
    print(f"SECTION-3 MODEL EQUILIBRIA GATE -- {len(records)} record(s)")
    print("=" * 112)
    print(f"{'model':<14}{'seed':>6}  {'verdict':<9}{'mean':>10}{'sd':>10}  cell")
    for r in records:
        mean = f"{r['mean']:.4f}" if "mean" in r else "-"
        sd = f"{r['sd']:.4f}" if "sd" in r else "-"
        print(f"{r.get('model','-'):<14}{str(r.get('seed','-')):>6}  "
              f"{r['status']:<9}{mean:>10}{sd:>10}  {r['tag']}")
        for e in r["errors"]:
            print(f"    !! {e}")
        for n in r.get("notes", []):
            print(f"    NOTE {n}")

    n_bad = sum(r["status"] != "PASS" for r in records)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ok": ok, "n_cells": len(cells),
                                   "n_failing": n_bad, "cells": records},
                                  indent=2, default=str))
        print(f"[check_s3m] verdict -> {out}")
    if ok:
        print("[check_s3m] PASS -- every model-seed cell matches the fixed "
              "surface, retrained in every round with a live KL anchor, "
              "served without parse failures, and shares one population "
              "and code provenance.")
        return 0
    print(f"[check_s3m] FAIL -- {n_bad} failing record(s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
