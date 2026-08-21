#!/usr/bin/env python3
"""Generate the PERFECT-PREDICTION k-SWEEP (2026-08-21). CPU only -- no
GPU, no LLM, no Condor.

THE BASELINE THIS PROVIDES. Under perfect prediction m(t) = x(t) the
pre-peer update collapses to Friedkin-Johnsen,

    z = (1 - beta_eff) x_innate + beta_eff x,   beta_eff = 1 - (1 - W) k,

so k controls how much of the PREVIOUS POPULATION STATE carries directly
into the next round:

    k = 1   Jiduan's stateless form -- the human component is pure
            innate, z = (1-W) x_innate + W m, and nothing of x survives
            except through the platform
    k < 1   direct carryover: a (1-k) share of the previous state enters
            the human component before the platform mixes in
    k = 0   no innate pull at all in the human component (h = x), so at
            W = .5 beta_eff = 1 and the pre-peer map is the identity

Note the direction, because it is easy to get backwards: at fixed W = .5,
LOWERING k RAISES beta_eff (k=1 -> .5, k=0 -> 1). More carryover means a
less anchored map, not a more anchored one.

GRID. k in {0, .2, .4, .6, .8, 1} x seeds {0, 42, 43} x W in {.5, 1},
300 rounds, both gates genuinely all_open, MovieLens Action, same
population / graph / peer operator as the paper.

  W = .5   the sweep proper -- beta_eff runs 1.0 down to 0.5
  W = 1    the STRUCTURAL CONTROL -- beta_eff = 1 for every k, because
           agents fully adopt the served vector and k cancels out of the
           algebra entirely. Every trajectory there must be identical
           across k at a matched peer seed, which check_pp_k_sweep.py
           enforces bit-for-bit.

OPEN GATES ARE MODES, NEVER THE NUMBER 1. Both gates are strict
inequalities, so eps = 1 still rejects a pair at distance exactly 1.
eps_social is set to a positive value that is INERT under all_open,
because 0 is how the no-peer condition is spelled everywhere else.

REUSE. Artifacts are addressed by a canonical filename encoding every
dial, so a cell that already exists -- in this sweep's directory or in
the earlier perfect_prediction/ namespace -- is reused rather than
regenerated. sim_perfect_predictor.py additionally refuses to overwrite
a completed artifact without --force, so this script is idempotent.

Usage:
  python gen_pp_k_sweep.py [--rounds 300] [--dry-run]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
REPO = HERE.parent.parent.parent

_spec = importlib.util.spec_from_file_location(
    "sim_pp", str(HERE / "sim_perfect_predictor.py"))
PP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PP)

KS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [0, 42, 43]
WS = [0.5, 1.0]
ROUNDS = 300
EPS_SOCIAL = 0.2      # inert under all_open; 0 means "no peers"
EPS_AI = 1.0          # inert under all_open
OUT_DIR = REPO / "notes" / "pofd" / "perfect_prediction_k_sweep"
LEGACY_DIR = REPO / "notes" / "pofd" / "perfect_prediction"
SIM = HERE / "sim_perfect_predictor.py"


def cell_cfg(k, w, seed, rounds):
    return {"innate_k": k, "w_plat": w, "eps_social": EPS_SOCIAL,
            "eps_ai": EPS_AI, "ai_gate_mode": "all_open",
            "peer_gate_mode": "all_open", "ab_sweeps": 1,
            "seed": seed, "rounds": rounds}


def locate(name):
    """The artifact if it already exists in either namespace."""
    for d in (OUT_DIR, LEGACY_DIR):
        p = d / name
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells, reused, made = [], 0, 0
    for w in WS:
        for k in KS:
            for seed in SEEDS:
                cfg = cell_cfg(k, w, seed, args.rounds)
                name = PP.artifact_name(cfg)
                found = locate(name)
                rec = {"innate_k": k, "w_plat": w, "seed": seed,
                       "rounds": args.rounds,
                       "beta_eff": PP.beta_eff(k, w),
                       "role": ("sweep" if w == 0.5
                                else "W=1 structural control"),
                       "artifact": name}
                if found is not None:
                    rec.update({"status": "reused", "path": str(found)})
                    reused += 1
                elif args.dry_run:
                    rec["status"] = "would-generate"
                else:
                    cmd = [sys.executable, str(SIM),
                           "--innate-k", f"{k:g}", "--w-plat", f"{w:g}",
                           "--eps-social", f"{EPS_SOCIAL:g}",
                           "--eps-ai", f"{EPS_AI:g}",
                           "--ai-gate-mode", "all_open",
                           "--peer-gate-mode", "all_open",
                           "--rounds", str(args.rounds),
                           "--seed", str(seed),
                           "--out-dir", str(OUT_DIR), "--quiet"]
                    r = subprocess.run(cmd, capture_output=True, text=True,
                                       env=dict(os.environ, USE_TF="0"))
                    if r.returncode != 0:
                        print(r.stdout + r.stderr, file=sys.stderr)
                        raise SystemExit(
                            f"[ksweep] generation FAILED for k={k} W={w} "
                            f"seed={seed}")
                    rec.update({"status": "generated",
                                "path": str(OUT_DIR / name)})
                    made += 1
                    print(f"[ksweep] k={k:<4g} W={w:<4g} seed={seed:<3} "
                          f"beta_eff={rec['beta_eff']:.2f}  generated")
                cells.append(rec)

    mf = {"ks": KS, "seeds": SEEDS, "ws": WS, "rounds": args.rounds,
          "eps_social": EPS_SOCIAL, "eps_ai": EPS_AI,
          "ai_gate_mode": "all_open", "peer_gate_mode": "all_open",
          "n_cells": len(cells), "n_reused": reused, "n_generated": made,
          "cells": cells}
    if not args.dry_run:
        with open(OUT_DIR / "manifest.json", "w") as fh:
            json.dump(mf, fh, indent=2)
    print(f"[ksweep] {len(cells)} cells: {reused} reused, {made} generated")
    for w in WS:
        print(f"[ksweep]   W={w:g}: beta_eff " +
              ", ".join(f"k={k:g}->{PP.beta_eff(k, w):.2f}" for k in KS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
