#!/usr/bin/env python3
"""HARD-GATED checker for the perfect-prediction k-sweep (2026-08-21).

Per artifact (delegated to check_perfect_predictor.check_artifact, which
re-simulates from the artifact's own config and compares byte-for-byte):
  * op_raw / twin_raw / pred_raw reproduce exactly
  * served == the START-OF-ROUND state, i.e. prediction really is perfect
  * environment hashes match, values finite and inside [0, 1]
  * the declared horizon equals the stored one

Sweep-level, added here:
  * BOTH gates genuinely open -- as MODES, not as the numeric value 1.
    Both tests are strict inequalities, so eps = 1 would still reject a
    pair at distance exactly 1; a numeric gate wearing an "open" label is
    the failure this checks for.
  * MEAN PRESERVATION. Starting from x(0) = innate, the pre-peer map is
    z = (1-b) innate + b x, whose mean is (1-b)m0 + b*m0 = m0 for ANY b,
    and the Deffuant peer step sends pairs to their midpoint, which
    conserves the mean exactly. So the population mean must equal the
    innate mean in EVERY round, at every k. This is an exact identity,
    not an approximation, so the tolerance is numerical only.
  * MATCHED POPULATION AND GRAPH across every cell -- one innate vector,
    one adjacency, bit-identical. A sweep built on two environments would
    make every cross-k comparison meaningless while looking well-formed.
  * CORRECT HORIZONS -- all cells the declared number of rounds.
  * W = 1 INVARIANCE ACROSS k. At W = 1 the pre-peer map is z = m = x for
    every k: the (1-W) factor kills the human component entirely, so k
    cancels out of the algebra. With a matched peer seed the trajectories
    must therefore be BIT-IDENTICAL across all k. This is the sweep's
    structural control -- if it fails, k is leaking somewhere it should
    not, and nothing in the W=.5 results can be trusted.

Usage:
  python check_pp_k_sweep.py [--dir DIR] [--legacy-dir DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
REPO = HERE.parent.parent.parent

_s1 = importlib.util.spec_from_file_location(
    "sim_pp", str(HERE / "sim_perfect_predictor.py"))
PP = importlib.util.module_from_spec(_s1)
_s1.loader.exec_module(PP)
_s2 = importlib.util.spec_from_file_location(
    "chk_pp", str(HERE / "check_perfect_predictor.py"))
CPP = importlib.util.module_from_spec(_s2)
_s2.loader.exec_module(CPP)

OUT_DIR = REPO / "notes" / "pofd" / "perfect_prediction_k_sweep"
LEGACY_DIR = REPO / "notes" / "pofd" / "perfect_prediction"
MEAN_TOL = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=OUT_DIR)
    ap.add_argument("--legacy-dir", type=Path, default=LEGACY_DIR)
    args = ap.parse_args()

    mf = json.load(open(args.dir / "manifest.json"))
    setup = PP.extract_loader()(
        REPO / "experiments/data/movielens/ml-100k", "Action")
    innate_ref = setup["innate"]
    m0 = float(innate_ref.mean())

    errs, loaded = [], {}
    for c in mf["cells"]:
        p = Path(c["path"])
        if not p.exists():
            errs.append(f"{c['artifact']}: missing at {p}")
            continue
        # full byte-for-byte replay from the artifact's own config
        errs += CPP.check_artifact(p, setup)
        d = torch.load(p, map_location="cpu", weights_only=False)
        cfg, op = d["config"], d["op_raw"]
        loaded[(c["w_plat"], c["innate_k"], c["seed"])] = (op, d)

        # -- gates genuinely open, as MODES --------------------------------
        for key in ("ai_gate_mode", "peer_gate_mode"):
            if cfg.get(key) != "all_open":
                errs.append(f"{c['artifact']}: {key}={cfg.get(key)!r} -- "
                            f"the sweep requires genuinely open gates, and "
                            f"a numeric threshold of 1 is NOT open (strict "
                            f"inequality rejects distance exactly 1)")
        if float(cfg.get("eps_social", 0.0)) <= 0:
            errs.append(f"{c['artifact']}: eps_social=0 is the NO-PEER "
                        f"condition and must not double as an open channel")

        # -- horizon --------------------------------------------------------
        if op.shape[0] != mf["rounds"]:
            errs.append(f"{c['artifact']}: {op.shape[0]} rounds != "
                        f"{mf['rounds']}")

        # -- matched population and graph -----------------------------------
        if not torch.equal(d["innate"].float(), innate_ref.float()):
            errs.append(f"{c['artifact']}: innate vector differs from the "
                        f"loader's -- the sweep is not on one environment")
        if cfg.get("adj_sha256") != CPP._sha((setup["adj"] > 0)
                                             .to(torch.uint8)):
            errs.append(f"{c['artifact']}: adjacency hash differs")

        # -- mean preservation (exact identity, all k) -----------------------
        drift = (op.mean(dim=1) - m0).abs()
        worst = float(drift.max())
        if worst > MEAN_TOL:
            errs.append(f"{c['artifact']}: population mean drifts from the "
                        f"innate mean by up to {worst:.3e} (round "
                        f"{int(drift.argmax())}) -- under perfect "
                        f"prediction with midpoint peers the mean is "
                        f"conserved exactly for every k")

        # -- beta_eff bookkeeping -------------------------------------------
        want_b = PP.beta_eff(c["innate_k"], c["w_plat"])
        if abs(float(cfg.get("beta_eff", -9)) - want_b) > 1e-12:
            errs.append(f"{c['artifact']}: beta_eff {cfg.get('beta_eff')} "
                        f"!= 1-(1-W)k = {want_b}")

    # -- W = 1 INVARIANCE ACROSS k -----------------------------------------
    # the sweep's structural control: at W=1 the pre-peer map is z = x for
    # every k, so a matched peer seed must give bit-identical trajectories
    by_seed = defaultdict(dict)
    for (w, k, seed), (op, _) in loaded.items():
        if w == 1.0:
            by_seed[seed][k] = op
    n_checked = 0
    for seed, byk in sorted(by_seed.items()):
        ks = sorted(byk)
        if len(ks) < 2:
            continue
        ref_k = ks[0]
        for k in ks[1:]:
            n_checked += 1
            if not torch.equal(byk[k], byk[ref_k]):
                gap = float((byk[k] - byk[ref_k]).abs().max())
                errs.append(
                    f"W=1 INVARIANCE seed {seed}: k={k:g} differs from "
                    f"k={ref_k:g} by up to {gap:.3e} -- at W=1 the human "
                    f"component is annihilated by (1-W)=0, so k cancels "
                    f"and the trajectories must be bit-identical")
    print(f"[check_ksweep] W=1 invariance: {n_checked} cross-k comparison(s)")

    # a negative control on the control: at W=.5 the SAME comparison must
    # FAIL, or the invariance check above is vacuous
    w5 = defaultdict(dict)
    for (w, k, seed), (op, _) in loaded.items():
        if w == 0.5:
            w5[seed][k] = op
    for seed, byk in sorted(w5.items()):
        ks = sorted(byk)
        if len(ks) >= 2 and torch.equal(byk[ks[0]], byk[ks[-1]]):
            errs.append(
                f"W=.5 seed {seed}: k={ks[0]:g} and k={ks[-1]:g} produced "
                f"IDENTICAL trajectories -- beta_eff should differ "
                f"({PP.beta_eff(ks[0], 0.5)} vs {PP.beta_eff(ks[-1], 0.5)}),"
                f" so k is not reaching the operator at all")
        break

    if errs:
        print(f"[check_ksweep] {len(errs)} FAILURE(S)", file=sys.stderr)
        for e in errs[:40]:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"[check_ksweep] PASS -- {len(loaded)} artifacts replayed "
          f"byte-for-byte, gates open, mean preserved, one environment, "
          f"W=1 invariant across k")
    return 0


if __name__ == "__main__":
    sys.exit(main())
