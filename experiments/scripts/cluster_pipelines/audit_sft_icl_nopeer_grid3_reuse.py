#!/usr/bin/env python3
"""Reuse audit for the three-seed no-peer SFT/ICL gate grid (2026-08-14).

Grid: 3 models x channels {k0, fz0, dyn, b0} x numeric gates
{0.05, 0.1, 0.2, 0.4, 1.0} x seeds {0, 42, 43} = 180 conceptual cells,
every cell AI_GATE_MODE=threshold (_ea1_ is the strict numeric 1.0 gate --
_eaopen_ runs are structurally non-reusable for it). Matching is by config
fields + trajectory completeness (exactly 30 rounds, (30, 723) op/pred),
shared with the reach/k0 audits, never tag similarity.

Cell statuses:
  reused     an exact complete run exists locally
  new        a genuinely informative job to queue
  reference  the k0 seed-42/43 repetitions: k0 is DETERMINISTIC given the
             frozen prediction map (bit-identical across seeds and hosts,
             proven by the pofdreachbase_ probes over seeds 0-45) and the
             peer-free operator draws nothing the trajectory consumes, so
             these cells map to the seed-0 k0 run instead of re-running --
             rerunning would manufacture artificial replication.

Writes experiments/condor/manifest_sft_icl_nopeer_grid3.json. With
--expect-* set, a count mismatch prints a DISCREPANCY report and exits 1
WITHOUT writing.

Usage:
  python3 audit_sft_icl_nopeer_grid3_reuse.py [--roots DIR ...] [--write]
      [--expect-reused N] [--expect-new N]
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "audit_reach", os.path.join(HERE, "audit_sft_icl_reach_reuse.py"))
AR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AR)
_spec_k0 = importlib.util.spec_from_file_location(
    "audit_k0", os.path.join(HERE, "audit_sft_k0_nopeer_reuse.py"))
AK = importlib.util.module_from_spec(_spec_k0)
_spec_k0.loader.exec_module(AK)

REPO = AR.REPO
MANIFEST = os.path.join(REPO, "experiments", "condor",
                        "manifest_sft_icl_nopeer_grid3.json")
ARMS = ["k0", "fz0", "dyn", "b0"]
GATES = [0.05, 0.1, 0.2, 0.4, 1.0]
SEEDS = [0, 42, 43]


def _num(v):
    return f"{v:g}".replace(".", "p")


def new_tag(model, arm, gate, seed):
    # the reach family/grammar: cells shared with the held full reach
    # wave carry BYTE-IDENTICAL tags (later release no-ops them)
    return f"pofdreach_{model}_{arm}_ea{_num(gate)}_w0p5_l0p2_es0_s{seed}"


def cell_want(model, arm, gate, seed):
    want = dict(AR.SHARED_WANT)
    want.update(AK.ARM_WANT_K0 if arm == "k0" else AR.ARM_WANT[arm])
    want["base_model"] = (AR.MODELS[model],)
    want["seed"] = (seed,)
    want["eps_ai"] = (gate,)
    want["ai_gate_mode"] = (AR.ABSENT, "threshold")
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=AR.DEFAULT_ROOTS)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--expect-reused", type=int, default=None)
    ap.add_argument("--expect-new", type=int, default=None)
    args = ap.parse_args()

    corpus = AR.scan(args.roots)
    print(f"[audit] scanned {len(corpus)} run dirs")

    cells, near_misses = [], []
    for model in AR.MODELS:
        for arm in ARMS:
            for gate in GATES:
                for seed in SEEDS:
                    want = cell_want(model, arm, gate, seed)
                    matches = []
                    for name, root, cfg in corpus:
                        fv = AR.field_verdict(cfg, want)
                        bad = {k: v for k, v in fv.items() if not v[2]}
                        if not bad:
                            matches.append((name, root, cfg))
                        elif (len(bad) <= 2 and cfg.get("base_model") ==
                                AR.MODELS[model]
                                and cfg.get("seed") == seed
                                and not cfg.get("eps", 1.0)
                                and not {"eps_ai", "kl_beta",
                                         "training_style",
                                         "icl_k"} & bad.keys()):
                            near_misses.append(
                                (model, arm, gate, seed, name,
                                 {k: (v[1] if v[1] is not AR.ABSENT
                                      else "<ABSENT>", v[0])
                                  for k, v in bad.items()}))
                    verified = []
                    for name, root, cfg in matches:
                        ok, note, arts = AR.completeness(
                            os.path.join(root, name))
                        if ok:
                            verified.append((sum(arts.values()), name,
                                             root, cfg, arts))
                        else:
                            near_misses.append(
                                (model, arm, gate, seed, name,
                                 {"completeness": (note, "complete")}))
                    verified.sort(key=lambda x: (-x[0], x[1]))
                    cell = {"model": model, "arm": arm, "gate": gate,
                            "seed": seed}
                    if verified:
                        _, name, root, cfg, arts = verified[0]
                        cell.update({
                            "status": "reused", "run_tag": name,
                            "source_root": os.path.relpath(root, REPO),
                            "artifacts": arts,
                            "recorded": {k: cfg.get(k)
                                         for k in AR.RECORDED_ONLY},
                            "alternates": [v[1] for v in verified[1:]]})
                    elif arm == "k0" and seed != 0:
                        # deterministic structural reference: maps to the
                        # seed-0 k0 run; NEVER queued
                        cell.update({"status": "reference",
                                     "run_tag": new_tag(model, "k0", gate,
                                                        0),
                                     "reference_seed": 0})
                    else:
                        cell.update({"status": "new",
                                     "run_tag": new_tag(model, arm, gate,
                                                        seed)})
                    cells.append(cell)

    # reference cells whose seed-0 anchor is itself a REUSED cell must
    # point at the reused tag, not the conceptual pofdreach_ one
    s0_tag = {(c["model"], c["arm"], c["gate"]): c["run_tag"]
              for c in cells if c["seed"] == 0}
    for c in cells:
        if c["status"] == "reference":
            c["run_tag"] = s0_tag[(c["model"], "k0", c["gate"])]

    reused = [c for c in cells if c["status"] == "reused"]
    new = [c for c in cells if c["status"] == "new"]
    refs = [c for c in cells if c["status"] == "reference"]

    print(f"\n== reused cells ({len(reused)}) ==")
    for c in reused:
        a = c["artifacts"]
        flags = "".join(k[0].upper() if v else k[0]
                        for k, v in (("gate_raw", a["gate_raw"]),
                                     ("twin_raw", a["twin_raw"]),
                                     ("icl_idx_raw", a["icl_idx_raw"])))
        print(f"  {c['model']:9s} {c['arm']:3s} ea={c['gate']!s:5s} "
              f"s{c['seed']:<2d} <- {c['run_tag']}  [{flags}]")
    print(f"\n== reference cells ({len(refs)}; k0 seed repetitions -> "
          f"the deterministic seed-0 run) ==")
    for c in refs[:6]:
        print(f"  {c['model']:9s} k0  ea={c['gate']!s:5s} s{c['seed']:<2d} "
              f"-> {c['run_tag']}")
    if len(refs) > 6:
        print(f"  ... {len(refs) - 6} more")
    if near_misses:
        print(f"\n== near-miss candidates rejected ({len(near_misses)}) ==")
        for model, arm, gate, seed, name, bad in near_misses[:20]:
            print(f"  {model} {arm} ea={gate} s{seed}: {name} -> {bad}")

    per_model = {m: sum(1 for c in new if c["model"] == m)
                 for m in AR.MODELS}
    per_arm = {a: sum(1 for c in new if c["arm"] == a) for a in ARMS}
    per_gate = {g: sum(1 for c in new if c["gate"] == g) for g in GATES}
    print(f"\n== counts ==")
    print(f"  conceptual cells: {len(cells)}")
    print(f"  reused (complete): {len(reused)}")
    print(f"  literally missing: {len(new) + len(refs)}")
    print(f"  redundant k0 seed repetitions (reference): {len(refs)}")
    print(f"  informative jobs to queue: {len(new)}")
    print(f"  new per model: {per_model}")
    print(f"  new per arm:   {per_arm}")
    print(f"  new per gate:  {per_gate}")

    bad_expect = []
    if args.expect_reused is not None and len(reused) != args.expect_reused:
        bad_expect.append(f"reused {len(reused)} != {args.expect_reused}")
    if args.expect_new is not None and len(new) != args.expect_new:
        bad_expect.append(f"new {len(new)} != {args.expect_new}")
    if bad_expect:
        print("\nDISCREPANCY: " + "; ".join(bad_expect))
        print("Manifest NOT written; never force reuse or scope changes.")
        sys.exit(1)

    manifest = {
        "wave": "sft_icl_nopeer_grid3",
        "audited": "2026-08-14 local corpora, by config fields + 30-round "
                   "completeness. Statuses: reused / new / reference (k0 "
                   "seed repetitions -> the deterministic seed-0 run; "
                   "never queued).",
        "roots": [os.path.relpath(r, REPO) for r in args.roots],
        "grid": {"models": AR.MODELS, "arms": ARMS, "gates": GATES,
                 "seeds": SEEDS, "n_agents": AR.N_AGENTS,
                 "n_rounds": AR.N_ROUNDS},
        "counts": {"cells": len(cells), "reused": len(reused),
                   "missing": len(new) + len(refs),
                   "reference_k0": len(refs), "new": len(new),
                   "new_per_model": per_model, "new_per_arm": per_arm,
                   "new_per_gate": {str(k): v
                                    for k, v in per_gate.items()},
                   "baselines": len(AR.MODELS)},
        "cells": cells,
        "baselines": [{"model": m, "seed": 0, "status": "reused",
                       "run_tag": AR.base_tag(m, 0)} for m in AR.MODELS],
    }
    if args.write:
        with open(MANIFEST, "w") as fh:
            json.dump(manifest, fh, indent=1)
            fh.write("\n")
        print(f"\n[audit] wrote {MANIFEST}")
    else:
        print("\n[audit] dry run (pass --write to write the manifest)")


if __name__ == "__main__":
    main()
