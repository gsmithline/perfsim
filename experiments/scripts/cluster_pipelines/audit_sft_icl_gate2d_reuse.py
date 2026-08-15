#!/usr/bin/env python3
"""Reuse audit for the Mistral two-dimensional gate grid (2026-08-15).

Grid: mistral7b x arms {b0 (ordinary SFT, beta=0), dyn (frozen weights,
live K=8 ICL refreshed every round)} x numeric AI gates
{0.05, 0.1, 0.2, 0.4, 1.0} x eps_social {0, 0.2, 0.4, 1.0} x seeds
{0, 42, 43} = 120 conceptual cells. EVERY cell is AI_GATE_MODE=threshold
-- eps_AI=1.0 is the real strict-< numeric gate, never all_open -- and
eps_social=1.0 is the real numeric peer-confidence gate. Matching is by
config fields + complete 30-round (30, 723) trajectories, never tag
similarity; the field surface is shared with the reach/peer02 audits with
eps (social) swept per cell.

Per manifest cell: arm, AI gate, social gate, seed, status (reused|new),
exact run tag, source root, CONFIG FINGERPRINT (sha256 over the matched
scientific fields' actual values), TRAJECTORY HASH (sha256 over
op_raw+pred_raw bytes), and VALIDATION STATUS (check_pofd_sanity verdict,
executed per reused run at audit time).

Writes experiments/condor/manifest_sft_icl_gate2d.json. With --expect-*
set, a count mismatch prints a DISCREPANCY report and exits 1 WITHOUT
writing.

Usage:
  python3 audit_sft_icl_gate2d_reuse.py [--roots DIR ...] [--write]
      [--expect-reused N] [--expect-new N] [--skip-validate]
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
_spec_p2 = importlib.util.spec_from_file_location(
    "audit_peer02", os.path.join(HERE, "audit_sft_icl_peer02_reuse.py"))
AP = importlib.util.module_from_spec(_spec_p2)
_spec_p2.loader.exec_module(AP)

REPO = AR.REPO
MANIFEST = os.path.join(REPO, "experiments", "condor",
                        "manifest_sft_icl_gate2d.json")
MODEL = "mistral7b"
ARMS = ["b0", "dyn"]
GATES = [0.05, 0.1, 0.2, 0.4, 1.0]
EPS_SOCIALS = [0.0, 0.2, 0.4, 1.0]
SEEDS = [0, 42, 43]


def _num(v):
    return f"{v:g}".replace(".", "p")


def new_tag(arm, gate, es, seed):
    return (f"pofdgate2d_{MODEL}_{arm}_ea{_num(gate)}_w0p5_l0p2"
            f"_es{_num(es)}_s{seed}")


def cell_want(arm, gate, es, seed):
    want = dict(AR.SHARED_WANT)
    want.update(AR.ARM_WANT[arm])
    want["base_model"] = (AR.MODELS[MODEL],)
    want["seed"] = (seed,)
    want["eps"] = (es,)                  # the swept peer-confidence gate
    want["eps_ai"] = (gate,)             # numeric strict-< threshold
    want["ai_gate_mode"] = (AR.ABSENT, "threshold")
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=AR.DEFAULT_ROOTS)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--expect-reused", type=int, default=None)
    ap.add_argument("--expect-new", type=int, default=None)
    ap.add_argument("--skip-validate", action="store_true",
                    help="skip the per-reused-run checker execution")
    args = ap.parse_args()

    corpus = AR.scan(args.roots)
    print(f"[audit] scanned {len(corpus)} run dirs")

    cells, near_misses = [], []
    for es in EPS_SOCIALS:
        for gate in GATES:
            for arm in ARMS:
                for seed in SEEDS:
                    want = cell_want(arm, gate, es, seed)
                    matches = []
                    for name, root, cfg in corpus:
                        fv = AR.field_verdict(cfg, want)
                        bad = {k: v for k, v in fv.items() if not v[2]}
                        if not bad:
                            matches.append((name, root, cfg))
                        elif (len(bad) <= 2 and cfg.get("base_model") ==
                                AR.MODELS[MODEL]
                                and cfg.get("seed") == seed
                                and not {"eps", "eps_ai", "kl_beta",
                                         "training_style",
                                         "icl_k"} & bad.keys()):
                            near_misses.append(
                                (arm, gate, es, seed, name,
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
                                (arm, gate, es, seed, name,
                                 {"completeness": (note, "complete")}))
                    verified.sort(key=lambda x: (-x[0], x[1]))
                    cell = {"model": MODEL, "arm": arm, "gate": gate,
                            "eps_social": es, "seed": seed}
                    if verified:
                        _, name, root, cfg, arts = verified[0]
                        rd = os.path.join(root, name)
                        cell.update({
                            "status": "reused", "run_tag": name,
                            "source_root": os.path.relpath(root, REPO),
                            "artifacts": arts,
                            "config_fingerprint": AP.fingerprint(cfg, want),
                            "trajectory_sha256": AP.traj_hash(rd),
                            "validation": ("SKIPPED" if args.skip_validate
                                           else AP.validate(rd)),
                            "recorded": {k: cfg.get(k)
                                         for k in AR.RECORDED_ONLY},
                            "alternates": [v[1] for v in verified[1:]]})
                    else:
                        cell.update({"status": "new",
                                     "run_tag": new_tag(arm, gate, es,
                                                        seed),
                                     "config_fingerprint": None,
                                     "trajectory_sha256": None,
                                     "validation": "PENDING"})
                    cells.append(cell)

    reused = [c for c in cells if c["status"] == "reused"]
    new = [c for c in cells if c["status"] == "new"]

    print(f"\n== reused cells ({len(reused)}) ==")
    for c in reused:
        a = c["artifacts"]
        flags = "".join(k[0].upper() if v else k[0]
                        for k, v in (("gate_raw", a["gate_raw"]),
                                     ("twin_raw", a["twin_raw"]),
                                     ("icl_idx_raw", a["icl_idx_raw"])))
        print(f"  {c['arm']:3s} ea={c['gate']!s:4s} es={c['eps_social']!s:3s} "
              f"s{c['seed']:<2d} <- {c['run_tag']}  [{flags}] "
              f"val={c['validation']}")
    if near_misses:
        print(f"\n== near-miss candidates rejected ({len(near_misses)}) ==")
        for arm, gate, es, seed, name, bad in near_misses[:25]:
            print(f"  {arm} ea={gate} es={es} s{seed}: {name} -> {bad}")

    per_es_r = {_num(e): sum(1 for c in reused if c["eps_social"] == e)
                for e in EPS_SOCIALS}
    per_es_n = {_num(e): sum(1 for c in new if c["eps_social"] == e)
                for e in EPS_SOCIALS}
    per_arm_n = {a: sum(1 for c in new if c["arm"] == a) for a in ARMS}
    print(f"\n== counts ==")
    print(f"  conceptual cells: {len(cells)}")
    print(f"  reused: {len(reused)}  per es: {per_es_r}")
    print(f"  new: {len(new)}  per es: {per_es_n}")
    print(f"  new per arm: {per_arm_n}")
    bad_val = [c["run_tag"] for c in reused
               if c["validation"] not in ("PASS", "SKIPPED")]
    if bad_val:
        print(f"  REUSED CELLS FAILING VALIDATION: {bad_val}")

    bad_expect = []
    if args.expect_reused is not None and len(reused) != args.expect_reused:
        bad_expect.append(f"reused {len(reused)} != {args.expect_reused}")
    if args.expect_new is not None and len(new) != args.expect_new:
        bad_expect.append(f"new {len(new)} != {args.expect_new}")
    if bad_val:
        bad_expect.append(f"{len(bad_val)} reused cells FAIL validation")
    if bad_expect:
        print("\nDISCREPANCY: " + "; ".join(bad_expect))
        print("Manifest NOT written; never force reuse or scope changes.")
        sys.exit(1)

    manifest = {
        "wave": "sft_icl_gate2d",
        "audited": "2026-08-15 local corpora, by config fields + 30-round "
                   "completeness; per-reused-cell config fingerprint, "
                   "trajectory sha256 (op_raw+pred_raw bytes), and "
                   "check_pofd_sanity verdict recorded at audit time.",
        "roots": [os.path.relpath(r, REPO) for r in args.roots],
        "grid": {"model": MODEL, "arms": ARMS, "gates": GATES,
                 "eps_socials": EPS_SOCIALS, "seeds": SEEDS,
                 "n_agents": AR.N_AGENTS, "n_rounds": AR.N_ROUNDS},
        "counts": {"cells": len(cells), "reused": len(reused),
                   "new": len(new), "reused_per_es": per_es_r,
                   "new_per_es": per_es_n, "new_per_arm": per_arm_n},
        "cells": cells,
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
