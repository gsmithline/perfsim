#!/usr/bin/env python3
"""Field-level reuse audit for the SECTION-3 FAMILY-GATE ABLATION
(2026-08-18, fam_gate_ablation).

Grid: the six family-prior checkpoints x arms {b0 = ordinary SFT,
b1 = forward-KL SFT beta=1} x ea {0.1, 0.2, 0.4, 1.0} (numeric
strict-< threshold) at the FIXED es=0.05, lam=0.2, W=0.5 canonical
Action surface, seed 0, 30 rounds = 48 conceptual cells on the exact
completed family-prior-scout code path.

Reuse is by EXACT FIELD-LEVEL match (the fig2 family-prior audit's
cell_want with the gate re-pinned per cell), never tag similarity.
save_raw_gen=True is a MATCHED field: the fam wave persists raw
generations and the pofdevo_ fully-evolving wave (which shares the
mistral b0 surface at these exact gates) does not -- without it the
evo runs would shadow the mistral b0 cells. The EXPECTED split is
hard-asserted: exactly the 12 completed ea=1 scout cells (6
checkpoints x b0/b1 at es0.05 s0) reuse; the 36 cells at ea
{0.1, 0.2, 0.4} are new. Any other split is a HARD FAIL -- report
it, do not force the count.

Usage:
  python audit_fam_gate_reuse.py [--roots R1 R2] [--print]
      [--write PATH]
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "audit_fam", os.path.join(HERE,
                              "audit_fig2_family_prior_reuse.py"))
AF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AF)
AR = AF.AR

MODELS = list(AF.MODELS)
ARMS = ["b0", "b1"]
GATES = [0.1, 0.2, 0.4, 1.0]
ES = 0.05
SEED = 0
MANIFEST_PATH = os.path.join(
    REPO, "experiments", "condor", "manifest_fam_gate_ablation.json")


def _num(v):
    return f"{v:g}".replace(".", "p")


def cell_tag(model, arm, gate):
    return (f"pofdfam_{model}_{arm}_ea{_num(gate)}_w0p5_l0p2"
            f"_es{_num(ES)}_s{SEED}")


def cell_want(model, arm, gate):
    want = AF.cell_want(model, arm, ES)
    want["eps_ai"] = (gate,)
    # the fam code path persists raw generations; the pofdevo_
    # fully-evolving wave shares the mistral b0 surface at these
    # gates but does NOT -- this field keeps it out
    want["save_raw_gen"] = (True,)
    return want


def audit(roots):
    runs = AR.scan(roots)
    cells, n_reused, n_new, unexpected = [], 0, 0, []
    for model in MODELS:
        for arm in ARMS:
            for gate in GATES:
                want = cell_want(model, arm, gate)
                hits = []
                for name, root, cfg in runs:
                    fv = AR.field_verdict(cfg, want)
                    if all(ok for _, _, ok in fv.values()):
                        hits.append((name, root))
                cell = {"model": model, "arm": arm, "gate": gate,
                        "es": ES, "seed": SEED,
                        "new_tag": cell_tag(model, arm, gate)}
                if not hits:
                    cell["status"] = "new"
                    n_new += 1
                else:
                    name, root = hits[0]
                    rd = os.path.join(root, name)
                    ok_c, note, arts = AR.completeness(rd)
                    if ok_c and not arts.get("twin_raw"):
                        ok_c, note = False, "twin_raw missing/empty"
                    if not ok_c:
                        cell["status"] = "new"
                        cell["incomplete_match"] = {"run_tag": name,
                                                    "note": note}
                        n_new += 1
                    else:
                        cell.update({"status": "reused",
                                     "run_tag": name, "run_dir": rd,
                                     "verdict": "PASS", "note": note})
                        n_reused += 1
                        if len(hits) > 1:
                            cell["extra_matches"] = [h[0]
                                                     for h in hits[1:]]
                is_reuse_slot = (gate == 1.0)
                if cell["status"] == "reused" and not is_reuse_slot:
                    unexpected.append(cell)
                if cell["status"] == "new" and is_reuse_slot:
                    unexpected.append(cell)
                cells.append(cell)
    return cells, n_reused, n_new, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=AR.DEFAULT_ROOTS)
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--write", nargs="?", const=MANIFEST_PATH,
                    default=None)
    args = ap.parse_args()

    cells, n_reused, n_new, unexpected = audit(args.roots)
    manifest = {"key": "fam_gate_ablation", "models": MODELS,
                "arms": ARMS, "gates": GATES, "es": ES, "seed": SEED,
                "n_cells": len(cells), "n_reused": n_reused,
                "n_new": n_new, "cells": cells}
    for c in cells:
        print(f"[audit_famg] {c['model']:<11} {c['arm']:<3} "
              f"ea{c['gate']:<4g} -> {c['status']}"
              + (f" ({c.get('run_tag')})"
                 if c["status"] == "reused" else ""),
              file=sys.stderr)
    print(f"[audit_famg] {n_reused} reused / {n_new} new of "
          f"{len(cells)} cells", file=sys.stderr)
    if unexpected or n_reused != 12 or n_new != 36:
        print(f"[audit_famg] HARD FAIL: expected 12 reused (the ea=1 "
              f"scout cells) + 36 new, got {n_reused} reused / "
              f"{n_new} new; unexpected: "
              f"{[(c['model'], c['arm'], c['gate'], c['status']) for c in unexpected]}",
              file=sys.stderr)
        sys.exit(1)
    if args.do_print:
        print(json.dumps(manifest, indent=2))
    if args.write:
        with open(args.write, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[audit_famg] wrote {args.write}", file=sys.stderr)


if __name__ == "__main__":
    main()
