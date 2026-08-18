#!/usr/bin/env python3
"""Field-level reuse audit for the SECTION-3 FAMILY-GATE ABLATION and
its SOCIAL-GATE EXTENSION (2026-08-18, fam_gate_ablation +
fam_gate_social).

Full conceptual surface: the six family-prior checkpoints x arms
{b0 = ordinary SFT, b1 = forward-KL SFT beta=1} x ea {0.1, 0.2, 0.4,
1.0} (numeric strict-< threshold) x es {0, 0.05, 0.2} at lam=0.2,
W=0.5, seed 0, 30 rounds on the exact completed family-prior-scout
code path = 144 conceptual cells.

Reuse is by EXACT FIELD-LEVEL match (the fig2 family-prior audit's
cell_want with gate and peer dose re-pinned per cell), never tag
similarity. save_raw_gen=True is a MATCHED field: the fam path
persists raw generations while the pofdevo_ / reach / k0 waves
(which share b0/b1 surfaces at several of these cells) do not --
without it they would shadow cells. Statuses:
  reused           a completed run field-matches (verdict PASS)
  covered_running  no completed match, but the tag is queued in the
                   fam_gate_ablation key (the es=0.05 wave the user
                   already submitted) -- NEVER re-queued here
  new              genuinely missing -> queues in fam_gate_social
One ea=1 slot (mistral7b b0 es0.2) is filled by the pofdgate2d run
that the fam scout ITSELF reused; it predates save_raw_gen, so the
strict field pass cannot see it. For ea=1 slots only, the audit
INHERITS the fam scout's committed manifest
(manifest_fig2_family_prior.json) verbatim -- the already-audited
occupant of that surface, marked "inherited" in the cell -- never a
blanket relaxation.

The EXPECTED split is hard-asserted: every es=0.05 cell reused or
covered (0 new); es=0.2 = 12 reused (the ea=1 scout cells) + 36 new;
es=0 = 48 new; total new = 84. Any other split is a HARD FAIL --
report it, do not force the count.

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
ESS = [0.0, 0.05, 0.2]
ES = 0.05          # the original 48-cell ablation surface
SEED = 0
ABLATION_CFG = os.path.join(
    REPO, "experiments", "condor",
    "configs_pofd_fam_gate_ablation.txt")
MANIFEST_PATH = os.path.join(
    REPO, "experiments", "condor", "manifest_fam_gate_social.json")


def _num(v):
    return f"{v:g}".replace(".", "p")


def cell_tag(model, arm, gate, es=ES):
    return (f"pofdfam_{model}_{arm}_ea{_num(gate)}_w0p5_l0p2"
            f"_es{_num(es)}_s{SEED}")


def cell_want(model, arm, gate, es=ES):
    want = AF.cell_want(model, arm, es)
    want["eps_ai"] = (gate,)
    # the fam code path persists raw generations; the pofdevo_ /
    # reach / k0 waves share b0/b1 surfaces at several cells but do
    # NOT -- this field keeps them out
    want["save_raw_gen"] = (True,)
    return want


def audit(roots):
    runs = AR.scan(roots)
    queued = set()
    if os.path.exists(ABLATION_CFG):
        queued = {line.split(",")[0].strip()
                  for line in open(ABLATION_CFG) if line.strip()}
    # the fam scout's committed audit: its reused occupants fill the
    # matching ea=1 slots here even when they predate save_raw_gen
    inherited = {}
    _fig2_mf = os.path.join(REPO, "experiments", "condor",
                            "manifest_fig2_family_prior.json")
    if os.path.exists(_fig2_mf):
        for c in json.load(open(_fig2_mf)).get("cells", []):
            if c.get("status") == "reused":
                inherited[(c["model"], c["arm"],
                           c["eps_social"])] = c
    cells, n_reused, n_covered, n_new, unexpected = [], 0, 0, 0, []
    for model in MODELS:
        for arm in ARMS:
            for gate in GATES:
                for es in ESS:
                    want = cell_want(model, arm, gate, es)
                    hits = []
                    for name, root, cfg in runs:
                        fv = AR.field_verdict(cfg, want)
                        if all(ok for _, _, ok in fv.values()):
                            hits.append((name, root))
                    tag = cell_tag(model, arm, gate, es)
                    cell = {"model": model, "arm": arm, "gate": gate,
                            "es": es, "seed": SEED, "new_tag": tag}
                    status = None
                    if hits:
                        name, root = hits[0]
                        rd = os.path.join(root, name)
                        ok_c, note, arts = AR.completeness(rd)
                        if ok_c and not arts.get("twin_raw"):
                            ok_c, note = False, "twin_raw missing"
                        if ok_c:
                            status = "reused"
                            cell.update({"run_tag": name,
                                         "run_dir": rd,
                                         "verdict": "PASS",
                                         "note": note})
                            n_reused += 1
                    if status is None and gate == 1.0:
                        inh = inherited.get((model, arm, es))
                        hit = (next(((n, r) for n, r, _ in runs
                                     if n == inh.get("run_tag")),
                                    None) if inh else None)
                        if hit:
                            rd = os.path.join(hit[1], hit[0])
                            ok_c, note, arts = AR.completeness(rd)
                            if ok_c and arts.get("twin_raw"):
                                status = "reused"
                                cell.update({
                                    "run_tag": hit[0], "run_dir": rd,
                                    "verdict": "PASS",
                                    "inherited": True,
                                    "note": ("inherited from "
                                             "manifest_fig2_family_"
                                             "prior")})
                                n_reused += 1
                    if status is None and tag in queued:
                        status = "covered_running"
                        cell["covering_key"] = "fam_gate_ablation"
                        n_covered += 1
                    if status is None:
                        status = "new"
                        n_new += 1
                    cell["status"] = status
                    # expectation map: es0.05 never new; es0.2 reused
                    # only at ea1; es0 always new
                    if es == 0.05 and status == "new":
                        unexpected.append(cell)
                    if es == 0.2 and status == "reused" and gate != 1.0:
                        unexpected.append(cell)
                    if es == 0.2 and status == "new" and gate == 1.0:
                        unexpected.append(cell)
                    if es == 0.0 and status != "new":
                        unexpected.append(cell)
                    cells.append(cell)
    return cells, n_reused, n_covered, n_new, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=AR.DEFAULT_ROOTS)
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--write", nargs="?", const=MANIFEST_PATH,
                    default=None)
    args = ap.parse_args()

    cells, n_reused, n_covered, n_new, unexpected = audit(args.roots)
    manifest = {"key": "fam_gate_social", "models": MODELS,
                "arms": ARMS, "gates": GATES, "ess": ESS,
                "seed": SEED, "n_cells": len(cells),
                "n_reused": n_reused, "n_covered": n_covered,
                "n_new": n_new, "cells": cells}
    for c in cells:
        print(f"[audit_famg] {c['model']:<11} {c['arm']:<3} "
              f"ea{c['gate']:<4g} es{c['es']:<4g} -> {c['status']}"
              + (f" ({c.get('run_tag')})"
                 if c["status"] == "reused" else ""),
              file=sys.stderr)
    print(f"[audit_famg] {n_reused} reused / {n_covered} covered "
          f"(running key) / {n_new} new of {len(cells)} cells",
          file=sys.stderr)
    if unexpected or n_new != 84:
        print(f"[audit_famg] HARD FAIL: expected 84 new (48 es0 + 36 "
              f"es0p2 below ea1) with every es0p05 cell reused or "
              f"covered; got {n_reused} reused / {n_covered} covered "
              f"/ {n_new} new; unexpected: "
              f"{[(c['model'], c['arm'], c['gate'], c['es'], c['status']) for c in unexpected]}",
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
