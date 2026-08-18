#!/usr/bin/env python3
"""Field-level reuse audit for the BOTTOM-20% SOURCE-IMPACT wave
(2026-08-18, mistral_bottom20_source_impact).

The wave's 36 conceptual cells are mistral7b x arms {b0, b0xa, d8} x
ea {0.1, 0.2, 0.4, 1.0} (numeric threshold) x seeds {0, 42, 43} on
the BOTTOM cohort (INNATE_CLAMP_MODE=bottom: the 145 agents with the
lowest innate Action opinions, agent-id tie-break), es=0, no peer
mode, 30 rounds, matched twin.

Reuse is by EXACT FIELD-LEVEL match (audit_sft_icl_reach_reuse
machinery: config fields + 30-round completeness + twin presence),
never tag similarity. The EXPECTED split is hard-asserted: exactly
the 12 completed no-peer-clamp b0 bottom cells reuse; all 12 b0xa and
12 d8 cells are new (the source-exclusion and personal-history arms
never ran on the bottom cohort). The old global live-K=8 dyn arm and
both graph-placement cohorts are excluded by construction
(training_style / innate_clamp_mode are matched fields). Any other
split is a HARD FAIL -- report it, do not force the count.

Usage:
  python audit_bottom20_reuse.py [--roots R1 R2] [--print]
      [--write PATH]
  --print writes the manifest JSON to stdout (for remote capture);
  --write saves it to PATH (default: the committed manifest path).
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "audit_reach", os.path.join(HERE, "audit_sft_icl_reach_reuse.py"))
AR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AR)

MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
ARMS = ["b0", "b0xa", "d8"]
GATES = [0.1, 0.2, 0.4, 1.0]
SEEDS = [0, 42, 43]
MANIFEST_PATH = os.path.join(
    REPO, "experiments", "condor",
    "manifest_bottom20_source_impact.json")
ABSENT = AR.ABSENT


def _num(v):
    return f"{v:g}".replace(".", "p")


def cell_tag(arm, gate, seed):
    return (f"pofdclamp_mistral7b_{arm}_bottom_ea{_num(gate)}"
            f"_w0p5_l0p2_es0_s{seed}")


def cell_want(arm, gate, seed):
    """Exact field surface a run must match to fill (arm, gate, seed).
    Builds on the reach SHARED_WANT (canonical no-peer Action
    environment) + the arm envelope, then pins the bottom clamp."""
    want = dict(AR.SHARED_WANT)
    if arm in ("b0", "b0xa"):
        want.update(AR.ARM_WANT["b0"])
    else:                                   # d8: personal-history ICL
        want.update({"training_style": ("frozen",), "kl_beta": (0.0,),
                     "use_lora": (False, 0),
                     "fresh_each_round": (False,),
                     "icl_k": (0,), "icl_days": (8,),
                     "icl_select": ("random",),
                     "icl_ctx_source": ("live",),
                     "icl_snapshot_round": (ABSENT, -1)})
    want["base_model"] = (MISTRAL,)
    want["seed"] = (seed,)
    want["eps_ai"] = (gate,)
    want["innate_clamp_mode"] = ("bottom",)
    want["innate_clamp_frac"] = (0.2,)
    want["innate_clamp_seed"] = (seed,)
    # tokenless no-peer wave: never a peer operator
    want["innate_clamp_peer_mode"] = (ABSENT,)
    # the exclusion flag separates b0 from b0xa at an otherwise
    # identical surface -- pin it EXPLICITLY on every arm
    want["sft_exclude_clamped"] = ((True,) if arm == "b0xa"
                                   else (ABSENT,))
    return want


def audit(roots):
    runs = AR.scan(roots)
    cells, n_reused, n_new = [], 0, 0
    unexpected = []
    for arm in ARMS:
        for gate in GATES:
            for seed in SEEDS:
                want = cell_want(arm, gate, seed)
                hits = []
                for name, root, cfg in runs:
                    fv = AR.field_verdict(cfg, want)
                    if all(ok for _, _, ok in fv.values()):
                        hits.append((name, root))
                cell = {"arm": arm, "gate": gate, "seed": seed,
                        "new_tag": cell_tag(arm, gate, seed)}
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
                        cell.update({"status": "reused", "run_tag": name,
                                     "run_dir": rd, "verdict": "PASS",
                                     "note": note})
                        n_reused += 1
                        if len(hits) > 1:
                            cell["extra_matches"] = [h[0]
                                                     for h in hits[1:]]
                if cell["status"] == "reused" and arm != "b0":
                    unexpected.append(cell)
                if cell["status"] == "new" and arm == "b0":
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
    manifest = {
        "key": "mistral_bottom20_source_impact",
        "arms": ARMS, "gates": GATES, "seeds": SEEDS,
        "n_cells": len(cells), "n_reused": n_reused, "n_new": n_new,
        "cells": cells,
    }
    for c in cells:
        print(f"[audit_b20] {c['arm']:<4} ea{c['gate']:<4g} "
              f"s{c['seed']:<2} -> {c['status']}"
              + (f" ({c.get('run_tag')})"
                 if c["status"] == "reused" else ""),
              file=sys.stderr)
    print(f"[audit_b20] {n_reused} reused / {n_new} new of "
          f"{len(cells)} cells", file=sys.stderr)
    if unexpected or n_reused != 12 or n_new != 24:
        # the design EXPECTS exactly the 12 completed b0 bottom cells
        # to reuse -- any other split means the archive changed under
        # us and the wave must be re-planned, not forced
        print(f"[audit_b20] HARD FAIL: expected 12 reused (all b0) + "
              f"24 new (b0xa/d8), got {n_reused} reused / {n_new} new;"
              f" unexpected cells: "
              f"{[(c['arm'], c['gate'], c['seed'], c['status']) for c in unexpected]}",
              file=sys.stderr)
        sys.exit(1)
    if args.do_print:
        print(json.dumps(manifest, indent=2))
    if args.write:
        with open(args.write, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[audit_b20] wrote {args.write}", file=sys.stderr)


if __name__ == "__main__":
    main()
