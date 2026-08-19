#!/usr/bin/env python3
"""Field-level reuse audit for the SECTION-4 THREE-SEED REPLICATION
of the bottom-20% fixed-vs-evolving comparison (2026-08-19,
mistral_bottom20_section4_repl).

The completed seed-0 surface (mistral_bottom20_source_impact fixed
b0/d8 + mistral_bottom20_evolving) is extended to seeds 42 and 43.
Target cells: seeds {42, 43} x conditions {fixed, evolving} x arms
{b0 = ordinary fresh SFT, d8 = frozen personal-history ICL
(ICL_K=0, ICL_DAYS=8)} x ea {0.1, 0.2, 0.4, 1} (numeric threshold)
x es {0, 0.05, 0.1, 0.2, 0.4, 1} = 192 cells, all mistral7b on the
canonical Action loop (W=0.5, lam=0.2, 30 rounds, matched twin).
FIXED pins the bottom-145 cohort (INNATE_CLAMP_MODE=bottom, cohort
seed = run seed) bit-exact at innate in population AND twin;
EVOLVING runs the identical code path with NO clamp key anywhere.

Reuse is by EXACT FIELD-LEVEL match (audit_sft_icl_reach_reuse
machinery: config fields + 30-round completeness + twin presence),
never tag similarity. The b0xa exclusion arm, the global live-K=8
dyn arm and both graph-placement cohorts are excluded by matched
fields (training_style / icl_days / innate_clamp_mode /
sft_exclude_clamped). There is NO forced expected split: the audit
reports exactly what the archive holds. Completed fixed-SFT no-peer
cells at seeds 42/43 (the tokenless mistral_innate_clamp_nopeer
originals) are the anticipated reuse pool, but any complete
field-level match counts and any surprise is REPORTED, not forced.

The only hard failure is a SAFETY one: a run directory whose name
equals a missing cell's new tag but whose config does not match the
surface (submitting that tag would no-op or write-race under the
idempotent exec).

Usage:
  python audit_bottom20_section4_repl.py [--roots R1 R2] [--print]
      [--write PATH]
  --print writes the manifest JSON to stdout (for remote capture);
  --write saves it to PATH (default: the committed manifest path).
"""
import argparse
import importlib.util
import json
import os
import sys

# remote capture pipes this file over ssh stdin (python - --print),
# where __file__ is unset: fall back to the pipelines cwd
HERE = (os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals() else os.getcwd())
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec_b = importlib.util.spec_from_file_location(
    "audit_b20", os.path.join(HERE, "audit_bottom20_reuse.py"))
AB = importlib.util.module_from_spec(_spec_b)
_spec_b.loader.exec_module(AB)
# SHARE AB's reach-audit instance: the ABSENT sentinel is a bare
# object(), so a second module load would carry a DIFFERENT sentinel
# and every absent-field want built by AB.cell_want would silently
# never match
AR = AB.AR

MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
CONDS = ["fixed", "evolving"]
ARMS = ["b0", "d8"]
GATES = [0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
SEEDS = [42, 43]
MANIFEST_PATH = os.path.join(
    REPO, "experiments", "condor",
    "manifest_bottom20_section4_repl.json")
ABSENT = AR.ABSENT

# d8 = frozen personal-history ICL: each agent sees ONLY its own
# eight most recent opinions (identical envelope to the completed
# seed-0 waves; global live-K=8 fails on icl_k/icl_days)
D8_WANT = {"training_style": ("frozen",), "kl_beta": (0.0,),
           "use_lora": (False, 0), "fresh_each_round": (False,),
           "icl_k": (0,), "icl_days": (8,), "icl_select": ("random",),
           "icl_ctx_source": ("live",),
           "icl_snapshot_round": (ABSENT, -1)}


def _num(v):
    return f"{v:g}".replace(".", "p")


def new_tag(cond, arm, gate, es, seed):
    """Tag a genuinely-missing cell would queue under. Every NEW
    fixed cell declares the stubborn operator (inert at es=0 -- the
    2026-08-18 b20 full-grid precedent); evolving tags are clampless
    by construction."""
    if cond == "fixed":
        return (f"pofdclamp_mistral7b_{arm}_bottom_stub_ea{_num(gate)}"
                f"_w0p5_l0p2_es{_num(es)}_s{seed}")
    return (f"pofdevo_mistral7b_{arm}_ea{_num(gate)}"
            f"_w0p5_l0p2_es{_num(es)}_s{seed}")


def cell_want(cond, arm, gate, es, seed):
    """Exact field surface a run must match to fill the cell."""
    if cond == "fixed":
        # the completed b20 audit surface, seed-parametric: bottom
        # clamp pinned, cohort seed = run seed, tokenless (no-peer)
        # accepted only at the b0 es=0 slots, stubborn elsewhere
        return AB.cell_want(arm, gate, es, seed)
    want = dict(AR.SHARED_WANT)
    want.update(AR.ARM_WANT["b0"] if arm == "b0" else D8_WANT)
    want["base_model"] = (MISTRAL,)
    want["seed"] = (seed,)
    want["eps_ai"] = (gate,)
    want["eps"] = (es,)
    # fully evolving: ANY clamp/exclusion key is a config lie
    for k in ("innate_clamp_mode", "innate_clamp_peer_mode",
              "innate_clamp_frac", "innate_clamp_seed",
              "sft_exclude_clamped"):
        want[k] = (ABSENT,)
    # the fig2 family waves carry save_raw_gen=True on an otherwise
    # matching surface; the pofdevo/pofdreach envelope never does
    want["save_raw_gen"] = (ABSENT, False)
    return want


def audit(roots):
    runs = AR.scan(roots)
    by_name = {name: (root, cfg) for name, root, cfg in runs}
    cells, n_reused, n_new = [], 0, 0
    surprises, hazards = [], []
    for seed in SEEDS:
        for cond in CONDS:
            for arm in ARMS:
                for gate in GATES:
                    for es in ESS:
                        want = cell_want(cond, arm, gate, es, seed)
                        hits = []
                        for name, root, cfg in runs:
                            fv = AR.field_verdict(cfg, want)
                            if all(ok for _, _, ok in fv.values()):
                                hits.append((name, root))
                        cell = {"cond": cond, "arm": arm,
                                "gate": gate, "es": es, "seed": seed,
                                "new_tag": new_tag(cond, arm, gate,
                                                   es, seed)}
                        if not hits:
                            cell["status"] = "new"
                            n_new += 1
                        else:
                            name, root = hits[0]
                            rd = os.path.join(root, name)
                            ok_c, note, arts = AR.completeness(rd)
                            if ok_c and not arts.get("twin_raw"):
                                ok_c, note = False, \
                                    "twin_raw missing/empty"
                            if not ok_c:
                                cell["status"] = "new"
                                cell["incomplete_match"] = {
                                    "run_tag": name, "note": note}
                                n_new += 1
                            else:
                                cell.update({"status": "reused",
                                             "run_tag": name,
                                             "run_dir": rd,
                                             "verdict": "PASS",
                                             "note": note})
                                n_reused += 1
                                if len(hits) > 1:
                                    cell["extra_matches"] = \
                                        [h[0] for h in hits[1:]]
                        # anticipated reuse pool: ONLY the fixed-SFT
                        # no-peer originals -- everything else
                        # reusing (or that pool missing) is a
                        # surprise worth flagging, never a failure
                        anticipated = (cond == "fixed"
                                       and arm == "b0" and es == 0.0)
                        if (cell["status"] == "reused") \
                                != anticipated:
                            surprises.append(cell)
                        if cell["status"] == "new" \
                                and cell["new_tag"] in by_name:
                            hazards.append(cell["new_tag"])
                        cells.append(cell)
    return cells, n_reused, n_new, surprises, hazards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=AR.DEFAULT_ROOTS)
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--write", nargs="?", const=MANIFEST_PATH,
                    default=None)
    args = ap.parse_args()

    cells, n_reused, n_new, surprises, hazards = audit(args.roots)
    manifest = {
        "key": "mistral_bottom20_section4_repl",
        "conds": CONDS, "arms": ARMS, "gates": GATES, "ess": ESS,
        "seeds": SEEDS,
        "n_cells": len(cells), "n_reused": n_reused, "n_new": n_new,
        "cells": cells,
    }
    for c in cells:
        print(f"[audit_b20r] {c['cond']:<8} {c['arm']:<2} "
              f"ea{c['gate']:<4g} es{c['es']:<4g} s{c['seed']} -> "
              f"{c['status']}"
              + (f" ({c.get('run_tag')})"
                 if c["status"] == "reused" else ""),
              file=sys.stderr)
    print(f"[audit_b20r] {n_reused} reused / {n_new} new of "
          f"{len(cells)} cells", file=sys.stderr)
    for c in surprises:
        print(f"[audit_b20r] SURPRISE: {c['cond']} {c['arm']} "
              f"ea{c['gate']:g} es{c['es']:g} s{c['seed']} is "
              f"{c['status']} (anticipated only fixed b0 es0 reuse)",
              file=sys.stderr)
    if hazards:
        # a dir already claims a missing cell's tag with a NON-
        # matching config: queueing it would no-op or write-race
        print(f"[audit_b20r] HARD FAIL: new-cell tag(s) occupied by "
              f"non-matching run dirs: {hazards}", file=sys.stderr)
        sys.exit(1)
    if args.do_print:
        print(json.dumps(manifest, indent=2))
    if args.write:
        with open(args.write, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[audit_b20r] wrote {args.write}", file=sys.stderr)


if __name__ == "__main__":
    main()
