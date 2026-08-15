#!/usr/bin/env python3
"""Reuse audit for the Figure-2 provider-separation replication (2026-08-15).

Figure 2 asks whether RETAINING THE ENTERING MODEL SIGNAL (SFT-KL at
beta=1, anchored to the base weights) preserves more provider-specific
population separation than ordinary SFT (beta=0). The cell set is
therefore:

  providers {qwen7b, olmo7b, mistral7b} x arms {b0, b1}
  x eps_AI 0.4 x eps_social {0, 0.2} x seeds {0, 42, 43}
  = 36 conceptual cells

canonical Action environment, W=0.5, lam=0.2, 30 rounds, nested
operator, matched twins, numeric threshold gates, forward KL.

Matching is by config fields + complete 30-round (30, 723)
trajectories, never tag similarity. b1 REQUIRES kl_direction=forward:
the reverse-KL era runs are a different intervention (RLHF-practice
comparison) and must never be silently folded into this figure.

New cells reuse the CANONICAL tag of their environment rather than
inventing a family, so a later broad release of that wave idempotently
no-ops them:
  es=0    -> pofdreach_<model>_b1_ea0p4_w0p5_l0p2_es0_s<seed>
  es=0.2  -> pofdws2f_<model>_b1_ea0p4_w0p5_l0p2_es0p2_s<seed>_fresh_data
Both families already have checker branches, so no checker change is
needed for this wave.

Writes experiments/condor/manifest_fig2_provider.json. With --expect-*
set, a count mismatch prints a DISCREPANCY report and exits 1 WITHOUT
writing.
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
                        "manifest_fig2_provider.json")
ARMS = ["b0", "b1"]
GATE = 0.4
EPS_SOCIALS = [0.0, 0.2]
SEEDS = [0, 42, 43]


def new_tag(model, arm, es, seed):
    if es == 0.0:
        return (f"pofdreach_{model}_{arm}_ea0p4_w0p5_l0p2_es0_s{seed}")
    return (f"pofdws2f_{model}_{arm}_ea0p4_w0p5_l0p2_es0p2_s{seed}"
            f"_fresh_data")


def cell_want(model, arm, es, seed):
    want = dict(AR.SHARED_WANT)
    want.update(AR.ARM_WANT[arm])
    want["base_model"] = (AR.MODELS[model],)
    want["seed"] = (seed,)
    want["eps"] = (es,)
    want["eps_ai"] = (GATE,)
    want["ai_gate_mode"] = (AR.ABSENT, "threshold")
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=AR.DEFAULT_ROOTS)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--expect-reused", type=int, default=None)
    ap.add_argument("--expect-new", type=int, default=None)
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    corpus = AR.scan(args.roots)
    print(f"[audit] scanned {len(corpus)} run dirs")

    cells, near_misses = [], []
    for es in EPS_SOCIALS:
        for model in AR.MODELS:
            for arm in ARMS:
                for seed in SEEDS:
                    want = cell_want(model, arm, es, seed)
                    verified = []
                    for name, root, cfg in corpus:
                        fv = AR.field_verdict(cfg, want)
                        bad = {k: v for k, v in fv.items() if not v[2]}
                        if bad:
                            if (len(bad) <= 2
                                    and cfg.get("base_model") ==
                                    AR.MODELS[model]
                                    and cfg.get("seed") == seed
                                    and not {"eps", "eps_ai",
                                             "training_style"} & bad.keys()):
                                near_misses.append(
                                    (model, arm, es, seed, name,
                                     {k: (v[1] if v[1] is not AR.ABSENT
                                          else "<ABSENT>", v[0])
                                      for k, v in bad.items()}))
                            continue
                        ok, note, arts = AR.completeness(
                            os.path.join(root, name))
                        if ok:
                            verified.append((sum(arts.values()), name,
                                             root, cfg, arts))
                    verified.sort(key=lambda x: (-x[0], x[1]))
                    cell = {"model": model, "arm": arm, "gate": GATE,
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
                            "alternates": [v[1] for v in verified[1:]]})
                    else:
                        cell.update({"status": "new",
                                     "run_tag": new_tag(model, arm, es,
                                                        seed),
                                     "validation": "PENDING"})
                    cells.append(cell)

    reused = [c for c in cells if c["status"] == "reused"]
    new = [c for c in cells if c["status"] == "new"]

    print(f"\n== coverage by (eps_social, model, arm) ==")
    for es in EPS_SOCIALS:
        for model in AR.MODELS:
            for arm in ARMS:
                got = [c for c in cells if c["eps_social"] == es
                       and c["model"] == model and c["arm"] == arm]
                have = sorted(c["seed"] for c in got
                              if c["status"] == "reused")
                miss = sorted(c["seed"] for c in got
                              if c["status"] == "new")
                flag = "" if not miss else f"   MISSING {miss}"
                print(f"  es={es!s:4s} {model:9s} {arm}: seeds {have}"
                      f"{flag}")
    print(f"\n== NEW jobs ({len(new)}) ==")
    for c in new:
        print(f"  {c['model']:9s} {c['arm']} es={c['eps_social']} "
              f"s{c['seed']:<2d} -> {c['run_tag']}")
    if near_misses:
        print(f"\n== near-miss candidates rejected ({len(near_misses)}) ==")
        for model, arm, es, seed, name, bad in near_misses[:15]:
            print(f"  {model} {arm} es={es} s{seed}: {name} -> {bad}")

    print(f"\n== counts ==")
    print(f"  conceptual cells: {len(cells)}  reused: {len(reused)}  "
          f"new: {len(new)}")
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
        "wave": "fig2_provider",
        "audited": "2026-08-15 local corpora, by config fields + 30-round "
                   "completeness; b1 requires kl_direction=forward.",
        "roots": [os.path.relpath(r, REPO) for r in args.roots],
        "grid": {"models": AR.MODELS, "arms": ARMS, "gate": GATE,
                 "eps_socials": EPS_SOCIALS, "seeds": SEEDS,
                 "n_agents": AR.N_AGENTS, "n_rounds": AR.N_ROUNDS},
        "counts": {"cells": len(cells), "reused": len(reused),
                   "new": len(new)},
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
