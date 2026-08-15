#!/usr/bin/env python3
"""Reuse audit for the one-seed context-depth x dual-gate grid (2026-08-15).

Grid: 3 models x 6 adaptation channels x numeric AI gates
{0.05, 0.1, 0.2, 0.4, 1.0} x numeric social gates {0, 0.2, 0.4, 1.0} x
seed 0 = 360 conceptual cells. Channels:

  b0   ordinary SFT (beta = 0), fresh LoRA each round
  k0   frozen no-context prompting (K = 0)
  fz0  frozen weights, FIXED K=8 context  (snapshot round 0)
  dyn  frozen weights, LIVE  K=8 context  (refreshed every round)
  f32  frozen weights, FIXED K=32 context (snapshot round 0)
  d32  frozen weights, LIVE  K=32 context (refreshed every round)

Every cell is AI_GATE_MODE=threshold: eps_AI=1.0 is the real strict-<
numeric gate, never all_open, and eps_social=1.0 is the real numeric
peer-confidence gate.

FIXED K=32 REQUIRES icl_snapshot_round == 0 (present, never absent).
The archived k32pri / k32noai runs are NOT fixed-K=32 context: they hold
icl_ctx_source "pristine" / "noai", i.e. the exemplar LABELS come from a
different source, and they carry no snapshot round at all. The
icl_ctx_source == "live" + snapshot requirements exclude them
structurally; a post-match guard re-asserts it per reused cell so no
future field-surface edit can silently admit them.

Matching is by config fields + complete 30-round (30, 723) trajectories,
never tag similarity. Per manifest cell: model, channel, AI gate, social
gate, seed, status (reused|new), exact run tag, source root, CONFIG
FINGERPRINT (sha256 over the matched fields' actual values), TRAJECTORY
HASH (sha256 over op_raw+pred_raw bytes) and VALIDATION STATUS
(check_pofd_sanity verdict, executed per reused run at audit time).

Writes experiments/condor/manifest_sft_icl_ctxgrid.json. With --expect-*
set, a count mismatch prints a DISCREPANCY report and exits 1 WITHOUT
writing.

Usage:
  python3 audit_sft_icl_ctxgrid_reuse.py [--roots DIR ...] [--write]
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
_spec_k0 = importlib.util.spec_from_file_location(
    "audit_k0", os.path.join(HERE, "audit_sft_k0_nopeer_reuse.py"))
AK = importlib.util.module_from_spec(_spec_k0)
_spec_k0.loader.exec_module(AK)
_spec_p2 = importlib.util.spec_from_file_location(
    "audit_peer02", os.path.join(HERE, "audit_sft_icl_peer02_reuse.py"))
AP = importlib.util.module_from_spec(_spec_p2)
_spec_p2.loader.exec_module(AP)

REPO = AR.REPO
MANIFEST = os.path.join(REPO, "experiments", "condor",
                        "manifest_sft_icl_ctxgrid.json")
ARMS = ["b0", "k0", "fz0", "dyn", "f32", "d32"]
GATES = [0.05, 0.1, 0.2, 0.4, 1.0]
EPS_SOCIALS = [0.0, 0.2, 0.4, 1.0]
SEED = 0

# K=32 channels. Both demand live-sourced exemplars with no day window;
# the ONLY difference is the snapshot round -- 0 freezes each agent's
# context verbatim after round 0, -1 (or an absent key, the pre-snapshot
# default) rebuilds it every round.
ARM_WANT_32 = {
    "f32": {"training_style": ("frozen",), "kl_beta": (0.0,),
            "use_lora": (False, 0), "fresh_each_round": (False,),
            "icl_k": (32,), "icl_days": (0,), "icl_select": ("random",),
            "icl_ctx_source": ("live",), "icl_snapshot_round": (0,)},
    "d32": {"training_style": ("frozen",), "kl_beta": (0.0,),
            "use_lora": (False, 0), "fresh_each_round": (False,),
            "icl_k": (32,), "icl_days": (0,), "icl_select": ("random",),
            "icl_ctx_source": ("live",),
            "icl_snapshot_round": (AR.ABSENT, -1)},
}
# post-match guard: (field, allowed values) re-asserted on every reused
# K=32 cell, so a pristine/noai/donor-context run can never be admitted
GUARD_32 = {"icl_k": (32,), "icl_ctx_source": ("live",)}

# EXCLUDED (model, arm) channels: combinations that cannot serve a
# parseable signal at all, so queueing them would only buy a constant.
# mistral7b at K=32 was measured on 2026-08-15 (seed-993 diagnostic,
# the first with working ICL-path DEBUG_GEN telemetry) at
# parse_fail_frac = 1.0000 every round: 100% of generations are prose
# with no digit ("To estimate the user's p..."), because under a
# 32-exemplar prompt the model stops obeying "respond with only the
# number". _parse then returns its 0.5 default for all 723 agents.
# Verified at MAX_NEW_TOKENS 6 and 24; raising the budget further is
# NOT a fix, since _parse takes the FIRST number and in prose that can
# be a rating quoted back from the profile (silently wrong values
# instead of an honest constant). qwen7b/olmo7b at K=32 are healthy and
# stay in the grid; mistral7b at K=8 is healthy and stays too.
# These cells remain in the manifest as status "excluded" -- the grid
# still accounts for all 360 conceptual cells -- but they never queue.
EXCLUDED_CHANNELS = {("mistral7b", "f32"), ("mistral7b", "d32")}
EXCLUDED_REASON = (
    "mistral7b serves no parseable signal at K=32: 100% digit-free "
    "generations (parse_fail_frac=1.0, seed-993 diagnostic 2026-08-15), "
    "so _parse returns its 0.5 default for every agent. Instruction-"
    "following failure under a 32-exemplar prompt; verified at "
    "MAX_NEW_TOKENS 6 and 24.")


def _num(v):
    return f"{v:g}".replace(".", "p")


def new_tag(model, arm, gate, es, seed=SEED):
    return (f"pofdctxgrid_{model}_{arm}_ea{_num(gate)}_w0p5_l0p2"
            f"_es{_num(es)}_s{seed}")


def arm_want(arm):
    if arm == "k0":
        return AK.ARM_WANT_K0
    if arm in ARM_WANT_32:
        return ARM_WANT_32[arm]
    return AR.ARM_WANT[arm]


def cell_want(model, arm, gate, es, seed=SEED):
    want = dict(AR.SHARED_WANT)
    want.update(arm_want(arm))
    want["base_model"] = (AR.MODELS[model],)
    want["seed"] = (seed,)
    want["eps"] = (es,)
    want["eps_ai"] = (gate,)
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

    cells, near_misses, guard_errs = [], [], []
    for es in EPS_SOCIALS:
        for model in AR.MODELS:
            for arm in ARMS:
                for gate in GATES:
                    want = cell_want(model, arm, gate, es)
                    matches = []
                    for name, root, cfg in corpus:
                        fv = AR.field_verdict(cfg, want)
                        bad = {k: v for k, v in fv.items() if not v[2]}
                        if not bad:
                            matches.append((name, root, cfg))
                        elif (len(bad) <= 2
                                and cfg.get("base_model") == AR.MODELS[model]
                                and cfg.get("seed") == SEED
                                and not {"eps", "eps_ai", "kl_beta",
                                         "training_style", "icl_k",
                                         "icl_ctx_source"} & bad.keys()):
                            near_misses.append(
                                (model, arm, gate, es, name,
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
                                (model, arm, gate, es, name,
                                 {"completeness": (note, "complete")}))
                    verified.sort(key=lambda x: (-x[0], x[1]))
                    cell = {"model": model, "arm": arm, "gate": gate,
                            "eps_social": es, "seed": SEED}
                    if verified:
                        _, name, root, cfg, arts = verified[0]
                        rd = os.path.join(root, name)
                        if arm in ARM_WANT_32:
                            for f, allowed in GUARD_32.items():
                                if cfg.get(f) not in allowed:
                                    guard_errs.append(
                                        f"{name}: {arm} cell has {f}="
                                        f"{cfg.get(f)!r}, wants one of "
                                        f"{allowed} (k32pri/k32noai-style "
                                        f"context must never be reused as "
                                        f"a K=32 population-context arm)")
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
                    elif (model, arm) in EXCLUDED_CHANNELS:
                        # no reusable run AND the channel cannot serve a
                        # signal -- record it, never queue it
                        cell.update({"status": "excluded",
                                     "run_tag": new_tag(model, arm, gate,
                                                        es),
                                     "excluded_reason": EXCLUDED_REASON,
                                     "config_fingerprint": None,
                                     "trajectory_sha256": None,
                                     "validation": "NOT RUN"})
                    else:
                        cell.update({"status": "new",
                                     "run_tag": new_tag(model, arm, gate,
                                                        es),
                                     "config_fingerprint": None,
                                     "trajectory_sha256": None,
                                     "validation": "PENDING"})
                    cells.append(cell)

    reused = [c for c in cells if c["status"] == "reused"]
    new = [c for c in cells if c["status"] == "new"]
    excluded = [c for c in cells if c["status"] == "excluded"]
    # an excluded channel must never have silently swallowed a usable
    # archived run: if one existed the cell would have been reused above
    assert not [c for c in reused
                if (c["model"], c["arm"]) in EXCLUDED_CHANNELS], \
        "an excluded channel matched an archived run -- re-examine"

    print(f"\n== reused cells ({len(reused)}) ==")
    for c in reused:
        a = c["artifacts"]
        flags = "".join(k[0].upper() if v else k[0]
                        for k, v in (("gate_raw", a["gate_raw"]),
                                     ("twin_raw", a["twin_raw"]),
                                     ("icl_idx_raw", a["icl_idx_raw"])))
        print(f"  {c['model']:9s} {c['arm']:3s} ea={c['gate']!s:4s} "
              f"es={c['eps_social']!s:3s} <- {c['run_tag']}  [{flags}] "
              f"val={c['validation']}")
    if near_misses:
        print(f"\n== near-miss candidates rejected ({len(near_misses)}) ==")
        for model, arm, gate, es, name, bad in near_misses[:20]:
            print(f"  {model} {arm} ea={gate} es={es}: {name} -> {bad}")

    def tally(rows, key):
        out = {}
        for r in rows:
            k = r[key] if not isinstance(r[key], float) else _num(r[key])
            out[k] = out.get(k, 0) + 1
        return out

    print("\n== counts ==")
    print(f"  conceptual cells: {len(cells)}")
    print(f"  reused: {len(reused)}  per model: {tally(reused, 'model')}")
    print(f"          per arm: {tally(reused, 'arm')}")
    print(f"          per es:  {tally(reused, 'eps_social')}")
    print(f"  new (QUEUED): {len(new)}  per model: "
          f"{tally(new, 'model')}")
    print(f"       per arm: {tally(new, 'arm')}")
    print(f"       per es:  {tally(new, 'eps_social')}")
    print(f"  excluded (NEVER queued): {len(excluded)}  per model: "
          f"{tally(excluded, 'model')}  per arm: "
          f"{tally(excluded, 'arm')}")
    if excluded:
        print(f"    reason: {EXCLUDED_REASON}")
    assert len(reused) + len(new) + len(excluded) == len(cells)
    bad_val = [c["run_tag"] for c in reused
               if c["validation"] not in ("PASS", "SKIPPED")]
    if bad_val:
        print(f"  REUSED CELLS FAILING VALIDATION: {bad_val}")
    if guard_errs:
        print("  K=32 CONTEXT-SOURCE GUARD TRIPPED:")
        for g in guard_errs:
            print(f"    {g}")

    bad_expect = []
    if args.expect_reused is not None and len(reused) != args.expect_reused:
        bad_expect.append(f"reused {len(reused)} != {args.expect_reused}")
    if args.expect_new is not None and len(new) != args.expect_new:
        bad_expect.append(f"new {len(new)} != {args.expect_new}")
    if bad_val:
        bad_expect.append(f"{len(bad_val)} reused cells FAIL validation")
    if guard_errs:
        bad_expect.append(f"{len(guard_errs)} K=32 guard violations")
    if bad_expect:
        print("\nDISCREPANCY: " + "; ".join(bad_expect))
        print("Manifest NOT written; never force reuse or scope changes.")
        sys.exit(1)

    manifest = {
        "wave": "sft_icl_ctxgrid",
        "audited": "2026-08-15 local corpora, by config fields + 30-round "
                   "completeness; per-reused-cell config fingerprint, "
                   "trajectory sha256 (op_raw+pred_raw bytes), and "
                   "check_pofd_sanity verdict recorded at audit time. "
                   "k32pri/k32noai runs are structurally excluded from "
                   "the K=32 channels (icl_ctx_source must be live).",
        "roots": [os.path.relpath(r, REPO) for r in args.roots],
        "grid": {"models": AR.MODELS, "arms": ARMS, "gates": GATES,
                 "eps_socials": EPS_SOCIALS, "seeds": [SEED],
                 "n_agents": AR.N_AGENTS, "n_rounds": AR.N_ROUNDS,
                 "arm_labels": {
                     "b0": "ordinary SFT (beta=0)",
                     "k0": "frozen no-context prompting (K=0)",
                     "fz0": "fixed K=8 context (snapshot round 0)",
                     "dyn": "live K=8 context (refreshed each round)",
                     "f32": "fixed K=32 context (snapshot round 0)",
                     "d32": "live K=32 context (refreshed each round)"}},
        "excluded_channels": sorted("/".join(c) for c
                                    in EXCLUDED_CHANNELS),
        "excluded_reason": EXCLUDED_REASON,
        "counts": {"cells": len(cells), "reused": len(reused),
                   "new": len(new), "excluded": len(excluded),
                   "reused_per_model": tally(reused, "model"),
                   "reused_per_arm": tally(reused, "arm"),
                   "reused_per_es": tally(reused, "eps_social"),
                   "new_per_model": tally(new, "model"),
                   "new_per_arm": tally(new, "arm"),
                   "new_per_es": tally(new, "eps_social"),
                   "excluded_per_arm": tally(excluded, "arm")},
        "cells": cells,
        "baselines": [{"model": m, "seed": SEED, "status": "reused",
                       "run_tag": AR.base_tag(m, SEED)} for m in AR.MODELS],
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
