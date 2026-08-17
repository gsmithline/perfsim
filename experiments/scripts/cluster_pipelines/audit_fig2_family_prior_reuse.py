#!/usr/bin/env python3
"""Reuse audit for the Figure-2 FAMILY-PRIOR SCOUT (2026-08-17).

The scout asks how much of each checkpoint family's zero-shot prior
survives the closed loop, across the SFT-retention dose. The cell set:

  checkpoints {qwen7b, qwen3_8b, olmo7b, olmo3_7b, mistral7b,
               ministral8b}
  x arms {b0 = ordinary SFT, b0p5 = forward-KL SFT beta=0.5,
          b1 = forward-KL SFT beta=1, k0 = frozen plain prompting}
  x eps_AI 1.0 (numeric strict-< threshold) x eps_social {0.05, 0.2}
  x seed 0 = 48 conceptual cells

canonical Action environment, W=0.5, lam=0.2, 30 rounds, nested
AI-then-peer operator, matched twins, greedy serving, fresh LoRA each
round on trained arms.

Matching is by config fields + complete 30-round (30, 723)
trajectories WITH a recorded matched twin, never tag similarity.
sft_kl arms REQUIRE kl_direction=forward (the reverse-KL era is a
different intervention). Innate-clamp runs are EXCLUDED by field
(innate_clamp_mode must be absent) -- the clamp waves ran this exact
gate/es surface with 145 agents pinned, which is a different
population. qwen3_8b REQUIRES chat_thinking=False; the other five
must lack the key entirely.

New cells take pofdfam_ tags (gen_pofd_sweep.py FAM block). Writes
experiments/condor/manifest_fig2_family_prior.json. With --expect-*
set, a count mismatch prints a DISCREPANCY report and exits 1 WITHOUT
writing; by default the audit REPORTS what it finds (never force a
particular reuse split).
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
                        "manifest_fig2_family_prior.json")
MODELS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "ministral8b": "mistralai/Ministral-8B-Instruct-2410",
}
ARMS = ["b0", "b0p5", "b1", "k0"]
GATE = 1.0
EPS_SOCIALS = [0.05, 0.2]
SEED = 0

ARM_WANT = {
    "b0": AR.ARM_WANT["b0"],
    "b1": AR.ARM_WANT["b1"],
    # the intermediate KL dose: the exact b1 envelope at beta=0.5
    "b0p5": {**AR.ARM_WANT["b1"], "kl_beta": (0.5,)},
    # frozen plain prompting: nothing trains, sft_* dials are inert
    "k0": {"training_style": ("frozen",), "kl_beta": (AR.ABSENT, 0.0),
           "use_lora": (False, 0), "fresh_each_round": (AR.ABSENT, False),
           "icl_k": (0,), "icl_days": (AR.ABSENT, 0)},
}


def new_tag(model, arm, es):
    return f"pofdfam_{model}_{arm}_ea1_w0p5_l0p2_es{f'{es:g}'.replace('.', 'p')}_s{SEED}"


def cell_want(model, arm, es):
    want = dict(AR.SHARED_WANT)
    want.update(ARM_WANT[arm])
    want["base_model"] = (MODELS[model],)
    want["seed"] = (SEED,)
    want["eps"] = (es,)
    want["eps_ai"] = (GATE,)
    want["ai_gate_mode"] = (AR.ABSENT, "threshold")
    # the clamp waves ran this exact (ea1, es) surface with 145 agents
    # permanently pinned -- a DIFFERENT population, never reusable here
    want["innate_clamp_mode"] = (AR.ABSENT,)
    want["innate_clamp_peer_mode"] = (AR.ABSENT,)
    # qwen3 must have run with thinking explicitly OFF; every other
    # checkpoint predates the dial and must lack the key
    want["chat_thinking"] = ((False,) if model == "qwen3_8b"
                             else (AR.ABSENT,))
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
        for model in MODELS:
            for arm in ARMS:
                want = cell_want(model, arm, es)
                verified = []
                for name, root, cfg in corpus:
                    fv = AR.field_verdict(cfg, want)
                    bad = {k: v for k, v in fv.items() if not v[2]}
                    if bad:
                        if (len(bad) <= 2
                                and cfg.get("base_model") == MODELS[model]
                                and not {"eps", "eps_ai",
                                         "training_style"} & bad.keys()):
                            near_misses.append(
                                (model, arm, es, name,
                                 {k: (v[1] if v[1] is not AR.ABSENT
                                      else "<ABSENT>", v[0])
                                  for k, v in bad.items()}))
                        continue
                    ok, note, arts = AR.completeness(
                        os.path.join(root, name))
                    if ok and not arts.get("twin_raw"):
                        ok, note = False, "no twin_raw (twin mandatory)"
                    if ok:
                        verified.append((sum(arts.values()), name,
                                         root, cfg, arts))
                verified.sort(key=lambda x: (-x[0], x[1]))
                cell = {"model": model, "arm": arm, "gate": GATE,
                        "eps_social": es, "seed": SEED}
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
                                 "run_tag": new_tag(model, arm, es),
                                 "validation": "PENDING"})
                cells.append(cell)

    reused = [c for c in cells if c["status"] == "reused"]
    new = [c for c in cells if c["status"] == "new"]

    print(f"\n== coverage by (eps_social, model) ==")
    for es in EPS_SOCIALS:
        for model in MODELS:
            got = [c for c in cells if c["eps_social"] == es
                   and c["model"] == model]
            have = sorted(c["arm"] for c in got
                          if c["status"] == "reused")
            miss = sorted(c["arm"] for c in got if c["status"] == "new")
            flag = "" if not miss else f"   MISSING {miss}"
            print(f"  es={es!s:5s} {model:12s}: reused {have}{flag}")
    print(f"\n== REUSED cells ({len(reused)}) ==")
    for c in reused:
        print(f"  {c['model']:12s} {c['arm']:4s} es={c['eps_social']} "
              f"<- {c['run_tag']} [{c['validation']}]")
    if near_misses:
        print(f"\n== near-miss candidates rejected ({len(near_misses)}) ==")
        for model, arm, es, name, bad in near_misses[:15]:
            print(f"  {model} {arm} es={es}: {name} -> {bad}")

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
        "wave": "fig2_family_prior",
        "audited": "2026-08-17 local corpora, by config fields + "
                   "30-round completeness + twin presence; sft_kl arms "
                   "require kl_direction=forward; clamp runs excluded "
                   "by field; qwen3_8b requires chat_thinking=False.",
        "roots": [os.path.relpath(r, REPO) for r in args.roots],
        "grid": {"models": MODELS, "arms": ARMS, "gate": GATE,
                 "eps_socials": EPS_SOCIALS, "seed": SEED,
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
