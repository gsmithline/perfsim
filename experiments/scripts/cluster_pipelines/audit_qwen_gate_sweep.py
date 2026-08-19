#!/usr/bin/env python3
"""Field-level reuse audit for the QWEN GATE SWEEP (2026-08-19,
qwen_gate_sweep).

The complete conceptual grid is 2 models x eps_AI {.05, .1, .2, .4, 1}
(numeric threshold) x eps_social {0, .05, .1, .2, .4, 1} x seed 0 = 60
cells of regularized SFT at lambda = 1 (the runner spells the KL
coefficient kl_beta; forward direction against the FIXED pristine
base) on the canonical Action environment: movielens Action, 723
agents, 30 rounds, W = .5, innate anchor k = .2, fresh LoRA each
round, matched no-platform twin (the runner instantiates it whenever
eps_social > 0; at eps_social = 0 the no-platform counterfactual is
innate and no twin is written), corrected nested AI-then-peer
operator, greedy serving.

  qwen7b    Qwen/Qwen2.5-7B-Instruct  -- default chat template
  qwen3_8b  Qwen/Qwen3-8B             -- CHAT_THINKING=0, i.e. the
            hybrid-reasoning template pinned OFF (config records
            chat_thinking=False; its ABSENCE would mean the thinking
            template ran and the cell is NOT reusable)

Reuse is by EXACT FIELD-LEVEL match (config fields + 30-round
completeness + twin presence where eps_social > 0), never tag
similarity, so completed cells from the family-prior / gate-ablation
/ reach waves fill the grid wherever they match exactly. Nothing is
forced: the audit reports whatever split the archive holds, and only
fails if a missing cell's would-be tag is already occupied by a
non-matching directory (which the idempotent exec would silently
no-op or race).

Usage:
  python audit_qwen_gate_sweep.py [--roots R1 R2] [--print]
      [--write PATH]
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
_spec = importlib.util.spec_from_file_location(
    "audit_reach", os.path.join(HERE, "audit_sft_icl_reach_reuse.py"))
AR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AR)

ABSENT = AR.ABSENT
MODELS = {"qwen7b": "Qwen/Qwen2.5-7B-Instruct",
          "qwen3_8b": "Qwen/Qwen3-8B"}
GATES = [0.05, 0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
SEED = 0
MANIFEST_PATH = os.path.join(
    REPO, "experiments", "condor", "manifest_qwen_gate_sweep.json")

SHARED = {
    "dataset": ("movielens",), "ml_target": ("Action",),
    "pop_model": ("ab",), "run_mode": ("loop",),
    "population_update": ("nested_ai_then_social_v1",),
    "w_plat": (0.5,), "innate_lambda": (0.2,),
    "gamma_bias": (0.0,), "platform_sus_scale": (1.0,),
    "anchor_mode": ("fixed",), "deploy_every": (1,),
    "data_regime": ("replace",), "n_rounds": (30,),
    "do_sample": (False,), "n_labeled": (723,), "train_cap": (723,),
    "seed_base_data": (True,), "pristine_frac": (0.0,),
    "canary_delta": (ABSENT, 0.0), "replay_frac": (ABSENT, 0.0),
    "teacher_label_delta": (ABSENT, 0.0),
    "icrh": (ABSENT, False), "feedback_mode": (ABSENT, "none"),
    "ai_gate_mode": (ABSENT, "threshold"),
    "pop_reset": (ABSENT, False), "ab_sweeps": (ABSENT, 1),
    "profile_drop_cols": (ABSENT, [], None),
    "profile_permute_cols": (ABSENT, [], None),
    "seed": (SEED,),
    # regularized SFT at lambda = 1, forward KL against the fixed base
    "training_style": ("sft_kl",), "kl_beta": (1.0,),
    "kl_direction": ("forward",),
    "kl_ref_adapter": (ABSENT, "", None),
    "use_lora": (True, 1), "lora_r": (512,), "sft_lr": (5e-5,),
    "sft_epochs": (1,), "sft_batch_size": (4,),
    "fresh_each_round": (True,), "icl_k": (0,), "icl_days": (0,),
}


def _num(v):
    return f"{v:g}".replace(".", "p")


def cell_want(model, gate, es):
    want = dict(SHARED)
    want["base_model"] = (MODELS[model],)
    want["eps_ai"] = (gate,)
    want["eps"] = (es,)
    # Qwen3's hybrid-reasoning template must be pinned OFF. The key is
    # written only when CHAT_THINKING carries a directive, so its
    # absence on a qwen3 run means the thinking template ran.
    want["chat_thinking"] = ((False,) if model == "qwen3_8b"
                             else (ABSENT,))
    return want


def new_tag(model, gate, es):
    return (f"pofdqgs_{model}_b1_ea{_num(gate)}_w0p5_l0p2"
            f"_es{_num(es)}_s{SEED}")


def audit(roots):
    runs = AR.scan(roots)
    by_name = {name: (root, cfg) for name, root, cfg in runs}
    cells, n_reused, n_new = [], 0, 0
    hazards = []
    for model in MODELS:
        for gate in GATES:
            for es in ESS:
                want = cell_want(model, gate, es)
                hits = []
                for name, root, cfg in runs:
                    fv = AR.field_verdict(cfg, want)
                    if all(ok for _, _, ok in fv.values()):
                        hits.append((name, root))
                cell = {"model": model, "gate": gate, "es": es,
                        "seed": SEED,
                        "new_tag": new_tag(model, gate, es)}
                picked = None
                for name, root in hits:
                    rd = os.path.join(root, name)
                    ok_c, note, arts = AR.completeness(rd)
                    # the twin exists only when the peer step is live
                    if ok_c and es > 0 and not arts.get("twin_raw"):
                        ok_c, note = False, "twin_raw missing/empty"
                    if ok_c:
                        picked = (name, rd, note)
                        break
                if picked is None:
                    cell["status"] = "new"
                    n_new += 1
                    if hits:
                        cell["incomplete_matches"] = [h[0]
                                                      for h in hits]
                else:
                    name, rd, note = picked
                    cell.update({"status": "reused", "run_tag": name,
                                 "run_dir": rd, "verdict": "PASS",
                                 "note": note})
                    n_reused += 1
                    others = [n for n, _ in hits if n != name]
                    if others:
                        cell["extra_matches"] = others
                if cell["status"] == "new" and cell["new_tag"] in by_name:
                    hazards.append(cell["new_tag"])
                cells.append(cell)
    return cells, n_reused, n_new, hazards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=AR.DEFAULT_ROOTS)
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--write", nargs="?", const=MANIFEST_PATH,
                    default=None)
    args = ap.parse_args()

    cells, n_reused, n_new, hazards = audit(args.roots)
    manifest = {
        "key": "qwen_gate_sweep",
        "models": list(MODELS), "gates": GATES, "ess": ESS,
        "seed": SEED,
        "n_cells": len(cells), "n_reused": n_reused, "n_new": n_new,
        "cells": cells,
    }
    for c in cells:
        print(f"[audit_qgs] {c['model']:<9} ea{c['gate']:<5g} "
              f"es{c['es']:<5g} -> {c['status']}"
              + (f" ({c.get('run_tag')})"
                 if c["status"] == "reused" else ""),
              file=sys.stderr)
    print(f"[audit_qgs] {n_reused} reused / {n_new} new of "
          f"{len(cells)} cells", file=sys.stderr)
    for model in MODELS:
        r = sum(1 for c in cells
                if c["model"] == model and c["status"] == "reused")
        print(f"[audit_qgs]   {model}: {r} reused / "
              f"{len(GATES) * len(ESS) - r} new", file=sys.stderr)
    if hazards:
        print(f"[audit_qgs] HARD FAIL: new-cell tag(s) occupied by "
              f"non-matching run dirs: {hazards}", file=sys.stderr)
        sys.exit(1)
    if args.do_print:
        print(json.dumps(manifest, indent=2))
    if args.write:
        with open(args.write, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[audit_qgs] wrote {args.write}", file=sys.stderr)


if __name__ == "__main__":
    main()
