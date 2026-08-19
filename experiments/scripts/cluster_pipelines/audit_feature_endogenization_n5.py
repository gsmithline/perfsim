#!/usr/bin/env python3
"""Field-level reuse audit for the FIVE-SEED EXTENSION of the main
feature-endogenization figure (2026-08-19,
feature_endogenization_n5).

The figure (plot_feature_endogenization_main.py, panels a/b) runs six
Qwen2.5-7B-Instruct conditions on the corrected peer environment
(movielens Action, 723 agents, eps_AI=0.4, eps_social=0.2, W=0.5,
innate anchor 0.2, 30 rounds, replace + fresh adapter each round,
nested AI-then-peer operator, matched twin, greedy serving) at seeds
{0, 42, 43}:

  nat_l0    ordinary SFT, KL coefficient lambda = 0   (pofdws2f_ b0)
  nat_l0p5  forward-KL SFT, lambda = .5               (pofdws2f_ b0p5)
  nat_l1    forward-KL SFT, lambda = 1                (pofdws2f_ b1)
  frozen    frozen weights, K = 0                     (pofdicls2_ k0)
  removed   lambda = 1 + PROFILE_DROP_COLS=gender     (pofdfegd_)
  permuted  lambda = 1 + PROFILE_PERMUTE_COLS=gender  (pofdfegp_)

(The runner spells the KL coefficient kl_beta and the tags spell it
b<...>; every DISPLAYED label calls it lambda.)

This audit extends the surface to seeds 44 and 45 -- 12 target cells.
It is self-verifying: the per-condition want surface is asserted
against ALL THREE established seeds first, so a want that does not
actually describe the existing experiment is a hard failure before
any reuse decision is made. Only then are seed-44/45 runs matched by
EXACT field-level comparison (never tag similarity), so a completed
seed-44/45 trajectory is reused rather than re-run, and a
same-tag-but-different-config directory is a hard failure instead of
a silent overwrite.

Usage:
  python audit_feature_endogenization_n5.py [--roots R1 R2] [--print]
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

QWEN = "Qwen/Qwen2.5-7B-Instruct"
ABSENT = AR.ABSENT
BASE_SEEDS = [0, 42, 43]
NEW_SEEDS = [44, 45]
CONDITIONS = ["nat_l0", "nat_l0p5", "nat_l1", "frozen", "removed",
              "permuted"]
MANIFEST_PATH = os.path.join(
    REPO, "experiments", "condor",
    "manifest_feature_endogenization_n5.json")

# the corrected peer environment shared by all six conditions
SHARED = {
    "dataset": ("movielens",), "ml_target": ("Action",),
    "pop_model": ("ab",), "run_mode": ("loop",),
    "population_update": ("nested_ai_then_social_v1",),
    "base_model": (QWEN,),
    "eps": (0.2,), "eps_ai": (0.4,), "w_plat": (0.5,),
    "innate_lambda": (0.2,), "gamma_bias": (0.0,),
    "platform_sus_scale": (1.0,), "anchor_mode": ("fixed",),
    "deploy_every": (1,), "data_regime": ("replace",),
    "n_rounds": (30,), "do_sample": (False,),
    "n_labeled": (723,), "train_cap": (723,),
    "seed_base_data": (True,), "pristine_frac": (0.0,),
    "canary_delta": (ABSENT, 0.0),
    "replay_frac": (ABSENT, 0.0),
    "teacher_label_delta": (ABSENT, 0.0),
    "icrh": (ABSENT, False), "feedback_mode": (ABSENT, "none"),
    "ai_gate_mode": (ABSENT, "threshold"),
    "pop_reset": (ABSENT, False), "ab_sweeps": (ABSENT, 1),
}
# LoRA SFT envelope shared by the five trained conditions
_TRAINED = {
    "use_lora": (True, 1), "lora_r": (512,), "sft_lr": (5e-5,),
    "sft_epochs": (1,), "sft_batch_size": (4,),
    "fresh_each_round": (True,), "icl_k": (0,), "icl_days": (0,),
}
# forward KL is pinned on every arm that HAS a KL term; the lambda=0
# arm carries no KL term, so kl_direction is inert there and is
# deliberately NOT matched (the s0 anchor predates the config field)
COND_WANT = {
    "nat_l0": dict(_TRAINED, training_style=("sft",),
                   kl_beta=(0.0,),
                   profile_drop_cols=(ABSENT, [], None),
                   profile_permute_cols=(ABSENT, [], None)),
    "nat_l0p5": dict(_TRAINED, training_style=("sft_kl",),
                     kl_beta=(0.5,), kl_direction=("forward",),
                     kl_ref_adapter=(ABSENT, "", None),
                     profile_drop_cols=(ABSENT, [], None),
                     profile_permute_cols=(ABSENT, [], None)),
    "nat_l1": dict(_TRAINED, training_style=("sft_kl",),
                   kl_beta=(1.0,), kl_direction=("forward",),
                   kl_ref_adapter=(ABSENT, "", None),
                   profile_drop_cols=(ABSENT, [], None),
                   profile_permute_cols=(ABSENT, [], None)),
    # frozen never trains: lora_r / sft_* are inert and unmatched
    "frozen": {"training_style": ("frozen",), "kl_beta": (0.0,),
               "use_lora": (False, 0), "fresh_each_round": (False,),
               "icl_k": (0,), "icl_days": (0,),
               "profile_drop_cols": (ABSENT, [], None),
               "profile_permute_cols": (ABSENT, [], None)},
    "removed": dict(_TRAINED, training_style=("sft_kl",),
                    kl_beta=(1.0,), kl_direction=("forward",),
                    kl_ref_adapter=(ABSENT, "", None),
                    profile_drop_cols=(["gender"],),
                    profile_permute_cols=(ABSENT, [], None)),
    "permuted": dict(_TRAINED, training_style=("sft_kl",),
                     kl_beta=(1.0,), kl_direction=("forward",),
                     kl_ref_adapter=(ABSENT, "", None),
                     profile_drop_cols=(ABSENT, [], None),
                     profile_permute_cols=(["gender"],)),
}


def cond_want(cond, seed):
    want = dict(SHARED)
    want.update(COND_WANT[cond])
    want["seed"] = (seed,)
    return want


def cond_tag(cond, seed):
    """Tag for a cell. Seed 0's lambda=0 anchor lives in the
    reverse-era pofdws2_ family (b0 has no KL term, so it is
    direction-free -- the 2026-07-30 reuse precedent); every other
    natural cell is pofdws2f_."""
    if cond == "nat_l0":
        fam = "pofdws2" if seed == 0 else "pofdws2f"
        return (f"{fam}_qwen7b_b0_ea0p4_w0p5_l0p2_es0p2"
                f"_s{seed}_fresh_data")
    if cond == "nat_l0p5":
        return ("pofdws2f_qwen7b_b0p5_ea0p4_w0p5_l0p2_es0p2"
                f"_s{seed}_fresh_data")
    if cond == "nat_l1":
        return ("pofdws2f_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2"
                f"_s{seed}_fresh_data")
    if cond == "frozen":
        return f"pofdicls2_qwen7b_w0p5_l0p2_es0p2_ea0p4_k0_s{seed}"
    if cond == "removed":
        return ("pofdfegd_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2"
                f"_s{seed}_fresh_data")
    return ("pofdfegp_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2"
            f"_s{seed}_fresh_data")


def _match(runs, want):
    hits = []
    for name, root, cfg in runs:
        fv = AR.field_verdict(cfg, want)
        if all(ok for _, _, ok in fv.values()):
            hits.append((name, root))
    return hits


def audit(roots):
    runs = AR.scan(roots)
    by_name = {name: (root, cfg) for name, root, cfg in runs}

    # -- self-verification: the want surface MUST describe all three
    # established seeds, by tag AND by fields, or the audit is
    # describing a different experiment than the figure plots
    base = []
    for cond in CONDITIONS:
        for seed in BASE_SEEDS:
            tag = cond_tag(cond, seed)
            entry = {"cond": cond, "seed": seed, "run_tag": tag}
            if tag not in by_name:
                entry["verdict"] = "MISSING"
            else:
                cfg = by_name[tag][1]
                fv = AR.field_verdict(cfg, cond_want(cond, seed))
                bad = {k: (str(exp), str(got))
                       for k, (exp, got, ok) in fv.items() if not ok}
                entry["verdict"] = "PASS" if not bad else "FIELD_MISMATCH"
                if bad:
                    entry["mismatch"] = bad
            base.append(entry)

    cells, n_reused, n_new = [], 0, 0
    hazards = []
    for cond in CONDITIONS:
        for seed in NEW_SEEDS:
            want = cond_want(cond, seed)
            hits = _match(runs, want)
            cell = {"cond": cond, "seed": seed,
                    "new_tag": cond_tag(cond, seed)}
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
            # a directory already holding this tag with a config that
            # does NOT match would be silently no-opped or raced by
            # the idempotent exec -- never queue it
            if cell["status"] == "new" and cell["new_tag"] in by_name:
                hazards.append(cell["new_tag"])
            cells.append(cell)
    return base, cells, n_reused, n_new, hazards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=AR.DEFAULT_ROOTS)
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--write", nargs="?", const=MANIFEST_PATH,
                    default=None)
    args = ap.parse_args()

    base, cells, n_reused, n_new, hazards = audit(args.roots)
    manifest = {
        "key": "feature_endogenization_n5",
        "conditions": CONDITIONS, "base_seeds": BASE_SEEDS,
        "new_seeds": NEW_SEEDS,
        "n_cells": len(cells), "n_reused": n_reused, "n_new": n_new,
        "baseline": base, "cells": cells,
    }
    for b in base:
        print(f"[audit_fe5] BASE {b['cond']:<9} s{b['seed']:<3} "
              f"{b['verdict']}"
              + (f" {b.get('mismatch')}" if b.get("mismatch") else ""),
              file=sys.stderr)
    bad_base = [b for b in base if b["verdict"] != "PASS"]
    if bad_base:
        print(f"[audit_fe5] HARD FAIL: the audited want surface does "
              f"not describe the established runs: "
              f"{[(b['cond'], b['seed'], b['verdict']) for b in bad_base]}"
              f" -- fix the surface before extending the seeds",
              file=sys.stderr)
        sys.exit(1)
    print(f"[audit_fe5] baseline: {len(base)}/{len(base)} established "
          f"cells match the surface exactly", file=sys.stderr)
    for c in cells:
        print(f"[audit_fe5] {c['cond']:<9} s{c['seed']:<3} -> "
              f"{c['status']}"
              + (f" ({c.get('run_tag')})"
                 if c["status"] == "reused" else ""),
              file=sys.stderr)
    print(f"[audit_fe5] {n_reused} reused / {n_new} new of "
          f"{len(cells)} cells", file=sys.stderr)
    if hazards:
        print(f"[audit_fe5] HARD FAIL: new-cell tag(s) occupied by "
              f"non-matching run dirs: {hazards}", file=sys.stderr)
        sys.exit(1)
    if args.do_print:
        print(json.dumps(manifest, indent=2))
    if args.write:
        with open(args.write, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[audit_fe5] wrote {args.write}", file=sys.stderr)


if __name__ == "__main__":
    main()
