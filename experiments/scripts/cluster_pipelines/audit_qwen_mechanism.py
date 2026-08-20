#!/usr/bin/env python3
"""Field-level reuse audit for the QWEN2.5 MECHANISM DIAGNOSTIC
(2026-08-20, qwen_mechanism_frozen).

PART A conceptual grid -- the paper regime, one platform condition per
cell, on the canonical Action environment (movielens Action, 723 agents,
seed 0, 30 rounds, W = .5, gamma = 0, numeric strict-< AI gate at
eps_AI = 1, fresh LoRA each round for the trained arms, corrected nested
AI-then-peer operator, greedy serving, matched no-platform twin):

    k in {.2, 1}  x  eps_social in {0, .05, .2, 1}  x  4 platform arms
      = 32 conceptual cells

The four platform arms are

    pp   perfect prediction, m(t) = x(t)      -- CPU oracle, NOT audited
                                                here (sim_perfect_predictor)
    k0   frozen Qwen2.5 prompting, K = D = 0  -- 8 GPU cells
    b0   ordinary fresh SFT, lambda = 0       -- 8 GPU cells
    b1   fresh forward-KL SFT, lambda = 1     -- 8 GPU cells

so 24 of the 32 are GPU cells and this script audits exactly those. The
8 perfect-prediction cells are cheap CPU replays and are generated fresh
in their own namespace every time.

WHY THE FROZEN ARM IS SPECIAL. A frozen K = D = 0 model never sees the
population: its parsed prediction vector is a CONSTANT, identical in
every round and independent of eps_AI and eps_social. That makes it a
single reusable object -- but it also means a frozen cell computed on
different silicon carries a DIFFERENT constant. The archived A100 cell
pofdreach_qwen7b_k0_ea1_w0p5_l0p2_es0_s0 differs from the H100 frozen
prior in 17 of 723 agents (MAE .0091, max .5). Reusing it for the
k = .2, eps_social = 0 cell would put a different served vector into one
corner of a grid whose whole purpose is to compare k = .2 against k = 1,
so it is REFUSED and superseded by a hardware-matched rerun. Every frozen
cell -- reused or new -- must therefore

  * sit on an H100-80GB,
  * carry a prediction vector that is constant across all 30 rounds, and
  * share ONE canonical sha256 with every other frozen cell.

The canonical hash is DERIVED here (the unique hash of the H100 frozen
cells) rather than hard-coded, and the audit hard-fails if the H100 cells
disagree instead of relaxing to a majority vote.

Nothing is forced. The audit reports whatever split the archive holds and
fails loudly on a mismatch; it never edits an expectation to match what
it found.

Usage:
  python audit_qwen_mechanism.py [--roots R1 R2] [--print] [--write PATH]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

import torch

# remote capture pipes this file over ssh stdin (python - --print), where
# __file__ is unset: fall back to the pipelines cwd
HERE = (os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals() else os.getcwd())
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "audit_reach", os.path.join(HERE, "audit_sft_icl_reach_reuse.py"))
AR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AR)

# ONE sentinel instance. Loading the reach module twice would create two
# distinct object()s and every absent-field want would silently fail to
# match, marking already-completed cells "new".
ABSENT = AR.ABSENT

MODEL = "Qwen/Qwen2.5-7B-Instruct"
KS = [0.2, 1.0]
ESS = [0.0, 0.05, 0.2, 1.0]
ARMS = ["k0", "b0", "b1"]
GATE = 1.0            # numeric strict-< eps_AI, NOT all_open
SEED = 0
N_ROUNDS = 30
N_AGENTS = 723
H100_MARKER = "H100"
MANIFEST_PATH = os.path.join(
    REPO, "experiments", "condor", "manifest_qwen_mechanism.json")

SHARED = {
    "dataset": ("movielens",), "ml_target": ("Action",),
    "pop_model": ("ab",), "run_mode": ("loop",),
    "population_update": ("nested_ai_then_social_v1",),
    "w_plat": (0.5,), "gamma_bias": (0.0,),
    "platform_sus_scale": (1.0,), "anchor_mode": ("fixed",),
    "deploy_every": (1,), "data_regime": ("replace",),
    "n_rounds": (N_ROUNDS,), "do_sample": (False,),
    "n_labeled": (723,), "train_cap": (723,), "seed_base_data": (True,),
    "pristine_frac": (0.0,), "canary_delta": (ABSENT, 0.0),
    "replay_frac": (ABSENT, 0.0), "teacher_label_delta": (ABSENT, 0.0),
    "icrh": (ABSENT, False), "feedback_mode": (ABSENT, "none"),
    "pop_reset": (ABSENT, False), "ab_sweeps": (ABSENT, 1),
    "profile_shuffle_p": (ABSENT, 0.0), "profile_sort_q": (ABSENT, 0.0),
    "profile_drop_cols": (ABSENT, [], None),
    "profile_permute_cols": (ABSENT, [], None),
    "seed": (SEED,), "base_model": (MODEL,),
    "eps_ai": (GATE,),
    # both gates on their strict-< numeric threshold. ABSENT is the
    # pre-mode default for the AI gate (2026-08-13) and the pre-mode
    # default for the peer gate (2026-08-20); every archived run
    # predates the peer-side key.
    "ai_gate_mode": (ABSENT, "threshold"),
    "peer_gate_mode": (ABSENT, "threshold"),
    # Qwen2.5 has no hybrid-reasoning template; the key is written only
    # when CHAT_THINKING carries a directive, so it must be absent here
    "chat_thinking": (ABSENT,),
    "icl_k": (0,), "icl_days": (0,),
}
ARM_WANT = {
    # frozen zero-shot serving: nothing trains, so lora_r / sft_* are
    # inert and deliberately NOT matched
    "k0": {"training_style": ("frozen",), "kl_beta": (0.0,),
           "use_lora": (False, 0), "fresh_each_round": (False,),
           "icl_snapshot_round": (ABSENT, -1)},
    # ordinary SFT carries no KL term, so kl_direction is inert and NOT
    # matched (the reverse-era b0 runs are the established precedent)
    "b0": {"training_style": ("sft",), "kl_beta": (0.0,),
           "use_lora": (True, 1), "lora_r": (512,), "sft_lr": (5e-5,),
           "sft_epochs": (1,), "sft_batch_size": (4,),
           "fresh_each_round": (True,), "kl_ref_adapter": (ABSENT, "", None)},
    "b1": {"training_style": ("sft_kl",), "kl_beta": (1.0,),
           "kl_direction": ("forward",), "use_lora": (True, 1),
           "lora_r": (512,), "sft_lr": (5e-5,), "sft_epochs": (1,),
           "sft_batch_size": (4,), "fresh_each_round": (True,),
           "kl_ref_adapter": (ABSENT, "", None)},
}


def _num(v):
    return f"{v:g}".replace(".", "p")


def cell_want(arm, k, es):
    want = dict(SHARED)
    want.update(ARM_WANT[arm])
    want["innate_lambda"] = (k,)
    want["eps"] = (es,)
    return want


def new_tag(arm, k, es):
    """Collision-safe tag in the NEW pofdqmech_ family. The anchor rides
    the established _l<k>_ token (a bare _k1_ token would collide with the
    ICL-K grammar, where _k0_ already spells the frozen arm)."""
    return (f"pofdqmech_qwen7b_{arm}_ea{_num(GATE)}_w0p5_l{_num(k)}"
            f"_es{_num(es)}_s{SEED}")


def gpu_name(cfg):
    return ((cfg.get("hardware") or {}).get("gpu_name") or "")


def pred_facts(run_dir):
    """(ok, note, sha256, constant) for a run's served prediction vector."""
    pt = os.path.join(run_dir, "trajectory.pt")
    try:
        d = torch.load(pt, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"trajectory.pt unreadable: {exc}", None, None
    pr = d.get("pred_raw")
    if pr is None or pr.numel() == 0:
        return False, "pred_raw missing/empty", None, None
    const = bool((pr == pr[0]).all())
    sha = hashlib.sha256(
        pr[0].contiguous().numpy().tobytes()).hexdigest()
    return True, "ok", sha, const


def audit(roots):
    runs = AR.scan(roots)
    by_name = {name: root for name, root, _ in runs}
    cells, hazards, notes = [], [], []

    for arm in ARMS:
        for k in KS:
            for es in ESS:
                want = cell_want(arm, k, es)
                hits = []
                for name, root, cfg in runs:
                    fv = AR.field_verdict(cfg, want)
                    if all(ok for _, _, ok in fv.values()):
                        hits.append((name, root, cfg))
                cell = {"arm": arm, "innate_k": k, "eps_social": es,
                        "gate": GATE, "seed": SEED,
                        "new_tag": new_tag(arm, k, es)}
                picked, rejected = None, []
                # deterministic order: sorted by tag, so two runs of the
                # audit on the same archive pick the same cell
                for name, root, cfg in sorted(hits, key=lambda h: h[0]):
                    rd = os.path.join(root, name)
                    ok_c, note, arts = AR.completeness(rd)
                    if ok_c and es > 0 and not arts.get("twin_raw"):
                        ok_c, note = False, "twin_raw missing/empty"
                    if ok_c and arm == "k0":
                        # frozen cells carry the extra hardware + constant
                        # -prediction requirements described in the module
                        # docstring
                        gn = gpu_name(cfg)
                        ok_p, pnote, sha, const = pred_facts(rd)
                        if not ok_p:
                            ok_c, note = False, pnote
                        elif H100_MARKER not in gn:
                            ok_c, note = False, (
                                f"frozen cell on {gn or 'unknown GPU'}, not "
                                f"{H100_MARKER} -- superseded")
                        elif not const:
                            ok_c, note = False, (
                                "frozen predictions are NOT constant across "
                                "rounds")
                        else:
                            cell["pred_sha256"] = sha
                            cell["gpu_name"] = gn
                    if ok_c:
                        picked = (name, rd, note)
                        break
                    rejected.append({"run_tag": name, "why": note})
                if picked is None:
                    cell["status"] = "new"
                    if rejected:
                        cell["rejected_matches"] = rejected
                    if cell["new_tag"] in by_name:
                        hazards.append(cell["new_tag"])
                else:
                    name, rd, note = picked
                    cell.update({"status": "reused", "run_tag": name,
                                 "run_dir": rd, "verdict": "PASS",
                                 "note": note})
                    if rejected:
                        cell["rejected_matches"] = rejected
                    others = [h[0] for h in sorted(hits, key=lambda h: h[0])
                              if h[0] != name]
                    if others:
                        cell["extra_matches"] = others
                cells.append(cell)

    # ---- canonical frozen prediction hash -------------------------------
    # DERIVED, never hard-coded: the unique sha256 shared by every reused
    # H100 frozen cell. Disagreement is a hard failure, not a vote.
    frozen_sha = sorted({c["pred_sha256"] for c in cells
                         if c["arm"] == "k0" and c["status"] == "reused"
                         and c.get("pred_sha256")})
    canonical = None
    if len(frozen_sha) == 1:
        canonical = frozen_sha[0]
    elif len(frozen_sha) > 1:
        notes.append(
            "HARD FAIL: reused H100 frozen cells do NOT share one "
            f"prediction hash: {frozen_sha}. The k-comparison would be "
            "contaminated; stopping rather than relaxing the requirement.")
    n_reused = sum(1 for c in cells if c["status"] == "reused")
    return cells, n_reused, len(cells) - n_reused, hazards, canonical, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=AR.DEFAULT_ROOTS)
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--write", nargs="?", const=MANIFEST_PATH, default=None)
    args = ap.parse_args()

    cells, n_reused, n_new, hazards, canonical, notes = audit(args.roots)
    manifest = {
        "key": "qwen_mechanism_frozen",
        "model": MODEL, "ks": KS, "ess": ESS, "arms": ARMS,
        "gate": GATE, "seed": SEED, "n_rounds": N_ROUNDS,
        "n_gpu_cells": len(cells), "n_reused": n_reused, "n_new": n_new,
        "n_perfect_prediction_cells": len(KS) * len(ESS),
        "n_conceptual_cells": (len(ARMS) + 1) * len(KS) * len(ESS),
        "canonical_frozen_pred_sha256": canonical,
        "cells": cells,
    }
    for c in cells:
        line = (f"[audit_qmech] {c['arm']:<3} k{c['innate_k']:<4g} "
                f"es{c['eps_social']:<5g} -> {c['status']}")
        if c["status"] == "reused":
            line += f" ({c['run_tag']})"
        for r in c.get("rejected_matches", []):
            line += f"\n[audit_qmech]      rejected {r['run_tag']}: {r['why']}"
        print(line, file=sys.stderr)
    print(f"[audit_qmech] {n_reused} reused / {n_new} new of {len(cells)} "
          f"GPU cells ({manifest['n_conceptual_cells']} conceptual, "
          f"{manifest['n_perfect_prediction_cells']} of them CPU oracle)",
          file=sys.stderr)
    print(f"[audit_qmech] canonical frozen pred sha256: {canonical}",
          file=sys.stderr)
    for n in notes:
        print(f"[audit_qmech] {n}", file=sys.stderr)
    if hazards:
        print(f"[audit_qmech] HARD FAIL: new-cell tag(s) already occupied "
              f"by non-matching run dirs: {hazards}", file=sys.stderr)
    if notes or hazards:
        sys.exit(1)
    if args.do_print:
        print(json.dumps(manifest, indent=2))
    if args.write:
        with open(args.write, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[audit_qmech] wrote {args.write}", file=sys.stderr)


if __name__ == "__main__":
    main()
