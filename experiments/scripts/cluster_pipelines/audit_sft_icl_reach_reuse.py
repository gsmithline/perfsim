#!/usr/bin/env python3
"""Reuse audit for the SFT-ICL reach wave (2026-08-13).

Scans local run corpora and maps every one of the 360 conceptual main cells

    3 models {qwen7b, olmo7b, mistral7b}
  x 4 arms   {b0 (sft), b1 (forward SFT-KL), fz0 (frozen round-0 K=8 ICL),
              dyn (live refreshed K=8 ICL)}
  x 6 gates  {0.05, 0.1, 0.2, 0.4, 0.7, open}
  x 5 seeds  {0, 42, 43, 44, 45}

to either an EXACT reusable existing run or a new pofdreach_ job. Matching is
BY CONFIG FIELDS AND TRAJECTORY COMPLETENESS, never by tag similarity: a
candidate must equal the reach wave's dial surface on every dynamics-
determining field (dataset/target/base model/operator/W/lam/gamma/eps/eps_ai/
seed/rounds/regime/fresh/style/beta/direction-at-b1/LoRA-512 budget for the
trained arms; K/select/source/snapshot for the frozen-context arms; greedy
serving; no replay/pristine/teacher/profile knobs anywhere) and hold a
complete 30-round trajectory with [30, 723] op_raw/pred_raw. Telemetry-only
fields (log_gender_gaps, ans_sample_k, ppl logging, wandb) are RECORDED but
not matched -- they draw no RNG the loop consumes and touch no dynamics.

Artifact availability is recorded per reused run: gate_raw (absent in runs
written before 2026-08-05 -- it reconstructs bit-exactly offline as
|clamp(pred,0,1) - x(t)| < eps_ai on the start-of-round opinion), twin_raw
(absent in the pre-WITH_TWIN es=0 runs -- at es=0 the no-platform twin is
innate to 1 float32 ulp, the esf precedent), and icl_idx_raw (absent in
icl2-era runs, which predate the exemplar logs).

Writes experiments/condor/manifest_sft_icl_reach.json (the machine-readable
cell map gen_pofd_sweep.py consumes) and prints the field-by-field reuse
table. With --expect-reused/--expect-new set, a count mismatch prints a
DISCREPANCY report and exits 1 WITHOUT writing the manifest.

Usage:
  python3 audit_sft_icl_reach_reuse.py [--roots DIR ...] [--write]
      [--expect-reused N] [--expect-new N]
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_ROOTS = [os.path.join(REPO, "runs", "pokec_gated_lm"),
                 os.path.join(REPO, "notes", "pofd", "cluster")]
MANIFEST = os.path.join(REPO, "experiments", "condor",
                        "manifest_sft_icl_reach.json")

MODELS = {"qwen7b": "Qwen/Qwen2.5-7B-Instruct",
          "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
          "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3"}
ARMS = ["b0", "b1", "fz0", "dyn"]
GATES = [0.05, 0.1, 0.2, 0.4, 0.7, "open"]
SEEDS = [0, 42, 43, 44, 45]
N_AGENTS = 723
N_ROUNDS = 30

# sentinel: "field may be absent OR hold one of these values" (older configs
# predate some keys; absent == the runner default the run actually executed)
ABSENT = object()

SHARED_WANT = {
    "dataset": ("movielens",), "ml_target": ("Action",),
    "pop_model": ("ab",), "run_mode": ("loop",),
    "population_update": ("nested_ai_then_social_v1",),
    "eps": (0.0,), "gamma_bias": (0.0,), "w_plat": (0.5,),
    "innate_lambda": (0.2,), "platform_sus_scale": (1.0,),
    "anchor_mode": ("fixed",), "deploy_every": (1,),
    "data_regime": ("replace",), "n_rounds": (30,), "do_sample": (False,),
    "canary_delta": (0.0,), "pop_reset": (False,), "ab_sweeps": (1,),
    "n_labeled": (723,), "train_cap": (723,), "seed_base_data": (True,),
    "pristine_frac": (0.0,), "replay_frac": (ABSENT, 0.0),
    "teacher_label_delta": (ABSENT, 0.0),
    "kl_ref_adapter": (ABSENT, "", None),
    "icrh": (ABSENT, False), "feedback_mode": (ABSENT, "none"),
    "profile_shuffle_p": (ABSENT, 0.0), "profile_sort_q": (ABSENT, 0.0),
    "profile_drop_cols": (ABSENT, [], None),
    "profile_permute_cols": (ABSENT, [], None),
    # numeric-gate cells: threshold mode (absent = pre-mode configs, which
    # ran the identical strict-< gate). The "open" gate requires an explicit
    # all_open config -- no archived run has one, by construction.
    "ai_gate_mode": (ABSENT, "threshold"),
}
ARM_WANT = {
    "b0": {"training_style": ("sft",), "kl_beta": (0.0,),
           "use_lora": (True, 1), "lora_r": (512,), "sft_lr": (5e-5,),
           "sft_epochs": (1,), "sft_batch_size": (4,),
           "fresh_each_round": (True,), "icl_k": (0,), "icl_days": (0,)},
    # b0 has no KL term -- kl_direction is inert and NOT matched (the
    # reverse-era pofdw2_ b0 runs are the fes/cube reuse precedent)
    "b1": {"training_style": ("sft_kl",), "kl_beta": (1.0,),
           "kl_direction": ("forward",), "use_lora": (True, 1),
           "lora_r": (512,), "sft_lr": (5e-5,), "sft_epochs": (1,),
           "sft_batch_size": (4,), "fresh_each_round": (True,),
           "icl_k": (0,), "icl_days": (0,)},
    # frozen arms: nothing trains, so lora_r/sft_* are inert and unmatched
    "fz0": {"training_style": ("frozen",), "kl_beta": (0.0,),
            "use_lora": (False, 0), "fresh_each_round": (False,),
            "icl_k": (8,), "icl_days": (0,), "icl_select": ("random",),
            "icl_ctx_source": ("live",), "icl_snapshot_round": (0,)},
    "dyn": {"training_style": ("frozen",), "kl_beta": (0.0,),
            "use_lora": (False, 0), "fresh_each_round": (False,),
            "icl_k": (8,), "icl_days": (0,), "icl_select": ("random",),
            "icl_ctx_source": ("live",),
            "icl_snapshot_round": (ABSENT, -1)},
}
RECORDED_ONLY = ["log_gender_gaps", "ans_sample_k", "log_ppl_dist", "host"]


def _num(v):
    s = f"{v:g}"
    return s.replace(".", "p")


def gate_tok(gate):
    return "eaopen" if gate == "open" else f"ea{_num(gate)}"


def reach_tag(model, arm, gate, seed):
    return f"pofdreach_{model}_{arm}_{gate_tok(gate)}_w0p5_l0p2_es0_s{seed}"


def base_tag(model, seed):
    return f"pofdreachbase_{model}_w0p5_l0p2_es0_s{seed}"


def field_verdict(cfg, want):
    """{field: (expected_tuple, got, ok)} for every matched field."""
    out = {}
    for k, allowed in want.items():
        got = cfg.get(k, ABSENT)
        ok = any((got is ABSENT and a is ABSENT)
                 or (got is not ABSENT and a is not ABSENT and got == a)
                 for a in allowed)
        out[k] = (allowed, got, ok)
    return out


def cell_want(model, arm, gate, seed):
    want = dict(SHARED_WANT)
    want.update(ARM_WANT[arm])
    want["base_model"] = (MODELS[model],)
    want["seed"] = (seed,)
    if gate == "open":
        want["ai_gate_mode"] = ("all_open",)
        # eps_ai is inert under all_open; not matched
    else:
        want["eps_ai"] = (gate,)
    return want


def scan(roots):
    """(dirname, root, cfg) for every run dir holding config.json."""
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            cj = os.path.join(root, name, "config.json")
            if os.path.exists(cj):
                try:
                    out.append((name, root, json.load(open(cj))))
                except (json.JSONDecodeError, OSError):
                    pass
    return out


def completeness(run_dir):
    """(ok, notes, artifacts) -- loads trajectory.pt, checks 30x723."""
    pt = os.path.join(run_dir, "trajectory.pt")
    if not os.path.exists(pt):
        return False, "no trajectory.pt", {}
    d = torch.load(pt, map_location="cpu", weights_only=False)
    traj = d.get("trajectory", [])
    op = d.get("op_raw")
    pr = d.get("pred_raw")
    if len(traj) != N_ROUNDS:
        return False, f"{len(traj)} trajectory rounds != {N_ROUNDS}", {}
    for key, t in (("op_raw", op), ("pred_raw", pr)):
        if t is None or tuple(t.shape) != (N_ROUNDS, N_AGENTS):
            return False, (f"{key} shape "
                           f"{None if t is None else tuple(t.shape)} != "
                           f"{(N_ROUNDS, N_AGENTS)}"), {}
    arts = {}
    for key in ("gate_raw", "twin_raw", "icl_idx_raw"):
        t = d.get(key)
        arts[key] = bool(t is not None and t.numel() > 0)
    return True, "complete", arts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--write", action="store_true",
                    help="write the manifest (default: dry-run print only)")
    ap.add_argument("--expect-reused", type=int, default=None)
    ap.add_argument("--expect-new", type=int, default=None)
    args = ap.parse_args()

    corpus = scan(args.roots)
    print(f"[audit] scanned {len(corpus)} run dirs under {len(args.roots)} "
          f"root(s)")

    cells = []
    near_misses = []
    for model in MODELS:
        for arm in ARMS:
            for gate in GATES:
                for seed in SEEDS:
                    want = cell_want(model, arm, gate, seed)
                    matches = []
                    for name, root, cfg in corpus:
                        fv = field_verdict(cfg, want)
                        bad = {k: v for k, v in fv.items() if not v[2]}
                        if not bad:
                            matches.append((name, root, cfg, fv))
                        elif (len(bad) <= 2 and cfg.get("base_model") ==
                                MODELS[model] and cfg.get("seed") == seed
                                and not cfg.get("eps", 1.0)
                                # a different dose/beta/style is a DIFFERENT
                                # cell, not a near miss of this one
                                and not {"eps_ai", "kl_beta",
                                         "training_style",
                                         "icl_k"} & bad.keys()):
                            near_misses.append(
                                (model, arm, gate, seed, name,
                                 {k: (v[1] if v[1] is not ABSENT
                                      else "<ABSENT>", v[0])
                                  for k, v in bad.items()}))
                    verified = []
                    for name, root, cfg, fv in matches:
                        ok, note, arts = completeness(
                            os.path.join(root, name))
                        if ok:
                            score = sum(arts.values())
                            verified.append((score, name, root, cfg, arts))
                        else:
                            near_misses.append(
                                (model, arm, gate, seed, name,
                                 {"completeness": (note, "complete")}))
                    verified.sort(key=lambda x: (-x[0], x[1]))
                    cell = {"model": model, "arm": arm,
                            "gate": gate if gate == "open" else float(gate),
                            "seed": seed}
                    if verified:
                        score, name, root, cfg, arts = verified[0]
                        cell.update({
                            "status": "reused", "run_tag": name,
                            "source_root": os.path.relpath(root, REPO),
                            "artifacts": arts,
                            "recorded": {k: cfg.get(k) for k in
                                         RECORDED_ONLY},
                            "alternates": [v[1] for v in verified[1:]],
                        })
                    else:
                        cell.update({"status": "new",
                                     "run_tag": reach_tag(model, arm, gate,
                                                          seed)})
                    cells.append(cell)

    reused = [c for c in cells if c["status"] == "reused"]
    new = [c for c in cells if c["status"] == "new"]

    # ---- field-by-field reuse table ----------------------------------------
    print("\n== matched fields per arm (expected values; ABSENT = key may "
          "predate the config surface) ==")
    for arm in ARMS:
        want = dict(SHARED_WANT)
        want.update(ARM_WANT[arm])
        parts = []
        for k in sorted(want):
            vals = "|".join("ABSENT" if v is ABSENT else repr(v)
                            for v in want[k])
            parts.append(f"{k}={vals}")
        print(f"  {arm}: " + "  ".join(parts))
    print("  (+ base_model per model slug, seed per cell, eps_ai per numeric "
          "gate; 'open' requires ai_gate_mode=all_open explicitly)")
    print("  recorded-not-matched (telemetry only): "
          + ", ".join(RECORDED_ONLY))

    print(f"\n== reused cells ({len(reused)}) ==")
    for c in reused:
        a = c["artifacts"]
        flags = "".join(k[0].upper() if v else k[0]
                        for k, v in (("gate_raw", a["gate_raw"]),
                                     ("twin_raw", a["twin_raw"]),
                                     ("icl_idx_raw", a["icl_idx_raw"])))
        alt = f"  (+{len(c['alternates'])} alt)" if c["alternates"] else ""
        print(f"  {c['model']:9s} {c['arm']:3s} ea={c['gate']!s:5s} "
              f"s{c['seed']:<2d} <- {c['run_tag']}  [{flags}]{alt}")
    print("  artifact flags: G/T/I present, g/t/i absent "
          "(gate_raw / twin_raw / icl_idx_raw)")

    if near_misses:
        print(f"\n== near-miss candidates rejected ({len(near_misses)}; "
              f"<=2 fields off or incomplete) ==")
        for model, arm, gate, seed, name, bad in near_misses[:40]:
            print(f"  {model} {arm} ea={gate} s{seed}: {name} -> {bad}")

    per_model = {m: sum(1 for c in new if c["model"] == m) for m in MODELS}
    per_arm = {a: sum(1 for c in new if c["arm"] == a) for a in ARMS}
    print(f"\n== counts ==")
    print(f"  conceptual main cells: {len(cells)}")
    print(f"  reused: {len(reused)}   new: {len(new)}")
    print(f"  new per model: {per_model}")
    print(f"  new per arm:   {per_arm}")
    print(f"  baseline probes: {len(MODELS) * len(SEEDS)}")

    bad_expect = []
    if args.expect_reused is not None and len(reused) != args.expect_reused:
        bad_expect.append(f"reused {len(reused)} != expected "
                          f"{args.expect_reused}")
    if args.expect_new is not None and len(new) != args.expect_new:
        bad_expect.append(f"new {len(new)} != expected {args.expect_new}")
    if bad_expect:
        print("\nDISCREPANCY: " + "; ".join(bad_expect))
        print("Manifest NOT written. Inspect the reuse table and near-miss "
              "report above; never force reuse on tag similarity.")
        sys.exit(1)

    manifest = {
        "wave": "sft_icl_reach",
        "audited": "2026-08-13 local corpora, by config fields + trajectory "
                   "completeness (this file is the audit record "
                   "gen_pofd_sweep.py hard-asserts against)",
        "roots": [os.path.relpath(r, REPO) for r in args.roots],
        "grid": {"models": MODELS, "arms": ARMS,
                 "gates": [g if g == "open" else float(g) for g in GATES],
                 "seeds": SEEDS, "n_agents": N_AGENTS, "n_rounds": N_ROUNDS},
        "counts": {"cells": len(cells), "reused": len(reused),
                   "new": len(new), "new_per_model": per_model,
                   "new_per_arm": per_arm,
                   "baselines": len(MODELS) * len(SEEDS)},
        "cells": cells,
        "baselines": [{"model": m, "seed": s, "run_tag": base_tag(m, s)}
                      for m in MODELS for s in SEEDS],
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
