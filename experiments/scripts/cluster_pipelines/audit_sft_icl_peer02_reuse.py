#!/usr/bin/env python3
"""Reuse audit for the eps_social=0.2 SFT/ICL channel table (2026-08-14).

Grid: 3 models x channels {b0, k0, fz0, dyn} x numeric AI gates
{0.1, 0.4} x eps_social=0.2 x seeds {0, 42, 43} = 72 conceptual cells,
every cell AI_GATE_MODE=threshold. Matching is by config fields + complete
30-round (30, 723) trajectories, never tag similarity; the field surface
is shared with the reach/k0 audits with eps (social) = 0.2 instead of 0.
No replay / pristine / teacher / profile / preference interventions.

Per manifest cell: model, channel, AI gate, social gate, seed, status
(reused|new), exact run tag, source root, CONFIG FINGERPRINT (sha256 over
the matched scientific fields' actual values), TRAJECTORY HASH (sha256
over op_raw+pred_raw bytes), and VALIDATION STATUS (check_pofd_sanity
verdict, executed per reused run at audit time).

Writes experiments/condor/manifest_sft_icl_peer02.json. With --expect-*
set, a count mismatch prints a DISCREPANCY report and exits 1 WITHOUT
writing.

Usage:
  python3 audit_sft_icl_peer02_reuse.py [--roots DIR ...] [--write]
      [--expect-reused N] [--expect-new N] [--skip-validate]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "audit_reach", os.path.join(HERE, "audit_sft_icl_reach_reuse.py"))
AR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AR)
_spec_k0 = importlib.util.spec_from_file_location(
    "audit_k0", os.path.join(HERE, "audit_sft_k0_nopeer_reuse.py"))
AK = importlib.util.module_from_spec(_spec_k0)
_spec_k0.loader.exec_module(AK)

REPO = AR.REPO
MANIFEST = os.path.join(REPO, "experiments", "condor",
                        "manifest_sft_icl_peer02.json")
CHECKER = os.path.join(HERE, "check_pofd_sanity.py")
ARMS = ["b0", "k0", "fz0", "dyn"]
GATES = [0.1, 0.4]
SEEDS = [0, 42, 43]
EPS_SOCIAL = 0.2


def _num(v):
    return f"{v:g}".replace(".", "p")


def new_tag(model, arm, gate, seed):
    return (f"pofdpeer2_{model}_{arm}_ea{_num(gate)}_w0p5_l0p2_es0p2"
            f"_s{seed}")


def cell_want(model, arm, gate, seed):
    want = dict(AR.SHARED_WANT)
    want.update(AK.ARM_WANT_K0 if arm == "k0" else AR.ARM_WANT[arm])
    want["base_model"] = (AR.MODELS[model],)
    want["seed"] = (seed,)
    want["eps"] = (EPS_SOCIAL,)          # the peer step is LIVE here
    want["eps_ai"] = (gate,)
    want["ai_gate_mode"] = (AR.ABSENT, "threshold")
    return want


def fingerprint(cfg, want):
    vals = {k: (cfg.get(k, "<ABSENT>")) for k in sorted(want)}
    return hashlib.sha256(
        json.dumps(vals, sort_keys=True, default=str).encode()).hexdigest()


def traj_hash(run_dir):
    d = torch.load(os.path.join(run_dir, "trajectory.pt"),
                   map_location="cpu", weights_only=False)
    h = hashlib.sha256()
    h.update(d["op_raw"].float().contiguous().numpy().tobytes())
    h.update(d["pred_raw"].float().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate(run_dir):
    p = subprocess.run([sys.executable, CHECKER, run_dir],
                       capture_output=True, text=True,
                       env={**os.environ, "USE_TF": "0"})
    return "PASS" if p.returncode == 0 else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=AR.DEFAULT_ROOTS)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--expect-reused", type=int, default=None)
    ap.add_argument("--expect-new", type=int, default=None)
    ap.add_argument("--skip-validate", action="store_true",
                    help="skip the per-reused-run checker execution")
    args = ap.parse_args()

    corpus = AR.scan(args.roots)
    print(f"[audit] scanned {len(corpus)} run dirs")

    cells, near_misses = [], []
    for model in AR.MODELS:
        for arm in ARMS:
            for gate in GATES:
                for seed in SEEDS:
                    want = cell_want(model, arm, gate, seed)
                    matches = []
                    for name, root, cfg in corpus:
                        fv = AR.field_verdict(cfg, want)
                        bad = {k: v for k, v in fv.items() if not v[2]}
                        if not bad:
                            matches.append((name, root, cfg))
                        elif (len(bad) <= 2 and cfg.get("base_model") ==
                                AR.MODELS[model]
                                and cfg.get("seed") == seed
                                and abs(cfg.get("eps", -1.0)
                                        - EPS_SOCIAL) < 1e-9
                                and not {"eps_ai", "kl_beta",
                                         "training_style",
                                         "icl_k"} & bad.keys()):
                            near_misses.append(
                                (model, arm, gate, seed, name,
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
                                (model, arm, gate, seed, name,
                                 {"completeness": (note, "complete")}))
                    verified.sort(key=lambda x: (-x[0], x[1]))
                    cell = {"model": model, "arm": arm, "gate": gate,
                            "eps_social": EPS_SOCIAL, "seed": seed}
                    if verified:
                        _, name, root, cfg, arts = verified[0]
                        rd = os.path.join(root, name)
                        cell.update({
                            "status": "reused", "run_tag": name,
                            "source_root": os.path.relpath(root, REPO),
                            "artifacts": arts,
                            "config_fingerprint": fingerprint(cfg, want),
                            "trajectory_sha256": traj_hash(rd),
                            "validation": ("SKIPPED" if args.skip_validate
                                           else validate(rd)),
                            "recorded": {k: cfg.get(k)
                                         for k in AR.RECORDED_ONLY},
                            "alternates": [v[1] for v in verified[1:]]})
                    else:
                        cell.update({"status": "new",
                                     "run_tag": new_tag(model, arm, gate,
                                                        seed),
                                     "config_fingerprint": None,
                                     "trajectory_sha256": None,
                                     "validation": "PENDING"})
                    cells.append(cell)

    reused = [c for c in cells if c["status"] == "reused"]
    new = [c for c in cells if c["status"] == "new"]

    print(f"\n== reused cells ({len(reused)}) ==")
    for c in reused:
        a = c["artifacts"]
        flags = "".join(k[0].upper() if v else k[0]
                        for k, v in (("gate_raw", a["gate_raw"]),
                                     ("twin_raw", a["twin_raw"]),
                                     ("icl_idx_raw", a["icl_idx_raw"])))
        print(f"  {c['model']:9s} {c['arm']:3s} ea={c['gate']!s:4s} "
              f"s{c['seed']:<2d} <- {c['run_tag']}  [{flags}] "
              f"val={c['validation']}")
    if near_misses:
        print(f"\n== near-miss candidates rejected ({len(near_misses)}) ==")
        for model, arm, gate, seed, name, bad in near_misses[:25]:
            print(f"  {model} {arm} ea={gate} s{seed}: {name} -> {bad}")

    per_model_r = {m: sum(1 for c in reused if c["model"] == m)
                   for m in AR.MODELS}
    per_model_n = {m: sum(1 for c in new if c["model"] == m)
                   for m in AR.MODELS}
    per_arm_n = {a: sum(1 for c in new if c["arm"] == a) for a in ARMS}
    print(f"\n== counts ==")
    print(f"  conceptual cells: {len(cells)}")
    print(f"  reused: {len(reused)}  per model: {per_model_r}")
    print(f"  new: {len(new)}  per model: {per_model_n}")
    print(f"  new per arm: {per_arm_n}")
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
        "wave": "sft_icl_peer02",
        "audited": "2026-08-14 local corpora, by config fields + 30-round "
                   "completeness; per-reused-cell config fingerprint, "
                   "trajectory sha256 (op_raw+pred_raw bytes), and "
                   "check_pofd_sanity verdict recorded at audit time.",
        "roots": [os.path.relpath(r, REPO) for r in args.roots],
        "grid": {"models": AR.MODELS, "arms": ARMS, "gates": GATES,
                 "eps_social": EPS_SOCIAL, "seeds": SEEDS,
                 "n_agents": AR.N_AGENTS, "n_rounds": AR.N_ROUNDS},
        "counts": {"cells": len(cells), "reused": len(reused),
                   "new": len(new), "reused_per_model": per_model_r,
                   "new_per_model": per_model_n,
                   "new_per_arm": per_arm_n},
        "cells": cells,
        # the common plain-prompting base-map cohort comes from the SAME
        # seed-0 pofdreachbase_ probes the no-peer grids use (innate +
        # m_base are bit-identical across seeds)
        "baselines": [{"model": m, "seed": 0, "status": "reused",
                       "run_tag": AR.base_tag(m, 0)} for m in AR.MODELS],
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
