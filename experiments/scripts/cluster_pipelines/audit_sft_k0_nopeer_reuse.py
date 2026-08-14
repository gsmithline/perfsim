#!/usr/bin/env python3
"""Reuse audit for the sft_k0_nopeer wave (2026-08-14).

Question: with peers off, does SFT transmit population information globally
through shared weights while frozen NO-context prompting (k0 = repeated
zero-shot serving; NOT adaptive ICL) acts only as a fixed external signal?

Grid: 3 models x arms {b0, b1, k0} x numeric gates
{0.05, 0.1, 0.2, 0.4, 0.7, 1.0} x seed 0 = 54 conceptual cells, every cell
AI_GATE_MODE=threshold (the 1.0 gate is the STRICT numeric threshold,
deliberately distinct from all_open -- completed _eaopen_ scout cells are
NOT reusable for _ea1_). Matching is by config fields + complete 30-round
trajectories, never tag similarity; the field surface is shared with the
reach audit (audit_sft_icl_reach_reuse) -- b0/b1 identical, k0 = frozen +
USE_LORA=0 + ICL_K=0 + ICL_DAYS=0 (icl_select/ctx_source/snapshot are
inert at K=0 and recorded, not matched). Feature/teacher/replay/pristine/
legacy-operator/altered-profile runs are excluded by those same fields.

Writes experiments/condor/manifest_sft_k0_nopeer.json (same schema as the
reach manifest; baselines = the three completed seed-0 pofdreachbase_
probes, reused). With --expect-* set, a count mismatch prints a DISCREPANCY
report and exits 1 WITHOUT writing.

Usage:
  python3 audit_sft_k0_nopeer_reuse.py [--roots DIR ...] [--write]
      [--expect-reused N] [--expect-new N]
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

REPO = AR.REPO
MANIFEST = os.path.join(REPO, "experiments", "condor",
                        "manifest_sft_k0_nopeer.json")
ARMS = ["b0", "b1", "k0"]
GATES = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
SEED = 0
ARM_WANT_K0 = {
    "training_style": ("frozen",), "kl_beta": (0.0,),
    "use_lora": (False, 0), "fresh_each_round": (False,),
    "icl_k": (0,), "icl_days": (0,),
}


def _num(v):
    return f"{v:g}".replace(".", "p")


def new_tag(model, arm, gate):
    # the SAME family/grammar as the reach wave, so pending b0/b1 cells
    # shared with the full reach grid carry BYTE-IDENTICAL tags and a
    # later full reach submission no-ops them
    return f"pofdreach_{model}_{arm}_ea{_num(gate)}_w0p5_l0p2_es0_s{SEED}"


def cell_want(model, arm, gate):
    want = dict(AR.SHARED_WANT)
    want.update(ARM_WANT_K0 if arm == "k0" else AR.ARM_WANT[arm])
    want["base_model"] = (AR.MODELS[model],)
    want["seed"] = (SEED,)
    want["eps_ai"] = (gate,)
    # numeric threshold ONLY -- an all_open config can never satisfy this,
    # so the completed _eaopen_ scout cells are structurally excluded
    want["ai_gate_mode"] = (AR.ABSENT, "threshold")
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=AR.DEFAULT_ROOTS)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--expect-reused", type=int, default=None)
    ap.add_argument("--expect-new", type=int, default=None)
    args = ap.parse_args()

    corpus = AR.scan(args.roots)
    print(f"[audit] scanned {len(corpus)} run dirs under {len(args.roots)} "
          f"root(s)")

    cells, near_misses = [], []
    for model in AR.MODELS:
        for arm in ARMS:
            for gate in GATES:
                want = cell_want(model, arm, gate)
                matches = []
                for name, root, cfg in corpus:
                    fv = AR.field_verdict(cfg, want)
                    bad = {k: v for k, v in fv.items() if not v[2]}
                    if not bad:
                        matches.append((name, root, cfg))
                    elif (len(bad) <= 2 and cfg.get("base_model") ==
                            AR.MODELS[model] and cfg.get("seed") == SEED
                            and not cfg.get("eps", 1.0)
                            and not {"eps_ai", "kl_beta", "training_style",
                                     "icl_k"} & bad.keys()):
                        near_misses.append(
                            (model, arm, gate, name,
                             {k: (v[1] if v[1] is not AR.ABSENT
                                  else "<ABSENT>", v[0])
                              for k, v in bad.items()}))
                verified = []
                for name, root, cfg in matches:
                    ok, note, arts = AR.completeness(
                        os.path.join(root, name))
                    if ok:
                        verified.append((sum(arts.values()), name, root,
                                         cfg, arts))
                    else:
                        near_misses.append((model, arm, gate, name,
                                            {"completeness": (note,
                                                              "complete")}))
                verified.sort(key=lambda x: (-x[0], x[1]))
                cell = {"model": model, "arm": arm, "gate": gate,
                        "seed": SEED}
                if verified:
                    _, name, root, cfg, arts = verified[0]
                    cell.update({"status": "reused", "run_tag": name,
                                 "source_root": os.path.relpath(root, REPO),
                                 "artifacts": arts,
                                 "recorded": {k: cfg.get(k) for k in
                                              AR.RECORDED_ONLY},
                                 "alternates": [v[1] for v in
                                                verified[1:]]})
                else:
                    cell.update({"status": "new",
                                 "run_tag": new_tag(model, arm, gate)})
                cells.append(cell)

    reused = [c for c in cells if c["status"] == "reused"]
    new = [c for c in cells if c["status"] == "new"]

    print("\n== matched fields per arm ==")
    for arm in ARMS:
        want = dict(AR.SHARED_WANT)
        want.update(ARM_WANT_K0 if arm == "k0" else AR.ARM_WANT[arm])
        parts = [f"{k}=" + "|".join("ABSENT" if v is AR.ABSENT else repr(v)
                                    for v in want[k]) for k in sorted(want)]
        print(f"  {arm}: " + "  ".join(parts))
    print("  (+ base_model per slug, seed=0, eps_ai per numeric gate, "
          "ai_gate_mode threshold-or-absent everywhere)")

    print(f"\n== reused cells ({len(reused)}) ==")
    for c in reused:
        a = c["artifacts"]
        flags = "".join(k[0].upper() if v else k[0]
                        for k, v in (("gate_raw", a["gate_raw"]),
                                     ("twin_raw", a["twin_raw"]),
                                     ("icl_idx_raw", a["icl_idx_raw"])))
        print(f"  {c['model']:9s} {c['arm']:3s} ea={c['gate']!s:5s} "
              f"<- {c['run_tag']}  [{flags}]"
              + (f"  (+{len(c['alternates'])} alt)" if c["alternates"]
                 else ""))
    if near_misses:
        print(f"\n== near-miss candidates rejected ({len(near_misses)}) ==")
        for model, arm, gate, name, bad in near_misses[:30]:
            print(f"  {model} {arm} ea={gate}: {name} -> {bad}")

    per_model = {m: sum(1 for c in new if c["model"] == m)
                 for m in AR.MODELS}
    per_arm = {a: sum(1 for c in new if c["arm"] == a) for a in ARMS}
    per_gate = {g: sum(1 for c in new if c["gate"] == g) for g in GATES}
    print(f"\n== counts ==")
    print(f"  conceptual cells: {len(cells)}")
    print(f"  reused: {len(reused)}   new: {len(new)}")
    print(f"  new per model: {per_model}")
    print(f"  new per arm:   {per_arm}")
    print(f"  new per gate:  {per_gate}")

    bad_expect = []
    if args.expect_reused is not None and len(reused) != args.expect_reused:
        bad_expect.append(f"reused {len(reused)} != {args.expect_reused}")
    if args.expect_new is not None and len(new) != args.expect_new:
        bad_expect.append(f"new {len(new)} != {args.expect_new}")
    if bad_expect:
        print("\nDISCREPANCY: " + "; ".join(bad_expect))
        print("Manifest NOT written; inspect the tables above. Never force "
              "reuse on tag similarity.")
        sys.exit(1)

    manifest = {
        "wave": "sft_k0_nopeer",
        "audited": "2026-08-14 local corpora, by config fields + 30-round "
                   "completeness (the record gen_pofd_sweep.py hard-asserts "
                   "against). k0 = frozen no-context prompting; _ea1_ is "
                   "the strict numeric threshold, never all_open.",
        "roots": [os.path.relpath(r, REPO) for r in args.roots],
        "grid": {"models": AR.MODELS, "arms": ARMS, "gates": GATES,
                 "seeds": [SEED], "n_agents": AR.N_AGENTS,
                 "n_rounds": AR.N_ROUNDS},
        "counts": {"cells": len(cells), "reused": len(reused),
                   "new": len(new), "new_per_model": per_model,
                   "new_per_arm": per_arm,
                   "new_per_gate": {str(k): v for k, v in per_gate.items()},
                   "baselines": len(AR.MODELS)},
        "cells": cells,
        # the completed seed-0 reach baseline probes are REUSED as the
        # common-cohort reference; nothing new queues for them
        "baselines": [{"model": m, "seed": SEED, "status": "reused",
                       "run_tag": AR.base_tag(m, SEED)}
                      for m in AR.MODELS],
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
