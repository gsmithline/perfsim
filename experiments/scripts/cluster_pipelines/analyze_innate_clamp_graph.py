#!/usr/bin/env python3
"""Innate-clamp GRAPH-PLACEMENT analysis (2026-08-17,
mistral_innate_clamp_graph_s0 + mistral_innate_clamp_graph_d8).

THE TEST: with 145 agents permanently pinned to innate and the peer
step LIVE under the one-sided STUBBORN operator, how does the fixed
cohort's PLACEMENT -- graph_clumped (one connected low-cut block,
41.5% responsive exposure) vs graph_scattered (distributed, 100%
exposure, 2.9x the cut edges) -- reshape what ordinary SFT (b0) and
the ICL arm do to the responsive population as the social gate opens?
The two masks are matched on innate and degree distributions
(identical joint-stratum quotas), so SCATTERED-minus-CLUMPED at a
matched (arm, ea, es) isolates the placement effect.

TWO ICL ARMS, TWO ANALYSES (--icl-arm, never mixed):
  d8 (DEFAULT, the corrected comparison): personal-history ICL --
     frozen weights, ICL_K=0, ICL_DAYS=8, each agent sees only its
     OWN latest eight opinions. The old cross-user _dyn_ runs are
     EXCLUDED. Coverage of all 96 conceptual cells (48 b0 + 48 d8)
     is HARD-ASSERTED before any output is written. Outputs go to
     notes/pofd/clamp_graph_d8_analysis/ (the old K8 analysis dir is
     never overwritten).
  dyn (legacy): the original cross-user live K=8 arm; lenient
     coverage, outputs to notes/pofd/clamp_graph_analysis/ exactly as
     before.

Read-only, descriptive, SEED 0 ONLY (no confidence intervals). All
baselines are IN-WAVE: es=0 cells run the same masks (the old no-peer
cohorts are different masks and are NOT baselines here).

Per cell (rounds 25-29 window + final round): responsive population SD
vs the matched twin's responsive SD, responsive displacement from the
twin, responsive AI acceptance + cumulative AI reach, responsive-to-
fixed mean and quantile-W1 gaps with normalized closure, and total
cut (fixed-responsive) pairs sampled/accepted plus the final fraction
of responsive agents reached through the cut.

Outputs: clamp_graph_per_cell.csv,
clamp_graph_placement_contrast.csv (scattered minus clumped),
clamp_graph_arm_contrast.csv (ICL arm minus b0 at matched cells).
"""
import argparse
import csv
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "analyze_reach", os.path.join(HERE, "analyze_sft_icl_reach.py"))
AN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN)
_spec_c = importlib.util.spec_from_file_location(
    "analyze_clamp", os.path.join(HERE, "analyze_innate_clamp.py"))
AC = importlib.util.module_from_spec(_spec_c)
_spec_c.loader.exec_module(AC)

NA = "NA"
MASKS = {"gclump": "graph_clumped", "gscat": "graph_scattered"}
GATES = [0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
OUT_DIR_OF_ARM = {
    "d8": os.path.join(REPO, "notes", "pofd", "clamp_graph_d8_analysis"),
    "dyn": os.path.join(REPO, "notes", "pofd", "clamp_graph_analysis"),
}
CONTRAST_METRICS = ["resp_std_ratio_late", "resp_std_ratio_final",
                    "resp_disp_twin_late", "gap_mean_late",
                    "gap_w1_late", "gap_mean_closure",
                    "fr_reach_final"]


def _num(v):
    return f"{v:g}".replace(".", "p")


def cell_tag(arm, gtok, gate, es):
    return (f"pofdclamp_mistral7b_{arm}_{gtok}_stub_ea{_num(gate)}"
            f"_w0p5_l0p2_es{_num(es)}_s0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--icl-arm", choices=["d8", "dyn"], default="d8",
                    help="ICL arm compared against b0: d8 = the "
                         "corrected personal-history wave (default, "
                         "hard-asserts full 96-cell coverage), dyn = "
                         "the legacy cross-user K=8 analysis")
    ap.add_argument("--out-dir", default=None,
                    help="default: per-arm directory, so the corrected "
                         "d8 analysis never overwrites the old K8 one")
    args = ap.parse_args()
    arms = ["b0", args.icl_arm]
    out_dir = args.out_dir or OUT_DIR_OF_ARM[args.icl_arm]

    per_cell, missing = [], []
    for gtok in MASKS:
        for arm in arms:
            for gate in GATES:
                for es in ESS:
                    tag = cell_tag(arm, gtok, gate, es)
                    rd = AN.find_run(args.roots, tag)
                    if rd is None:
                        missing.append(tag)
                        continue
                    # gap metrics computed for BOTH masks: the graph
                    # cohorts are innate-matched, so gap closure is
                    # meaningful in each (the innate gap is small but
                    # the placement question is about the cut)
                    row = {"mask": MASKS[gtok], "arm": arm,
                           "gate": gate, "eps_social": es,
                           "run_tag": tag,
                           **AC.cell_metrics(rd, "bottom")}
                    traj = AN.load(rd)["trajectory"]
                    row["fr_sampled_total"] = sum(
                        int(r.get("clamp_fr_sampled") or 0)
                        for r in traj)
                    row["fr_accepted_total"] = sum(
                        int(r.get("clamp_fr_accepted") or 0)
                        for r in traj)
                    row["fr_reach_final"] = traj[-1].get(
                        "clamp_fr_reach")
                    per_cell.append(row)
    n_total = len(MASKS) * len(arms) * len(GATES) * len(ESS)
    print(f"[clamp_graph] cells located: {len(per_cell)}/{n_total} "
          f"(arms b0/{args.icl_arm})")
    for tag in missing:
        print(f"  MISSING {tag}")
    if args.icl_arm == "d8" and missing:
        # the corrected comparison is reported ONLY on the complete
        # 96-cell surface -- a partial grid silently biases the
        # placement and arm contrasts
        print(f"[clamp_graph] HARD FAIL: {len(missing)} of {n_total} "
              f"conceptual cells missing -- no output written",
              file=sys.stderr)
        sys.exit(1)

    # placement effect: scattered minus clumped at matched cells
    contrast = []
    for arm in arms:
        for gate in GATES:
            for es in ESS:
                def get(mask):
                    r = [x for x in per_cell if x["mask"] == mask
                         and x["arm"] == arm and x["gate"] == gate
                         and x["eps_social"] == es]
                    return r[0] if r else None
                rc, rs = get("graph_clumped"), get("graph_scattered")
                if rc is None or rs is None:
                    continue
                out = {"arm": arm, "gate": gate, "eps_social": es}
                for m in CONTRAST_METRICS:
                    a, b = rs.get(m), rc.get(m)
                    out[f"d_{m}"] = (a - b if a not in (NA, None)
                                     and b not in (NA, None) else NA)
                contrast.append(out)

    # arm effect: the ICL arm minus b0 at matched (mask, gate, es)
    arm_contrast = []
    for mask in MASKS.values():
        for gate in GATES:
            for es in ESS:
                def get_arm(arm):
                    r = [x for x in per_cell if x["mask"] == mask
                         and x["arm"] == arm and x["gate"] == gate
                         and x["eps_social"] == es]
                    return r[0] if r else None
                rb, ri = get_arm("b0"), get_arm(args.icl_arm)
                if rb is None or ri is None:
                    continue
                out = {"mask": mask, "gate": gate, "eps_social": es}
                for m in CONTRAST_METRICS:
                    a, b = ri.get(m), rb.get(m)
                    out[f"d_{m}"] = (a - b if a not in (NA, None)
                                     and b not in (NA, None) else NA)
                arm_contrast.append(out)

    os.makedirs(out_dir, exist_ok=True)

    def write(name, rows):
        if not rows:
            print(f"[clamp_graph] {name}: no rows")
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(out_dir, name), "w",
                  newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys, restval=NA)
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"[clamp_graph] wrote {name} ({len(rows)} rows)")

    write("clamp_graph_per_cell.csv", per_cell)
    write("clamp_graph_placement_contrast.csv", contrast)
    write("clamp_graph_arm_contrast.csv", arm_contrast)

    print("\n== responsive std ratio vs twin, rounds 25-29 (seed 0 -- "
          "single seed, no intervals; cols = ea "
          + "/".join(f"{g:g}" for g in GATES) + ") ==")
    for gtok in MASKS:
        for arm in arms:
            print(f"  -- {MASKS[gtok]} {arm} --")
            for es in ESS:
                vals = []
                for gate in GATES:
                    r = [x for x in per_cell
                         if x["mask"] == MASKS[gtok]
                         and x["arm"] == arm and x["gate"] == gate
                         and x["eps_social"] == es]
                    v = (r[0]["resp_std_ratio_late"]
                         if r and r[0]["resp_std_ratio_late"]
                         not in (NA, None) else None)
                    vals.append(f"{v:.3f}" if v is not None
                                else "  -- ")
                print(f"    es={es:<4g}: " + "  ".join(vals))


if __name__ == "__main__":
    main()
