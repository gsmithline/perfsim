#!/usr/bin/env python3
"""Innate-clamp GRAPH-PLACEMENT analysis (2026-08-17,
mistral_innate_clamp_graph_s0).

THE TEST: with 145 agents permanently pinned to innate and the peer
step LIVE under the one-sided STUBBORN operator, how does the fixed
cohort's PLACEMENT -- graph_clumped (one connected low-cut block,
41.5% responsive exposure) vs graph_scattered (distributed, 100%
exposure, 2.9x the cut edges) -- reshape what ordinary SFT (b0) and
live K=8 ICL (dyn) do to the responsive population as the social gate
opens? The two masks are matched on innate and degree distributions
(identical joint-stratum quotas), so SCATTERED-minus-CLUMPED at a
matched (arm, ea, es) isolates the placement effect.

Read-only, descriptive, SEED 0 ONLY (no confidence intervals). All
baselines are IN-WAVE: es=0 cells run the same masks (the old no-peer
cohorts are different masks and are NOT baselines here).

Per cell (rounds 25-29 window + final round): responsive population SD
vs the matched twin's responsive SD, responsive displacement from the
twin, responsive AI acceptance + cumulative AI reach, responsive-to-
fixed mean and quantile-W1 gaps with normalized closure, and total
cut (fixed-responsive) pairs sampled/accepted plus the final fraction
of responsive agents reached through the cut.

Outputs notes/pofd/clamp_graph_analysis/: clamp_graph_per_cell.csv,
clamp_graph_placement_contrast.csv (scattered minus clumped).
"""
import argparse
import csv
import importlib.util
import os

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
ARMS = ["b0", "dyn"]
GATES = [0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
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
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "notes", "pofd", "clamp_graph_analysis"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    per_cell, missing = [], []
    for gtok in MASKS:
        for arm in ARMS:
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
    n_total = len(MASKS) * len(ARMS) * len(GATES) * len(ESS)
    print(f"[clamp_graph] cells located: {len(per_cell)}/{n_total}")
    for tag in missing:
        print(f"  MISSING {tag}")

    # placement effect: scattered minus clumped at matched cells
    contrast = []
    for arm in ARMS:
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

    def write(name, rows):
        if not rows:
            print(f"[clamp_graph] {name}: no rows")
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out_dir, name), "w",
                  newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys, restval=NA)
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"[clamp_graph] wrote {name} ({len(rows)} rows)")

    write("clamp_graph_per_cell.csv", per_cell)
    write("clamp_graph_placement_contrast.csv", contrast)

    print("\n== responsive std ratio vs twin, rounds 25-29 (seed 0 -- "
          "single seed, no intervals; cols = ea "
          + "/".join(f"{g:g}" for g in GATES) + ") ==")
    for gtok in MASKS:
        for arm in ARMS:
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
