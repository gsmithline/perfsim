#!/usr/bin/env python3
"""Graph-clamp CAUSAL SOURCE-EXCLUSION analysis (2026-08-18,
mistral_clamp_exclude_a).

THE ESTIMAND: the completed _b0_ SFT runs include the 145 fixed
agents' innate labels in every shared weight update; the _b0xa_ runs
are identical except SFT_EXCLUDE_CLAMPED=1 drops cohort A's rows from
every SFT batch while A stays fully present in the environment
(served, gated, pinned, stubborn peer pairing, matched twin). The
b0-minus-b0xa difference on the RESPONSIVE cohort B (578 agents) is
therefore the pathway from A into B THROUGH SHARED MODEL WEIGHTS:
  - no-peer cells (es=0): the PRIMARY estimand -- the only route from
    A to B is the weights, so any b0-vs-b0xa difference is
    weight-mediated by construction.
  - peer-enabled cells (es>0): the direct stubborn-peer pathway is
    IDENTICAL across the two conditions, so their difference is the
    ADDITIONAL weight-mediated pathway on top of it.
The personal-history _d8_ arm (frozen weights, each agent sees only
its own history) is the STRUCTURAL NULL and is reused, never rerun.

Coverage is HARD-ASSERTED: all 48 _b0xa_ cells plus their exact
matched 48 _b0_ and 48 _d8_ cells (2 masks x 4 AI gates x 6 social
gates) must be present or nothing is written. Read-only, descriptive,
SEED 0 ONLY -- exploratory, no confidence intervals.

Per matched cell, rounds 25-29 ("late") on cohort B only:
  pairwise b0 vs b0xa -- per-agent opinion MAE, opinion-distribution
  W1, served-prediction MAE and W1 (identical agent sets, so W1 is
  the exact mean |sort - sort|);
  per arm (b0 / b0xa / d8) -- responsive SD ratio vs the matched
  twin, displacement from twin, AI acceptance + cumulative reach, and
  the fixed-responsive peer reach through the cut.

Outputs (notes/pofd/clamp_graph_exclude_a_analysis/, never touching
the existing clamp_graph_analysis or clamp_graph_d8_analysis dirs):
  exclude_a_per_cell.csv       one row per (mask, arm, gate, es)
  exclude_a_pairwise.csv       one row per (mask, gate, es): b0-vs-
                               b0xa MAE/W1 on opinions + served preds
  exclude_a_arm_contrast.csv   b0xa_minus_b0 / d8_minus_b0 /
                               b0xa_minus_d8 on the shared metrics
"""
import argparse
import csv
import importlib.util
import os
import sys

import torch

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
ARMS = ["b0", "b0xa", "d8"]
GATES = [0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
LATE = range(25, 30)
OUT_DIR_DEFAULT = os.path.join(
    REPO, "notes", "pofd", "clamp_graph_exclude_a_analysis")
CONTRAST_METRICS = ["resp_std_ratio_late", "resp_std_ratio_final",
                    "resp_disp_twin_late", "resp_accept_late",
                    "resp_reach_final", "gap_mean_late", "gap_w1_late",
                    "gap_mean_closure", "fr_reach_final"]
CONTRASTS = [("b0xa_minus_b0", "b0xa", "b0"),
             ("d8_minus_b0", "d8", "b0"),
             ("b0xa_minus_d8", "b0xa", "d8")]


def _num(v):
    return f"{v:g}".replace(".", "p")


def cell_tag(arm, gtok, gate, es):
    return (f"pofdclamp_mistral7b_{arm}_{gtok}_stub_ea{_num(gate)}"
            f"_w0p5_l0p2_es{_num(es)}_s0")


def w1_sorted(a, b):
    """Exact 1-Wasserstein between equal-n empirical distributions."""
    return float((torch.sort(a).values - torch.sort(b).values)
                 .abs().mean())


def pairwise_metrics(d_a, d_b):
    """Late b0-vs-b0xa comparisons on the shared responsive cohort.
    Both runs must carry the bit-identical mask and innate (asserted
    by the caller)."""
    resp = ~d_a["innate_clamp_mask"].bool()
    out = {}
    for key, name in (("op_raw", "op"), ("pred_raw", "pred")):
        xa = d_a[key].float()
        xb = d_b[key].float()
        if name == "pred":
            xa, xb = xa.clamp(0.0, 1.0), xb.clamp(0.0, 1.0)
        out[f"{name}_mae_late"] = float(torch.stack(
            [(xa[t][resp] - xb[t][resp]).abs().mean()
             for t in LATE]).mean())
        out[f"{name}_w1_late"] = sum(
            w1_sorted(xa[t][resp], xb[t][resp])
            for t in LATE) / len(list(LATE))
        out[f"{name}_mae_final"] = float(
            (xa[-1][resp] - xb[-1][resp]).abs().mean())
        out[f"{name}_w1_final"] = w1_sorted(xa[-1][resp], xb[-1][resp])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    run_of, missing = {}, []
    for gtok in MASKS:
        for arm in ARMS:
            for gate in GATES:
                for es in ESS:
                    tag = cell_tag(arm, gtok, gate, es)
                    rd = AN.find_run(args.roots, tag)
                    if rd is None:
                        missing.append(tag)
                    else:
                        run_of[(gtok, arm, gate, es)] = rd
    n_total = len(MASKS) * len(ARMS) * len(GATES) * len(ESS)
    print(f"[exclude_a] cells located: {len(run_of)}/{n_total} "
          f"(arms {'/'.join(ARMS)})")
    for tag in missing:
        print(f"  MISSING {tag}")
    if missing:
        # the causal contrast is reported ONLY on the complete
        # 48-cell-per-arm surface -- a partial grid silently biases
        # every contrast
        print(f"[exclude_a] HARD FAIL: {len(missing)} of {n_total} "
              f"conceptual cells missing -- no output written",
              file=sys.stderr)
        sys.exit(1)

    per_cell, pairwise = [], []
    for gtok in MASKS:
        for gate in GATES:
            for es in ESS:
                loads = {arm: AN.load(run_of[(gtok, arm, gate, es)])
                         for arm in ARMS}
                m0 = loads["b0"]["innate_clamp_mask"].bool()
                for arm in ARMS:
                    if not torch.equal(
                            loads[arm]["innate_clamp_mask"].bool(), m0) \
                            or not torch.equal(loads[arm]["innate"],
                                               loads["b0"]["innate"]):
                        print(f"[exclude_a] HARD FAIL: mask/innate of "
                              f"{cell_tag(arm, gtok, gate, es)} does "
                              f"not match its b0 cell",
                              file=sys.stderr)
                        sys.exit(1)
                for arm in ARMS:
                    rd = run_of[(gtok, arm, gate, es)]
                    row = {"mask": MASKS[gtok], "arm": arm,
                           "gate": gate, "eps_social": es,
                           "run_tag": cell_tag(arm, gtok, gate, es),
                           **AC.cell_metrics(rd, "bottom")}
                    traj = loads[arm]["trajectory"]
                    row["fr_sampled_total"] = sum(
                        int(r.get("clamp_fr_sampled") or 0)
                        for r in traj)
                    row["fr_accepted_total"] = sum(
                        int(r.get("clamp_fr_accepted") or 0)
                        for r in traj)
                    row["fr_reach_final"] = traj[-1].get(
                        "clamp_fr_reach")
                    per_cell.append(row)
                pairwise.append({
                    "mask": MASKS[gtok], "gate": gate,
                    "eps_social": es,
                    **pairwise_metrics(loads["b0xa"], loads["b0"])})

    # arm contrasts on the shared per-arm metrics at matched cells
    arm_contrast = []
    for mask in MASKS.values():
        for gate in GATES:
            for es in ESS:
                def get_arm(arm):
                    return [x for x in per_cell if x["mask"] == mask
                            and x["arm"] == arm and x["gate"] == gate
                            and x["eps_social"] == es][0]
                for cname, hi, lo in CONTRASTS:
                    ra, rb = get_arm(hi), get_arm(lo)
                    out = {"mask": mask, "gate": gate,
                           "eps_social": es, "contrast": cname}
                    for m in CONTRAST_METRICS:
                        a, b = ra.get(m), rb.get(m)
                        out[f"d_{m}"] = (a - b if a not in (NA, None)
                                         and b not in (NA, None)
                                         else NA)
                    arm_contrast.append(out)

    os.makedirs(args.out_dir, exist_ok=True)

    def write(name, rows):
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
        print(f"[exclude_a] wrote {name} ({len(rows)} rows)")

    write("exclude_a_per_cell.csv", per_cell)
    write("exclude_a_pairwise.csv", pairwise)
    write("exclude_a_arm_contrast.csv", arm_contrast)

    # no-peer and peer-enabled summaries are reported SEPARATELY: at
    # es=0 the weight channel is the only A-to-B route (primary
    # estimand); at es>0 the direct peer pathway is identical across
    # conditions and the difference is the additional weight channel.
    for label, keep in (("NO-PEER (es=0): PRIMARY weight-channel "
                         "estimand", lambda e: e == 0.0),
                        ("PEER-ENABLED (es>0): additional "
                         "weight-mediated pathway",
                         lambda e: e > 0.0)):
        print(f"\n== {label} -- late b0-vs-b0xa on cohort B, rounds "
              f"25-29 (seed 0, single seed, no intervals; cols = ea "
              + "/".join(f"{g:g}" for g in GATES) + ") ==")
        for metric in ("op_mae_late", "op_w1_late", "pred_mae_late",
                       "pred_w1_late"):
            print(f"  -- {metric} --")
            for gtok in MASKS:
                for es in ESS:
                    if not keep(es):
                        continue
                    vals = []
                    for gate in GATES:
                        r = [x for x in pairwise
                             if x["mask"] == MASKS[gtok]
                             and x["gate"] == gate
                             and x["eps_social"] == es][0]
                        vals.append(f"{r[metric]:.4f}")
                    print(f"    {MASKS[gtok]:<15} es={es:<4g}: "
                          + "  ".join(vals))


if __name__ == "__main__":
    main()
