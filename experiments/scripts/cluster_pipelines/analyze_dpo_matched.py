#!/usr/bin/env python3
"""Preregistered analysis for the matched-randomness DPO pairs (2026-08-13).

Usage:  python analyze_dpo_matched.py [runs_dir] [out_csv]
        (defaults: runs/pokec_gated_lm, dpo_matched_pairs.csv)

Run ONLY after every pair is pulled, sanity-gated per arm
(check_pofd_sanity) and pair-gated (check_dpo_pair.py). Produces:

  - a per-pair CSV: gate, bank seed, late closed/open means, paired
    difference, late Wasserstein-1, first preference-divergence round,
    round-1 label-disagreement fraction, mean disagreement, low/high late
    state;
  - paired mean differences (closed - open) with 95% Student-t intervals
    per gate group -- no post-hoc cell selection: every completed pair of
    a group enters;
  - the distribution of late closed-open effects;
  - round-1 label disagreement vs late population divergence (per pair);
  - the fraction of wide-gate (ea0.4) bank seeds whose closed arm enters
    the LOW late state (mean < 0.5) vs HIGH;
  - the same table for the narrow-gate (ea0.2) expected-null control.

Bank seeds -- not agents, not rounds -- are the inferential replicates.
"""
import csv
import glob
import math
import sys
from pathlib import Path

import torch

T_CRIT_95 = {2: 4.302652729911275, 3: 3.182446305284263,
             4: 2.7764451051977987, 5: 2.5705818366147395,
             9: 2.2621571627409915, 10: 2.228138852,
             14: 2.144786688}
LATE = 5          # late window: final LATE rounds
LOW_THRESH = 0.5  # closed late mean below this = the displaced/low basin


def w1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.sort().values - b.sort().values).abs().mean())


def t_ci(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    crit = T_CRIT_95.get(n - 1)
    if crit is None:
        raise ValueError(f"no t critical value tabled for df={n - 1}")
    half = crit * sd / math.sqrt(n)
    return m, m - half, m + half


def main():
    runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/pokec_gated_lm")
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "dpo_matched_pairs.csv"
    rows = []
    for closed in sorted(glob.glob(str(runs / "pofddpomr_*_closed_*"))):
        cd = Path(closed)
        od = cd.parent / cd.name.replace("_closed_", "_open_")
        bd = cd.parent / cd.name.replace("_closed_", "_bank_")
        if not (od / "trajectory.pt").exists():
            print(f"[skip] {cd.name}: open arm missing")
            continue
        if not (bd / "pair_meta.json").exists():
            print(f"[skip] {cd.name}: pair not validated "
                  f"(run check_dpo_pair.py first)")
            continue
        dc = torch.load(cd / "trajectory.pt", map_location="cpu",
                        weights_only=False)
        do = torch.load(od / "trajectory.pt", map_location="cpu",
                        weights_only=False)
        cc = dc["config"]
        n_r = int(cc["n_rounds"])
        late = slice(n_r - LATE, n_r)
        opc, opo = dc["op_raw"].float(), do["op_raw"].float()
        to = do["trajectory"]
        first_div = next((t for t in range(1, n_r)
                          if int(to[t].get("dpo_label_disagree_n") or 0) > 0),
                         None)
        late_c = float(opc[late].mean())
        late_o = float(opo[late].mean())
        rows.append({
            "eps_ai": float(cc["eps_ai"]),
            "bank_seed": int(cc["dpo_bank_seed"]),
            "late_closed_mean": late_c,
            "late_open_mean": late_o,
            "paired_diff": late_c - late_o,
            "late_w1": w1(opc[late].flatten(), opo[late].flatten()),
            "first_divergence_round": first_div,
            "r1_disagree_frac": to[1].get("dpo_label_disagree_frac"),
            "mean_disagree_frac": sum(
                (to[t].get("dpo_label_disagree_frac") or 0.0)
                for t in range(1, n_r)) / (n_r - 1),
            "late_pop_divergence": w1(opc[-1], opo[-1]),
            "closed_late_state": "low" if late_c < LOW_THRESH else "high",
        })
    if not rows:
        sys.exit("no validated pairs found")
    cols = list(rows[0])
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[analyze] wrote {out_csv} ({len(rows)} pairs)\n")

    for ea in sorted({r["eps_ai"] for r in rows}, reverse=True):
        grp = [r for r in rows if r["eps_ai"] == ea]
        label = "WIDE gate" if ea == 0.4 else "narrow-gate control"
        diffs = [r["paired_diff"] for r in grp]
        w1s = [r["late_w1"] for r in grp]
        m, lo, hi = t_ci(diffs)
        mw, low_, hiw = t_ci(w1s)
        print(f"== ea={ea:g} ({label}, n={len(grp)} bank-seed pairs) ==")
        print(f"  paired late mean diff (closed-open): "
              f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] 95% t")
        print(f"  late W1(closed, open): {mw:.4f} [{low_:.4f}, {hiw:.4f}]")
        print("  late-effect distribution:",
              " ".join(f"{d:+.3f}" for d in sorted(diffs)))
        print("  first divergence rounds:",
              [r["first_divergence_round"] for r in grp])
        n_low = sum(1 for r in grp if r["closed_late_state"] == "low")
        print(f"  closed-arm late state: {n_low}/{len(grp)} low, "
              f"{len(grp) - n_low}/{len(grp)} high")
        print("  r1 disagreement vs late divergence (per pair):")
        for r in sorted(grp, key=lambda r: r["bank_seed"]):
            print(f"    bk{r['bank_seed']}: r1_disagree="
                  f"{(r['r1_disagree_frac'] or 0):.4f} -> late W1(pop) "
                  f"{r['late_pop_divergence']:.4f} "
                  f"({r['closed_late_state']})")
        print()


if __name__ == "__main__":
    main()
