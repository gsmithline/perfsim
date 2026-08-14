#!/usr/bin/env python3
"""Three-seed no-peer SFT/ICL gate-grid analysis (2026-08-14).

Descriptive analysis for manifest_sft_icl_nopeer_grid3.json: per-seed
values and across-seed summaries for every (model, arm, gate) cell of the
k0 / fz0 / dyn / b0 x {0.05, 0.1, 0.2, 0.4, 1.0} x seeds {0, 42, 43}
grid. SEEDS are the replicates -- never agents. Stochastic arms (b0, fz0,
dyn) get the paired three-seed mean and a 95% Student-t interval
(df = n_seeds - 1; t_crit(2) = 4.30265). k0 is a DETERMINISTIC structural
reference (frozen prediction map; the peer-free operator draws nothing the
trajectory consumes): reference cells map to the seed-0 run and NO
confidence interval is displayed for it -- where genuine extra k0 runs
exist (the two icls2x-era hardware replicates) they appear as per-seed
rows but the summary still reports the reference value without a CI.

Metrics per cell:
  1  common_reached_frac   fraction of the common K=0-rejected cohort
                           U_common(eps) = {i : |m_base_i - innate_i| >=
                           eps} (from the seed-0 pofdreachbase_ probe;
                           innate + m_base are bit-identical across seeds)
                           eventually gated. NA when the cohort is empty
                           (e.g. eps_ai = 1.0).
  2  own_recruited_frac    fraction of the arm's OWN round-0 rejects
                           accepted at some t >= 1 (NA on empty cohort)
  3  final_mad_twin        final-round mean |op - twin|
  4  final_w1_twin         final-round W1(op, twin)
  5  final_std_ratio       final-round std(op) / std(twin)
  6  accept_frac_final / accept_frac_mean   realized acceptance fraction
  7  first_entry_mean      mean first gated round over ever-gated agents
     gate_churn_mean       mean per-round gate churn vs previous round

Outputs (under --out-dir): grid3_per_seed.csv, grid3_summary.csv, and
per-metric heatmap tables grid3_heat_<metric>.csv (rows model x arm in
the order k0/fz0/dyn/b0, cols the five gates -- the no-peer heatmap
layout, one panel per model). --figures additionally renders the panels
with matplotlib (no figure titles -- narrative belongs in captions).

Do NOT run this on scientific outcomes before the production wave exists;
fixture-test only until then.
"""
import argparse
import csv
import importlib.util
import json
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "analyze_reach", os.path.join(HERE, "analyze_sft_icl_reach.py"))
AN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN)

NA = "NA"
T_CRIT = {1: 12.7062, 2: 4.30265, 3: 3.18245, 4: 2.77645}
ARM_ORDER = ["k0", "fz0", "dyn", "b0"]
HEAT_METRICS = ["common_reached_frac", "own_recruited_frac",
                "final_mad_twin", "final_w1_twin", "final_std_ratio",
                "accept_frac_final", "first_entry_mean",
                "gate_churn_mean"]


def cell_metrics(run_dir, base):
    d = AN.load(run_dir)
    cfg = d["config"]
    op = d["op_raw"].float()
    innate = d["innate"].float()
    n_r, n = op.shape
    gates, _ = AN.derive_gates(d)
    tw, _ = AN.twin_of(d)
    if base is not None and not torch.equal(base["innate"], innate):
        raise SystemExit(f"BASELINE/INNATE MISMATCH at {run_dir}")

    ever = torch.zeros(n, dtype=torch.bool)
    first = torch.full((n,), -1, dtype=torch.long)
    churns = []
    for t in range(n_r):
        g = gates[t]
        newly = g & ~ever
        first[newly] = t
        ever |= g
        if t:
            churns.append(float((g ^ gates[t - 1]).float().mean()))
    own = ~gates[0]
    own_rec = ((own & (first >= 1)).sum() / own.sum()).item() \
        if int(own.sum()) else NA
    if base is not None and cfg.get("eps_ai") is not None:
        common = (base["m_base"] - innate).abs() >= float(cfg["eps_ai"])
        common_frac = ((common & ever).sum() / common.sum()).item() \
            if int(common.sum()) else NA
    else:
        common_frac = NA
    std_tw = float(tw[-1].std())
    return {
        "common_reached_frac": common_frac,
        "own_recruited_frac": own_rec,
        "final_mad_twin": float((op[-1] - tw[-1]).abs().mean()),
        "final_w1_twin": AN.w1(op[-1], tw[-1]),
        "final_std_ratio": (float(op[-1].std()) / std_tw
                            if std_tw > 0 else NA),
        "accept_frac_final": float(gates[-1].float().mean()),
        "accept_frac_mean": float(gates.float().mean()),
        "first_entry_mean": (float(first[ever].float().mean())
                             if int(ever.sum()) else NA),
        "gate_churn_mean": (sum(churns) / len(churns) if churns else NA),
        "hw_hostname": (cfg.get("hardware") or {}).get("hostname", NA),
        "hw_gpu": (cfg.get("hardware") or {}).get("gpu_name", NA),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        REPO, "experiments", "condor",
        "manifest_sft_icl_nopeer_grid3.json"))
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "notes", "pofd", "grid3_analysis"))
    ap.add_argument("--figures", action="store_true",
                    help="render the per-model heatmap panels (matplotlib;"
                         " no titles)")
    args = ap.parse_args()
    man = json.load(open(args.manifest))
    os.makedirs(args.out_dir, exist_ok=True)

    bases = {}
    for b in man["baselines"]:
        rd = AN.find_run(args.roots, b["run_tag"])
        if rd is not None:
            d = AN.load(rd)
            bases[b["model"]] = {"m_base": d["pred_raw"][0].float(),
                                 "innate": d["innate"].float()}
    print(f"[grid3] baselines found: {len(bases)}/{len(man['baselines'])}")

    per_seed, missing = [], 0
    cache = {}
    for c in man["cells"]:
        ident = {"model": c["model"], "arm": c["arm"], "gate": c["gate"],
                 "seed": c["seed"], "status": c["status"],
                 "run_tag": c["run_tag"]}
        rd = AN.find_run(args.roots, c["run_tag"])
        if rd is None:
            missing += 1
            per_seed.append({**ident, "found": 0})
            continue
        if rd not in cache:
            cache[rd] = cell_metrics(rd, bases.get(c["model"]))
        per_seed.append({**ident, "found": 1, **cache[rd]})
    print(f"[grid3] cells located: {len(per_seed) - missing}/"
          f"{len(per_seed)}")

    # across-seed summaries: seeds are the replicates. k0 = deterministic
    # reference (no CI); stochastic arms need all three seeds found.
    summary = []
    for model in man["grid"]["models"]:
        for arm in man["grid"]["arms"]:
            for gate in man["grid"]["gates"]:
                rows = [r for r in per_seed
                        if r["model"] == model and r["arm"] == arm
                        and r["gate"] == gate and r.get("found") == 1]
                out = {"model": model, "arm": arm, "gate": gate,
                       "n_runs": len(rows)}
                for mkey in HEAT_METRICS:
                    vals = [r[mkey] for r in rows
                            if r.get(mkey) not in (NA, None)]
                    if not vals:
                        out[f"{mkey}_mean"] = NA
                        out[f"{mkey}_ci95"] = NA
                        continue
                    if arm == "k0":
                        # deterministic structural reference: report the
                        # seed-0 value; never a fake CI
                        s0 = [r[mkey] for r in rows if r["seed"] == 0
                              and r.get(mkey) not in (NA, None)]
                        out[f"{mkey}_mean"] = s0[0] if s0 else vals[0]
                        out[f"{mkey}_ci95"] = NA
                        continue
                    mean = sum(vals) / len(vals)
                    out[f"{mkey}_mean"] = mean
                    if len(vals) >= 2:
                        var = (sum((v - mean) ** 2 for v in vals)
                               / (len(vals) - 1))
                        out[f"{mkey}_ci95"] = (
                            T_CRIT[len(vals) - 1]
                            * math.sqrt(var / len(vals)))
                    else:
                        out[f"{mkey}_ci95"] = NA
                summary.append(out)

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
        print(f"[grid3] wrote {name} ({len(rows)} rows)")

    write("grid3_per_seed.csv", per_seed)
    write("grid3_summary.csv", summary)

    # heatmap tables: rows model x arm (k0/fz0/dyn/b0), cols the gates
    gates = man["grid"]["gates"]
    for mkey in HEAT_METRICS:
        rows = []
        for model in man["grid"]["models"]:
            for arm in ARM_ORDER:
                row = {"model": model, "arm": arm}
                for g in gates:
                    s = next((x for x in summary if x["model"] == model
                              and x["arm"] == arm and x["gate"] == g), {})
                    row[f"ea{g:g}"] = s.get(f"{mkey}_mean", NA)
                    row[f"ea{g:g}_ci95"] = s.get(f"{mkey}_ci95", NA)
                rows.append(row)
        write(f"grid3_heat_{mkey}.csv", rows)

    if args.figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        models = list(man["grid"]["models"])
        for mkey in HEAT_METRICS:
            fig, axes = plt.subplots(1, len(models),
                                     figsize=(4.2 * len(models), 3.2))
            for ax, model in zip(np.atleast_1d(axes), models):
                mat = np.full((len(ARM_ORDER), len(gates)), np.nan)
                for i, arm in enumerate(ARM_ORDER):
                    for j, g in enumerate(gates):
                        s = next((x for x in summary
                                  if x["model"] == model
                                  and x["arm"] == arm
                                  and x["gate"] == g), {})
                        v = s.get(f"{mkey}_mean", NA)
                        if v not in (NA, None):
                            mat[i, j] = float(v)
                im = ax.imshow(mat, aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(gates)),
                              [f"{g:g}" for g in gates])
                ax.set_yticks(range(len(ARM_ORDER)), ARM_ORDER)
                ax.set_xlabel(f"eps_AI ({model})")
                fig.colorbar(im, ax=ax, fraction=0.046)
            # NO figure titles: narrative belongs in the caption block
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir,
                                     f"grid3_heat_{mkey}.pdf"))
            plt.close(fig)
        print(f"[grid3] rendered {len(HEAT_METRICS)} heatmap panels")


if __name__ == "__main__":
    main()
