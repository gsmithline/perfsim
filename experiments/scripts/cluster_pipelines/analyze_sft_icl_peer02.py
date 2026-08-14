#!/usr/bin/env python3
"""eps_social=0.2 SFT/ICL channel-table analysis (2026-08-14).

Read-only, descriptive. Consumes manifest_sft_icl_peer02.json and writes
per-seed and across-seed summary CSVs for the table. SEEDS {0, 42, 43}
are the replicates -- never agents; every arm is stochastic here (the
peer step draws RNG), so every arm gets the mean, the sample standard
deviation, and the 95% Student-t interval (t_crit(2) = 4.30265).

Primary table metrics:
  - at eps_AI = 0.1: fraction of the COMMON plain-prompting-rejected
    cohort (|m_base - innate| >= 0.1 from the same seed-0 pofdreachbase_
    probes the no-peer grids use) accepted in >= 1 round
  - at eps_AI = 0.4: final-round mean |op - twin| (matched no-platform
    twin, which MOVES at es=0.2 -- twin_raw is REQUIRED, never innate)
  - at eps_AI = 0.4: final std(op) / std(twin)
Robustness: rounds 25-29 averages of the same displacement/W1/std-ratio
quantities. Also recorded per cell: final W1, acceptance fractions,
own-reject recruitment, first-entry mean, gate churn.

Gate masks: saved gate_raw where present (every NEW run); for reused
legacy runs the strict threshold gate is reconstructed from
clamp(pred_raw[t], 0, 1) against the start-of-round opinion (innate at
t=0, op_raw[t-1] after) and the reconstructed contact fraction is
CROSS-CHECKED against the saved per-round telemetry -- a mismatch is a
hard error, not a warning.

Do NOT run this on scientific outcomes before the production wave is
pulled; fixture-test only until then.
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
METRICS = ["common01_reached_frac", "own_recruited_frac",
           "final_mad_twin", "final_w1_twin", "final_std_ratio",
           "late_mad_twin", "late_w1_twin", "late_std_ratio",
           "accept_frac_final", "accept_frac_mean", "first_entry_mean",
           "gate_churn_mean"]


def gates_checked(d, run_tag):
    """[rounds, n] bool gates: saved gate_raw, else the strict-threshold
    reconstruction CROSS-CHECKED against saved contact telemetry."""
    op = d["op_raw"].float()
    gr = d.get("gate_raw")
    if gr is not None and gr.numel() > 0 and \
            tuple(gr.shape) == tuple(op.shape):
        return gr.bool(), "gate_raw"
    pred = d["pred_raw"].float()
    innate = d["innate"].float()
    eps_ai = float(d["config"]["eps_ai"])
    traj = d["trajectory"]
    rows = []
    for t in range(op.shape[0]):
        x0 = innate if t == 0 else op[t - 1]
        g = (pred[t].clamp(0.0, 1.0) - x0).abs() < eps_ai
        logged = traj[t].get("contact")
        if logged is not None and \
                abs(float(g.float().mean()) - float(logged)) > 1e-6:
            raise SystemExit(
                f"CONTACT CROSS-CHECK FAILED at {run_tag} round {t}: "
                f"reconstructed {float(g.float().mean()):.6f} != logged "
                f"{float(logged):.6f}")
        rows.append(g)
    return torch.stack(rows), "reconstructed"


def cell_metrics(run_dir, run_tag, base):
    d = AN.load(run_dir)
    op = d["op_raw"].float()
    innate = d["innate"].float()
    n_r, n = op.shape
    tw = d.get("twin_raw")
    if tw is None or tw.numel() == 0 or tuple(tw.shape) != tuple(op.shape):
        raise SystemExit(f"TWIN MISSING/SHORT at {run_tag} -- at es=0.2 "
                         f"the twin moves and is required, never innate")
    tw = tw.float()
    if base is not None and not torch.equal(base["innate"], innate):
        raise SystemExit(f"BASELINE/INNATE MISMATCH at {run_tag}")
    gates, gate_src = gates_checked(d, run_tag)

    ever = torch.zeros(n, dtype=torch.bool)
    first = torch.full((n,), -1, dtype=torch.long)
    churns = []
    for t in range(n_r):
        g = gates[t]
        first[g & ~ever] = t
        ever |= g
        if t:
            churns.append(float((g ^ gates[t - 1]).float().mean()))
    own = ~gates[0]
    if base is not None:
        common = (base["m_base"] - innate).abs() >= 0.1
        common01 = ((common & ever).sum() / common.sum()).item() \
            if int(common.sum()) else NA
    else:
        common01 = NA

    def window(lo, hi):
        mad = float(torch.stack([(op[t] - tw[t]).abs().mean()
                                 for t in range(lo, hi)]).mean())
        w1v = sum(AN.w1(op[t], tw[t]) for t in range(lo, hi)) / (hi - lo)
        ratios = [float(op[t].std()) / float(tw[t].std())
                  for t in range(lo, hi) if float(tw[t].std()) > 0]
        return mad, w1v, (sum(ratios) / len(ratios) if ratios else NA)

    late = window(25, 30) if n_r >= 30 else (NA, NA, NA)
    std_tw = float(tw[-1].std())
    return {
        "gate_source": gate_src,
        "common01_reached_frac": common01,
        "own_recruited_frac": (((own & (first >= 1)).sum()
                                / own.sum()).item()
                               if int(own.sum()) else NA),
        "final_mad_twin": float((op[-1] - tw[-1]).abs().mean()),
        "final_w1_twin": AN.w1(op[-1], tw[-1]),
        "final_std_ratio": (float(op[-1].std()) / std_tw
                            if std_tw > 0 else NA),
        "late_mad_twin": late[0], "late_w1_twin": late[1],
        "late_std_ratio": late[2],
        "accept_frac_final": float(gates[-1].float().mean()),
        "accept_frac_mean": float(gates.float().mean()),
        "first_entry_mean": (float(first[ever].float().mean())
                             if int(ever.sum()) else NA),
        "gate_churn_mean": (sum(churns) / len(churns) if churns else NA),
        "hw_hostname": (d["config"].get("hardware") or {}).get(
            "hostname", NA),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        REPO, "experiments", "condor", "manifest_sft_icl_peer02.json"))
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "notes", "pofd", "peer02_analysis"))
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
    print(f"[peer02] baselines found: {len(bases)}/"
          f"{len(man['baselines'])}")

    per_seed, missing = [], 0
    for c in man["cells"]:
        ident = {"model": c["model"], "arm": c["arm"], "gate": c["gate"],
                 "eps_social": c["eps_social"], "seed": c["seed"],
                 "status": c["status"], "run_tag": c["run_tag"]}
        rd = AN.find_run(args.roots, c["run_tag"])
        if rd is None:
            missing += 1
            per_seed.append({**ident, "found": 0})
            continue
        per_seed.append({**ident, "found": 1,
                         **cell_metrics(rd, c["run_tag"],
                                        bases.get(c["model"]))})
    print(f"[peer02] cells located: {len(per_seed) - missing}/"
          f"{len(per_seed)}")

    summary = []
    for model in man["grid"]["models"]:
        for arm in man["grid"]["arms"]:
            for gate in man["grid"]["gates"]:
                rows = [r for r in per_seed
                        if r["model"] == model and r["arm"] == arm
                        and r["gate"] == gate and r.get("found") == 1]
                out = {"model": model, "arm": arm, "gate": gate,
                       "n_seeds": len(rows)}
                for mkey in METRICS:
                    vals = [r[mkey] for r in rows
                            if r.get(mkey) not in (NA, None)]
                    if not vals:
                        out.update({f"{mkey}_mean": NA, f"{mkey}_sd": NA,
                                    f"{mkey}_ci95": NA})
                        continue
                    mean = sum(vals) / len(vals)
                    out[f"{mkey}_mean"] = mean
                    if len(vals) >= 2:
                        var = (sum((v - mean) ** 2 for v in vals)
                               / (len(vals) - 1))
                        out[f"{mkey}_sd"] = math.sqrt(var)
                        out[f"{mkey}_ci95"] = (
                            T_CRIT[len(vals) - 1]
                            * math.sqrt(var / len(vals)))
                    else:
                        out[f"{mkey}_sd"] = NA
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
        print(f"[peer02] wrote {name} ({len(rows)} rows)")

    write("peer02_per_seed.csv", per_seed)
    write("peer02_summary.csv", summary)


if __name__ == "__main__":
    main()
