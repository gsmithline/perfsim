#!/usr/bin/env python3
"""Zero-shot prior screen summary (2026-08-17, zsprior_screen).

Read-only, descriptive. For each candidate checkpoint's 1-round frozen
probe (pofdzsprior_<slug>_w0p5_l0p2_es0_s0) the prior is pred_raw[0]
over the paper's 723 MovieLens Action profiles. Reported per
checkpoint: mean, standard deviation, quantiles (min/5/25/50/75/95/
max), the number of DISTINCT predictions (a 723-agent prior collapsing
to a handful of values carries little per-agent signal), and the
parse-failure fraction from raw_gen_log.json.gz (0.0 required by the
checker; repeated here so the summary is self-contained).

Outputs notes/pofd/zsprior/zsprior_summary.csv.
"""
import argparse
import csv
import gzip
import importlib.util
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "analyze_reach", os.path.join(HERE, "analyze_sft_icl_reach.py"))
AN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN)

SLUGS = ["qwen3_8b", "olmo3_7b", "ministral8b", "mistralnemo"]
QS = [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "notes", "pofd", "zsprior"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for slug in SLUGS:
        tag = f"pofdzsprior_{slug}_w0p5_l0p2_es0_s0"
        rd = AN.find_run(args.roots, tag)
        if rd is None:
            print(f"MISSING {tag}")
            continue
        d = AN.load(rd)
        prior = d["pred_raw"].float()[0]
        pf = "NA"
        rg_path = os.path.join(rd, "raw_gen_log.json.gz")
        if os.path.exists(rg_path):
            with gzip.open(rg_path, "rt") as fh:
                pf = json.loads(fh.readline()).get("parse_fail_frac")
        row = {"checkpoint": slug,
               "base_model": d["config"].get("base_model"),
               "n_agents": int(prior.numel()),
               "mean": float(prior.mean()),
               "std": float(prior.std()),
               "n_distinct": int(prior.unique().numel()),
               "parse_fail_frac": pf}
        for q in QS:
            row[f"q{int(q * 100):02d}"] = float(torch.quantile(
                prior, torch.tensor(q)))
        rows.append(row)

    if rows:
        with open(os.path.join(args.out_dir, "zsprior_summary.csv"),
                  "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"[zsprior] wrote zsprior_summary.csv ({len(rows)} rows)")

    print("\n== zero-shot priors, 723 Action profiles, seed 0 ==")
    for r in rows:
        print(f"  {r['checkpoint']:12s} mean {r['mean']:.3f}  "
              f"std {r['std']:.3f}  median {r['q50']:.3f}  "
              f"[q05 {r['q05']:.3f}, q95 {r['q95']:.3f}]  "
              f"distinct {r['n_distinct']:>3d}  "
              f"parse_fail {r['parse_fail_frac']}")


if __name__ == "__main__":
    main()
