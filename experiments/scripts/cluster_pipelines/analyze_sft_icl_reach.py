#!/usr/bin/env python3
"""Descriptive reach analysis for the SFT-ICL reach wave (2026-08-13).

Reads the audited cell map (manifest_sft_icl_reach.json), locates every main
trajectory (reused or pofdreach_) and every pofdreachbase_ probe under the
given roots, and writes THREE tidy CSVs. Purely descriptive: no gate is
selected, no inferential statistic is computed, and no expected scientific
outcome is assumed. `NA` is written wherever a denominator is zero or a
required input (e.g. an unpulled baseline) is absent.

Definitions (all derived OFFLINE -- nothing here touches a simulation path):
  gate g_i(t)      saved gate_raw when present; otherwise re-derived through
                   the SHARED _gated_pop.ai_gate on clamp(pred_raw[t], 0, 1)
                   vs the start-of-round opinion (innate at t=0, op_raw[t-1]
                   after), the bit-exact reconstruction the checker verifies
                   on runs that do carry the mask.
  twin             saved twin_raw when present; else innate (the es=0
                   no-platform twin is innate to 1 float32 ulp -- recorded
                   in the twin_source column).
  own cohort       U_own = {i : g_i(0) = 0}; own first entry = first
                   accepted round t >= 1.
  common cohort    per NUMERIC gate eps: U_common(eps) = {i : |m_base_i -
                   innate_i| >= eps} from the matched baseline probe's
                   frozen no-context prediction vector m_base =
                   pred_raw[0]; first reach may occur at t = 0. The
                   baseline's innate vector must be bit-identical to the
                   arm's (verified; mismatch -> hard error). Undefined (NA)
                   for the all-open gate, whose cohort has no eps.

Outputs (under --out-dir):
  reach_runs.csv        one row per main cell: identity, status, hardware,
                        baseline provenance (m_base sha256, innate
                        identity), initial/final/cumulative accepted
                        fractions, own-cohort size + recruited fraction,
                        common-cohort size + reached fraction + absolute
                        population share, final-round twin displacement /
                        W1 / signed shift / std / std ratio.
  reach_rounds.csv      one row per run x round: accepted fraction,
                        cumulative accepted fraction, twin displacement,
                        W1, signed shift, std, std ratio, and (threshold
                        runs) the gate-margin distribution
                        eps_ai - |clamp(m)-x_start| (mean/p10/p50/p90).
  reach_first_entry.csv one row per run x agent: first gated round (-1 =
                        never), own-cohort membership + own first entry
                        (t>=1; -1 = never), common-cohort membership (NA
                        without a baseline).

Usage:
  python3 analyze_sft_icl_reach.py [--manifest PATH] [--roots DIR ...]
      [--out-dir DIR]
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec_gp = importlib.util.spec_from_file_location(
    "_gated_pop_reach", os.path.join(HERE, "_gated_pop.py"))
gp = importlib.util.module_from_spec(_spec_gp)
_spec_gp.loader.exec_module(gp)

NA = "NA"


def find_run(roots, tag):
    for root in roots:
        p = os.path.join(root, tag, "trajectory.pt")
        if os.path.exists(p):
            return os.path.join(root, tag)
    return None


def load(run_dir):
    return torch.load(os.path.join(run_dir, "trajectory.pt"),
                      map_location="cpu", weights_only=False)


def derive_gates(d):
    """[rounds, n] bool gate mask: saved gate_raw, else the checker-verified
    offline reconstruction through the shared ai_gate."""
    op = d["op_raw"].float()
    gr = d.get("gate_raw")
    if gr is not None and gr.numel() > 0 and \
            tuple(gr.shape) == tuple(op.shape):
        return gr.bool(), "gate_raw"
    pred = d["pred_raw"].float()
    innate = d["innate"].float()
    cfg = d["config"]
    mode = cfg.get("ai_gate_mode") or "threshold"
    eps_ai = float(cfg["eps_ai"])
    rows = []
    for t in range(op.shape[0]):
        x0 = innate if t == 0 else op[t - 1]
        rows.append(gp.ai_gate(pred[t].clamp(0.0, 1.0), x0, eps_ai, mode))
    return torch.stack(rows), "derived"


def twin_of(d):
    tw = d.get("twin_raw")
    if tw is not None and tw.numel() > 0 and \
            tuple(tw.shape) == tuple(d["op_raw"].shape):
        return tw.float(), "twin_raw"
    n_r = d["op_raw"].shape[0]
    return d["innate"].float().unsqueeze(0).expand(n_r, -1), "innate_es0"


def w1(a, b):
    return float((torch.sort(a).values - torch.sort(b).values).abs().mean())


def frac(num, den):
    return NA if den == 0 else num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        REPO, "experiments", "condor", "manifest_sft_icl_reach.json"))
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "notes", "pofd", "reach_analysis"))
    args = ap.parse_args()
    man = json.load(open(args.manifest))
    os.makedirs(args.out_dir, exist_ok=True)

    # baselines first: m_base per (model, seed) + provenance
    bases = {}
    for b in man["baselines"]:
        rd = find_run(args.roots, b["run_tag"])
        if rd is None:
            continue
        d = load(rd)
        m_base = d["pred_raw"][0].float()
        bases[(b["model"], b["seed"])] = {
            "m_base": m_base,
            "innate": d["innate"].float(),
            "hash": hashlib.sha256(
                m_base.contiguous().numpy().tobytes()).hexdigest(),
            "tag": b["run_tag"],
        }
    print(f"[reach] baselines found: {len(bases)}/{len(man['baselines'])}")

    runs_rows, round_rows, fe_rows = [], [], []
    n_missing = 0
    for c in man["cells"]:
        ident = {"model": c["model"], "arm": c["arm"], "gate": c["gate"],
                 "seed": c["seed"], "run_tag": c["run_tag"],
                 "status": c["status"]}
        rd = find_run(args.roots, c["run_tag"])
        if rd is None:
            n_missing += 1
            runs_rows.append({**ident, "found": 0})
            continue
        d = load(rd)
        cfg = d["config"]
        op = d["op_raw"].float()
        pred = d["pred_raw"].float()
        innate = d["innate"].float()
        n_r, n = op.shape
        gates, gate_src = derive_gates(d)
        tw, tw_src = twin_of(d)
        hw = cfg.get("hardware") or {}

        ever = torch.zeros(n, dtype=torch.bool)
        first = torch.full((n,), -1, dtype=torch.long)
        first_own = torch.full((n,), -1, dtype=torch.long)
        cum = []
        for t in range(n_r):
            g = gates[t]
            newly = g & ~ever
            first[newly] = t
            if t >= 1:
                own_new = newly & (first_own < 0)
                first_own[own_new] = t
            ever |= g
            cum.append(float(ever.float().mean()))
            # rounds CSV
            disp = float((op[t] - tw[t]).abs().mean())
            std_t = float(op[t].std())
            tw_std = float(tw[t].std())
            rr = {**ident, "round": t,
                  "accepted_frac": float(g.float().mean()),
                  "cum_accepted_frac": cum[-1],
                  "mean_abs_disp_twin": disp,
                  "w1_twin": w1(op[t], tw[t]),
                  "mean_shift_twin": float(op[t].mean() - tw[t].mean()),
                  "op_std": std_t,
                  "std_ratio_twin": (std_t / tw_std if tw_std > 0 else NA)}
            if (cfg.get("ai_gate_mode") or "threshold") == "threshold":
                x0 = innate if t == 0 else op[t - 1]
                margin = float(cfg["eps_ai"]) - \
                    (pred[t].clamp(0.0, 1.0) - x0).abs()
                rr.update({
                    "gate_margin_mean": float(margin.mean()),
                    "gate_margin_p10": float(margin.quantile(0.10)),
                    "gate_margin_p50": float(margin.quantile(0.50)),
                    "gate_margin_p90": float(margin.quantile(0.90))})
            else:
                rr.update({k: NA for k in ("gate_margin_mean",
                                           "gate_margin_p10",
                                           "gate_margin_p50",
                                           "gate_margin_p90")})
            round_rows.append(rr)

        own = ~gates[0]
        n_own = int(own.sum())
        own_recruited = int((own & (first_own >= 0)).sum())

        base = bases.get((c["model"], c["seed"]))
        if base is not None and not torch.equal(base["innate"], innate):
            raise SystemExit(
                f"BASELINE/INNATE MISMATCH: {c['run_tag']} innate differs "
                f"from {base['tag']} -- populations must be bit-identical")
        if base is not None and c["gate"] != "open":
            eps = float(c["gate"])
            common = (base["m_base"] - innate).abs() >= eps
            n_common = int(common.sum())
            common_reached = int((common & (first >= 0)).sum())
        else:
            common = None
            n_common = common_reached = None

        row = {**ident, "found": 1, "n_rounds": n_r, "n_agents": n,
               "gate_source": gate_src, "twin_source": tw_src,
               "hw_hostname": hw.get("hostname", NA),
               "hw_gpu": hw.get("gpu_name", NA),
               "hw_cc": hw.get("gpu_cc", NA),
               "hw_cuda": hw.get("cuda_version", NA),
               "hw_torch": hw.get("torch_version", NA),
               "hw_transformers": hw.get("transformers_version", NA),
               "baseline_tag": base["tag"] if base else NA,
               "baseline_m_base_sha256": base["hash"] if base else NA,
               "baseline_innate_identical": 1 if base else NA,
               "initial_accepted_frac": float(gates[0].float().mean()),
               "final_accepted_frac": float(gates[-1].float().mean()),
               "cum_accepted_frac": cum[-1],
               "own_cohort_size": n_own,
               "own_recruited_frac": frac(own_recruited, n_own),
               "common_cohort_size": (NA if n_common is None else n_common),
               "common_reached_frac": (NA if n_common is None
                                       else frac(common_reached, n_common)),
               "common_reached_pop_share": (NA if common_reached is None
                                            else common_reached / n),
               "final_mean_abs_disp_twin": float(
                   (op[-1] - tw[-1]).abs().mean()),
               "final_w1_twin": w1(op[-1], tw[-1]),
               "final_mean_shift_twin": float(op[-1].mean() - tw[-1].mean()),
               "final_op_std": float(op[-1].std()),
               "final_std_ratio_twin": (
                   float(op[-1].std() / tw[-1].std())
                   if float(tw[-1].std()) > 0 else NA)}
        runs_rows.append(row)

        for i in range(n):
            fe_rows.append({**ident, "agent": i,
                            "first_gated_round": int(first[i]),
                            "in_own_cohort": int(own[i]),
                            "own_first_entry_round": int(first_own[i]),
                            "in_common_cohort": (NA if common is None
                                                 else int(common[i]))})

    def write(name, rows):
        if not rows:
            print(f"[reach] {name}: no rows")
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out_dir, name), "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys, restval=NA)
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"[reach] wrote {name} ({len(rows)} rows)")

    write("reach_runs.csv", runs_rows)
    write("reach_rounds.csv", round_rows)
    write("reach_first_entry.csv", fe_rows)
    print(f"[reach] cells located: {len(man['cells']) - n_missing}/"
          f"{len(man['cells'])} ({n_missing} not pulled -- rows carry "
          f"found=0)")


if __name__ == "__main__":
    main()
