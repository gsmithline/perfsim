#!/usr/bin/env python3
"""Mistral 2-D gate grid analysis + diagonal-split heatmap (2026-08-15).

Read-only, descriptive. Consumes manifest_sft_icl_gate2d.json and writes
a per-seed CSV, an across-seed summary CSV (mean, sample SD, 95%
Student-t interval; SEEDS {0, 42, 43} are the replicates -- never
agents), and the diagonal-split heatmap (PDF + PNG): x = eps_AI, y =
eps_social, one square per environment, upper-left triangle = ordinary
SFT (b0, beta=0), lower-right triangle = live K=8 ICL (dyn), shared
diverging color scale centered at 1, the metric printed in each
triangle.

PRIMARY metric (colors + printed values): mean over rounds 25-29 of
std(op) / std(matched-twin op). Secondary (CSV only): final-round std
ratio, final/late MAD and W1 displacement from the twin, acceptance
fractions.

Twin policy: the matched no-platform twin at the SAME peer setting
(twin_raw). `innate` is permitted as the fallback ONLY for validated
legacy no-peer (es=0) reused cells, where the checker enforces
twin == innate to <= 1 float32 ulp; at es>0 a real simulated twin is
REQUIRED (hard error, never innate).

Gate masks: saved gate_raw where present (every NEW run); reconstructed
strict-threshold gates for legacy reused runs are CROSS-CHECKED against
the saved per-round contact telemetry (hard error on mismatch) via the
peer02 analyzer's shared gates_checked().

Cells not yet pulled appear with found=0, NA summary entries, and grey
NA triangles -- the full heatmap needs the production wave pulled.
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
_spec_p2 = importlib.util.spec_from_file_location(
    "analyze_peer02", os.path.join(HERE, "analyze_sft_icl_peer02.py"))
AP2 = importlib.util.module_from_spec(_spec_p2)
_spec_p2.loader.exec_module(AP2)

NA = "NA"
T_CRIT = {1: 12.7062, 2: 4.30265, 3: 3.18245, 4: 2.77645}
METRICS = ["late_std_ratio", "final_std_ratio",
           "final_mad_twin", "final_w1_twin",
           "late_mad_twin", "late_w1_twin",
           "accept_frac_final", "accept_frac_mean"]
ARM_LABEL = {"b0": "SFT (beta=0)", "dyn": "live ICL (K=8)"}


def twin_of(d, run_tag, es, status):
    """(twin [rounds, n], source). Real matched twin required at es>0;
    innate broadcast permitted only for validated legacy no-peer reuse."""
    op = d["op_raw"].float()
    tw = d.get("twin_raw")
    if tw is not None and tw.numel() > 0 and \
            tuple(tw.shape) == tuple(op.shape):
        return tw.float(), "twin_raw"
    if es == 0.0 and status == "reused":
        return (d["innate"].float().unsqueeze(0).expand_as(op)
                .contiguous(), "innate_fallback")
    raise SystemExit(f"TWIN MISSING/SHORT at {run_tag} -- at es>0 the "
                     f"twin moves and is required, never innate")


def cell_metrics(run_dir, run_tag, es, status):
    d = AN.load(run_dir)
    op = d["op_raw"].float()
    n_r, _ = op.shape
    tw, twin_src = twin_of(d, run_tag, es, status)
    gates, gate_src = AP2.gates_checked(d, run_tag)

    def std_ratio(t):
        s_tw = float(tw[t].std())
        return float(op[t].std()) / s_tw if s_tw > 0 else None

    def window(lo, hi):
        mad = float(torch.stack([(op[t] - tw[t]).abs().mean()
                                 for t in range(lo, hi)]).mean())
        w1v = sum(AN.w1(op[t], tw[t]) for t in range(lo, hi)) / (hi - lo)
        ratios = [r for r in (std_ratio(t) for t in range(lo, hi))
                  if r is not None]
        return mad, w1v, (sum(ratios) / len(ratios) if ratios else NA)

    late = window(25, 30) if n_r >= 30 else (NA, NA, NA)
    return {
        "gate_source": gate_src, "twin_source": twin_src,
        "late_std_ratio": late[2],
        "final_std_ratio": (std_ratio(n_r - 1)
                            if std_ratio(n_r - 1) is not None else NA),
        "final_mad_twin": float((op[-1] - tw[-1]).abs().mean()),
        "final_w1_twin": AN.w1(op[-1], tw[-1]),
        "late_mad_twin": late[0], "late_w1_twin": late[1],
        "accept_frac_final": float(gates[-1].float().mean()),
        "accept_frac_mean": float(gates.float().mean()),
        "hw_hostname": (d["config"].get("hardware") or {}).get(
            "hostname", NA),
    }


def heatmap(summary, grid, out_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Polygon

    gates, ess = grid["gates"], grid["eps_socials"]
    val = {(r["arm"], r["gate"], r["eps_social"]):
           r["late_std_ratio_mean"] for r in summary}
    nums = [v for v in val.values() if v != NA]
    # symmetric about the center so equal departures from 1 get equal
    # color intensity on both sides of the diverging scale
    half = max(abs(v - 1.0) for v in nums) if nums else 0.5
    half = max(half, 1e-3)
    norm = TwoSlopeNorm(vcenter=1.0, vmin=1.0 - half, vmax=1.0 + half)
    cmap = plt.get_cmap("RdBu_r")

    fig, ax = plt.subplots(figsize=(1.55 * len(gates) + 1.6,
                                    1.55 * len(ess) + 1.0))
    for yi, es in enumerate(ess):
        for xi, ea in enumerate(gates):
            for arm, verts, cx, cy in (
                    ("b0", [(xi, yi), (xi, yi + 1), (xi + 1, yi + 1)],
                     xi + 1 / 3, yi + 2 / 3),
                    ("dyn", [(xi, yi), (xi + 1, yi), (xi + 1, yi + 1)],
                     xi + 2 / 3, yi + 1 / 3)):
                v = val.get((arm, ea, es), NA)
                if v == NA:
                    face, txt, tcol = "0.88", "NA", "0.45"
                else:
                    face = cmap(norm(v))
                    lum = (0.299 * face[0] + 0.587 * face[1]
                           + 0.114 * face[2])
                    txt, tcol = f"{v:.2f}", ("w" if lum < 0.5 else "k")
                ax.add_patch(Polygon(verts, closed=True, facecolor=face,
                                     edgecolor="w", linewidth=0.8))
                ax.text(cx, cy, txt, ha="center", va="center",
                        fontsize=8.5, color=tcol)
            ax.add_patch(Polygon([(xi, yi), (xi + 1, yi + 1)],
                                 closed=False, fill=False,
                                 edgecolor="w", linewidth=0.8))
    for xi in range(len(gates) + 1):
        ax.axvline(xi, color="k", linewidth=0.6)
    for yi in range(len(ess) + 1):
        ax.axhline(yi, color="k", linewidth=0.6)
    ax.set_xlim(0, len(gates))
    ax.set_ylim(0, len(ess))
    ax.set_xticks([i + 0.5 for i in range(len(gates))])
    ax.set_xticklabels([f"{g:g}" for g in gates])
    ax.set_yticks([i + 0.5 for i in range(len(ess))])
    ax.set_yticklabels([f"{e:g}" for e in ess])
    ax.set_xlabel(r"$\varepsilon_{\mathrm{AI}}$")
    ax.set_ylabel(r"$\varepsilon_{\mathrm{social}}$")
    ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("rounds 25-29 mean  std(op) / std(twin)")
    # triangle key (legend, not a title -- figures carry no titles)
    fig.text(0.5, 0.005,
             "upper-left triangle: SFT ($\\beta=0$)   ·   "
             "lower-right triangle: live ICL ($K=8$)",
             ha="center", va="bottom", fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight", dpi=200)
        print(f"[gate2d] wrote {out_base}.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        REPO, "experiments", "condor", "manifest_sft_icl_gate2d.json"))
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "notes", "pofd", "gate2d_analysis"))
    args = ap.parse_args()
    man = json.load(open(args.manifest))
    os.makedirs(args.out_dir, exist_ok=True)

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
                                        c["eps_social"], c["status"])})
    print(f"[gate2d] cells located: {len(per_seed) - missing}/"
          f"{len(per_seed)}")

    summary = []
    for arm in man["grid"]["arms"]:
        for es in man["grid"]["eps_socials"]:
            for gate in man["grid"]["gates"]:
                rows = [r for r in per_seed
                        if r["arm"] == arm and r["gate"] == gate
                        and r["eps_social"] == es and r.get("found") == 1]
                out = {"model": man["grid"]["model"], "arm": arm,
                       "gate": gate, "eps_social": es,
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
        print(f"[gate2d] wrote {name} ({len(rows)} rows)")

    write("gate2d_per_seed.csv", per_seed)
    write("gate2d_summary.csv", summary)
    heatmap(summary, man["grid"],
            os.path.join(args.out_dir, "gate2d_heatmap"))


if __name__ == "__main__":
    main()
