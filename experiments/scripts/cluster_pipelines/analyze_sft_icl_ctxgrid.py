#!/usr/bin/env python3
"""Context-depth x dual-gate grid analysis + appendix heatmaps (2026-08-15).

Read-only, descriptive. Consumes manifest_sft_icl_ctxgrid.json and writes
a per-cell CSV, a channel summary CSV, and one appendix heatmap figure
per model (PDF + PNG) laying the six adaptation channels out as

    row 0 (weights change / context frozen):  SFT b0 | fixed K=8  | fixed K=32
    row 1 (weights frozen / context live):    K=0    | live  K=8  | live  K=32

so columns read as context DEPTH (none, 8, 32) and rows as whether the
served signal is refreshed. Every panel is a 5 x 4 heatmap: x = eps_AI,
y = eps_social, one square per environment, the metric printed in each
square, with ONE diverging color scale centered at 1 shared across all
panels of all three models (so panels and models are comparable).

PRIMARY metric: mean over rounds 25-29 of std(op) / std(matched-twin op)
-- above 1 the platform widened the population relative to its twin,
below 1 it compressed it. Secondary metrics (CSV only): final-round std
ratio, final/late MAD and W1 displacement from the twin, acceptance
fractions.

SEED 0 ONLY. This grid has ONE replicate per cell, so the CSVs carry NO
confidence intervals and none are computed -- the multi-seed t-intervals
used elsewhere in this project need seeds as replicates, and inventing a
spread from 723 agents would be a fabricated interval.

Twin policy (inherited from the gate2d analyzer): the matched
no-platform twin at the SAME peer setting; `innate` is permitted as the
fallback ONLY for validated legacy no-peer (es=0) reused cells, where
the checker enforces twin == innate to <= 1 float32 ulp. A real
simulated twin is REQUIRED whenever eps_social > 0. Gate masks come from
saved gate_raw where present, else the strict-threshold reconstruction
cross-checked against saved contact telemetry (hard error on mismatch).
"""
import argparse
import csv
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AN = _load("analyze_reach", "analyze_sft_icl_reach.py")
AG = _load("analyze_gate2d", "analyze_sft_icl_gate2d.py")

NA = "NA"
METRICS = AG.METRICS
# (row, col) -> arm; columns are context depth, rows are fixed vs live
PANELS = [[("b0", "SFT $\\beta=0$"), ("fz0", "fixed $K=8$"),
           ("f32", "fixed $K=32$")],
          [("k0", "$K=0$ prompting"), ("dyn", "live $K=8$"),
           ("d32", "live $K=32$")]]


def heatmaps(per_cell, grid, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    gates, ess = grid["gates"], grid["eps_socials"]
    val = {(r["model"], r["arm"], r["gate"], r["eps_social"]):
           r.get("late_std_ratio", NA)
           for r in per_cell if r.get("found") == 1}
    nums = [v for v in val.values() if v not in (NA, None)]
    if not nums:
        print("[ctxgrid] no located cells -- skipping figures")
        return
    half = max(max(abs(v - 1.0) for v in nums), 1e-3)
    norm = TwoSlopeNorm(vcenter=1.0, vmin=1.0 - half, vmax=1.0 + half)
    cmap = plt.get_cmap("RdBu_r")

    for model in grid["models"]:
        fig, axes = plt.subplots(
            2, 3, figsize=(3.05 * 3 + 1.5, 2.75 * 2 + 0.7))
        for ri, row in enumerate(PANELS):
            for ci, (arm, label) in enumerate(row):
                ax = axes[ri][ci]
                for yi, es in enumerate(ess):
                    for xi, ea in enumerate(gates):
                        v = val.get((model, arm, ea, es), NA)
                        if v in (NA, None):
                            face, txt, tcol = "0.88", "NA", "0.45"
                        else:
                            face = cmap(norm(v))
                            lum = (0.299 * face[0] + 0.587 * face[1]
                                   + 0.114 * face[2])
                            txt = f"{v:.2f}"
                            tcol = "w" if lum < 0.5 else "k"
                        ax.add_patch(plt.Rectangle(
                            (xi, yi), 1, 1, facecolor=face,
                            edgecolor="w", linewidth=0.8))
                        ax.text(xi + 0.5, yi + 0.5, txt, ha="center",
                                va="center", fontsize=7.5, color=tcol)
                ax.set_xlim(0, len(gates))
                ax.set_ylim(0, len(ess))
                ax.set_xticks([i + 0.5 for i in range(len(gates))])
                ax.set_xticklabels([f"{g:g}" for g in gates], fontsize=8)
                ax.set_yticks([i + 0.5 for i in range(len(ess))])
                ax.set_yticklabels([f"{e:g}" for e in ess], fontsize=8)
                ax.set_aspect("equal")
                # panel identifier (the channel), NOT a figure title
                ax.set_title(label, fontsize=9.5, pad=4)
                if ri == 1:
                    ax.set_xlabel(r"$\varepsilon_{\mathrm{AI}}$",
                                  fontsize=9)
                if ci == 0:
                    ax.set_ylabel(r"$\varepsilon_{\mathrm{social}}$",
                                  fontsize=9)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axes, fraction=0.028, pad=0.02)
        cb.set_label("rounds 25-29 mean  std(op) / std(twin)",
                     fontsize=9)
        base = os.path.join(out_dir, f"ctxgrid_heat_{model}")
        for ext in ("pdf", "png"):
            fig.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=200)
            print(f"[ctxgrid] wrote {os.path.basename(base)}.{ext}")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        REPO, "experiments", "condor", "manifest_sft_icl_ctxgrid.json"))
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "notes", "pofd", "ctxgrid_analysis"))
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    man = json.load(open(args.manifest))
    os.makedirs(args.out_dir, exist_ok=True)

    per_cell, missing = [], 0
    for c in man["cells"]:
        ident = {"model": c["model"], "arm": c["arm"], "gate": c["gate"],
                 "eps_social": c["eps_social"], "seed": c["seed"],
                 "status": c["status"], "run_tag": c["run_tag"]}
        rd = AN.find_run(args.roots, c["run_tag"])
        if rd is None:
            missing += 1
            per_cell.append({**ident, "found": 0})
            continue
        per_cell.append({**ident, "found": 1,
                         **AG.cell_metrics(rd, c["run_tag"],
                                           c["eps_social"], c["status"])})
    print(f"[ctxgrid] cells located: {len(per_cell) - missing}/"
          f"{len(per_cell)}")

    # channel summary: one row per (model, arm), averaging each metric
    # over the located environments. NO confidence intervals -- seed 0 is
    # the only replicate in this grid.
    summary = []
    for model in man["grid"]["models"]:
        for arm in man["grid"]["arms"]:
            rows = [r for r in per_cell if r["model"] == model
                    and r["arm"] == arm and r.get("found") == 1]
            out = {"model": model, "arm": arm,
                   "arm_label": man["grid"]["arm_labels"][arm],
                   "n_cells_found": len(rows), "n_seeds": 1,
                   "ci95": "NA (single seed -- no replicates)"}
            for mkey in METRICS:
                vals = [r[mkey] for r in rows
                        if r.get(mkey) not in (NA, None)]
                out[f"{mkey}_mean_over_cells"] = (
                    sum(vals) / len(vals) if vals else NA)
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
        print(f"[ctxgrid] wrote {name} ({len(rows)} rows)")

    if not args.figures_only:
        write("ctxgrid_per_cell.csv", per_cell)
        write("ctxgrid_channel_summary.csv", summary)
    heatmaps(per_cell, man["grid"], args.out_dir)


if __name__ == "__main__":
    main()
