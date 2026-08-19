#!/usr/bin/env python3
"""QWEN GATE SWEEP analysis (2026-08-19, qwen_gate_sweep).

The seed-0 grid: 2 checkpoints (Qwen2.5-7B-Instruct, Qwen3-8B with
the thinking template pinned OFF) x eps_AI {.05, .1, .2, .4, 1} x
eps_social {0, .05, .1, .2, .4, 1} = 60 trajectories of regularized
SFT at lambda = 1 (forward KL against the fixed pristine base) on the
canonical Action environment. All 60 are HARD-REQUIRED. Tags resolve
through the audited manifest, so reused cells keep their archived
tags (pofdfam_ / pofdesf_ / pofdw2f_ / pofdws2f_) and new cells use
the pofdqgs_ family.

Equilibrium = rounds 25-29 within each run. Per cell:
  eq_mean   population mean, averaged over the late window
  eq_sd     population SD, averaged over the late window
  w1_init   W_1 between the late-window population and the INITIAL
            population (innate). Equal-n, so the exact 1-Wasserstein
            distance is mean|sort(x) - sort(y)|, averaged over the
            window.
Nothing here reads the twin: the new cells carry one (WITH_TWIN=1)
and the older reused cells at eps_social = 0 do not, so every
reported statistic is a property of the population alone.

Outputs (notes/pofd/qwen_gate_sweep_analysis/):
  qgs_per_cell.csv          one row per (model, gate, es) with the
                            three statistics, the resolved run tag,
                            and the GPU architecture the run executed
                            on (the grid spans several waves and
                            hardware generations -- greedy generation
                            is only bit-reproducible within one
                            architecture, so the provenance is
                            recorded rather than assumed away)
  qgs_density_points.csv    the plotted densities, long form
  qgs_heatmaps_<model>.png/pdf   eq mean | eq SD | W1-from-initial
                            over the full 5x6 grid
  qgs_density_<model>.png/pdf    3x3 density grid at eps_AI
                            {.05, .4, 1} x eps_social {0, .2, 1};
                            each cell overlays the initial
                            population, the initial model
                            predictions (round 0 served values) and
                            the population equilibrium (rounds 25-29
                            pooled)
Both models are drawn on SHARED colour scales and SHARED density axes
so the two figures can be read against each other directly.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "analyze_reach", os.path.join(HERE, "analyze_sft_icl_reach.py"))
AN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN)

MODELS = ["qwen7b", "qwen3_8b"]
DISPLAY = {"qwen7b": "Qwen2.5-7B-Instruct", "qwen3_8b": "Qwen3-8B"}
GATES = [0.05, 0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
LATE = list(range(25, 30))
DENSITY_GATES = [0.05, 0.4, 1.0]
DENSITY_ESS = [0.0, 0.2, 1.0]
BINS = np.linspace(0.0, 1.0, 41)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])
MANIFEST_DEFAULT = os.path.join(
    REPO, "experiments", "condor", "manifest_qwen_gate_sweep.json")
OUT_DIR_DEFAULT = os.path.join(
    REPO, "notes", "pofd", "qwen_gate_sweep_analysis")


def manifest_tags(manifest):
    """{(model, gate, es): (tag, status)} -- reused cells keep their
    archived tag, new cells take the pofdqgs_ tag."""
    out = {}
    for c in manifest["cells"]:
        tag = c["run_tag"] if c["status"] == "reused" else c["new_tag"]
        out[(c["model"], c["gate"], c["es"])] = (tag, c["status"])
    return out


def gpu_arch(run_dir):
    try:
        with open(os.path.join(run_dir, "config.json")) as fh:
            hw = json.load(fh).get("hardware") or {}
    except (OSError, json.JSONDecodeError):
        return "unknown"
    name = hw.get("gpu_name") or ""
    for arch in ("H100", "A100", "A6000", "V100", "B200"):
        if arch in name:
            return arch
    return name or "unknown"


def w1(a, b):
    """Exact 1-Wasserstein distance between two equal-size samples."""
    return float(np.abs(np.sort(np.asarray(a, dtype=float))
                        - np.sort(np.asarray(b, dtype=float))).mean())


def density(values):
    """Normalised histogram density on the fixed [0, 1] grid."""
    counts, _ = np.histogram(np.asarray(values, dtype=float),
                             bins=BINS, density=True)
    return counts


def cell_stats(d):
    op = d["op_raw"].float().numpy()
    innate = d["innate"].float().numpy()
    late = op[LATE]
    return {
        "eq_mean": float(np.mean([r.mean() for r in late])),
        "eq_sd": float(np.mean([r.std(ddof=0) for r in late])),
        "w1_init": float(np.mean([w1(r, innate) for r in late])),
        "init_mean": float(innate.mean()),
        "init_sd": float(innate.std(ddof=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    with open(args.manifest) as fh:
        mf = json.load(fh)
    assert mf["key"] == "qwen_gate_sweep"
    tags = manifest_tags(mf)

    run_of, missing = {}, []
    for model in MODELS:
        for gate in GATES:
            for es in ESS:
                tag = tags[(model, gate, es)][0]
                rd = AN.find_run(args.roots, tag)
                if rd is None:
                    missing.append(tag)
                else:
                    run_of[(model, gate, es)] = rd
    n_total = len(MODELS) * len(GATES) * len(ESS)
    print(f"[qgs] cells located: {len(run_of)}/{n_total}")
    for tag in missing:
        print(f"  MISSING {tag}")
    if missing:
        print(f"[qgs] HARD FAIL: {len(missing)} of {n_total} "
              f"conceptual cells missing -- no output written",
              file=sys.stderr)
        sys.exit(1)

    loads = {k: AN.load(rd) for k, rd in run_of.items()}
    ref = loads[(MODELS[0], GATES[0], 0.0)]["innate"]
    for k, d in loads.items():
        if not torch.equal(d["innate"], ref):
            print(f"[qgs] HARD FAIL: {tags[k]} innate differs from "
                  f"the shared initial population", file=sys.stderr)
            sys.exit(1)

    per_cell, dens_rows = [], []
    for (model, gate, es), d in sorted(loads.items()):
        tag, status = tags[(model, gate, es)]
        per_cell.append({
            "model": model, "display": DISPLAY[model], "gate": gate,
            "eps_social": es, "run_tag": tag, "status": status,
            "gpu_arch": gpu_arch(run_of[(model, gate, es)]),
            **cell_stats(d)})
        if gate in DENSITY_GATES and es in DENSITY_ESS:
            op = d["op_raw"].float().numpy()
            series = {
                "initial_population": density(d["innate"].float()
                                              .numpy()),
                "initial_prediction": density(
                    np.clip(d["pred_raw"].float().numpy()[0], 0.0, 1.0)),
                "equilibrium": density(op[LATE].reshape(-1)),
            }
            for name, dens in series.items():
                for x, y in zip(CENTERS, dens):
                    dens_rows.append({"model": model, "gate": gate,
                                      "eps_social": es,
                                      "series": name,
                                      "opinion": float(x),
                                      "density": float(y)})

    os.makedirs(args.out_dir, exist_ok=True)

    def write(name, rows):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out_dir, name), "w",
                  newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys)
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"[qgs] wrote {name} ({len(rows)} rows)")

    write("qgs_per_cell.csv", per_cell)
    write("qgs_density_points.csv", dens_rows)

    def cell(model, gate, es):
        return [r for r in per_cell if r["model"] == model
                and r["gate"] == gate and r["eps_social"] == es][0]

    def grid(model, metric):
        return [[cell(model, g, e)[metric] for g in GATES]
                for e in ESS]

    METRICS = (("eq_mean", "equilibrium mean"),
               ("eq_sd", "equilibrium SD"),
               ("w1_init", r"$W_1$ from initial population"))
    for model in MODELS:
        print(f"\n== {DISPLAY[model]} (cols = ea "
              + "/".join(f"{g:g}" for g in GATES) + ") ==")
        for metric, label in METRICS:
            print(f"  -- {label} --")
            gr = grid(model, metric)
            for j, es in enumerate(ESS):
                print(f"    es={es:<5g}: "
                      + "  ".join(f"{v:.4f}" for v in gr[j]))
    n_arch = {}
    for r in per_cell:
        n_arch[r["gpu_arch"]] = n_arch.get(r["gpu_arch"], 0) + 1
    print(f"\n[qgs] GPU provenance across the 60 cells: {n_arch}")

    if args.no_fig:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Rectangle

    # SHARED colour scale per metric across both models
    scales = {}
    for metric, _ in METRICS:
        vals = [cell(m, g, e)[metric] for m in MODELS for g in GATES
                for e in ESS]
        scales[metric] = Normalize(vmin=min(vals), vmax=max(vals))

    for model in MODELS:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
        for ax, (metric, label) in zip(axes, METRICS):
            vals = grid(model, metric)
            norm = scales[metric]
            for j in range(len(ESS)):
                for i in range(len(GATES)):
                    ax.add_patch(Rectangle(
                        (i - 0.5, j - 0.5), 1.0, 1.0,
                        facecolor=plt.cm.viridis(norm(vals[j][i])),
                        edgecolor="white", lw=0.5))
                    ax.text(i, j, f"{vals[j][i]:.3f}", ha="center",
                            va="center", fontsize=6,
                            color=("white"
                                   if norm(vals[j][i]) < 0.55
                                   else "black"))
            fig.colorbar(ScalarMappable(norm=norm,
                                        cmap=plt.cm.viridis),
                         ax=ax, fraction=0.046, pad=0.04, label=label)
            ax.set_xlim(-0.5, len(GATES) - 0.5)
            ax.set_ylim(-0.5, len(ESS) - 0.5)
            ax.set_xticks(range(len(GATES)))
            ax.set_xticklabels([f"{g:g}" for g in GATES])
            ax.set_yticks(range(len(ESS)))
            ax.set_yticklabels([f"{e:g}" for e in ESS])
            ax.set_xlabel(r"$\varepsilon_{\mathrm{AI}}$")
            ax.set_ylabel(r"$\varepsilon_{\mathrm{social}}$")
            ax.set_aspect("equal")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(
                args.out_dir, f"qgs_heatmaps_{model}.{ext}"),
                dpi=220 if ext == "png" else None)
        plt.close(fig)
        print(f"[qgs] wrote qgs_heatmaps_{model}.png/pdf")

    # 3x3 density grids, shared y-limit across BOTH models
    dmax = max(r["density"] for r in dens_rows)
    styles = (("initial_population", "initial population", "#777777",
               "-"),
              ("initial_prediction", "initial model predictions",
               "#D55E00", "--"),
              ("equilibrium", "population equilibrium (r25-29)",
               "#0072B2", "-"))
    for model in MODELS:
        fig, axes = plt.subplots(
            len(DENSITY_ESS), len(DENSITY_GATES),
            figsize=(9.0, 7.4), sharex=True, sharey=True)
        for row, es in enumerate(reversed(DENSITY_ESS)):
            for col, gate in enumerate(DENSITY_GATES):
                ax = axes[row][col]
                for key, label, color, ls in styles:
                    ys = [r["density"] for r in dens_rows
                          if r["model"] == model and r["gate"] == gate
                          and r["eps_social"] == es
                          and r["series"] == key]
                    ax.plot(CENTERS, ys, color=color, linestyle=ls,
                            linewidth=1.3,
                            label=label if (row == 0 and col == 0)
                            else None)
                ax.set_ylim(0, dmax * 1.05)
                ax.set_xlim(0, 1)
                ax.spines[["top", "right"]].set_visible(False)
                if row == 0:
                    ax.set_title(
                        rf"$\varepsilon_{{\mathrm{{AI}}}}={gate:g}$",
                        fontsize=9)
                if col == 0:
                    ax.set_ylabel(
                        rf"$\varepsilon_{{\mathrm{{social}}}}"
                        rf"={es:g}$" + "\ndensity", fontsize=9)
                if row == len(DENSITY_ESS) - 1:
                    ax.set_xlabel("opinion", fontsize=9)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   frameon=False, fontsize=8.5,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(
                args.out_dir, f"qgs_density_{model}.{ext}"),
                dpi=220 if ext == "png" else None)
        plt.close(fig)
        print(f"[qgs] wrote qgs_density_{model}.png/pdf")


if __name__ == "__main__":
    main()
