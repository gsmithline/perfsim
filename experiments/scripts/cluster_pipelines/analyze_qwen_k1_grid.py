#!/usr/bin/env python3
"""QWEN2.5 FULL-ANCHOR (k=1) vs k=0.2 Section-3 grid (2026-08-19,
qwen_k1_grid).

Matched cells: arms {b0 = ordinary SFT, b1 = forward-KL SFT at
lambda_KL = 1} x eps_AI {0.1, 0.2, 0.4, 1} x eps_social
{0, 0.05, 0.2}, seed 0 = 24 pairs, 48 trajectories, ALL HARD-
REQUIRED. The two conditions differ in ONE dial, the FJ innate
anchor k: 0.2 in the completed Section-3 grid (pofdfam_ tags,
_l0p2_), 1 in this wave (pofdfamk1_ tags, _l1_). At k=1 the update is
memoryless -- each round re-anchors fully to innate and the platform
blend still moves gated agents.

Per cell, over the late window (rounds 25-29):
  mean   population mean opinion, averaged over the window
  sd     population SD, averaged over the window
  w1     W_1 between the late-window population and the INITIAL
         population (innate). Equal-n, so the exact 1-Wasserstein
         distance is mean|sort(x) - sort(y)|, averaged over the
         window.
and the matched k=1 minus k=0.2 differences of all three.

Nothing is read from the twin: both conditions carry one, but the
comparison is a property of the populations.

Outputs (notes/pofd/qwen_k1_grid_analysis/):
  k1_per_cell.csv    one row per (k, arm, gate, es) with the three
                     statistics plus the run tag and GPU architecture
  k1_contrast.csv    one row per (arm, gate, es) with the matched
                     k=1 - k=0.2 differences
  k1_contrast.png/pdf   per-arm heatmaps of the three differences
                     (x = eps_AI, y = eps_social), shared colour
                     scale per statistic across arms
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

ARMS = ["b0", "b1"]
GATES = [0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.2]
LATE = list(range(25, 30))
KS = [("k0p2", "pofdfam", "w0p5_l0p2"),
      ("k1", "pofdfamk1", "w0p5_l1")]
METRICS = (("mean", "late population mean"),
           ("sd", "late population SD"),
           ("w1", r"$W_1$ from initial population"))
OUT_DIR_DEFAULT = os.path.join(
    REPO, "notes", "pofd", "qwen_k1_grid_analysis")


def _num(v):
    return f"{v:g}".replace(".", "p")


def cell_tag(kind, arm, gate, es):
    prefix, wl = {k[0]: (k[1], k[2]) for k in KS}[kind]
    return (f"{prefix}_qwen7b_{arm}_ea{_num(gate)}_{wl}"
            f"_es{_num(es)}_s0")


def gpu_arch(run_dir):
    try:
        with open(os.path.join(run_dir, "config.json")) as fh:
            hw = json.load(fh).get("hardware") or {}
    except (OSError, json.JSONDecodeError):
        return "unknown"
    name = hw.get("gpu_name") or ""
    for a in ("H100", "A100", "A6000", "V100", "B200"):
        if a in name:
            return a
    return name or "unknown"


def w1(a, b):
    return float(np.abs(np.sort(np.asarray(a, dtype=float))
                        - np.sort(np.asarray(b, dtype=float))).mean())


def cell_stats(d):
    op = d["op_raw"].float().numpy()
    innate = d["innate"].float().numpy()
    late = op[LATE]
    return {"mean": float(np.mean([r.mean() for r in late])),
            "sd": float(np.mean([r.std(ddof=0) for r in late])),
            "w1": float(np.mean([w1(r, innate) for r in late]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    run_of, missing = {}, []
    for kind, _, _ in KS:
        for arm in ARMS:
            for gate in GATES:
                for es in ESS:
                    tag = cell_tag(kind, arm, gate, es)
                    rd = AN.find_run(args.roots, tag)
                    if rd is None:
                        missing.append(tag)
                    else:
                        run_of[(kind, arm, gate, es)] = rd
    n_total = len(KS) * len(ARMS) * len(GATES) * len(ESS)
    print(f"[k1] trajectories located: {len(run_of)}/{n_total}")
    for tag in missing:
        print(f"  MISSING {tag}")
    if missing:
        print(f"[k1] HARD FAIL: {len(missing)} of {n_total} matched "
              f"trajectories missing -- no output written",
              file=sys.stderr)
        sys.exit(1)

    loads = {k: AN.load(rd) for k, rd in run_of.items()}
    ref = loads[("k0p2", ARMS[0], GATES[0], 0.0)]["innate"]
    for k, d in loads.items():
        if not torch.equal(d["innate"], ref):
            print(f"[k1] HARD FAIL: {cell_tag(*k)} innate differs "
                  f"from the shared initial population",
                  file=sys.stderr)
            sys.exit(1)
        # the anchor must actually differ, and nothing else about the
        # environment may
        cfg = d.get("config") or {}
        want_lam = 1.0 if k[0] == "k1" else 0.2
        if abs(float(cfg.get("innate_lambda", -1)) - want_lam) > 1e-9:
            print(f"[k1] HARD FAIL: {cell_tag(*k)} innate_lambda="
                  f"{cfg.get('innate_lambda')!r} (want {want_lam})",
                  file=sys.stderr)
            sys.exit(1)

    per_cell = []
    for (kind, arm, gate, es), d in sorted(loads.items()):
        per_cell.append({
            "k": kind, "arm": arm, "gate": gate, "eps_social": es,
            "run_tag": cell_tag(kind, arm, gate, es),
            "gpu_arch": gpu_arch(run_of[(kind, arm, gate, es)]),
            **cell_stats(d)})

    def cell(kind, arm, gate, es):
        return [r for r in per_cell if r["k"] == kind
                and r["arm"] == arm and r["gate"] == gate
                and r["eps_social"] == es][0]

    contrast = []
    for arm in ARMS:
        for gate in GATES:
            for es in ESS:
                a = cell("k1", arm, gate, es)
                b = cell("k0p2", arm, gate, es)
                row = {"arm": arm, "gate": gate, "eps_social": es}
                for m, _ in METRICS:
                    row[f"{m}_k1"] = a[m]
                    row[f"{m}_k0p2"] = b[m]
                    row[f"d_{m}"] = a[m] - b[m]
                contrast.append(row)

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
        print(f"[k1] wrote {name} ({len(rows)} rows)")

    write("k1_per_cell.csv", per_cell)
    write("k1_contrast.csv", contrast)

    def crow(arm, gate, es):
        return [r for r in contrast if r["arm"] == arm
                and r["gate"] == gate and r["eps_social"] == es][0]

    for arm in ARMS:
        print(f"\n== {arm}: k=1 minus k=0.2 (cols = ea "
              + "/".join(f"{g:g}" for g in GATES) + ") ==")
        for m, label in METRICS:
            print(f"  -- d_{m} ({label}) --")
            for es in ESS:
                print(f"    es={es:<5g}: " + "  ".join(
                    f"{crow(arm, g, es)['d_' + m]:+.4f}"
                    for g in GATES))

    if args.no_fig:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Rectangle

    # shared scale per statistic across both arms
    norms = {}
    for m, _ in METRICS:
        lim = max(1e-6, max(abs(r[f"d_{m}"]) for r in contrast))
        norms[m] = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, axes = plt.subplots(len(ARMS), len(METRICS),
                             figsize=(13.0, 6.4))
    for i, arm in enumerate(ARMS):
        for j, (m, label) in enumerate(METRICS):
            ax = axes[i][j]
            for jj, es in enumerate(ESS):
                for ii, g in enumerate(GATES):
                    v = crow(arm, g, es)[f"d_{m}"]
                    ax.add_patch(Rectangle(
                        (ii - 0.5, jj - 0.5), 1.0, 1.0,
                        facecolor=plt.cm.RdBu_r(norms[m](v)),
                        edgecolor="white", lw=0.5))
                    ax.text(ii, jj, f"{v:+.3f}", ha="center",
                            va="center", fontsize=6.5)
            fig.colorbar(ScalarMappable(norm=norms[m],
                                        cmap=plt.cm.RdBu_r),
                         ax=ax, fraction=0.046, pad=0.04,
                         label=rf"$\Delta$ {label}")
            ax.set_xlim(-0.5, len(GATES) - 0.5)
            ax.set_ylim(-0.5, len(ESS) - 0.5)
            ax.set_xticks(range(len(GATES)))
            ax.set_xticklabels([f"{g:g}" for g in GATES])
            ax.set_yticks(range(len(ESS)))
            ax.set_yticklabels([f"{e:g}" for e in ESS])
            ax.set_xlabel(r"$\varepsilon_{\mathrm{AI}}$")
            ax.set_ylabel(rf"{arm}" + "\n"
                          + r"$\varepsilon_{\mathrm{social}}$")
            ax.set_aspect("equal")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out_dir, f"k1_contrast.{ext}"),
                    dpi=220 if ext == "png" else None)
    plt.close(fig)
    print("[k1] wrote k1_contrast.png/pdf")


if __name__ == "__main__":
    main()
