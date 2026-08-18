#!/usr/bin/env python3
"""FIXED-vs-FULLY-EVOLVING comparison for the bottom-20% wave
(2026-08-18, mistral_bottom20_evolving vs mistral_bottom20_source_
impact).

Matched cells: arm {b0 = ordinary SFT, d8 = personal-history ICL} x
ea {0.1, 0.2, 0.4, 1} x es {0, 0.05, 0.1, 0.2, 0.4, 1}, seed 0. The
FIXED condition pins cohort A (the 145 lowest-innate agents,
deterministic innate-then-id ranking) at innate; the EVOLVING
condition runs the identical code path with NO clamp -- A is defined
only ANALYTICALLY here, by recomputing the same ranking on the
shared innate vector. All 96 trajectories (48 fixed + 48 evolving)
are HARD-REQUIRED and must share the identical innate population;
every fixed run's stored mask must equal the recomputed bottom-145.

Per matched cell, rounds 25-29 ("late"):
  A and B mean and SD under each arm and condition (raw), plus B
  stats normalized by the SHARED initial B SD (std of innate over B
  -- one denominator for every condition; twin denominators are
  deliberately NOT used across conditions);
  fixed - evolving differences in B mean and B SD per arm;
  ICL - SFT difference in B SD within each condition;
  the difference-in-differences
    DiD = (SD_B[d8|fixed] - SD_B[b0|fixed])
        - (SD_B[d8|evolving] - SD_B[b0|evolving])
  -- does fixing A change the SFT-vs-ICL dispersion gap?

Outputs (notes/pofd/bottom20_evolving_analysis/):
  evolving_per_cell.csv    one row per (condition, arm, gate, es)
  evolving_contrast.csv    one row per (gate, es) with every contrast
  evolving_heatmaps.png/pdf  2x3 annotated grids: fixed-evolving B
      mean (SFT, ICL) + ICL-SFT B SD (fixed) on top; fixed-evolving
      B SD (SFT, ICL) + the DiD below. x = AI gate, y = peer gate.
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

ARMS = ["b0", "d8"]
GATES = [0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
LATE = range(25, 30)
CONDS = ["fixed", "evolving"]
OUT_DIR_DEFAULT = os.path.join(
    REPO, "notes", "pofd", "bottom20_evolving_analysis")


def _num(v):
    return f"{v:g}".replace(".", "p")


def fixed_tag(arm, gate, es):
    stub = "" if (arm == "b0" and es == 0.0) else "_stub"
    return (f"pofdclamp_mistral7b_{arm}_bottom{stub}_ea{_num(gate)}"
            f"_w0p5_l0p2_es{_num(es)}_s0")


def evo_tag(arm, gate, es):
    return (f"pofdevo_mistral7b_{arm}_ea{_num(gate)}"
            f"_w0p5_l0p2_es{_num(es)}_s0")


def cell_tag(cond, arm, gate, es):
    return (fixed_tag if cond == "fixed" else evo_tag)(arm, gate, es)


def cell_stats(d, mask, sd0):
    """Late A/B means and SDs (raw + B normalized by the shared
    initial B SD)."""
    op = d["op_raw"].float()
    b, a = ~mask, mask
    a_mean = float(torch.stack([op[t][a].mean() for t in LATE]).mean())
    a_sd = float(torch.stack([op[t][a].std() for t in LATE]).mean())
    b_mean = float(torch.stack([op[t][b].mean() for t in LATE]).mean())
    b_sd = float(torch.stack([op[t][b].std() for t in LATE]).mean())
    return {"a_mean_late": a_mean, "a_sd_late": a_sd,
            "b_mean_late": b_mean, "b_sd_late": b_sd,
            "b_sd_late_norm": b_sd / sd0,
            "b_mean_shift_norm":
                (b_mean - float(d["innate"].float()[b].mean())) / sd0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    run_of, missing = {}, []
    for cond in CONDS:
        for arm in ARMS:
            for gate in GATES:
                for es in ESS:
                    tag = cell_tag(cond, arm, gate, es)
                    rd = AN.find_run(args.roots, tag)
                    if rd is None:
                        missing.append(tag)
                    else:
                        run_of[(cond, arm, gate, es)] = rd
    n_total = len(CONDS) * len(ARMS) * len(GATES) * len(ESS)
    print(f"[b20_evo] trajectories located: {len(run_of)}/{n_total}")
    for tag in missing:
        print(f"  MISSING {tag}")
    if missing:
        print(f"[b20_evo] HARD FAIL: {len(missing)} of {n_total} "
              f"matched trajectories missing -- no output written",
              file=sys.stderr)
        sys.exit(1)

    loads = {k: AN.load(rd) for k, rd in run_of.items()}
    ref = loads[("evolving", "b0", GATES[0], 0.0)]
    innate = ref["innate"].float()
    # cohort A: the SAME lowest-innate 145 agents, ANALYTICALLY only
    order = sorted(range(innate.numel()),
                   key=lambda i: (float(innate[i]), i))
    mask = torch.zeros(innate.numel(), dtype=torch.bool)
    mask[torch.tensor(order[:145])] = True
    sd0 = float(innate[~mask].std())     # shared initial B SD
    for k, d in loads.items():
        if not torch.equal(d["innate"], ref["innate"]):
            print(f"[b20_evo] HARD FAIL: {cell_tag(*k)} innate "
                  f"differs from the shared population",
                  file=sys.stderr)
            sys.exit(1)
        cm = d.get("innate_clamp_mask")
        if k[0] == "fixed":
            if cm is None or not torch.equal(cm.bool(), mask):
                print(f"[b20_evo] HARD FAIL: {cell_tag(*k)} stored "
                      f"mask != the recomputed bottom-145",
                      file=sys.stderr)
                sys.exit(1)
        elif cm is not None and cm.numel():
            print(f"[b20_evo] HARD FAIL: {cell_tag(*k)} carries a "
                  f"clamp mask -- not fully evolving",
                  file=sys.stderr)
            sys.exit(1)

    per_cell = []
    for (cond, arm, gate, es), d in sorted(loads.items()):
        per_cell.append({"condition": cond, "arm": arm, "gate": gate,
                         "eps_social": es,
                         "run_tag": cell_tag(cond, arm, gate, es),
                         "b_sd0_shared": sd0,
                         **cell_stats(d, mask, sd0)})

    def st(cond, arm, gate, es):
        return [r for r in per_cell if r["condition"] == cond
                and r["arm"] == arm and r["gate"] == gate
                and r["eps_social"] == es][0]

    contrast = []
    for gate in GATES:
        for es in ESS:
            fb0 = st("fixed", "b0", gate, es)
            fd8 = st("fixed", "d8", gate, es)
            eb0 = st("evolving", "b0", gate, es)
            ed8 = st("evolving", "d8", gate, es)
            row = {"gate": gate, "eps_social": es,
                   "d_b_mean_sft": fb0["b_mean_late"]
                   - eb0["b_mean_late"],
                   "d_b_sd_sft": fb0["b_sd_late"] - eb0["b_sd_late"],
                   "d_b_mean_icl": fd8["b_mean_late"]
                   - ed8["b_mean_late"],
                   "d_b_sd_icl": fd8["b_sd_late"] - ed8["b_sd_late"],
                   "icl_minus_sft_b_sd_fixed":
                       fd8["b_sd_late"] - fb0["b_sd_late"],
                   "icl_minus_sft_b_sd_evolving":
                       ed8["b_sd_late"] - eb0["b_sd_late"]}
            row["did_b_sd"] = (row["icl_minus_sft_b_sd_fixed"]
                               - row["icl_minus_sft_b_sd_evolving"])
            for k in list(row):
                if k not in ("gate", "eps_social"):
                    row[f"{k}_norm"] = row[k] / sd0
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
        print(f"[b20_evo] wrote {name} ({len(rows)} rows)")

    write("evolving_per_cell.csv", per_cell)
    write("evolving_contrast.csv", contrast)

    def cgrid(metric):
        return [[[r for r in contrast if r["gate"] == g
                  and r["eps_social"] == e][0][metric]
                 for g in GATES] for e in ESS]

    print(f"\n[b20_evo] shared initial B SD = {sd0:.4f}")
    for metric, label in (
            ("d_b_sd_sft", "fixed - evolving B SD, SFT"),
            ("d_b_sd_icl", "fixed - evolving B SD, ICL"),
            ("did_b_sd", "DiD: fixing A's effect on the ICL-SFT "
                         "B SD gap")):
        print(f"\n== {label} (rounds 25-29, seed 0; cols = ea "
              + "/".join(f"{g:g}" for g in GATES) + ") ==")
        gr = cgrid(metric)
        for j, es in enumerate(ESS):
            print(f"  es={es:<4g}: "
                  + "  ".join(f"{v:+.4f}" for v in gr[j]))

    if not args.no_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.cm import ScalarMappable
        from matplotlib.patches import Rectangle

        panels = [
            ("d_b_mean_sft", "fixed $-$ evolving: $B$ mean (SFT)"),
            ("d_b_mean_icl", "fixed $-$ evolving: $B$ mean (ICL d8)"),
            ("icl_minus_sft_b_sd_fixed",
             "ICL $-$ SFT: $B$ SD ($A$ fixed)"),
            ("d_b_sd_sft", "fixed $-$ evolving: $B$ SD (SFT)"),
            ("d_b_sd_icl", "fixed $-$ evolving: $B$ SD (ICL d8)"),
            ("did_b_sd", "DiD: (ICL$-$SFT SD | fixed) $-$ "
                         "(ICL$-$SFT SD | evolving)"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6))
        for ax, (metric, label) in zip(axes.flat, panels):
            vals = cgrid(metric)
            lim = max(1e-6, max(abs(v) for row in vals for v in row))
            norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
            for j in range(len(ESS)):
                for i in range(len(GATES)):
                    ax.add_patch(Rectangle(
                        (i - 0.5, j - 0.5), 1.0, 1.0,
                        facecolor=plt.cm.RdBu_r(norm(vals[j][i])),
                        edgecolor="white", lw=0.5))
                    ax.text(i, j, f"{vals[j][i]:+.3f}", ha="center",
                            va="center", fontsize=6.5)
            fig.colorbar(ScalarMappable(norm=norm,
                                        cmap=plt.cm.RdBu_r),
                         ax=ax, fraction=0.046, pad=0.04,
                         label=label)
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
                args.out_dir, f"evolving_heatmaps.{ext}"),
                dpi=200 if ext == "png" else None)
        print("[b20_evo] wrote evolving_heatmaps.png/pdf")


if __name__ == "__main__":
    main()
