#!/usr/bin/env python3
"""BOTTOM-20% SOURCE-IMPACT analysis (2026-08-18 full-grid revision,
mistral_bottom20_source_impact).

Cohort A = the 145 agents with the LOWEST innate Action opinions
(deterministic innate-then-id ranking), pinned bit-exact at innate in
population and matched twin; cohort B = the other 578. Full grid:
arms {b0, b0xa, d8} x ea {0.1, 0.2, 0.4, 1} x es {0, 0.05, 0.1, 0.2,
0.4, 1}, SEED 0 ONLY = 72 conceptual cells, all HARD-REQUIRED, all
sharing the bit-identical A/B partition. Peer-enabled cells run the
one-sided stubborn operator (A-A inert, A-B moves only B, B-B
normal, identical operator on the twin); es=0 cells are the in-wave
baselines.

Per cell (rounds 25-29 late window, seed 0, no intervals):
  sd_ratio_late   responsive dispersion / matched-twin dispersion
  t_move          signed movement of B toward A:
                  T_a = mu_B^0 - mu_{B,a}^late (positive = toward A)
  A-B mean gap (initial/late) + normalized closure, quantile-W1(A,B)
  + normalized closure.

Outputs (notes/pofd/bottom20_impact_analysis/):
  bottom20_per_cell.csv       one row per (arm, gate, es)
  bottom20_contrast.csv       per (gate, es): b0xa - b0 and d8 - b0
                              on sd_ratio_late and t_move
  bottom20_grid_heatmaps.png/pdf   2x2: (a) diagonal-split SD-ratio
      heatmap, upper-left triangle = SFT with A (b0), lower-right =
      personal-history d8; (b) b0xa - b0 SD ratio; (c) diagonal-
      split signed movement toward A (same triangle convention);
      (d) b0xa - b0 signed movement. x = AI gate, y = peer gate.
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
_spec_c = importlib.util.spec_from_file_location(
    "analyze_clamp", os.path.join(HERE, "analyze_innate_clamp.py"))
AC = importlib.util.module_from_spec(_spec_c)
_spec_c.loader.exec_module(AC)

ARMS = ["b0", "b0xa", "d8"]
GATES = [0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
LATE = range(25, 30)
OUT_DIR_DEFAULT = os.path.join(
    REPO, "notes", "pofd", "bottom20_impact_analysis")


def _num(v):
    return f"{v:g}".replace(".", "p")


def cell_tag(arm, gate, es):
    stub = "" if (arm == "b0" and es == 0.0) else "_stub"
    return (f"pofdclamp_mistral7b_{arm}_bottom{stub}_ea{_num(gate)}"
            f"_w0p5_l0p2_es{_num(es)}_s0")


def cell_metrics(d, mask):
    op = d["op_raw"].float()
    tw = d["twin_raw"].float()
    innate = d["innate"].float()
    b, a = ~mask, mask
    mu_b0 = float(innate[b].mean())
    mu_b_late = float(torch.stack(
        [op[t][b].mean() for t in LATE]).mean())
    gap0 = float(innate[b].mean() - innate[a].mean())
    gap_late = float(torch.stack(
        [op[t][b].mean() - op[t][a].mean() for t in LATE]).mean())
    w1_0 = AC.w1_quantile(innate[b], innate[a])
    w1_late = sum(AC.w1_quantile(op[t][b], op[t][a])
                  for t in LATE) / len(list(LATE))
    ratios = []
    for t in LATE:
        s_tw = float(tw[t][b].std())
        if s_tw > 0:
            ratios.append(float(op[t][b].std()) / s_tw)
    return {
        "sd_ratio_late": (sum(ratios) / len(ratios)
                          if ratios else float("nan")),
        "t_move": mu_b0 - mu_b_late,
        "gap0_mean": gap0, "gap_late_mean": gap_late,
        "closure_mean": (1.0 - gap_late / gap0
                         if abs(gap0) > 1e-9 else float("nan")),
        "w1_0": w1_0, "w1_late": w1_late,
        "closure_w1": (1.0 - w1_late / w1_0
                       if w1_0 > 1e-9 else float("nan")),
    }


def split_heatmap(ax, ul, lr, cmap, norm, ul_lab, lr_lab):
    """Diagonal-split grid: cell (i, j) is cut bottom-left to
    top-right; the upper-left triangle renders ul[j][i], the
    lower-right lr[j][i]."""
    from matplotlib.patches import Polygon
    for j in range(len(ESS)):
        for i in range(len(GATES)):
            c0 = (i - 0.5, j - 0.5)
            c1 = (i + 0.5, j + 0.5)
            ax.add_patch(Polygon(
                [c0, (i - 0.5, j + 0.5), c1], closed=True,
                facecolor=cmap(norm(ul[j][i])), edgecolor="white",
                lw=0.5))
            ax.add_patch(Polygon(
                [c0, (i + 0.5, j - 0.5), c1], closed=True,
                facecolor=cmap(norm(lr[j][i])), edgecolor="white",
                lw=0.5))
            ax.text(i - 0.19, j + 0.21, f"{ul[j][i]:.2f}",
                    ha="center", va="center", fontsize=5.2)
            ax.text(i + 0.19, j - 0.21, f"{lr[j][i]:.2f}",
                    ha="center", va="center", fontsize=5.2)
    ax.text(0.02, 0.98, f"upper-left: {ul_lab}\nlower-right: {lr_lab}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7)


def flat_heatmap(ax, vals, cmap, norm):
    from matplotlib.patches import Rectangle
    for j in range(len(ESS)):
        for i in range(len(GATES)):
            ax.add_patch(Rectangle(
                (i - 0.5, j - 0.5), 1.0, 1.0,
                facecolor=cmap(norm(vals[j][i])), edgecolor="white",
                lw=0.5))
            ax.text(i, j, f"{vals[j][i]:+.2f}", ha="center",
                    va="center", fontsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    run_of, missing = {}, []
    for arm in ARMS:
        for gate in GATES:
            for es in ESS:
                tag = cell_tag(arm, gate, es)
                rd = AN.find_run(args.roots, tag)
                if rd is None:
                    missing.append(tag)
                else:
                    run_of[(arm, gate, es)] = rd
    n_total = len(ARMS) * len(GATES) * len(ESS)
    print(f"[bottom20] cells located: {len(run_of)}/{n_total}")
    for tag in missing:
        print(f"  MISSING {tag}")
    if missing:
        print(f"[bottom20] HARD FAIL: {len(missing)} of {n_total} "
              f"conceptual cells missing -- no output written",
              file=sys.stderr)
        sys.exit(1)

    loads = {k: AN.load(rd) for k, rd in run_of.items()}
    ref = loads[("b0", GATES[0], 0.0)]
    mask = ref["innate_clamp_mask"].bool()
    innate = ref["innate"].float()
    order = sorted(range(innate.numel()),
                   key=lambda i: (float(innate[i]), i))
    want_mask = torch.zeros(innate.numel(), dtype=torch.bool)
    want_mask[torch.tensor(order[:int(mask.sum())])] = True
    if not torch.equal(mask, want_mask) or int(mask.sum()) != 145 \
            or int((~mask).sum()) != 578:
        print("[bottom20] HARD FAIL: stored mask is not the "
              "deterministic bottom-145 / 578 partition",
              file=sys.stderr)
        sys.exit(1)
    for k, d in loads.items():
        if not torch.equal(d["innate_clamp_mask"].bool(), mask) or \
                not torch.equal(d["innate"], ref["innate"]):
            print(f"[bottom20] HARD FAIL: {cell_tag(*k)} does not "
                  f"share the A/B partition / innate population",
                  file=sys.stderr)
            sys.exit(1)

    per_cell = []
    for (arm, gate, es), d in sorted(loads.items()):
        per_cell.append({"arm": arm, "gate": gate, "eps_social": es,
                         "run_tag": cell_tag(arm, gate, es),
                         **cell_metrics(d, mask)})

    def cell(arm, gate, es):
        return [r for r in per_cell if r["arm"] == arm
                and r["gate"] == gate and r["eps_social"] == es][0]

    contrast = []
    for gate in GATES:
        for es in ESS:
            b0 = cell("b0", gate, es)
            xa = cell("b0xa", gate, es)
            d8 = cell("d8", gate, es)
            contrast.append({
                "gate": gate, "eps_social": es,
                "d_sd_ratio_b0xa_minus_b0":
                    xa["sd_ratio_late"] - b0["sd_ratio_late"],
                "d_sd_ratio_d8_minus_b0":
                    d8["sd_ratio_late"] - b0["sd_ratio_late"],
                "d_t_move_b0xa_minus_b0":
                    xa["t_move"] - b0["t_move"],
                "d_t_move_d8_minus_b0": d8["t_move"] - b0["t_move"],
                # the causal contrast: positive = including A's
                # labels pulled B farther toward A through weights
                "t_b0_minus_b0xa": b0["t_move"] - xa["t_move"],
            })

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
        print(f"[bottom20] wrote {name} ({len(rows)} rows)")

    write("bottom20_per_cell.csv", per_cell)
    write("bottom20_contrast.csv", contrast)

    def grid(arm, metric):
        return [[cell(arm, g, e)[metric] for g in GATES]
                for e in ESS]

    print("\n== responsive SD ratio vs twin, rounds 25-29 (seed 0, "
          "no intervals; cols = ea "
          + "/".join(f"{g:g}" for g in GATES) + ") ==")
    for arm in ARMS:
        print(f"  -- {arm} --")
        gr = grid(arm, "sd_ratio_late")
        for j, es in enumerate(ESS):
            print(f"    es={es:<4g}: "
                  + "  ".join(f"{v:.3f}" for v in gr[j]))
    print("\n== signed movement of B toward A (T = mu_B^0 - "
          "mu_B^late; positive = toward A) ==")
    for arm in ARMS:
        print(f"  -- {arm} --")
        gr = grid(arm, "t_move")
        for j, es in enumerate(ESS):
            print(f"    es={es:<4g}: "
                  + "  ".join(f"{v:+.4f}" for v in gr[j]))

    if not args.no_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize, TwoSlopeNorm
        from matplotlib.cm import ScalarMappable

        sd_b0 = grid("b0", "sd_ratio_late")
        sd_xa = grid("b0xa", "sd_ratio_late")
        sd_d8 = grid("d8", "sd_ratio_late")
        t_b0 = grid("b0", "t_move")
        t_xa = grid("b0xa", "t_move")
        t_d8 = grid("d8", "t_move")
        d_sd = [[sd_xa[j][i] - sd_b0[j][i]
                 for i in range(len(GATES))]
                for j in range(len(ESS))]
        d_t = [[t_xa[j][i] - t_b0[j][i] for i in range(len(GATES))]
               for j in range(len(ESS))]

        fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.6))
        (ax_sd, ax_dsd), (ax_t, ax_dt) = axes

        sd_all = [v for gr_ in (sd_b0, sd_d8) for row in gr_
                  for v in row]
        n_sd = Normalize(vmin=min(sd_all), vmax=max(sd_all))
        split_heatmap(ax_sd, sd_b0, sd_d8, plt.cm.viridis, n_sd,
                      "SFT with $A$ (b0)", "personal-history d8")
        fig.colorbar(ScalarMappable(norm=n_sd, cmap=plt.cm.viridis),
                     ax=ax_sd, fraction=0.046, pad=0.04,
                     label="responsive SD / twin SD (late)")

        lim = max(1e-6, max(abs(v) for row in d_sd for v in row))
        n_d = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        flat_heatmap(ax_dsd, d_sd, plt.cm.RdBu_r, n_d)
        fig.colorbar(ScalarMappable(norm=n_d, cmap=plt.cm.RdBu_r),
                     ax=ax_dsd, fraction=0.046, pad=0.04,
                     label="b0xa $-$ b0: SD ratio (late)")

        t_all = [v for gr_ in (t_b0, t_d8) for row in gr_
                 for v in row]
        t_lim = max(1e-6, max(abs(v) for v in t_all))
        n_t = TwoSlopeNorm(vmin=-t_lim, vcenter=0.0, vmax=t_lim)
        split_heatmap(ax_t, t_b0, t_d8, plt.cm.RdBu_r, n_t,
                      "SFT with $A$ (b0)", "personal-history d8")
        fig.colorbar(ScalarMappable(norm=n_t, cmap=plt.cm.RdBu_r),
                     ax=ax_t, fraction=0.046, pad=0.04,
                     label=r"movement toward $A$: "
                           r"$\mu_B^0-\mu_B^{\mathrm{late}}$")

        dt_lim = max(1e-6, max(abs(v) for row in d_t for v in row))
        n_dt = TwoSlopeNorm(vmin=-dt_lim, vcenter=0.0, vmax=dt_lim)
        flat_heatmap(ax_dt, d_t, plt.cm.RdBu_r, n_dt)
        fig.colorbar(ScalarMappable(norm=n_dt, cmap=plt.cm.RdBu_r),
                     ax=ax_dt, fraction=0.046, pad=0.04,
                     label=r"b0xa $-$ b0: movement toward $A$")

        for ax in (ax_sd, ax_dsd, ax_t, ax_dt):
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
                args.out_dir, f"bottom20_grid_heatmaps.{ext}"),
                dpi=220 if ext == "png" else None)
        print("[bottom20] wrote bottom20_grid_heatmaps.png/pdf")


if __name__ == "__main__":
    main()
