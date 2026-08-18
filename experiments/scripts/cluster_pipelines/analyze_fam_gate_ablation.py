#!/usr/bin/env python3
"""SECTION-3 FAMILY-GATE ABLATION analysis (2026-08-18,
fam_gate_ablation).

Grid (2026-08-18 SOCIAL-GATE EXTENSION): six checkpoints x SFT arms
{b0 = beta 0, b1 = forward-KL beta 1} x ea {0.1, 0.2, 0.4, 1} x es
{0, 0.05, 0.2} at lam=0.2 / W=0.5, seed 0 = 144 cells, all
HARD-REQUIRED (the ea=1 cells are the completed family-prior-scout
runs; es0.05 is the 48-cell ablation wave).

Equilibrium W1 between a checkpoint pair = the mean over rounds
25-29 of the exact 1-Wasserstein distance between their late
opinion populations (equal n=723, so W1 is mean |sort - sort|).
Prior W1 = the same distance between the pair's zero-shot M0 priors
(pred_raw[0] of the completed reachbase/zsprior probes).

Outputs (notes/pofd/fam_gate_analysis/):
  fam_gate_panels.png/pdf   three panels (Qwen / OLMo / Mistral):
      WITHIN-FAMILY equilibrium W1 vs the AI gate at es0.05, one
      line per beta
  fam_gate_grid.png/pdf     split-cell heatmaps over the full
      ea x es surface (beta=0 upper-left triangle, beta=1
      lower-right): within-family median and all-15-pair median
      equilibrium W1
  fam_gate_pairs.csv        all 15 checkpoint pairs: prior W1 +
      equilibrium W1 under both betas at EVERY (gate, es) + median
      row
  fam_gate_pairs.tex        the es0.05 gate axis as a LaTeX appendix
      tabular (booktabs)
"""
import argparse
import csv
import importlib.util
import itertools
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "analyze_reach", os.path.join(HERE, "analyze_sft_icl_reach.py"))
AN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN)
_spec_f = importlib.util.spec_from_file_location(
    "analyze_fam", os.path.join(HERE,
                                "analyze_fig2_family_prior.py"))
AF2 = importlib.util.module_from_spec(_spec_f)
_spec_f.loader.exec_module(AF2)

MODELS = ["qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b", "mistral7b",
          "ministral8b"]
DISPLAY = {"qwen7b": "Qwen2.5-7B", "qwen3_8b": "Qwen3-8B",
           "olmo7b": "OLMo2-7B", "olmo3_7b": "OLMo3-7B",
           "mistral7b": "Mistral-7B", "ministral8b": "Ministral-8B"}
FAMILIES = [("Qwen", ("qwen7b", "qwen3_8b")),
            ("OLMo", ("olmo7b", "olmo3_7b")),
            ("Mistral", ("mistral7b", "ministral8b"))]
ARMS = ["b0", "b1"]
GATES = [0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.2]
LATE = range(25, 30)
OUT_DIR_DEFAULT = os.path.join(
    REPO, "notes", "pofd", "fam_gate_analysis")


def _num(v):
    return f"{v:g}".replace(".", "p")


def cell_tag(model, arm, gate, es=0.05):
    return (f"pofdfam_{model}_{arm}_ea{_num(gate)}_w0p5_l0p2"
            f"_es{_num(es)}_s0")


def w1(a, b):
    return float((torch.sort(a).values - torch.sort(b).values)
                 .abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    run_of, missing = {}, []
    for model in MODELS:
        for arm in ARMS:
            for gate in GATES:
                for es in ESS:
                    tag = cell_tag(model, arm, gate, es)
                    # the mistral b0 ea1 es0p2 slot is filled by the
                    # gate2d run the fam scout itself reused
                    rd = AN.find_run(args.roots, tag)
                    if rd is None and gate == 1.0 and es == 0.2 \
                            and (model, arm) == ("mistral7b", "b0"):
                        rd = AN.find_run(
                            args.roots,
                            "pofdgate2d_mistral7b_b0_ea1_w0p5_l0p2"
                            "_es0p2_s0")
                    if rd is None:
                        missing.append(tag)
                    else:
                        run_of[(model, arm, gate, es)] = rd
    n_total = len(MODELS) * len(ARMS) * len(GATES) * len(ESS)
    print(f"[fam_gate] cells located: {len(run_of)}/{n_total}")
    for tag in missing:
        print(f"  MISSING {tag}")
    if missing:
        print(f"[fam_gate] HARD FAIL: {len(missing)} of {n_total} "
              f"cells missing -- no output written", file=sys.stderr)
        sys.exit(1)

    # late populations + M0 priors
    late_of = {}
    for k, rd in run_of.items():
        op = AN.load(rd)["op_raw"].float()
        late_of[k] = [op[t] for t in LATE]
    prior_of = {}
    for m in MODELS:
        rd = AN.find_run(args.roots, AF2.PRIOR_TAG[m])
        if rd is None:
            print(f"[fam_gate] HARD FAIL: M0 prior run missing: "
                  f"{AF2.PRIOR_TAG[m]}", file=sys.stderr)
            sys.exit(1)
        prior_of[m] = AN.load(rd)["pred_raw"].float()[0].clamp(0, 1)

    def eq_w1(m1, m2, arm, gate, es=0.05):
        return sum(w1(a, b)
                   for a, b in zip(late_of[(m1, arm, gate, es)],
                                   late_of[(m2, arm, gate, es)])
                   ) / len(list(LATE))

    pairs = list(itertools.combinations(MODELS, 2))
    rows = []
    for m1, m2 in pairs:
        row = {"pair": f"{DISPLAY[m1]} / {DISPLAY[m2]}",
               "prior_w1": w1(prior_of[m1], prior_of[m2])}
        for arm in ARMS:
            for gate in GATES:
                for es in ESS:
                    row[f"eq_w1_{arm}_ea{_num(gate)}_es{_num(es)}"] \
                        = eq_w1(m1, m2, arm, gate, es)
        rows.append(row)
    med = {"pair": "median (15 pairs)"}
    for key in rows[0]:
        if key != "pair":
            med[key] = statistics.median(r[key] for r in rows)
    rows.append(med)

    os.makedirs(args.out_dir, exist_ok=True)
    keys = list(rows[0])
    with open(os.path.join(args.out_dir, "fam_gate_pairs.csv"), "w",
              newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"[fam_gate] wrote fam_gate_pairs.csv ({len(rows)} rows)")

    # LaTeX appendix table (booktabs): prior W1 + both betas x gates
    gate_heads = " & ".join(f"{g:g}" for g in GATES)
    tex = [
        r"% AUTO-GENERATED by analyze_fam_gate_ablation.py -- do not",
        r"% edit by hand; rerun the analyzer.",
        r"\begin{tabular}{l r rrrr rrrr}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{$\beta=0$}"
        r" & \multicolumn{4}{c}{$\beta=1$} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r"pair & $W_1(M^0_i, M^0_j)$ & " + gate_heads + " & "
        + gate_heads + r" \\",
        r"\midrule",
    ]
    for r in rows:
        if r["pair"].startswith("median"):
            tex.append(r"\midrule")
        cells = [r["pair"], f"{r['prior_w1']:.3f}"]
        for arm in ARMS:
            for gate in GATES:
                cells.append(
                    f"{r[f'eq_w1_{arm}_ea{_num(gate)}_es0p05']:.3f}")
        tex.append(" & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    with open(os.path.join(args.out_dir, "fam_gate_pairs.tex"),
              "w") as fh:
        fh.write("\n".join(tex))
    print("[fam_gate] wrote fam_gate_pairs.tex")

    print("\n== within-family equilibrium W1 (rounds 25-29, seed 0; "
          "cols = ea " + "/".join(f"{g:g}" for g in GATES) + ") ==")
    for fam, (m1, m2) in FAMILIES:
        for arm in ARMS:
            vals = [eq_w1(m1, m2, arm, g) for g in GATES]
            print(f"  {fam:<8} {arm}: "
                  + "  ".join(f"{v:.4f}" for v in vals))

    if not args.no_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4),
                                 sharey=True)
        for ax, (fam, (m1, m2)) in zip(axes, FAMILIES):
            for arm, color, lab in (("b0", "tab:red", r"$\beta=0$"),
                                    ("b1", "tab:blue",
                                     r"$\beta=1$")):
                ax.plot(GATES, [eq_w1(m1, m2, arm, g) for g in GATES],
                        marker="o", ms=4, lw=1.5, color=color,
                        label=lab)
            ax.axhline(w1(prior_of[m1], prior_of[m2]), color="0.6",
                       lw=0.8, ls="--", label=r"prior $W_1$")
            ax.set_xscale("log")
            ax.set_xticks(GATES)
            ax.set_xticklabels([f"{g:g}" for g in GATES])
            ax.set_xlabel(r"$\varepsilon_{\mathrm{AI}}$")
            ax.text(0.04, 0.94,
                    f"{DISPLAY[m1]}\n{DISPLAY[m2]}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8)
        axes[0].set_ylabel(r"within-family equilibrium $W_1$")
        axes[0].legend(frameon=False, fontsize=8, loc="center left")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.out_dir,
                                     f"fam_gate_panels.{ext}"),
                        dpi=200 if ext == "png" else None)
        print("[fam_gate] wrote fam_gate_panels.png/pdf")

        # split-cell heatmaps over the full ea x es surface: beta=0
        # in the upper-left triangle, beta=1 in the lower-right;
        # left panel = median of the 3 within-family pairs, right =
        # median of all 15 pairs
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from matplotlib.patches import Polygon

        fam_pairs = [p for _, p in FAMILIES]

        def med_grid(arm, pool):
            return [[statistics.median(
                eq_w1(m1, m2, arm, g, e) for m1, m2 in pool)
                for g in GATES] for e in ESS]

        panels2 = [("within-family median", fam_pairs),
                   ("all-15-pair median", pairs)]
        fig2, axes2 = plt.subplots(1, 2, figsize=(9.6, 4.2))
        for ax, (lab, pool) in zip(axes2, panels2):
            ul = med_grid("b0", pool)     # beta = 0
            lr = med_grid("b1", pool)     # beta = 1
            vals = [v for gr_ in (ul, lr) for row in gr_
                    for v in row]
            norm = Normalize(vmin=min(vals), vmax=max(vals))
            cmap = plt.cm.viridis
            for j in range(len(ESS)):
                for i in range(len(GATES)):
                    c0 = (i - 0.5, j - 0.5)
                    c1 = (i + 0.5, j + 0.5)
                    ax.add_patch(Polygon(
                        [c0, (i - 0.5, j + 0.5), c1], closed=True,
                        facecolor=cmap(norm(ul[j][i])),
                        edgecolor="white", lw=0.5))
                    ax.add_patch(Polygon(
                        [c0, (i + 0.5, j - 0.5), c1], closed=True,
                        facecolor=cmap(norm(lr[j][i])),
                        edgecolor="white", lw=0.5))
                    ax.text(i - 0.19, j + 0.21, f"{ul[j][i]:.3f}",
                            ha="center", va="center", fontsize=5.6)
                    ax.text(i + 0.19, j - 0.21, f"{lr[j][i]:.3f}",
                            ha="center", va="center", fontsize=5.6)
            ax.text(0.02, 1.02, r"upper-left: $\beta=0$ · "
                                r"lower-right: $\beta=1$",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=7)
            fig2.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                          ax=ax, fraction=0.046, pad=0.04,
                          label=f"equilibrium $W_1$ ({lab})")
            ax.set_xlim(-0.5, len(GATES) - 0.5)
            ax.set_ylim(-0.5, len(ESS) - 0.5)
            ax.set_xticks(range(len(GATES)))
            ax.set_xticklabels([f"{g:g}" for g in GATES])
            ax.set_yticks(range(len(ESS)))
            ax.set_yticklabels([f"{e:g}" for e in ESS])
            ax.set_xlabel(r"$\varepsilon_{\mathrm{AI}}$")
            ax.set_ylabel(r"$\varepsilon_{\mathrm{social}}$")
            ax.set_aspect("equal")
        fig2.tight_layout()
        for ext in ("png", "pdf"):
            fig2.savefig(os.path.join(args.out_dir,
                                      f"fam_gate_grid.{ext}"),
                         dpi=220 if ext == "png" else None)
        print("[fam_gate] wrote fam_gate_grid.png/pdf")


if __name__ == "__main__":
    main()
