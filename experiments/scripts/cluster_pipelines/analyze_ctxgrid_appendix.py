#!/usr/bin/env python3
"""APPENDIX CANDIDATE: population-sampled context opens a cross-agent
route that personal-history-only context does not.

THE CONTROL.  Every arm below has FROZEN weights, so nothing the
platform learns can travel through parameters; the only thing that
varies is what the prompt is allowed to contain.

  k0   no context at all -- the platform's map is a static function of
       the agent's own profile
  fz0  K=8 exemplars drawn from OTHER agents, SNAPSHOT at round 0
  dyn  K=8 exemplars drawn from OTHER agents, REFRESHED every round
  f32  K=32 exemplars, snapshot at round 0
  d32  K=32 exemplars, refreshed every round

THE IDENTIFICATION IS live MINUS fixed, WITHIN A CELL.  fz0 and dyn are
identical in every respect -- same model, same weights, same K, same
prompt length, same gate, same peer dose -- except whether the
exemplars reflect THIS round's population or round 0's.  Only the live
arm can carry the current population into another agent's prompt, so a
systematic live-minus-fixed difference IS the cross-agent route, and it
cannot be attributed to context length, to the model, or to training.
The same contrast is repeated at K=32.

Two metrics, both from the existing context-grid analysis:
  late_std_ratio  rounds 25-29 mean of std(op) / std(no-AI twin).
                  Below 1 the platform CONTRACTED the population
                  relative to its twin; a live route should contract
                  more than a frozen snapshot.
  final_w1_twin   W1 displacement of the final population from its
                  matched no-AI twin. A live route should displace more.

PERSONAL-HISTORY-ONLY CONTEXT IS A DIFFERENT SURFACE.  The
personal-history arm (ICL_K=0, ICL_DAYS=8: each agent sees only its own
past) exists only in the Section-3 wave at W=1, k=1, S=100, both gates
open -- not on this grid's W=0.5, k=0.2 surface. It is therefore quoted
as a QUALITATIVE contrast in the JSON and never plotted on the same
axes, because no matched cell exists.

EXCLUSIONS, and why.  Mistral is dropped entirely:
  * f32/d32 are PARSER-CONTAMINATED. Under a 32-exemplar prompt Mistral
    stops emitting a number and explains instead; parse_fail_frac is
    1.0 in every round and every agent takes the parser's 0.5 default,
    so the trajectory records an instruction-following failure, not an
    opinion. Already marked `excluded` by the grid analysis.
  * k0/fz0 are STRUCTURALLY INCOMPLETE -- 7 of 20 grid cells present, so
    no paired live-minus-fixed contrast can be formed over the grid.
Qwen 2.5 and OLMo 2 are complete (20/20 in all five arms), which is why
no replacement grid is needed.

SINGLE SEED.  The whole grid is seed 0. Every number here is
DESCRIPTIVE: no confidence interval is computed and none should be
read in. The cell-level sign counts (out of 20 gate x peer cells) are
reported precisely so a reader can see how consistent the effect is
without a replicate to lean on.

Figures carry NO title (house rule); the caption block is written
beside the PDF.

  python analyze_ctxgrid_appendix.py
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "perfsim-ctxapp"))

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent

MODELS = ("qwen7b", "olmo7b")
LABEL = {"qwen7b": "Qwen 2.5", "olmo7b": "OLMo 2"}
ARMS = ("k0", "fz0", "dyn", "f32", "d32")
ARM_LABEL = {"k0": "none",
             "fz0": "fixed\n$K{=}8$", "dyn": "live\n$K{=}8$",
             "f32": "fixed\n$K{=}32$", "d32": "live\n$K{=}32$"}
PAIRS = (("dyn", "fz0", 8), ("d32", "f32", 32))
N_CELLS = 20
FIXED_C = "#4c72b0"
LIVE_C = "#c44e52"
NONE_C = "#7a7e83"
INK = "#202328"
EXCLUDED = {
    "mistral7b/f32": "parser-contaminated: parse_fail_frac 1.0 in every "
                     "round (100% digit-free generations under a "
                     "32-exemplar prompt); every agent served the 0.5 "
                     "parser default",
    "mistral7b/d32": "parser-contaminated: same failure as f32",
    "mistral7b/k0": "structurally incomplete: 7 of 20 grid cells present",
    "mistral7b/fz0": "structurally incomplete: 7 of 20 grid cells present",
}


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-cell",
                    default=str(REPO / "notes" / "pofd" / "ctxgrid_analysis"
                               / "ctxgrid_per_cell.csv"))
    ap.add_argument("--channel-summary",
                    default=str(REPO / "notes" / "pofd" / "ctxgrid_analysis"
                               / "ctxgrid_channel_summary.csv"))
    ap.add_argument("--gate-json", default=None,
                    help="check_pofd_sanity result to embed, if collected")
    ap.add_argument("--out-dir",
                    default=str(REPO / "notes" / "pofd"
                               / "ctxgrid_appendix"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.per_cell, newline="") as fh:
        rows = list(csv.DictReader(fh))
    with open(args.channel_summary, newline="") as fh:
        chan = {(r["model"], r["arm"]): r for r in csv.DictReader(fh)}

    by = {}
    seeds = set()
    for r in rows:
        by[(r["model"], r["arm"], r["gate"], r["eps_social"])] = r
        seeds.add(r["seed"])
    if seeds != {"0"}:
        print(f"[ctxapp] NOTE: seeds present = {sorted(seeds)}",
              file=sys.stderr)

    errs = []
    for m in MODELS:
        for a in ARMS:
            got = sum(1 for r in rows if r["model"] == m and r["arm"] == a
                      and r["found"] == "1")
            if got != N_CELLS:
                errs.append(f"{m}/{a}: {got} of {N_CELLS} cells found -- "
                            f"the paired contrast needs the full grid")

    # ---- arm-level, over the 20 gate x peer cells --------------------
    arm_rows = []
    for m in MODELS:
        for a in ARMS:
            vals = [num(by[(m, a, g, e)]["late_std_ratio"])
                    for (mm, aa, g, e) in by if mm == m and aa == a]
            w1s = [num(by[(m, a, g, e)]["final_w1_twin"])
                   for (mm, aa, g, e) in by if mm == m and aa == a]
            vals = [v for v in vals if v is not None]
            w1s = [v for v in w1s if v is not None]
            arm_rows.append({
                "model": m, "model_label": LABEL[m], "arm": a,
                "arm_label": chan.get((m, a), {}).get("arm_label", a),
                "n_cells": len(vals),
                "late_std_ratio_mean": st.mean(vals),
                "late_std_ratio_min": min(vals),
                "late_std_ratio_max": max(vals),
                "final_w1_twin_mean": st.mean(w1s),
                "final_w1_twin_min": min(w1s),
                "final_w1_twin_max": max(w1s),
            })

    # ---- the identification: live minus fixed, WITHIN a cell ---------
    pair_rows, pair_sum = [], []
    for m in MODELS:
        cells = sorted({(g, e) for (mm, aa, g, e) in by
                        if mm == m and aa == "dyn"})
        for live, fixed, K in PAIRS:
            d_sd, d_w1 = [], []
            for g, e in cells:
                a, b = by.get((m, live, g, e)), by.get((m, fixed, g, e))
                if not a or not b:
                    continue
                s_l, s_f = num(a["late_std_ratio"]), num(b["late_std_ratio"])
                w_l, w_f = num(a["final_w1_twin"]), num(b["final_w1_twin"])
                if None in (s_l, s_f, w_l, w_f):
                    continue
                d_sd.append(s_l - s_f)
                d_w1.append(w_l - w_f)
                pair_rows.append({
                    "model": m, "model_label": LABEL[m], "K": K,
                    "ai_gate": g, "eps_social": e,
                    "live_late_std_ratio": s_l, "fixed_late_std_ratio": s_f,
                    "d_late_std_ratio": s_l - s_f,
                    "live_final_w1_twin": w_l, "fixed_final_w1_twin": w_f,
                    "d_final_w1_twin": w_l - w_f,
                })
            pair_sum.append({
                "model": m, "model_label": LABEL[m], "K": K,
                "n_cells": len(d_sd),
                "d_late_std_ratio_mean": st.mean(d_sd),
                "d_late_std_ratio_median": st.median(d_sd),
                "n_cells_contracting": sum(1 for v in d_sd if v < 0),
                "d_final_w1_twin_mean": st.mean(d_w1),
                "d_final_w1_twin_median": st.median(d_w1),
                "n_cells_displacing": sum(1 for v in d_w1 if v > 0),
            })

    for name, data in (("ctxgrid_appendix_arms.csv", arm_rows),
                       ("ctxgrid_appendix_paired_cells.csv", pair_rows),
                       ("ctxgrid_appendix_paired_summary.csv", pair_sum)):
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            for r in data:
                w.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                            for k, v in r.items()})

    # ---- figure ------------------------------------------------------
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": .8,
                         "xtick.major.width": .8, "ytick.major.width": .8,
                         "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7),
                             gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
    x = np.arange(len(ARMS), dtype=float)
    for ax, m in zip(axes[:2], MODELS):
        ax.axhline(1.0, color="#b9bdc2", lw=1.0, ls=(0, (4, 2.5)), zorder=1)
        for i, a in enumerate(ARMS):
            r = next(v for v in arm_rows
                     if v["model"] == m and v["arm"] == a)
            c = (NONE_C if a == "k0" else
                 LIVE_C if a in ("dyn", "d32") else FIXED_C)
            ax.plot([x[i]] * 2, [r["late_std_ratio_min"],
                                 r["late_std_ratio_max"]],
                    color=c, lw=1.2, alpha=.55, solid_capstyle="round",
                    zorder=2)
            ax.plot([x[i]], [r["late_std_ratio_mean"]], "o", ms=4.8,
                    color=c, mec="white", mew=.7, zorder=3)
        for live, fixed, _K in PAIRS:
            i, j = ARMS.index(fixed), ARMS.index(live)
            ri = next(v for v in arm_rows
                      if v["model"] == m and v["arm"] == fixed)
            rj = next(v for v in arm_rows
                      if v["model"] == m and v["arm"] == live)
            ax.annotate("", xy=(x[j], rj["late_std_ratio_mean"]),
                        xytext=(x[i], ri["late_std_ratio_mean"]),
                        arrowprops=dict(arrowstyle="->", lw=.9,
                                        color="#6f7378", shrinkA=4,
                                        shrinkB=4), zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=6.0)
        ax.set_xlim(-.6, len(ARMS) - .4)
        # in-axes panel key, NOT a figure title (house rule: no titles)
        ax.text(.02, .04, LABEL[m], transform=ax.transAxes, fontsize=7.6,
                color="#3a3e43", va="bottom", ha="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("std(population) / std(no-AI twin)\n"
                       "rounds 25-29", labelpad=3)
    for ax in axes[:2]:
        ax.set_xlabel("population-sampled context", labelpad=2,
                      fontsize=7.2)

    ax = axes[2]
    ax.axhline(0.0, color="#b9bdc2", lw=1.0, ls=(0, (4, 2.5)), zorder=1)
    pos, ticks = [], []
    for k, (m, K) in enumerate([(m, K) for m in MODELS
                                for _, _, K in PAIRS]):
        vals = [r["d_late_std_ratio"] for r in pair_rows
                if r["model"] == m and r["K"] == K]
        jit = (np.arange(len(vals)) - len(vals) / 2) * .012
        ax.plot(k + jit, vals, "o", ms=2.6, color=LIVE_C, alpha=.6,
                mec="none", zorder=2)
        ax.plot([k - .26, k + .26], [st.mean(vals)] * 2, color=INK,
                lw=1.4, solid_capstyle="round", zorder=3)
        s = next(v for v in pair_sum if v["model"] == m and v["K"] == K)
        ax.annotate(f"{s['n_cells_contracting']}/{s['n_cells']}",
                    (k, max(vals)), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=6.2,
                    color="#4a4e53")
        pos.append(k)
        ticks.append(f"{LABEL[m].split()[0]}\n$K{{=}}{K}$")
    ax.set_xticks(pos)
    ax.set_xticklabels(ticks, fontsize=6.8)
    ax.set_xlim(-.6, len(pos) - .4)
    ax.set_ylabel("live $-$ fixed, within cell\n"
                  "$\\Delta$ std ratio", labelpad=3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.legend(handles=[
        Line2D([], [], color=NONE_C, marker="o", ms=4.8, lw=1.2,
               label="no context"),
        Line2D([], [], color=FIXED_C, marker="o", ms=4.8, lw=1.2,
               label="population context, round-0 snapshot"),
        Line2D([], [], color=LIVE_C, marker="o", ms=4.8, lw=1.2,
               label="population context, refreshed each round"),
    ], frameon=False, fontsize=6.8, loc="lower center", ncol=3,
        bbox_to_anchor=(.5, .005))
    fig.tight_layout(rect=(0, .10, 1, 1))
    pdf, png = (out / "ctxgrid_appendix.pdf", out / "ctxgrid_appendix.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    payload = {
        "claim": "population-sampled context opens a cross-agent route; "
                 "a round-0 snapshot of the same context does not, and "
                 "personal-history-only context cannot",
        "identification": (
            "live minus fixed WITHIN a gate x peer cell. The two arms "
            "match on model, frozen weights, K, prompt length, AI gate "
            "and peer dose; they differ only in whether the exemplars "
            "reflect this round's population. Only the live arm can "
            "carry the current population into another agent's prompt."),
        "surface": {"dataset": "movielens/Action", "n_agents": 723,
                    "w_plat": 0.5, "innate_lambda": 0.2, "seed": 0,
                    "grid": "5 AI gates x 4 eps_social = 20 cells",
                    "late_window": "rounds 25-29"},
        "status": "DESCRIPTIVE -- single seed, no confidence intervals; "
                  "cell-level sign counts are reported in their place",
        "metrics": {
            "late_std_ratio": "rounds 25-29 mean of std(op)/std(no-AI "
                              "twin); below 1 = contracted vs twin",
            "final_w1_twin": "W1 displacement of the final population "
                             "from its matched no-AI twin",
        },
        "personal_history_contrast": (
            "the personal-history-only arm (ICL_K=0, ICL_DAYS=8: each "
            "agent sees only its own past) exists only in the Section-3 "
            "wave at W=1, k=1, S=100, both gates open -- NOT on this "
            "grid's W=0.5, k=0.2 surface. There is no matched cell, so "
            "it is quoted qualitatively and never plotted on these axes. "
            "On its own surface that arm has no channel by which one "
            "agent's state can enter another agent's prompt, which is "
            "exactly the channel the live arms add here."),
        "excluded": EXCLUDED,
        "no_replacement_grid_needed": (
            "Qwen 2.5 and OLMo 2 are complete (20/20 cells in all five "
            "arms), so the control stands on the clean models"),
        "gate": {"errors": errs, "pass": not errs},
        "arms": arm_rows,
        "paired_live_minus_fixed": pair_sum,
    }
    if args.gate_json and Path(args.gate_json).exists():
        payload["check_pofd_sanity"] = json.loads(
            Path(args.gate_json).read_text())
    (out / "ctxgrid_appendix.json").write_text(json.dumps(payload, indent=2))

    cap = [
        "Population-sampled context opens a cross-agent route. All arms",
        "have FROZEN weights, so nothing can travel through parameters;",
        "only the prompt's contents vary. Left and centre: population",
        "spread over rounds 25-29 relative to the matched no-AI twin, per",
        "arm, as the mean over the 20 gate x peer cells with the",
        "cell-to-cell range; the dashed line is parity with the twin, and",
        "arrows run from a round-0 context snapshot to the same context",
        "refreshed each round. Right: the identifying contrast, live",
        "minus fixed WITHIN each cell -- the two arms match on model,",
        "weights, K, prompt length, gate and peer dose and differ only in",
        "whether the exemplars reflect the CURRENT population, so a",
        "systematic difference is the cross-agent route itself. One dot",
        "per cell; the bar is the mean; the annotation counts cells",
        "moving in the stated direction. The effect is unanimous in",
        "OLMo 2 (20/20 at both K) and directional but not cell-consistent",
        "in Qwen 2.5. MovieLens/Action, 723 agents, W=0.5, k=0.2, seed 0",
        "-- SINGLE SEED, so every value is descriptive and no confidence",
        "interval is implied. Mistral is excluded throughout: its K=32",
        "arms are parser-contaminated (100% digit-free generations, every",
        "agent served the parser's 0.5 default) and its K=0/K=8 arms cover",
        "only 7 of 20 cells.",
    ]
    (out / "ctxgrid_appendix_caption.txt").write_text("\n".join(cap) + "\n")

    if errs:
        print("[ctxapp] GATE FAIL:")
        for e in errs:
            print("   -", e)
    else:
        print(f"[ctxapp] GATE PASS: {len(MODELS)} clean models x "
              f"{len(ARMS)} arms x {N_CELLS} cells, all present")
    print(f"[ctxapp] excluded: {', '.join(sorted(EXCLUDED))}")
    hdr = (f"{'model':<10}{'arm':<5}{'n':>3}{'stdRatio':>10}"
           f"{'[min':>9}{'max]':>9}{'W1twin':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in arm_rows:
        print(f"{r['model_label']:<10}{r['arm']:<5}{r['n_cells']:>3}"
              f"{r['late_std_ratio_mean']:>10.4f}"
              f"{r['late_std_ratio_min']:>9.4f}"
              f"{r['late_std_ratio_max']:>9.4f}"
              f"{r['final_w1_twin_mean']:>9.4f}")
    hdr2 = (f"\n{'model':<10}{'K':>4}{'n':>4}{'d stdRatio':>12}"
            f"{'contracting':>13}{'d W1twin':>11}{'displacing':>12}")
    print(hdr2)
    print("-" * (len(hdr2) - 1))
    for s in pair_sum:
        print(f"{s['model_label']:<10}{s['K']:>4}{s['n_cells']:>4}"
              f"{s['d_late_std_ratio_mean']:>+12.4f}"
              f"{s['n_cells_contracting']:>9}/{s['n_cells']:<3}"
              f"{s['d_final_w1_twin_mean']:>+11.4f}"
              f"{s['n_cells_displacing']:>8}/{s['n_cells']:<3}")
    print(f"\n[ctxapp] wrote {out}/ctxgrid_appendix.{{pdf,png,json}} + "
          f"3 CSVs + _caption.txt")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
