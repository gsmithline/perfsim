#!/usr/bin/env python3
"""GREEDY vs SAMPLED robustness, per model, for the Section-3
personal-history ICL wave.

Greedy decoding is the MAIN-PAPER result; sampled serving (DO_SAMPLE=1,
GEN_TEMPERATURE=1.0, one draw per agent per round) is the robustness
check. The two arms share every other setting, so a per-model gap here
is attributable to the decoder alone.

WHAT IS AND IS NOT CLAIMED. Greedy decoding on frozen weights is
deterministic given the prompt, so its across-seed interval reflects the
Deffuant sweep only. A SAMPLED trajectory fluctuates by construction:
this figure therefore plots, for the sampled arm, the final-window MEAN
with a bar spanning its TEMPORAL variability (the round-to-round spread
of the population mean over the last N rounds) -- not a confidence
interval, and never labelled an equilibrium. The greedy arm keeps its
across-seed 95% Student-t interval. The two bar meanings differ and the
legend says so.

Figures carry NO title (house rule); the caption block is written beside
the PDF.

  python plot_section3_icl_decoding.py \
      --greedy-dir notes/pofd/section3_model_icl_greedy \
      --sampled-dir notes/pofd/section3_model_icl_sample_t1
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "perfsim-s3-decode"))

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent

ORDER = ("qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b",
         "mistral7b", "ministral8b")
DISPLAY = {
    "qwen7b": "Qwen\n2.5", "qwen3_8b": "Qwen\n3",
    "olmo7b": "OLMo\n2", "olmo3_7b": "OLMo\n3",
    "mistral7b": "Mistral", "ministral8b": "Ministral",
}
GREEDY_C = "#4c72b0"
SAMPLED_C = "#dd8452"
REFERENCE = "#696d73"
INK = "#202328"


def load_arm(d: Path, label):
    rows_p, sum_p = d / "model_equilibria.csv", d / "summary.json"
    if not rows_p.exists() or not sum_p.exists():
        sys.exit(f"[s3decode] {label}: analysis outputs absent under {d} -- "
                 f"run analyze_section3_model_equilibria.py --wave icl "
                 f"--decode ... first")
    with rows_p.open() as fh:
        rows = {r["model"]: r for r in csv.DictReader(fh)}
    if set(rows) != set(ORDER):
        sys.exit(f"[s3decode] {label}: expected the six models, got "
                 f"{sorted(rows)}")
    return rows, json.loads(sum_p.read_text())


def num(row, *names, default=None):
    for n in names:
        if n in row and row[n] not in ("", None):
            return float(row[n])
    if default is not None:
        return default
    sys.exit(f"[s3decode] none of {names} present in the analysis row")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--greedy-dir",
                    default=str(REPO / "notes" / "pofd"
                               / "section3_model_icl_greedy"))
    ap.add_argument("--sampled-dir",
                    default=str(REPO / "notes" / "pofd"
                               / "section3_model_icl_sample_t1"))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    g_rows, g_sum = load_arm(Path(args.greedy_dir), "greedy")
    s_rows, s_sum = load_arm(Path(args.sampled_dir), "sampled")
    out = Path(args.out_dir) if args.out_dir else (
        REPO / "notes" / "pofd" / "section3_icl_decoding")
    out.mkdir(parents=True, exist_ok=True)

    if s_sum.get("stochastic_arm") is not True:
        print("[s3decode] NOTE: the sampled directory does not declare "
              "stochastic_arm -- check it was analysed with "
              "--decode sample_t1", file=sys.stderr)
    perfect = float(g_sum["perfect_prediction_mean"])
    p_s = float(s_sum["perfect_prediction_mean"])
    if abs(perfect - p_s) > 1e-9:
        sys.exit(f"[s3decode] REFUSING: the arms disagree on the "
                 f"perfect-prediction mean ({perfect:.8f} vs {p_s:.8f})")

    recs = []
    for m in ORDER:
        gr = g_rows[m]
        sr = s_rows[m]
        g_mu = num(gr, "equilibrium_mean")
        g_lo, g_hi = num(gr, "ci95_low"), num(gr, "ci95_high")
        # the sampled arm is summarised by its final-window mean and the
        # TEMPORAL spread of that window -- not by a settled value
        s_mu = num(sr, "late_mean", "equilibrium_mean")
        s_t = num(sr, "temporal_sd", default=0.0)
        recs.append({
            "model": m, "greedy_mean": g_mu,
            "greedy_ci95_low": g_lo, "greedy_ci95_high": g_hi,
            "sampled_late_mean": s_mu, "sampled_temporal_sd": s_t,
            "sampled_round30_mean": num(sr, "round30_mean", "final_mean",
                                        default=float("nan")),
            "sampled_pop_sd_final": num(sr, "pop_sd_final", "final_postpeer_sd",
                                        default=float("nan")),
            "sampled_late_drift": num(sr, "late_drift", default=float("nan")),
            "sampled_minus_greedy": s_mu - g_mu,
            "perfect_prediction": perfect,
        })

    csv_p = out / "section3_icl_decoding.csv"
    with csv_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print(f"[s3decode] wrote {csv_p}")

    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": .8,
                         "xtick.major.width": .8, "ytick.major.width": .8,
                         "text.color": INK, "axes.labelcolor": INK})
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    x = np.arange(len(ORDER), dtype=float)
    off = .16
    ax.axhline(perfect, color=REFERENCE, lw=1.35, ls=(0, (4, 2.5)), zorder=1)
    for i, r in enumerate(recs):
        ax.plot([x[i] - off, x[i] + off],
                [r["greedy_mean"], r["sampled_late_mean"]],
                color="0.72", lw=1.0, zorder=2)
        lo, hi = r["greedy_ci95_low"], r["greedy_ci95_high"]
        if math.isfinite(lo) and math.isfinite(hi):
            ax.plot([x[i] - off] * 2, [lo, hi], color=GREEDY_C, lw=1.3,
                    solid_capstyle="round", zorder=3)
        t = r["sampled_temporal_sd"]
        if math.isfinite(t):
            ax.plot([x[i] + off] * 2,
                    [r["sampled_late_mean"] - t, r["sampled_late_mean"] + t],
                    color=SAMPLED_C, lw=1.3, solid_capstyle="round",
                    zorder=3)
        ax.plot([x[i] - off], [r["greedy_mean"]], "o", ms=4.6,
                color=GREEDY_C, mec="white", mew=.7, zorder=4)
        ax.plot([x[i] + off], [r["sampled_late_mean"]], "s", ms=4.4,
                color=SAMPLED_C, mec="white", mew=.7, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in ORDER])
    ax.set_ylabel("Population mean", labelpad=4)
    ax.set_xlim(-.6, len(ORDER) - .4)
    vals = [v for r in recs
            for v in (r["greedy_mean"], r["sampled_late_mean"])]
    lo_, hi_ = min(vals + [perfect]), max(vals + [perfect])
    pad = max(.02, .14 * (hi_ - lo_ if hi_ > lo_ else .1))
    ax.set_ylim(lo_ - pad, hi_ + pad)
    ax.text(-.52, perfect + .006, "perfect prediction", fontsize=7,
            color=REFERENCE, va="bottom")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], color=GREEDY_C, marker="o", ms=4.6, lw=1.3,
               label="greedy (main): equilibrium, 95% across-seed CI"),
        Line2D([], [], color=SAMPLED_C, marker="s", ms=4.4, lw=1.3,
               label="sampled $T{=}1$: final-window mean $\\pm$ temporal SD"),
    ], frameon=False, fontsize=6.8, loc="best", handlelength=1.6)
    fig.tight_layout()
    pdf, png = (out / "section3_icl_decoding.pdf",
                out / "section3_icl_decoding.png")
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"[s3decode] wrote {pdf} and {png}")

    lw = recs and s_rows[ORDER[0]].get("late_window", "10")
    cap = [
        "Decoding robustness for personal-history ICL. Circles: greedy",
        "serving (DO_SAMPLE=0), the main-paper arm -- the settled",
        "equilibrium with its across-seed 95% Student-t interval over",
        "seeds {0,42,43}. Squares: sampled serving (DO_SAMPLE=1,",
        "GEN_TEMPERATURE=1.0, one draw per agent per round) -- the mean",
        f"of the final {lw} round means, with a bar spanning the TEMPORAL",
        "standard deviation of those round means. THE TWO BARS MEAN",
        "DIFFERENT THINGS: seed uncertainty for greedy, round-to-round",
        "fluctuation for sampled. A sampled trajectory is not required to",
        "settle and no sampled value here is called an equilibrium. Both",
        "arms share MovieLens/Action, 723 agents, 30 rounds, S=100 sweeps,",
        "W=1, k=1, alpha=.5, both gates all_open, anch2, frozen weights,",
        "ICL_K=0, ICL_DAYS=8. Dashed line: perfect prediction",
        f"({perfect:.4f}), the common innate mean.",
    ]
    (out / "section3_icl_decoding_caption.txt").write_text(
        "\n".join(cap) + "\n")
    (out / "section3_icl_decoding_summary.json").write_text(json.dumps({
        "perfect_prediction_mean": perfect,
        "greedy_wave": g_sum.get("wave_key", g_sum.get("wave")),
        "sampled_wave": s_sum.get("wave_key", s_sum.get("wave")),
        "sampled_settling_required": s_sum.get("settling_required"),
        "models": recs}, indent=2))

    hdr = (f"{'model':<12} {'greedy':>9} {'sampled':>9} {'S-G':>8} "
           f"{'tempSD':>8} {'popSD':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in recs:
        print(f"{r['model']:<12} {r['greedy_mean']:>9.4f} "
              f"{r['sampled_late_mean']:>9.4f} "
              f"{r['sampled_minus_greedy']:>+8.4f} "
              f"{r['sampled_temporal_sd']:>8.4f} "
              f"{r['sampled_pop_sd_final']:>8.4f}")
    print(f"\nperfect prediction = {perfect:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
