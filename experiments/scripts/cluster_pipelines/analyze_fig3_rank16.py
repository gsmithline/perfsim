#!/usr/bin/env python3
"""LoRA-RANK ROBUSTNESS for Figure 3: does the qualitative reference
pull survive at a conventional rank?

WHAT IS AND IS NOT BEING CLAIMED.  The registered question is
qualitative: at r=16, does raising lambda from 0 to 2 still pull the
served map and the population toward the reference?  It is NOT a claim
that the transition in lambda is rank-invariant.  The exact means, the
location of the transition and the amount of displacement are all
allowed to differ between ranks, so this script deliberately does NOT
test for linear interpolation between the endpoints, nor for monotone
placement of lambda=2 between lambda=0 and lambda=inf.

THE CELLS.  beta = gamma = 1, alpha = .5, both gates open, anch2,
Qwen3-8B, MovieLens/Action 723 agents, 30 rounds, S=100 sweeps, fresh
LoRA and fresh optimizer each round, seed 0.

  r=16   lambda 0, 2    NEW (2 jobs)
  r=512  lambda 0, 2    REUSED archived cells, never rerun
  lambda=inf            the frozen endpoint -- RANK-INDEPENDENT, since a
                        frozen model instantiates no adapter at all, so
                        ONE point is shared by both ranks

THE TWO DISTANCES, and why both are needed.
  d(served, entering map)   how reference-like the served map is. The
                            entering map is the untrained Qwen3-8B's own
                            predictions (pofdzsprior_..., frozen, no
                            context, same prompt) -- the same vector the
                            lambda=inf endpoint replays.
  d(served, live labels)    how well the platform FIT what it was
                            trained on this round (labels_t = the
                            previous round's post-peer opinions; innate
                            at t=0).
The second is what makes check 4 possible: if r=16 ordinary SFT cannot
fit the live labels, then a rank-16 result that merely LOOKS
reference-like is confounded by UNDERCAPACITY and is not evidence of
retention.  That verdict is computed here, not asserted.

Figures carry NO title (house rule); the caption block is written
beside the PDF.

  python analyze_fig3_rank16.py
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "perfsim-f3r"))

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "condor"))

LATE = 5
RANKS = (16, 512)
LAMS = (0.0, 2.0)
FROZEN_SOURCE = "pofdzsprior_qwen3_8b_w0p5_l0p2_es0_s0"
FROZEN_REPLAY = "frz_k1_w1_eaopen_esopen_sw100_s0_r30.pt"
ROUNDS = 30
R16_C = "#c44e52"
R512_C = "#4c72b0"
REF_C = "#6f7378"
INK = "#202328"
# UNDERCAPACITY CRITERION. A ratio alone will not do: at beta=1 the
# rank-512 SFT arm sits at a fixed point and fits the live labels to
# EXACTLY 0.0000, so any nonzero rank-16 error divides by zero and would
# be branded a confound. The test is therefore: rank-16 ordinary SFT is
# undercapacity-confounded iff its late-window label-fit error exceeds
# BOTH an absolute floor AND a multiple of the rank-512 arm's. The
# absolute floor is set at the serving resolution scale -- the model
# emits ~2-decimal values, so an error of a few hundredths is the grid,
# not a failure to learn.
UNDERCAP_RATIO = 2.0
UNDERCAP_ABS = 0.05


def stats(op):
    means = op.mean(axis=1)
    tail = means[-LATE:]
    half = LATE // 2
    return {
        "final_mean": float(means[-1]),
        "final_sd": float(op[-1].std(ddof=0)),
        "late_mean": float(tail.mean()),
        "late_sd": float(np.mean(op[-LATE:].std(axis=1, ddof=0))),
        "late_range": float(tail.max() - tail.min()),
        "drift": float(tail[-half:].mean() - tail[:half].mean()),
    }


def served_stats(pred, op, innate, frozen):
    """Late-window served-map diagnostics."""
    lo = pred.shape[0] - LATE
    d_fz, d_lab, card, mode = [], [], [], []
    for t in range(lo, pred.shape[0]):
        d_fz.append(float(np.abs(pred[t] - frozen).mean()))
        labels = innate if t == 0 else op[t - 1]
        d_lab.append(float(np.abs(pred[t] - labels).mean()))
        vals = np.round(pred[t], 6)
        c = Counter(vals.tolist())
        card.append(len(c))
        mode.append(max(c.values()) / float(vals.size))
    return {
        "d_frozen_late": float(np.mean(d_fz)),
        "d_frozen_final": float(d_fz[-1]),
        "d_labels_late": float(np.mean(d_lab)),
        "d_labels_final": float(d_lab[-1]),
        "pred_cardinality_final": int(card[-1]),
        "pred_cardinality_late_mean": float(np.mean(card)),
        "pred_mode_share_final": float(mode[-1]),
        "pred_mode_share_late_mean": float(np.mean(mode)),
    }


def telemetry_stats(run_dir, rounds):
    p = Path(run_dir) / "telemetry.json"
    if not p.exists():
        return {}
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    rows = [r for r in rows if int(r.get("round", -1)) < rounds]
    out = {}
    for key in ("n_train", "l_init", "w_norm", "b_norm", "ba_norm",
                "grad_norm0", "grad_kl_norm0", "l_cc"):
        vals = [float(r[key]) for r in rows
                if r.get(key) is not None and np.isfinite(float(r[key]))]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_min"] = float(np.min(vals))
    out["n_training_rounds"] = len(
        [r for r in rows if r.get("l_init") is not None])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-root",
                    default=str(REPO / "notes" / "pofd" / "cluster"))
    ap.add_argument("--frozen-dir",
                    default=str(REPO / "notes" / "pofd" / "frozen_replay"))
    ap.add_argument("--out-dir",
                    default=str(REPO / "notes" / "pofd" / "fig3_rank16"))
    args = ap.parse_args()
    root, out = Path(args.runs_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import gen_pofd_sweep as g

    errs = []
    fz = torch.load(root / FROZEN_SOURCE / "trajectory.pt",
                    map_location="cpu", weights_only=False)
    if (fz.get("config") or {}).get("training_style") != "frozen":
        errs.append("the entering map's source run is not frozen")
    frozen = fz["pred_raw"][0].float().numpy()
    innate = fz["innate"].float().numpy()
    perfect = float(innate.mean())

    recs = []
    for rank in RANKS:
        for lam in LAMS:
            tag = (g.F3R_REUSED[lam] if rank == g.F3_RANK
                   else g.f3_tag(g.F3R_BETA, g.F3R_GAMMA, lam,
                                 lora_r=rank))
            p = root / tag / "trajectory.pt"
            if not p.exists():
                errs.append(f"MISSING r={rank} lambda={lam:g}: {p}")
                continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            cfg = d.get("config", {}) or {}
            if int(cfg.get("lora_r", -1)) != rank:
                errs.append(f"{tag}: lora_r={cfg.get('lora_r')} != {rank}")
            if abs(float(cfg.get("kl_beta", -1)) - lam) > 1e-12:
                errs.append(f"{tag}: kl_beta={cfg.get('kl_beta')}")
            if not np.array_equal(d["innate"].float().numpy(), innate):
                errs.append(f"{tag}: innate differs from the reference")
            op = d["op_raw"].float().numpy()[:ROUNDS]
            pred = d["pred_raw"].float().numpy()[:ROUNDS]
            rec = {"rank": rank, "lam": lam, "tag": tag,
                   "provenance": ("reused_r512" if rank == g.F3_RANK
                                  else "new_2026_08_28"),
                   "rounds_used": int(op.shape[0])}
            rec.update(stats(op))
            rec.update(served_stats(pred, op, innate, frozen))
            rec.update(telemetry_stats(p.parent, ROUNDS))
            recs.append(rec)

    # lambda = inf: ONE shared, rank-independent endpoint
    fr = torch.load(Path(args.frozen_dir) / FROZEN_REPLAY,
                    map_location="cpu", weights_only=False)
    fop = fr["op_raw"].float().numpy()[:ROUNDS]
    fpred = fr["pred_raw"].float().numpy()[:ROUNDS]
    inf_rec = {"rank": "shared", "lam": float("inf"),
               "tag": FROZEN_REPLAY, "provenance": "frozen_shared",
               "rounds_used": int(fop.shape[0])}
    inf_rec.update(stats(fop))
    inf_rec.update(served_stats(fpred, fop, innate, frozen))
    recs.append(inf_rec)

    with (out / "fig3_rank16.csv").open("w", newline="") as fh:
        keys = sorted({k for r in recs for k in r},
                      key=lambda k: (k not in ("rank", "lam", "tag",
                                               "provenance"), k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in recs:
            w.writerow({k: (f"{v:.8f}" if isinstance(v, float) else
                            r.get(k, "")) for k, v in
                        {kk: r.get(kk, "") for kk in keys}.items()})

    def get(rank, lam):
        return next((r for r in recs
                     if r["rank"] == rank and r["lam"] == lam), None)

    # ---- the registered qualitative checks, COMPUTED ------------------
    checks = {}
    a16, b16 = get(16, 0.0), get(16, 2.0)
    a512, b512 = get(512, 0.0), get(512, 2.0)
    if a16 and a512:
        ratio = (a16["d_labels_late"] / a512["d_labels_late"]
                 if a512["d_labels_late"] > 0 else float("inf"))
        confounded = bool(a16["d_labels_late"] > UNDERCAP_ABS
                          and ratio > UNDERCAP_RATIO)
        checks["4_undercapacity_confound"] = {
            "r16_label_fit": a16["d_labels_late"],
            "r512_label_fit": a512["d_labels_late"],
            "ratio_r16_over_r512": ratio,
            "abs_threshold": UNDERCAP_ABS,
            "ratio_threshold": UNDERCAP_RATIO,
            "criterion": "confounded iff label-fit exceeds BOTH the "
                         "absolute floor and the ratio threshold; the "
                         "ratio alone divides by zero when rank 512 sits "
                         "at an exact fixed point",
            "confounded": confounded,
            "reading": ("rank-16 ordinary SFT fits the live labels to "
                        "within the serving resolution, so a "
                        "reference-like result at lambda=2 is NOT "
                        "explained by undercapacity"
                        if not confounded else
                        "rank-16 ordinary SFT CANNOT fit the live labels; "
                        "the rank-16 comparison is CONFOUNDED BY "
                        "UNDERCAPACITY and is not evidence for retention"),
        }
        checks["1_r16_sft_learns_labels"] = {
            "d_labels_late": a16["d_labels_late"],
            "pop_late_mean": a16["late_mean"],
            "perfect_prediction": perfect,
            "gap_to_perfect": a16["late_mean"] - perfect,
            "r512_pop_late_mean": a512["late_mean"],
        }
    if a16 and b16:
        pulled = (b16["d_frozen_late"] < a16["d_frozen_late"])
        checks["2_r16_lambda2_more_reference_like"] = {
            "d_frozen_late_lambda0": a16["d_frozen_late"],
            "d_frozen_late_lambda2": b16["d_frozen_late"],
            "served_map_pulled_toward_reference": bool(pulled),
            "pop_late_lambda0": a16["late_mean"],
            "pop_late_lambda2": b16["late_mean"],
            "pop_moved_toward_frozen_endpoint": bool(
                abs(b16["late_mean"] - inf_rec["late_mean"])
                < abs(a16["late_mean"] - inf_rec["late_mean"])),
        }
    if a512 and b512:
        checks["r512_same_contrast_for_reference"] = {
            "d_frozen_late_lambda0": a512["d_frozen_late"],
            "d_frozen_late_lambda2": b512["d_frozen_late"],
            "served_map_pulled_toward_reference": bool(
                b512["d_frozen_late"] < a512["d_frozen_late"]),
        }
    checks["3_no_interpolation_required"] = (
        "the exact means, the transition location and the amount of "
        "displacement may differ between ranks; monotone placement of "
        "lambda=2 between the endpoints is NOT tested and NOT required")

    survives = bool(
        checks.get("2_r16_lambda2_more_reference_like", {})
        .get("served_map_pulled_toward_reference")
        and not checks.get("4_undercapacity_confound", {})
        .get("confounded", True))

    # ---- figure --------------------------------------------------------
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.size": 8.5, "axes.linewidth": .8,
                         "xtick.major.width": .8, "ytick.major.width": .8,
                         "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    xs = [0, 1, 2]
    xl = ["$\\lambda=0$", "$\\lambda=2$", "$\\lambda=\\infty$"]
    ax = axes[0]
    ax.axhline(perfect, color="#b9bdc2", lw=1.15, ls=(0, (4, 2.5)), zorder=1)
    for rank, c in ((16, R16_C), (512, R512_C)):
        ys = [get(rank, l)["late_mean"] for l in LAMS
              if get(rank, l)] + [inf_rec["late_mean"]]
        ax.plot(xs[:len(ys)], ys, "-o", ms=4.8, lw=1.2, color=c,
                mec="white", mew=.7, zorder=3)
    ax.plot([2], [inf_rec["late_mean"]], "D", ms=5.4, color=REF_C,
            mec="white", mew=.7, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(xl)
    ax.set_xlim(-.35, 2.35)
    ax.set_ylabel("population mean\nfinal five rounds", labelpad=3)
    ax.text(-.3, perfect + .006, "perfect prediction", fontsize=6.4,
            color="#8a8e93", va="bottom")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for rank, c in ((16, R16_C), (512, R512_C)):
        ys = [get(rank, l)["d_frozen_late"] for l in LAMS
              if get(rank, l)] + [inf_rec["d_frozen_late"]]
        ax.plot(xs[:len(ys)], ys, "-o", ms=4.8, lw=1.2, color=c,
                mec="white", mew=.7, zorder=3)
    ax.plot([2], [inf_rec["d_frozen_late"]], "D", ms=5.4, color=REF_C,
            mec="white", mew=.7, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(xl)
    ax.set_xlim(-.35, 2.35)
    ax.set_ylabel("served map distance from\n"
                  "the entering map (final five rounds)", labelpad=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[
        Line2D([], [], color=R16_C, marker="o", ms=4.8, lw=1.2,
               label="LoRA $r=16$ (new)"),
        Line2D([], [], color=R512_C, marker="o", ms=4.8, lw=1.2,
               label="LoRA $r=512$ (the figure's rank, reused)"),
        Line2D([], [], color=REF_C, marker="D", ms=5.0, lw=0,
               label="$\\lambda=\\infty$: frozen, rank-independent"),
    ], frameon=False, fontsize=6.7, loc="lower center", ncol=3,
        bbox_to_anchor=(.5, .005))
    fig.tight_layout(rect=(0, .105, 1, 1))
    fig.savefig(out / "fig3_rank16.pdf")
    fig.savefig(out / "fig3_rank16.png", dpi=200)
    plt.close(fig)

    payload = {
        "question": "does the QUALITATIVE reference pull survive at a "
                    "conventional LoRA rank?",
        "not_claimed": "that the transition in lambda is rank-invariant; "
                       "no interpolation or monotone placement is tested",
        "surface": {"model": "Qwen/Qwen3-8B", "dataset": "movielens/Action",
                    "n_agents": 723, "rounds": ROUNDS, "ab_sweeps": 100,
                    "beta_w_plat": 1.0, "gamma_innate_lambda": 1.0,
                    "deffuant_alpha": 0.5, "gates": "both all_open",
                    "operator": "nested_ai_anchored_then_social_v2",
                    "lora_targets": "q_proj, v_proj",
                    "lora_dropout": 0.05,
                    "lora_alpha": "2r by construction (32 at r=16, "
                                  "1024 at r=512)",
                    "fresh_adapter_and_optimizer_each_round": True,
                    "seed": 0},
        "entering_frozen_map": {"tag": FROZEN_SOURCE, "mean": float(
            frozen.mean()), "sd": float(frozen.std())},
        "perfect_prediction_mean": perfect,
        "lambda_inf_is_rank_independent": (
            "a frozen model instantiates no adapter, so there is exactly "
            "one lambda=inf point and both ranks share it"),
        "registered_checks": checks,
        "qualitative_reference_pull_survives_at_r16": survives,
        "gate": {"errors": errs, "pass": not errs},
        "cells": recs,
    }
    (out / "fig3_rank16.json").write_text(json.dumps(payload, indent=2))

    cap = [
        "LoRA-rank robustness for Figure 3. Left: post-peer population",
        "mean over the final five rounds. Right: how far the served map",
        "sits from the ENTERING FROZEN MAP -- the untrained Qwen3-8B's own",
        "predictions on the same 723 agents, which is also what the",
        "lambda=infinity endpoint serves. Red: a conventional LoRA rank of",
        "16 (new). Blue: the rank the figure uses, 512 (reused, not",
        "rerun). The lambda=infinity point is a single diamond because a",
        "frozen model instantiates no adapter, so that endpoint is",
        "rank-independent and shared. LORA_ALPHA is 2r at both ranks by",
        "construction, so the scaling ratio is held constant. Everything",
        "else is identical: beta=gamma=1, alpha=.5, both gates open, anch2",
        "operator, 30 rounds, S=100 sweeps, q/v targets, dropout .05, a",
        "fresh adapter and a fresh optimizer every round, seed 0. THE TEST",
        "IS QUALITATIVE: whether raising lambda still pulls the served map",
        "toward the reference. The exact means, the transition location",
        "and the displacement may differ between ranks, and no",
        "interpolation between the endpoints is claimed or tested. Dashed",
        f"line: perfect prediction ({perfect:.4f}).",
    ]
    (out / "fig3_rank16_caption.txt").write_text("\n".join(cap) + "\n")

    if errs:
        print("[f3r] GATE FAIL:")
        for e in errs:
            print("   -", e)
    else:
        print("[f3r] GATE PASS: ranks and lambdas match the cells; "
              "innate identical everywhere")
    hdr = (f"{'rank':>7}{'lam':>7}{'pop_fin':>9}{'pop_late':>10}"
           f"{'sd_fin':>9}{'drift':>9}{'d_frozen':>10}{'d_labels':>10}"
           f"{'card':>6}{'mode':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in recs:
        print(f"{str(r['rank']):>7}{r['lam']:>7g}{r['final_mean']:>9.4f}"
              f"{r['late_mean']:>10.4f}{r['final_sd']:>9.5f}"
              f"{r['drift']:>+9.4f}{r['d_frozen_late']:>10.4f}"
              f"{r['d_labels_late']:>10.4f}"
              f"{r['pred_cardinality_final']:>6}"
              f"{r['pred_mode_share_final']:>7.3f}")
    print(f"\nperfect prediction = {perfect:.4f}   "
          f"entering frozen map mean = {frozen.mean():.4f}")
    for k in sorted(checks):
        v = checks[k]
        print(f"\n[{k}]")
        if isinstance(v, str):
            print("   ", v)
        else:
            for kk, vv in v.items():
                shown = f"{vv:.6f}" if isinstance(vv, float) else str(vv)
                print(f"    {kk} = {shown}")
    print(f"\nQUALITATIVE REFERENCE PULL SURVIVES AT r=16: {survives}")
    print(f"[f3r] wrote {out}/fig3_rank16.{{csv,json,pdf,png}} + "
          f"_caption.txt")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
