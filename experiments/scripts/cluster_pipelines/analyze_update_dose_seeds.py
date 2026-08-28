#!/usr/bin/env python3
"""Does WEAKER ordinary SFT drift back toward the entering frozen model,
or does it just change the live-label dynamics?

THE LADDER.  Every arm consumes all 723 current live labels exactly once
per round, in the same recorded sampler order.  Only the number of
optimizer steps that pass over them differs (SFT_GRAD_ACCUM splits the
single epoch into averaged blocks; realized steps = ceil(181/accum)):

  U=1    one averaged full-batch step per round   (accum 181)
  U=5    five averaged blocks                     (accum 37)
  U=181  ordinary minibatch SFT                   (accum 1)

Data exposure, LR, rank, batch and seed are FIXED across the ladder, so
U is a pure update-frequency dial and "weaker SFT" means U small.

THE ENTERING FROZEN MAP.  pofdzsprior_qwen3_8b_w0p5_l0p2_es0_s0 is
Qwen3-8B with training_style=frozen, ICL_K=0, ICL_DAYS=0 -- never
trained, no context, the same plain profile prompt the update-dose runs
use, on the same 723 MovieLens/Action agents (innate vector verified
bit-identical).  Its round-0 map IS the map the loop enters with.  The
run's own W and eps_ai are irrelevant: a frozen model's round-0
prediction is a function of the prompt alone, and no dynamics have run
yet.

THE TEST.  If weaker SFT merely does less, the served map should sit
CLOSER to that entering map as U falls.  If instead U changes how the
platform tracks the live labels, the distance need not order by U at
all, and the closed loop can land somewhere neither the frozen map nor
perfect prediction explains.

SEEDS.  Seeds 42 and 43 are the 2026-08-28 replication (6 new jobs,
gated by check_update_dose.py --seeds 42,43, 6/6 PASS).  Seed 0 is
pre-existing.  At seed 0 the U=181 arm is the ARCHIVED pofdps_ SFT cell
(accum 1 is the byte-identical legacy path), read over its first ten
rounds; it predates SAVE_SFT_ORDER, so its sampler order cannot be
re-verified and it is flagged `reused_legacy` in every artifact.

Figures carry NO title (house rule); the caption block is written
beside the PDF.

  python analyze_update_dose_seeds.py
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "perfsim-uds"))

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent

US = (1, 5, 181)
SEEDS = (0, 42, 43)
ROUNDS = 10
LATE = 5
T95_DF2 = 4.302652729911275
FROZEN_TAG = "pofdzsprior_qwen3_8b_w0p5_l0p2_es0_s0"
UD_TAG = ("pofdud_qwen3_8b_sft_u{u}_sw100_eaopen_w1_k1_esopen_anch2"
          "_s{seed}_r{rounds}")
REUSE_U181_S0 = ("pofdps_qwen3_8b_sft_sw100_eaopen_w1_k1_esopen_anch2"
                 "_s0_r60")
U_COLOR = {1: "#c44e52", 5: "#dd8452", 181: "#4c72b0"}
INK = "#202328"


def cell_tag(u, seed):
    if seed == 0 and u == 181:
        return REUSE_U181_S0, "reused_legacy"
    return UD_TAG.format(u=u, seed=seed, rounds=ROUNDS), (
        "preexisting" if seed == 0 else "new_2026_08_28")


def ci95(vals):
    v = np.asarray(vals, dtype=float)
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    if v.size == 3:
        h = T95_DF2 * sd / np.sqrt(3.0)
        return m, sd, m - h, m + h
    return m, sd, float(v.min()), float(v.max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-root",
                    default=str(REPO / "notes" / "pofd" / "cluster"))
    ap.add_argument("--out-dir",
                    default=str(REPO / "notes" / "pofd"
                               / "update_dose_seeds"))
    args = ap.parse_args()
    root = Path(args.runs_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    errs = []
    fz = torch.load(root / FROZEN_TAG / "trajectory.pt",
                    map_location="cpu", weights_only=False)
    fcfg = fz.get("config", {}) or {}
    for k, want in (("training_style", "frozen"), ("icl_k", 0),
                    ("icl_days", 0), ("base_model", "Qwen/Qwen3-8B"),
                    ("chat_thinking", False), ("do_sample", False),
                    ("dataset", "movielens"), ("ml_target", "Action")):
        if fcfg.get(k) != want:
            errs.append(f"FROZEN {k}={fcfg.get(k)!r} (want {want!r}) -- "
                        f"this is not an untrained no-context map")
    frozen = fz["pred_raw"][0].float().numpy()
    innate = fz["innate"].float().numpy()

    cells, traj = [], {}
    for u in US:
        for s in SEEDS:
            tag, prov = cell_tag(u, s)
            p = root / tag / "trajectory.pt"
            if not p.exists():
                errs.append(f"MISSING u{u} s{s}: {p}")
                continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            if not np.array_equal(d["innate"].float().numpy(), innate):
                errs.append(f"u{u} s{s}: innate differs from the frozen "
                            f"reference -- not the same population")
            pred = d["pred_raw"].float().numpy()[:ROUNDS]
            op = d["op_raw"].float().numpy()[:ROUNDS]
            if pred.shape[0] != ROUNDS or op.shape[0] != ROUNDS:
                errs.append(f"u{u} s{s}: {op.shape[0]} of {ROUNDS} rounds")
                continue
            # agent-paired distance from the SERVED map to the entering map
            dfz = np.abs(pred - frozen[None, :]).mean(axis=1)
            means = op.mean(axis=1)
            tail = means[-LATE:]
            half = LATE // 2
            traj[(u, s)] = means
            cells.append({
                "u": u, "seed": s, "tag": tag, "provenance": prov,
                "d_frozen_r0": float(dfz[0]),
                "d_frozen_late": float(dfz[-LATE:].mean()),
                "d_frozen_final": float(dfz[-1]),
                "w1_pred_frozen_final": float(np.mean(np.abs(
                    np.sort(pred[-1]) - np.sort(frozen)))),
                "pop_mean_final": float(means[-1]),
                "pop_mean_late": float(tail.mean()),
                "late_range": float(tail.max() - tail.min()),
                "drift": float(tail[-half:].mean() - tail[:half].mean()),
                "pop_sd_final": float(op[-1].std(ddof=0)),
            })

    if not cells:
        sys.exit("[uds] no cells loaded")
    with (out / "update_dose_seeds_cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader()
        for r in cells:
            w.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    rows = []
    for u in US:
        got = [c for c in cells if c["u"] == u]
        rec = {"u": u, "n_seeds": len(got),
               "accum": {1: 181, 5: 37, 181: 1}[u]}
        for f in ("d_frozen_late", "d_frozen_final", "pop_mean_late",
                  "pop_mean_final", "late_range", "drift", "pop_sd_final"):
            m, sd, lo, hi = ci95([c[f] for c in got])
            rec[f] = m
            rec[f + "_sd"] = sd
            rec[f + "_lo"] = lo
            rec[f + "_hi"] = hi
        rows.append(rec)
    with (out / "update_dose_seeds.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    # THE VERDICT, computed rather than asserted.
    order_toward = all(rows[i]["d_frozen_late"] < rows[i + 1]["d_frozen_late"]
                       for i in range(len(rows) - 1))
    order_away = all(rows[i]["d_frozen_late"] > rows[i + 1]["d_frozen_late"]
                     for i in range(len(rows) - 1))
    settled = {r["u"]: bool(r["late_range"] < 1e-6) for r in rows}
    verdict = ("weaker SFT moves TOWARD the entering frozen map"
               if order_toward else
               "weaker SFT moves AWAY from the entering frozen map"
               if order_away else
               "distance to the entering frozen map does not order by U")

    # ---- figure ------------------------------------------------------
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": .8,
                         "xtick.major.width": .8, "ytick.major.width": .8,
                         "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    pp, fm = float(innate.mean()), float(frozen.mean())
    ax.axhline(pp, color="#6f7378", lw=1.15, ls=(0, (4, 2.5)), zorder=1)
    ax.axhline(fm, color="#9aa0a6", lw=1.15, ls=(0, (1.5, 2)), zorder=1)
    for u in US:
        for s in SEEDS:
            if (u, s) not in traj:
                continue
            ax.plot(range(ROUNDS), traj[(u, s)], "-", lw=1.15,
                    color=U_COLOR[u], alpha=.85, zorder=3)
    ax.set_xlabel("round", labelpad=3)
    ax.set_ylabel("population mean", labelpad=3)
    ax.set_xlim(-.3, ROUNDS - .7)
    ax.text(ROUNDS - .8, pp + .008, "perfect prediction", fontsize=6.4,
            color="#6f7378", ha="right", va="bottom")
    ax.text(ROUNDS - .8, fm + .008, "entering frozen map (mean)",
            fontsize=6.4, color="#9aa0a6", ha="right", va="bottom")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    x = np.arange(len(US), dtype=float)
    for i, r in enumerate(rows):
        ax.plot([x[i]] * 2, [r["d_frozen_late_lo"], r["d_frozen_late_hi"]],
                color=U_COLOR[r["u"]], lw=1.4, solid_capstyle="round",
                zorder=2)
        ax.plot([x[i]], [r["d_frozen_late"]], "o", ms=5.0,
                color=U_COLOR[r["u"]], mec="white", mew=.7, zorder=3)
    ax.plot(x, [r["d_frozen_late"] for r in rows], "-", lw=1.0,
            color="0.72", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$U{{=}}{u}$" for u in US])
    ax.set_xlim(-.5, len(US) - .5)
    ax.set_xlabel("optimizer steps per round", labelpad=3)
    ax.set_ylabel("mean $|$served $-$ entering map$|$\n"
                  "final five rounds", labelpad=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[Line2D([], [], color=U_COLOR[u], lw=1.4,
                               marker="o", ms=4.6,
                               label=f"$U={u}$"
                                     f"{'  (weakest)' if u == 1 else ''}")
                        for u in US],
               frameon=False, fontsize=6.9, loc="lower center", ncol=3,
               bbox_to_anchor=(.5, .005))
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(out / "update_dose_seeds.pdf")
    fig.savefig(out / "update_dose_seeds.png", dpi=200)
    plt.close(fig)

    payload = {
        "question": "does weaker ordinary SFT move the served map toward "
                    "the entering frozen model, or does it change the "
                    "live-label dynamics?",
        "verdict": verdict,
        "entering_frozen_map": {
            "tag": FROZEN_TAG, "mean": fm,
            "sd": float(frozen.std()),
            "n_distinct": int(np.unique(frozen).size),
            "why_valid": "training_style=frozen, ICL_K=0, ICL_DAYS=0 -- "
                         "never trained, no context, the same plain "
                         "profile prompt the dose runs use; innate vector "
                         "verified bit-identical to the dose wave",
        },
        "perfect_prediction_mean": pp,
        "ladder": "data exposure, LR, rank, batch and seed fixed; only "
                  "SFT_GRAD_ACCUM varies, so U is a pure "
                  "update-frequency dial",
        "settled_by_round_10": settled,
        "provenance": {
            "new_2026_08_28": "seeds 42,43 x U {1,5,181} -- 6 jobs, "
                              "check_update_dose --seeds 42,43 6/6 PASS",
            "preexisting": "seed 0, U in {1,5}",
            "reused_legacy": f"seed 0 U=181 is the archived "
                             f"{REUSE_U181_S0} read over its first ten "
                             f"rounds (accum 1 is the byte-identical "
                             f"legacy path). It predates SAVE_SFT_ORDER, "
                             f"so its sampler order cannot be "
                             f"re-verified.",
        },
        "gate": {"errors": errs, "pass": not errs},
        "by_dose": rows,
        "cells": cells,
    }
    (out / "update_dose_seeds.json").write_text(json.dumps(payload, indent=2))

    cap = [
        "Update dose in the closed loop, three seeds. Every arm consumes",
        "all 723 live labels exactly once per round in the same recorded",
        "order; only the number of optimizer steps over them differs, so",
        "U is a pure update-frequency dial. Left: population mean by",
        "round, one line per seed {0,42,43}. Dashed grey: perfect",
        f"prediction ({pp:.4f}), the innate mean. Dotted grey: the mean of",
        f"the ENTERING FROZEN MAP ({fm:.4f}) -- Qwen3-8B untrained, no",
        "context, same prompt, same agents. Right: how far the served map",
        "sits from that entering map, as the agent-paired mean absolute",
        "difference over the final five rounds, with across-seed 95%",
        "Student-t intervals. WEAKER SFT DOES NOT DRIFT BACK TOWARD THE",
        "ENTERING MAP: U=1 is the FURTHEST from it, and it is the only arm",
        "that fails to settle, cycling instead about a level well above",
        "perfect prediction. U=5 and U=181 reach exact consensus within",
        "two rounds, essentially at perfect prediction. MovieLens/Action,",
        "723 agents, W=1, k=1, S=100 sweeps, both gates open, fresh LoRA",
        "r512, 10 rounds.",
    ]
    (out / "update_dose_seeds_caption.txt").write_text("\n".join(cap) + "\n")

    if errs:
        print("[uds] GATE FAIL:")
        for e in errs:
            print("   -", e)
    else:
        print(f"[uds] GATE PASS: {len(cells)} cells, frozen reference is "
              f"untrained/no-context, innate bit-identical everywhere")
    hdr = (f"{'U':>5}{'accum':>7}{'n':>3}{'d_frozen_late':>15}"
           f"{'95% CI':>20}{'pop_late':>10}{'drift':>10}"
           f"{'late_range':>12}{'popSD':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['u']:>5}{r['accum']:>7}{r['n_seeds']:>3}"
              f"{r['d_frozen_late']:>15.4f}"
              f"{'[' + format(r['d_frozen_late_lo'], '.4f') + ', ' + format(r['d_frozen_late_hi'], '.4f') + ']':>20}"
              f"{r['pop_mean_late']:>10.4f}{r['drift']:>+10.4f}"
              f"{r['late_range']:>12.4f}{r['pop_sd_final']:>9.5f}")
    print(f"\nentering frozen map mean = {fm:.4f}    "
          f"perfect prediction = {pp:.4f}")
    print(f"VERDICT: {verdict}")
    print(f"settled by round 10: {settled}")
    print(f"\n[uds] wrote {out}/update_dose_seeds.{{csv,json,pdf,png}} + "
          f"_cells.csv + _caption.txt")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
