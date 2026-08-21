#!/usr/bin/env python3
"""HARD-GATED analyzer for the SFT TRAINING-DOSE scouts (2026-08-21).

THE QUESTION. Does a WEAKER SFT fit leave the served vector closer to the
entering Qwen model? Three families move exactly one dial each on the QWU
boundary surface:

  update   U optimizer updates      {1, 5, 20, 50, 100, 181}
  lr       learning rate            {1e-6, 3e-6, 1e-5, 3e-5, 5e-5}
  rank     LoRA rank                {1, 4, 8, 32, 128, 512}

ZERO IS FROZEN QWEN in all three: U=0, LR=0 and rank=0 all mean no
adaptation happened, which is the entering model exactly. The upper
endpoint of each family is the paper's current setting -- and rank 512
is the PAPER DEFAULT, unusually large for a LoRA, so the informative
band is 8-32 rather than the top end.

WHY ONE ROUND. The 100-round subsample wave drove the population to an
absorbing constant, after which the served vector and the training labels
are the SAME constant and the projection a equals 1 by construction --
it measures nothing. These cells train once on the innate labels and
serve once, so the served vector is read before any feedback exists.
This analyzer therefore looks ONLY at the first trained-and-served
vector and never touches the population trajectory.

THE TWO REFERENCES.
  m_base   the canonical frozen Qwen2.5 K=D=0 prediction vector -- the
           entering model's answer, hash-pinned. It is also the U=0 and
           LR=0 arm exactly: no training means no movement.
  m*       the TARGET: the innate population, which is what round 0
           trained on.

WHAT EACH DIAL ACTUALLY VARIES -- do not overclaim:
  U    is a TRAINING-DOSE dial, not a pure optimizer-step dial. U steps
       at batch 4 processes 4U of the 723 rows, so fewer updates also
       means fewer examples were ever seen.
  LR   limits how far the weights move. This does NOT test preservation
       of broad semantic capability -- only of the entering prediction
       map on these prompts.
  rank limits WHAT the update can represent, with alpha = 2r so the LoRA
       scaling alpha/r is constant across ranks. The cleanest capacity
       dial of the three -- but a small adapter may instead learn only a
       global scalar shift, which is why unique-value count and max mode
       share are reported alongside SD.

READING THE RESULT. The implicit-anchor hypothesis is supported only if
weaker SFT moves predictions toward frozen Qwen and away from the target
WHILE preserving Qwen's cross-agent structure -- with a low projection
residual and no collapse onto one value.

  STRUCTURE IS THE CRUX, because low rank (and a small LR, and few
  updates) limits the SHAPE of the change, not its MAGNITUDE. Even a
  rank-1 adapter can learn a large global shift and serve one value to
  every agent, which would sit close to Qwen in mean while retaining
  nothing of Qwen's per-agent map. The two cases are told apart by:

    near Qwen, corr_to_base high, many unique values
        -> Qwen's cross-agent structure survived: IMPLICIT ANCHOR
    near Qwen's MEAN, corr_to_base nan/low, one dominant value
        -> CAPACITY-INDUCED MODE COLLAPSE, not preservation

  corr is nan (not 0) for a constant output, so "no structure at all" is
  never silently reported as "uncorrelated".

Other failure modes to read off rather than explain away:
  high residual        -> neither reference explains the output
  non-monotone         -> report it; nothing here assumes an ordering
No expected ordering is encoded in the CSVs, the figure, or the tests.

Outputs (notes/pofd/sft_dose_analysis/):
  sft_dose_cells.csv        one row per arm, every metric
  sft_dose_preview.png/pdf  the three families side by side
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import torch                                               # noqa: E402

HERE = Path(os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
REPO = HERE.parent.parent.parent
DEFAULT_ROOTS = [REPO / "runs" / "pokec_gated_lm",
                 REPO / "notes" / "pofd" / "cluster"]
DEFAULT_OUT = REPO / "notes" / "pofd" / "sft_dose_analysis"
CANON_SHA = ("1674ee5f8d833f46de672791d933e1d3bdeefb07484c2d110dec84ce"
             "71da30bb")
FROZEN_SOURCES = ["pofdqmech_qwen7b_k0_ea1_w0p5_l1_es0p05_s0",
                  "pofdfam_qwen7b_k0_ea1_w0p5_l0p2_es0p05_s0"]
N_AGENTS = 723
STD_U, STD_LR, STD_RANK = 181, 5e-5, 512
UPDATES = [1, 5, 20, 50, 100, 181]
LRS = [1e-6, 3e-6, 1e-5, 3e-5, 5e-5]
RANKS = [1, 4, 8, 32, 128, 512]


def _lrtok(lr):
    return f"{lr:g}".replace("-", "m").replace(".", "p")


def tag_of(u, lr, rank):
    return (f"pofdsftdose_qwen7b_u{u}_lr{_lrtok(lr)}_rank{rank}"
            f"_eaopen_w1_l1_esopen_s0_r1")


def np_(t):
    return t.detach().cpu().float().numpy()


def w1(a, b):
    return float(np.abs(np.sort(a) - np.sort(b)).mean())


def w1_centered(a, b):
    return w1(a - a.mean(), b - b.mean())


def fit_a(m, m_star, m_base):
    """(a, residual RMSE) for m ~= a m* + (1-a) m_base. No intercept: the
    decomposition interpolates between two NAMED endpoints. a is never
    clipped -- a value outside [0,1] is a real statement. NEVER read a
    without the residual."""
    dv = m_star - m_base
    den = float(dv @ dv)
    if den < 1e-12:
        return float("nan"), float(np.sqrt(np.mean((m - m_base) ** 2)))
    a = float(((m - m_base) @ dv) / den)
    return a, float(np.sqrt(np.mean((m - (a * m_star
                                          + (1 - a) * m_base)) ** 2)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rmse_centered(a, b):
    """RMSE after removing each vector's mean: distance in SHAPE alone.
    A served vector that is Qwen plus a global offset has a large raw
    RMSE to Qwen but a near-zero centered one."""
    return rmse(a - a.mean(), b - b.mean())


def corr(a, b):
    """Pearson correlation across agents. THE discriminator between
    'preserved Qwen's cross-agent structure' and 'collapsed to one
    value near Qwen's mean': a constant output has zero variance, so the
    correlation is undefined and returned as nan rather than 0, which
    would read as 'uncorrelated' instead of 'no structure at all'."""
    sa, sb = float(a.std()), float(b.std())
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mae(a, b):
    return float(np.abs(a - b).mean())


def find(roots, tag):
    for r in roots:
        p = Path(r) / tag / "trajectory.pt"
        if p.exists():
            return Path(r) / tag
    return None


def load_base(roots):
    for tag in FROZEN_SOURCES:
        rd = find(roots, tag)
        if rd is None:
            continue
        d = torch.load(rd / "trajectory.pt", map_location="cpu",
                       weights_only=False)
        pr = d["pred_raw"]
        sha = hashlib.sha256(pr[0].contiguous().numpy().tobytes()).hexdigest()
        if sha != CANON_SHA:
            raise SystemExit(f"[sftd] {tag} sha {sha[:16]}... != canonical")
        return np_(pr[0]), tag
    raise SystemExit(f"[sftd] HARD FAIL: no canonical frozen source in "
                     f"{[str(r) for r in roots]}")


def cell_metrics(rd, m_base, target, family, dial):
    d = torch.load(Path(rd) / "trajectory.pt", map_location="cpu",
                   weights_only=False)
    cfg = d["config"]
    pred = np_(d["pred_raw"])[0]        # the FIRST trained-and-served vector
    if pred.shape[0] != N_AGENTS:
        raise SystemExit(f"[sftd] {rd}: {pred.shape[0]} agents")
    a, resid = fit_a(pred, target, m_base)
    uniq, counts = np.unique(np.round(pred, 6), return_counts=True)
    dose = (d.get("sft_dose") or [{}])[0]
    # parse-failure fraction from the raw generations
    pf = float("nan")
    rg = Path(rd) / "raw_gen_log.json.gz"
    if rg.exists():
        with gzip.open(rg, "rt") as fh:
            row0 = json.loads(fh.readline())
        raw = row0.get("raw") or []
        if raw:
            pf = sum(1 for x in raw
                     if re.search(r"\d", str(x)) is None) / len(raw)
    return {
        "family": family, "dial": dial,
        "U": int(cfg.get("max_steps") or 0),
        "lr": float(cfg.get("sft_lr") or 0.0),
        "rank": int(cfg.get("lora_r") or 0),
        "rmse_to_base": rmse(pred, m_base), "mae_to_base": mae(pred, m_base),
        "rmse_to_target": rmse(pred, target),
        "mae_to_target": mae(pred, target),
        "w1_to_base": w1(pred, m_base),
        "w1_to_base_centered": w1_centered(pred, m_base),
        "w1_to_target": w1(pred, target),
        "w1_to_target_centered": w1_centered(pred, target),
        # STRUCTURE vs LOCATION. Low rank limits the SHAPE of the update,
        # not its magnitude -- even rank 1 can learn a large global shift
        # and serve one value to everyone. These separate the two cases:
        #   near Qwen AND corr_to_base high AND many unique values
        #       -> Qwen's cross-agent structure survived: implicit anchor
        #   near Qwen's MEAN but corr_to_base nan/low, one dominant value
        #       -> capacity-induced mode collapse, not preservation
        "rmse_to_base_centered": rmse_centered(pred, m_base),
        "rmse_to_target_centered": rmse_centered(pred, target),
        "corr_to_base": corr(pred, m_base),
        "corr_to_target": corr(pred, target),
        "mean_shift_from_base": float(pred.mean() - m_base.mean()),
        "a": a, "resid_rmse": resid,
        "pred_mean": float(pred.mean()), "pred_sd": float(pred.std(ddof=1)),
        "n_unique": int(uniq.size),
        "max_mode_share": float(counts.max() / counts.sum()),
        "parse_fail_frac": pf,
        "actual_optimizer_steps": int(dose.get("global_step", -1)),
        "trainer_seed": int(dose.get("trainer_seed", -1)),
        "n_rows_seen": int(dose.get("n_rows", -1)),
        "train_loss": float(dose.get("train_loss", float("nan"))),
        "serve_eval_mode": bool(cfg.get("serve_eval_mode", False)),
        "git_sha": cfg.get("git_sha", ""),
        "gpu": (cfg.get("hardware") or {}).get("gpu_name", ""),
        "note": "no monotonicity assumed; read a WITH resid_rmse",
    }


def analyse(roots, out_dir):
    m_base, base_tag = load_base(roots)
    print(f"[sftd] base model: {base_tag} (sha {CANON_SHA[:12]}...)")

    # the target is the shared innate population used for the first fit
    probe = find(roots, tag_of(STD_U, STD_LR, STD_RANK))
    if probe is None:
        raise SystemExit(f"[sftd] HARD FAIL: the shared full-dose endpoint "
                         f"{tag_of(STD_U, STD_LR, STD_RANK)} is missing; "
                         f"submit qwen_sft_update_dose first")
    target = np_(torch.load(probe / "trajectory.pt", map_location="cpu",
                            weights_only=False)["innate"])

    rows, missing = [], []
    # U=0 / LR=0 / rank=0 are all the frozen vector: no job, no movement
    zero = {"family": "reference", "dial": 0.0, "U": 0, "lr": 0.0,
            "rank": 0, "a": 0.0, "resid_rmse": 0.0,
            "rmse_to_base": 0.0, "mae_to_base": 0.0,
            "rmse_to_target": rmse(m_base, target),
            "mae_to_target": mae(m_base, target),
            "w1_to_base": 0.0, "w1_to_base_centered": 0.0,
            "w1_to_target": w1(m_base, target),
            "w1_to_target_centered": w1_centered(m_base, target),
            "rmse_to_base_centered": 0.0,
            "rmse_to_target_centered": rmse_centered(m_base, target),
            "corr_to_base": 1.0, "corr_to_target": corr(m_base, target),
            "mean_shift_from_base": 0.0,
            "pred_mean": float(m_base.mean()),
            "pred_sd": float(m_base.std(ddof=1)),
            "n_unique": int(np.unique(np.round(m_base, 6)).size),
            "max_mode_share": float(
                np.unique(np.round(m_base, 6), return_counts=True)[1].max()
                / N_AGENTS),
            "parse_fail_frac": 0.0, "actual_optimizer_steps": 0,
            "trainer_seed": -1, "n_rows_seen": 0,
            "train_loss": float("nan"), "serve_eval_mode": True,
            "git_sha": "", "gpu": "n/a (frozen control, no GPU job)",
            "note": "U=0 / LR=0: the canonical frozen Qwen vector"}
    rows.append(zero)

    for fam, dials in (("update", UPDATES), ("lr", LRS), ("rank", RANKS)):
        for v in dials:
            u = v if fam == "update" else STD_U
            lr = v if fam == "lr" else STD_LR
            rk = v if fam == "rank" else STD_RANK
            rd = find(roots, tag_of(u, lr, rk))
            if rd is None:
                missing.append(f"{fam}={v} -> {tag_of(u, lr, rk)}")
            else:
                rows.append(cell_metrics(rd, m_base, target, fam, float(v)))
    if missing:
        raise SystemExit(
            f"[sftd] HARD FAIL: {len(missing)} cell(s) unavailable:\n  "
            + "\n  ".join(missing))

    out_dir.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(out_dir / "sft_dose_cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[sftd] wrote {out_dir / 'sft_dose_cells.csv'} ({len(rows)} rows)")
    report(rows)
    figure(rows, out_dir)
    return rows


def report(rows):
    print(f"\n[sftd] {'family':>8} {'dial':>10} {'->base':>8} {'->target':>9} "
          f"{'a':>8} {'resid':>8} {'SD':>7} {'uniq':>5} {'mode%':>6} {'pf':>5}")
    for r in rows:
        print(f"[sftd] {r['family']:>8} {r['dial']:>10g} "
              f"{r['rmse_to_base']:>8.4f} {r['rmse_to_target']:>9.4f} "
              f"{r['a']:>8.4f} {r['resid_rmse']:>8.4f} "
              f"{r['pred_sd']:>7.4f} {r['n_unique']:>5d} "
              f"{r['max_mode_share'] * 100:>5.1f}% "
              f"{r['parse_fail_frac']:>5.2f}")


def figure(rows, out_dir):
    fams = [("update", "optimizer updates $U$", True),
            ("lr", "learning rate", True),
            ("rank", "LoRA rank $r$", True)]
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 9.0))
    ref = rows[0]
    for j, (fam, xlab, logx) in enumerate(fams):
        sel = sorted([r for r in rows if r["family"] == fam],
                     key=lambda r: r["dial"])
        x = [r["dial"] for r in sel]
        panels = [
            (0, [("rmse_to_base", "to frozen Qwen", "#e8820c"),
                 ("rmse_to_target", "to target", "#2a6fb5")],
             "RMSE"),
            (1, [("a", "fitted $a$", "#111111"),
                 ("resid_rmse", "residual", "#c0392b")], "projection"),
            (2, [("pred_sd", "prediction SD", "#2e8b57"),
                 ("max_mode_share", "max mode share", "#8c8c8c"),
                 ("corr_to_base", "corr to Qwen", "#e8820c")],
             "structure vs collapse"),
        ]
        for i, series, ylab in panels:
            ax = axes[i][j]
            for field, lab, col in series:
                ax.plot(x, [r[field] for r in sel], marker="o", ms=4,
                        lw=1.5, color=col, label=lab)
                # the zero-dose reference: frozen Qwen, no training
                ax.axhline(ref[field] if field in ref else np.nan,
                           color=col, lw=0.7, ls=":", alpha=0.6)
            if logx:
                ax.set_xscale("log")
            if i == 2:
                ax.set_xlabel(xlab)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9)
            ax.grid(alpha=0.25, lw=0.5)
            if i == 0 and j == 0:
                ax.legend(fontsize=7.5, frameon=False)
            elif j == 0:
                ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"sft_dose_preview.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[sftd] wrote {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", action="append", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    analyse(args.runs_root or DEFAULT_ROOTS, args.out)
    print(f"[sftd] outputs in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
