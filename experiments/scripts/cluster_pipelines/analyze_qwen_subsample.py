#!/usr/bin/env python3
"""HARD-GATED analyzer for the OBSERVATION-RATE SUBSAMPLING wave
(2026-08-21, qwen_subsample).

THE HYPOTHESIS. Ordinary SFT implicitly leans MORE on the pretrained
model when it observes less of the population. Everything here is the
completed Wu-boundary b0 cell with one dial moved: how many of the 723
agents' labels reach the optimizer each round (14/36/72/181/362/723).
Serving is untouched -- all 723 are served in every arm -- so the arms
differ purely in observation.

THE DECOMPOSITION. Each round the served vector is regressed onto two
references:

    m(t)  ~=  a_t * m*(t)  +  (1 - a_t) * m_base

  m_base   the CANONICAL frozen Qwen2.5 K=D=0 prediction vector -- the
           entering model's answer, fixed forever and identical across
           every cell of this project (sha256 1674ee5f...da30bb).
  m*(t)    THIS RUN'S OWN training-label vector: innate at round 0 and
           op_raw[t-1] afterwards. That is the population state the
           round's adapter was actually fitted to.

a_t is then "how much of the served vector is explained by the data it
just trained on, versus the model it started from". a_t -> 1 means the
platform is reporting the population back to itself; a_t -> 0 means it
is reporting the pretrained prior regardless of what it saw.

WHY m*(t) IS THE RUN'S OWN LABELS AND NOT THE ORACLE. Using the separate
perfect-prediction trajectory would regress against a population that
this run never saw -- the oracle follows its own path, and by round 50
the two have diverged for reasons that have nothing to do with
observation rate. The quantity we want is the pull toward the data the
optimizer was handed, so the reference has to be that data.

FITTING. a_t is the ordinary least-squares coefficient of the projection
of (m - m_base) onto (m* - m_base):

    a_t = <m - m_base, m* - m_base> / <m* - m_base, m* - m_base>

with no intercept, because the decomposition is an interpolation between
two named endpoints, not a free linear model. a_t is NOT clipped to
[0, 1]: a value outside that range is a real statement (the served
vector lies outside the segment between prior and data) and clipping it
would hide exactly the case worth seeing. The residual RMSE reports how
well the two-endpoint model describes m at all -- a small a_t with a
large residual means "neither reference explains it", which is a
different finding from "it followed the prior".

NO MONOTONICITY IS ASSUMED OR ENFORCED. The late-a-versus-observation
curve is plotted as measured. If it is flat, or non-monotone, that is
the result.

Usage:
  python analyze_qwen_subsample.py [--runs-root DIR ...] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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
MANIFEST = REPO / "experiments" / "condor" / "manifest_qwen_subsample.json"
DEFAULT_ROOTS = [REPO / "runs" / "pokec_gated_lm",
                 REPO / "notes" / "pofd" / "cluster"]
DEFAULT_OUT = REPO / "notes" / "pofd" / "qwen_subsample_analysis"
# the canonical frozen Qwen2.5 K=D=0 prior, derived by
# audit_qwen_mechanism.py and pinned in three places already
CANON_SHA = ("1674ee5f8d833f46de672791d933e1d3bdeefb07484c2d110dec84ce"
             "71da30bb")
FROZEN_SOURCES = [
    "pofdqmech_qwen7b_k0_ea1_w0p5_l1_es0p05_s0",
    "pofdfam_qwen7b_k0_ea1_w0p5_l0p2_es0p05_s0",
]
LATE_WINDOW = 5
N_AGENTS = 723


def np_(t):
    return t.detach().cpu().float().numpy()


def load_base(roots):
    """The canonical frozen prediction vector, hash-verified."""
    import hashlib
    for tag in FROZEN_SOURCES:
        for root in roots:
            p = Path(root) / tag / "trajectory.pt"
            if not p.exists():
                continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            pr = d["pred_raw"]
            if not bool((pr == pr[0]).all()):
                raise SystemExit(f"[qss] {tag} frozen predictions are not "
                                 f"constant -- not a usable base model")
            sha = hashlib.sha256(pr[0].contiguous().numpy().tobytes()
                                 ).hexdigest()
            if sha != CANON_SHA:
                raise SystemExit(
                    f"[qss] {tag} prediction sha256 {sha[:16]}... != the "
                    f"canonical {CANON_SHA[:16]}... -- refusing to "
                    f"regress against a different prior")
            return np_(pr[0]), tag
    raise SystemExit(f"[qss] HARD FAIL: no canonical frozen source found "
                     f"in {[str(r) for r in roots]}; tried "
                     f"{FROZEN_SOURCES}")


def resolve_cells(roots):
    """{(observed, repeat_to): (run_dir, role)} for all 7 cells."""
    mf = json.load(open(MANIFEST))
    out, missing = {}, []
    for c in mf["cells"]:
        tag = c["run_tag"] if c["status"] == "reused" else c.get("new_tag")
        rd = None
        for root in roots:
            cand = Path(root) / tag
            if (cand / "trajectory.pt").exists():
                rd = cand
                break
        if rd is None:
            missing.append(f"{c['observed']} agents"
                           + (f" x{c['repeat_to']}" if c["repeat_to"]
                              else "") + f" -> {tag}")
        else:
            out[(c["observed"], c["repeat_to"])] = (rd, c["role"])
    if missing:
        raise SystemExit(
            f"[qss] HARD FAIL: {len(missing)} of {len(mf['cells'])} cells "
            f"unavailable (submit qwen_subsample first):\n  "
            + "\n  ".join(missing))
    return out, mf


def fit_a(m, m_star, m_base):
    """(a, residual_rmse) for m ~= a*m_star + (1-a)*m_base.

    Projection with no intercept: the decomposition interpolates between
    two NAMED endpoints, so a free constant would not mean anything.
    Returns a = nan when the two references coincide (no direction to
    project onto) rather than dividing by ~0 and reporting noise.
    """
    dv = m_star - m_base
    den = float(dv @ dv)
    if den < 1e-12:
        return float("nan"), float(np.sqrt(np.mean((m - m_base) ** 2)))
    a = float(((m - m_base) @ dv) / den)
    resid = m - (a * m_star + (1.0 - a) * m_base)
    return a, float(np.sqrt(np.mean(resid ** 2)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def analyse(roots, out_dir):
    cells, mf = resolve_cells(roots)
    m_base, base_tag = load_base(roots)
    print(f"[qss] base model: {base_tag} (sha {CANON_SHA[:12]}...)")

    rounds_rows, per_cell = [], []
    for (obs, rep), (rd, role) in sorted(cells.items()):
        d = torch.load(Path(rd) / "trajectory.pt", map_location="cpu",
                       weights_only=False)
        op, pred = np_(d["op_raw"]), np_(d["pred_raw"])
        innate = np_(d["innate"])
        n_rounds = op.shape[0]
        if op.shape[1] != N_AGENTS:
            raise SystemExit(f"[qss] {rd} has {op.shape[1]} agents")
        a_series = []
        for t in range(n_rounds):
            # m*(t): the labels THIS round trained on -- innate at round
            # 0, the preceding population afterwards. Never the oracle,
            # and never the same round's post-update opinions.
            m_star = innate if t == 0 else op[t - 1]
            a, res = fit_a(pred[t], m_star, m_base)
            a_series.append(a)
            rounds_rows.append({
                "observed": obs, "repeat_to": rep, "role": role,
                "observed_frac": round(obs / N_AGENTS, 6),
                "round": t, "a": a, "resid_rmse": res,
                "rmse_to_base": rmse(pred[t], m_base),
                "rmse_to_labels": rmse(pred[t], m_star),
                "pop_mean": float(op[t].mean()),
                "pop_sd": float(op[t].std(ddof=1)),
                "pred_mean": float(pred[t].mean()),
                "pred_sd": float(pred[t].std(ddof=1)),
            })
        late = list(range(n_rounds - LATE_WINDOW, n_rounds))
        la = [a_series[t] for t in late]
        per_cell.append({
            "observed": obs, "repeat_to": rep, "role": role,
            "observed_frac": round(obs / N_AGENTS, 6),
            "n_rounds": n_rounds,
            "late_rounds": f"{late[0]}-{late[-1]}",
            "late_a": float(np.mean(la)),
            "late_a_sd": float(np.std(la, ddof=1)),
            "late_resid_rmse": float(np.mean(
                [r["resid_rmse"] for r in rounds_rows
                 if r["observed"] == obs and r["repeat_to"] == rep
                 and r["round"] in late])),
            "late_rmse_to_base": float(np.mean(
                [r["rmse_to_base"] for r in rounds_rows
                 if r["observed"] == obs and r["repeat_to"] == rep
                 and r["round"] in late])),
            "late_rmse_to_labels": float(np.mean(
                [r["rmse_to_labels"] for r in rounds_rows
                 if r["observed"] == obs and r["repeat_to"] == rep
                 and r["round"] in late])),
            "final_pop_mean": float(op[-1].mean()),
            "final_pop_sd": float(op[-1].std(ddof=1)),
            "monotonicity": "NOT assumed or enforced",
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    _csv(out_dir / "subsample_rounds.csv", rounds_rows)
    _csv(out_dir / "subsample_per_cell.csv", per_cell)
    figure(rounds_rows, per_cell, out_dir)
    _report(per_cell)
    return rounds_rows, per_cell


def _csv(path, rows):
    if not rows:
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[qss] wrote {path} ({len(rows)} rows)")


def _report(per_cell):
    print(f"\n[qss] {'observed':>9} {'frac':>7} {'late a':>9} "
          f"{'resid':>9} {'->base':>9} {'->labels':>9} {'pop sd':>8}  role")
    for c in sorted(per_cell, key=lambda c: (c["repeat_to"], c["observed"])):
        print(f"[qss] {c['observed']:>9} {c['observed_frac']:>7.3f} "
              f"{c['late_a']:>9.4f} {c['late_resid_rmse']:>9.4f} "
              f"{c['late_rmse_to_base']:>9.4f} "
              f"{c['late_rmse_to_labels']:>9.4f} "
              f"{c['final_pop_sd']:>8.4f}  {c['role']}")


def figure(rounds_rows, per_cell, out_dir):
    """Preview figure: trajectories by observed fraction, and the late
    effective a against observation rate. Plotted as measured -- no
    monotonic fit, no ordering imposed."""
    obs_arms = sorted({(r["observed"], r["repeat_to"]) for r in rounds_rows})
    plain = [o for o in obs_arms if o[1] == 0]
    cmap = plt.get_cmap("viridis")
    col = {o: cmap(i / max(len(plain) - 1, 1))
           for i, o in enumerate(plain)}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    PANELS = [("a", "effective $a_t$ (weight on observed labels)"),
              ("resid_rmse", "residual RMSE"),
              ("pop_sd", "population SD")]
    for ax, (field, ylab) in zip(axes.flat, PANELS):
        for o in obs_arms:
            sel = sorted([r for r in rounds_rows
                          if (r["observed"], r["repeat_to"]) == o],
                         key=lambda r: r["round"])
            lab = (f"{o[0]} ({o[0] / N_AGENTS * 100:.0f}%)" if o[1] == 0
                   else f"{o[0]}$\\times${o[1]} compute-matched")
            ax.plot([r["round"] for r in sel], [r[field] for r in sel],
                    lw=1.4, label=lab,
                    color=(col.get(o) if o[1] == 0 else "#c0392b"),
                    ls=("-" if o[1] == 0 else "--"))
        ax.set_xlabel("round")
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    axes.flat[0].legend(fontsize=7.5, frameon=False, ncol=2)

    ax = axes.flat[3]
    pl = sorted([c for c in per_cell if c["repeat_to"] == 0],
                key=lambda c: c["observed"])
    ax.errorbar([c["observed_frac"] for c in pl], [c["late_a"] for c in pl],
                yerr=[c["late_a_sd"] for c in pl], marker="o", ms=5,
                lw=1.6, color="#2a6fb5", capsize=3, label="observation arm")
    for c in per_cell:
        if c["repeat_to"]:
            ax.errorbar([c["observed_frac"]], [c["late_a"]],
                        yerr=[c["late_a_sd"]], marker="D", ms=6,
                        color="#c0392b", capsize=3,
                        label="compute-matched")
    ax.set_xscale("log")
    ax.set_xlabel("observed fraction of the population")
    ax.set_ylabel("late effective $a$", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"qwen_subsample_preview.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[qss] wrote {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", action="append", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    analyse(args.runs_root or DEFAULT_ROOTS, args.out)
    print(f"[qss] outputs in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
