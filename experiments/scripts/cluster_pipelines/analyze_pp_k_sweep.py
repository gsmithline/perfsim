#!/usr/bin/env python3
"""Analyzer for the perfect-prediction k-sweep (2026-08-21). CPU only.

WHAT k DOES HERE. Under perfect prediction m(t) = x(t) the pre-peer
update is exactly Friedkin-Johnsen,

    z = (1 - beta_eff) x_innate + beta_eff x,   beta_eff = 1 - (1 - W) k,

so k sets how much of the PREVIOUS POPULATION STATE carries straight
into the next round. k = 1 is Jiduan's stateless form (the human
component is pure innate); k < 1 adds direct carryover of x.

READ THE DIRECTION CAREFULLY. At fixed W = .5, LOWERING k RAISES
beta_eff: k=1 -> .5, k=0 -> 1. More carryover means a LESS anchored map.
It is easy to assume the opposite, so beta_eff is carried in every CSV
row and printed next to every k.

THE t = 0 POINT. The figure plots the INNATE population at t = 0 and
then the post-update rounds 1..R, so the first plotted point is
unambiguous: it is the population before any round has run, not the
result of round 1. op_raw[i] is the state AFTER round i+1, which is why
the series is [innate] + op_raw rather than op_raw alone.

W = 1 IS THE CONTROL, NOT A CONDITION. There beta_eff = 1 for every k
because the (1-W) factor annihilates the human component, so all k
coincide. It is reported and checked, not swept.

Outputs (notes/pofd/perfect_prediction_k_sweep/):
  pp_k_sweep_rounds.csv      per k, seed, round
  pp_k_sweep_endpoints.csv   per k, seed: equilibrium SD, W1 from innate
  pp_k_sweep.png / .pdf      trajectories by k + a compact endpoint panel
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
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
_spec = importlib.util.spec_from_file_location(
    "sim_pp", str(HERE / "sim_perfect_predictor.py"))
PP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PP)

OUT_DIR = REPO / "notes" / "pofd" / "perfect_prediction_k_sweep"
SWEEP_W = 0.5
LATE_WINDOW = 5


def w1(a, b):
    """Equal-n 1-Wasserstein: mean |sort(a) - sort(b)|."""
    return float(np.abs(np.sort(a) - np.sort(b)).mean())


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return (d["op_raw"].float().numpy(), d["innate"].float().numpy(),
            d["config"])


def analyse(out_dir):
    mf = json.load(open(out_dir / "manifest.json"))
    ks, seeds, R = mf["ks"], mf["seeds"], mf["rounds"]
    rounds_rows, end_rows = [], []
    series = {}

    for c in mf["cells"]:
        if c["w_plat"] != SWEEP_W:
            continue
        op, innate, cfg = load(Path(c["path"]))
        k, seed = c["innate_k"], c["seed"]
        # t=0 is the INNATE population; t=1..R are the post-update rounds
        traj = np.concatenate([innate[None, :], op], axis=0)
        series[(k, seed)] = (traj, innate)
        for t in range(traj.shape[0]):
            v = traj[t]
            rounds_rows.append({
                "innate_k": k, "w_plat": SWEEP_W, "seed": seed,
                "beta_eff": PP.beta_eff(k, SWEEP_W),
                "t": t, "is_innate": int(t == 0),
                "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "range": float(v.max() - v.min()),
                "w1_from_innate": w1(v, innate),
            })
        late = traj[-LATE_WINDOW:]
        end_rows.append({
            "innate_k": k, "w_plat": SWEEP_W, "seed": seed,
            "beta_eff": PP.beta_eff(k, SWEEP_W), "rounds": R,
            "equilibrium_sd": float(np.mean(
                [x.std(ddof=1) for x in late])),
            "final_sd": float(traj[-1].std(ddof=1)),
            "final_mean": float(traj[-1].mean()),
            "innate_mean": float(innate.mean()),
            "mean_drift": float(traj[-1].mean() - innate.mean()),
            "w1_from_innate": w1(traj[-1], innate),
            "final_range": float(traj[-1].max() - traj[-1].min()),
            "note": ("lower k RAISES beta_eff at fixed W; more carryover "
                     "= less anchored"),
        })

    # the W=1 control, reported (its invariance is enforced by the checker)
    ctrl = [c for c in mf["cells"] if c["w_plat"] == 1.0]
    if ctrl:
        op, innate, _ = load(Path(ctrl[0]["path"]))
        end_rows.append({
            "innate_k": "all (identical)", "w_plat": 1.0, "seed": "all",
            "beta_eff": 1.0, "rounds": R,
            "equilibrium_sd": float(np.mean(
                [x.std(ddof=1) for x in op[-LATE_WINDOW:]])),
            "final_sd": float(op[-1].std(ddof=1)),
            "final_mean": float(op[-1].mean()),
            "innate_mean": float(innate.mean()),
            "mean_drift": float(op[-1].mean() - innate.mean()),
            "w1_from_innate": w1(op[-1], innate),
            "final_range": float(op[-1].max() - op[-1].min()),
            "note": "W=1 control: beta_eff=1 for every k, so all k coincide",
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    _csv(out_dir / "pp_k_sweep_rounds.csv", rounds_rows)
    _csv(out_dir / "pp_k_sweep_endpoints.csv", end_rows)
    figure(series, ks, seeds, end_rows, out_dir)
    _report(end_rows, ks)
    return rounds_rows, end_rows


def _csv(path, rows):
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
    print(f"[ksweep] wrote {path} ({len(rows)} rows)")


def _report(end_rows, ks):
    print(f"\n[ksweep] {'k':>6} {'beta_eff':>9} {'equil SD':>10} "
          f"{'W1 innate':>10} {'mean drift':>11}   (W=0.5, 3-seed mean)")
    for k in ks:
        sel = [r for r in end_rows if r["innate_k"] == k]
        if not sel:
            continue
        print(f"[ksweep] {k:>6g} {sel[0]['beta_eff']:>9.2f} "
              f"{np.mean([r['equilibrium_sd'] for r in sel]):>10.4e} "
              f"{np.mean([r['w1_from_innate'] for r in sel]):>10.4e} "
              f"{np.mean([r['mean_drift'] for r in sel]):>11.2e}")
    ctl = [r for r in end_rows if r["w_plat"] == 1.0]
    if ctl:
        print(f"[ksweep] W=1 control (all k identical): "
              f"equil SD {ctl[0]['equilibrium_sd']:.4e}, "
              f"W1 {ctl[0]['w1_from_innate']:.4e}")


def figure(series, ks, seeds, end_rows, out_dir):
    cmap = plt.get_cmap("viridis")
    col = {k: cmap(i / max(len(ks) - 1, 1)) for i, k in enumerate(ks)}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))

    for ax, field, ylab in ((axes[0][0], 0, "population mean"),
                            (axes[0][1], 1, "population SD")):
        for k in ks:
            per = [series[(k, s)][0] for s in seeds if (k, s) in series]
            if not per:
                continue
            vals = np.stack([[v.mean() if field == 0 else v.std(ddof=1)
                              for v in traj] for traj in per])
            t = np.arange(vals.shape[1])
            for row in vals:                      # thin: individual seeds
                ax.plot(t, row, lw=0.5, alpha=0.35, color=col[k])
            ax.plot(t, vals.mean(0), lw=1.8, color=col[k],
                    label=f"$k={k:g}$  ($\\beta_{{\\rm eff}}$="
                          f"{PP.beta_eff(k, SWEEP_W):.1f})")
        # t=0 is the innate population, before any round has run
        ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=0)
        if field == 0:
            # The mean is CONSERVED exactly here -- (1-b)m0 + b*m0 = m0 for
            # any beta_eff, and midpoint peer moves conserve it too. Left
            # on autoscale matplotlib zooms to the 1e-7 float residue and
            # draws what looks like violent oscillation; pinning the range
            # to a scale a reader cares about makes the flat line read as
            # flat, which is the actual result.
            m0 = float(list(series.values())[0][1].mean())
            ax.axhline(m0, color="#bbbbbb", lw=0.8, zorder=0)
            ax.set_ylim(m0 - 0.05, m0 + 0.05)
            ax.set_ylabel("population mean\n(conserved exactly; "
                          "$\\pm 0.05$ shown)", fontsize=9)
        else:
            ax.set_ylabel(ylab, fontsize=10)
        ax.set_xlabel("t   (0 = innate population, 1..R = post-update)")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].legend(fontsize=8, frameon=False, ncol=2)

    sw = [r for r in end_rows if r["w_plat"] == SWEEP_W]
    for ax, field, ylab in (
            (axes[1][0], "equilibrium_sd", "equilibrium SD"),
            (axes[1][1], "w1_from_innate", "$W_1$ from innate")):
        m = [np.mean([r[field] for r in sw if r["innate_k"] == k])
             for k in ks]
        e = [np.std([r[field] for r in sw if r["innate_k"] == k], ddof=1)
             for k in ks]
        ax.errorbar(ks, m, yerr=e, marker="o", ms=5, lw=1.6, capsize=3,
                    color="#2a6fb5")
        for i, k in enumerate(ks):
            ax.scatter([k], [m[i]], s=42, color=col[k], zorder=3)
        ax.set_xlabel("$k$   ($\\beta_{\\rm eff}=1-(1-W)k$; lower $k$ "
                      "= higher $\\beta_{\\rm eff}$)")
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"pp_k_sweep.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[ksweep] wrote {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    analyse(args.dir)
    print(f"[ksweep] outputs in {args.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
