#!/usr/bin/env python3
"""HARD-GATED analyzer for the REFERENCE-REPLAY pilot (pofdrr_, 2026-08-22).

THE MECHANISM, NAMED PRECISELY. Each round, a fraction q of the training
rows carry the population's own current opinions x_i(t) and the
remaining 1 - q carry a PINNED, FROZEN vector b -- the entering Qwen2.5
model's own K=D=0 predictions, computed once and never recomputed. This
is EXPLICIT DATA-SPACE REFERENCE REPLAY: specific rows of the training
set are literally overwritten with specific frozen numbers. It is NOT
"implicit anchoring" -- nothing is inferred, penalized, or regularized,
and the analyzer never describes it that way. q = 1 is ordinary SFT.

WHAT THIS REPORTS, AND IN WHICH DIRECTION. Nothing here assumes an
ordering across q. The working hypothesis in the lab notebook is that
lowering q pulls the population toward the frozen-model equilibrium and
retains more dispersion; this analyzer is written so that the OPPOSITE
result, or no result, would be reported just as plainly. Arms are
plotted in ladder order because that is how they are indexed, not
because a monotone curve is expected.

READOUTS (per arm)
  * End-of-round POST-PEER population mean and SD, rounds 1..T. That is
    the only state in the headline figure: no within-round intermediate
    (post-AI-mixture, pre-peer) state appears, because those are not the
    state the next round trains on and mixing the two makes a
    trajectory that no single operator generated.
  * Late-window 1-Wasserstein distance from the arm's opinion
    distribution to (a) the FROZEN-QWEN equilibrium and (b) the
    ORDINARY-SFT (q = 1) equilibrium. Two named endpoints, both taken
    from actual runs, never from an analytic guess.
  * Served-vector distance to the frozen prediction MAP b, both paired
    (agentwise MAE) and distributional (W1).
  * Mode / parse diagnostics: distinct served values, largest single-
    value share, parse-failure fraction. A served vector that has
    collapsed onto three values is a different object from one that
    still spans the prior, and every distance above reads differently
    in that light.

STATIONARITY IS TESTED AS MEAN DRIFT, NOT AS STEP SIZE. A fresh LoRA is
trained from scratch every round, so the per-round update never decays
toward zero the way an annealed optimizer's would; a step-size
threshold would label a perfectly settled run "still moving" forever.
What settles, if anything settles, is the OUTER loop: the population
mean over the late window. So the diagnostic is the drift of that mean
across the window (OLS slope per round and end-to-end change), reported
as measured with the threshold stated.

NO INDEPENDENT REPLICATES EXIST HERE. Rounds are successive states of
ONE trajectory and agents are coupled through a shared model and a
shared peer graph, so neither is a replicate: no confidence interval,
standard error, or significance test is computed over rounds or agents,
and none should be added later. With one seed per arm, every comparison
across q is DESCRIPTIVE.

Usage:
  python analyze_ref_replay.py [--runs-root DIR ...] [--out DIR]
                               [--arms 0.10,0.20,0.50,0.75,1.0]
                               [--late-window 10]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

# matplotlib writes a font cache on first import; point it somewhere
# writable before the import so this runs under a read-only HOME (CI,
# condor scratch, pytest sandboxes).
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(tempfile.gettempdir(), "mplconfig_rr"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import torch                                               # noqa: E402

HERE = Path(os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
REPO = HERE.parent.parent.parent
DEFAULT_ROOTS = [REPO / "runs" / "pokec_gated_lm",
                 REPO / "notes" / "pofd" / "cluster"]
DEFAULT_OUT = REPO / "notes" / "pofd" / "ref_replay"

N_AGENTS = 723
LADDER = (0.10, 0.20, 0.50, 0.75, 1.0)
SFT_ARM = 1.0                    # q = 1 IS ordinary SFT, by construction
LATE_WINDOW = 10                 # rounds pooled for the late-window reads
DRIFT_TOL = 2e-3                 # |end-to-end late-window mean change|
                                 # below this is reported as "settled";
                                 # stated, not hidden, and not a p-value
W1_GRID = 512
CANON_REF_SHA = (
    "1674ee5f8d833f46de672791d933e1d3bdeefb07484c2d110dec84ce71da30bb")
# frozen Qwen2.5 K=D=0 loops: the served vector is a CONSTANT there, so
# their late-window population IS the frozen-model equilibrium.
FROZEN_CANDIDATES = (
    "pofdqmech_qwen7b_k0_ea1_w0p5_l1_es0p05_s0",
    "pofdqmech_qwen7b_k0_ea1_w0p5_l1_es0_s0",
    "pofdqmech_qwen7b_k0_ea1_w0p5_l1_es0p2_s0",
    "pofdfam_qwen7b_k0_ea1_w0p5_l0p2_es0p05_s0",
)


def np_(t):
    return torch.as_tensor(t).detach().cpu().float().numpy()


def sha_vec(a):
    return hashlib.sha256(np.asarray(a, dtype=np.float32)
                          .tobytes()).hexdigest()


def w1(a, b, grid=W1_GRID):
    """1-Wasserstein between two samples via quantile interpolation on a
    shared probability grid (the two pools need not be the same size)."""
    qs = np.linspace(0.0, 1.0, grid)
    return float(np.mean(np.abs(np.quantile(np.asarray(a, dtype=float), qs)
                                - np.quantile(np.asarray(b, dtype=float),
                                              qs))))


def mode_stats(vec):
    """(distinct values, largest single-value share) of one served round."""
    vals = np.asarray(vec, dtype=np.float64)
    c = Counter(np.round(vals, 6).tolist())
    return len(c), float(max(c.values())) / float(len(vals))


def resolve_arms(roots, want_q):
    """{q: (run_dir, cfg)} for every requested arm. HARD FAILS naming the
    arms that are not on disk -- a partial ladder is not a dose ladder."""
    found, seen = {}, []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for rd in sorted(root.glob("pofdrr*")):
            p = rd / "trajectory.pt"
            if not p.exists():
                continue
            try:
                cfg = torch.load(p, map_location="cpu",
                                 weights_only=False).get("config") or {}
            except Exception:
                continue
            q = cfg.get("ref_replay_q")
            if q is None:
                continue
            seen.append((float(q), rd.name))
            found.setdefault(round(float(q), 6), (rd, cfg))
    missing = [q for q in want_q if round(q, 6) not in found]
    if missing:
        raise SystemExit(
            f"[rr] HARD FAIL: {len(missing)} requested arm(s) absent -- "
            f"q={missing}. Found on disk: "
            f"{sorted(set(seen)) or 'nothing under pofdrr*'}. Pull the "
            f"wave and gate it with check_ref_replay.py first; there is "
            f"no partial-ladder read.")
    return {round(q, 6): found[round(q, 6)] for q in want_q}


def load_frozen(roots, explicit=None):
    """(b, frozen_population_pool, tag) for the frozen-Qwen reference.

    b is the constant served vector, hash-verified against the canonical
    frozen-Qwen constant. The population pool is that run's late-window
    opinions -- the equilibrium a loop reaches when the platform serves
    the entering model's answers forever. HARD FAILS if no such run is
    available: the distance to the frozen equilibrium is a headline
    number and must not be silently replaced by a proxy.
    """
    cands = ([Path(explicit)] if explicit else
             [Path(r) / t for r in roots for t in FROZEN_CANDIDATES])
    tried = []
    for cand in cands:
        p = Path(cand) / "trajectory.pt"
        if not p.exists():
            tried.append(str(cand))
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        pred = np_(d["pred_raw"])
        if not bool((pred == pred[0]).all()):
            raise SystemExit(f"[rr] {cand} does not serve a CONSTANT "
                             f"vector -- it is not a frozen reference")
        sha = sha_vec(pred[0])
        if sha != CANON_REF_SHA:
            raise SystemExit(
                f"[rr] {cand} served sha256 {sha[:16]}... != the "
                f"canonical frozen-Qwen {CANON_REF_SHA[:16]}... -- "
                f"refusing to measure against a different prior")
        op = np_(d["op_raw"])
        pool = op[-min(LATE_WINDOW, op.shape[0]):].reshape(-1)
        return pred[0], pool, Path(cand).name
    raise SystemExit(
        f"[rr] HARD FAIL: no canonical frozen-Qwen reference run found "
        f"(tried {tried or [str(c) for c in cands]}). Pass "
        f"--frozen-run DIR or pull one of {list(FROZEN_CANDIDATES)}.")


def parse_fail_frac(traj, run_dir):
    """Per-round parse-failure fraction, or None when the run records no
    parse provenance (reported as unknown, never as zero)."""
    vals = [r.get("parse_fail_frac") for r in traj]
    if all(v is not None for v in vals) and vals:
        return [float(v) for v in vals]
    import gzip
    import re
    p = Path(run_dir) / "raw_gen_log.json.gz"
    if not p.exists():
        return None
    out = []
    try:
        with gzip.open(p, "rt") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                raw = json.loads(ln).get("raw") or []
                if not raw:
                    continue
                nod = sum(1 for x in raw
                          if re.search(r"\d", str(x)) is None)
                out.append(nod / float(len(raw)))
    except (OSError, json.JSONDecodeError):
        return None
    return out or None


def late_slice(n_rounds, window):
    lo = max(0, n_rounds - window)
    return list(range(lo, n_rounds))


def analyse(roots, out_dir, want_q=LADDER, window=LATE_WINDOW,
            frozen_run=None):
    arms = resolve_arms(roots, want_q)
    b, frozen_pool, frozen_tag = load_frozen(roots, frozen_run)
    print(f"[rr] frozen reference: {frozen_tag} "
          f"(sha {CANON_REF_SHA[:12]}..., pooled late window "
          f"n={frozen_pool.size})")

    # first pass: per-round series for every arm
    series, seeds = {}, set()
    for q, (rd, cfg) in arms.items():
        d = torch.load(Path(rd) / "trajectory.pt", map_location="cpu",
                       weights_only=False)
        op, pred = np_(d["op_raw"]), np_(d["pred_raw"])
        innate = np_(d["innate"])
        traj = d.get("trajectory") or []
        n_rounds = op.shape[0]
        pf = parse_fail_frac(traj, rd)
        rows = []
        for t in range(n_rounds):
            n_uniq, mode_share = mode_stats(pred[t])
            rows.append({
                "q": q, "arm": rd.name, "round": t,
                "completed_round": t + 1,
                "pop_mean": float(op[t].mean()),
                "pop_sd": float(op[t].std(ddof=1)),
                "pred_mean": float(pred[t].mean()),
                "pred_sd": float(pred[t].std(ddof=1)),
                "served_mae_to_b": float(np.mean(np.abs(pred[t] - b))),
                "served_w1_to_b": w1(pred[t], b),
                "served_unique": n_uniq,
                "served_max_mode_share": mode_share,
                "parse_fail_frac": (None if pf is None or t >= len(pf)
                                    else pf[t]),
            })
        series[q] = {"rows": rows, "op": op, "pred": pred,
                     "innate": innate, "n_rounds": n_rounds,
                     "name": rd.name, "cfg": cfg}
        seeds.add(cfg.get("seed", cfg.get("ref_replay_seed")))

    # the ordinary-SFT endpoint comes from the q = 1 arm itself
    sft_key = round(SFT_ARM, 6)
    if sft_key not in series:
        raise SystemExit(
            f"[rr] HARD FAIL: the q={SFT_ARM} (ordinary SFT) arm is "
            f"required -- it is one of the two named endpoints every "
            f"other arm is measured against")
    s1 = series[sft_key]
    sft_pool = s1["op"][late_slice(s1["n_rounds"], window)].reshape(-1)

    per_arm, round_rows = [], []
    for q in sorted(series):
        s = series[q]
        round_rows.extend(s["rows"])
        late = late_slice(s["n_rounds"], window)
        pool = s["op"][late].reshape(-1)
        lm = np.array([s["rows"][t]["pop_mean"] for t in late])
        # OUTER stationarity: drift of the population MEAN across the
        # late window. Not a step-size test -- a fresh LoRA trains every
        # round, so the per-round step never vanishes even when the
        # population has settled.
        xs = np.arange(len(lm), dtype=float)
        slope = float(np.polyfit(xs, lm, 1)[0]) if len(lm) > 1 else float("nan")
        drift = float(lm[-1] - lm[0]) if len(lm) > 1 else float("nan")
        pf_late = [s["rows"][t]["parse_fail_frac"] for t in late]
        pf_late = [v for v in pf_late if v is not None]
        per_arm.append({
            "q": q, "arm": s["name"],
            "n_live": s["cfg"].get("ref_replay_n_live"),
            "seed": s["cfg"].get("seed", s["cfg"].get("ref_replay_seed")),
            "n_rounds": s["n_rounds"],
            "late_rounds": f"{late[0] + 1}-{late[-1] + 1}",
            "late_pop_mean": float(pool.mean()),
            "late_pop_sd": float(pool.std(ddof=1)),
            "w1_to_frozen_equilibrium": w1(pool, frozen_pool),
            "w1_to_sft_equilibrium": w1(pool, sft_pool),
            "late_served_mae_to_b": float(np.mean(
                [s["rows"][t]["served_mae_to_b"] for t in late])),
            "late_served_w1_to_b": float(np.mean(
                [s["rows"][t]["served_w1_to_b"] for t in late])),
            "late_served_unique": float(np.mean(
                [s["rows"][t]["served_unique"] for t in late])),
            "late_served_max_mode_share": float(np.mean(
                [s["rows"][t]["served_max_mode_share"] for t in late])),
            "late_parse_fail_frac": (float(np.mean(pf_late)) if pf_late
                                     else None),
            "late_mean_slope_per_round": slope,
            "late_mean_drift": drift,
            "outer_stationarity": ("settled" if abs(drift) < DRIFT_TOL
                                   else "still drifting"),
            "stationarity_criterion":
                f"|late-window mean change| < {DRIFT_TOL} (MEAN DRIFT, "
                f"not per-round step size)",
            "replicates": "none -- one seed, one trajectory per arm",
            "direction": "not assumed; reported as measured",
        })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _csv(out_dir / "ref_replay_rounds.csv", round_rows)
    _csv(out_dir / "ref_replay_per_arm.csv", per_arm)
    population_figure(series, out_dir)
    _report(per_arm, seeds, frozen_tag, window)
    return round_rows, per_arm


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
    print(f"[rr] wrote {path} ({len(rows)} rows)")


def _report(per_arm, seeds, frozen_tag, window):
    descriptive = (len(seeds) == 1 and (0 in seeds or "0" in
                                        {str(s) for s in seeds}))
    print(f"\n[rr] {'q':>6} {'n_live':>7} {'late mean':>10} {'late sd':>8} "
          f"{'W1->frozen':>11} {'W1->SFT':>9} {'MAE->b':>8} "
          f"{'uniq':>6} {'mode':>6}  stationarity")
    for a in sorted(per_arm, key=lambda a: a["q"]):
        print(f"[rr] {a['q']:>6.2f} {str(a['n_live']):>7} "
              f"{a['late_pop_mean']:>10.4f} {a['late_pop_sd']:>8.4f} "
              f"{a['w1_to_frozen_equilibrium']:>11.4f} "
              f"{a['w1_to_sft_equilibrium']:>9.4f} "
              f"{a['late_served_mae_to_b']:>8.4f} "
              f"{a['late_served_unique']:>6.1f} "
              f"{a['late_served_max_mode_share']:>6.2f}  "
              f"{a['outer_stationarity']} (drift "
              f"{a['late_mean_drift']:+.4f})")
    print(f"[rr] frozen equilibrium from {frozen_tag}; ordinary-SFT "
          f"equilibrium is the q=1 arm; late window = last {window} "
          f"completed rounds.")
    if descriptive:
        print("[rr] DESCRIPTIVE: seed 0 only, one trajectory per arm. No "
              "confidence intervals are computed and none should be "
              "added -- rounds are successive states of one trajectory "
              "and agents are coupled through a shared model and peer "
              "graph, so neither is an independent replicate.")
    else:
        print(f"[rr] seeds present: {sorted(str(s) for s in seeds)}. Still "
              f"no CI over rounds or agents -- neither is a replicate.")
    print("[rr] mechanism: EXPLICIT DATA-SPACE REFERENCE REPLAY (frozen "
          "labels substituted into named training rows). No ordering "
          "across q is assumed or enforced anywhere above.")


def population_figure(series, out_dir):
    """Headline figure: END-OF-ROUND POST-PEER population mean and SD
    only, with the innate t=0 value as a dotted reference.

    NO TITLE (project convention -- the narrative lives in the caption
    block). No within-round intermediate state appears here.
    """
    qs = sorted(series)
    cmap = plt.get_cmap("viridis")
    colors = {q: cmap(i / max(len(qs) - 1, 1)) for i, q in enumerate(qs)}
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.75), sharex=True)
    innate = next(iter(series.values()))["innate"]
    innate_mean = float(innate.mean())
    innate_sd = float(innate.std(ddof=1))
    xmax = 1
    for q in qs:
        s = series[q]
        x = [r["completed_round"] for r in s["rows"]]
        xmax = max(xmax, max(x))
        style = {"color": colors[q], "lw": 2.1 if q == SFT_ARM else 1.45,
                 "ls": "-" if q != SFT_ARM else "-"}
        lab = (f"q = {q:g}" + ("  (ordinary SFT)" if q == SFT_ARM else ""))
        axes[0].plot(x, [r["pop_mean"] for r in s["rows"]], label=lab,
                     **style)
        axes[1].plot(x, [r["pop_sd"] for r in s["rows"]], label=lab,
                     **style)
    axes[0].axhline(innate_mean, color="0.45", lw=1.0, ls=":")
    axes[1].axhline(innate_sd, color="0.45", lw=1.0, ls=":")
    axes[0].set_ylabel("Population mean")
    axes[1].set_ylabel("Population SD")
    axes[1].set_ylim(bottom=-0.003)
    for ax in axes:
        ax.set_xlabel("Completed retraining round")
        ax.grid(alpha=0.22, lw=0.5)
        ax.tick_params(labelsize=8)
    axes[0].text(xmax, innate_mean + .004, "initial mean", ha="right",
                 va="bottom", fontsize=7, color="0.4")
    axes[1].text(xmax, innate_sd + .003, "initial SD", ha="right",
                 va="bottom", fontsize=7, color="0.4")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(qs),
               frameon=False, fontsize=7.2, handlelength=2.4,
               columnspacing=1.1, bbox_to_anchor=(.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, .84), w_pad=2.0)
    for ext in ("png", "pdf"):
        p = Path(out_dir) / f"ref_replay_population_mean_sd.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"[rr] wrote {p}")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", action="append", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", type=str, default=None,
                    help="comma-separated q values (default: full ladder)")
    ap.add_argument("--late-window", type=int, default=LATE_WINDOW)
    ap.add_argument("--frozen-run", type=Path, default=None)
    args = ap.parse_args(argv)
    want = (tuple(float(x) for x in args.arms.split(",")) if args.arms
            else LADDER)
    analyse(args.runs_root or DEFAULT_ROOTS, args.out, want_q=want,
            window=args.late_window, frozen_run=args.frozen_run)
    print(f"[rr] outputs in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
