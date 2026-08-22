#!/usr/bin/env python3
"""HARD-GATED analyzer for the FJ robustness wave (2026-08-21). CPU only.

THE QUESTION. Does the ordinary-SFT (b0) vs forward-KL-SFT (b1) result
survive under Jiduan Wu's model (homogeneous-parameter specialization)
in place of our Deffuant bounded-confidence peers? Six models x two
arms, seed 0, peer susceptibility alpha=.9, K=100 inner FJ steps per
outer round, 30 outer rounds. beta is the only population-side quantity
that varies: .5 for the core wave, 1 for the optional boundary wave.

HARD FAIL UNTIL THE GRID IS COMPLETE. Twelve trained cells and six
frozen controls. A partial grid gets refused rather than analysed,
because a per-model ordering read off whichever cells happened to finish
is a different claim from the one this wave was built to make.

WHAT IS AND IS NOT CLAIMED.
  * Seed 0 only. Everything here is DESCRIPTIVE. No ordering is called
    significant, and none is assumed in advance -- the Deffuant result
    may or may not survive, and both outcomes are reportable.
  * OUTER round 30 is NOT an equilibrium unless convergence says so.
    The INNER loop does converge -- at alpha=.9, K=100 contracts by
    ~0.9^100 -- but that is the FJ fixed point of one round's anchor,
    which says nothing about whether the outer model-population loop has
    settled. The two must not be conflated. The
    per-cell final step |x(30) - x(29)| is reported next to every late
    summary, and the header states plainly whether each cell converged.
    The model-independent control converges at round 13 under perfect
    prediction, which is a reason to EXPECT convergence, not evidence of
    it for a trained arm.
  * The frozen control is a POINT, not a trajectory. Under a stateless
    human component a constant predictor yields a constant population
    from round 1, so "distance to frozen" is distance to a fixed vector.
  * This is not a controlled comparison against the Deffuant wave. There
    is no natural correspondence between a bounded-confidence gate width
    and Wu's peer susceptibility alpha, so a surviving ordering is
    evidence of robustness, not of equivalence.

Metrics per cell: post-FJ population mean and SD, raw and mean-centered
1-Wasserstein distance from the innate distribution, distance to that
model's frozen control, and the late-round window (26-30) of each.

Outputs (notes/pofd/fj_robustness/):
  fj_robustness_cells.csv    one row per cell, late-window summaries
  fj_robustness_rounds.csv   per cell per round
  fj_robustness.png / .pdf   arms per model, with the frozen reference
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
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
OUT_DIR = REPO / "notes" / "pofd" / "fj_robustness"
CONDOR = REPO / "experiments" / "condor"
LATE = (26, 30)          # inclusive, 1-indexed rounds
# EQUILIBRIUM IS STATIONARITY, NOT A VANISHING STEP. A fresh LoRA is
# trained every round, so there is an irreducible per-round noise floor
# and |x(t)-x(t-1)| never goes to zero -- late-window steps here sit flat
# at 1e-3..7e-2 with a 21-25 vs 26-30 ratio of 0.85-1.13, i.e. plateaued
# rather than still decaying. Demanding a tiny step therefore mislabels a
# settled system as unconverged. What actually distinguishes settled from
# still-moving is whether the population MEAN is drifting.
CONV_TOL = 1e-6          # retained only for the controls, which are noiseless
EQ_DRIFT = 0.005         # late-window mean drift, on the [0,1] opinion scale
COLLAPSE_SD = 0.02       # below this the population is a point, not a spread
# theta is a ratio; when lambda=0 and lambda->infinity land in nearly the
# same place there is no baseline to interpolate along and the ratio
# explodes. Reported as undetermined rather than as an overshoot.
THETA_MIN_DENOM = 0.03


def w1(a, b):
    return float(np.abs(np.sort(np.asarray(a)) - np.sort(np.asarray(b))).mean())


def w1_centered(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return w1(a - a.mean(), b - b.mean())


def read_tags(path=None, key="fj_robustness"):
    if path is None:
        path = CONDOR / f"configs_pofd_{key}.txt"
    return _read_tags(path)


def _read_tags(path):
    """The conceptual grid, read from the ON-DISK config the jobs run
    from. Never re-derived from a tag grammar here -- deriving one string
    in two places is how the 5em05/5em5 mismatch happened."""
    if not Path(path).exists():
        raise SystemExit(f"[fjr] missing {path}; run gen_pofd_sweep.py")
    return [ln.split(",")[0].strip()
            for ln in Path(path).read_text().splitlines() if ln.strip()]


ARMS = ("b0", "b1", "b2", "b8")


def arm_of(tag):
    hits = [a for a in ARMS if f"_{a}_beta" in tag]
    if len(hits) != 1:
        raise SystemExit(f"[fjr] cannot read arm from {tag!r}")
    return hits[0]


def seed_of(tag):
    m = re.search(r"_s(\d+)_", tag)
    if not m:
        raise SystemExit(f"[fjr] cannot read seed from {tag!r}")
    return int(m.group(1))


def model_of(tag):
    """Everything between the family prefix and the arm token. Model
    slugs contain underscores (qwen3_8b), so this cannot be a positional
    split."""
    arm = arm_of(tag)
    return tag.split("pofdfj_")[1].split(f"_{arm}_beta")[0]


def load_cell(roots, tag):
    for r in roots:
        p = Path(r) / tag / "trajectory.pt"
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=False)
    return None


def analyse(roots, out_dir, frozen_dir=None, key="fj_robustness"):
    tags = read_tags(key=key)
    models = sorted({model_of(t) for t in tags})
    cells, missing = {}, []
    for t in tags:
        d = load_cell(roots, t)
        if d is None:
            missing.append(t)
        else:
            cells[t] = d
    if missing:
        raise SystemExit(
            f"[fjr] HARD FAIL: {len(missing)}/{len(tags)} trained cells "
            f"absent -- a partial grid is a different claim.\n  "
            + "\n  ".join(missing))

    frozen, frozen_beta = load_frozen(frozen_dir or (out_dir / "frozen"),
                                      models)
    absent = [m for m in models if m not in frozen]
    if absent:
        raise SystemExit(
            f"[fjr] HARD FAIL: no frozen (lambda -> infinity) control for "
            f"{absent}. Build them with fj_controls.py --frozen-npz from "
            f"the archived zero-shot vectors plus the extraction job.")
    # THE FROZEN CONTROL IS BETA-SPECIFIC. x_init = (1-beta) innate +
    # beta m, so a beta=.5 reference is simply the wrong vector for the
    # beta=1 wave -- and every "distance to frozen" would be quietly
    # measured against it.
    cell_betas = {round(float(d["config"].get("w_plat", -1)), 6)
                  for d in cells.values()}
    if len(cell_betas) != 1:
        raise SystemExit(f"[fjr] HARD FAIL: mixed beta across cells: "
                         f"{cell_betas}")
    cb = next(iter(cell_betas))
    for m, b in frozen_beta.items():
        if abs(float(b) - cb) > 1e-9:
            raise SystemExit(
                f"[fjr] HARD FAIL: the frozen control for {m} was built at "
                f"beta={b} but these cells ran at beta={cb}. Rebuild with "
                f"fj_controls.py --frozen-beta {cb}.")

    innate = next(iter(cells.values()))["innate"].float().numpy()
    rounds_rows, cell_rows = [], []
    # every arm must share one environment or no cross-arm comparison means
    # anything: same innate vector, same graph hash, same agent ordering
    _assert_shared_environment(cells)

    for tag, d in cells.items():
        op = d["op_raw"].float().numpy()
        cfg = d["config"]
        m, a = model_of(tag), arm_of(tag)
        fz = frozen[m]
        for t in range(op.shape[0]):
            v = op[t]
            rounds_rows.append({
                "model": m, "arm": a, "tag": tag, "t": t + 1,
                "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "w1_from_innate": w1(v, innate),
                "w1_from_innate_centered": w1_centered(v, innate),
                "w1_to_frozen": w1(v, fz),
                "rmse_to_frozen": float(np.sqrt(np.mean((v - fz) ** 2))),
                "step_from_prev": (float(np.abs(op[t] - op[t - 1]).max())
                                   if t else float("nan")),
            })
        lo, hi = LATE
        late = op[lo - 1:hi]
        final_step = float(np.abs(op[-1] - op[-2]).max())
        # the noise floor this cell actually sits at, and whether the mean
        # is still moving through it
        late_steps = [float(np.abs(op[t] - op[t - 1]).max())
                      for t in range(max(lo - 1, 1), hi)]
        prev_steps = [float(np.abs(op[t] - op[t - 1]).max())
                      for t in range(max(lo - 6, 1), max(lo - 1, 2))]
        noise_floor = float(np.mean(late_steps)) if late_steps else float("nan")
        step_ratio = (noise_floor / float(np.mean(prev_steps))
                      if prev_steps and np.mean(prev_steps) > 0
                      else float("nan"))
        mean_drift = float(op[hi - 1].mean() - op[lo - 1].mean())
        cell_rows.append({
            "model": m, "arm": a, "tag": tag, "seed": seed_of(tag),
            "beta": float(cfg.get("w_plat", float("nan"))),

            "K_inner": int(cfg.get("fj_inner_steps", -1)),
            "peer_alpha": float(cfg.get("fj_peer_alpha", float("nan"))),
            "fj_update_version": cfg.get("fj_update_version"),
            "rounds": int(op.shape[0]),
            "late_mean": float(late.mean()),
            "late_sd": float(np.mean([x.std(ddof=1) for x in late])),
            "late_w1_from_innate": float(np.mean([w1(x, innate) for x in late])),
            "late_w1_from_innate_centered": float(
                np.mean([w1_centered(x, innate) for x in late])),
            "late_w1_to_frozen": float(np.mean([w1(x, fz) for x in late])),
            "late_rmse_to_frozen": float(np.mean(
                [np.sqrt(np.mean((x - fz) ** 2)) for x in late])),
            "final_step": final_step,
            "noise_floor": noise_floor,
            "step_ratio": step_ratio,
            "late_mean_drift": mean_drift,
            # stationary MEAN through the noise floor, not a vanishing step
            "converged": bool(abs(mean_drift) <= EQ_DRIFT),
        })

    add_theta(cell_rows, frozen)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len({r["seed"] for r in cell_rows}) > 1:
        agg = across_seeds(cell_rows)
        _csv(out_dir / "fj_robustness_by_seed.csv", agg)
        verdict = seed_verdict(agg)
        _csv(out_dir / "fj_robustness_seed_verdict.csv", verdict)
        print(f"\n[fjr] SEED REPLICATES: "
              f"{sorted({r['seed'] for r in cell_rows})}")
        print(f"[fjr] {'model':<14} {'SD gap (b1-b0)':>15} "
              f"{'pooled seed sd':>15} {'separated?':>11}")
        for v in verdict:
            sep = ("yes" if v["separated"] else
                   ("no" if v["separated"] is False else "n/a"))
            print(f"[fjr] {v['model']:<14} {v['sd_gap_b1_minus_b0']:>+15.5f} "
                  f"{v['pooled_seed_sd']:>15.5f} {sep:>11}")
        print("[fjr] a gap inside the pooled seed sd is TRAINING NOISE, "
              "not an arm effect -- which a single-seed count cannot say.")
    _csv(out_dir / "fj_robustness_rounds.csv", rounds_rows)
    _csv(out_dir / "fj_robustness_cells.csv", cell_rows)
    report(cell_rows)
    figure(rounds_rows, cell_rows, frozen, models, out_dir)
    return cell_rows


def _assert_shared_environment(cells):
    innates, graphs, nprof = set(), set(), set()
    for tag, d in cells.items():
        import hashlib
        innates.add(hashlib.sha256(
            d["innate"].float().contiguous().numpy().tobytes()).hexdigest())
        graphs.add(d["config"].get("fj_graph_sha256"))
        p = d.get("profiles")
        nprof.add(len(p[list(p)[0]]) if isinstance(p, dict) else -1)
    if len(innates) != 1:
        raise SystemExit(f"[fjr] HARD FAIL: {len(innates)} distinct innate "
                         f"vectors across arms -- not one environment")
    if len(graphs) != 1:
        raise SystemExit(f"[fjr] HARD FAIL: {len(graphs)} distinct graph "
                         f"hashes across arms: {graphs}")
    if len(nprof) != 1:
        raise SystemExit(f"[fjr] HARD FAIL: profile counts differ: {nprof}")


def load_frozen(frozen_dir, models):
    """Frozen FJ controls, one static post-FJ vector per model, plus the
    beta each was built at so the caller can refuse a mismatch."""
    out, betas = {}, {}
    p = Path(frozen_dir)
    if not p.exists():
        return out, betas
    for m in models:
        f = p / f"frozen_{m}.pt"
        if f.exists():
            d = torch.load(f, map_location="cpu", weights_only=False)
            out[m] = d["post_fj"].float().numpy()
            betas[m] = d.get("beta", float("nan"))
    return out, betas


def add_theta(rows, frozen):
    """theta: where each arm's CONSENSUS VALUE sits between plain SFT
    (lambda=0 -> 0) and no training (lambda -> infinity -> 1).

    WHY THIS IS THE HEADLINE WHEN CELLS COLLAPSE. At beta=1 the
    population contracts to a point, so SD and every centered
    distributional metric go to ~0 and stop discriminating. What is left
    is WHERE each arm collapsed, and the natural coordinate for that is
    the interval between the two endpoints already in hand. theta turns a
    scatter of means into one number per cell with a fixed meaning, and
    it is signed by the endpoints -- the frozen model can sit above or
    below plain SFT and theta still reads "fraction of the way toward no
    training".

    It is a RATIO, so it means nothing when the endpoints nearly
    coincide; below THETA_MIN_DENOM it is reported as UNDETERMINED
    rather than as a large number. It is deliberately NOT clamped to
    [0, 1]: an arm overshooting past no-training is a real outcome and
    must stay visible.
    """
    base = {r["model"]: r for r in rows if r["arm"] == "b0"}
    for r in rows:
        b0, fz = base.get(r["model"]), frozen.get(r["model"])
        if b0 is None or fz is None:
            r["theta"] = float("nan")
            r["theta_denom"] = float("nan")
            r["theta_status"] = ("no lambda=0 reference" if b0 is None
                                 else "no frozen reference")
            continue
        den = float(np.mean(fz)) - b0["late_mean"]
        r["theta_denom"] = den
        if abs(den) < THETA_MIN_DENOM:
            r["theta"] = float("nan")
            r["theta_status"] = (f"undetermined: lambda=0 and lambda->inf "
                                 f"differ by only {den:+.4f}")
        else:
            r["theta"] = (r["late_mean"] - b0["late_mean"]) / den
            r["theta_status"] = "ok"
    return rows


def across_seeds(rows):
    """Collapse seed replicates to one row per (model, arm), carrying the
    spread so a count is never reported as if it had no error bar.

    WHY THIS EXISTS. The seed-0 wave reported orderings as counts across
    models -- 6/6, 5/6 -- and a count from one seed cannot say whether
    the gap exceeds training noise. The FJ operator is deterministic, so
    the ONLY thing a seed moves is the learner; the seed spread reported
    here IS the training-noise scale against which those gaps have to be
    read.
    """
    out = {}
    for r in rows:
        out.setdefault((r["model"], r["arm"]), []).append(r)
    agg = []
    for (m, a), rs in sorted(out.items()):
        row = {"model": m, "arm": a, "n_seeds": len(rs),
               "seeds": ",".join(str(x["seed"]) for x in sorted(
                   rs, key=lambda z: z["seed"]))}
        for f in ("late_mean", "late_sd", "late_w1_from_innate",
                  "late_w1_from_innate_centered", "late_w1_to_frozen",
                  "theta", "late_mean_drift"):
            v = np.array([x.get(f, float("nan")) for x in rs], dtype=float)
            row[f] = float(np.nanmean(v))
            # sample sd across seeds; nan at n=1 rather than a fake 0
            row[f + "_seed_sd"] = (float(np.nanstd(v, ddof=1))
                                   if len(rs) > 1 else float("nan"))
        row["all_converged"] = all(bool(x["converged"]) for x in rs)
        agg.append(row)
    return agg


def seed_verdict(agg):
    """Per model, does the b0-vs-b1 gap clear the seed noise?

    A gap is called SEPARATED only when it exceeds the pooled seed sd of
    the two arms. Anything smaller is reported as within noise -- which
    is the whole point of running replicates, and the thing a 6/6 count
    could not tell you.
    """
    lines = []
    for m in sorted({r["model"] for r in agg}):
        b0 = next((r for r in agg if r["model"] == m and r["arm"] == "b0"), None)
        b1 = next((r for r in agg if r["model"] == m and r["arm"] == "b1"), None)
        if not (b0 and b1):
            continue
        gap = b1["late_sd"] - b0["late_sd"]
        pooled = np.sqrt(np.nansum([b0["late_sd_seed_sd"] ** 2,
                                    b1["late_sd_seed_sd"] ** 2]))
        # plain bool, not np.bool_: it lands in a CSV and gets compared
        # with `is True` by callers
        sep = (bool(abs(gap) > pooled)
               if np.isfinite(pooled) and pooled > 0 else None)
        lines.append({"model": m, "sd_gap_b1_minus_b0": gap,
                      "pooled_seed_sd": float(pooled),
                      "separated": sep,
                      "n_seeds": min(b0["n_seeds"], b1["n_seeds"])})
    return lines


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
    print(f"[fjr] wrote {path} ({len(rows)} rows)")


def report(rows):
    lo, hi = LATE
    nconv = sum(1 for r in rows if r["converged"])
    print(f"\n[fjr] seed 0, DESCRIPTIVE. Late window = rounds {lo}-{hi}.")
    print(f"[fjr] at equilibrium (late-window mean drift <= {EQ_DRIFT}): "
          f"{nconv}/{len(rows)} cells. The per-round step does NOT vanish "
          f"here -- a fresh LoRA trains every round, so there is a noise "
          f"floor -- so stationarity of the MEAN is the test, not step size.")
    ncol = sum(1 for r in rows if r["late_sd"] < COLLAPSE_SD)
    if ncol:
        print(f"[fjr] {ncol}/{len(rows)} cells have late SD < {COLLAPSE_SD}: "
              f"the population is a POINT there, so SD and centered W1 stop "
              f"discriminating and theta (where it collapsed, 0 = plain SFT, "
              f"1 = no training) is the live quantity.")
    print(f"\n[fjr] {'model':<14} {'arm':>4} {'late mean':>10} {'late SD':>9} "
          f"{'W1 innate':>10} {'W1 cent':>9} {'W1 frozen':>10} "
          f"{'theta':>7} {'drift':>9} {'floor':>9} {'eq':>5}")
    for m in sorted({r["model"] for r in rows}):
        for a in ARMS:
            sel = [r for r in rows if r["model"] == m and r["arm"] == a]
            if not sel:
                continue
            r = sel[0]
            print(f"[fjr] {m:<14} {a:>4} {r['late_mean']:>10.4f} "
                  f"{r['late_sd']:>9.4f} {r['late_w1_from_innate']:>10.4f} "
                  f"{r['late_w1_from_innate_centered']:>9.4f} "
                  f"{r['late_w1_to_frozen']:>10.4f} "
                  f"{r.get('theta', float('nan')):>7.3f} "
                  f"{r['late_mean_drift']:>+9.5f} "
                  f"{r['noise_floor']:>9.2e} {str(r['converged']):>5}")
            if r.get("theta_status") not in (None, "ok"):
                print(f"[fjr] {'':<14} {'':>4} theta {r['theta_status']}")
        b0 = [r for r in rows if r["model"] == m and r["arm"] == "b0"]
        b1 = [r for r in rows if r["model"] == m and r["arm"] == "b1"]
        if b0 and b1:
            d = b1[0]["late_w1_from_innate"] - b0[0]["late_w1_from_innate"]
            who = "b1 further" if d > 0 else "b0 further"
            print(f"[fjr] {'':<14} {'':>4} delta W1(b1-b0) = {d:+.4f}  ({who})")
    print("\n[fjr] No ordering is asserted here; one seed, one "
          "configuration, and no correspondence to the Deffuant gate width.")


def figure(rounds_rows, cell_rows, frozen, models, out_dir):
    ncol = 3
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    col = {"b0": "#c1443c", "b1": "#2a6fb5", "b2": "#6a9a56",
           "b8": "#8a5fa8"}
    for ax, m in zip(axes, models):
        for a in ARMS:
            sel = sorted([r for r in rounds_rows
                          if r["model"] == m and r["arm"] == a],
                         key=lambda r: r["t"])
            if not sel:
                continue
            ax.plot([r["t"] for r in sel], [r["w1_from_innate"] for r in sel],
                    lw=1.7, color=col[a],
                    label=("ordinary SFT" if a == "b0"
                           else f"forward-KL $\\lambda$={a[1:]}"))
        if m in frozen:
            innate_v = None
            # the frozen control is a POINT: constant from round 1, so it
            # is drawn as a level, not a curve
            ax.axhline(np.mean([r["w1_to_frozen"] for r in rounds_rows
                                if r["model"] == m and r["arm"] == "b0"][:1]
                               or [np.nan]),
                       color="#888888", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("round")
        ax.set_ylabel("$W_1$ from innate", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.text(0.03, 0.94, m, transform=ax.transAxes, fontsize=9,
                va="top", ha="left")
    for ax in axes[len(models):]:
        ax.axis("off")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = Path(out_dir) / f"fj_robustness.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[fjr] wrote {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", type=Path,
                    default=[REPO / "notes" / "pofd" / "cluster",
                             REPO / "runs" / "pokec_gated_lm"])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--frozen-dir", type=Path, default=None)
    ap.add_argument("--key", default="fj_robustness",
                    help="fj_robustness or fj_robustness_beta1")
    args = ap.parse_args()
    analyse(args.roots, args.out_dir, args.frozen_dir, key=args.key)
    print(f"[fjr] outputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
