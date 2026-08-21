#!/usr/bin/env python3
"""HARD-GATED analyzer for the FJ robustness wave (2026-08-21). CPU only.

THE QUESTION. Does the ordinary-SFT (b0) vs forward-KL-SFT (b1) result
survive when Deffuant bounded-confidence peers are replaced by a LINEAR
FJ operator? Six models x two arms, seed 0, one configuration.

HARD FAIL UNTIL THE GRID IS COMPLETE. Twelve trained cells and six
frozen controls. A partial grid gets refused rather than analysed,
because a per-model ordering read off whichever cells happened to finish
is a different claim from the one this wave was built to make.

WHAT IS AND IS NOT CLAIMED.
  * Seed 0 only. Everything here is DESCRIPTIVE. No ordering is called
    significant, and none is assumed in advance -- the Deffuant result
    may or may not survive, and both outcomes are reportable.
  * Round 30 is NOT an equilibrium unless convergence says so. The
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
    and the FJ neighbour weight alpha, so a surviving ordering is
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
CONV_TOL = 1e-6


def w1(a, b):
    return float(np.abs(np.sort(np.asarray(a)) - np.sort(np.asarray(b))).mean())


def w1_centered(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return w1(a - a.mean(), b - b.mean())


def read_tags(path=CONDOR / "configs_pofd_fj_robustness.txt"):
    """The conceptual grid, read from the ON-DISK config the jobs run
    from. Never re-derived from a tag grammar here -- deriving one string
    in two places is how the 5em05/5em5 mismatch happened."""
    if not Path(path).exists():
        raise SystemExit(f"[fjr] missing {path}; run gen_pofd_sweep.py")
    return [ln.split(",")[0].strip()
            for ln in Path(path).read_text().splitlines() if ln.strip()]


def arm_of(tag):
    hits = [a for a in ("b0", "b1") if f"_{a}_beta" in tag]
    if len(hits) != 1:
        raise SystemExit(f"[fjr] cannot read arm from {tag!r}")
    return hits[0]


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


def analyse(roots, out_dir, frozen_dir=None):
    tags = read_tags()
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

    frozen = load_frozen(frozen_dir or (out_dir / "frozen"), models)
    absent = [m for m in models if m not in frozen]
    if absent:
        raise SystemExit(
            f"[fjr] HARD FAIL: no frozen (lambda -> infinity) control for "
            f"{absent}. Build them with fj_controls.py from the archived "
            f"zero-shot vectors (five reused) plus the extraction job.")

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
        cell_rows.append({
            "model": m, "arm": a, "tag": tag,
            "beta": float(cfg.get("w_plat", float("nan"))),
            "alpha": float(cfg.get("fj_alpha", float("nan"))),
            "n_inner": int(cfg.get("fj_inner_steps", -1)),
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
            # round 30 is round 30 unless this says otherwise
            "converged": bool(final_step <= CONV_TOL),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
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
    """Frozen FJ controls, one static post-FJ vector per model."""
    out = {}
    p = Path(frozen_dir)
    if not p.exists():
        return out
    for m in models:
        f = p / f"frozen_{m}.pt"
        if f.exists():
            out[m] = torch.load(f, map_location="cpu",
                                weights_only=False)["post_fj"].float().numpy()
    return out


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
    print(f"[fjr] converged (final step <= {CONV_TOL}): {nconv}/{len(rows)} "
          f"cells. Where False, round 30 is round 30, not an equilibrium.")
    print(f"\n[fjr] {'model':<14} {'arm':>4} {'late mean':>10} {'late SD':>9} "
          f"{'W1 innate':>10} {'W1 cent':>9} {'W1 frozen':>10} "
          f"{'final step':>11} {'conv':>5}")
    for m in sorted({r["model"] for r in rows}):
        for a in ("b0", "b1"):
            sel = [r for r in rows if r["model"] == m and r["arm"] == a]
            if not sel:
                continue
            r = sel[0]
            print(f"[fjr] {m:<14} {a:>4} {r['late_mean']:>10.4f} "
                  f"{r['late_sd']:>9.4f} {r['late_w1_from_innate']:>10.4f} "
                  f"{r['late_w1_from_innate_centered']:>9.4f} "
                  f"{r['late_w1_to_frozen']:>10.4f} {r['final_step']:>11.2e} "
                  f"{str(r['converged']):>5}")
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
    col = {"b0": "#c1443c", "b1": "#2a6fb5"}
    for ax, m in zip(axes, models):
        for a in ("b0", "b1"):
            sel = sorted([r for r in rounds_rows
                          if r["model"] == m and r["arm"] == a],
                         key=lambda r: r["t"])
            if not sel:
                continue
            ax.plot([r["t"] for r in sel], [r["w1_from_innate"] for r in sel],
                    lw=1.7, color=col[a],
                    label=("ordinary SFT" if a == "b0" else "forward-KL SFT"))
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
    args = ap.parse_args()
    analyse(args.roots, args.out_dir, args.frozen_dir)
    print(f"[fjr] outputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
