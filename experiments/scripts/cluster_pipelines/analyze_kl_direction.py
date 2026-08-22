#!/usr/bin/env python3
"""Forward vs reverse KL on the QWU surface -- figure and round-10 table.

RUN ON THE CLUSTER (or locally once the wave is pulled).

THE QUESTION. Does raising lambda move the population toward frozen
Qwen while PRESERVING heterogeneity, and does reverse KL show that more
clearly without collapsing the served map? A mean shift alone does not
answer it: a population can march toward the frozen mean while every
agent converges on one value, which is the opposite of retention. So
mean and SD are separate rows of the figure, and the table carries the
served map's dispersion, distinct-value count and largest-mode share
beside every distance.

THESE ARE 10-ROUND POST-PEER DYNAMICS, NOT EQUILIBRIA. The QWU cells
were still moving at round 10 of 100. Nothing here is labelled a fixed
point, and the axis label says so.

WHAT IS PLOTTED, EXACTLY. Rounds 1..10 are op_raw, which the runner
appends AFTER the peer sweeps (peers always run last), so every point
is an end-of-round post-peer population state. t=0 is the innate vector
-- a separate marker, never joined into the curve, because it is a
different object: the population before any platform round.

ARMS. Ten queued cells plus four reused/derived:
  perfect prediction   sim_perfect_predictor (CPU oracle, served = x)
  ordinary SFT lam=0   archived QWU b0
  forward lam=1        archived QWU b1
  frozen Qwen          replay_frozen_offline (CPU, constant served b)
The two archived cells ran 100 rounds; their first 10 are used. The
runner's population and peer streams are stateless in (seed, round), so
a prefix is the same trajectory a 10-round run would have produced --
but this is an ASSUMPTION about the stream, not a measurement, and it is
recorded in the output so a reader can weigh it.

Usage:
  python analyze_kl_direction.py --runs-root runs/pokec_gated_lm \\
      --out notes/pofd/figures
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(tempfile.gettempdir(), "perfsim-plot-cache"))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_AGENTS = 723
ROUNDS = 10
WS = (0.5, 1.0)
LAMS = (0.1, 1.0, 10.0)
QWU_B0 = "pofdqwu_qwen7b_b0_eaopen_w{wt}_l1_esopen_s0_r100"
QWU_B1 = "pofdqwu_qwen7b_b1_eaopen_w{wt}_l1_esopen_s0_r100"
KD = "pofdkd_qwen7b_{d}lam{lt}_eaopen_w{wt}_l1_esopen_s0_r10"


def _num(v):
    """project tag grammar: 0.5 -> 0p5, 1.0 -> 1, 0.1 -> 0p1, 10 -> 10"""
    s = f"{v:g}"
    return s.replace(".", "p")


def w1(a, b):
    """1-Wasserstein between two equal-size empirical distributions."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.shape != b.shape:
        n = min(a.size, b.size)
        qs = (np.arange(n) + 0.5) / n
        a = np.quantile(a, qs)
        b = np.quantile(b, qs)
    return float(np.abs(a - b).mean())


def load_gpu(run_dir):
    """(op[10,n], pred[10,n], innate[n], parse_fail_max) or None."""
    tf = os.path.join(run_dir, "trajectory.pt")
    if not os.path.exists(tf):
        return None
    d = torch.load(tf, map_location="cpu", weights_only=False)
    op = d.get("op_raw")
    pr = d.get("pred_raw")
    if op is None or pr is None or not torch.is_tensor(op) or op.numel() == 0:
        return None
    op = op.float()[:ROUNDS].numpy()
    pr = pr.float()[:ROUNDS].numpy()
    innate = d.get("innate")
    innate = innate.float().numpy() if torch.is_tensor(innate) else None
    pf = None
    gz = os.path.join(run_dir, "raw_gen_log.json.gz")
    if os.path.exists(gz):
        vals = [r.get("parse_fail_frac") for r in
                (json.loads(l) for l in gzip.open(gz, "rt"))]
        vals = [float(v) for v in vals[:ROUNDS] if v is not None]
        pf = max(vals) if vals else None
    if op.shape[0] < ROUNDS:
        return None
    return op, pr, innate, pf


def load_cpu(path):
    """Endpoint artifacts from sim_perfect_predictor / replay_frozen_offline.
    Returns (op[10,n], served[10,n]) -- key names differ between the two,
    so both spellings are accepted and a miss is loud."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    op = None
    for k in ("pp_raw", "op_raw", "pop_raw", "x_raw"):
        if torch.is_tensor(d.get(k)):
            op = d[k].float()[:ROUNDS].numpy()
            break
    served = None
    for k in ("served_raw", "pred_raw", "served"):
        if torch.is_tensor(d.get(k)):
            served = d[k].float()[:ROUNDS].numpy()
            break
    if op is None:
        raise KeyError(f"{path}: no population tensor; keys={list(d)}")
    return op, served


def collect(runs_root, pp_dir, fz_dir):
    """arms[w] -> list of (label, kind, op, pred, parse_fail). kind is one
    of perfect/sft0/fwd/rev/frozen and drives styling."""
    arms, innate, missing = {w: [] for w in WS}, None, []
    for w in WS:
        wt = _num(w)
        # --- CPU endpoints ---------------------------------------------
        pp = Path(pp_dir) / (f"pp_k1_w{wt}_eaopen_esopen_sw1_s0_r{ROUNDS}.pt")
        if pp.exists():
            op, sv = load_cpu(pp)
            arms[w].append(("perfect prediction", "perfect", op, sv, 0.0))
        else:
            missing.append(str(pp))
        fz = sorted(Path(fz_dir).glob(
            f"*k1_w{wt}_eaopen_esopen*s0_r{ROUNDS}.pt"))
        if fz:
            op, sv = load_cpu(fz[0])
            arms[w].append((r"frozen Qwen ($\lambda\to\infty$)", "frozen",
                            op, sv, 0.0))
        else:
            missing.append(f"{fz_dir}/*k1_w{wt}_eaopen_esopen*r{ROUNDS}.pt")
        # --- reused archived cells -------------------------------------
        for tmpl, lab, kind in ((QWU_B0, r"SFT $\lambda=0$", "sft0"),
                                (QWU_B1, r"forward $\lambda=1$", "fwd")):
            r = load_gpu(os.path.join(runs_root, tmpl.format(wt=wt)))
            if r is None:
                missing.append(tmpl.format(wt=wt))
                continue
            op, pr, inn, pf = r
            innate = innate if innate is not None else inn
            arms[w].append((lab, kind, op, pr, pf))
        # --- the queued wave -------------------------------------------
        for d_, kind in (("fwd", "fwd"), ("rev", "rev")):
            for lam in LAMS:
                if d_ == "fwd" and lam == 1.0:
                    continue                      # reused above
                tag = KD.format(d=d_, lt=_num(lam), wt=wt)
                r = load_gpu(os.path.join(runs_root, tag))
                if r is None:
                    missing.append(tag)
                    continue
                op, pr, inn, pf = r
                innate = innate if innate is not None else inn
                name = "forward" if d_ == "fwd" else "reverse"
                arms[w].append((rf"{name} $\lambda={lam:g}$", kind,
                                op, pr, pf))
    return arms, innate, missing


# forward = blues, reverse = reds, deepening with lambda; the endpoints
# are achromatic so neither direction can be confused with them
SHADE = {0.1: 0.42, 1.0: 0.62, 10.0: 0.86}
STYLE = {
    "perfect": dict(color="0.15", ls="--", lw=1.6, marker=None, zorder=5),
    "frozen": dict(color="0.45", ls=":", lw=2.0, marker=None, zorder=5),
    "sft0": dict(color="#2ca02c", ls="-", lw=1.5, marker="o", ms=3),
}


def style_for(kind, lam):
    if kind in STYLE:
        return dict(STYLE[kind])
    cmap = plt.get_cmap("Blues" if kind == "fwd" else "Reds")
    return dict(color=cmap(SHADE.get(lam, 0.6)), lw=1.6,
                ls="-" if kind == "fwd" else "-.",
                marker="s" if kind == "fwd" else "^", ms=3.4)


def lam_of(label):
    if "=0$" in label:
        return 0.0
    for lam in LAMS:
        if f"={lam:g}$" in label:
            return lam
    return None


def make_figure(arms, innate, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.2), sharex=True)
    t = np.arange(1, ROUNDS + 1)
    for col, w in enumerate(WS):
        for row, stat in enumerate(("mean", "sd")):
            ax = axes[row][col]
            for label, kind, op, _pred, _pf in arms[w]:
                y = op.mean(axis=1) if stat == "mean" else op.std(axis=1)
                ax.plot(t, y, label=label,
                        **style_for(kind, lam_of(label)))
            if innate is not None:
                y0 = float(innate.mean()) if stat == "mean" \
                    else float(innate.std())
                # t=0 is a POINT, never joined to the curves: it is the
                # population before any platform round, a different object
                ax.plot([0], [y0], marker="*", ms=11, color="k",
                        ls="none", zorder=6,
                        label="innate ($t=0$)" if row == 0 and col == 0
                        else None)
            ax.grid(alpha=.25, lw=.6)
            ax.set_xlim(-0.6, ROUNDS + 0.4)
            ax.set_xticks([0, 2, 4, 6, 8, 10])
            if row == 1:
                ax.set_xlabel("round (end-of-round, post-peer)")
            if col == 0:
                ax.set_ylabel("population mean" if stat == "mean"
                              else "population SD")
        axes[0][col].annotate(rf"$\beta={w:g}$", xy=(.5, 1.02),
                              xycoords="axes fraction", ha="center",
                              va="bottom", fontsize=11)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False,
               fontsize=8.5, bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=(0, .07, 1, .98))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"kl_direction_qwen_10round.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_dir / "kl_direction_qwen_10round.png"


def round10_table(arms, out_dir):
    rows = []
    for w in WS:
        fz = next((a for a in arms[w] if a[1] == "frozen"), None)
        fz_pop = fz[2][ROUNDS - 1] if fz else None
        fz_srv = fz[3][ROUNDS - 1] if (fz and fz[3] is not None) else None
        for label, kind, op, pred, pf in arms[w]:
            pop = op[ROUNDS - 1]
            srv = pred[ROUNDS - 1] if pred is not None else None
            r = {
                "beta": w, "arm": label.replace("$", "").replace("\\", ""),
                "kind": kind,
                "pop_W1_to_frozen": (w1(pop, fz_pop) if fz_pop is not None
                                     else float("nan")),
                "pop_mean": float(pop.mean()), "pop_sd": float(pop.std()),
            }
            if srv is not None:
                vals, counts = np.unique(np.round(srv, 6), return_counts=True)
                r.update({
                    "served_W1_to_frozen": (w1(srv, fz_srv)
                                            if fz_srv is not None
                                            else float("nan")),
                    "served_corr_frozen": (
                        float(np.corrcoef(srv, fz_srv)[0, 1])
                        if fz_srv is not None and srv.std() > 0
                        and fz_srv.std() > 0 else float("nan")),
                    "served_agree_frozen": (
                        float((srv == fz_srv).mean())
                        if fz_srv is not None else float("nan")),
                    "pred_sd": float(srv.std()),
                    "pred_n_distinct": int(vals.size),
                    "pred_mode_share": float(counts.max() / srv.size),
                })
            else:
                r.update({k: float("nan") for k in
                          ("served_W1_to_frozen", "served_corr_frozen",
                           "served_agree_frozen", "pred_sd",
                           "pred_mode_share")})
                r["pred_n_distinct"] = -1
            r["parse_fail_max"] = (float("nan") if pf is None else float(pf))
            rows.append(r)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "kl_direction_qwen_round10.csv"
    cols = ["beta", "arm", "kind", "pop_W1_to_frozen", "pop_mean", "pop_sd",
            "served_W1_to_frozen", "served_corr_frozen",
            "served_agree_frozen", "pred_sd", "pred_n_distinct",
            "pred_mode_share", "parse_fail_max"]
    with open(csv, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(
                (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]))
                for c in cols) + "\n")
    return rows, csv


def print_table(rows):
    hdr = (f"{'beta':>4} {'arm':<24} {'popW1':>7} {'mean':>7} {'sd':>7} "
           f"{'srvW1':>7} {'corr':>6} {'agree':>6} {'predSD':>7} "
           f"{'ndist':>5} {'mode':>6} {'pfail':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['beta']:>4g} {r['arm']:<24} "
              f"{r['pop_W1_to_frozen']:>7.4f} {r['pop_mean']:>7.4f} "
              f"{r['pop_sd']:>7.4f} {r['served_W1_to_frozen']:>7.4f} "
              f"{r['served_corr_frozen']:>6.3f} "
              f"{r['served_agree_frozen']:>6.3f} {r['pred_sd']:>7.4f} "
              f"{r['pred_n_distinct']:>5d} {r['pred_mode_share']:>6.3f} "
              f"{r['parse_fail_max']:>6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs/pokec_gated_lm")
    ap.add_argument("--pp-dir", default="notes/pofd/perfect_prediction")
    ap.add_argument("--fz-dir", default="notes/pofd/frozen_replay")
    ap.add_argument("--out", default="notes/pofd/figures")
    ap.add_argument("--allow-missing", action="store_true",
                    help="plot whatever is present. OFF by default: a "
                         "silently short ladder reads as a real result")
    args = ap.parse_args()

    arms, innate, missing = collect(args.runs_root, args.pp_dir, args.fz_dir)
    if missing and not args.allow_missing:
        print("[kd] HARD FAIL: missing arm(s) --", file=sys.stderr)
        for m in missing:
            print(f"        {m}", file=sys.stderr)
        print("[kd] gate and complete the wave, or pass --allow-missing "
              "to plot a deliberately partial ladder.", file=sys.stderr)
        return 2
    if missing:
        print(f"[kd] WARNING: plotting WITHOUT {len(missing)} arm(s): "
              f"{', '.join(missing)}")
    for w in WS:
        print(f"[kd] beta={w:g}: {len(arms[w])} arms")
    png = make_figure(arms, innate, args.out)
    rows, csv = round10_table(arms, args.out)
    print_table(rows)
    print(f"\n[kd] figure -> {png}")
    print(f"[kd] table  -> {csv}")
    print("[kd] 10-round post-peer dynamics, NOT equilibria.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
