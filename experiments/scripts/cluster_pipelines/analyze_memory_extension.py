#!/usr/bin/env python3
"""Section 3 MEMORY EXTENSION: does weakening the innate re-anchor move
the equilibrium toward the frozen model, and does the arm ordering hold?

k is the innate RE-ANCHOR strength (h = k*innate + (1-k)*x). k = 1 is the
stateless setup; k < 1 carries state forward. HYPOTHESIS: lower k weakens
the recurring pull back to innate opinions, so the platform's influence
on the equilibrium grows and the population lands nearer the frozen
model. Also tested: whether the ordering
  perfect prediction -> plain SFT -> lambda=1 -> lambda=8 -> frozen
survives as k falls.

MEAN AND SD ARE SEPARATE OUTCOMES. A shift of location is not
preservation of spread, and this tool never merges them.

PLOTTED STATES: exactly 61 per line -- the innate population at t = 0,
then the POST-PEER population after each of rounds 1..60. No within-round
intermediate state is exposed.

k = 1 and k = 0.2 come from the archived 100-round Section 3 cells,
truncated to the first 60 rounds (audited field-identical; truncation
assumes the population/peer streams are stateless in (seed, round)).

Run locally with threads pinned to 1.
"""
from __future__ import annotations
import argparse, os, sys, tempfile
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(tempfile.gettempdir(), "perfsim-plot-cache"))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.set_num_threads(1)

ROUNDS, W = 60, 0.5
KS = [1.0, 0.5, 0.2]
TRAINED = [("sft", "plain SFT"), ("fwdlam1", "forward $\\lambda=1$"),
           ("fwdlam8", "forward $\\lambda=8$")]
# the ordering under test, from most population-driven to most model-driven
ORDER = ["perfect", "sft", "fwdlam1", "fwdlam8", "frozen"]
LABEL = {"perfect": "perfect prediction", "sft": "plain SFT",
         "fwdlam1": "forward $\\lambda=1$", "fwdlam8": "forward $\\lambda=8$",
         "frozen": "frozen Qwen ($\\lambda\\to\\infty$)"}
COL = {"perfect": "0.15", "sft": "#2ca02c", "fwdlam1": "#4c72b0",
       "fwdlam8": "#c44e52", "frozen": "0.45"}
STY = {"perfect": "--", "sft": "-", "fwdlam1": "-", "fwdlam8": "-",
       "frozen": ":"}


def _num(v):
    return f"{v:g}".replace(".", "p")


def load_trained(root, arm, k):
    tag = (f"pofdmem_qwen3_8b_{arm}_eaopen_w0p5_k{_num(k)}"
           f"_esopen_anch2_s0_r60") if k == 0.5 else \
          (f"pofds3_qwen3_8b_{arm}_eaopen_w0p5_k{_num(k)}"
           f"_esopen_anch2_s0_r100")
    p = os.path.join(root, tag, "trajectory.pt")
    if not os.path.exists(p): return None, tag
    d = torch.load(p, map_location="cpu", weights_only=False)
    op = d["op_raw"].float().numpy()[:ROUNDS]
    if op.shape[0] < ROUNDS: return None, tag
    return (op, d["innate"].float().numpy()), tag


def load_cpu(path):
    if not os.path.exists(path): return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["op_raw"].float().numpy()[:ROUNDS], d["innate"].float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="notes/pofd/cluster")
    ap.add_argument("--pp-dir", default="notes/pofd/perfect_prediction")
    ap.add_argument("--fz-dir", default="notes/pofd/frozen_replay")
    ap.add_argument("--out", default="notes/pofd/section3_memory")
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data, missing, innate = {}, [], None
    for k in KS:
        for arm, _ in TRAINED:
            got, tag = load_trained(a.root, arm, k)
            if got is None: missing.append(tag)
            else:
                data[(k, arm)] = got[0]
                innate = innate if innate is not None else got[1]
        pp = os.path.join(a.pp_dir,
                          f"pp_k{_num(k)}_w{_num(W)}_eaopen_esopen_sw1_s0_r{ROUNDS}.pt")
        g = load_cpu(pp)
        if g is None: missing.append(pp)
        else: data[(k, "perfect")] = g[0]; innate = innate if innate is not None else g[1]
        fz = os.path.join(a.fz_dir,
                          f"frz_k{_num(k)}_w{_num(W)}_eaopen_esopen_sw1_s0_r{ROUNDS}.pt")
        g = load_cpu(fz)
        if g is None: missing.append(fz)
        else: data[(k, "frozen")] = g[0]; innate = innate if innate is not None else g[1]

    if missing and not a.allow_missing:
        print("[mem] HARD FAIL: missing cell(s):", file=sys.stderr)
        for m in missing: print(f"       {m}", file=sys.stderr)
        print("[mem] a silently short grid must not look complete; use "
              "--allow-missing to override.", file=sys.stderr)
        return 2
    if missing: print(f"[mem] WARNING: {len(missing)} cell(s) absent")

    # ---- CSVs ---------------------------------------------------------
    per = ["k,arm,round,post_peer_mean,post_peer_sd"]
    summ = ["k,arm,final_mean,final_sd,mean_51_60,sd_51_60,drift_mean,drift_sd,settled"]
    for k in KS:
        for arm in ORDER:
            if (k, arm) not in data: continue
            op = data[(k, arm)]
            m, s = op.mean(axis=1), op.std(axis=1)
            for t in range(op.shape[0]):
                per.append(f"{k:g},{arm},{t+1},{m[t]:.6f},{s[t]:.6f}")
            lm, ls = m[50:], s[50:]
            dm = float(lm[-5:].mean() - lm[:5].mean())
            ds = float(ls[-5:].mean() - ls[:5].mean())
            summ.append(f"{k:g},{arm},{m[-1]:.6f},{s[-1]:.6f},{lm.mean():.6f},"
                        f"{ls.mean():.6f},{dm:+.6f},{ds:+.6f},"
                        f"{'yes' if max(abs(dm),abs(ds))<=0.005 else 'no'}")
    open(os.path.join(a.out, "memory_per_round.csv"), "w").write("\n".join(per)+"\n")
    open(os.path.join(a.out, "memory_summary.csv"), "w").write("\n".join(summ)+"\n")

    # ---- figure: rows = mean / SD, cols = k ---------------------------
    fig, axes = plt.subplots(2, len(KS), figsize=(4.3*len(KS), 6.4), sharex=True)
    t = np.arange(0, ROUNDS+1)
    for c, k in enumerate(KS):
        for r, stat in enumerate(("mean", "sd")):
            ax = axes[r][c]
            for arm in ORDER:
                if (k, arm) not in data: continue
                op = data[(k, arm)]
                y0 = innate.mean() if stat == "mean" else innate.std()
                y = np.concatenate([[y0],
                                    op.mean(axis=1) if stat == "mean"
                                    else op.std(axis=1)])
                ax.plot(t, y, lw=1.5, color=COL[arm], ls=STY[arm],
                        label=LABEL[arm] if (r == 0 and c == 0) else None)
            ax.plot([0], [innate.mean() if stat == "mean" else innate.std()],
                    marker="*", ms=10, color="k", ls="none", zorder=6)
            ax.grid(alpha=.25, lw=.6); ax.set_xlim(-1, ROUNDS+1)
            if r == 1: ax.set_xlabel("round (post-peer, end of round)")
            if c == 0: ax.set_ylabel("population mean" if stat == "mean"
                                     else "population SD")
        axes[0][c].annotate(f"$k={k:g}$", xy=(.5, 1.02),
                            xycoords="axes fraction", ha="center",
                            va="bottom", fontsize=11)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=(0, .06, 1, .98))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"memory_extension.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- ordering test + report ---------------------------------------
    L = ["MEMORY EXTENSION -- does lower k move the equilibrium toward the",
         "frozen model, and does the arm ordering survive?", "",
         f"innate: mean {innate.mean():.4f}  SD {innate.std():.4f}", "",
         "Late-window (rounds 51-60) MEAN:", ""]
    L.append(f"{'k':>5}  " + "".join(f"{a_:<22}" for a_ in ORDER))
    for k in KS:
        row = f"{k:>5g}  "
        for arm in ORDER:
            row += (f"{data[(k,arm)].mean(axis=1)[50:].mean():<22.4f}"
                    if (k, arm) in data else f"{'--':<22}")
        L.append(row)
    L += ["", "Late-window (rounds 51-60) SD:", ""]
    L.append(f"{'k':>5}  " + "".join(f"{a_:<22}" for a_ in ORDER))
    for k in KS:
        row = f"{k:>5g}  "
        for arm in ORDER:
            row += (f"{data[(k,arm)].std(axis=1)[50:].mean():<22.4f}"
                    if (k, arm) in data else f"{'--':<22}")
        L.append(row)
    L += ["", "Distance of each arm's late MEAN to the frozen-model late MEAN",
          "(lower = nearer the frozen equilibrium):", ""]
    for k in KS:
        if (k, "frozen") not in data: continue
        fz = data[(k, "frozen")].mean(axis=1)[50:].mean()
        parts = []
        for arm in ORDER[:-1]:
            if (k, arm) in data:
                parts.append(f"{arm}={abs(data[(k,arm)].mean(axis=1)[50:].mean()-fz):.4f}")
        L.append(f"  k={k:g}: " + "  ".join(parts))
    L += ["", "Ordering test (perfect -> sft -> lam1 -> lam8 -> frozen, by",
          "distance to the frozen mean, should be DECREASING):", ""]
    for k in KS:
        if (k, "frozen") not in data: continue
        fz = data[(k, "frozen")].mean(axis=1)[50:].mean()
        ds = [abs(data[(k,arm)].mean(axis=1)[50:].mean()-fz)
              for arm in ORDER[:-1] if (k, arm) in data]
        mono = all(ds[i] >= ds[i+1]-1e-9 for i in range(len(ds)-1))
        L.append(f"  k={k:g}: {'HOLDS' if mono else 'BREAKS'}  ({', '.join(f'{d:.4f}' for d in ds)})")
    L += ["", "Read mean and SD separately: moving the location toward the",
          "frozen model is a different claim from preserving spread."]
    open(os.path.join(a.out, "memory_report.txt"), "w").write("\n".join(L)+"\n")
    print("\n".join(L))
    print(f"\n[mem] wrote CSVs, figure and report under {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
