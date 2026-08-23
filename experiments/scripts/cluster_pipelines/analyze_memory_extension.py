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

# ROUNDS is the analysis horizon; ENDPOINT_ROUNDS is the horizon the CPU
# endpoint artifacts were generated at. They differ on purpose: a 60-round
# perfect-prediction or frozen replay is deterministic given the seed, so
# its first 30 rounds ARE the 30-round run and the artifacts are reused by
# slicing rather than regenerated.
ROUNDS, ENDPOINT_ROUNDS, W = 30, 60, 0.5
KS = [1.0, 0.5, 0.2]
# S = COMPLETE Deffuant sweeps per retraining round, not pair interactions
SWEEPS = [1, 20, 100]
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


def load_trained(root, arm, S, k):
    # (S=1, k=1) and (S=1, k=0.2) are the archived 100-round Section 3
    # cells, truncated to 60; everything else is a pofdmem_ cell.
    tag = (f"pofds3_qwen3_8b_{arm}_eaopen_w0p5_k{_num(k)}"
           f"_esopen_anch2_s0_r100") if (S == 1 and k in (1.0, 0.2)) else \
          (f"pofdmem_qwen3_8b_{arm}_sw{S}_eaopen_w0p5_k{_num(k)}"
           f"_esopen_anch2_s0_r60")
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
    for S in SWEEPS:
        for k in KS:
            for arm, _ in TRAINED:
                got, tag = load_trained(a.root, arm, S, k)
                if got is None: missing.append(tag)
                else:
                    data[(S, k, arm)] = got[0]
                    innate = innate if innate is not None else got[1]
            for kind, dirp, key in (("pp", a.pp_dir, "perfect"),
                                    ("frz", a.fz_dir, "frozen")):
                f = os.path.join(dirp, f"{kind}_k{_num(k)}_w{_num(W)}"
                                       f"_eaopen_esopen_sw{S}_s0_r{ENDPOINT_ROUNDS}.pt")
                g = load_cpu(f)
                if g is None: missing.append(f)
                else:
                    data[(S, k, key)] = g[0]
                    innate = innate if innate is not None else g[1]

    if missing and not a.allow_missing:
        print("[mem] HARD FAIL: missing cell(s):", file=sys.stderr)
        for m in missing: print(f"       {m}", file=sys.stderr)
        print("[mem] a silently short grid must not look complete; use "
              "--allow-missing to override.", file=sys.stderr)
        return 2
    if missing: print(f"[mem] WARNING: {len(missing)} cell(s) absent")

    # ---- CSVs ---------------------------------------------------------
    per = ["sweeps,k,arm,round,post_peer_mean,post_peer_sd"]
    summ = ["sweeps,k,arm,final_mean,final_sd,mean_late,sd_late,drift_mean,drift_sd,settled"]
    for S in SWEEPS:
      for k in KS:
        for arm in ORDER:
            if (S, k, arm) not in data: continue
            op = data[(S, k, arm)]
            m, s = op.mean(axis=1), op.std(axis=1)
            for t in range(op.shape[0]):
                per.append(f"{S},{k:g},{arm},{t+1},{m[t]:.6f},{s[t]:.6f}")
            lm, ls = m[ROUNDS-10:], s[ROUNDS-10:]
            dm = float(lm[-5:].mean() - lm[:5].mean())
            ds = float(ls[-5:].mean() - ls[:5].mean())
            summ.append(f"{S},{k:g},{arm},{m[-1]:.6f},{s[-1]:.6f},{lm.mean():.6f},"
                        f"{ls.mean():.6f},{dm:+.6f},{ds:+.6f},"
                        f"{'yes' if max(abs(dm),abs(ds))<=0.005 else 'no'}")
    open(os.path.join(a.out, "memory_per_round.csv"), "w").write("\n".join(per)+"\n")
    open(os.path.join(a.out, "memory_summary.csv"), "w").write("\n".join(summ)+"\n")

    # ---- figure: 4 rows (S x {mean, SD}), 3 cols (k) -------------------
    rows_spec = [(S, stat) for S in SWEEPS for stat in ("mean", "sd")]
    fig, axes = plt.subplots(len(rows_spec), len(KS),
                             figsize=(4.3*len(KS), 3.1*len(rows_spec)),
                             sharex=True)
    t = np.arange(0, ROUNDS+1)
    for r, (S, stat) in enumerate(rows_spec):
        for c, k in enumerate(KS):
            ax = axes[r][c]
            for arm in ORDER:
                if (S, k, arm) not in data: continue
                op = data[(S, k, arm)]
                y0 = innate.mean() if stat == "mean" else innate.std()
                y = np.concatenate([[y0], op.mean(axis=1) if stat == "mean"
                                    else op.std(axis=1)])
                ax.plot(t, y, lw=1.5, color=COL[arm], ls=STY[arm],
                        label=LABEL[arm] if (r == 0 and c == 0) else None)
            ax.plot([0], [innate.mean() if stat == "mean" else innate.std()],
                    marker="*", ms=10, color="k", ls="none", zorder=6)
            ax.grid(alpha=.25, lw=.6); ax.set_xlim(-1, ROUNDS+1)
            if r == len(rows_spec)-1:
                ax.set_xlabel("round (post-peer, end of round)")
            if c == 0:
                ax.set_ylabel(f"$S={S}$\n" + ("population mean" if stat == "mean"
                                               else "population SD"))
            if r == 0:
                ax.annotate(f"$k={k:g}$", xy=(.5, 1.03),
                            xycoords="axes fraction", ha="center",
                            va="bottom", fontsize=11)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=(0, .04, 1, .99))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"memory_extension.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- the four tests -------------------------------------------------
    def late(S, k, arm, stat):
        op = data[(S, k, arm)]
        v = op.mean(axis=1) if stat == "mean" else op.std(axis=1)
        return float(v[ROUNDS-10:].mean())
    L = ["MEMORY EXTENSION x PEER STRENGTH", "",
         f"innate: mean {innate.mean():.4f}  SD {innate.std():.4f}",
         "S = COMPLETE Deffuant sweeps per retraining round.", ""]
    for stat, title in (("mean", f"Late-window (last 10 of {ROUNDS}) MEAN"),
                        ("sd", f"Late-window (last 10 of {ROUNDS}) SD")):
        L += [title, ""]
        L.append(f"{'S':>3} {'k':>5}  " + "".join(f"{x:<20}" for x in ORDER))
        for S in SWEEPS:
            for k in KS:
                row = f"{S:>3} {k:>5g}  "
                for arm in ORDER:
                    row += (f"{late(S,k,arm,stat):<20.4f}"
                            if (S, k, arm) in data else f"{'--':<20}")
                L.append(row)
        L.append("")
    L += ["TEST 1 -- does more peer interaction contract variance?", ""]
    for k in KS:
        for arm in ORDER:
            if (1, k, arm) in data and (20, k, arm) in data:
                a1, a20 = late(1,k,arm,"sd"), late(20,k,arm,"sd")
                L.append(f"  k={k:g} {arm:<12} SD {a1:.4f} -> {a20:.4f}  "
                         f"({'contracts' if a20 < a1 else 'DOES NOT contract'})")
    L += ["", "TEST 2 -- do the arms still select different MEANS once",
          f"variance has contracted (S=20)?", ""]
    for k in KS:
        ms = [(arm, late(20,k,arm,"mean")) for arm in ORDER if (20,k,arm) in data]
        if len(ms) >= 2:
            spread = max(v for _, v in ms) - min(v for _, v in ms)
            L.append(f"  k={k:g}: mean spread across arms = {spread:.4f}   "
                     + "  ".join(f"{a_}={v:.3f}" for a_, v in ms))
    L += ["", "TEST 3 -- does lower k move the equilibrium toward frozen?", ""]
    for S in SWEEPS:
        for arm in ORDER[:-1]:
            ds = [(k, abs(late(S,k,arm,"mean")-late(S,k,"frozen","mean")))
                  for k in KS if (S,k,arm) in data and (S,k,"frozen") in data]
            if len(ds) >= 2:
                mono = all(ds[i][1] >= ds[i+1][1]-1e-9 for i in range(len(ds)-1))
                L.append(f"  S={S:<3} {arm:<12} |mean - frozen| by k "
                         + " ".join(f"{k:g}:{d:.4f}" for k, d in ds)
                         + f"   {'nearer as k falls' if mono else 'NOT monotone'}")
    L += ["", "TEST 4 -- does the ordering perfect -> sft -> lam1 -> lam8 ->",
          "frozen persist (distance to frozen mean should DECREASE)?", ""]
    for S in SWEEPS:
        for k in KS:
            if (S,k,"frozen") not in data: continue
            fz = late(S,k,"frozen","mean")
            d = [abs(late(S,k,arm,"mean")-fz) for arm in ORDER[:-1]
                 if (S,k,arm) in data]
            if len(d) < 2: continue
            mono = all(d[i] >= d[i+1]-1e-9 for i in range(len(d)-1))
            L.append(f"  S={S:<3} k={k:<4g} {'HOLDS' if mono else 'BREAKS'}  "
                     f"({', '.join(f'{x:.4f}' for x in d)})")
    L += ["", "Mean and SD are SEPARATE outcomes: moving the location toward",
          "the frozen model is a different claim from preserving spread."]
    open(os.path.join(a.out, "memory_report.txt"), "w").write("\n".join(L)+"\n")
    print("\n".join(L))
    print(f"\n[mem] wrote CSVs, figure and report under {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
