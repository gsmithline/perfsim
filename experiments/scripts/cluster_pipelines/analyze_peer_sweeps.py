#!/usr/bin/env python3
"""Peer-sweep-strength analysis: does more within-round peer interaction
lower the post-peer SD plateau?

HYPOTHESIS, NOT EXPECTATION. The report states what the plateau does,
including "it did not move". Run locally with threads pinned to 1.

PLOTTED STATES: exactly 61 points per line -- the innate population at
t = 0, then the POST-PEER population after each of rounds 1..60. No
within-round intermediate state is ever exposed.

S = 1 comes from the archived 100-round Section 3 cells, TRUNCATED to
the first 60 rounds (audited field-identical; the truncation assumes the
population/peer streams are stateless in (seed, round)).

Outputs under notes/pofd/section3_peer_sweeps/:
  peer_sweeps_per_round.csv   arm, lam, sweeps, round, served SD (pre-peer),
                              post-peer SD, contraction ratio
  peer_sweeps_summary.csv     mean post-peer SD over rounds 51-60, final SD,
                              late drift, settled flag
  peer_sweeps_sd.{png,pdf}    3 panels (SFT, fwd 1, fwd 8), 4 lines each
  peer_sweeps_report.txt      the plain-language answer
"""
from __future__ import annotations
import argparse, os, sys, tempfile
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(tempfile.gettempdir(), "perfsim-plot-cache"))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.set_num_threads(1)

ROUNDS, LATE = 60, (50, 60)          # late window = rounds 51..60 (0-based 50:60)
ARMS = [("sft", 0.0, "ordinary SFT ($\\lambda=0$)"),
        ("fwdlam1", 1.0, "forward KL $\\lambda=1$"),
        ("fwdlam8", 8.0, "forward KL $\\lambda=8$")]
SWEEPS = [1, 5, 20, 100]
REUSED = {a: f"pofds3_qwen3_8b_{a}_eaopen_w1_k1_esopen_anch2_s0_r100"
          for a, _, _ in ARMS}
COL = {1: "#4c72b0", 5: "#55a868", 20: "#c44e52", 100: "#8172b2"}


def load(root, arm, S):
    tag = REUSED[arm] if S == 1 else \
        f"pofdps_qwen3_8b_{arm}_sw{S}_eaopen_w1_k1_esopen_anch2_s0_r60"
    p = os.path.join(root, tag, "trajectory.pt")
    if not os.path.exists(p):
        return None, tag
    d = torch.load(p, map_location="cpu", weights_only=False)
    op = d["op_raw"].float().numpy()[:ROUNDS]
    pr = d["pred_raw"].float().numpy()[:ROUNDS]
    if op.shape[0] < ROUNDS:
        return None, tag
    return (op, pr, d["innate"].float().numpy()), tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="notes/pofd/cluster")
    ap.add_argument("--out", default="notes/pofd/section3_peer_sweeps")
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data, missing = {}, []
    for arm, lam, _ in ARMS:
        for S in SWEEPS:
            got, tag = load(a.root, arm, S)
            if got is None: missing.append(tag)
            else: data[(arm, S)] = got
    if missing and not a.allow_missing:
        print("[ps] HARD FAIL: missing cell(s):", file=sys.stderr)
        for m in missing: print(f"       {m}", file=sys.stderr)
        print("[ps] a silently short grid must not look like a complete "
              "result; pass --allow-missing to override.", file=sys.stderr)
        return 2
    if missing:
        print(f"[ps] WARNING: plotting WITHOUT {len(missing)} cell(s)")

    per = ["arm,lam,sweeps,round,served_sd_pre_peer,post_peer_sd,contraction_ratio"]
    summ = ["arm,lam,sweeps,mean_post_sd_51_60,final_sd,late_drift,settled"]
    innate = next(iter(data.values()))[2] if data else None
    for arm, lam, _ in ARMS:
        for S in SWEEPS:
            if (arm, S) not in data: continue
            op, pr, _ = data[(arm, S)]
            pre, post = pr.std(axis=1), op.std(axis=1)
            for t in range(ROUNDS):
                per.append(f"{arm},{lam:g},{S},{t+1},{pre[t]:.6f},{post[t]:.6f},"
                           f"{post[t]/max(pre[t],1e-12):.6f}")
            late = post[LATE[0]:LATE[1]]
            # late drift: mean of the last 5 minus mean of the first 5 of the
            # late window. A fresh LoRA every round puts a noise floor under
            # any vanishing-step criterion, so drift is the honest test.
            drift = float(late[-5:].mean() - late[:5].mean())
            summ.append(f"{arm},{lam:g},{S},{late.mean():.6f},{post[-1]:.6f},"
                        f"{drift:+.6f},{'yes' if abs(drift) <= 0.005 else 'no'}")
    open(os.path.join(a.out, "peer_sweeps_per_round.csv"), "w").write("\n".join(per) + "\n")
    open(os.path.join(a.out, "peer_sweeps_summary.csv"), "w").write("\n".join(summ) + "\n")

    # ---- figure: 3 panels, 4 lines, shared y, 61 points --------------
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), sharey=True)
    t = np.arange(0, ROUNDS + 1)
    for ax, (arm, lam, lab) in zip(axes, ARMS):
        for S in SWEEPS:
            if (arm, S) not in data: continue
            op, _pr, inn = data[(arm, S)]
            # t=0 is the INNATE population, then post-peer rounds 1..60
            y = np.concatenate([[inn.std()], op.std(axis=1)])
            ax.plot(t, y, lw=1.5, color=COL[S], label=f"$S={S}$")
        if innate is not None:
            ax.plot([0], [innate.std()], marker="*", ms=11, color="k",
                    ls="none", zorder=6)
        ax.set_xlabel("round (post-peer, end of round)")
        ax.grid(alpha=.25, lw=.6); ax.set_xlim(-1, ROUNDS + 1)
        ax.annotate(lab, xy=(.5, 1.02), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("post-peer population SD")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"peer_sweeps_sd.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- report -------------------------------------------------------
    lines = ["PEER-SWEEP STRENGTH: does more within-round peer interaction",
             "lower the post-peer SD plateau?", "",
             "Late-window (rounds 51-60) mean post-peer SD:", ""]
    lines.append(f"{'arm':<24}" + "".join(f"S={S:<7}" for S in SWEEPS))
    verdicts = []
    for arm, lam, lab in ARMS:
        row, vals = f"{arm:<24}", []
        for S in SWEEPS:
            if (arm, S) in data:
                op = data[(arm, S)][0]
                v = float(op.std(axis=1)[LATE[0]:LATE[1]].mean())
                vals.append((S, v)); row += f"{v:<9.4f}"
            else: row += f"{'--':<9}"
        lines.append(row)
        if len(vals) >= 2:
            mono = all(vals[i][1] >= vals[i + 1][1] - 1e-9
                       for i in range(len(vals) - 1))
            drop = vals[0][1] - vals[-1][1]
            verdicts.append((arm, mono, drop, vals))
    lines += ["", "Verdict per arm (hypothesis: more sweeps -> lower plateau):"]
    for arm, mono, drop, vals in verdicts:
        lines.append(f"  {arm:<12} monotone decreasing in S: "
                     f"{'YES' if mono else 'NO'};  "
                     f"S=1 -> S={vals[-1][0]} change: {-drop:+.4f}")
    lines += ["", "Read with care: this is a HYPOTHESIS test, not a gate. A",
              "plateau that does not move is a real answer, and a plateau at",
              "~0 means the peer process alone drives the population to",
              "consensus at that sweep count."]
    open(os.path.join(a.out, "peer_sweeps_report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[ps] wrote CSVs, figure and report under {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
