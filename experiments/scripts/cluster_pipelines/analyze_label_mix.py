#!/usr/bin/env python3
"""Fixed/live label mixture: post-peer mean and SD per round, and where
round 5 sits between the frozen-Qwen and plain-SFT equilibria.

FIVE ROUNDS IS A DIRECTIONAL TEST, NOT AN EQUILIBRIUM. Every table says
so, and no convergence flag is emitted, because none would be honest.

q is the fraction of SFT rows carrying the agent's CURRENT post-peer
opinion; the remaining 1-q carry that agent's ORIGINAL frozen-Qwen
prediction. q = 0 pins EVERY label to the frozen prediction; q = 1 is ordinary
SFT and comes from the archived
pofdps_qwen3_8b_sft_sw100_... cell, truncated to 5 rounds.

Endpoints: the frozen-Qwen population at the SAME surface (W=1, k=1,
S=100) from replay_frozen_offline, and plain SFT = the q=1 arm itself.
"""
from __future__ import annotations
import argparse, os, sys, tempfile
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(tempfile.gettempdir(), "perfsim-plot-cache"))
import numpy as np, torch
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.set_num_threads(1)

N, ROUNDS = 723, 30
QS = [0.0, 0.25, 0.5, 0.75, 1.0]
SFT_TAG = "pofdps_qwen3_8b_sft_sw100_eaopen_w1_k1_esopen_anch2_s0_r60"
FRZ = "notes/pofd/frozen_replay/frz_k1_w1_eaopen_esopen_sw100_s0_r30.pt"


def _num(v):
    return f"{v:g}".replace(".", "p")


def load(root, q):
    tag = SFT_TAG if q == 1.0 else \
        f"pofdmix_qwen3_8b_q{_num(q)}_sw100_eaopen_w1_k1_esopen_anch2_s0_r{ROUNDS}"
    p = os.path.join(root, tag, "trajectory.pt")
    if not os.path.exists(p):
        return None, tag
    d = torch.load(p, map_location="cpu", weights_only=False)
    op = d["op_raw"].float().numpy()[:ROUNDS]
    if op.shape[0] < ROUNDS:
        return None, tag
    return (op, d["innate"].float().numpy()), tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="notes/pofd/cluster")
    ap.add_argument("--frozen", default=FRZ)
    ap.add_argument("--out", default="notes/pofd/section3_label_mix")
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data, missing, innate = {}, [], None
    for q in QS:
        got, tag = load(a.root, q)
        if got is None: missing.append(tag)
        else:
            data[q] = got[0]
            innate = innate if innate is not None else got[1]
    if not os.path.exists(a.frozen):
        missing.append(a.frozen); fz = None
    else:
        fz = torch.load(a.frozen, map_location="cpu",
                        weights_only=False)["op_raw"].float().numpy()[:ROUNDS]
    if missing and not a.allow_missing:
        print("[mix] HARD FAIL: missing:", file=sys.stderr)
        for m in missing: print(f"       {m}", file=sys.stderr)
        return 2
    if missing: print(f"[mix] WARNING: {len(missing)} missing")

    rows = ["q,round,post_peer_mean,post_peer_sd"]
    print(f"innate: mean {innate.mean():.4f}  SD {innate.std():.4f}")
    if fz is not None:
        print(f"frozen-Qwen (W=1,k=1,S=100) round {ROUNDS}: "
              f"mean {fz[-1].mean():.4f}  SD {fz[-1].std():.4f}")
    print("\nPOST-PEER MEAN by round")
    print(f"{'q':>6}  " + "".join(f"r{t+1:<8}" for t in range(ROUNDS)))
    for q in QS:
        if q not in data: continue
        m = data[q].mean(axis=1)
        print(f"{q:>6g}  " + "".join(f"{v:<9.4f}" for v in m))
        for t in range(ROUNDS):
            rows.append(f"{q:g},{t+1},{m[t]:.6f},{data[q].std(axis=1)[t]:.6f}")
    print("\nPOST-PEER SD by round")
    print(f"{'q':>6}  " + "".join(f"r{t+1:<8}" for t in range(ROUNDS)))
    for q in QS:
        if q not in data: continue
        print(f"{q:>6g}  " + "".join(f"{v:<9.4f}" for v in data[q].std(axis=1)))
    open(os.path.join(a.out, "label_mix_per_round.csv"), "w").write(
        "\n".join(rows) + "\n")

    # Paper-facing view: t=0 is the shared innate population; every later
    # point is the population after the complete platform + peer loop.
    if data:
        qs = [q for q in QS if q in data]
        cmap = plt.get_cmap("viridis")
        colors = {q: cmap(i / max(1, len(qs) - 1))
                  for i, q in enumerate(qs)}
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.35))
        rounds = np.arange(ROUNDS + 1)
        for q in qs:
            mean = np.r_[innate.mean(), data[q].mean(axis=1)]
            sd = np.r_[innate.std(), data[q].std(axis=1)]
            label = f"q={q:g}"
            axes[0].plot(rounds, mean, marker="o", ms=2.8, lw=2,
                         color=colors[q], label=label)
            axes[1].plot(rounds, sd, marker="o", ms=2.8, lw=2,
                         color=colors[q], label=label)

        axes[0].axhline(innate.mean(), color="0.45", ls=":", lw=1.5,
                        label="innate $t=0$")
        axes[1].axhline(innate.std(), color="0.45", ls=":", lw=1.5)
        axes[0].set_title("Post-peer population mean")
        axes[1].set_title("Post-peer population SD")
        for ax in axes[:2]:
            ax.set_xlabel("Completed retraining round")
            ax.grid(alpha=.22)
            ax.set_xlim(0, ROUNDS)
        axes[0].set_ylabel("Population mean")
        axes[1].set_ylabel("Population SD")

        final_means = np.array([data[q][-1].mean() for q in qs])
        axes[2].plot(qs, final_means, color="0.25", lw=1.3, zorder=1)
        for q, value in zip(qs, final_means):
            axes[2].scatter(q, value, s=64, color=colors[q],
                            edgecolor="white", linewidth=.8, zorder=2)
        axes[2].axhline(final_means[0], color=colors[qs[0]], ls=":", lw=1.5,
                        label="all frozen labels")
        axes[2].axhline(final_means[-1], color=colors[qs[-1]], ls=":", lw=1.5,
                        label="plain SFT")
        if .5 in data:
            axes[2].annotate("still drifting", (.5, data[.5][-1].mean()),
                             xytext=(7, -17), textcoords="offset points",
                             fontsize=9)
        axes[2].set_title("Round-30 population mean")
        axes[2].set_xlabel("Live-label fraction $q$")
        axes[2].set_ylabel("Population mean")
        axes[2].set_xticks(qs)
        axes[2].grid(alpha=.22)
        axes[2].legend(frameon=False, fontsize=8, loc="best")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=6,
                   frameon=False, bbox_to_anchor=(.5, 1.025))
        fig.tight_layout(rect=(0, 0, 1, .92))
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.out, f"label_mix_30round.{ext}"),
                        dpi=240, bbox_inches="tight")
        plt.close(fig)

    # ENDPOINTS: the matched q=0 and q=1 ARMS, not the CPU frozen replay.
    # q=0 trains every round on the frozen labels, so it is the trained
    # counterpart of the frozen model and the right zero for this ladder;
    # the CPU replay never trains and is reported alongside for reference.
    if 0.0 in data and 1.0 in data:
        fzm, sftm = float(data[0.0][-1].mean()), float(data[1.0][-1].mean())
        if fz is not None:
            print(f"\n(CPU frozen replay, never trained: "
                  f"{float(fz[-1].mean()):.4f})")
        print(f"\nROUND-{ROUNDS} MEAN vs the two endpoints")
        print(f"  q=0 all-fixed = {fzm:.4f}   plain SFT (q=1) = {sftm:.4f}")
        print(f"{'q':>6}{'mean':>10}{'d(frozen)':>12}{'d(SFT)':>10}"
              f"{'position':>11}")
        s2 = ["q,mean_r5,dist_frozen,dist_sft,frac_toward_frozen"]
        for q in QS:
            if q not in data: continue
            mv = float(data[q][-1].mean())
            df, ds = abs(mv - fzm), abs(mv - sftm)
            # 0 = sitting on plain SFT, 1 = sitting on frozen Qwen
            frac = (sftm - mv) / (sftm - fzm) if abs(sftm - fzm) > 1e-9 else float("nan")
            print(f"{q:>6g}{mv:>10.4f}{df:>12.4f}{ds:>10.4f}{frac:>11.3f}")
            s2.append(f"{q:g},{mv:.6f},{df:.6f},{ds:.6f},{frac:.4f}")
        open(os.path.join(a.out, "label_mix_round5.csv"), "w").write(
            "\n".join(s2) + "\n")
        print("\nposition: 0 = on plain SFT (q=1), 1 = on the q=0 all-fixed arm.")
        # ---- IS EACH ARM STILL DRIFTING? -----------------------------
        # A fresh LoRA every round puts a noise floor under any
        # vanishing-step test, so drift is measured as the change in the
        # MEAN between the two halves of the last 6 rounds, compared with
        # the round-to-round jitter over the same window. An arm whose
        # half-to-half shift exceeds that jitter is STILL MOVING and its
        # position must not be read as a settled value.
        print(f"\nSTILL DRIFTING? (last 6 rounds, halves compared)")
        print(f"{'q':>6}{'first3':>10}{'last3':>10}{'shift':>10}"
              f"{'jitter':>9}  verdict")
        for q in QS:
            if q not in data: continue
            m = data[q].mean(axis=1)[-6:]
            a3, b3 = float(m[:3].mean()), float(m[3:].mean())
            shift = b3 - a3
            jit = float(np.abs(np.diff(m)).mean())
            moving = abs(shift) > max(jit, 1e-4)
            print(f"{q:>6g}{a3:>10.4f}{b3:>10.4f}{shift:>+10.4f}{jit:>9.4f}"
                  f"  {'STILL DRIFTING' if moving else 'settled'}")
    # ---- figure: mean and SD, one line per q ---------------------------
    # ROUNDS+1 points per line: innate at t=0 then post-peer rounds 1..N.
    COL = {0.0: "0.15", 0.25: "#4c72b0", 0.5: "#c44e52",
           0.75: "#55a868", 1.0: "#8172b2"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    t = np.arange(0, ROUNDS + 1)
    for ax, stat in zip(axes, ("mean", "sd")):
        for q in QS:
            if q not in data: continue
            op = data[q]
            y0 = innate.mean() if stat == "mean" else innate.std()
            y = np.concatenate([[y0], op.mean(axis=1) if stat == "mean"
                                else op.std(axis=1)])
            ax.plot(t, y, lw=1.6, color=COL.get(q, "0.5"),
                    label=f"$q={q:g}$" + (" (all fixed)" if q == 0 else
                                          " (plain SFT)" if q == 1 else ""))
        ax.plot([0], [innate.mean() if stat == "mean" else innate.std()],
                marker="*", ms=11, color="k", ls="none", zorder=6)
        if stat == "mean" and fz is not None:
            ax.axhline(float(fz[-1].mean()), color="0.55", ls=":", lw=1.2)
            ax.annotate("frozen Qwen (CPU replay)",
                        xy=(ROUNDS * 0.55, float(fz[-1].mean())),
                        fontsize=8, color="0.4", va="bottom")
        ax.set_xlabel("round (post-peer, end of round)")
        ax.set_ylabel("population mean" if stat == "mean"
                      else "population SD")
        ax.grid(alpha=.25, lw=.6); ax.set_xlim(-0.6, ROUNDS + 0.6)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"label_mix.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[mix] figure -> {os.path.join(a.out, 'label_mix.png')}")

    print(f"\n[mix] {ROUNDS} rounds is a DIRECTIONAL test, NOT an equilibrium;\n      an arm flagged STILL DRIFTING has no settled position to quote.")
    print(f"[mix] wrote CSVs and figure under {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
