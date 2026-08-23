#!/usr/bin/env python3
"""Fixed/live label mixture: post-peer mean and SD per round, and where
round 5 sits between the frozen-Qwen and plain-SFT equilibria.

FIVE ROUNDS IS A DIRECTIONAL TEST, NOT AN EQUILIBRIUM. Every table says
so, and no convergence flag is emitted, because none would be honest.

q is the fraction of SFT rows carrying the agent's CURRENT post-peer
opinion; the remaining 1-q carry that agent's ORIGINAL frozen-Qwen
prediction. q = 1 is ordinary SFT and comes from the archived
pofdps_qwen3_8b_sft_sw100_... cell, truncated to 5 rounds.

Endpoints: the frozen-Qwen population at the SAME surface (W=1, k=1,
S=100) from replay_frozen_offline, and plain SFT = the q=1 arm itself.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch
torch.set_num_threads(1)

N, ROUNDS = 723, 10
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 1.0]
SFT_TAG = "pofdps_qwen3_8b_sft_sw100_eaopen_w1_k1_esopen_anch2_s0_r60"
FRZ = "notes/pofd/frozen_replay/frz_k1_w1_eaopen_esopen_sw100_s0_r10.pt"


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

    if fz is not None and 1.0 in data:
        fzm, sftm = float(fz[-1].mean()), float(data[1.0][-1].mean())
        print(f"\nROUND-{ROUNDS} MEAN vs the two endpoints")
        print(f"  frozen-Qwen = {fzm:.4f}   plain SFT (q=1) = {sftm:.4f}")
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
        print("\nposition: 0 = on plain SFT, 1 = on frozen Qwen.")
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
    print(f"\n[mix] {ROUNDS} rounds is a DIRECTIONAL test, NOT an equilibrium;\n      an arm flagged STILL DRIFTING has no settled position to quote.")
    print(f"[mix] wrote CSVs under {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
