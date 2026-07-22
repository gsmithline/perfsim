#!/usr/bin/env python3
"""Sanity gate for the pofd_ platform-only fresh-data runs.

Per run dir (needs trajectory.pt written by run_pokec_gated_lm.py), checks:
  1. CONFIG    the run really is the pofd design: eps(social)=0, w_plat=1,
               data_regime=replace, fresh_each_round=True, innate_lambda=0,
               pop=ab, mode=loop, canary=0, single sweep, no pop reset.
  2. NO-PEER   row['accepted'] (peer pairs that moved) == 0 in EVERY round.
  3. EXACT-COPY per round t, with x_before = innate (t=0) or op_raw[t-1]:
               gate_i = |served_i - x_before_i| < eps_ai. Accepted agents must
               land EXACTLY on the served prediction (W=1); rejected agents
               must be EXACTLY unchanged (no peer step, no innate anchor).
               Also cross-checks row['contact'] == gate fraction.
  4. FRESH     row['n_train'] present on every deploy round, == TRAIN_CAP=723,
               and NEVER grows round-over-round (accumulation signature).

Usage:
  python3 check_pofd_sanity.py <run_dir> [<run_dir> ...]
  python3 check_pofd_sanity.py runs/pokec_gated_lm/pofd_*_fresh_data
Exit 0 iff every run passes every check.
"""
import glob
import os
import sys

import torch

ATOL = 2e-6   # float32 blend arithmetic; W=1 makes the copy exact but the
              # stored tensors round-trip through cpu float32


def check_run(run_dir):
    path = os.path.join(run_dir, "trajectory.pt")
    if not os.path.exists(path):
        return [f"MISSING {path}"]
    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = d["config"]
    traj = d["trajectory"]
    op_raw = d["op_raw"].float()        # [rounds, n] opinions AFTER round t
    pred_raw = d["pred_raw"].float()    # [rounds, n] predictions SERVED in round t
    innate = d["innate"].float()
    errs = []

    # -- 1 CONFIG ------------------------------------------------------------
    want = {"eps": 0.0, "w_plat": 1.0, "innate_lambda": 0.0, "canary_delta": 0.0,
            "data_regime": "replace", "fresh_each_round": True, "pop_model": "ab",
            "run_mode": "loop", "ab_sweeps": 1, "pop_reset": False,
            "platform_sus_scale": 1.0, "dataset": "movielens"}
    for k, v in want.items():
        if cfg.get(k) != v:
            errs.append(f"CONFIG {k}={cfg.get(k)!r} (want {v!r})")
    eps_ai = float(cfg["eps_ai"])

    # -- 2 NO-PEER -----------------------------------------------------------
    bad = [r["round"] for r in traj if r.get("accepted", 0) != 0]
    if bad:
        errs.append(f"NO-PEER accepted!=0 in rounds {bad[:5]}{'...' if len(bad) > 5 else ''}")

    # -- 3 EXACT-COPY --------------------------------------------------------
    for t in range(op_raw.shape[0]):
        served = pred_raw[t].clamp(0.0, 1.0)
        if not torch.isfinite(served).all():
            errs.append(f"EXACT-COPY round {t}: non-finite predictions")
            continue
        x_before = innate if t == 0 else op_raw[t - 1]
        gate = (served - x_before).abs() < eps_ai
        d_acc = (op_raw[t][gate] - served[gate]).abs()
        if gate.any() and float(d_acc.max()) > ATOL:
            errs.append(f"EXACT-COPY round {t}: accepted opinion != prediction "
                        f"(max |diff| {float(d_acc.max()):.2e}, W=1 violated)")
        d_rej = (op_raw[t][~gate] - x_before[~gate]).abs()
        if (~gate).any() and float(d_rej.max()) > 0.0:
            errs.append(f"EXACT-COPY round {t}: rejected agent moved "
                        f"(max |diff| {float(d_rej.max()):.2e})")
        logged = traj[t].get("contact")
        if logged is not None and abs(logged - float(gate.float().mean())) > 1e-6:
            errs.append(f"EXACT-COPY round {t}: contact {logged:.6f} != "
                        f"gate frac {float(gate.float().mean()):.6f}")

    # -- 4 FRESH -------------------------------------------------------------
    sizes = [(r["round"], r["n_train"]) for r in traj
             if r.get("is_deploy") and "n_train" in r]
    if not sizes:
        errs.append("FRESH no n_train logged (pipeline predates the n_train patch?)")
    else:
        cap = int(cfg.get("train_cap") or 0) or 723
        wrong = [(t, n) for t, n in sizes if n != cap]
        if wrong:
            errs.append(f"FRESH n_train != {cap} at {wrong[:5]}")
        grew = [(t, n) for (t, n), (_, p) in zip(sizes[1:], sizes[:-1]) if n > p]
        if grew:
            errs.append(f"FRESH n_train GREW (accumulation?) at {grew[:5]}")
    return errs


def main():
    dirs = []
    for a in sys.argv[1:]:
        dirs += sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a]
    if not dirs:
        print(__doc__)
        sys.exit(2)
    n_fail = 0
    for rd in dirs:
        errs = check_run(rd)
        name = os.path.basename(rd.rstrip("/"))
        if errs:
            n_fail += 1
            print(f"FAIL {name}")
            for e in errs:
                print(f"     - {e}")
        else:
            print(f"PASS {name}  (no peer updates, W=1 exact copy, fresh data only)")
    print(f"[check_pofd_sanity] {len(dirs) - n_fail}/{len(dirs)} runs pass")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
