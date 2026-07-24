#!/usr/bin/env python3
"""Sanity gate for the pofd_ platform-only fresh-data runs.

Per run dir (needs trajectory.pt written by run_pokec_gated_lm.py), checks:
  1. CONFIG    the run really is the pofd design: eps(social)=0, w_plat=1,
               innate_lambda=0, pop=ab, mode=loop, canary=0, single sweep,
               no pop reset. Style/regime are TAG-AWARE:
               pofd_* dirs must be data_regime=replace + pristine_frac=0;
               pofdpf_* dirs (data-regime wave) accumulate + kl_beta=0 +
               style=sft + pristine_frac matching the _pf token;
               pofdbp* dirs (beta x pfrac interior) accumulate + style=sft_kl
               + kl_beta matching the _b token + pristine_frac matching _pf;
               pofdicl* dirs style=frozen, use_lora=0, fresh=False, and
               (icl_k, icl_days, icl_ctx_source) matching the arm token
               (k0/k8live/k32live/k32pri/d5/d10/d15/d30);
               pofddpo* dirs style=dpo, fresh=True, use_lora=1, and
               rlhf_feedback matching the closed/open token (DPO_BETA is
               env-only, verified via the submit configs instead). The prefix
               also covers pofddpon* (noisy wave: DO_SAMPLE=1, DPO_TAU=3 --
               both env-only, same config surface).
  2. NO-PEER   row['accepted'] (peer pairs that moved) == 0 in EVERY round.
  3. EXACT-COPY per round t, with x_before = innate (t=0) or op_raw[t-1]:
               gate_i = |served_i - x_before_i| < eps_ai. Accepted agents must
               land EXACTLY on the served prediction (W=1); rejected agents
               must be EXACTLY unchanged (no peer step, no innate anchor).
               Also cross-checks row['contact'] == gate fraction.
  4. FRESH     row['n_train'] present on every deploy round, == TRAIN_CAP=723,
               and NEVER grows round-over-round. n_train is logged POST-cap
               (run_pokec_gated_lm.py:1256), so it must hold 723 under
               accumulate too -- the pool grows, the batch never does.
               Skipped for pofdicl* (frozen: nothing trains); optional for
               pofddpo* (pair-based; checked only if logged).

Usage:
  python3 check_pofd_sanity.py <run_dir> [<run_dir> ...]
  python3 check_pofd_sanity.py runs/pokec_gated_lm/pofd_*_fresh_data
Exit 0 iff every run passes every check.
"""
import glob
import os
import re
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
    name = os.path.basename(run_dir.rstrip("/"))
    is_pfrac = name.startswith("pofdpf_")
    is_bp = name.startswith("pofdbp")      # covers pofdbpsmk_ too
    is_icl = name.startswith("pofdicl")    # covers pofdiclsmk_ too
    is_dpo = name.startswith("pofddpo")    # covers pofddposmk_ too
    want = {"eps": 0.0, "w_plat": 1.0, "innate_lambda": 0.0, "canary_delta": 0.0,
            "data_regime": "accumulate" if (is_pfrac or is_bp) else "replace",
            "pop_model": "ab",
            "run_mode": "loop", "ab_sweeps": 1, "pop_reset": False,
            "platform_sus_scale": 1.0, "dataset": "movielens"}
    if is_pfrac:
        want.update({"kl_beta": 0.0, "training_style": "sft",
                     "fresh_each_round": True})
    elif is_bp:
        # kl_beta varies per row -- checked against the _b tag token below
        want.update({"training_style": "sft_kl", "fresh_each_round": True})
    elif is_icl:
        # frozen weights: nothing trains, FRESH_EACH_ROUND deliberately 0
        want.update({"kl_beta": 0.0, "training_style": "frozen",
                     "pristine_frac": 0.0, "fresh_each_round": False,
                     "use_lora": 0, "icl_select": "random"})
    elif is_dpo:
        # DPO_BETA is env-only (not in config.json) -- verified via the submit
        # configs, not here. rlhf_feedback IS recorded and tag-checked below.
        want.update({"kl_beta": 0.0, "training_style": "dpo",
                     "pristine_frac": 0.0, "fresh_each_round": True,
                     "use_lora": 1})
    else:
        want.update({"pristine_frac": 0.0, "fresh_each_round": True})
    for k, v in want.items():
        if cfg.get(k) != v:
            errs.append(f"CONFIG {k}={cfg.get(k)!r} (want {v!r})")
    if is_pfrac or is_bp:
        m = re.search(r"_pf(\d+(?:p\d+)?)_", name)
        if m is None:
            errs.append(f"CONFIG no _pf token in dirname {name!r}")
        else:
            want_pf = float(m.group(1).replace("p", "."))
            got_pf = float(cfg.get("pristine_frac", -1.0))
            if abs(got_pf - want_pf) > 1e-9:
                errs.append(f"CONFIG pristine_frac={got_pf!r} (tag says {want_pf!r})")
    if is_bp:
        m = re.search(r"_b(\d+(?:p\d+)?)_ea", name)
        if m is None:
            errs.append(f"CONFIG no _b token in dirname {name!r}")
        else:
            want_b = float(m.group(1).replace("p", "."))
            got_b = float(cfg.get("kl_beta", -1.0))
            if abs(got_b - want_b) > 1e-9:
                errs.append(f"CONFIG kl_beta={got_b!r} (tag says {want_b!r})")
    if is_icl:
        ICL_ARM_WANT = {"k0": (0, 0, "live"), "k8live": (8, 0, "live"),
                        "k32live": (32, 0, "live"), "k32pri": (32, 0, "pristine"),
                        "d5": (0, 5, "live"), "d10": (0, 10, "live"),
                        "d15": (0, 15, "live"), "d30": (0, 30, "live")}
        m = re.search(r"_ea[\dp]+_([a-z0-9]+)_s\d", name)
        arm = m.group(1) if m else None
        if arm not in ICL_ARM_WANT:
            errs.append(f"CONFIG unknown icl arm token in dirname {name!r}")
        else:
            k_w, d_w, src_w = ICL_ARM_WANT[arm]
            got = (cfg.get("icl_k"), cfg.get("icl_days"), cfg.get("icl_ctx_source"))
            if got != (k_w, d_w, src_w):
                errs.append(f"CONFIG icl (k,days,src)={got!r} (arm {arm} wants "
                            f"{(k_w, d_w, src_w)!r})")
    if is_dpo:
        fb = "open" if "_open_" in name else ("closed" if "_closed_" in name else None)
        if fb is None:
            errs.append(f"CONFIG no closed/open token in dirname {name!r}")
        elif cfg.get("rlhf_feedback") != fb:
            errs.append(f"CONFIG rlhf_feedback={cfg.get('rlhf_feedback')!r} "
                        f"(tag says {fb!r})")
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
    # icl (frozen): nothing trains, no n_train ever -- skip. dpo: n_train is
    # logged by the shared telemetry when present; check it if there, but its
    # absence is not an error (the DPO learner consumes pairs, not rows).
    if is_icl:
        return errs
    sizes = [(r["round"], r["n_train"]) for r in traj
             if r.get("is_deploy") and "n_train" in r]
    if not sizes:
        if not is_dpo:
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
