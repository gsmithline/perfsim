"""Collapse at performatively stable points, across the continual/fresh x
replace/accumulate 2x2 (and the pristine arm).

For each run we (1) find the stable point -- where the loop settles -- and (2)
read the collapse there, instead of at an arbitrary final round (the collapse
cells are not equilibrated at 60). Two convergence notions:
  continual: weight-space performative stability, w_step=||theta_t-theta_{t-1}|| -> 0.
  fresh:     population fixed point, |op_mean_t - op_mean_{t-1}| -> 0 (no RGD).
We report, per environment (corner x eps x beta): did it converge and when,
and l_c0 / op_std / s_tag AT that stable point.

Run: python experiments/scripts/analyze_gated_2x2.py [runs_dir]
"""

import json
import os
import sys
import glob

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs/pokec_gated_lm"
TOL_OP = 0.003       # |d op_mean| below this = population settled
TOL_W = 0.05         # relative w_step below this = weights settled
WIN = 8              # consecutive settled rounds required


def corner(cfg):
    fresh = cfg.get("fresh_each_round", False)
    reg = cfg.get("data_regime", "replace")
    pf = cfg.get("pristine_frac", 0.0) or 0.0
    tag = ("fresh" if fresh else "continual") + "+" + reg
    if pf > 0:
        tag += f"+pristine{int(pf*100)}"
    return tag


def stable_round(signal, tol):
    """First round from which |signal| stays < tol to the end (settled); else None."""
    n = len(signal)
    for t in range(n):
        if all(abs(signal[k]) < tol for k in range(t, n)):
            return t if (n - t) >= WIN else None
    return None


def load(d):
    cfg = json.load(open(d + "/config.json"))
    tra = json.load(open(d + "/trajectory.json"))
    tel = [json.loads(l) for l in open(d + "/telemetry.json") if l.strip()]
    return cfg, tra, tel


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(RUNS, "g*"))):
        if not os.path.exists(d + "/trajectory.json"):
            continue
        try:
            cfg, tra, tel = load(d)
        except Exception:
            continue
        if cfg.get("run_mode") != "loop":
            continue
        opm = [r["op_mean"] for r in tra]
        d_op = [0.0] + [opm[i] - opm[i - 1] for i in range(1, len(opm))]
        # weight step (continual only; relative to w_norm)
        wstep = {r["round"]: r.get("w_step") for r in tel if r.get("w_step") is not None}
        wnorm = {r["round"]: r.get("w_norm") for r in tel if r.get("w_norm") is not None}
        fresh = cfg.get("fresh_each_round", False)
        if not fresh and wstep:
            rel = [(wstep.get(i, 0.0) / wnorm.get(i, 1.0)) if i in wstep else 0.0
                   for i in range(len(tra))]
            sr = stable_round(rel, TOL_W)
            conv_by = "w_step"
        else:
            sr = stable_round(d_op, TOL_OP)
            conv_by = "op_mean"
        idx = sr if sr is not None else len(tra) - 1
        lc0 = {r["round"]: r.get("l_c0") for r in tel}
        rows.append(dict(
            corner=corner(cfg), eps=cfg["eps"], beta=cfg["kl_beta"],
            converged=(sr is not None), at=idx, by=conv_by,
            lc0=lc0.get(idx, float("nan")), op_std=tra[idx]["op_std"],
            s=tra[idx].get("s_tag", float("nan")),
            wstep_final=list(wstep.values())[-1] if wstep else None))

    order = ["continual+replace", "continual+accumulate", "fresh+replace",
             "fresh+accumulate", "fresh+accumulate+pristine50"]
    print(f"COLLAPSE AT THE STABLE POINT  (runs in {RUNS})\n")
    print(f"{'corner':<30}{'eps':>5}{'beta':>6}{'conv?':>7}{'@rnd':>6}{'by':>9}"
          f"{'l_c0':>8}{'op_std':>8}{'s_tag':>7}")
    for c in order + sorted(set(r["corner"] for r in rows) - set(order)):
        cr = sorted([r for r in rows if r["corner"] == c], key=lambda r: (r["eps"], r["beta"]))
        for r in cr:
            cv = "yes" if r["converged"] else "NO"
            print(f"{r['corner']:<30}{r['eps']:>5}{r['beta']:>6}{cv:>7}{r['at']:>6}{r['by']:>9}"
                  f"{r['lc0']:>8.3f}{r['op_std']:>8.3f}{r['s']:>7.3f}")
        if cr:
            print()


if __name__ == "__main__":
    main()
