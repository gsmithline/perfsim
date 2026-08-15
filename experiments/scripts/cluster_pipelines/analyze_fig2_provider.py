#!/usr/bin/env python3
"""Figure-2 provider-separation analysis (2026-08-15).

THE TEST: does retaining the entering model signal (SFT-KL at beta=1,
anchored to the base weights) preserve more provider-specific population
separation than ordinary SFT (beta=0)?

Read-only, descriptive. Consumes manifest_fig2_provider.json: providers
{qwen7b, olmo7b, mistral7b} x arms {b0, b1} x eps_AI 0.4 x eps_social
{0, 0.2} x seeds {0, 42, 43}.

Reported:
  * PROVIDER-PAIR W1 for qwen-olmo, qwen-mistral and olmo-mistral, on
    the LATE POPULATION STATE of each arm/dose/seed.
  * Three-seed means with 95% Student-t intervals (t_crit(2)=4.30265).
    SEEDS are the replicates -- never agents.
  * The WITHIN-SEED contrast b1 - b0: separation is differenced inside
    each seed before averaging, so the interval is PAIRED and the
    seed-to-seed level of separation cancels.
  * ROUNDS 25-29 as a stability check: the same quantities averaged over
    the window, alongside the final round.
  * SEED-AVERAGED DENSITIES: each seed's opinion density is estimated
    SEPARATELY on a shared grid and the densities are then averaged.
    Agents are never pooled across seeds -- pooling would understate
    between-seed spread and silently inflate apparent sharpness.

TERMINOLOGY: these are LATE POPULATION STATES, not converged states.
Three seeds address replication; they do not establish convergence, and
nothing here should be described as a fixed point.

Outputs notes/pofd/fig2_provider/: fig2_pairs_per_seed.csv,
fig2_pairs_summary.csv, fig2_contrast_b1_minus_b0.csv,
fig2_density_seed_averaged.csv.
"""
import argparse
import csv
import importlib.util
import itertools
import json
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "analyze_reach", os.path.join(HERE, "analyze_sft_icl_reach.py"))
AN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN)

NA = "NA"
T_CRIT = {1: 12.7062, 2: 4.30265, 3: 3.18245, 4: 2.77645}
LATE_LO, LATE_HI = 25, 30
DENS_BINS = 50


def w1(a, b):
    """Wasserstein-1 between two equal-size empirical populations."""
    return float((torch.sort(a).values - torch.sort(b).values)
                 .abs().mean())


def mean_ci(vals):
    """(mean, sd, ci95) over SEED replicates; NA when n < 2."""
    vals = [v for v in vals if v not in (NA, None)]
    if not vals:
        return NA, NA, NA
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, NA, NA
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var), T_CRIT[len(vals) - 1] * math.sqrt(
        var / len(vals))


def load_states(manifest, roots):
    """{(model, arm, es, seed): {'final': [n], 'late': [rounds, n]}}"""
    out, missing = {}, []
    for c in manifest["cells"]:
        rd = AN.find_run(roots, c["run_tag"])
        key = (c["model"], c["arm"], c["eps_social"], c["seed"])
        if rd is None:
            missing.append((key, c["run_tag"], c["status"]))
            continue
        op = AN.load(rd)["op_raw"].float()
        if op.shape[0] < LATE_HI:
            missing.append((key, c["run_tag"], "short"))
            continue
        out[key] = {"final": op[-1], "late": op[LATE_LO:LATE_HI]}
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        REPO, "experiments", "condor", "manifest_fig2_provider.json"))
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "notes", "pofd", "fig2_provider"))
    args = ap.parse_args()
    man = json.load(open(args.manifest))
    os.makedirs(args.out_dir, exist_ok=True)

    models = list(man["grid"]["models"])
    arms = man["grid"]["arms"]
    ess = man["grid"]["eps_socials"]
    seeds = man["grid"]["seeds"]
    pairs = list(itertools.combinations(models, 2))

    states, missing = load_states(man, args.roots)
    print(f"[fig2] cells located: {len(states)}/{len(man['cells'])}")
    for key, tag, why in missing:
        print(f"  MISSING {key} <- {tag} ({why})")

    # -- provider-pair separation, per seed -----------------------------
    per_seed = []
    for es in ess:
        for arm in arms:
            for seed in seeds:
                for a, b in pairs:
                    ka = (a, arm, es, seed)
                    kb = (b, arm, es, seed)
                    if ka not in states or kb not in states:
                        continue
                    fa, fb = states[ka]["final"], states[kb]["final"]
                    la, lb = states[ka]["late"], states[kb]["late"]
                    late_w1 = sum(
                        w1(la[t], lb[t]) for t in range(la.shape[0])
                    ) / la.shape[0]
                    per_seed.append({
                        "eps_social": es, "arm": arm, "seed": seed,
                        "pair": f"{a}-{b}",
                        "w1_final": w1(fa, fb),
                        "w1_late_25_29": late_w1,
                        "mean_gap_final": abs(float(fa.mean())
                                              - float(fb.mean())),
                        "std_ratio_final": (float(fa.std())
                                            / float(fb.std())
                                            if float(fb.std()) > 0
                                            else NA)})

    # -- three-seed summary ---------------------------------------------
    summary = []
    for es in ess:
        for arm in arms:
            for a, b in pairs:
                rows = [r for r in per_seed if r["eps_social"] == es
                        and r["arm"] == arm
                        and r["pair"] == f"{a}-{b}"]
                out = {"eps_social": es, "arm": arm,
                       "pair": f"{a}-{b}", "n_seeds": len(rows)}
                for m in ("w1_final", "w1_late_25_29",
                          "mean_gap_final"):
                    mu, sd, ci = mean_ci([r[m] for r in rows])
                    out[f"{m}_mean"] = mu
                    out[f"{m}_sd"] = sd
                    out[f"{m}_ci95"] = ci
                summary.append(out)

    # -- within-seed contrast b1 - b0 (PAIRED) --------------------------
    # differencing inside a seed cancels that seed's overall separation
    # level, so the interval speaks to the ARM effect, not seed spread
    contrast = []
    for es in ess:
        for a, b in pairs:
            diffs_f, diffs_l, used = [], [], []
            for seed in seeds:
                def get(arm, key):
                    r = [x for x in per_seed if x["eps_social"] == es
                         and x["arm"] == arm and x["seed"] == seed
                         and x["pair"] == f"{a}-{b}"]
                    return r[0][key] if r else None
                f1, f0 = get("b1", "w1_final"), get("b0", "w1_final")
                l1, l0 = (get("b1", "w1_late_25_29"),
                          get("b0", "w1_late_25_29"))
                if None in (f1, f0, l1, l0):
                    continue
                diffs_f.append(f1 - f0)
                diffs_l.append(l1 - l0)
                used.append(seed)
            mu_f, sd_f, ci_f = mean_ci(diffs_f)
            mu_l, sd_l, ci_l = mean_ci(diffs_l)
            contrast.append({
                "eps_social": es, "pair": f"{a}-{b}",
                "n_paired_seeds": len(used), "seeds_used": str(used),
                "d_w1_final_mean": mu_f, "d_w1_final_sd": sd_f,
                "d_w1_final_ci95": ci_f,
                "d_w1_late_mean": mu_l, "d_w1_late_sd": sd_l,
                "d_w1_late_ci95": ci_l,
                "b1_preserves_more_separation": (
                    NA if mu_f == NA else bool(mu_f > 0))})

    # -- seed-averaged densities ----------------------------------------
    # each seed estimated separately on a shared grid, then averaged;
    # agents are NEVER pooled across seeds
    edges = torch.linspace(0.0, 1.0, DENS_BINS + 1)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    density = []
    for es in ess:
        for arm in arms:
            for model in models:
                per = []
                for seed in seeds:
                    st = states.get((model, arm, es, seed))
                    if st is None:
                        continue
                    h = torch.histogram(st["final"], bins=edges)[0]
                    per.append(h / max(1.0, float(h.sum())))
                if not per:
                    continue
                stack = torch.stack(per)
                mean_d = stack.mean(dim=0)
                sd_d = (stack.std(dim=0) if stack.shape[0] > 1
                        else torch.zeros_like(mean_d))
                for i, c in enumerate(centers):
                    density.append({
                        "eps_social": es, "arm": arm, "model": model,
                        "n_seeds": stack.shape[0],
                        "bin_center": round(c, 4),
                        "density_mean": float(mean_d[i]),
                        "density_sd_across_seeds": (
                            float(sd_d[i]) if stack.shape[0] > 1 else NA)})

    def write(name, rows):
        if not rows:
            print(f"[fig2] {name}: no rows")
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out_dir, name), "w",
                  newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys, restval=NA)
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"[fig2] wrote {name} ({len(rows)} rows)")

    write("fig2_pairs_per_seed.csv", per_seed)
    write("fig2_pairs_summary.csv", summary)
    write("fig2_contrast_b1_minus_b0.csv", contrast)
    write("fig2_density_seed_averaged.csv", density)

    print("\n== provider separation, three-seed mean +- 95% t "
          "(LATE POPULATION STATES, not converged) ==")
    for es in ess:
        for a, b in pairs:
            line = f"  es={es!s:4s} {a[:-2]}-{b[:-2]:9s}"
            for arm in arms:
                r = [x for x in summary if x["eps_social"] == es
                     and x["arm"] == arm and x["pair"] == f"{a}-{b}"]
                if r and r[0]["w1_final_mean"] != NA:
                    ci = r[0]["w1_final_ci95"]
                    ci_s = f"+-{ci:.3f}" if ci != NA else "(n<2)"
                    line += (f"  {arm} {r[0]['w1_final_mean']:.3f}"
                             f"{ci_s} [n={r[0]['n_seeds']}]")
                else:
                    line += f"  {arm} NA"
            print(line)
    print("\n== within-seed contrast W1(b1) - W1(b0) ==")
    for c in contrast:
        if c["d_w1_final_mean"] == NA:
            print(f"  es={c['eps_social']!s:4s} {c['pair']}: NA")
            continue
        ci = c["d_w1_final_ci95"]
        ci_s = f"+-{ci:.3f}" if ci != NA else "(n<2)"
        print(f"  es={c['eps_social']!s:4s} {c['pair']:22s} "
              f"final {c['d_w1_final_mean']:+.3f}{ci_s}   "
              f"late {c['d_w1_late_mean']:+.3f}  "
              f"[n={c['n_paired_seeds']} paired seeds]")


if __name__ == "__main__":
    main()
