#!/usr/bin/env python3
"""SECTION-4 THREE-SEED analysis: bottom-20% source effect and
responsive-population dispersion (2026-08-19,
mistral_bottom20_section4_repl + the completed seed-0 waves).

Cells: seeds {0, 42, 43} x conditions {fixed, evolving} x arms
{b0 = ordinary fresh SFT, d8 = frozen personal-history ICL} x ea
{0.1, 0.2, 0.4, 1} x es {0, 0.05, 0.1, 0.2, 0.4, 1} = 288 conceptual
trajectories, ALL HARD-REQUIRED. Seed-0 cells resolve through the
established tag schemes (pofdclamp bottom / pofdevo); seed-42/43
cells resolve through the audited replication manifest (reused cells
keep their archived tags -- pofdreach / pofdpeer2 / pofdgate2d /
pofdws2f / tokenless pofdclamp originals). Cohort A = the 145
lowest-innate agents (deterministic innate-then-id ranking): the
stored clamp mask on every fixed run must equal the recomputed
bottom-145; evolving runs must carry no mask, and every run must
share the bit-identical innate population.

Equilibrium estimator: mean over rounds 25-29 within each run. Every
statistic is computed separately per seed and then aggregated as the
three-seed mean with the 95% Student-t interval (df=2).

  (a) source effect
      T_a = mu_B^eq(A evolves) - mu_B^eq(A fixed), per arm
  (b) responsive dispersion in the FIXED condition
      SD(B platform) / SD(B matched no-platform twin)

The analyzer VERIFIES the structural null: at es=0 the personal-
history d8 source effect must be zero for every seed and AI gate
(frozen weights + own-history prompts + no peer step means no A
opinion can ever enter a B prompt, so the clamp cannot reach B).
The tolerance is HARDWARE-AWARE, because greedy LM generation is
only bit-reproducible within one GPU architecture (2026-08-19: the
d8/es0 cells diverge in cohort B from round 0 iff the fixed and
evolving runs landed on different architectures -- H100 vs A100 --
and are bit-identical whenever they match, over all 12 seed x gate
pairs):
  same architecture      -> EXACT zero required (tol 1e-9); any
                            violation is a hard failure, since a
                            genuine A->B pathway is the only
                            remaining explanation
  different / unknown    -> the measured |T_a| IS the empirical
                            generation-nondeterminism floor for the
                            whole grid; it is reported per cell and
                            hard-fails only above NULL_TOL_XHW,
                            where it would be large enough to
                            contaminate the effects being estimated
Every cell records its GPU architecture, and the floor is written
out so downstream reporting can quote it against the effect sizes.

Outputs (notes/pofd/bottom20_section4_3seed_analysis/ -- a NEW
directory; the seed-0 analyses are never overwritten):
  section4_per_seed_cells.csv   one row per (seed, condition, arm,
                                gate, es), incl. GPU architecture
  section4_null_floor.csv       the d8/es0 structural-null probe per
                                (seed, gate): T_a, the fixed/evolving
                                architecture pair, and the verdict
  section4_source_effect.csv    per (arm, gate, es): per-seed T_a,
                                mean, sd, 95% CI, excludes-zero
  section4_dispersion.csv       per (arm, gate, es): per-seed fixed
                                SD ratio, mean, sd, 95% CI,
                                excludes-one
  section4_contrast.csv         per (gate, es): paired SFT - ICL
                                source-effect contrast per seed,
                                mean, 95% CI
  section4_source_effect.png/pdf  diagonal-split heatmap (UL = SFT
      b0, LR = personal-history d8) of the three-seed mean T_a; '*'
      marks cells whose 95% CI excludes 0
  section4_sd_ratio.png/pdf       same split for the fixed-A SD
      ratio; '*' marks cells whose 95% CI excludes 1
  x = AI gate, y = peer gate; cell text is the three-seed mean at
  two decimals -- full intervals live in the CSVs only.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "analyze_reach", os.path.join(HERE, "analyze_sft_icl_reach.py"))
AN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN)

CONDS = ["fixed", "evolving"]
ARMS = ["b0", "d8"]
GATES = [0.1, 0.2, 0.4, 1.0]
ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
SEEDS = [0, 42, 43]
LATE = range(25, 30)
# 95% two-sided Student-t critical value at df = 2 (three seeds)
T_CRIT = 4.302652729911275
# structural null: bit-exact within one GPU architecture; across
# architectures the residual is greedy-generation nondeterminism,
# empirically <= 2e-3 on the B equilibrium mean vs effects of order
# 1e-1, so 5e-3 is the "large enough to contaminate" line
NULL_TOL = 1e-9
NULL_TOL_XHW = 5e-3
MANIFEST_DEFAULT = os.path.join(
    REPO, "experiments", "condor",
    "manifest_bottom20_section4_repl.json")
OUT_DIR_DEFAULT = os.path.join(
    REPO, "notes", "pofd", "bottom20_section4_3seed_analysis")


def _num(v):
    return f"{v:g}".replace(".", "p")


def seed0_tag(cond, arm, gate, es):
    """The completed seed-0 waves: tokenless pofdclamp bottom b0 at
    es=0, _stub_ elsewhere; pofdevo for evolving."""
    if cond == "fixed":
        stub = "" if (arm == "b0" and es == 0.0) else "_stub"
        return (f"pofdclamp_mistral7b_{arm}_bottom{stub}"
                f"_ea{_num(gate)}_w0p5_l0p2_es{_num(es)}_s0")
    return (f"pofdevo_mistral7b_{arm}_ea{_num(gate)}"
            f"_w0p5_l0p2_es{_num(es)}_s0")


def repl_tags(manifest):
    """{(cond, arm, gate, es, seed): tag} for seeds 42/43 from the
    audited manifest (reused cells keep their archived tags)."""
    out = {}
    for c in manifest["cells"]:
        tag = c["run_tag"] if c["status"] == "reused" else c["new_tag"]
        out[(c["cond"], c["arm"], c["gate"], c["es"], c["seed"])] = tag
    return out


def gpu_arch(run_dir):
    """Coarse GPU architecture of a run ('H100' / 'A100' / the raw
    name / 'unknown'). Greedy generation is bit-reproducible only
    within one architecture, so this is what the structural-null
    tolerance keys off."""
    try:
        with open(os.path.join(run_dir, "config.json")) as fh:
            hw = json.load(fh).get("hardware") or {}
    except (OSError, json.JSONDecodeError):
        return "unknown"
    name = hw.get("gpu_name") or ""
    for arch in ("H100", "A100", "A6000", "V100"):
        if arch in name:
            return arch
    return name or "unknown"


def tci3(vals):
    """(mean, sd, ci_lo, ci_hi): three-seed mean with the 95%
    Student-t interval (df = 2)."""
    n = len(vals)
    m = sum(vals) / n
    sd = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5
    half = T_CRIT * sd / n ** 0.5
    return m, sd, m - half, m + half


def excludes(ci_lo, ci_hi, ref):
    return ci_lo > ref or ci_hi < ref


def cell_stats(d, mask):
    """Late-window equilibrium stats over cohort B (and A's mean for
    reference); dispersion ratio uses the matched twin per round."""
    op = d["op_raw"].float()
    tw = d["twin_raw"].float()
    b, a = ~mask, mask
    mu_b = float(torch.stack([op[t][b].mean() for t in LATE]).mean())
    mu_a = float(torch.stack([op[t][a].mean() for t in LATE]).mean())
    sd_b = float(torch.stack([op[t][b].std() for t in LATE]).mean())
    ratios = []
    for t in LATE:
        s_tw = float(tw[t][b].std())
        if s_tw > 0:
            ratios.append(float(op[t][b].std()) / s_tw)
    return {"mu_b_eq": mu_b, "mu_a_eq": mu_a, "sd_b_late": sd_b,
            "sd_ratio_late": (sum(ratios) / len(ratios)
                              if ratios else float("nan"))}


def split_heatmap(ax, ul, lr, ul_mark, lr_mark, cmap, norm,
                  ul_lab, lr_lab):
    """Diagonal-split grid (bottom-left to top-right cut): the
    upper-left triangle renders ul[j][i], the lower-right lr[j][i];
    a trailing '*' marks cells whose 95% CI excludes the reference
    value. Cell text carries ONLY the three-seed mean at two
    decimals -- intervals live in the CSVs."""
    from matplotlib.patches import Polygon
    for j in range(len(ESS)):
        for i in range(len(GATES)):
            c0 = (i - 0.5, j - 0.5)
            c1 = (i + 0.5, j + 0.5)
            ax.add_patch(Polygon(
                [c0, (i - 0.5, j + 0.5), c1], closed=True,
                facecolor=cmap(norm(ul[j][i])), edgecolor="white",
                lw=0.5))
            ax.add_patch(Polygon(
                [c0, (i + 0.5, j - 0.5), c1], closed=True,
                facecolor=cmap(norm(lr[j][i])), edgecolor="white",
                lw=0.5))
            ax.text(i - 0.19, j + 0.21,
                    f"{ul[j][i]:.2f}" + ("*" if ul_mark[j][i] else ""),
                    ha="center", va="center", fontsize=5.4)
            ax.text(i + 0.19, j - 0.21,
                    f"{lr[j][i]:.2f}" + ("*" if lr_mark[j][i] else ""),
                    ha="center", va="center", fontsize=5.4)
    ax.text(0.02, 0.98, f"upper-left: {ul_lab}\nlower-right: {lr_lab}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7)


def style_axis(ax):
    ax.set_xlim(-0.5, len(GATES) - 0.5)
    ax.set_ylim(-0.5, len(ESS) - 0.5)
    ax.set_xticks(range(len(GATES)))
    ax.set_xticklabels([f"{g:g}" for g in GATES])
    ax.set_yticks(range(len(ESS)))
    ax.set_yticklabels([f"{e:g}" for e in ESS])
    ax.set_xlabel(r"$\varepsilon_{\mathrm{AI}}$")
    ax.set_ylabel(r"$\varepsilon_{\mathrm{social}}$")
    ax.set_aspect("equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    assert manifest["key"] == "mistral_bottom20_section4_repl"
    rtags = repl_tags(manifest)

    def tag_of(cond, arm, gate, es, seed):
        if seed == 0:
            return seed0_tag(cond, arm, gate, es)
        return rtags[(cond, arm, gate, es, seed)]

    run_of, missing = {}, []
    for seed in SEEDS:
        for cond in CONDS:
            for arm in ARMS:
                for gate in GATES:
                    for es in ESS:
                        tag = tag_of(cond, arm, gate, es, seed)
                        rd = AN.find_run(args.roots, tag)
                        if rd is None:
                            missing.append(tag)
                        else:
                            run_of[(cond, arm, gate, es, seed)] = rd
    n_total = (len(CONDS) * len(ARMS) * len(GATES) * len(ESS)
               * len(SEEDS))
    print(f"[sec4_3seed] trajectories located: "
          f"{len(run_of)}/{n_total}")
    for tag in missing:
        print(f"  MISSING {tag}")
    if missing:
        print(f"[sec4_3seed] HARD FAIL: {len(missing)} of {n_total} "
              f"conceptual trajectories missing -- no output written",
              file=sys.stderr)
        sys.exit(1)

    loads = {k: AN.load(rd) for k, rd in run_of.items()}
    ref = loads[("evolving", "b0", GATES[0], 0.0, 0)]
    innate = ref["innate"].float()
    order = sorted(range(innate.numel()),
                   key=lambda i: (float(innate[i]), i))
    mask = torch.zeros(innate.numel(), dtype=torch.bool)
    mask[torch.tensor(order[:145])] = True
    for k, d in loads.items():
        if not torch.equal(d["innate"], ref["innate"]):
            print(f"[sec4_3seed] HARD FAIL: {tag_of(*k)} innate "
                  f"differs from the shared population",
                  file=sys.stderr)
            sys.exit(1)
        cm = d.get("innate_clamp_mask")
        if k[0] == "fixed":
            if cm is None or not torch.equal(cm.bool(), mask):
                print(f"[sec4_3seed] HARD FAIL: {tag_of(*k)} stored "
                      f"mask != the recomputed bottom-145",
                      file=sys.stderr)
                sys.exit(1)
        elif cm is not None and cm.numel():
            print(f"[sec4_3seed] HARD FAIL: {tag_of(*k)} carries a "
                  f"clamp mask -- not fully evolving",
                  file=sys.stderr)
            sys.exit(1)

    per_cell = []
    for (cond, arm, gate, es, seed), d in sorted(
            loads.items(), key=lambda kv: (kv[0][4], kv[0][0],
                                           kv[0][1], kv[0][2],
                                           kv[0][3])):
        per_cell.append({"seed": seed, "condition": cond, "arm": arm,
                         "gate": gate, "eps_social": es,
                         "run_tag": tag_of(cond, arm, gate, es, seed),
                         "gpu_arch": gpu_arch(
                             run_of[(cond, arm, gate, es, seed)]),
                         **cell_stats(d, mask)})

    def st(cond, arm, gate, es, seed):
        return [r for r in per_cell if r["condition"] == cond
                and r["arm"] == arm and r["gate"] == gate
                and r["eps_social"] == es and r["seed"] == seed][0]

    # per-seed source effect and fixed dispersion ratio, then the
    # three-seed aggregation
    source_rows, disp_rows = [], []
    for arm in ARMS:
        for gate in GATES:
            for es in ESS:
                t_a = [st("evolving", arm, gate, es, s)["mu_b_eq"]
                       - st("fixed", arm, gate, es, s)["mu_b_eq"]
                       for s in SEEDS]
                m, sd, lo, hi = tci3(t_a)
                # hardware provenance per seed: greedy generation is
                # bit-reproducible only within one GPU architecture,
                # so a cross-architecture pair carries the
                # nondeterminism floor quantified by the d8/es0 probe
                hw_pair = {
                    f"gpu_pair_s{s}":
                        (st("fixed", arm, gate, es, s)["gpu_arch"]
                         + "/"
                         + st("evolving", arm, gate, es, s)["gpu_arch"])
                    for s in SEEDS}
                source_rows.append({
                    "arm": arm, "gate": gate, "eps_social": es,
                    **{f"t_a_s{s}": v for s, v in zip(SEEDS, t_a)},
                    "t_a_mean": m, "t_a_sd": sd,
                    "ci_lo": lo, "ci_hi": hi,
                    "ci_excludes_zero": excludes(lo, hi, 0.0),
                    "n_seeds_hardware_matched": sum(
                        1 for s in SEEDS
                        if st("fixed", arm, gate, es, s)["gpu_arch"]
                        == st("evolving", arm, gate, es,
                              s)["gpu_arch"]
                        != "unknown"),
                    **hw_pair})
                ratios = [st("fixed", arm, gate, es,
                             s)["sd_ratio_late"] for s in SEEDS]
                m, sd, lo, hi = tci3(ratios)
                disp_rows.append({
                    "arm": arm, "gate": gate, "eps_social": es,
                    **{f"sd_ratio_s{s}": v
                       for s, v in zip(SEEDS, ratios)},
                    "sd_ratio_mean": m, "sd_ratio_sd": sd,
                    "ci_lo": lo, "ci_hi": hi,
                    "ci_excludes_one": excludes(lo, hi, 1.0)})

    def srow(arm, gate, es):
        return [r for r in source_rows if r["arm"] == arm
                and r["gate"] == gate and r["eps_social"] == es][0]

    def drow(arm, gate, es):
        return [r for r in disp_rows if r["arm"] == arm
                and r["gate"] == gate and r["eps_social"] == es][0]

    # paired SFT - ICL source-effect contrast, within seed
    contrast_rows = []
    for gate in GATES:
        for es in ESS:
            diff = [srow("b0", gate, es)[f"t_a_s{s}"]
                    - srow("d8", gate, es)[f"t_a_s{s}"]
                    for s in SEEDS]
            m, sd, lo, hi = tci3(diff)
            contrast_rows.append({
                "gate": gate, "eps_social": es,
                **{f"t_a_sft_minus_icl_s{s}": v
                   for s, v in zip(SEEDS, diff)},
                "mean": m, "sd": sd, "ci_lo": lo, "ci_hi": hi,
                "ci_excludes_zero": excludes(lo, hi, 0.0)})

    # STRUCTURAL NULL: the no-peer personal-history source effect is
    # zero by construction (no A opinion can enter a B prompt).
    # Tolerance is hardware-aware -- see the module docstring.
    null_rows, null_bad = [], []
    print("[sec4_3seed] structural null (d8, es=0): T_a per "
          "seed/gate")
    for gate in GATES:
        r = srow("d8", gate, 0.0)
        for s in SEEDS:
            v = r[f"t_a_s{s}"]
            hw_f = st("fixed", "d8", gate, 0.0, s)["gpu_arch"]
            hw_e = st("evolving", "d8", gate, 0.0, s)["gpu_arch"]
            matched = (hw_f == hw_e and hw_f != "unknown")
            tol = NULL_TOL if matched else NULL_TOL_XHW
            ok = abs(v) <= tol
            null_rows.append({
                "seed": s, "gate": gate, "t_a": v,
                "gpu_fixed": hw_f, "gpu_evolving": hw_e,
                "hardware_matched": matched, "tol": tol,
                "verdict": "PASS" if ok else "FAIL"})
            print(f"  ea{gate:<4g} s{s:<3}: T_a={v:+.3e}  "
                  f"{hw_f}/{hw_e}"
                  f"{'  [matched -> exact]' if matched else ''}"
                  f"  {'PASS' if ok else 'FAIL'}")
            if not ok:
                null_bad.append((gate, s, v, hw_f, hw_e, tol))
    floor = max((abs(r["t_a"]) for r in null_rows
                 if not r["hardware_matched"]), default=0.0)
    n_match = sum(1 for r in null_rows if r["hardware_matched"])
    print(f"[sec4_3seed] structural null: {n_match}/{len(null_rows)} "
          f"probes hardware-matched (all bit-exact); "
          f"generation-nondeterminism floor from the "
          f"{len(null_rows) - n_match} cross-architecture probes: "
          f"|T_a| <= {floor:.2e}")
    if null_bad:
        print(f"[sec4_3seed] HARD FAIL: the no-peer personal-history "
              f"source effect exceeds its tolerance: {null_bad} -- "
              f"on a hardware-MATCHED probe this means the clamp "
              f"reached cohort B through a path that must not "
              f"exist; across architectures it means generation "
              f"nondeterminism is large enough to contaminate the "
              f"effects; no output written", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    def write(name, rows):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out_dir, name), "w",
                  newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys)
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"[sec4_3seed] wrote {name} ({len(rows)} rows)")

    write("section4_per_seed_cells.csv", per_cell)
    write("section4_source_effect.csv", source_rows)
    write("section4_dispersion.csv", disp_rows)
    write("section4_contrast.csv", contrast_rows)
    write("section4_null_floor.csv", null_rows)

    def grid(rows_fn, arm, key):
        return [[rows_fn(arm, g, e)[key] for g in GATES]
                for e in ESS]

    for arm in ARMS:
        print(f"\n== three-seed mean source effect T_a, {arm} "
              f"(cols = ea " + "/".join(f"{g:g}" for g in GATES)
              + "; * = 95% CI excludes 0) ==")
        gr = grid(srow, arm, "t_a_mean")
        mk = grid(srow, arm, "ci_excludes_zero")
        for j, es in enumerate(ESS):
            print(f"  es={es:<4g}: " + "  ".join(
                f"{v:+.4f}{'*' if m else ' '}"
                for v, m in zip(gr[j], mk[j])))
    for arm in ARMS:
        print(f"\n== three-seed mean fixed-A SD ratio, {arm} "
              f"(* = 95% CI excludes 1) ==")
        gr = grid(drow, arm, "sd_ratio_mean")
        mk = grid(drow, arm, "ci_excludes_one")
        for j, es in enumerate(ESS):
            print(f"  es={es:<4g}: " + "  ".join(
                f"{v:.3f}{'*' if m else ' '}"
                for v, m in zip(gr[j], mk[j])))

    if not args.no_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize, TwoSlopeNorm
        from matplotlib.cm import ScalarMappable

        # (a) source effect, split SFT / personal-history ICL
        t_b0 = grid(srow, "b0", "t_a_mean")
        t_d8 = grid(srow, "d8", "t_a_mean")
        m_b0 = grid(srow, "b0", "ci_excludes_zero")
        m_d8 = grid(srow, "d8", "ci_excludes_zero")
        lim = max(1e-6, max(abs(v) for gr_ in (t_b0, t_d8)
                            for row in gr_ for v in row))
        n_t = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        fig, ax = plt.subplots(figsize=(5.4, 6.2))
        split_heatmap(ax, t_b0, t_d8, m_b0, m_d8, plt.cm.RdBu_r,
                      n_t, "SFT (b0)", "personal-history d8")
        fig.colorbar(ScalarMappable(norm=n_t, cmap=plt.cm.RdBu_r),
                     ax=ax, fraction=0.046, pad=0.04,
                     label=r"$T_a=\mu_{B}^{\mathrm{eq}}"
                           r"(A\ \mathrm{evolves})-"
                           r"\mu_{B}^{\mathrm{eq}}"
                           r"(A\ \mathrm{fixed})$")
        style_axis(ax)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(
                args.out_dir, f"section4_source_effect.{ext}"),
                dpi=220 if ext == "png" else None)
        plt.close(fig)
        print("[sec4_3seed] wrote section4_source_effect.png/pdf")

        # (b) fixed-A dispersion ratio, same split
        s_b0 = grid(drow, "b0", "sd_ratio_mean")
        s_d8 = grid(drow, "d8", "sd_ratio_mean")
        k_b0 = grid(drow, "b0", "ci_excludes_one")
        k_d8 = grid(drow, "d8", "ci_excludes_one")
        vals = [v for gr_ in (s_b0, s_d8) for row in gr_ for v in row]
        n_s = Normalize(vmin=min(vals), vmax=max(vals))
        fig, ax = plt.subplots(figsize=(5.4, 6.2))
        split_heatmap(ax, s_b0, s_d8, k_b0, k_d8, plt.cm.viridis,
                      n_s, "SFT (b0)", "personal-history d8")
        fig.colorbar(ScalarMappable(norm=n_s, cmap=plt.cm.viridis),
                     ax=ax, fraction=0.046, pad=0.04,
                     label=r"SD($B$ platform) / SD($B$ twin), "
                           r"$A$ fixed")
        style_axis(ax)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(
                args.out_dir, f"section4_sd_ratio.{ext}"),
                dpi=220 if ext == "png" else None)
        plt.close(fig)
        print("[sec4_3seed] wrote section4_sd_ratio.png/pdf")


if __name__ == "__main__":
    main()
