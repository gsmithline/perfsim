#!/usr/bin/env python3
"""Routing-gap plot for any beta=W_PLAT family wave (--variant).

Two panels, sharing the eps_social axis:
  (a) raw MAE per method -- the Figure-5 metric, final-five-round
      agent-paired |evolving - fixed| on cohort B, SFT vs ICL
  (b) the ROUTING GAP  G = MAE_ICL - MAE_SFT, at round 15 and round 30,
      with G = 0 marked. Negative G = the shared-weight (SFT) route
      carries more of cohort A's influence into cohort B; positive G =
      the personal-history (ICL) route does.

Figures carry NO title (house rule); the narrative lives in the caption
block this script writes beside the PDF.

Run AFTER check_section4_gate.py --wave pilot_g1 passes. Seed 0 only --
descriptive, no confidence intervals are drawn or implied.

  python plot_s4g_pilot_g1.py --run-root <runs> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys

import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GEN_PATH = os.path.join(REPO, "experiments", "condor", "gen_pofd_sweep.py")
AN_PATH = os.path.join(HERE, "analyze_section4_gate.py")
LOG = "[s4g_gap_plot]"
N_AGENTS = 723
DEFAULT_VARIANT = "pilot_g1"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mae_window(fixed, evolving, mask_b, lo, hi):
    """The Figure-5 metric on rounds (lo, hi]: agent-paired
    |evolving - fixed| on cohort B, averaged over rounds and agents.
    Mirrors plot_section4_source_effect_main.paired_source_mae, which
    slices [-5:] of a 30-round run -- here the window is explicit so the
    same quantity can be read at round 15 and at round 30."""
    f = fixed["op_raw"].float()[lo:hi][:, mask_b]
    e = evolving["op_raw"].float()[lo:hi][:, mask_b]
    if not (torch.isfinite(f).all() and torch.isfinite(e).all()):
        raise ValueError("non-finite op_raw in the late window")
    return float((e - f).abs().mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", default=DEFAULT_VARIANT,
                    help="family variant to plot (pilot_g1, "
                         "scout_qwen3, scout, probe). For a CROSS-shaped "
                         "wave only the eps_social sweep is drawn, in "
                         "ascending es order.")
    ap.add_argument("--stem", default=None,
                    help="output filename stem (default s4g_<variant>)")
    ap.add_argument("--run-root",
                    default="/home/gsmithline/perfsim/runs/pokec_gated_lm")
    ap.add_argument("--out-dir", default=None,
                    help="default notes/pofd/s4g_<variant>/")
    ap.add_argument("--gen", default=GEN_PATH)
    ap.add_argument("--rounds", default="15,30",
                    help="comma-separated round marks for the gap panel")
    args = ap.parse_args(argv)
    variant = args.variant
    stem = args.stem or f"s4g_{variant}"
    if args.out_dir is None:
        args.out_dir = os.path.join(REPO, "notes", "pofd", stem)

    gen = _load(args.gen, "_gen_pilot")
    AN = _load(AN_PATH, "_an_pilot")
    fam = gen.S4G_VARIANTS[variant]
    # the eps_social axis, ASCENDING: for a cross-shaped wave this is
    # the "es" sweep only (its eps_AI sweep is a different figure)
    es_pts = (fam["points"]("es") if "ea_key" in fam
              else fam["points"]("all"))
    ess = sorted({float(e) for _a, e, _n in es_pts})
    gate = es_pts[0][0]
    n_rounds = int(es_pts[0][2])
    marks = [int(x) for x in args.rounds.split(",") if x.strip()]
    seed = int(gen.S4GP_SEEDS[0])

    # ---- load every pair once ------------------------------------------
    runs, missing = {}, []
    for arm in gen.S4GP_ARMS:
        for cond in gen.S4G_CONDS:
            for es in ess:
                tag = gen.s4gv_tag(arm, cond, gate, es, seed,
                                   fam["prefix"])
                rd = AN.find_run(args.run_root, tag)
                (missing.append(tag) if rd is None
                 else runs.__setitem__((arm, cond, es), AN.load(rd)))
    if missing:
        for t in missing:
            print(f"{LOG}   MISSING {t}")
        print(f"{LOG} HARD FAIL: {len(missing)} of {len(ess) * 4} cells "
              f"missing -- nothing written", file=sys.stderr)
        return 1
    ref = runs[(gen.S4GP_ARMS[0], "fixed", ess[0])]
    mask_b = ~AN.cohort_a_mask(ref["innate"].float())
    print(f"{LOG} {len(runs)} cells, cohort B = {int(mask_b.sum())} agents, "
          f"{n_rounds} rounds, seed {seed}")

    # ---- the metric at each round mark ---------------------------------
    rows, series = [], {}
    for r in marks:
        for es in ess:
            m = {}
            for arm in gen.S4GP_ARMS:
                m[arm] = mae_window(runs[(arm, "fixed", es)],
                                    runs[(arm, "evolving", es)],
                                    mask_b, r - 5, r)
            g = m["d8"] - m["b0"]
            series.setdefault(r, []).append((es, m["b0"], m["d8"], g))
            rows.append({"round": r, "window": f"{r - 4}-{r}",
                         "eps_social": es, "mae_sft": m["b0"],
                         "mae_icl": m["d8"], "routing_gap_G": g,
                         "icl_exactly_zero": m["d8"] == 0.0})
    os.makedirs(args.out_dir, exist_ok=True)
    csv_p = os.path.join(args.out_dir, f"{stem}_mae_gap.csv")
    with open(csv_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{LOG} wrote {csv_p} ({len(rows)} rows)")

    # ---- the four registered checks ------------------------------------
    last = max(marks)
    at_last = {es: (b, d, g) for es, b, d, g in series[last]}
    checks = {}
    d0 = at_last[0.0][1]
    checks["1_icl_zero_at_es0"] = {
        "value": d0, "pass": d0 == 0.0,
        "what": "ICL MAE is EXACTLY 0 at eps_social = 0"}
    closed = [g for es, _b, _d, g in series[last] if es <= 0.1]
    checks["2_gap_negative_when_peers_restricted"] = {
        "value": closed, "pass": all(g < 0 for g in closed),
        "what": "G < 0 at eps_social in {0, 0.1}"}
    gs = [g for _es, _b, _d, g in series[last]]
    crossed = any(a < 0 <= b for a, b in zip(gs, gs[1:]))
    checks["3_gap_crosses_zero"] = {
        "value": gs, "pass": crossed,
        "what": "G crosses zero as eps_social opens"}
    if len(marks) >= 2:
        a, b = sorted(marks)[-2:]
        pa = {es: g for es, _b, _d, g in series[a]}
        pb = {es: g for es, _b, _d, g in series[b]}
        drift = {es: pb[es] - pa[es] for es in pa}
        worst = max(abs(v) for v in drift.values())
        checks["4_late_window_stable"] = {
            "value": {f"{k:g}": v for k, v in drift.items()},
            "max_abs_drift": worst, "pass": worst <= 0.01,
            "what": f"|G(round {b}) - G(round {a})| <= 0.01 at every es"}
    print(f"\n{LOG} REGISTERED CHECKS (seed 0, descriptive)")
    for k in sorted(checks):
        c = checks[k]
        print(f"{LOG}   [{'PASS' if c['pass'] else 'FAIL'}] {k}: "
              f"{c['what']}")
        print(f"{LOG}          -> {c['value']}")

    # ---- the figure -----------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7,
                         "xtick.major.width": 0.7,
                         "ytick.major.width": 0.7})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    x = list(range(len(ess)))
    lab = [f"{e:g}" for e in ess]

    s_last = series[last]
    ax1.plot(x, [r[1] for r in s_last], "o-", color="#1f77b4", lw=1.4,
             ms=4, label="SFT (b0)")
    ax1.plot(x, [r[2] for r in s_last], "s-", color="#d62728", lw=1.4,
             ms=4, label="personal-history ICL (d8)")
    ax1.set_xticks(x); ax1.set_xticklabels(lab)
    ax1.set_xlabel(r"$\varepsilon_{\mathrm{social}}$")
    ax1.set_ylabel("cohort-$B$ MAE (fixed vs evolving)")
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax1.text(0.02, 0.02, f"round {last}", transform=ax1.transAxes,
             fontsize=6.5, color="0.35")

    for r, style in zip(sorted(marks), ["--", "-"]):
        ax2.plot(x, [row[3] for row in series[r]], style, marker="o",
                 ms=4, lw=1.4,
                 color="0.55" if style == "--" else "#2ca02c",
                 label=f"round {r}")
    ax2.axhline(0.0, color="0.2", lw=0.8, zorder=0)
    ax2.set_xticks(x); ax2.set_xticklabels(lab)
    ax2.set_xlabel(r"$\varepsilon_{\mathrm{social}}$")
    ax2.set_ylabel(r"routing gap $G=\mathrm{MAE}_{\mathrm{ICL}}"
                   r"-\mathrm{MAE}_{\mathrm{SFT}}$")
    ax2.legend(frameon=False, fontsize=6.5, loc="upper left")
    for a in (ax1, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    pdf = os.path.join(args.out_dir, f"{stem}_routing_gap.pdf")
    png = os.path.join(args.out_dir, f"{stem}_routing_gap.png")
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"{LOG} wrote {pdf} and {png}")

    w = fam.get("w_plat", gen.S4GP_W_PLAT)
    k = fam.get("innate_lambda", gen.W_LAMBDA)
    gtxt = "all_open" if gate == "open" else f"{gate:g}"
    cap = [
        f"{variant} -- routing gap. Left: the Figure-5 metric, the",
        f"final-five-round agent-paired mean |evolving - fixed| in cohort B",
        f"(n={int(mask_b.sum())}) at round {last}, for ordinary SFT and",
        "personal-history ICL. Right: the routing gap G = MAE_ICL -",
        "MAE_SFT at each round mark; G<0 means the shared-weight route",
        "carries more of cohort A's influence, G>0 the personal-history",
        f"route. {fam['slug']}, movielens Action, {n_rounds} rounds, "
        f"seed 0,",
        f"beta={w:g}, gamma(k)={k:g}, alpha=0.5, eps_AI={gtxt}, one "
        f"Deffuant sweep",
        "per round. SEED 0 ONLY -- descriptive, no intervals.",
    ]
    cap_p = os.path.join(args.out_dir, f"{stem}_caption.txt")
    with open(cap_p, "w") as fh:
        fh.write("\n".join(cap) + "\n")
    with open(os.path.join(args.out_dir,
                           f"{stem}_checks.json"), "w") as fh:
        json.dump({"variant": variant, "seed": seed, "rounds": n_rounds,
                   "round_marks": marks, "checks": checks,
                   "rows": rows}, fh, indent=2)
    print(f"{LOG} wrote {cap_p} and {stem}_checks.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
