#!/usr/bin/env python3
"""KL dose-response compared across population environments.

Same quantities as plot_beta_dose_response.py, but one row per population
environment at the shared gate eps_AI=0.4 (Qwen2.5-7B / MovieLens-Action,
seed 0, replace-only data, all available betas):

  row 1  full adoption      eps_soc=0,   W=1,   lambda=0    (pofd_ family)
  row 2  anchored           eps_soc=0,   W=0.5, lambda=0.2  (pofdw_ family)
  row 3  anchored + peers   eps_soc=0.2, W=0.5, lambda=0.2  (pofdws_ family)

Columns: (left) late-round signed gaps, (middle) late-round means,
(right) final-round high-mode shares. Per-column y-scales are shared across
rows so environments can be compared directly. The high-mode boundary is ONE
shared threshold from the pooled round-0 served distribution of all rows'
runs (round-0 models differ only by beta, not environment). Rows whose runs
carry an evolving no-AI twin (peer env) get a gray dotted twin reference.

Capture rule per row (exploratory, same as the single-env figure): shade
from the first beta whose final population low-mode share (1 - q^op_29)
exceeds LOWMODE_THR; hatch back to the last uncaptured beta (onset only
bracketed -- no runs in between).

Outputs (notes/pofd/figures/): beta_dose_response_envs_qwen.{png,pdf},
beta_dose_response_envs_qwen_points.csv. Caption printed.

Usage: python3 experiments/llm/plot_beta_dose_response_envs.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_empirical_return_maps as erm   # noqa: E402  (shared rcParams too)
import matplotlib.pyplot as plt            # noqa: E402

OUT_DIR = erm.OUT_DIR
LATE = 5
LOWMODE_THR = 0.25
EPS_AI = 0.4          # shared gate for the environment comparison
# Capture-band boundary. The pooled round-0 valley (~0.325) separates only the
# RAW low prior mode (~0.25); in the anchored environments captured agents
# park at the force-balance equilibrium x* = (2/3)0.25 + (1/3)abar ~ 0.375,
# ABOVE that valley, so a 0.325 rule is blind to them (verified: at beta=1 the
# anchored rows hold 0.22-0.29 of agents in [0.30, 0.45) and <0.04 below 0.30).
# 0.45 is the preregistered between-modes boundary and lies above BOTH capture
# destinations, so the same rule sees capture in every environment.
CAP_BOUND = 0.45
C_MODEL, C_POP, C_TWIN = "#0072B2", "#D55E00", "#888888"
ENVS = [   # (eps_soc, W, lambda, short label)
    (0.0, 1.0, 0.0, "full adoption\n$W{=}1$, $\\lambda{=}0$"),
    (0.0, 0.5, 0.2, "anchored\n$W{=}0.5$, $\\lambda{=}0.2$"),
    (0.2, 0.5, 0.2, "anchored + peers\n$\\varepsilon_{\\rm soc}{=}0.2$"),
]


def env_cell(candidates, eps_soc, w, lam):
    """Find the candidate cell for one environment at EPS_AI; return
    {beta: run} for every replace slot (most-recent on duplicates)."""
    for key, slots in candidates.items():
        if (key[0] == eps_soc and key[1] == EPS_AI and key[2] == w
                and key[3] == lam):
            out = {}
            for slot, rr in slots.items():
                if slot.startswith("rep:"):
                    r = sorted(rr, key=lambda x: x["mtime"])[-1]
                    if len(rr) > 1:
                        print(f"[select] dup in {slot}: keeping {r['tag']}")
                    out[float(slot[4:])] = r
            return out
    legacy_equivalent = (w == 1.0 and lam == 0.0 and eps_soc == 0.0)
    extra = "" if legacy_equivalent else (
        "\n  This environment (W<1, lambda>0, or peers live) is NOT invariant "
        "to the 2026-07-27 population-update correction, so erm.is_clean_loop "
        "now requires config population_update='nested_ai_then_social_v1'. "
        "The archived pofdw_/pofdws_ runs predate it and are deliberately "
        "excluded -- regenerate them before rebuilding this figure.")
    raise RuntimeError(f"no cell for eps_soc={eps_soc} W={w} lam={lam} "
                       f"eps_ai={EPS_AI}{extra}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _, _, candidates = erm.select_runs()
    env_runs = []
    for eps_soc, w, lam, label in ENVS:
        cell = env_cell(candidates, eps_soc, w, lam)
        betas = sorted(cell)
        print(f"\n[env] {label.replace(chr(10), ' ')}: betas {betas}")
        loaded = {}
        for b in betas:
            print(f"  beta={b:g}: {cell[b]['tag']}")
            loaded[b] = erm.load_run(cell[b])
        env_runs.append((label, loaded))
    beta_sets = [tuple(sorted(l)) for _, l in env_runs]
    assert len(set(beta_sets)) == 1, f"beta grids differ across envs: {beta_sets}"
    betas = sorted(env_runs[0][1])

    # one shared mode threshold from ALL rows' round-0 served distributions
    pool = {f"{i}:{b:g}": r for i, (_, l) in enumerate(env_runs)
            for b, r in l.items()}
    thr = erm.find_mode_threshold(pool)

    rows_out = []          # CSV rows
    env_stats = []         # per env: list of per-beta dicts + twin refs
    for label, loaded in env_runs:
        stats, twins_m, twins_q = [], [], []
        for b in betas:
            r = loaded[b]
            pm, om = r["pm"], r["om"]
            d = {
                "beta": b, "run_tag": r["tag"],
                "pop_gap": float((om[-LATE:] - pm[-LATE:]).mean()),
                "train_gap": float((pm[1:] - om[:-1])[-LATE:].mean()),
                "m_late": float(pm[-LATE:].mean()),
                "x_late": float(om[-LATE:].mean()),
                "q_pred_final": float(erm.occupancy(r["pred"], thr)[-1]),
                "q_op_final": float(erm.occupancy(r["op"], thr)[-1]),
                "low_pred45": 1 - float(erm.occupancy(r["pred"], CAP_BOUND)[-1]),
                "low_op45": 1 - float(erm.occupancy(r["op"], CAP_BOUND)[-1]),
            }
            if r["twin"] is not None:
                twins_m.append(float(r["twin"][-LATE:].mean()))
                twins_q.append(1 - float(erm.occupancy(r["twin"], CAP_BOUND)[-1]))
            stats.append(d)
        twin_m = float(np.mean(twins_m)) if twins_m else None
        twin_q = float(np.mean(twins_q)) if twins_q else None
        if twins_m and (max(twins_m) - min(twins_m) > 1e-6):
            print(f"[twin] {label.splitlines()[0]}: twin spread across betas "
                  f"{max(twins_m) - min(twins_m):.2e} (same seed; should be ~0)")
        env_stats.append((label, stats, twin_m, twin_q))

    # ---- per-env capture boundary + printed comparison table ----
    captures = []
    print(f"\n[capture] rule: final population share below {CAP_BOUND} "
          f"(sees both the raw mode ~0.25 and x*~0.375) > {LOWMODE_THR}")
    for label, stats, twin_m, twin_q in env_stats:
        cap = [d["low_op45"] > LOWMODE_THR for d in stats]
        onset = next((b for b, c in zip(betas, cap) if c), None)
        below = [b for b, c in zip(betas, cap) if not c
                 and (onset is None or b < onset)]
        prev = max(below) if below and onset is not None else None
        captures.append((onset, prev, cap))
        name = label.splitlines()[0]
        print(f"\n  {name}  (twin ref: "
              + (f"mean {twin_m:.3f}, low45 {twin_q:.3f})"
                 if twin_m is not None else "none saved)"))
        print(f"  {'beta':>5} {'pop_gap':>8} {'train_gap':>9} {'m_late':>7} "
              f"{'x_late':>7} {'low45_pred':>10} {'low45_op':>9}")
        for d in stats:
            print(f"  {d['beta']:5g} {d['pop_gap']:+8.4f} "
                  f"{d['train_gap']:+9.4f} {d['m_late']:7.4f} "
                  f"{d['x_late']:7.4f} {d['low_pred45']:10.3f} "
                  f"{d['low_op45']:9.3f}")
        print("  capture onset: "
              + (f"beta={onset:g}, bracketed in ({prev:g}, {onset:g}]"
                 if onset is not None and prev is not None else
                 f"beta={onset:g}" if onset is not None else "none"))

    # ---- figure: rows = envs, cols = quantities, shared per-column scale ----
    xs = np.log1p(betas)
    fig, axg = plt.subplots(3, 3, figsize=(7.0, 6.2), layout="constrained")
    col_heads = ["late-round gaps", "late-round means",
                 f"final share below {CAP_BOUND:g}"]
    gap_all = [v for _, s, _, _ in env_stats
               for d in s for v in (d["pop_gap"], d["train_gap"])]
    mean_all = [v for _, s, tm, _ in env_stats
                for d in s for v in (d["m_late"], d["x_late"])] + \
               [tm for _, _, tm, _ in env_stats if tm is not None]
    low_all = [v for _, s, _, _ in env_stats
               for d in s for v in (d["low_pred45"], d["low_op45"])]
    gpad = 0.08 * (max(gap_all) - min(gap_all))
    mpad = 0.08 * (max(mean_all) - min(mean_all))
    ylims = [(min(gap_all) - gpad, max(gap_all) + gpad),
             (min(mean_all) - mpad, max(mean_all) + mpad),
             (0.0, 1.18 * max(low_all))]

    letters = iter("abcdefghi")
    for ri, ((label, stats, twin_m, twin_q), (onset, prev, _)) in enumerate(
            zip(env_stats, captures)):
        for ci in range(3):
            ax = axg[ri][ci]
            ax.set_title(f"({next(letters)})", loc="left",
                         fontweight="bold", pad=3)
            if ri == 0:
                ax.set_title(col_heads[ci], loc="right", fontsize=7.5,
                             color="#666666", pad=3)
            ax.set_ylim(*ylims[ci])
            ax.set_xticks(xs, [f"{b:g}" for b in betas])
            if ri == 2:
                ax.set_xlabel("$\\beta_{\\mathrm{KL}}$  "
                              "($\\log(1{+}\\beta)$ spacing)")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if onset is not None:
                ax.axvspan(np.log1p(onset), xs[-1] * 1.04, color="#f5e1dd",
                           zorder=0)
                if prev is not None:
                    ax.axvspan(np.log1p(prev), np.log1p(onset),
                               facecolor="none", edgecolor="#d9c2bd",
                               hatch="///", lw=0, zorder=0)
        a, bx, c = axg[ri]
        a.text(-0.44, 0.5, label, transform=a.transAxes, rotation=90,
               ha="center", va="center", fontsize=7.5, color="#333333")
        a.axhline(0, color="#aaaaaa", lw=0.7, zorder=1)
        a.plot(xs, [d["pop_gap"] for d in stats], "o-", color=C_POP,
               lw=1.1, ms=3.2, label="population  $x_{t+1}-m_t$")
        a.plot(xs, [d["train_gap"] for d in stats], "s--", color=C_MODEL,
               lw=1.1, ms=3.2, label="post-training  $m_{t+1}-x_{t+1}$")
        a.set_ylabel("signed gap")
        bx.plot(xs, [d["m_late"] for d in stats], "s--", color=C_MODEL,
                lw=1.1, ms=3.2, label="model  $\\bar m$")
        bx.plot(xs, [d["x_late"] for d in stats], "o-", color=C_POP,
                lw=1.1, ms=3.2, label="population  $\\bar x$")
        bx.set_ylabel("mean (opinion units)")
        c.plot(xs, [d["low_pred45"] for d in stats], "s--", color=C_MODEL,
               lw=1.1, ms=3.2, label="served")
        c.plot(xs, [d["low_op45"] for d in stats], "o-", color=C_POP,
               lw=1.1, ms=3.2, label="opinions")
        c.axhline(LOWMODE_THR, color="#bbbbbb", lw=0.7, ls=":", zorder=1)
        c.set_ylabel(f"share below {CAP_BOUND:g}")
        if twin_m is not None:
            bx.axhline(twin_m, color=C_TWIN, lw=0.8, ls=":", zorder=1)
            bx.text(xs[-1] * 1.03, twin_m, "no-AI twin", fontsize=6,
                    color=C_TWIN, ha="right", va="bottom")
            c.axhline(twin_q, color=C_TWIN, lw=0.8, ls=":", zorder=1)
        if ri == 0:
            a.legend(loc="lower left", frameon=False, fontsize=5.5)
            bx.legend(loc="lower left", frameon=False, fontsize=5.5)
            c.legend(loc="upper left", frameon=False, fontsize=5.5)

    outs = []
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"beta_dose_response_envs_qwen.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None)
        outs.append(p)
    plt.close(fig)
    print(f"\n[fig] wrote {outs}")

    # ---- CSV ----
    csv_path = os.path.join(OUT_DIR, "beta_dose_response_envs_qwen_points.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["env", "eps_soc", "w_plat", "innate_lambda", "eps_ai",
                      "run_tag", "beta", "pop_gap_last5", "train_gap_last5",
                      "m_mean_last5", "x_mean_last5", "q_pred_final",
                      "q_op_final", "low_pred_045", "low_op_045",
                      "twin_mean_last5", "twin_low_045", "mode_threshold",
                      "capture_rule", "in_capture_region"])
        for (env, w, lam, _), (label, stats, twin_m, twin_q), (_, _, cap) in \
                zip([(e[0], e[1], e[2], e[3]) for e in ENVS],
                    env_stats, captures):
            name = label.splitlines()[0]
            for d, c in zip(stats, cap):
                wtr.writerow([name, env, w, lam, EPS_AI, d["run_tag"],
                              d["beta"], f"{d['pop_gap']:.6f}",
                              f"{d['train_gap']:.6f}", f"{d['m_late']:.6f}",
                              f"{d['x_late']:.6f}", f"{d['q_pred_final']:.6f}",
                              f"{d['q_op_final']:.6f}",
                              f"{d['low_pred45']:.6f}",
                              f"{d['low_op45']:.6f}",
                              f"{twin_m:.6f}" if twin_m is not None else "",
                              f"{twin_q:.6f}" if twin_q is not None else "",
                              f"{thr:.3f}",
                              f"low_op_{CAP_BOUND}>{LOWMODE_THR}", int(c)])
    print(f"[csv] wrote {csv_path}")

    print(f"""
[caption] beta_dose_response_envs_qwen
KL dose-response across population environments, one row per environment
(matched Qwen2.5-7B/MovieLens-Action runs, eps_AI={EPS_AI:g}, seed 0,
30 rounds, replace-only data, beta on log(1+beta) spacing; per-column
y-scales shared across rows). Columns: signed late-round gaps; final-{LATE}-round means; final-round share
of agents/serves below {CAP_BOUND:g}. The {CAP_BOUND:g} boundary is used
(rather than the round-0 density valley {thr:.3f}) because capture has a
DIFFERENT destination per environment: the raw prior mode ~0.25 under full
adoption, but the force-balance equilibrium x* ~ 0.375 under the innate
anchor -- x* lies above the valley, so a {thr:.3f} rule is blind to anchored
capture; {CAP_BOUND:g} lies above both destinations. Gray dotted line: the
matched no-AI twin (saved only for the peer environment). Shading per row:
capture observed where the final population share below {CAP_BOUND:g} exceeds
{LOWMODE_THR}; hatched band = transition interval (onset bracketed, no runs
between). Single seed; exploratory boundaries; no scaling-law claim.""")


if __name__ == "__main__":
    main()
