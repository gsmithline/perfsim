#!/usr/bin/env python3
"""Regularization dose-response: what the KL weight beta does to the loop.

Reuses run selection, loading, temporal-semantics validation, and the
high-mode threshold from plot_empirical_return_maps.py (same matched
Qwen2.5-7B / MovieLens-Action cell; replace-only data regime, every
available beta). Indexing verified there: pred[t] = m_t (served),
op[t] = x_{t+1} (population after responding to m_t), pred[t+1] = m_{t+1}.

Three panels over beta at log(1+beta) spacing:
  (a) late-round signed gaps: population gap x_{t+1}-m_t and
      post-training gap m_{t+1}-x_{t+1} (each averaged over the final
      5 rounds / transitions)
  (b) late-round means: model m and population x (final-5-round average)
  (c) final-round high-mode shares q^pred_29, q^op_29 (mode boundary from
      the pooled round-0 served distribution of these runs)

A POSSIBLE CAPTURE REGION [exploratory] is shaded from the first beta
whose final-round population low-mode share (1 - q^op_29) exceeds
LOWMODE_THR; the band back to the last uncaptured beta is hatched
"onset unresolved" because no runs exist between them. The boundary is
data-derived, single-seed, and makes no scaling-law claim.

Outputs (notes/pofd/figures/): beta_dose_response_qwen.{png,pdf},
beta_dose_response_qwen_points.csv. Caption text is printed.

Usage: python3 experiments/llm/plot_beta_dose_response.py
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
LATE = 5              # rounds / transitions averaged for "late-round" values
LOWMODE_THR = 0.25    # capture rule: final population low-mode share > this
C_MODEL, C_POP = "#0072B2", "#D55E00"   # Okabe-Ito; model blue, population verm.


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cell, chosen, _ = erm.select_runs()
    eps, eps_ai, w, lam = cell[0], cell[1], cell[2], cell[3]
    print(f"\n[select] cell: eps={eps:g} eps_ai={eps_ai:g} W={w:g} "
          f"lambda={lam:g}; replace-only beta sweep (the pristine arm is a "
          "different knob and is NOT part of this dose-response)")
    betas = sorted(float(s[4:]) for s in chosen if s.startswith("rep:"))
    runs = {}
    for b in betas:
        r = chosen[f"rep:{b:g}"]
        print(f"  beta={b:g}: {r['tag']}")
        runs[f"$\\beta{{=}}{b:g}$"] = erm.load_run(r)
    thr = erm.find_mode_threshold(runs)

    rows = []
    for b, r in zip(betas, runs.values()):
        pm, om = r["pm"], r["om"]
        qp = erm.occupancy(r["pred"], thr)
        qo = erm.occupancy(r["op"], thr)
        rows.append({
            "beta": b, "run_tag": r["tag"],
            # same-index om - pm is the serve/response pair (verified);
            # pm[1:] - om[:-1] is the post-training update on x_{t+1}
            "pop_gap": float((om[-LATE:] - pm[-LATE:]).mean()),
            "train_gap": float((pm[1:] - om[:-1])[-LATE:].mean()),
            "m_late": float(pm[-LATE:].mean()),
            "x_late": float(om[-LATE:].mean()),
            "q_pred_final": float(qp[-1]), "q_op_final": float(qo[-1]),
        })
    print(f"\n{'beta':>5} {'pop_gap':>8} {'train_gap':>9} {'m_late':>7} "
          f"{'x_late':>7} {'q_pred_f':>8} {'q_op_f':>7} {'low_op_f':>8}")
    for d in rows:
        print(f"{d['beta']:5g} {d['pop_gap']:+8.4f} {d['train_gap']:+9.4f} "
              f"{d['m_late']:7.4f} {d['x_late']:7.4f} {d['q_pred_final']:8.3f} "
              f"{d['q_op_final']:7.3f} {1 - d['q_op_final']:8.3f}")

    # ---- capture-region boundary (exploratory, data-derived) ----
    captured = [1 - d["q_op_final"] > LOWMODE_THR for d in rows]
    onset = next((b for b, c in zip(betas, captured) if c), None)
    prev = None
    if onset is not None:
        below = [b for b, c in zip(betas, captured) if not c and b < onset]
        prev = max(below) if below else None
    print(f"\n[capture] rule: final population low-mode share "
          f"(1 - q^op_29 at threshold {thr:.3f}) > {LOWMODE_THR}")
    if onset is None:
        print("[capture] no beta meets the rule -- no region shaded")
    else:
        print(f"[capture] first beta meeting the rule: {onset:g}"
              + (f"; last uncaptured beta: {prev:g} -> onset only bracketed "
                 f"in ({prev:g}, {onset:g}], hatched as unresolved"
                 if prev is not None else ""))

    # ---- pattern description (facts first, heuristic label flagged) ----
    lows = [1 - d["q_op_final"] for d in rows]
    jumps = [lows[i + 1] - lows[i] for i in range(len(lows) - 1)]
    print("[pattern] low-mode share steps between consecutive betas: "
          + ", ".join(f"{betas[i]:g}->{betas[i + 1]:g}: {j:+.3f}"
                      for i, j in enumerate(jumps)))
    big = max(jumps, key=abs)
    heur = ("threshold-like (one step carries most of the change)"
            if abs(big) > 0.6 * (max(lows) - min(lows) + 1e-12)
            else "gradual")
    mono = all(x <= y + 1e-9 for x, y in zip(lows, lows[1:]))
    print(f"[pattern] heuristic read: {heur}; low-mode share "
          f"{'monotone' if mono else 'NON-monotone'} in beta; late means on "
          "the uncaptured branch: "
          + ", ".join(f"b{d['beta']:g}={d['m_late']:.3f}" for d in rows))

    # ---- figure ----
    xs = np.log1p(betas)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), layout="constrained")
    heads = ["regularization creates opposing\nmodel-population gaps",
             "the loop jumps between\ntwo resting states",
             "high-mode occupancy collapses\nbeyond the threshold"]
    for ax, pl, head in zip(axes, "abc", heads):
        ax.set_title(f"({pl})", loc="left", fontweight="bold", pad=4)
        ax.set_title(head, loc="right", fontsize=7.5, color="#666666", pad=4)
        ax.set_xticks(xs, [f"{b:g}" for b in betas])
        ax.set_xlabel("$\\beta_{\\mathrm{KL}}$   ($\\log(1{+}\\beta)$ spacing)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if onset is not None:
            x1 = np.log1p(onset)
            ax.axvspan(x1, ax.get_xlim()[1] if False else xs[-1] * 1.04,
                       color="#f5e1dd", zorder=0)
            if prev is not None:
                ax.axvspan(np.log1p(prev), x1, facecolor="none",
                           edgecolor="#d9c2bd", hatch="///", lw=0, zorder=0)
    a, bx, c = axes

    a.axhline(0, color="#aaaaaa", lw=0.7, zorder=1)
    a.plot(xs, [d["pop_gap"] for d in rows], "o-", color=C_POP, lw=1.1, ms=3.5,
           label="population gap  $x_{t+1}-m_t$")
    a.plot(xs, [d["train_gap"] for d in rows], "s--", color=C_MODEL, lw=1.1,
           ms=3.5, label="post-training gap  $m_{t+1}-x_{t+1}$")
    a.set_ylabel(f"signed gap, final-{LATE}-round mean")
    a.legend(loc="upper left", frameon=False, fontsize=6)

    bx.plot(xs, [d["m_late"] for d in rows], "s--", color=C_MODEL, lw=1.1,
            ms=3.5, label="model  $\\bar m$")
    bx.plot(xs, [d["x_late"] for d in rows], "o-", color=C_POP, lw=1.1, ms=3.5,
            label="population  $\\bar x$")
    bx.set_ylabel(f"final-{LATE}-round mean (opinion units)")
    bx.legend(loc="lower left", frameon=False, fontsize=6)

    c.plot(xs, [d["q_pred_final"] for d in rows], "s--", color=C_MODEL, lw=1.1,
           ms=3.5, label="served  $q^{pred}_{29}$")
    c.plot(xs, [d["q_op_final"] for d in rows], "o-", color=C_POP, lw=1.1,
           ms=3.5, label="opinions  $q^{op}_{29}$")
    c.axhline(1 - LOWMODE_THR, color="#bbbbbb", lw=0.7, ls=":", zorder=1)
    c.set_ylabel(f"high-mode share (thr {thr:.3f})")
    c.set_ylim(0, 1.05)
    c.legend(loc="lower left", frameon=False, fontsize=6)
    if onset is not None:
        c.text(0.5 * (np.log1p(onset) + xs[-1] * 1.04), 1.03,
               "capture observed", fontsize=6, color="#a2624f",
               ha="center", va="top")
        if prev is not None:
            c.text(0.5 * (np.log1p(prev) + np.log1p(onset)), 0.28,
                   "transition\ninterval", fontsize=6, color="#a2624f",
                   ha="center", rotation=90)

    outs = []
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"beta_dose_response_qwen.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None)
        outs.append(p)
    plt.close(fig)
    print(f"\n[fig] wrote {outs}")

    # ---- CSV of every plotted point ----
    csv_path = os.path.join(OUT_DIR, "beta_dose_response_qwen_points.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["run_tag", "beta", "x_log1p_beta", "pop_gap_last5",
                      "train_gap_last5", "m_mean_last5", "x_mean_last5",
                      "q_pred_final", "q_op_final", "mode_threshold",
                      "lowmode_rule", "in_capture_region"])
        for d, cap in zip(rows, captured):
            wtr.writerow([d["run_tag"], d["beta"],
                          f"{np.log1p(d['beta']):.6f}",
                          f"{d['pop_gap']:.6f}", f"{d['train_gap']:.6f}",
                          f"{d['m_late']:.6f}", f"{d['x_late']:.6f}",
                          f"{d['q_pred_final']:.6f}",
                          f"{d['q_op_final']:.6f}", f"{thr:.3f}",
                          f"1-q_op_final>{LOWMODE_THR}", int(cap)])
    print(f"[csv] wrote {csv_path}")

    print(f"""
[caption] beta_dose_response_qwen
KL-regularization dose-response for the matched Qwen2.5-7B/MovieLens-Action
loop cell (723 agents, eps_ai={eps_ai:g}, full-adoption population, seed 0,
30 rounds, replace-only data; beta on log(1+beta) spacing). (a) Signed
late-round gaps: what the population leaves unrealized (x_(t+1)-m_t) and what
post-training then moves (m_(t+1)-x_(t+1)), final-{LATE}-round means.
(b) Final-{LATE}-round model and population means. (c) Final-round high-mode
shares of served predictions and opinions (mode boundary {thr:.3f} = round-0
density valley). Solid shading, "capture observed": betas whose final
population low-mode share exceeds {LOWMODE_THR} (first at beta={onset:g}).
Hatched band, "transition interval": the onset is only bracketed in
({prev:g}, {onset:g}] -- no runs exist between. Takeaway: a modest increase in
regularization between {prev:g} and {onset:g} changes both how the loop
stabilizes and which population state it produces. Single seed; no
scaling-law claim is made, and the beta grid is too coarse to locate the
transition more precisely.""")


if __name__ == "__main__":
    main()
