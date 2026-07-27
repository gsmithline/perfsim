#!/usr/bin/env python3
"""Two publication figures from the matched Qwen/MovieLens-Action pofd runs.

Figure 1 (peers_buffer_amplify_qwen): peers buffer, then amplify.
  (a) final population lower-band share (opinions < 0.45) vs beta, one line
      per environment (full adoption / anchored / anchored+peers), eps_AI=0.4
  (b) peer effect = (anchored+peers - anchored) lower-band share vs beta;
      negative = buffering, positive = amplification (a matched-arm
      difference, so it is labeled "peer effect", not a raw share)
  (c) final opinion distributions at beta=1, with Qwen's low prior mode
      (~0.25) and the anchored balance point x* (~0.375) marked

Figure 2 (feedback_capture_qwen): feedback produces capture.
  Full-adoption environment, beta in {0.2, 0.5, 1}.
  (a) model and population means over rounds
  (b) round-0 KL displacement (m_0(beta) vs the beta=0 round-0 serve --
      an explicit beta=0 baseline) separated from the additional
      round-0-to-29 movement (m_29 - m_0, feedback-driven)
  (c) population lower-band share (< 0.45) over rounds

All values from saved trajectories of config-matched runs (selection +
loading + temporal semantics from plot_empirical_return_maps.py; environment
cells from plot_beta_dose_response_envs.py). Single seed (s0) -- labeled on
the figures and in the captions. Outputs PNG + PDF + plotted-point CSV per
figure under notes/pofd/figures/.

Usage: python3 experiments/llm/plot_peer_feedback_figs.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_empirical_return_maps as erm       # noqa: E402
import plot_beta_dose_response_envs as env_mod  # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402

OUT_DIR = erm.OUT_DIR
CAP_BOUND = 0.45
LOW_MODE, X_STAR = 0.25, 0.375   # Qwen low prior mode; anchored balance point
ENV_STYLE = {   # (eps_soc, W, lambda) -> label, color, ls, marker
    (0.0, 1.0, 0.0): ("full adoption", "#888888", (0, (3, 1, 1, 1)), "^"),
    (0.0, 0.5, 0.2): ("anchored", "#E69F00", "--", "s"),
    (0.2, 0.5, 0.2): ("anchored + peers", "#0072B2", "-", "o"),
}
F2_BETAS = [0.2, 0.5, 1.0]
F2_COL = {0.2: "#6baed6", 0.5: "#3182bd", 1.0: "#08519c"}


def low_band(arr2d):
    """Per-round share of finite values below CAP_BOUND."""
    return 1.0 - erm.occupancy(arr2d, CAP_BOUND)


def save(fig, stem):
    outs = []
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None)
        outs.append(p)
    plt.close(fig)
    print(f"[fig] wrote {outs}")
    return outs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _, _, candidates = erm.select_runs()
    envs = {}     # env key -> {beta: loaded run}
    for key in ENV_STYLE:
        cell = env_mod.env_cell(candidates, *key)
        envs[key] = {b: erm.load_run(r) for b, r in sorted(cell.items())}
        print(f"[env] {ENV_STYLE[key][0]}: "
              + ", ".join(r["tag"] for r in cell.values()))
    betas = sorted(next(iter(envs.values())))
    assert all(sorted(v) == betas for v in envs.values()), "beta grids differ"
    xs = np.log1p(betas)

    # =============== Figure 1: peers buffer, then amplify ===============
    lower = {k: [float(low_band(v[b]["op"])[-1]) for b in betas]
             for k, v in envs.items()}
    anch = lower[(0.0, 0.5, 0.2)]
    peer = lower[(0.2, 0.5, 0.2)]
    peer_eff = [p - a for p, a in zip(peer, anch)]
    print("\n[fig1] final lower-band shares (op < 0.45):")
    for k, vals in lower.items():
        print(f"  {ENV_STYLE[k][0]:>17}: "
              + " ".join(f"b{b:g}={v:.3f}" for b, v in zip(betas, vals)))
    print("  peer effect (peers - anchored): "
          + " ".join(f"b{b:g}={v:+.3f}" for b, v in zip(betas, peer_eff)))

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.6),
                                  layout="constrained")
    heads = ["lower-band share grows\nwith the KL weight",
             "peers buffer weak anchoring,\namplify strong",
             "capture destinations differ\nby environment ($\\beta{=}1$)"]
    for ax, pl, head in zip((a, b, c), "abc", heads):
        ax.set_title(f"({pl})", loc="left", fontweight="bold", pad=3)
        ax.set_title(head, loc="right", fontsize=7, color="#666666", pad=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in (a, b):
        ax.set_xticks(xs, [f"{b_:g}" for b_ in betas])
        ax.set_xlabel("$\\beta_{\\mathrm{KL}}$  ($\\log(1{+}\\beta)$ spacing)")

    for k, vals in lower.items():
        lab, col, ls, mk = ENV_STYLE[k]
        a.plot(xs, vals, ls=ls, marker=mk, color=col, lw=1.1, ms=3.4,
               label=lab)
    a.set_ylabel(f"population share below {CAP_BOUND:g}")
    a.set_ylim(0, None)
    a.legend(loc="upper left", frameon=False, fontsize=6)

    b.axhline(0, color="#aaaaaa", lw=0.7, zorder=1)
    b.plot(xs, peer_eff, "o-", color="#0072B2", lw=1.2, ms=3.6)
    b.set_ylabel("peer effect on lower-band share\n(anchored+peers $-$ anchored)")
    b.text(0.04, 0.10, "buffering", transform=b.transAxes, fontsize=6.5,
           color="#555555", style="italic")
    b.text(0.96, 0.90, "amplification", transform=b.transAxes, fontsize=6.5,
           color="#555555", style="italic", ha="right")

    bins = np.linspace(0.0, 1.0, 51)
    hist_rows, dens_by_env = [], {}
    for k in ENV_STYLE:
        lab, col, ls, _ = ENV_STYLE[k]
        op_f = envs[k][1.0]["op"][-1].numpy()
        dens, edges = np.histogram(op_f, bins=bins, density=True)
        dens_by_env[k] = dens
        c.stairs(dens, edges, color=col, lw=1.1, ls="-" if ls == "-" else "--"
                 if ls == "--" else ":", label=lab)
        hist_rows += [(lab, edges[i], edges[i + 1], dens[i])
                      for i in range(len(dens))]
    # full adoption piles mass at the exact served values (near-point masses);
    # clip the axis to the anchored scale and say so, rather than letting one
    # spike flatten the two distributions of interest
    ymax = 1.2 * max(dens_by_env[k].max() for k in ENV_STYLE
                     if k != (0.0, 1.0, 0.0))
    c.set_ylim(0, ymax)
    gray = dens_by_env[(0.0, 1.0, 0.0)]
    if gray.max() > ymax:
        pk = 0.5 * (bins[gray.argmax()] + bins[gray.argmax() + 1])
        c.text(pk + 0.02, ymax * 0.97, f"clipped\n(peak {gray.max():.0f})",
               fontsize=5.5, color="#888888", va="top")
    for v, txt, ha in [(LOW_MODE, f"low mode {LOW_MODE:g} ", "right"),
                       (X_STAR, f" $x^*{{\\approx}}{X_STAR:g}$", "left")]:
        c.axvline(v, color="#999999", lw=0.7, ls=":", zorder=1)
        c.text(v, ymax * 0.80, txt, fontsize=6, color="#777777", ha=ha,
               va="top")
    c.set_xlim(0.05, 0.95)
    c.set_xlabel("final opinion (round 29)")
    c.set_ylabel("density")
    fig.text(0.998, 0.005, "single seed (s0)", ha="right", fontsize=6,
             color="#999999")
    save(fig, "peers_buffer_amplify_qwen")

    csv1 = os.path.join(OUT_DIR, "peers_buffer_amplify_qwen_points.csv")
    with open(csv1, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "env", "beta", "x_log1p_beta",
                    "lower_band_share_op_final", "peer_effect",
                    "bin_left", "bin_right", "density", "run_tag", "seed"])
        for k, vals in lower.items():
            lab = ENV_STYLE[k][0]
            for b_, v in zip(betas, vals):
                w.writerow(["a", lab, b_, f"{np.log1p(b_):.6f}", f"{v:.6f}",
                            "", "", "", "", envs[k][b_]["tag"], 0])
        for b_, v in zip(betas, peer_eff):
            w.writerow(["b", "peers-anchored", b_, f"{np.log1p(b_):.6f}", "",
                        f"{v:.6f}", "", "", "", "", 0])
        for lab, lo, hi, d in hist_rows:
            w.writerow(["c", lab, 1.0, "", "", "", f"{lo:.3f}", f"{hi:.3f}",
                        f"{d:.6f}", "", 0])
    print(f"[csv] wrote {csv1}")

    # =============== Figure 2: feedback produces capture ===============
    fa = envs[(0.0, 1.0, 0.0)]
    ref_m0 = float(fa[0.0]["pm"][0])   # beta=0 round-0 serve (explicit baseline)
    print(f"\n[fig2] full adoption; beta=0 round-0 serve (baseline) = "
          f"{ref_m0:.4f}")
    decomp = []
    for b_ in F2_BETAS:
        pm = fa[b_]["pm"]
        decomp.append({"beta": b_, "m0": float(pm[0]), "m29": float(pm[-1]),
                       "r0_disp": float(pm[0]) - ref_m0,
                       "loop_move": float(pm[-1]) - float(pm[0])})
        d = decomp[-1]
        print(f"  b{b_:g}: m0 {d['m0']:.3f} m29 {d['m29']:.3f} | round-0 KL "
              f"displacement {d['r0_disp']:+.3f}, loop movement "
              f"{d['loop_move']:+.3f}")

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.6),
                                  layout="constrained")
    heads = ["model and population\ndescend together",
             "round-0 KL displacement vs\nmovement added by the loop",
             "capture accumulates\nover rounds"]
    for ax, pl, head in zip((a, b, c), "abc", heads):
        ax.set_title(f"({pl})", loc="left", fontweight="bold", pad=3)
        ax.set_title(head, loc="right", fontsize=7, color="#666666", pad=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    rounds = np.arange(30)
    for b_ in F2_BETAS:
        r = fa[b_]
        a.plot(rounds, r["pm"], "-", color=F2_COL[b_], lw=1.1,
               label=f"$\\beta{{=}}{b_:g}$")
        a.plot(rounds, r["om"], ":", color=F2_COL[b_], lw=1.1)
    a.set_xlabel("round")
    a.set_ylabel("mean (opinion units)")
    a.legend(loc="lower left", frameon=False, fontsize=6)
    a.text(0.97, 0.70, "solid: model $\\bar m$\ndotted: population $\\bar x$",
           transform=a.transAxes, fontsize=6, color="#555555", ha="right",
           va="top")

    xpos = np.arange(len(F2_BETAS))
    b.axhline(ref_m0, color="#999999", lw=0.8, ls=":", zorder=1)
    b.text(xpos[-1] + 0.38, ref_m0, "$\\beta{=}0$\nround-0 serve", fontsize=6,
           color="#777777", ha="right", va="bottom")
    for i, d in enumerate(decomp):
        col = F2_COL[d["beta"]]
        b.plot([i, i], [d["m0"], d["m29"]], color=col, lw=1.2, zorder=2)
        b.plot(i, d["m0"], "o", mfc="white", mec=col, ms=5, mew=1.1, zorder=3)
        b.plot(i, d["m29"], "o", color=col, ms=5, zorder=3)
        b.annotate("", xy=(i, d["m29"]), xytext=(i, d["m0"]),
                   arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0))
        b.text(i + 0.09, d["m0"] + 0.006, f"r0 {d['r0_disp']:+.3f}",
               fontsize=6, color="#555555")
        b.text(i + 0.09, 0.5 * (d["m0"] + d["m29"]),
               f"loop {d['loop_move']:+.3f}", fontsize=6, color="#555555")
    b.set_xticks(xpos, [f"$\\beta{{=}}{b_:g}$" for b_ in F2_BETAS])
    b.set_xlim(-0.5, len(F2_BETAS) - 0.2)
    b.set_ylabel("mean served prediction")
    b.text(0.03, 0.05, "open: round 0\nfilled: round 29",
           transform=b.transAxes, fontsize=6, color="#555555")

    for b_ in F2_BETAS:
        c.plot(rounds, low_band(fa[b_]["op"]), "-", color=F2_COL[b_], lw=1.1,
               label=f"$\\beta{{=}}{b_:g}$")
    c.set_xlabel("round")
    c.set_ylabel(f"population share below {CAP_BOUND:g}")
    c.set_ylim(0, None)
    c.legend(loc="center right", frameon=False, fontsize=6)
    fig.text(0.998, 0.005, "single seed (s0)", ha="right", fontsize=6,
             color="#999999")
    save(fig, "feedback_capture_qwen")

    csv2 = os.path.join(OUT_DIR, "feedback_capture_qwen_points.csv")
    with open(csv2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "beta", "round", "m_t", "x_t",
                    "lower_band_share_op", "m0", "m29", "ref_beta0_m0",
                    "r0_displacement", "loop_movement", "run_tag", "seed"])
        for b_ in F2_BETAS:
            r = fa[b_]
            lb = low_band(r["op"])
            for t in range(30):
                w.writerow(["a,c", b_, t, f"{r['pm'][t]:.6f}",
                            f"{r['om'][t]:.6f}", f"{lb[t]:.6f}",
                            "", "", "", "", "", r["tag"], 0])
        for d in decomp:
            w.writerow(["b", d["beta"], "", "", "", "", f"{d['m0']:.6f}",
                        f"{d['m29']:.6f}", f"{ref_m0:.6f}",
                        f"{d['r0_disp']:.6f}", f"{d['loop_move']:.6f}",
                        fa[d["beta"]]["tag"], 0])
    print(f"[csv] wrote {csv2}")

    print(f"""
[caption] peers_buffer_amplify_qwen
Peers buffer weak anchoring and amplify strong anchoring (matched Qwen2.5-7B/
MovieLens-Action runs, eps_AI=0.4, seed 0, replace-only data). (a) Final
population lower-band share (opinions below {CAP_BOUND:g}, a boundary above
both capture destinations) vs beta for three population environments.
(b) Peer effect: the anchored+peers minus anchored difference in lower-band
share -- negative means peer interaction absorbs anchoring pressure
(buffering, strongest at beta=0.2), positive means it deepens capture
(amplification, at beta=1). (c) Final opinion distributions at beta=1: under
full adoption the captured mass sits at Qwen's low prior mode (~{LOW_MODE:g});
under the innate anchor it parks at the balance point x* (~{X_STAR:g}).
Single seed; onset locations are bracketed by a coarse beta grid.

[caption] feedback_capture_qwen
Feedback produces capture (full-adoption environment, beta in 0.2/0.5/1).
(a) Model (solid) and population (dotted) means over rounds: beta=0.5 STARTS
HIGH (m_0={decomp[1]['m0']:.3f}, barely displaced from the beta=0 round-0
serve {ref_m0:.3f}) and is walked down by repeated retraining on its own
consequences; beta=1 BEGINS ALREADY DISPLACED (m_0={decomp[2]['m0']:.3f}).
(b) The decomposition: round-0 KL displacement (relative to the explicit
beta=0 baseline) vs the additional round-0-to-29 movement. For beta=0.5 the
loop contributes {decomp[1]['loop_move']:+.3f} vs {decomp[1]['r0_disp']:+.3f}
at round 0; for beta=1 the split is {decomp[2]['loop_move']:+.3f} vs
{decomp[2]['r0_disp']:+.3f}. beta=0.2 drifts slightly UPWARD
({decomp[0]['loop_move']:+.3f}). (c) Population lower-band share over rounds:
capture accumulates round by round rather than appearing at deployment.
Single seed.""")


if __name__ == "__main__":
    main()
