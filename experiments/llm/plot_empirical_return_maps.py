#!/usr/bin/env python3
"""Empirical Simon-style performative return maps from existing pofd loop runs.

Decomposition visualized (three panels):
    m_t  --population response-->  x_{t+1}  --post-training-->  m_{t+1}
  (a) serving response:      m_t     vs x_{t+1}
  (b) post-training response: x_{t+1} vs m_{t+1}
  (c) full return relation:  m_t     vs m_{t+1}

TEMPORAL SEMANTICS (verified against
experiments/scripts/cluster_pipelines/run_pokec_gated_lm.py, main loop
lines ~960-1363):
  Within round t the runner (deploy_every=1 for all selected runs):
    1. TRAINS the learner on train_data (t=0: the innate seed batch;
       t>0: buffer data whose labels y are the round t-1 post-response
       opinions -- replace regime takes only the latest buffer entry,
       accumulate+pristine mixes a fixed fraction of the innate seed).
    2. PREDICTS: preds = lm(...) -> last_preds. This is the model
       deployed at round t, i.e. m_t.
    3. POPULATION RESPONDS to last_preds (gated blend, optional peer
       sweep / innate re-anchor) -> op.
    4. RECORDS row t: pred_* summaries from step-2 preds, op_* summaries
       from step-3 op; then op_raw.append(op), pred_raw.append(last_preds).
    5. BUFFERS y = op for next round's training pool.
  Therefore, at array index t:
    pred_raw[t] / pred_mean[t] = m_t      (served by the round-t model)
    op_raw[t]   / op_mean[t]   = x_{t+1}  (population AFTER responding to m_t)
    pred_raw[t+1]              = m_{t+1}  (next model, trained on op_raw[t])
  Same-index (pred, op) is the serve/response pair; the ONLY cross-index
  pairing is (op[t], pred[t+1]) for the post-training map. A 30-round run
  yields 30 (m_t, x_{t+1}) pairs and 29 full transitions.

Run selection is by config.json fields only (never tag parsing, except as
a final duplicate tiebreak via mtime). See select_runs() for the filter.

Outputs (all under notes/pofd/figures/):
  empirical_return_maps_mean_qwen.{png,pdf}        mean-state version
  empirical_return_maps_occupancy_qwen.{png,pdf}   high-mode occupancy version
  empirical_return_map_statefulness.png            thick-relation diagnostic
  empirical_return_maps_gender_qwen.{png,pdf}      observational M-F gap version
  empirical_return_maps_points.csv                 every plotted point
  empirical_return_maps_captions.txt               caption text per figure

Usage: python3 experiments/llm/plot_empirical_return_maps.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

DATA_ROOTS = [
    os.path.join(REPO, "notes", "pofd", "cluster"),
    os.path.join(REPO, "runs", "pokec_gated_lm"),
]
OUT_DIR = os.path.join(REPO, "notes", "pofd", "figures")

# Colorblind-safe styling: vermillion for the unanchored baseline, a single
# ordered blue ramp for the KL-strength sweep (monotone lightness encodes
# beta), green dash-dot for the data anchor. Line style + marker also differ
# so color alone never carries the distinction.
BETA_BLUES = {0.1: "#9ecae1", 0.2: "#6baed6", 0.5: "#3182bd", 1.0: "#08519c"}


def style_of(kind, beta=None):
    if kind == "rep" and beta == 0.0:
        return dict(color="#D55E00", ls="-", marker="o")
    if kind == "rep":
        blue = BETA_BLUES.get(beta, "#08519c")
        return dict(color=blue, ls="--", marker="s")
    return dict(color="#009E73", ls=(0, (3, 1, 1, 1)), marker="^")


STYLE = {}   # filled at selection time (label -> style), keyed by run label
MARK_ROUNDS = (0, 5, 10, 15, 20, 25)   # + final round, marked separately
FIXED_POINT_TOL_MEAN = 0.01            # |m_{t+1}-m_t| for final 5 transitions
FIXED_POINT_TOL_OCC = 0.02             # same criterion on occupancy scale
DEFAULT_MODE_THRESHOLD = 0.45          # preregistered fallback boundary
SENS_THRESHOLDS = (0.40, 0.45, 0.50)


def read_json_retry(path, attempts=6, sleep_s=0.3):
    """Retrying JSON read (mounted-FS flake guard). Raises after attempts --
    a failed read must never silently drop a run."""
    last = None
    for _ in range(attempts):
        try:
            with open(path) as fh:
                return json.load(fh)
        except OSError as exc:
            last = exc
            time.sleep(sleep_s)
    raise OSError(f"failed to read {path} after {attempts} attempts: {last}")


def load_pt_retry(path, attempts=6, sleep_s=0.3):
    last = None
    for _ in range(attempts):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except OSError as exc:
            last = exc
            time.sleep(sleep_s)
    raise OSError(f"failed to read {path} after {attempts} attempts: {last}")


def is_clean_loop(cfg):
    """Config-field filter for the matched comparison: Qwen2.5-7B, MovieLens
    Action (723 agents), 30-round loop, SFT (KL via kl_beta), fresh weights,
    no ICL/ICRH/feedback/canary/profile interventions, seed 0."""
    def g(key, default=None):
        return cfg.get(key, default)
    if "Qwen2.5-7B" not in str(g("base_model", "")):
        return False
    if g("dataset") is not None and g("dataset") != "movielens":
        return False
    if g("dataset") is None and g("n_labeled") != 723:   # legacy-config fallback
        return False
    if g("ml_target", "Action") != "Action":
        return False
    checks = [
        g("run_mode") == "loop", g("seed") == 0, g("n_rounds") == 30,
        g("deploy_every", 1) == 1,
        str(g("training_style", "sft")) in ("sft", "sft_kl"),
        g("fresh_each_round", False) is True,
        g("pop_model", "ab") == "ab",
        not g("icrh", False), g("icl_k", 0) == 0, g("icl_days", 0) == 0,
        g("feedback_mode", "none") == "none", not g("ab_retain", False),
        not g("pop_reset", False), g("ab_sweeps", 1) == 1,
        float(g("canary_delta", 0.0)) == 0.0,
        float(g("gamma_bias", 0.0)) == 0.0,
        float(g("profile_shuffle_p", 0.0)) == 0.0,
        float(g("profile_sort_q", 0.0)) == 0.0,
        not g("profile_drop_cols", []), not g("profile_permute_cols", []),
        not g("do_sample", False),
        population_update_ok(cfg),
    ]
    return all(checks)


NESTED_MARKER = "nested_ai_then_social_v1"


def population_update_ok(cfg):
    """Reject runs whose population dynamics predate the 2026-07-27 correction,
    UNLESS they sit at the legacy-equivalent corner.

    The corrected operator is
        h = k innate + (1-k) x ; z = (1-W)h + Wm if |m-x| < eps_AI else h ;
        x' = D_eps_social(z)
    Runs written before the fix applied the innate anchor AFTER the platform
    blend (diluting W m) and ran the peer sweep BEFORE it. At W=1, k=0,
    eps_social=0 the two are algebraically identical -- the anchor is a no-op
    and the peer step never fires -- so those runs stay usable. Anywhere else
    (W<1, k>0, or peers live) the two differ and MUST NOT be mixed."""
    w = float(cfg.get("w_plat", 1.0))
    k = float(cfg.get("innate_lambda", 0.0))
    eps_soc = float(cfg.get("eps", 0.0))
    legacy_equivalent = (w == 1.0 and k == 0.0 and eps_soc == 0.0)
    return legacy_equivalent or cfg.get("population_update") == NESTED_MARKER


def cell_key(cfg):
    """Population-environment cell: everything that must be held fixed so the
    only difference between arms is the post-training operator."""
    return (float(cfg.get("eps", 0.0)), float(cfg.get("eps_ai", 0.0)),
            float(cfg.get("w_plat", 1.0)), float(cfg.get("innate_lambda", 0.0)),
            str(cfg.get("pop_order", "peer_first")),
            str(cfg.get("anchor_mode", "fixed")),
            float(cfg.get("platform_sus_scale", 1.0)))


def intervention_of(cfg):
    """Map a config to a post-training intervention slot, or None.

    Replace-only runs enter one slot PER KL strength ("rep:<beta>"), so the
    whole beta sweep can be drawn; pristine retention (pristine_frac is
    authoritative -- such runs carry data_regime == "accumulate") requires
    weak/no KL for a clean data-anchor contrast."""
    beta = float(cfg.get("kl_beta", 0.0))
    pf = float(cfg.get("pristine_frac", 0.0))
    regime = str(cfg.get("data_regime", "replace"))
    if regime == "replace" and pf == 0.0:
        return f"rep:{beta:g}"
    if pf > 0.0 and beta == 0.0:
        return "pristine"
    return None


def select_runs():
    """Scan data roots, filter by config fields, pick the cell with full
    three-intervention coverage (preferring the highest-feedback gate, then
    the most recent runs), and one run per intervention."""
    candidates = {}   # cell_key -> {slot: [run dict]}
    scanned = failed = 0
    for root in DATA_ROOTS:
        if not os.path.isdir(root):
            continue
        for tag in sorted(os.listdir(root)):
            cfg_path = os.path.join(root, tag, "config.json")
            if not os.path.isfile(cfg_path):
                continue
            scanned += 1
            try:
                cfg = read_json_retry(cfg_path)
            except OSError as exc:
                failed += 1
                print(f"[WARN] unreadable config (counted, not skipped "
                      f"silently): {exc}", file=sys.stderr)
                continue
            if not is_clean_loop(cfg):
                continue
            slot = intervention_of(cfg)
            if slot is None:
                continue
            run = {"dir": os.path.join(root, tag), "tag": tag, "cfg": cfg,
                   "mtime": os.path.getmtime(cfg_path)}
            candidates.setdefault(cell_key(cfg), {}).setdefault(slot, []).append(run)
    if failed:
        raise OSError(f"{failed} config reads failed -- fix reads before "
                      "plotting (silent skips forbidden)")

    def has_three_way(v):
        # matched-coverage criterion for cell choice: replace at weak/no KL,
        # replace at strong KL, and a pristine-retention arm
        return ("rep:0" in v and "pristine" in v
                and any(s.startswith("rep:") and float(s[4:]) >= 0.5 for s in v))
    covered = {k: v for k, v in candidates.items() if has_three_way(v)}
    print(f"[select] scanned {scanned} run dirs; clean-loop cells with any "
          f"coverage: {len(candidates)}; full three-way coverage: {len(covered)}")
    for k, v in sorted(candidates.items()):
        eps, eps_ai, w, lam = k[0], k[1], k[2], k[3]
        print(f"  cell eps={eps:g} eps_ai={eps_ai:g} W={w:g} lam={lam:g}: "
              + ", ".join(f"{s}({len(r)})" for s, r in sorted(v.items())))
    if not covered:
        raise RuntimeError("no cell has all three interventions")

    # Preference: the paper's main peer cell (eps=0.2, W=0.5, lam=0.2) has NO
    # pristine-data runs, so full coverage only exists in the base full-adoption
    # environment. Among covered cells take the widest AI gate (highest
    # feedback), tiebreak on most recent runs.
    def cell_rank(k):
        runs = [r for slot in covered[k].values() for r in slot]
        return (k[1], np.median([r["mtime"] for r in runs]))
    chosen_cell = max(covered, key=cell_rank)

    chosen = {}
    for slot, runs in covered[chosen_cell].items():
        if slot == "pristine":     # canonical mid-level pristine fraction
            tgt = min((abs(float(r["cfg"]["pristine_frac"]) - 0.5), r["tag"])
                      for r in runs)[0]
            runs = [r for r in runs
                    if abs(float(r["cfg"]["pristine_frac"]) - 0.5) == tgt]
        if len(runs) > 1:
            runs = sorted(runs, key=lambda r: r["mtime"])
            print(f"[select] duplicate runs for slot {slot}; keeping most "
                  f"recent of: {[r['tag'] for r in runs]}")
        chosen[slot] = runs[-1]
    return chosen_cell, chosen, candidates


def load_run(run):
    """Load trajectory.json + trajectory.pt; validate shapes and same-index
    alignment between the json summaries and the raw arrays."""
    rows = read_json_retry(os.path.join(run["dir"], "trajectory.json"))
    pt = load_pt_retry(os.path.join(run["dir"], "trajectory.pt"))
    pred, op = pt["pred_raw"].float(), pt["op_raw"].float()
    T = len(rows)
    assert T == 30, f"{run['tag']}: expected 30 rounds, got {T}"
    assert pred.shape == op.shape == (T, 723), \
        f"{run['tag']}: raw shape {tuple(pred.shape)} vs {tuple(op.shape)}"
    n_nan = int(torch.isnan(pred).sum())
    if n_nan:
        print(f"[WARN] {run['tag']}: {n_nan} NaN predictions (nanmean used)")
    # off-by-one guard: json row t must summarize the SAME arrays as index t
    pm = np.array([r["pred_mean"] for r in rows])
    om = np.array([r["op_mean"] for r in rows])
    dp = float(np.max(np.abs(pm - pred.nanmean(1).numpy())))
    do = float(np.max(np.abs(om - op.nanmean(1).numpy())))
    assert dp < 1e-4 and do < 1e-4, \
        f"{run['tag']}: json/raw misalignment pred={dp:.2e} op={do:.2e}"
    twin = pt.get("twin_raw")
    if twin is None or not hasattr(twin, "numel") or twin.numel() == 0:
        twin = None   # base/anchored envs have no evolving no-AI twin saved
    else:
        twin = twin.float()
    run.update(rows=rows, pred=pred, op=op, pm=pm, om=om, twin=twin,
               profiles=pt["profiles"], innate=np.asarray(pt["innate"],
                                                          dtype=np.float64),
               op_std=np.array([r["op_std"] for r in rows]))
    print(f"[load] {run['tag']}: 30 rounds OK, json==raw means "
          f"(max dev pred {dp:.1e}, op {do:.1e}), NaN preds {n_nan}")
    return run


def find_mode_threshold(runs):
    """High-mode boundary from the ROUND-0 deployed prediction distribution
    only (pooled across the three arms): the density valley between the low
    and high modes, restricted to (0.30, 0.58). Falls back to 0.45 if the
    valley is unreliable (no clear mass on both sides)."""
    p0 = torch.cat([r["pred"][0] for r in runs.values()]).numpy()
    p0 = p0[np.isfinite(p0)]
    hist, edges = np.histogram(p0, bins=60, range=(0.0, 1.0))
    smooth = np.convolve(hist, np.ones(3) / 3.0, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    win = (centers > 0.30) & (centers < 0.58)
    valley = float(centers[win][np.argmin(smooth[win])])
    lo_mass = float((p0 < valley).mean())
    hi_mass = float((p0 > valley).mean())
    reliable = 0.02 < lo_mass < 0.98
    thr = valley if reliable else DEFAULT_MODE_THRESHOLD
    print(f"[mode] round-0 pooled preds n={len(p0)}: valley at {valley:.3f} "
          f"(mass below {lo_mass:.3f} / above {hi_mass:.3f}) -> "
          f"{'using valley' if reliable else 'valley unreliable, using 0.45'}; "
          f"threshold = {thr:.3f}")
    for s in SENS_THRESHOLDS:
        parts = []
        for lab, r in runs.items():
            q0 = float((r["pred"][0] > s).float().nanmean())
            qf = float((r["pred"][-1] > s).float().nanmean())
            parts.append(f"{lab}: r0 {q0:.3f} -> r29 {qf:.3f}")
        print(f"[mode] sensitivity thr={s:.2f} pred high-mode occupancy | "
              + " | ".join(parts))
    return thr


def occupancy(arr2d, thr):
    """Per-round fraction of finite values above thr."""
    fin = torch.isfinite(arr2d)
    return ((arr2d > thr) & fin).float().sum(1).div(fin.float().sum(1)).numpy()


def near_loop_consistent(m, tol):
    """Documented criterion: |m_{t+1}-m_t| < tol for each of the final five
    transitions. Returns (passes, max final-5 step)."""
    steps = np.abs(np.diff(m))[-5:]
    return bool(np.all(steps < tol)), float(steps.max())


# distinct annotation offsets per intervention so round labels of curves
# that share an endpoint region never overprint each other
ANN_OFFSETS = [(4, -8), (5, -10), (-13, 4)]


def style_axes(ax, lim, xlab, ylab, head, letter):
    ax.plot(lim, lim, color="#aaaaaa", lw=0.7, zorder=0)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(f"({letter})", loc="left", fontweight="bold", pad=4)
    ax.set_title(head, loc="right", fontsize=7.5, color="#666666", pad=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_traj(ax, x, y, st, i, label=None, round_text=True, star=False):
    """One trajectory: line + markers at rounds 0,5,...,25 and the final
    point, one mid-path direction arrow, optional r0/final round text."""
    ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.0, alpha=0.85,
            zorder=3, label=label)
    idx = [t for t in MARK_ROUNDS if t < len(x)]
    ax.plot(x[idx], y[idx], st["marker"], color=st["color"], ms=2.6,
            mew=0, ls="none", zorder=4)
    ax.plot(x[-1], y[-1], "*" if star else st["marker"], color=st["color"],
            ms=7 if star else 4.0, mec="white", mew=0.4, ls="none", zorder=5)
    if round_text:
        dx, dy = ANN_OFFSETS[i % len(ANN_OFFSETS)]
        # a near-closed trajectory would overprint its "0" and final-round
        # labels; keep only the final label in that case
        span = max(np.ptp(ax.get_xlim()), 1e-9)
        if np.hypot(x[-1] - x[0], y[-1] - y[0]) > 0.04 * span:
            ax.annotate("0", (x[0], y[0]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=6, color=st["color"])
        ax.annotate(str(len(x) - 1), (x[-1], y[-1]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=6, color=st["color"])
    k = max(1, len(x) // 3)      # one sparing direction arrow at ~1/3 of path
    ax.annotate("", xy=(x[k], y[k]), xytext=(x[k - 1], y[k - 1]),
                arrowprops=dict(arrowstyle="-|>", color=st["color"], lw=0.8),
                zorder=4)


def make_return_fig(series, lim, labels, out_stem, tol, ann_series=None,
                    formats=("png", "pdf")):
    """series: {intervention: (m, x_next)}; three panels (a)(b)(c).
    ann_series: labels that get round-number text (default: all)."""
    if ann_series is None:
        ann_series = list(series)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), layout="constrained")
    heads = ["serving response", "post-training response", "return relation"]
    xls = [labels["m"], labels["x"], labels["m"]]
    yls = [labels["x"], labels["m1"], labels["m1"]]
    for ax, h, xl, yl, pl in zip(axes, heads, xls, yls, "abc"):
        style_axes(ax, lim, xl, yl, h, pl)
    passes = {}
    for i, (lab, (m, xn)) in enumerate(series.items()):
        st = STYLE[lab]
        ok, mx = near_loop_consistent(m, tol)
        passes[lab] = (ok, mx)
        ann = lab in ann_series
        ann_i = ann_series.index(lab) if ann else i
        # (a) 30 pairs (m_t, x_{t+1}); (b),(c) 29 transitions ending at the
        # final valid transition (m_29 has no successor model).
        draw_traj(axes[0], m, xn, st, ann_i, label=lab, round_text=ann,
                  star=ok)
        draw_traj(axes[1], xn[:-1], m[1:], st, ann_i, round_text=False,
                  star=ok)
        draw_traj(axes[2], m[:-1], m[1:], st, ann_i, round_text=False,
                  star=ok)
        if ann:
            dx, dy = ANN_OFFSETS[ann_i % len(ANN_OFFSETS)]
            axes[2].annotate(str(len(m) - 1), (m[-2], m[-1]),
                             textcoords="offset points", xytext=(dx, dy),
                             fontsize=6, color=st["color"])
    axes[0].legend(loc=labels.get("legend_loc", "upper left"), frameon=False,
                   fontsize=6, handlelength=2.4, borderaxespad=0.2,
                   labelspacing=0.25)
    # diagonal meaning, placed in the empty off-diagonal corner of (c)
    axes[2].text(0.97, 0.06, "diagonal: unchanged\nafter one full cycle",
                 transform=axes[2].transAxes, fontsize=6, color="#888888",
                 ha="right", va="bottom")
    outs = []
    for ext in formats:
        p = os.path.join(OUT_DIR, f"{out_stem}.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None)
        outs.append(p)
    plt.close(fig)
    return outs, passes


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cell, chosen, _ = select_runs()
    eps, eps_ai, w, lam = cell[0], cell[1], cell[2], cell[3]

    print("\n[MISMATCH NOTE] the paper's main peer cell (eps_soc=0.2, W=0.5, "
          "lambda=0.2) has no pristine-data arm, so the matched three-way "
          "comparison uses the base full-adoption environment instead "
          f"(eps={eps:g}, W={w:g}, lambda={lam:g}: no peer dynamics, accepted "
          "agents adopt the served prediction outright).")
    print(f"\n[select] chosen cell: eps={eps:g} eps_ai={eps_ai:g} W={w:g} "
          f"lambda={lam:g} pop_order={cell[4]} anchor={cell[5]}")
    # full replace-only beta sweep (ascending), then the pristine arm last
    rep_betas = sorted(float(s[4:]) for s in chosen if s.startswith("rep:"))
    pf = float(chosen["pristine"]["cfg"]["pristine_frac"])
    order = [(f"rep:{b:g}", f"replace, $\\beta{{=}}{b:g}$", ("rep", b))
             for b in rep_betas]
    order.append(("pristine", f"pristine {pf * 100:g}%, $\\beta{{=}}0$",
                  ("pristine", None)))
    runs = {}
    for slot, lab, (kind, beta) in order:
        r = chosen[slot]
        c = r["cfg"]
        print(f"  {slot:>9} = {r['tag']}\n            regime={c['data_regime']} "
              f"pristine_frac={c['pristine_frac']:g} kl_beta={c['kl_beta']:g} "
              f"style={c['training_style']} seed={c['seed']} "
              f"rounds={c['n_rounds']} model={c['base_model']}")
        STYLE[lab] = style_of(kind, beta)
        runs[lab] = load_run(r)
    # round-number text only on the three reference arms (beta=0, strongest
    # beta, pristine); the intermediate-beta curves stay unannotated so the
    # shared endpoint clusters remain readable
    ann_series = [f"replace, $\\beta{{=}}{rep_betas[0]:g}$",
                  f"replace, $\\beta{{=}}{rep_betas[-1]:g}$",
                  order[-1][1]]

    # ---------------- version 1: mean state ----------------
    series_mean = {lab: (r["pm"], r["om"]) for lab, r in runs.items()}
    allv = np.concatenate([np.r_[m, x] for m, x in series_mean.values()])
    pad = 0.05 * (allv.max() - allv.min())
    lim = (float(allv.min() - pad), float(allv.max() + pad))
    labels_mean = {
        "m": "$m_t$  (mean served)",
        "x": "$x_{t+1}$  (mean opinion)",
        "m1": "$m_{t+1}$  (next served)",
    }
    outs_mean, pass_mean = make_return_fig(
        series_mean, lim, labels_mean, "empirical_return_maps_mean_qwen",
        FIXED_POINT_TOL_MEAN, ann_series)
    print(f"\n[mean] wrote {outs_mean}")
    for lab, (ok, mx) in pass_mean.items():
        print(f"[mean] {lab}: near-loop-consistent "
              f"(|m_(t+1)-m_t|<{FIXED_POINT_TOL_MEAN} for final 5 transitions): "
              f"{'PASS' if ok else 'FAIL'} (max final-5 step {mx:.4f}) -> "
              f"{'starred endpoint' if ok else 'round-30 endpoint, not a fixed point'}")

    # ---------------- version 2: high-mode occupancy ----------------
    thr = find_mode_threshold(runs)
    series_occ = {lab: (occupancy(r["pred"], thr), occupancy(r["op"], thr))
                  for lab, r in runs.items()}
    allq = np.concatenate([np.r_[q, qx] for q, qx in series_occ.values()])
    padq = 0.08 * (allq.max() - allq.min())
    limq = (float(allq.min() - padq), float(allq.max() + padq))
    labels_occ = {
        "m": "$q^{pred}_t$  (served, high-mode share)",
        "x": "$q^{op}_{t+1}$  (opinions)",
        "m1": "$q^{pred}_{t+1}$  (next served)",
        "legend_loc": "lower right",
    }
    outs_occ, pass_occ = make_return_fig(
        series_occ, limq, labels_occ,
        "empirical_return_maps_occupancy_qwen", FIXED_POINT_TOL_OCC,
        ann_series)
    print(f"\n[occ] wrote {outs_occ}  (threshold {thr:.3f})")
    for lab, (ok, mx) in pass_occ.items():
        print(f"[occ] {lab}: near-loop-consistent (<{FIXED_POINT_TOL_OCC}, "
              f"final 5): {'PASS' if ok else 'FAIL'} (max step {mx:.4f})")

    # ---------------- statefulness diagnostic ----------------
    # Do points with similar m_t map to different m_{t+1} depending on a
    # second state statistic (population std after response)? One panel per
    # intervention; y is the one-cycle change so vertical spread at fixed m_t
    # is the thickness of the relation.
    ncol = 3
    nrow = int(np.ceil(len(runs) / ncol))
    fig, axgrid = plt.subplots(nrow, ncol, figsize=(7.0, 2.1 * nrow),
                               sharey=True, layout="constrained")
    axes = np.atleast_1d(axgrid).ravel()
    for ax in axes[len(runs):]:
        ax.set_visible(False)
    allstd = np.concatenate([r["op_std"] for r in runs.values()])
    norm = matplotlib.colors.Normalize(allstd.min(), allstd.max())
    for ax, (lab, r) in zip(axes, runs.items()):
        m, sd = r["pm"], r["op_std"]
        sc = ax.scatter(m[:-1], m[1:] - m[:-1], c=sd[:-1], cmap="viridis",
                        norm=norm, s=12, lw=0)
        ax.axhline(0, color="#aaaaaa", lw=0.7, zorder=0)
        ax.set_title(lab, fontsize=7.5, color="#555555", pad=4)
        ax.set_xlabel("$m_t$")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # printed diagnostic: within narrow m_t bins, does op_std order m_{t+1}?
        rep = []
        for lo in np.arange(m.min(), m.max(), 0.01):
            sel = (m[:-1] >= lo) & (m[:-1] < lo + 0.01)
            if sel.sum() >= 5:
                cc = np.corrcoef(sd[:-1][sel], m[1:][sel])[0, 1]
                rep.append(f"m_t in [{lo:.3f},{lo + 0.01:.3f}) n={sel.sum()} "
                           f"corr(op_std, m_next)={cc:+.2f}")
        print(f"[state] {lab}: " + ("; ".join(rep) if rep else
              "no m_t bin with >=5 points (trajectory too monotone)"))
    for j in range(0, len(runs), ncol):
        axes[j].set_ylabel("$m_{t+1} - m_t$")
    cb = fig.colorbar(sc, ax=list(axes[:len(runs)]), fraction=0.03, pad=0.02)
    cb.set_label("population std after response", fontsize=7.5)
    p_state = os.path.join(OUT_DIR, "empirical_return_map_statefulness.png")
    fig.savefig(p_state, dpi=300)
    plt.close(fig)
    print(f"[state] wrote {p_state}")

    # ---------------- optional: observational gender-gap version ----------------
    # Per-agent arrays + profile gender labels exist, so the group-gap version
    # is computable. OBSERVATIONAL male-female differences among agents in this
    # run -- NOT the paired gender-counterfactual model effect (that lives in
    # the frozen demo-probe analysis, fig_feature.json claimA: Qwen/Action
    # paired prompt-flip gap ~ +0.12). No matched no-platform or gender-blind
    # runs exist in this cell, so no control curves are drawn.
    pt0 = load_pt_retry(os.path.join(chosen[f"rep:{rep_betas[0]:g}"]["dir"],
                                     "trajectory.pt"))
    genders = pt0["profiles"].get("gender")
    gender_rows = []
    if genders and set(genders) >= {"M", "F"}:
        gm = torch.tensor([g == "M" for g in genders])
        gf = torch.tensor([g == "F" for g in genders])
        innate = pt0["innate"].float()
        d_innate = float(innate[gm].mean() - innate[gf].mean())
        print(f"[gender] n_M={int(gm.sum())} n_F={int(gf.sum())}; innate "
              f"M-F gap = {d_innate:+.4f} (observational)")
        series_g = {}
        for lab, r in runs.items():
            dm = (r["pred"][:, gm].nanmean(1) - r["pred"][:, gf].nanmean(1)).numpy()
            dx = (r["op"][:, gm].nanmean(1) - r["op"][:, gf].nanmean(1)).numpy()
            series_g[lab] = (dm, dx)
        allg = np.concatenate([np.r_[a, b] for a, b in series_g.values()]
                              + [np.array([d_innate])])
        padg = 0.08 * (allg.max() - allg.min())
        limg = (float(allg.min() - padg), float(allg.max() + padg))
        labels_g = {
            "m": "$\\Delta m_t$  (M$-$F served)",
            "x": "$\\Delta x_{t+1}$  (M$-$F opinion)",
            "m1": "$\\Delta m_{t+1}$  (M$-$F next served)",
        }
        outs_g, _ = make_return_fig(series_g, limg, labels_g,
                                    "empirical_return_maps_gender_qwen",
                                    FIXED_POINT_TOL_MEAN, ann_series)
        print(f"[gender] wrote {outs_g} (observational gaps; innate gap "
              f"{d_innate:+.4f} shown only in caption, axes shared)")
        for lab, (dm, dx) in series_g.items():
            for t in range(30):
                gender_rows.append((lab, t, dm[t], dx[t],
                                    dm[t + 1] if t < 29 else ""))
    else:
        print("[gender] no gender labels in profiles -- extension skipped")

    # ---------------- auditable CSV ----------------
    csv_path = os.path.join(OUT_DIR, "empirical_return_maps_points.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["statistic", "run_tag", "intervention", "base_model",
                      "dataset", "ml_target", "seed", "eps", "eps_ai", "w_plat",
                      "innate_lambda", "data_regime", "pristine_frac",
                      "kl_beta", "round_t", "m_t", "x_next", "m_next",
                      "mode_threshold"])
        def cfg_cols(r):
            c = r["cfg"]
            return [r["tag"], c["base_model"], c.get("dataset", "movielens"),
                    c.get("ml_target", "Action"), c["seed"], c.get("eps", 0),
                    c["eps_ai"], c.get("w_plat", 1.0),
                    c.get("innate_lambda", 0.0), c["data_regime"],
                    c.get("pristine_frac", 0.0), c["kl_beta"]]
        for lab, r in runs.items():
            base = cfg_cols(r)
            m, xn = series_mean[lab]
            q, qx = series_occ[lab]
            for t in range(30):
                m1 = f"{m[t + 1]:.6f}" if t < 29 else ""
                wtr.writerow(["mean", base[0], lab] + base[1:]
                             + [t, f"{m[t]:.6f}", f"{xn[t]:.6f}", m1, ""])
                q1 = f"{q[t + 1]:.6f}" if t < 29 else ""
                wtr.writerow(["occupancy_high_mode", base[0], lab] + base[1:]
                             + [t, f"{q[t]:.6f}", f"{qx[t]:.6f}", q1,
                                f"{thr:.3f}"])
        for lab, t, dm, dx, dm1 in gender_rows:
            base = cfg_cols(runs[lab])
            wtr.writerow(["gender_gap_mean_MF", base[0], lab] + base[1:]
                         + [t, f"{dm:.6f}", f"{dx:.6f}",
                            dm1 if dm1 == "" else f"{dm1:.6f}", ""])
    print(f"[csv] wrote {csv_path}")

    # ---------------- captions ----------------
    def verdict(passes, tol):
        parts = []
        for lab, (ok, _) in passes.items():
            if ok:
                parts.append(f"{lab}: passes the near-loop-consistency "
                             f"criterion (all of the final five one-cycle "
                             f"changes < {tol})")
            else:
                parts.append(f"{lab}: round-30 endpoint only (criterion "
                             f"not met)")
        return "; ".join(parts)
    cap = f"""empirical_return_maps_mean_qwen
(a) Serving response: mean served prediction m_t vs mean population opinion
x_(t+1) after the round-t response. (b) Post-training response: x_(t+1) vs the
next deployed model's mean served prediction m_(t+1). (c) Full performative
return relation m_t vs m_(t+1); thin diagonal = unchanged after one full
serve-respond-retrain cycle. Empirical trajectories (rounds connected in time
order, markers every 5 rounds, arrow = direction, labeled endpoints = round 29)
from one matched Qwen2.5-7B / MovieLens-Action loop cell (723 agents, eps_ai=
{eps_ai:g}, full-adoption population: eps_soc={eps:g}, W={w:g}, lambda={lam:g},
seed 0, 30 rounds, fresh LoRA weights each round). Interventions differ only in
the post-training operator: replace-only data across the KL-strength sweep
beta in {{{", ".join(f"{b:g}" for b in rep_betas)}}} (KL to the base model,
blue ramp: darker = stronger), and accumulate with a fixed {pf * 100:g}%
pristine (innate-label) fraction at beta=0. Starred endpoints:
{verdict(pass_mean, FIXED_POINT_TOL_MEAN)}. The population response is
stateful, so these are empirical return relations, not single-valued response
curves. Single seed; no seed-level uncertainty is shown, and no interval is
computed across rounds (rounds are not independent replicates). The paper's
main peer cell has no pristine arm, hence the full-adoption cell here.

empirical_return_maps_occupancy_qwen
Same three maps on the high-mode occupancy statistic q_t = share of agents
above the mode boundary ({thr:.3f}, the density valley of the pooled round-0
served-prediction distribution; sensitivity at 0.40/0.45/0.50 printed by the
script). {verdict(pass_occ, FIXED_POINT_TOL_OCC)}. Single seed.

empirical_return_map_statefulness
One-cycle change m_(t+1)-m_t against m_t, colored by the population standard
deviation after the response, one panel per intervention. Vertical spread at
similar m_t that is ordered by color indicates the return relation is
state-dependent (thick), not a single-valued curve. Single seed; exploratory.

empirical_return_maps_gender_qwen (only if produced)
Observational male-female gaps (Delta = mean over M agents minus mean over F
agents) run through the same three maps. These are group differences among
agents in the deployed loop, NOT the paired gender-counterfactual model effect
(the frozen-probe analysis fig_feature.json claimA reports a ~+0.12 paired
prompt-flip gap for Qwen/Action, separate protocol). No matched no-platform or
gender-blind control runs exist in this cell, so none are shown. Single seed.
"""
    cap_path = os.path.join(OUT_DIR, "empirical_return_maps_captions.txt")
    with open(cap_path, "w") as fh:
        fh.write(cap)
    print(f"[captions] wrote {cap_path}")

    # ---------------- final validation summary ----------------
    print("\n[validate] per-run transition counts: "
          + ", ".join(f"{lab}: 30 (m,x) pairs, 29 full transitions"
                      for lab in runs))
    print("[validate] final plotted point in panels (b)/(c) is the "
          "(x_29 -> m_29) / (m_28 -> m_29) transition -- the last round's "
          "opinions (op_raw[29]) have no successor model and are used only "
          "in panel (a).")


if __name__ == "__main__":
    main()
