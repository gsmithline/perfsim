#!/usr/bin/env python3
"""FIGURE-6 LINE-PLOT CANDIDATE (compact) for the Section-4 corrected-gate
grid ``section4_gate_anch2_fig6``.

READS ONLY the analyzer's outputs -- never a trajectory:
  <analysis-dir>/section4_fig6_source_effect.csv   (required)
  <analysis-dir>/section4_fig6_method_gap.csv      (required; the paired
                                                     method gap G)
  <analysis-dir>/section4_fig6_summary.json         (optional; sign text,
                                                     window rule, gate)
written by analyze_section4_gate.py --wave fig6.

ENCODING
  x       eps_social in {0, .1, .3, 1} at evenly spaced CATEGORICAL
          positions labelled 0 / .1 / .3 / 1 (no numeric axis: the gates
          are levels, not a scale)
  y       T_a = mu_B^eq(A evolving) - mu_B^eq(A fixed), the three-seed
          mean of the per-paired-seed difference (EVOLVING MINUS FIXED;
          positive = an adaptive cohort A left cohort B higher than a
          pinned one did)
  colour  eps_AI in {0, .1, .3, 1}: ONE hue, light -> dark with eps_AI.
          eps_AI = 0 is the twin-derived pure peer-transmission baseline
          (gate closed, no served value enters) and is IDENTICAL for both
          methods -- both series are drawn there, on top of each other
  method  SFT (b0) = solid line + filled circles;
          personal-history ICL (d8) = dashed line + HOLLOW squares
  error   the 95% paired Student-t interval over the three seeds
          (df = 2) as thin, light bars with no caps
  zero    a faint horizontal line at T_a = 0

WHAT IS NOT DONE.  No heatmap, no smoothing, no interpolation, no
synthetic values: a series that is absent or incomplete is simply not
drawn; an UNSETTLED point (the analyzer's pair outcome != equilibrium)
is drawn RINGED, connected to nothing, and named in the printed caption,
so it can never be read as a level.  Line segments join only ADJACENT
settled points of one series.

THIRD OUTPUT -- THE PAIRED METHOD GAP.  G = T_SFT - T_ICL = t_a(b0,
seed) - t_a(d8, seed) per seed, three-seed mean with the df = 2 paired
t-interval (positive G = SFT's source effect exceeds ICL's), drawn
against eps_social (categorical x), one line per eps_AI on the same hue
scale, thin paired-t bars, a faint zero line.  At eps_AI = 0 both methods
are twin-derived and G is IDENTICALLY 0 -- it is drawn: it is the
structural anchor.  This is the direct test of whether SFT and ICL
approach each other as eps_social grows.

FILES (--out-dir; default notes/pofd/section4_fig6/previews; any out-dir
with a ``paper`` path component is REFUSED)
  section4_fig6_candidate.pdf/.png          single panel (the primary
                                            candidate)
  section4_fig6_candidate_2panel.pdf/.png   one panel per method (the
                                            alternative, for a visual
                                            choice)
  section4_fig6_candidate_gap.pdf/.png      G = T_SFT - T_ICL vs
                                            eps_social, one line per
                                            eps_AI
  section4_fig6_candidate_caption.txt       the printed caption block
The figures carry NO title text (no set_title, no suptitle -- the
project convention); the narrative is the caption block.

USAGE
  python plot_section4_fig6.py --analysis-dir RUNS/analysis/section4_gate_anch2_fig6 \\
      [--out-dir notes/pofd/section4_fig6/previews]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import textwrap

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "perfsim-s4fig6-mplcache"))

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_OUT = os.path.join(REPO, "notes", "pofd", "section4_fig6",
                           "previews")

SOURCE_CSV = "section4_fig6_source_effect.csv"
GAP_CSV = "section4_fig6_method_gap.csv"
SUMMARY_JSON = "section4_fig6_summary.json"
STEM = "section4_fig6_candidate"
T_A = "t_a_evolving_minus_fixed"
G = "g_sft_minus_icl"
NA = "NA"

# ------------------------------------------------------------- house ink
INK = "#202328"
GRID_GREY = "#d9dde2"
ZERO_GREY = "#b8bcc2"
# ONE hue for eps_AI (a magnitude): light -> dark, in eps_AI order
EA_RAMP = ["#a3bcdc", "#6f95c6", "#3b68a6", "#173f7a"]
ARM_LABEL = {"b0": "SFT", "d8": "personal-history ICL"}
ARM_ORDER = ["b0", "d8"]


def _f(v):
    """CSV cell -> float or None (NA / blank / non-numeric -> None)."""
    if v is None:
        return None
    v = str(v).strip()
    if v in ("", NA):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _b(v):
    return str(v).strip().lower() == "true"


def refuse_paper_dir(out_dir):
    parts = {p.lower() for p in os.path.abspath(out_dir).split(os.sep)}
    if "paper" in parts:
        print(f"[fig6] REFUSING --out-dir {out_dir!r}: figure candidates "
              f"never go under paper/", file=sys.stderr)
        sys.exit(1)


def cat_label(v):
    """0 -> '0', 0.1 -> '.1', 0.3 -> '.3', 1 -> '1'."""
    s = f"{v:g}"
    return s[1:] if s.startswith("0.") else s


def read_rows(analysis_dir):
    """The analyzer's source-effect rows, typed.  Every row keeps: arm,
    eps_ai, eps_social, status, settled, outcome, T_a mean / ci, and the
    served-cardinality minima (for the caption's quantization note)."""
    path = os.path.join(analysis_dir, SOURCE_CSV)
    if not os.path.exists(path):
        print(f"[fig6] {path} not found -- run analyze_section4_gate.py "
              f"--wave fig6 first", file=sys.stderr)
        sys.exit(1)
    with open(path) as fh:
        raw = list(csv.DictReader(fh))
    if not raw:
        print(f"[fig6] {path} is empty", file=sys.stderr)
        sys.exit(1)
    need = {"arm", "eps_ai", "eps_social", "status", f"{T_A}_mean",
            f"{T_A}_ci_lo", f"{T_A}_ci_hi"}
    miss = need - set(raw[0].keys())
    if miss:
        print(f"[fig6] {path} lacks columns {sorted(miss)} -- is this the "
              f"fig6 CSV (T_a primary)?", file=sys.stderr)
        sys.exit(1)
    rows = []
    for r in raw:
        rows.append({
            "arm": r["arm"],
            "eps_ai": float(r["eps_ai"]),
            "eps_social": float(r["eps_social"]),
            "status": r["status"],
            "settled": _b(r.get("settled", "False")),
            "outcome": r.get("outcome") or "",
            "mean": _f(r.get(f"{T_A}_mean")),
            "lo": _f(r.get(f"{T_A}_ci_lo")),
            "hi": _f(r.get(f"{T_A}_ci_hi")),
            "excl0": _b(r.get(f"{T_A}_ci_excludes_zero", "")),
            "served_min": min([v for v in (
                _f(r.get("served_distinct_fixed_min")),
                _f(r.get("served_distinct_evolving_min"))) if v is not None]
                or [None], key=lambda x: (x is None, x)),
        })
    return rows


def read_gap_rows(analysis_dir):
    """The analyzer's paired-method-gap rows, typed like read_rows."""
    path = os.path.join(analysis_dir, GAP_CSV)
    if not os.path.exists(path):
        print(f"[fig6] {path} not found -- run analyze_section4_gate.py "
              f"--wave fig6 (it writes the method gap beside the source "
              f"effect)", file=sys.stderr)
        sys.exit(1)
    with open(path) as fh:
        raw = list(csv.DictReader(fh))
    need = {"eps_ai", "eps_social", "status", f"{G}_mean", f"{G}_ci_lo",
            f"{G}_ci_hi"}
    miss = need - set(raw[0].keys()) if raw else need
    if miss:
        print(f"[fig6] {path} lacks columns {sorted(miss)}", file=sys.stderr)
        sys.exit(1)
    rows = []
    for r in raw:
        rows.append({
            "arm": "gap",
            "eps_ai": float(r["eps_ai"]),
            "eps_social": float(r["eps_social"]),
            "status": r["status"],
            "settled": _b(r.get("settled", "False")),
            "outcome": r.get("outcome") or "",
            "mean": _f(r.get(f"{G}_mean")),
            "lo": _f(r.get(f"{G}_ci_lo")),
            "hi": _f(r.get(f"{G}_ci_hi")),
            "excl0": _b(r.get(f"{G}_excludes_zero", "")),
            "served_min": None,
        })
    return rows


def read_summary(analysis_dir):
    path = os.path.join(analysis_dir, SUMMARY_JSON)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def classify(rows):
    """Split rows into drawable (complete, finite T_a), of which the
    unsettled are ringed; and the absent / incomplete ones."""
    drawable, absent = [], []
    for r in rows:
        ok = (r["status"] == "complete" and r["mean"] is not None
              and r["lo"] is not None and r["hi"] is not None)
        (drawable if ok else absent).append(r)
    return drawable, absent


def _rc():
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.4,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _style(ax, ess, ylabel):
    ax.axhline(0.0, color=ZERO_GREY, lw=0.6, zorder=0)
    ax.set_xlim(-0.5, len(ess) - 0.5)
    ax.set_xticks(range(len(ess)))
    ax.set_xticklabels([cat_label(e) for e in ess])
    ax.grid(axis="y", color=GRID_GREY, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.4, length=2.4, width=0.6)
    ax.set_xlabel(r"$\varepsilon_{\mathrm{social}}$", fontsize=9.4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.8)


def _series_style(arm, color):
    if arm == "b0":
        return dict(ls="-", marker="o", mfc=color, mec=color)
    if arm == "d8":
        return dict(ls=(0, (4, 2.5)), marker="s", mfc="white", mec=color)
    # the method gap: one series per eps_AI, solid line, filled diamonds
    return dict(ls="-", marker="D", mfc=color, mec=color)


def draw_panel(ax, drawable, ess, eas, arms, colors, dodge=0.075,
               arm_dodge=0.03):
    """Draw the series for the given arms on one axes.  Returns the list
    of (arm, eps_ai, eps_social) points ringed as unsettled."""
    ringed = []
    for i, ea in enumerate(eas):
        col = colors[ea]
        for arm in arms:
            off = (i - (len(eas) - 1) / 2.0) * dodge + (
                -arm_dodge if arm == "b0" else arm_dodge)
            st = _series_style(arm, col)
            pts = {}
            for r in drawable:
                if r["arm"] == arm and r["eps_ai"] == ea:
                    pts[r["eps_social"]] = r
            # segments join ADJACENT settled points only (no interpolation
            # across an absent or unsettled x)
            for j in range(len(ess) - 1):
                a, b = pts.get(ess[j]), pts.get(ess[j + 1])
                if a and b and a["settled"] and b["settled"]:
                    ax.plot([j + off, j + 1 + off], [a["mean"], b["mean"]],
                            color=col, lw=1.25, ls=st["ls"], zorder=3)
            for j, es in enumerate(ess):
                r = pts.get(es)
                if r is None:
                    continue
                x = j + off
                ax.errorbar([x], [r["mean"]],
                            yerr=[[r["mean"] - r["lo"]],
                                  [r["hi"] - r["mean"]]],
                            fmt="none", ecolor=col, elinewidth=0.6,
                            capsize=0, alpha=0.45, zorder=2)
                ax.plot([x], [r["mean"]], ls="none", marker=st["marker"],
                        ms=4.4, mfc=st["mfc"], mec=st["mec"], mew=0.9,
                        alpha=1.0 if r["settled"] else 0.75, zorder=4)
                if not r["settled"]:
                    ax.plot([x], [r["mean"]], ls="none", marker="o",
                            ms=10.5, mfc="none", mec=INK, mew=0.7,
                            zorder=5)
                    ringed.append((arm, ea, es, r["outcome"]))
    return ringed


def legend_handles(eas, colors, arms):
    from matplotlib.lines import Line2D
    hs = []
    for ea in eas:
        lab = (r"$\varepsilon_{\mathrm{AI}}=$" + cat_label(ea)
               + (" (no AI: twin)" if ea == 0 else ""))
        hs.append(Line2D([0], [0], color=colors[ea], lw=2.2, label=lab))
    for arm in arms:
        st = _series_style(arm, INK)
        hs.append(Line2D([0], [0], color=INK, lw=1.2, ls=st["ls"],
                         marker=st["marker"], ms=4.4, mfc=st["mfc"],
                         mec=INK, label=ARM_LABEL[arm]))
    return hs


YLABEL = (r"$T_a=\mu_B^{\mathrm{eq}}(\mathrm{A\ evolving})"
          r"-\mu_B^{\mathrm{eq}}(\mathrm{A\ fixed})$")
YLABEL_GAP = r"$G = T_a(\mathrm{SFT}) - T_a(\mathrm{ICL})$"


def _save(fig, out_dir, stem):
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=320 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.03)
        paths.append(p)
    print(f"[fig6] wrote {stem}.pdf/.png")
    return paths


def figure_single(drawable, ess, eas, colors, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _rc()
    fig, ax = plt.subplots(figsize=(4.3, 3.1))
    ringed = draw_panel(ax, drawable, ess, eas, ARM_ORDER, colors)
    _style(ax, ess, YLABEL)
    ax.legend(handles=legend_handles(eas, colors, ARM_ORDER), frameon=False,
              fontsize=7.2, loc="best", handlelength=2.4, labelspacing=0.35)
    fig.tight_layout()
    paths = _save(fig, out_dir, STEM)
    plt.close(fig)
    return paths, ringed


def figure_two_panel(drawable, ess, eas, colors, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _rc()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    ringed = []
    for ax, arm in zip(axes, ARM_ORDER):
        ringed += draw_panel(ax, drawable, ess, eas, [arm], colors,
                             arm_dodge=0.0)
        _style(ax, ess, YLABEL if arm == ARM_ORDER[0] else None)
        # panel identity as an ANNOTATION above the axes -- the figures
        # carry no title text (no set_title, no suptitle)
        ax.annotate(ARM_LABEL[arm], xy=(0.5, 1.02), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=9.2, color=INK)
    axes[0].legend(handles=legend_handles(eas, colors, []), frameon=False,
                   fontsize=7.2, loc="best", handlelength=2.4,
                   labelspacing=0.35)
    fig.tight_layout()
    paths = _save(fig, out_dir, f"{STEM}_2panel")
    plt.close(fig)
    return paths, ringed


def figure_gap(gap_drawable, ess, eas, colors, out_dir):
    """G = T_SFT - T_ICL vs eps_social, one line per eps_AI (same hue
    scale), paired-t bars, faint zero line, no titles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _rc()
    fig, ax = plt.subplots(figsize=(4.3, 3.1))
    ringed = draw_panel(ax, gap_drawable, ess, eas, ["gap"], colors,
                        arm_dodge=0.0)
    _style(ax, ess, YLABEL_GAP)
    hs = legend_handles(eas, colors, [])
    hs[0].set_label(hs[0].get_label().replace("(no AI: twin)",
                                              "(no AI: twin, G = 0)"))
    ax.legend(handles=hs, frameon=False, fontsize=7.2, loc="best",
              handlelength=2.4, labelspacing=0.35)
    fig.tight_layout()
    paths = _save(fig, out_dir, f"{STEM}_gap")
    plt.close(fig)
    return paths, ringed


def caption_gap(gap_rows, gap_drawable, gap_absent, gap_ringed, ess, eas):
    n_sig = sum(1 for r in gap_drawable if r["excl0"] and r["settled"])
    lines = [
        "",
        f"{STEM}_gap.pdf/.png -- the PAIRED METHOD GAP. y = G = T_a(SFT) -",
        "T_a(ICL), formed PER SEED from the same paired per-seed T_a of the",
        "two methods, then the three-seed mean and the 95% paired Student-t",
        "interval (df = 2). Positive G = SFT's source effect exceeds ICL's;",
        "G falling toward 0 with eps_social is the two methods approaching",
        "each other. One line per eps_AI on the same hue scale; at eps_AI =",
        "0 both methods are twin-derived and G is IDENTICALLY 0 -- drawn as",
        "the structural anchor. Faint line: G = 0. Segments join only",
        "adjacent settled points (both methods' pairs settled at every",
        "seed).",
        f"{len(gap_drawable)} of {len(gap_rows)} (eps_AI, eps_social) gap "
        f"points drawn; {n_sig} settled points have an interval excluding 0.",
    ]
    if gap_ringed:
        seen = {}
        for _, ea, es, oc in gap_ringed:
            seen.setdefault((ea, es), oc)
        lines.append(f"RINGED = UNSETTLED gap points ({len(seen)}): "
                     + "; ".join(f"eps_AI={cat_label(ea)} "
                                 f"eps_social={cat_label(es)} [{oc}]"
                                 for (ea, es), oc in seen.items()))
    else:
        lines.append("No gap point is ringed: every drawn gap point is "
                     "settled.")
    if gap_absent:
        lines.append(f"Gap points NOT DRAWN ({len(gap_absent)}): " + "; ".join(
            f"eps_AI={cat_label(r['eps_ai'])} "
            f"eps_social={cat_label(r['eps_social'])} [{r['status']}]"
            for r in gap_absent))
    return lines


def caption(rows, drawable, absent, ringed, ess, eas, summary, analysis_dir):
    n_drawn = len(drawable)
    n_sig = sum(1 for r in drawable if r["excl0"] and r["settled"])
    # eps_AI = 0 is exempt: the gate is closed there, nothing served enters
    quant = [r for r in rows if r["eps_ai"] != 0.0
             and r["served_min"] is not None and r["served_min"] <= 3]
    lines = [
        f"CAPTION -- {STEM}.pdf/.png, {STEM}_2panel.pdf/.png and "
        f"{STEM}_gap.pdf/.png (the figures carry no title text)",
        "",
        "Figure 6 candidate: the source effect of a non-adapting cohort A",
        "on the responsive majority (cohort B) across the matched gate",
        "grid, from the Section-4 corrected-gate wave "
        "(section4_gate_anch2_fig6).",
        "y = T_a = mu_B^eq(A evolving) - mu_B^eq(A fixed), EVOLVING MINUS",
        "FIXED: a positive value means a fully adaptive cohort A left",
        "cohort B HIGHER than a pinned cohort A did. Each point is the",
        "three-seed mean of the per-paired-seed difference; error bars are",
        "the 95% paired Student-t interval (df = 2). Equilibrium = the",
        "final five end-of-round post-peer rounds of the analysed artifact",
        "(rounds 26-30 of a 30-round run).",
        f"x = eps_social ({', '.join(cat_label(e) for e in ess)}) at "
        f"categorical positions;",
        f"colour = eps_AI ({', '.join(cat_label(e) for e in eas)}), one "
        f"hue light to dark.",
        "eps_AI = 0 is the twin-derived pure peer-transmission baseline",
        "(the strict gate is closed, no served value enters, the method",
        "drops out): it is IDENTICAL for both methods and both series are",
        "drawn there.",
        "SFT = solid line, filled circles; personal-history ICL = dashed",
        "line, hollow squares. Faint line: T_a = 0. Segments join only",
        "adjacent settled points; nothing is smoothed or interpolated.",
        f"{n_drawn} of {len(rows)} (method, eps_AI, eps_social) series "
        f"drawn; {n_sig} settled series have an interval excluding 0.",
    ]
    if ringed:
        seen = {}
        for arm, ea, es, oc in ringed:
            seen.setdefault((arm, ea, es), oc)
        lines.append(
            f"RINGED = UNSETTLED ({len(seen)}; the analyzer's late-window "
            f"drift test failed for a member of the pair, outcome in "
            f"brackets; NOT an equilibrium): " + "; ".join(
                f"{ARM_LABEL[a]} eps_AI={cat_label(ea)} "
                f"eps_social={cat_label(es)} [{oc}]"
                for (a, ea, es), oc in seen.items()))
    else:
        lines.append("No point is ringed: every drawn series is settled.")
    if absent:
        lines.append(
            f"NOT DRAWN ({len(absent)}, absent or incomplete): " + "; ".join(
                f"{ARM_LABEL.get(r['arm'], r['arm'])} "
                f"eps_AI={cat_label(r['eps_ai'])} "
                f"eps_social={cat_label(r['eps_social'])} [{r['status']}]"
                for r in absent))
    if quant:
        lines.append(
            f"SERVED-VALUE QUANTIZATION: {len(quant)} series have a member "
            f"whose late-window served map holds <= 3 distinct values "
            f"(see served_distinct / served_top_share in the analyzer "
            f"CSV); a null T_a there may be quantization, not absence of "
            f"effect.")
    if summary.get("gate_info"):
        lines.append(f"Gate verdict: {summary['gate_info']}.")
    if summary.get("coverage_note"):
        lines.append(f"Coverage: {summary['coverage_note']}.")
    lines += [
        f"Source: {os.path.join(analysis_dir, SOURCE_CSV)} (analyzer "
        f"output only; no trajectory was read).",
        "Exploratory: one wave, three seeds, one model, one dataset.",
    ]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compact Figure-6 line-plot candidate from the fig6 "
                    "analyzer CSV/JSON (never from trajectories).")
    ap.add_argument("--analysis-dir", required=True,
                    help=f"directory holding {SOURCE_CSV} "
                         f"(analyze_section4_gate.py --wave fig6 --out-dir)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help=f"where the PDF/PNG go (default {DEFAULT_OUT}; "
                         f"never under paper/)")
    args = ap.parse_args(argv)

    analysis_dir = os.path.abspath(args.analysis_dir)
    out_dir = os.path.abspath(args.out_dir)
    refuse_paper_dir(out_dir)
    rows = read_rows(analysis_dir)
    gap_rows = read_gap_rows(analysis_dir)
    summary = read_summary(analysis_dir)
    ess = sorted({r["eps_social"] for r in rows})
    eas = sorted({r["eps_ai"] for r in rows})
    if len(eas) > len(EA_RAMP):
        print(f"[fig6] {len(eas)} eps_AI levels but only {len(EA_RAMP)} "
              f"ramp steps", file=sys.stderr)
        sys.exit(1)
    colors = {ea: EA_RAMP[i] for i, ea in enumerate(eas)}
    drawable, absent = classify(rows)
    gap_drawable, gap_absent = classify(gap_rows)
    print(f"[fig6] analysis-dir: {analysis_dir}")
    print(f"[fig6] out-dir     : {out_dir}")
    print(f"[fig6] series      : {len(rows)} rows, {len(drawable)} drawable, "
          f"{len(absent)} absent/incomplete, "
          f"{sum(1 for r in drawable if not r['settled'])} unsettled")
    print(f"[fig6] gap         : {len(gap_rows)} rows, {len(gap_drawable)} "
          f"drawable, {sum(1 for r in gap_drawable if not r['settled'])} "
          f"unsettled")
    if summary.get("t_a_sign"):
        print(f"[fig6] sign        : {summary['t_a_sign']}")
    if summary.get("g_sign"):
        print(f"[fig6] G sign      : {summary['g_sign']}")

    os.makedirs(out_dir, exist_ok=True)
    paths, ringed = figure_single(drawable, ess, eas, colors, out_dir)
    paths2, _ = figure_two_panel(drawable, ess, eas, colors, out_dir)
    paths3, gap_ringed = figure_gap(gap_drawable, ess, eas, colors, out_dir)
    cap = caption(rows, drawable, absent, ringed, ess, eas, summary,
                  analysis_dir)
    # the gap block goes before the trailing source / exploratory lines
    cap = cap[:-2] + caption_gap(gap_rows, gap_drawable, gap_absent,
                                 gap_ringed, ess, eas) + cap[-2:]
    print("\n" + "\n".join(textwrap.fill(l, 78) if len(l) > 78 else l
                           for l in cap))
    cap_path = os.path.join(out_dir, f"{STEM}_caption.txt")
    with open(cap_path, "w") as fh:
        fh.write("\n".join(cap) + "\n")
    print(f"\n[fig6] wrote {STEM}_caption.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
