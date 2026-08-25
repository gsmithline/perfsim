#!/usr/bin/env python3
"""FIGURE-4 ANCHOR TRADE-OFF PREVIEW: one 4 x 5 histogram grid per
(model, es) of the fig4_anchor_tradeoff wave.

READS the analyzer's outputs in --analysis-dir (fig4_anchor_summary.json
and fig4_anchor_cells.csv, written by analyze_fig4_anchor.py) to learn
which artifact each of the 80 nominal cells resolves to (dup cells draw
their SOURCE's arrays), then the artifacts themselves for the raw
populations.

PANELS.  Rows: gamma = INNATE_LAMBDA = 1, .5, .2, 0 (top to bottom).
Columns: beta = W_PLAT = 0, .25, .5, .75, 1 (left to right).  Each panel
is a RAW fixed-width histogram on [0, 1]: np.histogram with bins =
np.linspace(0, 1, 51), counts on y, x and y SHARED across all 20 panels.
Three overlays per panel, drawn light-to-dark so all three stay visible
when they overlap:
  initial population       innate (seed-invariant), light filled step
  entering-model predictions
                           the model's zero-shot vector = pred_raw[0] of
                           the frozen replay (constant across rounds),
                           mid-alpha amber fill with a solid edge
  final post-peer population
                           op_raw[final round] of the source cell, blue
                           fill with a dark edge, on top
--with-frozen adds the lambda = infinity replay's final population as an
UNFILLED step outline (teal, dashed).

NO KDE, NO per-panel mean/SD text, NO title text of any kind (no
set_title, no suptitle -- the project convention): the row and column
labels are compact margin annotations in the axis-label style of
plot_section4_fig6.py, and there is ONE legend for the whole figure.
The narrative goes in the caption block printed to stdout and written
next to the figure.

FILES (--out-dir; default <analysis-dir>/previews; any out-dir with a
``paper`` path component is REFUSED)
  fig4_anchor_{model}_es{tok}.pdf/.png     tok = 0p05 / 0p2
  fig4_anchor_{model}_es{tok}_caption.txt

REFUSES (exit 1) when the analysis summary / CSV is missing and (exit 2)
when any trajectory or frozen artifact the CSV names is absent.

USAGE
  python plot_fig4_anchor.py --analysis-dir notes/pofd/fig4_anchor \\
      --run-root notes/pofd/cluster --frozen-dir notes/pofd/fig4_anchor/frozen
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "perfsim-f4a-mplcache"))

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent


def _load_sibling(name, modname):
    spec = importlib.util.spec_from_file_location(modname, str(HERE / name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


AN = _load_sibling("analyze_fig4_anchor.py", "_analyze_f4a_for_plot")

BINS = np.linspace(0.0, 1.0, 51)
FIGSIZE = (7.2, 5.6)
DPI = 320
STEM = "fig4_anchor"

# ------------------------------------------------------------- house ink
INK = "#202328"
INITIAL = "#7a7f87"
INITIAL_FILL = "#d9dde2"
MODEL = "#d97706"
MODEL_FILL = "#f2c078"
FINAL = "#356fb6"
FINAL_EDGE = "#1f4f8f"
FROZEN_EQ = "#1b8a78"
GRID_GREY = "#e4e7eb"

LABEL_INITIAL = "initial population (innate)"
LABEL_MODEL = "entering-model predictions (zero-shot)"
LABEL_FINAL = "final post-peer population"
LABEL_FROZEN = r"frozen model ($\lambda=\infty$) replay, final"


def refuse_paper_dir(out_dir):
    parts = {p.lower() for p in os.path.abspath(out_dir).split(os.sep)}
    if "paper" in parts:
        print(f"[f4a-plot] REFUSING --out-dir {out_dir!r}: figure "
              f"previews never go under paper/", file=sys.stderr)
        sys.exit(1)


def _rc():
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.6,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def cat_label(v):
    """1 -> '1', 0.5 -> '.5', 0.05 -> '.05', 0 -> '0'."""
    s = f"{float(v):g}"
    return s[1:] if s.startswith("0.") else s


# ------------------------------------------------------------ analysis
def read_analysis(analysis_dir):
    """(summary, rows) from the analyzer's out-dir; refuses (exit 1)
    when either file is missing or the CSV lacks the columns needed."""
    analysis_dir = Path(analysis_dir)
    sp = analysis_dir / AN.SUMMARY_JSON
    cp = analysis_dir / AN.CELLS_CSV
    for p in (sp, cp):
        if not p.exists():
            print(f"[f4a-plot] REFUSING: {p} not found -- run "
                  f"analyze_fig4_anchor.py first", file=sys.stderr)
            sys.exit(1)
    try:
        summary = json.loads(sp.read_text())
    except json.JSONDecodeError as e:
        print(f"[f4a-plot] REFUSING: {sp} unreadable: {e}", file=sys.stderr)
        sys.exit(1)
    with cp.open() as fh:
        rows = list(csv.DictReader(fh))
    need = {"model", "es", "beta", "gamma", "kind", "source_tag", "path",
            "horizon", "frozen_name", "settled", "cyclic", "outcome"}
    miss = need - set(rows[0].keys()) if rows else need
    if miss:
        print(f"[f4a-plot] REFUSING: {cp} lacks columns {sorted(miss)}",
              file=sys.stderr)
        sys.exit(1)
    return summary, rows


def _np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _resolve_path(path_str, source_tag, roots):
    """The CSV's recorded path when it still exists, else the tag under
    the given --run-root(s)."""
    p = Path(path_str) if path_str else None
    if p is not None and p.exists():
        return p
    return AN._find(source_tag, roots)


def load_panels(rows, grid, roots, frozen_dir, with_frozen):
    """{(model, es): {(gamma, beta): panel}} with panel = {innate,
    entering, final, frozen (or None), kind, source_tag, settled, cyclic,
    outcome}.  Trajectories are cached by path (dups share their
    source's), frozen replays by name.  Any absent artifact is a refusal
    (exit 2)."""
    traj, frz, absent = {}, {}, []
    frozen_dir = Path(frozen_dir)

    def _traj(row):
        key = (row["path"], row["source_tag"])
        if key not in traj:
            p = _resolve_path(row["path"], row["source_tag"], roots)
            if p is None:
                absent.append(f"trajectory {row['source_tag']}")
                traj[key] = None
            else:
                d = AN._load(p)
                h = int(row["horizon"])
                op = np.asarray(torch.as_tensor(d["op_raw"]).float().numpy(),
                                dtype=np.float64)
                if op.shape[0] < h:
                    absent.append(f"trajectory {row['source_tag']}: "
                                  f"{op.shape[0]} rounds < horizon {h}")
                    traj[key] = None
                else:
                    traj[key] = {"innate": _np(d["innate"]),
                                 "final": op[h - 1].copy()}
                del d
        return traj[key]

    def _frozen(name):
        if name not in frz:
            p = frozen_dir / name
            if not p.exists():
                absent.append(f"frozen replay {p}")
                frz[name] = None
            else:
                d = AN._load(p)
                pred = np.asarray(
                    torch.as_tensor(d["pred_raw"]).float().numpy(),
                    dtype=np.float64)
                if not np.array_equal(pred, np.broadcast_to(pred[0],
                                                            pred.shape)):
                    absent.append(f"frozen replay {name}: served vector "
                                  f"not constant across rounds")
                    frz[name] = None
                else:
                    op = np.asarray(
                        torch.as_tensor(d["op_raw"]).float().numpy(),
                        dtype=np.float64)
                    frz[name] = {"pred": pred[0].copy(),
                                 "final": op[-1].copy()}
                del d
        return frz[name]

    # the entering-model vector of a model: pred_raw[0] of ANY of its
    # beta > 0 frozen replays (the analyzer verified they agree)
    entering = {}
    for r in rows:
        if float(r["beta"]) > 0.0 and r["model"] not in entering:
            f = _frozen(r["frozen_name"])
            if f is not None:
                entering[r["model"]] = f["pred"]
    for m in grid.models:
        if m not in entering:
            absent.append(f"entering-model vector of {m} (no beta > 0 "
                          f"frozen replay)")

    panels = {}
    for r in rows:
        model, es = r["model"], float(r["es"])
        beta, gamma = float(r["beta"]), float(r["gamma"])
        t = _traj(r)
        f = _frozen(r["frozen_name"]) if with_frozen else None
        if t is None or (with_frozen and f is None) or model not in entering:
            continue
        panels.setdefault((model, es), {})[(gamma, beta)] = {
            "innate": t["innate"], "entering": entering[model],
            "final": t["final"], "frozen": f["final"] if f else None,
            "kind": r["kind"], "source_tag": r["source_tag"],
            "settled": str(r["settled"]).strip().lower() == "true",
            "cyclic": str(r["cyclic"]).strip().lower() == "true",
            "outcome": r["outcome"], "horizon": int(r["horizon"]),
        }
    if absent:
        print(f"[f4a-plot] REFUSING: {len(absent)} artifact(s) absent or "
              f"unusable", file=sys.stderr)
        for a in absent:
            print(f"    {a}", file=sys.stderr)
        sys.exit(2)
    for (model, es), pan in panels.items():
        want = {(g, b) for g in grid.gammas for b in grid.betas}
        miss = sorted(want - set(pan))
        if miss:
            print(f"[f4a-plot] REFUSING: {model} es={es:g} lacks "
                  f"{len(miss)} panel(s) in the analysis CSV: {miss}",
                  file=sys.stderr)
            sys.exit(2)
    return panels


# ------------------------------------------------------------- drawing
def _stairs(ax, values, **kw):
    counts, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=BINS)
    return ax.stairs(counts, BINS, **kw)


def draw_panel(ax, panel, with_frozen):
    """ONE StepPatch per overlay, light -> dark so all stay visible where
    they overlap: innate (light grey fill, grey edge), entering-model
    (amber mid-alpha fill, solid amber edge), final (blue translucent
    fill, dark blue edge) on top, then the frozen outline (unfilled,
    dashed teal) if asked.  Faces are translucent through their RGBA
    facecolor; edges stay opaque.  Returns the artists."""
    from matplotlib.colors import to_rgba
    arts = [
        _stairs(ax, panel["innate"], fill=True,
                facecolor=to_rgba(INITIAL_FILL, 0.95), edgecolor=INITIAL,
                lw=0.55, zorder=1, label=LABEL_INITIAL),
        _stairs(ax, panel["entering"], fill=True,
                facecolor=to_rgba(MODEL_FILL, 0.55), edgecolor=MODEL,
                lw=0.7, zorder=2, label=LABEL_MODEL),
        _stairs(ax, panel["final"], fill=True,
                facecolor=to_rgba(FINAL, 0.45), edgecolor=FINAL_EDGE,
                lw=0.8, zorder=3, label=LABEL_FINAL),
    ]
    if with_frozen and panel["frozen"] is not None:
        arts.append(_stairs(ax, panel["frozen"], fill=False,
                            edgecolor=FROZEN_EQ, lw=0.85, ls=(0, (3, 1.6)),
                            zorder=4, label=LABEL_FROZEN))
    return arts


def _style_axes(ax, bottom, left):
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", ".5", "1"])
    ax.grid(axis="y", color=GRID_GREY, lw=0.45)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.0, length=2.0, width=0.5, pad=1.5)
    ax.tick_params(axis="x", labelbottom=bottom)
    ax.tick_params(axis="y", labelleft=left)


def legend_handles(with_frozen):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hs = [Patch(facecolor=INITIAL_FILL, edgecolor=INITIAL, lw=0.55,
                label=LABEL_INITIAL),
          Patch(facecolor=MODEL_FILL, edgecolor=MODEL, lw=0.7, alpha=0.8,
                label=LABEL_MODEL),
          Patch(facecolor=FINAL, edgecolor=FINAL_EDGE, lw=0.8, alpha=0.6,
                label=LABEL_FINAL)]
    if with_frozen:
        hs.append(Line2D([0], [0], color=FROZEN_EQ, lw=0.9,
                         ls=(0, (3, 1.6)), label=LABEL_FROZEN))
    return hs


def draw_figure(panels, grid, with_frozen, ymax=None):
    """The 4 x 5 grid for one (model, es).  Returns the figure; the
    caller saves and closes it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _rc()
    gammas, betas = list(grid.gammas), list(grid.betas)
    fig, axes = plt.subplots(len(gammas), len(betas), figsize=FIGSIZE,
                             sharex=True, sharey=True, squeeze=False)
    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            ax = axes[i, j]
            draw_panel(ax, panels[(gamma, beta)], with_frozen)
            _style_axes(ax, bottom=(i == len(gammas) - 1), left=(j == 0))
    if ymax is not None:
        axes[0, 0].set_ylim(0.0, float(ymax))
    else:
        lo, hi = axes[0, 0].get_ylim()
        axes[0, 0].set_ylim(0.0, hi)
    left, right, top, bottom = 0.105, 0.99, 0.945, 0.15
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                        wspace=0.09, hspace=0.14)
    # compact margin labels (annotations, never titles)
    # numerals outside math mode: mathtext would set ".25" as ". 25"
    for j, beta in enumerate(betas):
        bb = axes[0, j].get_position()
        fig.text((bb.x0 + bb.x1) / 2.0, top + 0.012,
                 r"$\beta$ = " + f"{beta:g}", ha="center",
                 va="bottom", fontsize=8.4, color=INK)
    for i, gamma in enumerate(gammas):
        bb = axes[i, 0].get_position()
        fig.text(0.012, (bb.y0 + bb.y1) / 2.0,
                 r"$\gamma$ = " + f"{gamma:g}", ha="left",
                 va="center", rotation=90, fontsize=8.4, color=INK)
    fig.supylabel("agents per bin (50 bins on [0, 1])", x=0.043,
                  fontsize=7.8, color=INK)
    fig.supxlabel("opinion", y=bottom - 0.062, fontsize=8.2, color=INK)
    fig.legend(handles=legend_handles(with_frozen), loc="lower center",
               ncol=4 if with_frozen else 3, frameon=False, fontsize=7.0,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.6,
               columnspacing=1.3, handletextpad=0.5)
    return fig


def caption(model, es, panels, grid, with_frozen, summary):
    dups = sorted((g, b) for (g, b), p in panels.items() if p["kind"] == "dup")
    unsettled = sorted((g, b) for (g, b), p in panels.items()
                       if not p["settled"])
    horizons = sorted({p["horizon"] for p in panels.values()})
    lines = [
        f"CAPTION -- {STEM}_{model}_es{grid.num(es)}.pdf/.png (the figure "
        f"carries no title text)",
        "",
        f"Figure 4 anchor trade-off, model {model}, social threshold "
        f"eps_social = {es:g}. Rows: gamma = INNATE_LAMBDA = "
        + ", ".join(cat_label(g) for g in grid.gammas)
        + " (top to bottom). Columns: beta = W_PLAT = "
        + ", ".join(cat_label(b) for b in grid.betas)
        + " (left to right).",
        "Each panel: raw fixed-width histograms of the 723 agents on "
        "[0, 1], 50 bins, counts on y; x and y are shared across all "
        "20 panels. Grey filled: the initial population (innate, "
        "seed-invariant). Amber: the entering model's predictions, the "
        "zero-shot served vector (pred_raw[0] of the frozen replay; "
        "served values are .2f-quantized). Blue, on top: the final "
        "post-peer population (op_raw of the final round of the source "
        "cell; end-of-round state after the single Deffuant sweep).",
    ]
    if with_frozen:
        lines.append("Teal dashed outline: the final population of the "
                     "lambda = infinity replay, the same constant "
                     "served vector pushed through the identical "
                     "operator with no training.")
    lines.append(f"Horizon(s) analysed: {', '.join(str(h) for h in horizons)}"
                 f" rounds; the late window is the final ten.")
    if dups:
        lines.append(f"ALGEBRAIC DUPS ({len(dups)} panels draw their source "
                     f"cell's arrays): " + "; ".join(
                         f"gamma={cat_label(g)} beta={cat_label(b)} <- "
                         f"{panels[(g, b)]['source_tag']}"
                         for g, b in dups))
    if unsettled:
        lines.append(f"UNSETTLED ({len(unsettled)} panels; the analyzer's "
                     f"late-window test failed, NOT an equilibrium): "
                     + "; ".join(
                         f"gamma={cat_label(g)} beta={cat_label(b)} "
                         f"[{panels[(g, b)]['outcome']}"
                         f"{' cyclic' if panels[(g, b)]['cyclic'] else ''}]"
                         for g, b in unsettled))
    else:
        lines.append("Every panel's final population is settled by the "
                     "analyzer's late-window test.")
    if summary.get("git_sha"):
        lines.append(f"Wave git_sha: {', '.join(summary['git_sha'])}.")
    lines.append("Exploratory preview: one seed, one dataset (MovieLens "
                 "Action), one wave.")
    return lines


def _save(fig, out_dir, stem):
    paths = []
    for ext in ("pdf", "png"):
        p = Path(out_dir) / f"{stem}.{ext}"
        fig.savefig(str(p), dpi=DPI if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.03)
        paths.append(str(p))
    return paths


def build_figures(analysis_dir, roots, frozen_dir, grid, with_frozen,
                  ymax=None):
    """Yields (model, es, fig, caption_lines) for every (model, es) of
    the grid, in grid order.  The caller saves and closes."""
    summary, rows = read_analysis(analysis_dir)
    panels = load_panels(rows, grid, roots, frozen_dir, with_frozen)
    for model in grid.models:
        for es in grid.es:
            if (model, es) not in panels:
                print(f"[f4a-plot] REFUSING: no rows for {model} es={es:g} "
                      f"in the analysis CSV", file=sys.stderr)
                sys.exit(2)
            fig = draw_figure(panels[(model, es)], grid, with_frozen, ymax)
            yield (model, es, fig,
                   caption(model, es, panels[(model, es)], grid,
                           with_frozen, summary))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="4 x 5 histogram grids (gamma x beta) per (model, es) "
                    "for the Figure-4 anchor-tradeoff wave")
    ap.add_argument("--analysis-dir", required=True,
                    help="analyze_fig4_anchor.py --out-dir")
    ap.add_argument("--run-root", action="append", default=None)
    ap.add_argument("--frozen-dir", default=str(AN.DEFAULT_FROZEN_DIR))
    ap.add_argument("--out-dir", default=None,
                    help="default <analysis-dir>/previews; never under "
                         "paper/")
    ap.add_argument("--with-frozen", action="store_true",
                    help="add the lambda = infinity replay's final "
                         "population as an unfilled step outline")
    ap.add_argument("--ymax", type=float, default=None,
                    help="clip the shared y-limit (counts) so a consensus "
                         "spike does not flatten the reference histograms; "
                         "default: automatic")
    ap.add_argument("--gen", default=None,
                    help="path of the generator module (default: the "
                         "repo's gen_pofd_sweep.py)")
    args = ap.parse_args(argv)

    analysis_dir = Path(args.analysis_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir \
        else analysis_dir / "previews"
    refuse_paper_dir(str(out_dir))
    roots = [Path(r) for r in (args.run_root or AN.DEFAULT_RUN_ROOTS)]
    grid = AN._grid(gen_path=args.gen)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[f4a-plot] analysis-dir: {analysis_dir}")
    print(f"[f4a-plot] out-dir     : {out_dir}")

    import matplotlib.pyplot as plt
    written = []
    for model, es, fig, cap in build_figures(analysis_dir, roots,
                                             args.frozen_dir, grid,
                                             args.with_frozen, args.ymax):
        stem = f"{STEM}_{model}_es{grid.num(es)}"
        written += _save(fig, out_dir, stem)
        plt.close(fig)
        (out_dir / f"{stem}_caption.txt").write_text("\n".join(cap) + "\n")
        print("\n" + "\n".join(textwrap.fill(l, 78) if len(l) > 78 else l
                               for l in cap))
        print(f"\n[f4a-plot] wrote {stem}.pdf/.png + caption")
    print(f"[f4a-plot] {len(written)} file(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
