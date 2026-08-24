#!/usr/bin/env python3
"""ANALYZER for the FIGURE-4 REPLICATION AND CONVERGENCE wave
(pofdf4r_, key fig4_family_prior_repl30, 18 cells).

CPU only, and safe on a shared login node: OMP/MKL thread counts are
pinned to 1 BEFORE torch is imported, matplotlib runs on Agg, and at
most ONE run's tensors are alive at a time (each cell is reduced to
per-round scalars plus a handful of 723-vectors, then dropped).

GATE FIRST. This script computes numbers; check_fig4_repl.py decides
whether they mean anything.

====================================================================
WHAT IS MEASURED, EXACTLY
====================================================================
op_raw[t] is the END-OF-ROUND POST-PEER population state -- the runner
appends it after the peer sweep, and peers always run last. Every
"population" quantity below is that state, and it is labelled that way
in the CSV headers, the figures and the captions. The served vector for
round t is pred_raw[t-1]; it is a different object and lives in its own
column (served_mean).

Round indexing: round 1 .. 30 are op_raw[0] .. op_raw[29]. The innate
population is t = 0 and is used as a reference distribution, not as a
round.

THE TAIL WINDOW IS A CHOICE, SO IT IS A FLAG. The horizon is 30 rounds
(the user cut it from 100: "100 is too much"), and every window
statistic below is taken over the LAST TEN ROUNDS, 21-30 by default --
the direct analogue of the 81-100 window a 100-round run would use.
--window-start / --window-end move it, and the window actually used is
printed, written into every CSV row and recorded in the JSON, so the
choice is visible rather than buried in a constant.

A. EQUILIBRIUM (the tail window, rounds 21-30). Per checkpoint and
   seed, the mean of
   the per-round population mean and the mean of the per-round
   population SD over the window; then aggregated ACROSS SEEDS with a
   95% Student-t interval. The experimental unit is the SEED, so n = 3
   and df = 2 (t = 4.302652729911275). scipy.stats.t.ppf is used when
   scipy imports and the literal otherwise; which one ran is printed and
   recorded in the JSON as t_crit_source.

B. WASSERSTEIN (W1), with the same seed-level 95% intervals:
     w1_final_vs_innate  final post-peer population vs the innate/initial
                         distribution
     w1_final_vs_twin    final post-peer population vs the MATCHED
                         no-platform twin, twin_raw[-1]. The twin is the
                         counterfactual the runner already simulated on
                         the same seed and the same graph; it is READ,
                         never re-derived. This file contains no
                         dynamics.
     w1_final_vs_frozen  final population vs the FROZEN MODEL
                         distribution, i.e. pred_raw[0] of the frozen
                         control cell pofdfam_{slug}_k0_..._s0 -- exactly
                         the vector plot_sft_family_prior_one_row.py
                         draws as "entering model". Reported only when
                         that cell is under a run root; SKIPPED and named
                         otherwise.

C. CONVERGENCE over the tail window, per checkpoint and seed. The
   window is split in half -- 21-25 against 26-30 at the default -- and
   the two halves are compared:

       half-window drift  D = | mean(y[26..30]) - mean(y[21..25]) |
       fitted trend       T = | OLS slope of y over 21..30 | * 10
       settled  iff  D <= TOL and T <= TOL

   on the per-round mean and, separately, on the per-round SD. TOL
   defaults to 0.002 opinion units (--drift-tol), ~1.5% of the innate
   population SD.

   WHY BOTH TESTS. A fresh LoRA is trained every round, so the
   population carries an irreducible round-to-round jitter and a
   vanishing-step test can never fire. D catches a cell still
   translating whose trend is not linear; T catches a cell that
   oscillates with near-zero net drift while a real trend runs
   underneath.

   HOW STRONG THE WIGGLE PROTECTION IS AT THIS HORIZON, HONESTLY. The
   halves are FIVE rounds each here, not ten. A two-round excursion
   that sits inside one half still cancels out of D; one that STRADDLES
   the halves now moves D by 2/5 of its size instead of 1/5, so at 30
   rounds a straddling wiggle CAN push a genuinely flat cell over the
   tolerance. That is a real loss of discrimination relative to a
   100-round run and it is why the per-round jitter is reported next to
   every verdict (noise_floor = median |y[t] - y[t-1]| over the
   window): when drift is within a small multiple of the noise floor,
   read the verdict as "indistinguishable from flat at this horizon",
   not as a measured trend.

   "Was the horizon enough?" is answered from these verdicts, per
   checkpoint and overall, and is stated in the convergence report --
   together with the standing caveat that a 30-round tail is a MUCH
   WEAKER equilibrium claim than a 100-round one, whatever the
   verdicts say.

D. Tidy per-round CSV: model, seed, round, mean, sd, w1_to_twin,
   w1_to_innate, served_mean -- one row per (cell, round), 30 rounds
   per cell.

E. Figures, NO TITLES (project convention: narrative lives in the
   printed caption blocks, never in the figure). House style inherited
   from experiments/llm/plot_sft_family_prior_one_row.py: Gaussian KDE
   with BW = 0.025 evaluated on np.linspace(0, 1, 401), the
   INK / INITIAL / MODEL / POPULATION palette, the serif rcParams, the
   dashed-initial / dotted-model / solid-population line grammar.
   Panels are identified by in-axes annotations because a set_title is
   a title.

F. A printed caption block per figure. Each one states the TWO ways the
   replication differs from the published figure -- the gate reference
   AND the serving path -- so a shift is never read as evidence about
   the gate alone. See the note below.

====================================================================
DISPLAYED-vs-REPLICATED: TWO DIFFERENCES, NOT ONE
====================================================================
The archived Figure-4 configs carry NO serve_eval_mode key at all. That
field was added on 2026-08-21 with the fix that forces eval() for the
duration of generation; run_pokec_gated_lm.py's own comment at the
field says the fix makes "greedy decoding deterministic and LoRA
dropout off while serving". So the published figure served WITH LoRA
dropout live and this replication does not.

Any displayed-vs-replicated difference is therefore the AI-gate
reference (v1 x0 -> v2 anchored x') AND the serving path, together.
Neither can be attributed the shift on its own from these runs. This
sentence is repeated in both caption blocks and in the JSON verdict
(key displayed_vs_replicated) so it cannot be dropped on the way to a
paper.

PARTIAL COMPLETION IS NEVER AVERAGED OVER SILENTLY. Every missing cell
of the 18 is listed by expected tag; any checkpoint with fewer than
three seeds is labelled PARTIAL in the table, the CSV and the JSON, and
its interval is computed at its own n (df = n-1; n = 1 gets no
interval at all, not a zero-width one).

Outputs (--out-dir, default <run-root>/../fig4_repl_analysis)
  fig4_repl_per_round.csv        30 rows per cell (D)
  fig4_repl_equilibrium.csv      1 row per checkpoint (A + B)
  fig4_repl_convergence.txt      criterion, per-cell verdicts (C)
  fig4_repl_distributions.pdf/.png   equilibrium distributions (E)
  fig4_repl_convergence.pdf/.png     mean/SD over rounds (E)

Usage
  OMP_NUM_THREADS=1 python analyze_fig4_repl.py \\
      --run-root runs/pokec_gated_lm --run-root notes/pofd/cluster \\
      --out-dir notes/pofd/fig4_repl_analysis
  # a different tail window
  OMP_NUM_THREADS=1 python analyze_fig4_repl.py --run-root runs/... \\
      --out-dir ... --window-start 16 --window-end 30
"""
from __future__ import annotations

# Thread pinning FIRST -- before torch (and the BLAS it links) is
# imported, or the pin is a no-op on a shared login node.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("USE_TF", "0")

import argparse
import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "perfsim-fig4-repl"))

import numpy as np
import torch

torch.set_num_threads(1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = (os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals() else os.getcwd())
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

LOG = "[fig4_repl]"


def _load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The gate owns the wave's identity: the tag grammar, the checkpoint
# table, the horizon, the seed set, the equilibrium window and the
# archived Figure-4 cell names. Importing them means the analyzer and
# the gate cannot drift apart about what a cell IS.
CF = _load_sibling("check_fig4_repl", "check_fig4_repl.py")

MODELS = CF.MODELS
MODEL_ORDER = CF.MODEL_ORDER
SEEDS = CF.SEEDS
ROUNDS = CF.ROUNDS
N_AGENTS = CF.N_AGENTS
# DEFAULT tail window (the last 10 rounds of the horizon). Overridable
# per run with --window-start / --window-end; nothing below reads these
# module constants except the argparse defaults.
LATE_LO, LATE_HI = CF.LATE_LO, CF.LATE_HI
DEFAULT_ROOTS = CF.DEFAULT_ROOTS
FIG4_K0 = CF.FIG4_K0
expected_tag = CF.expected_tag

DEFAULT_DRIFT_TOL = 0.002
CONF = 0.95

# THE SENTENCE THAT MUST NOT BE DROPPED. Stated once, printed in both
# caption blocks and written to the JSON verdict, so a shift can never
# be read as evidence about the AI gate alone. The archived Figure-4
# configs carry NO serve_eval_mode key: that field arrived on
# 2026-08-21 with the fix that forces eval() during generation, which
# run_pokec_gated_lm.py describes as making "greedy decoding
# deterministic and LoRA dropout off while serving".
DISPLAYED_VS_REPLICATED = (
    "The replication differs from the published Figure 4 in TWO ways at "
    "once -- the AI-gate reference (v1, measured against the raw x0, "
    "vs v2, measured against the anchored x' = k innate + (1-k) x) and "
    "the serving path (the archived cells record no serve_eval_mode and "
    "so served with LoRA dropout live, while these runs serve in eval() "
    "mode) -- so no shift between the two may be attributed to the gate "
    "reference alone.")

# Display names, matching the published Figure 4 panel labels.
MODEL_LABEL = {"qwen7b": "Qwen 2.5", "qwen3_8b": "Qwen 3",
               "olmo7b": "OLMo 2", "olmo3_7b": "OLMo 3",
               "mistral7b": "Mistral", "ministral8b": "Ministral"}

# ---- house style, inherited from plot_sft_family_prior_one_row.py ----
GRID = np.linspace(0.0, 1.0, 401)
BW = 0.025
INK = "#202328"
INITIAL = "#858a91"
MODEL = "#d97706"
POPULATION = "#356fb6"
# the three seeds share the POPULATION colour and separate by dash
# pattern, so the figure still reads as "one population curve, three
# repeats" rather than as three different quantities.
SEED_DASH = {0: (0, ()), 42: (0, (4.0, 1.8)), 43: (0, (1.4, 1.5))}

RC = {
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.0,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


# ------------------------------------------------------------- numerics
def array(value) -> np.ndarray:
    """plot_sft_family_prior_one_row.array -- tensor or array to a flat
    float64 numpy vector."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1)


def kde(values: np.ndarray) -> np.ndarray:
    """plot_sft_family_prior_one_row.kde, verbatim in behaviour: a
    Gaussian KDE with the house bandwidth on the house grid."""
    values = array(values)
    z = (GRID[:, None] - values[None, :]) / BW
    return np.exp(-0.5 * z * z).sum(axis=1) / (
        values.size * BW * np.sqrt(2.0 * np.pi))


def w1(a, b) -> float:
    """1-Wasserstein between two empirical distributions
    (analyze_kl_direction.w1). Equal sizes here (723 vs 723), so this is
    the mean absolute difference of the sorted samples."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.shape != b.shape:
        n = min(a.size, b.size)
        qs = (np.arange(n) + 0.5) / n
        a = np.quantile(a, qs)
        b = np.quantile(b, qs)
    return float(np.abs(a - b).mean())


T_LITERAL = {1: 12.706204736432095, 2: 4.302652729911275,
             3: 3.182446305284263, 4: 2.7764451051977987}
T_CRIT_SOURCE = None


def t_crit(df, conf=CONF):
    """Two-sided Student-t critical value, and WHICH source produced it.

    scipy is used when it imports; the literal 97.5% quantiles are the
    fallback for the seed counts this wave can produce. The choice is
    recorded in T_CRIT_SOURCE and printed, because "n = 3 => t = 4.303"
    is an arithmetic claim a reader must be able to check.
    """
    global T_CRIT_SOURCE
    if df < 1:
        return float("nan")
    try:
        from scipy import stats
        T_CRIT_SOURCE = "scipy.stats.t.ppf"
        return float(stats.t.ppf(0.5 + conf / 2.0, df))
    except Exception:                                        # noqa: BLE001
        T_CRIT_SOURCE = ("literal 97.5% Student-t quantiles "
                         "(scipy unavailable)")
        return T_LITERAL.get(df, float("nan"))


def seed_agg(values):
    """Aggregate SEED-LEVEL values: (n, mean, sd, se, t, lo, hi).

    The seed is the experimental unit, so n is the number of seeds, not
    the number of agents or rounds. n < 2 returns a mean and nothing
    else -- a single seed has no interval, and a zero-width one would be
    a lie.
    """
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    n = len(vals)
    if n == 0:
        return (0, float("nan"), None, None, None, None, None)
    mean = float(np.mean(vals))
    if n < 2:
        return (n, mean, None, None, None, None, None)
    sd = float(np.std(vals, ddof=1))
    se = sd / np.sqrt(n)
    tc = t_crit(n - 1)
    return (n, mean, sd, se, tc, mean - tc * se, mean + tc * se)


def ols_slope(y):
    y = np.asarray(y, dtype=float)
    if y.size < 2 or not np.isfinite(y).all():
        return float("nan")
    x = np.arange(y.size, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def convergence(series, tol):
    """Half-window convergence test over the tail window.

    The window is split at n // 2, so a 10-round window compares 5
    rounds against 5. Returns a dict: settled, drift, trend, slope,
    range, noise_floor, first_half, second_half. See the module
    docstring for why both the half-window drift and the fitted trend
    have to pass -- and for how much weaker the 5-round halves of a
    30-round run are at rejecting a straddling two-round wiggle.
    """
    y = np.asarray(series, dtype=float)
    n = y.size
    out = {"n": int(n), "settled": False, "drift": float("nan"),
           "trend": float("nan"), "slope": float("nan"),
           "range": float("nan"), "noise_floor": float("nan"),
           "first_half": float("nan"), "second_half": float("nan")}
    if n < 4 or not np.isfinite(y).all():
        return out
    half = n // 2
    out["first_half"] = float(y[:half].mean())
    out["second_half"] = float(y[half:].mean())
    out["drift"] = abs(out["second_half"] - out["first_half"])
    out["slope"] = ols_slope(y)
    out["trend"] = abs(out["slope"]) * n
    out["range"] = float(y.max() - y.min())
    out["noise_floor"] = float(np.median(np.abs(np.diff(y))))
    out["settled"] = bool(out["drift"] <= tol and out["trend"] <= tol)
    return out


# ------------------------------------------------------------- loading
def resolve_tag(roots, tag):
    for root in roots:
        p = Path(root) / tag
        if (p / "trajectory.pt").exists():
            return p
    return None


def load_cell(run_dir):
    """Reduce ONE run to what the analysis needs, then drop its tensors.

    Returns a dict of small arrays (one scalar per round plus a few
    723-vectors) and scalars. Nothing here holds a [T, 723] tensor after
    the function returns, which is what keeps the peak footprint at one
    run.
    """
    d = torch.load(Path(run_dir) / "trajectory.pt", map_location="cpu",
                   weights_only=False)
    cfg = d.get("config") or {}
    cj = Path(run_dir) / "config.json"
    if cj.exists():
        try:
            cfg = json.loads(cj.read_text())
        except Exception:                                    # noqa: BLE001
            pass
    op = d["op_raw"].float().numpy()
    innate = array(d["innate"])
    tw = d.get("twin_raw")
    twin = tw.float().numpy() if (torch.is_tensor(tw) and tw.numel()) else None
    pr = d.get("pred_raw")
    pred = pr.float().numpy() if (torch.is_tensor(pr) and pr.numel()) else None
    n_rounds = op.shape[0]

    out = {
        "run_dir": str(run_dir),
        "config": cfg,
        "n_rounds": int(n_rounds),
        "innate": innate,
        "final_pop": op[-1].copy(),
        "mean": op.mean(axis=1),
        "sd": op.std(axis=1),
        "served_mean": (pred.mean(axis=1) if pred is not None
                        else np.full(n_rounds, np.nan)),
        "final_twin": twin[-1].copy() if twin is not None else None,
        "w1_to_innate": np.array([w1(op[t], innate)
                                  for t in range(n_rounds)]),
        "w1_to_twin": (np.array([w1(op[t], twin[t])
                                 for t in range(n_rounds)])
                       if twin is not None
                       else np.full(n_rounds, np.nan)),
    }
    del d, op, twin, pred
    return out


def load_frozen_served(roots, slug):
    """pred_raw[0] of the frozen control cell -- the "entering model"
    distribution plot_sft_family_prior_one_row.py draws.

    Returns (vector | None, tag, reason).
    """
    tag = FIG4_K0.format(slug=slug)
    p = resolve_tag(roots, tag)
    if p is None:
        return None, tag, (f"frozen control {tag} is not under the run "
                           f"root(s)")
    d = torch.load(p / "trajectory.pt", map_location="cpu",
                   weights_only=False)
    pr = d.get("pred_raw")
    if not torch.is_tensor(pr) or pr.numel() == 0:
        del d
        return None, tag, f"frozen control {tag} carries no pred_raw"
    vec = array(pr[0])
    del d, pr
    return vec, tag, ""


# ----------------------------------------------------------------- CSV
def _w(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.10g}"
    return "" if v is None else str(v)


def write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for r in rows:
            wr.writerow([_w(r.get(c)) for c in cols])
    return path


PER_ROUND_COLS = ["model", "seed", "round", "mean", "sd", "w1_to_twin",
                  "w1_to_innate", "served_mean"]

EQ_COLS = [
    "model", "base_model", "n_seeds", "seeds", "complete", "state_label",
    "late_lo", "late_hi",
    "pop_mean_mean", "pop_mean_sd", "pop_mean_se", "pop_mean_t",
    "pop_mean_ci_lo", "pop_mean_ci_hi",
    "pop_sd_mean", "pop_sd_sd", "pop_sd_se", "pop_sd_t",
    "pop_sd_ci_lo", "pop_sd_ci_hi",
    "w1_innate_mean", "w1_innate_ci_lo", "w1_innate_ci_hi",
    "w1_twin_mean", "w1_twin_ci_lo", "w1_twin_ci_hi",
    "w1_frozen_mean", "w1_frozen_ci_lo", "w1_frozen_ci_hi",
    "frozen_source", "frozen_status",
    "n_settled_mean", "n_settled_sd", "n_settled_both",
    "drift_tol", "t_crit_source",
]


# ------------------------------------------------------------- figures
def figure_distributions(cells, frozen, out_dir, stem, late_lo,
                         late_hi, rounds):
    """Per-checkpoint equilibrium distributions across the three seeds.

    NO TITLES. The panel's checkpoint is named by an in-axes annotation
    and the quantities by the shared legend, exactly as the project
    convention requires; the narrative is in the printed caption block.

    The solid POPULATION curves are the FINAL post-peer round, one per
    seed -- the same object the published Figure 4 draws, so the
    two are directly comparable. Tail-window statistics are in
    fig4_repl_equilibrium.csv rather than smeared into this KDE.
    """
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(MODEL_ORDER),
                                 figsize=(11.4, 2.05), sharex=True,
                                 sharey=True)
        curves = []
        for slug in MODEL_ORDER:
            for seed in SEEDS:
                c = cells.get((slug, seed))
                if c is not None:
                    curves.append(kde(c["final_pop"]))
                    curves.append(kde(c["innate"]))
            if frozen.get(slug) is not None:
                curves.append(kde(frozen[slug]))
        ymax = 1.04 * max((float(c.max()) for c in curves), default=1.0)

        for index, (ax, slug) in enumerate(zip(axes, MODEL_ORDER)):
            present = [s for s in SEEDS if (slug, s) in cells]
            if present:
                ax.plot(GRID, kde(cells[(slug, present[0])]["innate"]),
                        color=INITIAL, lw=1.55, ls=(0, (3.2, 2.0)),
                        zorder=2)
            if frozen.get(slug) is not None:
                ax.plot(GRID, kde(frozen[slug]), color=MODEL, lw=1.75,
                        ls=(0, (1.2, 1.35)), zorder=3)
            for seed in SEEDS:
                c = cells.get((slug, seed))
                if c is None:
                    continue
                ax.plot(GRID, kde(c["final_pop"]), color=POPULATION,
                        lw=1.9, ls=SEED_DASH[seed], zorder=4)
            label = MODEL_LABEL[slug]
            if len(present) < len(SEEDS):
                label += f" ({len(present)}/{len(SEEDS)} seeds)"
            ax.annotate(f"({chr(ord('a') + index)}) {label}",
                        xy=(0.5, 1.015), xycoords="axes fraction",
                        ha="center", va="bottom", fontsize=9.4,
                        fontweight="bold", color=INK, annotation_clip=False)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, ymax)
            ax.set_xticks((0.0, 0.5, 1.0))
            ax.set_xticklabels(("0", ".5", "1"))
            ax.grid(axis="y", color="#d9dde2", linewidth=0.52, alpha=0.72)
            ax.tick_params(axis="both", labelsize=8.6, length=2.2,
                           width=0.6, pad=1.4)
            for tl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
                tl.set_fontweight("bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if index > 0:
                ax.tick_params(labelleft=False)
        axes[0].set_ylabel("density", fontsize=9.4, fontweight="bold",
                           labelpad=2.0)
        fig.text(0.53, 0.035,
                 f"opinion (end-of-round post-peer, t = {rounds})",
                 ha="center", va="center", fontsize=9.4,
                 fontweight="bold")
        handles = [
            Line2D([0], [0], color=INITIAL, lw=1.55, ls=(0, (3.2, 2.0)),
                   label="initial population"),
            Line2D([0], [0], color=MODEL, lw=1.75, ls=(0, (1.2, 1.35)),
                   label="entering model (frozen control)"),
        ] + [Line2D([0], [0], color=POPULATION, lw=1.9, ls=SEED_DASH[s],
                    label=f"population, seed {s}") for s in SEEDS]
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False,
                   prop={"size": 8.8, "weight": "bold"},
                   handlelength=2.8, columnspacing=1.4)
        fig.subplots_adjust(left=0.045, right=0.997, top=0.735,
                            bottom=0.235, wspace=0.14)
        paths = _save(fig, out_dir, stem)
    return paths


def figure_convergence(cells, out_dir, stem, tol, late_lo, late_hi,
                       rounds):
    """Population mean (top row) and SD (bottom row) over rounds, one
    line per seed, with the tail window shaded.

    NO TITLES: checkpoints are named by in-axes annotations, quantities
    by the y-axis labels.
    """
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, len(MODEL_ORDER),
                                 figsize=(11.4, 3.9), sharex=True,
                                 squeeze=False)
        t = np.arange(1, rounds + 1)
        for row, key in enumerate(("mean", "sd")):
            lims = [np.inf, -np.inf]
            for slug in MODEL_ORDER:
                for seed in SEEDS:
                    c = cells.get((slug, seed))
                    if c is None:
                        continue
                    y = c[key]
                    lims[0] = min(lims[0], float(np.nanmin(y)))
                    lims[1] = max(lims[1], float(np.nanmax(y)))
            if not np.isfinite(lims[0]):
                lims = [0.0, 1.0]
            pad = 0.06 * max(lims[1] - lims[0], 1e-4)
            for col, slug in enumerate(MODEL_ORDER):
                ax = axes[row][col]
                ax.axvspan(late_lo, late_hi, color="#e8edf4", lw=0,
                           zorder=1)
                for seed in SEEDS:
                    c = cells.get((slug, seed))
                    if c is None:
                        continue
                    n = min(len(c[key]), rounds)
                    ax.plot(t[:n], c[key][:n], color=POPULATION, lw=1.15,
                            ls=SEED_DASH[seed], zorder=3)
                ax.set_xlim(1, rounds)
                ax.set_ylim(lims[0] - pad, lims[1] + pad)
                ax.grid(axis="y", color="#d9dde2", linewidth=0.5,
                        alpha=0.7)
                ax.tick_params(axis="both", labelsize=8.2, length=2.2,
                               width=0.6, pad=1.4)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if col > 0:
                    ax.tick_params(labelleft=False)
                if row == 0:
                    ax.annotate(MODEL_LABEL[slug], xy=(0.5, 1.02),
                                xycoords="axes fraction", ha="center",
                                va="bottom", fontsize=9.0,
                                fontweight="bold", color=INK,
                                annotation_clip=False)
                else:
                    ax.set_xticks((1, rounds // 2, rounds))
        axes[0][0].set_ylabel("population mean\n(post-peer)", fontsize=8.8,
                              fontweight="bold", labelpad=2.0)
        axes[1][0].set_ylabel("population SD\n(post-peer)", fontsize=8.8,
                              fontweight="bold", labelpad=2.0)
        fig.text(0.53, 0.022, "round", ha="center", va="center",
                 fontsize=9.4, fontweight="bold")
        handles = [Line2D([0], [0], color=POPULATION, lw=1.15,
                          ls=SEED_DASH[s], label=f"seed {s}")
                   for s in SEEDS]
        handles.append(Line2D([0], [0], color="#e8edf4", lw=7,
                              label=f"tail window "
                                    f"{late_lo}-{late_hi}"))
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
                   prop={"size": 8.8, "weight": "bold"}, handlelength=2.8,
                   columnspacing=1.6)
        fig.subplots_adjust(left=0.062, right=0.997, top=0.885,
                            bottom=0.115, wspace=0.16, hspace=0.16)
        paths = _save(fig, out_dir, stem)
    return paths


def _save(fig, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return [pdf, png]


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Analyzer for the Figure-4 replication wave "
                    "(pofdf4r_, 18 cells). CPU only.")
    ap.add_argument("--run-root", dest="run_roots", action="append",
                    default=None,
                    help="run root to scan; repeatable. Default: both "
                         "notes/pofd/cluster and runs/pokec_gated_lm.")
    ap.add_argument("--out-dir", default=None,
                    help="output directory. Default: a runs-adjacent "
                         "<run-root>/../fig4_repl_analysis. NEVER under "
                         "paper/.")
    ap.add_argument("--window-start", type=int, default=LATE_LO,
                    help=f"first round of the tail window, 1-indexed "
                         f"(default {LATE_LO} -- the last 10 rounds of "
                         f"the {ROUNDS}-round horizon)")
    ap.add_argument("--window-end", type=int, default=LATE_HI,
                    help=f"last round of the tail window, inclusive "
                         f"(default {LATE_HI})")
    ap.add_argument("--drift-tol", type=float, default=DEFAULT_DRIFT_TOL,
                    help="convergence tolerance in opinion units "
                         f"(default {DEFAULT_DRIFT_TOL})")
    ap.add_argument("--no-figs", action="store_true",
                    help="skip the figures; CSVs and the report still run")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the machine-readable summary here")
    args = ap.parse_args(argv)

    late_lo, late_hi = int(args.window_start), int(args.window_end)
    if not (1 <= late_lo <= late_hi <= ROUNDS):
        print(f"{LOG} usage error: --window-start/--window-end must "
              f"satisfy 1 <= start <= end <= {ROUNDS}; got "
              f"{late_lo}..{late_hi}", file=sys.stderr)
        return 2
    if late_hi - late_lo + 1 < 4:
        print(f"{LOG} usage error: the tail window {late_lo}-{late_hi} is "
              f"{late_hi - late_lo + 1} round(s). The convergence test "
              f"splits the window in half and fits a slope through it; "
              f"below 4 rounds that is arithmetic, not evidence.",
              file=sys.stderr)
        return 2

    roots = [Path(r) for r in (args.run_roots or DEFAULT_ROOTS)]
    live_roots = [r for r in roots if r.is_dir()]
    if not live_roots:
        print(f"{LOG} usage error: none of {[str(r) for r in roots]} is a "
              f"directory", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else (
        live_roots[0].parent / "fig4_repl_analysis")
    if "paper" in Path(out_dir).resolve().parts:
        print(f"{LOG} usage error: --out-dir {out_dir} is under paper/. "
              f"This tool never writes there.", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    # Resolve the t source ONCE, up front, so it is reported even when no
    # cell is present -- "which quantile did you use" must be answerable
    # from the run, not only from a run that had data.
    t_crit(len(SEEDS) - 1)

    # ---------------- load, one run's tensors at a time ---------------
    cells, missing = {}, []
    for slug in MODEL_ORDER:
        for seed in SEEDS:
            tag = expected_tag(slug, seed)
            p = resolve_tag(live_roots, tag)
            if p is None:
                missing.append((slug, seed, tag))
                continue
            try:
                c = load_cell(p)
            except Exception as exc:                         # noqa: BLE001
                print(f"{LOG} WARNING {tag}: unreadable ({exc}); treated "
                      f"as MISSING")
                missing.append((slug, seed, tag))
                continue
            if c["n_rounds"] < ROUNDS:
                print(f"{LOG} WARNING {tag}: {c['n_rounds']} rounds < "
                      f"{ROUNDS}; treated as MISSING (a truncated run is "
                      f"not this cell)")
                missing.append((slug, seed, tag))
                continue
            c["slug"], c["seed"], c["tag"] = slug, seed, tag
            cells[(slug, seed)] = c

    frozen, frozen_meta = {}, {}
    for slug in MODEL_ORDER:
        vec, tag, why = load_frozen_served(live_roots, slug)
        frozen[slug] = vec
        frozen_meta[slug] = {"tag": tag,
                             "status": "OK" if vec is not None else "SKIPPED",
                             "reason": why}

    partial = bool(missing)
    lo, hi = late_lo - 1, late_hi          # 0-based slice of the window
    win_len = late_hi - late_lo + 1
    half = win_len // 2
    half_a = f"{late_lo}-{late_lo + half - 1}"
    half_b = f"{late_lo + half}-{late_hi}"

    # ---------------- D. per-round CSV --------------------------------
    per_round = []
    for slug in MODEL_ORDER:
        for seed in SEEDS:
            c = cells.get((slug, seed))
            if c is None:
                continue
            for t in range(ROUNDS):
                per_round.append({
                    "model": slug, "seed": seed, "round": t + 1,
                    "mean": float(c["mean"][t]), "sd": float(c["sd"][t]),
                    "w1_to_twin": float(c["w1_to_twin"][t]),
                    "w1_to_innate": float(c["w1_to_innate"][t]),
                    "served_mean": float(c["served_mean"][t]),
                })
    pr_path = write_csv(out_dir / "fig4_repl_per_round.csv",
                        PER_ROUND_COLS, per_round)

    # ---------------- A + B + C ---------------------------------------
    eq_rows, conv_rows, summary = [], [], {}
    for slug in MODEL_ORDER:
        present = [s for s in SEEDS if (slug, s) in cells]
        if not present:
            eq_rows.append({"model": slug, "base_model": MODELS[slug],
                            "n_seeds": 0, "seeds": "", "complete": False,
                            "state_label": "ABSENT",
                            "late_lo": late_lo, "late_hi": late_hi,
                            "frozen_source": frozen_meta[slug]["tag"],
                            "frozen_status": frozen_meta[slug]["status"],
                            "drift_tol": args.drift_tol})
            continue
        means, sds, w1_inn, w1_twin, w1_frz = [], [], [], [], []
        settled_mean, settled_sd = 0, 0
        for seed in present:
            c = cells[(slug, seed)]
            win_mean = c["mean"][lo:hi]
            win_sd = c["sd"][lo:hi]
            means.append(float(win_mean.mean()))
            sds.append(float(win_sd.mean()))
            w1_inn.append(w1(c["final_pop"], c["innate"]))
            w1_twin.append(w1(c["final_pop"], c["final_twin"])
                           if c["final_twin"] is not None else float("nan"))
            if frozen[slug] is not None:
                w1_frz.append(w1(c["final_pop"], frozen[slug]))
            cm = convergence(win_mean, args.drift_tol)
            cs = convergence(win_sd, args.drift_tol)
            settled_mean += int(cm["settled"])
            settled_sd += int(cs["settled"])
            conv_rows.append({"model": slug, "seed": seed,
                              "mean": cm, "sd": cs,
                              "win_mean_first": float(
                                  win_mean[:half].mean()),
                              "win_mean_second": float(
                                  win_mean[half:].mean()),
                              "win_sd_first": float(win_sd[:half].mean()),
                              "win_sd_second": float(
                                  win_sd[half:].mean())})

        n_m, m_m, sd_m, se_m, t_m, lo_m, hi_m = seed_agg(means)
        n_s, m_s, sd_s, se_s, t_s, lo_s, hi_s = seed_agg(sds)
        _, wi_m, _, _, _, wi_lo, wi_hi = seed_agg(w1_inn)
        _, wt_m, _, _, _, wt_lo, wt_hi = seed_agg(w1_twin)
        if w1_frz:
            _, wf_m, _, _, _, wf_lo, wf_hi = seed_agg(w1_frz)
        else:
            wf_m = wf_lo = wf_hi = None
        complete = len(present) == len(SEEDS)
        both = min(settled_mean, settled_sd)
        eq_rows.append({
            "model": slug, "base_model": MODELS[slug],
            "n_seeds": len(present),
            "seeds": "|".join(str(s) for s in present),
            "complete": complete,
            "state_label": (f"equilibrium (rounds {late_lo}-{late_hi})"
                            if both == len(present)
                            else f"late-round state (rounds "
                                 f"{late_lo}-{late_hi})"),
            "late_lo": late_lo, "late_hi": late_hi,
            "pop_mean_mean": m_m, "pop_mean_sd": sd_m, "pop_mean_se": se_m,
            "pop_mean_t": t_m, "pop_mean_ci_lo": lo_m,
            "pop_mean_ci_hi": hi_m,
            "pop_sd_mean": m_s, "pop_sd_sd": sd_s, "pop_sd_se": se_s,
            "pop_sd_t": t_s, "pop_sd_ci_lo": lo_s, "pop_sd_ci_hi": hi_s,
            "w1_innate_mean": wi_m, "w1_innate_ci_lo": wi_lo,
            "w1_innate_ci_hi": wi_hi,
            "w1_twin_mean": wt_m, "w1_twin_ci_lo": wt_lo,
            "w1_twin_ci_hi": wt_hi,
            "w1_frozen_mean": wf_m, "w1_frozen_ci_lo": wf_lo,
            "w1_frozen_ci_hi": wf_hi,
            "frozen_source": frozen_meta[slug]["tag"],
            "frozen_status": frozen_meta[slug]["status"],
            "n_settled_mean": settled_mean, "n_settled_sd": settled_sd,
            "n_settled_both": both,
            "drift_tol": args.drift_tol,
            "t_crit_source": T_CRIT_SOURCE,
        })
        summary[slug] = {"n_seeds": len(present), "complete": complete,
                         "settled_mean": settled_mean,
                         "settled_sd": settled_sd}

    for row in eq_rows:
        row.setdefault("t_crit_source", T_CRIT_SOURCE)
    eq_path = write_csv(out_dir / "fig4_repl_equilibrium.csv", EQ_COLS,
                        eq_rows)

    # ---------------- printed tables ----------------------------------
    print(f"{LOG} run roots: {[str(r) for r in live_roots]}")
    print(f"{LOG} out dir  : {out_dir}")
    print(f"{LOG} cells    : {len(cells)} of "
          f"{len(MODEL_ORDER) * len(SEEDS)} present"
          + ("  [PARTIAL]" if partial else ""))
    if missing:
        print(f"{LOG} MISSING CELLS -- results below are PARTIAL and are "
              f"never averaged over an incomplete seed set silently:")
        for slug, seed, tag in missing:
            print(f"{LOG}   MISSING {slug:<12} s{seed:<3} {tag}")
    for slug in MODEL_ORDER:
        fm = frozen_meta[slug]
        if fm["status"] != "OK":
            print(f"{LOG} frozen control SKIPPED for {slug}: {fm['reason']}")

    print("\n" + "=" * 78)
    print(f"A. TAIL WINDOW, post-peer rounds {late_lo}-{late_hi} of "
          f"{ROUNDS}, aggregated across seeds")
    print(f"   95% Student-t intervals over the SEED as the experimental "
          f"unit (n=3 => df=2, t=4.303). t source: {T_CRIT_SOURCE}")
    print("=" * 78)
    hdr = (f"{'model':<13}{'n':>3}{'popMean':>9}{'ci_lo':>9}{'ci_hi':>9}"
           f"{'popSD':>9}{'ci_lo':>9}{'ci_hi':>9}  {'coverage':<10}")
    print(hdr)
    print("-" * len(hdr))

    def _f(v, w=9, p=4):
        return f"{'-':>{w}}" if v is None or not np.isfinite(
            float(v)) else f"{float(v):>{w}.{p}f}"

    for row in eq_rows:
        cov = ("complete" if row.get("complete") else
               f"PARTIAL {row.get('n_seeds', 0)}/{len(SEEDS)}")
        print(f"{row['model']:<13}{row.get('n_seeds', 0):>3}"
              f"{_f(row.get('pop_mean_mean'))}{_f(row.get('pop_mean_ci_lo'))}"
              f"{_f(row.get('pop_mean_ci_hi'))}{_f(row.get('pop_sd_mean'))}"
              f"{_f(row.get('pop_sd_ci_lo'))}{_f(row.get('pop_sd_ci_hi'))}"
              f"  {cov:<10}")

    print("\n" + "=" * 78)
    print("B. WASSERSTEIN (W1) from the FINAL post-peer population, with "
          "95% t intervals")
    print("   innate = the initial distribution; twin = twin_raw[-1], the "
          "matched")
    print("   no-platform counterfactual the runner already simulated "
          "(read, never")
    print("   re-derived); frozen = pred_raw[0] of "
          "pofdfam_{slug}_k0_..._s0, the same")
    print("   vector plot_sft_family_prior_one_row.py draws as 'entering "
          "model'.")
    print("=" * 78)
    hdr = (f"{'model':<13}{'W1_innate':>10}{'ci_lo':>9}{'ci_hi':>9}"
           f"{'W1_twin':>9}{'ci_lo':>9}{'ci_hi':>9}"
           f"{'W1_frozen':>10}{'ci_lo':>9}{'ci_hi':>9}  frozen")
    print(hdr)
    print("-" * len(hdr))
    for row in eq_rows:
        print(f"{row['model']:<13}{_f(row.get('w1_innate_mean'), 10)}"
              f"{_f(row.get('w1_innate_ci_lo'))}"
              f"{_f(row.get('w1_innate_ci_hi'))}"
              f"{_f(row.get('w1_twin_mean'))}"
              f"{_f(row.get('w1_twin_ci_lo'))}"
              f"{_f(row.get('w1_twin_ci_hi'))}"
              f"{_f(row.get('w1_frozen_mean'), 10)}"
              f"{_f(row.get('w1_frozen_ci_lo'))}"
              f"{_f(row.get('w1_frozen_ci_hi'))}  "
              f"{row.get('frozen_status')}")

    # ---------------- C. convergence report ---------------------------
    lines = []
    lines.append(f"CONVERGENCE OF THE FIGURE-4 REPLICATION WAVE "
                 f"(pofdf4r_, {ROUNDS} rounds)")
    lines.append("=" * 74)
    lines.append("")
    lines.append(f"CRITERION, over post-peer rounds {late_lo}-{late_hi} "
                 f"of {ROUNDS}, applied separately to the")
    lines.append("per-round population MEAN and the per-round population "
                 "SD:")
    lines.append("")
    lines.append(f"    half-window drift  D = |mean(y[{half_b}]) - "
                 f"mean(y[{half_a}])|")
    lines.append(f"    fitted trend       T = |OLS slope over "
                 f"{late_lo}..{late_hi}| * {win_len}")
    lines.append(f"    settled  iff  D <= {args.drift_tol:g}  AND  "
                 f"T <= {args.drift_tol:g}")
    lines.append("")
    lines.append("A fresh LoRA is trained every round, so the population "
                 "carries an")
    lines.append("irreducible round-to-round jitter and a vanishing-step "
                 "test can never")
    lines.append("fire. D catches a cell still translating whose trend is "
                 "not linear; T")
    lines.append("catches one oscillating with near-zero net drift while a "
                 "real trend runs")
    lines.append("underneath.")
    lines.append("")
    lines.append(f"HOW STRONG THIS IS AT A {ROUNDS}-ROUND HORIZON, "
                 f"HONESTLY. The halves here are")
    lines.append(f"{half} rounds each. A two-round excursion inside ONE "
                 f"half still cancels out")
    lines.append(f"of D, but one that STRADDLES the halves moves D by "
                 f"2/{half} of its size, so a")
    lines.append("genuinely flat cell can be pushed over the tolerance by "
                 "a wiggle. Compare")
    lines.append("every drift below against its noise column (median "
                 "|y[t]-y[t-1]| over the")
    lines.append("window): when drift sits within a small multiple of the "
                 "noise floor, read")
    lines.append("'NOT settled' as 'indistinguishable from flat at this "
                 "horizon', not as a")
    lines.append("measured trend.")
    lines.append("")
    lines.append(f"{'model':<13}{'seed':>5}{'quantity':>10}"
                 f"{'first10':>10}{'last10':>10}{'drift':>10}{'trend':>10}"
                 f"{'noise':>10}  verdict")
    lines.append("-" * 92)
    for row in conv_rows:
        for key, label in (("mean", "mean"), ("sd", "sd")):
            c = row[key]
            first = row[f"win_{key}_first"]
            second = row[f"win_{key}_second"]
            lines.append(
                f"{row['model']:<13}{row['seed']:>5}{label:>10}"
                f"{first:>10.5f}{second:>10.5f}{c['drift']:>10.5f}"
                f"{c['trend']:>10.5f}{c['noise_floor']:>10.5f}  "
                f"{'settled' if c['settled'] else 'NOT settled'}")
    lines.append("")
    n_cells = len(conv_rows)
    n_ok = sum(1 for r in conv_rows
               if r["mean"]["settled"] and r["sd"]["settled"])
    lines.append(f"SETTLED ON BOTH MEAN AND SD: {n_ok} of {n_cells} "
                 f"present cells.")
    lines.append("")
    # THE STANDING CAVEAT. Printed FIRST and in every branch, including
    # the all-settled one: the verdicts below are about a 10-round tail
    # of a 30-round run, and no arrangement of them turns that into the
    # evidence a 100-round run would give.
    lines.append("!" * 74)
    lines.append(f"HORIZON CAVEAT -- READ BEFORE THE VERDICT. This wave "
                 f"runs {ROUNDS} ROUNDS, not 100.")
    lines.append(f"Every statement below rests on a {win_len}-round tail "
                 f"({late_lo}-{late_hi}) split into {half}-round halves.")
    lines.append("A 30-round tail is a MUCH WEAKER equilibrium claim than "
                 "a 100-round one:")
    lines.append("it can only say the population is flat over the last "
                 "third of a short run,")
    lines.append("never that it has reached a fixed point. A slow drift "
                 "with a timescale")
    lines.append(f"longer than ~{ROUNDS} rounds is INVISIBLE to this test "
                 f"by construction, and the")
    lines.append("published Figure 4 shares that limit -- it is the same "
                 "horizon. Prefer")
    lines.append("\"flat over the last ten rounds\" to \"converged\" in "
                 "any text this feeds.")
    lines.append("!" * 74)
    lines.append("")
    if n_cells == 0:
        lines.append("WAS THE HORIZON ENOUGH? Unanswerable -- no cell is "
                     "present.")
    elif n_ok == n_cells and not partial:
        lines.append(f"DID EVERY CELL SETTLE WITHIN {ROUNDS} ROUNDS? Yes, "
                     f"on the complete 18-cell")
        lines.append("grid: both the population mean and the population "
                     "SD are flat to within")
        lines.append(f"{args.drift_tol:g} opinion units over rounds "
                     f"{late_lo}-{late_hi} on all three seeds.")
        lines.append("")
        lines.append(f"WAS THE HORIZON ENOUGH? NOT ESTABLISHED BY THAT. "
                     f"Settling on a {win_len}-round")
        lines.append(f"tail of a {ROUNDS}-round run is necessary, not "
                     f"sufficient: it rules out drift")
        lines.append(f"faster than the window and says nothing about "
                     f"drift slower than {ROUNDS} rounds.")
        lines.append("Report this as a flat tail at the published "
                     "horizon, and say so explicitly")
        lines.append("if the text needs the word 'equilibrium'.")
    elif n_ok == n_cells and partial:
        lines.append(f"DID EVERY PRESENT CELL SETTLE WITHIN {ROUNDS} "
                     f"ROUNDS? Yes -- but the grid is")
        lines.append("PARTIAL, so this is not a statement about the wave. "
                     "The missing cells are")
        lines.append("listed above and must be run before any claim is "
                     "made.")
        lines.append("")
        lines.append("WAS THE HORIZON ENOUGH? NOT ESTABLISHED BY THAT, "
                     "on two counts: the grid")
        lines.append(f"is incomplete, and settling on a {win_len}-round "
                     f"tail of a {ROUNDS}-round run is")
        lines.append("necessary rather than sufficient -- it rules out "
                     "drift faster than the")
        lines.append(f"window and says nothing about drift slower than "
                     f"{ROUNDS} rounds.")
    else:
        lines.append(f"DID EVERY CELL SETTLE WITHIN {ROUNDS} ROUNDS? NO. "
                     f"The cells below have not")
        lines.append(f"settled, so their {late_lo}-{late_hi} window is a "
                     f"LATE-ROUND STATE, never an")
        lines.append("equilibrium. Do not call it one in text or in a "
                     "caption. Check each one's")
        lines.append("drift against its noise column before calling it a "
                     "trend -- at this horizon")
        lines.append(f"the {half}-round halves are easily moved by a "
                     f"straddling two-round wiggle.")
        for row in conv_rows:
            if row["mean"]["settled"] and row["sd"]["settled"]:
                continue
            which = []
            if not row["mean"]["settled"]:
                which.append(
                    f"mean (drift {row['mean']['drift']:.5f} vs noise "
                    f"{row['mean']['noise_floor']:.5f})")
            if not row["sd"]["settled"]:
                which.append(
                    f"SD (drift {row['sd']['drift']:.5f} vs noise "
                    f"{row['sd']['noise_floor']:.5f})")
            lines.append(f"    {row['model']} s{row['seed']}: "
                         f"{' and '.join(which)} still moving")
    lines.append("")
    lines.append("DISPLAYED vs REPLICATED. " + DISPLAYED_VS_REPLICATED)
    if partial:
        lines.append("")
        lines.append("PARTIAL GRID. Missing cells:")
        for slug, seed, tag in missing:
            lines.append(f"    MISSING {slug} s{seed}  {tag}")
    conv_path = out_dir / "fig4_repl_convergence.txt"
    conv_path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    # ---------------- E + F. figures and captions ---------------------
    figs = []
    if not args.no_figs and cells:
        figs += figure_distributions(cells, frozen, out_dir,
                                     "fig4_repl_distributions",
                                     late_lo, late_hi, ROUNDS)
        figs += figure_convergence(cells, out_dir,
                                   "fig4_repl_convergence", args.drift_tol,
                                   late_lo, late_hi, ROUNDS)

        n_frz = sum(1 for s in MODEL_ORDER if frozen[s] is not None)
        print("\n" + "=" * 78)
        print("CAPTION -- fig4_repl_distributions (the figure carries NO "
              "title text)")
        print("=" * 78)
        print(
            f"Opinion distributions of the Figure-4 condition at three "
            f"seeds, at the SAME\n{ROUNDS}-round horizon the published "
            f"figure uses. Each panel is one checkpoint;\ncurves are "
            f"Gaussian kernel density estimates (bandwidth {BW}) over "
            f"{N_AGENTS} agents on\nmovielens Action, the same estimator "
            f"and palette as the published Figure 4.\nGrey dashed: the "
            f"initial (innate) population, identical across checkpoints "
            f"and\nseeds because the movielens population and its 10-NN "
            f"graph are a deterministic\nfunction of the dataset. Orange "
            f"dotted: the entering model, pred_raw[0] of the\nfrozen "
            f"control pofdfam_{{slug}}_k0_ea1_w0p5_l0p2_es0p05_s0 "
            f"({n_frz} of {len(MODEL_ORDER)} present).\nBlue: the "
            f"END-OF-ROUND POST-PEER population at round {ROUNDS}, one "
            f"line per seed\n(solid = 0, dashed = 42, dotted = 43). "
            f"Forward-KL SFT at lambda = 1, LoRA r=512\nretrained fresh "
            f"each round, beta (w_plat) = 0.5, innate anchor k = 0.2, AI "
            f"gate\neps = 1 (numeric strict-<), peer gate eps_social = "
            f"0.05, homophily gamma = 0.\nSeed-to-seed spread within a "
            f"panel is the replication error of the published\npanel; "
            f"tail-window statistics for rounds {late_lo}-{late_hi} are in "
            f"fig4_repl_equilibrium.csv.\n{DISPLAYED_VS_REPLICATED}")
        print("\n" + "=" * 78)
        print("CAPTION -- fig4_repl_convergence (the figure carries NO "
              "title text)")
        print("=" * 78)
        print(
            f"Convergence of the population over the {ROUNDS} rounds of "
            f"the wave. Top row: the\nmean of the END-OF-ROUND POST-PEER "
            f"population; bottom row: its standard\ndeviation. One line "
            f"per seed (solid = 0, dashed = 42, dotted = 43); each column "
            f"is\none checkpoint. The shaded band is the tail window, "
            f"rounds {late_lo}-{late_hi}, over which the\nreported "
            f"statistics are taken. A cell counts as settled when the "
            f"half-window\ndrift |mean({half_b}) - mean({half_a})| and the "
            f"fitted trend |slope|*{win_len} are both at\nor below "
            f"{args.drift_tol:g} opinion units, applied separately to the "
            f"mean and the SD;\n{n_ok} of {n_cells} present cells meet it. "
            f"A {ROUNDS}-round horizon makes this a MUCH WEAKER\n"
            f"equilibrium claim than a 100-round one: a flat "
            f"{win_len}-round tail rules out drift\nfaster than the "
            f"window and is blind to drift slower than the run, so read "
            f"it as\n\"flat over the last {win_len} rounds\", not as a "
            f"demonstrated fixed point. Each row\nshares one y-range "
            f"across checkpoints so panel heights are comparable and no\n"
            f"panel silently switches to offset notation.\n"
            f"{DISPLAYED_VS_REPLICATED}")

    # ---------------- JSON --------------------------------------------
    payload = {
        "wave": "fig4_family_prior_repl30",
        "run_roots": [str(r) for r in live_roots],
        "out_dir": str(out_dir),
        "rounds": ROUNDS, "seeds": list(SEEDS),
        "late_window": [late_lo, late_hi],
        "late_window_halves": [half_a, half_b],
        "horizon_caveat": (
            f"{ROUNDS} rounds, not 100. Every window statistic rests on a "
            f"{win_len}-round tail ({late_lo}-{late_hi}) split into "
            f"{half}-round halves. A flat tail here rules out drift "
            f"faster than the window and is blind to drift slower than "
            f"the run; it is not a demonstrated fixed point. The "
            f"published Figure 4 shares this limit -- it is the same "
            f"horizon."),
        "displayed_vs_replicated": DISPLAYED_VS_REPLICATED,
        "drift_tol": args.drift_tol,
        "t_crit_source": T_CRIT_SOURCE,
        "n_cells_present": len(cells),
        "n_cells_total": len(MODEL_ORDER) * len(SEEDS),
        "partial": partial,
        "missing": [{"model": m, "seed": s, "expected_tag": t}
                    for m, s, t in missing],
        "frozen_controls": frozen_meta,
        "equilibrium": eq_rows,
        "convergence": [
            {"model": r["model"], "seed": r["seed"],
             "mean": r["mean"], "sd": r["sd"]} for r in conv_rows],
        "outputs": {"per_round_csv": str(pr_path),
                    "equilibrium_csv": str(eq_path),
                    "convergence_txt": str(conv_path),
                    "figures": [str(p) for p in figs]},
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2,
                                                  default=str))
        print(f"\n{LOG} summary -> {args.json_out}")

    print(f"\n{LOG} wrote {pr_path}")
    print(f"{LOG} wrote {eq_path}")
    print(f"{LOG} wrote {conv_path}")
    for p in figs:
        print(f"{LOG} wrote {p}")
    if partial:
        print(f"{LOG} DONE (PARTIAL: {len(cells)}/"
              f"{len(MODEL_ORDER) * len(SEEDS)} cells). Do not present "
              f"these as the complete three-seed replication.")
    else:
        print(f"{LOG} DONE (complete 18-cell grid)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
