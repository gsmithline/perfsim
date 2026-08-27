#!/usr/bin/env python3
"""SECTION-4 CORRECTED-GATE analyzer -- ONE analyzer for TWO waves:

  --wave section4_gate_anch2       (alias v1)   the original 72-cell wave,
                                                 2026-08-24; DEFAULT, its
                                                 behaviour is unchanged
  --wave section4_gate_anch2_fig6  (alias fig6) the Figure-6 grid, 192
                                                 cells, 2026-08-25

Both grids are READ FROM THE GENERATOR (experiments/condor/gen_pofd_sweep
.py, importlib-loaded the way check_fig3_full_loop.py does it): the
original from S4G_GATES / S4G_ESS / S4G_SEEDS, the Figure-6 grid from
s4g2_cells() -> (arm, cond, ea, es, seed, kind).  Nothing about either grid
is hard-coded here; see "FIGURE-6 MODE" below for what the second wave
adds.  The paragraphs that follow describe the original wave.

WHAT THIS WAVE IS. The published Section-4 experiment -- Mistral-7B,
movielens Action, 723 agents, bottom-20% FIXED source cohort vs a fully
EVOLVING population -- re-run under the CORRECTED AI gate: the acceptance
test is |m - x'| < eps_AI (STRICT) on the ANCHORED opinion x' = k*innate +
(1-k)*x,
i.e. config ``population_update == "nested_ai_anchored_then_social_v2"``
(the tag token ``anch2``).  Everything else is held at the Section-4
surface: W_PLAT = 0.5, INNATE_LAMBDA (k) = 0.2, homophily gamma = 0,
30 rounds, AI_GATE_MODE = threshold.

SCIENTIFIC QUESTION (unchanged from the original Section 4). Does a FIXED
(non-adapting) source cohort change collective outcomes and dispersion
relative to a fully EVOLVING population, and how does that depend on the
AI gate (eps_AI) and the social gate (eps_social)?

GRID -- 72 cells, all conceptually required:
  arms {b0 = ordinary SFT, d8 = frozen personal-history ICL (ICL_DAYS=8)}
  x conditions {fixed = bottom-20% innate clamp (145 lowest-innate agents
    pinned, one-sided stubborn peer operator), evolving = no clamp}
  x eps_ai {0.2, 1.0}  x  eps_social {0.0, 0.2, 1.0}  x  seeds {0, 42, 43}

TAG GRAMMAR (pinned by the generator; parsed here, never guessed):
  pofds4g_mistral7b_{arm}_{fixb20|evoall}_anch2_ea{EA}_w0p5_l0p2_es{ES}_s{SEED}
  arm in {b0, d8};  fixb20 = fixed, evoall = evolving;
  EA in {0p2, 1};   ES in {0, 0p2, 1};   SEED in {0, 42, 43}.

====================================================================
WHAT IS READ, AND WHAT THE NUMBERS MEAN
====================================================================
Artifacts: ``<run_root>/<tag>/trajectory.pt``, loaded with
``torch.load(..., map_location="cpu", weights_only=False)``.

  op_raw   [T, 723]  THE END-OF-ROUND POST-PEER POPULATION STATE.  Peer
                     sweeps run LAST inside a round, so op_raw[t] is the
                     state after the AI blend AND the Deffuant sweep of
                     round t.  No within-round intermediate state is read,
                     emitted, or plotted anywhere in this file, and every
                     column label and caption says "post-peer".
  twin_raw [T, 723]  the MATCHED no-platform process the runner already
                     simulated.  It is USED, never reimplemented.  When a
                     run carries no twin (the runner only forces one at
                     eps_social > 0 or WITH_TWIN=1), the fallback is the
                     INNATE vector broadcast over rounds, which is the
                     no-platform process EXACTLY at k > 0: the twin update
                     h = k*innate + (1-k)*x has innate as a fixed point and
                     the twin starts at innate, so with no AI blend and no
                     peer step it never leaves it.  The fallback is
                     recorded per cell in the ``twin_source`` column
                     (twin_raw | innate_es0) -- inherited verbatim from
                     analyze_sft_icl_reach.twin_of.
  pred_raw [T, 723]  the round's per-agent model prediction.  ``served`` is
                     what actually reached the population: the runner
                     serves clamp(pred + canary, 0, 1) and canary is 0 on
                     this whole surface, so served_mean = mean of
                     clamp(pred_raw[t], 0, 1).  pred_mean_raw (unclamped)
                     is emitted beside it so the clamp is visible.
  innate   [723]     the initial population; bit-identical across every
                     cell of the wave (asserted).
  innate_clamp_mask / _count / _seed   present on clamped (fixed) runs only.

COHORT A / COHORT B.  Cohort A = the 145 lowest-innate agents under the
deterministic (innate value, agent id) ranking -- the same rule
_gated_pop.innate_clamp_mask(mode="bottom") uses, so the cohort is a
function of ``innate`` alone.  In the FIXED condition A is the clamped
source cohort; in the EVOLVING condition A IS AN ANALYSIS MASK ONLY --
those agents evolve like everyone else and the mask exists purely so that
a fixed/evolving pair is masked IDENTICALLY.  Guarantees enforced here:
  * every located cell carries a bit-identical ``innate`` (torch.equal
    against the first cell loaded), and each fixed/evolving pair is
    re-checked pairwise through a sha256 of the innate bytes;
  * on a fixed run the STORED innate_clamp_mask must equal the recomputed
    bottom-145 mask, and op_raw[t][A] must equal innate[A] bit-exactly for
    every round (the clamp really held);
  * an evolving run must carry NO clamp mask.
Any violation is a FATAL structural failure: the fixed/evolving contrast
would not be a contrast, so nothing is written.

====================================================================
DEFINITIONS INHERITED FROM analyze_bottom20_section4_3seed.py
====================================================================
So the corrected-gate numbers are directly comparable to the published
v1-gate ones, the following are taken over VERBATIM (source of truth:
experiments/scripts/cluster_pipelines/analyze_bottom20_section4_3seed.py):

  * LATE WINDOW = ``range(25, 30)`` -- op_raw indices 25..29, the last five
    post-peer states of a 30-round run (rounds 26..30 in the runner's
    1-based round numbering; the per-round CSV carries BOTH indexings).
    VERIFIED against the prior analyzer, which uses exactly
    ``LATE = range(25, 30)`` and indexes ``op[t]`` for t in LATE.
  * mu_b_eq   = mean over LATE of op_raw[t][B].mean()   (also mu_a_eq)
  * sd_b_late = mean over LATE of op_raw[t][B].std()
  * sd_ratio_late = mean over LATE of
        op_raw[t][B].std() / twin_raw[t][B].std(),  rounds with a zero
    twin SD skipped -- SD(B platform) / SD(B matched no-platform twin).
  * three-seed aggregation ``tci3``: mean, sample sd (ddof=1), and the 95%
    two-sided Student-t interval at df = 2, half-width
    T_CRIT * sd / sqrt(3).
  * T_CRIT = 4.302652729911275, the LITERAL df=2 critical value (equal to
    scipy.stats.t.ppf(0.975, 2)).  The literal is used rather than scipy so
    the login-node run has no scipy dependency; the test asserts the two
    agree when scipy is importable.
  * ``excludes(lo, hi, ref)`` -- interval-excludes-a-reference test, and
    the ci_excludes_zero / ci_excludes_one column names.
  * the cohort-A/B masking rule, the innate bit-equality and stored-mask
    assertions, and the ``gpu_arch`` provenance reader (config.json
    hardware.gpu_name -> H100 / A100 / ...).
  * the d8 / eps_social = 0 STRUCTURAL-NULL probe with its hardware-aware
    tolerances (NULL_TOL = 1e-9 within one GPU architecture, NULL_TOL_XHW
    = 5e-3 across architectures), because it applies unchanged here: with
    frozen weights, own-history prompts and no peer step, no cohort-A
    opinion can enter a cohort-B prompt, so the source effect on B must be
    zero by construction.  The AI-gate correction creates no new path.
  * drift / half-window convention: ``half = n // 2``; first half = the
    first floor(n/2) window rounds, second half = the rest (n = 5 -> 2 vs
    3), inherited from analyze_section3.convergence.

DEFINITIONS ADDED HERE (not in the prior analyzer)
  * SIGN.  The headline source effect in this file is
    ``delta_mu_b = mu_b_eq(fixed) - mu_b_eq(evolving)``  (fixed minus
    evolving, as this wave's spec asks).  The prior analyzer's T_a is the
    OPPOSITE sign (evolving minus fixed), so every source-effect row also
    carries ``t_a_evolving_minus_fixed_*`` = -delta, and the two are
    written side by side rather than silently reconciled.
  * population-mean source effect ``delta_mu_pop`` -- reported but
    MECHANICALLY CONFOUNDED: in the fixed condition 145 of 723 agents are
    pinned at innate by construction, so a population-mean difference
    mixes the clamp's arithmetic with any real spillover.  Cohort B is the
    honest collective-outcome channel and is the headline.
  * DISPERSION vs the OTHER CONDITION (this wave's spec): population SD
    and cohort-B SD, fixed vs evolving, with the fixed/evolving SD RATIO
    computed PER SEED and then aggregated (the paired ratio, not a ratio
    of means).  The inherited twin-referenced ``sd_ratio_late`` is
    reported per condition alongside it.
  * per-round tidy CSV, W1(population, twin) per round -- W1 between two
    equal-size empirical populations,
    ``(sort(a) - sort(b)).abs().mean()``, the house definition
    (analyze_sft_icl_reach.w1 / analyze_fig2_provider.w1).
  * DRIFT / ROBUSTNESS FLAG per series: the statistic recomputed on the
    first half and the second half of the late window;
    ``drift = second half - first half`` per seed, then the three-seed
    mean.  Flagged when |drift| > --drift-tol (default 0.002 opinion
    units, the analyze_section3 tolerance on the same population scale)
    or when |drift| exceeds the three-seed 95% CI half-width.  A
    two-round wiggle can then never be read as a trend.

PARTIAL COVERAGE.  Every one of the 72 cells is required.  Missing cells
are listed by exact tag, coverage is written to its own CSV, and a series
is aggregated ONLY over seeds where BOTH conditions are present AND all
three seeds are paired.  An incomplete series is emitted with
status=incomplete and NA aggregates -- it is never averaged over a short
seed set, and it is never plotted.  The figures and the printed report
carry a PARTIAL banner whenever anything is missing.

EXIT CODES
  0  complete 72/72, structural checks passed
  1  FATAL structural failure (innate mismatch, clamp mask mismatch,
     wrong operator marker, config disagrees with the tag, too few
     rounds) -- nothing is written
  2  outputs written but coverage is PARTIAL
  3  outputs written and complete, but the d8/eps_social=0 structural
     null was violated: the results are SUSPECT (a marker file
     SUSPECT_NULL_VIOLATION.txt is written next to them)

OUTPUTS (--out-dir; defaults to a runs-ADJACENT analysis directory, and
the tool REFUSES any out-dir with a ``paper`` path component)
  section4_gate_per_round.csv      one row per (arm, cond, ea, es, seed, round)
  section4_gate_cells.csv          one row per cell: late-window scalars
  section4_gate_source_effect.csv  per (arm, ea, es): fixed - evolving
  section4_gate_dispersion.csv     per (arm, ea, es): SD comparison + ratio
  section4_gate_null_probe.csv     the d8/es=0 structural-null verdicts
  section4_gate_coverage.csv       one row per required cell: present / missing
  section4_gate_source_effect.pdf/.png    NO TITLES
  section4_gate_dispersion.pdf/.png       NO TITLES
  section4_gate_captions.txt       the printed caption blocks

PERFORMANCE / ETIQUETTE. Runs on the cluster LOGIN NODE: OMP_NUM_THREADS
and MKL_NUM_THREADS are pinned to 1 BEFORE torch is imported,
torch.set_num_threads(1) after, matplotlib is forced to "Agg", and at most
ONE run's tensors are ever resident -- each trajectory is reduced to
scalars and dropped before the next is opened.

====================================================================
FIGURE-6 MODE  (--wave section4_gate_anch2_fig6, alias fig6)
====================================================================
GRID -- 192 cells from gen_pofd_sweep.s4g2_cells():
  arms {b0, d8} x conds {fixed, evolving} x eps_ai {0, .1, .3, 1}
  x eps_social {0, .1, .3, 1} x seeds {0, 42, 43}, each cell tagged
  kind in {gpu, witness, twin}: 144 gpu + 4 witness + 44 twin-derived
  (the counts are the generator's S4G2_N_GPU / _N_WITNESS / _N_TWIN and
  are asserted at import, never typed here).
  Tags come from gen_pofd_sweep.s4g_tag(arm, cond, gate, es, seed,
  prefix='pofds4g', rounds=None); rounds=60/100 appends _r60/_r100 (the
  horizon-extension artifacts).  For every cell with a run the LONGEST
  available horizon is analysed: _r100 > _r60 > base tag.

eps_AI = 0 IS THE TWIN.  gp.ai_gate is a STRICT inequality |m - x'| <
eps_AI, so at eps_AI = 0 the gate is closed for every agent in every
round: the served vector never enters, the METHOD drops out, and the
population IS the matched no-AI twin the runner already simulated as
twin_raw.  Hence
  * 'twin' cells have NO run dir.  Their population is twin_raw of ANY
    base-horizon run at the same (cond, eps_social, seed); every base run
    at that (cond, es, seed) must carry a bit-identical twin_raw (sha256,
    HARD FAIL otherwise), and the same cohort-B late-window statistics
    are computed on it.  b0 and d8 at eps_AI = 0 therefore share ONE
    value (method collapse) -- by construction, and asserted.
  * 'witness' cells (generator S4G2_EA0_WITNESS: b0 and d8, FIXED and
    EVOLVING, es = .3, seed 0, actually run at eps_AI = 0) are analysed
    from op_raw normally -- a fixed witness through the full clamp checks
    (stored mask == bottom cohort, cohort A pinned every round) -- and
    HARD FAIL unless op_raw is bit-identical to twin_raw (torch.equal).
  * a run that exists for a 'twin' cell is treated as a witness (same
    identity check) and noted.

THE FIGURE-6 QUANTITY.  T_a = mu_B^eq(A evolving) - mu_B^eq(A fixed),
column ``t_a_evolving_minus_fixed_*`` -- EVOLVING MINUS FIXED, the
published Section-4 analyzer's sign; a POSITIVE T_a means a fully
adaptive cohort A left the responsive majority HIGHER than a pinned
cohort A did.  In fig6 mode T_a is the PRIMARY column everywhere (CSV,
printed table, JSON) and the inherited ``delta_mu_b`` (fixed minus
evolving, = -T_a) is carried beside it.  It is formed PER PAIRED SEED
(fixed and evolving at the same seed), then the three-seed mean and the
95% PAIRED Student-t interval (n = 3, df = 2, T_CRIT_DF2) over the
per-seed differences.  Equilibrium = the FINAL FIVE post-peer rounds of
the analysed artifact: op_raw indices n-5..n-1, i.e. LATE = range(25, 30)
= rounds 26-30 (1-indexed) for the 30-round base tag (VERIFIED at import:
late_window(30) == list(LATE) == [25, 26, 27, 28, 29]), 56-60 for _r60,
96-100 for _r100.

SETTLED / UNSETTLED -- three tests per CELL, ALL must hold (tol =
--drift-tol, default 0.002 opinion units):
  (a) final-5 half-split (2 vs 3, LATE_H1 / LATE_H2):
      |mean_B(second half) - mean_B(first half)| <= tol   (``mu_b_drift``)
  (b) final-10 half-split: |mean_B(final five) - mean_B(the five rounds
      before them)| <= tol, i.e. rounds 26-30 vs 21-25 of a 30-round run
      (``late10_drift``) -- a slow trend the final five cannot see;
  (c) the RANGE (max - min) of the five late-window round means of
      cohort B <= 2*tol (``late5_range``) -- a short cycle can average
      out of (a) and (b) but cannot hide its amplitude.
``settled_a/b/c`` are written per cell and ``settled`` = a and b and c.
CYCLIC: an unsettled cell whose last 10 consecutive round-mean
differences alternate in sign on >= 70% of the 9 steps between them
(``cycle_alternation``) is classified ``cyclic`` -- a long-run / cyclic
outcome, NEVER an equilibrium.  A PAIR is unsettled when EITHER member
is; its outcome is 'cyclic' if either member cycles, else 'extend_to_60'
(base horizon analysed) / 'extend_to_100' (a _r60 analysed).
``section4_fig6_extension_request.json`` is written into --out-dir with
BOTH members of every unsettled pair as {arm, cond, eps_ai, eps_social,
seed, rounds, reason}; the reason names which of (a)/(b)/(c) each member
failed and starts with "cyclic:" for a cyclic pair (the generator's
s4g2_ext_requests() requires matched pairs).  A pair whose member is
twin-derived has no GPU run to extend: it is listed under
``twin_derived_unsettled`` (outside "cells"); a pair still unsettled at
100 rounds is 'unsettled_at_100' and listed under ``not_extendable``.

PAIRED METHOD GAP.  For every (eps_AI, eps_social),
  G = T_SFT - T_ICL = t_a(b0, seed) - t_a(d8, seed)   PER SEED (paired on
seed, from the same paired per-seed T_a), then the three-seed mean and
the df=2 paired t-interval; columns ``g_sft_minus_icl_s{seed}``,
``_mean``, ``_sd``, ``_ci_lo``, ``_ci_hi``, ``_excludes_zero`` and its own
half-window drift, in ``section4_fig6_method_gap.csv``, the JSON block
``method_gap`` and a printed table.
SIGN: positive G = SFT's source effect exceeds ICL's.
At eps_AI = 0 both methods are twin-derived, so G is IDENTICALLY 0
(asserted) -- the structural anchor of the G plot.

SERVED-VALUE CARDINALITY.  For every cell with a run, over the late
window: ``served_distinct`` = number of distinct values in pred_raw
(pooled over the window's rounds and agents, finite values), and
``served_top_share`` = the fraction equal to the single most common
value.  Twin-derived cells carry 'n/a (gate closed)'.  These are printed
NEXT TO T_a so a quantized served map (Qwen's binary 0.25/0.65 is the
cautionary tale) can never masquerade as a null effect.

HARD FAIL (exit 1, NOTHING written) in fig6 mode on: any required cell
absent (a gpu/witness run missing at every horizon, or a twin cell with
no base run to derive from), --gate-json absent or not reporting a
passing verdict (--allow-ungated is REJECTED in fig6 mode), an unpaired
seed / a horizon mismatch inside a pair, twin disagreement, a witness
failing op_raw == twin_raw, a run without twin_raw, or any inherited
structural violation.  Partial output is NOT allowed in fig6 mode.

FIG6 EXIT CODES
  0  complete and every pair settled
  1  HARD FAIL (nothing written)
  2  outputs written, but >= 1 pair unsettled: the extension request
     carries the cells to extend
  3  outputs written, but the d8/eps_social=0 structural null failed

FIG6 OUTPUTS (--out-dir, stem section4_fig6_)
  section4_fig6_per_round.csv / _cells.csv / _source_effect.csv /
  _method_gap.csv / _dispersion.csv / _null_probe.csv / _coverage.csv /
  _captions.txt
  section4_fig6_summary.json          always written
  section4_fig6_extension_request.json always written (cells may be [])
  section4_fig6_source_effect.pdf/.png, _dispersion.pdf/.png  NO TITLES
  (the compact Figure-6 candidate is drawn by plot_section4_fig6.py from
  the CSV/JSON above, never from trajectories)

USAGE
  OMP_NUM_THREADS=1 python analyze_section4_gate.py \\
      --run-root /home/gsmithline/perfsim/runs/pokec_gated_lm \\
      --out-dir  /home/gsmithline/perfsim/runs/analysis/section4_gate_anch2 \\
      [--no-figs] [--json OUT.json] [--drift-tol 0.002]
  OMP_NUM_THREADS=1 python analyze_section4_gate.py --wave fig6 \\
      --run-root /home/gsmithline/perfsim/runs/pokec_gated_lm \\
      --gate-json /home/gsmithline/perfsim/runs/analysis/s4g_fig6_gate.json \\
      --out-dir  /home/gsmithline/perfsim/runs/analysis/section4_gate_anch2_fig6
"""
from __future__ import annotations

# Login-node etiquette: torch reads the BLAS/OpenMP thread limits at IMPORT
# time, so they are pinned here, above every other import.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("USE_TF", "0")

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
import tempfile
import textwrap

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "perfsim-s4gate-mplcache"))

import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# ---------------------------------------------------------------- the grid
# BOTH grids are read from the generator so this file can never disagree
# with the cells that were submitted.  The generator is importlib-loaded
# from its path (check_fig3_full_loop._load_gen does the same); its main()
# is guarded, so the load has no side effects.
GEN_PATH = os.path.join(REPO, "experiments", "condor", "gen_pofd_sweep.py")


def _load_gen():
    spec = importlib.util.spec_from_file_location("_gen_s4gate", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gen_s4gate"] = mod
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen()

KEY = GEN.S4G_KEY                       # "section4_gate_anch2"
KEY_FIG6 = GEN.S4G2_KEY                 # "section4_gate_anch2_fig6"
WAVE_ALIASES = {"v1": KEY, KEY: KEY, "fig6": KEY_FIG6, KEY_FIG6: KEY_FIG6}
TAG_PREFIX = "pofds4g"
MODEL_SLUG = "mistral7b"
ARM_LABEL = {"b0": "ordinary SFT",
             "d8": "frozen personal-history ICL (8 days)"}
COND_TOK = dict(GEN.S4G_COND_TOK)       # {"fixed": "fixb20", "evolving": "evoall"}
TOK_COND = {v: k for k, v in COND_TOK.items()}
W_PLAT = float(GEN.W_WPLAT)             # 0.5
INNATE_LAMBDA = float(GEN.W_LAMBDA)     # 0.2, k, the anchor weight
N_ROUNDS = int(GEN.S4G_ROUNDS)          # 30
CELL_KINDS = ("gpu", "witness", "twin")


class WaveGrid:
    """One wave's grid, as the generator declares it.

    cells   [(arm, cond, ea, es, seed, kind)], kind in CELL_KINDS
    tag()   gen_pofd_sweep.s4g_tag, with rounds=None -> the base tag and
            rounds=60/100 -> the _r60/_r100 extension tag (fig6 only)
    """

    def __init__(self, key, cells, gates, ess, seeds, arms, conds, fig6,
                 stem, ext_rounds_ok=()):
        self.key = key
        self.cells = list(cells)
        self.gates = list(gates)
        self.ess = list(ess)
        self.seeds = list(seeds)
        self.arms = list(arms)
        self.conds = list(conds)
        self.fig6 = bool(fig6)
        self.stem = stem
        self.ext_rounds_ok = tuple(int(r) for r in ext_rounds_ok)
        self.kind_of = {c[:5]: c[5] for c in self.cells}
        self.keys = [c[:5] for c in self.cells]
        self.n_cells = len(self.cells)
        assert len(set(self.keys)) == self.n_cells, "duplicate cells"
        self.n_kind = {k: sum(1 for c in self.cells if c[5] == k)
                       for k in CELL_KINDS}

    def tag(self, arm, cond, ea, es, seed, rounds=None):
        return GEN.s4g_tag(arm, cond, ea, es, seed, prefix=TAG_PREFIX,
                           rounds=rounds)

    def horizons(self):
        """Artifact horizons to look for, LONGEST FIRST; None = base tag."""
        if not self.fig6:
            return [None]
        return sorted(self.ext_rounds_ok, reverse=True) + [None]

    def __repr__(self):
        return (f"WaveGrid({self.key}: {self.n_cells} cells, "
                f"ea={self.gates}, es={self.ess}, seeds={self.seeds})")


def load_grids():
    v1 = WaveGrid(
        key=GEN.S4G_KEY,
        cells=[(arm, cond, ea, es, seed, "gpu")
               for arm in GEN.S4G_ARMS for cond in GEN.S4G_CONDS
               for ea in GEN.S4G_GATES for es in GEN.S4G_ESS
               for seed in GEN.S4G_SEEDS],
        gates=GEN.S4G_GATES, ess=GEN.S4G_ESS, seeds=GEN.S4G_SEEDS,
        arms=GEN.S4G_ARMS, conds=GEN.S4G_CONDS, fig6=False,
        stem="section4_gate")
    fig6 = WaveGrid(
        key=GEN.S4G2_KEY, cells=GEN.s4g2_cells(),
        gates=GEN.S4G2_GATES, ess=GEN.S4G2_ESS, seeds=GEN.S4G2_SEEDS,
        arms=GEN.S4G2_ARMS, conds=GEN.S4G2_CONDS, fig6=True,
        stem="section4_fig6", ext_rounds_ok=GEN.S4G2_EXT_ROUNDS_OK)
    assert v1.n_cells == GEN.S4G_N_TOTAL == 72, v1.n_cells
    assert fig6.n_cells == GEN.S4G2_N_CELLS == 192, fig6.n_cells
    assert fig6.n_kind == {"gpu": GEN.S4G2_N_GPU,
                           "witness": GEN.S4G2_N_WITNESS,
                           "twin": GEN.S4G2_N_TWIN}, fig6.n_kind
    # the seed set, arms and conditions are WAVE-WIDE (the fig6 grid
    # declares S4G2_SEEDS = S4G_SEEDS); the per-seed helpers rely on it
    assert fig6.seeds == v1.seeds and fig6.arms == v1.arms \
        and fig6.conds == v1.conds
    return {v1.key: v1, fig6.key: fig6}


GRIDS = load_grids()
GRID_V1 = GRIDS[KEY]
GRID_FIG6 = GRIDS[KEY_FIG6]

# The DEFAULT wave's grid under the historical names.  Every function
# below takes an explicit ``grid`` (None -> GRID_V1), so these are the
# original wave's values -- read from the generator, not typed here.
ARMS = GRID_V1.arms
CONDS = GRID_V1.conds
EAS = GRID_V1.gates
ESS = GRID_V1.ess
SEEDS = GRID_V1.seeds
N_CELLS = GRID_V1.n_cells                                            # 72


def _grid(grid):
    return GRID_V1 if grid is None else grid

# ------------------------------------------- inherited numeric conventions
LATE = range(25, 30)         # analyze_bottom20_section4_3seed.py, verbatim
LATE_IDX = list(LATE)
N_LATE = len(LATE_IDX)       # the FINAL FIVE post-peer rounds


def late_window(n_rounds):
    """op_raw indices of the final five post-peer rounds of an n-round
    artifact: n-5..n-1.  For the 30-round base tag this IS the inherited
    LATE = range(25, 30) = rounds 26-30 in the runner's 1-based numbering
    (asserted below); for a _r60 / _r100 extension it is 55..59 / 95..99
    (rounds 56-60 / 96-100)."""
    n_rounds = int(n_rounds)
    if n_rounds < N_LATE:
        raise ValueError(f"{n_rounds} rounds < {N_LATE} needed")
    return list(range(n_rounds - N_LATE, n_rounds))


assert late_window(N_ROUNDS) == LATE_IDX == [25, 26, 27, 28, 29]
assert [t + 1 for t in late_window(N_ROUNDS)] == [26, 27, 28, 29, 30]
# 95% two-sided Student-t critical value at df = 2 (three seeds).
# LITERAL, equal to scipy.stats.t.ppf(0.975, 2); no scipy on the login node.
T_CRIT_DF2 = 4.302652729911275
T_CRIT_SOURCE = "literal 4.302652729911275 == scipy.stats.t.ppf(0.975, 2)"
CLAMP_FRAC = 0.20
EXPECTED_N_AGENTS = 723
EXPECTED_N_CLAMP = 145       # round(0.20 * 723)
POP_UPDATE_V2 = "nested_ai_anchored_then_social_v2"
AI_GATE_MODE = "threshold"
# structural null (d8, eps_social = 0): bit-exact within one GPU
# architecture; across architectures the residue is greedy-generation
# nondeterminism, and 5e-3 is the "large enough to contaminate" line.
NULL_TOL = 1e-9
NULL_TOL_XHW = 5e-3
DEFAULT_DRIFT_TOL = 0.002    # analyze_section3.py tolerance, opinion units
RANGE_TOL_MULT = 2           # (c): late-window range of round means <= 2*tol
CYCLE_WINDOW = 10            # last 10 consecutive round-mean differences ...
CYCLE_ALTERNATION_MIN = 0.7  # ... alternating in sign on >= 70% of the steps

NA = "NA"

# ------------------------------------------------------- house figure ink
INK = "#202328"
GRID_GREY = "#d9dde2"
ARM_COLOR = {"b0": "#356fb6", "d8": "#d97706"}
ARM_MARKER = {"b0": "o", "d8": "s"}


# ============================================================ tag grammar
def _num(v):
    """0.2 -> '0p2', 1.0 -> '1', 0.0 -> '0' (the wave-wide convention)."""
    return f"{v:g}".replace(".", "p")


def _unnum(tok):
    """'0p2' -> 0.2, '1' -> 1.0 (inverse of _num)."""
    return float(tok.replace("p", "."))


def cell_tag(arm, cond, ea, es, seed, rounds=None):
    """The pinned tag for one cell -- gen_pofd_sweep.s4g_tag, so the
    grammar lives in exactly one place.  rounds=None -> the base tag;
    rounds=60/100 -> the _r60/_r100 horizon-extension tag."""
    return GEN.s4g_tag(arm, cond, ea, es, seed, prefix=TAG_PREFIX,
                       rounds=rounds)


# the grammar is pinned by the generator and cross-checked here at import
assert cell_tag("b0", "fixed", 0.2, 0.0, 42) == (
    "pofds4g_mistral7b_b0_fixb20_anch2_ea0p2_w0p5_l0p2_es0_s42")
assert cell_tag("d8", "evolving", 0.1, 0.3, 0, rounds=60) == (
    "pofds4g_mistral7b_d8_evoall_anch2_ea0p1_w0p5_l0p2_es0p3_s0_r60")

_TAG_RE = re.compile(
    r"^" + TAG_PREFIX + r"_(?P<model>[a-z0-9_]+?)"
    r"_(?P<arm>b0|d8)_(?P<cond_tok>fixb20|evoall)_anch2"
    r"_ea(?P<ea>[0-9p]+)_w(?P<w>[0-9p]+)_l(?P<l>[0-9p]+)"
    r"_es(?P<es>[0-9p]+)_s(?P<seed>[0-9]+)(?:_r(?P<rounds>[0-9]+))?$")


def parse_tag(tag, grid=None):
    """Parse a wave tag -> dict, or None if it is not one of ours.

    Returns model / arm / cond / eps_ai / w_plat / innate_lambda /
    eps_social / seed / rounds (None for a base tag, 60/100 for an
    extension) and in_grid: whether the parsed cell is one the given wave
    declares (default: the original 72-cell wave, where an extension tag
    is never in grid).  Round-trips with cell_tag for in-grid cells.
    """
    g = _grid(grid)
    m = _TAG_RE.match(tag)
    if m is None:
        return None
    out = {"tag": tag,
           "model": m.group("model"),
           "arm": m.group("arm"),
           "cond": TOK_COND[m.group("cond_tok")],
           "eps_ai": _unnum(m.group("ea")),
           "w_plat": _unnum(m.group("w")),
           "innate_lambda": _unnum(m.group("l")),
           "eps_social": _unnum(m.group("es")),
           "seed": int(m.group("seed")),
           "rounds": (int(m.group("rounds")) if m.group("rounds")
                      else None)}
    key = (out["arm"], out["cond"], out["eps_ai"], out["eps_social"],
           out["seed"])
    out["kind"] = g.kind_of.get(key)
    out["in_grid"] = (out["model"] == MODEL_SLUG
                      and key in g.kind_of
                      and out["w_plat"] == W_PLAT
                      and out["innate_lambda"] == INNATE_LAMBDA
                      and (out["rounds"] is None
                           or out["rounds"] in g.ext_rounds_ok))
    return out


def scan_run_root(run_root, grid=None):
    """Every pofds4g_ tag physically present under run_root (a run counts
    only when it has a trajectory.pt), parsed.  Used to report tags that
    exist but are NOT in the declared grid -- a grammar drift in the
    generator must not vanish silently."""
    found = []
    if not os.path.isdir(run_root):
        return found
    for name in sorted(os.listdir(run_root)):
        if not name.startswith(TAG_PREFIX + "_"):
            continue
        if not os.path.exists(os.path.join(run_root, name,
                                           "trajectory.pt")):
            continue
        p = parse_tag(name, grid)
        found.append(p if p is not None else {"tag": name, "in_grid": False,
                                              "unparsed": True})
    return found


# ============================================================ io helpers
def find_run(run_root, tag):
    """The run directory for a tag, or None."""
    p = os.path.join(run_root, tag, "trajectory.pt")
    return os.path.join(run_root, tag) if os.path.exists(p) else None


def load(run_dir):
    return torch.load(os.path.join(run_dir, "trajectory.pt"),
                      map_location="cpu", weights_only=False)


def gpu_arch(run_dir):
    """Coarse GPU architecture ('H100'/'A100'/raw name/'unknown') from
    config.json hardware.gpu_name -- inherited from
    analyze_bottom20_section4_3seed.gpu_arch.  Greedy generation is
    bit-reproducible only within one architecture, which is what the
    structural-null tolerance keys off."""
    try:
        with open(os.path.join(run_dir, "config.json")) as fh:
            hw = json.load(fh).get("hardware") or {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return "unknown"
    name = hw.get("gpu_name") or ""
    for arch in ("H100", "A100", "A6000", "V100", "B200", "L40"):
        if arch in name:
            return arch
    return name or "unknown"


def has_real_twin(d):
    """Does the artifact carry a saved twin_raw shaped like op_raw?"""
    tw = d.get("twin_raw")
    return (tw is not None and torch.is_tensor(tw) and tw.numel() > 0
            and tuple(tw.shape) == tuple(d["op_raw"].shape))


def twin_sha(d):
    """sha256 of twin_raw's bytes (float32) -- lets every base run at one
    (cond, eps_social, seed) be asserted twin-identical without holding
    two runs in memory.  None when the run carries no real twin."""
    if not has_real_twin(d):
        return None
    t = d["twin_raw"].detach().cpu().float().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def twin_derived_artifact(d, ea=0.0):
    """The eps_AI = 0 cell DERIVED from a run at the same (cond,
    eps_social, seed): a trajectory-shaped dict whose population IS the
    run's twin_raw (the matched no-AI process; at eps_AI = 0 the strict
    gate |m - x'| < 0 is closed for everyone, so the served vector never
    enters and the method drops out).  pred_raw is None: nothing was
    served.  The config is the source run's with eps_ai rewritten to 0,
    and a fixed run's clamp mask is carried so the inherited structural
    checks (clamp held, mask == bottom cohort) run on the twin too."""
    if not has_real_twin(d):
        raise ValueError("no real twin_raw to derive from")
    cfg = dict(d.get("config") or {})
    cfg["eps_ai"] = float(ea)
    cfg["derived_from_run_tag"] = cfg.get("run_tag")
    out = {"config": cfg, "op_raw": d["twin_raw"], "twin_raw": d["twin_raw"],
           "pred_raw": None, "innate": d["innate"]}
    for k in ("innate_clamp_mask", "innate_clamp_count", "innate_clamp_mode",
              "innate_clamp_frac", "innate_clamp_seed",
              "innate_clamp_peer_mode"):
        if k in d:
            out[k] = d[k]
    return out


SERVED_NA = "n/a (gate closed)"


def served_cardinality(pred, late_idx):
    """Late-window served-value cardinality: the number of DISTINCT values
    in pred_raw pooled over the window's rounds and agents (finite values
    only), the share held by the single most common value, and that
    value.  A served map with 1-3 distinct values cannot carry a graded
    effect, so these are printed next to T_a."""
    vals = pred[late_idx].reshape(-1).float()
    finite = vals[torch.isfinite(vals)]
    n_nan = int(vals.numel() - finite.numel())
    if finite.numel() == 0:
        return {"served_distinct": 0, "served_top_share": None,
                "served_top_value": None, "served_n_finite": 0,
                "served_n_nan": n_nan}
    uniq, counts = torch.unique(finite, return_counts=True)
    i = int(counts.argmax())
    return {"served_distinct": int(uniq.numel()),
            "served_top_share": float(counts[i]) / float(finite.numel()),
            "served_top_value": float(uniq[i]),
            "served_n_finite": int(finite.numel()),
            "served_n_nan": n_nan}


def twin_of(d):
    """(twin [T, n], source).  The saved matched no-platform process when
    present; else the innate vector broadcast over rounds, which IS the
    no-platform process at k > 0 (innate is a fixed point of
    h = k*innate + (1-k)*x and the twin starts there).  Inherited from
    analyze_sft_icl_reach.twin_of -- no counterfactual is simulated here."""
    tw = d.get("twin_raw")
    op = d["op_raw"]
    if tw is not None and tw.numel() > 0 and tuple(tw.shape) == tuple(op.shape):
        return tw.float(), "twin_raw"
    n_r = op.shape[0]
    return d["innate"].float().unsqueeze(0).expand(n_r, -1), "innate_es0"


# ============================================================ statistics
def w1(a, b):
    """Wasserstein-1 between two equal-size empirical populations (house
    definition: analyze_sft_icl_reach.w1 / analyze_fig2_provider.w1)."""
    return float((torch.sort(a).values - torch.sort(b).values).abs().mean())


MAE_COL = "mae_b_paired"
MAE_DEF = ("mae_b_paired = mean over the late window of mean_i "
           "|op_B(fixed)[t][i] - op_B(evolving)[t][i]| over the cohort-B "
           "agents i, the two members PAIRED BY AGENT ID at the same "
           "window position (2026-08-27). The DIRECT fixed-vs-evolving "
           "transmission magnitude: T_a / delta_mu_b is the difference of "
           "cohort-B MEANS and cancels when cohort A's influence spreads B "
           "without shifting it; the paired MAE does not. Sign-free "
           "(>= 0); 0 exactly for the d8/eps_social=0 structural null.")


def op_b_window(d, mask_a, late_idx):
    """[len(late_idx), n_B] float32: the artifact's op_raw over the late
    window on cohort B only -- the ONLY tensor that outlives a cell's
    load, kept so the fixed/evolving pair can be contrasted agent by
    agent (5 x 578 floats). A twin-derived artifact's op_raw IS its
    source run's twin_raw, so the same window applies."""
    op = d["op_raw"].float()
    idx = torch.tensor(list(late_idx), dtype=torch.long)
    return op[idx][:, ~mask_a].clone()


def paired_mae_b(win_f, win_e):
    """(mae, mae_h1, mae_h2): the agent-paired cohort-B MAE between the
    fixed and evolving windows over the whole window and its two halves
    (half_split of the window positions: 2 vs 3 for a 5-round window).
    Both windows must have the same shape (same window length, same
    cohort B)."""
    if tuple(win_f.shape) != tuple(win_e.shape):
        raise ValueError(f"paired windows differ in shape "
                         f"{tuple(win_f.shape)} vs {tuple(win_e.shape)}")
    per_round = (win_f - win_e).abs().mean(dim=1)          # [T_win]
    h1, h2 = half_split(list(range(int(per_round.numel()))))
    return (float(per_round.mean()),
            float(per_round[h1].mean()), float(per_round[h2].mean()))


def cohort_a_mask(innate, frac=CLAMP_FRAC):
    """Bool [n] mask of cohort A = the round(frac*n) LOWEST-innate agents
    under the deterministic (innate value, agent id) ranking.

    This is _gated_pop.innate_clamp_mask(mode="bottom") reimplemented on
    the analysis side exactly as analyze_bottom20_section4_3seed.py does
    it, so the EVOLVING condition -- which stores no mask -- can be masked
    identically to its FIXED partner.  Every fixed run's STORED mask is
    checked against this reconstruction."""
    innate = innate.float()
    n = int(innate.numel())
    n_a = int(round(float(frac) * n))
    if not 0 < n_a < n:
        raise ValueError(f"clamp frac {frac!r} gives a degenerate cohort "
                         f"({n_a} of {n})")
    order = sorted(range(n), key=lambda i: (float(innate[i]), i))
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.tensor(order[:n_a], dtype=torch.long)] = True
    return mask


def innate_sha(innate):
    """sha256 of the innate bytes -- lets a fixed/evolving pair be
    re-asserted bit-identical without holding two runs in memory."""
    t = innate.detach().cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def tci3(vals):
    """(mean, sd, ci_lo, ci_hi): three-seed mean with the 95% Student-t
    interval at df = 2 -- analyze_bottom20_section4_3seed.tci3, verbatim.

    Refuses n != 3: an incomplete seed set must never be averaged into a
    row that looks like a three-seed result."""
    n = len(vals)
    if n != len(SEEDS):
        raise ValueError(f"tci3 needs exactly {len(SEEDS)} seed values, "
                         f"got {n}")
    m = sum(vals) / n
    sd = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5
    half = T_CRIT_DF2 * sd / n ** 0.5
    return m, sd, m - half, m + half


def excludes(ci_lo, ci_hi, ref):
    """Does the interval exclude the reference value? (inherited)"""
    return bool(ci_lo > ref or ci_hi < ref)


def half_split(idx):
    """(first half, second half) of a late window, the
    analyze_section3.convergence convention: half = n // 2, so a 5-round
    window splits 2 / 3.  Guarantees a two-round wiggle is visible as a
    half-to-half difference instead of hiding inside one mean."""
    h = len(idx) // 2
    return list(idx[:h]), list(idx[h:])


LATE_H1, LATE_H2 = half_split(LATE_IDX)


# ==================================================== per-cell reduction
def reduce_cell(d, mask_a, late_idx=None):
    """Reduce ONE trajectory to (per-round rows, late-window scalars).

    Every tensor touched here is local, so the caller can drop the whole
    trajectory before opening the next one.  op_raw[t] is the END-OF-ROUND
    POST-PEER state; nothing else is read.

    late_idx: the equilibrium window (default the inherited LATE_IDX =
    25..29 of a 30-round run; fig6 passes late_window(n_rounds) so a
    _r60/_r100 extension is read on ITS final five rounds).  The halves
    are half_split(late_idx) (2 vs 3).  pred_raw may be None (a
    twin-derived eps_AI = 0 cell: nothing was served) -- then every served
    column is None / SERVED_NA.
    """
    late_idx = LATE_IDX if late_idx is None else list(late_idx)
    h1, h2 = half_split(late_idx)
    op = d["op_raw"].float()
    tw, twin_source = twin_of(d)
    pred = d.get("pred_raw")
    pred = None if pred is None else pred.float()
    innate = d["innate"].float()
    a, b = mask_a, ~mask_a
    n_r = int(op.shape[0])
    if late_idx[-1] >= n_r:
        raise ValueError(f"late window {late_idx} outside {n_r} rounds")

    rounds = []
    for t in range(n_r):
        x, xt = op[t], tw[t]
        if pred is not None:
            served = pred[t].clamp(0.0, 1.0)
            served_mean = float(torch.nanmean(served))
            pred_mean_raw = float(torch.nanmean(pred[t]))
        else:
            served_mean, pred_mean_raw = None, None
        rounds.append({
            "round": t,                       # op_raw index (late window)
            "round_1based": t + 1,            # the runner's round number
            "in_late_window": t in late_idx,
            "pop_mean": float(x.mean()),
            "pop_sd": float(x.std()),
            "a_mean": float(x[a].mean()),
            "a_sd": float(x[a].std()),
            "b_mean": float(x[b].mean()),
            "b_sd": float(x[b].std()),
            "w1_twin_pop": w1(x, xt),
            "w1_twin_b": w1(x[b], xt[b]),
            "served_mean": served_mean,
            "pred_mean_raw": pred_mean_raw,
            "twin_source": twin_source,
        })

    def wmean(idx, key):
        if any(rounds[t][key] is None for t in idx):
            return None
        return sum(rounds[t][key] for t in idx) / len(idx)

    # twin-referenced dispersion ratio, inherited: mean over the window of
    # SD(B platform)/SD(B twin), skipping any round with a zero twin SD.
    def twin_ratio(idx, sel):
        vals = []
        for t in idx:
            s_tw = float(tw[t][sel].std())
            if s_tw > 0:
                vals.append(float(op[t][sel].std()) / s_tw)
        return sum(vals) / len(vals) if vals else float("nan")

    late = {
        "n_rounds": n_r,
        "late_rounds_op_raw": f"{late_idx[0]}-{late_idx[-1]}",
        "late_rounds_1based": f"{late_idx[0] + 1}-{late_idx[-1] + 1}",
        "twin_source": twin_source,
        "n_a": int(a.sum()), "n_b": int(b.sum()),
        "innate_mean": float(innate.mean()),
        "innate_sd": float(innate.std()),
        "innate_a_mean": float(innate[a].mean()),
        "innate_b_mean": float(innate[b].mean()),
        "innate_b_sd": float(innate[b].std()),
        "mu_pop_eq": wmean(late_idx, "pop_mean"),
        "mu_a_eq": wmean(late_idx, "a_mean"),
        "mu_b_eq": wmean(late_idx, "b_mean"),
        "sd_pop_late": wmean(late_idx, "pop_sd"),
        "sd_a_late": wmean(late_idx, "a_sd"),
        "sd_b_late": wmean(late_idx, "b_sd"),
        "w1_twin_pop_late": wmean(late_idx, "w1_twin_pop"),
        "w1_twin_b_late": wmean(late_idx, "w1_twin_b"),
        "served_mean_late": wmean(late_idx, "served_mean"),
        "sd_ratio_late": twin_ratio(late_idx, b),       # inherited name
        "sd_ratio_pop_twin_late": twin_ratio(late_idx, slice(None)),
        # half-window values, for the drift / robustness flag
        "mu_b_h1": wmean(h1, "b_mean"),
        "mu_b_h2": wmean(h2, "b_mean"),
        "mu_pop_h1": wmean(h1, "pop_mean"),
        "mu_pop_h2": wmean(h2, "pop_mean"),
        "sd_b_h1": wmean(h1, "b_sd"),
        "sd_b_h2": wmean(h2, "b_sd"),
        "sd_pop_h1": wmean(h1, "pop_sd"),
        "sd_pop_h2": wmean(h2, "pop_sd"),
        "innate_sha256": innate_sha(innate),
    }
    # per-CELL convergence statistics; the settled verdict against
    # --drift-tol is applied by the caller (settle_verdict), which knows
    # the tolerance
    #   (a) final-5 half-split drift, second half minus first half
    late["mu_b_drift"] = late["mu_b_h2"] - late["mu_b_h1"]
    b_means = [r["b_mean"] for r in rounds]
    #   (b) final-10 half-split: the final five minus the five before them
    n_late = len(late_idx)
    prev_idx = [t - n_late for t in late_idx]
    if prev_idx[0] >= 0:
        late["mu_b_prev5"] = sum(b_means[t] for t in prev_idx) / n_late
        late["late10_drift"] = late["mu_b_eq"] - late["mu_b_prev5"]
        late["late10_rounds_1based"] = (
            f"{prev_idx[0] + 1}-{prev_idx[-1] + 1} vs "
            f"{late_idx[0] + 1}-{late_idx[-1] + 1}")
    else:
        late["mu_b_prev5"] = None
        late["late10_drift"] = None
        late["late10_rounds_1based"] = None
    #   (c) range of the five late-window round means
    lv = [b_means[t] for t in late_idx]
    late["late5_range"] = max(lv) - min(lv)
    #   cycle detector: sign alternation of the last CYCLE_WINDOW
    #   consecutive round-mean differences (fraction of the steps
    #   between consecutive differences on which the sign flips)
    last = late_idx[-1]
    if last - CYCLE_WINDOW >= 0:
        diffs = [b_means[t] - b_means[t - 1]
                 for t in range(last - CYCLE_WINDOW + 1, last + 1)]
        steps = len(diffs) - 1
        flips = sum(1 for i in range(steps) if diffs[i] * diffs[i + 1] < 0)
        late["cycle_alternation"] = flips / steps
    else:
        late["cycle_alternation"] = None
    if pred is not None:
        late.update(served_cardinality(pred, late_idx))
    else:
        late.update({"served_distinct": SERVED_NA,
                     "served_top_share": SERVED_NA,
                     "served_top_value": None, "served_n_finite": None,
                     "served_n_nan": None})
    return rounds, late


def settle_verdict(late, tol):
    """The per-cell SETTLED verdict from reduce_cell's statistics:
      settled_a  |final-5 half-split drift| <= tol
      settled_b  |final-10 half-split drift| <= tol
      settled_c  range of the five late round means <= RANGE_TOL_MULT*tol
      settled    a and b and c
      cyclic     not settled AND the last-10-difference sign alternation
                 >= CYCLE_ALTERNATION_MIN  (a long-run / cyclic outcome)
    """
    d5, d10 = late.get("mu_b_drift"), late.get("late10_drift")
    rg, alt = late.get("late5_range"), late.get("cycle_alternation")
    a = d5 is not None and abs(d5) <= tol
    b = d10 is not None and abs(d10) <= tol
    c = rg is not None and rg <= RANGE_TOL_MULT * tol
    settled = bool(a and b and c)
    cyclic = bool((not settled) and alt is not None
                  and alt >= CYCLE_ALTERNATION_MIN)
    return {"settled_a": bool(a), "settled_b": bool(b), "settled_c": bool(c),
            "settled": settled, "cyclic": cyclic, "drift_tol": tol,
            "range_tol": RANGE_TOL_MULT * tol}


def structural_checks(d, key, mask_a, ref_sha, tag=None, horizon=None):
    """FATAL structural problems with one cell, as a list of strings.

    Empty list == the cell can take part in the fixed/evolving contrast.
    horizon: when given (fig6), the artifact must hold EXACTLY that many
    rounds (and config n_rounds, if present, must agree) -- an extension
    tag that is not the extension it claims to be is a lie.
    """
    arm, cond, ea, es, seed = key
    tag = tag or cell_tag(*key)
    bad = []
    cfg = d.get("config") or {}
    op = d["op_raw"]
    innate = d["innate"]

    if horizon is None:
        if int(op.shape[0]) < max(LATE_IDX) + 1:
            bad.append(f"{tag}: {int(op.shape[0])} rounds < "
                       f"{max(LATE_IDX) + 1} needed for the late window")
    else:
        if int(op.shape[0]) != int(horizon):
            bad.append(f"{tag}: op_raw holds {int(op.shape[0])} rounds, the "
                       f"tag's horizon is {horizon}")
        nr = cfg.get("n_rounds")
        if nr is not None and int(nr) != int(horizon):
            bad.append(f"{tag}: config n_rounds={nr!r} != tag horizon "
                       f"{horizon}")
    if int(innate.numel()) != int(op.shape[1]):
        bad.append(f"{tag}: innate has {int(innate.numel())} agents, "
                   f"op_raw has {int(op.shape[1])}")
    if ref_sha is not None and innate_sha(innate) != ref_sha:
        bad.append(f"{tag}: innate differs from the shared wave population "
                   f"(sha mismatch) -- the A/B partition would not be the "
                   f"same partition")

    pu = cfg.get("population_update")
    if pu != POP_UPDATE_V2:
        bad.append(f"{tag}: population_update={pu!r}, expected "
                   f"{POP_UPDATE_V2!r} -- this is the CORRECTED-gate wave, "
                   f"the 'anch2' token must be true by construction")
    gm = cfg.get("ai_gate_mode")
    if gm is not None and gm != AI_GATE_MODE:
        bad.append(f"{tag}: ai_gate_mode={gm!r}, expected {AI_GATE_MODE!r}")
    for field, want in (("eps_ai", ea), ("eps", es), ("w_plat", W_PLAT),
                        ("innate_lambda", INNATE_LAMBDA), ("seed", seed)):
        got = cfg.get(field)
        if got is not None and abs(float(got) - float(want)) > 1e-9:
            bad.append(f"{tag}: config {field}={got!r} disagrees with the "
                       f"tag ({want!r}) -- the tag would be a lie")
    gb = cfg.get("gamma_bias")
    if gb is not None and abs(float(gb)) > 1e-9:
        bad.append(f"{tag}: gamma_bias={gb!r}, expected 0 (no homophily "
                   f"selection bias anywhere in this wave)")

    cm = d.get("innate_clamp_mask")
    if cond == "fixed":
        if cm is None or not torch.is_tensor(cm) or cm.numel() == 0:
            bad.append(f"{tag}: FIXED condition carries no "
                       f"innate_clamp_mask")
        elif not torch.equal(cm.bool(), mask_a):
            bad.append(f"{tag}: stored innate_clamp_mask != the recomputed "
                       f"bottom-{int(mask_a.sum())} cohort")
        else:
            cnt = d.get("innate_clamp_count")
            if cnt is not None and int(cnt) != int(mask_a.sum()):
                bad.append(f"{tag}: innate_clamp_count={cnt} != "
                           f"{int(mask_a.sum())}")
            mode = cfg.get("innate_clamp_mode",
                           d.get("innate_clamp_mode"))
            if mode is not None and mode != "bottom":
                bad.append(f"{tag}: innate_clamp_mode={mode!r}, expected "
                           f"'bottom'")
            # the clamp must actually have HELD every round
            opa = d["op_raw"].float()[:, mask_a]
            inn_a = d["innate"].float()[mask_a]
            if not torch.equal(opa, inn_a.unsqueeze(0).expand_as(opa)):
                worst = float((opa - inn_a.unsqueeze(0)).abs().max())
                bad.append(f"{tag}: cohort A moved in a FIXED run "
                           f"(max |op - innate| = {worst:.3e})")
    else:
        if cm is not None and torch.is_tensor(cm) and cm.numel() > 0:
            bad.append(f"{tag}: EVOLVING condition carries a clamp mask -- "
                       f"not a fully evolving population")
    return bad


# ============================================================ csv output
def _cell(v):
    if v is None:
        return NA
    if isinstance(v, float) and not math.isfinite(v):
        return NA
    return v


def write_csv(out_dir, name, rows):
    if not rows:
        print(f"[s4g] SKIP {name} (no rows)")
        return None
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    path = os.path.join(out_dir, name)
    with open(path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({c: _cell(r.get(c)) for c in cols})
    print(f"[s4g] wrote {name} ({len(rows)} rows, {len(cols)} cols)")
    return path


# ========================================================== aggregation
def agg_block(prefix, per_seed, ref=0.0, ref_name="zero"):
    """Per-seed columns + the three-seed mean / sd / 95% t interval for one
    statistic.  per_seed is {seed: value}; a short or non-finite set yields
    per-seed columns and NA aggregates, never a partial average."""
    out = {f"{prefix}_s{s}": per_seed.get(s) for s in SEEDS}
    vals = [per_seed.get(s) for s in SEEDS]
    ok = (len(per_seed) == len(SEEDS)
          and all(v is not None and math.isfinite(v) for v in vals))
    if not ok:
        out.update({f"{prefix}_mean": None, f"{prefix}_sd": None,
                    f"{prefix}_ci_lo": None, f"{prefix}_ci_hi": None,
                    f"{prefix}_ci_excludes_{ref_name}": None})
        return out
    m, sd, lo, hi = tci3(vals)
    out.update({f"{prefix}_mean": m, f"{prefix}_sd": sd,
                f"{prefix}_ci_lo": lo, f"{prefix}_ci_hi": hi,
                f"{prefix}_ci_excludes_{ref_name}": excludes(lo, hi, ref)})
    return out


def drift_block(prefix, per_seed_h1, per_seed_h2, tol, ci_half):
    """First-half vs second-half robustness flag for one series.

    drift_s = (second half) - (first half) per seed; the three-seed mean is
    the reported drift.  Flagged when it exceeds --drift-tol, and
    separately when it exceeds the three-seed CI half-width, so a
    two-round wiggle inside the late window can never be read as a trend.
    """
    out = {}
    drifts = {}
    for s in SEEDS:
        h1, h2 = per_seed_h1.get(s), per_seed_h2.get(s)
        if h1 is None or h2 is None:
            continue
        drifts[s] = h2 - h1
    out.update({f"{prefix}_drift_s{s}": drifts.get(s) for s in SEEDS})
    if len(drifts) == len(SEEDS) and all(math.isfinite(v)
                                        for v in drifts.values()):
        dm = sum(drifts.values()) / len(drifts)
        out[f"{prefix}_drift_mean"] = dm
        out[f"{prefix}_drift_tol"] = tol
        out[f"{prefix}_drift_flag"] = bool(abs(dm) > tol)
        out[f"{prefix}_drift_exceeds_ci"] = (
            bool(abs(dm) > ci_half) if (ci_half is not None
                                        and math.isfinite(ci_half))
            else None)
    else:
        out[f"{prefix}_drift_mean"] = None
        out[f"{prefix}_drift_tol"] = tol
        out[f"{prefix}_drift_flag"] = None
        out[f"{prefix}_drift_exceeds_ci"] = None
    return out


def _ci_half(row, prefix):
    lo, hi = row.get(f"{prefix}_ci_lo"), row.get(f"{prefix}_ci_hi")
    if lo is None or hi is None:
        return None
    return abs(hi - lo) / 2.0


T_A_COL = "t_a_evolving_minus_fixed"
T_A_SIGN = ("t_a_evolving_minus_fixed = mu_B^eq(A evolving) - mu_B^eq(A "
            "fixed): EVOLVING MINUS FIXED (the published Section-4 "
            "analyzer's sign); positive = a fully adaptive cohort A left "
            "the responsive majority HIGHER than a pinned cohort A did. "
            "delta_mu_b = -t_a (fixed minus evolving).")


G_COL = "g_sft_minus_icl"
G_SIGN = ("g_sft_minus_icl = T_a(SFT, b0) - T_a(ICL, d8) PER SEED (paired "
          "on seed): positive G = SFT's source effect exceeds ICL's; "
          "identically 0 at eps_AI = 0 (both methods twin-derived).")


def source_effect_block(d_b, primary_t_a):
    """The paired source-effect columns from the per-seed differences
    d_b = {seed: mu_b_eq(fixed) - mu_b_eq(evolving)}.

    Both signs are always written.  primary_t_a=False (original wave):
    delta_mu_b is the full block and t_a carries the prior analyzer's
    sign beside it, exactly as before.  primary_t_a=True (fig6): T_a =
    evolving - fixed is the full PRIMARY block (per seed, three-seed mean,
    sd, paired 95% t interval, excludes-zero) and delta_mu_b follows.
    Formed on the negated per-seed differences, so T_a's interval is the
    same paired interval mirrored (lo/hi swap), never a re-estimate.
    """
    delta = agg_block("delta_mu_b", d_b, 0.0, "zero")
    ta = agg_block(T_A_COL, {s: -v for s, v in d_b.items()}, 0.0, "zero")
    if primary_t_a:
        out = dict(ta)
        out.update(delta)
        return out
    out = dict(delta)
    for s in SEEDS:
        out[f"{T_A_COL}_s{s}"] = ta[f"{T_A_COL}_s{s}"]
    for suffix in ("mean", "ci_lo", "ci_hi"):
        out[f"{T_A_COL}_{suffix}"] = ta[f"{T_A_COL}_{suffix}"]
    return out


def pair_outcome(f, e, grid):
    """Settled verdict + outcome for ONE fixed/evolving pair at one seed
    (fig6).  Unsettled = EITHER member failed the drift tolerance.  The
    outcome names the next horizon the generator can run (extend_to_60 /
    extend_to_100); a pair with a twin-derived member has no GPU run to
    extend; a pair unsettled at the last allowed horizon is reported as
    such.  An unsettled pair is NEVER 'equilibrium'."""
    settled = bool(f.get("settled")) and bool(e.get("settled"))
    if settled:
        return True, "equilibrium"
    if f.get("cyclic") or e.get("cyclic"):
        return False, "cyclic"
    if f.get("analysed_from") == "twin_raw" or \
            e.get("analysed_from") == "twin_raw":
        return False, "unsettled_twin_derived"
    h = int(f.get("horizon") or N_ROUNDS)
    nxt = [r for r in sorted(grid.ext_rounds_ok) if r > h]
    if nxt:
        return False, f"extend_to_{nxt[0]}"
    return False, f"unsettled_at_{h}"


def _flags(c):
    """'a:T b:F c:T' -- which of the three settled tests a cell passed."""
    return " ".join(f"{k}:{'T' if c.get('settled_' + k) else 'F'}"
                    for k in ("a", "b", "c"))


def _served_summary(cells_by_seed, key):
    vals = [cells_by_seed[s].get(key) for s in SEEDS if s in cells_by_seed]
    nums = [v for v in vals if isinstance(v, (int, float))
            and not isinstance(v, bool)]
    return vals, nums


def build_source_rows(cells, drift_tol, grid=None, win_b=None):
    """B. SOURCE EFFECT per (arm, eps_ai, eps_social), aggregated over the
    three PAIRED seeds.  Original wave: delta_mu_b = fixed - evolving is
    primary.  fig6: T_a = evolving - fixed is primary, and every row also
    carries the per-pair settled verdicts / outcomes / horizons and the
    late-window served-value cardinality of both members.

    win_b: {cell key: op_b_window(...)} -- when both members of a pair
    carry a window, the row also gets the agent-paired cohort-B MAE
    block (MAE_COL: per seed, three-seed mean / sd / 95% t interval,
    half-window drift flag). Absent windows leave the block NA."""
    g = _grid(grid)
    rows = []
    for arm in g.arms:
        for ea in g.gates:
            for es in g.ess:
                paired = [s for s in SEEDS
                          if (arm, "fixed", ea, es, s) in cells
                          and (arm, "evolving", ea, es, s) in cells]
                complete = len(paired) == len(SEEDS)
                d_b, d_pop, d_b_h1, d_b_h2 = {}, {}, {}, {}
                mae, mae_h1, mae_h2 = {}, {}, {}
                gpu_pair, sha_ok = {}, {}
                fx, ev = {}, {}
                for s in paired:
                    f = cells[(arm, "fixed", ea, es, s)]
                    e = cells[(arm, "evolving", ea, es, s)]
                    fx[s], ev[s] = f, e
                    d_b[s] = f["mu_b_eq"] - e["mu_b_eq"]
                    d_pop[s] = f["mu_pop_eq"] - e["mu_pop_eq"]
                    d_b_h1[s] = f["mu_b_h1"] - e["mu_b_h1"]
                    d_b_h2[s] = f["mu_b_h2"] - e["mu_b_h2"]
                    gpu_pair[s] = f'{f["gpu_arch"]}/{e["gpu_arch"]}'
                    sha_ok[s] = f["innate_sha256"] == e["innate_sha256"]
                    if win_b is not None:
                        wf = win_b.get((arm, "fixed", ea, es, s))
                        we = win_b.get((arm, "evolving", ea, es, s))
                        if wf is not None and we is not None:
                            mae[s], mae_h1[s], mae_h2[s] = paired_mae_b(wf, we)
                row = {"arm": arm, "arm_label": ARM_LABEL[arm],
                       "eps_ai": ea, "eps_social": es,
                       "n_seeds_paired": len(paired),
                       "seeds_paired": "|".join(str(s) for s in paired),
                       "status": "complete" if complete else "incomplete"}
                if g.fig6:
                    # settled verdict FIRST, so the outcome sits beside T_a
                    outcomes = {}
                    for s in paired:
                        outcomes[s] = pair_outcome(fx[s], ev[s], g)
                    n_settled = sum(1 for v in outcomes.values() if v[0])
                    unsettled = sorted({v[1] for v in outcomes.values()
                                        if not v[0]})
                    row["settled"] = (complete and n_settled == len(SEEDS))
                    row["outcome"] = ("equilibrium" if row["settled"]
                                      else "|".join(unsettled) or
                                      "incomplete")
                    row["n_pairs_settled"] = n_settled
                    hs = sorted({int(fx[s].get("horizon") or 0)
                                 for s in paired})
                    row["horizon"] = "|".join(str(h) for h in hs)
                row.update(source_effect_block(d_b, primary_t_a=g.fig6))
                # the agent-paired transmission magnitude, beside T_a
                row.update(agg_block(MAE_COL, mae, 0.0, "zero"))
                row.update(drift_block(MAE_COL, mae_h1, mae_h2, drift_tol,
                                       _ci_half(row, MAE_COL)))
                row.update(agg_block("delta_mu_pop", d_pop, 0.0, "zero"))
                row.update(drift_block("delta_mu_b", d_b_h1, d_b_h2,
                                       drift_tol,
                                       _ci_half(row, "delta_mu_b")))
                if g.fig6:
                    for s in SEEDS:
                        oc = outcomes.get(s)
                        row[f"pair_settled_s{s}"] = (None if oc is None
                                                     else oc[0])
                        row[f"pair_outcome_s{s}"] = (None if oc is None
                                                     else oc[1])
                        row[f"pair_horizon_s{s}"] = (
                            fx[s].get("horizon") if s in fx else None)
                        row[f"pair_twin_derived_s{s}"] = (
                            (fx[s].get("analysed_from") == "twin_raw"
                             or ev[s].get("analysed_from") == "twin_raw")
                            if s in fx else None)
                        for cond, src in (("fixed", fx), ("evolving", ev)):
                            c = src.get(s)
                            row[f"mu_b_drift_{cond}_s{s}"] = (
                                c.get("mu_b_drift") if c else None)
                            row[f"late10_drift_{cond}_s{s}"] = (
                                c.get("late10_drift") if c else None)
                            row[f"late5_range_{cond}_s{s}"] = (
                                c.get("late5_range") if c else None)
                            row[f"settled_flags_{cond}_s{s}"] = (
                                _flags(c) if c else None)
                            row[f"cyclic_{cond}_s{s}"] = (
                                c.get("cyclic") if c else None)
                            row[f"cycle_alternation_{cond}_s{s}"] = (
                                c.get("cycle_alternation") if c else None)
                        # per-seed T_a on the two halves, for the paired
                        # method gap's own drift
                        row[f"t_a_h1_s{s}"] = (-d_b_h1[s] if s in d_b_h1
                                               else None)
                        row[f"t_a_h2_s{s}"] = (-d_b_h2[s] if s in d_b_h2
                                               else None)
                    row["drift_tol"] = drift_tol
                    row["range_tol"] = RANGE_TOL_MULT * drift_tol
                    for cond, src in (("fixed", fx), ("evolving", ev)):
                        vals, nums = _served_summary(src, "served_distinct")
                        row[f"served_distinct_{cond}"] = "|".join(
                            str(v) for v in vals)
                        vals, tops = _served_summary(src, "served_top_share")
                        row[f"served_top_share_{cond}"] = "|".join(
                            (f"{v:.3f}" if isinstance(v, float) else str(v))
                            for v in vals)
                        row[f"served_distinct_{cond}_min"] = (
                            min(nums) if nums else None)
                        row[f"served_top_share_{cond}_max"] = (
                            max(tops) if tops else None)
                    row["kind_fixed"] = "|".join(
                        str(fx[s].get("kind")) for s in paired)
                    row["kind_evolving"] = "|".join(
                        str(ev[s].get("kind")) for s in paired)
                row.update({f"gpu_pair_s{s}": gpu_pair.get(s)
                            for s in SEEDS})
                row["n_seeds_hardware_matched"] = sum(
                    1 for s in paired
                    if gpu_pair[s].split("/")[0] == gpu_pair[s].split("/")[1]
                    != "unknown")
                row["innate_pair_bit_identical"] = (
                    all(sha_ok.values()) if sha_ok else None)
                rows.append(row)
    return rows


def build_method_gap_rows(source_rows, drift_tol, grid):
    """D. PAIRED METHOD GAP per (eps_AI, eps_social):
    G = t_a(b0, seed) - t_a(d8, seed) PER SEED from the same paired
    per-seed T_a, then the three-seed mean and the df=2 paired t
    interval, its own half-window drift, and the settled verdict of the
    six cells behind it.  Positive G = SFT's source effect exceeds ICL's.
    At eps_AI = 0 both arms are twin-derived, so G is identically 0."""
    rows = []
    for ea in grid.gates:
        for es in grid.ess:
            rb, rd = _pick(source_rows, "b0", ea, es), \
                _pick(source_rows, "d8", ea, es)
            row = {"eps_ai": ea, "eps_social": es, "arms": "b0-d8",
                   "sign": "positive = SFT source effect exceeds ICL's"}
            if rb is None or rd is None:
                row["status"] = "incomplete"
                rows.append(row)
                continue
            complete = (rb["status"] == "complete"
                        and rd["status"] == "complete")
            row["status"] = "complete" if complete else "incomplete"
            g, h1, h2, settled = {}, {}, {}, {}
            for s in SEEDS:
                tb, td = rb.get(f"{T_A_COL}_s{s}"), rd.get(f"{T_A_COL}_s{s}")
                if tb is None or td is None:
                    continue
                g[s] = tb - td
                if rb.get(f"t_a_h1_s{s}") is not None \
                        and rd.get(f"t_a_h1_s{s}") is not None:
                    h1[s] = rb[f"t_a_h1_s{s}"] - rd[f"t_a_h1_s{s}"]
                    h2[s] = rb[f"t_a_h2_s{s}"] - rd[f"t_a_h2_s{s}"]
                pb, pd = rb.get(f"pair_settled_s{s}"), \
                    rd.get(f"pair_settled_s{s}")
                settled[s] = (bool(pb) and bool(pd)) if (
                    pb is not None and pd is not None) else None
            n_set = sum(1 for v in settled.values() if v)
            row["settled"] = bool(complete and n_set == len(SEEDS))
            outs = sorted({o for r in (rb, rd)
                           for o in (r.get("outcome") or "").split("|")
                           if o and o != "equilibrium"})
            row["outcome"] = ("equilibrium" if row["settled"]
                              else "|".join(outs) or "incomplete")
            row["n_pairs_settled"] = n_set
            blk = agg_block(G_COL, g, 0.0, "zero")
            blk[f"{G_COL}_excludes_zero"] = blk.pop(f"{G_COL}_ci_excludes_zero")
            row.update(blk)
            row.update(drift_block(G_COL, h1, h2, drift_tol,
                                   _ci_half(row, G_COL)))
            for s in SEEDS:
                row[f"pair_settled_s{s}"] = settled.get(s)
                row[f"pair_outcome_b0_s{s}"] = rb.get(f"pair_outcome_s{s}")
                row[f"pair_outcome_d8_s{s}"] = rd.get(f"pair_outcome_s{s}")
            row["t_a_sft_mean"] = rb.get(f"{T_A_COL}_mean")
            row["t_a_icl_mean"] = rd.get(f"{T_A_COL}_mean")
            row["horizon"] = "|".join(sorted({str(rb.get("horizon")),
                                              str(rd.get("horizon"))}))
            rows.append(row)
    return rows


def _member_reason(row, cond, s, drift_tol):
    def _fmt(v, sign=True):
        if v is None:
            return "NA"
        return f"{v:+.5f}" if sign else f"{v:.5f}"
    flags = row.get(f"settled_flags_{cond}_s{s}") or ""
    failed = [k for k in ("a", "b", "c") if f"{k}:F" in flags]
    verdict = ("passed (a)(b)(c)" if not failed
               else "FAILED " + "".join(f"({k})" for k in failed))
    txt = (f"{cond} {verdict} [a final-5 half-split drift "
           f"{_fmt(row.get(f'mu_b_drift_{cond}_s{s}'))}, b final-10 "
           f"half-split drift {_fmt(row.get(f'late10_drift_{cond}_s{s}'))} "
           f"(tol {drift_tol:g}), c final-5 range "
           f"{_fmt(row.get(f'late5_range_{cond}_s{s}'), sign=False)} "
           f"(tol {RANGE_TOL_MULT * drift_tol:g})]")
    if row.get(f"cyclic_{cond}_s{s}"):
        alt = row.get(f"cycle_alternation_{cond}_s{s}")
        txt += (f" CYCLIC (last-{CYCLE_WINDOW} difference sign alternation "
                f"{alt:.2f} >= {CYCLE_ALTERNATION_MIN:g})")
    return txt


def build_extension_request(source_rows, grid, drift_tol, run_root):
    """The section4_fig6_extension_request.json payload: BOTH members of
    every unsettled pair (matched, as gen_pofd_sweep.s4g2_ext_requests
    demands), each as {arm, cond, eps_ai, eps_social, seed, rounds,
    reason}; the reason names which of (a)/(b)/(c) each member failed and
    starts with "cyclic:" for a cyclic pair.  Pairs whose member is
    twin-derived cannot be extended by a GPU job and are listed under
    "twin_derived_unsettled"; pairs already at the last allowed horizon
    under "not_extendable" -- both OUTSIDE "cells"."""
    req, twin_unsettled, not_ext = [], [], []
    for row in source_rows:
        arm, ea, es = row["arm"], row["eps_ai"], row["eps_social"]
        for s in SEEDS:
            oc = row.get(f"pair_outcome_s{s}")
            if oc is None or oc == "equilibrium":
                continue
            h = int(row.get(f"pair_horizon_s{s}") or N_ROUNDS)
            reason = (("cyclic: " if oc == "cyclic" else "")
                      + "; ".join(_member_reason(row, c, s, drift_tol)
                                  for c in grid.conds)
                      + f"; analysed at {h} rounds; pair outcome {oc}")
            entry = {"arm": arm, "eps_ai": ea, "eps_social": es, "seed": s,
                     "outcome": oc, "reason": reason}
            nxt = [r for r in sorted(grid.ext_rounds_ok) if r > h]
            if row.get(f"pair_twin_derived_s{s}"):
                twin_unsettled.append(entry)
            elif not nxt:
                not_ext.append(entry)
            else:
                for cond in grid.conds:
                    req.append({"arm": arm, "cond": cond, "eps_ai": ea,
                                "eps_social": es, "seed": s,
                                "rounds": nxt[0], "reason": reason})
    return {
        "key": grid.key,
        "generated_by": "analyze_section4_gate.py --wave fig6",
        "run_root": run_root,
        "drift_tol": drift_tol,
        "settled_rule": (
            f"a cell is settled only if ALL hold: (a) |final-5 half-split "
            f"drift| <= tol, (b) |final-10 half-split drift| (final five "
            f"vs the five before) <= tol, (c) range of the five late "
            f"round means <= {RANGE_TOL_MULT}*tol; an unsettled cell whose "
            f"last {CYCLE_WINDOW} round-mean differences alternate in sign "
            f"on >= {CYCLE_ALTERNATION_MIN:.0%} of the steps is 'cyclic'; "
            f"a pair is unsettled when EITHER member is"),
        "pairing": ("both members of every unsettled pair are listed; "
                    "gen_pofd_sweep.s4g2_ext_requests() rejects an "
                    "unpaired request"),
        "n_cells": len(req),
        "cells": req,
        "twin_derived_unsettled": twin_unsettled,
        "not_extendable": not_ext,
        "note": ("twin_derived_unsettled pairs have an eps_AI = 0 member "
                 "derived from twin_raw (no GPU run to extend); "
                 "not_extendable pairs are unsettled at the last allowed "
                 "horizon; neither is in 'cells' and neither is ever "
                 "called an equilibrium"),
    }


def build_dispersion_rows(cells, drift_tol, grid=None):
    """C. DISPERSION: fixed vs evolving population SD and cohort-B SD on the
    same late window, with the paired fixed/evolving SD ratio."""
    g = _grid(grid)
    rows = []
    for arm in g.arms:
        for ea in g.gates:
            for es in g.ess:
                paired = [s for s in SEEDS
                          if (arm, "fixed", ea, es, s) in cells
                          and (arm, "evolving", ea, es, s) in cells]
                complete = len(paired) == len(SEEDS)
                acc = {k: {} for k in (
                    "sd_b_fixed", "sd_b_evolving", "delta_sd_b",
                    "sd_ratio_b", "sd_pop_fixed", "sd_pop_evolving",
                    "delta_sd_pop", "sd_ratio_pop",
                    "sd_ratio_twin_b_fixed", "sd_ratio_twin_b_evolving",
                    "ratio_b_h1", "ratio_b_h2",
                    "delta_sd_b_h1", "delta_sd_b_h2")}
                for s in paired:
                    f = cells[(arm, "fixed", ea, es, s)]
                    e = cells[(arm, "evolving", ea, es, s)]
                    acc["sd_b_fixed"][s] = f["sd_b_late"]
                    acc["sd_b_evolving"][s] = e["sd_b_late"]
                    acc["delta_sd_b"][s] = f["sd_b_late"] - e["sd_b_late"]
                    acc["sd_ratio_b"][s] = (
                        f["sd_b_late"] / e["sd_b_late"]
                        if e["sd_b_late"] > 0 else float("nan"))
                    acc["sd_pop_fixed"][s] = f["sd_pop_late"]
                    acc["sd_pop_evolving"][s] = e["sd_pop_late"]
                    acc["delta_sd_pop"][s] = (f["sd_pop_late"]
                                              - e["sd_pop_late"])
                    acc["sd_ratio_pop"][s] = (
                        f["sd_pop_late"] / e["sd_pop_late"]
                        if e["sd_pop_late"] > 0 else float("nan"))
                    acc["sd_ratio_twin_b_fixed"][s] = f["sd_ratio_late"]
                    acc["sd_ratio_twin_b_evolving"][s] = e["sd_ratio_late"]
                    acc["ratio_b_h1"][s] = (
                        f["sd_b_h1"] / e["sd_b_h1"]
                        if e["sd_b_h1"] > 0 else float("nan"))
                    acc["ratio_b_h2"][s] = (
                        f["sd_b_h2"] / e["sd_b_h2"]
                        if e["sd_b_h2"] > 0 else float("nan"))
                    acc["delta_sd_b_h1"][s] = f["sd_b_h1"] - e["sd_b_h1"]
                    acc["delta_sd_b_h2"][s] = f["sd_b_h2"] - e["sd_b_h2"]
                row = {"arm": arm, "arm_label": ARM_LABEL[arm],
                       "eps_ai": ea, "eps_social": es,
                       "n_seeds_paired": len(paired),
                       "seeds_paired": "|".join(str(s) for s in paired),
                       "status": "complete" if complete else "incomplete"}
                for pre, ref, nm in (
                        ("sd_b_fixed", 0.0, "zero"),
                        ("sd_b_evolving", 0.0, "zero"),
                        ("delta_sd_b", 0.0, "zero"),
                        ("sd_ratio_b", 1.0, "one"),
                        ("sd_pop_fixed", 0.0, "zero"),
                        ("sd_pop_evolving", 0.0, "zero"),
                        ("delta_sd_pop", 0.0, "zero"),
                        ("sd_ratio_pop", 1.0, "one"),
                        ("sd_ratio_twin_b_fixed", 1.0, "one"),
                        ("sd_ratio_twin_b_evolving", 1.0, "one")):
                    row.update(agg_block(pre, acc[pre], ref, nm))
                row.update(drift_block("sd_ratio_b", acc["ratio_b_h1"],
                                       acc["ratio_b_h2"], drift_tol,
                                       _ci_half(row, "sd_ratio_b")))
                row.update(drift_block("delta_sd_b", acc["delta_sd_b_h1"],
                                       acc["delta_sd_b_h2"], drift_tol,
                                       _ci_half(row, "delta_sd_b")))
                rows.append(row)
    return rows


def build_null_rows(cells, grid=None):
    """The d8 / eps_social = 0 STRUCTURAL NULL, inherited from
    analyze_bottom20_section4_3seed.py: frozen weights + own-history
    prompts + no peer step means NO cohort-A opinion can reach a cohort-B
    prompt, so the source effect on B is zero BY CONSTRUCTION.  The
    corrected AI gate changes what the gate is measured against, not who
    is in whose prompt, so the null still holds.

    Tolerance is hardware-aware: bit-exact (1e-9) when the fixed and
    evolving runs landed on the same GPU architecture, otherwise the
    measured residue IS this wave's greedy-generation nondeterminism floor
    and only fails above NULL_TOL_XHW.
    """
    g = _grid(grid)
    rows = []
    for ea in g.gates:
        for s in SEEDS:
            kf, ke = ("d8", "fixed", ea, 0.0, s), ("d8", "evolving", ea, 0.0, s)
            if kf not in cells or ke not in cells:
                rows.append({"seed": s, "eps_ai": ea, "delta_mu_b": None,
                             "delta_sd_b": None, "gpu_fixed": None,
                             "gpu_evolving": None, "hardware_matched": None,
                             "tol": None, "verdict": "MISSING"})
                continue
            f, e = cells[kf], cells[ke]
            dmu = f["mu_b_eq"] - e["mu_b_eq"]
            dsd = f["sd_b_late"] - e["sd_b_late"]
            hw_f, hw_e = f["gpu_arch"], e["gpu_arch"]
            matched = (hw_f == hw_e and hw_f != "unknown")
            tol = NULL_TOL if matched else NULL_TOL_XHW
            rows.append({"seed": s, "eps_ai": ea, "delta_mu_b": dmu,
                         "delta_sd_b": dsd, "gpu_fixed": hw_f,
                         "gpu_evolving": hw_e, "hardware_matched": matched,
                         "tol": tol,
                         "verdict": "PASS" if abs(dmu) <= tol else "FAIL"})
    return rows


# =============================================================== figures
def _pick(rows, arm, ea, es):
    for r in rows:
        if r["arm"] == arm and r["eps_ai"] == ea and r["eps_social"] == es:
            return r
    return None


def _flag_of(r, prefix):
    """The dagger flag for one series row: the inherited half-window
    drift flag of the statistic, OR (fig6 rows) a pair outcome that is
    not 'equilibrium' -- an unsettled series is never drawn as a level
    without the mark."""
    fl = bool(r.get(f"{prefix}_drift_flag"))
    oc = r.get("outcome")
    if oc is not None:
        fl = fl or (oc != "equilibrium")
    return fl


def _series(rows, arm, ea, prefix, grid=None):
    """(x positions, y, yerr_lo, yerr_hi, flagged) for one arm at one
    eps_ai.  Incomplete series are DROPPED, never plotted as if they were
    three-seed results."""
    g = _grid(grid)
    xs, ys, lo, hi, flag = [], [], [], [], []
    for j, es in enumerate(g.ess):
        r = _pick(rows, arm, ea, es)
        if r is None or r["status"] != "complete":
            continue
        m = r.get(f"{prefix}_mean")
        if m is None or not math.isfinite(m):
            continue
        xs.append(j)
        ys.append(m)
        lo.append(m - r[f"{prefix}_ci_lo"])
        hi.append(r[f"{prefix}_ci_hi"] - m)
        flag.append(_flag_of(r, prefix))
    return xs, ys, lo, hi, flag


def _style_panel(ax, ylabel=None, xlabel=True, grid=None):
    g = _grid(grid)
    ax.set_xlim(-0.45, len(g.ess) - 0.55)
    ax.set_xticks(range(len(g.ess)))
    ax.set_xticklabels([f"{e:g}" for e in g.ess])
    ax.grid(axis="y", color=GRID_GREY, lw=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.6, length=2.4, width=0.6)
    if xlabel:
        ax.set_xlabel(r"$\varepsilon_{\mathrm{social}}$", fontsize=9.4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.2)


def _panel_tag(ax, text):
    """Panel identity as an ANNOTATION above the axes.  The project
    convention is that paper figures carry NO title text: no set_title,
    no suptitle, anywhere in this module."""
    ax.annotate(text, xy=(0.5, 1.02), xycoords="axes fraction",
                ha="center", va="bottom", fontsize=9.4, color=INK)


def _plot_series(ax, rows, ea, prefix, dodge=0.09, lw=1.4, ls="-",
                 label_suffix="", grid=None):
    """Draw both arms at one eps_AI. Returns True if any drift flag was
    marked, and the number of series actually drawn."""
    g = _grid(grid)
    drew_flag, drew = False, 0
    for arm in g.arms:
        off = -dodge if arm == "b0" else dodge
        xs, ys, lo, hi, flag = _series(rows, arm, ea, prefix, g)
        if not xs:
            continue
        xs = [x + off for x in xs]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=ARM_COLOR[arm],
                    marker=ARM_MARKER[arm], ms=4.6, lw=lw, ls=ls,
                    capsize=2.6, elinewidth=0.9,
                    label=ARM_LABEL[arm] + label_suffix, zorder=3)
        drew += 1
        for x, y, fl in zip(xs, ys, flag):
            if fl:
                ax.annotate("\u2020", xy=(x, y), xytext=(0, 7),
                            textcoords="offset points", ha="center",
                            va="bottom", fontsize=9, color=ARM_COLOR[arm])
                drew_flag = True
    return drew_flag, drew


def _rc():
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.6,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _foot(fig, text, y):
    """Wrapped footer note under the axes.  A footer is NOT a title: the
    figures carry no title text, and the narrative lives in the printed
    caption block."""
    fig.text(0.005, y, "\n".join(textwrap.wrap(text, 118)),
             fontsize=6.9, ha="left", va="top", color=INK)


def _save(fig, out_dir, stem):
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=320 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.03)
        paths.append(p)
    print(f"[s4g] wrote {stem}.pdf/.png")
    return paths


def figure_source(rows, out_dir, cover_note, drift_tol, grid=None):
    g = _grid(grid)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _rc()
    n_p = len(g.gates)
    fig, axes = plt.subplots(1, n_p, figsize=(1.8 * n_p + 3.6, 3.0),
                             sharey=True, squeeze=False)
    flagged, n_drawn = False, 0
    if g.fig6:
        prefix = T_A_COL
        ylab = (r"$T_a=\mu_B^{\mathrm{eq}}(\mathrm{A\ evolving})"
                r"-\mu_B^{\mathrm{eq}}(\mathrm{A\ fixed})$")
    else:
        prefix = "delta_mu_b"
        ylab = (r"$\mu_B^{\mathrm{late}}(\mathrm{fixed})"
                r"-\mu_B^{\mathrm{late}}"
                r"(\mathrm{evolving})$")
    for i, ea in enumerate(g.gates):
        ax = axes[0][i]
        ax.axhline(0.0, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=1)
        fl, nd = _plot_series(ax, rows, ea, prefix, grid=g)
        flagged |= fl
        n_drawn += nd
        _style_panel(ax, ylabel=ylab if i == 0 else None, grid=g)
        _panel_tag(ax, r"$\varepsilon_{\mathrm{AI}}=%g$" % ea)
    if n_drawn:
        axes[0][0].legend(frameon=False, fontsize=8.2, loc="best")
    else:
        axes[0][0].annotate("no complete series", xy=(0.5, 0.5),
                            xycoords="axes fraction", ha="center",
                            va="center", fontsize=9, color="0.45")
    foot = (f"post-peer late window = the final five rounds of each "
            f"artifact (op_raw {LATE_IDX[0]}-{LATE_IDX[-1]} at 30 rounds); "
            f"error bars = 95% paired Student-t interval over seeds "
            f"{', '.join(str(s) for s in SEEDS)} (df=2); {cover_note}"
            if g.fig6 else
            f"post-peer late window = op_raw rounds "
            f"{LATE_IDX[0]}-{LATE_IDX[-1]}; error bars = 95% Student-t "
            f"interval over seeds {', '.join(str(s) for s in SEEDS)} "
            f"(df=2); {cover_note}")
    if flagged:
        foot += (f"; \u2020 = half-window drift exceeds {drift_tol:g} "
                 f"{'or the pair is unsettled ' if g.fig6 else ''}"
                 f"(see the CSV)")
    fig.tight_layout()
    _foot(fig, foot, -0.035)
    return _save(fig, out_dir, f"{g.stem}_source_effect")


def figure_dispersion(rows, out_dir, cover_note, drift_tol, grid=None):
    g = _grid(grid)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _rc()
    n_p = len(g.gates)
    fig, axes = plt.subplots(2, n_p, figsize=(1.8 * n_p + 3.6, 5.2),
                             sharey="row", sharex=True, squeeze=False)
    flagged, n_drawn = False, 0
    specs = [("sd_ratio_b", r"SD$_B$(fixed) / SD$_B$(evolving)"),
             ("sd_ratio_pop", r"SD$_{\mathrm{pop}}$(fixed) / "
                              r"SD$_{\mathrm{pop}}$(evolving)")]
    for r_i, (prefix, ylab) in enumerate(specs):
        for i, ea in enumerate(g.gates):
            ax = axes[r_i][i]
            ax.axhline(1.0, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=1)
            fl, nd = _plot_series(ax, rows, ea, prefix, grid=g)
            flagged |= fl
            n_drawn += nd
            _style_panel(ax, ylabel=ylab if i == 0 else None,
                         xlabel=(r_i == len(specs) - 1), grid=g)
            if r_i == 0:
                _panel_tag(ax, r"$\varepsilon_{\mathrm{AI}}=%g$" % ea)
    if n_drawn:
        axes[0][0].legend(frameon=False, fontsize=8.2, loc="best")
    else:
        axes[0][0].annotate("no complete series", xy=(0.5, 0.5),
                            xycoords="axes fraction", ha="center",
                            va="center", fontsize=9, color="0.45")
    foot = (f"post-peer late window = op_raw rounds "
            f"{LATE_IDX[0]}-{LATE_IDX[-1]}; ratios formed PER SEED then "
            f"averaged; error bars = 95% Student-t interval over seeds "
            f"{', '.join(str(s) for s in SEEDS)} (df=2); {cover_note}")
    if flagged:
        foot += f"; \u2020 = half-window drift exceeds {drift_tol:g}"
    fig.tight_layout()
    _foot(fig, foot, -0.018)
    return _save(fig, out_dir, f"{g.stem}_dispersion")


# =============================================================== captions
def caption_source(source_rows, cover_note, drift_tol, partial, shape):
    n_sig = sum(1 for r in source_rows
                if r["status"] == "complete"
                and r.get("delta_mu_b_ci_excludes_zero"))
    n_done = sum(1 for r in source_rows if r["status"] == "complete")
    return [
        "CAPTION -- section4_gate_source_effect.pdf/.png "
        "(the figure carries no title text)",
        "",
        "Source effect of a non-adapting source cohort under the CORRECTED",
        f"AI gate. Mistral-7B on movielens Action, {shape['n_agents']} "
        f"agents, {shape['n_rounds']} rounds,",
        "W=0.5, k=0.2, homophily gamma=0, AI gate measured against the",
        "ANCHORED opinion x' = k*innate + (1-k)*x (config",
        f"population_update = {POP_UPDATE_V2}).",
        "Each point is the late-window cohort-B mean opinion in the FIXED",
        f"condition (the {shape['n_a']} lowest-innate agents pinned at "
        f"their innate",
        "value, one-sided stubborn peer operator) MINUS the same quantity",
        "in the fully EVOLVING condition, so a positive value means the",
        "non-adapting source cohort left the responsive majority HIGHER",
        "than a fully adaptive population did.",
        f"Cohort B = the other {shape['n_b']} agents;",
        f"in the evolving condition the same {shape['n_a']}-agent cohort A "
        f"is an",
        "ANALYSIS MASK only, reconstructed from the bit-identical innate",
        "vector so both members of a pair are masked identically.",
        f"The late window is op_raw rounds {LATE_IDX[0]}-{LATE_IDX[-1]}, the",
        "last five END-OF-ROUND POST-PEER states (peer sweeps run last",
        "inside a round). Panels are the two AI gates; series are the two",
        "arms; error bars are 95% Student-t intervals over the three seeds",
        f"({', '.join(str(s) for s in SEEDS)}, df=2, t={T_CRIT_DF2:.4f}).",
        f"{n_sig} of {n_done} complete cells have an interval excluding 0.",
        "A \u2020 marks a series whose late-window half-to-half drift "
        "exceeds",
        f"{drift_tol:g} opinion units, i.e. one that has not settled and",
        "should not be read as a level.",
        f"Coverage: {cover_note}."
        + (" RESULTS ARE PARTIAL." if partial else ""),
        "Exploratory: one wave, three seeds, one model, one dataset.",
    ]


def caption_fig6_source(source_rows, cover_note, drift_tol, shape, grid,
                        gate_info):
    """The fig6 caption block: T_a with its sign convention stated, the
    eps_AI = 0 twin derivation, the settled rule, every unsettled pair by
    name, and the served-value cardinality summary."""
    n_done = sum(1 for r in source_rows if r["status"] == "complete")
    n_sig = sum(1 for r in source_rows if r["status"] == "complete"
                and r.get(f"{T_A_COL}_ci_excludes_zero"))
    unsettled = [r for r in source_rows if r.get("outcome") != "equilibrium"]
    # eps_AI = 0 is exempt: the witness runs did generate, but the closed
    # gate let nothing served enter the population
    quantized = [r for r in source_rows
                 if r["eps_ai"] != 0.0
                 and any(isinstance(r.get(f"served_distinct_{c}_min"), int)
                         and r[f"served_distinct_{c}_min"] <= 3
                         for c in grid.conds)]
    lines = [
        f"CAPTION -- {grid.stem}_source_effect.pdf/.png "
        "(the figure carries no title text)",
        "",
        "Figure-6 quantity: T_a = mu_B^eq(A evolving) - mu_B^eq(A fixed),",
        "EVOLVING MINUS FIXED. A positive T_a means a fully adaptive",
        "cohort A left the responsive majority (cohort B) HIGHER than a",
        "pinned cohort A did; the inherited delta_mu_b = -T_a is in the CSV.",
        f"Mistral-7B on movielens Action, {shape['n_agents']} agents, "
        f"W=0.5, k=0.2,",
        "homophily gamma=0, corrected AI gate on the anchored opinion",
        f"x' = k*innate + (1-k)*x (population_update = {POP_UPDATE_V2}).",
        f"Cohort A = the {shape['n_a']} lowest-innate agents (pinned in "
        f"the FIXED",
        "condition, an ANALYSIS MASK in the EVOLVING one); cohort B = the",
        f"other {shape['n_b']} agents.",
        "Equilibrium = the FINAL FIVE end-of-round post-peer states of the",
        "analysed artifact (op_raw rounds 25-29 = rounds 26-30 of a",
        "30-round run; 56-60 of a _r60 extension; 96-100 of a _r100).",
        "T_a is formed PER PAIRED SEED, then the three-seed mean and the",
        f"95% paired Student-t interval (df=2, t={T_CRIT_DF2:.4f}) over "
        f"the",
        "per-seed differences are the point and the error bar.",
        "eps_AI = 0 is the matched no-AI twin: the strict gate |m - x'| <",
        "eps_AI is closed for everyone, the method drops out, and the",
        "population is twin_raw of the runs at the same (condition,",
        "eps_social, seed) -- verified bit-identical across those runs and",
        "against the two witness runs (op_raw == twin_raw). The eps_AI = 0",
        "value is therefore IDENTICAL for both methods (pure peer",
        "transmission baseline).",
        f"Panels are the {len(grid.gates)} AI gates "
        f"({', '.join(f'{e:g}' for e in grid.gates)}); x is eps_social "
        f"({', '.join(f'{e:g}' for e in grid.ess)});",
        "series are the two methods.",
        f"{n_sig} of {n_done} complete series have an interval excluding 0.",
        "A pair is SETTLED when both members' late-window half-to-half",
        f"drift (2 vs 3 rounds) is within {drift_tol:g} opinion units; "
        f"an unsettled",
        "pair is never called an equilibrium and is marked \u2020.",
    ]
    if unsettled:
        lines.append(f"UNSETTLED series ({len(unsettled)}): " + "; ".join(
            f"{r['arm']} ea={r['eps_ai']:g} es={r['eps_social']:g} "
            f"[{r['outcome']}]" for r in unsettled))
        lines.append("Extension requests for these pairs are in "
                     f"{grid.stem}_extension_request.json.")
    else:
        lines.append("Every pair is settled at the analysed horizon.")
    n_cyc = sum(1 for r in unsettled if "cyclic" in (r.get("outcome") or ""))
    lines.append(
        "SETTLED requires all of (a) final-5 half-split drift, (b) final-10")
    lines.append(
        f"half-split drift (rounds 26-30 vs 21-25) within {drift_tol:g}, and "
        f"(c) the")
    lines.append(
        f"range of the five late round means within {RANGE_TOL_MULT * drift_tol:g}; "
        f"a sign-alternating")
    lines.append(
        f"unsettled cell is CYCLIC ({n_cyc} cyclic series here), never an "
        f"equilibrium.")
    lines.append(
        "Paired method gap G = T_a(SFT) - T_a(ICL) per seed (positive = SFT's")
    lines.append(
        f"source effect exceeds ICL's; G = 0 at eps_AI = 0 by construction) "
        f"is in {grid.stem}_method_gap.csv.")
    if quantized:
        lines.append(
            f"SERVED-VALUE QUANTIZATION: {len(quantized)} series have a "
            f"member whose late-window served map holds <= 3 distinct "
            f"values (served_distinct / served_top_share in the CSV): a "
            f"null T_a there may be quantization, not absence of effect.")
    else:
        lines.append("No series has a member with <= 3 distinct served "
                     "values in the late window (served_distinct / "
                     "served_top_share in the CSV).")
    lines += [
        f"Gate verdict: {gate_info}.",
        f"Coverage: {cover_note}.",
        "Exploratory: one wave, three seeds, one model, one dataset.",
    ]
    return lines


def caption_dispersion(disp_rows, cover_note, drift_tol, partial, shape,
                       grid=None):
    g = _grid(grid)
    n_sig = sum(1 for r in disp_rows
                if r["status"] == "complete"
                and r.get("sd_ratio_b_ci_excludes_one"))
    n_done = sum(1 for r in disp_rows if r["status"] == "complete")
    return [
        f"CAPTION -- {g.stem}_dispersion.pdf/.png "
        "(the figure carries no title text)",
        "",
        "Dispersion under a fixed versus an evolving source cohort, same",
        "corrected-gate wave. Top row: the ratio of late-window cohort-B",
        "opinion SD in the FIXED condition to the same SD in the EVOLVING",
        "condition. Bottom row: the same ratio for the FULL population.",
        "Ratios are formed WITHIN a seed and then averaged, so the interval",
        "is a paired three-seed interval; a value above 1 means the",
        "non-adapting source cohort left MORE opinion spread than a fully",
        "adaptive population did, below 1 means less.",
        "The full-population row is reported for completeness but is",
        f"MECHANICALLY CONFOUNDED: in the fixed condition {shape['n_a']} of "
        f"{shape['n_agents']} agents",
        "are pinned at innate by construction, which by itself changes the",
        "population SD. Cohort B is the honest dispersion channel.",
        f"Late window = op_raw rounds {LATE_IDX[0]}-{LATE_IDX[-1]} (the last",
        "five end-of-round post-peer states"
        + (" of a 30-round artifact; the final five of an extension)."
           if g.fig6 else ")."),
        f"Panels are the {len(g.gates)} AI gates,",
        "series the two arms, error bars 95% Student-t intervals over the",
        f"three seeds (df=2, t={T_CRIT_DF2:.4f}). Dashed line: ratio = 1.",
        f"{n_sig} of {n_done} complete cells have a cohort-B SD-ratio",
        "interval excluding 1.",
        f"A \u2020 marks a series whose half-to-half drift exceeds "
        f"{drift_tol:g}.",
        "The twin-referenced dispersion of the published Section 4 --",
        "SD(B platform)/SD(B matched no-platform twin), per condition --",
        f"is in {g.stem}_dispersion.csv rather than this figure.",
        f"Coverage: {cover_note}."
        + (" RESULTS ARE PARTIAL." if partial else ""),
        "Exploratory: one wave, three seeds, one model, one dataset.",
    ]


# =================================================================== main
def default_out_dir(run_root, grid=None):
    """A runs-ADJACENT analysis directory: a sibling of the run root, never
    inside it (so a tag scan cannot trip over the analysis) and NEVER under
    paper/."""
    parent = os.path.dirname(os.path.abspath(run_root.rstrip(os.sep)))
    return os.path.join(parent, "analysis", _grid(grid).key)


def read_gate_verdict(path):
    """(ok, info) from a check_section4_gate.py --json verdict.  The
    checker writes the top-level verdict as ``pass`` (and ``ok`` per
    cell); both spellings are accepted at the top level.  ok is False
    for a missing / unreadable / verdict-less file."""
    if path is None:
        return None, "no --gate-json given"
    if not os.path.exists(path):
        return False, f"--gate-json {path!r} does not exist"
    try:
        with open(path) as fh:
            js = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        return False, f"--gate-json {path!r} unreadable: {e}"
    if not isinstance(js, dict):
        return False, f"--gate-json {path!r} is not a JSON object"
    v = js.get("ok", js.get("pass"))
    if v is None:
        return False, (f"--gate-json {path!r} carries neither 'ok' nor "
                       f"'pass'")
    info = (f"{'ok' if v else 'FAILED'} (wave={js.get('wave')!r}, "
            f"cells present {js.get('n_cells_present')}/"
            f"{js.get('n_cells_total')}, failed {js.get('n_cells_failed')}, "
            f"{os.path.basename(path)})")
    return bool(v), info


def refuse_paper_dir(out_dir):
    parts = {p.lower() for p in os.path.abspath(out_dir).split(os.sep)}
    if "paper" in parts:
        print(f"[s4g] REFUSING --out-dir {out_dir!r}: analysis artifacts "
              f"never go under paper/", file=sys.stderr)
        sys.exit(1)


def print_table(rows, title, mean_key, mark_key, fmt="%+.4f", grid=None):
    g = _grid(grid)
    prefix = mean_key[:-5]
    print(f"\n== {title} ==")
    print("   " + " ".join(f"{'es=' + f'{e:g}':>12}" for e in g.ess))
    for arm in g.arms:
        for ea in g.gates:
            cellstr = []
            for es in g.ess:
                r = _pick(rows, arm, ea, es)
                if r is None or r["status"] != "complete" \
                        or r.get(mean_key) is None:
                    cellstr.append(f"{'--':>12}")
                    continue
                s = fmt % r[mean_key]
                s += "*" if (mark_key and r.get(mark_key)) else " "
                s += "\u2020" if _flag_of(r, prefix) else " "
                cellstr.append(f"{s:>12}")
            print(f"  {arm:<3} ea={ea:<4g}" + " ".join(cellstr))
    print("  (* = 95% CI excludes the reference; \u2020 = half-window "
          "drift exceeds the tolerance"
          + (" or the pair is unsettled)" if g.fig6 else ")"))


def print_method_gap_table(gap_rows, grid):
    print(f"\n== paired method gap G = T_a(SFT) - T_a(ICL) [{G_COL}] ==")
    print(f"  {G_SIGN}")
    print("        " + " ".join(f"{'es=' + f'{e:g}':>12}" for e in grid.ess))
    for ea in grid.gates:
        cellstr = []
        for es in grid.ess:
            r = next((x for x in gap_rows if x["eps_ai"] == ea
                      and x["eps_social"] == es), None)
            m = None if r is None else r.get(f"{G_COL}_mean")
            if r is None or r["status"] != "complete" or m is None:
                cellstr.append(f"{'--':>12}")
                continue
            s = f"{m:+.4f}"
            s += "*" if r.get(f"{G_COL}_excludes_zero") else " "
            s += "\u2020" if _flag_of(r, G_COL) else " "
            cellstr.append(f"{s:>12}")
        print(f"  ea={ea:<4g}" + " ".join(cellstr))
    print("  (* = 95% paired CI excludes 0; \u2020 = drift flag or an "
          "unsettled member pair; G == 0 at ea=0 by construction)")


def print_fig6_detail(rows, grid):
    """T_a per series NEXT TO the settled verdict and the served-value
    cardinality of both members, so a quantized served map can never
    masquerade as a null effect."""
    print(f"\n== FIG6 detail: {T_A_COL} (mean [95% paired CI]) with the "
          f"settled verdict and late-window served cardinality ==")
    print(f"  {T_A_SIGN}")
    hdr = (f"  {'arm':<3} {'ea':>4} {'es':>4} {'T_a':>9} "
           f"{'[ci_lo, ci_hi]':>22} {'MAE_B':>8} {'outcome':>22} {'hz':>6} "
           f"{'distinct f|e':>16} {'top-share f|e':>18}")
    print(hdr)
    warn = []
    for arm in grid.arms:
        for ea in grid.gates:
            for es in grid.ess:
                r = _pick(rows, arm, ea, es)
                if r is None:
                    continue
                m = r.get(f"{T_A_COL}_mean")
                if m is None:
                    t_s, ci_s = "--", "--"
                else:
                    t_s = f"{m:+.4f}"
                    ci_s = (f"[{r[f'{T_A_COL}_ci_lo']:+.4f}, "
                            f"{r[f'{T_A_COL}_ci_hi']:+.4f}]")
                oc = r.get("outcome") or "--"
                mv = r.get(f"{MAE_COL}_mean")
                mae_s = "--" if mv is None else f"{mv:.4f}"
                dist = (f"{r.get('served_distinct_fixed', '')}|"
                        f"{r.get('served_distinct_evolving', '')}")
                top = (f"{r.get('served_top_share_fixed', '')}|"
                       f"{r.get('served_top_share_evolving', '')}")
                dist = dist.replace(SERVED_NA, "n/a")
                top = top.replace(SERVED_NA, "n/a")
                print(f"  {arm:<3} {ea:>4g} {es:>4g} {t_s:>9} {ci_s:>22} "
                      f"{mae_s:>8} {oc:>22} {r.get('horizon', ''):>6} "
                      f"{dist:>16} {top:>18}")
                mins = [r.get(f"served_distinct_{c}_min")
                        for c in grid.conds]
                mins = [v for v in mins if isinstance(v, int)]
                if ea != 0.0 and mins and min(mins) <= 3:
                    warn.append(f"{arm} ea={ea:g} es={es:g} "
                                f"(min distinct {min(mins)})")
    print(f"  MAE_B = {MAE_COL} three-seed mean (agent-paired cohort-B "
          f"MAE, fixed vs evolving; see the MAE table)")
    print("  distinct = number of distinct pred_raw values pooled over the "
          "late window (per seed, f|e = fixed|evolving); top-share = "
          "fraction at the single most common value; n/a = eps_AI=0 "
          "twin-derived (gate closed, nothing served; a witness run's map "
          "is reported but exempt from the warning below)")
    if warn:
        print(f"  ***** SERVED MAP QUANTIZED (<= 3 distinct values) in "
              f"{len(warn)} series: " + "; ".join(warn)
              + " -- a null T_a there may be quantization *****")


def locate_cells(run_root, grid):
    """Existence pass (no tensors).  Returns (located, missing, coverage,
    base_runs).

    located[key] = {run_dir, tag, horizon, kind, analysed_from,
                    derived_from}; for a gpu/witness cell the LONGEST
    available horizon (_r100 > _r60 > base) is chosen; a twin cell with
    no run is resolved to analysed_from='twin_raw' from ANY base-horizon
    run at the same (cond, eps_social, seed); a twin cell that does have
    a run is analysed from it as a witness (noted).
    base_runs[(cond, es, seed)] = [(key, run_dir, tag)] of the base-tag
    artifacts, which feed the twin derivation and the twin-agreement
    check even when the cell itself is analysed at a longer horizon.
    """
    located, missing, coverage, base_runs = {}, [], [], {}
    for (arm, cond, ea, es, seed, kind) in grid.cells:
        key = (arm, cond, ea, es, seed)
        base_tag = grid.tag(arm, cond, ea, es, seed)
        base_rd = find_run(run_root, base_tag)
        if base_rd is not None:
            base_runs.setdefault((cond, es, seed), []).append(
                (key, base_rd, base_tag))
        found = None
        if kind == "twin":
            if base_rd is not None:
                found = {"run_dir": base_rd, "tag": base_tag,
                         "horizon": N_ROUNDS, "kind": kind,
                         "analysed_from": "op_raw", "derived_from": None,
                         "note": "twin cell WITH a run: verified as a "
                                 "witness"}
        else:
            for r in grid.horizons():
                tag = base_tag if r is None else grid.tag(arm, cond, ea, es,
                                                          seed, rounds=r)
                rd = base_rd if r is None else find_run(run_root, tag)
                if rd is not None:
                    found = {"run_dir": rd, "tag": tag,
                             "horizon": N_ROUNDS if r is None else int(r),
                             "kind": kind, "analysed_from": "op_raw",
                             "derived_from": None, "note": None}
                    break
            if found is None:
                missing.append(base_tag)
        if found is not None:
            located[key] = found
    # twin-derived cells: any base run at the same (cond, es, seed)
    for (arm, cond, ea, es, seed, kind) in grid.cells:
        key = (arm, cond, ea, es, seed)
        if kind != "twin" or key in located:
            continue
        srcs = base_runs.get((cond, es, seed), [])
        tag = grid.tag(arm, cond, ea, es, seed)
        if not srcs:
            missing.append(f"{tag} [twin-derived: no base run at "
                           f"({cond}, es={es:g}, seed={seed})]")
            continue
        located[key] = {"run_dir": None, "tag": tag, "horizon": N_ROUNDS,
                        "kind": kind, "analysed_from": "twin_raw",
                        "derived_from": srcs[0][2], "note": None}
    for (arm, cond, ea, es, seed, kind) in grid.cells:
        key = (arm, cond, ea, es, seed)
        loc = located.get(key)
        coverage.append({
            "arm": arm, "cond": cond, "eps_ai": ea, "eps_social": es,
            "seed": seed, "kind": kind,
            "run_tag": loc["tag"] if loc else grid.tag(arm, cond, ea, es,
                                                       seed),
            "present": loc is not None,
            "horizon": loc["horizon"] if loc else None,
            "analysed_from": loc["analysed_from"] if loc else None,
            "run_dir": loc["run_dir"] if loc else None,
            "derived_from": loc["derived_from"] if loc else None,
            "note": loc["note"] if loc else None})
    return located, missing, coverage, base_runs


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Section-4 corrected-gate analyzer: fixed vs evolving "
                    "source cohort under the anchored AI gate, for the "
                    "original 72-cell wave (default) or the Figure-6 grid "
                    "(--wave fig6).")
    ap.add_argument("--wave", default=KEY, choices=sorted(WAVE_ALIASES),
                    help=f"which wave's grid to analyse (default {KEY}; "
                         f"aliases v1 / fig6)")
    ap.add_argument("--run-root",
                    default=os.path.join(REPO, "runs", "pokec_gated_lm"),
                    help="directory holding <tag>/trajectory.pt "
                         "(cluster: /home/gsmithline/perfsim/runs/"
                         "pokec_gated_lm)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: a runs-adjacent "
                         "analysis dir; never under paper/)")
    ap.add_argument("--no-figs", action="store_true",
                    help="CSVs, captions and report only")
    ap.add_argument("--json", default=None,
                    help="write a machine-readable summary here (fig6 "
                         "mode always also writes <out-dir>/"
                         "section4_fig6_summary.json)")
    ap.add_argument("--drift-tol", type=float, default=DEFAULT_DRIFT_TOL,
                    help=f"half-window drift tolerance in opinion units "
                         f"(default {DEFAULT_DRIFT_TOL:g})")
    ap.add_argument("--gate-json", default=None,
                    help="check_section4_gate.py --json verdict; REQUIRED "
                         "and must pass in fig6 mode, optional otherwise")
    ap.add_argument("--allow-ungated", action="store_true",
                    help="original wave only: go on when --gate-json "
                         "reports a failed verdict (REJECTED in fig6 mode)")
    args = ap.parse_args(argv)

    g = GRIDS[WAVE_ALIASES[args.wave]]
    stem = g.stem
    run_root = os.path.abspath(args.run_root)
    out_dir = os.path.abspath(args.out_dir if args.out_dir
                              else default_out_dir(run_root, g))
    refuse_paper_dir(out_dir)
    print(f"[s4g] key      : {g.key}  ({g.n_cells} cells: "
          + ", ".join(f"{g.n_kind[k]} {k}" for k in CELL_KINDS
                      if g.n_kind[k]) + ")")
    print(f"[s4g] grid     : ea {g.gates} x es {g.ess} x seeds {g.seeds} "
          f"(read from gen_pofd_sweep.py)")
    print(f"[s4g] run-root : {run_root}")
    print(f"[s4g] out-dir  : {out_dir}")
    print(f"[s4g] window   : op_raw rounds {LATE_IDX[0]}-{LATE_IDX[-1]} "
          f"(END-OF-ROUND POST-PEER states; LATE = range(25, 30), "
          f"inherited from analyze_bottom20_section4_3seed.py"
          + ("; = rounds 26-30 1-indexed; an extension artifact is read "
             "on ITS final five rounds)" if g.fig6 else ")"))
    print(f"[s4g] t crit   : {T_CRIT_SOURCE}")
    if g.fig6:
        print(f"[s4g] SIGN     : {T_A_SIGN}")

    # ---- 0. the gate verdict
    if g.fig6 and args.allow_ungated:
        print("[s4g] HARD FAIL: --allow-ungated is not accepted in fig6 "
              "mode -- the Figure-6 numbers are never computed on an "
              "ungated wave", file=sys.stderr)
        return 1
    gate_ok, gate_info = read_gate_verdict(args.gate_json)
    if g.fig6:
        if args.gate_json is None:
            print("[s4g] HARD FAIL: fig6 mode requires --gate-json (the "
                  "check_section4_gate.py verdict) and it must pass",
                  file=sys.stderr)
            return 1
        print(f"[s4g] gate     : {gate_info}")
        if not gate_ok:
            print(f"[s4g] HARD FAIL: the gate verdict is not a pass -- "
                  f"nothing is written ({gate_info})", file=sys.stderr)
            return 1
    else:
        if args.gate_json is None:
            print("[s4g] gate     : none given (the original wave accepts "
                  "an ungated run; pass --gate-json to pin it)")
        else:
            print(f"[s4g] gate     : {gate_info}")
            if not gate_ok and not args.allow_ungated:
                print(f"[s4g] HARD FAIL: the gate verdict is not a pass "
                      f"({gate_info}); use --allow-ungated to override "
                      f"in the original wave", file=sys.stderr)
                return 1
            if not gate_ok:
                print("[s4g] ***** --allow-ungated: proceeding on a FAILED "
                      "gate verdict; the numbers are suspect *****")

    # ---- 1. locate every required cell (existence only; no tensors yet)
    located, missing, coverage, base_runs = locate_cells(run_root, g)
    partial = bool(missing)
    n_from_run = sum(1 for v in located.values()
                     if v["analysed_from"] == "op_raw")
    n_from_twin = len(located) - n_from_run
    n_ext = sum(1 for v in located.values() if v["horizon"] != N_ROUNDS)
    print(f"[s4g] located  : {len(located)}/{g.n_cells} cells "
          f"({n_from_run} from a run, {n_from_twin} twin-derived, "
          f"{n_ext} at an extended horizon)")
    for tag in missing:
        print(f"  MISSING {tag}")
    for key in g.keys:
        loc = located.get(key)
        if loc and loc.get("note"):
            print(f"  NOTE {loc['tag']}: {loc['note']}")

    # tags that exist on disk but are not in the declared grid
    stray = [f["tag"] for f in scan_run_root(run_root, g)
             if not f.get("in_grid")]
    if stray:
        print(f"[s4g] NOTE: {len(stray)} {TAG_PREFIX}_ run(s) under the run "
              f"root are NOT in the declared {g.n_cells}-cell grid "
              f"(grammar drift or a smoke wave?):")
        for t in stray[:20]:
            print(f"  UNEXPECTED {t}")

    if g.fig6 and missing:
        print(f"\n[s4g] HARD FAIL: {len(missing)} required cell(s) absent "
              f"-- fig6 mode allows NO partial output", file=sys.stderr)
        for tag in missing:
            print(f"  {tag}", file=sys.stderr)
        return 1
    if not located:
        print(f"[s4g] HARD FAIL: no cell of {g.key} found under {run_root} "
              f"-- nothing to analyse", file=sys.stderr)
        return 1

    # ---- 2. one pass, ONE run's tensors resident at a time
    per_round_rows, cells, fatal = [], {}, []
    state = {"ref_sha": None, "mask_a": None}
    twin_shas = {}          # (cond, es, seed) -> {tag: sha256(twin_raw)}
    derived = {}            # (cond, es, seed) -> (rounds, late, tag, arch)
    win_b = {}              # cell key -> op_b_window (5 x n_B floats)
    twin_needed = {(k[1], k[3], k[4]) for k, v in located.items()
                   if v["analysed_from"] == "twin_raw"}

    def init_ref(d, tag):
        state["mask_a"] = cohort_a_mask(d["innate"])
        state["ref_sha"] = innate_sha(d["innate"])
        n_ag, n_a = int(d["innate"].numel()), int(state["mask_a"].sum())
        print(f"[s4g] cohort A : {n_a} of {n_ag} agents "
              f"(bottom {CLAMP_FRAC:g} by the innate-then-id ranking; "
              f"reference run {tag})")
        if n_ag != EXPECTED_N_AGENTS or n_a != EXPECTED_N_CLAMP:
            print(f"[s4g] NOTE: this wave is specified at "
                  f"{EXPECTED_N_AGENTS} agents / {EXPECTED_N_CLAMP} "
                  f"clamped; this run has {n_ag} / {n_a}")

    def note_twin(d, key, tag, run_dir):
        """Record a BASE-horizon run's twin sha; derive the eps_AI = 0
        cells at its (cond, es, seed) from its twin_raw if not yet done."""
        cse = (key[1], key[3], key[4])
        sha = twin_sha(d)
        if sha is None:
            fatal.append(f"{tag}: no twin_raw shaped like op_raw -- fig6 "
                         f"needs the matched no-AI twin on every run "
                         f"(the eps_AI = 0 cells are derived from it)")
            return
        twin_shas.setdefault(cse, {})[tag] = sha
        if cse in twin_needed and cse not in derived:
            tkey = next(k for k, v in located.items()
                        if v["analysed_from"] == "twin_raw"
                        and (k[1], k[3], k[4]) == cse)
            dd = twin_derived_artifact(d)
            dtag = f"{g.tag(*tkey)} [twin-derived from {tag}]"
            bad = structural_checks(dd, tkey, state["mask_a"],
                                    state["ref_sha"], tag=dtag,
                                    horizon=N_ROUNDS)
            if bad:
                fatal.extend(bad)
                return
            rounds, late = reduce_cell(dd, state["mask_a"],
                                       late_window(N_ROUNDS))
            derived[cse] = (rounds, late, tag, gpu_arch(run_dir), sha,
                            op_b_window(dd, state["mask_a"],
                                        late_window(N_ROUNDS)))

    for key in g.keys:
        loc = located.get(key)
        if loc is None or loc["analysed_from"] != "op_raw":
            continue
        rd, tag, horizon = loc["run_dir"], loc["tag"], loc["horizon"]
        arm, cond, ea, es, seed = key
        cse = (cond, es, seed)
        # the BASE artifact of an extended cell still feeds the twin
        # derivation and the twin-agreement check
        if g.fig6 and horizon != N_ROUNDS:
            base = [b for b in base_runs.get(cse, []) if b[0] == key]
            if base:
                _, brd, btag = base[0]
                db = load(brd)
                if state["mask_a"] is None:
                    init_ref(db, btag)
                bad = structural_checks(db, key, state["mask_a"],
                                        state["ref_sha"], tag=btag,
                                        horizon=N_ROUNDS)
                if bad:
                    fatal.extend(bad)
                else:
                    note_twin(db, key, btag, brd)
                del db
        d = load(rd)
        if state["mask_a"] is None:
            init_ref(d, tag)
        bad = structural_checks(d, key, state["mask_a"], state["ref_sha"],
                                tag=tag, horizon=(horizon if g.fig6
                                                  else None))
        if g.fig6 and not has_real_twin(d):
            bad.append(f"{tag}: no twin_raw shaped like op_raw -- fig6 "
                       f"needs the matched no-AI twin on every run")
        if g.fig6 and ea == 0.0 and has_real_twin(d):
            # a run at eps_AI = 0 is a WITNESS: the strict gate is closed,
            # so the population must BE the twin, bit for bit
            op_, tw_ = d["op_raw"].float(), d["twin_raw"].float()
            if not torch.equal(op_, tw_):
                worst = float((op_ - tw_).abs().max())
                bad.append(f"{tag}: eps_AI = 0 WITNESS but op_raw != "
                           f"twin_raw (max |diff| = {worst:.3e}) -- the "
                           f"gate did not close, so the eps_AI = 0 "
                           f"twin derivation is WRONG for this wave")
            del op_, tw_
        if bad:
            fatal.extend(bad)
            del d
            continue
        t_sha = twin_sha(d)
        if g.fig6 and horizon == N_ROUNDS:
            note_twin(d, key, tag, rd)
        late_idx = late_window(horizon) if g.fig6 else LATE_IDX
        rounds, late = reduce_cell(d, state["mask_a"], late_idx)
        win_b[key] = op_b_window(d, state["mask_a"], late_idx)
        del d                                  # drop before the next open
        for r in rounds:
            per_round_rows.append({
                "arm": arm, "cond": cond, "eps_ai": ea, "eps_social": es,
                "seed": seed, "run_tag": tag, **r})
        late.update({"arm": arm, "cond": cond, "eps_ai": ea,
                     "eps_social": es, "seed": seed, "run_tag": tag,
                     "gpu_arch": gpu_arch(rd)})
        if g.fig6:
            late.update({"kind": loc["kind"], "horizon": horizon,
                         "analysed_from": "op_raw", "derived_from": None,
                         "twin_sha256": t_sha})
            late.update(settle_verdict(late, args.drift_tol))
        cells[key] = late

    if g.fig6:
        # fill the twin-derived cells (method collapse BY CONSTRUCTION:
        # every arm at one (cond, es, seed) gets the same derived stats)
        for key in g.keys:
            loc = located.get(key)
            if loc is None or loc["analysed_from"] != "twin_raw":
                continue
            arm, cond, ea, es, seed = key
            got = derived.get((cond, es, seed))
            if got is None:
                if not any(m.startswith(loc["derived_from"]) for m in fatal):
                    fatal.append(f"{loc['tag']}: twin-derived cell could "
                                 f"not be derived (no base run at "
                                 f"({cond}, es={es:g}, seed={seed}) "
                                 f"passed the checks)")
                continue
            rounds, late0, src_tag, arch, sha, win = got
            win_b[key] = win
            tag = loc["tag"]
            for r in rounds:
                per_round_rows.append({
                    "arm": arm, "cond": cond, "eps_ai": ea,
                    "eps_social": es, "seed": seed,
                    "run_tag": f"{tag} [twin-derived from {src_tag}]",
                    **r})
            late = dict(late0)
            late.update({"arm": arm, "cond": cond, "eps_ai": ea,
                         "eps_social": es, "seed": seed, "run_tag": tag,
                         "gpu_arch": arch, "kind": "twin",
                         "horizon": N_ROUNDS, "analysed_from": "twin_raw",
                         "derived_from": src_tag,
                         "twin_sha256": sha})
            late.update(settle_verdict(late, args.drift_tol))
            cells[key] = late
        # twin agreement across every base run at one (cond, es, seed)
        for cse in sorted(twin_shas):
            shas = twin_shas[cse]
            if len(set(shas.values())) > 1:
                fatal.append(
                    f"twin_raw DISAGREES across the base runs at "
                    f"(cond={cse[0]}, es={cse[1]:g}, seed={cse[2]}): "
                    + ", ".join(f"{t}={s[:10]}" for t, s in sorted(
                        shas.items()))
                    + " -- the eps_AI = 0 derivation is not well-defined")
        # method collapse at eps_AI = 0, asserted numerically: every
        # eps_AI = 0 cell at one (cond, es, seed) -- twin-derived or
        # witness -- must carry the same late-window cohort-B mean
        by_cse = {}
        for key, c in cells.items():
            if key[2] == 0.0:
                by_cse.setdefault((key[1], key[3], key[4]), []).append(
                    (key, c["mu_b_eq"]))
        for cse, items in sorted(by_cse.items()):
            vals = [v for _, v in items]
            if max(vals) - min(vals) > 1e-12:
                fatal.append(
                    f"eps_AI = 0 METHOD COLLAPSE violated at (cond={cse[0]}"
                    f", es={cse[1]:g}, seed={cse[2]}): mu_b_eq = "
                    + ", ".join(f"{k[0]}={v:.6f}" for k, v in items))
        # pairing: both members present at the SAME horizon
        for arm in g.arms:
            for ea in g.gates:
                for es in g.ess:
                    for s in SEEDS:
                        kf = (arm, "fixed", ea, es, s)
                        ke = (arm, "evolving", ea, es, s)
                        if kf in cells and ke in cells:
                            hf, he = cells[kf]["horizon"], cells[ke]["horizon"]
                            if hf != he:
                                fatal.append(
                                    f"UNPAIRED horizon at {arm} ea={ea:g} "
                                    f"es={es:g} seed={s}: fixed analysed "
                                    f"at {hf} rounds, evolving at {he} -- "
                                    f"extensions must come in matched "
                                    f"pairs")
                        elif (kf in cells) != (ke in cells) and not fatal:
                            fatal.append(f"UNPAIRED seed at {arm} ea={ea:g} "
                                         f"es={es:g} seed={s}")

    if fatal:
        print(f"\n[s4g] HARD FAIL: {len(fatal)} structural violation(s) -- "
              f"the fixed/evolving contrast would not be a contrast, so no "
              f"output is written:", file=sys.stderr)
        for msg in fatal:
            print(f"  {msg}", file=sys.stderr)
        return 1

    # ---- 3. aggregate
    source_rows = build_source_rows(cells, args.drift_tol, g, win_b=win_b)
    disp_rows = build_dispersion_rows(cells, args.drift_tol, g)
    null_rows = build_null_rows(cells, g)
    gap_rows = build_method_gap_rows(source_rows, args.drift_tol, g) \
        if g.fig6 else []
    if g.fig6:
        # G is identically 0 at eps_AI = 0: both arms are the same twin
        for r in gap_rows:
            if r["eps_ai"] == 0.0 and r["status"] == "complete":
                bad0 = [s for s in SEEDS if r.get(f"{G_COL}_s{s}") != 0.0]
                if bad0:
                    print(f"[s4g] HARD FAIL: method gap at eps_AI=0, "
                          f"es={r['eps_social']:g} is not identically 0 "
                          f"(seeds {bad0}) -- the twin derivation is broken",
                          file=sys.stderr)
                    return 1

    n_complete = sum(1 for r in source_rows if r["status"] == "complete")
    cover_note = (f"{len(located)}/{g.n_cells} cells present, "
                  f"{n_complete}/{len(source_rows)} (arm, "
                  f"eps_AI, eps_social) series complete over all "
                  f"{len(SEEDS)} seeds")
    if g.fig6:
        cover_note += (f"; {n_from_twin} eps_AI=0 cells twin-derived, "
                       f"{n_ext} cells at an extended horizon")
    if partial:
        print(f"\n[s4g] ***** PARTIAL COVERAGE ***** {cover_note}. "
              f"Incomplete series are written with status=incomplete and NA "
              f"aggregates, and are NOT plotted: a short seed set is never "
              f"averaged into a three-seed number.")

    # ---- 4. structural null (d8, eps_social = 0)
    print("\n[s4g] structural null (d8, eps_social=0): cohort-B source "
          "effect must be 0 by construction")
    n_fail = 0
    for r in null_rows:
        if r["verdict"] == "MISSING":
            print(f"  ea{r['eps_ai']:<4g} s{r['seed']:<3}: MISSING pair")
            continue
        print(f"  ea{r['eps_ai']:<4g} s{r['seed']:<3}: "
              f"delta_mu_b={r['delta_mu_b']:+.3e}  "
              f"{r['gpu_fixed']}/{r['gpu_evolving']}"
              f"{'  [matched -> exact]' if r['hardware_matched'] else ''}"
              f"  tol={r['tol']:.0e}  {r['verdict']}")
        n_fail += r["verdict"] == "FAIL"
    xhw = [abs(r["delta_mu_b"]) for r in null_rows
           if r["verdict"] != "MISSING" and not r["hardware_matched"]]
    if xhw:
        print(f"[s4g] cross-architecture generation-nondeterminism floor "
              f"from {len(xhw)} probe(s): |delta_mu_b| <= {max(xhw):.2e}")

    # ---- 5. write
    os.makedirs(out_dir, exist_ok=True)
    write_csv(out_dir, f"{stem}_per_round.csv", per_round_rows)
    write_csv(out_dir, f"{stem}_cells.csv",
              [cells[k] for k in g.keys if k in cells])
    write_csv(out_dir, f"{stem}_source_effect.csv", source_rows)
    if g.fig6:
        write_csv(out_dir, f"{stem}_method_gap.csv", gap_rows)
    write_csv(out_dir, f"{stem}_dispersion.csv", disp_rows)
    write_csv(out_dir, f"{stem}_null_probe.csv", null_rows)
    write_csv(out_dir, f"{stem}_coverage.csv", coverage)

    ext_req, n_unsettled = None, 0
    if g.fig6:
        ext_req = build_extension_request(source_rows, g, args.drift_tol,
                                          run_root)
        ext_path = os.path.join(out_dir, f"{stem}_extension_request.json")
        with open(ext_path, "w") as fh:
            json.dump(ext_req, fh, indent=2)
        n_unsettled = sum(1 for r in source_rows for s in SEEDS
                          if r.get(f"pair_outcome_s{s}") not in
                          (None, "equilibrium"))
        n_cyclic = sum(1 for r in source_rows for s in SEEDS
                       if r.get(f"pair_outcome_s{s}") == "cyclic")
        print(f"[s4g] wrote {stem}_extension_request.json "
              f"({ext_req['n_cells']} cells = "
              f"{ext_req['n_cells'] // 2} matched pairs to extend, "
              f"{n_cyclic} of them cyclic; "
              f"{len(ext_req['twin_derived_unsettled'])} twin-derived and "
              f"{len(ext_req['not_extendable'])} not-extendable unsettled "
              f"pair(s) listed outside 'cells')")
        print_table(source_rows,
                    f"three-seed T_a = mu_B^eq(evolving) - mu_B^eq(fixed) "
                    f"[{T_A_COL}]",
                    f"{T_A_COL}_mean", f"{T_A_COL}_ci_excludes_zero",
                    grid=g)
        print_fig6_detail(source_rows, g)
        print_method_gap_table(gap_rows, g)
    else:
        print_table(source_rows,
                    "three-seed source effect  mu_B(fixed) - mu_B(evolving)",
                    "delta_mu_b_mean", "delta_mu_b_ci_excludes_zero",
                    grid=g)
    print_table(source_rows,
                f"three-seed agent-paired cohort-B MAE  "
                f"mean_i |op_B(fixed) - op_B(evolving)| [{MAE_COL}]",
                f"{MAE_COL}_mean", None, "%.4f", grid=g)
    print(f"  {MAE_DEF}")
    print_table(disp_rows,
                "three-seed cohort-B SD ratio  SD_B(fixed) / SD_B(evolving)",
                "sd_ratio_b_mean", "sd_ratio_b_ci_excludes_one", "%.4f",
                grid=g)

    any_cell = next(iter(cells.values()))
    shape = {"n_agents": any_cell["n_a"] + any_cell["n_b"],
             "n_a": any_cell["n_a"], "n_b": any_cell["n_b"],
             "n_rounds": any_cell["n_rounds"]}
    if g.fig6:
        caps = [caption_fig6_source(source_rows, cover_note, args.drift_tol,
                                    shape, g, gate_info),
                caption_dispersion(disp_rows, cover_note, args.drift_tol,
                                   partial, shape, g)]
    else:
        caps = [caption_source(source_rows, cover_note, args.drift_tol,
                               partial, shape),
                caption_dispersion(disp_rows, cover_note, args.drift_tol,
                                   partial, shape)]
    figs = []
    if not args.no_figs:
        figs += figure_source(source_rows, out_dir, cover_note,
                              args.drift_tol, g)
        figs += figure_dispersion(disp_rows, out_dir, cover_note,
                                  args.drift_tol, g)
    cap_path = os.path.join(out_dir, f"{stem}_captions.txt")
    with open(cap_path, "w") as fh:
        for block in caps:
            print("\n" + "\n".join(block))
            fh.write("\n".join(block) + "\n\n")
    print(f"\n[s4g] wrote {stem}_captions.txt")

    if n_fail:
        marker = os.path.join(out_dir, "SUSPECT_NULL_VIOLATION.txt")
        with open(marker, "w") as fh:
            fh.write(
                f"{n_fail} of {len(null_rows)} d8/eps_social=0 structural-"
                f"null probes FAILED.\nWith frozen weights, own-history "
                f"prompts and no peer step no cohort-A opinion can reach a "
                f"cohort-B prompt, so a nonzero cohort-B source effect "
                f"there means either a path that must not exist or "
                f"generation nondeterminism large enough to contaminate "
                f"the estimates. Treat every number in this directory as "
                f"SUSPECT until it is explained.\n")
        print(f"\n[s4g] ***** the d8/eps_social=0 structural null FAILED on "
              f"{n_fail} probe(s): results in this directory are SUSPECT "
              f"*****", file=sys.stderr)

    summary = {
        "key": g.key, "wave": g.key, "mode": "fig6" if g.fig6 else "v1",
        "run_root": run_root, "out_dir": out_dir,
        "grid": {"arms": g.arms, "conds": g.conds, "eps_ai": g.gates,
                 "eps_social": g.ess, "seeds": g.seeds,
                 "grid_source": "experiments/condor/gen_pofd_sweep.py"},
        "n_cells_expected": g.n_cells, "n_cells_located": len(located),
        "missing_tags": missing, "unexpected_tags": stray,
        "partial": partial, "coverage_note": cover_note,
        "late_window_op_raw_rounds": [LATE_IDX[0], LATE_IDX[-1]],
        "late_window_halves": [LATE_H1, LATE_H2],
        "late_window_rule": ("the final five post-peer rounds of the "
                             "analysed artifact; = op_raw 25-29 = rounds "
                             "26-30 (1-indexed) at 30 rounds"),
        "op_raw_semantics": "END-OF-ROUND POST-PEER population state",
        "t_crit_df2": T_CRIT_DF2, "t_crit_source": T_CRIT_SOURCE,
        "drift_tol": args.drift_tol,
        "population_update_required": POP_UPDATE_V2,
        "inherited_from": "analyze_bottom20_section4_3seed.py",
        "mae_b_paired_column": MAE_COL,
        "mae_b_paired_definition": MAE_DEF,
        "gate_json": args.gate_json, "gate_ok": gate_ok,
        "gate_info": gate_info,
        "n_series_complete": n_complete,
        "n_series": len(source_rows),
        "null_probe_failures": n_fail,
        "figures": figs,
        "source_effect": source_rows, "dispersion": disp_rows,
        "null_probe": null_rows,
    }
    if g.fig6:
        summary.update({
            "primary_column": T_A_COL,
            "t_a_sign": T_A_SIGN,
            "g_sign": G_SIGN,
            "method_gap": gap_rows,
            "settled_rule": ext_req["settled_rule"],
            "n_pairs_cyclic": n_cyclic,
            "n_cells_from_run": n_from_run,
            "n_cells_twin_derived": n_from_twin,
            "n_cells_extended_horizon": n_ext,
            "n_pairs_unsettled": n_unsettled,
            "n_series_settled": sum(1 for r in source_rows
                                    if r.get("settled")),
            "extension_request": {
                "path": ext_path, "n_cells": ext_req["n_cells"],
                "n_twin_derived_unsettled": len(
                    ext_req["twin_derived_unsettled"])},
            "twin_sha256_by_cond_es_seed": {
                f"{c}|{es:g}|{s}": sorted(set(v.values()))
                for (c, es, s), v in sorted(twin_shas.items())},
            "cells": [cells[k] for k in g.keys if k in cells],
            "coverage": coverage,
        })
        js_path = os.path.join(out_dir, f"{stem}_summary.json")
        with open(js_path, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"[s4g] wrote {stem}_summary.json")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"[s4g] wrote {args.json}")

    if n_fail:
        return 3
    if g.fig6:
        if n_unsettled:
            print(f"\n[s4g] ***** {n_unsettled} pair(s) UNSETTLED at the "
                  f"analysed horizon ({n_cyclic} cyclic): NOT an "
                  f"equilibrium; extension request written *****")
            return 2
        return 0
    if partial:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
