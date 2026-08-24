#!/usr/bin/env python3
"""SECTION-4 CORRECTED-GATE analyzer (key ``section4_gate_anch2``, 72 GPU
cells, 2026-08-24).

WHAT THIS WAVE IS. The published Section-4 experiment -- Mistral-7B,
movielens Action, 723 agents, bottom-20% FIXED source cohort vs a fully
EVOLVING population -- re-run under the CORRECTED AI gate: the acceptance
test is |m - x'| <= eps_AI on the ANCHORED opinion x' = k*innate + (1-k)*x,
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

USAGE
  OMP_NUM_THREADS=1 python analyze_section4_gate.py \\
      --run-root /home/gsmithline/perfsim/runs/pokec_gated_lm \\
      --out-dir  /home/gsmithline/perfsim/runs/analysis/section4_gate_anch2 \\
      [--no-figs] [--json OUT.json] [--drift-tol 0.002]
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
KEY = "section4_gate_anch2"
TAG_PREFIX = "pofds4g"
MODEL_SLUG = "mistral7b"
ARMS = ["b0", "d8"]
ARM_LABEL = {"b0": "ordinary SFT",
             "d8": "frozen personal-history ICL (8 days)"}
CONDS = ["fixed", "evolving"]
COND_TOK = {"fixed": "fixb20", "evolving": "evoall"}
TOK_COND = {v: k for k, v in COND_TOK.items()}
EAS = [0.2, 1.0]
ESS = [0.0, 0.2, 1.0]
SEEDS = [0, 42, 43]
W_PLAT = 0.5
INNATE_LAMBDA = 0.2          # k, the anchor weight
N_ROUNDS = 30
N_CELLS = len(ARMS) * len(CONDS) * len(EAS) * len(ESS) * len(SEEDS)  # 72

# ------------------------------------------- inherited numeric conventions
LATE = range(25, 30)         # analyze_bottom20_section4_3seed.py, verbatim
LATE_IDX = list(LATE)
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


def cell_tag(arm, cond, ea, es, seed):
    """The pinned tag for one cell of this wave."""
    return (f"{TAG_PREFIX}_{MODEL_SLUG}_{arm}_{COND_TOK[cond]}_anch2"
            f"_ea{_num(ea)}_w{_num(W_PLAT)}_l{_num(INNATE_LAMBDA)}"
            f"_es{_num(es)}_s{seed}")


_TAG_RE = re.compile(
    r"^" + TAG_PREFIX + r"_(?P<model>[a-z0-9_]+?)"
    r"_(?P<arm>b0|d8)_(?P<cond_tok>fixb20|evoall)_anch2"
    r"_ea(?P<ea>[0-9p]+)_w(?P<w>[0-9p]+)_l(?P<l>[0-9p]+)"
    r"_es(?P<es>[0-9p]+)_s(?P<seed>[0-9]+)$")


def parse_tag(tag):
    """Parse a wave tag -> dict, or None if it is not one of ours.

    Returns model / arm / cond / eps_ai / w_plat / innate_lambda /
    eps_social / seed, and in_grid: whether the parsed cell is one of the
    72 the wave declares.  Round-trips with cell_tag for in-grid cells.
    """
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
           "seed": int(m.group("seed"))}
    out["in_grid"] = (out["model"] == MODEL_SLUG
                      and out["arm"] in ARMS
                      and out["eps_ai"] in EAS
                      and out["eps_social"] in ESS
                      and out["seed"] in SEEDS
                      and out["w_plat"] == W_PLAT
                      and out["innate_lambda"] == INNATE_LAMBDA)
    return out


def scan_run_root(run_root):
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
        p = parse_tag(name)
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
def reduce_cell(d, mask_a):
    """Reduce ONE trajectory to (per-round rows, late-window scalars).

    Every tensor touched here is local, so the caller can drop the whole
    trajectory before opening the next one.  op_raw[t] is the END-OF-ROUND
    POST-PEER state; nothing else is read.
    """
    op = d["op_raw"].float()
    tw, twin_source = twin_of(d)
    pred = d["pred_raw"].float()
    innate = d["innate"].float()
    a, b = mask_a, ~mask_a
    n_r = int(op.shape[0])

    rounds = []
    for t in range(n_r):
        x, xt = op[t], tw[t]
        served = pred[t].clamp(0.0, 1.0)
        rounds.append({
            "round": t,                       # op_raw index (late window)
            "round_1based": t + 1,            # the runner's round number
            "pop_mean": float(x.mean()),
            "pop_sd": float(x.std()),
            "a_mean": float(x[a].mean()),
            "a_sd": float(x[a].std()),
            "b_mean": float(x[b].mean()),
            "b_sd": float(x[b].std()),
            "w1_twin_pop": w1(x, xt),
            "w1_twin_b": w1(x[b], xt[b]),
            "served_mean": float(torch.nanmean(served)),
            "pred_mean_raw": float(torch.nanmean(pred[t])),
            "twin_source": twin_source,
        })

    def wmean(idx, key):
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
        "twin_source": twin_source,
        "n_a": int(a.sum()), "n_b": int(b.sum()),
        "innate_mean": float(innate.mean()),
        "innate_sd": float(innate.std()),
        "innate_a_mean": float(innate[a].mean()),
        "innate_b_mean": float(innate[b].mean()),
        "innate_b_sd": float(innate[b].std()),
        "mu_pop_eq": wmean(LATE_IDX, "pop_mean"),
        "mu_a_eq": wmean(LATE_IDX, "a_mean"),
        "mu_b_eq": wmean(LATE_IDX, "b_mean"),
        "sd_pop_late": wmean(LATE_IDX, "pop_sd"),
        "sd_a_late": wmean(LATE_IDX, "a_sd"),
        "sd_b_late": wmean(LATE_IDX, "b_sd"),
        "w1_twin_pop_late": wmean(LATE_IDX, "w1_twin_pop"),
        "w1_twin_b_late": wmean(LATE_IDX, "w1_twin_b"),
        "served_mean_late": wmean(LATE_IDX, "served_mean"),
        "sd_ratio_late": twin_ratio(LATE_IDX, b),       # inherited name
        "sd_ratio_pop_twin_late": twin_ratio(LATE_IDX, slice(None)),
        # half-window values, for the drift / robustness flag
        "mu_b_h1": wmean(LATE_H1, "b_mean"),
        "mu_b_h2": wmean(LATE_H2, "b_mean"),
        "mu_pop_h1": wmean(LATE_H1, "pop_mean"),
        "mu_pop_h2": wmean(LATE_H2, "pop_mean"),
        "sd_b_h1": wmean(LATE_H1, "b_sd"),
        "sd_b_h2": wmean(LATE_H2, "b_sd"),
        "sd_pop_h1": wmean(LATE_H1, "pop_sd"),
        "sd_pop_h2": wmean(LATE_H2, "pop_sd"),
        "innate_sha256": innate_sha(innate),
    }
    return rounds, late


def structural_checks(d, key, mask_a, ref_sha):
    """FATAL structural problems with one cell, as a list of strings.

    Empty list == the cell can take part in the fixed/evolving contrast.
    """
    arm, cond, ea, es, seed = key
    tag = cell_tag(*key)
    bad = []
    cfg = d.get("config") or {}
    op = d["op_raw"]
    innate = d["innate"]

    if int(op.shape[0]) < max(LATE_IDX) + 1:
        bad.append(f"{tag}: {int(op.shape[0])} rounds < "
                   f"{max(LATE_IDX) + 1} needed for the late window")
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


def build_source_rows(cells, drift_tol):
    """B. SOURCE EFFECT: fixed minus evolving on the late window, per
    (arm, eps_ai, eps_social), aggregated over the three seeds."""
    rows = []
    for arm in ARMS:
        for ea in EAS:
            for es in ESS:
                paired = [s for s in SEEDS
                          if (arm, "fixed", ea, es, s) in cells
                          and (arm, "evolving", ea, es, s) in cells]
                complete = len(paired) == len(SEEDS)
                d_b, d_pop, d_b_h1, d_b_h2 = {}, {}, {}, {}
                gpu_pair, sha_ok = {}, {}
                for s in paired:
                    f = cells[(arm, "fixed", ea, es, s)]
                    e = cells[(arm, "evolving", ea, es, s)]
                    d_b[s] = f["mu_b_eq"] - e["mu_b_eq"]
                    d_pop[s] = f["mu_pop_eq"] - e["mu_pop_eq"]
                    d_b_h1[s] = f["mu_b_h1"] - e["mu_b_h1"]
                    d_b_h2[s] = f["mu_b_h2"] - e["mu_b_h2"]
                    gpu_pair[s] = f'{f["gpu_arch"]}/{e["gpu_arch"]}'
                    sha_ok[s] = f["innate_sha256"] == e["innate_sha256"]
                row = {"arm": arm, "arm_label": ARM_LABEL[arm],
                       "eps_ai": ea, "eps_social": es,
                       "n_seeds_paired": len(paired),
                       "seeds_paired": "|".join(str(s) for s in paired),
                       "status": "complete" if complete else "incomplete"}
                row.update(agg_block("delta_mu_b", d_b, 0.0, "zero"))
                row.update(agg_block("delta_mu_pop", d_pop, 0.0, "zero"))
                # the prior analyzer's sign, so the corrected-gate numbers
                # sit next to the published ones without a mental flip
                for s in SEEDS:
                    v = d_b.get(s)
                    row[f"t_a_evolving_minus_fixed_s{s}"] = (
                        None if v is None else -v)
                for suffix in ("mean", "ci_lo", "ci_hi"):
                    v = row.get(f"delta_mu_b_{suffix}")
                    row[f"t_a_evolving_minus_fixed_{suffix}"] = (
                        None if v is None else -v)
                if row["t_a_evolving_minus_fixed_ci_lo"] is not None:
                    row["t_a_evolving_minus_fixed_ci_lo"], \
                        row["t_a_evolving_minus_fixed_ci_hi"] = (
                            row["t_a_evolving_minus_fixed_ci_hi"],
                            row["t_a_evolving_minus_fixed_ci_lo"])
                row.update(drift_block("delta_mu_b", d_b_h1, d_b_h2,
                                       drift_tol,
                                       _ci_half(row, "delta_mu_b")))
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


def build_dispersion_rows(cells, drift_tol):
    """C. DISPERSION: fixed vs evolving population SD and cohort-B SD on the
    same late window, with the paired fixed/evolving SD ratio."""
    rows = []
    for arm in ARMS:
        for ea in EAS:
            for es in ESS:
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


def build_null_rows(cells):
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
    rows = []
    for ea in EAS:
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


def _series(rows, arm, ea, prefix):
    """(x positions, y, yerr_lo, yerr_hi, flagged) for one arm at one
    eps_ai.  Incomplete series are DROPPED, never plotted as if they were
    three-seed results."""
    xs, ys, lo, hi, flag = [], [], [], [], []
    for j, es in enumerate(ESS):
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
        flag.append(bool(r.get(f"{prefix}_drift_flag")))
    return xs, ys, lo, hi, flag


def _style_panel(ax, ylabel=None, xlabel=True):
    ax.set_xlim(-0.45, len(ESS) - 0.55)
    ax.set_xticks(range(len(ESS)))
    ax.set_xticklabels([f"{e:g}" for e in ESS])
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
                 label_suffix=""):
    """Draw both arms at one eps_AI. Returns True if any drift flag was
    marked, and the number of series actually drawn."""
    drew_flag, drew = False, 0
    for arm in ARMS:
        off = -dodge if arm == "b0" else dodge
        xs, ys, lo, hi, flag = _series(rows, arm, ea, prefix)
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


def figure_source(rows, out_dir, cover_note, drift_tol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _rc()
    fig, axes = plt.subplots(1, len(EAS), figsize=(7.2, 3.0), sharey=True,
                             squeeze=False)
    flagged, n_drawn = False, 0
    for i, ea in enumerate(EAS):
        ax = axes[0][i]
        ax.axhline(0.0, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=1)
        fl, nd = _plot_series(ax, rows, ea, "delta_mu_b")
        flagged |= fl
        n_drawn += nd
        _style_panel(ax, ylabel=(r"$\mu_B^{\mathrm{late}}(\mathrm{fixed})"
                                r"-\mu_B^{\mathrm{late}}"
                                r"(\mathrm{evolving})$")
                     if i == 0 else None)
        _panel_tag(ax, r"$\varepsilon_{\mathrm{AI}}=%g$" % ea)
    if n_drawn:
        axes[0][0].legend(frameon=False, fontsize=8.2, loc="best")
    else:
        axes[0][0].annotate("no complete series", xy=(0.5, 0.5),
                            xycoords="axes fraction", ha="center",
                            va="center", fontsize=9, color="0.45")
    foot = (f"post-peer late window = op_raw rounds "
            f"{LATE_IDX[0]}-{LATE_IDX[-1]}; error bars = 95% Student-t "
            f"interval over seeds {', '.join(str(s) for s in SEEDS)} "
            f"(df=2); {cover_note}")
    if flagged:
        foot += (f"; \u2020 = half-window drift exceeds {drift_tol:g} "
                 f"(see the CSV)")
    fig.tight_layout()
    _foot(fig, foot, -0.035)
    return _save(fig, out_dir, "section4_gate_source_effect")


def figure_dispersion(rows, out_dir, cover_note, drift_tol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _rc()
    fig, axes = plt.subplots(2, len(EAS), figsize=(7.2, 5.2), sharey="row",
                            sharex=True, squeeze=False)
    flagged, n_drawn = False, 0
    specs = [("sd_ratio_b", r"SD$_B$(fixed) / SD$_B$(evolving)"),
             ("sd_ratio_pop", r"SD$_{\mathrm{pop}}$(fixed) / "
                              r"SD$_{\mathrm{pop}}$(evolving)")]
    for r_i, (prefix, ylab) in enumerate(specs):
        for i, ea in enumerate(EAS):
            ax = axes[r_i][i]
            ax.axhline(1.0, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=1)
            fl, nd = _plot_series(ax, rows, ea, prefix)
            flagged |= fl
            n_drawn += nd
            _style_panel(ax, ylabel=ylab if i == 0 else None,
                         xlabel=(r_i == len(specs) - 1))
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
    return _save(fig, out_dir, "section4_gate_dispersion")


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


def caption_dispersion(disp_rows, cover_note, drift_tol, partial, shape):
    n_sig = sum(1 for r in disp_rows
                if r["status"] == "complete"
                and r.get("sd_ratio_b_ci_excludes_one"))
    n_done = sum(1 for r in disp_rows if r["status"] == "complete")
    return [
        "CAPTION -- section4_gate_dispersion.pdf/.png "
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
        "five end-of-round post-peer states). Panels are the two AI gates,",
        "series the two arms, error bars 95% Student-t intervals over the",
        f"three seeds (df=2, t={T_CRIT_DF2:.4f}). Dashed line: ratio = 1.",
        f"{n_sig} of {n_done} complete cells have a cohort-B SD-ratio",
        "interval excluding 1.",
        f"A \u2020 marks a series whose half-to-half drift exceeds "
        f"{drift_tol:g}.",
        "The twin-referenced dispersion of the published Section 4 --",
        "SD(B platform)/SD(B matched no-platform twin), per condition --",
        "is in section4_gate_dispersion.csv rather than this figure.",
        f"Coverage: {cover_note}."
        + (" RESULTS ARE PARTIAL." if partial else ""),
        "Exploratory: one wave, three seeds, one model, one dataset.",
    ]


# =================================================================== main
def default_out_dir(run_root):
    """A runs-ADJACENT analysis directory: a sibling of the run root, never
    inside it (so a tag scan cannot trip over the analysis) and NEVER under
    paper/."""
    parent = os.path.dirname(os.path.abspath(run_root.rstrip(os.sep)))
    return os.path.join(parent, "analysis", KEY)


def refuse_paper_dir(out_dir):
    parts = {p.lower() for p in os.path.abspath(out_dir).split(os.sep)}
    if "paper" in parts:
        print(f"[s4g] REFUSING --out-dir {out_dir!r}: analysis artifacts "
              f"never go under paper/", file=sys.stderr)
        sys.exit(1)


def print_table(rows, title, mean_key, mark_key, fmt="%+.4f"):
    print(f"\n== {title} ==")
    print("   " + " ".join(f"{'es=' + f'{e:g}':>12}" for e in ESS))
    for arm in ARMS:
        for ea in EAS:
            cellstr = []
            for es in ESS:
                r = _pick(rows, arm, ea, es)
                if r is None or r["status"] != "complete" \
                        or r.get(mean_key) is None:
                    cellstr.append(f"{'--':>12}")
                    continue
                s = fmt % r[mean_key]
                s += "*" if r.get(mark_key) else " "
                s += "\u2020" if r.get(
                    f"{mean_key[:-5]}_drift_flag") else " "
                cellstr.append(f"{s:>12}")
            print(f"  {arm:<3} ea={ea:<4g}" + " ".join(cellstr))
    print("  (* = 95% CI excludes the reference; \u2020 = half-window "
          "drift exceeds the tolerance)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Section-4 corrected-gate (section4_gate_anch2) "
                    "analyzer: fixed vs evolving source cohort under the "
                    "anchored AI gate.")
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
                    help="write a machine-readable summary here")
    ap.add_argument("--drift-tol", type=float, default=DEFAULT_DRIFT_TOL,
                    help=f"half-window drift tolerance in opinion units "
                         f"(default {DEFAULT_DRIFT_TOL:g})")
    args = ap.parse_args(argv)

    run_root = os.path.abspath(args.run_root)
    out_dir = os.path.abspath(args.out_dir if args.out_dir
                              else default_out_dir(run_root))
    refuse_paper_dir(out_dir)
    print(f"[s4g] key      : {KEY}")
    print(f"[s4g] run-root : {run_root}")
    print(f"[s4g] out-dir  : {out_dir}")
    print(f"[s4g] window   : op_raw rounds {LATE_IDX[0]}-{LATE_IDX[-1]} "
          f"(END-OF-ROUND POST-PEER states; LATE = range(25, 30), "
          f"inherited from analyze_bottom20_section4_3seed.py)")
    print(f"[s4g] t crit   : {T_CRIT_SOURCE}")

    # ---- 1. locate every required cell (existence only; no tensors yet)
    grid = [(arm, cond, ea, es, seed)
            for arm in ARMS for cond in CONDS
            for ea in EAS for es in ESS for seed in SEEDS]
    assert len(grid) == N_CELLS, (len(grid), N_CELLS)
    located, missing, coverage = {}, [], []
    for key in grid:
        tag = cell_tag(*key)
        rd = find_run(run_root, tag)
        coverage.append({
            "arm": key[0], "cond": key[1], "eps_ai": key[2],
            "eps_social": key[3], "seed": key[4], "run_tag": tag,
            "present": rd is not None,
            "run_dir": rd if rd is not None else None})
        if rd is None:
            missing.append(tag)
        else:
            located[key] = rd
    partial = bool(missing)
    print(f"[s4g] located  : {len(located)}/{N_CELLS} trajectories")
    for tag in missing:
        print(f"  MISSING {tag}")

    # tags that exist on disk but are not in the declared grid
    stray = [f["tag"] for f in scan_run_root(run_root)
             if not f.get("in_grid")]
    if stray:
        print(f"[s4g] NOTE: {len(stray)} {TAG_PREFIX}_ run(s) under the run "
              f"root are NOT in the declared 72-cell grid (grammar drift "
              f"or a smoke wave?):")
        for t in stray[:20]:
            print(f"  UNEXPECTED {t}")

    if not located:
        print(f"[s4g] HARD FAIL: no cell of {KEY} found under {run_root} -- "
              f"nothing to analyse", file=sys.stderr)
        return 1

    # ---- 2. one pass, ONE run's tensors resident at a time
    per_round_rows, cells, fatal = [], {}, []
    ref_sha, mask_a = None, None
    for key in grid:
        if key not in located:
            continue
        rd = located[key]
        d = load(rd)
        if mask_a is None:
            mask_a = cohort_a_mask(d["innate"])
            ref_sha = innate_sha(d["innate"])
            n_ag, n_a = int(d["innate"].numel()), int(mask_a.sum())
            print(f"[s4g] cohort A : {n_a} of {n_ag} agents "
                  f"(bottom {CLAMP_FRAC:g} by the innate-then-id ranking; "
                  f"reference run {cell_tag(*key)})")
            if n_ag != EXPECTED_N_AGENTS or n_a != EXPECTED_N_CLAMP:
                print(f"[s4g] NOTE: this wave is specified at "
                      f"{EXPECTED_N_AGENTS} agents / {EXPECTED_N_CLAMP} "
                      f"clamped; this run has {n_ag} / {n_a}")
        bad = structural_checks(d, key, mask_a, ref_sha)
        if bad:
            fatal.extend(bad)
            del d
            continue
        rounds, late = reduce_cell(d, mask_a)
        del d                                  # drop before the next open
        arm, cond, ea, es, seed = key
        for r in rounds:
            per_round_rows.append({
                "arm": arm, "cond": cond, "eps_ai": ea, "eps_social": es,
                "seed": seed, "run_tag": cell_tag(*key), **r})
        late.update({"arm": arm, "cond": cond, "eps_ai": ea,
                     "eps_social": es, "seed": seed,
                     "run_tag": cell_tag(*key), "gpu_arch": gpu_arch(rd)})
        cells[key] = late

    if fatal:
        print(f"\n[s4g] HARD FAIL: {len(fatal)} structural violation(s) -- "
              f"the fixed/evolving contrast would not be a contrast, so no "
              f"output is written:", file=sys.stderr)
        for msg in fatal:
            print(f"  {msg}", file=sys.stderr)
        return 1

    # ---- 3. aggregate
    source_rows = build_source_rows(cells, args.drift_tol)
    disp_rows = build_dispersion_rows(cells, args.drift_tol)
    null_rows = build_null_rows(cells)

    n_complete = sum(1 for r in source_rows if r["status"] == "complete")
    cover_note = (f"{len(located)}/{N_CELLS} cells present, "
                  f"{n_complete}/{len(source_rows)} (arm, "
                  f"eps_AI, eps_social) series complete over all "
                  f"{len(SEEDS)} seeds")
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
    write_csv(out_dir, "section4_gate_per_round.csv", per_round_rows)
    write_csv(out_dir, "section4_gate_cells.csv",
              [cells[k] for k in grid if k in cells])
    write_csv(out_dir, "section4_gate_source_effect.csv", source_rows)
    write_csv(out_dir, "section4_gate_dispersion.csv", disp_rows)
    write_csv(out_dir, "section4_gate_null_probe.csv", null_rows)
    write_csv(out_dir, "section4_gate_coverage.csv", coverage)

    print_table(source_rows,
                "three-seed source effect  mu_B(fixed) - mu_B(evolving)",
                "delta_mu_b_mean", "delta_mu_b_ci_excludes_zero")
    print_table(disp_rows,
                "three-seed cohort-B SD ratio  SD_B(fixed) / SD_B(evolving)",
                "sd_ratio_b_mean", "sd_ratio_b_ci_excludes_one", "%.4f")

    any_cell = next(iter(cells.values()))
    shape = {"n_agents": any_cell["n_a"] + any_cell["n_b"],
             "n_a": any_cell["n_a"], "n_b": any_cell["n_b"],
             "n_rounds": any_cell["n_rounds"]}
    caps = [caption_source(source_rows, cover_note, args.drift_tol, partial,
                           shape),
            caption_dispersion(disp_rows, cover_note, args.drift_tol,
                               partial, shape)]
    figs = []
    if not args.no_figs:
        figs += figure_source(source_rows, out_dir, cover_note,
                              args.drift_tol)
        figs += figure_dispersion(disp_rows, out_dir, cover_note,
                                  args.drift_tol)
    cap_path = os.path.join(out_dir, "section4_gate_captions.txt")
    with open(cap_path, "w") as fh:
        for block in caps:
            print("\n" + "\n".join(block))
            fh.write("\n".join(block) + "\n\n")
    print("\n[s4g] wrote section4_gate_captions.txt")

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
        "key": KEY, "run_root": run_root, "out_dir": out_dir,
        "n_cells_expected": N_CELLS, "n_cells_located": len(located),
        "missing_tags": missing, "unexpected_tags": stray,
        "partial": partial, "coverage_note": cover_note,
        "late_window_op_raw_rounds": [LATE_IDX[0], LATE_IDX[-1]],
        "late_window_halves": [LATE_H1, LATE_H2],
        "op_raw_semantics": "END-OF-ROUND POST-PEER population state",
        "t_crit_df2": T_CRIT_DF2, "t_crit_source": T_CRIT_SOURCE,
        "drift_tol": args.drift_tol,
        "population_update_required": POP_UPDATE_V2,
        "inherited_from": "analyze_bottom20_section4_3seed.py",
        "n_series_complete": n_complete,
        "n_series": len(source_rows),
        "null_probe_failures": n_fail,
        "figures": figs,
        "source_effect": source_rows, "dispersion": disp_rows,
        "null_probe": null_rows,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"[s4g] wrote {args.json}")

    if n_fail:
        return 3
    if partial:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
