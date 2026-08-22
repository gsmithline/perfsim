#!/usr/bin/env python3
"""GATE for the Section 3 RETENTION wave (pofds3_, 2026-08).

RUN THIS BEFORE ANYTHING IS ANALYZED, PLOTTED, OR PULLED.

CPU only. Set OMP_NUM_THREADS=1 (and USE_TF=0 for a local run) before
invoking; the script pins torch to a single thread itself so it can be
run on a login node or a small CPU condor slot without becoming a
multithreaded job.

====================================================================
RELATIONSHIP TO check_kl_direction.py -- READ THIS FIRST
====================================================================
check_kl_direction.py gates the ARCHIVED forward-vs-reverse wave
(pofdkd_, 10 rounds, Qwen2.5 only, k = 1, lambda in {.1, 1, 10}, both
arms sft_kl). It is NOT modified by this file and its behaviour must
not change: it certifies a wave that is already on record.

This file INHERITS its constants (N_AGENTS, the exact H100 SKU string)
and its output helper by import, and re-states the per-run gate because
Section 3 changes it on seven axes. Every delta is listed here so the
two gates can be diffed by reading, not by guessing:

  INHERITED VERBATIM IN SPIRIT (same clause, same field, same location)
    tag-vs-config direction agreement; kl_ref_adapter empty;
    anchor_mode "fixed"; base_model pinned to the tag's slug;
    training_style / use_lora / fresh_each_round; the surface pins
    (seed, dataset, ml_target, n_labeled, ab_sweeps, pop_model,
    icl_k/icl_days, lora_r, sft_epochs, epoch_size, train_cap, sft_lr);
    H100; serve_eval_mode; pred_raw/op_raw shape + finiteness; innate
    present; parse_fail_frac read from raw_gen_log.json.gz with the
    per-round 723-agent completeness check; l_init / grad_norm0 /
    grad_kl_norm0 read from telemetry.json with ROUND 0 EXEMPT from the
    KL-gradient rule.

  WIDENED (the archived gate would reject a legitimate Section 3 cell)
    1. two checkpoints, not one   -> MODELS, and the Qwen3 thinking gate
    2. k in {1, 0.2}, not k == 1  -> ENVS
    3. lambda ladder {0, .1, .5, 1, 2, 4, 8} + reverse {1, 8}
    4. an UNREGULARIZED arm exists (sft, lambda = 0, training_style
       "sft"); check_kl_direction hard-fails training_style != "sft_kl"
    5. horizon 100 (smoke 3), not 10
    6. both gate MODES pinned to all_open (the archived wave read eps
       as a numeric threshold pin)
    7. cells may arrive under a FOREIGN tag via attested reuse

  INVERTED -- THE ONE BEHAVIOURAL DIFFERENCE THAT MATTERS
    check_kl_direction HARD-FAILS a served map that is constant across
    agents, and HARD-FAILS a served vector that is bit-identical in
    every round. Those are correct for a wave whose question is
    "does reverse KL preserve heterogeneity", where a degenerate map
    means the answer cannot be read.

    For Section 3 they are WRONG. This wave asks what a rising lambda
    does to a population, and the frozen model it climbs toward serves
    a near-binary map. A collapsed or bit-static served map is a
    LEGITIMATE SCIENTIFIC OUTCOME here, and a gate that rejected it
    would delete the finding. So Section 3 does NOT fail either
    condition when parsing and training provenance are valid; it
    REPORTS them (see served_map_stats and the PER-CELL REPORT block).

--------------------------------------------------------------------
WHAT THIS WAVE CLAIMS, AND THEREFORE WHAT THIS GATE PROTECTS
--------------------------------------------------------------------
The claim is a LADDER: as the KL coefficient lambda rises, the trained
population is pulled toward the frozen model's attractor, and the
question is what happens to HETEROGENEITY on the way. Every failure
mode below would let a fake ladder pass:

  DIRECTION   config.kl_direction must equal the token the tag carries.
              A "rev" tag over a forward run inverts the robustness
              check and is invisible downstream.
  LAMBDA      config.kl_beta must equal the tag's lambda, exactly. The
              ladder IS the lambda axis.
  ENVIRONMENT (w_plat, innate_lambda) must be one of the three declared
              pairs. (1, 0.2) is NOT in this design; a run there is a
              different experiment wearing the wave's prefix.
  REFERENCE   kl_ref_adapter MUST be empty and anchor_mode "fixed". If
              one arm anchored to a trained adapter, its lambda is not
              on the same axis as the others'.
  OPERATOR    the round operator (population_update marker) must be the
              one EXPECTED FOR THAT SOURCE (see below), and the AI/peer
              gates must both be all_open -- the wave's tag says
              eaopen/esopen and the whole ladder is a gate-free
              comparison.
  SHARED WORLD every cell must sit on the SAME agents, in the same
              order, on the same graph, with the same peer stream.
              Under gamma = 0 and an all_open peer gate the peer-pair
              and acceptance counts are a function of (graph, seed, n)
              ALONE -- opinions cannot touch them -- so those two
              per-round counters are an exact cross-arm identity test
              for graph/order/RNG. See _peer_stream_key().
  PARSE       parse_fail_frac must be exactly 0 in every round. A parse
              failure is recorded as a confident constant, so a nonzero
              rate is never a rounding matter.
  TRAINING    the optimizer must have moved (grad_norm0 not all zero),
              and for a REGULARIZED arm the anchor term must actually
              have contributed a gradient in at least one round AFTER
              round 0. Round 0 is EXEMPT and must be: a fresh LoRA at
              round 0 IS the reference, so the divergence and its
              gradient are legitimately ~0 there.

--------------------------------------------------------------------
WHAT THIS GATE DELIBERATELY DOES *NOT* FAIL
--------------------------------------------------------------------
A CONSTANT OR COARSE SERVED MAP IS NOT A FAILURE. Frozen Qwen serves a
binary map to almost every agent; a strongly regularized arm is
supposed to approach that. Collapse is a legitimate scientific outcome
of this experiment, and a gate that rejected it would delete the
finding. So when parsing and training provenance are valid, the served
map's degeneracy is REPORTED, per cell, never failed:

  distinct served values, largest mode share, top-3 mode share,
  effective modes = exp(Shannon entropy of the exact value
  distribution), served SD

reported both for the final round and averaged over the equilibrium
window (rounds 81-100).

--------------------------------------------------------------------
THE OPERATOR MARKER: PER-SOURCE, NEVER "EITHER IS FINE"
--------------------------------------------------------------------
Two round-operator markers exist:

  nested_ai_then_social_v1            AI gate measured against x0
  nested_ai_anchored_then_social_v2   AI gate measured against the
                                      anchored x' = k innate + (1-k) x0

THE EQUIVALENCE, AND EXACTLY WHERE IT COMES FROM. In
_gated_pop.py:205-206 gp.ai_gate returns an all-ones mask as its FIRST
statement when mode == "all_open", and the reference vector is not read
until :209, on the "threshold" path. So under an all_open AI gate the
two markers select the identical agents and drive the identical update:
they are the same operator, as arithmetic, not as approximation.

THAT IS NOT A LICENCE TO ACCEPT EITHER MARKER ANYWHERE. This gate keeps
an EXPECTED-MARKER TABLE keyed by SOURCE (EXPECTED_MARKER below) and
validates every cell against its own entry:

  new Section 3 cells (pofds3_ / pofds3smk_)  MUST be v2. A v1 here
      means the run did not come from the current tree.
  the 4 archived QWU reuse cells              MUST be v1. A v2 here
      means the artifact on disk is not the one the reuse audit read.
  any source not in the table                 REJECTED. Refusing to
      guess is the point.

And the equivalence argument is only ALLOWED to apply once
ai_gate_mode == "all_open" AND peer_gate_mode == "all_open" have been
verified on that same cell -- a v1 artifact carrying a numeric AI gate
is rejected outright, because there the two operators genuinely differ.

The marker actually found is recorded for every cell (report
["population_update"]) and flows into the verdict JSON and the
analyzer's provenance columns, so the decision is auditable rather than
asserted.

Some existing plotters hard-assert v1 (e.g.
experiments/llm/plot_feature_endogenization_beta_final.py:88,
audit_qwen_gate_sweep.py:63). Those files are NOT modified and that
pattern is deliberately NOT copied here: a blanket single-marker assert
would reject every new Section 3 cell.

--------------------------------------------------------------------
REUSED CELLS
--------------------------------------------------------------------
About four conceptual cells are satisfied by archived Qwen2.5 QWU runs
whose tags are NOT in this wave's grammar. Such a cell is accepted ONLY
when the field-level manifest (notes/pofd/section3/reuse_manifest.json,
written by the reuse audit) marks it REUSE *and* carries an artifact
hash that matches what is on disk. See _manifest_verdict().

For a manifest-attested reuse cell exactly three checks are relaxed,
and every relaxation is printed in a PROVENANCE DEVIATIONS block:

  1. tag grammar        -- an archived cell is bound to its conceptual
                           slot by CONFIG FIELDS plus the manifest hash,
                           never by its name.
  2. serve_eval_mode    -- the field was introduced 2026-08-21; its
                           ABSENCE on an older run means "not verifiable
                           from the artifact", not "false". Present-and-
                           false is still a hard failure.
  3. population_update  -- an archived cell is EXPECTED to carry v1 and
                           is rejected if it carries v2. See the
                           per-source table above; this is a different
                           expected value, not a waived check.

Everything else -- kl_beta, kl_direction, w_plat, innate_lambda, both
gate modes, base_model, seed, LoRA setup, horizon, kl_ref_adapter,
anchor_mode, H100, parse, telemetry, shapes, finiteness, shared world --
is re-checked directly from the artifact for reused cells too.

A reused cell with n_rounds > 100 is NOT silently truncated. A trained
trajectory's first 100 rounds are a valid prefix in principle, but that
is a claim about the run's RNG and data path, so it must be admitted
explicitly by the manifest field "horizon_prefix_ok": true rather than
assumed by this script.

--------------------------------------------------------------------
Usage
--------------------------------------------------------------------
  OMP_NUM_THREADS=1 python check_section3.py \\
      --runs-root runs/pokec_gated_lm \\
      --reuse-manifest notes/pofd/section3/reuse_manifest.json \\
      --frozen notes/pofd/frozen_replay/frz_k1_w0p5_eaopen_esopen_sw1_s0_r300.pt

  # explicit dirs also work (shell glob)
  OMP_NUM_THREADS=1 python check_section3.py runs/pokec_gated_lm/pofds3_*_r100

  # the 3-round smoke cell (relaxed HORIZON only)
  OMP_NUM_THREADS=1 python check_section3.py --smoke \\
      runs/pokec_gated_lm/pofds3smk_*_r3

Exit codes: 0 = pass, 1 = hard failure, 2 = usage/input error.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

# transformers' TF probe deadlocks on some machines and nothing here
# needs it; torch is pinned to one thread so this never becomes a
# multithreaded job on a shared login node.
os.environ.setdefault("USE_TF", "0")

import torch

torch.set_num_threads(1)

# remote capture pipes this file over ssh stdin, where __file__ is unset
HERE = (os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals() else os.getcwd())
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _load_sibling(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(HERE, file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The archived forward-vs-reverse gate. Imported for its constants so the
# agent count, the base checkpoint string and the exact H100 SKU string
# have ONE home; see the RELATIONSHIP block above for why the per-run
# gate itself is re-stated rather than called. NOT modified.
#
# NOT reused: CK.fail. It hard-codes the "[check_kd]" log prefix, so
# calling it here would label every Section 3 failure as a failure of the
# archived wave's gate. A three-line local emitter is cheaper than that
# confusion.
CK = _load_sibling("check_kl_direction", "check_kl_direction.py")

# ---------------------------------------------------------------- pins
N_AGENTS = CK.N_AGENTS              # 723
PROD_ROUNDS = 100
SMOKE_ROUNDS = 3
PROD_PREFIX = "pofds3_"
SMOKE_PREFIX = "pofds3smk_"
LATE_LO, LATE_HI = 81, 100          # equilibrium window, 1-based, inclusive
SEED = 0
H100 = CK.H100                      # "NVIDIA H100 80GB HBM3"
H100_MARKER = "H100"

# BOTH roots. The archived QWU cells that satisfy ~4 conceptual slots
# live under notes/pofd/cluster/, NOT under runs/pokec_gated_lm/, so a
# single-root scan would report them as absent and the reuse would
# silently turn into four new GPU jobs.
DEFAULT_ROOTS = [os.path.join(REPO, "runs", "pokec_gated_lm"),
                 os.path.join(REPO, "notes", "pofd", "cluster")]

MODELS = {"qwen7b": CK.BASE,        # "Qwen/Qwen2.5-7B-Instruct"
          "qwen3_8b": "Qwen/Qwen3-8B"}

# (w_plat = beta, innate_lambda = k). (1, 0.2) is deliberately ABSENT.
ENVS = ((0.5, 1.0), (1.0, 1.0), (0.5, 0.2))
ENV_LABEL = {(0.5, 1.0): "main", (1.0, 1.0): "wu", (0.5, 0.2): "mem"}

FWD_LAMBDAS = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0)
REV_LAMBDAS = (1.0, 8.0)
# reverse KL is a labelled ROBUSTNESS CHECK and runs only in the two k=1
# environments. forward is the PRIMARY ladder.
REV_ENVS = ((0.5, 1.0), (1.0, 1.0))

# The tag's operator token. "anch2" names the runner's AI_GATE_REFERENCE
# default ("anchor"), which writes population_update =
# "nested_ai_anchored_then_social_v2". The token is spelled after the
# thing that actually exists in the tree, so a new cell can never be
# mistaken for an archived one.
OPTOK = "anch2"
POP_UPDATE_V1 = "nested_ai_then_social_v1"        # gate on x0 (archived)
POP_UPDATE_V2 = "nested_ai_anchored_then_social_v2"   # gate on x' (2026-08-22)
OPTOK_POP_UPDATE = {"anch2": POP_UPDATE_V2}

# --- THE EXPECTED-MARKER TABLE, KEYED BY SOURCE ----------------------
# Never "v1 or v2 is fine". Each source has ONE expected marker and a
# deviation is a hard failure in both directions. See the docstring for
# the equivalence argument (_gated_pop.py:205-206 vs :209) and for why
# it is gated on both gate modes being all_open.
SOURCE_S3_NEW = "section3_new"
SOURCE_QWU_REUSE = "qwu_reuse"
SOURCE_CPU_ENDPOINT = "cpu_endpoint"
EXPECTED_MARKER = {
    # new GPU cells from the current tree
    SOURCE_S3_NEW: POP_UPDATE_V2,
    # the 4 archived Qwen2.5 QWU cells, as audited
    SOURCE_QWU_REUSE: POP_UPDATE_V1,
    # newly generated pp_*/frz_* CPU endpoints. sim_perfect_predictor now
    # passes gate_on="anchor" explicitly and records v2;
    # replay_frozen_offline inherits it through PP.build_config. Endpoints
    # written BEFORE that fix label themselves v1 and are deliberately NOT
    # rewritten -- they are accepted only under all_open/all_open and are
    # reported as a provenance line, exactly as the archived QWU cells are.
    SOURCE_CPU_ENDPOINT: POP_UPDATE_V2,
}

# The ONLY archived tags this wave may reuse, each pinned to the exact
# conceptual slot it may fill. A manifest cannot relabel one of these
# into a different slot.
QWU_REUSE_SLOTS = {
    "pofdqwu_qwen7b_b0_eaopen_w0p5_l1_esopen_s0_r100":
        ("qwen7b", "sft", 0.5, 1.0),
    "pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100":
        ("qwen7b", "sft", 1.0, 1.0),
    "pofdqwu_qwen7b_b1_eaopen_w0p5_l1_esopen_s0_r100":
        ("qwen7b", "fwdlam1", 0.5, 1.0),
    "pofdqwu_qwen7b_b1_eaopen_w1_l1_esopen_s0_r100":
        ("qwen7b", "fwdlam1", 1.0, 1.0),
}

# Canonical frozen SOURCE runs, per checkpoint: the run whose pred_raw[0]
# is the frozen served map every "distance to the frozen model" is
# measured against. The Qwen2.5 sha is pinned (derived by
# audit_qwen_mechanism.py and asserted in three places). The Qwen3-8B sha
# is NOT pinned -- it has not been re-derived from pred_raw[0] -- so it is
# taken as a parameter or DERIVED at runtime and RECORDED, never invented
# here. See check_frozen().
FROZEN_SOURCE = {
    "qwen7b": "pofdqmech_qwen7b_k0_ea1_w0p5_l1_es0p05_s0",
    "qwen3_8b": "pofdzsprior_qwen3_8b_w0p5_l0p2_es0_s0",
}

# movielens Action, 723 agents: sha256 over the float32 innate vector.
# Read off notes/pofd/perfect_prediction/pp_k1_w0p5_eaopen_esopen_sw1_s0_r300.pt
# (config["innate_sha256"], reproduced by recomputing over the stored
# tensor). Pinned so a cell built on a different agent set or a different
# agent ORDER cannot enter the grid.
CANONICAL_INNATE_SHA = (
    "be34f284f929e2198996a37b080c03eef5750e1917d90269cd3fde81a7b31b19")

# The ONE canonical frozen Qwen2.5 K=D=0 served vector on H100-80GB
# (mechanism diagnostic, 2026-08-20), sha256 over float32 bytes. Same
# constant as check_pofd_sanity.QMECH_CANONICAL_PRED_SHA and
# experiments/condor/manifest_qwen_mechanism.json. There is NO canonical
# constant for Qwen3-8B; supply it with --qwen3-frozen-sha.
QMECH_CANONICAL_PRED_SHA = (
    "1674ee5f8d833f46de672791d933e1d3bdeefb07484c2d110dec84ce71da30bb")

# Config fields that must hold the SAME value in every cell of the wave,
# reused cells included.
#
# This is check_kl_direction.check_one's `pins` dict with the Section 3
# deltas applied. That dict is a local, so it cannot be imported; the
# values here were verified field-by-field against the archived reuse
# cell notes/pofd/cluster/pofdqwu_qwen7b_b0_eaopen_w0p5_l1_esopen_s0_r100
# (lora_r 512, sft_lr 5e-05, sft_epochs 1, epoch_size 100, train_cap 723,
# icl_k/icl_days 0, ab_sweeps 1, n_labeled 723) so a reuse cell is not
# rejected by a pin invented here.
#
# DELTAS vs the archived gate: eps / eps_ai drop out of the pins (both
# gates are all_open, so both thresholds are INERT -- see INERT_FIELDS),
# and ai_gate_mode / peer_gate_mode are pinned to "all_open" instead.
#
# ASSUMPTION, FLAGGED: lora_r / sft_lr / sft_epochs / epoch_size /
# train_cap are inherited from the QWU wave, not restated in the Section
# 3 contract. If gen_pofd_sweep.py chose different values this gate fails
# LOUDLY, which is the intended failure direction.
SHARED_PINS = {
    "dataset": "movielens",
    "ml_target": "Action",
    "n_labeled": N_AGENTS,
    "train_cap": N_AGENTS,
    "pop_model": "ab",
    "ab_sweeps": 1,
    "seed": SEED,
    "seed_base_data": True,
    "anchor_mode": "fixed",
    "ai_gate_mode": "all_open",
    "peer_gate_mode": "all_open",
    "use_lora": True,
    "fresh_each_round": True,
    "icl_k": 0,
    "icl_days": 0,
    "lora_r": 512,
    "sft_epochs": 1,
    "epoch_size": 100,
}
SHARED_PINS_FLOAT = {"sft_lr": 5e-5}

# recorded but INERT under all_open gates; reported, never failed on value
INERT_FIELDS = ("eps", "eps_ai")

# fields required to be IDENTICAL across the grid but whose value this
# gate does not pin (it only pins consistency)
CONSISTENCY_FIELDS = ("max_steps", "population_update", "git_sha")

ABSENT = object()


# ------------------------------------------------------------- helpers
def _num(v: float) -> str:
    """project tag grammar: 0.5 -> '0p5', 1 -> '1', 0.2 -> '0p2'."""
    return f"{float(v):g}".replace(".", "p").replace("-", "m")


def _unnum(s: str):
    """Inverse of _num, with a round-trip guard. '0p50' is NOT 0.5."""
    try:
        v = float(s.replace("m", "-").replace("p", "."))
    except ValueError:
        return None
    return v if _num(v) == s else None


def _sha_t(t) -> str:
    """sha256 over a tensor's float32 bytes (check_pofd_sanity._sha_t)."""
    a = torch.as_tensor(t).detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def _sha_obj(o) -> str:
    return hashlib.sha256(
        json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _feq(a, b, tol=1e-9):
    a, b = _as_float(a), _as_float(b)
    return a is not None and b is not None and abs(a - b) <= tol


def served_map_stats(vec) -> dict:
    """Degeneracy of ONE served vector. Reported, never failed.

    Values are rounded to 1e-6 before counting: the served map is parsed
    from decimal text, so exact float equality is the right notion, and
    the rounding only removes float32 print dust.

    effective modes = exp(Shannon entropy of the EXACT value
    distribution) -- not a binned histogram. A map that serves two values
    to 98.9% of agents must read as ~2 modes, and a 50-bin histogram
    would not say that.
    """
    import numpy as np
    v = np.asarray(vec, dtype=np.float64).ravel()
    n = v.size
    if n == 0:
        return {"n_distinct": -1, "mode_share": float("nan"),
                "top3_share": float("nan"), "eff_modes": float("nan"),
                "sd": float("nan"), "mean": float("nan")}
    vals, counts = np.unique(np.round(v, 6), return_counts=True)
    p = counts / float(n)
    order = np.sort(p)[::-1]
    h = float(-(p * np.log(p)).sum())
    return {
        "n_distinct": int(vals.size),
        "mode_share": float(order[0]),
        "top3_share": float(order[:3].sum()),
        "eff_modes": float(math.exp(h)),
        "sd": float(v.std()),
        "mean": float(v.mean()),
    }


# ------------------------------------------------------------ tag grammar
_ARM_RE = re.compile(r"^(?:sft|fwdlam(?P<f>[0-9pm]+)|revlam(?P<r>[0-9pm]+))$")
_TAG_RE = re.compile(
    r"^pofds3(?P<smk>smk)?_"
    r"(?P<model>qwen7b|qwen3_8b)_"
    r"(?P<arm>[a-z0-9]+)_"
    r"eaopen_w(?P<beta>[0-9pm]+)_k(?P<k>[0-9pm]+)_esopen_"
    r"(?P<optok>[a-z0-9]+)_s(?P<seed>\d+)_r(?P<rounds>\d+)$")


def arm_semantics(arm: str):
    """arm token -> (training_style, kl_beta, kl_direction|None) or None.

    'sft' is direction-NEUTRAL: lambda is 0, so no KL term exists and
    whatever kl_direction the runner happened to record is meaningless.
    """
    m = _ARM_RE.match(arm)
    if not m:
        return None
    if arm == "sft":
        return ("sft", 0.0, None)
    if m.group("f") is not None:
        lam = _unnum(m.group("f"))
        return None if lam is None else ("sft_kl", lam, "forward")
    lam = _unnum(m.group("r"))
    return None if lam is None else ("sft_kl", lam, "reverse")


def parse_tag(tag: str):
    """(slot_dict, errs). slot_dict is None when the tag is not in the
    Section 3 grammar at all (which is how a reuse cell arrives)."""
    m = _TAG_RE.match(tag)
    if not m:
        return None, []
    errs = []
    beta, k = _unnum(m.group("beta")), _unnum(m.group("k"))
    if beta is None:
        errs.append(f"tag w-token {m.group('beta')!r} is not canonical "
                    f"(0.5 must spell '0p5', 1 must spell '1')")
    if k is None:
        errs.append(f"tag k-token {m.group('k')!r} is not canonical")
    sem = arm_semantics(m.group("arm"))
    if sem is None:
        errs.append(f"tag arm {m.group('arm')!r} is not a Section 3 arm "
                    f"(want sft, fwdlam<L>, revlam<L>)")
    if m.group("optok") != OPTOK:
        errs.append(f"tag operator token {m.group('optok')!r} != {OPTOK!r}")
    if int(m.group("seed")) != SEED:
        errs.append(f"tag seed {m.group('seed')} != {SEED}")
    slot = {
        "smoke": bool(m.group("smk")),
        "model": m.group("model"),
        "arm": m.group("arm"),
        "beta": beta, "k": k,
        "sem": sem,
        "optok": m.group("optok"),
        "seed": int(m.group("seed")),
        "tag_rounds": int(m.group("rounds")),
    }
    return slot, errs


def slot_tag(model, arm, beta, k, rounds=PROD_ROUNDS, smoke=False):
    pre = SMOKE_PREFIX if smoke else PROD_PREFIX
    return (f"{pre}{model}_{arm}_eaopen_w{_num(beta)}_k{_num(k)}"
            f"_esopen_{OPTOK}_s{SEED}_r{rounds}")


def conceptual_grid():
    """The 50 conceptual TRAINED cells, in report order."""
    slots = []
    for model in MODELS:
        for beta, k in ENVS:
            slots.append((model, "sft", beta, k))
            for lam in FWD_LAMBDAS:
                slots.append((model, f"fwdlam{_num(lam)}", beta, k))
            if (beta, k) in REV_ENVS:
                for lam in REV_LAMBDAS:
                    slots.append((model, f"revlam{_num(lam)}", beta, k))
    return slots


# -------------------------------------------------------- reuse manifest
# Accepted spellings for the manifest's cell fields. The reuse audit is
# written by a different agent; rather than guess ONE spelling and fail
# opaquely, every plausible spelling is listed here and a miss names the
# whole required schema.
_MF_MODEL = ("model", "model_slug", "base_model_slug")
_MF_ARM = ("arm", "arm_token")
_MF_BETA = ("beta", "w_plat", "w")
_MF_K = ("k", "innate_lambda", "innate_k")
_MF_TAG = ("run_tag", "tag", "reuse_tag", "source_run_tag")
_MF_DIR = ("run_dir", "reuse_dir", "source_run_dir", "path")
_MF_STATUS = ("status", "verdict", "decision")
# hash field -> which computed digest it must equal
_MF_HASHES = {
    "pred_raw_sha256": "pred_raw",
    "op_raw_sha256": "op_raw",
    "config_sha256": "config",
    "innate_sha256": "innate",
    # generic: must match ANY of the computed digests
    "artifact_sha256": "*",
    "sha256": "*",
    "trajectory_sha256": "*",
}
_REUSE_WORDS = {"reuse", "reused", "pass", "ok", "accept", "accepted"}

MANIFEST_SCHEMA_HELP = """\
required reuse-manifest schema (notes/pofd/section3/reuse_manifest.json):

  {"key": "section3", "cells": [
     {"model": "qwen7b",          # or model_slug
      "arm": "fwdlam1",           # Section 3 arm token
      "beta": 0.5,                # or w_plat
      "k": 1,                     # or innate_lambda
      "status": "reused",         # or verdict: "REUSE"
      "run_tag": "pofdqwu_...",   # the ARCHIVED tag
      "run_dir": "runs/pokec_gated_lm/pofdqwu_...",
      "pred_raw_sha256": "<sha256 of trajectory.pt['pred_raw'] f32 bytes>",
      "op_raw_sha256":   "<sha256 of trajectory.pt['op_raw']  f32 bytes>",
      "horizon_prefix_ok": false  # only if n_rounds > 100 is admitted
     }, ...]}

At least ONE hash field must be present per reused cell. A generic
"artifact_sha256" is accepted if it equals any of the computed digests.\
"""


def _mf_get(cell, names, default=ABSENT):
    for n in names:
        if n in cell:
            return cell[n]
    return default


def load_manifest(path, out):
    """(by_tag, by_slot, ok). Absence is NOT an error until a run dir
    actually needs attestation."""
    p = Path(path)
    if not p.exists():
        out.append(f"[check_s3] note: no reuse manifest at {p} -- every run "
                   f"dir must then carry a {PROD_PREFIX}* tag")
        return {}, {}, True
    try:
        mf = json.loads(p.read_text())
    except Exception as e:                                # noqa: BLE001
        out.append(f"[check_s3] FAIL manifest: {p} is not readable JSON ({e})")
        return {}, {}, False
    cells = mf.get("cells")
    if not isinstance(cells, list):
        out.append(f"[check_s3] FAIL manifest: {p} has no 'cells' list.\n"
                   f"{MANIFEST_SCHEMA_HELP}")
        return {}, {}, False
    by_tag, by_slot, ok = {}, {}, True
    for i, c in enumerate(cells):
        if not isinstance(c, dict):
            out.append(f"[check_s3] FAIL manifest: cells[{i}] is not an object")
            ok = False
            continue
        tag = _mf_get(c, _MF_TAG)
        if tag is not ABSENT and tag:
            by_tag[str(tag)] = c
            by_tag[os.path.basename(str(tag).rstrip("/"))] = c
        d = _mf_get(c, _MF_DIR)
        if d is not ABSENT and d:
            by_tag.setdefault(os.path.basename(str(d).rstrip("/")), c)
        model = _mf_get(c, _MF_MODEL)
        arm = _mf_get(c, _MF_ARM)
        beta = _as_float(_mf_get(c, _MF_BETA, None))
        k = _as_float(_mf_get(c, _MF_K, None))
        if ABSENT not in (model, arm) and beta is not None and k is not None:
            by_slot[(str(model), str(arm), beta, k)] = c
    return by_tag, by_slot, ok


def _manifest_says_reuse(cell) -> bool:
    st = _mf_get(cell, _MF_STATUS, "")
    return str(st).strip().lower() in _REUSE_WORDS


def _manifest_verdict(cell, digests, tag, out):
    """Hard requirement for a NON-grammar run dir: the manifest must mark
    it REUSE and at least one recorded hash must match the artifact."""
    ok = True
    if not _manifest_says_reuse(cell):
        out.append(f"[check_s3] FAIL {tag}: manifest entry does not say "
                   f"REUSE (status={_mf_get(cell, _MF_STATUS, '<absent>')!r}) "
                   f"-- an archived cell enters the grid only on an explicit "
                   f"reuse verdict")
        ok = False
    present = [k for k in _MF_HASHES if k in cell and cell[k]]
    if not present:
        out.append(f"[check_s3] FAIL {tag}: manifest entry carries NO artifact "
                   f"hash. Tag similarity is not provenance.\n"
                   f"{MANIFEST_SCHEMA_HELP}\n"
                   f"           computed here: " +
                   "; ".join(f"{k}={v}" for k, v in digests.items()))
        return False, []
    matched = []
    for key in present:
        want = str(cell[key]).strip().lower()
        which = _MF_HASHES[key]
        if which == "*":
            hit = [n for n, v in digests.items() if v == want]
            if hit:
                matched.append(f"{key}~{hit[0]}")
            else:
                out.append(
                    f"[check_s3] FAIL {tag}: manifest {key}={want[:16]}... "
                    f"matches none of the computed digests (" +
                    "; ".join(f"{n}={v[:16]}..." for n, v in digests.items()) +
                    ")")
                ok = False
        else:
            got = digests.get(which)
            if got is None:
                out.append(f"[check_s3] FAIL {tag}: manifest records {key} but "
                           f"{which} could not be read from the artifact")
                ok = False
            elif got != want:
                out.append(f"[check_s3] FAIL {tag}: manifest {key}="
                           f"{want[:16]}... != artifact {which} sha256 "
                           f"{got[:16]}... -- the manifest was written against "
                           f"a DIFFERENT artifact than the one on disk")
                ok = False
            else:
                matched.append(key)
    return ok, matched


# ------------------------------------------------------------- artifacts
def _read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()
            if l.strip()]


def _read_gz_jsonl(path):
    with gzip.open(path, "rt") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _peer_stream_key(rows):
    """Per-round (peer_pairs, accepted), the cross-arm world fingerprint.

    WHY THIS IS AN EXACT IDENTITY TEST HERE. gp.ab_sweep draws pairs with
    weights d**(-gamma) * adj[ini]; the project pins gamma = 0, so the
    weights collapse to the adjacency row and pair SELECTION does not
    read opinions at all. The Luby priorities come from the dedicated
    peer Generator, which is seeded from the run seed and consumed in a
    fixed pattern. Acceptance is the only opinion-dependent step, and the
    peer gate here is all_open, which accepts unconditionally.

    So under (gamma = 0, peer_gate_mode = all_open, same seed, same
    graph, same agent order) these two counters are bit-identical across
    every arm of the wave. Any difference means a different graph, a
    different agent ORDER, a different seed, or a different sweep count.
    """
    out = []
    for r in rows:
        if "peer_pairs" not in r or "accepted" not in r:
            return None
        out.append((int(r["round"]) if r.get("round") is not None else -1,
                    int(r["peer_pairs"]), int(r["accepted"])))
    return tuple(out)


def check_one(run_dir, smoke, mf_by_tag, out, notes):
    """Gate ONE run dir. Returns a per-cell record (never None)."""
    run_dir = str(run_dir).rstrip("/")
    tag = os.path.basename(run_dir)
    rec = {"tag": tag, "run_dir": run_dir, "ok": False, "slot": None,
           "reuse": False, "deviations": [], "report": {}}
    want_rounds = SMOKE_ROUNDS if smoke else PROD_ROUNDS

    cf, tf = os.path.join(run_dir, "config.json"), \
        os.path.join(run_dir, "trajectory.pt")
    if not os.path.exists(cf):
        out.append(f"[check_s3] FAIL {tag}: no config.json")
        return rec
    if not os.path.exists(tf):
        out.append(f"[check_s3] FAIL {tag}: no trajectory.pt -- the run did "
                   f"not finish")
        return rec
    try:
        cfg = json.load(open(cf))
    except Exception as e:                                # noqa: BLE001
        out.append(f"[check_s3] FAIL {tag}: config.json unreadable ({e})")
        return rec
    try:
        d = torch.load(tf, map_location="cpu", weights_only=False)
    except Exception as e:                                # noqa: BLE001
        out.append(f"[check_s3] FAIL {tag}: trajectory.pt unreadable ({e})")
        return rec

    ok = True
    slot, tag_errs = parse_tag(tag)
    for e in tag_errs:
        out.append(f"[check_s3] FAIL {tag}: {e}")
        ok = False
    source_kind = SOURCE_S3_NEW if slot is not None else None

    # ---------------- reuse attestation, if this is not our grammar ----
    if slot is None:
        op_t, pr_t = d.get("op_raw"), d.get("pred_raw")
        inn_t = d.get("innate")
        digests = {}
        if torch.is_tensor(pr_t) and pr_t.numel():
            digests["pred_raw"] = _sha_t(pr_t)
        if torch.is_tensor(op_t) and op_t.numel():
            digests["op_raw"] = _sha_t(op_t)
        if torch.is_tensor(inn_t) and inn_t.numel():
            digests["innate"] = _sha_t(inn_t)
        digests["config"] = _sha_obj(cfg)
        cell = mf_by_tag.get(tag)
        if cell is None:
            out.append(
                f"[check_s3] FAIL {tag}: tag is not in the Section 3 grammar "
                f"({PROD_PREFIX}{{model}}_{{arm}}_eaopen_w{{b}}_k{{k}}_esopen_"
                f"{OPTOK}_s0_r100) and no reuse-manifest entry names it. A run "
                f"dir enters this grid by grammar OR by attested reuse, never "
                f"by resemblance.\n{MANIFEST_SCHEMA_HELP}")
            return rec
        mok, matched = _manifest_verdict(cell, digests, tag, out)
        ok &= mok
        model = str(_mf_get(cell, _MF_MODEL, ""))
        arm = str(_mf_get(cell, _MF_ARM, ""))
        beta = _as_float(_mf_get(cell, _MF_BETA, None))
        k = _as_float(_mf_get(cell, _MF_K, None))
        sem = arm_semantics(arm)
        if model not in MODELS or sem is None or beta is None or k is None:
            out.append(f"[check_s3] FAIL {tag}: reuse-manifest entry does not "
                       f"name a Section 3 slot (model={model!r} arm={arm!r} "
                       f"beta={beta!r} k={k!r})")
            return rec
        # WHICH archived source is this? The expected operator marker and
        # the permitted slot both hang off the answer, so an unregistered
        # source is rejected rather than defaulted.
        if tag in QWU_REUSE_SLOTS:
            source_kind = SOURCE_QWU_REUSE
            pinned = QWU_REUSE_SLOTS[tag]
            if (model, arm, beta, k) != pinned:
                out.append(
                    f"[check_s3] FAIL {tag}: the reuse manifest assigns this "
                    f"archived cell to slot {(model, arm, beta, k)}, but "
                    f"{tag} is pinned to {pinned}. A manifest cannot relabel "
                    f"an archived run into a different conceptual cell.")
                ok = False
        else:
            out.append(
                f"[check_s3] FAIL {tag}: not a registered reuse source. The "
                f"only archived tags this wave may reuse are "
                f"{sorted(QWU_REUSE_SLOTS)}. An unregistered source has no "
                f"expected operator marker, so it cannot be validated.")
            return rec
        slot = {"smoke": False, "model": model, "arm": arm, "beta": beta,
                "k": k, "sem": sem, "optok": None, "seed": SEED,
                "tag_rounds": None}
        rec["reuse"] = True
        rec["deviations"].append(
            f"tag grammar waived: archived tag {tag!r} (source "
            f"{source_kind!r}), bound to slot by config fields + manifest "
            f"hash ({', '.join(matched) or 'none'})")
    rec["slot"] = (slot["model"], slot["arm"], slot["beta"], slot["k"])
    style, want_lam, want_dir = slot["sem"]

    if smoke and not rec["reuse"] and not slot["smoke"]:
        out.append(f"[check_s3] FAIL {tag}: --smoke wants a {SMOKE_PREFIX}* "
                   f"cell")
        ok = False
    if not smoke and slot.get("smoke"):
        out.append(f"[check_s3] FAIL {tag}: a smoke cell cannot stand in for a "
                   f"production cell")
        ok = False

    def bad(msg):
        nonlocal ok
        out.append(f"[check_s3] FAIL {tag}: {msg}")
        ok = False

    # ---------------- horizon ------------------------------------------
    n_rounds = cfg.get("n_rounds")
    if slot.get("tag_rounds") is not None and slot["tag_rounds"] != want_rounds:
        bad(f"tag declares r{slot['tag_rounds']}, this mode wants "
            f"r{want_rounds}")
    if int(n_rounds or -1) != want_rounds:
        if rec["reuse"] and int(n_rounds or -1) > want_rounds:
            cell = mf_by_tag.get(tag, {})
            if cell.get("horizon_prefix_ok") is True:
                rec["deviations"].append(
                    f"horizon: archived n_rounds={n_rounds}; first "
                    f"{want_rounds} rounds admitted as a prefix by the "
                    f"manifest (horizon_prefix_ok)")
            else:
                bad(f"n_rounds={n_rounds} != {want_rounds}. A longer archived "
                    f"run is a PREFIX CANDIDATE, not an equivalent cell: the "
                    f"claim that its first {want_rounds} rounds equal a "
                    f"{want_rounds}-round run is a statement about the data "
                    f"path and the RNG stream, so it must be admitted by the "
                    f"manifest ('horizon_prefix_ok': true), not assumed here")
        else:
            bad(f"n_rounds={n_rounds} != {want_rounds}")

    # ---------------- checkpoint, arm, direction, lambda ---------------
    if cfg.get("base_model") != MODELS[slot["model"]]:
        bad(f"base_model={cfg.get('base_model')!r}, slot {slot['model']!r} "
            f"means {MODELS[slot['model']]!r}")
    if cfg.get("training_style") != style:
        bad(f"training_style={cfg.get('training_style')!r} != {style!r} for "
            f"arm {slot['arm']!r}")
    got_lam = _as_float(cfg.get("kl_beta"))
    if got_lam is None or not _feq(got_lam, want_lam):
        bad(f"kl_beta={cfg.get('kl_beta')!r} != {want_lam:g} -- lambda IS the "
            f"ladder axis")
    got_dir = cfg.get("kl_direction")
    if want_dir is None:
        # lambda = 0: no KL term exists, so any recorded direction is
        # meaningless. Report it; do not read a ladder position from it.
        rec["report"]["sft_kl_direction_recorded"] = got_dir
        if got_lam not in (None,) and not _feq(got_lam, 0.0):
            bad("the sft arm must carry kl_beta == 0")
    else:
        if got_dir != want_dir:
            bad(f"DIRECTION MISMATCH: arm {slot['arm']!r} means "
                f"{want_dir!r}, config recorded {got_dir!r}. This is the one "
                f"failure that would invert the robustness check and it is "
                f"invisible in every downstream artifact")
        if got_dir not in ("forward", "reverse"):
            bad(f"kl_direction={got_dir!r} is neither direction")

    # ---------------- reference is the RAW base ------------------------
    if cfg.get("kl_ref_adapter"):
        bad(f"kl_ref_adapter={cfg.get('kl_ref_adapter')!r} must be EMPTY -- "
            f"the anchor is the raw base checkpoint, which is what makes "
            f"lambda mean the same thing on every rung")

    # ---------------- environment --------------------------------------
    if not _feq(cfg.get("w_plat"), slot["beta"]):
        bad(f"w_plat={cfg.get('w_plat')!r} != tag beta {slot['beta']:g}")
    if not _feq(cfg.get("innate_lambda"), slot["k"]):
        bad(f"innate_lambda={cfg.get('innate_lambda')!r} != tag k "
            f"{slot['k']:g}")
    env = (slot["beta"], slot["k"])
    if env not in ENVS:
        bad(f"environment (beta={slot['beta']:g}, k={slot['k']:g}) is not one "
            f"of {[(f'{b:g}', f'{kk:g}') for b, kk in ENVS]}"
            + (" -- (1, 0.2) is deliberately NOT in this design"
               if env == (1.0, 0.2) else ""))
    if want_dir == "reverse" and env not in REV_ENVS:
        bad(f"reverse KL is declared only in {[ENV_LABEL[e] for e in REV_ENVS]}"
            f", not in environment {ENV_LABEL.get(env, env)}")
    if want_dir == "reverse" and not any(_feq(want_lam, l)
                                         for l in REV_LAMBDAS):
        bad(f"reverse lambda {want_lam:g} is not in {REV_LAMBDAS}")
    if want_dir == "forward" and not any(_feq(want_lam, l)
                                         for l in FWD_LAMBDAS):
        bad(f"forward lambda {want_lam:g} is not in {FWD_LAMBDAS}")

    # ---------------- operator marker: PER SOURCE, never "either" --------
    # The v1/v2 equivalence holds ONLY under an all_open AI gate
    # (_gated_pop.py:205-206 returns the all-ones mask before the
    # reference is read at :209). So the gate modes are established
    # FIRST, and the equivalence is not permitted to apply otherwise.
    marker = cfg.get("population_update")
    rec["report"]["population_update"] = marker
    rec["report"]["source_kind"] = source_kind
    gates_open = (cfg.get("ai_gate_mode") == "all_open"
                  and cfg.get("peer_gate_mode") == "all_open")
    rec["report"]["gates_all_open"] = bool(gates_open)
    want_marker = EXPECTED_MARKER.get(source_kind)
    if marker is None:
        bad("population_update marker ABSENT -- a run written before "
            "2026-07-27 used the legacy order (peer sweep, gated blend, then "
            "an innate re-anchor per sweep), which is a DIFFERENT round "
            "operator, not an inert relabelling")
    elif marker not in (POP_UPDATE_V1, POP_UPDATE_V2):
        bad(f"unknown population_update {marker!r} -- refusing to guess which "
            f"operator ran")
    elif want_marker is None:
        bad(f"source {source_kind!r} has no entry in EXPECTED_MARKER, so "
            f"there is no expected operator to check {marker!r} against. "
            f"Register the source before admitting the cell")
    elif marker != want_marker:
        bad(f"OPERATOR MARKER MISMATCH: source {source_kind!r} must record "
            f"{want_marker!r}, this cell recorded {marker!r}. "
            + ("A v1 marker on a new Section 3 cell means the run did not "
               "come from the current tree."
               if source_kind == SOURCE_S3_NEW else
               "A v2 marker on an archived cell means the artifact on disk "
               "is not the one the reuse audit read.")
            + " The v1/v2 equivalence under all_open gates is about the "
              "ARITHMETIC of one round; it is not a licence to accept "
              "either marker from either source.")
    elif marker == POP_UPDATE_V1 and not gates_open:
        bad(f"population_update={marker!r} with gates "
            f"ai={cfg.get('ai_gate_mode')!r}/"
            f"peer={cfg.get('peer_gate_mode')!r}: the v1/v2 equivalence "
            f"holds ONLY when the AI gate is all_open (_gated_pop.py:205-206 "
            f"returns before the reference is read at :209). With a numeric "
            f"gate the two operators genuinely differ, so this artifact "
            f"cannot stand in for a v2 cell")
    elif marker != OPTOK_POP_UPDATE[OPTOK] and gates_open:
        rec["deviations"].append(
            f"operator marker {marker!r} -- the EXPECTED value for source "
            f"{source_kind!r}; numerically identical to "
            f"{OPTOK_POP_UPDATE[OPTOK]!r} here because both gates are "
            f"all_open (_gated_pop.py:205-206 vs :209)")
    recorded_ref = cfg.get("ai_gate_reference")
    if recorded_ref is not None and marker in (POP_UPDATE_V1, POP_UPDATE_V2):
        implied = "x0" if marker == POP_UPDATE_V1 else "anchor"
        if recorded_ref != implied:
            bad(f"ai_gate_reference={recorded_ref!r} but population_update="
                f"{marker!r} implies {implied!r} -- the marker and the "
                f"recorded reference disagree about which operator ran")

    # ---------------- shared pins ---------------------------------------
    for key, val in SHARED_PINS.items():
        got = cfg.get(key, ABSENT)
        if isinstance(val, bool):
            if got is ABSENT or bool(got) != val:
                bad(f"{key}={got if got is not ABSENT else '<absent>'!r}, "
                    f"expected {val!r}")
        elif got is ABSENT or got != val:
            bad(f"{key}={got if got is not ABSENT else '<absent>'!r}, "
                f"expected {val!r}")
    for key, val in SHARED_PINS_FLOAT.items():
        if not _feq(cfg.get(key), val, tol=1e-12):
            bad(f"{key}={cfg.get(key)!r}, expected {val!r}")
    rec["report"]["inert"] = {k: cfg.get(k) for k in INERT_FIELDS}

    # ---------------- Qwen3 thinking template ---------------------------
    ct = cfg.get("chat_thinking", ABSENT)
    if slot["model"] == "qwen3_8b":
        if ct is ABSENT:
            bad("chat_thinking is ABSENT on a Qwen3 cell -- the key is written "
                "only when CHAT_THINKING carries a directive, so its absence "
                "means the hybrid-reasoning template RAN. Thinking must be "
                "explicitly disabled (CHAT_THINKING=0)")
        elif ct is not False and ct != 0:
            bad(f"chat_thinking={ct!r} on a Qwen3 cell; must be False")
    elif ct is not ABSENT and (ct is True or ct == 1):
        bad(f"chat_thinking={ct!r} on {slot['model']} -- a non-default chat "
            f"template was used")

    # ---------------- hardware / serving provenance ----------------------
    hw = cfg.get("hardware") or {}
    gpu = hw.get("gpu_name") or ""
    # exact-match, same as check_kl_direction.check_one
    if gpu != H100:
        bad(f"gpu_name={gpu or '<absent>'!r}, expected {H100!r}: greedy "
            f"decoding is bit-reproducible only within one GPU architecture, "
            f"and the archived A100 frozen cell differs from the H100 prior "
            f"in 17 of 723 agents")
    rec["report"]["gpu_name"] = gpu
    rec["report"]["torch_version"] = hw.get("torch_version")
    rec["report"]["transformers_version"] = hw.get("transformers_version")
    sem_flag = cfg.get("serve_eval_mode", ABSENT)
    if sem_flag is ABSENT:
        if rec["reuse"]:
            rec["deviations"].append(
                "serve_eval_mode ABSENT (field introduced 2026-08-21): LoRA "
                "dropout during serving is NOT certifiable from this "
                "artifact; only git history plus timestamps separate it from "
                "a pre-fix run")
        else:
            bad("serve_eval_mode is absent -- a cell generated for this wave "
                "postdates the field, so absence means the runner is not the "
                "one this wave declares")
    elif sem_flag is not True:
        bad(f"serve_eval_mode={sem_flag!r}; LoRA dropout may have been live "
            f"while serving")

    # ---------------- the trajectory tensors -----------------------------
    op = d.get("op_raw")
    pr = d.get("pred_raw")
    shape = (want_rounds, N_AGENTS)
    for name, t in (("op_raw", op), ("pred_raw", pr)):
        if not torch.is_tensor(t) or t.numel() == 0:
            bad(f"{name} missing or empty")
            continue
        tt = t.float()
        eff = tuple(tt.shape)
        if rec["reuse"] and eff[:1] > shape[:1] and eff[1:] == shape[1:]:
            tt = tt[:want_rounds]
            eff = tuple(tt.shape)
        if eff != shape:
            bad(f"{name} shape {tuple(t.shape)} != {shape} -- incomplete "
                f"{'serving' if name == 'pred_raw' else 'population record'}")
            continue
        if not torch.isfinite(tt).all():
            n_bad = int((~torch.isfinite(tt)).sum())
            bad(f"{name} holds {n_bad} non-finite value(s)")
        if name == "op_raw":
            rec["op"] = tt
        else:
            rec["pred"] = tt

    innate = d.get("innate")
    if not torch.is_tensor(innate) or innate.numel() != N_AGENTS:
        bad(f"innate vector missing or not {N_AGENTS} long -- it is t=0 of "
            f"every trajectory")
    else:
        ish = _sha_t(innate)
        rec["report"]["innate_sha256"] = ish
        if ish != CANONICAL_INNATE_SHA:
            bad(f"innate sha256 {ish[:16]}... != canonical "
                f"{CANONICAL_INNATE_SHA[:16]}... -- a different agent set, or "
                f"the same agents in a different ORDER")
    profiles = d.get("profiles")
    if profiles is None:
        bad("profiles missing from trajectory.pt -- the agent identities "
            "cannot be compared across arms")
    else:
        rec["report"]["profiles_sha256"] = _sha_obj(profiles)

    # ---------------- parse failures: exactly zero ------------------------
    gz = os.path.join(run_dir, "raw_gen_log.json.gz")
    if not os.path.exists(gz):
        bad("no raw_gen_log.json.gz -- SAVE_RAW_GEN was off, so the parse-"
            "failure rate cannot be established from the artifact. The "
            "trajectory rows carry summaries of the ALREADY-PARSED vector, "
            "which is exactly where a 100%-failure round looks clean")
    else:
        try:
            rows = _read_gz_jsonl(gz)
        except Exception as e:                            # noqa: BLE001
            rows = None
            bad(f"raw_gen_log.json.gz unreadable ({e})")
        if rows is not None:
            if len(rows) < want_rounds:
                bad(f"raw_gen_log has {len(rows)} round(s), expected "
                    f"{want_rounds}")
            elif len(rows) > want_rounds and not rec["reuse"]:
                bad(f"raw_gen_log has {len(rows)} round(s), expected "
                    f"{want_rounds}")
            rows = rows[:want_rounds]
            bad_rounds = [(r.get("round"), r.get("parse_fail_frac"))
                          for r in rows
                          if r.get("parse_fail_frac") is None
                          or float(r["parse_fail_frac"]) != 0.0]
            if bad_rounds:
                bad(f"parse failures in {len(bad_rounds)} round(s), e.g. round "
                    f"{bad_rounds[0][0]} frac={bad_rounds[0][1]}. A parse "
                    f"failure is recorded as a confident constant, so ANY "
                    f"nonzero rate is a hard failure")
            short = [(r.get("round"), len(r.get("parsed") or []))
                     for r in rows if len(r.get("parsed") or []) != N_AGENTS]
            if short:
                bad(f"round {short[0][0]} parsed {short[0][1]} of {N_AGENTS} "
                    f"agents -- incomplete serving")

    # ---------------- telemetry: training and the anchor term -------------
    tel_p = os.path.join(run_dir, "telemetry.json")
    if not os.path.exists(tel_p):
        bad("no telemetry.json -- l_init and the gradient norms live there, "
            "NOT in trajectory.pt, so training cannot be confirmed")
    else:
        try:
            tel = _read_jsonl(tel_p)
        except Exception as e:                            # noqa: BLE001
            tel = None
            bad(f"telemetry.json unreadable ({e})")
        if tel is not None:
            tel = tel[:want_rounds] if len(tel) >= want_rounds else tel
            if len(tel) != want_rounds:
                bad(f"telemetry has {len(tel)} round(s), expected "
                    f"{want_rounds}")
            li = [_as_float(r.get("l_init")) for r in tel]
            if any(v is None for v in li):
                bad(f"l_init missing in {sum(v is None for v in li)} round(s) "
                    f"-- optimizer telemetry incomplete")
            elif not all(math.isfinite(v) for v in li):
                bad("non-finite training loss (l_init)")
            gn = [_as_float(r.get("grad_norm0")) for r in tel]
            if any(v is None for v in gn):
                bad(f"grad_norm0 missing in {sum(v is None for v in gn)} "
                    f"round(s) -- optimizer telemetry incomplete")
            elif max(gn) == 0.0:
                bad("grad_norm0 is 0 in EVERY round -- the optimizer never "
                    "moved, so this arm did not train")
            elif not all(math.isfinite(v) for v in gn):
                bad("non-finite grad_norm0")
            klg = [_as_float(r.get("grad_kl_norm0")) for r in tel]
            has_kl = any(v is not None for v in klg)
            if want_lam > 0:
                if not has_kl:
                    bad("no grad_kl_norm0 recorded on a REGULARIZED arm -- "
                        "cannot confirm the KL term was ever applied")
                else:
                    vals = [v for v in klg if v is not None]
                    if any(v is None for v in klg):
                        bad(f"grad_kl_norm0 missing in "
                            f"{sum(v is None for v in klg)} round(s)")
                    if not all(math.isfinite(v) for v in vals):
                        bad("non-finite KL gradient norm")
                    after0 = [v for v, r in zip(klg, tel)
                              if v is not None
                              and int(r.get("round", -1)) != 0]
                    if after0 and max(after0) <= 0.0:
                        bad("grad_kl_norm0 is 0 in every round AFTER round 0 "
                            "-- the anchor contributed no gradient, so this "
                            "arm is ordinary SFT wearing a lambda. (Round 0 "
                            "is exempt and must be: a fresh LoRA at round 0 "
                            "IS the reference.)")
                    rec["report"]["grad_kl_norm0_max_after_r0"] = (
                        max(after0) if after0 else float("nan"))
            elif has_kl and max(v for v in klg if v is not None) > 0.0:
                bad("the sft arm (lambda = 0) recorded a NONZERO anchor "
                    "gradient -- it was not trained at lambda = 0")
            rec["peer_key"] = None
            rec["report"]["l_init_first"] = li[0] if li and li[0] else None

    # ---------------- shared world: peer stream fingerprint ---------------
    traj_rows = d.get("trajectory")
    if not isinstance(traj_rows, list) or len(traj_rows) < want_rounds:
        bad(f"trajectory rows missing or short "
            f"({len(traj_rows) if isinstance(traj_rows, list) else None} < "
            f"{want_rounds})")
    else:
        key = _peer_stream_key(traj_rows[:want_rounds])
        if key is None:
            bad("trajectory rows carry no peer_pairs/accepted counters -- the "
                "cross-arm graph/order/RNG identity cannot be established")
        else:
            rec["peer_key"] = key

    # ---------------- probe set (reported) --------------------------------
    ps = os.path.join(run_dir, "probe_set.json")
    if os.path.exists(ps):
        try:
            pj = json.loads(Path(ps).read_text())
            rec["report"]["probe_n"] = len(pj.get("agent_idx") or [])
        except Exception:                                 # noqa: BLE001
            rec["report"]["probe_n"] = -1
    else:
        notes.append(f"[check_s3] note {tag}: no probe_set.json")
        rec["report"]["probe_n"] = None

    # ---------------- COLLAPSE REPORT (never a failure) --------------------
    # THE ONE BEHAVIOURAL INVERSION vs check_kl_direction.check_one, which
    # hard-fails BOTH of the conditions measured here:
    #   "served map is CONSTANT across agents in every round"
    #   "served vector is bit-identical in EVERY round"
    # Section 3 must not: it is asking what a rising lambda does to a
    # population that is climbing toward a frozen model whose own served
    # map is near-binary, so degeneracy is the measurement, not a defect.
    # Both conditions are recorded below and printed; neither touches
    # `ok`. Nothing else about the archived gate's semantics is changed.
    if "pred" in rec:
        pred = rec["pred"]
        lo = min(LATE_LO, pred.shape[0])
        rec["report"]["final"] = served_map_stats(pred[-1].numpy())
        win = pred[lo - 1:LATE_HI] if pred.shape[0] >= LATE_LO else pred
        per = [served_map_stats(win[i].numpy()) for i in range(win.shape[0])]
        rec["report"]["late"] = {
            k: float(sum(p[k] for p in per) / len(per)) for k in
            ("n_distinct", "mode_share", "top3_share", "eff_modes", "sd",
             "mean")}
        rec["report"]["late_window"] = ([lo, min(LATE_HI, pred.shape[0])]
                                        if pred.shape[0] >= LATE_LO
                                        else [1, pred.shape[0]])
        # REPORTED, NOT FAILED (see above)
        rec["report"]["pred_constant_across_agents_every_round"] = bool(
            float(pred.std(dim=1, unbiased=False).max()) == 0.0)
        rec["report"]["pred_bit_identical_across_rounds"] = bool(
            pred.shape[0] > 1 and float((pred - pred[0]).abs().max()) == 0.0)
    if "op" in rec:
        opv = rec["op"]
        rec["report"]["pop_final_mean"] = float(opv[-1].mean())
        rec["report"]["pop_final_sd"] = float(opv[-1].std(unbiased=False))

    rec["cfg"] = cfg
    rec["ok"] = ok
    return rec


# --------------------------------------------------------- frozen anchor
def check_frozen(paths, model_sha_overrides, out, notes):
    """Gate the frozen-model reference artifacts the figure anchors on.

    The trained cells anchor on the raw base checkpoint, which leaves no
    per-run hash. What IS hashable, and what the whole lambda ladder is
    measured against, is the frozen model's SERVED VECTOR: pred_raw[0] of
    the frozen SOURCE run, carried into the offline propagation artifacts
    (notes/pofd/frozen_replay/frz_*.pt) by replay_frozen_offline.py.

    NAMING HAZARD, LOUD ON PURPOSE. The frz_* filename encodes k, W, both
    gate modes, sweeps, seed and horizon but NOT the model. Section 3
    needs SIX of these (2 checkpoints x 3 environments) and the two
    checkpoints collide pairwise on filename. So the artifact's own
    config["base_model"] is the only provenance there is, it is checked
    here, and a same-name pair of different checkpoints is reported.

    Qwen2.5's canonical served-map sha256 is pinned. Qwen3-8B's is NOT
    (it has not been re-derived from pred_raw[0]); it is taken from
    --qwen3-8b-frozen-sha when supplied, otherwise DERIVED at runtime,
    RECORDED in the verdict, and cross-checked for agreement across every
    Qwen3 artifact in the invocation. It is never invented as a constant.
    """
    ok, seen, derived = True, [], {}
    for p in paths:
        p = Path(p)
        name = p.name
        kind = "pp" if name.startswith("pp_") else "frz"
        lbl = "perfect-prediction" if kind == "pp" else "frozen"
        if not p.exists():
            out.append(f"[check_s3] FAIL {lbl} {name}: file does not exist")
            ok = False
            continue
        try:
            a = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:                            # noqa: BLE001
            out.append(f"[check_s3] FAIL {lbl} {name}: unreadable ({e})")
            ok = False
            continue
        fcfg = a.get("config") or {}

        def cbad(msg, _n=name, _l=lbl):
            nonlocal ok
            out.append(f"[check_s3] FAIL {_l} {_n}: {msg}")
            ok = False

        # ---- schema: identical to a GPU run's, so one loader serves both
        for key in ("op_raw", "twin_raw", "pred_raw", "innate"):
            t = a.get(key)
            if not torch.is_tensor(t) or t.numel() == 0:
                cbad(f"{key} missing or empty. CPU endpoints carry the SAME "
                     f"schema as a GPU run (config / op_raw / twin_raw / "
                     f"pred_raw / innate); twin_raw is the matched "
                     f"no-platform process and must not be re-simulated")
        want_platform = ("perfect_prediction" if kind == "pp"
                         else "frozen_offline_replay")
        if fcfg.get("platform") != want_platform:
            cbad(f"platform={fcfg.get('platform')!r}, expected "
                 f"{want_platform!r}")

        # ---- environment and operator dials
        if fcfg.get("ai_gate_mode") != "all_open" or \
                fcfg.get("peer_gate_mode") != "all_open":
            cbad(f"gates ai={fcfg.get('ai_gate_mode')!r} "
                 f"peer={fcfg.get('peer_gate_mode')!r}, both must be "
                 f"all_open")
        if int(fcfg.get("seed", -1)) != SEED:
            cbad(f"seed={fcfg.get('seed')!r}, expected {SEED}")
        if int(fcfg.get("ab_sweeps", -1)) != 1:
            cbad(f"ab_sweeps={fcfg.get('ab_sweeps')!r}, expected 1")
        if not _feq(fcfg.get("gamma_bias"), 0.0):
            cbad(f"gamma_bias={fcfg.get('gamma_bias')!r}, expected 0 (the "
                 f"project pins it, and it is what makes the peer stream "
                 f"opinion-independent)")
        k_e, w_e = _as_float(fcfg.get("innate_k")), _as_float(fcfg.get("w_plat"))
        if (w_e, k_e) not in ENVS:
            cbad(f"(beta={w_e!r}, k={k_e!r}) is not a Section 3 environment "
                 f"{[(f'{b:g}', f'{kk:g}') for b, kk in ENVS]}")

        # ---- operator marker: PER SOURCE (see EXPECTED_MARKER)
        marker = fcfg.get("population_update")
        gates_open = (fcfg.get("ai_gate_mode") == "all_open"
                      and fcfg.get("peer_gate_mode") == "all_open")
        want_marker = EXPECTED_MARKER[SOURCE_CPU_ENDPOINT]
        if marker not in (POP_UPDATE_V1, POP_UPDATE_V2):
            cbad(f"population_update={marker!r} -- refusing to guess which "
                 f"operator ran")
        elif marker == POP_UPDATE_V1:
            # Pre-2026-08-22 endpoints label themselves v1. They are NOT
            # rewritten (project instruction), and under all_open/all_open
            # the two markers are the same operator
            # (_gated_pop.py:205-206 returns before :209 reads the
            # reference). Acceptable, and REPORTED -- never silent.
            if not gates_open:
                cbad(f"population_update={marker!r} with a non-all_open gate: "
                     f"the v1/v2 equivalence does not hold there")
            else:
                notes.append(
                    f"[check_s3] PROVENANCE {lbl} {name}: population_update="
                    f"{marker!r} (pre-2026-08-22 CPU endpoint; expected value "
                    f"for a newly generated one is {want_marker!r}). Accepted "
                    f"because both gates are all_open, which makes the two "
                    f"markers the same operator (_gated_pop.py:205-206 vs "
                    f":209). The artifact is deliberately NOT rewritten.")
        elif marker != want_marker:
            cbad(f"population_update={marker!r}, expected {want_marker!r}")

        pred = a.get("pred_raw")
        sha = None
        slug = None
        if torch.is_tensor(pred) and pred.numel():
            if kind == "frz":
                if not bool((pred == pred[0]).all()):
                    nvary = int((pred != pred[0]).any(dim=0).sum())
                    cbad(f"served vector is NOT constant across rounds "
                         f"({nvary} of {pred.shape[1]} agents vary). A frozen "
                         f"K=D=0 model cannot see the population, so this "
                         f"artifact is not what it claims")
                sha = _sha_t(pred[0])
                rec_sha = fcfg.get("frozen_pred_sha256")
                if rec_sha and rec_sha != sha:
                    cbad(f"recorded frozen_pred_sha256 {rec_sha[:16]}... != "
                         f"actual {sha[:16]}...")
                base = fcfg.get("base_model")
                slug = next((s for s, b in MODELS.items() if b == base), None)
                if slug is None:
                    cbad(f"base_model={base!r} is not one of this wave's "
                         f"checkpoints. The frz_* filename does NOT encode "
                         f"the model, so the artifact config is the only "
                         f"provenance there is")
                else:
                    src = fcfg.get("source_run_tag")
                    want = model_sha_overrides.get(slug)
                    if want is None and slug == "qwen7b":
                        want = QMECH_CANONICAL_PRED_SHA
                    if want is None:
                        # NOT pinned here as a constant: derive, record,
                        # and require every artifact of this checkpoint in
                        # the invocation to agree.
                        prev = derived.get(slug)
                        if prev is None:
                            derived[slug] = sha
                            notes.append(
                                f"[check_s3] DERIVED frozen served-map sha256 "
                                f"for {slug} = {sha} (from {name}; source "
                                f"{src or FROZEN_SOURCE.get(slug)}). NOT YET "
                                f"PINNED anywhere -- recorded in the verdict "
                                f"JSON so an audit can pin it. Pass "
                                f"--qwen3-8b-frozen-sha to gate against a "
                                f"pinned value instead.")
                        elif prev != sha:
                            cbad(f"served-map sha256 {sha[:16]}... disagrees "
                                 f"with {prev[:16]}... derived from another "
                                 f"{slug} artifact in this invocation -- two "
                                 f"different frozen priors would contaminate "
                                 f"every distance drawn across the grid")
                    elif sha != want:
                        cbad(f"served-map sha256 {sha[:16]}... != canonical "
                             f"{want[:16]}... for {slug}")
                    # THE VECTOR IS THE INVARIANT, NOT THE SOURCE TAG. A
                    # frozen K=D=0 model never sees the population, so the
                    # same served map comes out of any environment's frozen
                    # cell -- the archived frz_* artifacts were in fact
                    # extracted from pofdfam_qwen7b_k0_ea1_w0p5_l0p2_es0p05_s0
                    # and carry the canonical qmech sha byte-for-byte. So a
                    # source-tag difference under a MATCHING sha changes no
                    # conclusion and is reported, not failed; a sha mismatch
                    # is the hard failure and is handled just above.
                    if src and FROZEN_SOURCE.get(slug) and \
                            src != FROZEN_SOURCE[slug]:
                        lvl = ("PROVENANCE" if sha == want
                               else "PROVENANCE (sha ALSO differs)")
                        notes.append(
                            f"[check_s3] {lvl} frozen {name}: extracted from "
                            f"{src!r}, not the nominated canonical source "
                            f"{FROZEN_SOURCE[slug]!r}"
                            + (" -- served map is byte-identical to the "
                               "canonical vector, so the distances are "
                               "unaffected." if sha == want else "."))
            else:
                # perfect prediction serves x, so the served map is NOT
                # constant and must not be hashed as a prior
                sha = None
        inn = a.get("innate")
        if torch.is_tensor(inn) and _sha_t(inn) != CANONICAL_INNATE_SHA:
            cbad("innate sha256 differs from the canonical agent set")
        seen.append({"file": name, "path": str(p), "kind": kind,
                     "model": slug, "k": k_e, "w_plat": w_e,
                     "sha256": sha, "population_update": marker,
                     "ai_gate_mode": fcfg.get("ai_gate_mode"),
                     "peer_gate_mode": fcfg.get("peer_gate_mode"),
                     "rounds": int(pred.shape[0])
                     if torch.is_tensor(pred) else -1})

    # the frz_* filename does not encode the model: two checkpoints at the
    # same (k, W) collide. Section 3 needs 2 x 3 of them, so say it loudly.
    by_name = {}
    for s in seen:
        if s["kind"] == "frz":
            by_name.setdefault(s["file"], set()).add(s["model"])
    for fn, models in by_name.items():
        if len(models) > 1:
            out.append(f"[check_s3] FAIL frozen {fn}: the same filename was "
                       f"supplied for checkpoints {sorted(models)}. frz_* "
                       f"names encode k/W/gates/seed/rounds but NOT the "
                       f"model, and this wave needs 2 checkpoints x 3 "
                       f"environments. Separate them (per-model subdirs) "
                       f"before anything is analyzed.")
            ok = False
    return ok, seen, derived


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(
        description="Section 3 retention wave gate (CPU only)")
    ap.add_argument("runs", nargs="*",
                    help="run dirs; if omitted, --roots are scanned")
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS,
                    help="run roots to scan (both are needed: the archived "
                         "QWU reuse cells live under notes/pofd/cluster/)")
    ap.add_argument("--reuse-manifest",
                    default="notes/pofd/section3/reuse_manifest.json")
    ap.add_argument("--frozen", nargs="*", default=[],
                    help="frozen-propagation artifacts to gate (frz_*.pt); "
                         "Section 3 needs 6 = 2 checkpoints x 3 environments")
    ap.add_argument("--perfect", nargs="*", default=[],
                    help="perfect-prediction artifacts to gate (pp_*.pt); "
                         "Section 3 needs 3, one per environment")
    ap.add_argument("--qwen3-8b-frozen-sha", dest="qwen3_sha", default=None,
                    help="canonical Qwen3-8B frozen served-map sha256")
    ap.add_argument("--qwen7b-frozen-sha", dest="qwen7b_sha", default=None,
                    help="override the pinned Qwen2.5-7B frozen sha256")
    ap.add_argument("--smoke", action="store_true",
                    help=f"gate a {SMOKE_ROUNDS}-round {SMOKE_PREFIX}* cell "
                         f"instead of the wave (relaxes the HORIZON only)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="do not hard-fail on absent cells (still reported "
                         "loudly, and the verdict line says PARTIAL)")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the machine-readable verdict here")
    args = ap.parse_args()

    out, notes = [], []
    roots = [Path(r) for r in args.roots]
    live_roots = [r for r in roots if r.is_dir()]
    run_dirs = list(args.runs)
    mf_by_tag, mf_by_slot, mf_ok = load_manifest(args.reuse_manifest, out)
    if not run_dirs:
        if not live_roots:
            print(f"[check_s3] usage error: none of {[str(r) for r in roots]} "
                  f"is a directory and no run dirs were given",
                  file=sys.stderr)
            return 2
        if len(live_roots) < len(roots):
            notes.append(f"[check_s3] note: scanned only "
                         f"{[str(r) for r in live_roots]}; missing "
                         f"{[str(r) for r in roots if r not in live_roots]}")
        pre = SMOKE_PREFIX if args.smoke else PROD_PREFIX
        for root in live_roots:
            run_dirs += [str(p) for p in sorted(root.iterdir())
                         if p.is_dir() and p.name.startswith(pre)]
        # Archived reuse cells never match the prefix, so a prefix scan
        # alone would report their four slots as ABSENT. Pull them in by
        # the manifest's own run_dir / run_tag, resolved against BOTH
        # roots (the QWU cells sit under notes/pofd/cluster/).
        for cell in list(mf_by_slot.values()):
            if not _manifest_says_reuse(cell):
                continue
            d = _mf_get(cell, _MF_DIR)
            t = _mf_get(cell, _MF_TAG)
            cands = []
            if d is not ABSENT and d:
                p = Path(str(d))
                cands.append(p if p.is_absolute() else Path(REPO) / p)
            if t is not ABSENT and t:
                cands += [root / str(t) for root in live_roots]
            hit = next((str(c) for c in cands if (c / "config.json").exists()),
                       None)
            if hit is None and cands:
                out.append(
                    f"[check_s3] FAIL manifest: reuse cell "
                    f"{t if t is not ABSENT else d!r} is not resolvable under "
                    f"{[str(r) for r in live_roots]} -- a manifest that names "
                    f"a run nobody can open is not attestation")
                mf_ok = False
            elif hit is not None:
                run_dirs.append(hit)
        run_dirs = sorted(set(run_dirs))
    if not run_dirs:
        print(f"[check_s3] usage error: no run dirs found under "
              f"{[str(r) for r in live_roots]}. Nothing to gate is NOT a "
              f"pass.", file=sys.stderr)
        return 2

    allok = mf_ok
    recs = [check_one(rd, args.smoke, mf_by_tag, out, notes)
            for rd in run_dirs]
    allok &= all(r["ok"] for r in recs)

    # ---------------- cross-cell identity --------------------------------
    good = [r for r in recs if r["ok"]]
    def _uniq(field):
        return {r["report"].get(field) for r in good
                if r["report"].get(field) is not None}

    for field, why in (
            ("innate_sha256", "a different agent set or agent ORDER"),
            ("profiles_sha256", "different agent profiles")):
        vals = _uniq(field)
        if len(vals) > 1:
            out.append(f"[check_s3] FAIL wave: {len(vals)} distinct {field} "
                       f"across the grid -- {why}. Every cell must sit on the "
                       f"same world.")
            for r in good:
                out.append(f"           {r['tag']}: "
                           f"{str(r['report'].get(field))[:16]}...")
            allok = False

    keys = {}
    for r in good:
        k = r.get("peer_key")
        if k is not None:
            keys.setdefault(k, []).append(r["tag"])
    if len(keys) > 1:
        out.append(
            f"[check_s3] FAIL wave: {len(keys)} distinct peer-stream "
            f"fingerprints (per-round peer_pairs/accepted). With gamma = 0 "
            f"and an all_open peer gate those counters cannot depend on "
            f"opinions, so a difference means a different graph, agent order, "
            f"seed or sweep count.")
        for k, tags in keys.items():
            out.append(f"           group of {len(tags)}: {tags[0]} ...")
        allok = False

    gpus = {r["report"].get("gpu_name") for r in good
            if r["report"].get("gpu_name")}
    if len(gpus) > 1:
        out.append(f"[check_s3] FAIL wave: mixed GPU SKUs {sorted(gpus)} -- "
                   f"greedy decoding is bit-reproducible only within one "
                   f"architecture")
        allok = False
    tvs = {r["report"].get("transformers_version") for r in good
           if not r["reuse"] and r["report"].get("transformers_version")}
    if len(tvs) > 1:
        out.append(f"[check_s3] FAIL wave: NEW cells span "
                   f"{len(tvs)} transformers versions {sorted(tvs)} -- "
                   f"generation is not comparable across them")
        allok = False
    reused_tvs = {r["report"].get("transformers_version") for r in good
                  if r["reuse"] and r["report"].get("transformers_version")}
    if reused_tvs - tvs:
        notes.append(f"[check_s3] note: reused cells carry transformers "
                     f"version(s) {sorted(reused_tvs - tvs)}, the new cells "
                     f"{sorted(tvs)}")
    for field in CONSISTENCY_FIELDS:
        vals = {json.dumps(r["cfg"].get(field), default=str) for r in good
                if "cfg" in r}
        if len(vals) > 1 and field != "population_update":
            notes.append(f"[check_s3] note: {field} is not identical across "
                         f"the grid: {sorted(vals)}")

    # ---------------- grid completeness -----------------------------------
    slots = conceptual_grid()
    by_slot, dupes = {}, []
    for r in recs:
        if r["slot"] is None:
            continue
        if r["slot"] in by_slot:
            dupes.append((r["slot"], by_slot[r["slot"]]["tag"], r["tag"]))
        by_slot[r["slot"]] = r
    for slot, a, b in dupes:
        out.append(f"[check_s3] FAIL wave: two run dirs claim the same "
                   f"conceptual cell {slot}: {a} and {b}")
        allok = False
    missing = [s for s in slots if s not in by_slot]
    present = len(slots) - len(missing)

    # ---------------- CPU endpoints (frozen + perfect prediction) ----------
    fz_seen, derived = [], {}
    endpoints = list(args.frozen) + list(args.perfect)
    if endpoints:
        overrides = {}
        if args.qwen3_sha:
            overrides["qwen3_8b"] = args.qwen3_sha.strip().lower()
        if args.qwen7b_sha:
            overrides["qwen7b"] = args.qwen7b_sha.strip().lower()
        fok, fz_seen, derived = check_frozen(endpoints, overrides, out, notes)
        allok &= fok
    else:
        notes.append("[check_s3] note: no --frozen / --perfect artifact "
                     "given, so the frozen reference hash was NOT checked. "
                     "The lambda ladder is measured against that vector; "
                     "gate it before any distance is quoted.")

    # ------------------------------------------------------------- print
    for line in out:
        print(line)
    for line in notes:
        print(line)

    print("\n" + "=" * 78)
    print("PER-CELL REPORT -- served-map degeneracy is REPORTED, NEVER FAILED.")
    print("A collapsed or coarse served map is a legitimate outcome of this")
    print("experiment; the gate only certifies that parsing and training")
    print("provenance are valid. (check_kl_direction.py hard-fails the two")
    print("flag columns below; Section 3 deliberately does not.) Late window")
    print(f"= post-peer rounds {LATE_LO}-{LATE_HI}.")
    print("  flags: A = served map constant across agents in EVERY round")
    print("         R = served vector bit-identical in EVERY round")
    print("=" * 78)
    hdr = (f"{'cell':<46} {'ok':>4} {'fl':>3} {'ndist':>6} {'mode':>6} "
           f"{'top3':>6} {'effmod':>7} {'predSD':>7} {'popMean':>8} "
           f"{'popSD':>7}")
    print(hdr)
    print("-" * len(hdr))
    for slot in slots:
        r = by_slot.get(slot)
        name = f"{slot[0]}/{ENV_LABEL.get((slot[2], slot[3]), '?')}" \
               f"(b{_num(slot[2])},k{_num(slot[3])})/{slot[1]}"
        if r is None:
            print(f"{name:<46} {'---':>4}   ABSENT")
            continue
        lt = r["report"].get("late") or {}
        flags = ("A" if r["report"].get(
                     "pred_constant_across_agents_every_round") else ".") + \
                ("R" if r["report"].get(
                     "pred_bit_identical_across_rounds") else ".")
        print(f"{name:<46} {'ok' if r['ok'] else 'FAIL':>4} {flags:>3} "
              f"{lt.get('n_distinct', float('nan')):>6.1f} "
              f"{lt.get('mode_share', float('nan')):>6.3f} "
              f"{lt.get('top3_share', float('nan')):>6.3f} "
              f"{lt.get('eff_modes', float('nan')):>7.2f} "
              f"{lt.get('sd', float('nan')):>7.4f} "
              f"{r['report'].get('pop_final_mean', float('nan')):>8.4f} "
              f"{r['report'].get('pop_final_sd', float('nan')):>7.4f}")

    devs = [(r["tag"], dv) for r in recs for dv in r["deviations"]]
    if devs:
        print("\n" + "=" * 78)
        print("PROVENANCE DEVIATIONS (manifest-attested reuse only)")
        print("=" * 78)
        for tag, dv in devs:
            print(f"  {tag}: {dv}")

    if fz_seen:
        print("\nCPU ENDPOINTS (frozen propagation + perfect prediction)")
        for s in fz_seen:
            sha = s["sha256"]
            print(f"  {s['file']:<46} {s['kind']:<4} model={s['model']} "
                  f"k={s['k']} W={s['w_plat']} rounds={s['rounds']} "
                  f"marker={s['population_update']} "
                  f"sha={sha[:16] + '...' if sha else '-'}")

    print("\n" + "=" * 78)
    if missing:
        print(f"GRID COMPLETENESS: {present} of {len(slots)} conceptual cells "
              f"present -- {len(missing)} ABSENT")
        print("A silently short grid must not look like a complete result.")
        for s in missing:
            print(f"  ABSENT  {s[0]:<9} {ENV_LABEL.get((s[2], s[3]), '?'):<5} "
                  f"beta={s[2]:<4g} k={s[3]:<4g} arm={s[1]:<10} "
                  f"expected tag {slot_tag(s[0], s[1], s[2], s[3])}")
        if not args.allow_partial:
            allok = False
            print("[check_s3] the absent cells above are a HARD FAILURE. Pass "
                  "--allow-partial to gate a deliberately partial grid.")
    else:
        print(f"GRID COMPLETENESS: all {len(slots)} conceptual cells present")
    print("=" * 78)

    verdict = {
        "wave": "section3", "smoke": args.smoke,
        "n_runs": len(recs), "n_cells_present": present,
        "n_cells_total": len(slots),
        "missing": [{"model": s[0], "arm": s[1], "beta": s[2], "k": s[3],
                     "expected_tag": slot_tag(*s)} for s in missing],
        "pass": bool(allok),
        "partial": bool(missing),
        "cells": [{"tag": r["tag"], "run_dir": r["run_dir"], "ok": r["ok"],
                   "slot": r["slot"], "reuse": r["reuse"],
                   "deviations": r["deviations"], "report": r["report"]}
                  for r in recs],
        "cpu_endpoints": fz_seen,
        # frozen served-map sha256 values DERIVED at runtime because no
        # canonical constant exists for that checkpoint yet. Recorded so
        # an audit can pin them; never hard-coded in this file.
        "derived_frozen_sha256": derived,
        "expected_marker_table": EXPECTED_MARKER,
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(verdict, indent=2))
        print(f"[check_s3] verdict -> {args.json_out}")

    if allok and missing:
        print(f"[check_s3] PASS (PARTIAL GRID: {present}/{len(slots)} cells) "
              f"-- direction and lambda match every tag, the reference is the "
              f"raw base, serving is complete, zero parse failures, the anchor "
              f"gradient is live on every regularized arm, and all cells share "
              f"one world. DO NOT present this as the complete Section 3 grid.")
        return 0
    if allok:
        print(f"[check_s3] PASS -- {len(recs)} run(s), {present}/{len(slots)} "
              f"conceptual cells: direction and lambda match every tag, the "
              f"reference is the raw base, serving is complete, zero parse "
              f"failures, the anchor gradient is live on every regularized "
              f"arm, and all cells share one world.")
        return 0
    print(f"[check_s3] FAILED -- see above ({len(recs)} run(s) inspected)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
