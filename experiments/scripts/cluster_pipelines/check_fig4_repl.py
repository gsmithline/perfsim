#!/usr/bin/env python3
"""GATE for the FIGURE-4 REPLICATION AND CONVERGENCE wave
(pofdf4r_, key fig4_family_prior_repl30, 18 GPU jobs).

RUN THIS BEFORE ANYTHING IS ANALYZED, PLOTTED OR QUOTED.

CPU only, and safe on a shared login node: OMP/MKL thread counts are
pinned to 1 BEFORE torch is imported and torch is pinned again after,
so this never becomes a multithreaded job. At most one run's tensors
are held at a time.

====================================================================
WHAT THE WAVE CLAIMS
====================================================================
Figure 4 of the current draft
(paper/bialign_neurips27/neurips_2026_sft_icl_rework.tex:407,
figures/sft_family_prior_one_row.pdf, rendered by
experiments/llm/plot_sft_family_prior_one_row.py) shows ONE seed-0,
30-round cell per checkpoint:

    pofdfam_{slug}_b1_ea1_w0p5_l0p2_es0p05_s0     (the displayed arm)
    pofdfam_{slug}_k0_ea1_w0p5_l0p2_es0p05_s0     (the frozen control)

This wave re-runs EXACTLY that displayed condition at three seeds, at
the SAME 30-round horizon: 6 checkpoints x 3 seeds = 18 cells. So the
claim is NOT "a new experiment" and not even "a longer one"; it is "the
SAME experiment, three times". The horizon was 100 in an earlier draft
of this wave and the user cut it to 30 ("100 is too much"), which
TIGHTENS what this gate can demand: at a matched horizon the new cells
are configuration-identical to the displayed Figure-4 cells apart from
the seed and the operator provenance marker, so n_rounds is now checked
for EQUALITY in --against-fig4 rather than exempted. Every failure mode
that would fake the claim is gated here:

  GRAMMAR/COVERAGE  the parsed (slug, seed) set must be the FULL 18-cell
                    product. A short grid is reported cell by cell and
                    is a hard failure unless --allow-partial is passed.
  OPERATOR          population_update must be
                    "nested_ai_anchored_then_social_v2" AND
                    ai_gate_reference must be "anchor". The archived
                    pofdfam_ cells carry the OLD
                    "nested_ai_then_social_v1" (see _POP_UPDATE_MARKER,
                    run_pokec_gated_lm.py:213-227); a v1 run here means
                    the job did not come from the current tree.
                    PRECISELY WHAT THE DIFFERENCE IS: v1 tests
                    |m - x0| < eps_AI, v2 tests |m - h| < eps_AI with
                    h = k innate + (1-k) x0. That is different
                    arithmetic, but at THIS wave's eps_AI = 1 the gate
                    is wide enough that the two agree on essentially
                    every pair, so a v1 cell would not necessarily
                    trace a different population -- it would be the
                    wrong PROVENANCE, and the wave declares v2 in its
                    tag. Hard-failed on that ground, not on a claim
                    that the numbers must differ.
  MODEL             config base_model must be the slug's exact HF id.
  SEED              config seed must equal the tag's seed.
  SHARED WORLD      at a given seed the innate vector must be
                    BIT-IDENTICAL across all six checkpoints (see the
                    honest note below), and the seeds must actually have
                    done something: two cells of the same checkpoint at
                    different seeds may not carry bit-identical
                    trajectories.
  HORIZON           n_rounds == 30 and len(trajectory) >= 30, with
                    op_raw / pred_raw / twin_raw all [30, 723].
  PARSE             parse_fail_frac == 0 in EVERY round, read from
                    raw_gen_log.json.gz (a GZIPPED JSONL file). It is
                    NOT in trajectory.pt; a checker that looked there
                    would read None and pass vacuously.
  GRID              every dial of the displayed condition, pinned.

--------------------------------------------------------------------
AN HONEST CORRECTION TO THE "SHARED POPULATION/GRAPH" SPEC
--------------------------------------------------------------------
The wave spec asked for two things here: (a) the innate vector is
bit-identical across the six checkpoints at a given seed, and (b)
DIFFERENT SEEDS GIVE DIFFERENT INNATE VECTORS.

(a) is right and is gated as a hard failure.

(b) IS FALSE ON THIS SURFACE, and gating it would fail all 18
legitimate cells. On movielens the population and the graph are pure
functions of (dataset, target, knn): load_movielens_setup
(run_pokec_gated_lm.py:423-457) takes no seed and derives
innate = (Pl[target] - 1) / 4 plus a deterministic 10-NN graph over the
largest connected component. No RNG is consulted. Verified on disk:
the six archived pofdfam_ b1 cells AND an archived seed-42 cell all
carry innate sha256
be34f284f929e2198996a37b080c03eef5750e1917d90269cd3fde81a7b31b19.

So the seed here moves the PEER/TRAINING RNG, not the world. This gate
therefore replaces (b) with two checks that are true and stronger:

  * every cell's innate vector must equal the CANONICAL movielens-Action
    723-agent vector (CANONICAL_INNATE_SHA) -- one pin covering all 18
    cells, so a different agent set or a different agent ORDER cannot
    enter the grid at any seed;
  * for each checkpoint, no two seeds may carry a BIT-IDENTICAL op_raw.
    That is the check that actually establishes "the seed did something",
    and it is a hard failure.

The cross-seed innate identity is REPORTED as a fact of the loader,
never failed. If you want the pin relaxed (a different dataset, a
different target), pass --innate-sha to name the expected vector or
--no-innate-pin to fall back to consistency-only.

--------------------------------------------------------------------
--against-fig4: FIELD-BY-FIELD AGAINST THE PUBLISHED CELL
--------------------------------------------------------------------
Per checkpoint, the archived Figure-4 seed-0 cell
pofdfam_{slug}_b1_ea1_w0p5_l0p2_es0p05_s0 is loaded from the run root
and its config is compared FIELD BY FIELD with the new seed-0 cell.
Four buckets, all printed:

  EXPECTED      seed, population_update, ai_gate_reference, run_tag.
                Exactly the differences the wave is FOR, plus run_tag,
                which is a pure function of them. n_rounds is NOT on
                this list any more: the horizons now match, so 30 == 30
                is ASSERTED and a moving horizon is a hard failure with
                its own message.
  ENVIRONMENT   host, hardware, git_sha. Where and by which tree the
                job computed. Reported, never failed -- these are not
                the experiment.
  CODE DELTA    a field present in exactly ONE of the two configs
                because the runner GAINED it after the archived run
                (2026-08-17). Each one is registered in
                ONE_SIDED_REGISTRY with the value that must be found
                and a note on whether it is behaviour-neutral. An
                UNREGISTERED one-sided field is a HARD FAILURE: the
                point is that nothing is waived without being named.
                NOTE THE ONE THAT IS NOT NEUTRAL: serve_eval_mode.
                The archived cell predates the 2026-08-21 fix that
                forces eval() for the duration of generation, so it
                served with LoRA dropout live. The replication does
                not. That is a real difference in the serving path and
                it is printed as such, not buried.
  HARD FAIL     any other difference, listed as field / old / new.

An absent archived cell marks that checkpoint SKIPPED and is counted in
the verdict line. It never silently passes.

--------------------------------------------------------------------
ARTIFACT FACTS THIS FILE RELIES ON
--------------------------------------------------------------------
  runs live at <run_root>/<tag>/; BOTH roots are accepted the same way
  (runs/pokec_gated_lm and notes/pofd/cluster), exactly as
  plot_sft_family_prior_one_row.py resolves them.
  trajectory.pt : torch.load(..., map_location="cpu",
                  weights_only=False) -> config, op_raw [T,723],
                  twin_raw [T,723], pred_raw [T,723], innate [723],
                  trajectory (list of per-round rows).
  op_raw[t]     : the END-OF-ROUND POST-PEER state (peers run last).
  parse_fail_frac -> raw_gen_log.json.gz (gzipped JSONL).
  l_init / grad_* -> telemetry.json (JSONL).
  Neither is in trajectory.pt. Both trajectory.pt and
  raw_gen_log.json.gz are written ONLY at run completion, so their
  absence means "did not finish", never "not saved".
  WITH_TWIN is NOT a config field -- the runner records it only by
  writing a non-empty twin_raw, so that is what is checked.

Usage
  OMP_NUM_THREADS=1 python check_fig4_repl.py --run-root runs/pokec_gated_lm
  OMP_NUM_THREADS=1 python check_fig4_repl.py --run-root runs/pokec_gated_lm \\
      --run-root notes/pofd/cluster --against-fig4 \\
      --json notes/pofd/fig4_repl/check_verdict.json

Exit codes: 0 = pass, 1 = hard failure, 2 = usage/input error.
"""
from __future__ import annotations

# Thread pinning FIRST: both variables must be set before torch (and the
# BLAS it links) is imported, or the pin is a no-op and this becomes a
# multithreaded job on a shared login node.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# transformers' TF probe deadlocks on some machines and nothing here
# needs transformers at all.
os.environ.setdefault("USE_TF", "0")

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

HERE = (os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals() else os.getcwd())
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

LOG = "[check_f4r]"

# ---------------------------------------------------------------- pins
N_AGENTS = 723
PROD_PREFIX = "pofdf4r_"
ROUNDS = 30
SEEDS = (0, 42, 43)
# tail window, 1-based inclusive: the last 10 rounds of the horizon, the
# direct analogue of the 81-100 window an earlier 100-round draft used.
LATE_LO, LATE_HI = 21, 30

# slug -> the EXACT HuggingFace id. Restated here rather than imported so
# a test of this gate is not testing the thing it imports its
# expectations from.
MODELS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "ministral8b": "mistralai/Ministral-8B-Instruct-2410",
}
MODEL_ORDER = ("qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b", "mistral7b",
               "ministral8b")

# The two round-operator markers the runner can emit
# (run_pokec_gated_lm.py:213-227, keyed on AI_GATE_REFERENCE).
POP_UPDATE_V1 = "nested_ai_then_social_v1"              # gate on x0
POP_UPDATE_V2 = "nested_ai_anchored_then_social_v2"     # gate on x'
WANT_POP_UPDATE = POP_UPDATE_V2
WANT_GATE_REF = "anchor"

# The tag's operator-provenance token, and what it must mean.
OPTOK = "anch2"
OPTOK_POP_UPDATE = {OPTOK: POP_UPDATE_V2}

# movielens Action, 723 agents: sha256 over the float32 innate vector.
# Same constant as check_section3.CANONICAL_INNATE_SHA; re-derived here
# from the six archived pofdfam_ b1 cells on disk.
CANONICAL_INNATE_SHA = (
    "be34f284f929e2198996a37b080c03eef5750e1917d90269cd3fde81a7b31b19")

# BOTH roots, resolved the same way plot_sft_family_prior_one_row.py
# resolves them (notes/pofd/cluster first, then runs/pokec_gated_lm).
DEFAULT_ROOTS = (os.path.join(REPO, "notes", "pofd", "cluster"),
                 os.path.join(REPO, "runs", "pokec_gated_lm"))

# The archived Figure-4 cells, per checkpoint. The b1 cell is the
# displayed forward-KL lambda=1 arm; the k0 cell is the frozen control
# the figure draws as "entering model".
FIG4_B1 = "pofdfam_{slug}_b1_ea1_w0p5_l0p2_es0p05_s0"
FIG4_K0 = "pofdfam_{slug}_k0_ea1_w0p5_l0p2_es0p05_s0"

# ---- the displayed condition, as config fields ----------------------
# Verified field-by-field against the archived cell
# notes/pofd/cluster/pofdfam_qwen7b_b1_ea1_w0p5_l0p2_es0p05_s0/config.json
# so a pin invented here cannot reject a legitimate replication.
# `eps` is eps_social. `innate_lambda` is the paper's gamma / innate
# anchor k. `w_plat` is the paper's beta.
GRID_PINS = {
    "dataset": "movielens",
    "ml_target": "Action",
    "n_labeled": N_AGENTS,
    "train_cap": N_AGENTS,
    "training_style": "sft_kl",
    "kl_direction": "forward",
    "kl_ref_adapter": "",
    "anchor_mode": "fixed",
    "ai_gate_mode": "threshold",
    "pop_model": "ab",
    "ab_sweeps": 1,
    "use_lora": True,
    "fresh_each_round": True,
    "icl_k": 0,
    "icl_days": 0,
    "lora_r": 512,
    "sft_epochs": 1,
    "sft_batch_size": 4,
    "epoch_size": 100,
    "seed_base_data": True,
    "save_raw_gen": True,
    "n_rounds": ROUNDS,
    "deploy_every": 1,
    "data_regime": "replace",
    "feedback_mode": "none",
    "icrh": False,
    "do_sample": False,
    "pristine_frac": 0.0,
    "replay_frac": 0.0,
    "teacher_label_delta": 0.0,
}
GRID_PINS_FLOAT = {
    "kl_beta": 1.0,
    "eps_ai": 1.0,
    "eps": 0.05,            # eps_social
    "w_plat": 0.5,          # the paper's beta
    "innate_lambda": 0.2,   # the paper's gamma / innate anchor k
    "gamma_bias": 0.0,      # homophily gamma column: pinned 0 everywhere
    "sft_lr": 5e-5,
}
# recorded from 2026-08-20 only; the archived cells ran the same numeric
# peer gate, which was then implicit. Absent == "threshold".
PEER_GATE_MODE_DEFAULT = "threshold"

# ---- --against-fig4 buckets -----------------------------------------
# The differences the wave EXISTS to create, plus run_tag, which is a
# pure function of seed / horizon / the operator token.
# n_rounds is deliberately ABSENT: the replication runs the archived
# horizon, so equality is asserted by _against_n_rounds() below and any
# difference falls through to the hard-fail bucket.
AGAINST_EXPECTED_DIFF = ("seed", "population_update",
                         "ai_gate_reference", "run_tag")
# Where and by which tree the job computed. Not the experiment.
AGAINST_ENV_PROVENANCE = ("host", "hardware", "git_sha", "hostname")

ANY = object()

# Fields present in exactly ONE of the two configs because the runner
# gained (or dropped) them after the archived Figure-4 cells were
# written on 2026-08-17. Every entry names the value that must be found
# and says whether the field is behaviour-neutral. An UNREGISTERED
# one-sided field is a hard failure -- see the docstring.
ONE_SIDED_REGISTRY = {
    "peer_gate_mode": (
        PEER_GATE_MODE_DEFAULT,
        "recorded from 2026-08-20. BEHAVIOUR-NEUTRAL: the archived cell "
        "ran the same numeric peer gate at eps_social=0.05; the field "
        "only makes it legible."),
    "serve_eval_mode": (
        True,
        "recorded from 2026-08-21. *** NOT BEHAVIOUR-NEUTRAL ***: it "
        "records the fix that forces eval() for the duration of "
        "generation, so the replication serves with LoRA dropout OFF "
        "while the archived Figure-4 cell served with it live. This is "
        "the one real difference in the serving path between the "
        "published figure and this wave, and it is reported, not "
        "waived."),
    "git_sha": (
        ANY,
        "provenance: the tree that produced the run. Different by "
        "construction."),
    "fj_update_version": (
        "legacy",
        "FJ robustness record, 2026-08-21. BEHAVIOUR-NEUTRAL here: "
        "'legacy' IS the operator the archived cell ran, and every fj_* "
        "field is inert at pop_model='ab'."),
    "fj_peer_alpha": (ANY, "inert at pop_model='ab' (FJ record, "
                           "2026-08-21)"),
    "fj_internal_anchor_coef": (ANY, "inert at pop_model='ab'"),
    "fj_peer_sus_convention": (ANY, "descriptive string, inert at "
                                    "pop_model='ab'"),
    "fj_inner_steps": (ANY, "inert at pop_model='ab'"),
    "fj_human_component": (ANY, "descriptive string, inert at "
                                "pop_model='ab'"),
    "fj_model": (ANY, "descriptive string, inert at pop_model='ab'"),
    "fj_recurrence": (ANY, "descriptive string, inert at "
                           "pop_model='ab'"),
    "ai_gate_reference": (
        WANT_GATE_REF,
        "recorded from 2026-08-22; the operator field the wave is FOR. "
        "Also listed in AGAINST_EXPECTED_DIFF for cells where the "
        "archived config happens to carry it."),
}


# ------------------------------------------------------------- helpers
def _sha_t(t) -> str:
    """sha256 over a tensor's float32 bytes (check_pofd_sanity._sha_t)."""
    a = torch.as_tensor(t).detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _feq(a, b, tol=1e-9):
    a, b = _as_float(a), _as_float(b)
    return a is not None and b is not None and abs(a - b) <= tol


def _read_gz_jsonl(path):
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


# --------------------------------------------------------- tag grammar
# pofdf4r_{slug}_b1_anch2_ea1_w0p5_l0p2_es0p05_r30_s{SEED}
#
# The slug is captured loosely and validated against MODELS so an
# unknown checkpoint reads as "unknown checkpoint", not as "malformed
# tag" -- two different operator errors that need two different
# messages. Every other token is FIXED: this wave re-runs exactly one
# condition, so any variation in it is a different experiment wearing
# the wave's prefix.
TAG_RE = re.compile(
    r"^" + PROD_PREFIX + r"(?P<slug>.+?)"
    r"_b1_(?P<optok>[a-z0-9]+)_ea1_w0p5_l0p2_es0p05_r(?P<rounds>\d+)"
    r"_s(?P<seed>\d+)$")

TAG_TEMPLATE = (PROD_PREFIX + "{slug}_b1_" + OPTOK +
                "_ea1_w0p5_l0p2_es0p05_r{rounds}_s{seed}")


def expected_tag(slug, seed, rounds=ROUNDS):
    """The one legal tag for a (checkpoint, seed) cell."""
    return TAG_TEMPLATE.format(slug=slug, rounds=rounds, seed=seed)


def parse_tag(tag):
    """(slot | None, [errors]).

    slot = {"slug", "seed", "rounds", "optok", "base_model"}.
    """
    m = TAG_RE.match(tag)
    if m is None:
        return None, [
            f"tag is not in the fig4-replication grammar "
            f"{TAG_TEMPLATE.format(slug='{slug}', rounds=ROUNDS, seed='{seed}')}"
            f" (slug in {sorted(MODELS)}, seed in {list(SEEDS)})"]
    errs = []
    slug = m.group("slug")
    if slug not in MODELS:
        errs.append(f"unknown checkpoint slug {slug!r}; the wave runs "
                    f"{sorted(MODELS)}")
    optok = m.group("optok")
    if optok != OPTOK:
        errs.append(f"operator token {optok!r} in the tag; this wave is "
                    f"pinned to {OPTOK!r} <-> population_update "
                    f"{POP_UPDATE_V2!r}")
    rounds = int(m.group("rounds"))
    if rounds != ROUNDS:
        errs.append(f"tag horizon r{rounds}; the wave is a {ROUNDS}-round "
                    f"replication and a shorter run is a different object")
    seed = int(m.group("seed"))
    if seed not in SEEDS:
        errs.append(f"seed {seed} in the tag; the wave runs seeds "
                    f"{list(SEEDS)}")
    if errs:
        return None, errs
    return ({"slug": slug, "seed": seed, "rounds": rounds, "optok": optok,
             "base_model": MODELS[slug]}, [])


def full_grid():
    """The 18-cell product, in report order."""
    return [(slug, seed) for slug in MODEL_ORDER for seed in SEEDS]


# ------------------------------------------------------------ one cell
def check_one(run_dir, out, notes, innate_sha_pin=CANONICAL_INNATE_SHA):
    """Gate ONE run dir. Returns a per-cell record (never None).

    Only one run's tensors are alive inside this call; everything the
    caller needs afterwards is a scalar or a sha256.
    """
    run_dir = str(run_dir).rstrip("/")
    tag = os.path.basename(run_dir)
    rec = {"tag": tag, "run_dir": run_dir, "ok": False, "slot": None,
           "report": {}, "cfg": None}

    cf = os.path.join(run_dir, "config.json")
    tf = os.path.join(run_dir, "trajectory.pt")
    if not os.path.exists(cf):
        out.append(f"{LOG} FAIL {tag}: no config.json")
        return rec
    if not os.path.exists(tf):
        out.append(f"{LOG} FAIL {tag}: no trajectory.pt -- it is written "
                   f"ONLY at run completion, so the run did not finish")
        return rec
    try:
        cfg = json.load(open(cf))
    except Exception as exc:                                # noqa: BLE001
        out.append(f"{LOG} FAIL {tag}: config.json unreadable ({exc})")
        return rec
    rec["cfg"] = cfg

    ok = True

    def bad(msg):
        nonlocal ok
        out.append(f"{LOG} FAIL {tag}: {msg}")
        ok = False

    # -------------------------------------------- 1. grammar
    slot, tag_errs = parse_tag(tag)
    for e in tag_errs:
        bad(e)
    if slot is None:
        return rec
    rec["slot"] = (slot["slug"], slot["seed"])

    # -------------------------------------------- 2. the round OPERATOR
    # Named first because everything downstream is measured on the state
    # this operator produced. v1 tests |m - x0| < eps_AI and v2 tests
    # |m - h| < eps_AI; at eps_AI = 1 they agree on essentially every
    # pair, so this gate is enforcing PROVENANCE -- the tag says anch2
    # and the artifact must say so too -- not claiming the trajectories
    # would differ.
    pu = cfg.get("population_update")
    if pu != WANT_POP_UPDATE:
        if pu == POP_UPDATE_V1:
            bad(f"population_update is the OLD operator {POP_UPDATE_V1!r} "
                f"(AI gate measured against the raw start-of-round x0). "
                f"This wave's tag declares _{OPTOK}_ and requires "
                f"{WANT_POP_UPDATE!r} (gate measured against the anchored "
                f"x' = k innate + (1-k) x). The archived pofdfam_ cells "
                f"are exactly the v1 runs this wave replaces, so a v1 "
                f"artifact under a pofdf4r_ tag is the OLD run wearing "
                f"the NEW name -- rejected on provenance even though at "
                f"eps_ai=1 the two gate references agree on essentially "
                f"every pair.")
        else:
            bad(f"population_update={pu!r}; expected "
                f"{WANT_POP_UPDATE!r}. The runner can emit only "
                f"{POP_UPDATE_V1!r} or {POP_UPDATE_V2!r} "
                f"(_POP_UPDATE_MARKER, run_pokec_gated_lm.py:213-227).")
    gr = cfg.get("ai_gate_reference")
    if gr != WANT_GATE_REF:
        bad(f"ai_gate_reference={gr!r}; the wave pins {WANT_GATE_REF!r} in "
            f"the sub environment. Absent means a config written before "
            f"2026-08-22, i.e. not this tree.")
    rec["report"]["population_update"] = pu
    rec["report"]["ai_gate_reference"] = gr

    # -------------------------------------------- 3. the CHECKPOINT
    got_base = cfg.get("base_model")
    if got_base != slot["base_model"]:
        bad(f"base_model={got_base!r} but the tag says {slot['slug']!r} = "
            f"{slot['base_model']!r}")
    ct = cfg.get("chat_thinking", "<absent>")
    if slot["slug"] == "qwen3_8b":
        # CHAT_THINKING=0 -> apply_chat_template(enable_thinking=False)
        if ct is True or ct == 1:
            bad("chat_thinking is TRUE for qwen3_8b -- Qwen3 hybrid "
                "reasoning must be OFF (CHAT_THINKING=0)")
        elif ct == "<absent>":
            bad("qwen3_8b records no chat_thinking; the run left "
                "CHAT_THINKING unset, so the thinking switch was never "
                "passed to apply_chat_template and 'thinking off' is not "
                "establishable from the artifact")
        elif ct not in (False, 0):
            bad(f"chat_thinking={ct!r} for qwen3_8b; expected False/0")
    elif ct is True or ct == 1:
        bad(f"chat_thinking is TRUE for {slot['slug']!r}; only Qwen3 "
            f"carries a thinking directive and it must be off")
    elif ct != "<absent>":
        # present-and-false on a non-hybrid template renders unchanged,
        # so this is a recording deviation, not a behavioural one.
        notes.append(f"{LOG} note: {tag} records chat_thinking={ct!r} for "
                     f"{slot['slug']!r}, which carries no thinking "
                     f"directive; the template renders unchanged")
    rec["report"]["chat_thinking"] = None if ct == "<absent>" else ct

    # -------------------------------------------- 4. seed
    if cfg.get("seed") != slot["seed"]:
        bad(f"config seed={cfg.get('seed')!r} but the tag says "
            f"s{slot['seed']}")

    # -------------------------------------------- 5. grid fields
    for key, want in GRID_PINS.items():
        got = cfg.get(key, "<absent>")
        if isinstance(want, bool):
            if got == "<absent>" or bool(got) is not want:
                bad(f"{key}={got!r}, expected {want!r}")
        elif got != want:
            bad(f"{key}={got!r}, expected {want!r}")
    for key, want in GRID_PINS_FLOAT.items():
        if not _feq(cfg.get(key), want):
            bad(f"{key}={cfg.get(key)!r}, expected {want!r}")
    pgm = cfg.get("peer_gate_mode", PEER_GATE_MODE_DEFAULT)
    if pgm != PEER_GATE_MODE_DEFAULT:
        bad(f"peer_gate_mode={pgm!r}; eps_social=0.05 is a NUMERIC peer "
            f"gate and the displayed condition ran "
            f"{PEER_GATE_MODE_DEFAULT!r}")

    # -------------------------------------------- 6. tensors
    try:
        d = torch.load(tf, map_location="cpu", weights_only=False)
    except Exception as exc:                                # noqa: BLE001
        out.append(f"{LOG} FAIL {tag}: trajectory.pt unreadable ({exc})")
        return rec

    if int(cfg.get("n_rounds", -1)) != ROUNDS:
        bad(f"n_rounds={cfg.get('n_rounds')!r}, expected {ROUNDS}")
    rows = d.get("trajectory")
    n_rows = len(rows) if isinstance(rows, (list, tuple)) else -1
    if n_rows < ROUNDS:
        bad(f"trajectory has {n_rows} row(s), need >= {ROUNDS}")

    shapes_ok = True
    for name in ("op_raw", "pred_raw", "twin_raw"):
        t = d.get(name)
        if not torch.is_tensor(t) or t.numel() == 0:
            if name == "twin_raw":
                bad("twin_raw is empty -- WITH_TWIN is NOT a config field, "
                    "so a non-empty twin_raw is the ONLY evidence the "
                    "matched no-platform twin was simulated, and the "
                    "analyzer measures W1 to it")
            else:
                bad(f"{name} missing or empty")
            shapes_ok = False
            continue
        if tuple(t.shape) != (ROUNDS, N_AGENTS):
            bad(f"{name} shape {tuple(t.shape)} != {(ROUNDS, N_AGENTS)}")
            shapes_ok = False
            continue
        if not torch.isfinite(t).all():
            bad(f"{name} has non-finite values")

    inn = d.get("innate")
    if not torch.is_tensor(inn) or inn.numel() != N_AGENTS:
        bad(f"innate missing or not {N_AGENTS} agents -- it is t=0 of the "
            f"trajectory and cannot be reconstructed")
    else:
        if not torch.isfinite(inn).all():
            bad("innate has non-finite values")
        isha = _sha_t(inn)
        rec["report"]["innate_sha256"] = isha
        if innate_sha_pin and isha != innate_sha_pin:
            bad(f"innate sha256 {isha[:16]}... != the canonical "
                f"movielens-Action 723-agent vector "
                f"{innate_sha_pin[:16]}... -- a different agent set or a "
                f"different agent ORDER. Pass --innate-sha/--no-innate-pin "
                f"only if the surface deliberately changed.")

    if shapes_ok:
        op = d["op_raw"].float()
        rec["report"]["op_sha256"] = _sha_t(op)
        rec["report"]["pop_final_mean"] = float(op[-1].mean())
        rec["report"]["pop_final_sd"] = float(op[-1].std())
        lo, hi = LATE_LO - 1, LATE_HI
        rec["report"]["late_mean"] = float(op[lo:hi].mean())
        rec["report"]["late_sd"] = float(op[lo:hi].std(dim=1).mean())
        pr = d["pred_raw"].float()
        rec["report"]["served_final_mean"] = float(pr[-1].mean())
        rec["report"]["served_final_sd"] = float(pr[-1].std())

    prof = d.get("profiles")
    if prof is not None:
        rec["report"]["profiles_sha256"] = hashlib.sha256(
            json.dumps(prof, sort_keys=True, default=str).encode()).hexdigest()

    # Any graph fingerprint the artifact happens to carry. pop_model="ab"
    # runs record none (fj_graph_sha256 is written only on the FJ path),
    # so this is "check it IF it is there", exactly as specified -- never
    # "invent one".
    fp = {}
    for key in ("fj_graph_sha256", "graph_sha256", "adj_sha256",
                "innate_sha256"):
        if cfg.get(key):
            fp[f"config.{key}"] = cfg[key]
    for key in ("graph_sha256", "adj_sha256"):
        if isinstance(d.get(key), str):
            fp[f"trajectory.{key}"] = d[key]
    rec["report"]["graph_fingerprints"] = fp

    # release the big tensors before the next cell is opened
    del d

    # -------------------------------------------- 7. zero parse failures
    gz = os.path.join(run_dir, "raw_gen_log.json.gz")
    if not os.path.exists(gz):
        bad("no raw_gen_log.json.gz -- SAVE_RAW_GEN=1 writes it at run "
            "completion and it is the ONLY home of parse_fail_frac "
            "(trajectory.pt does not carry it; a checker that looked "
            "there would read None and pass vacuously)")
    else:
        try:
            grows = _read_gz_jsonl(gz)
        except Exception as exc:                            # noqa: BLE001
            grows = None
            bad(f"raw_gen_log.json.gz unreadable ({exc})")
        if grows is not None:
            if len(grows) < ROUNDS:
                bad(f"raw_gen_log.json.gz has {len(grows)} round(s), need "
                    f"{ROUNDS}")
            missing = [r.get("round") for r in grows
                       if r.get("parse_fail_frac") is None]
            if missing:
                bad(f"{len(missing)} round(s) carry no parse_fail_frac, "
                    f"e.g. round {missing[0]}")
            nz = [(r.get("round"), float(r["parse_fail_frac"]))
                  for r in grows if r.get("parse_fail_frac") is not None
                  and float(r["parse_fail_frac"]) != 0.0]
            if nz:
                bad(f"parse failures in {len(nz)} round(s) -- a parse "
                    f"failure is recorded as a confident constant, so a "
                    f"nonzero rate is never rounding. e.g. round "
                    f"{nz[0][0]}: {nz[0][1]:g}")
            short = [(r.get("round"), len(r.get("parsed") or []))
                     for r in grows if r.get("parsed") is not None
                     and len(r["parsed"]) != N_AGENTS]
            if short:
                bad(f"round {short[0][0]} parsed {short[0][1]} of "
                    f"{N_AGENTS} agents -- serving is incomplete")
            vals = [float(r["parse_fail_frac"]) for r in grows
                    if r.get("parse_fail_frac") is not None]
            rec["report"]["parse_fail_max"] = max(vals) if vals else None

    # -------------------------------------------- 8. training actually ran
    # Not in the wave spec, but a fresh-LoRA forward-KL arm whose
    # optimizer never moved would sail through every check above.
    tp = os.path.join(run_dir, "telemetry.json")
    if not os.path.exists(tp):
        notes.append(f"{LOG} note: {tag} has no telemetry.json, so "
                     f"'the optimizer moved' is not establishable from the "
                     f"artifact (l_init / grad_* live only there)")
    else:
        try:
            tel = _read_jsonl(tp)
        except Exception as exc:                            # noqa: BLE001
            tel = []
            notes.append(f"{LOG} note: {tag} telemetry.json unreadable "
                         f"({exc})")
        gn = [float(r["grad_norm0"]) for r in tel
              if r.get("grad_norm0") is not None]
        if gn and max(gn) == 0.0:
            bad("grad_norm0 is zero in every round -- the learner never "
                "trained")
        kg = [float(r["grad_kl_norm0"]) for r in tel
              if r.get("grad_kl_norm0") is not None]
        # round 0 is EXEMPT: a fresh LoRA at round 0 IS the reference, so
        # the divergence and its gradient are legitimately ~0 there.
        if len(kg) > 1 and max(kg[1:]) <= 0.0:
            bad("the forward-KL anchor gradient is zero in every round "
                "after round 0 -- lambda=1 was recorded but never "
                "applied")
        rec["report"]["n_telemetry_rounds"] = len(tel)

    hw = cfg.get("hardware") or {}
    rec["report"]["gpu_name"] = hw.get("gpu_name")
    rec["report"]["transformers_version"] = hw.get("transformers_version")
    rec["report"]["git_sha"] = cfg.get("git_sha")
    rec["ok"] = ok
    return rec


# ------------------------------------------------- --against-fig4 diff
def compare_configs(old, new, slug):
    """Field-by-field comparison of the archived Figure-4 cell against the
    new seed-0 cell.

    Returns (hard_fails, expected, environment, code_delta) -- four lists
    of (field, old_value, new_value, note). Nothing is dropped: every
    differing field lands in exactly one bucket, and only `hard_fails`
    affects the verdict.

    n_rounds gets its OWN clause rather than riding the generic path.
    The wave used to run a longer horizon, so "n_rounds differs" used to
    be the expected case; now the horizons match and the equality is the
    POINT of the comparison. Asserting it explicitly means the message a
    reader gets says so, instead of a generic "this field changed".
    """
    hard, expected, env, delta = [], [], [], []

    # --- the horizon: EQUALITY, asserted ------------------------------
    o_nr, n_nr = old.get("n_rounds", "<absent>"), new.get("n_rounds",
                                                          "<absent>")
    if not (_feq(o_nr, n_nr) or o_nr == n_nr):
        hard.append((
            "n_rounds", o_nr, n_nr,
            f"HORIZON MISMATCH. This wave replicates the displayed cell at "
            f"its OWN horizon ({ROUNDS} rounds), so the two must be EQUAL "
            f"-- the earlier 100-round draft of this wave is the only "
            f"reason a difference here was ever expected. A replication "
            f"run over a different number of rounds is not the same "
            f"experiment and its equilibrium window is not the same "
            f"window."))
    elif _as_float(n_nr) is not None and int(float(n_nr)) != ROUNDS:
        hard.append((
            "n_rounds", o_nr, n_nr,
            f"both configs agree on {n_nr!r} rounds, but this wave is "
            f"pinned to {ROUNDS}"))

    for field in sorted(set(old) | set(new)):
        if field == "n_rounds":
            continue        # handled above, with its own message
        in_old, in_new = field in old, field in new
        ov = old.get(field, "<absent>")
        nv = new.get(field, "<absent>")
        if in_old and in_new:
            same = (ov == nv)
            if not same:
                fo, fn = _as_float(ov), _as_float(nv)
                if fo is not None and fn is not None:
                    same = abs(fo - fn) <= 1e-12
            if same:
                continue
            if field in AGAINST_EXPECTED_DIFF:
                expected.append((field, ov, nv,
                                 "the difference this wave exists to "
                                 "create (the horizon is NOT one of "
                                 "them any more -- it is asserted "
                                 "equal)"))
            elif field in AGAINST_ENV_PROVENANCE:
                env.append((field, ov, nv,
                            "where/by which tree the job computed; not "
                            "the experiment"))
            else:
                hard.append((field, ov, nv,
                             "UNEXPECTED: this field defines the "
                             "experiment and it changed"))
            continue
        # one-sided: added or dropped since the archived run
        if field in AGAINST_ENV_PROVENANCE:
            env.append((field, ov, nv, "provenance field, one-sided"))
            continue
        if field in ONE_SIDED_REGISTRY:
            want, note = ONE_SIDED_REGISTRY[field]
            present = nv if in_new else ov
            if want is not ANY:
                match = (present == want)
                if not match:
                    fw, fp = _as_float(want), _as_float(present)
                    if fw is not None and fp is not None:
                        match = abs(fw - fp) <= 1e-12
                    elif isinstance(want, bool):
                        match = bool(present) is want
                if not match:
                    hard.append((field, ov, nv,
                                 f"registered one-sided field but the "
                                 f"value is not the registered "
                                 f"{want!r}: {note}"))
                    continue
            delta.append((field, ov, nv, note))
            continue
        hard.append((field, ov, nv,
                     "one-sided and UNREGISTERED: the field exists in "
                     "only one of the two configs and nothing in "
                     "ONE_SIDED_REGISTRY says why. Classify it there "
                     "(with the value it must carry and whether it is "
                     "behaviour-neutral) rather than letting it pass."))
    return hard, expected, env, delta


def resolve_tag(roots, tag):
    """First root under which <root>/<tag>/config.json exists, or None.

    Both roots are accepted the same way plot_sft_family_prior_one_row.py
    accepts them.
    """
    for root in roots:
        p = Path(root) / tag
        if (p / "config.json").exists():
            return str(p)
    return None


def against_fig4(roots, new_by_slot, out, notes):
    """Per checkpoint: compare the archived Figure-4 cell to the new
    seed-0 cell. Returns (ok, records)."""
    ok = True
    recs = []
    for slug in MODEL_ORDER:
        arch_tag = FIG4_B1.format(slug=slug)
        arch_dir = resolve_tag(roots, arch_tag)
        new_rec = new_by_slot.get((slug, 0))
        if new_rec is None or new_rec.get("cfg") is None:
            recs.append({"slug": slug, "status": "SKIPPED",
                         "archived_tag": arch_tag,
                         "reason": f"the new seed-0 cell "
                                   f"{expected_tag(slug, 0)} is absent or "
                                   f"unreadable, so there is nothing to "
                                   f"compare"})
            notes.append(f"{LOG} against-fig4 {slug}: SKIPPED -- new seed-0 "
                         f"cell absent")
            continue
        if arch_dir is None:
            recs.append({"slug": slug, "status": "SKIPPED",
                         "archived_tag": arch_tag,
                         "reason": f"the archived Figure-4 cell "
                                   f"{arch_tag} is not present under "
                                   f"{[str(r) for r in roots]}"})
            notes.append(
                f"{LOG} against-fig4 {slug}: SKIPPED -- archived cell "
                f"{arch_tag} is absent under {[str(r) for r in roots]}. "
                f"NOT a pass: the published condition was not compared "
                f"for this checkpoint.")
            continue
        try:
            arch_cfg = json.load(open(os.path.join(arch_dir, "config.json")))
        except Exception as exc:                            # noqa: BLE001
            out.append(f"{LOG} FAIL against-fig4 {slug}: archived "
                       f"config.json unreadable ({exc})")
            ok = False
            recs.append({"slug": slug, "status": "FAIL",
                         "archived_tag": arch_tag,
                         "reason": f"archived config unreadable ({exc})"})
            continue
        hard, exp, env, delta = compare_configs(arch_cfg, new_rec["cfg"],
                                                slug)
        if hard:
            ok = False
            out.append(f"{LOG} FAIL against-fig4 {slug}: "
                       f"{len(hard)} field(s) differ from the published "
                       f"Figure-4 cell {arch_tag} beyond the allowed "
                       f"seed / n_rounds / operator set:")
            for field, ov, nv, why in hard:
                out.append(f"           {field}: old={ov!r} new={nv!r}  "
                           f"({why})")
        recs.append({"slug": slug,
                     "status": "FAIL" if hard else "OK",
                     "archived_tag": arch_tag,
                     "archived_dir": arch_dir,
                     "new_tag": new_rec["tag"],
                     "hard_fails": [list(x) for x in hard],
                     "expected_diffs": [list(x) for x in exp],
                     "environment_diffs": [list(x) for x in env],
                     "code_deltas": [list(x) for x in delta]})
    return ok, recs


# ------------------------------------------------------------- reports
def cross_cell_checks(recs, out, notes, innate_sha_pin):
    """Shared world, and 'the seed did something'. Returns ok."""
    ok = True
    good = [r for r in recs if r["ok"] and r["slot"] is not None]

    # --- innate bit-identity across the six checkpoints, per seed -----
    by_seed = {}
    for r in good:
        sha = r["report"].get("innate_sha256")
        if sha:
            by_seed.setdefault(r["slot"][1], {}).setdefault(sha, []).append(
                r["slot"][0])
    for seed in sorted(by_seed):
        groups = by_seed[seed]
        if len(groups) > 1:
            ok = False
            out.append(
                f"{LOG} FAIL wave: at seed {seed} the innate vector is NOT "
                f"bit-identical across checkpoints -- {len(groups)} "
                f"distinct vectors. SEED_BASE_DATA=1 builds the SAME "
                f"population and graph for every model, so the six cells "
                f"at one seed must sit on one world; they are not "
                f"comparable otherwise.")
            for sha, slugs in groups.items():
                out.append(f"           {sha[:16]}...: {sorted(slugs)}")

    # --- the cross-SEED fact, reported (see the module docstring) -----
    all_shas = {sha for groups in by_seed.values() for sha in groups}
    if len(by_seed) > 1:
        if len(all_shas) == 1:
            notes.append(
                f"{LOG} note: the innate vector is bit-identical across "
                f"ALL seeds as well ({sorted(all_shas)[0][:16]}...). That "
                f"is CORRECT on this surface and is not a finding: "
                f"load_movielens_setup takes no seed, so innate = "
                f"(rating - 1)/4 over the 10-NN LCC is a pure function of "
                f"(dataset, target). The seed moves the peer/training RNG, "
                f"not the world -- which is why 'seeds give different "
                f"innate vectors' is NOT gated here.")
        else:
            notes.append(
                f"{LOG} note: {len(all_shas)} distinct innate vectors "
                f"across seeds. On movielens that is unexpected (the "
                f"loader consults no RNG); check whether the dataset or "
                f"target changed.")

    # --- graph fingerprints, where the artifacts carry any ------------
    fp_by_seed = {}
    for r in good:
        for key, val in (r["report"].get("graph_fingerprints") or {}).items():
            fp_by_seed.setdefault((r["slot"][1], key), {}).setdefault(
                val, []).append(r["slot"][0])
    if not fp_by_seed:
        notes.append(f"{LOG} note: no graph fingerprint is recorded on this "
                     f"path (fj_graph_sha256 is written only for "
                     f"pop_model='fj'), so the shared-graph claim rests on "
                     f"the innate/profiles identity above plus the "
                     f"deterministic 10-NN construction in "
                     f"load_movielens_setup.")
    for (seed, key), groups in sorted(fp_by_seed.items()):
        if len(groups) > 1:
            ok = False
            out.append(f"{LOG} FAIL wave: at seed {seed} the graph "
                       f"fingerprint {key} takes {len(groups)} distinct "
                       f"values across checkpoints: "
                       f"{ {k[:16]: sorted(v) for k, v in groups.items()} }")

    # --- profiles identity, per seed ----------------------------------
    prof_by_seed = {}
    for r in good:
        sha = r["report"].get("profiles_sha256")
        if sha:
            prof_by_seed.setdefault(r["slot"][1], {}).setdefault(
                sha, []).append(r["slot"][0])
    for seed in sorted(prof_by_seed):
        if len(prof_by_seed[seed]) > 1:
            ok = False
            out.append(f"{LOG} FAIL wave: at seed {seed} the agent profiles "
                       f"differ across checkpoints "
                       f"({len(prof_by_seed[seed])} distinct) -- the six "
                       f"cells are not on one population")

    # --- the seed must have DONE something ----------------------------
    by_model = {}
    for r in good:
        sha = r["report"].get("op_sha256")
        if sha:
            by_model.setdefault(r["slot"][0], {}).setdefault(sha, []).append(
                r["slot"][1])
    for slug in sorted(by_model):
        for sha, seeds in by_model[slug].items():
            if len(seeds) > 1:
                ok = False
                out.append(
                    f"{LOG} FAIL wave: {slug} carries a BIT-IDENTICAL "
                    f"op_raw at seeds {sorted(seeds)}. The seed feeds the "
                    f"peer and training RNG, so two seeds producing the "
                    f"same {ROUNDS}x{N_AGENTS} trajectory means the seed never "
                    f"reached the runner -- the three-seed claim would be "
                    f"one run counted three times.")

    # --- one hardware family ------------------------------------------
    gpus = {r["report"].get("gpu_name") for r in good
            if r["report"].get("gpu_name")}
    if len(gpus) > 1:
        notes.append(f"{LOG} note: cells span GPU SKUs {sorted(gpus)}; "
                     f"greedy decoding is bit-reproducible only within one "
                     f"architecture, so cross-seed differences partly "
                     f"reflect hardware")
    tvs = {r["report"].get("transformers_version") for r in good
           if r["report"].get("transformers_version")}
    if len(tvs) > 1:
        notes.append(f"{LOG} note: cells span transformers versions "
                     f"{sorted(tvs)}; generation is not strictly "
                     f"comparable across them")
    return ok


def print_table(by_slot):
    hdr = (f"{'cell':<58} {'ok':>5} {'pfmax':>7} {'popMean':>8} "
           f"{'popSD':>7} {'lateMean':>9} {'lateSD':>8} {'srvMean':>8}")
    print(hdr)
    print("-" * len(hdr))
    for slug, seed in full_grid():
        name = f"{slug}/s{seed}"
        r = by_slot.get((slug, seed))
        if r is None:
            print(f"{name:<58} {'---':>5}   ABSENT")
            continue
        rp = r["report"]

        def f(key, width, prec):
            val = rp.get(key)
            if val is None:
                return f"{'-':>{width}}"
            return f"{val:>{width}.{prec}f}"

        print(f"{name:<58} {'ok' if r['ok'] else 'FAIL':>5} "
              f"{f('parse_fail_max', 7, 4)} {f('pop_final_mean', 8, 4)} "
              f"{f('pop_final_sd', 7, 4)} {f('late_mean', 9, 4)} "
              f"{f('late_sd', 8, 4)} {f('served_final_mean', 8, 4)}")


def print_against(recs):
    print("\n" + "=" * 78)
    print("AGAINST THE PUBLISHED FIGURE 4 (field by field, seed-0 cells)")
    print(f"The horizons MATCH ({ROUNDS} rounds), so n_rounds is asserted "
          f"EQUAL, not exempted:")
    print("a moving horizon is a HARD FAIL with its own message.")
    print("EXPECTED = the differences the wave exists to create.")
    print("ENVIRONMENT = where/by which tree it computed; never failed.")
    print("CODE DELTA = a field the runner gained after 2026-08-17, each")
    print("             registered with the value it must carry. Read the")
    print("             serve_eval_mode note: it is NOT behaviour-neutral.")
    print("HARD FAIL = anything else.")
    print("=" * 78)
    for r in recs:
        if r["status"] == "SKIPPED":
            print(f"  {r['slug']:<12} SKIPPED -- {r['reason']}")
            continue
        print(f"  {r['slug']:<12} {r['status']}  vs {r['archived_tag']}")
        for label, key in (("EXPECTED", "expected_diffs"),
                           ("ENVIRONMENT", "environment_diffs"),
                           ("CODE DELTA", "code_deltas"),
                           ("HARD FAIL", "hard_fails")):
            for field, ov, nv, why in r.get(key) or []:
                sov = str(ov)
                snv = str(nv)
                print(f"      {label:<11} {field}: old={sov[:52]} "
                      f"new={snv[:52]}")
                if label in ("CODE DELTA", "HARD FAIL"):
                    print(f"                  {why}")


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Gate for the Figure-4 replication wave (pofdf4r_, "
                    "18 cells). CPU only.")
    ap.add_argument("runs", nargs="*",
                    help="explicit run dirs; if omitted the --run-root(s) "
                         "are scanned for pofdf4r_* cells")
    ap.add_argument("--run-root", dest="run_roots", action="append",
                    default=None,
                    help="run root to scan; repeatable. Default: both "
                         "notes/pofd/cluster and runs/pokec_gated_lm, the "
                         "two roots plot_sft_family_prior_one_row.py uses.")
    ap.add_argument("--against-fig4", action="store_true",
                    help="compare each new seed-0 cell field by field with "
                         "the archived Figure-4 cell "
                         "pofdfam_{slug}_b1_ea1_w0p5_l0p2_es0p05_s0")
    ap.add_argument("--allow-partial", action="store_true",
                    help="do not hard-fail on absent cells (still reported "
                         "cell by cell, and the verdict says PARTIAL)")
    ap.add_argument("--innate-sha", default=CANONICAL_INNATE_SHA,
                    help="expected sha256 of the float32 innate vector "
                         "(default: the canonical movielens-Action "
                         "723-agent vector)")
    ap.add_argument("--no-innate-pin", action="store_true",
                    help="drop the innate sha pin and check only that the "
                         "six checkpoints at a seed agree")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the machine-readable verdict here")
    args = ap.parse_args(argv)

    roots = [Path(r) for r in (args.run_roots or DEFAULT_ROOTS)]
    live_roots = [r for r in roots if r.is_dir()]
    out, notes = [], []
    pin = None if args.no_innate_pin else (args.innate_sha or "").strip().lower()

    run_dirs = list(args.runs)
    if not run_dirs:
        if not live_roots:
            print(f"{LOG} usage error: none of "
                  f"{[str(r) for r in roots]} is a directory and no run "
                  f"dirs were given", file=sys.stderr)
            return 2
        if len(live_roots) < len(roots):
            notes.append(f"{LOG} note: scanned only "
                         f"{[str(r) for r in live_roots]}; absent: "
                         f"{[str(r) for r in roots if r not in live_roots]}")
        for root in live_roots:
            run_dirs += [str(p) for p in sorted(root.iterdir())
                         if p.is_dir() and p.name.startswith(PROD_PREFIX)]
        run_dirs = sorted(set(run_dirs))
    if not run_dirs:
        print(f"{LOG} usage error: no {PROD_PREFIX}* run dirs found under "
              f"{[str(r) for r in live_roots]}. Nothing to gate is NOT a "
              f"pass.", file=sys.stderr)
        return 2

    # one run's tensors at a time: check_one loads, digests and drops
    recs = [check_one(rd, out, notes, innate_sha_pin=pin) for rd in run_dirs]
    allok = all(r["ok"] for r in recs)

    by_slot, dupes = {}, []
    for r in recs:
        if r["slot"] is None:
            continue
        if r["slot"] in by_slot:
            dupes.append((r["slot"], by_slot[r["slot"]]["tag"], r["tag"]))
        by_slot[r["slot"]] = r
    for slot, a, b in dupes:
        out.append(f"{LOG} FAIL wave: two run dirs claim cell {slot}: "
                   f"{a} and {b}")
        allok = False

    allok &= cross_cell_checks(recs, out, notes, pin)

    against = []
    if args.against_fig4:
        aok, against = against_fig4(live_roots or roots, by_slot, out, notes)
        allok &= aok

    grid = full_grid()
    missing = [s for s in grid if s not in by_slot]
    present = len(grid) - len(missing)

    for line in out:
        print(line)
    for line in notes:
        print(line)

    print("\n" + "=" * 78)
    print(f"PER-CELL REPORT -- {PROD_PREFIX}* , {ROUNDS} rounds, "
          f"{N_AGENTS} agents. op_raw[t] is the END-OF-ROUND POST-PEER")
    print(f"state (peers run last). Late window = post-peer rounds "
          f"{LATE_LO}-{LATE_HI}.")
    print("=" * 78)
    print_table(by_slot)

    if args.against_fig4:
        print_against(against)
        skipped = [r["slug"] for r in against if r["status"] == "SKIPPED"]
        if skipped:
            print(f"\n{LOG} against-fig4 SKIPPED for {skipped} -- the "
                  f"published condition was NOT compared for those "
                  f"checkpoints. That is not a pass for them.")

    print("\n" + "=" * 78)
    if missing:
        print(f"GRID COVERAGE: {present} of {len(grid)} cells present -- "
              f"{len(missing)} MISSING")
        print("A silently short grid must not look like a complete result.")
        for slug, seed in missing:
            print(f"  MISSING  {slug:<12} s{seed:<3} expected tag "
                  f"{expected_tag(slug, seed)}")
        if not args.allow_partial:
            allok = False
            print(f"{LOG} the missing cells above are a HARD FAILURE. Pass "
                  f"--allow-partial to gate a deliberately partial grid.")
    else:
        print(f"GRID COVERAGE: all {len(grid)} cells present "
              f"({len(MODEL_ORDER)} checkpoints x {len(SEEDS)} seeds)")
    print("=" * 78)

    verdict = {
        "wave": "fig4_family_prior_repl30",
        "prefix": PROD_PREFIX,
        "rounds": ROUNDS,
        "seeds": list(SEEDS),
        "models": {k: MODELS[k] for k in MODEL_ORDER},
        "want_population_update": WANT_POP_UPDATE,
        "want_ai_gate_reference": WANT_GATE_REF,
        "innate_sha_pin": pin,
        "n_runs": len(recs),
        "n_cells_present": present,
        "n_cells_total": len(grid),
        "missing": [{"model": s[0], "seed": s[1],
                     "expected_tag": expected_tag(*s)} for s in missing],
        "partial": bool(missing),
        "pass": bool(allok),
        "cells": [{"tag": r["tag"], "run_dir": r["run_dir"], "ok": r["ok"],
                   "slot": r["slot"], "report": r["report"]} for r in recs],
        "against_fig4": against,
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(verdict, indent=2))
        print(f"{LOG} verdict -> {args.json_out}")

    if allok and missing:
        print(f"{LOG} PASS (PARTIAL GRID: {present}/{len(grid)} cells) -- "
              f"operator v2/anchor on every cell, exact checkpoint ids, "
              f"seeds match their tags, one shared world per seed, "
              f"{ROUNDS} rounds, zero parse failures. DO NOT present this "
              f"as the complete three-seed replication.")
        return 0
    if allok:
        print(f"{LOG} PASS -- {len(recs)} run(s), {present}/{len(grid)} "
              f"cells: operator v2/anchor on every cell, exact checkpoint "
              f"ids, seeds match their tags, one shared world per seed, "
              f"{ROUNDS} rounds, zero parse failures.")
        return 0
    print(f"{LOG} FAILED -- see above ({len(recs)} run(s) inspected)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
