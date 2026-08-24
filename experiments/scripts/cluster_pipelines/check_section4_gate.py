#!/usr/bin/env python3
"""Gate for the SECTION-4 CORRECTED-GATE wave (pofds4g_, 2026-08-24).

The wave re-runs the completed Mistral bottom-20% FIXED-vs-EVOLVING
Section-4 experiment under the CORRECTED AI gate:

    |m - x'| <= eps_AI   with x' = k*innate + (1-k)*x   (the ANCHORED
                                                         opinion)

instead of the raw start-of-round opinion x0. In the runner that is
AI_GATE_REFERENCE=anchor, which makes

    config["population_update"] == "nested_ai_anchored_then_social_v2"
    config["ai_gate_reference"] == "anchor"

(see _POP_UPDATE_MARKER in run_pokec_gated_lm.py). EVERY archived
Section-4 cell carries the OLD "nested_ai_then_social_v1" marker, so a
run recording v1 is a HARD FAILURE here, not a warning: it is an
archived-semantics run wearing a corrected-gate tag, and nothing about
this wave's claim survives it. That is the single most important gate in
this file.

WHAT IS CHECKED (each item HARD-FAILS; nothing in this file warns):
  1. GRAMMAR + COVERAGE -- every present run dir parses under the pinned
     tag grammar, and the parsed (arm, cond, ea, es, seed) set equals the
     full 72-cell product. Missing cells are listed explicitly and are a
     hard failure; a short grid never looks like a complete result.
  2. OPERATOR -- population_update == nested_ai_anchored_then_social_v2
     AND ai_gate_reference == "anchor". v1 fails by name.
  3. GRID FIELDS, TAG vs CONFIG IN BOTH DIRECTIONS -- eps_ai, eps
     (social), w_plat, innate_lambda, homophily gamma, n_rounds, base
     model, dataset/target, seed, train_cap, n_labeled, kl_direction,
     the numeric threshold gates, and the arm envelope (style, kl_beta,
     icl_k, icl_days, use_lora, fresh_each_round). Both directions means
     the config value is re-rendered into the tag grammar and compared to
     the token, and the arm/condition are INFERRED BACK from the config
     and compared to the tag -- a table lookup in one direction only
     cannot catch a tag that lies about a config it also matches.
  4. CONDITION INTEGRITY --
       fixed:    innate_clamp_mode=="bottom", frac==0.2, seed==run seed,
                 peer mode=="stubborn", count==145, and the cohort is
                 RECONSTRUCTED from innate (the deterministic
                 innate-then-id ranking) rather than trusted; the clamped
                 rows equal innate BIT-EXACTLY in op_raw AND twin_raw at
                 every recorded round.
       evolving: the config carries NO innate_clamp_* key at all and the
                 trajectory carries no clamp artifact.
  5. COHORT PAIRING -- for each (arm, ea, es, seed) the fixed and
     evolving members of the pair must share the innate vector
     BIT-EXACTLY, hence the same reconstructed bottom-145 cohort. The
     whole wave must in fact sit on ONE innate vector: with
     PROFILE_SHUFFLE_P=0 and no routing treatment, load_movielens_setup
     is a pure function of (dataset, target) -- it takes no seed at all,
     innate is (rating-1)/4 on the LCC of a cosine 10-NN graph -- and
     gp.innate_clamp_mask("bottom") is a pure sort with no RNG draw. So
     innate, the graph and the cohort are SEED-INVARIANT.
  5b. SEED-DISTINCTNESS -- the direct consequence of 5. Because nothing
     about the world depends on the seed, config["seed"] is the ONLY
     per-run evidence that a seed reached the runner, and that field is
     written from the environment rather than observed from behaviour.
     So within each (arm, cond, ea, es) cell, no two seeds may produce a
     BIT-IDENTICAL op_raw. Two identical trajectories mean the seed never
     reached the training/serving stream: invisible in every per-run
     field, and it would silently collapse the three-seed confidence
     intervals to a single observation. Compared by sha256 over op_raw,
     so no two runs' tensors are ever resident at once. A missing third
     seed is a COVERAGE failure (already fatal), not a pass here.
  6. TWIN present, correctly shaped, finite, in [0,1] and non-degenerate.
  7. ZERO PARSE FAILURES -- read from raw_gen_log.json.gz when that
     artifact exists. IT DOES NOT EXIST FOR THIS WAVE: the S4G sub
     templates in gen_pofd_sweep.py do not set SAVE_RAW_GEN=1, and the
     runner writes raw_gen_log.json.gz (the only place parse_fail_frac is
     recorded) only under SAVE_RAW_GEN=1. So the default behaviour is:
       * raw log present  -> every round must report parse_fail_frac == 0
                             and a full 723-value parsed vector;
       * raw log absent and config does not claim save_raw_gen -> fall
         back to the EQUIVALENT evidence in trajectory.pt: a parse
         failure writes NaN into the served vector (run_pokec_gated_lm
         maps an unparsable generation to float("nan")), so ANY
         non-finite entry in pred_raw is a hard failure. The fallback is
         announced per cell and in the verdict; it is not silent.
       * raw log absent while config says save_raw_gen -> hard failure.
     --require-raw-gen makes the absence itself a hard failure.
  8. len(trajectory) >= n_rounds, with op_raw/pred_raw/twin_raw shaped
     [n_rounds, 723].
  9. --smoke gates the 3-round pofds4gsmk_ cells (4 of them: both arms x
     both conditions at ea=1, es=0.2, seed 0) and HONOURS THE RUN ROOT IT
     IS PASSED. (check_section3.py --smoke has a bug where the run dir it
     is given is ignored; that bug is deliberately not copied.)

The clamp logic is not re-invented here: the cohort reconstruction, the
"bit-exact in population AND twin" assertion, the stubborn-peer
treatment of the responsive twin, and the d8 personal-history replay all
MIRROR check_pofd_sanity.check_run's "-- 1j CLAMP" block (and its "1j
EVO" counterpart at `if is_evo:`), which is itself the algorithm of
_gated_pop.innate_clamp_mask(mode="bottom"). Each mirrored piece names
its source in a comment.

--------------------------------------------------------------------
Usage
--------------------------------------------------------------------
  # the production wave, on the cluster login node (threads are pinned
  # inside this module, before torch is imported)
  python check_section4_gate.py \\
      --run-root /home/gsmithline/perfsim/runs/pokec_gated_lm

  # the 4-job 3-round smoke
  python check_section4_gate.py --smoke \\
      --run-root /home/gsmithline/perfsim/runs/pokec_gated_lm

  # gate exactly the tags in a file (a deliberately partial pull)
  python check_section4_gate.py --run-root RUNS --tags-file tags.txt

  # machine-readable verdict
  python check_section4_gate.py --run-root RUNS --json /tmp/s4g.json

Exit codes: 0 = every check passed, 1 = hard failure, 2 = usage error.
"""
from __future__ import annotations

import os

# PERFORMANCE / SHARED-NODE HYGIENE: this gate may run on the cluster
# LOGIN NODE, so BLAS fan-out is pinned BEFORE torch is imported (after
# the import the env vars no longer take effect). USE_TF=0 keeps
# transformers' TensorFlow probe out of the way if anything on the path
# pulls it in.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("USE_TF", "0")

import argparse          # noqa: E402
import gzip              # noqa: E402
import hashlib           # noqa: E402
import importlib.util    # noqa: E402
import json              # noqa: E402
import re                # noqa: E402
import sys               # noqa: E402
from pathlib import Path  # noqa: E402

import torch             # noqa: E402

torch.set_num_threads(1)

# remote capture pipes checkers over ssh stdin, where __file__ is unset
HERE = (os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals() else os.getcwd())

# the ONE cohort definition (gp.innate_clamp_mask / gp.innate_clamp_hash)
# is shared with the runner via _gated_pop.py, so the deployed clamp and
# this reconstruction can never drift. Loaded exactly as the runner and
# check_pofd_sanity load it.
_GP_PATH = os.path.join(HERE, "_gated_pop.py")
_spec_gp = importlib.util.spec_from_file_location("_gated_pop_s4g", _GP_PATH)
gp = importlib.util.module_from_spec(_spec_gp)
_spec_gp.loader.exec_module(gp)

LOG = "[check_s4g]"

# ---------------------------------------------------------------- design
DEFAULT_RUN_ROOT = "/home/gsmithline/perfsim/runs/pokec_gated_lm"

PROD_PREFIX = "pofds4g"
SMOKE_PREFIX = "pofds4gsmk"
# scan prefixes carry the separator: "pofds4gsmk_..." also startswith
# "pofds4g", so a bare-prefix scan in production mode would silently
# swallow the smokes.
PROD_SCAN = PROD_PREFIX + "_"
SMOKE_SCAN = SMOKE_PREFIX + "_"

OP_TOKEN = "anch2"                 # <-> nested_ai_anchored_then_social_v2
OP_INFIX = "_" + OP_TOKEN + "_"
WANT_MARKER = "nested_ai_anchored_then_social_v2"
OLD_MARKER = "nested_ai_then_social_v1"
WANT_GATE_REF = "anchor"

N = 723
PROD_ROUNDS = 30
SMOKE_ROUNDS = 3
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SLUG = "mistral7b"

ARMS = ("b0", "d8")
CONDS = ("fixed", "evolving")
COND_TOK = {"fixed": "fixb20", "evolving": "evoall"}
TOK_COND = {v: k for k, v in COND_TOK.items()}
EAS = (0.2, 1.0)
ESS = (0.0, 0.2, 1.0)
SEEDS = (0, 42, 43)
W_PLAT = 0.5                       # the paper's beta
INNATE_LAMBDA = 0.2                # the paper's gamma / innate anchor k
GAMMA_BIAS = 0.0                   # homophily selection bias: always 0
CLAMP_FRAC = 0.2
CLAMP_COUNT = 145                  # round(0.2 * 723)
SMOKE_EA, SMOKE_ES, SMOKE_SEED = 1.0, 0.2, 0
N_TOTAL = len(ARMS) * len(CONDS) * len(EAS) * len(ESS) * len(SEEDS)   # 72

# TAG GRAMMAR (pinned). PARSED, never table-looked-up: a new dose or seed
# must fail as a GRID error naming the value, not as a "malformed tag".
TAG_RE = re.compile(
    r"^(?P<pre>pofds4gsmk|pofds4g)"
    r"_(?P<slug>[a-z0-9]+)"
    r"_(?P<arm>b0|d8)"
    r"_(?P<cond>fixb20|evoall)"
    r"_(?P<op>anch2)"
    r"_ea(?P<ea>[0-9p]+)"
    r"_w(?P<w>[0-9p]+)"
    r"_l(?P<l>[0-9p]+)"
    r"_es(?P<es>[0-9p]+)"
    r"_s(?P<seed>\d+)$")

# Everything HELD FIXED across all 72 cells, byte-matched to the
# completed Section-4 surface. Values that are true by construction of
# the sub template's env are pinned here anyway: "true by construction"
# is a claim about the generator, and this file gates the ARTIFACT.
PINS = {
    "base_model": BASE_MODEL,
    "dataset": "movielens",
    "ml_target": "Action",
    "n_labeled": N,
    "train_cap": N,
    "kl_direction": "forward",
    "ai_gate_mode": "threshold",
    "peer_gate_mode": "threshold",
    "gamma_bias": GAMMA_BIAS,
    "w_plat": W_PLAT,
    "innate_lambda": INNATE_LAMBDA,
    "pop_model": "ab",
    "run_mode": "loop",
    "anchor_mode": "fixed",
    "data_regime": "replace",
    "deploy_every": 1,
    "platform_sus_scale": 1.0,
    "canary_delta": 0.0,
    "ab_sweeps": 1,
    "epoch_size": 100,
    "sft_epochs": 1,
    "sft_batch_size": 4,
    "lora_r": 512,
    "sft_lr": 5e-5,
    "pristine_frac": 0.0,
    "replay_frac": 0.0,
    "teacher_label_delta": 0.0,
    "kl_ref_adapter": "",
    "feedback_mode": "none",
    "icrh": False,
    "do_sample": False,
    "seed_base_data": True,
    "serve_eval_mode": True,
    "fj_update_version": "legacy",
}

# The two arms, as config surfaces. b0 = ordinary SFT; d8 = frozen
# personal-history ICL (each prompt carries only that agent's OWN last 8
# recorded opinions -- ICL_K=0, so no cross-user exemplar exists).
ARM_WANT = {
    "b0": {"training_style": "sft", "kl_beta": 0.0, "icl_k": 0,
           "icl_days": 0, "use_lora": True, "fresh_each_round": True},
    "d8": {"training_style": "frozen", "kl_beta": 0.0, "icl_k": 0,
           "icl_days": 8, "use_lora": False, "fresh_each_round": False},
}
# clamp keys that must be ABSENT from an evolving config (and from an
# evolving trajectory)
CLAMP_CFG_KEYS = ("innate_clamp_mode", "innate_clamp_frac",
                  "innate_clamp_seed", "innate_clamp_peer_mode",
                  "sft_exclude_clamped")
CLAMP_TRAJ_KEYS = ("innate_clamp_mask", "innate_clamp_count",
                   "innate_clamp_mode", "innate_clamp_frac",
                   "innate_clamp_seed", "innate_clamp_hash",
                   "innate_clamp_peer_mode", "clamp_fr_touch_raw")


# ------------------------------------------------------------- grammar
def _num(v):
    """float -> the tag's number grammar. Mirrors gen_pofd_sweep._num:
    0.2 -> '0p2', 1.0 -> '1', 0.0 -> '0'."""
    return f"{float(v):g}".replace(".", "p")


def _unnum(tok):
    """the tag's number grammar -> float: '0p2' -> 0.2, '1' -> 1.0."""
    return float(tok.replace("p", "."))


def render_tag(arm, cond, ea, es, seed, smoke=False):
    pre = SMOKE_PREFIX if smoke else PROD_PREFIX
    return (f"{pre}_{MODEL_SLUG}_{arm}_{COND_TOK[cond]}_{OP_TOKEN}"
            f"_ea{_num(ea)}_w{_num(W_PLAT)}_l{_num(INNATE_LAMBDA)}"
            f"_es{_num(es)}_s{int(seed)}")


def expected_cells(smoke):
    """The conceptual grid: 72 production cells, or the 4-cell smoke."""
    if smoke:
        return [(arm, cond, SMOKE_EA, SMOKE_ES, SMOKE_SEED)
                for arm in ARMS for cond in CONDS]
    return [(arm, cond, ea, es, seed)
            for seed in SEEDS for arm in ARMS for cond in CONDS
            for ea in EAS for es in ESS]


def parse_tag(tag, smoke):
    """(info, errs). info is None when nothing downstream can be trusted.

    Every value is PARSED out of the tag and then required to round-trip
    back through the same grammar, so a token that parses to the right
    float in the wrong spelling ('ea0p20', 'es1p0', 's042') is caught.
    """
    errs = []
    # THE operator token, checked before anything else so its absence is
    # reported as itself rather than as a generic grammar miss.
    if OP_INFIX not in tag:
        return None, [f"tag carries no {OP_INFIX!r} token -- every "
                      f"Section-4 corrected-gate tag MUST declare the "
                      f"anchored operator ({WANT_MARKER})"]
    m = TAG_RE.match(tag)
    if m is None:
        return None, [f"tag is not in the pofds4g grammar "
                      f"{PROD_PREFIX}_{MODEL_SLUG}_<b0|d8>_<fixb20|evoall>"
                      f"_{OP_TOKEN}_ea<EA>_w0p5_l0p2_es<ES>_s<SEED>"]
    pre, slug = m.group("pre"), m.group("slug")
    arm, cond = m.group("arm"), TOK_COND[m.group("cond")]
    ea, es = _unnum(m.group("ea")), _unnum(m.group("es"))
    w, lam = _unnum(m.group("w")), _unnum(m.group("l"))
    seed = int(m.group("seed"))
    if (pre == SMOKE_PREFIX) != bool(smoke):
        errs.append(f"smoke/production prefix mismatch: prefix {pre!r} with "
                    f"--smoke={bool(smoke)}; a smoke cell can never stand in "
                    f"for a production cell (or the reverse)")
    if slug != MODEL_SLUG:
        errs.append(f"model slug {slug!r}; this wave is {MODEL_SLUG}-only")
    rebuilt = (f"{pre}_{slug}_{arm}_{COND_TOK[cond]}_{OP_TOKEN}_ea{_num(ea)}"
               f"_w{_num(w)}_l{_num(lam)}_es{_num(es)}_s{seed}")
    if rebuilt != tag:
        errs.append(f"tag numbers do not round-trip through the pinned "
                    f"grammar (would be spelled {rebuilt!r}) -- two "
                    f"spellings of one cell make coverage unprovable")
    if w != W_PLAT:
        errs.append(f"tag says w{m.group('w')} (= {w:g}); the wave is "
                    f"W_PLAT={W_PLAT:g}")
    if lam != INNATE_LAMBDA:
        errs.append(f"tag says l{m.group('l')} (= {lam:g}); the wave is "
                    f"INNATE_LAMBDA={INNATE_LAMBDA:g}")
    if smoke:
        if ea != SMOKE_EA or es != SMOKE_ES or seed != SMOKE_SEED:
            errs.append(f"smoke cell must be ea{_num(SMOKE_EA)} "
                        f"es{_num(SMOKE_ES)} s{SMOKE_SEED}; got ea{_num(ea)} "
                        f"es{_num(es)} s{seed}")
    else:
        if ea not in EAS:
            errs.append(f"eps_ai {ea:g} is not in the grid {list(EAS)}")
        if es not in ESS:
            errs.append(f"eps_social {es:g} is not in the grid {list(ESS)}")
        if seed not in SEEDS:
            errs.append(f"seed {seed} is not in the grid {list(SEEDS)}")
    info = {"pre": pre, "arm": arm, "cond": cond, "ea": ea, "es": es,
            "seed": seed, "cell": (arm, cond, ea, es, seed)}
    return info, errs


# --------------------------------------------------------------- cohort
def bottom_cohort_mask(innate, n_frozen):
    """Boolean [n] mask of the n_frozen LOWEST innate opinions, agent id
    as the deterministic tie-break.

    MIRRORED from check_pofd_sanity.check_run's "-- 1j CLAMP" block (its
    `order_cl` / `want_ids` reconstruction), which is in turn the exact
    algorithm of _gated_pop.innate_clamp_mask(mode="bottom"). Kept as an
    independent line of code on purpose: a helper bug must not be able to
    self-certify, so a fixed run is checked against BOTH this and
    gp.innate_clamp_mask.
    """
    n = int(innate.numel())
    order = sorted(range(n), key=lambda i: (float(innate[i]), i))
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.tensor(sorted(order[:n_frozen]), dtype=torch.long)] = True
    return mask


def _sha_t(t):
    """sha256 over a tensor's raw float32 bytes -- bit-identity, not
    closeness."""
    return hashlib.sha256(
        t.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()


def _bit_eq_rows(block, vec):
    """Every row of `block` [T,k] bit-identical to `vec` [k]."""
    return bool((block == vec.unsqueeze(0)).all())


# ------------------------------------------------------------ one cell
def check_one(run_dir, smoke, require_raw_gen):
    """Gate ONE run dir. Returns a record of scalars/hashes only: no
    tensor outlives this call, so the gate holds at most one run's
    trajectory in memory at a time (login-node budget)."""
    run_dir = str(run_dir).rstrip("/")
    tag = os.path.basename(run_dir)
    rec = {"run_dir": run_dir, "tag": tag, "cell": None, "errs": [],
           "notes": [], "parse_evidence": None, "innate_sha256": None,
           "cohort_sha256": None, "n_rounds": None, "pop_final_mean": None,
           "pop_final_sd": None, "op_twin_l1": None,
           "op_sha256": None}
    errs = rec["errs"]

    def bad(msg):
        errs.append(msg)

    info, terrs = parse_tag(tag, smoke)
    errs.extend(terrs)
    if info is None:
        return rec
    rec["cell"] = info["cell"]
    arm, cond = info["arm"], info["cond"]
    want_rounds = SMOKE_ROUNDS if smoke else PROD_ROUNDS

    tp = os.path.join(run_dir, "trajectory.pt")
    if not os.path.exists(tp):
        bad("no trajectory.pt -- trajectory.pt and raw_gen_log.json.gz are "
            "written ONLY at run completion, so this cell is incomplete "
            "(config.json alone is written at launch and proves nothing)")
        return rec
    d = torch.load(tp, map_location="cpu", weights_only=False)
    cfg = d.get("config")
    if not isinstance(cfg, dict) or not cfg:
        bad("trajectory.pt carries no config dict")
        del d
        return rec

    # --- 2. THE OPERATOR. The most important gate in this file. --------
    pu = cfg.get("population_update", "<absent>")
    if pu == OLD_MARKER:
        bad(f"population_update={OLD_MARKER!r} -- this is the OLD, ARCHIVED "
            f"gate reference (|m - x0| on the RAW start-of-round opinion). "
            f"The corrected-gate wave requires {WANT_MARKER!r} "
            f"(AI_GATE_REFERENCE=anchor, |m - x'| on the ANCHORED opinion). "
            f"An archived-semantics run wearing a {OP_TOKEN!r} tag cannot "
            f"stand in for a cell of this wave")
    elif pu != WANT_MARKER:
        bad(f"population_update={pu!r}, expected {WANT_MARKER!r} -- an "
            f"absent or unknown round-operator marker is a hard failure, "
            f"never a silent fallback")
    gr = cfg.get("ai_gate_reference", "<absent>")
    if gr != WANT_GATE_REF:
        bad(f"ai_gate_reference={gr!r}, expected {WANT_GATE_REF!r} (an "
            f"absent field means the run predates 2026-08-22 and gated on "
            f"x0)")
    # config.json, written at launch, must tell the same story as the
    # trajectory config written at completion
    cjp = os.path.join(run_dir, "config.json")
    if os.path.exists(cjp):
        try:
            cj = json.loads(open(cjp).read())
        except (ValueError, OSError) as e:
            bad(f"config.json unreadable: {e}")
            cj = {}
        for k in ("population_update", "ai_gate_reference"):
            if k in cj and cj.get(k) != cfg.get(k):
                bad(f"config.json {k}={cj.get(k)!r} disagrees with "
                    f"trajectory.pt {k}={cfg.get(k)!r}")

    # --- 3. GRID FIELDS, TAG vs CONFIG in BOTH directions -------------
    # forward: the config value must equal what the tag says
    for key, tok, want in (("eps_ai", f"ea{_num(info['ea'])}", info["ea"]),
                           ("eps", f"es{_num(info['es'])}", info["es"])):
        got = cfg.get(key, None)
        if got is None or abs(float(got) - want) > 1e-12:
            bad(f"{key}={got!r} but the tag says {tok} (= {want:g}) -- the "
                f"queue column did not reach the runner")
        else:
            # backward: re-render the CONFIG value into the tag grammar
            # and require the token itself
            if _num(float(got)) != _num(want):
                bad(f"{key}={got!r} renders as {_num(float(got))!r}, tag "
                    f"token {tok!r}")
    if int(cfg.get("seed", -1)) != info["seed"]:
        bad(f"seed={cfg.get('seed')!r} but the tag says s{info['seed']}")
    if int(cfg.get("n_rounds", -1)) != want_rounds:
        bad(f"n_rounds={cfg.get('n_rounds')!r}, expected {want_rounds}")
    for key, want in PINS.items():
        got = cfg.get(key, "<absent>")
        if isinstance(want, bool):
            if got == "<absent>" or bool(got) is not want:
                bad(f"{key}={got!r}, expected {want!r}")
        elif isinstance(want, float):
            if got == "<absent>" or not isinstance(got, (int, float)) or \
                    abs(float(got) - want) > 1e-12:
                bad(f"{key}={got!r}, expected {want!r}")
        elif got != want:
            bad(f"{key}={got!r}, expected {want!r}")
    # the ARM, forward and backward
    for key, want in ARM_WANT[arm].items():
        got = cfg.get(key, "<absent>")
        if isinstance(want, bool):
            if got == "<absent>" or bool(got) is not want:
                bad(f"arm {arm}: {key}={got!r}, expected {want!r}")
        elif isinstance(want, float):
            if got == "<absent>" or not isinstance(got, (int, float)) or \
                    abs(float(got) - want) > 1e-12:
                bad(f"arm {arm}: {key}={got!r}, expected {want!r}")
        elif got != want:
            bad(f"arm {arm}: {key}={got!r}, expected {want!r}")
    inferred = [a for a, wnt in ARM_WANT.items()
                if all(bool(cfg.get(k)) == bool(v) if isinstance(v, bool)
                       else cfg.get(k) == v for k, v in wnt.items())]
    if inferred != [arm]:
        bad(f"the config envelope reads back as arm(s) {inferred or ['?']} "
            f"while the tag claims {arm!r} -- style/kl_beta/icl_k/icl_days/"
            f"use_lora/fresh_each_round must identify exactly one arm")
    # the CONDITION, backward: a clamp key in the config IS the fixed
    # condition; its absence IS the evolving condition
    cfg_clamped = "innate_clamp_mode" in cfg
    if cfg_clamped != (cond == "fixed"):
        bad(f"tag condition {cond!r} ({COND_TOK[cond]}) but the config "
            f"{'carries' if cfg_clamped else 'carries no'} innate_clamp_mode "
            f"-- the tag and the config disagree about whether 145 agents "
            f"are pinned")

    # --- 8. shapes / horizon ------------------------------------------
    traj = d.get("trajectory")
    if not isinstance(traj, (list, tuple)):
        bad("trajectory (per-round rows) missing or not a list")
        traj = []
    if len(traj) < want_rounds:
        bad(f"trajectory holds {len(traj)} rows < n_rounds {want_rounds} "
            f"-- the run is short")
    op = d.get("op_raw")
    pr = d.get("pred_raw")
    tw = d.get("twin_raw")
    inn = d.get("innate")
    shapes_ok = True
    for nm, t in (("op_raw", op), ("pred_raw", pr), ("twin_raw", tw)):
        if not torch.is_tensor(t) or t.numel() == 0:
            bad(f"{nm} missing/empty")
            shapes_ok = False
        elif tuple(t.shape) != (want_rounds, N):
            bad(f"{nm} shape "
                f"{tuple(t.shape) if torch.is_tensor(t) else None} != "
                f"{(want_rounds, N)}")
            shapes_ok = False
    if not torch.is_tensor(inn) or tuple(inn.shape) != (N,):
        bad(f"innate shape "
            f"{tuple(inn.shape) if torch.is_tensor(inn) else None} != {(N,)}")
        shapes_ok = False
    if not shapes_ok:
        del d
        return rec
    op = op.float()
    pr = pr.float()
    tw = tw.float()
    inn = inn.float()
    rec["n_rounds"] = int(op.shape[0])

    if not torch.isfinite(op).all():
        bad("op_raw has non-finite values")
    elif float(op.min()) < -1e-6 or float(op.max()) > 1 + 1e-6:
        bad(f"op_raw outside [0,1]: [{float(op.min()):.4f}, "
            f"{float(op.max()):.4f}]")
    if not torch.isfinite(inn).all():
        bad("innate has non-finite values")

    # --- 6. TWIN present and non-degenerate ---------------------------
    # WITH_TWIN=1 leaves NO config field (the runner records no with_twin
    # key), so the artifact itself is the only evidence the counterfactual
    # was simulated -- hence twin_raw is gated on shape, finiteness, range
    # and dispersion rather than on a flag.
    if not torch.isfinite(tw).all():
        bad("twin_raw has non-finite values")
    elif float(tw.min()) < -1e-6 or float(tw.max()) > 1 + 1e-6:
        bad(f"twin_raw outside [0,1]: [{float(tw.min()):.4f}, "
            f"{float(tw.max()):.4f}]")
    elif float(tw.std()) == 0.0:
        bad(f"twin_raw is CONSTANT ({float(tw.reshape(-1)[0]):g}) over every "
            f"agent and round -- a degenerate twin is not a counterfactual "
            f"(innate itself is heterogeneous, so the no-platform path "
            f"cannot be constant)")
    rec["op_twin_l1"] = float((op - tw).abs().mean())
    # the deployed trajectory's fingerprint, for the wave-level
    # SEED-DISTINCTNESS check. Hashed here and the tensor dropped, so
    # two runs are never resident at once.
    rec["op_sha256"] = _sha_t(op)
    rec["pop_final_mean"] = float(op[-1].mean())
    rec["pop_final_sd"] = float(op[-1].std())

    # --- 5. the innate vector and the reconstructed cohort ------------
    rec["innate_sha256"] = _sha_t(inn)
    want_frozen = int(round(CLAMP_FRAC * int(inn.numel())))
    if want_frozen != CLAMP_COUNT:
        bad(f"frac {CLAMP_FRAC:g} of {int(inn.numel())} agents gives "
            f"{want_frozen} frozen, expected {CLAMP_COUNT}")
    rec_mask = bottom_cohort_mask(inn, want_frozen)
    rec["cohort_sha256"] = gp.innate_clamp_hash(rec_mask)

    # --- 4. CONDITION INTEGRITY ---------------------------------------
    if cond == "fixed":
        # MIRRORS check_pofd_sanity.check_run's "-- 1j CLAMP" block.
        cl_mode = cfg.get("innate_clamp_mode", "<absent>")
        cl_frac = cfg.get("innate_clamp_frac", None)
        cl_seed = cfg.get("innate_clamp_seed", None)
        cl_peer = cfg.get("innate_clamp_peer_mode", "<absent>")
        if cl_mode != "bottom":
            bad(f"fixed: innate_clamp_mode={cl_mode!r}, expected 'bottom' "
                f"(the 145 LOWEST-innate agents)")
        if cl_frac is None or abs(float(cl_frac) - CLAMP_FRAC) > 1e-12:
            bad(f"fixed: innate_clamp_frac={cl_frac!r}, expected "
                f"{CLAMP_FRAC:g}")
        if cl_seed is None or int(cl_seed) != info["seed"]:
            bad(f"fixed: innate_clamp_seed={cl_seed!r} != run seed "
                f"{info['seed']} (the cohort seed rides the run seed)")
        if cl_peer != "stubborn":
            bad(f"fixed: innate_clamp_peer_mode={cl_peer!r}, expected "
                f"'stubborn' (the one-sided peer operator, inert at es=0)")
        if bool(cfg.get("sft_exclude_clamped")):
            bad("fixed: sft_exclude_clamped is set -- b0xa source exclusion "
                "is NOT part of this wave (SFT_EXCLUDE_CLAMPED=0)")
        cm = d.get("innate_clamp_mask")
        if not torch.is_tensor(cm) or cm.numel() == 0:
            bad("fixed: innate_clamp_mask missing/empty in trajectory.pt")
        elif cm.dtype != torch.bool or tuple(cm.shape) != (N,):
            bad(f"fixed: innate_clamp_mask dtype/shape {cm.dtype}/"
                f"{tuple(cm.shape)} (want bool [{N}])")
        else:
            cm = cm.bool()
            got_frozen = int(cm.sum())
            if got_frozen != CLAMP_COUNT:
                bad(f"fixed: mask pins {got_frozen} agents, expected exactly "
                    f"{CLAMP_COUNT}")
            if int(d.get("innate_clamp_count", -1)) != got_frozen:
                bad(f"fixed: innate_clamp_count="
                    f"{d.get('innate_clamp_count')!r} != mask sum "
                    f"{got_frozen}")
            for k in ("innate_clamp_mode", "innate_clamp_frac",
                      "innate_clamp_seed"):
                if d.get(k, "<absent>") != cfg.get(k, "<absent>"):
                    bad(f"fixed: trajectory {k}={d.get(k)!r} != config "
                        f"{cfg.get(k)!r}")
            want_hash = gp.innate_clamp_hash(cm)
            if d.get("innate_clamp_hash") != want_hash:
                bad(f"fixed: innate_clamp_hash="
                    f"{str(d.get('innate_clamp_hash'))[:16]!r}... does not "
                    f"match the stored mask ({want_hash[:16]!r}...) -- mask "
                    f"corrupted or tampered")
            # RECONSTRUCT the cohort; never trust the stored mask. Both
            # the shared helper and this file's independent ranking, so a
            # bug in either cannot self-certify.
            if not torch.equal(rec_mask, cm):
                bad(f"fixed: the stored mask is NOT the {CLAMP_COUNT} "
                    f"lowest-innate agents under the deterministic "
                    f"innate-then-id ranking -- "
                    f"{int((rec_mask ^ cm).sum())} agents differ")
            try:
                helper = gp.innate_clamp_mask(inn, "bottom", CLAMP_FRAC,
                                              int(cl_seed or 0))
                if not torch.equal(helper, cm):
                    bad(f"fixed: mask does not reconstruct from (innate, "
                        f"'bottom', {CLAMP_FRAC:g}, {cl_seed!r}) -- "
                        f"{int((helper ^ cm).sum())} agents differ")
            except (ValueError, TypeError) as e:
                bad(f"fixed: mask reconstruction impossible: {e}")
            # BIT-EXACT in BOTH the deployed population and the twin, at
            # every recorded round (the check_pofd_sanity CLAMP assertion)
            if not _bit_eq_rows(op[:, cm], inn[cm]):
                nbad = int((op[:, cm] != inn[cm].unsqueeze(0))
                           .any(dim=0).sum())
                bad(f"fixed: {nbad} pinned agents drift off innate in op_raw "
                    f"(max |diff| "
                    f"{float((op[:, cm] - inn[cm]).abs().max()):.2e}) -- the "
                    f"clamp must be bit-exact")
            if not _bit_eq_rows(tw[:, cm], inn[cm]):
                nbad = int((tw[:, cm] != inn[cm].unsqueeze(0))
                           .any(dim=0).sum())
                bad(f"fixed: {nbad} pinned agents drift off innate in "
                    f"twin_raw (max |diff| "
                    f"{float((tw[:, cm] - inn[cm]).abs().max()):.2e}) -- the "
                    f"clamp holds in the matched twin too")
            # STUBBORN-PEER invariants (check_pofd_sanity: with a live
            # clamp-peer operator the RESPONSIVE twin MOVES, so it is
            # never compared to innate; what must exist is the operator's
            # own per-round telemetry).
            ft = d.get("clamp_fr_touch_raw")
            if not torch.is_tensor(ft) or ft.numel() == 0:
                bad("fixed: clamp_fr_touch_raw missing/empty -- the stubborn "
                    "peer operator records per-round fixed->responsive reach")
            elif tuple(ft.shape) != (want_rounds, N):
                bad(f"fixed: clamp_fr_touch_raw shape {tuple(ft.shape)} != "
                    f"{(want_rounds, N)}")
            elif bool(ft.bool()[:, cm].any()):
                bad("fixed: clamp_fr_touch_raw marks a PINNED agent as "
                    "reached -- the reach mask lives on the responsive "
                    "subset only")
            # the responsive complement must actually be alive
            if int((~cm).sum()) and not bool(
                    (op[:, ~cm] != inn[~cm].unsqueeze(0)).any()):
                bad(f"fixed: not one of the {int((~cm).sum())} responsive "
                    f"agents ever leaves innate -- the clamp was applied "
                    f"beyond its mask")
    else:
        # EVOLVING: MIRRORS check_pofd_sanity's `if is_evo:` section --
        # no clamp, no fixed cohort, and NO clamp key anywhere.
        for k in CLAMP_CFG_KEYS:
            if k in cfg:
                bad(f"evolving: config carries {k}={cfg.get(k)!r} -- a "
                    f"fully-evolving run has no clamp and no fixed agents, "
                    f"so the key must be ABSENT (absent == off, the audit "
                    f"convention)")
        for k in CLAMP_TRAJ_KEYS:
            v = d.get(k)
            present = (v.numel() > 0) if torch.is_tensor(v) else (v is not None)
            if present:
                bad(f"evolving: trajectory carries clamp artifact {k} -- a "
                    f"fully-evolving run must not carry one")

    # --- d8 personal-history artifacts (both conditions) --------------
    # MIRRORS the "d8 PERSONAL-HISTORY replay" preamble of
    # check_pofd_sanity's CLAMP/EVO sections: ICL_K=0 means NO cross-user
    # exemplar may exist, and the rendered personal histories are a
    # mandatory artifact. The byte-level replay of the rendered sentence
    # is left to check_pofd_sanity, which owns it.
    if int(cfg.get("icl_days") or 0) > 0:
        if int(cfg.get("icl_k") or 0) != 0:
            bad("d8: icl_k>0 -- cross-user exemplars are forbidden in the "
                "personal-history arm")
        for k in ("icl_idx_raw", "icl_val_raw"):
            v = d.get(k)
            if torch.is_tensor(v) and v.numel():
                bad(f"d8: {k} non-empty -- cross-user exemplar artifacts "
                    f"must not exist")
        if os.path.exists(os.path.join(run_dir, "icl_ctx_log.json.gz")):
            bad("d8: icl_ctx_log.json.gz present -- no cross-user context "
                "may be rendered")
        if not os.path.exists(os.path.join(run_dir, "icl_days_log.json.gz")):
            bad("d8: icl_days_log.json.gz missing -- the rendered "
                "personal-history contexts are mandatory")

    # --- 7. ZERO PARSE FAILURES ---------------------------------------
    gz = os.path.join(run_dir, "raw_gen_log.json.gz")
    if os.path.exists(gz):
        rec["parse_evidence"] = "raw_gen_log"
        try:
            with gzip.open(gz, "rt") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
        except (OSError, ValueError) as e:
            bad(f"raw_gen_log.json.gz unreadable: {e}")
            rows = []
        if [r.get("round") for r in rows] != list(range(want_rounds)):
            bad(f"raw_gen_log holds rounds "
                f"{[r.get('round') for r in rows][:5]}... (want "
                f"0..{want_rounds - 1})")
        nz = [r for r in rows if float(r.get("parse_fail_frac", 1.0)) != 0.0]
        if nz:
            bad(f"parse failures in {len(nz)} round(s), e.g. round "
                f"{nz[0].get('round')} at parse_fail_frac="
                f"{nz[0].get('parse_fail_frac')!r}")
        short = [r for r in rows if len(r.get("parsed") or []) != N]
        if short:
            bad(f"round {short[0].get('round')} parsed "
                f"{len(short[0].get('parsed') or [])} of {N} agents")
    elif bool(cfg.get("save_raw_gen")) or require_raw_gen:
        rec["parse_evidence"] = "missing"
        bad("raw_gen_log.json.gz missing -- parse_fail_frac is recorded "
            "NOWHERE else, so the parse rate is not establishable"
            + (" (config claims save_raw_gen)"
               if cfg.get("save_raw_gen") else " (--require-raw-gen)"))
    else:
        # SAVE_RAW_GEN is NOT set by the S4G sub templates, so
        # raw_gen_log.json.gz does not exist for this wave and
        # parse_fail_frac cannot be read. The equivalent evidence in
        # trajectory.pt: run_pokec_gated_lm maps an unparsable generation
        # to float("nan") in the served vector, so a parse failure is
        # exactly a non-finite pred_raw entry.
        rec["parse_evidence"] = "pred_raw_nan"
        rec["notes"].append(
            "no raw_gen_log.json.gz (SAVE_RAW_GEN is not set for this wave); "
            "parse failures gated on non-finite pred_raw instead")
        nnan = int((~torch.isfinite(pr)).sum())
        if nnan:
            rounds_bad = torch.nonzero(
                (~torch.isfinite(pr)).any(dim=1)).flatten().tolist()
            bad(f"{nnan} non-finite entries in pred_raw over "
                f"{len(rounds_bad)} round(s), first round {rounds_bad[0]} -- "
                f"an unparsable generation is stored as NaN, so this IS a "
                f"parse failure")

    del d, op, pr, tw, inn
    return rec


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Section-4 corrected-gate wave (pofds4g_) gate; CPU only")
    ap.add_argument("--run-root", default=DEFAULT_RUN_ROOT,
                    help=f"directory holding the run dirs (default the "
                         f"cluster path {DEFAULT_RUN_ROOT})")
    ap.add_argument("--smoke", action="store_true",
                    help=f"gate the {SMOKE_ROUNDS}-round {SMOKE_PREFIX}_ "
                         f"cells (4 jobs: both arms x both conditions at "
                         f"ea{_num(SMOKE_EA)} es{_num(SMOKE_ES)} "
                         f"s{SMOKE_SEED}) under --run-root")
    ap.add_argument("--tags-file", default=None,
                    help="file of tags (one per line, # comments allowed) to "
                         "gate INSTEAD of the full product; coverage is then "
                         "checked against that list")
    ap.add_argument("--require-raw-gen", action="store_true",
                    help="treat a missing raw_gen_log.json.gz as a hard "
                         "failure instead of falling back to the pred_raw "
                         "NaN evidence (this wave does not set SAVE_RAW_GEN)")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the machine-readable verdict here")
    args = ap.parse_args(argv)

    root = Path(args.run_root)
    if not root.is_dir():
        print(f"{LOG} usage error: --run-root {args.run_root!r} is not a "
              f"directory", file=sys.stderr)
        return 2
    scan = SMOKE_SCAN if args.smoke else PROD_SCAN

    # ---- what we EXPECT ------------------------------------------------
    cells = expected_cells(args.smoke)
    tag_of_cell = {c: render_tag(*c, smoke=args.smoke) for c in cells}
    out = []
    if args.tags_file:
        try:
            raw = Path(args.tags_file).read_text().splitlines()
        except OSError as e:
            print(f"{LOG} usage error: --tags-file {args.tags_file!r}: {e}",
                  file=sys.stderr)
            return 2
        wanted = [ln.strip() for ln in raw
                  if ln.strip() and not ln.strip().startswith("#")]
        if not wanted:
            print(f"{LOG} usage error: --tags-file {args.tags_file!r} holds "
                  f"no tags. Nothing to gate is NOT a pass.", file=sys.stderr)
            return 2
        keep, keep_tags = [], []
        for t in wanted:
            info, terrs = parse_tag(t, args.smoke)
            if info is None or terrs:
                for e in (terrs or ["unparseable"]):
                    out.append(f"FAIL --tags-file {t}: {e}")
                continue
            keep.append(info["cell"])
            keep_tags.append(t)
        # the tags file REPLACES the product as the coverage target, so a
        # deliberately partial pull can be gated without the 72-cell
        # completeness check turning every absent cell into noise
        cells = [c for c in expected_cells(args.smoke) if c in set(keep)]
        tag_of_cell = {c: render_tag(*c, smoke=args.smoke) for c in cells}
        run_dirs = [str(root / t) for t in keep_tags]
    else:
        run_dirs = sorted(str(p) for p in root.iterdir()
                          if p.is_dir() and p.name.startswith(scan))
    if not run_dirs:
        for line in out:
            print(f"{LOG} {line}")
        if out:
            print(f"{LOG} FAILED -- every requested tag was rejected before "
                  f"it could be opened.")
            return 1
        print(f"{LOG} usage error: no {scan}* run dirs under {root}. "
              f"Nothing to gate is NOT a pass.", file=sys.stderr)
        return 2

    # ---- gate each run, one trajectory in memory at a time -------------
    recs = []
    for rd in run_dirs:
        if not os.path.isdir(rd):
            out.append(f"FAIL {os.path.basename(rd)}: no such run dir under "
                       f"{root}")
            recs.append({"run_dir": rd, "tag": os.path.basename(rd),
                         "cell": None, "errs": ["run dir does not exist"],
                         "notes": [], "parse_evidence": None,
                         "innate_sha256": None, "cohort_sha256": None,
                         "n_rounds": None, "pop_final_mean": None,
                         "pop_final_sd": None, "op_twin_l1": None,
                         "op_sha256": None})
            continue
        recs.append(check_one(rd, args.smoke, args.require_raw_gen))
    for r in recs:
        for e in r["errs"]:
            out.append(f"FAIL {r['tag']}: {e}")
        for n in r["notes"]:
            out.append(f"NOTE {r['tag']}: {n}")

    # ---- 1. coverage ---------------------------------------------------
    by_cell, dupes = {}, []
    for r in recs:
        c = r["cell"]
        if c is None:
            continue
        if c in by_cell:
            dupes.append((c, by_cell[c]["tag"], r["tag"]))
        by_cell[c] = r
    for c, a, b in dupes:
        out.append(f"FAIL wave: two run dirs claim the same conceptual cell "
                   f"{c}: {a} and {b}")
    extra = sorted(c for c in by_cell if c not in set(cells))
    for c in extra:
        out.append(f"FAIL wave: {by_cell[c]['tag']} parses to cell {c}, which "
                   f"is NOT in the expected grid")
    missing = [c for c in cells if c not in by_cell]
    present = len(cells) - len(missing)

    # ---- 5. cohort pairing + one-world ---------------------------------
    pair_fail = 0
    pairs = sorted({(c[0], c[2], c[3], c[4]) for c in cells})
    for arm, ea, es, seed in pairs:
        fx = by_cell.get((arm, "fixed", ea, es, seed))
        ev = by_cell.get((arm, "evolving", ea, es, seed))
        if fx is None or ev is None:
            continue                      # already a coverage failure
        if fx["innate_sha256"] is None or ev["innate_sha256"] is None:
            continue                      # already a per-cell failure
        if fx["innate_sha256"] != ev["innate_sha256"]:
            out.append(
                f"FAIL pair {arm}/ea{_num(ea)}/es{_num(es)}/s{seed}: the "
                f"fixed and evolving members sit on DIFFERENT innate vectors "
                f"({fx['innate_sha256'][:16]}... vs "
                f"{ev['innate_sha256'][:16]}...) -- {fx['tag']} vs "
                f"{ev['tag']}. The comparison is between one population's "
                f"cohort A being pinned and evolving; two populations make "
                f"it meaningless")
            pair_fail += 1
        elif fx["cohort_sha256"] != ev["cohort_sha256"]:
            out.append(
                f"FAIL pair {arm}/ea{_num(ea)}/es{_num(es)}/s{seed}: same "
                f"innate but DIFFERENT reconstructed bottom-{CLAMP_COUNT} "
                f"cohort ({fx['cohort_sha256'][:16]}... vs "
                f"{ev['cohort_sha256'][:16]}...)")
            pair_fail += 1
    inn_shas = {r["innate_sha256"] for r in recs
                if r["innate_sha256"] is not None}
    if len(inn_shas) > 1:
        out.append(
            f"FAIL wave: {len(inn_shas)} distinct innate vectors across the "
            f"grid. load_movielens_setup is a pure function of (dataset, "
            f"target) here -- PROFILE_SHUFFLE_P=0 and no routing treatment -- "
            f"so innate does not even depend on the run seed. A difference "
            f"means a different agent set or a different agent ORDER.")
        for r in recs:
            if r["innate_sha256"]:
                out.append(f"     {r['tag']}: {r['innate_sha256'][:16]}...")

    # ---- 5b. SEED-DISTINCTNESS ------------------------------------------
    # innate, the 10-NN graph and the bottom-145 cohort are SEED-INVARIANT
    # for movielens (load_movielens_setup takes no seed;
    # gp.innate_clamp_mask("bottom") is a pure sort with no RNG draw), so
    # config["seed"] -- written from the environment, never observed -- is
    # the only per-run evidence that a seed reached the runner. The
    # BEHAVIOURAL evidence is that two seeds of one cell must not produce
    # the same trajectory. Compared by the op_raw sha256 taken in
    # check_one, so no two runs' tensors are ever resident at once.
    seed_fail = 0
    seed_groups = {}
    for c, r in by_cell.items():
        if r.get("op_sha256") is None:
            continue                      # already a per-cell failure
        seed_groups.setdefault((c[0], c[1], c[2], c[3]), []).append(
            (c[4], r["tag"], r["op_sha256"]))
    for g, members in sorted(seed_groups.items()):
        if len(members) < 2:
            # a missing seed is COVERAGE (already fatal above), not a pass
            # or a failure of this check
            continue
        by_sha = {}
        for seed, tag, sha in sorted(members):
            by_sha.setdefault(sha, []).append((seed, tag))
        for sha, hits in by_sha.items():
            if len(hits) < 2:
                continue
            out.append(
                f"FAIL seed-distinctness {g[0]}/{g[1]}/ea{_num(g[2])}/"
                f"es{_num(g[3])}: seeds {[h[0] for h in hits]} produced a "
                f"BIT-IDENTICAL op_raw (sha256 {sha[:16]}...) -- "
                f"{', '.join(h[1] for h in hits)}. Nothing about this world "
                f"depends on the seed (innate, the 10-NN graph and the "
                f"cohort are seed-invariant), so config['seed'] -- written "
                f"from the environment, not observed -- is the only other "
                f"evidence a seed reached the runner. Identical trajectories "
                f"mean it never reached the training/serving stream, which "
                f"collapses the three-seed intervals to ONE observation "
                f"while every per-run field still looks correct.")
            seed_fail += 1

    # ---- print ---------------------------------------------------------
    for line in out:
        print(f"{LOG} {line}")

    hdr = (f"{'cell':<66} {'verdict':>7} {'rounds':>6} {'popMean':>8} "
           f"{'popSD':>7} {'opTwinL1':>9} {'parse':>13}")
    print("\n" + "=" * len(hdr))
    print(f"PER-CELL REPORT -- {'SMOKE' if args.smoke else 'PRODUCTION'} "
          f"grid, {present}/{len(cells)} cells present")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        r = by_cell.get(c)
        name = tag_of_cell[c]
        if r is None:
            print(f"{name:<66} {'ABSENT':>7} {'-':>6} {'-':>8} {'-':>7} "
                  f"{'-':>9} {'-':>13}")
            continue
        print(f"{r['tag']:<66} "
              f"{'PASS' if not r['errs'] else 'FAIL':>7} "
              f"{(r['n_rounds'] if r['n_rounds'] is not None else -1):>6} "
              f"{(r['pop_final_mean'] if r['pop_final_mean'] is not None else float('nan')):>8.4f} "
              f"{(r['pop_final_sd'] if r['pop_final_sd'] is not None else float('nan')):>7.4f} "
              f"{(r['op_twin_l1'] if r['op_twin_l1'] is not None else float('nan')):>9.4f} "
              f"{str(r['parse_evidence']):>13}")
    for c in extra:
        r = by_cell[c]
        print(f"{r['tag']:<66} {'EXTRA':>7} {'-':>6} {'-':>8} {'-':>7} "
              f"{'-':>9} {str(r['parse_evidence']):>13}")

    fallbacks = [r["tag"] for r in recs
                 if r["parse_evidence"] == "pred_raw_nan"]
    if fallbacks:
        print(f"\n{LOG} PARSE-RATE EVIDENCE: {len(fallbacks)} cell(s) have no "
              f"raw_gen_log.json.gz (the S4G sub templates do not set "
              f"SAVE_RAW_GEN=1, and parse_fail_frac lives nowhere else), so "
              f"zero-parse-failure was gated on non-finite pred_raw -- the "
              f"same event, since an unparsable generation is stored as NaN. "
              f"Pass --require-raw-gen to make the absence itself fatal.")

    print("\n" + "=" * len(hdr))
    if missing:
        print(f"GRID COMPLETENESS: {present} of {len(cells)} cells present -- "
              f"{len(missing)} ABSENT. A silently short grid must not look "
              f"like a complete result.")
        for c in missing:
            print(f"  ABSENT  arm={c[0]:<3} cond={c[1]:<8} ea={c[2]:<4g} "
                  f"es={c[3]:<4g} seed={c[4]:<3} expected tag "
                  f"{tag_of_cell[c]}")
    else:
        print(f"GRID COMPLETENESS: all {len(cells)} cells present")
    print("=" * len(hdr))

    n_fail_cells = sum(1 for r in recs if r["errs"])
    allok = (n_fail_cells == 0 and not missing and not dupes and not extra
             and seed_fail == 0
             and pair_fail == 0 and len(inn_shas) <= 1
             and not any(l.startswith("FAIL") for l in out))
    verdict = {
        "wave": "section4_gate_anch2",
        "smoke": bool(args.smoke),
        "run_root": str(root),
        "operator_required": WANT_MARKER,
        "ai_gate_reference_required": WANT_GATE_REF,
        "n_runs": len(recs),
        "n_cells_present": present,
        "n_cells_total": len(cells),
        "n_cells_failed": n_fail_cells,
        "n_pair_failures": pair_fail,
        "n_seed_distinctness_failures": seed_fail,
        "missing": [{"arm": c[0], "cond": c[1], "eps_ai": c[2],
                     "eps_social": c[3], "seed": c[4],
                     "expected_tag": tag_of_cell[c]} for c in missing],
        "duplicate_cells": [{"cell": list(c), "tags": [a, b]}
                            for c, a, b in dupes],
        "unexpected_cells": [{"cell": list(c), "tag": by_cell[c]["tag"]}
                             for c in extra],
        "innate_sha256_distinct": sorted(inn_shas),
        "parse_evidence_fallback": fallbacks,
        "pass": bool(allok),
        "cells": [{"tag": r["tag"], "run_dir": r["run_dir"],
                   "cell": list(r["cell"]) if r["cell"] else None,
                   "ok": not r["errs"], "errors": r["errs"],
                   "notes": r["notes"],
                   "parse_evidence": r["parse_evidence"],
                   "n_rounds": r["n_rounds"],
                   "innate_sha256": r["innate_sha256"],
                   "cohort_sha256": r["cohort_sha256"],
                   "pop_final_mean": r["pop_final_mean"],
                   "pop_final_sd": r["pop_final_sd"],
                   "op_twin_l1": r["op_twin_l1"],
                   "op_sha256": r["op_sha256"]} for r in recs],
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(verdict, indent=2))
        print(f"{LOG} verdict -> {args.json_out}")

    if allok:
        print(f"{LOG} PASS -- {len(recs)} run(s), {present}/{len(cells)} "
              f"cells: every tag carries {OP_INFIX!r} and every config the "
              f"{WANT_MARKER} operator at ai_gate_reference=anchor; grid "
              f"fields agree with the tags in both directions; the "
              f"bottom-{CLAMP_COUNT} cohort reconstructs and is bit-exact in "
              f"population and twin on every fixed cell; every evolving cell "
              f"is clamp-free; each fixed/evolving pair shares one innate "
              f"vector and one cohort; no two seeds of a cell share a "
              f"trajectory (seed-distinctness); twins are "
              f"non-degenerate; zero parse failures.")
        return 0
    print(f"{LOG} FAILED -- {n_fail_cells} of {len(recs)} run(s) failed, "
          f"{len(missing)} cell(s) absent, {len(dupes)} duplicate, "
          f"{len(extra)} unexpected, {pair_fail} pair mismatch(es), "
          f"{seed_fail} seed-distinctness collision(s). See the FAIL "
          f"lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
