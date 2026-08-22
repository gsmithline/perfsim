#!/usr/bin/env python3
"""HARD GATE for the Jiduan Wu / Pokec replication (pofdwu_ family).

Run it on every pull, BEFORE the analyzer, and never analyse a run it
refuses:

    python3 experiments/scripts/cluster_pipelines/check_jiduan_pokec.py \\
        runs/pokec_gated_lm/pofdwu_*

WHY THIS IS ITS OWN FILE AND NOT A SECTION OF check_pofd_sanity.py.
Every earlier wave shares one dataset (MovieLens, 723 agents, everyone
labelled) and one homogeneous FJ parameterization. This wave shares
neither. It runs on Pokec's own 2163-node LCC, splits the population
into an OBSERVED set O (the FIRST 1730 rows, y_label2163.pk) and a
HELD-OUT set U (the LAST 433, y_unlabel_label2163.pk), and takes alpha
and beta PER AGENT from the dataset. The failure modes that matter here
-- a held-out opinion reaching the optimizer, a model prediction
overwriting an observed value, held-out truth appearing in a prompt --
simply do not exist in the other waves, and the checks for them do not
belong bolted onto a function that gates 800 MovieLens runs.

WHAT THE OPERATOR IS SUPPOSED TO BE (FJ_UPDATE_VERSION=wu1):

    served^(t)_i   = x^(t)_i                    for i in O   (passthrough)
                   = m^(t)_i                    for i in U   (the model)
    x_init^(t)_i   = (1 - beta_i) x^innate_i + beta_i served^(t)_i
    u^(0)          = x_init^(t)
    u^(l+1)_i      = (1 - alpha_i) x_init^(t)_i
                     + alpha_i (P u^(l))_i,     l = 0 .. K-1
    x^(t+1)        = u^(K)

alpha_i is PEER SUSCEPTIBILITY, beta_i PLATFORM SUSCEPTIBILITY, both
scaled by FJ_ALPHA_SCALE / FJ_BETA_SCALE. The human component is the raw
innate opinion with no carryover (k = 1), so the ONLY channel between
outer rounds is the platform.

NOTHING BELOW TRUSTS A CONFIG FIELD ON ITS OWN. The declared recurrence
is REPLAYED -- all K inner steps, every round, from the repo's own copy
of the Pokec graph and parameter vectors -- and the config is only ever
used to say what should have been replayed.

THE ALPHA TRAP THIS DATASET WALKS STRAIGHT INTO. The dataset file is
called hetero_peer_sus2163.pkl and holds alpha (mean .8909), but
FJWorld.peer_sus is STUBBORNNESS = 1 - alpha. Feeding the file through
unchanged runs peer susceptibility .1091 -- near-opposite dynamics --
and every downstream number stays finite, ordered and plausible. So the
inverted convention is replayed TOO, and a run whose trajectory matches
it is refused by name.

u^(1) vs u^(K). At alpha ~ .89 and K = 100 the inner loop CONVERGES
(contraction ~ .89^100), so u^(K) has forgotten where it started and the
final-state replay cannot falsify a stale-state initialisation. u^(1) is
affine in u^(0) with coefficient alpha, so it still carries it. That is
the entire reason fj_u1_raw exists, and a run without it is refused.

TWO THINGS THAT ARE NOT THE SAME AND MUST NOT BE CONFLATED: the INNER FJ
loop converges by construction at this alpha and K; whether the OUTER
model-population loop has settled is an empirical question this checker
does not answer (the analyzer does).

THE ARTIFACT CONTRACT. Every field below is REQUIRED, and a missing one
is a hard failure that names the field rather than a crash:

  config   dataset, pop_model, fj_update_version, fj_inner_steps,
           n_rounds, fj_peer_source, fj_platform_source, fj_alpha_scale,
           fj_beta_scale, fj_observed_passthrough, fj_alpha_raw_sha256,
           fj_beta_raw_sha256, fj_alpha_realized_sha256,
           fj_beta_realized_sha256, fj_alpha_realized_mean,
           fj_beta_realized_mean, wu_icl_mode, wu_icl_k, wu_icl_d,
           n_labeled, training_style, kl_beta, use_lora,
           fresh_each_round, serve_eval_mode, hardware
           routing runs additionally: routing_treat_frac,
           routing_treat_seed, routing_treat_value, and (treatment only)
           routing_treat_idx_sha256 + routing_treat_n
  tensors  op_raw, model_pred_raw, served_raw, fj_x_init_raw, fj_u1_raw,
           observed_mask, train_idx_raw, train_y_raw, innate
  file     wu_ctx_log.json.gz for every run whose ICL mode is not "none";
           gzip JSONL, one wu_context.round_log_line per round.

model_pred_raw MAY BE NaN ON O and only there: under passthrough the
model is never asked about an observed agent, and that absence has to be
representable or "not asked" and "answered exactly x_O" become the same
artifact. An infinity anywhere is always a failure.

ROUTING IS A SOURCE INJECTION AT OBSERVED AGENTS. The runner draws the
cohort from the OBSERVED pool and rewrites those agents' INNATE opinion
before anything reads it, so the treatment propagates consistently into
x(0), the FJ anchor, the round-0 SFT labels and the served vector. The
cohort is a function of (ROUTING_TREAT_SEED, frac, |O|) and NOT of the
run seed, so it is RECOMPUTED here rather than read; the CONTROL twin
runs at frac 0 and carries no cohort at all, and the pair is proved by
differencing the two innate vectors against that recomputed set.

PASSTHROUGH SEMANTICS. "served_O = x_O" has two defensible readings and
they are NOT interchangeable: x_O can be the LIVE opinion the observed
agent holds entering the round (innate at t=0, the previous post-FJ
state after that), or the STATIC observation recorded once at t=0.
`live` is the default because the platform in Wu's model observes the
population it is currently serving. The alternative is available as
--passthrough innate, and a run that matches NEITHER is refused with
both distances printed, so the failure says which reading it is closest
to instead of leaving the reader to guess.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import re
import sys
from pathlib import Path

import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
REPO = HERE.parent.parent.parent

# wu_context.py is the ONE home for what a context mode means, how a
# context entry is audited, how the served vector is assembled and how a
# routing cohort is hashed. The runner uses it, its tests use it, and so
# does this checker -- a second copy of "which mode is the extension"
# would be a second answer.
_wuc_spec = importlib.util.spec_from_file_location(
    "wu_context_chk", str(HERE / "wu_context.py"))
wuc = importlib.util.module_from_spec(_wuc_spec)
_wuc_spec.loader.exec_module(wuc)

# ---- the Pokec environment, pinned -------------------------------------
WU_N = 2163                 # LCC size
WU_N_OBSERVED = 1730        # O -- the FIRST rows (y_label2163.pk)
WU_N_HELDOUT = 433          # U -- the LAST rows (y_unlabel_label2163.pk)
WU_GRAPH_NODES = 2163
WU_GRAPH_EDGES = 2346
WU_INNER = 100              # K_FJ
WU_ROUNDS = 50              # T, outer rounds
WU_H100 = "NVIDIA H100 80GB HBM3"
# sha256 over the float32 bytes of the vectors the runner builds, in the
# runner's own row order (y_label followed by y_unlabel; the graph
# node-ordered by profiles["user_id"]). Computed from examples/pokec on
# 2026-08-22. These are the "wrong dataset / wrong node order / wrong
# vector" gate: any of the three defects moves at least one of them.
WU_INNATE_SHA = (
    "c89ac8333ee364be65227af5bcda5af88f56c4324528dc09c682ab6592070331")
WU_ALPHA_RAW_SHA = (
    "87668d2b18a56b2bad04a2862b9fc11a80205efc59013dd8caf2a33500f36860")
WU_BETA_RAW_SHA = (
    "b51329df7e2d5aebda1377de6424fc8bc74f1d8e9260943948e394f14fe3d7d0")
WU_GRAPH_SHA = (
    "8fc093ababf781bcbbf7712598360cbcbee0c5200aa60e0140a96974a820844f")
WU_ALPHA_RAW_MEAN = 0.8908604383      # peer susceptibility, NOT stubbornness
WU_BETA_RAW_MEAN = 0.8890184760       # platform susceptibility
WU_POKEC_DIR = REPO / "examples" / "pokec"

# float32 accumulation over K=100 inner steps x 50 outer rounds; the
# operator is a contraction, so error does not compound across rounds
WU_REPLAY_TOL = 2e-5
WU_EXACT_TOL = 1e-6         # for identities that are copies, not blends
WU_MEAN_TOL = 1e-6

WU_PASSTHROUGH_MODES = ("live", "innate")
WU_PASSTHROUGH_DEFAULT = "live"

# ---- arm semantics -----------------------------------------------------
# Deliberately RESTATED here rather than imported from gen_pofd_sweep.py:
# the generator and the checker are meant to be independent witnesses to
# what an arm token means, and tests/test_jiduan_pokec_infra.py asserts
# the two tables agree. Importing would make a generator typo invisible.
#   arm -> (training_style, kl_beta, wu_icl_mode, wu_icl_k, wu_icl_d)
WU_ARM_SEMANTICS = {
    "b0": ("sft", 0.0, "none", 0, 0),
    "b0p1": ("sft_kl", 0.1, "none", 0, 0),
    "b0p5": ("sft_kl", 0.5, "none", 0, 0),
    "b1": ("sft_kl", 1.0, "none", 0, 0),
    "b10": ("sft_kl", 10.0, "none", 0, 0),
    "frz": ("frozen", 0.0, "none", 0, 0),
    "octx8": ("frozen", 0.0, "observed_context", 8, 0),
    "phist8": ("frozen", 0.0, "prediction_history", 0, 8),
    "ehist8": ("frozen", 0.0, "expressed_history", 0, 8),
}
WU_TRAINED_ARMS = ("b0", "b0p1", "b0p5", "b1", "b10")
# STRICT vs EXTENSION is wu_context's call, not this file's: a mechanism
# is an EXTENSION when it shows the model something Wu's platform cannot
# observe. The platform has its own past OUTPUTS (prediction_history is
# strict); it does not observe held-out agents' realised opinions
# (expressed_history is the extension).
WU_STRICT_ICL_MODES = tuple(wuc.STRICT_MODES)
WU_EXTENSION_ICL_MODES = tuple(wuc.EXTENSION_MODES)
# what the displayed numbers ARE, per mode -- the artifact field the
# history is checked against
WU_HISTORY_TENSOR = {"prediction_history": "served_raw",
                     "expressed_history": "op_raw"}

WU_REQUIRED_CFG = (
    "dataset", "pop_model", "fj_update_version", "fj_inner_steps",
    "n_rounds", "fj_peer_source", "fj_platform_source", "fj_alpha_scale",
    "fj_beta_scale", "fj_observed_passthrough", "fj_alpha_raw_sha256",
    "fj_beta_raw_sha256", "fj_alpha_realized_sha256",
    "fj_beta_realized_sha256", "fj_alpha_realized_mean",
    "fj_beta_realized_mean", "wu_icl_mode", "wu_icl_k", "wu_icl_d",
    "n_labeled", "training_style", "kl_beta", "use_lora",
    "fresh_each_round", "serve_eval_mode", "hardware",
)
WU_REQUIRED_ART = (
    "op_raw", "model_pred_raw", "served_raw", "fj_x_init_raw",
    "fj_u1_raw", "observed_mask", "train_idx_raw", "train_y_raw",
    "innate",
)
WU_ROUTING_CFG = ("routing_treat_frac", "routing_treat_seed",
                  "routing_treat_value")

# the runner's dedicated cohort RNG offset. The cohort is a function of
# (ROUTING_TREAT_SEED, frac, |O|) and NOT of the run seed, which is what
# makes it recomputable here and reusable across twins.
WU_ROUTE_COHORT_STREAM = 611_000


# ------------------------------------------------------------------ utils

def _sha_t(t):
    a = torch.as_tensor(t).detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def _f(t):
    return torch.as_tensor(t).detach().cpu().float()


def arm_of(tag):
    """The arm token, read with the `_<arm>_pa` delimiter. A positional
    split cannot work: model slugs contain underscores (qwen3_8b), and a
    bare prefix match would make b1 swallow b10."""
    hits = [a for a in WU_ARM_SEMANTICS if f"_{a}_pa" in tag]
    return hits[0] if len(hits) == 1 else None


def route_side_of(tag):
    if "_rtT_" in tag:
        return "T"
    if "_rtC_" in tag:
        return "C"
    return None


def rounds_of(tag):
    m = re.search(r"_r(\d+)(?:smoke)?$", tag)
    return int(m.group(1)) if m else None


def seed_of(tag):
    m = re.search(r"_s(\d+)_r\d+", tag)
    return int(m.group(1)) if m else None


def twin_stem(tag):
    """The tag with the routing side removed -- treatment and control
    collapse onto the same stem, which is how the pair is found."""
    return tag.replace("_rtT_", "_rt_").replace("_rtC_", "_rt_")


# ---- the environment the replay runs against ---------------------------
# A module-level cache so tests can inject a toy environment (4 agents, a
# hand-built ring) without touching the repo dataset, exactly the idiom
# check_pofd_sanity.py uses for the MovieLens graph.
_WU_ENV_CACHE = {}


def wu_env(pokec_dir=None):
    """innate, alpha_raw, beta_raw and the row-normalized graph, rebuilt
    from the repo so the replay never depends on the run having saved
    them. Returns None when the dataset is not reachable -- the caller
    turns that into a hard failure rather than a silent skip."""
    if "env" in _WU_ENV_CACHE:
        return _WU_ENV_CACHE["env"]
    try:
        import networkx as nx
        import numpy as np
        sys.path.insert(0, str(REPO))
        from perfsim.environments.dynamics import normalize_adjacency
        p = Path(pokec_dir or WU_POKEC_DIR)
        with open(p / "lcc_profiles_relation_to_smoking.pk", "rb") as fh:
            df = pickle.load(fh)
        with open(p / "lcc_graph_relation_to_smoking.pk", "rb") as fh:
            graph = pickle.load(fh)
        pp = p / "parametric_params"
        with open(pp / "y_label2163.pk", "rb") as fh:
            y_lab = pickle.load(fh)
        with open(pp / "y_unlabel_label2163.pk", "rb") as fh:
            y_unlab = pickle.load(fh)
        with open(pp / "hetero_peer_sus2163.pkl", "rb") as fh:
            alpha = pickle.load(fh)
        with open(pp / "hetero_platform_sus2163.pkl", "rb") as fh:
            beta = pickle.load(fh)
        adj = torch.tensor(
            nx.to_numpy_array(graph, nodelist=df["user_id"].tolist()),
            dtype=torch.float32)
        env = {
            "innate": torch.tensor(
                np.asarray(list(y_lab) + list(y_unlab), dtype=np.float64),
                dtype=torch.float32),
            "alpha_raw": torch.tensor(np.asarray(alpha), dtype=torch.float32),
            "beta_raw": torch.tensor(np.asarray(beta), dtype=torch.float32),
            "W": normalize_adjacency(adj).float(),
            "n_nodes": int(graph.number_of_nodes()),
            "n_edges": int(graph.number_of_edges()),
            "n_observed": len(y_lab),
            "n_heldout": len(y_unlab),
        }
    except Exception:                                   # pragma: no cover
        env = None
    _WU_ENV_CACHE["env"] = env
    return env


# ------------------------------------------------------------- the checks

def check_jiduan_run(name, d, *, expect_rounds=WU_ROUNDS, expect_n=WU_N,
                     expect_observed=WU_N_OBSERVED,
                     expect_heldout=WU_N_HELDOUT, expect_inner=WU_INNER,
                     env=None, passthrough=WU_PASSTHROUGH_DEFAULT,
                     expect_peer_source="dataset",
                     expect_platform_source="dataset", pin_pokec=None):
    """Return a list of failure strings; empty means the run may be
    analysed. `expect_*` are parameters so tests can run a 4-agent toy
    while production keeps 2163 / 1730 / 433. `pin_pokec` defaults to
    "is this the production size", which is what decides whether the
    canonical Pokec hashes apply."""
    errs = []
    if passthrough not in WU_PASSTHROUGH_MODES:
        return [f"WU {name}: unknown passthrough mode {passthrough!r}"]
    if pin_pokec is None:
        pin_pokec = (expect_n == WU_N and expect_observed == WU_N_OBSERVED)

    cfg = d.get("config")
    if not isinstance(cfg, dict):
        return [f"WU {name}: no config dict in the artifact"]

    # -- (0) EVERY REQUIRED FIELD IS PRESENT, NAMED ONE BY ONE.
    # A KeyError three checks later tells the reader nothing; a list of
    # the fields the runner did not write tells them exactly what to fix.
    missing_cfg = [k for k in WU_REQUIRED_CFG if k not in cfg]
    if missing_cfg:
        errs.append(f"WU {name}: config is missing required field(s): "
                    f"{', '.join(missing_cfg)}")
    missing_art = [k for k in WU_REQUIRED_ART
                   if d.get(k) is None
                   or (torch.is_tensor(d.get(k)) and d[k].numel() == 0)]
    if "fj_u1_raw" in missing_art:
        errs.append(
            f"WU {name}: no fj_u1_raw. At alpha ~ {WU_ALPHA_RAW_MEAN:.3f} "
            f"and K = {expect_inner} the inner loop CONVERGES, so u^(K) "
            f"has forgotten its initialisation and no final-state replay "
            f"can prove u^(0) = x_init. u^(1) is the only witness; "
            f"without it the run is unverifiable, not merely incomplete")
    other_missing = [k for k in missing_art if k != "fj_u1_raw"]
    if other_missing:
        errs.append(f"WU {name}: artifact is missing required tensor(s): "
                    f"{', '.join(other_missing)}")
    if missing_cfg or missing_art:
        return errs          # nothing below can be trusted without them

    # -- (1) THE ENVIRONMENT: right dataset, right node order, right
    #        vectors. Any of the three defects moves a hash.
    env = env if env is not None else wu_env()
    if env is None:
        return errs + [f"WU {name}: the Pokec dataset is not reachable at "
                       f"{WU_POKEC_DIR}, so the recurrence cannot be "
                       f"replayed and nothing here can be verified"]
    innate = _f(env["innate"])
    alpha_raw = _f(env["alpha_raw"])
    beta_raw = _f(env["beta_raw"])
    W = _f(env["W"])
    n = innate.shape[0]

    if cfg.get("dataset") != "pokec":
        errs.append(f"WU {name}: dataset={cfg.get('dataset')!r}, expected "
                    f"'pokec' -- this wave is Wu's own dataset or nothing")
    if pin_pokec:
        for label, got, want in (
                ("innate", _sha_t(innate), WU_INNATE_SHA),
                ("alpha_raw", _sha_t(alpha_raw), WU_ALPHA_RAW_SHA),
                ("beta_raw", _sha_t(beta_raw), WU_BETA_RAW_SHA),
                ("graph", _sha_t(W), WU_GRAPH_SHA)):
            if got != want:
                errs.append(
                    f"WU {name}: the replay {label} vector hashes {got[:16]}"
                    f"..., not the canonical Pokec {want[:16]}... -- wrong "
                    f"dataset, wrong node order or a changed file")
        if (env.get("n_nodes"), env.get("n_edges")) != (WU_GRAPH_NODES,
                                                        WU_GRAPH_EDGES):
            errs.append(f"WU {name}: graph is {env.get('n_nodes')} nodes / "
                        f"{env.get('n_edges')} edges, expected "
                        f"{WU_GRAPH_NODES}/{WU_GRAPH_EDGES}")
    if n != expect_n:
        errs.append(f"WU {name}: the environment has {n} agents, expected "
                    f"{expect_n}")
        return errs
    if expect_observed + expect_heldout != expect_n:
        errs.append(f"WU {name}: |O| {expect_observed} + |U| "
                    f"{expect_heldout} != N {expect_n}")
        return errs
    # THE RUN'S OWN INNATE IS WHAT ANCHORED IT, and it equals the dataset
    # vector everywhere EXCEPT a routed cohort (the routing treatment
    # rewrites innate at the injected agents). The routing section proves
    # that difference is exactly the cohort; here the untreated case is
    # required to be bit-identical.
    innate_run = _f(d["innate"])
    if innate_run.shape != innate.shape:
        errs.append(f"WU {name}: the run's innate has "
                    f"{tuple(innate_run.shape)} entries, the dataset "
                    f"{tuple(innate.shape)}")
        return errs
    if route_side_of(name) != "T" and _sha_t(innate_run) != _sha_t(innate):
        errs.append(f"WU {name}: the run's innate vector is not the Pokec "
                    f"innate vector this replay uses -- different data, a "
                    f"different row order, or an undeclared intervention")

    # -- (2) THE OBSERVED / HELD-OUT SPLIT IS THE DATASET'S OWN.
    # O is a PREFIX by construction (innate = y_label ++ y_unlabel), so a
    # mask that is the right size but the wrong set is still wrong.
    obs = d["observed_mask"]
    obs = obs.bool() if torch.is_tensor(obs) else torch.as_tensor(obs).bool()
    if obs.shape != (n,):
        errs.append(f"WU {name}: observed_mask {tuple(obs.shape)} != ({n},)")
        return errs
    n_obs = int(obs.sum())
    if n_obs != expect_observed:
        errs.append(f"WU {name}: |O| = {n_obs}, expected {expect_observed}")
    if not bool(obs[:expect_observed].all()) or bool(obs[expect_observed:].any()):
        errs.append(f"WU {name}: observed_mask is not the first "
                    f"{expect_observed} rows -- the O/U split does not "
                    f"follow the dataset's own y_label / y_unlabel order")
    if int((~obs).sum()) != expect_heldout:
        errs.append(f"WU {name}: |U| = {int((~obs).sum())}, expected "
                    f"{expect_heldout}")
    O = obs
    U = ~obs

    # -- (3) THE OPERATOR IS THE OPT-IN ONE, AT THE DECLARED K AND T.
    if cfg.get("fj_update_version") != "wu1":
        errs.append(f"WU {name}: fj_update_version="
                    f"{cfg.get('fj_update_version')!r} -- the LEGACY FJ "
                    f"operator starts its inner loop from the previous "
                    f"population and never applies W_PLAT; it is not this "
                    f"recurrence and its numbers are not comparable")
    if cfg.get("pop_model") != "fj":
        errs.append(f"WU {name}: pop_model={cfg.get('pop_model')!r}")
    n_inner = int(cfg.get("fj_inner_steps", -1))
    if n_inner != expect_inner:
        errs.append(f"WU {name}: fj_inner_steps={n_inner}, expected K = "
                    f"{expect_inner}")

    op = _f(d["op_raw"])
    pred = _f(d["model_pred_raw"])
    served = _f(d["served_raw"])
    x_init_saved = _f(d["fj_x_init_raw"])
    u1_saved = _f(d["fj_u1_raw"])
    want_shape = (expect_rounds, n)
    bad_shape = [nm for nm, t in (("op_raw", op), ("model_pred_raw", pred),
                                  ("served_raw", served),
                                  ("fj_x_init_raw", x_init_saved),
                                  ("fj_u1_raw", u1_saved))
                 if tuple(t.shape) != want_shape]
    if bad_shape:
        errs.append(f"WU {name}: wrong prediction/state lengths -- "
                    + ", ".join(f"{nm} {tuple(d[nm].shape)}"
                                for nm in bad_shape)
                    + f"; all must be {want_shape}")
        return errs
    if int(cfg.get("n_rounds", -1)) != expect_rounds:
        errs.append(f"WU {name}: config n_rounds={cfg.get('n_rounds')} but "
                    f"the horizon under test is {expect_rounds}")

    # -- (4) SERVING HYGIENE: finite, in range, parsed, eval mode.
    # NaN ON THE OBSERVED SET IS LEGAL IN model_pred_raw AND ONLY THERE.
    # Under passthrough the model is never asked about an observed agent,
    # and that absence has to be REPRESENTABLE -- otherwise "the model
    # was not asked" and "the model happened to answer x_O" are the same
    # artifact. Everywhere else, and everywhere in every other tensor,
    # non-finite is a failure.
    for nm, t in (("served_raw", served), ("op_raw", op),
                  ("fj_x_init_raw", x_init_saved),
                  ("fj_u1_raw", u1_saved)):
        if not torch.isfinite(t).all():
            errs.append(f"WU {name}: {nm} has non-finite entries")
    pred_u = pred[:, U]
    if not torch.isfinite(pred_u).all():
        errs.append(f"WU {name}: model_pred_raw has non-finite entries on "
                    f"the HELD-OUT set, which is the only set the model "
                    f"is asked about")
    elif float(pred_u.min()) < 0.0 or float(pred_u.max()) > 1.0:
        errs.append(f"WU {name}: model_pred_raw leaves [0, 1] on the "
                    f"held-out set")
    pred_o = pred[:, O]
    bad_o = torch.isinf(pred_o) | ((~torch.isnan(pred_o))
                                   & ((pred_o < 0.0) | (pred_o > 1.0)))
    if bool(bad_o.any()):
        errs.append(f"WU {name}: model_pred_raw on the observed set is "
                    f"neither a value in [0, 1] nor NaN -- an infinity "
                    f"there is a serving failure, not an 'unasked' marker")
    if torch.isfinite(op).all() and (float(op.min()) < 0.0
                                     or float(op.max()) > 1.0):
        errs.append(f"WU {name}: op_raw leaves [0, 1]")
    if cfg.get("serve_eval_mode") is not True:
        errs.append(f"WU {name}: serve_eval_mode="
                    f"{cfg.get('serve_eval_mode')!r} -- serving must run "
                    f"in eval mode or greedy decoding is not reproducible")
    for row in (d.get("trajectory") or []):
        pf = row.get("parse_fail")
        if pf is not None and float(pf) != 0.0:
            errs.append(f"WU {name}: round {row.get('t', row.get('round'))} "
                        f"parse_fail={pf} -- an unparsed generation becomes "
                        f"a silent constant and drives the population")
            break
    gpu = (cfg.get("hardware") or {}).get("gpu_name")
    if pin_pokec and gpu != WU_H100:
        errs.append(f"WU {name}: gpu {gpu!r} != {WU_H100!r}; greedy "
                    f"decoding is only bit-reproducible within one "
                    f"architecture")

    # -- (5) NO SCALAR alpha OR beta IN A DATASET-SOURCED RUN.
    # A scalar sitting next to a per-agent source is exactly the
    # ambiguity this wave exists to remove: whichever one the operator
    # picked, the artifact would look well-formed.
    if cfg.get("fj_peer_source") != expect_peer_source:
        errs.append(f"WU {name}: fj_peer_source="
                    f"{cfg.get('fj_peer_source')!r}, expected "
                    f"{expect_peer_source!r}")
    if cfg.get("fj_platform_source") != expect_platform_source:
        errs.append(f"WU {name}: fj_platform_source="
                    f"{cfg.get('fj_platform_source')!r}, expected "
                    f"{expect_platform_source!r}")
    c_alpha = float(cfg.get("fj_alpha_scale", float("nan")))
    c_beta = float(cfg.get("fj_beta_scale", float("nan")))
    alpha = alpha_raw * c_alpha
    beta = beta_raw * c_beta
    if expect_peer_source == "dataset" and c_alpha > 0:
        if int(torch.unique(alpha).numel()) <= 1:
            errs.append(f"WU {name}: the realized alpha is CONSTANT across "
                        f"agents under fj_peer_source='dataset' -- a "
                        f"scalar peer susceptibility in the exact "
                        f"heterogeneous key")
        for scalar_field in ("fj_peer_alpha", "fj_alpha"):
            v = cfg.get(scalar_field)
            if v is not None:
                errs.append(f"WU {name}: {scalar_field}={v} is recorded "
                            f"alongside fj_peer_source='dataset' -- two "
                            f"sources for one parameter, and the artifact "
                            f"cannot say which one ran")
    if expect_platform_source == "dataset" and c_beta > 0:
        if int(torch.unique(beta).numel()) <= 1:
            errs.append(f"WU {name}: the realized beta is CONSTANT across "
                        f"agents under fj_platform_source='dataset' -- a "
                        f"scalar platform susceptibility in the exact "
                        f"heterogeneous key")
        for scalar_field in ("fj_beta", "fj_platform_beta"):
            v = cfg.get(scalar_field)
            if v is not None:
                errs.append(f"WU {name}: {scalar_field}={v} is recorded "
                            f"alongside fj_platform_source='dataset'")

    # -- (6) THE VECTORS THE RUN CLAIMS ARE THE VECTORS THIS REPLAY USES
    for label, raw, realized, raw_key, real_key, mean_key in (
            ("alpha", alpha_raw, alpha, "fj_alpha_raw_sha256",
             "fj_alpha_realized_sha256", "fj_alpha_realized_mean"),
            ("beta", beta_raw, beta, "fj_beta_raw_sha256",
             "fj_beta_realized_sha256", "fj_beta_realized_mean")):
        if cfg.get(raw_key) != _sha_t(raw):
            errs.append(f"WU {name}: {raw_key}={str(cfg.get(raw_key))[:16]}"
                        f"... != the repo's {label} vector "
                        f"{_sha_t(raw)[:16]}...")
        if cfg.get(real_key) != _sha_t(realized):
            errs.append(f"WU {name}: {real_key} does not equal "
                        f"sha256({label}_raw * c_{label}) -- the scale "
                        f"recorded in the config is not the scale applied")
        if abs(float(cfg.get(mean_key, float("nan")))
               - float(realized.mean())) > WU_MEAN_TOL:
            errs.append(f"WU {name}: {mean_key}={cfg.get(mean_key)} != "
                        f"{float(realized.mean()):.7f}")

    # -- (7) THE ALPHA COMPLEMENT, ELEMENTWISE.
    # The dataset ships alpha; the world must be built with 1 - alpha.
    # If the run records the stubbornness it used, it is compared
    # elementwise; the behavioural test is the inverted replay in (10).
    stub_sha = cfg.get("fj_peer_sus_sha256")
    if stub_sha is not None and stub_sha != _sha_t(1.0 - alpha):
        errs.append(f"WU {name}: the recorded stubbornness vector is not "
                    f"1 - alpha elementwise -- FJWorld.peer_sus is the "
                    f"ANCHOR weight, and passing alpha through unchanged "
                    f"runs peer susceptibility 1 - alpha instead")
    stub_mean = cfg.get("fj_peer_sus_mean")
    if stub_mean is not None and abs(float(stub_mean)
                                     - float((1.0 - alpha).mean())) > WU_MEAN_TOL:
        errs.append(f"WU {name}: fj_peer_sus_mean={stub_mean} != "
                    f"{float((1.0 - alpha).mean()):.7f} = mean(1 - alpha)")

    # -- (8) NO DEFFUANT OR AI-GATE PARAMETER MAY APPLY.
    # This is Wu's model. A stray gate or sweep setting must not sit in
    # the environment claiming to apply: FJ mixing and platform exposure
    # are unconditional, so any of these would name something that never
    # ran -- or, worse, something that did.
    if cfg.get("fj_observed_passthrough") not in (True, 1):
        errs.append(f"WU {name}: fj_observed_passthrough="
                    f"{cfg.get('fj_observed_passthrough')!r} must be on -- "
                    f"the platform knows O's opinions by construction")
    for key, ok in (("ai_gate_mode", ("threshold", None)),
                    ("peer_gate_mode", ("threshold", None)),
                    ("anchor_mode", ("fixed", None)),
                    ("data_regime", ("replace", None)),
                    ("run_mode", ("loop", None))):
        if cfg.get(key) not in ok:
            errs.append(f"WU {name}: {key}={cfg.get(key)!r} -- FJ has no "
                        f"gates and this wave has no accumulation; a "
                        f"non-default value must never be accepted")
    for key in ("eps", "eps_ai", "gamma_bias", "canary_delta",
                "pristine_frac", "replay_frac", "shuffle_p", "sort_q"):
        v = cfg.get(key)
        if v not in (None, 0, 0.0, False):
            errs.append(f"WU {name}: {key}={v} must be 0 -- it is a "
                        f"Deffuant/AI-gate or data-regime parameter and "
                        f"has no meaning under wu1")
    if cfg.get("ab_sweeps") not in (None, 1):
        errs.append(f"WU {name}: ab_sweeps={cfg.get('ab_sweeps')} -- the "
                    f"Deffuant sweep count must not apply here")
    if cfg.get("pop_reset"):
        errs.append(f"WU {name}: pop_reset must be False")

    # -- (9) ARM SEMANTICS, read from the tag, gated against the config.
    arm = arm_of(name)
    if arm is None:
        errs.append(f"WU {name}: cannot read a unique arm token from the "
                    f"tag (expected one of {sorted(WU_ARM_SEMANTICS)})")
    else:
        w_style, w_kl, w_mode, w_k, w_d = WU_ARM_SEMANTICS[arm]
        if cfg.get("training_style") != w_style:
            errs.append(f"WU {name}: arm {arm} declares training_style="
                        f"{cfg.get('training_style')!r}, expected "
                        f"{w_style!r}")
        if float(cfg.get("kl_beta", -1)) != w_kl:
            errs.append(f"WU {name}: arm {arm} has kl_beta="
                        f"{cfg.get('kl_beta')}, expected {w_kl}")
        if w_kl > 0 and cfg.get("kl_direction") != "forward":
            errs.append(f"WU {name}: {arm} must be FORWARD KL, got "
                        f"{cfg.get('kl_direction')!r}")
        if w_kl > 0 and cfg.get("kl_ref_adapter"):
            errs.append(f"WU {name}: {arm} must have NO reference adapter")
        if (cfg.get("wu_icl_mode"), int(cfg.get("wu_icl_k", -1)),
                int(cfg.get("wu_icl_d", -1))) != (w_mode, w_k, w_d):
            errs.append(
                f"WU {name}: arm {arm} means "
                f"(mode={w_mode}, K={w_k}, D={w_d}) but the run recorded "
                f"(mode={cfg.get('wu_icl_mode')!r}, "
                f"K={cfg.get('wu_icl_k')}, D={cfg.get('wu_icl_d')})")
        # FRESH-ADAPTER SEMANTICS. Every primary SFT arm trains a NEW
        # LoRA each round; a carried-over adapter turns 50 rounds of
        # retraining into one long fine-tune and there is no way to tell
        # from the trajectory afterwards.
        if arm in WU_TRAINED_ARMS:
            if cfg.get("fresh_each_round") is not True:
                errs.append(f"WU {name}: fresh_each_round="
                            f"{cfg.get('fresh_each_round')!r} in the "
                            f"primary SFT arm {arm}; the adapter must be "
                            f"rebuilt every round, not carried")
            if not cfg.get("use_lora"):
                errs.append(f"WU {name}: use_lora={cfg.get('use_lora')!r} "
                            f"in trained arm {arm}")
        else:
            if cfg.get("use_lora"):
                errs.append(f"WU {name}: use_lora={cfg.get('use_lora')!r} "
                            f"in FROZEN arm {arm}")
            if cfg.get("fresh_each_round"):
                errs.append(f"WU {name}: fresh_each_round set in FROZEN "
                            f"arm {arm}")

    # -- (10) OBSERVED PASSTHROUGH: the model never speaks for O.
    x_entry = torch.cat([innate_run.unsqueeze(0), op[:-1]], dim=0)  # live
    x_static = innate_run.unsqueeze(0).expand(expect_rounds, n)     # innate
    want_obs = x_entry if passthrough == "live" else x_static
    gap_sel = (served - want_obs)[:, O].abs()
    gap = float(gap_sel.max()) if gap_sel.numel() else 0.0
    if gap > WU_EXACT_TOL:
        alt = x_static if passthrough == "live" else x_entry
        alt_gap = float((served - alt)[:, O].abs().max())
        t_bad = int(gap_sel.max(dim=1).values.argmax())
        i_bad = int(gap_sel[t_bad].argmax())
        why = ""
        if float((served[t_bad, i_bad] - pred[t_bad, i_bad]).abs()) <= WU_EXACT_TOL:
            why = (" -- the served value at that agent IS the model's "
                   "prediction, i.e. a model prediction REPLACED an "
                   "observed opinion")
        errs.append(
            f"WU {name}: observed passthrough violated: served_O differs "
            f"from x_O ({passthrough}) by up to {gap:.3e} (round {t_bad}, "
            f"agent {i_bad}); against the '"
            f"{'innate' if passthrough == 'live' else 'live'}' reading the "
            f"gap is {alt_gap:.3e}{why}")
    # the model's own vector must still be recorded for U even when the
    # served vector is a passthrough on O -- otherwise there is nothing
    # to score the prediction with
    if float((served - pred)[:, U].abs().max()) > WU_EXACT_TOL:
        route = route_side_of(name)
        if route is None:
            errs.append(f"WU {name}: served_U differs from model_pred_raw "
                        f"on the held-out set in a run with no routing "
                        f"intervention -- something rewrote the model's "
                        f"output")

    # -- (11) NOBODY TRAINS ON A HELD-OUT AGENT.
    tidx = d["train_idx_raw"]
    tidx = tidx if torch.is_tensor(tidx) else torch.as_tensor(tidx)
    tidx = tidx.long()
    if tidx.dim() == 1:
        tidx = tidx.unsqueeze(0)
    if int(tidx.numel()):
        lo, hi = int(tidx.min()), int(tidx.max())
        if lo < 0 or hi >= n:
            errs.append(f"WU {name}: train_idx_raw leaves [0, {n}) "
                        f"(min {lo}, max {hi})")
        else:
            leaked = torch.unique(tidx[~O[tidx]])
            if leaked.numel():
                errs.append(
                    f"WU {name}: the optimizer saw {int(leaked.numel())} "
                    f"HELD-OUT agent(s), e.g. index {int(leaked[0])} -- "
                    f"training on U destroys the only thing the held-out "
                    f"set is for")
    if int(cfg.get("n_labeled", -1)) != expect_observed:
        errs.append(f"WU {name}: n_labeled={cfg.get('n_labeled')} != "
                    f"|O| = {expect_observed}; the training prefix and the "
                    f"observed set must be the same set")
    # SFT labels: round 0 is innate, round t is the previous post-FJ state
    ty = _f(d["train_y_raw"])
    if ty.dim() == 1:
        ty = ty.unsqueeze(0)
    if ty.shape[0] and ty.shape == tidx.shape:
        want_lab = torch.gather(x_entry[:ty.shape[0]], 1, tidx[:ty.shape[0]])
        bad = (ty - want_lab).abs()
        if float(bad.max()) > WU_EXACT_TOL:
            t_bad = int(bad.max(dim=1).values.argmax())
            errs.append(
                f"WU {name}: round-{t_bad} SFT labels are not the opinions "
                f"those agents held entering the round (round 0 = innate, "
                f"round t = the round-(t-1) post-FJ population)")
    elif ty.shape[0]:
        errs.append(f"WU {name}: train_y_raw {tuple(ty.shape)} does not "
                    f"line up with train_idx_raw {tuple(tidx.shape)}, so "
                    f"the labels cannot be attributed to agents")

    # -- (12) ROUTING, when the tag says there is any.
    errs += _routing_errs(name, cfg, innate_run, innate, O, n,
                          expect_observed)

    # -- (13) THE ANCHOR AND THE INNER LOOP, REPLAYED IN FULL.
    x0 = (1.0 - beta).unsqueeze(0) * innate_run.unsqueeze(0) + \
        beta.unsqueeze(0) * served
    if float((x0 - x_init_saved).abs().max()) > WU_REPLAY_TOL:
        errs.append(
            f"WU {name}: fj_x_init_raw is not "
            f"(1 - beta_i) innate + beta_i served (max gap "
            f"{float((x0 - x_init_saved).abs().max()):.3e}) -- either beta "
            f"was not applied per agent or the served vector recorded is "
            f"not the one the anchor was built from")
    a = alpha.unsqueeze(0)
    Wt = W.t().contiguous()
    u = x0.clone()
    u1 = None
    for _ in range(n_inner):
        u = (1.0 - a) * x0 + a * (u @ Wt)
        if u1 is None:
            u1 = u.clone()
    if u1 is not None and float((u1 - u1_saved).abs().max()) > WU_REPLAY_TOL:
        # u^(1) is affine in u^(0); a mismatch here is a WRONG
        # INITIALISATION, and the stale-state alternative is named
        # explicitly so the failure says which one it looks like. The
        # alternative is evaluated at the FIRST offending round, not
        # pooled over all of them -- a stale start cannot show at t=0
        # (there is no previous population), and pooling would make
        # every stale run read as "matches neither".
        per_round = (u1 - u1_saved).abs().max(dim=1).values
        bad = int((per_round > WU_REPLAY_TOL).nonzero()[0])
        stale0 = torch.cat([innate_run.unsqueeze(0), op[:-1]], dim=0)
        stale_u1 = (1.0 - a[0]) * x0[bad] + a[0] * (Wt.t() @ stale0[bad])
        why = ("it matches a PREVIOUS-POPULATION start, i.e. the inner "
               "loop was seeded from the last round's state"
               if float((u1_saved[bad] - stale_u1).abs().max()) <= WU_REPLAY_TOL
               else "it matches neither u^(0) = x_init nor a "
                    "previous-population start")
        errs.append(f"WU {name}: round {bad} u^(1) != (1 - alpha_i) x_init "
                    f"+ alpha_i P x_init -- {why}, so u^(0) was not this "
                    f"round's anchor")
    replay_gap = float((u - op).abs().max())
    if replay_gap > WU_REPLAY_TOL:
        errs.append(f"WU {name}: the trajectory does not replay the "
                    f"declared recurrence (max gap {replay_gap:.3e} > "
                    f"{WU_REPLAY_TOL}) at K={n_inner}, c_alpha={c_alpha}, "
                    f"c_beta={c_beta}")

    # -- (14) THE INVERTED CONVENTION IS DISTINGUISHABLE AND REJECTED.
    # Round 0 alone settles it and costs one extra inner loop instead of
    # fifty. If the two conventions were indistinguishable here the test
    # would be vacuous, so that is checked before the verdict.
    if c_alpha > 0:
        inv = 1.0 - alpha
        x0_0 = x0[:1]
        u_ok, u_inv = x0_0.clone(), x0_0.clone()
        for _ in range(n_inner):
            u_ok = (1.0 - alpha) * x0_0 + alpha * (u_ok @ Wt)
            u_inv = (1.0 - inv) * x0_0 + inv * (u_inv @ Wt)
        sep = float((u_ok - u_inv).abs().max())
        if sep > WU_REPLAY_TOL:
            if float((u_inv - op[:1]).abs().max()) <= WU_REPLAY_TOL:
                errs.append(
                    f"WU {name}: round 0 matches the INVERTED alpha "
                    f"convention -- the world was built with alpha as the "
                    f"anchor weight, so it ran peer susceptibility "
                    f"{float((1.0 - alpha).mean()):.4f} instead of "
                    f"{float(alpha.mean()):.4f}. Every downstream number "
                    f"is still finite, ordered and wrong")

    # -- (15) THE CONTEXT LOG, for every run whose ICL mode is not none.
    errs += _ctx_errs(name, cfg, d, served, op, x_entry, expect_observed,
                      expect_rounds)
    return errs


def routing_cohort(frac, seed, n_observed, n):
    """The routed cohort, RECOMPUTED rather than read.

    The runner draws it from the OBSERVED pool with a dedicated generator
    seeded by ROUTING_TREAT_SEED alone -- deliberately not by the run
    seed, so a twin can reuse the same cohort. That makes the cohort a
    pure function of (frac, cohort seed, |O|), which is what lets this
    checker rebuild it and lets the CONTROL twin carry no cohort at all
    and still be comparable.
    """
    pool = torch.arange(min(int(n_observed), int(n)))
    k = int(round(float(frac) * pool.numel()))
    gen = torch.Generator().manual_seed(int(seed) + WU_ROUTE_COHORT_STREAM)
    return pool[torch.randperm(pool.numel(), generator=gen)[:k]] \
        .sort().values.long()


def _routing_errs(name, cfg, innate_run, innate_ds, O, n, n_observed):
    """Routing is a SOURCE INJECTION at agents the platform can SEE.

    The runner rewrites the treated agents' INNATE opinion before
    anything reads it, so the treatment reaches x(0), the FJ anchor, the
    round-0 SFT labels and (through the passthrough) the served vector,
    all consistently. Two things therefore have to hold and are checked
    against the dataset vector rather than against a stored mask:
      * the cohort sits inside O -- injecting into U would rewrite the
        very truth the wave is scored against;
      * innate differs from the dataset EXACTLY on the cohort. An
        intervention that leaked outside its cohort is not the
        intervention the tag names.
    """
    errs = []
    side = route_side_of(name)
    frac = cfg.get("routing_treat_frac")
    moved = (innate_run - innate_ds).abs() > WU_EXACT_TOL
    if side is None:
        if frac not in (None, 0, 0.0):
            errs.append(f"WU {name}: routing_treat_frac={frac} in a run "
                        f"whose tag carries no routing side -- the tag and "
                        f"the intervention disagree")
        if bool(moved.any()):
            errs.append(f"WU {name}: innate differs from the dataset at "
                        f"{int(moved.sum())} agent(s) but the tag declares "
                        f"no routing treatment")
        return errs
    missing = [k for k in WU_ROUTING_CFG if k not in cfg]
    if missing:
        return [f"WU {name}: routing run is missing config field(s): "
                f"{', '.join(missing)}"]
    if side == "C":
        if float(frac) != 0.0:
            errs.append(f"WU {name}: CONTROL twin ran with "
                        f"routing_treat_frac={frac} -- a control that "
                        f"injects is not a control")
        if bool(moved.any()):
            errs.append(f"WU {name}: CONTROL twin's innate differs from the "
                        f"dataset at {int(moved.sum())} agent(s); the "
                        f"control arm must leave the population untouched")
        return errs
    # -- treatment
    if not 0.0 < float(frac) <= 1.0:
        errs.append(f"WU {name}: TREATMENT twin has routing_treat_frac="
                    f"{frac}, which selects no cohort")
        return errs
    val = float(cfg["routing_treat_value"])
    if not 0.0 <= val <= 1.0:
        errs.append(f"WU {name}: routing_treat_value={val} is outside "
                    f"[0, 1]; the runner refuses it, so an artifact "
                    f"carrying it did not come from this code path")
    cohort = routing_cohort(frac, cfg["routing_treat_seed"], n_observed, n)
    if cohort.numel() == 0:
        errs.append(f"WU {name}: the recomputed cohort is empty")
        return errs
    if not bool(O[cohort].all()):
        errs.append(f"WU {name}: the routed cohort contains "
                    f"{int((~O[cohort]).sum())} HELD-OUT agent(s) -- the "
                    f"injection may only touch agents the platform "
                    f"observes, or it rewrites the truth being scored")
    sha = cfg.get("routing_treat_idx_sha256")
    if sha is None:
        errs.append(f"WU {name}: TREATMENT twin records no "
                    f"routing_treat_idx_sha256, so the cohort it used "
                    f"cannot be tied to the one this checker rebuilt")
    elif sha != wuc.idx_sha256(cohort.numpy()):
        errs.append(f"WU {name}: the cohort recomputed from "
                    f"(frac={frac}, seed={cfg['routing_treat_seed']}, "
                    f"|O|={n_observed}) does not match "
                    f"routing_treat_idx_sha256 -- the run treated a "
                    f"different set of agents than its parameters name")
    if cfg.get("routing_treat_n") not in (None, int(cohort.numel())):
        errs.append(f"WU {name}: routing_treat_n="
                    f"{cfg.get('routing_treat_n')} != {int(cohort.numel())}")
    mask = torch.zeros(n, dtype=torch.bool)
    mask[cohort] = True
    on = float((innate_run[mask] - val).abs().max())
    if on > WU_EXACT_TOL:
        errs.append(f"WU {name}: TREATMENT twin did not set the cohort's "
                    f"innate to {val} (max gap {on:.3e})")
    outside = moved & (~mask)
    if bool(outside.any()):
        errs.append(f"WU {name}: the treatment moved {int(outside.sum())} "
                    f"agent(s) OUTSIDE its routed cohort -- the "
                    f"intervention is not confined to the set the tag and "
                    f"the parameters name")
    return errs


def check_routing_pair(name_t, d_t, name_c, d_c, env=None):
    """The paired half: treatment and control must differ on EXACTLY the
    routed cohort and nowhere else.

    This is the "source masks differ" gate, and it is deliberately not a
    comparison of two stored masks: the control carries no cohort (it
    runs at frac 0), so the cohort is recomputed from the TREATMENT's
    parameters and the two innate vectors are differenced. A pair whose
    difference set is not that cohort is two experiments, not a twin
    pair, and the gap between them is not the effect of routing.
    """
    errs = []
    if route_side_of(name_t) != "T" or route_side_of(name_c) != "C":
        return [f"WU pair {name_t} / {name_c}: the two tags are not a "
                f"treatment/control pair"]
    cfg_t = d_t.get("config") or {}
    cfg_c = d_c.get("config") or {}
    if twin_stem(name_t) != twin_stem(name_c):
        errs.append(f"WU pair {name_t} / {name_c}: the tags differ in more "
                    f"than the routing side, so they are not twins")
    for k in ("seed", "n_labeled", "fj_alpha_scale", "fj_beta_scale",
              "wu_icl_mode", "training_style", "fj_inner_steps",
              "routing_treat_seed"):
        if cfg_t.get(k) != cfg_c.get(k):
            errs.append(f"WU pair {name_t} / {name_c}: {k} differs "
                        f"({cfg_t.get(k)!r} vs {cfg_c.get(k)!r}); the twins "
                        f"are not matched")
    if float(cfg_c.get("routing_treat_frac", 0.0)) != 0.0:
        errs.append(f"WU pair {name_t} / {name_c}: the control twin ran at "
                    f"routing_treat_frac={cfg_c.get('routing_treat_frac')}")
    it, ic = _f(d_t["innate"]), _f(d_c["innate"])
    if it.shape != ic.shape:
        return errs + [f"WU pair {name_t} / {name_c}: innate lengths differ"]
    n = it.shape[0]
    n_obs = int(cfg_t.get("n_labeled", WU_N_OBSERVED))
    cohort = routing_cohort(cfg_t.get("routing_treat_frac", 0.0),
                            cfg_t.get("routing_treat_seed", 0), n_obs, n)
    want = torch.zeros(n, dtype=torch.bool)
    want[cohort] = True
    got = (it - ic).abs() > WU_EXACT_TOL
    # THE TEST IS CONTAINMENT, NOT EQUALITY. An injected value can
    # coincide with an agent's existing innate opinion -- on Pokec a
    # nontrivial share of the population already sits at 0.5 -- so a
    # cohort member that "did not move" is arithmetic, not a defect.
    # What would be a defect is a difference OUTSIDE the cohort.
    outside = got & ~want
    if bool(outside.any()):
        errs.append(f"WU pair {name_t} / {name_c}: the treatment and control "
                    f"SOURCE MASKS differ at {int(outside.sum())} agent(s) "
                    f"OUTSIDE the declared cohort -- the twins are not a "
                    f"matched pair, so their difference is not the effect "
                    f"of routing")
    val = cfg_t.get("routing_treat_value")
    if val is not None and cohort.numel():
        on_t = float((it[cohort] - float(val)).abs().max())
        if on_t > WU_EXACT_TOL:
            errs.append(f"WU pair {name_t} / {name_c}: the treatment's "
                        f"innate on the cohort is not the injected value "
                        f"{val} (max gap {on_t:.3e})")
        if bool((ic[cohort] - float(val)).abs().max() <= WU_EXACT_TOL) \
                and int(cohort.numel()) > 1:
            errs.append(f"WU pair {name_t} / {name_c}: the CONTROL twin's "
                        f"innate already equals the injected value across "
                        f"the whole cohort -- the two arms are the same run")
    return errs


def _load_ctx_rows(d):
    """wu_ctx_log is gzip JSONL -- one round_log_line per line. Accepts an
    already-loaded list (tests) or the path on disk."""
    log = d.get("wu_ctx_log")
    if log is not None:
        return list(log), None
    p = d.get("_wu_ctx_log_path")
    if not (p and Path(p).exists()):
        return None, None
    try:
        rows = []
        with gzip.open(p, "rt") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        return rows, None
    except Exception as e:
        return None, f"{type(e).__name__}"


def _ctx_errs(name, cfg, d, served, op, x_entry, n_observed, expect_rounds):
    """The in-context arms, gated against the run's own record.

    THE LEAK THIS EXISTS TO CATCH. observed_context exemplars must come
    from O, because O is what Wu's platform observes; an exemplar drawn
    from U puts a HELD-OUT TRUTH in the prompt and the model is then
    scored on U having been told the answer. The personal-memory modes
    must show the agent its OWN recorded history and nothing else --
    prediction_history the SERVED values, expressed_history the POST-FJ
    opinions -- and at round 0 there is no history at all, so a non-empty
    one was seeded from outside the run.

    wu_context.audit_entry is the per-entry proof (ids observed, values
    equal to x_O[ids], rendered text displaying exactly those values, in
    order) and lives in ONE place; what is added here is the check that
    the values are THIS RUN'S OWN recorded history, which the runner's
    helper cannot know.
    """
    errs = []
    mode = cfg.get("wu_icl_mode")
    if mode not in wuc.MODES:
        return [f"WU {name}: wu_icl_mode={mode!r} is not one of {wuc.MODES}"]
    rows, parse_fail = _load_ctx_rows(d)
    if parse_fail:
        return [f"WU {name}: wu_ctx_log.json.gz will not parse "
                f"({parse_fail}); a context nobody can read is a context "
                f"nobody has checked"]
    if mode == "none":
        for r in (rows or []):
            for ag in (r.get("agents") or []):
                if ag.get("ids") or ag.get("values"):
                    return [f"WU {name}: wu_icl_mode='none' but the context "
                            f"log carries exemplars at round "
                            f"{r.get('round')}"]
        return errs
    if rows is None:
        return [f"WU {name}: wu_icl_mode={mode!r} but no wu_ctx_log.json.gz "
                f"-- an in-context arm whose contexts were never recorded "
                f"cannot be shown free of held-out truth"]
    if len(rows) != expect_rounds:
        errs.append(f"WU {name}: the context log has {len(rows)} rounds, "
                    f"the run has {expect_rounds}")
    import numpy as np
    observed_ids = np.arange(int(n_observed), dtype=np.int64)
    hist = {"prediction_history": served, "expressed_history": op}.get(mode)
    depth = int(cfg.get("wu_icl_d", 0))
    k_ctx = int(cfg.get("wu_icl_k", 0))
    for r in rows:
        for k in ("round", "mode", "wu_icl_k", "wu_icl_d", "history_source",
                  "wu_icl_extension", "agents"):
            if k not in r:
                return errs + [f"WU {name}: a context log round is missing "
                               f"the required key {k!r}"]
        t = int(r["round"])
        if not 0 <= t < expect_rounds:
            errs.append(f"WU {name}: context log round {t} outside the run")
            break
        if r["mode"] != mode:
            errs.append(f"WU {name}: context log round {t} mode="
                        f"{r['mode']!r} != config wu_icl_mode={mode!r}")
            break
        if (int(r["wu_icl_k"]), int(r["wu_icl_d"])) != (k_ctx, depth):
            errs.append(f"WU {name}: context log round {t} carries "
                        f"(K={r['wu_icl_k']}, D={r['wu_icl_d']}), the "
                        f"config says (K={k_ctx}, D={depth})")
            break
        if r["history_source"] != wuc.HISTORY_SOURCE[mode]:
            errs.append(f"WU {name}: context log round {t} history_source="
                        f"{r['history_source']!r} does not match mode "
                        f"{mode!r} -- what the displayed numbers ARE is not "
                        f"what the mode says they are")
            break
        if bool(r["wu_icl_extension"]) != wuc.is_extension(mode):
            errs.append(f"WU {name}: context log round {t} marks "
                        f"extension={r['wu_icl_extension']} for mode "
                        f"{mode!r}, which wu_context classifies as "
                        f"{'an extension' if wuc.is_extension(mode) else 'strict'}")
            break
        for entry in (r.get("agents") or []):
            for k in ("agent", "ids", "values", "text", "history_source",
                      "mode", "extension"):
                if k not in entry:
                    return errs + [f"WU {name}: context log round {t} has an "
                                   f"agent record missing {k!r}"]
            bad = wuc.audit_entry(entry, observed_ids=observed_ids,
                                  opinion=x_entry[t])
            if bad:
                lead = ("HELD-OUT TRUTH LEAK -- " if any(
                    "HELD-OUT" in b for b in bad) else "")
                errs.append(f"WU {name}: {lead}round {t} agent "
                            f"{entry.get('agent')}: " + "; ".join(bad))
                return errs
            i = int(entry["agent"])
            vals = [float(v) for v in entry["values"]]
            if mode == "observed_context":
                if len(vals) != k_ctx:
                    errs.append(f"WU {name}: round {t} agent {i} carries "
                                f"{len(vals)} exemplars, not K={k_ctx}")
                    return errs
                continue
            # personal memory: the values must be THIS RUN'S OWN record
            want = [float(hist[s, i]) for s in range(max(0, t - depth), t)]
            if any(v != v for v in want):
                # the platform never spoke for this agent, so there is no
                # history for it to show
                if vals:
                    errs.append(
                        f"WU {name}: round {t} agent {i} has no recorded "
                        f"{wuc.HISTORY_SOURCE[mode]} history in this run "
                        f"(the platform was never asked about it), yet "
                        f"carries {len(vals)} value(s) -- those did not "
                        f"come from this run")
                    return errs
                continue
            off = [j for j, (a, b) in enumerate(zip(vals, want))
                   if abs(a - b) > WU_EXACT_TOL]
            if len(vals) != len(want) or off:
                errs.append(
                    f"WU {name}: round {t} agent {i}: the "
                    f"{wuc.HISTORY_SOURCE[mode]} history shown is not this "
                    f"run's own last {depth} value(s) for that agent (got "
                    f"{len(vals)}, expected {len(want)})"
                    + ("" if not off else
                       f"; first mismatch at position {off[0]}")
                    + (" -- at round 0 there is nothing to remember, so a "
                       "non-empty history is seeded from outside the run"
                       if t == 0 else ""))
                return errs
    return errs


# ---------------------------------------------------------------- driver

def check_run_dir(run_dir, **kw):
    path = os.path.join(run_dir, "trajectory.pt")
    name = os.path.basename(str(run_dir).rstrip("/"))
    if not os.path.exists(path):
        return name, None, [f"MISSING {path}"]
    d = torch.load(path, map_location="cpu", weights_only=False)
    ctx = os.path.join(run_dir, "wu_ctx_log.json.gz")
    if os.path.exists(ctx):
        d["_wu_ctx_log_path"] = ctx
    r = rounds_of(name)
    if r is not None:
        kw.setdefault("expect_rounds", r)
    return name, d, check_jiduan_run(name, d, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--passthrough", default=WU_PASSTHROUGH_DEFAULT,
                    choices=WU_PASSTHROUGH_MODES,
                    help="what x_O means in served_O = x_O")
    ap.add_argument("--pokec-dir", default=None)
    args = ap.parse_args()
    if args.pokec_dir:
        _WU_ENV_CACHE.pop("env", None)
        if wu_env(args.pokec_dir) is None:
            print(f"[check_jiduan_pokec] cannot load the Pokec dataset "
                  f"from {args.pokec_dir}", file=sys.stderr)
            sys.exit(2)

    dirs = []
    for a in args.run_dirs:
        dirs += sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a]
    loaded, n_fail = {}, 0
    for rd in dirs:
        name, d, errs = check_run_dir(rd, passthrough=args.passthrough)
        loaded[name] = d
        if errs:
            n_fail += 1
            print(f"FAIL {name}")
            for e in errs:
                print(f"     - {e}")
        else:
            print(f"PASS {name}  (wu1 replayed at K={WU_INNER}, per-agent "
                  f"alpha/beta, passthrough on O, trained on O only)")
    # PAIRED routing check -- only possible when both twins are present,
    # and silence here would be the easiest thing in the world to miss
    stems = {}
    for name, d in loaded.items():
        side = route_side_of(name)
        if side and d is not None:
            stems.setdefault(twin_stem(name), {})[side] = (name, d)
    for stem, pair in sorted(stems.items()):
        if set(pair) != {"T", "C"}:
            print(f"NOTE {stem}: only the "
                  f"{'treatment' if 'T' in pair else 'control'} twin is "
                  f"present, so the source masks were NOT compared")
            continue
        errs = check_routing_pair(*pair["T"], *pair["C"])
        if errs:
            n_fail += 1
            print(f"FAIL pair {stem}")
            for e in errs:
                print(f"     - {e}")
        else:
            print(f"PASS pair {stem}  (identical routed cohort, matched "
                  f"frac/seed)")
    print(f"[check_jiduan_pokec] {len(dirs) - n_fail}/{len(dirs)} runs pass")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
