#!/usr/bin/env python3
"""HARD-GATED analyzer for the Qwen2.5 mechanism diagnostic (2026-08-20).

WHAT THE EXPERIMENT ANSWERS. Five conditions at matched (k, eps_social)
form a ladder from "no platform at all" to "a real retrained LLM":

  twin        no platform
  perfect     m(t) = x(t): an exact population echo, CPU oracle
  frozen      the STATIC entering Qwen2.5 prediction map (K = D = 0)
  sft         ordinary fresh SFT, lambda = 0
  sft_kl      fresh forward-KL SFT, lambda = 1

At the Wu boundary (Part C only) a SIXTH condition joins them:

  icl         personal-history ICL, D = 8, K = 0, frozen weights

which is MEMORY WITHOUT LEARNING: no gradients touch the model, but the
served value tracks each agent's own last 8 opinions. It sits between the
static frozen map (no adaptation at all) and b0/b1 (parametric
retraining), and answers whether the loop needs weight updates at all or
whether per-agent history alone reproduces the feedback dynamics. It has
no Part-A counterpart, so it appears only in the Wu-boundary figure.

and the contrasts are read STRICTLY as follows. These strings are the
ones written into mechanism_contrasts.csv, so the causal language cannot
drift between the code and the write-up:

  perfect - twin      the effect of CLOSING an exact population-echo loop
  frozen - perfect    the effect of replacing that echo with the static
                      entering Qwen prediction map
  sft - perfect       the AGGREGATE parametric-retraining gap. NOT
                      "optimizer error". Ordinary SFT still starts from
                      pretrained Qwen weights and a finite-rank LoRA, so
                      this contrast bundles pretrained initialization,
                      limited capacity, finite optimization, parameters
                      shared across agents, greedy decoding, parsing, and
                      generalization across profiles.
  sft_kl - sft        explicit forward-KL reference retention
  sft_kl - frozen     how learning from the EVOLVING population changes an
                      already-retained model signal

THEORY MAPPING (Wu et al., "Reaching a Consensus in Predictive Loops",
arXiv:2603.12137). Our pre-peer update is

    z = (1 - W) [ k x_innate + (1 - k) x ] + W m.

Under perfect prediction m = x it is exactly Friedkin-Johnsen with

    z = (1 - beta_eff) x_innate + beta_eff x,   beta_eff = 1 - (1 - W) k.

  * the paper setting k = .2, W = .5 has beta_eff = .9
  * k = 1, W = .5 has beta_eff = .5, so k = 1 ALONE IS NOT A CONSENSUS
    LIMIT -- it is a LESS anchored pre-peer map
  * for non-perfect Qwen predictions, k = 1 gives the direct Wu-style
    form z = (1 - W) x_innate + W m
  * the relevant high-susceptibility boundary is therefore k = 1, W = 1
  * at W = 1 the pre-peer population EQUALS the served prediction vector
    and k becomes algebraically irrelevant
  * Wu et al.'s consensus result requires perfect prediction AND platform
    susceptibility approaching one. Perfect prediction at finite
    susceptibility can retain a heterogeneous equilibrium.

Our randomized Deffuant midpoint process is NOT Wu et al.'s deterministic
FJ operator. With a connected graph and genuinely open peer interactions
it is a randomized-gossip analogue that should converge to consensus
under perfect prediction at W = 1. That is a qualitative limiting-case
CORRESPONDENCE, not a replication of their theorem, and it is described
that way everywhere in this pipeline.

WARNING, restated in mechanism_per_cell.csv: comparing k = .2 with k = 1
at fixed W = .5 changes beta_eff (.9 -> .5) as well as innate/state
anchoring. It is NOT a pure memory ablation. And W is POPULATION
SUSCEPTIBILITY to the served output -- W = .5 vs W = 1 is not a
regularization comparison.

LOCATION VERSUS HETEROGENEITY. Reporting MAE or RMSE alone cannot
separate a common directional shift from heterogeneous errors, so every
cell carries both a location measure and a shape measure:

  mean difference        equilibrium-LOCATION movement
  SD, range              dispersion
  mean-centered W1       distributional SHAPE after removing location
  raw W1                 location and shape combined

  large raw W1 + large mean gap + small centered W1  -> mainly a shift
  large centered W1 / SD gap / range gap             -> changed
                                                        heterogeneity
  perfect prediction, bounded peers, no pretrained signal -> whatever
      heterogeneity remains is the NONLINEAR DYNAMICS
  frozen or sft_kl exceeding the perfect-prediction centered-shape
      effect -> an additional model-signal contribution
  sft differing from perfect -> aggregate practical-retraining effect

LABEL TIMING. Prediction error is measured against the labels that
actually trained the model serving that round: the initial training
labels at round 0, and the PRECEDING recorded population afterwards
(op_raw[t-1]). Predictions are never compared to post-intervention
opinions from the same round.

RNG MATCHING, stated plainly. The CPU oracle seeds its peer generator
exactly as the runner does (seed + 424243), but the cluster runs build
torch.Generator(device="cuda"), whose stream differs from a CPU
generator at the same seed. So oracle-vs-oracle comparisons are exactly
RNG-matched and oracle-vs-LLM comparisons are DISTRIBUTIONAL. Every
oracle-vs-LLM statistic here is therefore a W1 / moment / shape
quantity; no per-agent paired difference against the oracle is reported,
because it would not mean what it looks like. Each LLM cell's own twin
is the run's recorded twin_raw; at eps_social = 0 the twin has no RNG at
all (no peer step) and is computed exactly.

Usage:
  python analyze_qwen_mechanism_diagnostic.py [--runs-root DIR ...]
      [--out DIR] [--part a|c|all]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import torch                                               # noqa: E402

HERE = Path(os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
REPO = HERE.parent.parent.parent

_spec = importlib.util.spec_from_file_location(
    "sim_pp", str(HERE / "sim_perfect_predictor.py"))
PP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PP)

MANIFEST = REPO / "experiments" / "condor" / "manifest_qwen_mechanism.json"
DEFAULT_ROOTS = [REPO / "runs" / "pokec_gated_lm",
                 REPO / "notes" / "pofd" / "cluster"]
PP_DIR = REPO / "notes" / "pofd" / "perfect_prediction"
FRZ_DIR = REPO / "notes" / "pofd" / "frozen_replay"
DEFAULT_OUT = REPO / "notes" / "pofd" / "qwen_mechanism_analysis"

KS = [0.2, 1.0]
ESS = [0.0, 0.05, 0.2, 1.0]
ARMS = {"k0": "frozen", "b0": "sft", "b1": "sft_kl"}
CONDITIONS = ["twin", "perfect", "frozen", "sft", "sft_kl"]
# Part C carries one extra arm that has no Part-A counterpart: d8 =
# personal-history ICL (D=8, K=0, frozen weights). It is memory WITHOUT
# learning, sitting between the static frozen map and parametric
# retraining, so it gets its own colour and only appears in the
# Wu-boundary figure.
WU_CONDITIONS = CONDITIONS + ["icl"]
COLORS = {"twin": "#8c8c8c", "perfect": "#111111", "frozen": "#e8820c",
          "sft": "#2a6fb5", "sft_kl": "#c0392b", "icl": "#2e8b57"}
LABELS = {"twin": "no-platform twin", "perfect": "perfect prediction",
          "frozen": "frozen Qwen", "sft": "ordinary SFT",
          "sft_kl": "regularized SFT",
          "icl": "personal-history ICL ($D{=}8$)"}
WU_ARMS = {"b0": "sft", "b1": "sft_kl", "d8": "icl"}
# declared tolerance for the numerical cluster count: two opinions are in
# the same cluster when they are within this of each other
CLUSTER_TOL = 1e-4
LATE_WINDOW = 5             # last 5 rounds; the exact rounds go in the CSV
QUANTILES = np.linspace(0.0, 1.0, 7)
WU_WS = [0.5, 1.0]
WU_MARK_ROUNDS = [30, 100]
ORACLE_WS = [0.5, 0.9, 1.0]
ORACLE_PEERS = [("nopeer", 0.0, "threshold"), ("thr0p05", 0.05, "threshold"),
                ("thr0p2", 0.2, "threshold"), ("open", 0.2, "all_open")]
ORACLE_PEER_LABEL = {"nopeer": "no peers", "thr0p05": "threshold .05",
                     "thr0p2": "threshold .2", "open": "all-open"}
ORACLE_ROUNDS = 300


# ---------------------------------------------------------------- metrics
def w1(a, b):
    """Equal-n 1-Wasserstein: mean |sort(a) - sort(b)|."""
    return float(np.abs(np.sort(a) - np.sort(b)).mean())


def w1_centered(a, b):
    """W1 after removing each sample's mean -- pure SHAPE, no location."""
    return w1(a - a.mean(), b - b.mean())


def shape_vector(x):
    """Centered seven-quantile shape descriptor."""
    return np.quantile(x - x.mean(), QUANTILES)


def cluster_count(x, tol=CLUSTER_TOL):
    """Numerical clusters at a DECLARED tolerance: sort, then split
    wherever consecutive opinions are more than tol apart."""
    s = np.sort(x)
    return int(1 + (np.diff(s) > tol).sum()) if s.size else 0


def slope(y):
    """Least-squares slope of y against its index."""
    if len(y) < 2:
        return float("nan")
    t = np.arange(len(y), dtype=float)
    return float(np.polyfit(t, np.asarray(y, dtype=float), 1)[0])


def round_metrics(x, ini, prev, served, labels, pp_x):
    """Every per-round quantity for one condition."""
    row = {
        "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
        "var": float(x.var(ddof=1)),
        "range": float(x.max() - x.min()),
        "w1_from_initial": w1(x, ini),
        "n_clusters": cluster_count(x),
    }
    for i, q in enumerate(shape_vector(x)):
        row[f"shape_q{i}"] = float(q)
    row["w1_prev_round"] = w1(x, prev) if prev is not None else float("nan")
    if pp_x is not None:
        row["w1_to_perfect"] = w1(x, pp_x)
        row["w1_to_perfect_centered"] = w1_centered(x, pp_x)
        row["mean_diff_to_perfect"] = float(x.mean() - pp_x.mean())
    else:
        row["w1_to_perfect"] = float("nan")
        row["w1_to_perfect_centered"] = float("nan")
        row["mean_diff_to_perfect"] = float("nan")
    if served is not None and labels is not None:
        d = served - labels
        row.update({
            "pred_mean": float(served.mean()),
            "pred_sd": float(served.std(ddof=1)),
            "pred_bias": float(d.mean()),
            "pred_mae": float(np.abs(d).mean()),
            "pred_rmse": float(np.sqrt((d ** 2).mean())),
            "pred_w1_centered": w1_centered(served, labels),
        })
    else:
        for k in ("pred_mean", "pred_sd", "pred_bias", "pred_mae",
                  "pred_rmse", "pred_w1_centered"):
            row[k] = float("nan")
    return row


# ----------------------------------------------------------------- loading
def load_traj(run_dir):
    d = torch.load(Path(run_dir) / "trajectory.pt", map_location="cpu",
                   weights_only=False)
    return d


def np_(t):
    return t.detach().cpu().float().numpy()


def analytic_twin(innate, k, rounds):
    """The no-platform twin when eps_social = 0: the peer step is inert,
    so the twin is x <- k innate + (1-k) x with NO RNG at all and is
    computed exactly rather than read off a run that may predate the
    WITH_TWIN flag."""
    out, x = [], innate.copy()
    for _ in range(rounds):
        x = k * innate + (1.0 - k) * x
        out.append(x.copy())
    return np.stack(out)


def pp_path(k, w, es, rounds, ai="threshold", peer="threshold", ea=1.0):
    cfg = {"innate_k": k, "w_plat": w, "eps_social": es, "eps_ai": ea,
           "ai_gate_mode": ai, "peer_gate_mode": peer, "ab_sweeps": 1,
           "seed": 0, "rounds": rounds}
    return PP_DIR / PP.artifact_name(cfg)


def frz_path(k, w, es, rounds, ai="all_open", peer="all_open", ea=1.0):
    cfg = {"innate_k": k, "w_plat": w, "eps_social": es, "eps_ai": ea,
           "ai_gate_mode": ai, "peer_gate_mode": peer, "ab_sweeps": 1,
           "seed": 0, "rounds": rounds}
    return FRZ_DIR / ("frz" + PP.artifact_name(cfg)[2:])


def require(paths, what):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(
            f"[qmech] HARD FAIL: {what} missing {len(missing)} of "
            f"{len(paths)}:\n  " + "\n  ".join(missing))


def resolve_gpu_cells(roots):
    """{(arm, k, es): run_dir} for all 24 Part-A GPU cells. Reused cells
    come from the audited manifest; new cells are looked up by their
    generated tag. Missing cells are a HARD FAIL naming every one."""
    mf = json.load(open(MANIFEST))
    out, missing = {}, []
    for c in mf["cells"]:
        key = (c["arm"], c["innate_k"], c["eps_social"])
        if c["status"] == "reused":
            rd = Path(c["run_dir"])
            if not (rd / "trajectory.pt").exists():
                # the manifest was written on the cluster; re-resolve the
                # same tag against the local roots
                rd = None
                for root in roots:
                    cand = Path(root) / c["run_tag"]
                    if (cand / "trajectory.pt").exists():
                        rd = cand
                        break
            if rd is None:
                missing.append(f"{key} reused tag {c['run_tag']}")
                continue
            out[key] = rd
        else:
            rd = None
            for root in roots:
                cand = Path(root) / c["new_tag"]
                if (cand / "trajectory.pt").exists():
                    rd = cand
                    break
            if rd is None:
                missing.append(f"{key} NEW tag {c['new_tag']} (not run yet?)")
                continue
            out[key] = rd
    if missing:
        raise SystemExit(
            f"[qmech] HARD FAIL: {len(missing)} of {len(mf['cells'])} "
            f"Part-A GPU cells unavailable:\n  " + "\n  ".join(missing))
    return out


def wu_tag(arm, w):
    def n(v):
        return f"{v:g}".replace(".", "p")
    return (f"pofdqwu_qwen7b_{arm}_eaopen_w{n(w)}_l1_esopen_s0_r100")


def resolve_wu_cells(roots):
    out, missing = {}, []
    for w in WU_WS:
        for arm in WU_ARMS:
            tag = wu_tag(arm, w)
            rd = None
            for root in roots:
                cand = Path(root) / tag
                if (cand / "trajectory.pt").exists():
                    rd = cand
                    break
            if rd is None:
                missing.append(tag)
            else:
                out[(WU_ARMS[arm], w)] = rd
    if missing:
        raise SystemExit(
            f"[qmech] HARD FAIL: {len(missing)} of "
            f"{len(WU_WS) * len(WU_ARMS)} Wu-limit GPU cells unavailable "
            f"(submit qwen_wu_limit / qwen_wu_limit_icl first):\n  "
            + "\n  ".join(missing))
    return out


# -------------------------------------------------------------- part A
def check_shared_environment(run_dirs, oracle_paths):
    """HARD GATE: every cell in the grid must sit on the SAME Action
    population and graph, with a bit-identical innate vector. A grid
    built on two different environments would make every cross-cell
    contrast meaningless while looking perfectly well-formed."""
    ref, ref_name, errs = None, None, []
    for rd in run_dirs:
        d = load_traj(rd)
        v = d["innate"].float()
        if ref is None:
            ref, ref_name = v, str(rd)
        elif not torch.equal(v, ref):
            errs.append(f"innate vector of {rd} differs from {ref_name}")
        n_ag = d["op_raw"].shape[1]
        if n_ag != ref.shape[0]:
            errs.append(f"{rd} has {n_ag} agents, expected {ref.shape[0]}")
    for p in oracle_paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        if not torch.equal(d["innate"].float(), ref):
            errs.append(f"innate vector of {p} differs from {ref_name}")
    if errs:
        raise SystemExit("[qmech] HARD FAIL: shared-environment check:\n  "
                         + "\n  ".join(errs))
    print(f"[qmech] shared environment OK: innate bit-identical across "
          f"{len(run_dirs)} run(s) + {len(oracle_paths)} oracle cell(s)")


def check_matched_twins(runs):
    """HARD GATE: cells that share (k, eps_social) share a seed, an
    environment and a peer-RNG construction, so their recorded twins
    must agree. Arms differ only in the platform, and the twin has no
    platform in it."""
    errs = []
    for k in KS:
        for es in ESS:
            if es <= 0:
                continue        # es=0 twins are RNG-free; checked exactly
            ref, ref_arm = None, None
            for arm in ARMS:
                tw = load_traj(runs[(arm, k, es)]).get("twin_raw")
                if tw is None or tw.numel() == 0:
                    continue
                if ref is None:
                    ref, ref_arm = tw.float(), arm
                elif not torch.equal(tw.float(), ref):
                    errs.append(
                        f"k={k:g} es={es:g}: twin of arm {arm} differs "
                        f"from arm {ref_arm} (max "
                        f"{float((tw.float() - ref).abs().max()):.3e})")
    if errs:
        raise SystemExit("[qmech] HARD FAIL: matched-twin check:\n  "
                         + "\n  ".join(errs))
    print("[qmech] matched twins agree across arms at every (k, eps_social)")


def analyse_part_a(roots, out_dir):
    runs = resolve_gpu_cells(roots)
    oracles = [pp_path(k, 0.5, es, 30) for k in KS for es in ESS]
    require(oracles, "Part-A perfect-prediction oracle cells")
    check_shared_environment(list(runs.values()), oracles)
    check_matched_twins(runs)

    rows, per_cell = [], []
    for k in KS:
        for es in ESS:
            ppd = torch.load(pp_path(k, 0.5, es, 30), map_location="cpu",
                             weights_only=False)
            innate = np_(ppd["innate"])
            pp_op = np_(ppd["op_raw"])
            pp_pred = np_(ppd["pred_raw"])
            pp_twin = np_(ppd["twin_raw"])
            n_rounds = pp_op.shape[0]

            series = {"perfect": (pp_op, pp_pred)}
            for arm, cond in ARMS.items():
                d = load_traj(runs[(arm, k, es)])
                series[cond] = (np_(d["op_raw"]), np_(d["pred_raw"]))
            # the twin: the oracle's own CPU twin is exactly RNG-matched
            # to the oracle, and at es=0 the twin has no RNG at all
            series["twin"] = (pp_twin, None)

            for cond in CONDITIONS:
                op, pred = series[cond]
                for t in range(n_rounds):
                    labels = innate if t == 0 else op[t - 1]
                    m = round_metrics(
                        op[t], innate, op[t - 1] if t else None,
                        None if pred is None else pred[t],
                        None if pred is None else labels,
                        pp_op[t])
                    rows.append({"part": "A", "condition": cond,
                                 "innate_k": k, "w_plat": 0.5,
                                 "eps_social": es, "round": t, **m})
                late = list(range(n_rounds - LATE_WINDOW, n_rounds))
                lm = [float(op[t].mean()) for t in late]
                last = op[-1]
                pc = {"part": "A", "condition": cond, "innate_k": k,
                      "w_plat": 0.5, "eps_social": es,
                      "beta_eff_if_perfect": PP.beta_eff(k, 0.5),
                      "late_rounds": f"{late[0]}-{late[-1]}",
                      "late_mean": float(np.mean(lm)),
                      "late_slope": slope(lm),
                      "final_mean": float(last.mean()),
                      "final_sd": float(last.std(ddof=1)),
                      "final_range": float(last.max() - last.min()),
                      "final_n_clusters": cluster_count(last),
                      "cluster_tol": CLUSTER_TOL,
                      "final_w1_from_initial": w1(last, innate),
                      "final_w1_to_perfect": w1(last, pp_op[-1]),
                      "final_w1_to_perfect_centered":
                          w1_centered(last, pp_op[-1]),
                      "final_mean_diff_to_perfect":
                          float(last.mean() - pp_op[-1].mean()),
                      "k_comparison_warning":
                          "k=.2 vs k=1 at fixed W changes beta_eff too; "
                          "NOT a pure memory ablation",
                      }
                if pred is not None:
                    lab = np.stack([innate if t == 0 else op[t - 1]
                                    for t in range(n_rounds)])
                    pc["final_pred_mae"] = float(
                        np.abs(pred[-1] - lab[-1]).mean())
                    pc["late_pred_mae"] = float(
                        np.abs(pred[late] - lab[late]).mean())
                else:
                    pc["final_pred_mae"] = float("nan")
                    pc["late_pred_mae"] = float("nan")
                per_cell.append(pc)

    # ---- contrasts, with the causal reading fixed in the CSV ----------
    CONTRASTS = [
        ("perfect", "twin",
         "effect of closing an exact population-echo loop"),
        ("frozen", "perfect",
         "effect of replacing the echo with the static entering Qwen "
         "prediction map"),
        ("sft", "perfect",
         "AGGREGATE parametric-retraining gap (pretrained init, finite "
         "LoRA capacity, finite optimization, shared parameters, greedy "
         "decoding, parsing, generalization) -- NOT optimizer error"),
        ("sft_kl", "sft",
         "explicit forward-KL reference retention"),
        ("sft_kl", "frozen",
         "how learning from the evolving population changes a retained "
         "model signal"),
    ]
    by = {(r["condition"], r["innate_k"], r["eps_social"]): r
          for r in per_cell}
    contrasts = []
    for a, b, reading in CONTRASTS:
        for k in KS:
            for es in ESS:
                ra, rb = by[(a, k, es)], by[(b, k, es)]
                contrasts.append({
                    "contrast": f"{a} - {b}", "reading": reading,
                    "innate_k": k, "eps_social": es, "w_plat": 0.5,
                    "d_late_mean": ra["late_mean"] - rb["late_mean"],
                    "d_final_sd": ra["final_sd"] - rb["final_sd"],
                    "d_final_range": ra["final_range"] - rb["final_range"],
                    "d_final_n_clusters":
                        ra["final_n_clusters"] - rb["final_n_clusters"],
                    "w1_final": None, "w1_final_centered": None,
                })
    # raw / centered W1 between the two conditions' final populations
    finals = {}
    for k in KS:
        for es in ESS:
            ppd = torch.load(pp_path(k, 0.5, es, 30), map_location="cpu",
                             weights_only=False)
            finals[("perfect", k, es)] = np_(ppd["op_raw"])[-1]
            finals[("twin", k, es)] = np_(ppd["twin_raw"])[-1]
            for arm, cond in ARMS.items():
                finals[(cond, k, es)] = np_(
                    load_traj(runs[(arm, k, es)])["op_raw"])[-1]
    for row in contrasts:
        a, b = row["contrast"].split(" - ")
        fa = finals[(a, row["innate_k"], row["eps_social"])]
        fb = finals[(b, row["innate_k"], row["eps_social"])]
        row["w1_final"] = w1(fa, fb)
        row["w1_final_centered"] = w1_centered(fa, fb)

    write_csv(out_dir / "mechanism_rounds.csv", rows)
    write_csv(out_dir / "mechanism_per_cell.csv", per_cell)
    write_csv(out_dir / "mechanism_contrasts.csv", contrasts)
    figure_mechanism(per_cell, out_dir)
    return rows, per_cell, contrasts


# -------------------------------------------------------------- part C/D/E
def analyse_oracle_grid(out_dir):
    """Part E: 12 cheap oracle cells separating susceptibility from
    nonlinear peer selection."""
    paths = [pp_path(1.0, w, es, ORACLE_ROUNDS, ai="all_open", peer=pg)
             for w in ORACLE_WS for _, es, pg in ORACLE_PEERS]
    require(paths, "Part-E oracle grid cells")
    rows = []
    for w in ORACLE_WS:
        for label, es, pg in ORACLE_PEERS:
            d = torch.load(pp_path(1.0, w, es, ORACLE_ROUNDS,
                                   ai="all_open", peer=pg),
                           map_location="cpu", weights_only=False)
            op, innate = np_(d["op_raw"]), np_(d["innate"])
            last = op[-1]
            rows.append({
                "innate_k": 1.0, "w_plat": w, "peer_condition": label,
                "eps_social": es, "peer_gate_mode": pg,
                "beta_eff": PP.beta_eff(1.0, w),
                "rounds": ORACLE_ROUNDS,
                "final_mean": float(last.mean()),
                "mean_drift_from_initial":
                    float(last.mean() - innate.mean()),
                "final_sd": float(last.std(ddof=1)),
                "final_var": float(last.var(ddof=1)),
                "final_range": float(last.max() - last.min()),
                "final_n_clusters": cluster_count(last),
                "cluster_tol": CLUSTER_TOL,
                "sd_r30": float(op[29].std(ddof=1)),
                "sd_r100": float(op[99].std(ddof=1)),
                "sd_r300": float(op[299].std(ddof=1)),
                "w1_from_initial": w1(last, innate),
                "note": ("W is POPULATION SUSCEPTIBILITY to the served "
                         "output, not a regularization dial"),
            })
    write_csv(out_dir / "wu_oracle_grid.csv", rows)
    return rows


def analyse_part_c(roots, out_dir):
    """Parts C + D: the Wu boundary with real Qwen, plus the CPU oracle
    and offline frozen controls extended to 300 rounds."""
    runs = resolve_wu_cells(roots)
    require([pp_path(1.0, w, 0.2, ORACLE_ROUNDS, ai="all_open",
                     peer="all_open") for w in WU_WS],
            "Part-D perfect-prediction controls")
    require([frz_path(1.0, w, 0.2, ORACLE_ROUNDS) for w in WU_WS],
            "Part-D offline frozen replays")

    rows = []
    for w in WU_WS:
        ppd = torch.load(pp_path(1.0, w, 0.2, ORACLE_ROUNDS, ai="all_open",
                                 peer="all_open"), map_location="cpu",
                         weights_only=False)
        innate = np_(ppd["innate"])
        pp_op, pp_pred = np_(ppd["op_raw"]), np_(ppd["pred_raw"])
        frd = torch.load(frz_path(1.0, w, 0.2, ORACLE_ROUNDS),
                         map_location="cpu", weights_only=False)
        series = {
            "perfect": (pp_op, pp_pred, ORACLE_ROUNDS),
            "frozen": (np_(frd["op_raw"]), np_(frd["pred_raw"]),
                       ORACLE_ROUNDS),
            "twin": (np_(ppd["twin_raw"]), None, ORACLE_ROUNDS),
        }
        for cond in ("sft", "sft_kl", "icl"):
            d = load_traj(runs[(cond, w)])
            op = np_(d["op_raw"])
            series[cond] = (op, np_(d["pred_raw"]), op.shape[0])
        for cond, (op, pred, nr) in series.items():
            for t in range(nr):
                labels = innate if t == 0 else op[t - 1]
                m = round_metrics(
                    op[t], innate, op[t - 1] if t else None,
                    None if pred is None else pred[t],
                    None if pred is None else labels,
                    pp_op[t])
                rows.append({"part": "C", "condition": cond, "innate_k": 1.0,
                             "w_plat": w, "eps_social": 0.2,
                             "ai_gate_mode": "all_open",
                             "peer_gate_mode": "all_open",
                             "round": t, "horizon": nr, **m})
    write_csv(out_dir / "wu_limit_convergence.csv", rows)
    report_wu_rounds(rows)
    oracle = analyse_oracle_grid(out_dir)
    figure_wu(rows, oracle, out_dir)
    return rows, oracle


def report_wu_rounds(rows):
    """Every condition at rounds 30 and 100; the CPU oracle and the
    offline frozen replay also at 300, where the LLM arms simply do not
    exist. Their absence is printed as '--', never extrapolated."""
    print("\n[qmech] Wu-boundary conditions at fixed rounds "
          "(1-indexed round r reads trajectory index r-1)")
    hdr = f"{'W':>4}  {'condition':<20} {'round':>5}  {'mean':>10} " \
          f"{'sd':>11} {'range':>11} {'clusters':>8}"
    print("[qmech] " + hdr)
    for w in WU_WS:
        for cond in WU_CONDITIONS:
            for r in (30, 100, ORACLE_ROUNDS):
                sel = [x for x in rows if x["condition"] == cond
                       and x["w_plat"] == w and x["round"] == r - 1]
                if not sel:
                    if r == ORACLE_ROUNDS:
                        print(f"[qmech] {w:>4g}  {LABELS[cond]:<20} "
                              f"{r:>5}  {'--':>10} {'--':>11} {'--':>11} "
                              f"{'--':>8}   (no LLM horizon beyond 100)")
                    continue
                x = sel[0]
                print(f"[qmech] {w:>4g}  {LABELS[cond]:<20} {r:>5}  "
                      f"{x['mean']:>10.6f} {x['sd']:>11.3e} "
                      f"{x['range']:>11.3e} {x['n_clusters']:>8d}")


# ----------------------------------------------------------------- output
def write_csv(path, rows):
    if not rows:
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    print(f"[qmech] wrote {path} ({len(rows)} rows)")


def figure_mechanism(per_cell, out_dir):
    """Figure 1: paper-regime mechanism decomposition. Rows = k, columns =
    mean / SD / centered W1 to perfect prediction / prediction MAE, x =
    the social gate, five consistently coloured conditions, one shared
    legend, light reference lines at the initial mean and SD, and no text
    annotations inside the panels."""
    ppd = torch.load(pp_path(0.2, 0.5, 0.0, 30), map_location="cpu",
                     weights_only=False)
    innate = np_(ppd["innate"])
    ini_mean, ini_sd = float(innate.mean()), float(innate.std(ddof=1))

    COLS = [("late_mean", "population mean", ini_mean),
            ("final_sd", "population SD", ini_sd),
            ("final_w1_to_perfect_centered",
             "centered $W_1$ to perfect", None),
            ("late_pred_mae", "prediction MAE", None)]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.2), sharex=True)
    xs = np.arange(len(ESS))
    for i, k in enumerate(KS):
        for j, (field, ylab, ref) in enumerate(COLS):
            ax = axes[i][j]
            if ref is not None:
                ax.axhline(ref, color="#bbbbbb", lw=0.8, zorder=0)
            for cond in CONDITIONS:
                ys = [next(r[field] for r in per_cell
                           if r["condition"] == cond and r["innate_k"] == k
                           and r["eps_social"] == es) for es in ESS]
                if np.all(np.isnan(ys)):
                    continue
                ax.plot(xs, ys, marker="o", ms=4, lw=1.6,
                        color=COLORS[cond], label=LABELS[cond])
            if i == 0:
                ax.set_title(ylab, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"$k={k:g}$", fontsize=11)
            if i == 1:
                ax.set_xticks(xs)
                ax.set_xticklabels([f"{e:g}" for e in ESS])
                ax.set_xlabel(r"$\varepsilon_{\mathrm{social}}$")
            ax.grid(alpha=0.25, lw=0.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        p = out_dir / f"qwen_mechanism_diagnostic.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[qmech] wrote {p}")
    plt.close(fig)


def figure_wu(rows, oracle, out_dir):
    """Figure 2: Wu-limit convergence. Columns = W, rows = mean / SD (log)
    / centered W1 to perfect / prediction MAE, x = round. Rounds 30 and
    100 marked. The CPU oracle and the offline frozen replay run to 300;
    the LLM curves stop at their real 100-round horizon and are NOT
    visually extended. A compact oracle-only heatmap of final SD over
    W x peer condition sits at the bottom."""
    ROWS = [("mean", "population mean", False),
            ("sd", "population SD", True),
            ("w1_to_perfect_centered", "centered $W_1$ to perfect", False),
            ("pred_mae", "prediction MAE", False)]
    fig = plt.figure(figsize=(11.0, 11.5))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1.05],
                          hspace=0.35, wspace=0.22)
    for j, w in enumerate(WU_WS):
        for i, (field, ylab, logy) in enumerate(ROWS):
            ax = fig.add_subplot(gs[i, j])
            for cond in WU_CONDITIONS:
                sel = [r for r in rows if r["condition"] == cond
                       and r["w_plat"] == w]
                if not sel:
                    continue
                sel.sort(key=lambda r: r["round"])
                ys = [r[field] for r in sel]
                if np.all(np.isnan(ys)):
                    continue
                ax.plot([r["round"] for r in sel], ys, lw=1.5,
                        color=COLORS[cond], label=LABELS[cond])
            for mr in WU_MARK_ROUNDS:
                ax.axvline(mr, color="#cccccc", lw=0.8, ls=":", zorder=0)
            if logy:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"$W={w:g}$", fontsize=11)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9)
            if i == len(ROWS) - 1:
                ax.set_xlabel("round")
            ax.grid(alpha=0.25, lw=0.5)
            if i == 0 and j == 0:
                handles, labels = ax.get_legend_handles_labels()

    # compact oracle-only heatmap: final SD over W x peer condition
    axh = fig.add_subplot(gs[4, 0])
    peers = [p[0] for p in ORACLE_PEERS]
    M = np.array([[next(r["final_sd"] for r in oracle
                        if r["w_plat"] == w and r["peer_condition"] == p)
                   for p in peers] for w in ORACLE_WS])
    im = axh.imshow(np.log10(np.maximum(M, 1e-12)), cmap="viridis",
                    aspect="auto")
    axh.set_xticks(range(len(peers)))
    axh.set_xticklabels([ORACLE_PEER_LABEL[p] for p in peers], fontsize=8)
    axh.set_yticks(range(len(ORACLE_WS)))
    axh.set_yticklabels([f"$W={w:g}$" for w in ORACLE_WS], fontsize=8)
    axh.set_ylabel("oracle only: $\\log_{10}$ final SD", fontsize=8)
    cb = fig.colorbar(im, ax=axh, fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=7)

    axl = fig.add_subplot(gs[4, 1])
    axl.axis("off")
    axl.legend(handles, labels, loc="center", ncol=2, frameon=False,
               fontsize=8.5)
    for ext in ("png", "pdf"):
        p = out_dir / f"qwen_wu_limit.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[qmech] wrote {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", action="append", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--part", choices=("a", "c", "all"), default="all")
    args = ap.parse_args()
    roots = args.runs_root or DEFAULT_ROOTS
    args.out.mkdir(parents=True, exist_ok=True)

    if args.part in ("a", "all"):
        analyse_part_a(roots, args.out)
    if args.part in ("c", "all"):
        analyse_part_c(roots, args.out)
    print(f"[qmech] outputs in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
