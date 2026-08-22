#!/usr/bin/env python3
"""ANALYZER for the Section 3 RETENTION wave (pofds3_, 2026-08).

CPU only. Run with OMP_NUM_THREADS=1 (and USE_TF=0 locally); torch is
pinned to one thread here so this never becomes a multithreaded job on a
shared login node.

GATE FIRST. This script computes numbers; check_section3.py decides
whether they mean anything. Pass --verdict <check_section3 --json output>
so the provenance manifest records what was certified.

====================================================================
WHAT IS PLOTTED AND TABULATED, EXACTLY
====================================================================
Every trajectory has EXACTLY 101 points:

  t = 0        the INNATE population, as a real first point
  t = 1..100   op_raw[t-1], the END-OF-ROUND POST-PEER population state
               (the runner appends it after gp.ab_sweep; peers always
               run last)

No within-round intermediate state is ever read, emitted, or mixed in.
The served vector for round t is pred_raw[t-1], which is a different
object from the population and is kept in its own columns.

ONE LOADER FOR CPU AND GPU. sim_perfect_predictor.py and
replay_frozen_offline.py write the SAME schema a GPU run writes --
config / op_raw / twin_raw / pred_raw / innate -- so there is a single
load path here rather than two.

NOTHING IS RE-SIMULATED. The matched no-platform process is the
`twin_raw` tensor the runner and the oracle already write; t=0 is the
`innate` field; the frozen served map is pred_raw[0] of the frozen
propagation artifact. This file contains no dynamics.

====================================================================
THE CONVERGENCE CRITERION (chosen, and why)
====================================================================
A fresh LoRA is trained EVERY round, so the served map -- and through it
the population -- carries an irreducible round-to-round jitter. A
vanishing-step test like |x(t+1) - x(t)| < 1e-6 can therefore never fire,
and using one would label every cell "not converged" regardless of the
science. The diagnostic is LATE-WINDOW MEAN DRIFT instead, over the
equilibrium window (post-peer rounds 81-100):

  half-window drift  D = | mean(m[91..100]) - mean(m[81..90]) |
  fitted trend       T = | OLS slope of m over 81..100 | * 20
  converged_mean  iff  D <= TOL  AND  T <= TOL
  converged_sd    iff  the same two tests on the per-round SD s[t]
  converged       iff  converged_mean AND converged_sd

with TOL = --drift-tol, default 0.002 opinion units.

WHY BOTH TESTS. D catches a cell that is still translating but whose
trend is not linear; T catches a cell that oscillates with near-zero net
drift while a real trend runs underneath. Either alone is fooled by the
other's failure mode.

WHY 0.002. It is ~1.5% of the innate population SD (0.13743800) and
~0.3% of the innate mean (0.63005400), and it sits far below the
arm-to-arm separations the figure has to show (the innate-to-frozen mean
gap in this setup is O(0.1)). It is also comfortably ABOVE the fresh-LoRA
jitter floor, which is reported per cell as noise_floor_mean = median
|m[t] - m[t-1]| over the window, so a reader can see the margin rather
than trust the constant. Every raw quantity (D, T, the window range, the
noise floor, the tolerance) is written to the late CSV, so re-flagging at
another tolerance needs no re-run.

A cell that fails is FLAGGED and labelled a "late-round state (rounds
81-100)", NEVER an equilibrium -- including in the state_label column and
in the convergence report. The frozen propagation baseline is run through
the identical test: if the target itself has not settled, the phrase
"distance to the frozen equilibrium" is not earned, and the report says
so.

====================================================================
DIRECTIONS ARE NOT COMPARABLE AT EQUAL LAMBDA
====================================================================
EQUAL NUMERICAL LAMBDA DOES NOT MEAN EQUAL EFFECTIVE STRENGTH ACROSS KL
DIRECTIONS. KL(p_ref||p_theta) and KL(p_theta||p_ref) have different
curvature at the same coefficient, so "forward lambda 8 vs reverse lambda
8" is not a controlled contrast. Forward is the PRIMARY ladder; reverse
lambda in {1, 8} is a LABELLED ROBUSTNESS CHECK.

This tool therefore emits full curves and the ACHIEVED distance to the
frozen model for every arm, tags every row with direction_role, and
deliberately provides no same-lambda cross-direction summary row for a
headline to be read off. Compare arms by what they ACHIEVED, not by the
number in their name.

====================================================================
Outputs (--out, default notes/pofd/section3_analysis)
====================================================================
  section3_per_round.csv         101 rows per cell
  section3_late_equilibrium.csv  1 row per cell
  section3_provenance.json       reuse-manifest passthrough + resolution
  section3_convergence.txt       criterion, per-cell verdicts, cautions
  section3_grid_exploratory.png/.pdf   complete-grid figure, NO titles

Usage
  OMP_NUM_THREADS=1 python analyze_section3.py \\
      --reuse-manifest notes/pofd/section3/reuse_manifest.json \\
      --verdict notes/pofd/section3/check_verdict.json \\
      --out notes/pofd/section3_analysis
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(tempfile.gettempdir(), "perfsim-plot-cache"))

import numpy as np
import torch

torch.set_num_threads(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = (os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals() else os.getcwd())
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _load_sibling(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- REUSED, not re-implemented ------------------------------------
# CS  : the Section 3 gate -- tag grammar, the conceptual grid, the
#       reuse-manifest reader, served_map_stats, every pinned constant.
#       One definition of "what a Section 3 cell is", shared by the gate
#       and the analyzer.
# AKD : the archived forward-vs-reverse analyzer. UNMODIFIED. Reused for
#       w1() (the 1-Wasserstein between equal-size empirical
#       distributions) and _num() (the tag number grammar), which are
#       pure and exactly right, and for its STYLE / style_for() palette
#       so the two waves' figures stay visually consistent.
#
#       NOT reused: AKD.load_gpu / AKD.load_cpu. Both truncate to the
#       module global AKD.ROUNDS = 10 and between them return only 2 of
#       the 4 tensors this wave needs (no twin_raw, no innate, no
#       config). Rebinding another module's global to 100 to borrow a
#       loader is a worse dependency than one honest loader -- and since
#       CPU and GPU artifacts share a schema, ONE loader is strictly
#       simpler than the two AKD carries.
CS = _load_sibling("check_section3", "check_section3.py")
AKD = _load_sibling("analyze_kl_direction", "analyze_kl_direction.py")

w1 = AKD.w1
_num = AKD._num

N_AGENTS = CS.N_AGENTS
ROUNDS = CS.PROD_ROUNDS
LATE_LO, LATE_HI = CS.LATE_LO, CS.LATE_HI
ENVS = CS.ENVS
ENV_LABEL = CS.ENV_LABEL
MODELS = CS.MODELS
DEFAULT_DRIFT_TOL = 0.002

# innate reference values, for the tolerance rationale in the report
INNATE_MEAN = 0.6300539970397949
INNATE_SD = 0.1374379992485046

PP_SEARCH = ("notes/pofd/perfect_prediction_anchored_baseline",
             "notes/pofd/perfect_prediction_k_sweep",
             "notes/pofd/perfect_prediction")
FROZEN_DIR = "notes/pofd/frozen_replay"


# ------------------------------------------------------------ loading
def load_artifact(path, rounds=ROUNDS):
    """ONE loader for CPU endpoints and GPU runs -- they share a schema.

    `path` may be a run directory (uses <dir>/trajectory.pt) or a .pt
    file. Returns a dict of float32 numpy arrays truncated to `rounds`,
    plus the config and the raw round count. Raises with the path named.
    """
    p = Path(path)
    f = p / "trajectory.pt" if p.is_dir() else p
    if not f.exists():
        raise FileNotFoundError(f"{f} does not exist")
    d = torch.load(f, map_location="cpu", weights_only=False)
    out = {"path": str(f), "config": d.get("config") or {}}
    if p.is_dir():
        cj = p / "config.json"
        if cj.exists():
            out["config"] = json.loads(cj.read_text())
    n_raw = None
    for key in ("op_raw", "pred_raw", "twin_raw"):
        t = d.get(key)
        if not torch.is_tensor(t) or t.numel() == 0:
            out[key] = None
            continue
        t = t.float()
        n_raw = t.shape[0] if n_raw is None else n_raw
        out[key] = t[:rounds].numpy()
    inn = d.get("innate")
    out["innate"] = inn.float().numpy() if torch.is_tensor(inn) else None
    out["n_rounds_raw"] = int(n_raw) if n_raw is not None else -1
    if out["op_raw"] is None or out["op_raw"].shape[0] < rounds:
        got = None if out["op_raw"] is None else out["op_raw"].shape[0]
        raise ValueError(f"{f}: op_raw has {got} round(s), need {rounds}")
    if out["innate"] is None:
        raise ValueError(f"{f}: no innate vector -- it is t=0 of the "
                         f"trajectory and cannot be reconstructed")
    return out


def read_parse_fail(run_dir, rounds=ROUNDS):
    gz = Path(run_dir) / "raw_gen_log.json.gz"
    if not gz.exists():
        return [float("nan")] * rounds
    vals = []
    with gzip.open(gz, "rt") as fh:
        for line in fh:
            if line.strip():
                v = json.loads(line).get("parse_fail_frac")
                vals.append(float("nan") if v is None else float(v))
    vals = vals[:rounds]
    return vals + [float("nan")] * (rounds - len(vals))


TEL_KEYS = ("l_init", "l_cc", "l_c0", "l_0c", "l_00",
            "grad_norm0", "grad_kl_norm0", "grad_cos0", "grad_ratio0")


def read_telemetry(run_dir, rounds=ROUNDS):
    p = Path(run_dir) / "telemetry.json"
    rows = []
    if p.exists():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = rows[:rounds]
    out = {k: [float("nan")] * rounds for k in TEL_KEYS}
    for i, r in enumerate(rows):
        for k in TEL_KEYS:
            v = r.get(k)
            if v is not None:
                try:
                    out[k][i] = float(v)
                except (TypeError, ValueError):
                    pass
    return out


# ------------------------------------------------------------ metrics
def aligned(a, b):
    """Agent-aligned (per-index) mean absolute distance. Unlike W1 this
    notices a population that has the right DISTRIBUTION but the wrong
    agents holding it."""
    return float(np.abs(np.asarray(a) - np.asarray(b)).mean())


def rmse(a, b):
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean()))


def _nanmean(v):
    """np.nanmean over a possibly all-NaN slice, without the warning. An
    all-NaN slice is the NORMAL case for grad_kl_norm0 on the lambda = 0
    arm (there is no anchor term to record), so it must not look like an
    error."""
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def ols_slope(y):
    y = np.asarray(y, dtype=float)
    if y.size < 2 or not np.isfinite(y).all():
        return float("nan")
    x = np.arange(y.size, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def convergence(series, tol):
    """(converged, half-window drift, |slope|*window, range, noise floor).

    See the module docstring: late-window MEAN DRIFT, not a vanishing
    step, because a fresh LoRA every round puts a floor under the step.
    """
    y = np.asarray(series, dtype=float)
    n = y.size
    if n < 4 or not np.isfinite(y).all():
        return False, float("nan"), float("nan"), float("nan"), float("nan")
    half = n // 2
    drift = abs(float(y[half:].mean() - y[:half].mean()))
    trend = abs(ols_slope(y)) * n
    rng = float(y.max() - y.min())
    floor = float(np.median(np.abs(np.diff(y)))) if n > 1 else float("nan")
    return bool(drift <= tol and trend <= tol), drift, trend, rng, floor


def served_stats(vec):
    """CS.served_map_stats -- one definition of served-map degeneracy,
    shared with the gate so the CSV and the gate's report cannot drift."""
    return CS.served_map_stats(vec)


def served_vs_frozen(srv, fz):
    if srv is None or fz is None:
        return float("nan"), float("nan"), float("nan")
    d = w1(srv, fz)
    agree = float((np.asarray(srv) == np.asarray(fz)).mean())
    corr = float("nan")
    if np.std(srv) > 0 and np.std(fz) > 0:
        corr = float(np.corrcoef(srv, fz)[0, 1])
    return d, agree, corr


# -------------------------------------------------------- baselines
def _pp_name(k, w, rounds):
    return (f"pp_k{_num(k)}_w{_num(w)}_eaopen_esopen_sw1"
            f"_s{CS.SEED}_r{rounds}.pt")


def _frz_name(k, w, rounds):
    return (f"frz_k{_num(k)}_w{_num(w)}_eaopen_esopen_sw1"
            f"_s{CS.SEED}_r{rounds}.pt")


def _pp_command(k, w, rounds):
    return (f"OMP_NUM_THREADS=1 python experiments/scripts/cluster_pipelines/"
            f"sim_perfect_predictor.py --innate-k {k:g} --w-plat {w:g} "
            f"--eps-social 0.2 --eps-ai 1 --ai-gate-mode all_open "
            f"--peer-gate-mode all_open --sweeps 1 --gamma 0 "
            f"--rounds {rounds} --seed {CS.SEED}")


def _frz_command(k, w, rounds, model):
    src = CS.FROZEN_SOURCE.get(model, "<frozen K=D=0 H100 run dir>")
    return (f"OMP_NUM_THREADS=1 python experiments/scripts/cluster_pipelines/"
            f"replay_frozen_offline.py --from-run <ROOT>/{src} "
            f"--innate-k {k:g} --w-plat {w:g} --eps-social 0.2 --eps-ai 1 "
            f"--ai-gate-mode all_open --peer-gate-mode all_open "
            f"--rounds {rounds} --seed {CS.SEED} "
            f"--out-dir notes/pofd/frozen_replay/{model}")


def resolve_pp(k, w, rounds, search, notes):
    """The matched perfect-prediction endpoint for one environment.

    A LONGER artifact is accepted as a prefix and said so out loud: the
    CPU oracle seeds its peer generator once and consumes it
    sequentially, and the r10 and r300 artifacts were verified
    bit-identical over their first 10 rounds. That is a measurement, not
    an assumption, but it is still recorded.
    """
    want = _pp_name(k, w, rounds)
    for d in search:
        p = Path(REPO) / d / want
        if p.exists():
            return str(p), None
    cands = []
    for d in search:
        dd = Path(REPO) / d
        if not dd.is_dir():
            continue
        for p in sorted(dd.glob(_pp_name(k, w, "*"))):
            try:
                r = int(p.stem.rsplit("_r", 1)[1])
            except (IndexError, ValueError):
                continue
            if r > rounds:
                cands.append((r, p))
    if cands:
        r, p = min(cands)
        notes.append(f"[s3] PREFIX: perfect prediction (beta={w:g}, k={k:g}) "
                     f"uses the first {rounds} of {r} rounds from {p.name}; "
                     f"{want} does not exist. CPU endpoint streams are seeded "
                     f"once and consumed sequentially (verified bit-identical "
                     f"between the r10 and r300 artifacts).")
        return str(p), f"prefix of r{r}"
    return None, want


def resolve_frozen(model, k, w, rounds, frozen_dir, explicit, notes):
    """The matched frozen propagation endpoint for (model, environment).

    THE NAMING HAZARD, HANDLED. frz_* filenames encode k / W / gates /
    sweeps / seed / rounds but NOT the model, and Section 3 needs SIX of
    them (2 checkpoints x 3 environments), so the two checkpoints collide
    pairwise on filename. Binding is therefore by the artifact's OWN
    config["base_model"], never by path. An explicit --frozen-map entry
    wins; otherwise the directory is scanned RECURSIVELY (so per-model
    subdirs work) and every candidate is opened and identified.
    """
    if (model, k, w) in explicit:
        return explicit[(model, k, w)], None
    want = _frz_name(k, w, rounds)
    root = Path(REPO) / frozen_dir if not os.path.isabs(frozen_dir) \
        else Path(frozen_dir)
    hits, longer = [], []
    if root.is_dir():
        for p in sorted(root.rglob("frz_*.pt")):
            try:
                a = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:                             # noqa: BLE001
                continue
            c = a.get("config") or {}
            slug = next((s for s, b in MODELS.items()
                         if b == c.get("base_model")), None)
            if slug != model:
                continue
            if not (CS._feq(c.get("innate_k"), k)
                    and CS._feq(c.get("w_plat"), w)):
                continue
            nr = int(c.get("rounds", -1))
            if nr == rounds:
                hits.append(p)
            elif nr > rounds:
                longer.append((nr, p))
    if len(hits) > 1:
        notes.append(f"[s3] AMBIGUOUS: {len(hits)} frozen artifacts claim "
                     f"({model}, beta={w:g}, k={k:g}): "
                     f"{[str(h) for h in hits]}")
        return None, f"ambiguous: {[h.name for h in hits]}"
    if hits:
        return str(hits[0]), None
    if longer:
        nr, p = min(longer)
        notes.append(f"[s3] PREFIX: frozen propagation ({model}, beta={w:g}, "
                     f"k={k:g}) uses the first {rounds} of {nr} rounds from "
                     f"{p}; {want} does not exist.")
        return str(p), f"prefix of r{nr}"
    return None, want


# ------------------------------------------------------------- cells
def resolve_cells(roots, reuse_manifest, notes, errs):
    """conceptual slot -> {run_dir, tag, source_kind, reuse, ...}."""
    roots = [Path(r) for r in roots if Path(r).is_dir()]
    out_lines = []
    mf_by_tag, mf_by_slot, _ok = CS.load_manifest(reuse_manifest, out_lines)
    for l in out_lines:
        notes.append(l.replace("[check_s3]", "[s3]"))
    found = {}
    for root in roots:
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            slot, tag_errs = CS.parse_tag(p.name)
            if slot is None or tag_errs or slot["smoke"]:
                continue
            key = (slot["model"], slot["arm"], slot["beta"], slot["k"])
            if key in found:
                errs.append(f"two run dirs claim {key}: "
                            f"{found[key]['run_dir']} and {p}")
                continue
            found[key] = {"run_dir": str(p), "tag": p.name,
                          "source_kind": CS.SOURCE_S3_NEW, "reuse": False}
    for cell in mf_by_slot.values():
        if not CS._manifest_says_reuse(cell):
            continue
        model = str(CS._mf_get(cell, CS._MF_MODEL, ""))
        arm = str(CS._mf_get(cell, CS._MF_ARM, ""))
        beta = CS._as_float(CS._mf_get(cell, CS._MF_BETA, None))
        k = CS._as_float(CS._mf_get(cell, CS._MF_K, None))
        tag = CS._mf_get(cell, CS._MF_TAG, None)
        d = CS._mf_get(cell, CS._MF_DIR, None)
        key = (model, arm, beta, k)
        cands = []
        if d not in (None, CS.ABSENT):
            pp = Path(str(d))
            cands.append(pp if pp.is_absolute() else Path(REPO) / pp)
        if tag not in (None, CS.ABSENT):
            cands += [r / str(tag) for r in roots]
        hit = next((c for c in cands if (c / "trajectory.pt").exists()), None)
        if hit is None:
            errs.append(f"reuse cell {tag!r} for slot {key} is not resolvable "
                        f"under {[str(r) for r in roots]}")
            continue
        if key in found:
            errs.append(f"slot {key} is claimed by BOTH a new run "
                        f"({found[key]['tag']}) and a reuse entry ({tag})")
            continue
        found[key] = {"run_dir": str(hit), "tag": str(tag),
                      "source_kind": CS.SOURCE_QWU_REUSE, "reuse": True}
    return found, mf_by_tag


def arm_meta(arm):
    style, lam, direction = CS.arm_semantics(arm)
    role = "robustness_check" if direction == "reverse" else "primary_ladder"
    return lam, (direction or "none"), role


# -------------------------------------------------------------- rows
PER_ROUND_COLS = [
    "model", "arm", "beta", "k", "env", "direction", "lam", "direction_role",
    "reuse", "source_kind", "population_update", "run_tag", "t",
    "pop_mean", "pop_sd",
    "w1_innate", "aligned_innate", "rmse_innate",
    "w1_twin", "aligned_twin",
    "w1_pp", "aligned_pp",
    "w1_frozen_t", "aligned_frozen_t",
    "w1_frozen_eq", "aligned_frozen_eq",
    "w1_consec", "aligned_consec",
    "pred_mean", "pred_sd", "pred_n_distinct", "pred_mode_share",
    "pred_top3_share", "pred_eff_modes",
    "served_w1_frozen", "served_agree_frozen", "served_corr_frozen",
    "parse_fail_frac",
] + list(TEL_KEYS)

LATE_COLS = [
    "model", "arm", "beta", "k", "env", "direction", "lam", "direction_role",
    "reuse", "source_kind", "population_update", "run_tag", "gpu_name",
    "transformers_version", "n_rounds_raw",
    "late_lo", "late_hi", "drift_tol",
    "late_pop_mean", "late_pop_sd", "late_mean_jitter",
    "late_mean_drift", "late_mean_trend", "late_mean_range",
    "late_mean_noise_floor",
    "late_sd_drift", "late_sd_trend", "late_sd_range", "late_sd_noise_floor",
    "late_mean_slope", "late_sd_slope",
    "converged_mean", "converged_sd", "converged", "state_label",
    "late_w1_innate", "late_aligned_innate",
    "late_w1_twin", "late_aligned_twin",
    "late_w1_pp", "late_aligned_pp",
    "late_w1_frozen_eq", "late_aligned_frozen_eq",
    "late_w1_consec",
    "late_pred_sd", "late_pred_n_distinct", "late_pred_mode_share",
    "late_pred_top3_share", "late_pred_eff_modes",
    "late_served_w1_frozen", "late_served_agree_frozen",
    "late_served_corr_frozen",
    "parse_fail_max",
    "late_l_init", "late_grad_norm0", "late_grad_kl_norm0",
    "pp_source", "frozen_source", "frozen_eq_converged",
    "provenance_deviations",
]


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


# ------------------------------------------------------------ figure
COLS = ("mean", "sd", "dfz", "modes")
COL_LABEL = {"mean": "population mean", "sd": "population SD",
             "dfz": "$W_1$ to frozen equilibrium",
             "modes": "served effective modes (log)"}
COL_KEY = {"mean": "pop_mean", "sd": "pop_sd", "dfz": "w1_frozen_eq",
           "modes": "pred_eff_modes"}


def exploratory_figure(series, out_dir, drift_tol):
    """Complete-grid exploratory figure. NO TITLES (project convention);
    facets are identified by axis labels and left-edge annotations.

    ABSOLUTE READABLE SCALES. Every column shares ONE y-range across all
    six rows, computed from the data actually plotted. That is what stops
    matplotlib from switching a panel to offset notation
    (e.g. "1e-7 + 6.3005e-1" on the perfect-prediction mean, which is
    mean-preserving to 1e-7 and would otherwise render as a full-height
    zoom on float noise). The served-modes column is log-scaled instead,
    because perfect prediction serves ~500 effective modes while a
    collapsed arm serves ~2, and a shared LINEAR range there would flatten
    every trained arm onto the axis.
    """
    rows = [(m, e) for m in MODELS for e in ENVS]
    fig, axes = plt.subplots(len(rows), len(COLS),
                             figsize=(15.0, 2.5 * len(rows)),
                             sharex=True, squeeze=False)
    t = np.arange(0, ROUNDS + 1)
    handles = {}
    lims = {c: [np.inf, -np.inf] for c in COLS}

    def _track(col, y):
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if col == "modes":
            y = y[y > 0]
        if y.size:
            lims[col][0] = min(lims[col][0], float(y.min()))
            lims[col][1] = max(lims[col][1], float(y.max()))

    for ri, (model, env) in enumerate(rows):
        for ci, what in enumerate(COLS):
            ax = axes[ri][ci]
            drew = False
            for key, s in sorted(series.items()):
                if key[0] != model or (key[2], key[3]) != env:
                    continue
                lam, direction, _role = arm_meta(key[1])
                st = style_for_arm(direction, lam)
                y = s[COL_KEY[what]]
                lbl = arm_label(key[1])
                ln, = ax.plot(t[:len(y)], y, label=lbl, **st)
                handles.setdefault(lbl, ln)
                _track(what, y)
                drew = True
            for kind, s in sorted(series.get(("__base__", model, env),
                                             {}).items()):
                st = dict(AKD.STYLE[kind])
                y = s[COL_KEY[what]]
                lbl = {"perfect": "perfect prediction",
                       "frozen": r"frozen ($\lambda\to\infty$)"}[kind]
                ln, = ax.plot(t[:len(y)], y, label=lbl, **st)
                handles.setdefault(lbl, ln)
                _track(what, y)
                drew = True
            ax.axvspan(LATE_LO, LATE_HI, color="0.85", zorder=0, lw=0)
            ax.grid(alpha=.25, lw=.6)
            ax.set_xlim(0, ROUNDS)
            if not drew:
                ax.annotate("no data", xy=(.5, .5),
                            xycoords="axes fraction", ha="center",
                            va="center", fontsize=9, color="0.45")
                ax.set_yticks([])
            if ri == len(rows) - 1:
                ax.set_xlabel("round $t$  (0 = innate; $t\\geq1$ post-peer)")
            if ri == 0:
                ax.annotate(COL_LABEL[what], xy=(.5, 1.04),
                            xycoords="axes fraction", ha="center",
                            va="bottom", fontsize=10)
        axes[ri][0].set_ylabel(
            f"{model}\n" + r"$\beta=%g$, $k=%g$" % env + f"\n({ENV_LABEL[env]})",
            fontsize=8.5)

    # one absolute range per column, applied to every row
    for ci, what in enumerate(COLS):
        lo, hi = lims[what]
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        for ri in range(len(rows)):
            ax = axes[ri][ci]
            if what == "modes":
                ax.set_yscale("log")
                ax.set_ylim(max(1.0, lo * 0.8), hi * 1.25)
            else:
                pad = 0.05 * (hi - lo) if hi > lo else max(1e-3, abs(hi) * .05)
                ax.set_ylim(lo - pad, hi + pad)
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    order = sorted(handles)
    fig.legend([handles[k] for k in order], order, loc="lower center",
               ncol=6, frameon=False, fontsize=8,
               bbox_to_anchor=(.5, -0.012))
    fig.text(.005, .002,
             f"shaded band = equilibrium window (rounds {LATE_LO}-{LATE_HI}); "
             f"convergence flag uses late-window mean drift, tol={drift_tol:g}",
             fontsize=7.5, ha="left")
    fig.tight_layout(rect=(0, .055, 1, .985))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"section3_grid_exploratory.{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# Forward = Blues solid squares, reverse = Reds dash-dot triangles,
# lambda = 0 green circles: AKD.style_for's palette, EXTENDED with a
# shade ramp over this wave's six forward rungs (AKD.SHADE only knows
# {0.1, 1, 10}, so every Section 3 rung would otherwise draw in one
# colour). AKD is not modified.
S3_LAMBDAS = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0)


def style_for_arm(direction, lam):
    if lam == 0:
        return dict(AKD.STYLE["sft0"])
    st = AKD.style_for("fwd" if direction == "forward" else "rev", lam)
    idx = S3_LAMBDAS.index(lam) if lam in S3_LAMBDAS else 2
    shade = 0.34 + 0.60 * idx / max(1, len(S3_LAMBDAS) - 1)
    cmap = plt.get_cmap("Blues" if direction == "forward" else "Reds")
    st["color"] = cmap(shade)
    st["ms"] = 0.0
    st["marker"] = None
    st["lw"] = 1.5 if direction == "forward" else 1.1
    return st


def arm_label(arm):
    lam, direction, _ = arm_meta(arm)
    if lam == 0:
        return r"SFT $\lambda=0$"
    d = "forward" if direction == "forward" else "reverse"
    suffix = "" if direction == "forward" else " (robustness)"
    return rf"{d} $\lambda={lam:g}${suffix}"


# -------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Section 3 retention analyzer (CPU only)")
    ap.add_argument("--roots", nargs="*", default=CS.DEFAULT_ROOTS)
    ap.add_argument("--reuse-manifest",
                    default="notes/pofd/section3/reuse_manifest.json")
    ap.add_argument("--verdict", default=None,
                    help="check_section3.py --json output, copied into the "
                         "provenance manifest")
    ap.add_argument("--pp-search", nargs="*", default=list(PP_SEARCH))
    ap.add_argument("--frozen-dir", default=FROZEN_DIR)
    ap.add_argument("--frozen-map", nargs="*", default=[], metavar="M:B:K=PATH",
                    help="explicit frozen binding, e.g. "
                         "qwen3_8b:0.5:1=notes/pofd/frozen_replay/qwen3_8b/"
                         "frz_k1_w0p5_eaopen_esopen_sw1_s0_r100.pt")
    ap.add_argument("--out", default="notes/pofd/section3_analysis")
    ap.add_argument("--drift-tol", type=float, default=DEFAULT_DRIFT_TOL)
    ap.add_argument("--allow-missing", action="store_true",
                    help="analyze a deliberately partial grid. OFF by "
                         "default: a silently short grid reads as a result")
    args = ap.parse_args()

    notes, errs = [], []
    out_dir = Path(args.out) if os.path.isabs(args.out) \
        else Path(REPO) / args.out

    explicit = {}
    for spec in args.frozen_map:
        try:
            lhs, path = spec.split("=", 1)
            m, b, k = lhs.split(":")
            explicit[(m, float(k), float(b))] = path
        except ValueError:
            print(f"[s3] usage error: --frozen-map entry {spec!r} is not "
                  f"MODEL:BETA:K=PATH", file=sys.stderr)
            return 2

    cells, mf_by_tag = resolve_cells(args.roots, args.reuse_manifest,
                                     notes, errs)
    slots = CS.conceptual_grid()
    missing_cells = [s for s in slots if s not in cells]

    # ---- baselines, one per environment (+ per model for frozen) -----
    pp_art, fz_art, missing_bases = {}, {}, []
    for env in ENVS:
        w, k = env
        path, why = resolve_pp(k, w, ROUNDS, args.pp_search, notes)
        if path is None:
            missing_bases.append(
                f"perfect prediction (beta={w:g}, k={k:g}): {why} not found "
                f"under {args.pp_search}\n      generate with: "
                f"{_pp_command(k, w, ROUNDS)}")
        else:
            pp_art[env] = load_artifact(path)
            pp_art[env]["_src"] = path
        for model in MODELS:
            path, why = resolve_frozen(model, k, w, ROUNDS, args.frozen_dir,
                                       explicit, notes)
            if path is None:
                missing_bases.append(
                    f"frozen propagation ({model}, beta={w:g}, k={k:g}): "
                    f"{why} not found under {args.frozen_dir}\n"
                    f"      generate with: {_frz_command(k, w, ROUNDS, model)}")
            else:
                fz_art[(model, env)] = load_artifact(path)
                fz_art[(model, env)]["_src"] = path

    hard = bool(errs) or (bool(missing_cells or missing_bases)
                          and not args.allow_missing)
    if errs:
        print("[s3] HARD FAIL: cell resolution errors --", file=sys.stderr)
        for e in errs:
            print(f"        {e}", file=sys.stderr)
    if missing_bases:
        print(f"[s3] HARD FAIL: {len(missing_bases)} of "
              f"{len(ENVS) * (1 + len(MODELS))} CPU endpoint artifacts are "
              f"missing (3 perfect-prediction + 6 frozen propagation are "
              f"required) --", file=sys.stderr)
        for m in missing_bases:
            print(f"    {m}", file=sys.stderr)
    if missing_cells:
        print(f"[s3] {'HARD FAIL' if not args.allow_missing else 'WARNING'}: "
              f"{len(missing_cells)} of {len(slots)} conceptual cells absent "
              f"--", file=sys.stderr)
        for s in missing_cells:
            print(f"        {CS.slot_tag(*s)}", file=sys.stderr)
    if hard:
        print("[s3] refusing to emit CSVs or a figure over an incomplete "
              "grid. Complete the wave and the endpoints, or pass "
              "--allow-missing to analyze a deliberately partial grid.",
              file=sys.stderr)
        return 2
    if missing_cells or missing_bases:
        print(f"[s3] WARNING: PARTIAL GRID -- {len(slots) - len(missing_cells)}"
              f"/{len(slots)} cells, {len(missing_bases)} endpoint(s) missing. "
              f"Every output below is partial and must be labelled as such.")

    # ---- frozen equilibrium targets, and whether they are equilibria --
    fz_eq, fz_conv = {}, {}
    for key, a in fz_art.items():
        op = a["op_raw"]
        win = op[LATE_LO - 1:LATE_HI]
        fz_eq[key] = win.mean(axis=0)
        conv, *_ = convergence(op[LATE_LO - 1:LATE_HI].mean(axis=1),
                               args.drift_tol)
        fz_conv[key] = bool(conv)
        if not conv:
            notes.append(
                f"[s3] CAUTION: the frozen propagation baseline for "
                f"{key[0]} (beta={key[1][0]:g}, k={key[1][1]:g}) has NOT "
                f"converged by the same diagnostic. Its rounds "
                f"{LATE_LO}-{LATE_HI} mean state is a LATE-ROUND STATE, so "
                f"'distance to the frozen equilibrium' is really 'distance "
                f"to the frozen late-round state' for every arm measured "
                f"against it.")

    # ---- per-cell computation ----------------------------------------
    per_round_rows, late_rows, series = [], [], {}
    for slot in slots:
        if slot not in cells:
            continue
        model, arm, beta, k = slot
        env = (beta, k)
        info = cells[slot]
        lam, direction, role = arm_meta(arm)
        try:
            a = load_artifact(info["run_dir"])
        except Exception as e:                            # noqa: BLE001
            print(f"[s3] HARD FAIL: {info['tag']}: {e}", file=sys.stderr)
            return 2
        cfg = a["config"]
        op, pred, twin, innate = (a["op_raw"], a["pred_raw"], a["twin_raw"],
                                  a["innate"])
        pp = pp_art.get(env)
        fz = fz_art.get((model, env))
        fzeq = fz_eq.get((model, env))
        fz_served = fz["pred_raw"][0] if fz is not None \
            and fz["pred_raw"] is not None else None
        pfail = read_parse_fail(info["run_dir"])
        tel = read_telemetry(info["run_dir"])

        # t = 0 is the innate population, as a real point; t = 1..100 are
        # post-peer states. Exactly 101 points, no intermediates.
        states = [innate] + [op[i] for i in range(ROUNDS)]
        pp_states = ([pp["innate"]] + [pp["op_raw"][i] for i in range(ROUNDS)]
                     if pp is not None else None)
        fz_states = ([fz["innate"]] + [fz["op_raw"][i] for i in range(ROUNDS)]
                     if fz is not None else None)
        tw_states = ([innate] + [twin[i] for i in range(ROUNDS)]
                     if twin is not None and twin.shape[0] >= ROUNDS else None)
        assert len(states) == ROUNDS + 1, "trajectory must have 101 points"

        base = {"model": model, "arm": arm, "beta": beta, "k": k,
                "env": ENV_LABEL[env], "direction": direction, "lam": lam,
                "direction_role": role, "reuse": info["reuse"],
                "source_kind": info["source_kind"],
                "population_update": cfg.get("population_update"),
                "run_tag": info["tag"]}
        cur = {c: [] for c in ("pop_mean", "pop_sd", "w1_frozen_eq",
                               "pred_eff_modes")}
        for t in range(ROUNDS + 1):
            x = states[t]
            r = dict(base)
            r["t"] = t
            r["pop_mean"] = float(x.mean())
            r["pop_sd"] = float(x.std())
            r["w1_innate"] = w1(x, innate)
            r["aligned_innate"] = aligned(x, innate)
            r["rmse_innate"] = rmse(x, innate)
            if tw_states is not None:
                r["w1_twin"] = w1(x, tw_states[t])
                r["aligned_twin"] = aligned(x, tw_states[t])
            if pp_states is not None:
                r["w1_pp"] = w1(x, pp_states[t])
                r["aligned_pp"] = aligned(x, pp_states[t])
            if fz_states is not None:
                r["w1_frozen_t"] = w1(x, fz_states[t])
                r["aligned_frozen_t"] = aligned(x, fz_states[t])
            if fzeq is not None:
                r["w1_frozen_eq"] = w1(x, fzeq)
                r["aligned_frozen_eq"] = aligned(x, fzeq)
            if t > 0:
                r["w1_consec"] = w1(x, states[t - 1])
                r["aligned_consec"] = aligned(x, states[t - 1])
                srv = pred[t - 1] if pred is not None else None
                if srv is not None:
                    st = served_stats(srv)
                    r["pred_mean"] = st["mean"]
                    r["pred_sd"] = st["sd"]
                    r["pred_n_distinct"] = st["n_distinct"]
                    r["pred_mode_share"] = st["mode_share"]
                    r["pred_top3_share"] = st["top3_share"]
                    r["pred_eff_modes"] = st["eff_modes"]
                    d_, ag, co = served_vs_frozen(srv, fz_served)
                    r["served_w1_frozen"] = d_
                    r["served_agree_frozen"] = ag
                    r["served_corr_frozen"] = co
                r["parse_fail_frac"] = pfail[t - 1]
                for kk in TEL_KEYS:
                    r[kk] = tel[kk][t - 1]
            per_round_rows.append(r)
            cur["pop_mean"].append(r["pop_mean"])
            cur["pop_sd"].append(r["pop_sd"])
            cur["w1_frozen_eq"].append(r.get("w1_frozen_eq", float("nan")))
            cur["pred_eff_modes"].append(r.get("pred_eff_modes", float("nan")))
        series[slot] = cur

        # ---- late window -------------------------------------------------
        lo, hi = LATE_LO, LATE_HI
        m_late = np.array(cur["pop_mean"][lo:hi + 1], dtype=float)
        s_late = np.array(cur["pop_sd"][lo:hi + 1], dtype=float)
        cm, dm, tm, rm, fm = convergence(m_late, args.drift_tol)
        cs_, ds, ts, rs, fs = convergence(s_late, args.drift_tol)
        win_states = states[lo:hi + 1]
        win_pred = pred[lo - 1:hi] if pred is not None else None
        mean_state = np.mean(np.stack(win_states, axis=0), axis=0)
        lr = dict(base)
        lr.update({
            "gpu_name": (cfg.get("hardware") or {}).get("gpu_name"),
            "transformers_version":
                (cfg.get("hardware") or {}).get("transformers_version"),
            "n_rounds_raw": a["n_rounds_raw"],
            "late_lo": lo, "late_hi": hi, "drift_tol": args.drift_tol,
            "late_pop_mean": float(m_late.mean()),
            "late_pop_sd": float(s_late.mean()),
            "late_mean_jitter": float(m_late.std()),
            "late_mean_drift": dm, "late_mean_trend": tm,
            "late_mean_range": rm, "late_mean_noise_floor": fm,
            "late_sd_drift": ds, "late_sd_trend": ts,
            "late_sd_range": rs, "late_sd_noise_floor": fs,
            "late_mean_slope": ols_slope(m_late),
            "late_sd_slope": ols_slope(s_late),
            "converged_mean": cm, "converged_sd": cs_,
            "converged": bool(cm and cs_),
            "state_label": ("equilibrium" if (cm and cs_)
                            else f"late-round state (rounds {lo}-{hi}), "
                                 f"NOT an equilibrium"),
            "late_w1_innate": w1(mean_state, innate),
            "late_aligned_innate": aligned(mean_state, innate),
            "late_w1_consec": _nanmean(
                [w1(states[t], states[t - 1]) for t in range(lo, hi + 1)]),
            "parse_fail_max": float(np.nanmax(pfail)) if np.isfinite(pfail).any() else
                float("nan"),
            "late_l_init": _nanmean(tel["l_init"][lo - 1:hi]),
            "late_grad_norm0": _nanmean(tel["grad_norm0"][lo - 1:hi]),
            "late_grad_kl_norm0":
                _nanmean(tel["grad_kl_norm0"][lo - 1:hi]),
            "pp_source": pp["_src"] if pp is not None else None,
            "frozen_source": fz["_src"] if fz is not None else None,
            "frozen_eq_converged": fz_conv.get((model, env)),
            "provenance_deviations": "",
        })
        if tw_states is not None:
            tw_mean = np.mean(np.stack(tw_states[lo:hi + 1], axis=0), axis=0)
            lr["late_w1_twin"] = w1(mean_state, tw_mean)
            lr["late_aligned_twin"] = aligned(mean_state, tw_mean)
        if pp_states is not None:
            pp_mean = np.mean(np.stack(pp_states[lo:hi + 1], axis=0), axis=0)
            lr["late_w1_pp"] = w1(mean_state, pp_mean)
            lr["late_aligned_pp"] = aligned(mean_state, pp_mean)
        if fzeq is not None:
            lr["late_w1_frozen_eq"] = w1(mean_state, fzeq)
            lr["late_aligned_frozen_eq"] = aligned(mean_state, fzeq)
        if win_pred is not None:
            per = [served_stats(win_pred[i]) for i in range(win_pred.shape[0])]
            for src, dst in (("sd", "late_pred_sd"),
                             ("n_distinct", "late_pred_n_distinct"),
                             ("mode_share", "late_pred_mode_share"),
                             ("top3_share", "late_pred_top3_share"),
                             ("eff_modes", "late_pred_eff_modes")):
                lr[dst] = float(np.mean([p[src] for p in per]))
            trio = [served_vs_frozen(win_pred[i], fz_served)
                    for i in range(win_pred.shape[0])]
            lr["late_served_w1_frozen"] = float(np.mean([t_[0] for t_ in trio]))
            lr["late_served_agree_frozen"] = float(
                np.mean([t_[1] for t_ in trio]))
            lr["late_served_corr_frozen"] = _nanmean([t_[2] for t_ in trio])
        late_rows.append(lr)

    # ---- REFERENCE arms -------------------------------------------------
    # Perfect prediction and the frozen propagation are written into the
    # SAME two CSVs, as arms "__perfect__" and "__frozen__" with
    # direction_role "reference". The preview figure then reads one table
    # instead of re-opening .pt files and re-deriving the endpoints, and
    # the lambda -> infinity end of the ladder is a real row rather than a
    # number the plotter invents.
    for env in ENVS:
        for model in MODELS:
            b = {}
            fzeq = fz_eq.get((model, env))
            for kind, art in (("__perfect__", pp_art.get(env)),
                              ("__frozen__", fz_art.get((model, env)))):
                if art is None:
                    continue
                st = [art["innate"]] + [art["op_raw"][i]
                                        for i in range(ROUNDS)]
                tw = ([art["innate"]] + [art["twin_raw"][i]
                                         for i in range(ROUNDS)]
                      if art["twin_raw"] is not None
                      and art["twin_raw"].shape[0] >= ROUNDS else None)
                pr_b = art["pred_raw"]
                fz_srv = (fz_art[(model, env)]["pred_raw"][0]
                          if (model, env) in fz_art else None)
                rbase = {"model": model, "arm": kind, "beta": env[0],
                         "k": env[1], "env": ENV_LABEL[env],
                         "direction": "none", "lam": "",
                         "direction_role": "reference", "reuse": False,
                         "source_kind": CS.SOURCE_CPU_ENDPOINT,
                         "population_update":
                             (art["config"] or {}).get("population_update"),
                         "run_tag": Path(art["_src"]).name}
                means, sds, dfz, modes = [], [], [], []
                for t in range(ROUNDS + 1):
                    x = st[t]
                    r = dict(rbase)
                    r["t"] = t
                    r["pop_mean"] = float(x.mean())
                    r["pop_sd"] = float(x.std())
                    r["w1_innate"] = w1(x, art["innate"])
                    r["aligned_innate"] = aligned(x, art["innate"])
                    r["rmse_innate"] = rmse(x, art["innate"])
                    if tw is not None:
                        r["w1_twin"] = w1(x, tw[t])
                        r["aligned_twin"] = aligned(x, tw[t])
                    if fzeq is not None:
                        r["w1_frozen_eq"] = w1(x, fzeq)
                        r["aligned_frozen_eq"] = aligned(x, fzeq)
                    if t > 0:
                        r["w1_consec"] = w1(x, st[t - 1])
                        r["aligned_consec"] = aligned(x, st[t - 1])
                        if pr_b is not None:
                            sstat = served_stats(pr_b[t - 1])
                            r["pred_mean"] = sstat["mean"]
                            r["pred_sd"] = sstat["sd"]
                            r["pred_n_distinct"] = sstat["n_distinct"]
                            r["pred_mode_share"] = sstat["mode_share"]
                            r["pred_top3_share"] = sstat["top3_share"]
                            r["pred_eff_modes"] = sstat["eff_modes"]
                            d_, ag, co = served_vs_frozen(pr_b[t - 1], fz_srv)
                            r["served_w1_frozen"] = d_
                            r["served_agree_frozen"] = ag
                            r["served_corr_frozen"] = co
                    per_round_rows.append(r)
                    means.append(r["pop_mean"])
                    sds.append(r["pop_sd"])
                    dfz.append(r.get("w1_frozen_eq", float("nan")))
                    modes.append(r.get("pred_eff_modes", float("nan")))
                b[{"__perfect__": "perfect",
                   "__frozen__": "frozen"}[kind]] = {
                    "pop_mean": means, "pop_sd": sds,
                    "w1_frozen_eq": dfz, "pred_eff_modes": modes}
                m_l = np.array(means[LATE_LO:LATE_HI + 1], dtype=float)
                s_l = np.array(sds[LATE_LO:LATE_HI + 1], dtype=float)
                cm, dm, tm, rm, fm = convergence(m_l, args.drift_tol)
                cs2, ds, ts, rs, fs = convergence(s_l, args.drift_tol)
                mean_state = np.mean(np.stack(st[LATE_LO:LATE_HI + 1], axis=0),
                                     axis=0)
                lr = dict(rbase)
                lr.update({
                    "n_rounds_raw": art["n_rounds_raw"],
                    "late_lo": LATE_LO, "late_hi": LATE_HI,
                    "drift_tol": args.drift_tol,
                    "late_pop_mean": float(m_l.mean()),
                    "late_pop_sd": float(s_l.mean()),
                    "late_mean_jitter": float(m_l.std()),
                    "late_mean_drift": dm, "late_mean_trend": tm,
                    "late_mean_range": rm, "late_mean_noise_floor": fm,
                    "late_sd_drift": ds, "late_sd_trend": ts,
                    "late_sd_range": rs, "late_sd_noise_floor": fs,
                    "late_mean_slope": ols_slope(m_l),
                    "late_sd_slope": ols_slope(s_l),
                    "converged_mean": cm, "converged_sd": cs2,
                    "converged": bool(cm and cs2),
                    "state_label": ("equilibrium" if (cm and cs2)
                                    else f"late-round state (rounds "
                                         f"{LATE_LO}-{LATE_HI}), NOT an "
                                         f"equilibrium"),
                    "late_w1_innate": w1(mean_state, art["innate"]),
                    "late_aligned_innate": aligned(mean_state, art["innate"]),
                    "frozen_source": (fz_art[(model, env)]["_src"]
                                      if (model, env) in fz_art else None),
                    "pp_source": (pp_art[env]["_src"] if env in pp_art
                                  else None),
                    "frozen_eq_converged": fz_conv.get((model, env)),
                    "provenance_deviations": "",
                })
                if fzeq is not None:
                    lr["late_w1_frozen_eq"] = w1(mean_state, fzeq)
                    lr["late_aligned_frozen_eq"] = aligned(mean_state, fzeq)
                if pr_b is not None:
                    wp = pr_b[LATE_LO - 1:LATE_HI]
                    per = [served_stats(wp[i]) for i in range(wp.shape[0])]
                    for src, dst in (("sd", "late_pred_sd"),
                                     ("n_distinct", "late_pred_n_distinct"),
                                     ("mode_share", "late_pred_mode_share"),
                                     ("top3_share", "late_pred_top3_share"),
                                     ("eff_modes", "late_pred_eff_modes")):
                        lr[dst] = float(np.mean([p[src] for p in per]))
                late_rows.append(lr)
            if b:
                series[("__base__", model, env)] = b

    # ---- outputs ------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    pr = write_csv(out_dir / "section3_per_round.csv", PER_ROUND_COLS,
                   per_round_rows)
    lt = write_csv(out_dir / "section3_late_equilibrium.csv", LATE_COLS,
                   late_rows)

    mf_raw = None
    mfp = Path(args.reuse_manifest)
    if not mfp.is_absolute():
        mfp = Path(REPO) / mfp
    if mfp.exists():
        try:
            mf_raw = json.loads(mfp.read_text())
        except Exception:                                 # noqa: BLE001
            mf_raw = {"error": f"unreadable: {mfp}"}
    verdict = None
    if args.verdict and Path(args.verdict).exists():
        verdict = json.loads(Path(args.verdict).read_text())
    prov = {
        "wave": "section3",
        "rounds": ROUNDS, "late_window": [LATE_LO, LATE_HI],
        "drift_tol": args.drift_tol,
        "n_cells_present": len(slots) - len(missing_cells),
        "n_cells_total": len(slots),
        "missing_cells": [CS.slot_tag(*s) for s in missing_cells],
        "missing_endpoints": missing_bases,
        "expected_marker_table": CS.EXPECTED_MARKER,
        "cells": {f"{m}|{a}|{b:g}|{k:g}": {
            **cells[(m, a, b, k)],
            "population_update": next(
                (r["population_update"] for r in late_rows
                 if (r["model"], r["arm"], r["beta"], r["k"]) == (m, a, b, k)),
                None)}
            for (m, a, b, k) in slots if (m, a, b, k) in cells},
        "perfect_prediction": {ENV_LABEL[e]: pp_art[e]["_src"]
                               for e in pp_art},
        "frozen_propagation": {f"{m}|{ENV_LABEL[e]}": fz_art[(m, e)]["_src"]
                               for (m, e) in fz_art},
        "frozen_eq_converged": {f"{m}|{ENV_LABEL[e]}": fz_conv[(m, e)]
                                for (m, e) in fz_conv},
        "reuse_manifest": mf_raw,
        "check_section3_verdict": verdict,
        "notes": notes,
    }
    pv = out_dir / "section3_provenance.json"
    pv.write_text(json.dumps(prov, indent=2, default=str))

    # ---- convergence report -------------------------------------------
    n_flag = sum(1 for r in late_rows if not r["converged"])
    lines = [
        "SECTION 3 CONVERGENCE REPORT",
        "=" * 78,
        f"equilibrium window : post-peer rounds {LATE_LO}-{LATE_HI}",
        f"tolerance          : {args.drift_tol:g} opinion units "
        f"(~{100 * args.drift_tol / INNATE_SD:.1f}% of the innate SD "
        f"{INNATE_SD:.6f}, ~{100 * args.drift_tol / INNATE_MEAN:.2f}% of the "
        f"innate mean {INNATE_MEAN:.6f})",
        "",
        "CRITERION. A fresh LoRA is trained every round, so the served map",
        "and the population carry an irreducible round-to-round jitter and a",
        "vanishing-step test (e.g. 1e-6) can never fire. The diagnostic is",
        "late-window MEAN DRIFT:",
        "",
        "    D = |mean(m[91..100]) - mean(m[81..90])|      half-window drift",
        "    T = |OLS slope of m over 81..100| * 20        fitted trend",
        "    converged_mean  iff  D <= tol  AND  T <= tol",
        "    converged_sd    iff  the same two tests on the per-round SD",
        "    converged       iff  both",
        "",
        "D catches a cell still translating without a linear trend; T catches",
        "one oscillating at near-zero net drift with a real trend underneath.",
        "The per-cell noise floor (median |m[t]-m[t-1]| over the window) is",
        "reported beside them so the margin is visible rather than asserted.",
        "Every raw quantity is in section3_late_equilibrium.csv, so",
        "re-flagging at another tolerance needs no re-run.",
        "",
        "A cell that fails is a LATE-ROUND STATE, not an equilibrium.",
        "=" * 78,
        "",
    ]
    hdr = (f"{'cell':<44} {'conv':>5} {'D_mean':>9} {'T_mean':>9} "
           f"{'floor':>9} {'D_sd':>9} {'T_sd':>9} {'d(frozen)':>10}")
    lines += [hdr, "-" * len(hdr)]
    for r in late_rows:
        name = f"{r['model']}/{r['env']}/{r['arm']}"
        lines.append(
            f"{name:<44} {'yes' if r['converged'] else 'NO':>5} "
            f"{r['late_mean_drift']:>9.5f} {r['late_mean_trend']:>9.5f} "
            f"{r['late_mean_noise_floor']:>9.5f} "
            f"{r['late_sd_drift']:>9.5f} {r['late_sd_trend']:>9.5f} "
            f"{r.get('late_w1_frozen_eq', float('nan')):>10.5f}")
    lines += [
        "",
        f"{n_flag} of {len(late_rows)} cells are FLAGGED as late-round "
        f"states, not equilibria.",
        "",
        "=" * 78,
        "READING THE LADDER",
        "=" * 78,
        "EQUAL NUMERICAL LAMBDA IS NOT EQUAL EFFECTIVE STRENGTH ACROSS KL",
        "DIRECTIONS. KL(p_ref||p_theta) and KL(p_theta||p_ref) differ in",
        "curvature at the same coefficient, so a forward-vs-reverse pair at",
        "matched lambda is not a controlled contrast and no such summary row",
        "is emitted here. Forward is the PRIMARY ladder; reverse lambda in",
        "{1, 8} is a LABELLED ROBUSTNESS CHECK (direction_role column).",
        "Compare arms by the distance they ACHIEVED to the frozen model",
        "(late_w1_frozen_eq / late_aligned_frozen_eq), not by the number in",
        "their name.",
        "",
        "Served-map degeneracy (distinct values, mode share, top-3 share,",
        "effective modes, SD) is REPORTED, not failed: the frozen model this",
        "ladder climbs toward serves a near-binary map, so collapse is a",
        "legitimate outcome and its measurement is the point.",
    ]
    if missing_cells or missing_bases:
        lines = ([f"*** PARTIAL GRID: {len(slots) - len(missing_cells)}"
                  f"/{len(slots)} cells, {len(missing_bases)} endpoint(s) "
                  f"missing. NOT the complete Section 3 result. ***", ""]
                 + lines)
    for n in notes:
        lines.append(n)
    cv = out_dir / "section3_convergence.txt"
    cv.write_text("\n".join(lines) + "\n")

    figs = exploratory_figure(series, out_dir, args.drift_tol)

    print("\n".join(notes))
    print(f"\n[s3] per-round CSV   -> {pr}  ({len(per_round_rows)} rows, "
          f"{ROUNDS + 1} per cell)")
    print(f"[s3] late CSV        -> {lt}  ({len(late_rows)} cells)")
    print(f"[s3] provenance      -> {pv}")
    print(f"[s3] convergence     -> {cv}  ({n_flag} flagged)")
    for f in figs:
        print(f"[s3] figure          -> {f}")
    print(f"[s3] t=0 is the innate population; t=1..{ROUNDS} are end-of-round "
          f"POST-PEER states. {ROUNDS + 1} points per trajectory, no "
          f"within-round intermediates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
