#!/usr/bin/env python3
"""HARD-GATED analyzer for the Jiduan Wu / Pokec replication. CPU only.

    python3 experiments/scripts/cluster_pipelines/analyze_jiduan_pokec.py \\
        --grid prior

Run check_jiduan_pokec.py FIRST. This script assumes the operator is
already proved; it does not re-prove it, and analysing an ungated run is
how a wrong-convention trajectory becomes a figure.

THE SETUP. Pokec LCC, N = 2163. The FIRST 1730 rows are the OBSERVED set
O, the LAST 433 the HELD-OUT set U. The platform is served O's own
opinions and must PREDICT U. alpha_i (peer susceptibility, dataset mean
.8909) and beta_i (platform, .8890) are per agent; K_FJ = 100 inner
steps; T = 50 outer rounds; seed 0 is the primary.

THE PRIMARY ESTIMANDS LIVE ON U, AND ONLY ON U:
  * W1 / RMSE / correlation between the served prediction on U and the
    FROZEN model's prediction map on U -- how far training has moved the
    platform away from the model it started as.
  * W1 / RMSE between the served prediction on U and the CURRENT held-out
    population on U -- how well the platform is tracking the people it
    cannot see.
  * mean and SD of the served prediction on U.
  * mean, SD and W1-from-innate of the final FJ population on U.
Full-population and observed-set numbers are computed too, but they are
SECONDARY and labelled as such everywhere they appear: O's served value
is its own opinion by construction, so any full-population "accuracy"
is four fifths tautology.

WHAT IS AND IS NOT CLAIMED.
  * SEED 0 IS DESCRIPTIVE. Where only seed 0 exists, every number is
    reported as a description of one run. Nothing is called significant.
  * NO ORDERING IS ASSUMED OR ENCODED. Differences are reported signed,
    with both arms named; there is no "expected" direction anywhere in
    this file, and there must never be one -- a replication that only
    recognises one outcome has stopped being a replication.
  * ROUNDS ARE NOT REPLICATES AND NEITHER ARE AGENTS. 50 rounds of one
    trajectory is one trajectory, and 2163 agents on one graph are one
    population -- they share the graph, the innate vector and the
    learner. No standard error is ever taken over rounds or agents. The
    only spread this file reports is ACROSS SEEDS, and it is nan at one
    seed rather than a fake zero.
  * INNER-FJ CONVERGENCE AND OUTER CONVERGENCE ARE DIFFERENT THINGS.
    The INNER loop converges BY CONSTRUCTION: at alpha ~ .89 and K = 100
    it contracts by ~ .89^100, which is what Wu's model specifies. That
    says nothing at all about whether the OUTER model-population loop
    has settled, and round 50 is NOT an equilibrium until the outer test
    says so. The outer test is LATE-WINDOW MEAN DRIFT, not a vanishing
    per-round step: a fresh LoRA is trained every round, so there is an
    irreducible per-round noise floor and demanding a tiny step would
    mislabel a settled system as unconverged.
  * STRICT WU-COMPATIBLE CONTEXT IS NOT THE OBSERVATION-SEMANTIC
    EXTENSION, and which is which is wu_context.is_extension()'s call,
    not this file's. Its rule: a mechanism is an EXTENSION when it shows
    the model something Wu's platform CANNOT observe. observed_context
    (exemplars from O) and prediction_history (the platform's own past
    outputs) are both inside the platform's information set and are
    STRICT; expressed_history shows held-out agents' realised opinions,
    which the platform never sees, and is the EXTENSION. The two classes
    are kept in separate panels with separate labels and are never
    averaged together.

FIGURES carry NO TITLES (project convention; the narrative goes in the
caption block printed at the end). Headline figures plot population
states BETWEEN COMPLETE OUTER ROUNDS only: innate at t = 0, then the
post-FJ state after rounds 1..T. No within-round state appears.

THE CONTROLS HERE ARE THE MINIMAL IN-LINE SET -- perfect prediction, no
platform, and the frozen map -- computed so every figure has a reference
without a second invocation. jiduan_controls.py is the fuller model-free
control suite for this wave and is what the jiduan_pokec_controls key
points at; this file does not depend on it, so neither can break the
other.

Outputs (notes/pofd/jiduan_pokec_replication/):
  jiduan_pokec_rounds.csv            one row per cell per round per subset
  jiduan_pokec_cells.csv             one row per cell per subset
  jiduan_pokec_controls.csv          model-independent CPU controls
  jiduan_pokec_prior_retention.png/.pdf
  jiduan_pokec_sft_icl.png/.pdf
  jiduan_pokec_environment_dose.png/.pdf
  jiduan_pokec_routing.png/.pdf
"""
from __future__ import annotations

import argparse
import csv
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
OUT_DIR = REPO / "notes" / "pofd" / "jiduan_pokec_replication"
CONDOR = REPO / "experiments" / "condor"

PREFIX = "pofdwu_"
FROZEN_PREFIX = "pofdwuzs_"
ARMS = ("b0", "b0p1", "b0p5", "b1", "b10", "frz", "octx8", "phist8",
        "ehist8")
TRAINED_ARMS = ("b0", "b0p1", "b0p5", "b1", "b10")
# STRICT vs EXTENSION is wu_context.is_extension()'s call and has exactly
# one home; the mirror here is asserted against it by the tests. Its
# rule: a mechanism is an EXTENSION when it shows the model something
# Wu's platform cannot observe. The platform has its own past OUTPUTS
# (prediction_history is strict); it does NOT observe held-out agents'
# realised opinions (expressed_history is the extension).
EXTENSION_ARMS = ("ehist8",)
STRICT_ARMS = tuple(a for a in ARMS if a not in EXTENSION_ARMS)
ARM_LABEL = {
    "b0": "SFT $\\lambda$=0", "b0p1": "SFT $\\lambda$=0.1",
    "b0p5": "SFT $\\lambda$=0.5", "b1": "SFT $\\lambda$=1",
    "b10": "SFT $\\lambda$=10", "frz": "frozen, no context",
    "octx8": "observed context K=8", "phist8": "prediction history D=8",
    "ehist8": "expressed history D=8",
}
LAMBDA_OF = {"b0": 0.0, "b0p1": 0.1, "b0p5": 0.5, "b1": 1.0, "b10": 10.0}

N_TOTAL = 2163
N_OBSERVED = 1730
N_HELDOUT = 433
INNER = 100
ROUNDS = 50
LATE = (41, 50)          # inclusive, 1-indexed outer rounds
# EQUILIBRIUM IS STATIONARITY OF THE MEAN, NOT A VANISHING STEP. A fresh
# LoRA trains every round, so |x(t) - x(t-1)| has a floor it never goes
# below. Demanding a tiny step therefore mislabels a settled system as
# unconverged; what distinguishes settled from still-moving is whether
# the population MEAN is still going anywhere.
EQ_DRIFT = 0.005         # on the [0, 1] opinion scale
SUBSETS = ("U", "O", "full")
SUBSET_NOTE = {"U": "PRIMARY (held out)", "O": "secondary (observed)",
               "full": "secondary (all agents)"}


# ----------------------------------------------------------- tag parsing

def _unnum(tok):
    return float(tok.replace("p", "."))


def parse_tag(tag):
    """Read a pofdwu_ tag back into its cell. The tag grammar has exactly
    one home (gen_pofd_sweep.py); this parses, it does not re-derive --
    deriving one string in two places is how a whole wave goes missing."""
    if not tag.startswith(PREFIX):
        return None
    hits = [a for a in ARMS if f"_{a}_pa" in tag]
    if len(hits) != 1:
        return None
    arm = hits[0]
    head, rest = tag.split(f"_{arm}_pa", 1)
    model = head[len(PREFIX):]
    toks = ("pa" + rest).split("_")
    if len(toks) < 5:
        return None
    pa, pb, ink = toks[0], toks[1], toks[2]
    if not (pa.startswith("pa") and pb.startswith("pb")
            and ink.startswith("in")):
        return None
    route = None
    tail = toks[3:]
    if tail and tail[0].startswith("rt"):
        route = tail[0][2:]
        tail = tail[1:]
    if len(tail) != 2 or not tail[0].startswith("s") \
            or not tail[1].startswith("r"):
        return None
    src = {"d": "dataset", "h": "homogeneous"}
    if pa[2] not in src or pb[2] not in src:
        return None
    return {
        "tag": tag, "model": model, "arm": arm,
        "peer_source": src[pa[2]], "ca": _unnum(pa[3:]),
        "platform_source": src[pb[2]], "cb": _unnum(pb[3:]),
        "inner": int(ink[2:]), "route": route,
        "seed": int(tail[0][1:]),
        "rounds": int(tail[1][1:].replace("smoke", "")),
        "smoke": tail[1].endswith("smoke"),
    }


def read_config_tags(condor=None):
    """Every pofdwu_ tag this project has generated, read from the
    ON-DISK config files the jobs run from."""
    condor = Path(condor or CONDOR)
    out = []
    for p in sorted(condor.glob("configs_pofd_jiduan_pokec_*.txt")):
        for ln in p.read_text().splitlines():
            if ln.strip():
                out.append(ln.split(",")[0].strip())
    return out


def frozen_model_of(tag):
    return (tag[len(FROZEN_PREFIX):].rsplit("_s", 1)[0]
            if tag.startswith(FROZEN_PREFIX) else None)


# ------------------------------------------------------ conceptual grids
# Each grid is a list of CELL SPECS. A spec is resolved to a tag by
# matching against the parsed on-disk tags, so a cell that was never
# generated is a hard failure here rather than a silently short figure.

def _spec(model, arm, ca=1.0, cb=1.0, seed=0, route=None):
    return {"model": model, "arm": arm, "ca": ca, "cb": cb, "seed": seed,
            "route": route}


PRIOR_MODELS = ["qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b", "mistral7b",
                "ministral8b"]
SEED_MODELS = ["qwen7b", "olmo7b"]
SEEDS = [42, 43]
ICL_MODELS = ["qwen7b", "mistral7b"]
ICL_ARMS = ["b0", "b1", "phist8", "octx8", "frz"]
ENV_MODEL = "qwen7b"
ENV_ARMS = ["b0", "b1", "phist8"]
ENV_SCALES = [0.0, 0.5, 1.0]
ROUTE_MODEL = "qwen7b"
ROUTE_ARMS = ["b0", "phist8", "frz"]
ROUTE_CAS = [0.0, 1.0]


def env_pairs():
    pairs = [(ca, 1.0) for ca in ENV_SCALES]
    for cb in ENV_SCALES:
        if (1.0, cb) not in pairs:
            pairs.append((1.0, cb))
    return pairs


def grid_specs(grid):
    if grid == "prior":
        return [_spec(m, a) for m in PRIOR_MODELS for a in ("b0", "b1")]
    if grid == "prior_seeds":
        return ([_spec(m, a) for m in SEED_MODELS for a in ("b0", "b1")]
                + [_spec(m, a, seed=s) for s in SEEDS
                   for m in SEED_MODELS for a in ("b0", "b1")])
    if grid == "ladder":
        return [_spec("qwen7b", a) for a in
                ("b0", "b0p1", "b0p5", "b1", "b10")]
    if grid == "icl":
        return [_spec(m, a) for m in ICL_MODELS for a in ICL_ARMS]
    if grid == "environment":
        # c_beta = 0 removes the platform from the recurrence entirely,
        # so the three arms are the SAME population trajectory and the
        # grid holds ONE cell there, under the canonical b0 arm.
        out = []
        for ca, cb in env_pairs():
            out += ([_spec(ENV_MODEL, "b0", ca, cb)] if cb == 0.0
                    else [_spec(ENV_MODEL, a, ca, cb) for a in ENV_ARMS])
        return out
    if grid == "routing":
        return [_spec(ROUTE_MODEL, a, ca=ca, seed=s, route=r)
                for s in [0] + SEEDS for a in ROUTE_ARMS
                for ca in ROUTE_CAS for r in ("T", "C")]
    raise SystemExit(f"[wu] unknown grid {grid!r}")


def resolve(specs, tags):
    """spec -> tag, or a HARD FAIL naming every cell that has none. A
    partial grid is a different claim from the one the wave was built to
    make, so it is refused rather than analysed."""
    parsed = [p for p in (parse_tag(t) for t in tags) if p and not p["smoke"]]
    out, missing, ambiguous = [], [], []
    for s in specs:
        hit = [p for p in parsed
               if p["model"] == s["model"] and p["arm"] == s["arm"]
               and abs(p["ca"] - s["ca"]) < 1e-9
               and abs(p["cb"] - s["cb"]) < 1e-9
               and p["seed"] == s["seed"] and p["route"] == s["route"]]
        if not hit:
            missing.append(s)
        elif len(hit) > 1:
            ambiguous.append((s, [h["tag"] for h in hit]))
        else:
            out.append(dict(s, tag=hit[0]["tag"], inner=hit[0]["inner"],
                            rounds=hit[0]["rounds"]))
    if missing:
        raise SystemExit(
            "[wu] HARD FAIL: the conceptual grid is incomplete -- "
            f"{len(missing)} cell(s) were never generated:\n  "
            + "\n  ".join(str(s) for s in missing)
            + "\n  Run gen_pofd_sweep.py; a partial grid is a different "
              "claim.")
    if ambiguous:
        raise SystemExit(f"[wu] HARD FAIL: ambiguous cells {ambiguous}")
    return out


# --------------------------------------------------------------- metrics

def w1(a, b):
    return float(np.abs(np.sort(np.asarray(a, dtype=float))
                        - np.sort(np.asarray(b, dtype=float))).mean())


def rmse(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        # a constant vector has no correlation; nan says so, and 0 would
        # be a claim
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _nanmean(xs):
    """nan when EVERY entry is nan (no frozen reference, or the t=0 row),
    without the all-nan-slice warning. nan is the right answer there: 0
    would be a claim."""
    v = np.asarray(list(xs), dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def load_cell(roots, tag):
    for r in roots:
        p = Path(r) / tag / "trajectory.pt"
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=False)
    return None


def frozen_map(roots, model):
    """The frozen model's prediction over all agents, from the one-round
    extraction. The extraction runs the ordinary serving path, so it
    writes pred_raw; newer artifacts also carry model_pred_raw. Both are
    accepted and the FIRST round is the map by construction (EPS_AI=0
    means nothing moved)."""
    d = load_cell(roots, f"{FROZEN_PREFIX}{model}_s0_r1")
    if d is None:
        return None
    for key in ("model_pred_raw", "pred_raw"):
        if d.get(key) is not None and torch.is_tensor(d[key]) and d[key].numel():
            return d[key].float().numpy()[0]
    return None


def _idx(n_total, n_observed, subset):
    if subset == "O":
        return np.arange(0, n_observed)
    if subset == "U":
        return np.arange(n_observed, n_total)
    return np.arange(0, n_total)


def round_rows(spec, d, frozen, n_observed):
    """One row per (cell, round, subset).

    t = 0 is the INNATE state -- a complete-round boundary with no round
    behind it, so it carries population statistics and no served ones.
    t >= 1 is the POST-FJ population after outer round t. Nothing
    within-round ever appears here.
    """
    op = d["op_raw"].float().numpy()
    served = d["served_raw"].float().numpy() if d.get("served_raw") is not None \
        else d["pred_raw"].float().numpy()
    pred = (d["model_pred_raw"].float().numpy()
            if d.get("model_pred_raw") is not None else served)
    innate = d["innate"].float().numpy()
    n_total = innate.shape[0]
    rows = []
    for subset in SUBSETS:
        ix = _idx(n_total, n_observed, subset)
        inn = innate[ix]
        fz = None if frozen is None else frozen[ix]
        for t in range(0, op.shape[0] + 1):
            state = inn if t == 0 else op[t - 1][ix]
            row = {
                "grid_tag": spec["tag"], "model": spec["model"],
                "arm": spec["arm"], "ca": spec["ca"], "cb": spec["cb"],
                "seed": spec["seed"], "route": spec["route"] or "",
                "channel": ("strict_wu" if spec["arm"] in STRICT_ARMS
                            else "observation_extension"),
                "subset": subset, "subset_role": SUBSET_NOTE[subset],
                "t": t, "state": "innate" if t == 0 else "post_fj",
                "pop_mean": float(state.mean()),
                "pop_sd": float(state.std(ddof=1)),
                "pop_w1_from_innate": w1(state, inn),
                "pop_rmse_from_innate": rmse(state, inn),
            }
            if t == 0:
                row.update({k: float("nan") for k in
                            ("served_mean", "served_sd", "w1_pred_to_frozen",
                             "rmse_pred_to_frozen", "corr_pred_to_frozen",
                             "w1_pred_to_pop", "rmse_pred_to_pop",
                             "step_from_prev")})
            else:
                sv = served[t - 1][ix]
                pr = pred[t - 1][ix]
                # the population the platform was trying to describe when
                # it spoke: the state ENTERING the round
                entering = inn if t == 1 else op[t - 2][ix]
                row.update({
                    "served_mean": float(sv.mean()),
                    "served_sd": float(sv.std(ddof=1)),
                    "w1_pred_to_frozen": (float("nan") if fz is None
                                          else w1(pr, fz)),
                    "rmse_pred_to_frozen": (float("nan") if fz is None
                                            else rmse(pr, fz)),
                    "corr_pred_to_frozen": (float("nan") if fz is None
                                            else corr(pr, fz)),
                    "w1_pred_to_pop": w1(pr, entering),
                    "rmse_pred_to_pop": rmse(pr, entering),
                    "step_from_prev": float(np.abs(state - entering).max()),
                })
            rows.append(row)
    return rows


def cell_rows(spec, rows, d):
    """Late-window summary per (cell, subset), with the OUTER stationarity
    test kept explicitly apart from the INNER-loop convergence."""
    cfg = d.get("config") or {}
    a_mean = float(cfg.get("fj_alpha_realized_mean", float("nan")))
    inner = int(cfg.get("fj_inner_steps", spec.get("inner", INNER)))
    out = []
    for subset in SUBSETS:
        sel = sorted([r for r in rows if r["subset"] == subset],
                     key=lambda r: r["t"])
        # the late window follows the horizon that actually ran. A short
        # run (a smoke, a killed job) gets the LAST rounds it has, and
        # the window it was measured on rides the row -- silently
        # dropping the cell would turn a truncated run into an absent
        # one, which reads as a missing cell rather than a short one.
        lo, hi = LATE
        horizon = sel[-1]["t"] if sel else 0
        if horizon < hi:
            lo, hi = max(1, horizon - (LATE[1] - LATE[0])), horizon
        late = [r for r in sel if lo <= r["t"] <= hi]
        if not late:
            continue
        steps = [r["step_from_prev"] for r in late[1:]]
        prev = [r["step_from_prev"] for r in sel
                if max(1, lo - 5) <= r["t"] < lo]
        floor = float(np.mean(steps)) if steps else float("nan")
        ratio = (floor / float(np.mean(prev))
                 if prev and np.mean(prev) > 0 else float("nan"))
        drift = late[-1]["pop_mean"] - late[0]["pop_mean"]
        row = {
            "tag": spec["tag"], "model": spec["model"], "arm": spec["arm"],
            "lam": LAMBDA_OF.get(spec["arm"], float("nan")),
            "ca": spec["ca"], "cb": spec["cb"], "seed": spec["seed"],
            "route": spec["route"] or "",
            "channel": ("strict_wu" if spec["arm"] in STRICT_ARMS
                        else "observation_extension"),
            "subset": subset, "subset_role": SUBSET_NOTE[subset],
            "rounds": sel[-1]["t"], "late_window": f"{lo}-{hi}",
            # INNER FJ loop: converged BY CONSTRUCTION at this alpha and
            # K -- a property of the specification, not a finding, and
            # explicitly NOT evidence about the outer loop
            "K_inner": inner,
            "alpha_realized_mean": a_mean,
            "inner_contraction_bound": (float(a_mean) ** inner
                                        if a_mean == a_mean else float("nan")),
            "inner_converged_by_construction": bool(
                a_mean == a_mean and a_mean ** inner < 1e-3),
            # OUTER loop: an empirical question, tested on the MEAN
            "late_mean": float(np.mean([r["pop_mean"] for r in late])),
            "late_sd": float(np.mean([r["pop_sd"] for r in late])),
            "late_pop_w1_from_innate": float(np.mean(
                [r["pop_w1_from_innate"] for r in late])),
            "late_served_mean": _nanmean(r["served_mean"] for r in late),
            "late_served_sd": _nanmean(r["served_sd"] for r in late),
            "late_w1_pred_to_frozen": _nanmean(
                r["w1_pred_to_frozen"] for r in late),
            "late_rmse_pred_to_frozen": _nanmean(
                r["rmse_pred_to_frozen"] for r in late),
            "late_corr_pred_to_frozen": _nanmean(
                r["corr_pred_to_frozen"] for r in late),
            "late_w1_pred_to_pop": _nanmean(
                r["w1_pred_to_pop"] for r in late),
            "late_rmse_pred_to_pop": _nanmean(
                r["rmse_pred_to_pop"] for r in late),
            "outer_noise_floor": floor,
            "outer_step_ratio": ratio,
            "outer_late_mean_drift": drift,
            "outer_stationary": bool(abs(drift) <= EQ_DRIFT),
        }
        out.append(row)
    return out


def across_seeds(rows):
    """Collapse seed replicates to one row per cell identity, CARRYING
    the spread. The FJ operator is deterministic, so a seed moves only
    the learner: this spread IS the training-noise scale, and it is the
    only spread in this file. At one seed it is nan, never 0."""
    grouped = {}
    for r in rows:
        key = (r["model"], r["arm"], r["ca"], r["cb"], r["route"],
               r["subset"])
        grouped.setdefault(key, []).append(r)
    fields = ("late_mean", "late_sd", "late_pop_w1_from_innate",
              "late_served_mean", "late_served_sd",
              "late_w1_pred_to_frozen", "late_rmse_pred_to_frozen",
              "late_corr_pred_to_frozen", "late_w1_pred_to_pop",
              "outer_late_mean_drift")
    agg = []
    for key, rs in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        row = dict(zip(("model", "arm", "ca", "cb", "route", "subset"), key))
        row["n_seeds"] = len(rs)
        row["seeds"] = ",".join(str(x["seed"])
                                for x in sorted(rs, key=lambda z: z["seed"]))
        for f in fields:
            v = np.array([x.get(f, float("nan")) for x in rs], dtype=float)
            row[f] = _nanmean(v)
            row[f + "_seed_sd"] = (float(np.nanstd(v, ddof=1))
                                   if len(rs) > 1 and np.isfinite(v).any()
                                   else float("nan"))
        row["all_outer_stationary"] = all(bool(x["outer_stationary"])
                                          for x in rs)
        agg.append(row)
    return agg


def pair_gap(agg, arm_a, arm_b, field="late_w1_pred_to_frozen",
             subset="U"):
    """The signed difference between two arms, with the pooled seed sd
    beside it. DIRECTION-NEUTRAL by construction: the caller names both
    arms, the sign is reported as-is, and `separated` says only whether
    the gap exceeds seed noise -- never which side is preferable."""
    out = []
    keys = {(r["model"], r["ca"], r["cb"], r["route"]) for r in agg
            if r["subset"] == subset}
    for k in sorted(keys, key=str):
        def pick(arm):
            return next((r for r in agg
                         if r["subset"] == subset and r["arm"] == arm
                         and (r["model"], r["ca"], r["cb"], r["route"]) == k),
                        None)
        ra, rb = pick(arm_a), pick(arm_b)
        if not (ra and rb):
            continue
        gap = rb[field] - ra[field]
        pooled = float(np.sqrt(np.nansum(
            [ra.get(field + "_seed_sd", np.nan) ** 2,
             rb.get(field + "_seed_sd", np.nan) ** 2])))
        out.append({"model": k[0], "ca": k[1], "cb": k[2], "route": k[3],
                    "subset": subset, "field": field,
                    "arm_a": arm_a, "arm_b": arm_b,
                    "gap_b_minus_a": gap, "pooled_seed_sd": pooled,
                    "separated": (bool(abs(gap) > pooled)
                                  if np.isfinite(pooled) and pooled > 0
                                  else None),
                    "n_seeds": min(ra["n_seeds"], rb["n_seeds"])})
    return out


# ------------------------------------------------- model-free CPU controls

def fj_apply(innate, served, alpha, beta, W, n_inner):
    x0 = (1.0 - beta) * innate + beta * served
    u = x0.copy()
    for _ in range(n_inner):
        u = (1.0 - alpha) * x0 + alpha * (W @ u)
    return u


def run_control(kind, env, *, ca=1.0, cb=1.0, rounds=ROUNDS,
                n_inner=INNER, frozen=None, n_observed=N_OBSERVED):
    """The controls that need no language model at all.

      perfect      the platform predicts U exactly: served = x entering
                   the round, everywhere. The upper bound on adaptation.
      no_platform  c_beta = 0: nothing the platform says can reach the
                   population, so x = M x_innate from round 1 onward.
      frozen       a constant prediction map on U.

    NOTE the frozen control is NOT a constant trajectory here, unlike the
    MovieLens FJ wave: the observed passthrough feeds the evolving
    population back into the anchor every round even when the model never
    changes. That difference is a property of Wu's observation semantics
    and is worth reporting, not smoothing over.
    """
    innate = np.asarray(env["innate"], dtype=float)
    alpha = np.asarray(env["alpha_raw"], dtype=float) * ca
    beta = np.asarray(env["beta_raw"], dtype=float) * cb
    W = np.asarray(env["W"], dtype=float)
    n = innate.shape[0]
    x = innate.copy()
    traj = []
    for _ in range(rounds):
        served = x.copy()
        if kind == "perfect":
            pass                      # served = the entering population
        elif kind == "no_platform":
            beta = beta * 0.0
        elif kind == "frozen":
            if frozen is None:
                raise ValueError("frozen control needs a frozen map")
            served = x.copy()
            served[n_observed:] = np.asarray(frozen, dtype=float)[n_observed:]
        else:
            raise ValueError(f"unknown control {kind!r}")
        served[:n_observed] = x[:n_observed]        # passthrough on O
        x = fj_apply(innate, served, alpha, beta, W, n_inner)
        traj.append(x.copy())
    return np.stack(traj)


def control_rows(env, frozen_by_model, rounds=ROUNDS, n_inner=INNER,
                 n_observed=N_OBSERVED):
    innate = np.asarray(env["innate"], dtype=float)
    rows = []
    jobs = [("perfect", None, "qwen7b"), ("no_platform", None, "qwen7b")]
    jobs += [("frozen", v, m) for m, v in sorted(frozen_by_model.items())
             if v is not None]
    for kind, fz, model in jobs:
        cb = 0.0 if kind == "no_platform" else 1.0
        traj = run_control(kind, env, cb=cb, rounds=rounds,
                           n_inner=n_inner, frozen=fz,
                           n_observed=n_observed)
        for subset in SUBSETS:
            ix = _idx(innate.shape[0], n_observed, subset)
            for t in range(traj.shape[0]):
                rows.append({
                    "control": kind, "model": model if kind == "frozen" else "",
                    "subset": subset, "t": t + 1,
                    "pop_mean": float(traj[t][ix].mean()),
                    "pop_sd": float(traj[t][ix].std(ddof=1)),
                    "pop_w1_from_innate": w1(traj[t][ix], innate[ix]),
                    "step_from_prev": float(
                        np.abs(traj[t] - (innate if t == 0
                                          else traj[t - 1]))[ix].max()),
                })
    return rows


# ---------------------------------------------------------------- output

def _csv(path, rows):
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[wu] wrote {path} ({len(rows)} rows)")


def _style(ax, xlabel, ylabel):
    """No titles anywhere -- this project's figures carry none, and the
    narrative goes in the caption block."""
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)


# Panel labels sit INSIDE the axes because the figures carry no titles.
# A translucent backing washes out whatever curve runs behind it, which
# looks like faded data; an opaque one hides it outright. So the label
# gets an opaque cartouche AND the axes get headroom above the data, so
# there is nothing behind it to hide.
_LBL = dict(fontsize=9, va="top", ha="left", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=1.0,
                      pad=1.5))


def _panel_label(ax, text, fontsize=9):
    kw = dict(_LBL)
    kw["fontsize"] = fontsize
    ax.text(0.035, 0.96, text, transform=ax.transAxes, **kw)


def _headroom(ax, frac=0.28):
    """Room above the data for the in-axes label. Called AFTER plotting."""
    lo, hi = ax.get_ylim()
    if hi > lo:
        ax.set_ylim(lo, hi + frac * (hi - lo))


def _save(fig, out_dir, stem):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = Path(out_dir) / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[wu] wrote {p}")
    plt.close(fig)


ARM_COLOR = {"b0": "#c1443c", "b0p1": "#d98a3a", "b0p5": "#6a9a56",
             "b1": "#2a6fb5", "b10": "#8a5fa8", "frz": "#777777",
             "octx8": "#2f8f8f", "phist8": "#b2478f", "ehist8": "#7a6a3a"}
SEED_LS = {0: "-", 42: "--", 43: ":"}


def _series(rounds_rows, subset, **match):
    sel = [r for r in rounds_rows if r["subset"] == subset
           and all(r[k] == v for k, v in match.items())]
    sel.sort(key=lambda r: r["t"])
    return sel


def fig_prior_retention(rounds_rows, out_dir, models, subset="U"):
    """Population states between COMPLETE outer rounds only: innate at
    t=0, then post-FJ after 1..T. One panel per model; every seed drawn
    RAW, never averaged."""
    ncol = 3
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, models):
        for arm in ("b0", "b1"):
            for sd in sorted({r["seed"] for r in rounds_rows}):
                s = _series(rounds_rows, subset, model=m, arm=arm, seed=sd,
                            route="")
                if not s:
                    continue
                ax.plot([r["t"] for r in s],
                        [r["pop_sd"] for r in s], lw=1.5,
                        color=ARM_COLOR[arm], ls=SEED_LS.get(sd, "-"),
                        label=(f"{ARM_LABEL[arm]}, seed {sd}"))
        _style(ax, "complete outer round", "held-out opinion SD")
        _headroom(ax)
        _panel_label(ax, m)
    for ax in axes[len(models):]:
        ax.axis("off")
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "jiduan_pokec_prior_retention")


def fig_sft_icl(rounds_rows, out_dir, models, subset="U"):
    """STRICT Wu-compatible channels on the left, the OBSERVATION-
    SEMANTIC EXTENSION on the right. They are different objects and are
    never drawn on shared axes or averaged together.

    When a grid contains no extension cell the second column is DROPPED
    rather than left blank, and the surviving panel says so -- an empty
    axis reads as missing data, which is a different claim from "this
    grid asks only the strict question".
    """
    has_ext = any(r["arm"] in EXTENSION_ARMS and r["route"] == ""
                  for r in rounds_rows)
    cols = (STRICT_ARMS, EXTENSION_ARMS) if has_ext else (STRICT_ARMS,)
    # SEPARATE PANELS, SHARED SCALE. The separation is semantic -- the
    # columns are different objects -- but the magnitudes still have to
    # be readable against each other, so y is shared within a row.
    fig, axes = plt.subplots(len(models), len(cols),
                             figsize=(4.6 * len(cols), 3.0 * len(models)),
                             sharex=True, sharey="row", squeeze=False)
    for row, m in enumerate(models):
        for col, arms in enumerate(cols):
            ax = axes[row][col]
            for arm in arms:
                s = _series(rounds_rows, subset, model=m, arm=arm, seed=0,
                            route="")
                if not s:
                    continue
                ax.plot([r["t"] for r in s],
                        [r["pop_w1_from_innate"] for r in s], lw=1.6,
                        color=ARM_COLOR[arm], label=ARM_LABEL[arm])
            _style(ax, "complete outer round",
                   "$W_1$ from innate (held out)")
            _headroom(ax)
            if col == 0 and not has_ext:
                lbl = (f"{m}\nstrict Wu-compatible context\n"
                       f"(no observation-semantic extension cell here)")
            elif col == 0:
                lbl = f"{m}\nstrict Wu-compatible context"
            else:
                lbl = f"{m}\nobservation-semantic extension"
            _panel_label(ax, lbl, fontsize=8)
            ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "jiduan_pokec_sft_icl")


def fig_environment_dose(cells, out_dir, subset="U"):
    """Two dose axes, one row each: c_alpha at c_beta=1, and c_beta at
    c_alpha=1. The shared centre point appears in both rows, which is
    what it is."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    sel = [c for c in cells if c["subset"] == subset and not c["route"]]
    # At c_beta = 0 the platform cannot reach the population at all, so
    # every arm IS the same trajectory and the grid holds one cell. All
    # three curves are drawn through that single shared point -- which is
    # the true statement -- and it is annotated so the shared point does
    # not read as three coincidental agreements.
    shared0 = next((c for c in sel if abs(c["cb"]) < 1e-9), None)
    for ax, (axis, fixed) in zip(axes, (("ca", "cb"), ("cb", "ca"))):
        for arm in ENV_ARMS:
            pts = sorted([c for c in sel if c["arm"] == arm
                          and abs(c[fixed] - 1.0) < 1e-9
                          and c[axis] > 0.0],
                         key=lambda c: c[axis])
            if axis == "ca":
                pts = sorted([c for c in sel if c["arm"] == arm
                              and abs(c[fixed] - 1.0) < 1e-9],
                             key=lambda c: c[axis])
            elif shared0 is not None:
                pts = [shared0] + pts
            if not pts:
                continue
            ax.plot([p[axis] for p in pts],
                    [p["late_sd"] for p in pts], marker="o", lw=1.5,
                    color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        if axis == "cb" and shared0 is not None:
            ax.annotate("platform disconnected:\nall arms coincide",
                        xy=(0.0, shared0["late_sd"]),
                        xytext=(0.06, 0.12), textcoords="axes fraction",
                        fontsize=7, color="#444444",
                        arrowprops=dict(arrowstyle="-", lw=0.7,
                                        color="#888888"))
        _style(ax, ("peer-susceptibility scale $c_\\alpha$"
                    if axis == "ca" else
                    "platform-susceptibility scale $c_\\beta$"),
               "late-window held-out SD")
        ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "jiduan_pokec_environment_dose")


def fig_routing(rounds_rows, out_dir, subset="U"):
    """Paired twins, one panel per (arm, c_alpha): treatment solid,
    control dashed, and EVERY SEED DRAWN RAW -- one line per seed, never
    a mean over seeds. Seeds are not labelled individually; a twelve-item
    legend hides the data it is describing, and the identity of an
    individual seed carries no information anyway."""
    from matplotlib.lines import Line2D
    cas = sorted({r["ca"] for r in rounds_rows if r["route"]})
    arms = [a for a in ROUTE_ARMS
            if any(r["arm"] == a and r["route"] for r in rounds_rows)]
    fig, axes = plt.subplots(max(len(arms), 1), max(len(cas), 1),
                             figsize=(4.3 * max(len(cas), 1),
                                      2.8 * max(len(arms), 1)),
                             sharex=True, sharey="row", squeeze=False)
    for i, arm in enumerate(arms):
        for j, ca in enumerate(cas):
            ax = axes[i][j]
            for side, ls in (("T", "-"), ("C", "--")):
                for sd in sorted({r["seed"] for r in rounds_rows}):
                    s = _series(rounds_rows, subset, arm=arm, ca=ca,
                                route=side, seed=sd)
                    if not s:
                        continue
                    ax.plot([r["t"] for r in s],
                            [r["pop_mean"] for r in s], lw=1.3, ls=ls,
                            alpha=0.85, color=ARM_COLOR[arm])
            _style(ax, "complete outer round", "held-out opinion mean")
            _headroom(ax)
            _panel_label(ax, f"{ARM_LABEL[arm]}\n$c_\\alpha$ = {ca:g}",
                         fontsize=8)
    n_seeds = len({r["seed"] for r in rounds_rows if r["route"]})
    axes[0][-1].legend(handles=[
        Line2D([], [], color="#444444", ls="-", lw=1.3, label="treated"),
        Line2D([], [], color="#444444", ls="--", lw=1.3, label="control"),
        Line2D([], [], color="none",
               label=f"one line per seed ({n_seeds})")],
        fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "jiduan_pokec_routing")


# ---------------------------------------------------------------- report

def report(cells, agg, grid):
    seeds = sorted({c["seed"] for c in cells})
    windows = sorted({c["late_window"] for c in cells})
    print(f"\n[wu] grid={grid}. Late window(s) = outer rounds "
          f"{', '.join(windows) or '-'}.")
    if seeds == [0]:
        print("[wu] SEED 0 ONLY -- everything below is DESCRIPTIVE. No "
              "ordering is called significant and none is assumed.")
    else:
        print(f"[wu] seeds {seeds}: the only spread reported here is "
              f"ACROSS SEEDS. Rounds are not replicates and neither are "
              f"agents.")
    prim = [c for c in cells if c["subset"] == "U"]
    n_inner_ok = sum(1 for c in prim if c["inner_converged_by_construction"])
    n_outer_ok = sum(1 for c in prim if c["outer_stationary"])
    print(f"[wu] INNER FJ loop: converged by construction in "
          f"{n_inner_ok}/{len(prim)} cells (contraction "
          f"alpha^K at K={prim[0]['K_inner'] if prim else INNER}). This is "
          f"a property of the specification and is NOT evidence about the "
          f"outer loop.")
    print(f"[wu] OUTER loop: stationary (late-window |mean drift| <= "
          f"{EQ_DRIFT}) in {n_outer_ok}/{len(prim)} cells. The per-round "
          f"step does NOT vanish -- a fresh LoRA trains every round, so "
          f"there is a noise floor -- which is why the mean is the test.")
    print(f"\n[wu] HELD-OUT SET U ({N_HELDOUT} agents) -- PRIMARY")
    print(f"[wu] {'model':<13} {'arm':>7} {'ca':>4} {'cb':>4} {'rt':>3} "
          f"{'seed':>5} {'popSD':>8} {'servSD':>8} {'W1->fz':>8} "
          f"{'r(fz)':>7} {'W1->pop':>8} {'drift':>9} {'stat':>5}")
    for c in sorted(prim, key=lambda r: (r["model"], r["arm"], r["ca"],
                                         r["cb"], r["route"], r["seed"])):
        print(f"[wu] {c['model']:<13} {c['arm']:>7} {c['ca']:>4.2g} "
              f"{c['cb']:>4.2g} {c['route'] or '-':>3} {c['seed']:>5} "
              f"{c['late_sd']:>8.4f} {c['late_served_sd']:>8.4f} "
              f"{c['late_w1_pred_to_frozen']:>8.4f} "
              f"{c['late_corr_pred_to_frozen']:>7.3f} "
              f"{c['late_w1_pred_to_pop']:>8.4f} "
              f"{c['outer_late_mean_drift']:>+9.5f} "
              f"{str(c['outer_stationary']):>5}")
    print("\n[wu] SECONDARY subsets (O and full) are in the CSVs. On O the "
          "served value is the agent's own opinion by construction, so a "
          "full-population fit is mostly tautology and is never the "
          "headline.")
    gaps = pair_gap(agg, "b0", "b1")
    if gaps:
        print("\n[wu] signed b0 -> b1 differences in W1(prediction, frozen "
              "map) on U. Sign is reported as-is; `separated` says only "
              "whether the gap exceeds seed noise, never which side is "
              "preferable.")
        for g in gaps:
            sep = ("yes" if g["separated"] else
                   ("no" if g["separated"] is False else "n/a (1 seed)"))
            print(f"[wu] {g['model']:<13} ca={g['ca']:<4.2g} "
                  f"cb={g['cb']:<4.2g} gap(b1-b0)="
                  f"{g['gap_b_minus_a']:+.5f}  pooled seed sd="
                  f"{g['pooled_seed_sd']:.5f}  separated={sep}")
    strict = [c for c in prim if c["channel"] == "strict_wu"]
    ext = [c for c in prim if c["channel"] == "observation_extension"]
    if strict and ext:
        print(f"\n[wu] {len(strict)} strictly Wu-compatible cell(s) and "
              f"{len(ext)} observation-semantic EXTENSION cell(s). They "
              f"are reported side by side and never pooled: Wu's platform "
              f"has no memory of its own past outputs, so the extension "
              f"answers a different question.")
    print("\n[wu] CAPTION MATERIAL (the figures carry no titles): Pokec "
          f"LCC, N={N_TOTAL}; observed set O = the first {N_OBSERVED} "
          f"rows, held-out set U = the last {N_HELDOUT}. Per-agent "
          f"alpha_i and beta_i from the dataset, K_FJ={INNER} inner steps "
          f"per outer round, T={ROUNDS} outer rounds. States shown are "
          f"innate at t=0 and the post-FJ population after each complete "
          f"outer round.")


# ------------------------------------------------------------------ main

def analyse(roots, out_dir, grid="prior", condor=None, n_observed=N_OBSERVED,
            controls=True):
    specs = resolve(grid_specs(grid), read_config_tags(condor))
    cells, missing = [], []
    for s in specs:
        d = load_cell(roots, s["tag"])
        if d is None:
            missing.append(s["tag"])
        else:
            cells.append((s, d))
    if missing:
        raise SystemExit(
            f"[wu] HARD FAIL: {len(missing)}/{len(specs)} cells of the "
            f"'{grid}' grid have no trajectory.pt -- a partial grid is a "
            f"different claim.\n  " + "\n  ".join(missing))
    models = sorted({s["model"] for s in specs})
    frozen = {m: frozen_map(roots, m) for m in models}
    absent = [m for m, v in frozen.items() if v is None]
    if absent:
        raise SystemExit(
            f"[wu] HARD FAIL: no frozen prediction map for {absent}. The "
            f"primary held-out estimand is distance to it, so it is not "
            f"optional. Submit jiduan_pokec_frozen; no archived Pokec "
            f"frozen vector exists to reuse.")
    _assert_shared_environment(cells)

    rounds_rows, cellrows = [], []
    for s, d in cells:
        rr = round_rows(s, d, frozen[s["model"]], n_observed)
        rounds_rows += rr
        cellrows += cell_rows(s, rr, d)
    agg = across_seeds(cellrows)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    _csv(Path(out_dir) / "jiduan_pokec_rounds.csv", rounds_rows)
    _csv(Path(out_dir) / "jiduan_pokec_cells.csv", cellrows)
    if len({c["seed"] for c in cellrows}) > 1:
        _csv(Path(out_dir) / "jiduan_pokec_by_seed.csv", agg)
    if controls:
        env = _env_from_cells(cells)
        if env is not None:
            _csv(Path(out_dir) / "jiduan_pokec_controls.csv",
                 control_rows(env, frozen, rounds=cellrows[0]["rounds"],
                              n_observed=n_observed))
    report(cellrows, agg, grid)
    figures(rounds_rows, cellrows, out_dir, models, grid)
    return cellrows


def _env_from_cells(cells):
    """The controls need innate, alpha, beta and the graph. Taken from the
    checker's loader so there is one home for the environment."""
    try:
        sys.path.insert(0, str(HERE))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "chk_wu_env", str(HERE / "check_jiduan_pokec.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        env = mod.wu_env()
    except Exception:
        env = None
    return env


def _assert_shared_environment(cells):
    """Every arm must share ONE environment or no cross-arm comparison
    means anything: same innate vector, same graph, same alpha/beta, same
    observed/held-out split."""
    import hashlib

    def sha(t):
        return hashlib.sha256(torch.as_tensor(t).detach().cpu().float()
                              .contiguous().numpy().tobytes()).hexdigest()
    innates, graphs, alphas, betas, splits = set(), set(), set(), set(), set()
    for s, d in cells:
        cfg = d.get("config") or {}
        innates.add(sha(d["innate"]))
        graphs.add(cfg.get("fj_graph_sha256"))
        alphas.add(cfg.get("fj_alpha_raw_sha256"))
        betas.add(cfg.get("fj_beta_raw_sha256"))
        splits.add(int(cfg.get("n_labeled", -1)))
    for label, s_ in (("innate vector", innates), ("graph hash", graphs),
                      ("alpha vector", alphas), ("beta vector", betas),
                      ("observed/held-out split", splits)):
        if len(s_) != 1:
            raise SystemExit(f"[wu] HARD FAIL: {len(s_)} distinct {label}s "
                             f"across the grid -- not one environment: {s_}")


def figures(rounds_rows, cellrows, out_dir, models, grid):
    if grid in ("prior", "prior_seeds"):
        fig_prior_retention(rounds_rows, out_dir, models)
    if grid in ("icl", "ladder"):
        fig_sft_icl(rounds_rows, out_dir, models)
    if grid == "environment":
        fig_environment_dose(cellrows, out_dir)
    if grid == "routing":
        fig_routing(rounds_rows, out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roots", nargs="*", type=Path,
                    default=[REPO / "notes" / "pofd" / "cluster",
                             REPO / "runs" / "pokec_gated_lm"])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--grid", default="prior",
                    choices=["prior", "prior_seeds", "ladder", "icl",
                             "environment", "routing"])
    ap.add_argument("--controls-only", action="store_true",
                    help="write only the model-independent CPU controls")
    ap.add_argument("--no-controls", action="store_true")
    args = ap.parse_args()
    if args.controls_only:
        env = _env_from_cells([])
        if env is None:
            raise SystemExit("[wu] the Pokec dataset is not reachable, so "
                             "the controls cannot be computed")
        models = sorted({m for m in PRIOR_MODELS})
        fz = {m: frozen_map(args.roots, m) for m in models}
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _csv(args.out_dir / "jiduan_pokec_controls.csv",
             control_rows(env, {m: v for m, v in fz.items() if v is not None}))
        return 0
    analyse(args.roots, args.out_dir, grid=args.grid,
            controls=not args.no_controls)
    print(f"[wu] outputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
