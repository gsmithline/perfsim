#!/usr/bin/env python3
"""GATE for the redesigned Figure 3 (reference_retention_memory_equilibrium),
UNIFIED 108-CELL GRID (2026-08-24 v2).

WHAT THIS GATES, AND WHY IT IS NOT check_pofd_sanity.
The figure's claim is that every finite-lambda point ran the COMPLETE
recursive loop: retrain a fresh adapter on the current population labels,
serve, mix into the population at beta, run S peer sweeps, and feed the
resulting post-peer opinions into the next round.  A trajectory.pt that
merely exists proves none of that.

THE RETRAINING WITNESS -- what is hard and what is diagnostic.
  HARD    per-round training telemetry.  telemetry.json must exist, be
          non-empty, and carry a training record (l_init present,
          n_train > 0) for EVERY round of the cell's horizon.  That is
          the recorded evidence that an optimizer actually ran each
          round.  Absent, empty, or missing any round => FAIL.
  HARD    (lambda > 0 only) the KL anchor gradient grad_kl_norm0 must
          not be identically zero after round 0 (round 0 is exempt: a
          fresh LoRA IS the reference there).
  NOTE    "the served vector moved across rounds" is DIAGNOSTIC ONLY.
          A valid loop can settle: the model retrains every round and
          legitimately reproduces the same served map once the
          population stops changing.  A constant served map is printed
          as a note, never a failure -- the telemetry is the witness.
  NOTE    likewise a constant op_raw (possible at full consensus).

THE THREE NAMES THAT COLLIDE IN THIS REPO, once:
    beta   = W_PLAT, platform susceptibility      -> config w_plat
    gamma  = INNATE_LAMBDA, the innate re-anchor  -> config innate_lambda
    lambda = kl_beta, the forward-KL coefficient  -> config kl_beta
The homophily gamma is a DIFFERENT gamma and must stay 0.0.

CELL KINDS.  108 unique cells, three kinds, only one of them a GPU run:
    gpu     finite lambda, beta in (0, 1]           -- trajectory.pt
    twin    beta = 0 (lambda drops out)             -- twin_raw of any
                                                       cell at that gamma
    frozen  lambda = inf (model frozen, population  -- notes/pofd/
            loop still recursive)                      frozen_replay
    (beta = 1 deduplicates gamma: it drops out of the operator at W = 1.)
The grid is READ FROM THE GENERATOR (gen_pofd_sweep.f3_cells) rather
than restated here, so the checker and the wave cannot disagree about
what the figure is.

EXTENSIONS.  If experiments/condor/fig3_extension_request.json exists,
its cells' 60/100-round artifacts are checked too when present (full
checks at the extended horizon); a requested-but-not-yet-run extension
prints as PENDING-EXT and does NOT fail the base gate -- the ANALYZER's
--paper mode is what refuses to call an unsettled cell an equilibrium.

HARD-FAIL, NEVER WARN, on the base grid: absent or ungated cells fail;
there is no "analyze what is present" mode, because a Figure 3 drawn
from a partial grid is a different figure.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
GEN = REPO / "experiments" / "condor" / "gen_pofd_sweep.py"
DEFAULT_RUN_ROOTS = (
    Path("/home/gsmithline/perfsim/runs/pokec_gated_lm"),
    REPO / "runs" / "pokec_gated_lm",
    REPO / "notes" / "pofd" / "cluster",
)
DEFAULT_FROZEN_DIR = REPO / "notes" / "pofd" / "frozen_replay"

N_AGENTS = 723
TOL = 1e-9


def _load_gen():
    spec = importlib.util.spec_from_file_location("_gen_f3", str(GEN))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gen_f3"] = mod
    spec.loader.exec_module(mod)
    return mod


def _num(v):
    return f"{v:g}".replace(".", "p")


def _resolve(tag, roots):
    for r in roots:
        p = Path(r) / tag / "trajectory.pt"
        if p.exists():
            return p
    return None


def _frozen_name(beta, gamma, sweeps, rounds, seed=0):
    return (f"frz_k{_num(gamma)}_w{_num(beta)}_eaopen_esopen"
            f"_sw{sweeps}_s{seed}_r{rounds}.pt")


def _sha_t(t):
    a = t.detach().cpu().contiguous().float().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def _finite(t):
    return bool(torch.isfinite(torch.as_tensor(t)).all())


def _read_jsonl(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


# --------------------------------------------------------------- 1. config
def check_config(cfg, beta, gamma, lam, rounds, sweeps, model_id, errs):
    """Every dial the figure's caption asserts, checked against the CELL,
    not against whatever the run happened to record."""
    def eq(key, want, label=None):
        got = cfg.get(key, "<absent>")
        if isinstance(want, float):
            ok = isinstance(got, (int, float)) and abs(float(got) - want) <= TOL
        else:
            ok = got == want
        if not ok:
            errs.append(f"CONFIG {label or key}={got!r} (want {want!r})")

    eq("w_plat", float(beta), "beta/w_plat")
    eq("innate_lambda", float(gamma), "gamma/innate_lambda")
    eq("kl_beta", float(lam), "lambda/kl_beta")
    eq("ab_sweeps", int(sweeps))
    eq("n_rounds", int(rounds))
    eq("seed", 0)
    eq("dataset", "movielens")
    eq("ml_target", "Action")
    eq("base_model", model_id)
    eq("kl_direction", "forward")
    eq("ai_gate_mode", "all_open")
    eq("peer_gate_mode", "all_open")
    eq("icl_k", 0)
    eq("train_cap", N_AGENTS)
    eq("n_labeled", N_AGENTS)
    eq("lora_r", 512)
    # the corrected operator -- the anch2 token in the tag must be true
    if cfg.get("population_update") != "nested_ai_anchored_then_social_v2":
        errs.append(
            f"OPERATOR population_update={cfg.get('population_update')!r} "
            f"-- Figure 3 requires the CORRECTED anchored operator "
            f"'nested_ai_anchored_then_social_v2'; "
            f"'nested_ai_then_social_v1' gates on the raw x0")
    if cfg.get("ai_gate_reference") != "anchor":
        errs.append(f"OPERATOR ai_gate_reference="
                    f"{cfg.get('ai_gate_reference')!r} (want 'anchor')")
    # homophily gamma is a DIFFERENT gamma and is 0 in every pofd wave
    hg = cfg.get("homophily_gamma", cfg.get("gamma_bias", 0.0))
    if hg not in (0, 0.0):
        errs.append(f"CONFIG homophily gamma={hg!r} (want 0.0) -- this is "
                    f"NOT the innate re-anchor gamma")
    # the trained arms must actually be adaptive
    if lam == 0.0 and cfg.get("training_style") not in ("sft", None):
        errs.append(f"CONFIG training_style={cfg.get('training_style')!r} "
                    f"(lambda=0 is ordinary SFT)")
    if lam > 0.0 and cfg.get("training_style") not in ("sft_kl", None):
        errs.append(f"CONFIG training_style={cfg.get('training_style')!r} "
                    f"(lambda>0 is sft_kl)")
    if cfg.get("use_lora") in (0, False):
        errs.append("CONFIG use_lora is off -- a finite-lambda cell must train")
    if cfg.get("fresh_each_round") in (0, False):
        errs.append("CONFIG fresh_each_round is off -- Figure 3 retrains a "
                    "FRESH adapter every round")


# ------------------------------------------------------- 2/3. shape, finite
def check_arrays(d, rounds, errs):
    for key in ("op_raw", "twin_raw", "pred_raw"):
        if key not in d:
            errs.append(f"ARTIFACT {key} absent")
            continue
        a = torch.as_tensor(d[key]).float()
        if a.ndim != 2 or a.shape[1] != N_AGENTS:
            errs.append(f"ARTIFACT {key} shape {tuple(a.shape)} "
                        f"(want [>={rounds}, {N_AGENTS}])")
            continue
        if a.shape[0] < rounds:
            errs.append(f"ARTIFACT {key} has {a.shape[0]} rounds "
                        f"(want >= {rounds})")
        if not _finite(a):
            errs.append(f"ARTIFACT {key} holds non-finite values")
        if key != "pred_raw":
            lo, hi = float(a.min()), float(a.max())
            if lo < -TOL or hi > 1.0 + TOL:
                errs.append(f"ARTIFACT {key} out of [0,1]: [{lo:.4g},{hi:.4g}]")
    tr = d.get("trajectory", [])
    if len(tr) < rounds:
        errs.append(f"ARTIFACT trajectory has {len(tr)} rows (want >= {rounds})")


# ------------------------------------------------------- 4. THE FULL LOOP
def check_full_loop(d, run_dir, lam, rounds, errs, notes):
    """The claim the figure rests on: this cell RETRAINED and FED BACK
    every round.

    HARD WITNESS: the per-round training telemetry.  One record per
    round with l_init (the optimizer saw an initial loss) and n_train > 0
    (it saw rows).  A run that skipped training in any round cannot fake
    this, and a run that legitimately settled still produces it.

    DIAGNOSTICS ONLY: "the served vector moved" and "the population
    moved".  A settled loop may serve a bit-identical map round after
    round -- retraining every time and reproducing itself -- so a
    constant map is NOT proof of a broken loop; the 2026-08-24 audit
    demoted it from a failure to a note."""
    tel = Path(run_dir) / "telemetry.json"
    if not tel.exists():
        errs.append(
            "FULL-LOOP telemetry.json ABSENT -- there is no recorded "
            "evidence that this cell ever trained; the trajectory alone "
            "cannot distinguish a full loop from a frozen replay")
        return
    rows = _read_jsonl(tel)
    if not rows:
        errs.append("FULL-LOOP telemetry.json is EMPTY -- no training "
                    "records at all")
        return
    trained = {int(r["round"]) for r in rows
               if "l_init" in r and int(r.get("n_train", 0) or 0) > 0}
    missing = sorted(set(range(rounds)) - trained)
    if missing:
        errs.append(
            f"FULL-LOOP telemetry carries a training record for only "
            f"{len(trained)} of {rounds} round(s); missing rounds "
            f"{missing[:8]}{'...' if len(missing) > 8 else ''} -- the "
            f"loop did not retrain every round")
    # a KL-regularised cell must actually have an anchor gradient after r0
    if lam > 0.0:
        gk = [r.get("grad_kl_norm0") for r in rows
              if int(r.get("round", 0)) > 0
              and r.get("grad_kl_norm0") is not None]
        if gk and all(abs(float(v)) <= 0.0 for v in gk):
            errs.append(f"FULL-LOOP lambda={lam:g} but the KL gradient "
                        f"norm is identically 0 after round 0 -- the "
                        f"reference anchor never bound")
        elif not gk and rounds > 1:
            errs.append(f"FULL-LOOP lambda={lam:g} but telemetry records "
                        f"no grad_kl_norm0 after round 0 -- cannot show "
                        f"the anchor ever bound")
    # ---- diagnostics, never failures ----
    pred = torch.as_tensor(d["pred_raw"]).float()[:rounds]
    op = torch.as_tensor(d["op_raw"]).float()[:rounds]
    if pred.shape[0] >= 2 and \
            len({_sha_t(pred[t]) for t in range(pred.shape[0])}) == 1:
        notes.append("served map constant across rounds (legitimate for a "
                     "settled loop; telemetry above is the witness)")
    if op.shape[0] >= 2 and \
            len({_sha_t(op[t]) for t in range(op.shape[0])}) == 1:
        notes.append("population constant across rounds (full consensus "
                     "can do this; telemetry above is the witness)")


# ---------------------------------------------------- 5. zero parse failures
def check_parse(run_dir, pred, rounds, errs):
    """parse_fail_frac lives ONLY in raw_gen_log.json.gz (gzipped JSONL)
    -- never in trajectory.pt.  This wave sets SAVE_RAW_GEN=1, so the
    strict gate applies; the NaN fallback covers archived reuse cells."""
    raw = Path(run_dir) / "raw_gen_log.json.gz"
    if raw.exists():
        bad = []
        with gzip.open(raw, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pf = rec.get("parse_fail_frac")
                if pf is not None and float(pf) > 0.0:
                    bad.append((rec.get("round"), float(pf)))
        if bad:
            errs.append(f"PARSE non-zero parse_fail_frac in {len(bad)} "
                        f"round(s): {bad[:4]}")
        return "raw_gen_log"
    n_nan = int((~torch.isfinite(pred[:rounds])).sum())
    if n_nan:
        errs.append(f"PARSE {n_nan} non-finite served value(s) -- an "
                    f"unparsable generation is stored as NaN")
    return "pred_raw_nan"


# ============================================================ per-cell check
def check_gpu_cell(tag, beta, gamma, lam, rounds, sweeps, model_id, roots,
                   common_rounds=None):
    errs, notes = [], []
    common_rounds = rounds if common_rounds is None else common_rounds
    path = _resolve(tag, roots)
    if path is None:
        return {"tag": tag, "status": "ABSENT", "errors":
                ["no trajectory.pt under any run root"], "notes": []}
    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = d.get("config", {}) or {}
    check_config(cfg, beta, gamma, lam, rounds, sweeps, model_id, errs)
    check_arrays(d, rounds, errs)
    if all(not e.startswith("ARTIFACT") for e in errs):
        check_full_loop(d, path.parent, lam, rounds, errs, notes)
        src = check_parse(path.parent, torch.as_tensor(d["pred_raw"]).float(),
                          rounds, errs)
    else:
        src = "n/a"
    op = torch.as_tensor(d["op_raw"]).float()[:rounds]
    return {"tag": tag, "status": "PASS" if not errs else "FAIL",
            "errors": errs, "notes": notes, "parse_src": src,
            "rounds": int(op.shape[0]),
            "mean": float(op[-1].mean()), "sd": float(op[-1].std()),
            # hashed over the COMMON horizon, never the cell's own: a
            # reused 60-round twin must compare equal to an identical
            # 30-round one
            "twin_sha": _sha_t(torch.as_tensor(d["twin_raw"]).float()
                               [:common_rounds])
            if "twin_raw" in d else None,
            "innate_sha": _sha_t(torch.as_tensor(d["innate"]).float())
            if "innate" in d else None}


def check_frozen_cell(beta, gamma, sweeps, rounds, frozen_dir, model_id):
    """lambda = inf.  The population loop must still be recursive --
    what is frozen is the MODEL, so pred_raw must be constant while
    op_raw moves.  For the frozen kind constancy of the served map IS
    the definitional check, not a diagnostic."""
    errs = []
    name = _frozen_name(beta, gamma, sweeps, rounds)
    p = Path(frozen_dir) / name
    if not p.exists():
        alt = Path(frozen_dir) / _frozen_name(beta, gamma, sweeps, 60)
        if alt.exists():
            p = alt
        else:
            return {"tag": name, "status": "ABSENT",
                    "errors": [f"no frozen replay at {name} (or _r60)"],
                    "notes": []}
    d = torch.load(p, map_location="cpu", weights_only=False)
    cfg = d.get("config", {}) or {}
    if cfg.get("platform") != "frozen_offline_replay":
        errs.append(f"FROZEN platform={cfg.get('platform')!r}")
    if cfg.get("population_update") != "nested_ai_anchored_then_social_v2":
        errs.append(f"FROZEN population_update="
                    f"{cfg.get('population_update')!r} (want the anchored v2)")
    # provenance: the source checkpoint. The 2026-08-24 audit caught nine
    # artifacts whose replay_note claimed Qwen2.5 over a Qwen3-8B vector;
    # base_model is the structured field and must name the real source.
    if cfg.get("base_model") != model_id:
        errs.append(f"FROZEN base_model={cfg.get('base_model')!r} "
                    f"(want {model_id!r} -- the source run's checkpoint)")
    for key, want in (("w_plat", float(beta)), ("innate_k", float(gamma)),
                      ("ab_sweeps", int(sweeps))):
        got = cfg.get(key, "<absent>")
        if not isinstance(got, (int, float)) or abs(float(got) - want) > TOL:
            errs.append(f"FROZEN {key}={got!r} (want {want!r})")
    op = torch.as_tensor(d["op_raw"]).float()
    pred = torch.as_tensor(d["pred_raw"]).float()
    if op.shape[0] < rounds:
        errs.append(f"FROZEN {op.shape[0]} rounds (want >= {rounds})")
    if not _finite(op) or not _finite(pred):
        errs.append("FROZEN non-finite values")
    if pred.ndim == 2 and pred.shape[0] >= 2:
        if len({_sha_t(pred[t]) for t in range(min(pred.shape[0], rounds))}) != 1:
            errs.append("FROZEN served vector MOVES across rounds -- the "
                        "model is supposed to be frozen")
    if op.shape[0] >= 2 and beta > 0.0 and \
            len({_sha_t(op[t]) for t in range(min(op.shape[0], rounds))}) == 1:
        errs.append("FROZEN population is constant -- the population loop "
                    "must still run recursively at lambda=inf")
    return {"tag": name, "status": "PASS" if not errs else "FAIL",
            "errors": errs, "notes": [], "rounds": int(op.shape[0]),
            "mean": float(op[min(rounds, op.shape[0]) - 1].mean()),
            "sd": float(op[min(rounds, op.shape[0]) - 1].std())}


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser(
        description="gate the redesigned Figure 3 grid (hard-fails on any "
                    "absent or ungated cell)")
    ap.add_argument("--run-root", action="append", default=None,
                    help="repeatable; defaults to the cluster path, "
                         "runs/pokec_gated_lm and notes/pofd/cluster")
    ap.add_argument("--frozen-dir", default=str(DEFAULT_FROZEN_DIR))
    ap.add_argument("--smoke", action="store_true",
                    help="gate the 3-round smoke cell instead of the grid")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    roots = [Path(r) for r in (args.run_root or DEFAULT_RUN_ROOTS)]
    g = _load_gen()
    model_id = g.FAM_MODELS[g.F3_MODEL]["base_model"]

    records, ok = [], True
    if args.smoke:
        row = g.f3_smoke_rows()[0]
        c = [x.strip() for x in row.split(",")]
        rec = check_gpu_cell(c[0], float(c[11]), float(c[14]), float(c[2]),
                             g.F3_SMOKE_ROUNDS, g.F3_SWEEPS, model_id, roots)
        rec["kind"] = "gpu"
        records.append(rec)
    else:
        # the grid comes from the generator, so the checker and the wave
        # can never disagree about what Figure 3 is
        twin_by_gamma = {}
        for (beta, gamma, lam, kind) in g.f3_cells():
            if kind == "gpu":
                tag = (g.F3_REUSED.get((beta, gamma, lam))
                       or g.f3_tag(beta, gamma, lam))
                rounds = 60 if tag.endswith("_r60") else g.F3_ROUNDS
                gam = 1.0 if gamma is None else gamma
                rec = check_gpu_cell(tag, beta, gam, lam, rounds,
                                     g.F3_SWEEPS, model_id, roots,
                                     common_rounds=g.F3_ROUNDS)
                # the beta=0 column is read off these twins
                if rec.get("twin_sha"):
                    twin_by_gamma.setdefault(gam, set()).add(rec["twin_sha"])
            elif kind == "frozen":
                gam = 1.0 if gamma is None else gamma
                rec = check_frozen_cell(beta, gam, g.F3_SWEEPS,
                                        g.F3_ROUNDS, args.frozen_dir,
                                        model_id)
            else:                      # twin: no artifact of its own
                continue
            rec.update(kind=kind, beta=beta, gamma=gamma,
                       lam=(None if lam is None else float(lam)))
            records.append(rec)

        # the beta = 0 column: one twin per gamma must serve every lambda.
        for gam, shas in sorted(twin_by_gamma.items()):
            if len(shas) > 1:
                records.append({
                    "tag": f"<beta=0 twin, gamma={gam:g}>", "kind": "twin",
                    "status": "FAIL", "notes": [], "errors": [
                        f"{len(shas)} DIFFERENT twin trajectories at "
                        f"gamma={gam:g} -- at beta=0 the served vector "
                        f"cannot reach the population, so every lambda "
                        f"must share one twin"]})
            else:
                records.append({"tag": f"<beta=0 twin, gamma={gam:g}>",
                                "kind": "twin", "status": "PASS",
                                "errors": [], "notes": []})
        missing_gamma = sorted(set(g.F3_GAMMAS) - set(twin_by_gamma))
        if missing_gamma:
            records.append({"tag": "<beta=0 column>", "kind": "twin",
                            "status": "FAIL", "notes": [], "errors": [
                                f"no run carries a twin at gamma "
                                f"{missing_gamma} -- the beta=0 column of "
                                f"Figure 3 cannot be drawn"]})

        # requested horizon extensions: full checks when present,
        # PENDING-EXT (non-failing) when the job has not run yet
        for (beta, gamma, lam, rounds) in g.f3x_requests():
            gam = 1.0 if gamma is None else gamma
            tag = g.f3_tag(beta, gam, lam, rounds=rounds)
            if _resolve(tag, roots) is None:
                records.append({"tag": tag, "kind": "ext",
                                "status": "PENDING-EXT", "errors": [],
                                "notes": [f"requested {rounds}-round "
                                          f"extension not run yet"]})
                continue
            rec = check_gpu_cell(tag, beta, gam, lam, rounds,
                                 g.F3_SWEEPS, model_id, roots,
                                 common_rounds=g.F3_ROUNDS)
            rec["kind"] = "ext"
            records.append(rec)

    print("=" * 118)
    print(f"FIGURE-3 GRID GATE -- {len(records)} cell(s)")
    print("=" * 118)
    print(f"{'cell':<74}{'kind':<8}{'verdict':<12}{'mean':>9}{'sd':>9}")
    print("-" * 118)
    for r in sorted(records, key=lambda r: (r.get("kind", ""), r["tag"])):
        if r["status"] not in ("PASS", "PENDING-EXT"):
            ok = False
        m = f"{r['mean']:.4f}" if r.get("mean") is not None else "-"
        s = f"{r['sd']:.4f}" if r.get("sd") is not None else "-"
        print(f"{r['tag']:<74}{r.get('kind',''):<8}{r['status']:<12}"
              f"{m:>9}{s:>9}")
        for e in r["errors"]:
            print(f"    !! {e}")
        for n in r.get("notes", []):
            print(f"    NOTE {n}")
    print("=" * 118)
    n_bad = sum(1 for r in records
                if r["status"] not in ("PASS", "PENDING-EXT"))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"ok": ok, "n_cells": len(records), "n_failing": n_bad,
             "cells": records}, indent=2, default=str))
        print(f"[check_f3] verdict -> {args.json}")
    if ok:
        print(f"[check_f3] PASS -- {len(records)} cell(s): every finite-"
              f"lambda cell carries a per-round training record for every "
              f"round (the HARD retraining witness) with a live KL anchor "
              f"where lambda > 0, every config matches its cell, the "
              f"anchored anch2 operator is recorded everywhere, lambda=inf "
              f"is frozen in the model but recursive in the population "
              f"with Qwen3-8B provenance, the beta=0 twins agree across "
              f"lambda, and there are zero parse failures.")
        return 0
    print(f"[check_f3] FAIL -- {n_bad} of {len(records)} cell(s). Figure 3 "
          f"MUST NOT replace the paper placeholder until this is clean.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
