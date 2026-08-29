#!/usr/bin/env python3
"""GATE for the CORRECTED continual-adaptation control (pofdcac_,
2026-08-29): fresh vs carried LoRA weights under the anch2 operator.

WHAT THE WAVE CLAIMS, and therefore what is checked:

 1. THE OPERATOR IS ACTUALLY CORRECTED, AND THE CORRECTION CAN BITE.
    population_update == nested_ai_anchored_then_social_v2 and
    ai_gate_reference == "anchor" -- and the AI gate is a NUMERIC
    THRESHOLD, never all_open.  _gated_pop.ai_gate returns an all-ones
    mask under all_open before the gate reference is ever read, so at an
    open-gate surface anch2 and the legacy operator are numerically
    identical and this control would be a tautology.  An all_open cell
    is therefore a HARD FAILURE here, not a variant.
 2. THE TWO ARMS DIFFER IN EXACTLY ONE DIAL.  For every (lambda, seed)
    the fresh and continual configs must agree on every recorded dial
    except fresh_each_round, and the tag's arm token must agree with
    that flag -- otherwise a continual cell could be filed as a fresh
    one and the contrast would be backwards.
 3. THE LOOP RETRAINED EVERY ROUND on all 723 live labels, with a live
    KL anchor where lambda > 0 and NO KL term where lambda = 0.
 4. THE ADAPTER ACTUALLY MOVED.  b_norm/ba_norm, not w_norm: PEFT
    initialises lora_A at random and lora_B at exactly zero, so ||theta||
    is nonzero before any update and cannot witness training.
 5. ZERO MALFORMED GENERATIONS, counted rather than bounded.  This wave
    sets SAVE_RAW_GEN=1, which the archived fec wave did not; there the
    malformed rate could only be bounded from the served values.  Here
    the log is mandatory, every raw string must be well formed, and the
    logged `parsed` vector must equal pred_raw round by round.

  python check_continual_anch2.py [--smoke]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import gzip
import json
import math
import re
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "condor"))

N = 723
DEFAULT_ROOTS = (Path("/home/gsmithline/perfsim/runs/pokec_gated_lm"),
                 REPO / "runs" / "pokec_gated_lm",
                 REPO / "notes" / "pofd" / "cluster")
STRICT_NUM = re.compile(r"^\s*(\d*\.\d+|\d+(?:\.\d*)?)\s*$")
CORRECTED = "nested_ai_anchored_then_social_v2"


def _find(tag, roots):
    for r in roots:
        p = Path(r) / tag / "trajectory.pt"
        if p.exists():
            return p
    return None


def _jsonl(p):
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def check_cell(g, lam, arm, fresh, seed, rounds, roots, smoke=False):
    tag = g.cac_tag(lam, arm, seed, rounds, smoke)
    errs, rec = [], {"tag": tag, "lam": lam, "arm": arm, "seed": seed}
    p = _find(tag, roots)
    if p is None:
        return {**rec, "status": "ABSENT", "errors": ["no trajectory.pt"]}
    d = torch.load(p, map_location="cpu", weights_only=False)
    cfg = d.get("config", {}) or {}
    rec["config"] = cfg

    # ---- 1. operator, and a gate where the correction can bite --------
    if cfg.get("population_update") != CORRECTED:
        errs.append(f"OPERATOR population_update="
                    f"{cfg.get('population_update')!r} (want {CORRECTED!r})")
    if cfg.get("ai_gate_reference") != "anchor":
        errs.append(f"OPERATOR ai_gate_reference="
                    f"{cfg.get('ai_gate_reference')!r} (want 'anchor')")
    if cfg.get("ai_gate_mode") == "all_open":
        errs.append("OPERATOR ai_gate_mode=all_open -- the AI gate then "
                    "returns all-ones BEFORE the gate reference is read, "
                    "so anch2 and the legacy operator are numerically "
                    "identical and this control proves nothing")
    elif cfg.get("ai_gate_mode") not in ("threshold", None):
        errs.append(f"OPERATOR ai_gate_mode={cfg.get('ai_gate_mode')!r}")

    # ---- 2. the cell is the cell it claims to be ----------------------
    for key, want in (("kl_beta", float(lam)), ("seed", seed),
                      ("n_rounds", rounds), ("w_plat", g.CAC_W_PLAT),
                      ("innate_lambda", g.CAC_INNATE_LAMBDA),
                      ("eps", g.CAC_EPS_SOCIAL), ("eps_ai", g.CAC_EPS_AI),
                      ("ab_sweeps", g.CAC_SWEEPS), ("lora_r", 512),
                      ("train_cap", N), ("n_labeled", N),
                      ("data_regime", "replace"), ("dataset", "movielens"),
                      ("ml_target", "Action"), ("pristine_frac", 0.0),
                      ("training_style", "sft" if lam == 0 else "sft_kl")):
        got = cfg.get(key, "<absent>")
        ok = (isinstance(got, (int, float)) and not isinstance(got, bool)
              and abs(float(got) - want) <= 1e-9
              if isinstance(want, float) else got == want)
        if not ok:
            errs.append(f"CONFIG {key}={got!r} (want {want!r})")
    if lam > 0 and cfg.get("kl_direction") != "forward":
        errs.append(f"CONFIG kl_direction={cfg.get('kl_direction')!r}")
    if bool(cfg.get("fresh_each_round")) is not bool(fresh):
        errs.append(f"CONFIG fresh_each_round="
                    f"{cfg.get('fresh_each_round')!r} but the tag says "
                    f"{arm!r} -- the arms would be swapped")
    hg = cfg.get("homophily_gamma", cfg.get("gamma_bias", 0.0))
    if hg not in (0, 0.0):
        errs.append(f"CONFIG homophily gamma={hg!r} (want 0.0)")

    # ---- 3. complete, finite trajectory -------------------------------
    for name in ("op_raw", "pred_raw", "twin_raw"):
        t = d.get(name)
        if not torch.is_tensor(t) or tuple(t.shape) != (rounds, N):
            errs.append(f"ARTIFACT {name} shape "
                        f"{tuple(t.shape) if torch.is_tensor(t) else None} "
                        f"!= {(rounds, N)}")
            continue
        t = t.float()
        if not torch.isfinite(t).all():
            errs.append(f"ARTIFACT {name} non-finite")
        elif float(t.min()) < -1e-6 or float(t.max()) > 1 + 1e-6:
            errs.append(f"ARTIFACT {name} outside [0,1]")

    # ---- 4. retrained every round; anchor present iff lambda > 0 ------
    rows = _jsonl(p.parent / "telemetry.json")
    trained = {int(r["round"]) for r in rows
               if "l_init" in r and int(r.get("n_train", 0) or 0) == N}
    missing = sorted(set(range(rounds)) - trained)
    if missing:
        errs.append(f"TRAINING only {len(trained)} of {rounds} round(s) "
                    f"trained on all {N} live labels; missing "
                    f"{missing[:6]}")
    gk = [float(r["grad_kl_norm0"]) for r in rows
          if r.get("grad_kl_norm0") is not None]
    if lam == 0.0:
        live = [v for v in gk if abs(v) > 0.0]
        if live:
            errs.append(f"TRAINING lambda=0 but a KL anchor gradient is "
                        f"nonzero in {len(live)} round(s) (max "
                        f"{max(live):.4g}) -- an SFT cell carries no KL")
    else:
        post = [float(r["grad_kl_norm0"]) for r in rows
                if int(r.get("round", 0)) > 0
                and r.get("grad_kl_norm0") is not None]
        if not post:
            errs.append("TRAINING lambda>0 but no grad_kl_norm0 recorded "
                        "after round 0 -- cannot show the anchor bound")
        elif all(v == 0.0 for v in post):
            errs.append("TRAINING lambda>0 but the KL gradient is "
                        "identically 0 after round 0")
        elif not all(math.isfinite(v) for v in post):
            errs.append("TRAINING non-finite KL gradient norm")
        else:
            rec["grad_kl_norm0_mean"] = sum(post) / len(post)

    # ---- 5. the adapter actually moved --------------------------------
    bn = [(int(r["round"]), float(r["b_norm"]), float(r.get("ba_norm", 0)))
          for r in rows if r.get("b_norm") is not None]
    if not bn:
        errs.append("ADAPTER no b_norm telemetry -- ||B|| is the only "
                    "witness the optimizer moved anything (w_norm is "
                    "nonzero even with B identically zero)")
    else:
        dead = [x for x in bn if x[1] <= 0.0 or x[2] <= 0.0]
        if dead:
            errs.append(f"ADAPTER B or BA identically zero in "
                        f"{len(dead)} round(s), e.g. {dead[:3]}")
        rec["b_norm_min"] = min(b for _, b, _ in bn)
        rec["w_norm_first_last"] = [
            next((float(r["w_norm"]) for r in rows if r.get("w_norm")
                  is not None), None),
            next((float(r["w_norm"]) for r in reversed(rows)
                  if r.get("w_norm") is not None), None)]

    # ---- 6. zero MALFORMED generations, counted -----------------------
    gz = p.parent / "raw_gen_log.json.gz"
    if not gz.exists():
        errs.append("PARSE raw_gen_log.json.gz ABSENT -- the parser "
                    "stores a finite default on failure, so the "
                    "malformed rate cannot be established anywhere else "
                    "(SAVE_RAW_GEN=1 is mandatory for this wave)")
    else:
        grows = []
        try:
            with gzip.open(gz, "rt") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        grows.append(json.loads(line))
        except (OSError, ValueError):
            errs.append("PARSE raw_gen_log.json.gz unreadable")
        got = [r.get("round") for r in grows]
        if got != list(range(rounds)):
            errs.append(f"PARSE raw log must carry rounds 0..{rounds - 1} "
                        f"once, in order; got {got[:6]}")
        pred = d.get("pred_raw")
        bad, mism, total = [], [], 0
        for r in grows:
            for i, txt in enumerate(r.get("raw") or []):
                total += 1
                m = STRICT_NUM.match(str(txt).strip())
                if not m or not (0.0 <= float(m.group(1)) <= 1.0):
                    bad.append((r.get("round"), i, str(txt)[:14]))
            t = r.get("round")
            vals = r.get("parsed") or []
            if torch.is_tensor(pred) and isinstance(t, int) \
                    and t < pred.shape[0] and len(vals) == N:
                pv = torch.tensor([float(v) for v in vals])
                dmax = float((pv.clamp(0, 1)
                              - pred[t].float().clamp(0, 1)).abs().max())
                if dmax > 1e-6:
                    mism.append((t, dmax))
            if r.get("parse_fail_frac") not in (0, 0.0):
                errs.append(f"PARSE round {t} parse_fail_frac="
                            f"{r.get('parse_fail_frac')!r} (want 0)")
                break
        if bad:
            errs.append(f"PARSE {len(bad)}/{total} malformed generation(s), "
                        f"e.g. {bad[:3]}")
        if mism:
            errs.append(f"PARSE the logged `parsed` vector != pred_raw in "
                        f"{len(mism)} round(s), e.g. {mism[:3]} -- the raw "
                        f"log does not describe this trajectory")
        rec["n_generations"] = total
        rec["n_malformed"] = len(bad)

    op = d["op_raw"].float() if torch.is_tensor(d.get("op_raw")) else None
    if op is not None and op.shape[0] >= rounds:
        rec["final_mean"] = float(op[rounds - 1].mean())
        rec["final_sd"] = float(op[rounds - 1].std())
    rec["status"] = "PASS" if not errs else "FAIL"
    rec["errors"] = errs
    return rec


PAIR_KEYS = ("kl_beta", "seed", "n_rounds", "w_plat", "innate_lambda",
             "eps", "eps_ai", "ab_sweeps", "lora_r", "sft_lr",
             "sft_epochs", "sft_batch_size", "train_cap", "n_labeled",
             "data_regime", "dataset", "ml_target", "base_model",
             "kl_direction", "training_style", "population_update",
             "ai_gate_mode", "ai_gate_reference", "pristine_frac",
             "icl_k", "icl_days", "do_sample", "seed_base_data")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-root", action="append", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    roots = [Path(r) for r in (args.run_root or DEFAULT_ROOTS)]
    import gen_pofd_sweep as g

    recs = []
    if args.smoke:
        recs.append(check_cell(g, 2.0, "adcont", 0, 0,
                               g.CAC_SMOKE_ROUNDS, roots, smoke=True))
    else:
        for lam in g.CAC_LAMS:
            for arm, fresh in g.CAC_ARMS:
                for seed in g.CAC_SEEDS:
                    recs.append(check_cell(g, lam, arm, fresh, seed,
                                           g.CAC_ROUNDS, roots))
        # THE PAIRING: one dial may differ, and only one
        by = {(r["lam"], r["arm"], r["seed"]): r for r in recs
              if r.get("config")}
        for lam in g.CAC_LAMS:
            for seed in g.CAC_SEEDS:
                a = by.get((lam, "adfresh", seed))
                b = by.get((lam, "adcont", seed))
                if not a or not b:
                    continue
                diff = [k for k in PAIR_KEYS
                        if a["config"].get(k) != b["config"].get(k)]
                if diff:
                    recs.append({
                        "tag": f"<pairing lambda={lam:g} seed={seed}>",
                        "status": "FAIL", "errors": [
                            f"fresh and continual differ on {diff} -- the "
                            f"arms must differ ONLY in fresh_each_round"]})

    ok = all(r["status"] == "PASS" for r in recs)
    hdr = (f"{'cell':<62}{'verdict':<9}{'mean':>8}{'sd':>9}"
           f"{'malformed':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        print(f"{r['tag'][:60]:<62}{r['status']:<9}"
              f"{r.get('final_mean', float('nan')):>8.4f}"
              f"{r.get('final_sd', float('nan')):>9.5f}"
              f"{r.get('n_malformed', '-'):>11}")
        for e in r["errors"]:
            print(f"    !! {e}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"ok": ok, "cells": [{k: v for k, v in r.items()
                                  if k != "config"} for r in recs]},
            indent=2, default=str))
    print(f"\n[check_cac] {'PASS' if ok else 'FAIL'} -- "
          f"{sum(1 for r in recs if r['status'] == 'PASS')}/{len(recs)} "
          f"cell(s): corrected operator on a NUMERIC gate where it can "
          f"bite, arms differing only in fresh_each_round, every round "
          f"retrained on all 723 live labels, the adapter demonstrably "
          f"moved, and zero malformed generations.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
