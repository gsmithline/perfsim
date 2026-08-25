#!/usr/bin/env python3
"""Gate the matched open-gate, cross-model Section 3 equilibrium wave.

The required surface is read from ``gen_pofd_sweep.py``: six models x
three seeds, beta=1, gamma=1, forward-KL lambda=2, Deffuant alpha=.5,
100 complete peer sweeps, both gates open, and 30 recursive rounds.
Every production cell is fresh under one provenance; no lambda=0 cell is
part of this figure.  Perfect prediction is the shared innate-mean reference
and therefore needs no model job.

Parse failures are STRICT here: the runner stores a FINITE 0.5 for an
unparsable generation, so pred_raw can never reveal one; parse_fail_frac
lives only in raw_gen_log.json.gz.  Every cell of this wave runs with
SAVE_RAW_GEN=1 and reuses no archived cell, so the log is REQUIRED (a
missing log is a failure, never a fallback), must carry exactly rounds
0..n_rounds-1 once each, parse_fail_frac == 0 in every round, and 723
parsed values per round.

Every generation must also be WELL-FORMED: a number in [0, 1] at the start
of the string (leading-dot ".64" allowed), and the logged parsed value must
equal that number.  This is stricter than the runner's legacy parser, which
read Mistral-7B's ".64 (" as 64 -> 1.0 without flagging a failure; the
check proves, generation by generation, that a cell's served values are
what the model wrote.

PROVENANCE EXEMPTION (S3M_RERUN): Mistral-7B was rerun under
PARSE_MODE=strict with new tags.  The gate admits exactly two git SHAs --
one shared by every kept cell, one shared by every rerun cell -- requires
parse_mode == "strict" on rerun cells and legacy on kept cells, and records
both SHAs in the verdict.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import torch

from check_fig3_full_loop import (
    _load_gen,
    _resolve,
    _sha_t,
    check_arrays,
    check_full_loop,
)

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
DEFAULT_RUN_ROOTS = (
    Path("/home/gsmithline/perfsim/runs/pokec_gated_lm"),
    REPO / "runs" / "pokec_gated_lm",
    REPO / "notes" / "pofd" / "cluster",
)
N_AGENTS = 723
TOL = 1e-9


# Mirror of HFCausalLMModel._parse_strict (perfsim/models/hf_causal_lm.py);
# duplicated so the login-node gate never imports transformers/peft.
# tests/test_section3_model_equilibria.py pins the two to agree.
_STRICT_RE = re.compile(r"^\s*(\d*\.\d+|\d+(?:\.\d*)?)")


def strict_parse(text):
    """(value, ok): well-formed number in [0,1] at the start, else (None, False)."""
    m = _STRICT_RE.match(text or "")
    if m is None:
        return None, False
    try:
        v = float(m.group(1))
    except ValueError:
        return None, False
    if not 0.0 <= v <= 1.0:
        return None, False
    return v, True


def check_raw_generations(run_dir, rounds, errs):
    """STRICT zero-parse-failure gate (no NaN fallback, no escape hatch).

    The Figure-3 helper ``check_parse`` tolerates a missing log by falling
    back to a NaN scan of pred_raw, which is right for that wave's archived
    reuse cells but CANNOT detect a parse failure (the runner stores a
    finite 0.5).  This wave has no archived cells and pins SAVE_RAW_GEN=1,
    so the log itself is the evidence and its absence is a failure."""
    gz = Path(run_dir) / "raw_gen_log.json.gz"
    if not gz.exists():
        errs.append("PARSE raw_gen_log.json.gz ABSENT -- parse_fail_frac is "
                    "recorded nowhere else and the parser stores a finite "
                    "0.5 on failure, so this cell's parse rate cannot be "
                    "established (SAVE_RAW_GEN=1 is mandatory here)")
        return
    rows = []
    try:
        with gzip.open(gz, "rt") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (OSError, ValueError, json.JSONDecodeError) as e:
        errs.append(f"PARSE raw_gen_log.json.gz unreadable: {e}")
        return
    got = [r.get("round") for r in rows]
    if got != list(range(rounds)):
        absent = [t for t in range(rounds) if t not in got]
        errs.append(f"PARSE raw_gen_log must carry exactly rounds "
                    f"0..{rounds - 1} in order, once; it holds "
                    f"{got[:6]}{'...' if len(got) > 6 else ''} -- "
                    f"{len(absent)} round(s) without a logged parse rate")
    bad = []
    for r in rows:
        v = r.get("parse_fail_frac")
        ok = (isinstance(v, (int, float)) and not isinstance(v, bool)
              and float(v) == 0.0)
        if not ok:
            bad.append((r.get("round"), v))
    if bad:
        errs.append(f"PARSE parse_fail_frac must be exactly 0 in every "
                    f"round (an absent field counts as a failure); "
                    f"{len(bad)} bad round(s), e.g. {bad[:4]}")
    short = [(r.get("round"), len(r.get("parsed") or []))
             for r in rows if len(r.get("parsed") or []) != N_AGENTS]
    if short:
        errs.append(f"PARSE {len(short)} round(s) parsed fewer than "
                    f"{N_AGENTS} agents, e.g. {short[:4]}")
    # WELL-FORMED generations: every raw string must strict-parse, and the
    # value the run actually served (parsed[i]) must equal it.  A legacy
    # cell whose parser silently clamped ".64 (" -> 1.0 fails here.
    malformed, mismatched, total = [], [], 0
    for r in rows:
        raws = r.get("raw") or []
        parsed = r.get("parsed") or []
        if len(raws) != len(parsed):
            errs.append(f"PARSE round {r.get('round')} logs {len(raws)} raw "
                        f"strings but {len(parsed)} parsed values")
            continue
        for i, (txt, pv) in enumerate(zip(raws, parsed)):
            total += 1
            v, ok = strict_parse(txt)
            if not ok:
                malformed.append((r.get("round"), i, str(txt)[:20]))
            elif abs(float(pv) - v) > 1e-6:
                mismatched.append((r.get("round"), i, str(txt)[:20],
                                   float(pv)))
    if malformed:
        errs.append(f"PARSE {len(malformed)}/{total} generation(s) are not "
                    f"a well-formed number in [0,1] at the start of the "
                    f"string, e.g. {malformed[:3]} -- the served value is "
                    f"not what the model wrote")
    if mismatched:
        errs.append(f"PARSE {len(mismatched)}/{total} served value(s) "
                    f"differ from the number the model wrote, e.g. "
                    f"{mismatched[:3]}")
    return {"generations": total, "malformed": len(malformed),
            "mismatched": len(mismatched)}


def check_kl_witness_every_round(run_dir, rounds, errs):
    """Per-round LIVE KL-gradient witness.

    ``check_full_loop`` accepts a single positive grad_kl_norm0 in any round
    after round 0.  This wave rests on the anchor binding in EVERY round
    (a fresh adapter is regularised toward the same reference each round),
    and the pulled Figure-3 lambda=2 fresh-LoRA cells record a positive
    grad_kl_norm0 in all 30 rounds, so require it round by round."""
    tel = Path(run_dir) / "telemetry.json"
    if not tel.exists():
        return  # check_full_loop already reported the absence
    rows = []
    for line in tel.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    by_round = {}
    for r in rows:
        if "grad_kl_norm0" in r and r.get("round") is not None:
            by_round[int(r["round"])] = r["grad_kl_norm0"]
    bad = []
    for t in range(rounds):
        v = by_round.get(t)
        try:
            live = v is not None and float(v) > 0.0
        except (TypeError, ValueError):
            live = False
        if not live:
            bad.append((t, v))
    if bad:
        errs.append(f"KL-WITNESS grad_kl_norm0 must be recorded and > 0 in "
                    f"every round 0..{rounds - 1}; {len(bad)} round(s) fail, "
                    f"e.g. {bad[:4]}")


def check_runtime_open_gates(d, rounds, errs):
    """Runtime evidence (not just config) that both gates were open in
    EVERY round: the trajectory rows carry contact (the AI-gate open
    fraction, 1.0 under all_open), peer_gate_mode, and the all-open peer
    invariant peer_pairs == accepted == 723 * ab_sweeps."""
    tr = d.get("trajectory", []) or []
    cfg = d.get("config", {}) or {}
    sweeps = cfg.get("ab_sweeps")
    if not isinstance(sweeps, int) or sweeps <= 0:
        errs.append(f"GATE-RUNTIME config ab_sweeps={sweeps!r} unusable")
        return
    want_pairs = N_AGENTS * sweeps
    bad = []
    for t in range(min(rounds, len(tr))):
        row = tr[t] or {}
        try:
            contact_ok = abs(float(row.get("contact")) - 1.0) <= TOL
        except (TypeError, ValueError):
            contact_ok = False
        mode_ok = row.get("peer_gate_mode") == "all_open"
        pairs, acc = row.get("peer_pairs"), row.get("accepted")
        pairs_ok = (isinstance(pairs, int) and isinstance(acc, int)
                    and pairs == want_pairs and acc == pairs)
        if not (contact_ok and mode_ok and pairs_ok):
            bad.append((t, row.get("contact"), row.get("peer_gate_mode"),
                        pairs, acc))
    if len(tr) < rounds:
        bad.append(("rows", len(tr), "<", rounds))
    if bad:
        errs.append(f"GATE-RUNTIME every round must record contact == 1.0, "
                    f"peer_gate_mode == 'all_open' and peer_pairs == "
                    f"accepted == {want_pairs}; {len(bad)} round(s) fail, "
                    f"e.g. {bad[:3]}")


def _eq(cfg, key, want, errs):
    got = cfg.get(key, "<absent>")
    if isinstance(want, float):
        ok = isinstance(got, (int, float)) and abs(float(got) - want) <= TOL
    else:
        ok = got == want
    if not ok:
        errs.append(f"CONFIG {key}={got!r} (want {want!r})")


def check_cell(tag, model, seed, rounds, g, roots, strict=False):
    errs, notes = [], []
    path = _resolve(tag, roots)
    if path is None:
        return {"tag": tag, "model": model, "seed": seed,
                "status": "ABSENT", "errors": ["trajectory.pt absent"],
                "notes": []}

    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = d.get("config", {}) or {}
    expected = {
        "w_plat": g.S3M_BETA,
        "innate_lambda": g.S3M_GAMMA,
        "kl_beta": g.S3M_LAMBDA,
        "deffuant_alpha": g.S3M_ALPHA,
        "ab_sweeps": g.S3M_SWEEPS,
        "n_rounds": rounds,
        "seed": seed,
        "dataset": "movielens",
        "ml_target": "Action",
        "base_model": g.FAM_MODELS[model]["base_model"],
        "training_style": "sft_kl",
        "kl_direction": "forward",
        "kl_ref_adapter": "",
        "ai_gate_mode": "all_open",
        "peer_gate_mode": "all_open",
        "ai_gate_reference": "anchor",
        "population_update": "nested_ai_anchored_then_social_v2",
        "pop_model": "ab",
        "icl_k": 0,
        "train_cap": N_AGENTS,
        "n_labeled": N_AGENTS,
        "seed_base_data": True,
        "lora_r": 512,
        "sft_epochs": 1,
        "sft_batch_size": 4,
        "sft_lr": 5e-5,
        "use_lora": True,
        "fresh_each_round": True,
        "save_raw_gen": True,
        "serve_eval_mode": True,
        "do_sample": False,
    }
    for key, want in expected.items():
        _eq(cfg, key, want, errs)
    if not isinstance(cfg.get("git_sha"), str) or not cfg["git_sha"].strip():
        errs.append("CONFIG git_sha absent or empty")
    hg = cfg.get("homophily_gamma", cfg.get("gamma_bias", 0.0))
    if hg not in (0, 0.0):
        errs.append(f"CONFIG homophily gamma={hg!r} (want 0.0)")
    if model == "qwen3_8b" and cfg.get("chat_thinking") is not False:
        errs.append(f"CONFIG Qwen3 chat_thinking={cfg.get('chat_thinking')!r} "
                    "(want False)")
    pm = cfg.get("parse_mode", "legacy")
    if strict and pm != "strict":
        errs.append(f"CONFIG parse_mode={pm!r} (rerun cell must be 'strict')")
    if not strict and pm != "legacy":
        errs.append(f"CONFIG parse_mode={pm!r} (kept cell must be legacy; "
                    "a strict-parsed cell needs the rerun tag)")

    check_arrays(d, rounds, errs)
    check_runtime_open_gates(d, rounds, errs)
    if all(not e.startswith("ARTIFACT") for e in errs):
        check_full_loop(d, path.parent, g.S3M_LAMBDA, rounds, errs, notes)
        check_kl_witness_every_round(path.parent, rounds, errs)
        gen_stats = check_raw_generations(path.parent, rounds, errs)
    else:
        gen_stats = None

    op = torch.as_tensor(d.get("op_raw", torch.empty(0))).float()
    rec = {
        "tag": tag,
        "model": model,
        "seed": seed,
        "status": "PASS" if not errs else "FAIL",
        "errors": errs,
        "notes": notes,
        "git_sha": cfg.get("git_sha"),
        "parse_mode": pm,
        "rerun": bool(strict),
        "generations": gen_stats,
        "innate_sha": (_sha_t(torch.as_tensor(d["innate"]).float())
                       if "innate" in d else None),
        "twin_sha": (_sha_t(torch.as_tensor(d["twin_raw"]).float()[:rounds])
                     if "twin_raw" in d else None),
    }
    if op.ndim == 2 and op.shape[0] >= rounds:
        rec["mean"] = float(op[rounds - 1].mean())
        rec["sd"] = float(op[rounds - 1].std())
    return rec


def main():
    ap = argparse.ArgumentParser(
        description="gate the Section 3 open-gate cross-model equilibrium wave")
    ap.add_argument("--run-root", action="append", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="the 3-round OLMo-3 smoke (legacy parse)")
    ap.add_argument("--rerun-smoke", action="store_true",
                    help="the 3-round Mistral-7B PARSE_MODE=strict smoke")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    roots = [Path(r) for r in (args.run_root or DEFAULT_RUN_ROOTS)]
    g = _load_gen()
    smoke = args.smoke or args.rerun_smoke
    if args.smoke and args.rerun_smoke:
        ap.error("--smoke and --rerun-smoke are exclusive")
    if args.smoke:
        cells = [(g.S3M_SMOKE_MODEL, g.S3M_SMOKE_SEED,
                  g.s3m_tag(g.S3M_SMOKE_MODEL, g.S3M_SMOKE_SEED,
                            rounds=g.S3M_SMOKE_ROUNDS, smoke=True),
                  g.S3M_SMOKE_ROUNDS, False)]
    elif args.rerun_smoke:
        cells = [(g.S3M_RERUN_SMOKE_MODEL, g.S3M_SMOKE_SEED,
                  g.s3m_tag(g.S3M_RERUN_SMOKE_MODEL, g.S3M_SMOKE_SEED,
                            rounds=g.S3M_SMOKE_ROUNDS, smoke=True,
                            strict=True),
                  g.S3M_SMOKE_ROUNDS, True)]
    else:
        # rerun models resolve to their strict-parse tags (s3m_cell_tag)
        cells = [(model, seed, g.s3m_cell_tag(model, seed), g.S3M_ROUNDS,
                  model in g.S3M_RERUN)
                 for model in g.S3M_MODELS for seed in g.S3M_SEEDS]

    records = [check_cell(tag, model, seed, rounds, g, roots, strict=strict)
               for model, seed, tag, rounds, strict in cells]
    provenance = None

    # All production cells must share the same population; code provenance
    # is one SHA per group (kept cells / strict-parse rerun cells).
    if not smoke:
        innate = {r.get("innate_sha") for r in records if r.get("innate_sha")}
        if len(innate) != 1:
            records.append({"tag": "<cross-cell innate>", "model": "-",
                            "seed": "-", "status": "FAIL", "notes": [],
                            "errors": [f"{len(innate)} distinct innate hashes; "
                                       "all model cells must share one population"]})
        kept = {r.get("git_sha") for r in records
                if r.get("git_sha") and not r.get("rerun")}
        rerun = {r.get("git_sha") for r in records
                 if r.get("git_sha") and r.get("rerun")}
        n_kept = sum(1 for r in records if "rerun" in r and not r["rerun"])
        n_rerun = sum(1 for r in records if r.get("rerun"))
        if len(kept) != 1:
            records.append({"tag": "<kept-cell provenance>", "model": "-",
                            "seed": "-", "status": "FAIL", "notes": [],
                            "errors": [f"{len(kept)} distinct git SHAs across "
                                       f"the {n_kept} kept cells; they must "
                                       "share one provenance"]})
        if n_rerun and len(rerun) != 1:
            records.append({"tag": "<rerun-cell provenance>", "model": "-",
                            "seed": "-", "status": "FAIL", "notes": [],
                            "errors": [f"{len(rerun)} distinct git SHAs across "
                                       f"the {n_rerun} rerun cells; they must "
                                       "share one provenance"]})
        provenance = {
            "kept_cells": n_kept, "kept_sha": sorted(kept),
            "rerun_cells": n_rerun, "rerun_sha": sorted(rerun),
            "rerun_models": sorted(g.S3M_RERUN),
            "exemption": ("S3M_RERUN: Mistral-7B rerun under PARSE_MODE="
                          "strict after the legacy parser read '.64 (' as "
                          "1.0; kept cells are proven well-formed on every "
                          "generation (see 'generations' per cell)"),
        }
        for seed in g.S3M_SEEDS:
            twins = {r.get("twin_sha") for r in records
                     if r.get("seed") == seed and r.get("twin_sha")}
            if len(twins) != 1:
                records.append({"tag": f"<seed {seed} twins>", "model": "-",
                                "seed": seed, "status": "FAIL", "notes": [],
                                "errors": [f"{len(twins)} distinct no-platform "
                                           "twins across models"]})

    ok = all(r["status"] == "PASS" for r in records)
    print("=" * 112)
    print(f"SECTION-3 MODEL EQUILIBRIA GATE -- {len(records)} record(s)")
    print("=" * 112)
    print(f"{'model':<14}{'seed':>6}  {'verdict':<9}{'mean':>10}{'sd':>10}  cell")
    for r in records:
        mean = f"{r['mean']:.4f}" if "mean" in r else "-"
        sd = f"{r['sd']:.4f}" if "sd" in r else "-"
        print(f"{r.get('model','-'):<14}{str(r.get('seed','-')):>6}  "
              f"{r['status']:<9}{mean:>10}{sd:>10}  {r['tag']}")
        for e in r["errors"]:
            print(f"    !! {e}")
        for n in r.get("notes", []):
            print(f"    NOTE {n}")

    n_bad = sum(r["status"] != "PASS" for r in records)
    if provenance:
        print(f"PROVENANCE kept {provenance['kept_cells']} cell(s) @ "
              f"{provenance['kept_sha']}; rerun {provenance['rerun_cells']} "
              f"cell(s) ({', '.join(provenance['rerun_models'])}) @ "
              f"{provenance['rerun_sha']} -- {provenance['exemption']}")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ok": ok, "n_cells": len(cells),
                                   "n_failing": n_bad, "cells": records,
                                   "provenance": provenance},
                                  indent=2, default=str))
        print(f"[check_s3m] verdict -> {out}")
    if ok:
        print("[check_s3m] PASS -- every model-seed cell matches the fixed "
              "surface, retrained in every round with a live KL anchor, "
              "served well-formed numbers without parse failures, and "
              "shares one population and one provenance per group.")
        return 0
    print(f"[check_s3m] FAIL -- {n_bad} failing record(s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
