#!/usr/bin/env python3
"""Gate the SECTION-3 PERSONAL-HISTORY ICL wave (section3_model_icl,
tags pofds3i_) -- the Figure-3(a) analogue with reference-regularized
SFT replaced by FROZEN personal-history ICL.

WHAT THIS FILE PROVES, and why each check exists.

1. THE ENVIRONMENT IS EXACT. Every config field of the S3M surface is
   pinned to the same value (W=1, k=1, S=100, alpha=.5, both gates
   all_open, anch2 operator, movielens/Action, 723 agents, greedy
   serving, seeds 0/42/43), so the only difference between the two
   waves is the adaptation channel. A wave that drifts on any of these
   cannot be paired with the SFT one.

2. THE WEIGHTS ARE FROZEN. training_style == "frozen", use_lora False,
   kl_beta 0, sft_epochs 0, fresh_each_round False -- and, because a
   config field is a claim rather than evidence, NO adapter artifact may
   exist in the run dir (round0_adapter/, trl/) and no KL/grad witness
   may be recorded. There is no optimizer here, so anything an optimizer
   would leave behind is a contradiction.

3. THE PERSONAL-HISTORY LOG IS COMPLETE AND REPLAYS BYTE-EXACTLY.
   icl_days_log.json.gz must hold exactly rounds 0..n-1, each with 723
   rendered sentences, and every sentence must replay CHARACTER FOR
   CHARACTER from (innate, op_raw). That single replay simultaneously
   proves: the context is the agent's OWN last <= 8 post-peer opinions,
   in the right window and order, and nothing else.

4. NO CROSS-AGENT CONTEXT. icl_k == 0, icl_ctx_log.json.gz absent, and
   the cross-user exemplar tensors (icl_idx_raw / icl_val_raw) empty.
   The replay in (3) already forbids a foreign value from appearing;
   these are the independent structural statements of the same fact.

5. THE TRAJECTORY IS COMPLETE. op_raw / pred_raw shaped [rounds, 723],
   finite, in [0,1], with a non-degenerate twin, and the runtime
   open-gate evidence (contact == 1.0, peer_gate_mode all_open,
   peer_pairs == accepted == 723 * S) present in EVERY round.

6. ZERO PARSE FAILURES AND ZERO MALFORMED GENERATIONS, ON EVERY MODEL.
   PARSE_MODE=strict is pinned wave-wide, so a malformed generation is
   COUNTED rather than silently clamped to a default. MISTRAL IS NOT
   EXEMPT: this file has no provenance exemption, no per-model tolerance
   and no legacy-parser path. The S3M wave needed a Mistral-only rerun
   because the legacy parser read ".64 (" as 1.0 with no failure
   flagged; here that is a hard failure like any other.

Read-only. CPU only -- no model is loaded, and one trajectory is
resident at a time.

  python check_section3_model_icl.py [--smoke] [--json out.json]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
DEFAULT_RUN_ROOTS = (
    Path("/home/gsmithline/perfsim/runs/pokec_gated_lm"),
    REPO / "runs" / "pokec_gated_lm",
    REPO / "notes" / "pofd" / "cluster",
)
LOG = "[check_s3i]"
N_AGENTS = 723
TOL = 1e-9
ICL_DAYS = 8


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The SFT wave's gate supplies the strict raw-generation reader and the
# runtime open-gate evidence; the Section-4 gate supplies the byte-exact
# personal-history replay. Both are imported rather than re-implemented
# so a fix to either lands here too.
S3M = _load(HERE / "check_section3_model_equilibria.py", "_s3m_gate")
S4G = _load(HERE / "check_section4_gate.py", "_s4g_gate")


def _load_gen():
    return _load(REPO / "experiments" / "condor" / "gen_pofd_sweep.py",
                 "_gen_s3i")


def _resolve(tag, roots):
    for r in roots:
        p = Path(r) / tag / "trajectory.pt"
        if p.exists():
            return p
    return None


def _sha_t(t):
    return hashlib.sha256(
        t.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()


# ------------------------------------------------------------- checks
def check_frozen(run_dir, d, cfg, errs):
    """(2) No optimizer ran. The config says frozen; the RUN DIR must
    agree -- an adapter or a KL/grad witness is physical evidence that
    something trained."""
    for name in ("round0_adapter", "trl"):
        if (run_dir / name).exists():
            errs.append(f"FROZEN {name}/ exists in the run dir -- a frozen "
                        f"arm has no adapter and no trainer state")
    for k in ("grad_kl_norm0", "witness_steps", "witness_lora_b_norm",
              "witness_kl_last"):
        for t, row in enumerate(d.get("trajectory", []) or []):
            if isinstance(row, dict) and row.get(k) is not None:
                errs.append(f"FROZEN trajectory round {t} records {k}="
                            f"{row.get(k)!r} -- an optimizer witness on a "
                            f"frozen arm")
                break


def check_no_cross_agent_context(run_dir, d, cfg, errs):
    """(4) Nothing from another agent may reach a prompt."""
    if int(cfg.get("icl_k") or 0) != 0:
        errs.append(f"LOCALITY icl_k={cfg.get('icl_k')!r} -- cross-user "
                    f"exemplars are forbidden on this wave")
    if (run_dir / "icl_ctx_log.json.gz").exists():
        errs.append("LOCALITY icl_ctx_log.json.gz present -- that log only "
                    "exists when cross-user context is rendered")
    for k in ("icl_idx_raw", "icl_val_raw"):
        v = d.get(k)
        if torch.is_tensor(v) and v.numel():
            errs.append(f"LOCALITY {k} is non-empty ({tuple(v.shape)}) -- a "
                        f"cross-user exemplar artifact")


def check_personal_history(run_dir, d, rounds, cfg, errs, rec):
    """(3) The rendered contexts exist for every round and replay
    byte-exactly from (innate, op_raw)."""
    p = run_dir / "icl_days_log.json.gz"
    if not p.exists():
        errs.append("HISTORY icl_days_log.json.gz ABSENT -- the rendered "
                    "personal-history contexts are the only evidence of "
                    "what each prompt carried, and are mandatory")
        return
    try:
        with gzip.open(p, "rt") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    except (OSError, ValueError) as e:
        errs.append(f"HISTORY icl_days_log.json.gz unreadable: {e}")
        return
    got = [r.get("round") for r in rows]
    if got != list(range(rounds)):
        errs.append(f"HISTORY log must hold exactly rounds 0..{rounds - 1} "
                    f"once each, in order; it holds {got[:6]}"
                    f"{'...' if len(got) > 6 else ''}")
        return
    short = [r.get("round") for r in rows
             if not isinstance(r.get("ctx"), list)
             or len(r["ctx"]) != N_AGENTS]
    if short:
        errs.append(f"HISTORY round {short[0]} does not carry {N_AGENTS} "
                    f"rendered contexts")
        return
    innate = torch.as_tensor(d["innate"]).float()
    op = torch.as_tensor(d["op_raw"]).float()
    target = cfg.get("ml_target") or "Action"
    fail = S4G.replay_personal_history(innate, op, rows, ICL_DAYS, target)
    if fail is not None:
        errs.append(f"HISTORY personal-history context is OFF the byte-exact "
                    f"(innate, op_raw) replay at round {fail[0]} agent "
                    f"{fail[1]}: {fail[2]}")
    else:
        rec["history_replay"] = "byte-exact"


def check_arrays(d, rounds, errs):
    """(5) Complete, finite, in-range trajectory with a live twin."""
    for name in ("op_raw", "pred_raw"):
        t = d.get(name)
        if not torch.is_tensor(t) or tuple(t.shape) != (rounds, N_AGENTS):
            errs.append(f"ARTIFACT {name} shape "
                        f"{tuple(t.shape) if torch.is_tensor(t) else None} "
                        f"!= {(rounds, N_AGENTS)}")
            continue
        t = t.float()
        if not torch.isfinite(t).all():
            errs.append(f"ARTIFACT {name} has non-finite values")
        elif name == "op_raw" and (float(t.min()) < -1e-6
                                   or float(t.max()) > 1 + 1e-6):
            errs.append(f"ARTIFACT op_raw outside [0,1]: "
                        f"[{float(t.min()):.4f}, {float(t.max()):.4f}]")
    inn = d.get("innate")
    if not torch.is_tensor(inn) or tuple(inn.shape) != (N_AGENTS,):
        errs.append("ARTIFACT innate missing or wrong shape")
    tw = d.get("twin_raw")
    if not torch.is_tensor(tw) or tw.numel() == 0:
        errs.append("ARTIFACT twin_raw missing (WITH_TWIN=1 is pinned)")
    elif float(tw.float().std()) == 0.0:
        errs.append("ARTIFACT twin_raw is CONSTANT -- a degenerate twin is "
                    "not a counterfactual")


def check_cell(tag, model, seed, rounds, g, roots, arm="greedy"):
    errs, notes = [], []
    path = _resolve(tag, roots)
    if path is None:
        return {"tag": tag, "model": model, "seed": seed, "status": "ABSENT",
                "errors": ["trajectory.pt absent"], "notes": []}
    run_dir = path.parent
    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = d.get("config", {}) or {}
    rec = {"tag": tag, "model": model, "seed": seed}

    # (1) the environment, field by field -- the S3M surface with the
    # ICL arm substituted, and NOTHING else moved
    expected = {
        "w_plat": g.S3I_BETA,
        "innate_lambda": g.S3I_GAMMA,
        "deffuant_alpha": g.S3I_ALPHA,
        "ab_sweeps": g.S3I_SWEEPS,
        "n_rounds": rounds,
        "seed": seed,
        "dataset": "movielens",
        "ml_target": "Action",
        "base_model": g.FAM_MODELS[model]["base_model"],
        "ai_gate_mode": "all_open",
        "peer_gate_mode": "all_open",
        "ai_gate_reference": "anchor",
        "population_update": "nested_ai_anchored_then_social_v2",
        "pop_model": "ab",
        "train_cap": N_AGENTS,
        "n_labeled": N_AGENTS,
        "seed_base_data": True,
        "save_raw_gen": True,
        "serve_eval_mode": True,
        # (7) THE DECODING ARM. do_sample is pinned explicitly on both
        # arms: "which decoder ran" is the difference between the
        # main-paper result and the robustness check, and a runner
        # default is not a record.
        "do_sample": g.S3I_DECODE[arm]["do_sample"] == 1,
        # (2) the ARM: frozen personal history
        "training_style": "frozen",
        "kl_beta": 0.0,
        "use_lora": False,
        "fresh_each_round": False,
        "sft_epochs": 0,
        "icl_k": 0,
        "icl_days": ICL_DAYS,
        # (6) strict parsing, wave-wide, no exemption
        "parse_mode": "strict",
    }
    for key, want in expected.items():
        S3M._eq(cfg, key, want, errs)
    dec = g.S3I_DECODE[arm]
    if dec["temperature"] is not None:
        S3M._eq(cfg, "gen_temperature", float(dec["temperature"]), errs)
    if dec["tok"] not in tag:
        errs.append(f"DECODE tag does not carry the '{dec['tok']}' token "
                    f"-- a cell that does not say which decoder ran "
                    f"cannot be audited")
    if not isinstance(cfg.get("git_sha"), str) or not cfg["git_sha"].strip():
        errs.append("CONFIG git_sha absent or empty")
    hg = cfg.get("homophily_gamma", cfg.get("gamma_bias", 0.0))
    if hg not in (0, 0.0):
        errs.append(f"CONFIG homophily gamma={hg!r} (want 0.0)")
    if model == "qwen3_8b" and cfg.get("chat_thinking") is not False:
        errs.append(f"CONFIG Qwen3 chat_thinking={cfg.get('chat_thinking')!r} "
                    f"(want False)")

    check_arrays(d, rounds, errs)
    check_frozen(run_dir, d, cfg, errs)
    check_no_cross_agent_context(run_dir, d, cfg, errs)
    S3M.check_runtime_open_gates(d, rounds, errs)
    gen_stats = None
    if all(not e.startswith("ARTIFACT") for e in errs):
        check_personal_history(run_dir, d, rounds, cfg, errs, rec)
        # (6) the strict raw-generation gate: the log must exist, cover
        # every round, report parse_fail_frac == 0, carry 723 parsed
        # values, and every raw string must BE the served value.
        gen_stats = S3M.check_raw_generations(run_dir, rounds, errs)

    op = torch.as_tensor(d.get("op_raw", torch.empty(0))).float()
    pr = torch.as_tensor(d.get("pred_raw", torch.empty(0))).float()
    # fingerprints for the wave-level DECODING evidence below
    rec["pred_sha"] = _sha_t(pr) if pr.numel() else None
    rec["pred0_sha"] = _sha_t(pr[0]) if pr.ndim == 2 and pr.shape[0] else None
    rec["arm"] = arm
    rec.update({
        "status": "PASS" if not errs else "FAIL",
        "errors": errs, "notes": notes,
        "git_sha": cfg.get("git_sha"),
        "parse_mode": cfg.get("parse_mode", "legacy"),
        "generations": gen_stats,
        "innate_sha": (_sha_t(torch.as_tensor(d["innate"]).float())
                       if "innate" in d else None),
    })
    if op.ndim == 2 and op.shape[0] >= rounds:
        rec["mean"] = float(op[rounds - 1].mean())
        rec["sd"] = float(op[rounds - 1].std())
    del d
    return rec


def main():
    ap = argparse.ArgumentParser(
        description="gate the Section-3 personal-history ICL wave")
    ap.add_argument("--run-root", action="append", default=None)
    ap.add_argument("--arm", default="greedy",
                    choices=("greedy", "sample_t1"),
                    help="greedy = DO_SAMPLE=0 (main paper); sample_t1 = "
                         "DO_SAMPLE=1 at GEN_TEMPERATURE=1.0 (robustness)")
    ap.add_argument("--smoke", action="store_true",
                    help="the 3-round Mistral-7B smoke of the chosen arm")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    roots = [Path(r) for r in (args.run_root or DEFAULT_RUN_ROOTS)]
    g = _load_gen()
    if args.smoke:
        cells = [(g.S3I_SMOKE_MODEL, g.S3I_SMOKE_SEED,
                  g.s3i_tag(g.S3I_SMOKE_MODEL, g.S3I_SMOKE_SEED, args.arm,
                            rounds=g.S3I_SMOKE_ROUNDS, smoke=True),
                  g.S3I_SMOKE_ROUNDS)]
    else:
        cells = [(m, s, g.s3i_cell_tag(m, s, args.arm), g.S3I_ROUNDS)
                 for m in g.S3I_MODELS for s in g.S3I_SEEDS]

    recs = [check_cell(tag, m, s, r, g, roots, args.arm)
            for m, s, tag, r in cells]

    # wave-level: one innate vector, and (production) distinct seeds
    inn = {r["innate_sha"] for r in recs if r.get("innate_sha")}
    wave_errs = []
    if len(inn) > 1:
        wave_errs.append(f"WAVE {len(inn)} distinct innate vectors -- "
                         f"movielens innate is a pure function of "
                         f"(dataset, target), so the cells are not one "
                         f"population")
    # (7b) THE SEED CONTROLS SAMPLING -- as evidence, not assertion.
    # run_pokec_gated_lm seeds the global torch stream once and the
    # Deffuant sweep draws from its OWN generator, so on this frozen arm
    # generation is the only global-RNG consumer.
    #   sampled: the three seeds must produce DIFFERENT pred_raw, or the
    #            seed never reached the sampler and the three "replicates"
    #            are one observation.
    #   greedy : decoding is deterministic and every agent's round-0
    #            prompt is innate-only, so the three seeds must AGREE at
    #            round 0. Disagreement there means something other than
    #            the decoder moved.
    if not args.smoke:
        by_cell = {}
        for r in recs:
            if r.get("pred_sha"):
                by_cell.setdefault(r["model"], []).append(r)
        for model, rs in sorted(by_cell.items()):
            if len(rs) < 2:
                continue
            if args.arm == "sample_t1":
                shas_p = {r["pred_sha"] for r in rs}
                if len(shas_p) < len(rs):
                    wave_errs.append(
                        f"DECODE {model}: {len(rs)} sampled seeds share a "
                        f"pred_raw fingerprint -- the seed did not reach "
                        f"the sampler, so these are not replicates")
            else:
                z = {r["pred0_sha"] for r in rs if r.get("pred0_sha")}
                if len(z) > 1:
                    wave_errs.append(
                        f"DECODE {model}: greedy seeds disagree at ROUND 0 "
                        f"({len(z)} distinct pred_raw[0]) -- with frozen "
                        f"weights and identical innate-only prompts, "
                        f"greedy decoding must be seed-invariant there")
    shas = sorted({r["git_sha"] for r in recs if r.get("git_sha")})
    if len(shas) > 1:
        wave_errs.append(f"WAVE {len(shas)} distinct git_sha across the "
                         f"cells ({[s[:10] for s in shas]}) -- this wave "
                         f"claims one code provenance and has NO exemption")

    hdr = (f"{'cell':<62} {'verdict':>7} {'mean':>8} {'sd':>7} "
           f"{'parse':>7} {'history':>11}")
    print("=" * len(hdr))
    n_pass = sum(1 for r in recs if r["status"] == "PASS")
    print(f"SECTION-3 PERSONAL-HISTORY ICL [{args.arm}] -- "
          f"{'SMOKE' if args.smoke else 'PRODUCTION'} grid, "
          f"{n_pass}/{len(recs)} cells PASS")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        print(f"{r['tag']:<62} {r['status']:>7} "
              f"{r.get('mean', float('nan')):>8.4f} "
              f"{r.get('sd', float('nan')):>7.4f} "
              f"{str(r.get('parse_mode')):>7} "
              f"{str(r.get('history_replay', '-')):>11}")
    for r in recs:
        for e in r["errors"]:
            print(f"{LOG} FAIL {r['tag']}: {e}")
    for e in wave_errs:
        print(f"{LOG} FAIL {e}")

    ok = n_pass == len(recs) and not wave_errs
    verdict = {
        "wave": g.s3i_arm_key(args.arm, args.smoke), "arm": args.arm,
        "decoding": dict(g.S3I_DECODE[args.arm]),
        "smoke": bool(args.smoke),
        "n_cells": len(recs), "n_pass": n_pass,
        # "ok" is the key analyze_section3_model_equilibria.gate_binds_wave
        # reads; "pass" mirrors the Section-4 gates' spelling
        "ok": bool(ok), "pass": bool(ok),
        "parse_policy": ("PARSE_MODE=strict wave-wide; NO per-model "
                         "exemption and no legacy-parser path"),
        "git_sha": shas, "innate_sha": sorted(inn),
        "wave_errors": wave_errs, "cells": recs,
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(verdict, indent=2))
        print(f"{LOG} verdict -> {args.json}")
    if ok:
        print(f"{LOG} PASS -- {len(recs)} cell(s): environment exact; "
              f"weights frozen with no adapter or optimizer witness; "
              f"personal histories complete and byte-exact; no cross-agent "
              f"context; trajectories complete with both gates open every "
              f"round; zero parse failures and zero malformed generations "
              f"under PARSE_MODE=strict on every model, Mistral included; "
              f"decoding is {args.arm} and the seed demonstrably controls "
              f"it.")
        return 0
    print(f"{LOG} FAILED -- {len(recs) - n_pass} of {len(recs)} cell(s) "
          f"failed, {len(wave_errs)} wave-level violation(s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
