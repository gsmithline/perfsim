#!/usr/bin/env python3
"""section3_frontier_icl: the personal-history ICL analogue of Figure 3(a),
served by a FROZEN frontier API model instead of a local checkpoint.

WHY A SEPARATE GENERATOR.  gen_pofd_sweep.py builds every GPU wave in the
project from one shared set of templates and cross-wave collision
assertions. This wave shares none of that: no GPU request, no adapter, no
SFT, a different secret-handling story, and a hard cost cap. Weaving it in
would regenerate unrelated .sub files for waves that are running. It
therefore reuses the SURFACE CONSTANTS from that file (imported, not
copied, so they cannot drift) and emits its own configs and sub.

THE SURFACE IS THE S3I ONE, EXACTLY.  MovieLens/Action, 723 agents, 30
rounds, 100 Deffuant sweeps, W_PLAT=1, INNATE_LAMBDA=1, alpha=.5, both
gates all_open, the corrected anch2 operator, ICL_K=0, ICL_DAYS=8, seeds
0/42/43. The ONLY difference from section3_model_icl is which model
serves. That is the point: it makes the frontier models drop into the
existing Figure-3(a) comparison.

TAGS ARE DISJOINT BY CONSTRUCTION.  pofds3f_ (frontier) cannot collide
with pofds3i_ (local ICL) or pofds3m_ (local SFT), and the model token is
the SLUG with separators normalised, so two providers of the same family
are different cells.

  python gen_frontier_icl.py --models google/gemini-3.7-flash@Google-AI-Studio
  python gen_frontier_icl.py --models ... --seeds 0,42,43 --estimate
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

# ---- the surface, imported from the wave this one mirrors --------------
from gen_pofd_sweep import (                               # noqa: E402
    S3I_SWEEPS, S3I_ROUNDS, S3I_SEEDS, S3I_ICL_DAYS, S3I_BETA, S3I_GAMMA,
    S3I_ALPHA, S3_EPS_SOCIAL, S3_OP_TOKEN,
)

PREFIX = "pofds3f"
SMOKE_PREFIX = "pofds3fsmk"
N_AGENTS = 723
SMOKE_ROUNDS = 3
ARM = "d8"

# Requests are exactly one per agent per round: the serving loop asks each
# agent once. Asserted below rather than trusted.
REQ_PER_MODEL_SEED = N_AGENTS * S3I_ROUNDS          # 21,690
REQ_PER_MODEL_3SEED = REQ_PER_MODEL_SEED * 3        # 65,070
REQ_PER_SMOKE = N_AGENTS * SMOKE_ROUNDS             # 2,169

# Measured on the S3I prompts: the MovieLens profile block plus an
# eight-value history renders to ~250 input tokens; a bare number costs a
# handful of output tokens. Both are REPLACED by live-canary numbers
# before any production estimate is quoted -- see --canary-usage.
EST_IN_TOK = 250
EST_OUT_TOK = 8


def model_token(slug: str, provider: str) -> str:
    """A filesystem- and tag-safe token that still identifies the exact
    model AND provider, because the same slug served by two providers is
    two different measurements."""
    s = re.sub(r"[^a-z0-9]+", "", slug.split("/")[-1].lower())
    p = re.sub(r"[^a-z0-9]+", "", provider.lower())
    return f"{s}-{p}"


def tag(slug: str, provider: str, seed: int, rounds: int = S3I_ROUNDS,
        smoke: bool = False) -> str:
    pre = SMOKE_PREFIX if smoke else PREFIX
    return (f"{pre}_{model_token(slug, provider)}_{ARM}_greedy"
            f"_sw{S3I_SWEEPS}_eaopen_w1_k1_esopen_{S3_OP_TOKEN}"
            f"_s{seed}_r{rounds}")


def estimate_cost(n_requests: int, price_in: float, price_out: float,
                  in_tok: int = EST_IN_TOK, out_tok: int = EST_OUT_TOK) -> dict:
    """USD. price_in/price_out are per MILLION tokens."""
    cin = n_requests * in_tok / 1e6 * price_in
    cout = n_requests * out_tok / 1e6 * price_out
    return {"requests": n_requests, "in_tokens": n_requests * in_tok,
            "out_tokens": n_requests * out_tok,
            "cost_in_usd": round(cin, 2), "cost_out_usd": round(cout, 2),
            "cost_total_usd": round(cin + cout, 2)}


# NO GPU REQUEST. This wave is a network client: the work happens in
# someone else's datacentre. Asking for an H100 would idle one for hours.
SUB = """\
# section3_frontier_icl -- {kind}
# CPU ONLY: the model is served over the network, so this wave asks for
# no accelerator at all -- requesting one would hold a GPU idle for the
# whole wave while the work happens in someone else's datacentre.
#
# THE KEY IS NEVER IN THIS FILE, in `arguments`, or in `environment`.
# The job reads OPENROUTER_API_KEY_FILE, a mode-0600 path on shared home.
# A key in a submit file is a key in the schedd's job classad, which is
# world-readable to the pool.
universe           = vanilla
executable         = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
should_transfer_files = NO
getenv             = False

request_cpus       = 2
request_memory     = 8G
request_disk       = 8G

environment        = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn DATASET=movielens ML_TARGET=Action MODEL_BACKEND=openrouter OPENROUTER_API_KEY_FILE=/home/gsmithline/.openrouter_key OR_MODEL=$(orModel) OR_PROVIDER=$(orProvider) OR_MAX_TOKENS={or_max_tokens} OR_TEMPERATURE=0 OR_TOP_P=1 OR_REASONING_MODE={reasoning_mode} OR_CONCURRENCY={concurrency} OR_RPS={rps} OR_MAX_REQUESTS=$(orMaxRequests) OR_MAX_COST=$(orMaxCost) OR_CACHE=/home/gsmithline/perfsim/runs/pokec_gated_lm/$(tag)/or_cache.sqlite TRAINING_STYLE=frozen SFT_EPOCHS=0 USE_LORA=0 KL_BETA=0 FRESH_EACH_ROUND=0 LOG_PERPLEXITY=0 LOG_ANSWER_DIST=0 ANS_SAMPLE_K=0 PARSE_MODE=strict SAVE_RAW_GEN=1 LOG_GENDER_GAPS=1 EPS_AI=1.0 AI_GATE_MODE=all_open PEER_GATE_MODE=all_open EPS={eps_social} GAMMA_BIAS=0.0 ICL_K=0 ICL_DAYS={icl_days} ICL_SELECT=random ICL_CTX_SOURCE=live W_PLAT={wplat} INNATE_LAMBDA={lam} DEFFUANT_ALPHA={alpha} AB_SWEEPS={sweeps} AI_GATE_REFERENCE=anchor WITH_TWIN=1 TRAIN_CAP={n} N_LABELED={n} N_ROUNDS=$(nrounds) SEED=$(seed) SEED_BASE_DATA=1"

arguments          = "$(tag) frozen"
output             = logs/$(tag).out
error              = logs/$(tag).err
log                = logs/$(tag).log

on_exit_hold       = (ExitCode =!= 0)
periodic_release   = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove    = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag,seed,nrounds,orModel,orProvider,orMaxRequests,orMaxCost from configs_pofd_{key}.txt
"""


def build(models, seeds, rounds, smoke, max_cost_per_cell):
    rows, meta = [], []
    for slug, provider in models:
        for seed in seeds:
            t = tag(slug, provider, seed, rounds, smoke)
            n_req = N_AGENTS * rounds
            # +2% head-room so a single retry-after-cache-miss cannot trip
            # the cap mid-wave; the COST cap is the real control.
            rows.append(f"{t},{seed},{rounds},{slug},{provider},"
                        f"{int(n_req * 1.02)},{max_cost_per_cell}")
            meta.append({"tag": t, "model": slug, "provider": provider,
                         "seed": seed, "rounds": rounds, "requests": n_req})
    return rows, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="comma-separated slug@Provider (provider spaces as -)")
    ap.add_argument("--seeds", default=",".join(str(s) for s in S3I_SEEDS))
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE_ROUNDS}-round scientific smoke")
    ap.add_argument("--max-cost-per-cell", type=float, default=25.0)
    ap.add_argument("--price", default="",
                    help="slug@Provider=in/out per Mtok, for the estimate")
    ap.add_argument("--canary-usage", default="",
                    help="in,out mean tokens MEASURED by the live canary; "
                         "replaces the catalog-based guess")
    ap.add_argument("--write", action="store_true",
                    help="write configs + sub (default: estimate only)")
    args = ap.parse_args()

    models = []
    for m in args.models.split(","):
        if "@" not in m:
            raise SystemExit(f"--models entry {m!r} must be slug@Provider")
        slug, prov = m.split("@", 1)
        if "auto" in slug or "latest" in slug:
            raise SystemExit(
                f"{slug!r} is a moving alias; a paper wave must pin an "
                f"exact dated slug.")
        models.append((slug.strip(), prov.replace("-", " ").strip()))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rounds = SMOKE_ROUNDS if args.smoke else S3I_ROUNDS

    in_tok, out_tok = EST_IN_TOK, EST_OUT_TOK
    src = "catalog pricing + estimated token counts"
    if args.canary_usage:
        in_tok, out_tok = (int(x) for x in args.canary_usage.split(","))
        src = "catalog pricing + LIVE-CANARY measured token counts"

    prices = {}
    for entry in filter(None, args.price.split(",")):
        k, v = entry.split("=")
        pin, pout = v.split("/")
        # only the PROVIDER half carries dashes-for-spaces; the slug's own
        # hyphens (gemini-3.7-flash) must survive verbatim
        kslug, kprov = k.split("@", 1)
        prices[f"{kslug}@{kprov.replace('-', ' ')}"] = (float(pin), float(pout))

    # THE COUNTS THE USER SPECIFIED, ASSERTED rather than believed.
    assert REQ_PER_MODEL_SEED == 21_690, REQ_PER_MODEL_SEED
    assert REQ_PER_MODEL_3SEED == 65_070, REQ_PER_MODEL_3SEED

    rows, meta = build(models, seeds, rounds, args.smoke,
                       args.max_cost_per_cell)
    total_req = sum(m["requests"] for m in meta)
    print(f"[frontier] {len(rows)} cell(s), {rounds} rounds, "
          f"{N_AGENTS} agents -> {total_req:,} requests")
    print(f"[frontier] estimate source: {src} "
          f"({in_tok} in / {out_tok} out tokens per request)\n")

    hdr = (f"{'model@provider':<44}{'seeds':>6}{'requests':>10}"
           f"{'$in':>9}{'$out':>9}{'$total':>9}")
    print(hdr); print("-" * len(hdr))
    grand = 0.0
    for slug, prov in models:
        key = f"{slug}@{prov}"
        n = N_AGENTS * rounds * len(seeds)
        if key in prices:
            e = estimate_cost(n, *prices[key], in_tok, out_tok)
            grand += e["cost_total_usd"]
            print(f"{key:<44}{len(seeds):>6}{n:>10,}"
                  f"{e['cost_in_usd']:>9.2f}{e['cost_out_usd']:>9.2f}"
                  f"{e['cost_total_usd']:>9.2f}")
        else:
            print(f"{key:<44}{len(seeds):>6}{n:>10,}"
                  f"{'?':>9}{'?':>9}{'?':>9}  (no --price given)")
    if grand:
        print(f"{'TOTAL':<44}{'':>6}{total_req:>10,}{'':>9}{'':>9}"
              f"{grand:>9.2f}")
    print("\n  Estimates are NOT a cap. OR_MAX_COST is enforced per cell "
          "in-process\n  and raises rather than truncating the run.")

    if not args.write:
        print("\n[frontier] estimate only. Pass --write to emit configs+sub; "
              "submission is still a separate, explicit step.")
        return 0

    key = "section3_frontier_icl" + ("_smoke" if args.smoke else "")
    cfg = HERE / f"configs_pofd_{key}.txt"
    cfg.write_text("\n".join(rows) + "\n")
    sub = SUB.format(
        kind=("3-ROUND SCIENTIFIC SMOKE" if args.smoke
              else "PRODUCTION: open-gate cross-model equilibria, "
                   "personal-history ICL, frozen frontier API model"),
        key=key, or_max_tokens=16, concurrency=8, rps=4,
        reasoning_mode="disabled",
        eps_social=f"{S3_EPS_SOCIAL:g}", icl_days=S3I_ICL_DAYS,
        wplat=f"{S3I_BETA:g}", lam=f"{S3I_GAMMA:g}",
        alpha=f"{S3I_ALPHA:g}", sweeps=S3I_SWEEPS, op=S3_OP_TOKEN, n=N_AGENTS)
    # AUDIT BEFORE WRITE. A sub that fails either check must never reach
    # disk, or a later "the file looks fine" glance would be checking the
    # artifact of a failed build.
    assert "request_gpus" not in sub, "this wave must not request a GPU"
    assert "OPENROUTER_API_KEY=" not in sub, "the key must never be in the sub"
    assert "sk-or-" not in sub, "a literal key must never be in the sub"
    (HERE / f"at_pofd_{key}.sub").write_text(sub)
    (HERE / f"manifest_{key}.json").write_text(json.dumps(
        {"key": key, "rounds": rounds, "agents": N_AGENTS,
         "requests_total": total_req, "estimate_source": src,
         "cells": meta}, indent=1))
    print(f"\n[frontier] wrote configs_pofd_{key}.txt, at_pofd_{key}.sub, "
          f"manifest_{key}.json")
    print("[frontier] NOT SUBMITTED. Review, then submit deliberately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
