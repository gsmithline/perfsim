#!/usr/bin/env python3
"""Prove that a production cell's seeded cache reproduces its gated smoke
EXACTLY, using no network at all.

WHY. Seeding a 20-round cell from a 3-round smoke cache is only safe if
the first three rounds are bit-identical: same prompts, same served
values, same history blocks, same provenance. Copying a file and assuming
that is how a wave silently becomes two experiments. This runs the real
runner in CACHE-ONLY mode against the PRODUCTION cache, for 3 rounds,
into a scratch directory, then diffs against the accepted smoke.

Any cache miss is a hard failure (CacheMissError), so the replay cannot
quietly buy a missing response.

  python verify_cache_reuse.py            # all six cells
"""
from __future__ import annotations
import json, os, subprocess, sys, tempfile, gzip
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs" / "pokec_gated_lm"
RUNNER = REPO / "experiments/scripts/cluster_pipelines/run_pokec_gated_lm.py"
PRE = Path(__file__).with_name("or_preflight.py")

CELLS = [("openai/gpt-5.6-sol-20260709", "Azure", "gpt56sol-azure"),
         ("anthropic/claude-opus-5-20260723", "Amazon Bedrock",
          "claudeopus5-amazonbedrock"),
         ("moonshotai/kimi-k3-20260715", "Morph", "kimik3-morph")]
SUF = "_greedy_sw100_eaopen_w1_k1_esopen_anch2_s0"


def env_for(slug, prov, depth, out_dir, cache, rounds):
    pf = subprocess.run([sys.executable, str(PRE), "--model", slug,
                         "--provider", prov, "--seed", "0", "--json"],
                        capture_output=True, text=True)
    if pf.returncode != 0:
        raise SystemExit(f"preflight failed: {pf.stderr}")
    i = json.loads(pf.stdout)
    e = dict(os.environ)
    e.update({
        "RUN_TAG": "verify", "OUT_DIR": str(out_dir),
        "OR_CACHE": str(cache), "OR_CACHE_ONLY": "1",
        "DATASET": "movielens", "ML_TARGET": "Action",
        "MODEL_BACKEND": "openrouter", "OR_MODEL": i["resolved_model"],
        "OR_PROVIDER": prov, "OR_EXPECTED_CANONICAL": i["canonical_slug"],
        "OR_TEMPERATURE": "0" if i["supports_temperature"] else "",
        "OR_SEED": "0" if i["supports_seed"] else "",
        "OR_MAX_TOKENS": "32", "OR_REASONING_MODE": "disabled",
        "OR_TOP_P": "1", "OR_REQUIRE_PARAMETERS": "0", "OR_ZDR": "1",
        "OR_MAX_REQUESTS": "1", "OR_MAX_COST": "0.001",
        "TRAINING_STYLE": "frozen", "SFT_EPOCHS": "0", "USE_LORA": "0",
        "KL_BETA": "0", "FRESH_EACH_ROUND": "0", "LOG_PERPLEXITY": "0",
        "LOG_ANSWER_DIST": "0", "ANS_SAMPLE_K": "0", "PARSE_MODE": "strict",
        "SAVE_RAW_GEN": "1", "LOG_GENDER_GAPS": "1",
        "ICL_K": "0", "ICL_DAYS": str(depth), "ICL_SELECT": "random",
        "ICL_CTX_SOURCE": "live", "POP_MODEL": "ab",
        "AI_GATE_MODE": "all_open", "PEER_GATE_MODE": "all_open",
        "AI_GATE_REFERENCE": "anchor", "EPS_AI": "1.0", "EPS": "0.2",
        "GAMMA_BIAS": "0.0", "W_PLAT": "1", "INNATE_LAMBDA": "1",
        "DEFFUANT_ALPHA": "0.5", "AB_SWEEPS": "100",
        "N_ROUNDS": str(rounds), "WITH_TWIN": "1", "TRAIN_CAP": "723",
        "N_LABELED": "723", "SEED": "0", "SEED_BASE_DATA": "1",
        "RESOURCE_GUARD": "0",
        "WANDB_MODE": "disabled", "WANDB_DISABLED": "true", "USE_TF": "0",
    })
    return e


def main() -> int:
    ok = True
    for slug, prov, tok in CELLS:
        for depth in (0, 8):
            smoke = RUNS / f"pofds3fsmk_{tok}_d{depth}{SUF}_r3"
            prod = RUNS / f"pofds3f_{tok}_d{depth}{SUF}_r20"
            label = f"{tok} D={depth}"
            if not (smoke / "trajectory.pt").is_file():
                print(f"[verify] {label}: FAIL, no gated smoke"); ok = False
                continue
            with tempfile.TemporaryDirectory() as td:
                r = subprocess.run(
                    [sys.executable, str(RUNNER)], cwd=str(REPO),
                    env=env_for(slug, prov, depth, td,
                                prod / "or_cache.sqlite", 3),
                    capture_output=True, text=True)
                tj = Path(td) / "trajectory.pt"
                if not tj.is_file():
                    tail = (r.stdout + r.stderr)[-400:]
                    print(f"[verify] {label}: FAIL, replay produced nothing\n"
                          f"          {tail}")
                    ok = False
                    continue
                a = torch.load(tj, map_location="cpu", weights_only=False)
                b = torch.load(smoke / "trajectory.pt", map_location="cpu",
                               weights_only=False)
                checks = {}
                for k in ("op_raw", "pred_raw", "twin_raw"):
                    if k in a and k in b:
                        checks[k] = bool(torch.equal(a[k][:3].float(),
                                                     b[k][:3].float()))
                if depth > 0:
                    ha = [json.loads(l) for l in
                          gzip.open(Path(td) / "icl_days_log.json.gz", "rt")
                          if l.strip()]
                    hb = [json.loads(l) for l in
                          gzip.open(smoke / "icl_days_log.json.gz", "rt")
                          if l.strip()]
                    checks["history_blocks"] = (
                        [x["ctx"] for x in ha[:3]] == [x["ctx"] for x in hb[:3]])
                else:
                    checks["no_history_log"] = not (
                        Path(td) / "icl_days_log.json.gz").exists()
                pa = [json.loads(l) for l in
                      gzip.open(Path(td) / "or_provenance.json.gz", "rt")
                      if l.strip()]
                checks["provenance_text"] = all(
                    [x["text"] for x in pa[i]["records"]] ==
                    [x["text"] for x in _smoke_prov(smoke)[i]["records"]]
                    for i in range(3))
                bad = [k for k, v in checks.items() if not v]
                ok &= not bad
                print(f"[verify] {label}: {'PASS' if not bad else 'FAIL ' + str(bad)}"
                      f"  ({', '.join(k for k in checks)})")
    print(f"\n[verify] {'ALL CELLS BYTE-IDENTICAL' if ok else 'MISMATCH -- do not launch'}")
    return 0 if ok else 1


def _smoke_prov(smoke):
    return [json.loads(l) for l in
            gzip.open(smoke / "or_provenance.json.gz", "rt") if l.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
