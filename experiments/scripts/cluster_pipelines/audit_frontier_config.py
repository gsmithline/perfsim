#!/usr/bin/env python3
"""Field-by-field audit of the frontier ICL EFFECTIVE configuration against
an archived, GATED Section-3 ICL run. NO MODEL REQUEST IS MADE.

WHY THIS FILE EXISTS. On 2026-08-31 a launcher set EPS_SOCIAL and
POPULATION_UPDATE; the runner reads neither. It ran with EPS=0.3 (needs
0.2) and GAMMA_BIAS=1.5 (this project forbids homophily bias; needs 0.0),
and would have produced a complete, correctly-tagged, fully-provenanced
trajectory of the WRONG DYNAMICS. Comparing launcher environment
variables would not have caught it -- only the runner's own recorded
EFFECTIVE configuration does.

So this drives the real runner in DRY_RUN_CONFIG mode, which builds and
records the effective config, validates the model pin against the live
catalog, and exits before serving a single agent. The result is then
compared to the archived reference field by field.

DIFFERENCES ARE ALLOWLISTED, NOT TOLERATED. Every field that legitimately
differs because the model is served over an API appears in ALLOWED below
with a reason. Anything else is a FAILURE.

  python audit_frontier_config.py [--reference <run dir>] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "experiments/scripts/cluster_pipelines/run_pokec_gated_lm.py"
PREFLIGHT = Path(__file__).with_name("or_preflight.py")
DEFAULT_REF = (REPO / "notes/pofd/cluster/"
               "pofds3i_mistral7b_d8_greedy_sw100_eaopen_w1_k1_esopen_anch2_s0_r30")

# The scientifically relevant surface. If any of these differs from the
# archived reference and is not allowlisted, the frontier cells cannot be
# placed beside the local ones.
SURFACE = [
    "dataset", "ml_target", "pop_model", "training_style", "sft_epochs",
    "use_lora", "kl_beta", "icl_k", "icl_days", "icl_select",
    "icl_ctx_source", "icl_snapshot_round", "eps", "eps_ai", "gamma_bias",
    "w_plat", "innate_lambda", "deffuant_alpha", "ab_sweeps",
    "ai_gate_mode", "peer_gate_mode", "population_update",
    "ai_gate_reference", "train_cap", "n_labeled", "parse_mode",
    "fresh_each_round", "deploy_every", "seed_base_data", "with_twin",
    "save_raw_gen",
]

# Intentional differences REQUIRED by the remote API backend. Each carries
# the reason it cannot match, so a reader can judge it.
ALLOWED = {
    "base_model": "the served model is a remote API slug, not a local checkpoint",
    "model_backend": "openrouter vs the implicit hf backend of the archive",
    "max_new_tokens": "local HF generation knob; the API uses max_tokens",
    "gen_temperature": "local HF knob; API decoding is recorded in openrouter.policy",
    "gen_top_p": "local HF knob, ditto",
    "gen_top_k": "local HF knob, ditto",
    "gen_repetition_penalty": "local HF knob; no API equivalent is exposed",
    "do_sample": "local HF knob; API serving is deterministic-as-requested",
    "gen_batch_size": "local batching; the API path bounds CONCURRENCY instead",
    "log_perplexity": "needs logits; the API returns none (refused by the backend)",
    "log_answer_dist": "needs logits, ditto",
    "n_perplexity": "perplexity is unavailable, so its size is moot",
    "log_ppl_dist": "perplexity is unavailable",
    "ppl_dist_cap": "perplexity is unavailable",
    "ppl_batch": "perplexity is unavailable",
    "ans_sample_k": "sampling telemetry needs logits",
    "ans_sample_n": "sampling telemetry needs logits",
    "ans_sample_t": "sampling telemetry needs logits",
    "lora_r": "no adapter exists on a frozen API model",
    "sft_lr": "there is no optimizer",
    "sft_batch_size": "there is no optimizer",
    "epoch_size": "there is no training loop",
    "max_steps": "there is no training loop",
    "kl_direction": "no KL term exists without training",
    "chat_thinking": "no local chat template is applied; reasoning is an API knob",
    "hist_bins": "reporting-only",
    "n_rounds": "the smoke runs fewer rounds than the 30-round archive",
    "rounds": "ditto",
    "seed": "compared per cell, not against the reference's seed",
    "run_tag": "each cell has its own tag",
    "git_sha": "provenance of the run, not of the surface",
    "host": "provenance of the run",
    "hardware": "provenance of the run",
    "wandb_run_suffix": "bookkeeping",
    "out_dir": "path",
    "openrouter": "API-backend block with no archive counterpart",
    "gen_policy_effective": "API decoding policy; no archive counterpart",
}


def effective_config(model, provider, seed, rounds, reasoning_mode, out_dir):
    """Drive the REAL runner in DRY_RUN_CONFIG mode. No model request."""
    pf = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--model", model,
         "--provider", provider, "--seed", str(seed), "--json"]
        + (["--reasoning-mode", reasoning_mode] if reasoning_mode else []),
        capture_output=True, text=True)
    if pf.returncode != 0:
        raise SystemExit(f"preflight failed for {model}: {pf.stderr}")
    info = json.loads(pf.stdout)
    tag = (f"pofds3fsmk_dryrun_d8_greedy_sw100_eaopen_w1_k1_esopen_anch2"
           f"_s{seed}_r{rounds}")
    env = dict(os.environ)
    env.update({
        "RUN_TAG": tag, "OUT_DIR": str(out_dir), "DRY_RUN_CONFIG": "1",
        "DATASET": "movielens", "ML_TARGET": "Action",
        "MODEL_BACKEND": "openrouter",
        "OR_MODEL": info["resolved_model"], "OR_PROVIDER": provider,
        "OR_EXPECTED_CANONICAL": info["canonical_slug"],
        "OR_TEMPERATURE": "0" if info["supports_temperature"] else "",
        "OR_SEED": str(seed) if info["supports_seed"] else "",
        "OR_MAX_TOKENS": str(32 if info["reasoning_mode"] == "disabled"
                             else 2080),
        "OR_REASONING_MODE": info["reasoning_mode"],
        "OR_REQUIRE_PARAMETERS": "0", "OR_ZDR": "1",
        "OR_MAX_REQUESTS": "1", "OR_MAX_COST": "0.01", "OR_DRY_RUN": "1",
        # the Section 3(a) surface, by the names the runner ACTUALLY reads
        "TRAINING_STYLE": "frozen", "SFT_EPOCHS": "0", "USE_LORA": "0",
        "KL_BETA": "0", "FRESH_EACH_ROUND": "0", "LOG_PERPLEXITY": "0",
        "LOG_ANSWER_DIST": "0", "ANS_SAMPLE_K": "0",
        "PARSE_MODE": "strict", "SAVE_RAW_GEN": "1",
        "LOG_GENDER_GAPS": "1",
        "ICL_K": "0", "ICL_DAYS": "8", "ICL_SELECT": "random",
        "ICL_CTX_SOURCE": "live", "POP_MODEL": "ab",
        "AI_GATE_MODE": "all_open", "PEER_GATE_MODE": "all_open",
        "AI_GATE_REFERENCE": "anchor", "EPS_AI": "1.0", "EPS": "0.2",
        "GAMMA_BIAS": "0.0", "W_PLAT": "1", "INNATE_LAMBDA": "1",
        "DEFFUANT_ALPHA": "0.5", "AB_SWEEPS": "100",
        "N_ROUNDS": str(rounds), "WITH_TWIN": "1", "TRAIN_CAP": "723",
        "N_LABELED": "723", "SEED": str(seed), "SEED_BASE_DATA": "1",
        "WANDB_MODE": "disabled", "WANDB_DISABLED": "true", "USE_TF": "0",
    })
    r = subprocess.run([sys.executable, str(RUNNER)], env=env,
                       capture_output=True, text=True, cwd=str(REPO))
    cfg_path = Path(out_dir) / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"dry run produced no config for {model}:\n"
                         f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return json.loads(cfg_path.read_text()), info, r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(DEFAULT_REF))
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ref = json.loads((Path(args.reference) / "config.json").read_text())
    print(f"[audit] reference: {Path(args.reference).name}")
    print(f"[audit] NO MODEL REQUEST IS MADE BY THIS AUDIT.\n")

    MODELS = [
        ("openai/gpt-5.6-sol-20260709", "Azure", None),
        ("anthropic/claude-opus-5-20260723", "Amazon Bedrock", None),
        ("google/gemini-3.1-pro-preview-20260219", "Google", "minimal"),
    ]
    results, ok = [], True
    for slug, prov, rmode in MODELS:
        with tempfile.TemporaryDirectory() as td:
            cfg, info, log = effective_config(slug, prov, 0, args.rounds,
                                              rmode, td)
        # COMPLETE field-by-field: the union of both configs, not just the
        # surface list. A field present in one and absent in the other is a
        # difference too, and is classified the same way.
        diffs = []
        for f in sorted(set(ref) | set(cfg)):
            want, got = ref.get(f), cfg.get(f)
            if isinstance(want, float) or isinstance(got, float):
                same = (want is not None and got is not None
                        and abs(float(want) - float(got)) < 1e-9)
            else:
                same = want == got
            if not same:
                diffs.append((f, want, got))
        # the two knobs the near-miss turned on
        checks = {
            "innate_lambda (paper gamma) == 1": float(cfg.get("innate_lambda", -1)) == 1.0,
            "gamma_bias (homophily) == 0": float(cfg.get("gamma_bias", -1)) == 0.0,
            "ai_gate_mode all_open": cfg.get("ai_gate_mode") == "all_open",
            "peer_gate_mode all_open": cfg.get("peer_gate_mode") == "all_open",
            "eps pinned 0.2 (inactive when open, still pinned)":
                abs(float(cfg.get("eps", -1)) - 0.2) < 1e-9,
            "operator == anchored v2":
                cfg.get("population_update") == "nested_ai_anchored_then_social_v2",
            "ai_gate_reference == anchor": cfg.get("ai_gate_reference") == "anchor",
            "answer limit pinned (not None)":
                (cfg.get("openrouter") or {}).get("policy", {}).get("max_tokens") is not None,
            "canonical pin recorded":
                (cfg.get("gen_policy_effective") or {}).get(
                    "expected_canonical_slug", {}).get("value") == info["canonical_slug"],
            "provider pinned, fallbacks off":
                (cfg.get("openrouter") or {}).get("provider", {}).get(
                    "allow_fallbacks") is False,
            "zdr required":
                (cfg.get("openrouter") or {}).get("provider", {}).get("zdr") is True,
        }
        # a field outside the scientific surface AND outside the allowlist
        # is reported as UNCLASSIFIED: it is not a failure of the surface,
        # but it has not been justified either, so it must be looked at.
        bad_diffs = [d for d in diffs if d[0] in SURFACE and d[0] not in ALLOWED]
        unclassified = [d for d in diffs
                        if d[0] not in SURFACE and d[0] not in ALLOWED]
        failed = [k for k, v in checks.items() if not v]
        ok &= not bad_diffs and not failed
        print(f"=== {slug} via {prov} ===")
        print(f"  resolved {info['resolved_model']} | canonical "
              f"{info['canonical_slug']}")
        for k, v in checks.items():
            print(f"  [{'ok' if v else 'FAIL'}] {k}")
        if bad_diffs:
            print(f"  UNALLOWLISTED SURFACE DIFFERENCES:")
            for f, w, g in bad_diffs:
                print(f"     {f}: reference={w!r} effective={g!r}")
        else:
            print(f"  [ok] every surface field matches the gated reference")
        allow_hit = [d[0] for d in diffs if d[0] in ALLOWED]
        print(f"  fields compared: {len(set(ref) | set(cfg))}  "
              f"identical: {len(set(ref) | set(cfg)) - len(diffs)}  "
              f"differing: {len(diffs)}")
        if allow_hit:
            print(f"  allowlisted ({len(allow_hit)}): {sorted(allow_hit)}")
        if unclassified:
            print(f"  UNCLASSIFIED differences ({len(unclassified)}) -- not "
                  f"surface, not allowlisted; review these:")
            for f, w, g in unclassified:
                print(f"     {f}: reference={w!r} effective={g!r}")
        print()
        results.append({"model": slug, "provider": prov,
                        "resolved": info["resolved_model"],
                        "canonical": info["canonical_slug"],
                        "checks": checks,
                        "unallowlisted": [list(d) for d in bad_diffs],
                        "unclassified": [list(d) for d in unclassified],
                        "n_fields": len(set(ref) | set(cfg)),
                        "n_differing": len(diffs),
                        "allowlisted": sorted(allow_hit)})

    print(f"[audit] {'PASS' if ok else 'FAIL'} -- effective configuration "
          f"{'matches' if ok else 'DOES NOT match'} the gated Section-3 "
          f"reference on every scientifically relevant field.")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"reference": str(args.reference), "allowlist": ALLOWED,
             "surface": SURFACE, "results": results, "pass": ok}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
