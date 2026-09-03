#!/usr/bin/env python3
"""OPT-IN live connectivity canary. AT MOST THREE PAID REQUESTS.

This is the only script in the repo that spends money by design, and it
is never run automatically: no test imports it, no wave depends on it,
and it refuses to run without --i-understand-this-costs-money.

WHAT IT IS FOR. Three things cannot be learned from the catalog:
  - whether the KEY works from THIS host (and, on the cluster, whether the
    worker node has egress at all);
  - whether the pinned provider actually honours temperature, max_tokens
    and reasoning-off under require_parameters=true -- the catalog says
    what an endpoint ADVERTISES, not what it does;
  - the REAL token usage of our prompt, which is what turns a catalog
    price into a credible wave estimate.

It sends the genuine MovieLens ICL prompt, not a toy, so the measured
token counts are the ones a production wave would pay.

  python or_canary.py --model google/gemini-3.7-flash \
      --provider "Google AI Studio" --i-understand-this-costs-money
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from perfsim.models.openrouter_client import (            # noqa: E402
    Budget, DecodingPolicy, OpenRouterClient, ProviderPin, assert_canonical,
    load_api_key, validate_key,
)

MAX_REQUESTS = 3

# The real thing: a MovieLens/Action profile with an eight-day personal
# history, exactly as the wave renders it.
PROMPTS = [
    ("Estimate how much this user likes Action movies based on their "
     "profile.\nProfile:\n- age: 24\n- gender: male\n- occupation: "
     "technician\n- average rating of Drama movies: 3.4 out of 5\n"
     "- average rating of Comedy movies: 3.9 out of 5\n\n"
     "This user's own opinion of Action movies over the most recent days "
     "(oldest to newest): 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.63, 0.65."
     "\n\nOutput a single number in [0, 1] (1 = loves Action, 0 = "
     "dislikes Action). Respond with only the number, e.g. 0.42."),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--n", type=int, default=3,
                    help=f"requests to send (hard max {MAX_REQUESTS})")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--i-understand-this-costs-money", action="store_true",
                    dest="confirmed")
    args = ap.parse_args()

    if not args.confirmed:
        print("[canary] refusing to spend money without "
              "--i-understand-this-costs-money", file=sys.stderr)
        return 2
    n = min(args.n, MAX_REQUESTS)
    if args.n > MAX_REQUESTS:
        print(f"[canary] clamping {args.n} -> {MAX_REQUESTS} requests")

    load_api_key(required=True)          # raises without echoing the key
    print("[canary] key present. Checking limits (metadata only)...")
    try:
        meta = validate_key()
        print(f"[canary] key OK: "
              f"free_tier={meta.get('is_free_tier')} "
              f"limit={meta.get('limit')} "
              f"remaining={meta.get('limit_remaining')}")
    except Exception as e:                                  # noqa: BLE001
        print(f"[canary] /key check failed: {e}", file=sys.stderr)
        return 1

    # PREFLIGHT, not assumption. The decoding policy is derived from the
    # pinned endpoint's live supported_parameters: sending a knob it does
    # not expose makes it ineligible under require_parameters=true and the
    # route fails. GPT-5.x exposes no temperature; Claude Opus 5 exposes no
    # seed. Free: public metadata, no key.
    import subprocess
    pf = subprocess.run(
        [sys.executable,
         str(Path(__file__).with_name("or_preflight.py")),
         "--model", args.model, "--provider", args.provider,
         "--seed", str(args.seed), "--json"],
        capture_output=True, text=True)
    if pf.returncode != 0:
        print(f"[canary] preflight failed:\n{pf.stderr}", file=sys.stderr)
        return 2
    info = json.loads(pf.stdout)
    model_id = info["resolved_model"]
    canonical = info["canonical_slug"]
    print(f"[canary] {args.model} -> {model_id} (canonical {canonical}) "
          f"via {args.provider}; temperature="
          f"{'0' if info['supports_temperature'] else 'OMITTED'}, "
          f"seed={'sent' if info['supports_seed'] else 'unsupported'}, "
          f"zdr={info['zdr']}")
    assert_canonical(model_id, canonical, when="at canary start")

    pin = ProviderPin(order=(args.provider,), allow_fallbacks=False,
                      require_parameters=False, data_collection="deny",
                      zdr=True)
    policy = DecodingPolicy(
        temperature=(0.0 if info["supports_temperature"] else None),
        top_p=1.0, max_tokens=args.max_tokens,
        seed=(args.seed if info["supports_seed"] else None),
        reasoning_mode=os.environ.get("OR_REASONING_MODE", "disabled"))
    budget = Budget(max_requests=MAX_REQUESTS, max_estimated_cost_usd=0.50,
                    max_realized_cost_usd=0.50, max_concurrency=1,
                    requests_per_second=1.0)
    client = OpenRouterClient(model=model_id, provider=pin, policy=policy,
                              budget=budget, cache=None)

    # the SAME prompt n times: repeat variance under temperature 0 is
    # exactly the reproducibility question, and it is free to ask here
    batch = [[{"role": "user", "content": PROMPTS[0]}] for _ in range(n)]
    print(f"[canary] sending {n} request(s) to {model_id} via "
          f"{args.provider} (allow_fallbacks=false, zdr=true, "
          f"data_collection=deny)...")
    provs = client.complete_many_sync(batch)

    texts = [p.text for p in provs]
    pin_tok = [p.prompt_tokens for p in provs]
    out_tok = [p.completion_tokens for p in provs]
    cost = sum(p.cost_usd or 0.0 for p in provs)
    print(f"\n[canary] responses: {texts}")
    print(f"[canary] resolved model:    {provs[0].resolved_model}")
    print(f"[canary] resolved provider: {provs[0].resolved_provider}")
    print(f"[canary] finish reasons:    {[p.finish_reason for p in provs]}")
    print(f"[canary] system fingerprint:{provs[0].system_fingerprint}")
    print(f"[canary] tokens in/out:     {pin_tok} / {out_tok}")
    print(f"[canary] latency (s):       {[round(p.latency_s,2) for p in provs]}")
    print(f"[canary] realized cost:     ${cost:.6f}")
    identical = len(set(texts)) == 1
    print(f"[canary] repeat determinism: {'IDENTICAL' if identical else 'DIFFERED'} "
          f"across {n} calls at temperature 0")
    if not identical:
        print("[canary] NOTE: differing outputs at T=0 are EXPECTED on some "
              "providers. Temperature 0 removes sampling, not the "
              "provider's batching/kernel nondeterminism. Report it; do "
              "not claim determinism.")
    print(f"\n[canary] FOR THE WAVE ESTIMATE, pass:  --canary-usage "
          f"{int(sum(pin_tok)/len(pin_tok))},{int(sum(out_tok)/len(out_tok))}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"requested": args.model, "model": model_id,
             "canonical_slug": canonical, "provider": args.provider,
             "policy": policy.to_body(),
             "n": n, "texts": texts, "identical": identical,
             "mean_prompt_tokens": sum(pin_tok) / len(pin_tok),
             "mean_completion_tokens": sum(out_tok) / len(out_tok),
             "realized_cost_usd": cost,
             "provenance": [p.to_dict() for p in provs]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
