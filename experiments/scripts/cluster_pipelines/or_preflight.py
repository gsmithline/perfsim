#!/usr/bin/env python3
"""Resolve a model + provider against the LIVE catalog and derive the
decoding policy the endpoint can actually honour. Free: public metadata
only, no key, no inference.

WHY THIS EXISTS. With require_parameters=true and allow_fallbacks=false,
sending a knob the pinned endpoint does not expose makes that endpoint
ineligible and the request fails. The support matrix differs per model
AND per provider (GPT-5.x exposes seed but no temperature; Claude Opus 5
exposes temperature only on Azure; Gemini exposes both), so hard-coding a
policy per model is a stale lookup waiting to happen. This derives it.

It also resolves a CANONICAL SLUG to the routable id. The dated forms
(openai/gpt-5.6-sol-20260709) are canonical_slugs, which the API will not
accept as `model`; the :batch variant shares the slug and is excluded,
being a different product.

  eval "$(python or_preflight.py --model openai/gpt-5.6-sol-20260709 \
             --provider OpenAI --seed 0)"
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

BASE = "https://openrouter.ai/api/v1"


def get(u):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={"User-Agent": "perfsim"}),
        timeout=60).read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    models = get(f"{BASE}/models")["data"]
    ids = {m["id"] for m in models}
    if args.model in ids:
        mid = args.model
    else:
        cands = [m["id"] for m in models
                 if m.get("canonical_slug") == args.model
                 and not m["id"].endswith(":batch")]
        if len(cands) != 1:
            print(f"# FATAL: {args.model!r} is neither a routable id nor a "
                  f"canonical_slug with exactly one non-batch model "
                  f"(matches: {cands})", file=sys.stderr)
            return 2
        mid = cands[0]

    # THE PIN. Whatever the user named, the wave is pinned to the DATED
    # canonical slug the routable id carries right now, and every round
    # re-checks it. A provider that re-points the id at a new build mid-wave
    # would otherwise stitch two models into one trajectory.
    canon = next((m.get("canonical_slug") for m in models
                  if m["id"] == mid), None)
    if canon is None:
        print(f"# FATAL: {mid} carries no canonical_slug", file=sys.stderr)
        return 2
    if args.model != mid and args.model != canon:
        print(f"# FATAL: requested {args.model!r} but {mid} carries "
              f"canonical_slug {canon!r}", file=sys.stderr)
        return 2

    eps = get(f"{BASE}/models/{mid}/endpoints")["data"]["endpoints"]
    zdr = {e["provider_name"] for e in
           get(f"{BASE}/models/{mid}/endpoints?zdr=true")["data"]["endpoints"]}
    match = [e for e in eps if e["provider_name"] == args.provider]
    if not match:
        print(f"# FATAL: {mid} has no endpoint from provider "
              f"{args.provider!r}; available: "
              f"{sorted({e['provider_name'] for e in eps})}", file=sys.stderr)
        return 2
    e = match[0]
    sp = set(e.get("supported_parameters") or [])

    if args.provider not in zdr:
        print(f"# FATAL: {mid} via {args.provider} is NOT available under "
              f"zero-data-retention routing; refusing to relax zdr",
              file=sys.stderr)
        return 2
    if "max_tokens" not in sp:
        print(f"# FATAL: {mid} via {args.provider} does not expose "
              f"max_tokens; an explicit completion limit is required",
              file=sys.stderr)
        return 2

    temp = "0" if "temperature" in sp else ""     # "" => omit the knob
    seed = str(args.seed) if ("seed" in sp and args.seed is not None) else ""
    price = e.get("pricing") or {}
    info = {
        "requested": args.model, "resolved_model": mid,
        "canonical_slug": canon,
        "provider": args.provider, "zdr": True,
        "supports_temperature": "temperature" in sp,
        "supports_seed": "seed" in sp,
        "supports_reasoning_control": "reasoning" in sp,
        "context_length": e.get("context_length"),
        "max_completion_tokens": e.get("max_completion_tokens"),
        "price_in_per_mtok": float(price.get("prompt", 0)) * 1e6,
        "price_out_per_mtok": float(price.get("completion", 0)) * 1e6,
    }
    if args.json:
        print(json.dumps(info, indent=1))
        return 0
    print(f"export OR_MODEL={mid}")
    print(f"export OR_PROVIDER={json.dumps(args.provider)}")
    print(f"export OR_TEMPERATURE={json.dumps(temp)}")
    print(f"export OR_SEED={json.dumps(seed)}")
    # REASONING. Some endpoints refuse to disable it (Gemini 3.1 Pro:
    # "Reasoning is mandatory ... cannot be disabled", HTTP 400). The client
    # falls back from disabled to minimal on that error and records it, but
    # starting in the right mode avoids a wasted round-trip per request.
    reasoning_mode = "disabled"
    # COMPLETION BUDGET. A reasoning endpoint spends most of the budget on
    # reasoning tokens before emitting the number: Gemini 3.1 Pro used 301
    # reasoning tokens for a 4-character answer, so a 16-token budget
    # returns finish_reason=length and no content at all.
    max_tokens = 16 if "reasoning" not in sp else 2048
    print(f"export OR_EXPECTED_CANONICAL={canon}")
    print(f"export OR_REASONING_MODE={reasoning_mode}")
    print(f"export OR_MAX_TOKENS={max_tokens}")
    print(f"export OR_PRICE_IN={info['price_in_per_mtok']:.4f}")
    print(f"export OR_PRICE_OUT={info['price_out_per_mtok']:.4f}")
    print(f"# resolved {args.model} -> {mid} via {args.provider}: "
          f"temperature={'sent' if temp else 'OMITTED (unsupported)'}, "
          f"seed={'sent' if seed else 'not supported'}, zdr=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
