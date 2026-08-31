#!/usr/bin/env python3
"""Candidate-model manifest from OpenRouter's live Models API.

WHAT THIS IS FOR. Picking a frontier model for a paper wave is a
REPRODUCIBILITY decision, not a shopping decision: the slug must be exact
and dated, the provider must be pinnable, the endpoint must actually
accept the parameters we require, and the whole thing must be available
under zero-data-retention. This script reports those facts and NOTHING
ELSE -- it does not choose, and it does not rank by "best".

COSTS NOTHING. /api/v1/models and /api/v1/models/{author}/{slug}/endpoints
are public metadata endpoints; no key is sent and no inference happens.

ZDR is determined the only honest way available: the same query is run
with and without ?zdr=true, and an endpoint counts as ZDR-capable only if
it survives the filtered query. A provider that vanishes under the filter
is reported as NOT ZDR rather than assumed fine.

  python or_catalog.py [--out manifest.json] [--authors openai,anthropic,google]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://openrouter.ai/api/v1"
# The parameters a paper run REQUIRES the endpoint to honour. require_parameters
# =true makes OpenRouter refuse an endpoint that cannot, so an endpoint missing
# one of these is unusable for us and is reported as such.
REQUIRED_PARAMS = ("temperature", "max_tokens")
NICE_PARAMS = ("seed", "top_p", "reasoning")


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "perfsim-catalog"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _iso(ts) -> str | None:
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(int(ts), timezone.utc).strftime("%Y-%m-%d")
    except Exception:                                       # noqa: BLE001
        return None


def _price(p: dict, key: str) -> float | None:
    v = p.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", default="openai,anthropic,google")
    ap.add_argument("--top-n", type=int, default=4,
                    help="newest N per author to report")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    authors = [a.strip() for a in args.authors.split(",") if a.strip()]

    allm = _get(f"{BASE}/models")["data"]
    zdr_ids = {m["id"] for m in _get(f"{BASE}/models?zdr=true")["data"]}

    rows = []
    for author in authors:
        cand = [m for m in allm if m["id"].startswith(f"{author}/")]
        # newest first; a paper run wants a current, dated slug
        cand.sort(key=lambda m: m.get("created") or 0, reverse=True)
        for m in cand[:args.top_n]:
            mid = m["id"]
            try:
                eps = _get(f"{BASE}/models/{mid}/endpoints")["data"]["endpoints"]
            except Exception as e:                          # noqa: BLE001
                eps = []
                print(f"[catalog] WARN {mid}: endpoints unavailable ({e})",
                      file=sys.stderr)
            try:
                zeps = {e["provider_name"] for e in
                        _get(f"{BASE}/models/{mid}/endpoints?zdr=true")
                        ["data"]["endpoints"]}
            except Exception:                               # noqa: BLE001
                zeps = set()
            for e in eps:
                sp = set(e.get("supported_parameters") or [])
                rows.append({
                    "author": author,
                    "model_id": mid,
                    "canonical_slug": m.get("canonical_slug"),
                    "name": m.get("name"),
                    "created": _iso(m.get("created")),
                    "knowledge_cutoff": m.get("knowledge_cutoff"),
                    "provider": e.get("provider_name"),
                    "context_length": e.get("context_length"),
                    "max_completion_tokens": e.get("max_completion_tokens"),
                    "quantization": e.get("quantization"),
                    "price_in_per_mtok": (
                        _price(e.get("pricing") or {}, "prompt") or 0) * 1e6,
                    "price_out_per_mtok": (
                        _price(e.get("pricing") or {}, "completion") or 0) * 1e6,
                    "supported_parameters": sorted(sp),
                    "has_required_params": all(p in sp for p in REQUIRED_PARAMS),
                    "supports_seed": "seed" in sp,
                    "supports_reasoning_control": "reasoning" in sp,
                    "zdr_model_level": mid in zdr_ids,
                    "zdr_this_provider": e.get("provider_name") in zeps,
                    "uptime_last_30m": e.get("uptime_last_30m"),
                })

    rows.sort(key=lambda r: (r["author"], r["model_id"], r["provider"] or ""))
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": BASE,
        "note": ("Candidates only. No model is selected here. ZDR is "
                 "verified by re-querying with ?zdr=true, never assumed."),
        "required_params": list(REQUIRED_PARAMS),
        "rows": rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"[catalog] wrote {args.out} ({len(rows)} model x provider rows)")

    hdr = (f"{'model_id':<42}{'provider':<22}{'created':<12}{'ctx':>9}"
           f"{'$in':>8}{'$out':>8}{'seed':>6}{'zdr':>5}{'req':>5}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['model_id']:<42}{(r['provider'] or '?'):<22}"
              f"{(r['created'] or '?'):<12}{r['context_length'] or 0:>9}"
              f"{r['price_in_per_mtok']:>8.2f}{r['price_out_per_mtok']:>8.2f}"
              f"{('yes' if r['supports_seed'] else 'no'):>6}"
              f"{('Y' if r['zdr_this_provider'] else 'n'):>5}"
              f"{('Y' if r['has_required_params'] else 'n'):>5}")
    print("\n  $in/$out are USD per MILLION tokens. seed: endpoint advertises "
          "a seed parameter.\n  zdr: this PROVIDER survives ?zdr=true. req: "
          "endpoint honours temperature+max_tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
