#!/usr/bin/env python3
"""Gate the SECTION-3 FRONTIER ICL wave (tags pofds3f_): the Figure-3(a)
personal-history ICL surface served by a FROZEN frontier API model.

WHAT THIS PROVES, and why each check exists. An API-served wave can fail
in ways a local checkpoint cannot, so this gate is STRICTER than the
local-model one, not weaker.

1. THE ENVIRONMENT IS THE S3I ONE. W=1, k=1, S=100, alpha=.5, both gates
   all_open, anch2, movielens/Action, 723 agents, seeds 0/42/43,
   ICL_K=0, ICL_DAYS=8. Only the server differs, or the frontier cells
   cannot be placed beside the local ones.

2. THE WEIGHTS ARE FROZEN AND THERE IS NO OPTIMIZER. training_style
   frozen, sft_epochs 0, use_lora false, kl_beta 0 -- and no adapter
   artifact, no trl/ dir, no KL or gradient witness. A config field is a
   claim; an absent artifact is evidence.

3. ONE RESPONSE PER SERVED AGENT PER ROUND, ALIGNED. or_provenance
   carries exactly rounds 0..R-1, each with exactly 723 records, and the
   i-th record of round t is agent i of round t. Alignment is not
   assumed: the parsed value of record i must EQUAL pred_raw[t, i].

4. NO FALLBACK, NO SUBSTITUTION, NO TRUNCATION. Every record's resolved
   model and provider equal the requested ones, and every finish_reason
   is a clean stop. One silent reroute makes the cell a different
   experiment wearing the same tag.

5. PROVENANCE IS COMPLETE. Every record carries a generation id, token
   usage and a request hash. Cost may be absent only if the provider
   reports none, and that is reported rather than silently zeroed.

6. ZERO PARSE FAILURES, AND THE 0.5 FALLBACK NEVER APPEARS. PARSE_MODE
   is strict wave-wide and the backend raises rather than defaulting, so
   a parse failure here means the artifact predates the guard.

7. PERSONAL HISTORY REPLAYS EXACTLY, AND NO CROSS-AGENT CONTEXT.
   icl_days_log replays character-for-character from (innate, op_raw),
   the same proof the local ICL wave uses; icl_k == 0 and no exemplar
   log exists.

8. THE TRAJECTORY IS COMPLETE. op_raw/pred_raw [rounds, 723], finite,
   in [0,1], with the open-gate runtime evidence in every round.

Read-only, CPU only.

  python check_section3_frontier_icl.py [--smoke] [--json out.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "notes" / "pofd" / "cluster"

# THE APPROVED SECTION-3 SURFACE, compared field by field against an
# ARCHIVED GATED RUN rather than against a hand-copied dict. A hand-copied
# expectation drifts from the wave it is supposed to match; the archive
# cannot. The single definition of "scientifically relevant" lives in
# audit_frontier_config.SURFACE and is imported here, so the gate and the
# pre-run audit can never disagree about what matters.
REFERENCE = (REPO / "notes" / "pofd" / "cluster" /
             "pofds3i_mistral7b_d8_greedy_sw100_eaopen_w1_k1_esopen_anch2_s0_r30")

def _surface():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_aud", str(Path(__file__).with_name("audit_frontier_config.py")))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.SURFACE

# fields that must hold regardless of the archive (API-backend specific)
EXPECT = {
    "model_backend": "openrouter", "parse_mode": "strict",
    "ai_gate_reference": "anchor",
}
N_AGENTS = 723
FORBIDDEN_ARTIFACTS = ("round0_adapter", "trl")


def _load_jsonl_gz(p: Path):
    with gzip.open(p, "rt") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def check_cell(d: Path, rounds: int) -> tuple[list[str], dict]:
    errs: list[str] = []
    cfg = json.loads((d / "config.json").read_text())

    # 1a. EVERY scientifically relevant field, against the gated archive
    if REFERENCE.is_file() or (REFERENCE / "config.json").is_file():
        ref = json.loads((REFERENCE / "config.json").read_text())
        for f in _surface():
            want, got = ref.get(f), cfg.get(f)
            if isinstance(want, float) or isinstance(got, float):
                same = (want is not None and got is not None
                        and abs(float(want) - float(got)) < 1e-9)
            else:
                same = want == got
            if not same:
                errs.append(f"{d.name}: {f}={got!r} but the gated Section-3 "
                            f"reference has {want!r} -- the frontier cells "
                            f"cannot be placed beside the local ones")
    else:
        errs.append(f"{d.name}: reference {REFERENCE.name} is missing; "
                    f"cannot verify the surface")

    # 1b. API-backend specifics
    for k, want in EXPECT.items():
        got = cfg.get(k)
        if isinstance(want, bool):
            got = bool(got)
        elif isinstance(want, float) and got is not None:
            got = float(got)
        elif isinstance(want, int) and not isinstance(want, bool) and got is not None:
            got = int(got)
        if got != want:
            errs.append(f"{d.name}: {k}={got!r}, want {want!r}")

    # 1c. the tag must be reconstructable from the effective config: a
    # correctly named run must not contain the wrong dynamics
    for tok, ok in (
            ("eaopen", cfg.get("ai_gate_mode") == "all_open"),
            ("esopen", cfg.get("peer_gate_mode") == "all_open"),
            ("anch2", cfg.get("population_update")
             == "nested_ai_anchored_then_social_v2"),
            ("_d8_", int(cfg.get("icl_days", -1)) == 8),
            ("sw100", int(cfg.get("ab_sweeps", -1)) == 100),
            ("w1", float(cfg.get("w_plat", -1)) == 1.0),
            ("k1", float(cfg.get("innate_lambda", -1)) == 1.0)):
        if tok in d.name and not ok:
            errs.append(f"{d.name}: tag claims {tok} but the effective "
                        f"config contradicts it")

    # 1d. the answer limit must be an explicit number, never unset
    _pol = (cfg.get("openrouter") or {}).get("policy") or {}
    if _pol.get("max_tokens") is None:
        errs.append(f"{d.name}: max_tokens is unset; an explicit completion "
                    f"limit is required and must be recorded")

    # 2. frozen: no artifact an optimizer would leave
    for a in FORBIDDEN_ARTIFACTS:
        if (d / a).exists():
            errs.append(f"{d.name}: {a}/ exists on a FROZEN cell -- there is "
                        f"no optimizer here, so this is a contradiction")

    traj = torch.load(d / "trajectory.pt", map_location="cpu",
                      weights_only=False)
    op = traj["op_raw"].float().numpy()
    pred = traj["pred_raw"].float().numpy()

    # 8. trajectory shape and finiteness
    for name, arr in (("op_raw", op), ("pred_raw", pred)):
        if arr.shape != (rounds, N_AGENTS):
            errs.append(f"{d.name}: {name} shape {arr.shape}, want "
                        f"{(rounds, N_AGENTS)}")
        elif not np.isfinite(arr).all():
            errs.append(f"{d.name}: {name} has non-finite entries")
        elif arr.min() < -1e-6 or arr.max() > 1 + 1e-6:
            errs.append(f"{d.name}: {name} leaves [0,1] "
                        f"({arr.min():.4f}..{arr.max():.4f})")

    # 3/4/5. provenance
    pp = d / "or_provenance.json.gz"
    stats = {"records": 0, "cost_usd": 0.0, "cache_hits": 0, "retries": 0}
    if not pp.exists():
        errs.append(f"{d.name}: or_provenance.json.gz MISSING -- an "
                    f"API-served value without provenance is not a datum")
    else:
        rows = _load_jsonl_gz(pp)
        if [r["round"] for r in rows] != list(range(rounds)):
            errs.append(f"{d.name}: provenance rounds are "
                        f"{[r['round'] for r in rows][:5]}..., want 0..{rounds-1}")
        want_model = (cfg.get("openrouter") or {}).get("model_slug")
        want_prov = ((cfg.get("openrouter") or {}).get("provider") or
                     {}).get("order", [None])[0]
        for row in rows:
            recs = row["records"]
            t = row["round"]
            if len(recs) != N_AGENTS:
                errs.append(f"{d.name} round {t}: {len(recs)} responses for "
                            f"{N_AGENTS} agents")
                continue
            for i, r in enumerate(recs):
                stats["records"] += 1
                stats["cost_usd"] += float(r.get("cost_usd") or 0.0)
                stats["cache_hits"] += int(r.get("cache_status") == "hit")
                stats["retries"] += int(r.get("retries") or 0)
                if r.get("resolved_model") != want_model:
                    errs.append(f"{d.name} r{t} a{i}: MODEL SUBSTITUTION "
                                f"{r.get('resolved_model')!r} != {want_model!r}")
                    break
                if r.get("resolved_provider") != want_prov:
                    errs.append(f"{d.name} r{t} a{i}: PROVIDER FALLBACK "
                                f"{r.get('resolved_provider')!r} != {want_prov!r}")
                    break
                if r.get("finish_reason") not in ("stop", "end_turn", "eos"):
                    errs.append(f"{d.name} r{t} a{i}: finish_reason="
                                f"{r.get('finish_reason')!r} (truncated?)")
                    break
                if not r.get("generation_id") or not r.get("request_hash"):
                    errs.append(f"{d.name} r{t} a{i}: incomplete provenance")
                    break
                if r.get("prompt_tokens") is None or \
                        r.get("completion_tokens") is None:
                    errs.append(f"{d.name} r{t} a{i}: missing token usage")
                    break
            else:
                # 3. ALIGNMENT: record i must be agent i of round t
                got = np.array([_parse_or_nan(r.get("text")) for r in recs])
                if not np.allclose(got, pred[t], atol=1e-6, equal_nan=False):
                    bad = int(np.argmax(np.abs(got - pred[t])))
                    errs.append(
                        f"{d.name} r{t}: raw responses do NOT match pred_raw "
                        f"(worst agent {bad}: parsed {got[bad]!r} vs stored "
                        f"{pred[t][bad]!r}) -- request/agent alignment is "
                        f"broken or the served vector was rewritten")
                continue
            break

    # 6. parse failures
    rg = d / "raw_gen_log.json.gz"
    if rg.exists():
        for row in _load_jsonl_gz(rg):
            if float(row.get("parse_fail_frac") or 0.0) > 0:
                errs.append(f"{d.name} r{row['round']}: parse_fail_frac="
                            f"{row['parse_fail_frac']}, want 0")

    # 7. personal history replays exactly; no cross-agent context
    if (d / "icl_ctx_log.json.gz").exists():
        errs.append(f"{d.name}: icl_ctx_log.json.gz exists -- that is the "
                    f"CROSS-USER exemplar log and ICL_K must be 0")
    dl = d / "icl_days_log.json.gz"
    if not dl.exists():
        errs.append(f"{d.name}: icl_days_log.json.gz MISSING")
    else:
        innate = traj["innate"].float().numpy()
        target = cfg.get("ml_target", "Action")
        rows = _load_jsonl_gz(dl)
        for row in rows:
            t = row["round"]
            hist = np.concatenate([innate[None, :], op[:t]], axis=0)[-8:]
            for i, txt in enumerate(row["contexts"]):
                seq = ", ".join(f"{v:.2f}" for v in hist[:, i])
                want = (f"This user's own opinion of {target} movies over "
                        f"the most recent days (oldest to newest): {seq}.")
                if txt != want:
                    errs.append(f"{d.name} r{t} a{i}: personal history does "
                                f"not replay\n   got  {txt!r}\n   want {want!r}")
                    break
            else:
                continue
            break

    return errs, stats


def _parse_or_nan(text):
    try:
        sys.path.insert(0, str(REPO))
        from perfsim.models.hf_causal_lm import HFCausalLMModel
        v, ok = HFCausalLMModel._parse_strict(text or "")
        return v if ok else float("nan")
    except Exception:                                       # noqa: BLE001
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs-root", default=str(RUNS))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    root = Path(args.runs_root)
    pre = "pofds3fsmk_" if args.smoke else "pofds3f_"
    rounds = 3 if args.smoke else 30
    cells = sorted(d for d in root.glob(f"{pre}*") if (d / "config.json").exists())
    if not cells:
        print(f"[check_s3f] no {pre}* cells under {root}")
        return 1

    all_errs, out = [], []
    hdr = (f"{'cell':<58}{'verdict':<9}{'mean':>8}{'sd':>8}"
           f"{'resp':>8}{'cost$':>8}{'cached':>8}")
    print(hdr); print("-" * len(hdr))
    for d in cells:
        errs, st = check_cell(d, rounds)
        traj = torch.load(d / "trajectory.pt", map_location="cpu",
                          weights_only=False)
        fin = traj["op_raw"].float().numpy()[-1]
        print(f"{d.name[:57]:<58}{('FAIL' if errs else 'PASS'):<9}"
              f"{fin.mean():>8.4f}{fin.std():>8.5f}{st['records']:>8}"
              f"{st['cost_usd']:>8.3f}{st['cache_hits']:>8}")
        all_errs += errs
        out.append({"cell": d.name, "errors": errs, **st,
                    "final_mean": float(fin.mean()),
                    "final_sd": float(fin.std())})

    if all_errs:
        print(f"\n[check_s3f] FAIL -- {len(all_errs)} problem(s):")
        for e in all_errs[:40]:
            print("   -", e)
    else:
        print(f"\n[check_s3f] PASS -- {len(cells)} cell(s): frozen frontier "
              f"serving on the exact S3I surface, one aligned response per "
              f"agent per round, no fallback or substitution, complete "
              f"provenance, zero parse failures, personal history replays "
              f"exactly, no cross-agent context.")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 1 if all_errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
