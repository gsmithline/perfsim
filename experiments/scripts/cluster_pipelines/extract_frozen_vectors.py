#!/usr/bin/env python3
"""Extract the frozen (lambda -> infinity) zero-shot vectors for the FJ
robustness wave. RUN ON THE CLUSTER; writes a small npz to pull.

REUSE IS EARNED, NOT ASSUMED. An archived vector is accepted only if
every one of these matches, checked here in code rather than by eye:

  base_model                  the exact checkpoint id
  dataset / ml_target         movielens / Action
  n_labeled, agent count      723, and 723 profile rows
  training_style              "frozen"
  use_lora                    False -- no adapter of any kind
  icl_k, icl_days             0 / 0 -- no context in the prompt
  pred_raw constant           identical in every round, which is what
                              proves the model never saw the population
  gpu_name                    NVIDIA H100 80GB HBM3, EXACTLY. Greedy
                              decoding is only bit-reproducible within an
                              architecture: the archived A100 frozen cell
                              differs from the H100 prior in 17/723
                              agents (MAE .0091, max .5), so mixing them
                              would measure one model against a
                              different-hardware reference.
  transformers / torch        5.5.4 / 2.5.1

AND, across every qualifying run for a model, the prediction hash must be
UNIQUE. Two different hashes means the "frozen prior" is not well defined
for that model and no vector is emitted -- the wave takes an extraction
job instead of quietly picking one.

A note on serve_eval_mode: it is absent from these artifacts (the field
postdates them) and that is safe here, because use_lora=False means no
LoRA dropout exists and HF's from_pretrained returns a model in eval
mode. It is also confirmed empirically: current code, with the eval-mode
fix, reproduces the pinned Qwen2.5 hash exactly.

Usage (cluster):
  python extract_frozen_vectors.py --out ~/fj_frozen.npz
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

H100 = "NVIDIA H100 80GB HBM3"
MODELS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "ministral8b": "mistralai/Ministral-8B-Instruct-2410",
}
N_AGENTS = 723


def qualifies(cfg, pred_raw):
    """(ok, reason). Reason is returned even on success so a run that
    nearly qualifies can be told apart from one that is far off."""
    hw = cfg.get("hardware") or {}
    if cfg.get("training_style") != "frozen":
        return False, f"training_style={cfg.get('training_style')!r}"
    if cfg.get("use_lora"):
        return False, "use_lora"
    if cfg.get("icl_k") not in (0, None):
        return False, f"icl_k={cfg.get('icl_k')}"
    if cfg.get("icl_days") not in (0, None):
        return False, f"icl_days={cfg.get('icl_days')}"
    if cfg.get("dataset") != "movielens" or cfg.get("ml_target") != "Action":
        return False, f"dataset={cfg.get('dataset')}/{cfg.get('ml_target')}"
    if cfg.get("n_labeled") != N_AGENTS:
        return False, f"n_labeled={cfg.get('n_labeled')}"
    if hw.get("gpu_name") != H100:
        return False, f"gpu={hw.get('gpu_name')!r}"
    if hw.get("transformers_version") != "5.5.4":
        return False, f"transformers={hw.get('transformers_version')}"
    if hw.get("torch_version") != "2.5.1":
        return False, f"torch={hw.get('torch_version')}"
    if pred_raw.numel() == 0 or pred_raw.shape[1] != N_AGENTS:
        return False, f"pred_raw={tuple(pred_raw.shape)}"
    if pred_raw.shape[0] > 1 and float((pred_raw - pred_raw[0]).abs().max()) > 0:
        return False, "pred_raw not constant across rounds"
    if not torch.isfinite(pred_raw).all():
        return False, "non-finite predictions"
    return True, "ok"


def scan(runs_root):
    found = defaultdict(lambda: defaultdict(list))
    inv = {v: k for k, v in MODELS.items()}
    for p in sorted(glob.glob(os.path.join(runs_root, "*", "trajectory.pt"))):
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        cfg = d.get("config") or {}
        slug = inv.get(cfg.get("base_model"))
        if slug is None:
            continue
        pr = d.get("pred_raw")
        if pr is None or not torch.is_tensor(pr):
            continue
        pr = pr.float()
        ok, _ = qualifies(cfg, pr)
        if not ok:
            continue
        nprof = -1
        prof = d.get("profiles")
        if isinstance(prof, dict) and prof:
            nprof = len(prof[list(prof)[0]])
        if nprof not in (-1, N_AGENTS):
            continue
        sha = hashlib.sha256(pr[0].contiguous().numpy().tobytes()).hexdigest()
        found[slug][sha].append(os.path.basename(os.path.dirname(p)))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs/pokec_gated_lm")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    found = scan(args.runs_root)
    vecs, sources, problems = {}, {}, []
    for slug in MODELS:
        shas = found.get(slug, {})
        if not shas:
            problems.append(f"{slug}: NO qualifying vector -> extraction job")
            continue
        if len(shas) > 1:
            problems.append(
                f"{slug}: {len(shas)} DISTINCT prediction hashes across "
                f"qualifying runs -- the frozen prior is not well defined, "
                f"refusing to pick one: "
                + "; ".join(f"{s[:12]} x{len(t)}" for s, t in shas.items()))
            continue
        sha, tags = next(iter(shas.items()))
        p = os.path.join(args.runs_root, tags[0], "trajectory.pt")
        d = torch.load(p, map_location="cpu", weights_only=False)
        vecs[slug] = d["pred_raw"].float()[0].numpy()
        sources[slug] = f"{tags[0]} sha256={sha}"
        print(f"[frozen] {slug:<14} {sha[:16]}...  x{len(tags)} qualifying "
              f"run(s), e.g. {tags[0]}")

    for p in problems:
        print(f"[frozen] {p}", file=sys.stderr)
    if not vecs:
        print("[frozen] nothing to write", file=sys.stderr)
        return 1
    np.savez(args.out, sources=np.array(sources, dtype=object), **vecs)
    print(f"[frozen] wrote {args.out} with {len(vecs)}/{len(MODELS)} models")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
