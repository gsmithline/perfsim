#!/usr/bin/env python3
"""OFFLINE POPULATION REPLAY under a real frozen-Qwen prediction vector
(2026-08-20, qwen_mechanism diagnostic). CPU only.

WHAT THIS IS, EXACTLY. A frozen model prompted with K = D = 0 never
sees the population: the prompt is the agent's static profile, decoding is
greedy, and nothing trains. Its parsed prediction vector is therefore a
CONSTANT -- bit-identical in every round, and independent of eps_AI and
eps_social. Eight archived H100 cells confirm this empirically (constant
across all 30 rounds, one shared sha256).

So generating that same 723-vector again for 300 rounds on a GPU would
burn hours to recompute a constant. Instead this script loads the vector
from a REAL archived H100 run and replays the population and peer process
around it offline, through the IDENTICAL operator path the LLM runs use
(sim_perfect_predictor.simulate with a served_fn hook -- there is no
second copy of the dynamics anywhere).

BE CLEAR ABOUT WHAT IS AND IS NOT MEASURED. The served values are genuine
H100 outputs of whatever checkpoint the SOURCE RUN served (the artifact
records base_model and replay_note straight from the source config --
never hardcode a model name here: nine 2026-08-24 artifacts briefly
claimed Qwen2.5 while replaying a Qwen3-8B vector). The population dynamics are a faithful offline
replay. What is NOT re-measured is the model: this cannot detect a frozen
model whose predictions would have drifted, because the whole premise --
verified, not assumed -- is that they do not. Every artifact records
platform="frozen_offline_replay" plus the source run tag and the vector's
sha256, and the analyzer labels these curves as replays.

The vector is hardware-specific. The archived A100 frozen cell differs
from the H100 prior in 17 of 723 agents (MAE .0091, max .5), so the
source run must be an H100 cell and its hash is checked against the
canonical one the audit derived.

Usage:
  python replay_frozen_offline.py --from-run <RUN_DIR> \\
      --innate-k 1 --w-plat 1 --eps-social .2 \\
      --ai-gate-mode all_open --peer-gate-mode all_open \\
      --rounds 300 --seed 0 [--expect-sha <sha256>]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
REPO = HERE.parent.parent.parent

_spec = importlib.util.spec_from_file_location(
    "sim_pp", str(HERE / "sim_perfect_predictor.py"))
PP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PP)

DEFAULT_OUT = REPO / "notes" / "pofd" / "frozen_replay"
H100_MARKER = "H100"


def load_frozen_vector(run_dir, expect_sha=None, require_h100=True,
                       gpu_marker=None):
    """The constant served vector of an archived frozen K=D=0 run.

    Hard-fails unless the run is on an H100, its predictions are constant
    across every recorded round, and (when given) its sha256 equals the
    canonical one. Returns (vector, sha256, source_meta)."""
    run_dir = Path(run_dir)
    import json
    cfg = json.load(open(run_dir / "config.json"))
    gpu = ((cfg.get("hardware") or {}).get("gpu_name") or "")
    # gpu_marker (2026-08-26): the hardware class the CALLER declares its
    # wave runs on (e.g. "NVIDIA A100-SXM4-80GB" for the Figure-4 anchor
    # wave while the H100 pool is down). Default keeps the H100 rule.
    marker = gpu_marker or H100_MARKER
    if require_h100 and marker not in gpu:
        raise SystemExit(
            f"[frozen] REFUSED: {run_dir.name} ran on {gpu or 'unknown GPU'}, "
            f"not a {marker}. Greedy decoding is bit-reproducible only "
            f"within one GPU architecture and the archived A100 frozen cell "
            f"differs from the H100 prior in 17 of 723 agents; a wave's "
            f"prior must come from the wave's own hardware class.")
    if cfg.get("training_style") != "frozen" or cfg.get("icl_k") not in (0,) \
            or cfg.get("icl_days") not in (0,):
        raise SystemExit(
            f"[frozen] REFUSED: {run_dir.name} is not a frozen K=D=0 run "
            f"(training_style={cfg.get('training_style')!r}, "
            f"icl_k={cfg.get('icl_k')!r}, icl_days={cfg.get('icl_days')!r})")
    d = torch.load(run_dir / "trajectory.pt", map_location="cpu",
                   weights_only=False)
    pr = d.get("pred_raw")
    if pr is None or pr.numel() == 0:
        raise SystemExit(f"[frozen] {run_dir.name}: pred_raw missing/empty")
    if not bool((pr == pr[0]).all()):
        nvary = int((pr != pr[0]).any(dim=0).sum())
        raise SystemExit(
            f"[frozen] REFUSED: {run_dir.name} predictions are NOT constant "
            f"across rounds ({nvary} of {pr.shape[1]} agents vary). A frozen "
            f"K=D=0 model cannot see the population, so this run is not what "
            f"it claims to be.")
    vec = pr[0].clone()
    sha = hashlib.sha256(vec.contiguous().numpy().tobytes()).hexdigest()
    if expect_sha and sha != expect_sha:
        raise SystemExit(
            f"[frozen] REFUSED: {run_dir.name} prediction sha256 {sha} != "
            f"canonical {expect_sha}. Mixing two frozen priors into one grid "
            f"would contaminate every comparison drawn across it.")
    return vec, sha, {"source_run_tag": cfg.get("run_tag") or run_dir.name,
                      "source_run_dir": str(run_dir),
                      "source_gpu_name": gpu,
                      "source_n_rounds": int(pr.shape[0]),
                      "base_model": cfg.get("base_model")}


def main():
    ap = argparse.ArgumentParser(
        description="offline population replay under a real frozen-Qwen "
                    "prediction vector (CPU)")
    ap.add_argument("--from-run", required=True,
                    help="archived H100 frozen K=D=0 run directory")
    ap.add_argument("--expect-gpu", default=None,
                    help="hardware class the source run must have served on "
                         "(substring of config.hardware.gpu_name); default "
                         "H100. The Figure-4 anchor wave passes "
                         "'NVIDIA A100-SXM4-80GB'.")
    ap.add_argument("--expect-sha", default=None,
                    help="canonical prediction sha256 (from the audit "
                         "manifest); mismatch is a hard failure")
    ap.add_argument("--innate-k", type=float, required=True)
    ap.add_argument("--w-plat", type=float, required=True)
    ap.add_argument("--eps-social", type=float, required=True)
    ap.add_argument("--eps-ai", type=float, default=1.0)
    ap.add_argument("--ai-gate-mode", default="threshold",
                    choices=("threshold", "all_open"))
    ap.add_argument("--peer-gate-mode", default="threshold",
                    choices=("threshold", "all_open"))
    ap.add_argument("--sweeps", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=0.0)
    # Deffuant alpha (2026-08-24): sim_perfect_predictor.build_config
    # started reading args.alpha, which this parser did not define -- the
    # replay path died with AttributeError before writing anything. The
    # default 0.5 is the value every archived frozen artifact was built
    # with (sim_perfect_predictor.simulate defaults deffuant_alpha=0.5 and
    # artifact_name emits NO _a token at 0.5, which is why the existing
    # filenames carry none), so this restores the old behaviour exactly
    # rather than choosing a new one.
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Deffuant step size; 0.5 reproduces every "
                         "archived frozen replay")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not 0.0 <= args.alpha <= 0.5:
        ap.error("--alpha must lie in [0, 0.5]")
    if args.peer_gate_mode == "all_open" and args.eps_social <= 0:
        ap.error("--peer-gate-mode all_open with --eps-social 0 is "
                 "contradictory (eps_social=0 is the NO-PEER condition)")

    vec, sha, src = load_frozen_vector(args.from_run, args.expect_sha,
                                       gpu_marker=args.expect_gpu)
    setup = PP.extract_loader()(
        REPO / "experiments/data/movielens/ml-100k", "Action")
    if vec.shape[0] != int(setup["n"]):
        raise SystemExit(f"[frozen] vector length {vec.shape[0]} != "
                         f"{setup['n']} agents")

    # the frozen replay runs the same corrected operator PP does
    cfg = PP.build_config(args, setup, gate_on="anchor")
    cfg.update({
        "platform": "frozen_offline_replay",
        "expected_gpu": args.expect_gpu or H100_MARKER,
        "frozen_pred_sha256": sha,
        # the model name comes from the SOURCE RUN's config, never a
        # literal: the 2026-08-24 audit caught nine artifacts whose note
        # said Qwen2.5 while the source was a Qwen3-8B run
        "replay_note": (f"offline population replay; served values are real "
                        f"H100 {src.get('base_model', 'UNKNOWN-MODEL')} "
                        f"frozen K=D=0 outputs, held constant as the "
                        f"archived run demonstrates they are"),
        **src,
    })
    # eps_social is a real dial here (it drives the peer gate), and the
    # perfect-prediction beta_eff identity does NOT apply to a frozen
    # served vector -- drop the field rather than record a misleading one
    cfg.pop("beta_eff", None)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / ("frz" + PP.artifact_name(cfg)[2:])
    if out.exists() and not args.force:
        print(f"[frozen] {out.name} already exists -- refusing to "
              f"overwrite. Pass --force to regenerate.", file=sys.stderr)
        return 0

    const = vec.clone()
    op_raw, twin_raw, served_raw = PP.simulate(
        setup, innate_k=cfg["innate_k"], w_plat=cfg["w_plat"],
        eps_social=cfg["eps_social"], eps_ai=cfg["eps_ai"],
        rounds=cfg["rounds"], seed=cfg["seed"],
        ai_gate_mode=cfg["ai_gate_mode"],
        peer_gate_mode=cfg["peer_gate_mode"],
        ab_sweeps=cfg["ab_sweeps"], gamma=cfg["gamma_bias"],
        served_fn=lambda x, t: const, require_open_gate=False)

    torch.save({"config": cfg, "op_raw": op_raw, "twin_raw": twin_raw,
                "pred_raw": served_raw,
                "innate": setup["innate"].clone()}, out)
    if not args.quiet:
        fin = op_raw[-1]
        print(f"[frozen] replay k={cfg['innate_k']:g} W={cfg['w_plat']:g} "
              f"ai={cfg['ai_gate_mode']} peer={cfg['peer_gate_mode']} "
              f"rounds={cfg['rounds']} from {src['source_run_tag']}")
        print(f"[frozen]   served vector sha256 {sha[:16]}... "
              f"mean {float(const.mean()):.6f} sd {float(const.std()):.6f}")
        print(f"[frozen]   final mean {float(fin.mean()):.6f} "
              f"sd {float(fin.std()):.6f} "
              f"range {float(fin.max() - fin.min()):.4e}")
        print(f"[frozen] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
