#!/usr/bin/env python3
"""PERFECT-PREDICTION oracle for the pofd population dynamics -- CPU only,
no LLM, no GPU, no Condor. Tracked pipeline version (2026-08-20,
qwen_mechanism diagnostic); supersedes the untracked
notes/pofd/sim_perfect_predictor.py, which is left in place and is NEVER
read or relabelled by anything here.

The platform serves each agent exactly their current opinion,

    m(t) = x(t)                                  (perfect prediction)

so the AI gate |m - x| = 0 is open for every agent under any positive
threshold, and the implemented nested operator

    h(t) = k x^innate + (1 - k) x(t)
    z(t) = (1 - W) h(t) + W m(t)                 (gated agents)
    x(t+1) = D_eps_social( z(t) )                (peer sweeps last)

collapses to a pure Friedkin-Johnsen pre-peer step

    z(t) = (1 - beta_eff) x^innate + beta_eff x(t),
    beta_eff = 1 - (1 - W) k.

THE POINT OF beta_eff. Perfect prediction does NOT mean "the platform does
nothing": it dilutes the innate anchor from k to (1 - W) k, because the
attention share W is diverted onto a signal that merely echoes the current
state. The oracle is dynamically inert only when beta_eff = 1, i.e. W = 1
or k = 0. This is the quantity that maps our setting onto Wu et al.,
"Reaching a Consensus in Predictive Loops" (arXiv:2603.12137):

  * the paper regime k = .2, W = .5  ->  beta_eff = .9
  * k = 1, W = .5                    ->  beta_eff = .5, so k = 1 ALONE is
    NOT a consensus limit -- it is a LESS anchored pre-peer map, not a
    more anchored one
  * with imperfect (real Qwen) predictions, k = 1 gives the direct
    Wu-style form z = (1 - W) x^innate + W m
  * the high-susceptibility boundary is therefore k = 1, W = 1, where the
    pre-peer population equals the served vector and k drops out of the
    algebra entirely
  * Wu et al.'s consensus result needs BOTH perfect prediction AND
    platform susceptibility approaching one. Perfect prediction at finite
    susceptibility can and does retain a heterogeneous equilibrium.

CORRESPONDENCE, NOT REPLICATION. Our peer operator is a randomized
Deffuant midpoint process on a graph, not Wu et al.'s deterministic
Friedkin-Johnsen operator. With a connected graph and genuinely open peer
interaction it is a randomized-gossip analogue that should converge to
consensus under perfect prediction at W = 1. That is a qualitative
limiting-case correspondence and is described as such everywhere in this
pipeline -- it is not a proof, and it does not reproduce their theorem.

GENUINELY OPEN GATES. eps_ai = 1 and eps_social = 1 are NUMERIC
thresholds, and both gates are strict inequalities, so a pair or an
(agent, prediction) sitting at distance exactly 1 is still REJECTED. The
open conditions are therefore spelled as modes, never as the number 1:
--ai-gate-mode all_open, --peer-gate-mode all_open. The filename records
which was used.

MATCHING THE LLM RUNS -- read this before comparing.
The population operators (gp.nested_presocial_update, gp.ab_sweep) are
imported from the same _gated_pop module the runner uses, and the peer
generator is seeded exactly as the runner seeds it (seed + 424243). What
is NOT identical is the DEVICE: the cluster runs build
torch.Generator(device="cuda"), whose stream differs from a CPU generator
at the same seed. So the oracle is matched to an LLM run on graph, innate
vector, operator, gate semantics, horizon and seed -- but its realized
peer PAIRS are its own. Oracle-vs-oracle comparisons (every identity
check and the whole Part E grid) are exactly RNG-matched; oracle-vs-LLM
comparisons are distributional, and the analyzer treats them that way.
The oracle's twin, likewise, is the CPU twin of the oracle, not a
bit-copy of the run's in-GPU twin.

Usage:
  python sim_perfect_predictor.py --innate-k .2 --w-plat .5 \
      --eps-social .2 --rounds 30 --seed 0
  python sim_perfect_predictor.py --innate-k 1 --w-plat 1 \
      --eps-social .2 --peer-gate-mode all_open --ai-gate-mode all_open \
      --rounds 300 --seed 0
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import _gated_pop as gp                                      # noqa: E402
from perfsim.environments.dynamics.fj import (               # noqa: E402
    normalize_adjacency)

RUNNER = HERE / "run_pokec_gated_lm.py"
# default artifact namespace: NEW, and deliberately not the legacy
# notes/pofd/cluster/_pp_baseline/ppbase_*.pt namespace, so no archived
# oracle is ever overwritten, relabelled, or silently mixed in
DEFAULT_OUT = REPO / "notes" / "pofd" / "perfect_prediction"
PEER_SEED_OFFSET = 424243          # the runner's ab_gen / ab_gen_cf offset


def extract_loader():
    """Pull load_movielens_setup out of the runner WITHOUT importing it
    (the runner's top-level transformers import hangs on some machines).
    AST extraction means the environment cannot drift from the runner's:
    if the loader changes, this changes with it."""
    tree = ast.parse(RUNNER.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef)
              and n.name == "load_movielens_setup")
    ns = {"pd": pd, "np": np, "nx": nx, "torch": torch, "Path": Path,
          "normalize_adjacency": normalize_adjacency}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(RUNNER),
                 "exec"), ns)
    return ns["load_movielens_setup"]


def _sha(t):
    return hashlib.sha256(
        t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _num(v):
    return f"{v:g}".replace(".", "p").replace("-", "m")


def beta_eff(k, w):
    """Effective self-weight of the pre-peer map under perfect
    prediction: z = (1 - beta_eff) innate + beta_eff x."""
    return 1.0 - (1.0 - w) * k


def artifact_name(cfg):
    """Filename encoding every dial that changes the trajectory.

    Legacy midpoint cells retain their canonical names; nondefault Deffuant
    alpha values receive an explicit ``_a`` token.
    """
    ea = ("eaopen" if cfg["ai_gate_mode"] == "all_open"
          else f"ea{_num(cfg['eps_ai'])}")
    es = ("esopen" if cfg["peer_gate_mode"] == "all_open"
          else f"es{_num(cfg['eps_social'])}")
    alpha = float(cfg.get("deffuant_alpha", 0.5))
    alpha_token = "" if alpha == 0.5 else f"_a{_num(alpha)}"
    return (f"pp_k{_num(cfg['innate_k'])}_w{_num(cfg['w_plat'])}"
            f"_{ea}_{es}_sw{cfg['ab_sweeps']}"
            f"{alpha_token}_s{cfg['seed']}_r{cfg['rounds']}.pt")


def simulate(setup, *, innate_k, w_plat, eps_social, eps_ai, rounds, seed,
             ai_gate_mode="threshold", peer_gate_mode="threshold",
             ab_sweeps=1, gamma=0.0, deffuant_alpha=0.5,
             dtype=torch.float32, gate_on="anchor",
             served_fn=None, require_open_gate=None, accepted_out=None):
    """Population trajectory under a served signal + its matched twin.

    served_fn(x, t) -> the vector the platform serves in round t, given
    the START-OF-ROUND state x. Default None = PERFECT PREDICTION, i.e.
    served = x.clone() exactly, which is what this module is for. The hook
    exists so the offline frozen replay (replay_frozen_offline.py) drives
    the IDENTICAL operator path with a constant served vector instead of
    maintaining a second copy of the dynamics.

    require_open_gate defaults to True under perfect prediction (|m - x|
    is identically 0, so a closed gate would mean the operator is wrong)
    and False otherwise -- a frozen model legitimately has agents outside
    a numeric gate.

    Uses the IMPLEMENTED operators (gp.nested_presocial_update, gp.ab_sweep)
    rather than a re-derived closed form, so an operator change cannot pass
    unnoticed here. Returns (pp_raw, twin_raw, served_raw) as
    [rounds, n] tensors recorded AFTER the peer sweeps, matching the
    op_raw convention of trajectory.pt.

    dtype defaults to float32 -- the precision the LLM runs carry, and the
    precision every production oracle cell is generated in. float64 exists
    for the STRUCTURAL beta_eff identity checks only: the two identities
    are exact algebra, but in float32 the two parameterizations evaluate z
    through different products (0.5*(0.2 i + 0.8 x) + 0.5 x versus
    0.1 i + 0.9 x) and land 1 ulp apart, which the bounded-confidence peer
    gate then amplifies. See check_perfect_predictor.py.
    """
    innate = setup["innate"].clone().to(dtype)
    adj = (setup["adj"] > 0).to(dtype)
    w_agent = (w_plat * setup["platform_sus"]).clamp(0.0, 1.0).to(dtype)

    # both streams seeded exactly as the runner seeds ab_gen / ab_gen_cf,
    # so the oracle and its own twin see the identical peer-pair pattern
    gen_pp = torch.Generator().manual_seed(seed + PEER_SEED_OFFSET)
    gen_cf = torch.Generator().manual_seed(seed + PEER_SEED_OFFSET)

    perfect = served_fn is None
    if perfect:
        def served_fn(x, t):                     # noqa: F811
            return x.clone()
    if require_open_gate is None:
        require_open_gate = perfect

    x_pp = innate.clone()
    x_cf = innate.clone()
    pp_raw, twin_raw, served_raw = [], [], []
    for t in range(rounds):
        # under perfect prediction m(t) = x(t) EXACTLY -- served is a copy
        # of the START-OF-ROUND state, taken before the operator touches
        # anything
        served = served_fn(x_pp, t).to(dtype)
        served_raw.append(served.clone())
        # gate_on is passed EXPLICITLY (2026-08-22). It used to be omitted,
        # which meant this oracle silently inherited gp's default -- and
        # that default flipped from "x0" to "anchor" when the AI-gate
        # anchor fix landed, so the artifact went on recording the v1
        # marker while executing the v2 operator. Numerically inert on
        # this surface (both gates all_open => ai_gate returns an
        # all-ones mask at _gated_pop.py:205-206 before the reference is
        # ever read at :209), but the provenance was wrong, and an oracle
        # that misreports which operator it ran is worse than useless.
        # Named here so a future default change cannot move it again.
        # gate_on is now a PARAMETER (2026-08-29) rather than a literal:
        # replaying a PRE-CORRECTION trajectory needs gate_on="x0", and
        # forcing "anchor" onto a v1 source silently replays different
        # dynamics. It is inert wherever the AI gate is all_open (the
        # mask is all-ones before the reference is read) and wherever
        # k = 0 (x' == x0), which is why this went unnoticed. The default
        # stays "anchor", so every existing caller is unchanged.
        x_pp, gate = gp.nested_presocial_update(
            x_pp, served, innate, innate_k, w_agent, eps_ai,
            gate_mode=ai_gate_mode, gate_on=gate_on)
        if require_open_gate and not bool(gate.all()):
            raise AssertionError(
                "perfect prediction must open every AI gate (|m - x| = 0); "
                f"{int((~gate).sum())} agents rejected at eps_ai="
                f"{eps_ai!r}, mode={ai_gate_mode!r}")
        acc = 0
        for _ in range(ab_sweeps):
            acc += gp.ab_sweep(x_pp, adj, eps_social, gamma, gen=gen_pp,
                               gate_mode=peer_gate_mode,
                               alpha=deffuant_alpha)
        if accepted_out is not None:
            accepted_out.append(int(acc))
        pp_raw.append(x_pp.detach().clone())
        # matched no-platform twin, exactly the runner's ab_x_cf branch:
        # human component only, then the SAME peer dynamics
        x_cf = innate_k * innate + (1.0 - innate_k) * x_cf
        for _ in range(ab_sweeps):
            gp.ab_sweep(x_cf, adj, eps_social, gamma, gen=gen_cf,
                        gate_mode=peer_gate_mode,
                        alpha=deffuant_alpha)
        twin_raw.append(x_cf.detach().clone())
    return (torch.stack(pp_raw), torch.stack(twin_raw),
            torch.stack(served_raw))


# The ONE mapping from the executed gate reference to the operator name
# recorded in the artifact. gate_on became a PARAMETER on 2026-08-29 (so
# a pre-correction trajectory can be replayed exactly), which means a
# hardcoded marker here would label an x0 run as anch2 -- precisely the
# mislabelling test_section3_checker was written to catch. Derive it.
OPERATOR_FOR_GATE_ON = {
    "anchor": "nested_ai_anchored_then_social_v2",
    "x0": "nested_ai_then_social_v1",
}


def operator_marker(gate_on: str) -> str:
    """The population_update string for the reference actually executed."""
    try:
        return OPERATOR_FOR_GATE_ON[gate_on]
    except KeyError:
        raise ValueError(
            f"gate_on={gate_on!r} has no operator marker; refusing to "
            f"label an artifact with an operator it did not run") from None


def build_config(args, setup, gate_on):
    """gate_on is REQUIRED, not defaulted: a default here is how a
    run silently labels itself with an operator it did not execute."""
    return {
        "platform": "perfect_prediction",
        # DERIVED from what simulate() actually executed, never asserted.
        # Artifacts written before 2026-08-22 carry the v1 string and were
        # produced by the same operator on this surface -- both gates
        # all_open makes v1 and v2 numerically identical -- so old
        # artifacts stay valid and are NOT rewritten. A consumer that needs
        # to tell them apart should validate the expected marker PER
        # SOURCE rather than treat the two strings as interchangeable.
        "population_update": operator_marker(gate_on),
        "gate_on": gate_on,
        "innate_k": float(args.innate_k),
        "w_plat": float(args.w_plat),
        "beta_eff": beta_eff(float(args.innate_k), float(args.w_plat)),
        "eps_social": float(args.eps_social),
        "eps_ai": float(args.eps_ai),
        "ai_gate_mode": args.ai_gate_mode,
        "peer_gate_mode": args.peer_gate_mode,
        "ab_sweeps": int(args.sweeps),
        "deffuant_alpha": float(args.alpha),
        "gamma_bias": float(args.gamma),
        "rounds": int(args.rounds),
        "seed": int(args.seed),
        "peer_seed": int(args.seed) + PEER_SEED_OFFSET,
        "peer_rng_device": "cpu",
        "dataset": "movielens", "ml_target": "Action",
        "n_agents": int(setup["n"]),
        "innate_sha256": _sha(setup["innate"]),
        "adj_sha256": _sha((setup["adj"] > 0).to(torch.uint8)),
        "platform_sus_sha256": _sha(setup["platform_sus"]),
        "sim_version": "sim_perfect_predictor/2026-08-24-alpha",
    }


def main():
    ap = argparse.ArgumentParser(
        description="perfect-prediction (m(t) = x(t)) oracle, CPU only")
    ap.add_argument("--innate-k", type=float, default=None,
                    help="innate anchor k (the runner's INNATE_LAMBDA)")
    ap.add_argument("--lam", type=float, default=None,
                    help="DEPRECATED alias for --innate-k. The name 'lam' "
                         "is ambiguous in this project (it also spells the "
                         "KL coefficient); kept only so older invocations "
                         "do not silently mean something else.")
    ap.add_argument("--w-plat", type=float, required=True,
                    help="platform susceptibility W")
    ap.add_argument("--eps-social", type=float, required=True,
                    help="Deffuant confidence width (inert under "
                         "--peer-gate-mode all_open, still recorded)")
    ap.add_argument("--eps-ai", type=float, default=1.0,
                    help="AI gate width; any positive value opens it here "
                         "since |m - x| = 0")
    ap.add_argument("--ai-gate-mode", default="threshold",
                    choices=("threshold", "all_open"))
    ap.add_argument("--peer-gate-mode", default="threshold",
                    choices=("threshold", "all_open"))
    ap.add_argument("--sweeps", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="symmetric Deffuant compromise rate in [0, 0.5]")
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="gamma_bias; the project pins it at 0")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true",
                    help="overwrite a COMPLETED artifact (refused "
                         "otherwise)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.innate_k is None and args.lam is None:
        ap.error("--innate-k is required (--lam is a deprecated alias)")
    if args.innate_k is not None and args.lam is not None \
            and args.innate_k != args.lam:
        ap.error(f"--innate-k {args.innate_k!r} and the deprecated --lam "
                 f"{args.lam!r} disagree; pass only --innate-k")
    if args.innate_k is None:
        print("[pp] WARNING: --lam is DEPRECATED and ambiguous in this "
              "project; use --innate-k", file=sys.stderr)
        args.innate_k = args.lam
    if args.eps_ai <= 0 and args.ai_gate_mode != "all_open":
        ap.error("a zero-width AI gate never opens; pass --eps-ai > 0 or "
                 "--ai-gate-mode all_open")
    if args.peer_gate_mode == "all_open" and args.eps_social <= 0:
        ap.error("--peer-gate-mode all_open with --eps-social 0 is "
                 "contradictory (eps_social=0 is the NO-PEER condition)")
    if not 0.0 <= args.alpha <= 0.5:
        ap.error("--alpha must lie in [0, 0.5]")

    setup = extract_loader()(
        REPO / "experiments/data/movielens/ml-100k", "Action")
    # PP always serves the corrected operator; the marker is derived
    # from this same value rather than restated as a constant.
    pp_gate_on = "anchor"
    cfg = build_config(args, setup, gate_on=pp_gate_on)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / artifact_name(cfg)
    if out.exists() and not args.force:
        try:
            prev = torch.load(out, map_location="cpu", weights_only=False)
            done = int(prev["config"]["rounds"])
        except Exception:
            done = -1
        if done == cfg["rounds"]:
            print(f"[pp] {out.name} already complete ({done} rounds) -- "
                  f"refusing to overwrite. Pass --force to regenerate.",
                  file=sys.stderr)
            return 0
        print(f"[pp] {out.name} exists but is not a completed "
              f"{cfg['rounds']}-round artifact (found {done}); pass "
              f"--force to replace it.", file=sys.stderr)
        return 1

    pp_raw, twin_raw, served_raw = simulate(
        setup, innate_k=cfg["innate_k"], w_plat=cfg["w_plat"],
        eps_social=cfg["eps_social"], eps_ai=cfg["eps_ai"],
        rounds=cfg["rounds"], seed=cfg["seed"],
        ai_gate_mode=cfg["ai_gate_mode"],
        peer_gate_mode=cfg["peer_gate_mode"],
        ab_sweeps=cfg["ab_sweeps"], gamma=cfg["gamma_bias"],
        deffuant_alpha=cfg["deffuant_alpha"])

    payload = {"config": cfg, "op_raw": pp_raw, "twin_raw": twin_raw,
               "pred_raw": served_raw, "innate": setup["innate"].clone()}
    torch.save(payload, out)

    if not args.quiet:
        fin, ini = pp_raw[-1], setup["innate"]
        print(f"[pp] k={cfg['innate_k']:g} W={cfg['w_plat']:g} "
              f"beta_eff={cfg['beta_eff']:.4g} "
              f"ai={cfg['ai_gate_mode']} peer={cfg['peer_gate_mode']} "
              f"alpha={cfg['deffuant_alpha']:g} "
              f"eps_social={cfg['eps_social']:g} rounds={cfg['rounds']} "
              f"seed={cfg['seed']}")
        print(f"[pp]   innate mean {float(ini.mean()):.6f} "
              f"sd {float(ini.std()):.6f}")
        print(f"[pp]   final  mean {float(fin.mean()):.6f} "
              f"sd {float(fin.std()):.6f} "
              f"range {float(fin.max() - fin.min()):.6e}")
        print(f"[pp]   twin   mean {float(twin_raw[-1].mean()):.6f} "
              f"sd {float(twin_raw[-1].std()):.6f}")
        print(f"[pp] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
