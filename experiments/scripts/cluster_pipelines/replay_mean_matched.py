#!/usr/bin/env python3
"""MEAN-MATCHED PREDICTION-MAP CONTROLS (CPU replay, no new GPU jobs).

THE OBJECTION THIS ANSWERS.  A served prediction map moves the
population.  But how much of that is the map, and how much is just its
SCALAR MEAN?  If serving the same mean to everybody reproduced the
outcome, the whole effect would be mean arithmetic and nothing about
WHICH agent is told WHAT would matter.

THE THREE POLICIES.  Each drives the IDENTICAL operator path
(sim_perfect_predictor.simulate via its served_fn hook -- there is no
second copy of the dynamics anywhere), on the same innate vector, the
same graph, the same seed and the same peer RNG stream:

  real       served[t] = the source run's recorded pred_raw[t]
  constant   served[t] = mean(pred_raw[t]) broadcast to all agents.
             SAME scalar mean every round, ZERO cross-agent structure.
  shuffled   served[t] = pred_raw[t][perm] for ONE fixed permutation,
             reused in every round.  Same mean AND the same multiset of
             values every round -- only the assignment of value to agent
             is destroyed.  Several permutations are drawn and reported
             as a spread, because one draw is one draw.

Reading the two contrasts:
  real vs constant   what the map's cross-agent SHAPE buys over its mean
  real vs shuffled   what TARGETING buys -- the same values, delivered
                     to the right agents rather than to arbitrary ones

WHY THE COMPARISON IS INTERNAL, NOT AGAINST THE SOURCE RUN.  The
runner draws peer pairs with a generator on the MODEL's device
(run_pokec_gated_lm.py:2457, `torch.Generator(device=ab_device)`), so an
archived GPU run's peer sequence came from a CUDA stream.  A CPU replay
at the same seed draws a DIFFERENT sequence, and no amount of care makes
it match.  That is invisible wherever pair order does not matter -- at
S=100 with an open peer gate the population reaches consensus whatever
the order, which is why the Figure-3 corner reproduces its source to
5e-4 -- and dominant where it does: at S=1 with a bounded-confidence
gate the twin already differs by ~2e-1 in ROUND 0, and the twin never
sees the served map at all.

So the three policies are compared WITH EACH OTHER, under one shared
peer stream, not against the source trajectory.  That contrast is the
valid one: the policies differ only in the served map, everything else
including the RNG being identical by construction.

AND THE EFFECT IS MEASURED AGAINST A NOISE FLOOR.  The `real` policy is
replayed under several PEER SEEDS; the spread it shows is how much the
outcome moves for no reason but the peer draw.  A real-vs-control
difference smaller than that floor is not resolvable and is reported as
such rather than as an effect.  Divergence from the source run is
reported too, as a diagnostic, never as a pass/fail.

WHAT THIS IS NOT.  It is OPEN-LOOP.  The model is never re-queried, so
the controls answer "what would this sequence of maps have produced?",
not "what would the model have done had it seen a different
population".  A closed-loop version needs GPU jobs; this deliberately
does not.  Every artifact records open_loop: true.

  python replay_mean_matched.py --workers 6
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
ML = REPO / "experiments" / "data" / "movielens" / "ml-100k"

_spec = importlib.util.spec_from_file_location(
    "sim_pp", str(HERE / "sim_perfect_predictor.py"))
PP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PP)

ROUNDS = 30
SEED = 0
# THE ENVIRONMENT IS READ FROM EACH SOURCE'S OWN CONFIG, not fixed
# globally. The first pass ran only the Figure-3 corner (W=1, k=1, both
# gates open, S=100) and found the controls indistinguishable from the
# real map -- because 100 open-gate Deffuant sweeps drive the population
# to consensus within the round, so only the served MEAN can survive.
# That is a property of the peer regime, not a general statement, so the
# main environment (W=.5, k=.2, numeric gates, S=1), where dispersion is
# preserved, has to be run too.
FALLBACK_ENV = dict(innate_k=1.0, w_plat=1.0, eps_social=0.2, eps_ai=1.0,
                    ai_gate_mode="all_open", peer_gate_mode="all_open",
                    ab_sweeps=100, gamma=0.0, deffuant_alpha=0.5)


# THE OPERATOR MUST FOLLOW THE SOURCE. sim_perfect_predictor.simulate
# used to hardcode gate_on="anchor" (the corrected v2 reference), so
# replaying a PRE-CORRECTION run reproduced different dynamics: the AI
# gate measured |m - x'| where the source measured |m - x0|. Inert at an
# all_open gate and at k = 0, which is why the Figure-3 corner replayed
# faithfully while the main environment (numeric gate, k = .2) diverged
# by up to .49 and was correctly rejected by the fidelity gate.
GATE_ON = {"nested_ai_then_social_v1": "x0",
           "nested_ai_anchored_then_social_v2": "anchor"}


def env_of(cfg):
    """Replay dials straight from the source run's config."""
    if not cfg:
        return dict(FALLBACK_ENV)
    # two schemas: LLM runs record innate_lambda/eps; the CPU replay
    # artifacts record innate_k/eps_social. Accept either, and fail loudly
    # rather than defaulting -- a wrong dial here silently replays the
    # wrong dynamics.
    def pick(*names):
        for n in names:
            if n in cfg and cfg[n] is not None:
                return float(cfg[n])
        raise KeyError(f"none of {names} in the source config")
    return dict(
        innate_k=pick("innate_lambda", "innate_k"),
        w_plat=pick("w_plat"),
        eps_social=pick("eps", "eps_social"),
        eps_ai=pick("eps_ai"),
        ai_gate_mode=cfg.get("ai_gate_mode") or "threshold",
        peer_gate_mode=cfg.get("peer_gate_mode") or "threshold",
        ab_sweeps=int(cfg["ab_sweeps"]),
        gamma=float(cfg.get("gamma_bias") or 0.0),
        deffuant_alpha=float(cfg.get("deffuant_alpha") or 0.5),
        gate_on=_gate_on(cfg))


def _gate_on(cfg):
    op = cfg.get("population_update")
    if op not in GATE_ON:
        raise KeyError(
            f"population_update={op!r} has no known gate reference; "
            f"refusing to guess -- the wrong one silently replays "
            f"different dynamics")
    return GATE_ON[op]
N_PERM = 5
PEER_SEEDS = (0, 1, 2)      # the noise floor: peer draw alone

SOURCES = [
    ("lam0_r512", "run", "pofdps_qwen3_8b_sft_sw100_eaopen_w1_k1_esopen"
                         "_anch2_s0_r60"),
    ("lam2_r512", "run", "pofdlam_qwen3_8b_fwdlam2_sw100_eaopen_w1_k1"
                         "_esopen_anch2_s0_r30"),
    ("lam2_r16", "run", "pofdf3_qwen3_8b_fwdlam2_sw100_eaopen_w1_k1"
                        "_esopen_anch2_rank16_s0_r30"),
    ("laminf_frozen", "replay", "frz_k1_w1_eaopen_esopen_sw100_s0_r30.pt"),
    # the MAIN environment: W=.5, k=.2, numeric gates, S=1 -- dispersion
    # survives here, so cross-agent structure has room to matter
    ("main_lam0", "run", "pofdws2_qwen7b_b0_ea0p4_w0p5_l0p2_es0p2_s0"
                         "_fresh_data"),
    ("main_lam0p5", "run", "pofdws2f_qwen7b_b0p5_ea0p4_w0p5_l0p2_es0p2_s0"
                           "_fresh_data"),
    ("main_lam1", "run", "pofdws2f_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2_s0"
                         "_fresh_data"),
]


def _sha(t):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(t, dtype=np.float32)).tobytes()
    ).hexdigest()[:16]


def load_source(kind, name, runs_root, frozen_dir):
    p = (Path(frozen_dir) / name if kind == "replay"
         else Path(runs_root) / name / "trajectory.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    return (d["pred_raw"].float()[:ROUNDS], d["op_raw"].float()[:ROUNDS],
            (d.get("config") or {}))


def run_one(task):
    """One replay. Top-level so ProcessPoolExecutor can pickle it."""
    label, policy, pred_np, perm, k, env, pseed = task
    pred = torch.from_numpy(pred_np)
    setup = PP.extract_loader()(ML, "Action")
    if policy == "real":
        def served(x, t):
            return pred[t].clone()
    elif policy == "constant":
        def served(x, t):
            return torch.full_like(pred[t], float(pred[t].mean()))
    elif policy == "shuffled":
        idx = torch.from_numpy(perm)

        def served(x, t):
            return pred[t][idx].clone()
    else:
        raise ValueError(policy)
    op, twin, sv = PP.simulate(setup, rounds=ROUNDS, seed=pseed,
                               served_fn=served, require_open_gate=False,
                               **env)
    return {"label": label, "policy": policy, "perm_k": k,
            "peer_seed": pseed, "op": op.numpy(),
            "served_sha": _sha(sv.numpy())}


def summarize(op):
    m = op.mean(axis=1)
    tail = m[-5:]
    half = 2
    return {"final_mean": float(m[-1]), "final_sd": float(op[-1].std()),
            "late_mean": float(tail.mean()),
            "late_drift": float(tail[-half:].mean() - tail[:half].mean())}


def w1(a, b):
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-root",
                    default=str(REPO / "notes" / "pofd" / "cluster"))
    ap.add_argument("--frozen-dir",
                    default=str(REPO / "notes" / "pofd" / "frozen_replay"))
    ap.add_argument("--out-dir",
                    default=str(REPO / "notes" / "pofd" / "mean_matched"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks, sources = [], {}
    rng = np.random.default_rng(12345)
    for label, kind, name in SOURCES:
        pred, op, cfg = load_source(kind, name, args.runs_root,
                                    args.frozen_dir)
        env = env_of(cfg)
        sources[label] = {"tag": name, "kind": kind, "env": env,
                          "source_op": op.numpy(),
                          "pred": pred.numpy(),
                          "pred_sha": _sha(pred.numpy()),
                          "n_distinct_final": int(
                              np.unique(pred.numpy()[-1]).size),
                          "lora_r": cfg.get("lora_r"),
                          "kl_beta": cfg.get("kl_beta")}
        pn = pred.numpy()
        perms = [rng.permutation(pn.shape[1]).astype(np.int64)
                 for _ in range(args.n_perm)]
        for ps in PEER_SEEDS:
            tasks.append((label, "real", pn, None, -1, env, ps))
            tasks.append((label, "constant", pn, None, -1, env, ps))
            for k, pm in enumerate(perms):
                tasks.append((label, "shuffled", pn, pm, k, env, ps))

    print(f"[mm] {len(tasks)} replays over {args.workers} worker(s) "
          f"({len(SOURCES)} sources x (real + constant + "
          f"{args.n_perm} shuffles))", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_one, tasks))

    by = {}
    for r in results:
        by.setdefault((r["label"], r["policy"]), []).append(r)

    import csv
    errs, rows = [], []
    for label, _kind, _name in SOURCES:
        src = sources[label]
        s_op = src["source_op"]
        # every policy must share the peer stream it is compared against
        for ps in PEER_SEEDS:
            reals = [r for r in by[(label, "real")] if r["peer_seed"] == ps]
            if len(reals) != 1:
                errs.append(f"{label}: {len(reals)} real replays at peer "
                            f"seed {ps} (want exactly 1)")
        for policy in ("real", "constant", "shuffled"):
            for r in by[(label, policy)]:
                m = r["op"].mean(axis=1)
                base = next(x for x in by[(label, "real")]
                            if x["peer_seed"] == r["peer_seed"])
                rows.append({
                    "source": label, "tag": src["tag"],
                    "kl_beta": src["kl_beta"], "lora_r": src["lora_r"],
                    "ab_sweeps": src["env"]["ab_sweeps"],
                    "policy": policy, "perm_k": r["perm_k"],
                    "peer_seed": r["peer_seed"],
                    "final_mean": float(m[-1]),
                    "final_sd": float(r["op"][-1].std()),
                    "late_mean": float(m[-5:].mean()),
                    "d_final_mean_vs_real": float(
                        m[-1] - base["op"][-1].mean()),
                    "w1_final_vs_real": w1(r["op"][-1], base["op"][-1]),
                })
    with (out / "mean_matched_cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    summary = []
    for label, _k, _n in SOURCES:
        src = sources[label]
        realm = {r["peer_seed"]: r for r in by[(label, "real")]}
        # NOISE FLOOR: how far the outcome moves for no reason but the
        # peer draw. Everything below this is unresolvable.
        rm = [float(r["op"][-1].mean()) for r in realm.values()]
        floor_mean = float(np.max(rm) - np.min(rm))
        floor_w1 = float(np.max([w1(a["op"][-1], b["op"][-1])
                                 for a in realm.values()
                                 for b in realm.values()]))
        # EFFECTS: paired within peer seed, then averaged over seeds
        d_const, w_const, d_shuf, w_shuf = [], [], [], []
        for ps in PEER_SEEDS:
            base = realm[ps]
            for r in by[(label, "constant")]:
                if r["peer_seed"] == ps:
                    d_const.append(float(r["op"][-1].mean()
                                         - base["op"][-1].mean()))
                    w_const.append(w1(r["op"][-1], base["op"][-1]))
            for r in by[(label, "shuffled")]:
                if r["peer_seed"] == ps:
                    d_shuf.append(float(r["op"][-1].mean()
                                        - base["op"][-1].mean()))
                    w_shuf.append(w1(r["op"][-1], base["op"][-1]))
        # DIAGNOSTIC ONLY: distance from the source run, which a CPU
        # replay of a GPU-peer run cannot be expected to reproduce
        src_div = float(np.abs(np.mean([r["op"] for r in realm.values()],
                                       axis=0) - s_op).max())
        summary.append({
            "source": label, "tag": src["tag"], "env": src["env"],
            "kl_beta": src["kl_beta"], "lora_r": src["lora_r"],
            "served_distinct_final": src["n_distinct_final"],
            "real_final_mean": float(np.mean(rm)),
            "noise_floor_mean_range": floor_mean,
            "noise_floor_w1_max": floor_w1,
            "d_mean_constant": float(np.mean(d_const)),
            "w1_constant": float(np.mean(w_const)),
            "d_mean_shuffled": float(np.mean(d_shuf)),
            "w1_shuffled": float(np.mean(w_shuf)),
            "constant_resolvable": bool(
                abs(float(np.mean(d_const))) > floor_mean
                or float(np.mean(w_const)) > floor_w1),
            "shuffled_resolvable": bool(
                abs(float(np.mean(d_shuf))) > floor_mean
                or float(np.mean(w_shuf)) > floor_w1),
            "source_divergence_diagnostic": src_div,
        })
    flat = [{**{k: v for k, v in r.items() if k != "env"},
             "w_plat": r["env"]["w_plat"], "innate_k": r["env"]["innate_k"],
             "ab_sweeps": r["env"]["ab_sweeps"],
             "ai_gate_mode": r["env"]["ai_gate_mode"]} for r in summary]
    with (out / "mean_matched.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]))
        w.writeheader()
        for r in flat:
            w.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    (out / "mean_matched.json").write_text(json.dumps({
        "question": "how much of a served map's effect is its scalar "
                    "mean, and how much is which agent gets which value?",
        "policies": {
            "real": "the source run's recorded pred_raw[t]",
            "constant": "mean(pred_raw[t]) broadcast to all agents -- "
                        "same scalar mean, no cross-agent structure",
            "shuffled": f"pred_raw[t][perm] for one fixed permutation, "
                        f"reused every round -- same mean AND same "
                        f"multiset, targeting destroyed; "
                        f"{args.n_perm} permutations",
        },
        "open_loop": True,
        "open_loop_caveat": (
            "the model is never re-queried, so these answer 'what would "
            "this sequence of maps have produced', not 'what would the "
            "model have done facing a different population'. A "
            "closed-loop version needs GPU jobs and is not run here."),
        "operator": "sim_perfect_predictor.simulate via served_fn -- the "
                    "identical path the LLM runs use",
        "environment_per_source": {k: v["env"] for k, v in
                                   sources.items()},
        "rounds": ROUNDS, "seed": SEED,
        "why_not_compared_to_the_source": (
            "the runner draws peer pairs with a generator on the MODEL's "
            "device, so an archived GPU run's peer sequence came from a "
            "CUDA stream; a CPU replay at the same seed draws a different "
            "one and cannot match. The policies are therefore compared "
            "with each other under one shared peer stream. "
            "source_divergence_diagnostic records the gap to the source "
            "run and is a diagnostic, never a pass/fail."),
        "noise_floor": (
            "the real policy replayed under peer seeds "
            f"{list(PEER_SEEDS)}; its spread is movement caused by the "
            "peer draw alone. A real-vs-control difference below it is "
            "NOT resolvable and is reported as such."),
        "gate": {"errors": errs, "pass": not errs},
        "by_source": summary,
    }, indent=2))

    if errs:
        print("[mm] GATE FAIL:")
        for e in errs:
            print("   -", e)
    else:
        print(f"[mm] GATE PASS: one real replay per peer seed "
              f"{list(PEER_SEEDS)}; policies share the peer stream they "
              f"are compared against")
    hdr = (f"{'source':<15}{'S':>4}{'card':>5}{'real':>9}"
           f"{'W1_const':>10}{'W1_shuf':>10}{'floor':>9}"
           f"{'const?':>8}{'shuf?':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for s in summary:
        print(f"{s['source']:<15}{s['env']['ab_sweeps']:>4}"
              f"{s['served_distinct_final']:>5}"
              f"{s['real_final_mean']:>9.4f}"
              f"{s['w1_constant']:>10.4f}{s['w1_shuffled']:>10.4f}"
              f"{s['noise_floor_w1_max']:>9.4f}"
              f"{('YES' if s['constant_resolvable'] else 'no'):>8}"
              f"{('YES' if s['shuffled_resolvable'] else 'no'):>7}")
    print("\n  W1_const / W1_shuf: distance from the real map's outcome, "
          "paired within peer seed.")
    print("  floor: the largest real-vs-real distance across peer seeds "
          "-- movement from the peer draw alone.")
    print("  const?/shuf?: is the effect above that floor, i.e. "
          "resolvable at all.")
    print(f"\n[mm] wrote {out}/mean_matched.{{csv,json}} + _cells.csv")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
