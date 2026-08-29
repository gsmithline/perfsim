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

THE FIDELITY CHECK IS A HARD GATE.  The `real` policy replays the
recorded served sequence, so it must REPRODUCE the source run's own
op_raw.  If it does not, the replay is not the same dynamics and every
counterfactual below is void.  Measured, not assumed.

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
# the Figure-3 corner: beta = gamma = 1, both gates open, S = 100
ENV = dict(innate_k=1.0, w_plat=1.0, eps_social=0.2, eps_ai=1.0,
           ai_gate_mode="all_open", peer_gate_mode="all_open",
           ab_sweeps=100, gamma=0.0, deffuant_alpha=0.5)
# the replay reproduces a real run to ~5e-4 (float32 + bounded
# confidence); anything above this is a different dynamics, not noise
FIDELITY_TOL = 5e-3
N_PERM = 5

SOURCES = [
    ("lam0_r512", "run", "pofdps_qwen3_8b_sft_sw100_eaopen_w1_k1_esopen"
                         "_anch2_s0_r60"),
    ("lam2_r512", "run", "pofdlam_qwen3_8b_fwdlam2_sw100_eaopen_w1_k1"
                         "_esopen_anch2_s0_r30"),
    ("lam2_r16", "run", "pofdf3_qwen3_8b_fwdlam2_sw100_eaopen_w1_k1"
                        "_esopen_anch2_rank16_s0_r30"),
    ("laminf_frozen", "replay", "frz_k1_w1_eaopen_esopen_sw100_s0_r30.pt"),
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
    label, policy, pred_np, perm, k = task
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
    op, twin, sv = PP.simulate(setup, rounds=ROUNDS, seed=SEED,
                               served_fn=served, **ENV)
    return {"label": label, "policy": policy, "perm_k": k,
            "op": op.numpy(), "served_sha": _sha(sv.numpy())}


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
        sources[label] = {"tag": name, "kind": kind,
                          "source_op": op.numpy(),
                          "pred": pred.numpy(),
                          "pred_sha": _sha(pred.numpy()),
                          "n_distinct_final": int(
                              np.unique(pred.numpy()[-1]).size),
                          "lora_r": cfg.get("lora_r"),
                          "kl_beta": cfg.get("kl_beta")}
        pn = pred.numpy()
        tasks.append((label, "real", pn, None, -1))
        tasks.append((label, "constant", pn, None, -1))
        for k in range(args.n_perm):
            tasks.append((label, "shuffled", pn,
                          rng.permutation(pn.shape[1]).astype(np.int64), k))

    print(f"[mm] {len(tasks)} replays over {args.workers} worker(s) "
          f"({len(SOURCES)} sources x (real + constant + "
          f"{args.n_perm} shuffles))", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_one, tasks))

    by = {}
    for r in results:
        by.setdefault((r["label"], r["policy"]), []).append(r)

    errs, rows = [], []
    for label, _kind, _name in SOURCES:
        src = sources[label]
        s_op = src["source_op"]
        real = by[(label, "real")][0]
        # HARD GATE: the real replay must reproduce the source run
        fid = float(np.abs(real["op"] - s_op).max())
        if fid > FIDELITY_TOL:
            errs.append(f"{label}: the `real` replay does not reproduce "
                        f"the source op_raw (max |diff| {fid:.3e} > "
                        f"{FIDELITY_TOL:.0e}) -- the counterfactuals are "
                        f"not the same dynamics and mean nothing")
        base = summarize(real["op"])
        for policy in ("real", "constant", "shuffled"):
            for r in by[(label, policy)]:
                s = summarize(r["op"])
                rows.append({
                    "source": label, "tag": src["tag"],
                    "kl_beta": src["kl_beta"], "lora_r": src["lora_r"],
                    "served_distinct_final": src["n_distinct_final"],
                    "policy": policy, "perm_k": r["perm_k"],
                    "fidelity_max_abs_vs_source": (
                        fid if policy == "real" else ""),
                    **s,
                    "d_final_mean_vs_real": s["final_mean"]
                    - base["final_mean"],
                    "w1_final_vs_real": w1(r["op"][-1], real["op"][-1]),
                    "paired_mae_final_vs_real": float(np.abs(
                        r["op"][-1] - real["op"][-1]).mean()),
                })

    import csv
    with (out / "mean_matched_cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    summary = []
    for label, _k, _n in SOURCES:
        src = sources[label]
        real = summarize(by[(label, "real")][0]["op"])
        const = summarize(by[(label, "constant")][0]["op"])
        sh = [summarize(r["op"]) for r in by[(label, "shuffled")]]
        sh_w1 = [w1(r["op"][-1], by[(label, "real")][0]["op"][-1])
                 for r in by[(label, "shuffled")]]
        c_w1 = w1(by[(label, "constant")][0]["op"][-1],
                  by[(label, "real")][0]["op"][-1])
        summary.append({
            "source": label, "tag": src["tag"],
            "kl_beta": src["kl_beta"], "lora_r": src["lora_r"],
            "served_distinct_final": src["n_distinct_final"],
            "real_final_mean": real["final_mean"],
            "constant_final_mean": const["final_mean"],
            "shuffled_final_mean_mean": float(np.mean(
                [s["final_mean"] for s in sh])),
            "shuffled_final_mean_min": float(np.min(
                [s["final_mean"] for s in sh])),
            "shuffled_final_mean_max": float(np.max(
                [s["final_mean"] for s in sh])),
            "d_mean_constant_minus_real": const["final_mean"]
            - real["final_mean"],
            "d_mean_shuffled_minus_real": float(np.mean(
                [s["final_mean"] for s in sh])) - real["final_mean"],
            "w1_constant_vs_real": c_w1,
            "w1_shuffled_vs_real_mean": float(np.mean(sh_w1)),
            "w1_shuffled_vs_real_min": float(np.min(sh_w1)),
            "w1_shuffled_vs_real_max": float(np.max(sh_w1)),
            "real_final_sd": real["final_sd"],
            "constant_final_sd": const["final_sd"],
            "shuffled_final_sd_mean": float(np.mean(
                [s["final_sd"] for s in sh])),
        })
    with (out / "mean_matched.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        for r in summary:
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
        "environment": {**{k: v for k, v in ENV.items()},
                        "rounds": ROUNDS, "seed": SEED},
        "fidelity_gate": {"tol": FIDELITY_TOL,
                          "meaning": "the `real` replay must reproduce "
                                     "the source run's own op_raw"},
        "gate": {"errors": errs, "pass": not errs},
        "by_source": summary,
    }, indent=2))

    if errs:
        print("[mm] GATE FAIL:")
        for e in errs:
            print("   -", e)
    else:
        print(f"[mm] GATE PASS: every `real` replay reproduces its source "
              f"run to < {FIDELITY_TOL:.0e}")
    hdr = (f"{'source':<15}{'card':>5}{'real':>9}{'constant':>10}"
           f"{'shuffled':>10}{'d_const':>9}{'d_shuf':>9}"
           f"{'W1_const':>10}{'W1_shuf':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for s in summary:
        print(f"{s['source']:<15}{s['served_distinct_final']:>5}"
              f"{s['real_final_mean']:>9.4f}{s['constant_final_mean']:>10.4f}"
              f"{s['shuffled_final_mean_mean']:>10.4f}"
              f"{s['d_mean_constant_minus_real']:>+9.4f}"
              f"{s['d_mean_shuffled_minus_real']:>+9.4f}"
              f"{s['w1_constant_vs_real']:>10.4f}"
              f"{s['w1_shuffled_vs_real_mean']:>10.4f}")
    print(f"\n[mm] wrote {out}/mean_matched.{{csv,json}} + _cells.csv")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
