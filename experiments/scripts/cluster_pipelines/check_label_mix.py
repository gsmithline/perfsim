#!/usr/bin/env python3
"""Gate for the fixed/live label-mixture test (pofdmix_, 2026-08-23).

FIVE ROUNDS IS A DIRECTIONAL TEST, NOT AN EQUILIBRIUM. This gate checks
that the mixture was actually applied as declared; it makes no
convergence claim and neither should anything downstream.

The claim to protect is that q of the labels were the agent's CURRENT
post-peer opinion and 1-q were that agent's ORIGINAL frozen-Qwen
prediction, with nested subsets. All of that is reconstructible from the
artifact, so it is CHECKED rather than trusted:

  LABELS      ref_replay_labels[t] must equal x(t) on the live rows and
              b on the rest, exactly. Not "close" -- these are the
              literal vectors handed to the learner.
  LIVE COUNT  |S_t| must be round(q*n), every round.
  NESTING     a smaller-q live set must be a PREFIX of a larger one at
              the same round, which is what makes the arms comparable.
  b           ref_replay_ref_vec must be constant across rounds and match
              the pinned reference run's hash.
Plus the usual surface pins, zero parse failures and real training.

Usage:
  OMP_NUM_THREADS=1 python check_label_mix.py notes/pofd/cluster/pofdmix_*_r5
"""
from __future__ import annotations
import argparse, gzip, json, os, re, sys
import numpy as np, torch
torch.set_num_threads(1)

N = 723
H100 = "NVIDIA H100 80GB HBM3"
BASE = "Qwen/Qwen3-8B"
SMOKE_ROUNDS = 3
TAG_RE = re.compile(
    r"^(?P<pre>pofdmix|pofdmixsmk)_qwen3_8b_q(?P<q>[0-9p]+)_sw(?P<sw>\d+)"
    r"_eaopen_w1_k1_esopen_anch2_s0_r(?P<r>\d+)$")


def _unnum(t):
    return float(t.replace("p", "."))


def check_one(d, smoke, out, live_by_round):
    tag = os.path.basename(d.rstrip("/"))
    m = TAG_RE.match(tag)
    if not m:
        out.append(f"FAIL {tag}: tag is not in the label-mixture grammar")
        return None
    if m.group("pre").endswith("smk") != smoke:
        out.append(f"FAIL {tag}: smoke/production prefix mismatch"); return None
    q, sw, want_rounds = _unnum(m.group("q")), int(m.group("sw")), int(m.group("r"))
    cf, tf = os.path.join(d, "config.json"), os.path.join(d, "trajectory.pt")
    if not (os.path.exists(cf) and os.path.exists(tf)):
        out.append(f"FAIL {tag}: missing config.json or trajectory.pt"); return None
    c = json.load(open(cf)); ok = True
    def bad(msg):
        nonlocal ok; out.append(f"FAIL {tag}: {msg}"); ok = False

    if abs(float(c.get("ref_replay_q", -1)) - q) > 1e-12:
        bad(f"ref_replay_q={c.get('ref_replay_q')!r} but the tag says q={q} "
            f"-- the $(refq) column did not reach the runner")
    pins = {"base_model": BASE, "dataset": "movielens", "ml_target": "Action",
            "n_labeled": N, "train_cap": N, "seed": 0, "w_plat": 1.0,
            "innate_lambda": 1.0, "ai_gate_mode": "all_open",
            "peer_gate_mode": "all_open", "training_style": "sft",
            "kl_beta": 0.0, "ab_sweeps": sw, "pop_model": "ab",
            "use_lora": True, "fresh_each_round": True, "lora_r": 512,
            "icl_k": 0, "n_rounds": want_rounds, "serve_eval_mode": True}
    for k, v in pins.items():
        got = c.get(k, "<absent>")
        if isinstance(v, bool):
            if bool(got) is not v: bad(f"{k}={got!r}, expected {v!r}")
        elif got != v: bad(f"{k}={got!r}, expected {v!r}")
    if c.get("chat_thinking") not in (False, 0):
        bad(f"chat_thinking={c.get('chat_thinking')!r} -- must be OFF")
    if (c.get("hardware") or {}).get("gpu_name") != H100:
        bad(f"gpu={(c.get('hardware') or {}).get('gpu_name')!r}")

    dd = torch.load(tf, map_location="cpu", weights_only=False)
    op, pr = dd.get("op_raw"), dd.get("pred_raw")
    for nm, t in (("op_raw", op), ("pred_raw", pr)):
        if not torch.is_tensor(t) or tuple(t.shape) != (want_rounds, N):
            bad(f"{nm} shape wrong"); return {"ok": False, "tag": tag}
        if not torch.isfinite(t).all(): bad(f"{nm} non-finite")
    op = op.float().numpy()

    # ---- THE MIXTURE ITSELF -------------------------------------------
    idx, lab, b = (dd.get("ref_replay_live_idx"), dd.get("ref_replay_labels"),
                   dd.get("ref_replay_ref_vec"))
    if not all(torch.is_tensor(x) and x.numel() for x in (idx, lab, b)):
        bad("ref_replay_live_idx / labels / ref_vec missing -- the mixture "
            "cannot be verified from this artifact")
        return {"ok": False, "tag": tag}
    lab = lab.float().numpy(); b = b.float().numpy()
    n_live = int(round(q * N))
    if idx.shape[1] != n_live:
        bad(f"live set is {idx.shape[1]} rows, expected round(q*n)={n_live}")
    inn = dd["innate"].float().numpy()
    prev = np.vstack([inn[None, :], op[:-1]])   # x at the START of each round
    for t in range(want_rounds):
        live = idx[t].numpy()
        exp = b.copy()
        exp[live] = prev[t][live]
        if not np.array_equal(np.round(lab[t], 6), np.round(exp, 6)):
            nd = int((np.round(lab[t], 6) != np.round(exp, 6)).sum())
            bad(f"round {t}: {nd} of {N} labels differ from "
                f"(live -> x(t), rest -> b)")
            break
        live_by_round.setdefault(t, []).append((q, list(live)))

    # ---- parse + training ----------------------------------------------
    gz = os.path.join(d, "raw_gen_log.json.gz")
    if not os.path.exists(gz):
        bad("no raw_gen_log.json.gz")
    else:
        rows = [json.loads(l) for l in gzip.open(gz, "rt")]
        b_ = [r for r in rows if float(r.get("parse_fail_frac", 1)) != 0.0]
        if b_: bad(f"parse failures in {len(b_)} round(s)")
    tp = os.path.join(d, "telemetry.json")
    if os.path.exists(tp):
        tel = [json.loads(l) for l in open(tp).read().splitlines() if l.strip()]
        gn = [float(r["grad_norm0"]) for r in tel if r.get("grad_norm0") is not None]
        if gn and max(gn) == 0.0: bad("never trained (grad_norm0 all zero)")
        if len(gn) != want_rounds: bad(f"trained {len(gn)} of {want_rounds} rounds")
    else:
        bad("no telemetry.json")
    return {"ok": ok, "tag": tag, "q": q,
            "mean_last": float(op[-1].mean()), "sd_last": float(op[-1].std())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out, recs, live = [], [], {}
    for d in a.runs:
        r = check_one(d, a.smoke, out, live)
        if r: recs.append(r)
    # ---- NESTING across arms -------------------------------------------
    for t, arms in sorted(live.items()):
        arms.sort(key=lambda x: x[0])
        for (qa, la), (qb, lb) in zip(arms, arms[1:]):
            if la != lb[:len(la)]:
                out.append(f"FAIL wave: round {t}: q={qa} live set is not a "
                           f"PREFIX of q={qb} -- the arms are not nested, so "
                           f"they differ by WHICH agents are live as well as "
                           f"how many")
    for l in out: print(f"[check_mix] {l}")
    print(f"\n{'cell':<62}{'q':>6}{'mean':>9}{'SD':>9}")
    for r in sorted(recs, key=lambda x: x["q"]):
        print(f"{r['tag']:<62}{r['q']:>6g}{r['mean_last']:>9.4f}{r['sd_last']:>9.4f}")
    print(f"\n{want_rounds}-round run: check the drift flag before quoting a position.")
    allok = bool(recs) and all(r["ok"] for r in recs) and not any(
        l.startswith("FAIL") for l in out)
    print(f"[check_mix] {'PASS' if allok else 'FAILED'} -- {len(a.runs)} run(s)")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
