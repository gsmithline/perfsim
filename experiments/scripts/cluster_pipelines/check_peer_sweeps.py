#!/usr/bin/env python3
"""Gate for the peer-sweep-strength wave (pofdps_, 2026-08-23).

Run LOCALLY or with threads pinned to 1; never as multithreaded work on
a login node.

The wave holds the Section 3 Qwen3 surface fixed and varies ONLY
AB_SWEEPS, so the gate is built around the ways that variation could be
fake: a sweeps column that never reached the runner, a horizon that is
not what the tag claims, or a "post-peer" state that is not post-peer.

NEAR-ZERO SD IS PERMITTED AND REPORTED, NEVER FAILED. The hypothesis is
that more sweeps lower the plateau; a plateau at ~0 is that hypothesis
coming true, not a broken run.

Usage:
  OMP_NUM_THREADS=1 python check_peer_sweeps.py --smoke runs/.../pofdpssmk_*_r3
  OMP_NUM_THREADS=1 python check_peer_sweeps.py notes/pofd/cluster/pofdps_*_r60
"""
from __future__ import annotations
import argparse, gzip, json, math, os, re, sys
import numpy as np, torch
torch.set_num_threads(1)

N = 723
H100 = "NVIDIA H100 80GB HBM3"
BASE = "Qwen/Qwen3-8B"
# horizon is read off the TAG (pofdps_ ran 60, pofdmem_ runs 30)
SMOKE_ROUNDS = 3
# Accepts BOTH families. pofdps_ pins W=1, k=1 and varies only the sweep
# count; pofdmem_ varies W, k and S. Parsing them with one expression
# keeps a single gate rather than two that can drift apart, and W/k are
# read FROM THE TAG and checked against the config rather than assumed.
TAG_RE = re.compile(
    r"^(?P<pre>pofdps|pofdpssmk|pofdmem|pofdmemsmk|pofdlam|pofdlamsmk)"
    r"_qwen3_8b_(?P<arm>sft|fwdlam[0-9p]+)_sw(?P<sw>\d+)_eaopen_"
    r"w(?P<w>[0-9p]+)_k(?P<k>[0-9p]+)_esopen_anch2_s0_r(?P<r>\d+)$")


def _unnum(tok):
    """tag number grammar back to a float: 0p5 -> 0.5, 1 -> 1.0"""
    return float(tok.replace("p", "."))
def arm_lambda(arm):
    """lambda from the arm token. Parsed rather than table-looked-up, so a
    new dose (fwdlam0p25, fwdlam2, ...) does not need the gate edited --
    an unlisted arm used to fail as a GRAMMAR error, which reads like a
    malformed tag rather than a missing table entry."""
    return 0.0 if arm == "sft" else _unnum(arm[len("fwdlam"):])


def check_one(d, smoke, out):
    tag = os.path.basename(d.rstrip("/"))
    m = TAG_RE.match(tag)
    if not m:
        out.append(f"FAIL {tag}: tag is not in the peer-sweep grammar"); return None
    if m.group("pre").endswith("smk") != smoke:
        out.append(f"FAIL {tag}: smoke/production prefix mismatch"); return None
    arm, sw = m.group("arm"), int(m.group("sw"))
    want_rounds = int(m.group("r"))
    if smoke and want_rounds != SMOKE_ROUNDS:
        out.append(f"FAIL {tag}: smoke horizon {want_rounds} != {SMOKE_ROUNDS}")
        return None
    cf, tf = os.path.join(d, "config.json"), os.path.join(d, "trajectory.pt")
    if not (os.path.exists(cf) and os.path.exists(tf)):
        out.append(f"FAIL {tag}: missing config.json or trajectory.pt"); return None
    c = json.load(open(cf)); ok = True
    def bad(msg):
        nonlocal ok; out.append(f"FAIL {tag}: {msg}"); ok = False

    # --- the varied dial, and that it actually reached the runner -----
    if int(c.get("ab_sweeps", -1)) != sw:
        bad(f"ab_sweeps={c.get('ab_sweeps')!r} but the tag says sw{sw} -- the "
            f"$(sweeps) column did not reach the runner")
    if c.get("pop_model") != "ab":
        bad(f"pop_model={c.get('pop_model')!r}; AB_SWEEPS>1 requires ab")
    # --- everything that must be HELD fixed ---------------------------
    tag_w, tag_k = _unnum(m.group("w")), _unnum(m.group("k"))
    pins = {"base_model": BASE, "dataset": "movielens", "ml_target": "Action",
            "n_labeled": N, "seed": 0, "w_plat": tag_w, "innate_lambda": tag_k,
            "ai_gate_mode": "all_open", "peer_gate_mode": "all_open",
            "kl_direction": "forward", "kl_ref_adapter": "",
            "anchor_mode": "fixed", "use_lora": True, "fresh_each_round": True,
            "lora_r": 512, "sft_epochs": 1, "train_cap": N, "icl_k": 0,
            "icl_days": 0, "n_rounds": want_rounds, "serve_eval_mode": True}
    for k, v in pins.items():
        got = c.get(k, "<absent>")
        if isinstance(v, bool):
            if bool(got) is not v: bad(f"{k}={got!r}, expected {v!r}")
        elif got != v: bad(f"{k}={got!r}, expected {v!r}")
    if c.get("chat_thinking") not in (False, 0):
        bad(f"chat_thinking={c.get('chat_thinking')!r} -- Qwen3 thinking must be OFF")
    if float(c.get("kl_beta", -1)) != arm_lambda(arm):
        bad(f"kl_beta={c.get('kl_beta')!r}, expected {arm_lambda(arm)}")
    if (c.get("hardware") or {}).get("gpu_name") != H100:
        bad(f"gpu={(c.get('hardware') or {}).get('gpu_name')!r}")

    dd = torch.load(tf, map_location="cpu", weights_only=False)
    op, pr = dd.get("op_raw"), dd.get("pred_raw")
    for nm, t in (("op_raw", op), ("pred_raw", pr)):
        if not torch.is_tensor(t) or tuple(t.shape) != (want_rounds, N):
            bad(f"{nm} shape {tuple(t.shape) if torch.is_tensor(t) else None} "
                f"!= {(want_rounds, N)}"); return {"ok": False, "tag": tag}
        if not torch.isfinite(t).all(): bad(f"{nm} has non-finite values")
    op, pr = op.float().numpy(), pr.float().numpy()

    # --- zero parse failures ------------------------------------------
    gz = os.path.join(d, "raw_gen_log.json.gz")
    if not os.path.exists(gz):
        bad("no raw_gen_log.json.gz -- parse rate not establishable")
    else:
        rows = [json.loads(l) for l in gzip.open(gz, "rt")]
        b = [r for r in rows if float(r.get("parse_fail_frac", 1)) != 0.0]
        if b: bad(f"parse failures in {len(b)} round(s), e.g. round {b[0].get('round')}")
        s = [r for r in rows if len(r.get("parsed") or []) != N]
        if s: bad(f"round {s[0].get('round')} parsed {len(s[0]['parsed'])} of {N}")

    # --- fresh retraining every round ---------------------------------
    tp = os.path.join(d, "telemetry.json")
    if not os.path.exists(tp):
        bad("no telemetry.json -- cannot confirm retraining")
    else:
        tel = [json.loads(l) for l in open(tp).read().splitlines() if l.strip()]
        gn = [float(r["grad_norm0"]) for r in tel if r.get("grad_norm0") is not None]
        if len(gn) != want_rounds: bad(f"grad_norm0 for {len(gn)} of {want_rounds} rounds")
        if gn and max(gn) == 0.0: bad("grad_norm0 zero in every round -- never trained")
        if arm_lambda(arm) > 0:
            kg = [float(r["grad_kl_norm0"]) for r in tel
                  if r.get("grad_kl_norm0") is not None]
            if len(kg) > 1 and max(kg[1:]) <= 0:
                bad("anchor gradient zero in every round after round 0")

    # --- the peer block must weakly CONTRACT the PRE-PEER SD ----------
    # A peer sweep is an averaging step, so it cannot INCREASE dispersion
    # within the round. The quantity it acts on is the PRE-PEER
    # population, which equals the served vector ONLY at W = 1. At W < 1
    # agents partly keep their own state, so the pre-peer population is
    # the blend the runner forms before the sweeps:
    #     h      = k*innate + (1-k)*x_prev          (human component)
    #     x_pre  = (1-W)*h + W*m                    (gate open => no mask)
    # Comparing the served vector to the post-peer state at W < 1 compares
    # two different objects and reports expansions that never happened --
    # which is exactly what this check did on its first run.
    inn = dd.get("innate")
    inn = inn.float().numpy() if torch.is_tensor(inn) else None
    if inn is not None and tag_w < 1.0:
        prev = np.vstack([inn[None, :], op[:-1]])
        h = tag_k * inn[None, :] + (1.0 - tag_k) * prev
        x_pre = (1.0 - tag_w) * h + tag_w * pr
    else:
        x_pre = pr
    pre, post = x_pre.std(axis=1), op.std(axis=1)
    viol = [(i, float(pre[i]), float(post[i]))
            for i in range(want_rounds) if post[i] > pre[i] + 1e-6]
    if viol:
        bad(f"peer block EXPANDED served SD in {len(viol)} round(s), e.g. round "
            f"{viol[0][0]}: served {viol[0][1]:.5f} -> post-peer {viol[0][2]:.5f}. "
            f"A sweep is an averaging step; it cannot add dispersion.")
    ratio = float(np.mean(post / np.maximum(pre, 1e-12)))
    return {"ok": ok, "tag": tag, "arm": arm, "sweeps": sw, "w": tag_w,
            "k": tag_k, "lam": arm_lambda(arm), "post_sd_last": float(post[-1]),
            "pre_sd_last": float(pre[-1]), "contract": ratio,
            "n_viol": len(viol)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out, recs = [], []
    for d in a.runs:
        r = check_one(d, a.smoke, out)
        if r: recs.append(r)
    for l in out: print(f"[check_ps] {l}")
    print(f"\n{'cell':<62}{'sw':>5}{'preSD':>8}{'postSD':>8}{'contract':>9}")
    for r in sorted(recs, key=lambda x: (x["lam"], x["sweeps"])):
        print(f"{r['tag']:<62}{r['sweeps']:>5}{r['pre_sd_last']:>8.4f}"
              f"{r['post_sd_last']:>8.4f}{r['contract']:>9.4f}")
    print("\nnear-zero post-peer SD is REPORTED, not failed: a plateau at ~0 is "
          "the hypothesis coming true.")
    allok = bool(recs) and all(r["ok"] for r in recs) and not any(
        l.startswith("FAIL") for l in out)
    print(f"[check_ps] {'PASS' if allok else 'FAILED'} -- {len(a.runs)} run(s)")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
