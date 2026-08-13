#!/usr/bin/env python3
"""Paired invariant checker for MATCHED-RANDOMNESS DPO runs (2026-08-13).

Usage:  python check_dpo_pair.py <closed_run_dir> [--write-meta]

Given the CLOSED arm's run dir, derives the open dir and the shared bank
(_closed_ -> _open_ / _bank_) and verifies the full paired contract:

  1. both arms complete (>= n_rounds trajectory rows) and their configs are
     identical except the feedback source (allowed diffs: run_tag,
     rlhf_feedback, dpo_bank_mode, host, gpu_name);
  2. the shared bank's candidate ids/strings, validity/tie masks, and BT
     uniforms are internally hash-consistent, and BOTH arms' per-round row
     hashes (dpo_cand_hash / dpo_unif_hash) equal the bank's;
  3. round-0 identity: the closed and open arms carry the SAME round-0
     preference-label orientation, served predictions, and population state
     (all three cross-checked against the bank's round-0 record), the
     reader marks dpo_r0_shared and zero round-0 disagreement, and the bank
     round-0 adapter snapshot digest verifies;
  4. the open arm's judge equals its no-platform twin: rlhf_feedback=open
     and twin_raw stays exactly on innate (eps_social=0);
  5. divergence telemetry is self-consistent: reader disagreement keys
     present every round, first-divergence round == first round with
     dpo_label_disagree_n > 0.

Exit 0 + (optionally) pair_meta.json on success; exit 1 with the first
failure otherwise. Corrupted candidates/uniforms/hashes/round-0 state and
arm-config drift all land here or in DpoBank.read_round's own hash checks.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from perfsim.learners.lm.dpo_bank import DpoBank, state_digest  # noqa: E402

ALLOWED_CFG_DIFF = {"run_tag", "rlhf_feedback", "dpo_bank_mode", "host",
                    "gpu_name"}


def fail(msg):
    print(f"FAIL {msg}")
    sys.exit(1)


def main():
    closed_dir = Path(sys.argv[1]).resolve()
    write_meta = "--write-meta" in sys.argv[2:]
    name = closed_dir.name
    if "_closed_" not in name:
        fail(f"{name}: not a closed-arm dir")
    open_dir = closed_dir.parent / name.replace("_closed_", "_open_")
    bank_dir = closed_dir.parent / name.replace("_closed_", "_bank_")

    dc = torch.load(closed_dir / "trajectory.pt", map_location="cpu",
                    weights_only=False)
    do = torch.load(open_dir / "trajectory.pt", map_location="cpu",
                    weights_only=False)
    cc, co = dc["config"], do["config"]
    n_rounds = int(cc["n_rounds"])
    for tag, d in (("closed", dc), ("open", do)):
        if len(d["trajectory"]) < n_rounds:
            fail(f"{tag} arm incomplete: {len(d['trajectory'])}/{n_rounds}")

    # 1 -- configs equal except the feedback source
    diff = {k for k in set(cc) | set(co) if cc.get(k) != co.get(k)}
    if diff - ALLOWED_CFG_DIFF:
        fail(f"arm configs differ beyond feedback source: "
             f"{sorted(diff - ALLOWED_CFG_DIFF)}")
    if not (cc.get("rlhf_feedback") == "closed"
            and co.get("rlhf_feedback") == "open"):
        fail(f"feedback arms wrong: closed={cc.get('rlhf_feedback')!r} "
             f"open={co.get('rlhf_feedback')!r}")
    if not (cc.get("dpo_bank_mode") == "write"
            and co.get("dpo_bank_mode") == "read"):
        fail("bank modes wrong (want closed=write, open=read)")
    if not (cc.get("dpo_matched") and co.get("dpo_matched")):
        fail("dpo_matched marker missing")

    bank = DpoBank(str(bank_dir), int(cc["dpo_bank_seed"]))
    meta = bank.read_meta()

    # 2 -- bank internal hashes + both arms' row hashes match the bank
    tc, to = dc["trajectory"], do["trajectory"]
    for t in range(n_rounds):
        rec = bank.read_round(t)   # raises on internal hash corruption
        for arm, rows in (("closed", tc), ("open", to)):
            if t == 0 and arm == "open":
                continue  # round 0 is the fork: reader carries writer stats
            for key, want in (("dpo_cand_hash", rec["cand_hash"][:16]),
                              ("dpo_unif_hash", rec["unif_hash"][:16])):
                if rows[t].get(key) != want:
                    fail(f"{arm} round {t}: {key}={rows[t].get(key)!r} != "
                         f"bank {want!r}")

    # 3 -- round-0 identity through the bank's round-0 record
    r0 = bank.read_round0_state()  # raises on snapshot digest corruption
    if state_digest(r0["snapshot"]) != r0["snapshot_digest"]:
        fail("round-0 snapshot digest mismatch")
    if not torch.equal(dc["pred_raw"][0].float(), r0["preds"].float()):
        fail("closed round-0 served preds != bank record")
    if not torch.equal(do["pred_raw"][0].float(), r0["preds"].float()):
        fail("open round-0 served preds != bank record (fork not exact)")
    if not torch.equal(dc["op_raw"][0].float(), r0["x0"].float()):
        fail("closed round-0 population != bank record")
    if not torch.equal(do["op_raw"][0].float(), r0["x0"].float()):
        fail("open round-0 population != bank record (fork not exact)")
    if tc[0].get("dpo_orient_hash") != r0["stats"].get("dpo_orient_hash"):
        fail("closed round-0 label orientation != bank record")
    if to[0].get("dpo_orient_hash") != r0["stats"].get("dpo_orient_hash"):
        fail("open round-0 label orientation != writer's (fork not shared)")
    if not to[0].get("dpo_r0_shared"):
        fail("open round 0 lacks the dpo_r0_shared marker")
    if to[0].get("dpo_label_disagree_n") != 0:
        fail(f"open round-0 disagreement "
             f"{to[0].get('dpo_label_disagree_n')!r} != 0")

    # 4 -- the open judge IS the no-platform twin (eps_social=0 -> innate)
    if float(co.get("eps", -1.0)) != 0.0:
        fail(f"open arm eps={co.get('eps')!r} (matched design wants 0)")
    innate = do["innate"].float()
    tw_err = float((do["twin_raw"].float()
                    - innate.unsqueeze(0)).abs().max())
    if tw_err != 0.0:
        fail(f"open twin drifts off innate (max |twin-innate| = {tw_err})")

    # 5 -- divergence telemetry
    first_div = None
    for t in range(1, n_rounds):
        n_dis = to[t].get("dpo_label_disagree_n")
        if n_dis is None:
            fail(f"open round {t} missing dpo_label_disagree_n")
        if first_div is None and int(n_dis) > 0:
            first_div = t

    late = slice(n_rounds - 5, n_rounds)
    op_c = dc["op_raw"].float()
    op_o = do["op_raw"].float()
    summary = {
        "closed_tag": cc["run_tag"], "open_tag": co["run_tag"],
        "bank_seed": int(cc["dpo_bank_seed"]),
        "train_seed": int(cc["dpo_train_seed"]),
        "pop_seed": int(cc["seed"]), "eps_ai": float(cc["eps_ai"]),
        "n_rounds": n_rounds,
        "pristine_digest": meta["pristine_digest"],
        "hosts": {"closed": cc.get("host"), "open": co.get("host")},
        "gpus": {"closed": cc.get("gpu_name"), "open": co.get("gpu_name")},
        "first_divergence_round": first_div,
        "r1_disagree_frac": to[1].get("dpo_label_disagree_frac"),
        "mean_disagree_frac": float(sum(
            to[t].get("dpo_label_disagree_frac") or 0.0
            for t in range(1, n_rounds)) / (n_rounds - 1)),
        "late_mean_diff_closed_minus_open": float(
            op_c[late].mean() - op_o[late].mean()),
        "pair_complete": True,
    }
    print(f"PASS {name}: pair valid; first divergence round {first_div}; "
          f"late closed-open mean diff "
          f"{summary['late_mean_diff_closed_minus_open']:+.4f}")
    if write_meta:
        (bank_dir / "pair_meta.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True))
        print(f"[check_dpo_pair] wrote {bank_dir / 'pair_meta.json'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
