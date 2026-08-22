#!/usr/bin/env python3
"""Reuse audit for the Section 3 retention wave (section3_retention, 2026-08-22).

Decides, for each of the four PLAUSIBLE archived candidates, whether it can
stand in for a Section 3 production cell. Matching is BY EXACT CONFIG FIELD
VALUE AND TRAJECTORY COMPLETENESS, NEVER by tag similarity: the QWU tags and
the pofds3_ tags are different strings by construction, so a tag comparison
would either reject everything or, worse, accept a near-neighbour that differs
on a dynamics-determining dial.

THE CANDIDATES (the only archived 100-round, both-gates-open, k=1 Qwen2.5
cells that sit on the Section 3 surface at all):

    pofdqwu_qwen7b_b0_eaopen_w0p5_l1_esopen_s0_r100 -> (qwen7b, sft,      env1)
    pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100   -> (qwen7b, sft,      env2)
    pofdqwu_qwen7b_b1_eaopen_w0p5_l1_esopen_s0_r100 -> (qwen7b, fwdlam1,  env1)
    pofdqwu_qwen7b_b1_eaopen_w1_l1_esopen_s0_r100   -> (qwen7b, fwdlam1,  env2)

env3 (beta=0.5, k=0.2) has NO archived counterpart, and neither does Qwen3-8B,
so 46 of the 50 conceptual trained cells are new jobs regardless of what this
audit finds.

TWO KNOWN DEVIATIONS, both recorded in the manifest so a reader can disagree
with the decision rather than having to rediscover it:

(1) serve_eval_mode ABSENT.
    The four configs carry neither serve_eval_mode nor git_sha. Both keys are
    written side by side by run_pokec_gated_lm.py (lines 1719 and 1721) and
    BOTH were introduced by commit 3f4e06e (2026-08-21 15:07:32 +0200) -- a
    full DAY AFTER these runs were launched. Their absence is therefore
    evidence about the CONFIG SCHEMA, not about the serving path, and carries
    no information either way about whether LoRA dropout was active.

    The serving fix itself is commit 9ee5136 (2026-08-20 16:31:06 +0200),
    which patched perfsim/models/hf_causal_lm.py::_generate to force .eval()
    and restore the caller's mode in `finally`. It did NOT touch
    run_pokec_gated_lm.py at all.

    Whether these runs served with dropout OFF is therefore a question about
    the cluster checkout at launch time, and it is settled DIRECTLY by the
    cluster repo reflog (read-only, `git reflog` in ~/perfsim):

        9ee5136 HEAD@{2026-08-20 16:32:22 +0200}: pull: Fast-forward
        c25b2cf HEAD@{2026-08-20 16:57:24 +0200}: pull: Fast-forward
        79b133a HEAD@{2026-08-20 17:06:11 +0200}: pull: Fast-forward

    The four config.json files were written at 2026-08-20 17:07, one minute
    after HEAD advanced to 79b133a, and `git merge-base --is-ancestor 9ee5136
    79b133a` is TRUE. Every reflog entry in the whole 2026-08 window is a
    Fast-forward -- the only non-fast-forward entries in the last 200 are four
    `reset: moving to HEAD` no-ops from 2026-07-12..15. So the tree that ran
    these four jobs provably CONTAINED the eval-mode fix.

    This is direct evidence, and it supersedes the weaker indirect argument
    (that the run dir pofdqwu_qwen7b_b1_eaopen_w1_l1_esopen_s0_r3smoke2 has
    config.json at 2026-08-20 16:36:41, and the "smoke2" token itself was
    created by 9ee5136, so a dir wearing that token cannot predate it). Both
    are recorded in the manifest; the reflog one is what the verdict rests on.

(2) population_update == "nested_ai_then_social_v1".
    A run launched today writes "nested_ai_anchored_then_social_v2" instead
    (run_pokec_gated_lm.py:946 defaults AI_GATE_REFERENCE to "anchor";
    _POP_UPDATE_MARKER at :219-226 maps that to the v2 string). The two
    operators differ ONLY in which vector the AI gate measures distance FROM.
    Under ai_gate_mode="all_open" that distance is never read: _gated_pop.py
    ai_gate() lines 205-206 return an all-ones mask before touching the
    reference argument. Both cells here are all_open, so v1 and v2 are the
    SAME FUNCTION on this surface, bit for bit. Recorded, not fatal.

SCOUTS ARE NEVER REUSABLE. The pofdkd_* cells are 10-round direction scouts.
A 100-round production arm cannot be satisfied by a 10-round trajectory under
any argument, so any candidate whose n_rounds != 100 (or whose trajectory is
short) is REJECTED with reason "SCOUT_HORIZON" before any other field is even
considered. Passing one on --extra-candidates does not change that.

Writes notes/pofd/section3/reuse_manifest.json with, per candidate: the tag,
the ABSOLUTE artifact path, the verdict, every field compared as
(expected, actual, ok), the deviation record above, and the sha256 of the
served vector.

Usage:
  python3 audit_section3_reuse.py [--roots DIR ...] [--write]
      [--extra-candidates TAG ...] [--expect-reuse N]
"""
import argparse
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_ROOTS = [os.path.join(REPO, "notes", "pofd", "cluster"),
                 os.path.join(REPO, "runs", "pokec_gated_lm")]
MANIFEST = os.path.join(REPO, "notes", "pofd", "section3", "reuse_manifest.json")

S3_KEY = "section3_retention"
N_AGENTS = 723
N_ROUNDS = 100

QWEN25 = "Qwen/Qwen2.5-7B-Instruct"

# sentinel: "this key may be ABSENT, or hold one of the listed values".
# An absent key means the run executed the runner default of its era, which
# for every key below is the value we want -- but the distinction is kept
# visible in the manifest rather than silently folded into a match.
ABSENT = "<ABSENT>"


def _absent_ok(allowed):
    return ABSENT in allowed


# ---------------------------------------------------------------------------
# THE PINNED SECTION 3 TRAINING SURFACE.
# Every key here is dynamics-determining or budget-determining. Telemetry-only
# keys (host, hardware.hostname, log_ppl_dist, ans_sample_*, n_probe,
# tel_eval_cap, grad_norm_n, grad_decomp) are RECORDED but never matched: they
# consume no RNG the loop reads and touch no update.
SHARED_WANT = {
    # identity / data
    "base_model":        (QWEN25,),
    "dataset":           ("movielens",),
    "ml_target":         ("Action",),
    "n_labeled":         (723,),
    "train_cap":         (723,),
    "seed":              (0,),
    "seed_base_data":    (True,),
    # horizon
    "n_rounds":          (N_ROUNDS,),
    "deploy_every":      (1,),
    "epoch_size":        (100,),
    "max_steps":         (1,),
    # operator / gates
    "ai_gate_mode":      ("all_open",),
    "peer_gate_mode":    ("all_open",),
    "eps":               (0.2,),
    "gamma_bias":        (0.0,),
    "ab_sweeps":         (1,),
    "pop_model":         ("ab",),
    "anchor_mode":       ("fixed",),
    "run_mode":          ("loop",),
    "data_regime":       ("replace",),
    "platform_sus_scale": (1.0,),
    # training budget
    "use_lora":          (True, 1),
    "lora_r":            (512,),
    "sft_lr":            (5e-5,),
    "sft_epochs":        (1,),
    "sft_batch_size":    (4,),
    "fresh_each_round":  (True,),
    "kl_ref_adapter":    ("",),
    # context: Section 3 is the NO-ICL surface
    "icl_k":             (0,),
    "icl_days":          (0,),
    # serving
    "do_sample":         (False,),
    "save_raw_gen":      (True,),
    # knobs that must be OFF
    "canary_delta":      (0.0,),
    "pop_reset":         (False,),
    "pristine_frac":     (0.0,),
    "replay_frac":       (ABSENT, 0.0),
    "teacher_label_delta": (ABSENT, 0.0),
    "icrh":              (ABSENT, False),
    "feedback_mode":     (ABSENT, "none"),
    "profile_shuffle_p": (ABSENT, 0.0),
    "profile_sort_q":    (ABSENT, 0.0),
}

# eps_ai is deliberately NOT matched: under ai_gate_mode="all_open" the gate
# returns before reading it (_gated_pop.py:205-206), so its stored value
# (1.0 in the archive) is inert. It is RECORDED so the reader can see it.
RECORDED_ONLY = ["eps_ai", "host", "log_ppl_dist", "ans_sample_k", "n_probe",
                 "tel_eval_cap", "grad_norm_n", "grad_decomp", "pop_order",
                 "rlhf_feedback", "population_update"]

ARM_WANT = {
    # ordinary SFT: lambda = 0, no KL term at all. kl_direction is INERT here
    # (kl_beta multiplies it to zero) and is NOT matched -- this is why the
    # sft arm is direction-neutral and its tag carries no direction token.
    "sft":      {"training_style": ("sft",),    "kl_beta": (0.0,)},
    "fwdlam1":  {"training_style": ("sft_kl",), "kl_beta": (1.0,),
                 "kl_direction": ("forward",)},
}

# (beta, k) -> env name. Section 3 pins exactly three.
ENVS = {
    "env1": {"w_plat": 0.5, "innate_lambda": 1.0},
    "env2": {"w_plat": 1.0, "innate_lambda": 1.0},
    "env3": {"w_plat": 0.5, "innate_lambda": 0.2},
}

# The four plausible candidates, each mapped to the S3 cell it would satisfy.
CANDIDATES = [
    ("pofdqwu_qwen7b_b0_eaopen_w0p5_l1_esopen_s0_r100", "sft",     "env1"),
    ("pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100",   "sft",     "env2"),
    ("pofdqwu_qwen7b_b1_eaopen_w0p5_l1_esopen_s0_r100", "fwdlam1", "env1"),
    ("pofdqwu_qwen7b_b1_eaopen_w1_l1_esopen_s0_r100",   "fwdlam1", "env2"),
]

# ---------------------------------------------------------------------------
# The two known deviations, verbatim, so they land in the manifest.
DEVIATIONS = {
    "serve_eval_mode_absent": {
        "what": "config.json carries neither serve_eval_mode nor git_sha.",
        "why_absent": (
            "Both keys are written adjacently by run_pokec_gated_lm.py "
            "(serve_eval_mode at line 1719, git_sha at line 1721) and both "
            "were introduced by commit 3f4e06e (2026-08-21 15:07:32 +0200), "
            "one day AFTER these runs launched. The absence is a config-SCHEMA "
            "fact, not a serving fact."),
        "fix_commit": "9ee5136 (2026-08-20 16:31:06 +0200)",
        "fix_touched": "perfsim/models/hf_causal_lm.py::_generate "
                       "(forces .eval(), restores mode in finally)",
        "fix_did_not_touch": "run_pokec_gated_lm.py",
        "direct_evidence": (
            "Cluster repo reflog (~/perfsim, read-only): "
            "9ee5136 HEAD@{2026-08-20 16:32:22}, "
            "c25b2cf HEAD@{2026-08-20 16:57:24}, "
            "79b133a HEAD@{2026-08-20 17:06:11}, all 'pull: Fast-forward'. "
            "The four config.json files were written 2026-08-20 17:07, one "
            "minute after HEAD reached 79b133a, and "
            "`git merge-base --is-ancestor 9ee5136 79b133a` is TRUE. The only "
            "non-fast-forward reflog entries in the last 200 are four "
            "'reset: moving to HEAD' no-ops from 2026-07-12..15. The tree that "
            "ran these jobs therefore CONTAINED the eval-mode fix."),
        "indirect_corroboration": (
            "pofdqwu_qwen7b_b1_eaopen_w1_l1_esopen_s0_r3smoke2/config.json is "
            "dated 2026-08-20 16:36:41 on the cluster, and the 'smoke2' token "
            "(QWU_SMOKE_TOKEN) was itself created by 9ee5136, so a dir wearing "
            "that token cannot have been generated by a pre-fix tree."),
        "decision": "PERMITS REUSE",
        "decision_basis": "direct (reflog ancestry), not inference",
        "how_to_disagree": (
            "Show that the cluster checkout at 2026-08-20 17:07 was NOT "
            "79b133a -- e.g. a dirty working tree with hf_causal_lm.py "
            "reverted, which the reflog cannot see. Nothing in the archive "
            "rules that out; it is judged implausible, not impossible."),
    },
    "population_update_v1": {
        "what": 'population_update == "nested_ai_then_social_v1"; a run '
                'launched today writes "nested_ai_anchored_then_social_v2".',
        "why": ("run_pokec_gated_lm.py:946 defaults AI_GATE_REFERENCE to "
                "'anchor'; _POP_UPDATE_MARKER (lines 219-226) maps 'anchor' -> "
                "v2 and 'x0' -> v1. gen_pofd_sweep.py never sets "
                "AI_GATE_REFERENCE, so the runner default decides."),
        "numerical_impact": (
            "ZERO on this surface. v1 and v2 differ only in the vector the AI "
            "gate measures distance FROM (gate_reference, _gated_pop.py:212-238). "
            "Under ai_gate_mode='all_open', ai_gate() returns an all-ones mask "
            "at _gated_pop.py:205-206 BEFORE reading that reference at all, so "
            "the two operators are the same function bit-for-bit. The runner's "
            "own docstring states the same at _gated_pop.py:257-260."),
        "decision": "PERMITS REUSE",
        "how_to_disagree": (
            "Only if some downstream consumer keys on the marker STRING rather "
            "than the operator. Several plotters do assert "
            "population_update == 'nested_ai_then_social_v1' -- any Section 3 "
            "analyzer must accept BOTH markers, since the wave will mix them."),
    },
    "hgate_v2_marker": {
        "what": 'The spec asked for provenance marker '
                '"nested_ai_then_social_hgate_v2".',
        "finding": ("NO SUCH STRING EXISTS anywhere in the repo. `grep -rn "
                    "hgate` returns zero hits in .py/.sh/.md. The only two "
                    "markers the runner can ever write are "
                    "'nested_ai_then_social_v1' and "
                    "'nested_ai_anchored_then_social_v2'."),
        "decision": "DO NOT INVENT. The audit does not require, expect, or "
                    "emit an hgate marker.",
    },
}


def sha256_f32(t):
    """sha256 over raw float32 bytes -- the project convention
    (_gated_pop.ref_replay_hash)."""
    return hashlib.sha256(
        t.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def field_verdict(cfg, want):
    """{field: {"expected": [...], "actual": v, "ok": bool}} for every
    matched field. Absent keys surface as the ABSENT sentinel so the manifest
    distinguishes 'key missing' from 'key present and wrong'."""
    out = {}
    for k, allowed in want.items():
        got = cfg.get(k, ABSENT)
        if got is ABSENT or got == ABSENT:
            ok = _absent_ok(allowed)
            got = ABSENT
        else:
            ok = any(a != ABSENT and got == a for a in allowed)
        out[k] = {"expected": list(allowed), "actual": got, "ok": bool(ok)}
    return out


def cell_want(arm, env):
    want = dict(SHARED_WANT)
    want.update(ARM_WANT[arm])
    want["w_plat"] = (ENVS[env]["w_plat"],)
    want["innate_lambda"] = (ENVS[env]["innate_lambda"],)
    return want


def find_run(tag, roots):
    for root in roots:
        d = os.path.join(root, tag)
        if os.path.exists(os.path.join(d, "config.json")):
            return os.path.abspath(d)
    return None


def artifact_check(run_dir):
    """Completeness of the trajectory: 100x723 finite pred_raw/op_raw, plus
    the artifacts the config cannot speak for.

    WITH_TWIN is read from the environment by the runner (_env_int("WITH_TWIN",
    0) at run_pokec_gated_lm.py:2324) and is NEVER written to config.json, so
    the ONLY way to confirm WITH_TWIN=1 is a non-empty twin_raw here."""
    out = {"ok": False, "notes": [], "artifacts": {}}
    pt = os.path.join(run_dir, "trajectory.pt")
    if not os.path.exists(pt):
        out["notes"].append("no trajectory.pt")
        return out, None
    d = torch.load(pt, map_location="cpu", weights_only=False)
    traj = d.get("trajectory", [])
    if len(traj) != N_ROUNDS:
        out["notes"].append(f"{len(traj)} trajectory rounds != {N_ROUNDS}")
    for key in ("pred_raw", "op_raw"):
        t = d.get(key)
        shp = None if t is None else tuple(t.shape)
        if shp != (N_ROUNDS, N_AGENTS):
            out["notes"].append(f"{key} shape {shp} != {(N_ROUNDS, N_AGENTS)}")
            continue
        if not bool(torch.isfinite(t).all()):
            n_bad = int((~torch.isfinite(t)).sum())
            out["notes"].append(f"{key} has {n_bad} non-finite entries")
        out["artifacts"][key] = {"shape": list(shp), "finite": True}
    tw = d.get("twin_raw")
    twin_ok = tw is not None and tw.numel() > 0 and \
        tuple(tw.shape) == (N_ROUNDS, N_AGENTS)
    out["artifacts"]["twin_raw"] = {
        "present": bool(twin_ok),
        "shape": None if tw is None else list(tw.shape),
        "note": "non-empty twin_raw is the ONLY evidence of WITH_TWIN=1; "
                "the runner never writes it to config.json",
    }
    if not twin_ok:
        out["notes"].append("twin_raw missing/empty -> WITH_TWIN=1 unproven")
    gr = d.get("gate_raw")
    out["artifacts"]["gate_raw"] = {
        "present": bool(gr is not None and gr.numel() > 0),
        "shape": None if gr is None else list(gr.shape)}
    out["ok"] = not out["notes"]
    if not out["notes"]:
        out["notes"].append("complete")
    return out, d


def audit_one(tag, arm, env, roots):
    rec = {"tag": tag, "s3_cell": {"model": "qwen7b", "arm": arm, "env": env},
           "path": None, "verdict": "REJECT", "reject_reasons": [],
           "fields": {}, "recorded_only": {}, "artifacts": {},
           "served_sha256": None}
    run_dir = find_run(tag, roots)
    if run_dir is None:
        rec["reject_reasons"].append("NOT_FOUND: no config.json under any root")
        return rec
    rec["path"] = run_dir
    cfg = json.load(open(os.path.join(run_dir, "config.json")))

    # HARD GUARD, evaluated FIRST: a scout can never satisfy a production arm.
    n_rounds = cfg.get("n_rounds")
    if n_rounds != N_ROUNDS:
        rec["reject_reasons"].append(
            f"SCOUT_HORIZON: n_rounds={n_rounds} != {N_ROUNDS}. A short-horizon "
            f"scout (e.g. the 10-round pofdkd_* direction scouts) can NEVER "
            f"satisfy a 100-round production arm -- the equilibrium window is "
            f"post-peer rounds 81-100, which such a run does not contain.")
        rec["fields"] = field_verdict(cfg, cell_want(arm, env))
        return rec

    want = cell_want(arm, env)
    rec["fields"] = field_verdict(cfg, want)
    bad = sorted(k for k, v in rec["fields"].items() if not v["ok"])
    if bad:
        rec["reject_reasons"].append("FIELD_MISMATCH: " + ", ".join(
            f"{k}(want {rec['fields'][k]['expected']}, "
            f"got {rec['fields'][k]['actual']!r})" for k in bad))

    hw = cfg.get("hardware", {}) or {}
    gpu = hw.get("gpu_name", ABSENT)
    rec["fields"]["hardware.gpu_name"] = {
        "expected": ["NVIDIA H100 80GB HBM3"], "actual": gpu,
        "ok": gpu == "NVIDIA H100 80GB HBM3"}
    if gpu != "NVIDIA H100 80GB HBM3":
        rec["reject_reasons"].append(f"GPU_MISMATCH: {gpu!r}")

    # serve_eval_mode: absent on these runs. Recorded with the full deviation
    # argument rather than matched, because absence carries no information
    # (see DEVIATIONS["serve_eval_mode_absent"]).
    rec["fields"]["serve_eval_mode"] = {
        "expected": [True, ABSENT],
        "actual": cfg.get("serve_eval_mode", ABSENT),
        "ok": True,
        "note": "ABSENT accepted -- the FIELD postdates the run by a day; the "
                "FIX is proven present by cluster reflog ancestry. See "
                "deviations.serve_eval_mode_absent."}

    for k in RECORDED_ONLY:
        rec["recorded_only"][k] = cfg.get(k, ABSENT)
    rec["recorded_only"]["hardware"] = hw

    arts, d = artifact_check(run_dir)
    rec["artifacts"] = arts
    if not arts["ok"]:
        rec["reject_reasons"].append(
            "ARTIFACT_INCOMPLETE: " + "; ".join(arts["notes"]))
    if d is not None and d.get("pred_raw") is not None:
        pr = d["pred_raw"]
        rec["served_sha256"] = {
            "pred_raw_full_f32": sha256_f32(pr),
            "pred_raw_round0_f32": sha256_f32(pr[0]),
            "convention": "sha256 over raw float32 bytes, contiguous "
                          "(_gated_pop.ref_replay_hash)",
        }
        if d.get("op_raw") is not None:
            rec["served_sha256"]["op_raw_full_f32"] = sha256_f32(d["op_raw"])

    rec["verdict"] = "REUSE" if not rec["reject_reasons"] else "REJECT"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--write", action="store_true",
                    help="write the manifest (default: dry-run print only)")
    ap.add_argument("--extra-candidates", nargs="*", default=[],
                    help="extra TAGs to audit as (arm,env) guesses; scouts "
                         "are refused on horizon regardless")
    ap.add_argument("--expect-reuse", type=int, default=None)
    args = ap.parse_args()

    roots = [os.path.abspath(r) for r in args.roots]
    print(f"[audit_section3_reuse] key={S3_KEY}")
    print(f"[audit_section3_reuse] roots: {roots}")

    recs = [audit_one(t, arm, env, roots) for t, arm, env in CANDIDATES]
    for tag in args.extra_candidates:
        # An unmapped extra is audited against env1/sft purely so the horizon
        # guard and the field table still run and REPORT; it is not a claim
        # that the tag belongs to that cell.
        r = audit_one(tag, "sft", "env1", roots)
        r["s3_cell"]["note"] = ("UNMAPPED extra candidate; audited against "
                                "sft/env1 only so the horizon guard reports")
        recs.append(r)

    n_reuse = sum(r["verdict"] == "REUSE" for r in recs)
    total_cells = 50  # 2 models x 3 envs x 7 + 2 models x 2 envs x 2 reverse
    print()
    for r in recs:
        c = r["s3_cell"]
        print(f"  {r['verdict']:6s}  {r['tag']}")
        print(f"          -> cell (qwen7b, {c['arm']}, {c['env']})")
        print(f"          path: {r['path']}")
        if r["served_sha256"]:
            print(f"          served sha256(pred_raw f32): "
                  f"{r['served_sha256']['pred_raw_full_f32']}")
        for why in r["reject_reasons"]:
            print(f"          REJECT: {why}")
    print()
    print(f"[audit_section3_reuse] REUSE {n_reuse} / {len(recs)} audited")
    print(f"[audit_section3_reuse] conceptual trained cells: {total_cells}; "
          f"new GPU jobs needed: {total_cells - n_reuse}")

    manifest = {
        "sweep_key": S3_KEY,
        "generated_by": os.path.relpath(os.path.abspath(__file__), REPO),
        "horizon": {"n_rounds": N_ROUNDS, "n_agents": N_AGENTS,
                    "equilibrium_window": "post-peer rounds 81-100"},
        "conceptual_trained_cells": total_cells,
        "n_reuse": n_reuse,
        "n_new_gpu_jobs": total_cells - n_reuse,
        "matching_policy": (
            "EXACT FIELD MATCH on every dynamics- and budget-determining "
            "config key, plus trajectory completeness. Tag similarity is "
            "NEVER used. eps_ai is recorded but not matched (inert under "
            "all_open). kl_direction is not matched on the sft arm (inert at "
            "kl_beta=0)."),
        "scout_policy": (
            "Any candidate with n_rounds != 100 is REJECTED with "
            "SCOUT_HORIZON before any other field is considered. The 10-round "
            "pofdkd_* cells are scouts and can never satisfy a production arm."),
        "deviations": DEVIATIONS,
        "candidates": recs,
    }

    if args.expect_reuse is not None and n_reuse != args.expect_reuse:
        print(f"[audit_section3_reuse] DISCREPANCY: expected "
              f"{args.expect_reuse} reusable, got {n_reuse}. "
              f"Manifest NOT written.", file=sys.stderr)
        return 1

    if args.write:
        os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)
            f.write("\n")
        print(f"[audit_section3_reuse] wrote {MANIFEST}")
    else:
        print("[audit_section3_reuse] dry run; pass --write to emit "
              f"{os.path.relpath(MANIFEST, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
