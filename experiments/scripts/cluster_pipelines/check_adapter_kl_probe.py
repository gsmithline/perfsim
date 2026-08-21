#!/usr/bin/env python3
"""HARD-GATED checker for the adapter KL / soft-decode probe (2026-08-21).

Nothing in the probe's output is trustworthy unless the probe was
measuring what it claims. These are the conditions under which the
soft-decode and KL numbers mean anything; each is a hard failure.

  1  CANONICAL BASE. The probe's own greedy served vector must hash to
     the pinned frozen Qwen2.5 K=D=0 vector. This is the single check
     that ties the probe's serving path to the archived runs: prompts,
     chat template, padding, decoding settings and hardware all have to
     agree or the hash does not. A probe that silently prompts
     differently would produce beautifully self-consistent numbers about
     a model nobody ran.

  2  ONE BASE FOR EVERY ADAPTER. Each adapter cell recomputes the base
     soft value under disable_adapter(). Those must agree with the base
     stage AND with each other. If peft ever leaves an adapter partially
     active, this drifts -- and every KL in that cell is measured
     against a contaminated reference while still looking finite and
     ordered.

  3  SUPPORT ACTUALLY COVERS THE MASS. Soft values are expectations over
     a truncated numeric support, renormalized. That is only honest if
     the mass outside the support is small, for BOTH models. Checked
     against a stated bound rather than assumed, because an adapter that
     moved probability onto non-numeric tokens would otherwise be
     reported as a confident soft value.

  4  t* CARRIES THE VALUE UNCERTAINTY. The soft value is read at one
     position. If other positions carry comparable leverage the single-
     position summary is hiding structure, so the leverage share at t*
     is checked and reported.

  5  COMPLETE, DISTINCT, WELL-FORMED. All 15 dose adapters present
     exactly once, tags matching the on-disk condor configs; KL finite
     and non-negative in both directions; distinct adapters produce
     distinct KL vectors (identical vectors mean the adapter never got
     applied -- the failure mode that looks most like a real result).

  6  GREEDY CROSS-CHECK. For agents whose adapter argmax does not
     diverge before t*, the teacher-forced greedy value must equal the
     value that adapter actually served in its dose run. This is the
     link between the probe's reference frame and the served-opinion
     measurement it exists to explain. It is REPORTED per adapter and
     gated only loosely: legitimate early divergence breaks the
     correspondence for some agents, and pretending otherwise would
     turn a known limitation into a false alarm.

Usage:
  python check_adapter_kl_probe.py [--dir runs/adapter_kl_probe]
                                   [--runs-root runs/pokec_gated_lm]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
REPO = HERE.parent.parent.parent

_spec = importlib.util.spec_from_file_location(
    "probe_akl", str(HERE / "probe_adapter_kl.py"))
AKL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AKL)

TAIL_MAX = 0.02          # mass allowed outside the numeric support
BASE_DEV_MAX = 1e-6      # base soft value must be reproduced exactly
TSTAR_SHARE_MIN = 0.5    # t* must carry at least this share of leverage
# The fixed-tail value map substitutes at t* and keeps the BASE's
# remaining tokens, so it reproduces the LEADING decimal of the adapter's
# answer, not its second digit -- the adapter emits its own tail. Exact
# equality is therefore the wrong test (it scores 1-3%); agreement is
# checked at the resolution the frame actually has, and the residual is
# reported so the analysis can state its own precision.
VALUE_TOL = 0.06         # one leading-decimal step, minus rounding
GREEDY_AGREE_MIN = 0.90  # at VALUE_TOL, not at exact equality


def check(dirpath, runs_root, expect_agents=AKL.N_AGENTS):
    errs, notes = [], []
    d = Path(dirpath)
    mf = json.load(open(d / "probe_manifest.json"))
    base = torch.load(d / "base_probe.pt", map_location="cpu",
                      weights_only=False)

    # -- 1 canonical base -------------------------------------------------
    sha = AKL.sha_vec(base["served"])
    if mf["n_agents"] != expect_agents:
        errs.append(f"n_agents {mf['n_agents']} != {expect_agents} -- this "
                    f"is a truncated (smoke) probe, not a gateable run")
    elif sha != AKL.CANON_SHA:
        errs.append(f"base served sha {sha[:16]}... != canonical "
                    f"{AKL.CANON_SHA[:16]}... -- the probe is not "
                    f"reproducing the archived frozen Qwen serving path")
    if mf.get("hash_gate") != "enforced" and mf["n_agents"] == expect_agents:
        errs.append(f"hash_gate={mf.get('hash_gate')!r} on a full run")

    # -- 1a the serving frame ---------------------------------------------
    # generate() applies the model's logits processors; under
    # do_sample=False that still includes repetition_penalty. A probe that
    # scored the RAW distribution while the runs served a PENALIZED one is
    # measuring a different object -- which is how the first two attempts
    # failed, with argmax disagreement concentrated at the decision digit.
    if "repetition_penalty" not in mf:
        errs.append("manifest does not record repetition_penalty -- cannot "
                    "tell which decoding frame this probe scored")
    else:
        notes.append(f"serving frame: repetition_penalty="
                     f"{mf['repetition_penalty']}")

    # -- 1b teacher-forced alignment --------------------------------------
    # With the serving frame reproduced, a residual mismatch is bf16 noise
    # between cached incremental decoding and one full-sequence forward,
    # which can only flip a nearly tied position. A mismatch at a
    # CONFIDENT position means the span is misaligned.
    tm = mf.get("tf_mismatch")
    if tm is None:
        errs.append("manifest has no tf_mismatch record -- this probe "
                    "predates the alignment diagnostic and cannot be gated")
    else:
        notes.append(
            f"teacher-forced argmax mismatches vs generate(): {tm['n']} "
            f"({100 * tm['frac_of_agents']:.1f}% of agents) at positions "
            f"{tm['positions']}, max margin {tm['max_margin']:.2e}, median "
            f"{tm['median_margin']:.2e}")
        pn = tm.get("path_noise")
        if pn:
            notes.append(
                f"measured path-noise floor (re-scored at batch "
                f"{pn['batch']}): {pn['n_flips']} argmax flips at t*, worst "
                f"flipped margin {pn['max_margin_flipped']:.3e}, max |dlogp| "
                f"{pn['max_abs_logp_diff']:.3e}")
        # a real misalignment is BROAD and CONFIDENT; path noise is
        # neither. Both bounds come from the observed signature of the
        # repetition-penalty fault (11% of agents, margins to 0.50).
        fmax = float(tm.get("frac_max", 0.02))
        hard = float(tm.get("margin_hard", 0.30))
        if float(tm["frac_of_agents"]) > fmax:
            errs.append(f"teacher-forced mismatch reaches "
                        f"{100 * tm['frac_of_agents']:.1f}% of agents > "
                        f"{100 * fmax:.0f}% -- too broad to be path noise")
        if float(tm["max_margin"]) > hard:
            errs.append(f"a teacher-forced mismatch sits at margin "
                        f"{tm['max_margin']:.4f} > {hard} -- a position that "
                        f"confident cannot flip from float noise")

    # -- 4 t* carries the value uncertainty -------------------------------
    shares = []
    for levs in base["leverage"]:
        a = np.asarray(levs, dtype=np.float64)
        tot = float(a.sum())
        shares.append(1.0 if tot <= 0 else float(a.max() / tot))
    shares = np.asarray(shares)
    notes.append(f"leverage share at t*: mean {shares.mean():.3f}, "
                 f"min {shares.min():.3f}")
    if float(shares.mean()) < TSTAR_SHARE_MIN:
        errs.append(f"t* carries only {shares.mean():.3f} of the value "
                    f"leverage on average -- the single-position soft "
                    f"value is not a faithful summary")
    tb = np.asarray(base["tail_base"])
    notes.append(f"base tail mass off the support: max {tb.max():.5f}, "
                 f"mean {tb.mean():.5f} over {len(base['support'])} "
                 f"support tokens")
    if float(tb.max()) > TAIL_MAX:
        errs.append(f"base tail mass off the numeric support reaches "
                    f"{tb.max():.4f} > {TAIL_MAX} -- the renormalized "
                    f"soft value is not trustworthy")
    to = np.asarray(base["topm_outside_support"])
    if float(to.max()) > TAIL_MAX:
        errs.append(f"the base puts up to {to.max():.4f} of its top-M mass "
                    f"OUTSIDE the numeric support -- the value map is "
                    f"dropping tokens the model actually favours")

    # -- 5 completeness ---------------------------------------------------
    want = AKL.read_dose_tags(REPO / "experiments" / "condor")
    got = list(mf["tags"])
    if len(got) != len(set(got)):
        errs.append("duplicate adapter tags in the manifest")
    if set(got) != set(want):
        errs.append(f"adapter set mismatch: missing {sorted(set(want) - set(got))}, "
                    f"unexpected {sorted(set(got) - set(want))}")

    kl_sig, per_tag = {}, {}
    for tag in got:
        p = d / f"adapter_{tag}.pt"
        if not p.exists():
            errs.append(f"{tag}: missing {p.name}")
            continue
        r = torch.load(p, map_location="cpu", weights_only=False)
        per_tag[tag] = r
        for k in ("kl_fwd_sum", "kl_rev_sum", "kl_fwd_tstar", "kl_rev_tstar",
                  "kl_served_sum", "kl_served_tstar"):
            v = np.asarray(r[k], dtype=np.float64)
            if not np.isfinite(v).all():
                errs.append(f"{tag}: {k} has non-finite entries")
            if float(v.min()) < 0.0:
                errs.append(f"{tag}: {k} has negative entries "
                            f"(min {v.min():.3e}) -- KL cannot be negative")
        # -- 2 one base for every adapter
        dev = float(np.abs(np.asarray(r["soft_base_recheck"])
                           - np.asarray(base["soft_base"])).max())
        if dev > BASE_DEV_MAX:
            errs.append(f"{tag}: base soft value recomputed under "
                        f"disable_adapter() differs from the base stage by "
                        f"{dev:.3e} -- the adapter is leaking into its own "
                        f"reference")
        # -- 3 support covers the mass
        ta = float(np.asarray(r["tail_adapter"]).max())
        if ta > TAIL_MAX:
            errs.append(f"{tag}: adapter tail mass reaches {ta:.4f} > "
                        f"{TAIL_MAX} -- soft value renormalized over a "
                        f"support the adapter has left")
        kl_sig[tag] = AKL.sha_vec(np.asarray(r["kl_fwd_sum"],
                                             dtype=np.float32))

    dupes = {}
    for tag, s in kl_sig.items():
        dupes.setdefault(s, []).append(tag)
    for s, tags in dupes.items():
        if len(tags) > 1:
            errs.append(f"identical KL vectors for {tags} -- distinct "
                        f"adapters cannot score identically; the adapter "
                        f"was probably never applied")

    # -- 6 greedy cross-check against the dose runs -----------------------
    frame_mae = []
    for tag, r in per_tag.items():
        traj = Path(runs_root) / tag / "trajectory.pt"
        if not traj.exists():
            notes.append(f"{tag}: no trajectory.pt, greedy cross-check skipped")
            continue
        served = torch.load(traj, map_location="cpu",
                            weights_only=False)["pred_raw"][0]
        served = np.asarray(served, dtype=np.float64).reshape(-1)
        gtf = np.asarray(r["greedy_tf"], dtype=np.float64)
        fd = np.asarray(r["first_div"], dtype=np.float64)
        ts = np.asarray(base["tstar"], dtype=np.float64)
        ok = (fd < 0) | (fd >= ts)          # no divergence before t*
        ok &= np.isfinite(gtf)
        if ok.sum() == 0:
            errs.append(f"{tag}: every agent diverges before t* -- the "
                        f"teacher-forced frame does not describe this "
                        f"adapter at all")
            continue
        dv = np.abs(gtf[ok] - served[ok])
        agree = float(np.mean(dv <= VALUE_TOL))
        notes.append(f"{tag}: greedy cross-check {100 * agree:.1f}% within "
                     f"{VALUE_TOL} on {int(ok.sum())}/{len(ok)} agents "
                     f"(MAE {dv.mean():.4f}, P95 {np.percentile(dv, 95):.3f}, "
                     f"max {dv.max():.2f}; exact {100 * np.mean(dv < 1e-9):.1f}%)")
        frame_mae.append(float(dv.mean()))
        if agree < GREEDY_AGREE_MIN:
            errs.append(f"{tag}: teacher-forced greedy value is within "
                        f"{VALUE_TOL} of the served value for only "
                        f"{100 * agree:.1f}% of agents (MAE {dv.mean():.4f}) "
                        f"-- the probe's frame does not reproduce the dose "
                        f"run's leading decimal")
    if frame_mae:
        # the analysis must not read a trend smaller than its own frame
        # error, so state that error once, plainly, at the end
        notes.append(f"FRAME RESOLUTION: mean |teacher-forced - served| = "
                     f"{np.mean(frame_mae):.4f} across adapters. Differences "
                     f"between cells smaller than this are NOT resolvable "
                     f"by the fixed-tail soft value.")
    return errs, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path,
                    default=REPO / "runs" / "adapter_kl_probe")
    ap.add_argument("--runs-root", type=Path,
                    default=REPO / "runs" / "pokec_gated_lm")
    args = ap.parse_args()
    errs, notes = check(args.dir, args.runs_root)
    for n in notes:
        print(f"[check_akl] {n}")
    if errs:
        print(f"[check_akl] {len(errs)} FAILURE(S)", file=sys.stderr)
        for e in errs[:40]:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("[check_akl] PASS -- canonical base reproduced, one shared "
          "reference across adapters, support covers the mass, t* carries "
          "the leverage, all adapters present and distinct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
