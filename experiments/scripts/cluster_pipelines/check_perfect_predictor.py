#!/usr/bin/env python3
"""Gate for the perfect-prediction oracle artifacts (2026-08-20).

Every artifact written by sim_perfect_predictor.py (and the offline
frozen replays from replay_frozen_offline.py) is RECONSTRUCTED from its
own recorded config and compared byte-for-byte against what is stored.
Nothing here trusts a stored number it did not recompute.

Per artifact:
  * op_raw / twin_raw / pred_raw reproduce byte-for-byte from the config
  * perfect prediction really is perfect: pred_raw[t] equals the
    START-OF-ROUND state exactly (innate at t=0, op_raw[t-1] after)
  * the innate vector, adjacency and platform susceptibility hash to the
    recorded values (the environment did not drift under the artifact)
  * every value finite and inside [0, 1]
  * the declared horizon matches the stored length
  * the AI gate is open for every agent every round (|m - x| = 0)

Structural checks across artifacts:
  * beta_eff = 1 - (1 - W) k, so (k=.2, W=.5) and (k=1, W=.9) are the
    SAME pre-peer map, and so are (k=0, W=.5) and (k=1, W=1)
  * the W=1, k=1, all-open oracle preserves the population mean and
    reaches SD < 1e-5 by round 300

ON "BYTE-IDENTICAL" FOR THE IDENTITY PAIRS -- read this, it is a real
result and not a caveat to skim.

  beta_eff = 1 pair, (k=0, W=.5) vs (k=1, W=1): byte-identical, in
  float32 AND float64, at every peer gate. Both reduce z to EXACTLY x
  (k=0 gives h = x and then .5x + .5x = x exactly in binary; k=1, W=1
  gives z = 0*innate + 1*x), so there is no rounding to disagree about.

  beta_eff = .9 pair, (k=.2, W=.5) vs (k=1, W=.9): exact ALGEBRA, but
  NOT byte-identical in floating point. The two parameterizations
  evaluate the same linear map through different products --
  0.5*(0.2 i + 0.8 x) + 0.5 x versus 0.1 i + 0.9 x -- and 0.1/0.2/0.8/0.9
  are not binary-exact, so they land 1 ulp apart (1.19e-07 in float32).
  Measured consequences at seed 0, 30 rounds:
      no peers (es=0):     stays 1 ulp for the whole trajectory
      es=.2 or es=1:       stays ~2e-07
      es=.05:              grows to 4.9e-02 IN THE FIRST ROUND and
                           ~1.1e-01 by round 30
  The last line is the bounded-confidence nonlinearity doing exactly
  what Part E is about: a 1-ulp perturbation flips a pair's acceptance
  at the confidence boundary, and the flip propagates. In float64 the
  same pair agrees to <= 2.2e-08 with peers and byte-exactly without.

  So this checker asserts the identity where it is a statement about
  arithmetic -- on the PRE-PEER map, at every round of the trajectory,
  to within a few ulp -- and REPORTS the realized trajectory divergence
  rather than pretending it is zero. Claiming byte-identity for this
  pair would be false.

Usage:
  python check_perfect_predictor.py [ARTIFACT ...]
  python check_perfect_predictor.py --dir notes/pofd/perfect_prediction
  python check_perfect_predictor.py --structural      # identities only
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
sys.path.insert(0, str(HERE))
import _gated_pop as gp                                    # noqa: E402

# a few ulp of float32 at opinion scale; the pre-peer identity is exact
# algebra evaluated through two different product orders
IDENTITY_ULP_TOL = 1e-6
CONSENSUS_SD = 1e-5
CONSENSUS_ROUNDS = 300
CONSENSUS_MEAN_TOL = 1e-6


def _sha(t):
    return hashlib.sha256(
        t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def check_artifact(path, setup):
    """Errors for one oracle / frozen-replay artifact."""
    errs = []
    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = d["config"]
    op, tw, pr = d["op_raw"], d["twin_raw"], d["pred_raw"]
    name = os.path.basename(path)
    kind = cfg.get("platform", "?")

    # -- environment provenance ------------------------------------------
    for key, tensor in (("innate_sha256", setup["innate"]),
                        ("adj_sha256", (setup["adj"] > 0).to(torch.uint8)),
                        ("platform_sus_sha256", setup["platform_sus"])):
        if key in cfg and cfg[key] != _sha(tensor):
            errs.append(f"{name}: {key} mismatch -- the environment drifted "
                        f"under this artifact")
    if not torch.equal(d["innate"].float(), setup["innate"].float()):
        errs.append(f"{name}: stored innate vector != the loader's")

    # -- horizon ---------------------------------------------------------
    # A declared horizon that does not match the stored one makes every
    # later comparison ill-defined (and would raise a shape error rather
    # than report a finding), so it is fatal for this artifact and we
    # return immediately instead of limping on.
    r = int(cfg["rounds"])
    fatal = []
    for key, t in (("op_raw", op), ("twin_raw", tw), ("pred_raw", pr)):
        if t.shape[0] != r:
            fatal.append(f"{name}: {key} has {t.shape[0]} rounds, config "
                         f"declares {r}")
    if op.shape[1] != int(setup["n"]):
        fatal.append(f"{name}: {op.shape[1]} agents != {int(setup['n'])}")
    if fatal:
        return errs + fatal

    # -- finite and in range ---------------------------------------------
    for key, t in (("op_raw", op), ("twin_raw", tw), ("pred_raw", pr)):
        if not bool(torch.isfinite(t).all()):
            errs.append(f"{name}: {key} has non-finite values")
        elif float(t.min()) < -1e-6 or float(t.max()) > 1 + 1e-6:
            errs.append(f"{name}: {key} out of [0,1] "
                        f"[{float(t.min()):.4f}, {float(t.max()):.4f}]")

    # -- perfect prediction really is perfect ----------------------------
    if kind == "perfect_prediction":
        for t in range(op.shape[0]):
            x0 = d["innate"] if t == 0 else op[t - 1]
            if not torch.equal(pr[t], x0):
                nd = int((pr[t] != x0).sum())
                errs.append(f"{name}: round {t} served vector != the "
                            f"start-of-round state ({nd} agents differ) -- "
                            f"this is not perfect prediction")
                break
    elif kind == "frozen_offline_replay":
        # the frozen served vector must be the SAME constant every round
        if not bool((pr == pr[0]).all()):
            errs.append(f"{name}: frozen replay served vector is not "
                        f"constant across rounds")
        sha = _sha(pr[0])
        if cfg.get("frozen_pred_sha256") and sha != cfg["frozen_pred_sha256"]:
            errs.append(f"{name}: replayed served vector sha256 != the "
                        f"recorded source hash")
    else:
        errs.append(f"{name}: unknown platform {kind!r}")

    # -- byte-for-byte replay --------------------------------------------
    served_fn = None
    if kind == "frozen_offline_replay":
        const = pr[0].clone()
        served_fn = (lambda x, t: const)
    op2, tw2, pr2 = PP.simulate(
        setup, innate_k=cfg["innate_k"], w_plat=cfg["w_plat"],
        eps_social=cfg["eps_social"], eps_ai=cfg["eps_ai"],
        rounds=r, seed=cfg["seed"],
        ai_gate_mode=cfg["ai_gate_mode"],
        peer_gate_mode=cfg["peer_gate_mode"],
        ab_sweeps=cfg["ab_sweeps"], gamma=cfg["gamma_bias"],
        served_fn=served_fn,
        require_open_gate=(kind == "perfect_prediction"))
    for key, a, b in (("op_raw", op, op2), ("twin_raw", tw, tw2),
                      ("pred_raw", pr, pr2)):
        if not torch.equal(a, b):
            errs.append(f"{name}: {key} does NOT reproduce byte-for-byte "
                        f"from its own config (max |diff| "
                        f"{float((a - b).abs().max()):.3e})")

    # -- the AI gate is open for every agent, every round ----------------
    if kind == "perfect_prediction":
        w_agent = (cfg["w_plat"] * setup["platform_sus"]).clamp(0.0, 1.0)
        for t in range(op.shape[0]):
            x0 = d["innate"] if t == 0 else op[t - 1]
            g = gp.ai_gate(pr[t], x0, cfg["eps_ai"], cfg["ai_gate_mode"])
            if not bool(g.all()):
                errs.append(f"{name}: round {t} AI gate closed for "
                            f"{int((~g).sum())} agents under perfect "
                            f"prediction")
                break
        del w_agent

    # -- beta_eff bookkeeping --------------------------------------------
    if kind == "perfect_prediction":
        want_b = PP.beta_eff(cfg["innate_k"], cfg["w_plat"])
        if abs(float(cfg.get("beta_eff", -9)) - want_b) > 1e-12:
            errs.append(f"{name}: recorded beta_eff "
                        f"{cfg.get('beta_eff')!r} != 1-(1-W)k = {want_b}")
    return errs


def _prepeer(setup, k, w, x, dtype=torch.float32):
    """One pre-peer perfect-prediction step at state x."""
    innate = setup["innate"].to(dtype)
    w_agent = (w * setup["platform_sus"]).clamp(0.0, 1.0).to(dtype)
    z, gate = gp.nested_presocial_update(x, x.clone(), innate, k, w_agent,
                                         1.0, gate_mode="threshold")
    assert bool(gate.all())
    return z


def structural_checks(setup, verbose=True):
    """The two beta_eff identities and the W=1 consensus assertion."""
    errs = []

    PAIRS = [((0.2, 0.5), (1.0, 0.9), "beta_eff=0.9"),
             ((0.0, 0.5), (1.0, 1.0), "beta_eff=1.0")]
    for (k1, w1), (k2, w2), label in PAIRS:
        b1, b2 = PP.beta_eff(k1, w1), PP.beta_eff(k2, w2)
        if abs(b1 - b2) > 1e-12:
            errs.append(f"IDENTITY {label}: beta_eff {b1} != {b2}")
            continue
        # (a) the ARITHMETIC claim: the two parameterizations are the same
        # pre-peer map. Checked at every state the trajectory actually
        # visits, not just at innate, and to a few ulp -- see the module
        # docstring for why byte-identity is false for the .9 pair.
        for es in (0.0, 0.05, 0.2, 1.0):
            a, _, _ = PP.simulate(setup, innate_k=k1, w_plat=w1,
                                  eps_social=es, eps_ai=1.0, rounds=30,
                                  seed=0)
            worst = 0.0
            for t in range(a.shape[0]):
                x = setup["innate"] if t == 0 else a[t - 1]
                z1 = _prepeer(setup, k1, w1, x)
                z2 = _prepeer(setup, k2, w2, x)
                worst = max(worst, float((z1 - z2).abs().max()))
            if worst > IDENTITY_ULP_TOL:
                errs.append(f"IDENTITY {label} es={es:g}: pre-peer maps "
                            f"differ by {worst:.3e} > {IDENTITY_ULP_TOL:.0e} "
                            f"-- this is NOT float rounding")
            # (b) the REALIZED trajectory divergence, reported not asserted
            b, _, _ = PP.simulate(setup, innate_k=k2, w_plat=w2,
                                  eps_social=es, eps_ai=1.0, rounds=30,
                                  seed=0)
            gap = float((a - b).abs().max())
            if verbose:
                print(f"[identity] {label} es={es:<5g} pre-peer "
                      f"{worst:.2e} | realized trajectory {gap:.2e}"
                      + ("  <- bounded-confidence amplification"
                         if gap > 1e-3 else ""))
        # the beta_eff=1 pair IS byte-identical; assert that, since a
        # regression there would mean the operator stopped reducing to x
        if label == "beta_eff=1.0":
            for es in (0.0, 0.05, 0.2, 1.0):
                a, _, _ = PP.simulate(setup, innate_k=k1, w_plat=w1,
                                      eps_social=es, eps_ai=1.0, rounds=30,
                                      seed=0)
                b, _, _ = PP.simulate(setup, innate_k=k2, w_plat=w2,
                                      eps_social=es, eps_ai=1.0, rounds=30,
                                      seed=0)
                if not torch.equal(a, b):
                    errs.append(
                        f"IDENTITY beta_eff=1 es={es:g}: (k=0,W=.5) and "
                        f"(k=1,W=1) are NOT byte-identical (max "
                        f"{float((a - b).abs().max()):.3e}); both reduce z "
                        f"to exactly x, so this is a real regression")

    # -- consensus at the Wu boundary ------------------------------------
    op, _, _ = PP.simulate(setup, innate_k=1.0, w_plat=1.0, eps_social=0.2,
                           eps_ai=1.0, rounds=CONSENSUS_ROUNDS, seed=0,
                           ai_gate_mode="all_open",
                           peer_gate_mode="all_open")
    ini_mean = float(setup["innate"].mean())
    fin = op[-1]
    drift = abs(float(fin.mean()) - ini_mean)
    sd = float(fin.std())
    rng = float(fin.max() - fin.min())
    if drift > CONSENSUS_MEAN_TOL:
        errs.append(f"CONSENSUS mean drifted by {drift:.3e} > "
                    f"{CONSENSUS_MEAN_TOL:.0e} -- perfect prediction at "
                    f"W=1 with midpoint peers must preserve the mean")
    if sd >= CONSENSUS_SD:
        errs.append(f"CONSENSUS SD {sd:.3e} not below {CONSENSUS_SD:.0e} "
                    f"by round {CONSENSUS_ROUNDS}")
    if verbose:
        print(f"[consensus] k=1 W=1 all-open, round {CONSENSUS_ROUNDS}: "
              f"mean {float(fin.mean()):.8f} (drift {drift:.2e}) "
              f"sd {sd:.3e} range {rng:.3e}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", nargs="*", type=Path)
    ap.add_argument("--dir", type=Path, action="append", default=None,
                    help="check every .pt in this directory")
    ap.add_argument("--structural", action="store_true",
                    help="run ONLY the identity + consensus checks")
    ap.add_argument("--skip-structural", action="store_true")
    args = ap.parse_args()

    setup = PP.extract_loader()(
        REPO / "experiments/data/movielens/ml-100k", "Action")

    paths = list(args.artifacts)
    for dpath in (args.dir or []):
        paths.extend(sorted(Path(dpath).glob("*.pt")))
    if args.structural:
        paths = []

    all_errs = []
    for p in paths:
        errs = check_artifact(p, setup)
        print(f"[check_pp] {'FAIL' if errs else 'PASS'}  {p.name}")
        for e in errs:
            print(f"[check_pp]   {e}")
        all_errs.extend(errs)

    if not args.skip_structural:
        all_errs.extend(structural_checks(setup))

    if all_errs:
        print(f"[check_pp] {len(all_errs)} FAILURE(S)", file=sys.stderr)
        return 1
    print(f"[check_pp] OK -- {len(paths)} artifact(s), structural checks "
          f"{'skipped' if args.skip_structural else 'passed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
