#!/usr/bin/env python3
"""HARD-GATED checker for the REFERENCE-REPLAY pilot (pofdrr_, 2026-08-22).

WHAT THE PILOT DOES. Ordinary SFT retrains each round on the labels
y_i(t) = x_i(t) -- the population's own current opinions. This pilot
replaces a fraction of those labels with a PINNED, FROZEN vector b:

    y_i(t) = x_i(t)   for i in the live set L(t),   |L(t)| = n_live
    y_i(t) = b_i      for every other agent

b is the canonical frozen Qwen2.5 K=D=0 prediction vector -- the entering
model's own answer, computed once and never recomputed. The dial is
q = n_live / N, and q = 1 must reduce EXACTLY to ordinary SFT.

Call this EXPLICIT DATA-SPACE REFERENCE REPLAY. It is not "implicit
anchoring": nothing is inferred, regularized, or penalized. Specific
rows of the training set are literally overwritten with specific frozen
numbers, and this checker reconstructs every one of them.

NOTHING HERE ENCODES A DIRECTION. Whether lowering q moves the
population toward b, away from it, or nowhere is the measurement. Every
check below is about whether the mechanism RAN AS DECLARED.

WHAT IS CHECKED (each is a hard failure)

  CONTRACT     ref_replay_live_idx [T, n_live], ref_replay_labels [T, N],
               ref_replay_ref_vec [N], and config ref_replay_q /
               ref_replay_seed / ref_replay_n_live / ref_replay_ref_run /
               ref_replay_ref_sha256, plus op_raw / pred_raw / innate /
               trajectory. A missing field is a NAMED failure, never a
               traceback: an artifact that predates the contract must
               say so in words.

  LABELS       every label reconstructed and compared elementwise, every
               round: y equals x(t) on the live rows and b on the rest,
               where x(t) = innate at t=0 and op_raw[t-1] afterwards.

  DRIFT        the non-live rows must carry the ORIGINAL b in every
               round -- bit-identical to round 0 and to
               ref_replay_ref_vec. The failure this rules out is
               ACCUMULATION: replaying last round's replayed labels, so
               the "frozen" reference quietly becomes a moving average of
               the loop's own output while every shape still checks out.

  LIVE SET     regenerated from (ref_replay_seed, round) with the
               project's standard stream -- torch.Generator().manual_seed
               (seed + t) then randperm(N) -- and compared elementwise IN
               ORDER against ref_replay_live_idx.

  NESTING      the live set is a PREFIX of that round's permutation, so
               every smaller q's live set is a subset of every larger
               q's at the same round. Checked structurally inside one arm
               (against the smaller rungs of the production ladder) and
               directly across arms when several run dirs are given.
               Without nesting the arms are not a dose ladder, they are
               five unrelated samples.

  LADDER       n_live must be exactly the value the declared q names:
               {.10, .20, .50, .75, 1} -> {72, 145, 362, 542, 723} at
               N = 723. Off-ladder q is refused rather than rounded.

  ORDER        live indices inside [0, N) and unique within a round, and
               -- when the run records a row order -- that order exactly
               arange(N). NOTE what actually pins row j to agent j is the
               elementwise label identity above; a permuted label matrix
               fails LABELS immediately, including at q = 1 (y[j] would
               be x[perm[j]], not x[j]).

  SHAPE        723 rows, and 181 optimizer steps in EVERY round. The
               arms differ in WHICH labels the optimizer sees, never in
               how much compute it spends -- if steps move with q, the
               comparison is confounded and no distance below means
               anything.

  REFERENCE    b is N finite values in [0, 1]; sha256(b) equals the
               run's own ref_replay_ref_sha256; and that sha equals the
               canonical frozen-Qwen constant. When the source run named
               by ref_replay_ref_run is pulled alongside, b must
               re-derive from it bit-exactly.

  SFT-IDENTITY q = 1 must reduce EXACTLY to ordinary SFT: live set = all
               agents, labels == x(t) everywhere, and no row carrying b
               by coincidence of the mechanism. This is the arm every
               other arm is read against, so it is checked as an
               identity, not as an approximation.

  SURFACE      the declared horizon, beta = 1 (forward SFT-KL), gamma =
               k = 1, BOTH gates all_open (a MODE, never the number 1 --
               both gates are strict inequalities), Qwen2.5-7B-Instruct,
               an H100 SKU, and fresh-adapter semantics true. A fresh
               LoRA every round is what makes "the optimizer sees these
               labels" a statement about this round rather than about
               accumulated weights.

  SERVING      predictions finite, in range, one per agent, and every
               generation parsing -- a digit-free generation takes the
               0.5 default and would be read as a real prediction.

Usage:
  python check_ref_replay.py <run_dir> [<run_dir> ...]
  python check_ref_replay.py runs/pokec_gated_lm/pofdrr_*
Exit 0 iff every run passes every check AND the arms nest.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
REPO = HERE.parent.parent.parent

# -- production constants -------------------------------------------------
N_AGENTS = 723            # the paper's MovieLens Action cohort
OPT_STEPS = 181           # ceil(723 / 4): fixed compute in EVERY arm
N_ROUNDS = 100            # the pilot's declared horizon
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
H100_SKU = "H100"         # substring match: the SKU string carries memory
# The ONE canonical frozen Qwen2.5 K=D=0 prediction vector on H100-80GB,
# derived by audit_qwen_mechanism.py and pinned in check_pofd_sanity.py,
# gen_pofd_sweep.py and the mechanism manifest. b IS this vector: the
# whole pilot is about replaying the entering model's own answers, so if
# this hash does not match, the run replayed some OTHER model.
CANON_REF_SHA = (
    "1674ee5f8d833f46de672791d933e1d3bdeefb07484c2d110dec84ce71da30bb")

# q -> n_live at N = 723. Pinned rather than computed: round(0.5 * 723)
# lands on a .5 tie and would depend on the rounding rule.
Q_LADDER_723 = {0.10: 72, 0.20: 145, 0.50: 362, 0.75: 542, 1.0: 723}
LABEL_TOL = 0.0           # labels are COPIES; nothing here is arithmetic
VEC_TOL = 1e-6            # b range check slack


def q_key(q):
    return round(float(q), 6)


def n_live_for(q, n_agents=N_AGENTS):
    """The live-set size the declared q names, or None if q is off-ladder.

    The ladder is pinned at the production N and derived by rounding for
    the toy sizes the tests use. Off-ladder q returns None so the caller
    can REFUSE it instead of silently rounding a typo into a valid arm.
    """
    k = q_key(q)
    if n_agents == N_AGENTS:
        return Q_LADDER_723.get(k)
    if k not in {q_key(x) for x in Q_LADDER_723}:
        return None
    return int(round(k * n_agents))


def sha_vec(t):
    """sha256 over the float32 bytes -- the project's vector fingerprint."""
    a = torch.as_tensor(t).detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def live_perm(seed, t, n_agents):
    """The round's full permutation. The live set is its PREFIX, which is
    what makes the arms nested: the draw never reads n_live, so every arm
    at this (seed, round) walks the same ordering."""
    g = torch.Generator().manual_seed(int(seed) + int(t))
    return torch.randperm(int(n_agents), generator=g)


def canonical_sha_agrees():
    """(ok, message): does check_pofd_sanity pin the same constant?

    The constant lives in several files by design; they must agree or one
    of them is stale. Loaded lazily and defensively -- a sibling module
    that will not import is a NOTE here, not a failure of this run.
    """
    try:
        spec = importlib.util.spec_from_file_location(
            "_chk_pofd_for_rr", str(HERE / "check_pofd_sanity.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:                        # pragma: no cover
        return True, (f"could not cross-check the canonical sha against "
                      f"check_pofd_sanity.py ({type(e).__name__}: {e})")
    other = getattr(mod, "QMECH_CANONICAL_PRED_SHA", None)
    if other != CANON_REF_SHA:
        return False, (f"canonical frozen-Qwen sha disagrees with "
                       f"check_pofd_sanity.QMECH_CANONICAL_PRED_SHA "
                       f"({other!r}) -- the pinned copies have drifted")
    return True, "canonical frozen-Qwen sha agrees with check_pofd_sanity"


def _tensor(d, key):
    """Fetch a tensor field, or None. Empty counts as absent (the runner
    writes an empty tensor when a feature never fired)."""
    v = d.get(key)
    if v is None:
        return None
    try:
        v = torch.as_tensor(v)
    except Exception:
        return None
    return v if v.numel() else None


def check_ref_replay(run_dir, n_agents=N_AGENTS, opt_steps=OPT_STEPS,
                     expect_rounds=N_ROUNDS, base_model=BASE_MODEL,
                     canon_sha=CANON_REF_SHA, require_canon_sha=True,
                     require_surface=True, runs_roots=()):
    """(errs, info) for one pofdrr_ run dir.

    Never raises on a malformed artifact: every missing or wrong-shaped
    field becomes a named entry in errs. info carries what the arm-level
    nesting check needs plus the diagnostics worth printing.
    """
    errs, info = [], {"run_dir": str(run_dir), "name":
                      os.path.basename(str(run_dir).rstrip("/"))}
    name = info["name"]
    path = os.path.join(str(run_dir), "trajectory.pt")
    if not os.path.exists(path):
        return [f"MISSING {path}"], info
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return [f"UNREADABLE {path}: {type(e).__name__}: {e}"], info
    if not isinstance(d, dict):
        return [f"UNREADABLE {path}: not a dict artifact"], info

    if not name.startswith("pofdrr"):
        errs.append(f"TAG {name!r} is not a pofdrr_ run -- this checker "
                    f"gates the reference-replay pilot only")
    is_smoke = name.startswith("pofdrrsmk")

    cfg = d.get("config")
    if not isinstance(cfg, dict):
        return errs + ["CONTRACT config missing from trajectory.pt"], info

    # ---- 0 CONTRACT: presence, before anything is dereferenced ----------
    traj = d.get("trajectory")
    if not isinstance(traj, list) or not traj:
        errs.append("CONTRACT trajectory missing or empty")
        traj = []
    core = {}
    for key in ("op_raw", "pred_raw", "innate"):
        t = _tensor(d, key)
        if t is None:
            errs.append(f"CONTRACT {key} missing/empty from trajectory.pt")
        core[key] = None if t is None else t.float()
    live_idx = _tensor(d, "ref_replay_live_idx")
    labels = _tensor(d, "ref_replay_labels")
    ref_vec = _tensor(d, "ref_replay_ref_vec")
    if live_idx is None:
        errs.append("CONTRACT ref_replay_live_idx missing/empty -- the "
                    "live set is unprovable")
    if labels is None:
        errs.append("CONTRACT ref_replay_labels missing/empty -- the "
                    "labels handed to the learner are unprovable")
    if ref_vec is None:
        errs.append("CONTRACT ref_replay_ref_vec missing/empty -- the "
                    "pinned frozen vector b is unprovable")
    CFG_KEYS = ("ref_replay_q", "ref_replay_seed", "ref_replay_n_live",
                "ref_replay_ref_run", "ref_replay_ref_sha256")
    for k in CFG_KEYS:
        if cfg.get(k) is None:
            errs.append(f"CONTRACT config.{k} missing -- the reference "
                        f"replay cannot be reconstructed without it")

    q = cfg.get("ref_replay_q")
    seed = cfg.get("ref_replay_seed")
    n_live_cfg = cfg.get("ref_replay_n_live")
    info.update({"q": q, "seed": seed, "n_live": n_live_cfg,
                 "n_rounds": None})

    # ---- 1 LADDER: q and n_live must name each other --------------------
    want_n = None
    if q is not None:
        try:
            want_n = n_live_for(float(q), n_agents)
        except (TypeError, ValueError):
            errs.append(f"LADDER ref_replay_q={q!r} is not a number")
        if want_n is None and q is not None:
            errs.append(
                f"LADDER ref_replay_q={q!r} is off the pilot ladder "
                f"{sorted(Q_LADDER_723)} -- refusing to round an "
                f"unrecognised dose into a valid arm")
    if want_n is not None and n_live_cfg is not None:
        if int(n_live_cfg) != want_n:
            errs.append(f"LADDER ref_replay_n_live={n_live_cfg} but q="
                        f"{q} at N={n_agents} names {want_n}")
    # the tag, when it carries a q token, is a third independent statement
    m_q = re.search(r"_q(\d+(?:p\d+)?)_", name)
    if m_q is not None and q is not None:
        tag_q = float(m_q.group(1).replace("p", "."))
        # _q10_ / _q1p0_ both mean the dose, written with or without the
        # decimal point; accept either spelling of the same number
        if abs(tag_q - float(q)) > 1e-9 and \
                abs(tag_q / 100.0 - float(q)) > 1e-9:
            errs.append(f"LADDER tag says q={m_q.group(1)} but config "
                        f"ref_replay_q={q}")

    # ---- 2 SHAPES -------------------------------------------------------
    op = core["op_raw"]
    pred = core["pred_raw"]
    innate = core["innate"]
    n_rounds = None
    if op is not None:
        if op.dim() != 2 or op.shape[1] != n_agents:
            errs.append(f"SHAPE op_raw {tuple(op.shape)} -- want "
                        f"[rounds, {n_agents}]")
        else:
            n_rounds = int(op.shape[0])
            info["n_rounds"] = n_rounds
    if pred is not None and (pred.dim() != 2 or pred.shape[1] != n_agents):
        errs.append(f"SHAPE pred_raw {tuple(pred.shape)} -- want "
                    f"[rounds, {n_agents}] (wrong prediction length)")
    if innate is not None and tuple(innate.shape) != (n_agents,):
        errs.append(f"SHAPE innate {tuple(innate.shape)} -- want "
                    f"[{n_agents}]")
    if labels is not None:
        if labels.dim() != 2 or labels.shape[1] != n_agents:
            errs.append(f"SHAPE ref_replay_labels {tuple(labels.shape)} "
                        f"-- want [rounds, {n_agents}] (row count must be "
                        f"exactly {n_agents}: one label per agent)")
        elif n_rounds is not None and labels.shape[0] != n_rounds:
            errs.append(f"SHAPE ref_replay_labels has "
                        f"{labels.shape[0]} rounds, op_raw has {n_rounds}")
    if live_idx is not None:
        if live_idx.dim() != 2:
            errs.append(f"SHAPE ref_replay_live_idx "
                        f"{tuple(live_idx.shape)} -- want [rounds, n_live]")
        else:
            if n_live_cfg is not None and \
                    live_idx.shape[1] != int(n_live_cfg):
                errs.append(f"SHAPE ref_replay_live_idx holds "
                            f"{live_idx.shape[1]} live ids per round, "
                            f"config ref_replay_n_live={n_live_cfg}")
            if n_rounds is not None and live_idx.shape[0] != n_rounds:
                errs.append(f"SHAPE ref_replay_live_idx has "
                            f"{live_idx.shape[0]} rounds, op_raw has "
                            f"{n_rounds}")
    if ref_vec is not None and tuple(ref_vec.shape) != (n_agents,):
        errs.append(f"SHAPE ref_replay_ref_vec {tuple(ref_vec.shape)} -- "
                    f"want [{n_agents}]")
    if traj and n_rounds is not None and len(traj) != n_rounds:
        errs.append(f"SHAPE trajectory holds {len(traj)} rows, op_raw has "
                    f"{n_rounds} rounds")

    # ---- 3 HORIZON ------------------------------------------------------
    want_rounds = cfg.get("n_rounds")
    if want_rounds is None:
        errs.append("SURFACE config.n_rounds missing")
    elif n_rounds is not None and int(want_rounds) != n_rounds:
        errs.append(f"HORIZON op_raw holds {n_rounds} rounds but config "
                    f"declares n_rounds={want_rounds} (truncated run?)")
    if not is_smoke and want_rounds is not None and \
            int(want_rounds) != expect_rounds:
        errs.append(f"HORIZON n_rounds={want_rounds} -- the pilot's "
                    f"declared horizon is {expect_rounds} rounds "
                    f"(pofdrrsmk_ tags carry the short smokes)")

    # ---- 4 REFERENCE VECTOR b -------------------------------------------
    b = None
    if ref_vec is not None and tuple(ref_vec.shape) == (n_agents,):
        b = ref_vec.float()
        if not bool(torch.isfinite(b).all()):
            errs.append("REFERENCE ref_replay_ref_vec has non-finite "
                        "values")
        elif float(b.min()) < -VEC_TOL or float(b.max()) > 1 + VEC_TOL:
            errs.append(f"REFERENCE ref_replay_ref_vec out of [0,1]: "
                        f"[{float(b.min()):.4f}, {float(b.max()):.4f}]")
        got_sha = sha_vec(b)
        info["ref_sha"] = got_sha
        declared = cfg.get("ref_replay_ref_sha256")
        if declared is not None and got_sha != declared:
            errs.append(f"REFERENCE sha256(b)={got_sha[:16]}... != config "
                        f"ref_replay_ref_sha256={str(declared)[:16]}... "
                        f"-- the saved vector is not the one the config "
                        f"claims was replayed")
        if require_canon_sha and got_sha != canon_sha:
            errs.append(f"REFERENCE sha256(b)={got_sha[:16]}... != the "
                        f"canonical frozen-Qwen vector "
                        f"{canon_sha[:16]}... -- this run replayed some "
                        f"OTHER model's answers")
        # when the source run is pulled alongside, b must re-derive
        src = cfg.get("ref_replay_ref_run")
        if src:
            src_dir = None
            for root in runs_roots:
                cand = Path(root) / str(src)
                if (cand / "trajectory.pt").exists():
                    src_dir = cand
                    break
            if src_dir is not None:
                try:
                    sd = torch.load(src_dir / "trajectory.pt",
                                    map_location="cpu", weights_only=False)
                    spr = torch.as_tensor(sd["pred_raw"]).float()
                    if spr.dim() != 2 or spr.shape[1] != n_agents:
                        errs.append(f"REFERENCE source run {src!r} serves "
                                    f"{tuple(spr.shape)}, incompatible "
                                    f"with this run's {n_agents} agents")
                    elif not bool((spr == spr[0]).all()):
                        errs.append(f"REFERENCE source run {src!r} does "
                                    f"not serve a constant vector -- it "
                                    f"is not a frozen reference")
                    elif not torch.equal(spr[0], b):
                        errs.append(f"REFERENCE b does not re-derive from "
                                    f"its source run {src!r} "
                                    f"({int((spr[0] != b).sum())} agents "
                                    f"differ)")
                except Exception as e:
                    errs.append(f"REFERENCE source run {src!r} unreadable: "
                                f"{type(e).__name__}: {e}")

    # ---- 5 LIVE SET + NESTING (within the arm) --------------------------
    ok_live_shape = (live_idx is not None and live_idx.dim() == 2
                     and n_rounds is not None
                     and live_idx.shape[0] == n_rounds)
    # set False by pass A below: with an out-of-range or repeated id the
    # live MASK cannot be built at all, so the label replay is skipped
    # rather than allowed to raise
    live_ids_ok = ok_live_shape
    if ok_live_shape and seed is not None:
        n_l = int(live_idx.shape[1])
        li = live_idx.long()
        bad_range = bad_uniq = bad_perm = bad_nest = None
        # PASS A: the recorded ids are canonical agents, once each.
        for t in range(n_rounds):
            row = li[t]
            if int(row.min()) < 0 or int(row.max()) >= n_agents:
                bad_range = t
                break
            if int(torch.unique(row).numel()) != n_l:
                bad_uniq = t
                break
        # PASS B: the live set IS the (seed, round) permutation prefix,
        # elementwise and in order.
        if bad_range is None and bad_uniq is None:
            for t in range(n_rounds):
                want = live_perm(seed, t, n_agents)[:n_l]
                if not torch.equal(li[t], want):
                    bad_perm = (t, int((li[t] != want).sum()))
                    break
        # PASS C: NESTING, run on the RECORDED set and INDEPENDENTLY of
        # pass B -- every smaller rung of the ladder must be contained in
        # this arm's live set at this round. Kept separate on purpose: a
        # live set that merely reorders its ids still nests, while one
        # drawn from a different slice of the permutation does not, and
        # the two failures mean different things.
        if bad_range is None and bad_uniq is None:
            for t in range(n_rounds):
                rowset = set(li[t].tolist())
                perm = live_perm(seed, t, n_agents)
                for q_small in sorted(Q_LADDER_723):
                    n_s = n_live_for(q_small, n_agents)
                    if n_s is None or n_s >= n_l:
                        continue
                    if not set(perm[:n_s].tolist()) <= rowset:
                        bad_nest = (t, q_small)
                        break
                if bad_nest:
                    break
        if bad_range is not None:
            errs.append(f"ORDER round {bad_range}: a live index is "
                        f"outside the canonical agent range "
                        f"0..{n_agents - 1}")
        if bad_uniq is not None:
            errs.append(f"ORDER round {bad_uniq}: the live set repeats an "
                        f"agent -- a permutation prefix cannot")
        if bad_perm is not None:
            errs.append(f"LIVE round {bad_perm[0]}: ref_replay_live_idx "
                        f"differs from the (ref_replay_seed={seed}, "
                        f"round) permutation prefix in {bad_perm[1]} of "
                        f"{n_l} positions -- the live set is not the "
                        f"deterministic draw it claims to be")
        if bad_nest is not None:
            errs.append(f"NESTING round {bad_nest[0]}: the q={bad_nest[1]} "
                        f"live set is NOT contained in this arm's live "
                        f"set -- the arms are not a nested dose ladder")
        live_ids_ok = bad_range is None and bad_uniq is None

    # ---- 6 ROW ORDER (when recorded) ------------------------------------
    order = _tensor(d, "ref_replay_row_idx")
    if order is None:
        order = _tensor(d, "sft_order_idx_raw")
    if order is not None and n_rounds is not None:
        want_ord = torch.arange(n_agents)
        if order.dim() != 2 or order.shape[1] != n_agents:
            errs.append(f"ORDER recorded row order {tuple(order.shape)} "
                        f"-- want [rounds, {n_agents}]")
        else:
            off = [t for t in range(min(n_rounds, order.shape[0]))
                   if not torch.equal(order[t].long(), want_ord)]
            if off:
                errs.append(f"ORDER recorded row order is not the "
                            f"canonical 0..{n_agents - 1} in rounds "
                            f"{off[:5]} -- ref_replay_labels row j must "
                            f"be agent j")

    # ---- 7 LABELS + DRIFT ------------------------------------------------
    ok_lab = (labels is not None and labels.dim() == 2
              and labels.shape[1] == n_agents and n_rounds is not None
              and labels.shape[0] == n_rounds and op is not None
              and innate is not None and b is not None and ok_live_shape
              and live_ids_ok)
    if ok_lab:
        y = labels.float()
        n_l = int(live_idx.shape[1])
        li = live_idx.long()
        bad_live = bad_ref = bad_drift = None
        prev_mask = None
        for t in range(n_rounds):
            x_t = innate if t == 0 else op[t - 1]
            mask = torch.zeros(n_agents, dtype=torch.bool)
            mask[li[t]] = True
            # DRIFT first, and stated WITHOUT reference to b: rows that
            # are non-live in BOTH this round and the last must carry
            # bit-identical labels. That is what "the reference does not
            # move" means operationally, and it names accumulation --
            # replaying the loop's own previous output -- independently
            # of whether the stored b is itself intact.
            if prev_mask is not None:
                both = (~mask) & (~prev_mask)
                if bool(both.any()) and \
                        not torch.equal(y[t][both], y[t - 1][both]):
                    bad_drift = (t, float((y[t][both]
                                           - y[t - 1][both]).abs().max()),
                                 int(both.sum()))
                    break
            prev_mask = mask
            # live rows carry the population's own current opinions
            dl = (y[t][mask] - x_t[mask]).abs()
            if bool(mask.any()) and float(dl.max()) > LABEL_TOL:
                bad_live = (t, float(dl.max()), int((dl > LABEL_TOL).sum()))
                break
            # non-live rows carry the PINNED reference, not a recent copy
            dr = (y[t][~mask] - b[~mask]).abs()
            if bool((~mask).any()) and float(dr.max()) > LABEL_TOL:
                bad_ref = (t, float(dr.max()), int((dr > LABEL_TOL).sum()))
                break
        if bad_drift is not None:
            errs.append(f"DRIFT round {bad_drift[0]}: labels changed on "
                        f"{bad_drift[2]} rows that were non-live in this "
                        f"round AND the last (max |diff| "
                        f"{bad_drift[1]:.3e}) -- the replayed reference "
                        f"is accumulating the loop's own output instead "
                        f"of staying pinned")
        if bad_live is not None:
            errs.append(f"LABELS round {bad_live[0]}: {bad_live[2]} live "
                        f"rows do not carry x(t) (max |diff| "
                        f"{bad_live[1]:.3e}) -- a live agent's label must "
                        f"be its own current opinion")
        if bad_ref is not None:
            errs.append(f"LABELS round {bad_ref[0]}: {bad_ref[2]} "
                        f"non-live rows do not carry b (max |diff| "
                        f"{bad_ref[1]:.3e})")
        if bad_drift is not None:
            errs.append(f"DRIFT round {bad_drift[0]}: the non-live labels "
                        f"moved off the ORIGINAL frozen vector by "
                        f"{bad_drift[1]:.3e} -- this is accumulation "
                        f"(replaying the loop's own output), not a "
                        f"pinned reference")

        # ---- 8 SFT IDENTITY at q = 1 ------------------------------------
        if q is not None and abs(float(q) - 1.0) < 1e-12:
            if n_l != n_agents:
                errs.append(f"SFT-IDENTITY q=1 but the live set holds "
                            f"{n_l} of {n_agents} agents")
            else:
                bad_q1 = [t for t in range(n_rounds)
                          if not torch.equal(
                              y[t], innate if t == 0 else op[t - 1])]
                if bad_q1:
                    errs.append(
                        f"SFT-IDENTITY q=1 must reduce EXACTLY to "
                        f"ordinary SFT, but the labels differ from x(t) "
                        f"in rounds {bad_q1[:5]} -- the q=1 arm is the "
                        f"reference every other arm is read against")

    # ---- 9 OPTIMIZER STEPS (fixed compute) -------------------------------
    dose = d.get("sft_dose") or []
    step_rows = []
    if dose:
        step_rows = [(r.get("round"), r.get("global_step")) for r in dose]
    elif traj:
        for r in traj:
            v = r.get("opt_steps", r.get("global_step"))
            if v is not None:
                step_rows.append((r.get("round"), v))
    if not step_rows:
        # DERIVED FALLBACK, and it is named as such in the notes.
        # sft_dose telemetry postdates 2026-08-21, so the REUSED q=1 arm
        # (the completed QWU b0 cell) legitimately has none -- requiring
        # measured steps there would reject the one arm this design gets
        # for free. Step count is a deterministic function of the config:
        # ceil(rows / batch) * epochs. Derive it, check it, and record
        # that it was DERIVED rather than observed, so a reader can tell
        # a measured claim from an arithmetic one.
        rows = cfg.get("n_labeled")
        bs = cfg.get("sft_batch_size")
        ep = cfg.get("sft_epochs")
        if not all(isinstance(v, int) and v > 0 for v in (rows, bs, ep)):
            errs.append("STEPS no per-round optimizer-step provenance and "
                        "no config to derive it from (n_labeled / "
                        "sft_batch_size / sft_epochs) -- 'fixed compute "
                        "across arms' is the load-bearing claim of this "
                        "design and cannot be assumed")
        else:
            derived = -(-rows // bs) * ep      # ceil division
            info["steps_source"] = "derived"
            info["steps_derived"] = derived
            info["steps_note"] = (
                f"derived {derived} from config (ceil({rows}/{bs})x{ep}); "
                f"this run predates sft_dose telemetry, so compute is "
                f"checked by ARITHMETIC, not measurement")
            if derived != opt_steps:
                errs.append(f"STEPS derived optimizer steps {derived} != "
                            f"{opt_steps} -- compute must not move with q")
    else:
        info["steps_source"] = "measured"
    if step_rows:
        bad_steps = [(t, s) for t, s in step_rows if int(s) != opt_steps]
        if bad_steps:
            errs.append(f"STEPS optimizer steps != {opt_steps} in "
                        f"{len(bad_steps)} round(s), first "
                        f"{bad_steps[:3]} -- compute must not move with q")
        if n_rounds is not None and len(step_rows) != n_rounds:
            errs.append(f"STEPS step provenance covers {len(step_rows)} "
                        f"rounds, the run holds {n_rounds}")

    # ---- 10 SURFACE ------------------------------------------------------
    if require_surface:
        if float(cfg.get("kl_beta", -1)) != 1.0:
            errs.append(f"SURFACE kl_beta={cfg.get('kl_beta')!r} -- the "
                        f"pilot surface is beta=1")
        if cfg.get("training_style") != "sft_kl":
            errs.append(f"SURFACE training_style="
                        f"{cfg.get('training_style')!r} -- want 'sft_kl' "
                        f"at beta=1")
        if cfg.get("kl_direction") != "forward":
            errs.append(f"SURFACE kl_direction="
                        f"{cfg.get('kl_direction')!r} -- forward KL is "
                        f"the canonical direction for new waves")
        if float(cfg.get("innate_lambda", -1)) != 1.0:
            errs.append(f"SURFACE innate_lambda="
                        f"{cfg.get('innate_lambda')!r} -- the pilot "
                        f"surface is gamma = k = 1")
        for key in ("ai_gate_mode", "peer_gate_mode"):
            if cfg.get(key) != "all_open":
                errs.append(
                    f"SURFACE {key}={cfg.get(key)!r} -- both gates must "
                    f"be all_open. Open is a MODE, never the number 1: "
                    f"both gates are strict inequalities, so eps=1 "
                    f"REJECTS a pair at distance exactly 1")
        if float(cfg.get("eps", 0.0)) <= 0:
            errs.append(f"SURFACE eps={cfg.get('eps')!r} -- eps_social=0 "
                        f"is the NO-PEER condition and cannot also mean "
                        f"an open peer channel")
        if cfg.get("base_model") != base_model:
            errs.append(f"SURFACE base_model={cfg.get('base_model')!r} "
                        f"-- want {base_model!r}")
        gpu = (cfg.get("hardware") or {}).get("gpu_name") or ""
        if H100_SKU not in gpu:
            errs.append(f"SURFACE ran on {gpu or 'unknown GPU'}, not an "
                        f"{H100_SKU} -- the frozen reference vector is "
                        f"only bit-reproducible on its own SKU")
        if cfg.get("fresh_each_round") is not True:
            errs.append(f"SURFACE fresh_each_round="
                        f"{cfg.get('fresh_each_round')!r} -- fresh-adapter "
                        f"semantics are mandatory: without them the "
                        f"round's labels are not what the served vector "
                        f"is a function of")

    # ---- 11 SERVING ------------------------------------------------------
    if pred is not None and pred.dim() == 2:
        if not bool(torch.isfinite(pred).all()):
            errs.append("SERVING non-finite predictions (parse-fail NaN)")
        elif float(pred.min()) < -VEC_TOL or float(pred.max()) > 1 + VEC_TOL:
            errs.append(f"SERVING predictions out of [0,1]: "
                        f"[{float(pred.min()):.4f}, "
                        f"{float(pred.max()):.4f}]")
    pf = [(r.get("round"), r.get("parse_fail_frac")) for r in traj
          if r.get("parse_fail_frac") is not None]
    rg_path = os.path.join(str(run_dir), "raw_gen_log.json.gz")
    have_raw = os.path.exists(rg_path)
    if not pf and not have_raw:
        errs.append("SERVING no parse provenance (trajectory "
                    "parse_fail_frac or raw_gen_log.json.gz) -- a "
                    "digit-free generation parses to the 0.5 default and "
                    "would be read as a real prediction")
    bad_pf = [(t, v) for t, v in pf if float(v) != 0.0]
    if bad_pf:
        errs.append(f"SERVING parse failures in {len(bad_pf)} round(s), "
                    f"first {bad_pf[:3]} -- every generation must parse")
    if have_raw:
        try:
            with gzip.open(rg_path, "rt") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    row = json.loads(ln)
                    raw = row.get("raw") or []
                    if len(raw) != n_agents:
                        errs.append(f"SERVING round {row.get('round')}: "
                                    f"{len(raw)} raw generations, want "
                                    f"{n_agents}")
                        break
                    nod = sum(1 for x in raw
                              if re.search(r"\d", str(x)) is None)
                    if nod:
                        errs.append(
                            f"SERVING round {row.get('round')}: {nod} of "
                            f"{n_agents} generations contain no digit -- "
                            f"those parse to the 0.5 default")
                        break
        except (OSError, json.JSONDecodeError) as e:
            errs.append(f"SERVING raw_gen_log.json.gz unreadable: "
                        f"{type(e).__name__}: {e}")

    info["live_idx"] = live_idx.long() if live_ids_ok else None
    return errs, info


def check_arms(infos, n_agents=N_AGENTS):
    """Cross-arm NESTING: at every round, a smaller q's live set must be
    a subset of every larger q's. Only arms whose live sets survived the
    per-run checks are compared (a broken arm already failed loudly)."""
    errs = []
    usable = [i for i in infos
              if i.get("live_idx") is not None and i.get("q") is not None]
    usable.sort(key=lambda i: float(i["q"]))
    for a, bnxt in zip(usable, usable[1:]):
        if float(a["q"]) == float(bnxt["q"]):
            continue
        la, lb = a["live_idx"], bnxt["live_idx"]
        rounds = min(la.shape[0], lb.shape[0])
        for t in range(rounds):
            if not set(la[t].tolist()) <= set(lb[t].tolist()):
                errs.append(
                    f"NESTING-ARMS round {t}: q={a['q']} ({a['name']}) "
                    f"live set is NOT a subset of q={bnxt['q']} "
                    f"({bnxt['name']}) -- the arms are not nested, so "
                    f"differences between them are not a dose response")
                break
        if a.get("seed") is not None and bnxt.get("seed") is not None \
                and a["seed"] != bnxt["seed"]:
            errs.append(f"NESTING-ARMS {a['name']} seed={a['seed']} vs "
                        f"{bnxt['name']} seed={bnxt['seed']} -- nested "
                        f"arms must share one live-set stream")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--n-agents", type=int, default=N_AGENTS)
    ap.add_argument("--opt-steps", type=int, default=OPT_STEPS)
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    ap.add_argument("--runs-root", action="append", default=None,
                    help="extra root(s) to resolve ref_replay_ref_run in")
    args = ap.parse_args(argv)

    roots = [Path(r) for r in (args.runs_root or [])] + \
            [REPO / "runs" / "pokec_gated_lm",
             REPO / "notes" / "pofd" / "cluster"]
    ok_sha, sha_msg = canonical_sha_agrees()
    print(f"[check_rr] {sha_msg}")

    total, infos = 0, []
    bad = 0
    for rd in sorted(args.run_dirs):
        errs, info = check_ref_replay(
            rd, n_agents=args.n_agents, opt_steps=args.opt_steps,
            expect_rounds=args.rounds, runs_roots=roots)
        if not ok_sha:
            errs = [f"REFERENCE {sha_msg}"] + errs
        infos.append(info)
        total += 1
        nm = info["name"]
        if errs:
            bad += 1
            print(f"[check_rr] FAIL {nm} ({len(errs)})")
            for e in errs[:25]:
                print(f"    - {e}")
        else:
            print(f"[check_rr] pass {nm} (q={info.get('q')}, "
                  f"n_live={info.get('n_live')}, "
                  f"rounds={info.get('n_rounds')})")
    arm_errs = check_arms(infos, n_agents=args.n_agents)
    for e in arm_errs:
        print(f"[check_rr]     - {e}")
    if bad or arm_errs:
        print(f"[check_rr] {bad}/{total} run(s) FAILED"
              + (f" + {len(arm_errs)} cross-arm failure(s)"
                 if arm_errs else ""), file=sys.stderr)
        return 1
    print(f"[check_rr] PASS -- {total} run(s): every label reconstructed, "
          f"live sets deterministic and nested, b canonical and pinned, "
          f"compute fixed across arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
