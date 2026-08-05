#!/usr/bin/env python3
"""Sanity gate for the pofd_ platform-only fresh-data runs.

Per run dir (needs trajectory.pt written by run_pokec_gated_lm.py), checks:
  1. CONFIG    the run really is the pofd design: eps(social)=0, w_plat=1,
               innate_lambda=0, pop=ab, mode=loop, canary=0, single sweep,
               no pop reset. Style/regime are TAG-AWARE:
               pofd_* dirs must be data_regime=replace + pristine_frac=0;
               pofdpf* dirs (data-regime wave, incl. the env2/env3 ports
               pofdpf2_/pofdpfs2_) accumulate + kl_beta=0 +
               style=sft + pristine_frac matching the _pf token;
               pofdbp* dirs (beta x pfrac interior) accumulate + style=sft_kl
               + kl_beta matching the _b token + pristine_frac matching _pf;
               pofdicl* dirs style=frozen, use_lora=0, fresh=False, and
               (icl_k, icl_days, icl_ctx_source) matching the arm token
               (k0/k8live/k32live/k32pri/d5/d10/d15/d30);
               pofddpo* dirs style=dpo, fresh=True, use_lora=1, and
               rlhf_feedback matching the closed/open token (DPO_BETA is
               env-only, verified via the submit configs instead). The prefix
               also covers pofddpon* (noisy wave: DO_SAMPLE=1, DPO_TAU=3 --
               both env-only, same config surface).
  2. NO-PEER   row['accepted'] (peer pairs that moved) == 0 in EVERY round.
  3. EXACT-COPY per round t, with x(t) = innate (t=0) or op_raw[t-1] -- the
               saved opinion is post-social, so with eps_social=0 it IS the
               state the next round starts from and the state the training
               buffer labels carry. The replayed update depends on the run's
               config population_update marker:

               "nested_ai_then_social_v1" (runs from 2026-07-27):
                 h = k innate + (1-k) x(t)
                 z = (1-W) h + W m   if |m - x(t)| < eps_AI   else h
                 x(t+1) = D_eps_social(z)
                 -- gate on the START-OF-ROUND opinion, mixture once per round,
                 peer sweeps last. W=1 returns z = m for every k.

               marker ABSENT (archived runs): the superseded order, gated blend
               first and the innate re-anchor over everyone after it,
                 accepted: op = (1-lam)[(1-W) x(t) + W served] + lam innate
                 rejected: op = (1-lam) x(t) + lam innate
               kept so the archive stays auditable. At W=1, lam=0, eps_social=0
               the two are identical, so those runs pass under either.

               W/lam come from the _w/_l dirname tokens (checked against the
               config); no tokens -> W=1, lam=0. Also cross-checks
               row['contact'] == the gate fraction computed on x(t).
               pofdws* (_es token, eps_social>0): peer moves are RNG-pairwise
               and cannot be replayed offline -- these runs get peer-alive,
               finite in-range opinions, twin telemetry every round and
               twin_raw shape == op_raw shape. Marked peer runs additionally
               get mean(op_raw[t]) == mean(z(t)): every Deffuant move sends a
               pair to its midpoint, so the peer sweep conserves the population
               mean exactly, and the equality fails if the mixture did not run
               BEFORE the peer step.
  4. FRESH     row['n_train'] present on every deploy round, == TRAIN_CAP=723,
               and NEVER grows round-over-round. n_train is logged POST-cap
               (run_pokec_gated_lm.py:1256), so it must hold 723 under
               accumulate too -- the pool grows, the batch never does.
               Skipped for pofdicl* (frozen: nothing trains); optional for
               pofddpo* (pair-based; checked only if logged).

Usage:
  python3 check_pofd_sanity.py <run_dir> [<run_dir> ...]
  python3 check_pofd_sanity.py runs/pokec_gated_lm/pofd_*_fresh_data
Exit 0 iff every run passes every check.
"""
import glob
import os
import re
import sys

import torch

ATOL = 2e-6   # float32 blend arithmetic; W=1 makes the copy exact but the
              # stored tensors round-trip through cpu float32


def check_run(run_dir):
    path = os.path.join(run_dir, "trajectory.pt")
    if not os.path.exists(path):
        return [f"MISSING {path}"]
    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = d["config"]
    traj = d["trajectory"]
    op_raw = d["op_raw"].float()        # [rounds, n] opinions AFTER round t
    pred_raw = d["pred_raw"].float()    # [rounds, n] predictions SERVED in round t
    innate = d["innate"].float()
    errs = []

    # -- 1 CONFIG ------------------------------------------------------------
    name = os.path.basename(run_dir.rstrip("/"))
    is_pfrac = name.startswith("pofdpf")   # covers pofdpf2_/pofdpfs2[smk]_ too
    is_bp = name.startswith("pofdbp")      # covers pofdbpsmk_ too
    is_icl = name.startswith("pofdicl")    # covers pofdiclsmk_ too
    # dpo branch covers pofddpo/pofddpon/pofddposmk AND the W-wave twins
    # pofdwdpo/pofdwdpon/pofdwdposmk (same config surface, W/lam via tokens)
    is_dpo = name.startswith(("pofddpo", "pofdwdpo"))
    # continual-weights fec families (covers pofdws2fcsmk_ too)
    is_cont = name.startswith(("pofdws2fc", "pofdfegdc", "pofdfegpc"))
    # controlled-teacher fe wave: pofdtch_ = one-round transformed-label
    # teacher training runs; pofdtfe_ = the env3 loop arms whose KL ref is a
    # teacher adapter (covers pofdtfesmk_ too)
    is_tch = name.startswith("pofdtch")
    is_tfe = name.startswith("pofdtfe")
    # _w/_l/_es tokens (pofdw*/pofdws* waves): W_PLAT, INNATE_LAMBDA and
    # EPS_SOCIAL move off their pofd defaults (1.0 / 0.0 / 0.0). Absent
    # tokens keep the original W=1 no-peer design.
    m_w = re.search(r"_w(\d+(?:p\d+)?)_", name)
    m_l = re.search(r"_l(\d+(?:p\d+)?)_", name)
    m_es = re.search(r"_es(\d+(?:p\d+)?)_", name)
    want_w = float(m_w.group(1).replace("p", ".")) if m_w else 1.0
    want_l = float(m_l.group(1).replace("p", ".")) if m_l else 0.0
    want_es = float(m_es.group(1).replace("p", ".")) if m_es else 0.0
    is_social = want_es > 0.0
    want = {"eps": want_es, "w_plat": want_w, "innate_lambda": want_l,
            "canary_delta": 0.0,
            "data_regime": "accumulate" if (is_pfrac or is_bp) else "replace",
            "pop_model": "ab",
            "run_mode": "loop", "ab_sweeps": 1, "pop_reset": False,
            "platform_sus_scale": 1.0, "dataset": "movielens"}
    if is_pfrac:
        want.update({"kl_beta": 0.0, "training_style": "sft",
                     "fresh_each_round": True})
    elif is_bp:
        # kl_beta varies per row -- checked against the _b tag token below
        want.update({"training_style": "sft_kl", "fresh_each_round": True})
    elif is_icl:
        # frozen weights: nothing trains, FRESH_EACH_ROUND deliberately 0
        want.update({"kl_beta": 0.0, "training_style": "frozen",
                     "pristine_frac": 0.0, "fresh_each_round": False,
                     "use_lora": 0, "icl_select": "random"})
    elif is_dpo:
        # DPO_BETA is env-only (not in config.json) -- verified via the submit
        # configs, not here. rlhf_feedback IS recorded and tag-checked below.
        want.update({"kl_beta": 0.0, "training_style": "dpo",
                     "pristine_frac": 0.0, "fresh_each_round": True,
                     "use_lora": 1})
    elif is_cont:
        # continual-weights families (fec wave): the adapter persists across
        # rounds (FRESH_EACH_ROUND=0). Data protocol unchanged -- replace,
        # n_train capped -- so _fresh_errs still applies below.
        want.update({"pristine_frac": 0.0, "fresh_each_round": False})
    elif is_tch:
        # one-round teacher training on transformed labels: plain SFT, no
        # KL, natural profiles; delta itself is gated against the tag below.
        # pofdtchr_ = the random-even-split twin (synthetic groups A/B, no
        # prompt feature -- memorization-only carrier).
        want.update({"kl_beta": 0.0, "training_style": "sft",
                     "pristine_frac": 0.0, "fresh_each_round": True,
                     "n_rounds": 1, "log_gender_gaps": True,
                     "profile_drop_cols": [], "profile_permute_cols": []})
        if name.startswith("pofdtchr"):
            want.update({"teacher_label_col": "random_even",
                         "teacher_label_fav": "A"})
        else:
            want.update({"teacher_label_col": "gender",
                         "teacher_label_fav": "M"})
    elif is_tfe:
        # teacher-referenced env3 loops: forward SFT-KL b1, the delta must
        # be OFF (the group signal lives only in the reference weights);
        # ref path + profile controls are arm-gated below. pofdtfer_ = the
        # random-even-split twin.
        want.update({"kl_beta": 1.0, "training_style": "sft_kl",
                     "kl_direction": "forward", "pristine_frac": 0.0,
                     "fresh_each_round": True, "teacher_label_delta": 0.0,
                     "log_gender_gaps": True})
        if name.startswith("pofdtfer"):
            want.update({"teacher_label_col": "random_even",
                         "teacher_label_fav": "A"})
        else:
            want.update({"teacher_label_col": "gender"})
    else:
        want.update({"pristine_frac": 0.0, "fresh_each_round": True})
    # olmo7brom model-slot token (2026-08-05, OLMo Romance fe mirror): the
    # first non-Action ML_TARGET in any wave -- gate the dataset dial and
    # the base model on the token. Non-rom runs keep the original surface
    # (no ml_target gate; older configs predate the key).
    if "_olmo7brom_" in name:
        want["ml_target"] = "Romance"
        if "OLMo-2" not in str(cfg.get("base_model", "")):
            errs.append(f"CONFIG base_model={cfg.get('base_model')!r} "
                        f"(olmo7brom tag wants an OLMo-2 model)")
    for k, v in want.items():
        if cfg.get(k) != v:
            errs.append(f"CONFIG {k}={cfg.get(k)!r} (want {v!r})")
    if is_pfrac or is_bp:
        m = re.search(r"_pf(\d+(?:p\d+)?)_", name)
        if m is None:
            errs.append(f"CONFIG no _pf token in dirname {name!r}")
        else:
            want_pf = float(m.group(1).replace("p", "."))
            got_pf = float(cfg.get("pristine_frac", -1.0))
            if abs(got_pf - want_pf) > 1e-9:
                errs.append(f"CONFIG pristine_frac={got_pf!r} (tag says {want_pf!r})")
    if is_bp:
        m = re.search(r"_b(\d+(?:p\d+)?)_ea", name)
        if m is None:
            errs.append(f"CONFIG no _b token in dirname {name!r}")
        else:
            want_b = float(m.group(1).replace("p", "."))
            got_b = float(cfg.get("kl_beta", -1.0))
            if abs(got_b - want_b) > 1e-9:
                errs.append(f"CONFIG kl_beta={got_b!r} (tag says {want_b!r})")
    if is_icl:
        ICL_ARM_WANT = {"k0": (0, 0, "live"), "k8live": (8, 0, "live"),
                        "k32live": (32, 0, "live"), "k32pri": (32, 0, "pristine"),
                        "k32noai": (32, 0, "noai"),
                        "d5": (0, 5, "live"), "d10": (0, 10, "live"),
                        "d15": (0, 15, "live"), "d30": (0, 30, "live")}
        m = re.search(r"_ea[\dp]+_([a-z0-9]+)_s\d", name)
        arm = m.group(1) if m else None
        if arm not in ICL_ARM_WANT:
            errs.append(f"CONFIG unknown icl arm token in dirname {name!r}")
        else:
            k_w, d_w, src_w = ICL_ARM_WANT[arm]
            got = (cfg.get("icl_k"), cfg.get("icl_days"), cfg.get("icl_ctx_source"))
            if got != (k_w, d_w, src_w):
                errs.append(f"CONFIG icl (k,days,src)={got!r} (arm {arm} wants "
                            f"{(k_w, d_w, src_w)!r})")
        # icl endogenization families (2026-08-04, fei wave): the profile
        # treatment is the ONLY delta vs pofdicls2_ -- gate it. Every other
        # icl family must run with untouched profiles (key absent in configs
        # that predate the profile knobs -> treated as empty).
        want_drop = ["gender"] if name.startswith("pofdicls2gd") else []
        want_perm = ["gender"] if name.startswith("pofdicls2gp") else []
        got_drop = cfg.get("profile_drop_cols") or []
        got_perm = cfg.get("profile_permute_cols") or []
        if got_drop != want_drop:
            errs.append(f"CONFIG profile_drop_cols={got_drop!r} "
                        f"(want {want_drop!r})")
        if got_perm != want_perm:
            errs.append(f"CONFIG profile_permute_cols={got_perm!r} "
                        f"(want {want_perm!r})")
    if is_dpo:
        fb = "open" if "_open_" in name else ("closed" if "_closed_" in name else None)
        if fb is None:
            errs.append(f"CONFIG no closed/open token in dirname {name!r}")
        elif cfg.get("rlhf_feedback") != fb:
            errs.append(f"CONFIG rlhf_feedback={cfg.get('rlhf_feedback')!r} "
                        f"(tag says {fb!r})")
        # dpo_* keys exist only in configs written from 2026-08-03 on (the
        # full-epoch wdpo2e wave onward); older DPO runs skip these gates.
        if "dpo_beta" in cfg:
            m = re.search(r"_db([0-9p]+)_", name)
            if m:
                want_db = float(m.group(1).replace("p", "."))
                if abs(float(cfg["dpo_beta"]) - want_db) > 1e-9:
                    errs.append(f"CONFIG dpo_beta={cfg['dpo_beta']!r} "
                                f"(tag says {want_db})")
        if "dpo_max_steps" in cfg and ("wdpo2e_" in name or "wdpos2e_" in name
                                       or "wdpos2esmk_" in name):
            if int(cfg["dpo_max_steps"]) > 0:
                errs.append(f"CONFIG dpo_max_steps={cfg['dpo_max_steps']!r} "
                            f"(full-epoch family wants <=0)")
    if is_tch:
        m = re.search(r"_d([pm])0p08_", name)
        if m is None:
            errs.append(f"CONFIG no _dp0p08/_dm0p08 token in dirname {name!r}")
        else:
            want_delta = 0.08 if m.group(1) == "p" else -0.08
            got_delta = float(cfg.get("teacher_label_delta", 0.0))
            if abs(got_delta - want_delta) > 1e-9:
                errs.append(f"CONFIG teacher_label_delta={got_delta!r} "
                            f"(tag says {want_delta!r})")
            # transformed-label gate: the saved round-0 training batch must
            # equal clip(innate + delta * (2*1[gender==M] - 1), 0, 1); the x
            # side must stay pristine innate (the transform is label-only)
            gt = d.get("gender_true")
            b0p = os.path.join(run_dir, "round0_batch.pt")
            if gt is None:
                errs.append("TEACHER gender_true missing from trajectory.pt")
            elif not os.path.exists(b0p):
                errs.append("TEACHER round0_batch.pt missing")
            else:
                b0 = torch.load(b0p, map_location="cpu", weights_only=False)
                idx = b0["agent_idx"].long()
                fav = cfg.get("teacher_label_fav", "M")
                sign = torch.tensor([1.0 if g == fav else -1.0 for g in gt])
                want_y = (innate + want_delta * sign).clamp(0.0, 1.0)[idx]
                got_y = b0["y"].squeeze(-1).float()
                dmax = float((got_y - want_y).abs().max())
                if dmax > 1e-6:
                    errs.append(f"TEACHER round-0 labels differ from "
                                f"clip(innate + delta*sign) by max {dmax:.2e}")
                dx = float((b0["x"].squeeze(-1).float() - innate[idx]).abs().max())
                if dx > 0:
                    errs.append(f"TEACHER round-0 x differs from innate "
                                f"(max {dx:.2e}) -- transform must be label-only")
        if any("gg_pred_true" not in r for r in traj):
            errs.append("TEACHER gg_pred_true missing from trajectory rows")
    if is_tfe:
        if name.startswith("pofdtfer"):
            # random-even-split wave: refs are the pofdtchr_ teachers; the
            # neutral arm is REUSED from pofdtfe_ (identical physics), so
            # tneu appears here only if someone runs it anyway. No profile
            # controls exist (nothing displayed marks the synthetic group).
            TFE_REF_SUFFIX = {
                "tpos": "pofdtchr_qwen7b_dp0p08_ea0p4_w0p5_l0p2_s0/round0_adapter",
                "tneu": "pofdw2_qwen7b_b0_ea0p4_w0p5_l0p2_s0_fresh_data/round0_adapter",
                "tneg": "pofdtchr_qwen7b_dm0p08_ea0p4_w0p5_l0p2_s0/round0_adapter",
            }
        else:
            TFE_REF_SUFFIX = {
                "tpos": "pofdtch_qwen7b_dp0p08_ea0p4_w0p5_l0p2_s0/round0_adapter",
                "tneu": "pofdw2_qwen7b_b0_ea0p4_w0p5_l0p2_s0_fresh_data/round0_adapter",
                "tneg": "pofdtch_qwen7b_dm0p08_ea0p4_w0p5_l0p2_s0/round0_adapter",
            }
            TFE_REF_SUFFIX["tposgd"] = TFE_REF_SUFFIX["tposgp"] = TFE_REF_SUFFIX["tpos"]
        TFE_PROF = {"tposgd": (["gender"], []), "tposgp": ([], ["gender"])}
        m = re.search(r"_ea[\dp]+_(t[a-z]+)_w", name)
        arm = m.group(1) if m else None
        if arm not in TFE_REF_SUFFIX:
            errs.append(f"CONFIG unknown tfe arm token in dirname {name!r}")
        else:
            ref = cfg.get("kl_ref_adapter") or ""
            if not ref.endswith(TFE_REF_SUFFIX[arm]):
                errs.append(f"CONFIG kl_ref_adapter={ref!r} (arm {arm} wants "
                            f"...{TFE_REF_SUFFIX[arm]!r})")
            want_drop, want_perm = TFE_PROF.get(arm, ([], []))
            if cfg.get("profile_drop_cols") != want_drop:
                errs.append(f"CONFIG profile_drop_cols="
                            f"{cfg.get('profile_drop_cols')!r} (arm {arm} "
                            f"wants {want_drop!r})")
            if cfg.get("profile_permute_cols") != want_perm:
                errs.append(f"CONFIG profile_permute_cols="
                            f"{cfg.get('profile_permute_cols')!r} (arm {arm} "
                            f"wants {want_perm!r})")
            # gg telemetry every round; displayed-group keys exist except in
            # the drop arm and the random-split wave (nothing displayed
            # marks the group in either)
            need = ["gg_pred_true", "gg_op_true", "gg_twin_true",
                    "gg_teacher", "gg_r2_inc_true"]
            if arm != "tposgd" and not name.startswith("pofdtfer"):
                need += ["gg_pred_disp", "gg_op_disp", "gg_r2_inc_disp"]
            bad = [r["round"] for r in traj if any(k not in r for k in need)]
            if bad:
                errs.append(f"CONFIG gg telemetry keys missing in rounds "
                            f"{bad[:5]}{'...' if len(bad) > 5 else ''}")
    if is_tch or is_tfe:
        m_s = re.search(r"_s(\d+)(?:_|$)", name)
        if m_s and cfg.get("seed") != int(m_s.group(1)):
            errs.append(f"CONFIG seed={cfg.get('seed')!r} "
                        f"(tag says {m_s.group(1)})")
    eps_ai = float(cfg["eps_ai"])

    # -- 2 NO-PEER / PEER-ALIVE ----------------------------------------------
    if is_social:
        # peer step is ON by design: require it actually fired somewhere
        if not any(r.get("accepted", 0) > 0 for r in traj):
            errs.append("PEER-ALIVE eps_social>0 but accepted==0 every round "
                        "(peer step never fired?)")
    else:
        bad = [r["round"] for r in traj if r.get("accepted", 0) != 0]
        if bad:
            errs.append(f"NO-PEER accepted!=0 in rounds {bad[:5]}{'...' if len(bad) > 5 else ''}")

    # -- 3 EXACT-COPY (exact composed blend when W<1 or lam>0) ---------------
    # pofdws* (eps_social>0): peer moves are RNG-pairwise and cannot be
    # replayed offline -- swap the exact-update check for the weaker gate:
    # finite in-range opinions/predictions, the SIMULATED twin present
    # (twin telemetry every round + twin_raw matching op_raw in shape).
    nested = cfg.get("population_update") == "nested_ai_then_social_v1"
    w = float(cfg.get("w_plat", 1.0))
    lam = float(cfg.get("innate_lambda", 0.0))
    if is_social:
        if not (torch.isfinite(op_raw).all() and torch.isfinite(pred_raw).all()):
            errs.append("SOCIAL non-finite opinions or predictions")
        if float(op_raw.min()) < -1e-6 or float(op_raw.max()) > 1 + 1e-6:
            errs.append(f"SOCIAL opinions out of [0,1]: "
                        f"[{float(op_raw.min()):.3f}, {float(op_raw.max()):.3f}]")
        if nested:
            # Peer moves are RNG-pairwise and cannot be replayed offline, but
            # every Deffuant move sends a pair to its midpoint, so the peer
            # sweep CONSERVES the population mean exactly. Under
            # nested_ai_then_social_v1 the peer sweep runs LAST, on z, so
            # mean(op_raw[t]) must equal mean(z(t)) computed from op_raw[t-1].
            # Under the legacy order (peers first) it would not.
            for t in range(op_raw.shape[0]):
                served = pred_raw[t].clamp(0.0, 1.0)
                if not torch.isfinite(served).all():
                    errs.append(f"SOCIAL round {t}: non-finite predictions")
                    continue
                x0 = innate if t == 0 else op_raw[t - 1]
                h = lam * innate + (1.0 - lam) * x0
                gate = (served - x0).abs() < eps_ai
                z = torch.where(gate, (1.0 - w) * h + w * served, h)
                dmean = abs(float(op_raw[t].mean()) - float(z.mean()))
                if dmean > 1e-4:
                    errs.append(f"SOCIAL round {t}: mean(op_raw) differs from "
                                f"mean(z) by {dmean:.2e} -- peer sweep is "
                                f"mean-conserving, so the nested update did NOT "
                                f"run before the peer step (gate on x0, "
                                f"W={w:g} lam={lam:g})")
                logged = traj[t].get("contact")
                if logged is not None and \
                        abs(logged - float(gate.float().mean())) > 1e-6:
                    errs.append(f"SOCIAL round {t}: contact {logged:.6f} != "
                                f"gate-on-x0 frac "
                                f"{float(gate.float().mean()):.6f} -- the AI "
                                f"gate must use the start-of-round opinion")
        no_twin = [r["round"] for r in traj if "twin_mean" not in r]
        if no_twin:
            errs.append(f"SOCIAL twin telemetry missing in rounds {no_twin[:5]}")
        tw = d.get("twin_raw")
        if tw is None or tw.numel() == 0:
            errs.append("SOCIAL twin_raw missing/empty in trajectory.pt "
                        "(pipeline predates the twin_raw patch?)")
        elif tuple(tw.shape) != tuple(op_raw.shape):
            errs.append(f"SOCIAL twin_raw shape {tuple(tw.shape)} != "
                        f"op_raw {tuple(op_raw.shape)}")
        if is_icl:
            # frozen weights: nothing trains, no n_train ever (same skip as
            # the no-peer path below) -- peer-env icl runs (pofdicls2_)
            return errs
        return errs + _fresh_errs(cfg, traj, is_dpo)
    for t in range(op_raw.shape[0]):
        served = pred_raw[t].clamp(0.0, 1.0)
        if not torch.isfinite(served).all():
            errs.append(f"EXACT-COPY round {t}: non-finite predictions")
            continue
        # x_before is the START-OF-ROUND opinion x(t): with eps_social=0 the
        # peer step is inert, so the saved op_raw[t-1] (post-social) IS the
        # state the next round starts from and the state the buffer labels
        # carry. Both versions gate on it.
        x_before = innate if t == 0 else op_raw[t - 1]
        gate = (served - x_before).abs() < eps_ai
        if nested:
            # population_update="nested_ai_then_social_v1":
            #   h = lam innate + (1-lam) x(t)
            #   z = (1-W) h + W m  if |m - x(t)| < eps_AI  else h
            #   x(t+1) = D_eps_social(z) = z here (eps_social == 0)
            h = lam * innate + (1.0 - lam) * x_before
            expect = torch.where(gate, (1.0 - w) * h + w * served, h)
        else:
            # LEGACY (marker absent): gated_blend on x_before, then the innate
            # re-anchor over EVERYONE -- reproduce the archived runs exactly
            x_mid = torch.where(gate, (1.0 - w) * x_before + w * served, x_before)
            expect = (1.0 - lam) * x_mid + lam * innate if lam > 0 else x_mid
        d_acc = (op_raw[t][gate] - expect[gate]).abs()
        if gate.any() and float(d_acc.max()) > ATOL:
            errs.append(f"EXACT-COPY round {t}: accepted opinion != blend "
                        f"(max |diff| {float(d_acc.max()):.2e}, "
                        f"W={w:g} lam={lam:g} violated)")
        d_rej = (op_raw[t][~gate] - expect[~gate]).abs()
        rej_tol = ATOL if lam > 0 else 0.0   # lam=0 -> rejected untouched, exact
        if (~gate).any() and float(d_rej.max()) > rej_tol:
            errs.append(f"EXACT-COPY round {t}: rejected agent off the "
                        f"anchor path (max |diff| {float(d_rej.max()):.2e})")
        logged = traj[t].get("contact")
        if logged is not None and abs(logged - float(gate.float().mean())) > 1e-6:
            errs.append(f"EXACT-COPY round {t}: contact {logged:.6f} != "
                        f"gate-on-x0 frac {float(gate.float().mean()):.6f} "
                        f"(the AI gate must use the start-of-round opinion)")

    # -- 4 FRESH -------------------------------------------------------------
    # icl (frozen): nothing trains, no n_train ever -- skip.
    if is_icl:
        return errs
    return errs + _fresh_errs(cfg, traj, is_dpo)


def _fresh_errs(cfg, traj, is_dpo):
    """FRESH check: n_train == TRAIN_CAP on every deploy round, never grows.
    dpo: n_train is logged by the shared telemetry when present; check it if
    there, but its absence is not an error (the learner consumes pairs)."""
    errs = []
    sizes = [(r["round"], r["n_train"]) for r in traj
             if r.get("is_deploy") and "n_train" in r]
    if not sizes:
        if not is_dpo:
            errs.append("FRESH no n_train logged (pipeline predates the n_train patch?)")
    else:
        cap = int(cfg.get("train_cap") or 0) or 723
        wrong = [(t, n) for t, n in sizes if n != cap]
        if wrong:
            errs.append(f"FRESH n_train != {cap} at {wrong[:5]}")
        grew = [(t, n) for (t, n), (_, p) in zip(sizes[1:], sizes[:-1]) if n > p]
        if grew:
            errs.append(f"FRESH n_train GREW (accumulation?) at {grew[:5]}")
    return errs


def main():
    dirs = []
    for a in sys.argv[1:]:
        dirs += sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a]
    if not dirs:
        print(__doc__)
        sys.exit(2)
    n_fail = 0
    for rd in dirs:
        errs = check_run(rd)
        name = os.path.basename(rd.rstrip("/"))
        if errs:
            n_fail += 1
            print(f"FAIL {name}")
            for e in errs:
                print(f"     - {e}")
        elif re.search(r"_es\d", name):
            print(f"PASS {name}  (peer step live, twin simulated, fresh data only)")
        else:
            print(f"PASS {name}  (no peer updates, exact platform blend, fresh data only)")
    print(f"[check_pofd_sanity] {len(dirs) - n_fail}/{len(dirs)} runs pass")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
