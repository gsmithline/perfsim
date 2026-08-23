#!/usr/bin/env python3
"""Generate + verify the pofd_ sweep configs (platform-only closed loop, fresh data).

Design (2026-07-22): every social channel is off and the platform channel is
total -- eps_social=0 (ab_sweep accepts pairs only if |xi-xj| < eps, so 0 means
NO peer updates ever), W_PLAT=1.0 with all-ones movielens platform_sus (gated
agents' opinions become EXACTLY the served prediction), FRESH_EACH_ROUND=1 +
DATA_REGIME=replace (the round-t retrain starts from the pristine adapter and
sees ONLY the single most recent round of data -- no replay, no accumulation
through weights). Grid: beta {0,0.1,0.2,0.5,1} x eps_AI {0.05,0.1,0.2,0.4}.
beta=0 -> style sft, beta>0 -> sft_kl (project convention).

ACTIVE_MODELS / SEEDS control scale (2026-07-22 scope: qwen7b x seed 0 = 20
jobs). To extend, add entries there, rerun, and clone the model's .sub from
at_pofd_qwen7b.sub with the env deltas noted in MODELS.

pofdpf_ data-regime wave (2026-07-22): the DATA-side regularization dial,
qwen7b, beta=0 (KL off -> isolates the data knob). DATA_REGIME=accumulate +
PRISTINE_FRAC=pf: each 723-row retrain batch pins round(pf*723) rows to the
round-0 innate seed and draws the rest uniformly from the accumulated loop
buffer (mix_pristine_data). pf=1 -> true pristine (every retrain sees ONLY
innate data; the data-space analog of beta->inf), pf=0 -> plain accumulate
(innate share decays like 1/t). The already-run replace beta=0 row is the
no-regularization endpoint beyond pf=0.

Usage:
  python3 gen_pofd_sweep.py            # write configs_pofd_<model>.txt + smoke
  python3 gen_pofd_sweep.py --verify   # assert on-disk configs match the grid
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# short name -> (BASE_MODEL, env deltas to note in the model's .sub)
MODELS = {
    "qwen7b":   ("Qwen/Qwen2.5-7B-Instruct", "none (reference sub)"),
    "llama8b":  ("meta-llama/Llama-3.1-8B-Instruct", "HF_HUB_OFFLINE=1"),
    "gemma12b": ("google/gemma-3-12b-it",
                 "ENV_NAME=opdyn_gemma HF_HUB_OFFLINE=1 PPL_BATCH=8"),
    "olmo7b":   ("allenai/OLMo-2-1124-7B-Instruct",
                 "HF_HOME=/lustre/fast/fast/gsmithline/hf_cache "
                 "HF_HUB_OFFLINE=1 PPL_BATCH=16, request_memory 160G disk 60G"),
    # 4th model, promoted 2026-08-12 off the mlatZ_cand screen (prior:
    # 38 distinct values, corr 0.414 -- the only smooth/non-mode-collapsed
    # prior in the program). Runner SFT marker "[/INST]" pre-existing.
    "mistral7b": ("mistralai/Mistral-7B-Instruct-v0.3",
                  "HF_HOME=/lustre/fast/fast/gsmithline/hf_cache "
                  "HF_HUB_OFFLINE=1"),
}
ACTIVE_MODELS = ["qwen7b", "gemma12b", "olmo7b"]
SEEDS = [0]
BETAS = [0.0, 0.1, 0.2, 0.5, 1.0]
# eps_ai=0.0 (zero-dose anchor) DROPPED 2026-07-22 after the olmo7b wave
# confirmed it fully inert -- strict `< eps_ai` gate never opens, acceptance
# 0.00 and op_bias/op_std delta +0.000 in every beta row (see BATCHES.md).
# The olmo ea0 runs are kept in notes/pofd/cluster/ as the control.
EPS_AIS = [0.05, 0.1, 0.2, 0.4]

# fixed columns (arg order of run_one_pokec_gated.sh):
# deploy_every=1, regime=replace, pscale=1.0, anchor=fixed, pop=ab,
# eps=0.0 (SOCIAL radius -> peer step off), gamma=0.0, wplat=1.0 (total
# adoption), mode=loop, canary=0.0; eps_ai is the per-row gate width.
ROW = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, ab, "
       "0.0, 0.0, 1.0, loop, 0.0, {eps_ai}")

# pofdpf_ data-regime wave: same fixed columns but regime=accumulate, beta=0
# (-> style sft), plus a 16th column pfrac -> PRISTINE_FRAC=$(pfrac) in the .sub.
PFRACS = [1.0, 0.75, 0.5, 0.25, 0.0]
PFRAC_MODELS = ["qwen7b", "olmo7b"]   # olmo added 2026-07-23, same 20-cell grid
ROW_PF = ("{tag}, sft, 0, {seed}, 1, accumulate, 1.0, fixed, ab, "
          "0.0, 0.0, 1.0, loop, 0.0, {eps_ai}, {pfrac}")

# pofdbp_ wave (2026-07-23): the beta x pfrac INTERIOR for the regularization
# phase diagram. The two edges already exist -- replace row across beta (pofd_
# grid) and beta=0 column across pf (pofdpf_ wave); this fills beta>0 x
# accumulate/pf so displacement/error heatmaps + the low-error low-displacement
# frontier can be drawn. Scoped to eps_AI {0.2, 0.4}: 0.2 is the data-anchor
# figure's eps, 0.4 is where the weight anchor bifurcates qwen (prior capture).
# beta>0 -> style sft_kl always (no beta=0 rows here; that column is pofdpf_).
BP_BETAS = [0.1, 0.2, 0.5, 1.0]
BP_EPS = [0.2, 0.4]
BP_MODEL = "qwen7b"
ROW_BP = ("{tag}, sft_kl, {beta}, {seed}, 1, accumulate, 1.0, fixed, ab, "
          "0.0, 0.0, 1.0, loop, 0.0, {eps_ai}, {pfrac}")

# pofdicl_ wave (2026-07-23): adaptation via CONTEXT -- frozen weights, no
# gradients, same pofd population channel (eps_social=0, INNATE_LAMBDA=0,
# W_PLAT=1). Arms per eps_AI: k0 = pure frozen baseline; k8live/k32live = K
# exemplar lines (other users, labels from the LIVE loop buffer -- the moving
# data anchor, replace-analog); k32pri = same K but labels are the users'
# INNATE opinions (fixed anchor, pristine-analog); d5/d10/d15/d30 = agent's OWN
# last-D-day history (personal memory, no cross-user context). The D dial has
# the same moving-vs-fixed structure: history starts at innate and turns loop-
# mediated after acceptances, so d5 shows only recent loop opinions (self-
# replace) while d30 >= horizon always includes the innate start (self-
# accumulate). noai is SKIPPED on
# purpose: with eps_social=0 the no-AI twin never moves, so noai == pristine.
ICL_ARMS = [("k0", 0, 0, "live"), ("k8live", 8, 0, "live"),
            ("k32live", 32, 0, "live"), ("k32pri", 32, 0, "pristine"),
            ("d5", 0, 5, "live"), ("d10", 0, 10, "live"),
            ("d15", 0, 15, "live"), ("d30", 0, 30, "live")]
ICL_MODEL = "qwen7b"
# olmo7b twin of the ICL wave (2026-07-27): same 8 arms x eps_AI grid.
# Frozen weights (nothing trains), W=1/lam=0/eps_social=0 -- unaffected by
# the population-update correction. Model deltas live in the .sub.
ICL_MODELS = ["qwen7b", "olmo7b"]
ROW_ICL = ("{tag}, frozen, 0, {seed}, 1, replace, 1.0, fixed, ab, "
           "0.0, 0.0, 1.0, loop, 0.0, {eps_ai}, {iclk}, {icldays}, {iclsrc}")

# pofddpo_ wave (2026-07-23): preference-based adaptation, same pofd channel,
# INNATE_LAMBDA=0 (the old rlhf waves used 0.2 -- NOT here), FRESH_EACH_ROUND=1.
# RLHF_FEEDBACK closed = preferences graded by the model's own deployed
# population (moving anchor); open = graded by the no-AI twin, which under
# eps_social=0 stays at innate forever -- the fixed data anchor. DPO_BETA is
# the implicit-KL-to-reference strength: the DPO analog of the KL knob.
DPO_BETAS = [0.1, 0.5]
DPO_FEEDBACKS = ["closed", "open"]
DPO_MODEL = "qwen7b"
# olmo7b twin of the sharp DPO wave (2026-07-27): same grid, same pofd channel
# (W=1, lam=0, eps_social=0 -- unaffected by the population-update correction),
# so it is directly comparable to pofddpo_ and to the olmo7b SFT/KL wave. Model
# deltas live in the .sub (separate HF cache, PPL_BATCH=16, 160G/60G).
DPO_MODELS = ["qwen7b", "olmo7b"]
ROW_DPO = ("{tag}, dpo, 0, {seed}, 1, replace, 1.0, fixed, ab, "
           "0.0, 0.0, 1.0, loop, 0.0, {eps_ai}, {rlhf}, {dpobeta}")

# pofddpon_ wave (2026-07-24): NOISY DPO -- same grid as the drift-prone half of
# pofddpo_ (eps_AI {0.2,0.4}) but with the two determinism knobs released:
# DO_SAMPLE=1 (serving = ONE draw from the label distribution instead of greedy
# mode -- the finite-sampling channel) and DPO_TAU=3 (soft BT grading; a 0.1
# accuracy gap wins p=0.57 instead of 0.77 at tau=12). Tests whether the 2-round
# snap-to-fixed-point of pofddpo_ is intrinsic to preference learning or an
# artifact of the sharp settings. Tag prefix pofddpon starts with "pofddpo" ON
# PURPOSE: check_pofd_sanity's dpo branch applies unchanged (TAU/DO_SAMPLE are
# env-only, like DPO_BETA -- verified via the submit configs).
DPON_EPS = [0.2, 0.4]

# pofdw_ wave (2026-07-24): FIRST population-realism rung. Two knobs at once,
# both textbook: W_PLAT=0.5 = canonical Deffuant mixing rate mu (the platform
# step IS a one-sided Deffuant interaction, bound eps_AI, rate W; the whole
# W=1 program was the mu=1 degenerate corner) + INNATE_LAMBDA=0.2 = Friedkin-
# Johnsen stubbornness. Displacement stops being absorbing: captured agents
# equilibrate between the served value and innate, stranded agents heal. W
# rides queue column 12; LAMBDA is env-only in the w-subs (runner arg 15 is
# never passed). Tags carry _w0p5_l0p2 tokens; check_pofd_sanity reads them and
# verifies the update exactly.
#
# SUPERSEDED 2026-07-27 by the pofdw2_/pofdws2_ waves. The original pofdw_/
# pofdws_ runs used the pre-correction round operator, which applied the innate
# anchor AFTER the platform blend (so W m was diluted by (1-lam), and W=1 did
# NOT recover m) and ran the peer sweep BEFORE the blend. The corrected
# operator, config population_update="nested_ai_then_social_v1", is
#
#     h_i(t) = k x_innate,i + (1-k) x_i(t)
#     z_i(t) = (1-W) h_i(t) + W m_i(t)   if |m_i(t) - x_i(t)| < eps_AI
#            = h_i(t)                    otherwise
#     x(t+1) = D_eps_social(z(t))
#
# gate on the START-OF-ROUND opinion x_i(t), mixture ONCE per round, peer
# (Deffuant) sweeps LAST. W=1 now gives z = m for every k, so the environment
# ladder W=1 -> W<1 -> eps_social>0 is properly nested. At W=1, k=0,
# eps_social=0 the two operators are algebraically identical, so the pofd_,
# pofdpf_, pofdbp_ and pofdicl_ families are UNAFFECTED and stay valid. The
# archived pofdw_/pofdws_ runs are not, and must not be mixed with pofdw2_/
# pofdws2_ output -- erm.is_clean_loop enforces the marker.
#
# Phase 1 = the decisive W=1 batteries re-run: core beta x eps grid (pofdw_) +
# sharp DPO (pofdwdpo_, prefix deliberately NOT starting with "pofddpo" -- the
# checker's dpo branch matches both) + noisy DPO (pofdwdpon_). ANS_SAMPLE_K=16
# entropy probe ON in all w-subs.
W_WPLAT = 0.5
W_LAMBDA = 0.2
W_MODEL = "qwen7b"
ROW_W = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, ab, "
         "0.0, 0.0, 0.5, loop, 0.0, {eps_ai}")
ROW_WDPO = ("{tag}, dpo, 0, {seed}, 1, replace, 1.0, fixed, ab, "
            "0.0, 0.0, 0.5, loop, 0.0, {eps_ai}, {rlhf}, {dpobeta}")

# pofdws_ wave (2026-07-24): SECOND population-realism rung -- the peer step
# comes ON. Same (W=0.5, lam=0.2) point as pofdw_ plus EPS_SOCIAL=0.2 (queue
# col 10; classic Deffuant bounded-confidence radius; the ab_sweep moves
# in-radius pairs to their midpoint = mu 0.5). The no-AI twin is NO LONGER
# frozen innate -- run_pokec_gated_lm now instantiates the simulated twin
# whenever eps > 0 (same peer sweeps, mirrored RNG, never gated) and saves
# its per-agent trajectory to trajectory.pt (twin_raw). Displacement is
# measured vs twin_raw, not innate. Peer moves are RNG-pairwise, so the
# checker CANNOT replay the exact update -- pofdws* runs get the weaker gate
# (config + peer-alive + twin present + finiteness) in check_pofd_sanity.
W_EPS_SOCIAL = 0.2
ROW_WS = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, ab, "
          "0.2, 0.0, 0.5, loop, 0.0, {eps_ai}")

# forward-KL waves (probe 2026-07-28, made CANONICAL 2026-07-29): the anchor
# as FORWARD KL(ref || pi) -- mass-covering; the SFT+KL optimum is then CE to
# the mixture (data + beta*base)/(1+beta) -- instead of reverse KL(pi || ref)
# (mode-seeking; the RLHF convention, used by every wave before 2026-07-28).
# The W=1 probe showed direction moves the capture THRESHOLD, not the
# strong-anchor endpoint (b0p5: fwd kills the 337-agent camp; b1: same ~330
# camp, Jaccard 0.86, either way). Decision: forward is the paper's canonical
# anchor from here on; reverse waves remain as the RLHF-practice comparison.
# KL_DIRECTION=forward is env-only, set in the at_pofd_qwen7b_*f.sub files and
# recorded in config.json. beta=0 rows are direction-independent -- figures
# reuse the reverse-wave b0 runs rather than duplicating them under new tags.
# 2026-07-29 (same day, user): w2f/ws2f expanded from key cells to the FULL
# beta>0 grids (FKL_BETAS x EPS_AIS = 16 cells each), so the forward waves
# can fully replace w2/ws2 in every figure; w1f stays at the ea0p4 dial.
# 2026-07-29 (later, user): olmo7b w2f/ws2f twins added -- same 16-cell grids,
# testing the qwen headline (peers dissolve the reverse b1_ea0p4 camp under
# forward, 198->3) on the IN-RANGE 0.75 prior where w1f showed forward is not
# protective (b0p1 captures MORE, b0p5 halved-not-killed, b1 collapses anyway).
FKL_BETAS = [0.1, 0.2, 0.5, 1.0]

# pofdw2fpt_ pristine-TEACHER wave (2026-08-04, user/reviewer): same forward-
# KL W=0.5 environment as w2f, but the KL reference is Train(base, D_pristine)
# -- the w2 b0 run's round0_adapter (base + one SFT epoch on the 723 innate
# labels; measured corr(pred, innate) 0.879, mae 0.046, mean 0.643, vs the
# raw base prior's bimodal 0.25/0.65 with corr 0.04) merged onto the base via
# KL_REF_ADAPTER (env-only, set in the .sub, recorded in config.json). Claim
# under test: the population effect depends on the CONTENT of the reference
# distribution, not only on the presence of a KL penalty. The triangle: w2f =
# same beta, base ref (pretrained prior); THIS = same beta, population ref (a
# model representation of the original population); pf2 = the same anchor
# content in DATA space (hard labels, ordinary SFT rows -- a dataset cannot
# be a token-level KL teacher, it has no next-token distribution). Scoped to
# ea {0.2, 0.4} (pf2 precedent: the data-anchor figure's eps + the qwen
# bifurcation eps) x FKL_BETAS = 8 jobs, seed 0. b0 is anchor-free (no KL
# term) -- reuse the pofdw2_ b0 runs. Smoke REQUIRED before the wave:
# KL_REF_ADAPTER is a new code path (peft merge into the frozen ref).
PT_EPS = [0.2, 0.4]

# eps-social dose-response at the forward-KL headline cell (2026-07-29, user):
# qwen7b, forward KL, b=1, ea=0.4, lam=0.2 FIXED -- sweep the repair channel
# to locate the transition where peers start beating the mass-covering anchor
# (the ws2f b1_ea0p4 camp dissolves at es=0.2, attributable 198->3; the w2f
# es=0 twin holds 234). W=0.5 arm: es {0, 0.10, 0.15, 0.20, 0.25}, where
# es=0 IS pofdw2f_qwen7b_b1_ea0p4 and es=0.20 IS pofdws2f_qwen7b_b1_ea0p4 --
# REUSED, not re-run (identical physics; WITH_TWIN only adds telemetry), so
# only the 3 remaining doses run. W=1 arm: es {0, 0.20, 0.30, 0.40} -- ALL
# 4 run: lam=0.2 differs from the w1f environment (lam=0, ungated agents
# frozen) and W=1 + peers is a new combination, so nothing is reusable.
# WITH_TWIN=1 rides the .sub env: every run (incl. es=0) saves the matched
# no-platform twin_raw (RNG-mirrored peers; deterministic innate drift at
# es=0).
# Flow: seed 0 locates the transition -> fill ESF_REPL_POINTS with the
# important (w, es) cells -> regenerate -> submit qwen7b_esf_repl (s42+s43).
# targeted fe expansion (2026-08-02, user), key qwen7b_fex: 8 missing
# cells of the natural-gender fe protocol (b1, forward KL, W=0.5,
# lam=0.2, replace + fresh adapter, corrected operator, WITH_TWIN=1),
# staying in the pofdesf_ family:
#   (1) ea0p4, es {0.10, 0.15, 0.25} x s {42, 43} -- the W=0.5
#       transition doses at replication seeds; tags BYTE-IDENTICAL to
#       the corresponding esf_repl rows (never submitted -- if
#       esf_repl runs later those 6 no-op via the idempotent exec)
#   (2) ea0p2, es {0.10, 0.25} x s0 -- extends the family to a second
#       eps_AI dial (esf_tag/esf_rows gained an ea param; ROW_ESF
#       feeds {eps_ai}); es=0.2 counterpart exists as pofdws2f_ b1_ea0p2
# AUDIT 2026-08-02: all 8 tags absent on cluster (no partials), no
# other-family equivalents. No smoke: env identical to the validated
# esf sub, only queue-fed dials differ. Checker: pofdesf_ -> generic
# social branch (es>0, fresh), as validated for the esf s0 scan.
# corrected-env ports of ICL + DPO (2026-07-29, user): the frozen in-context
# arm and the sharp on-policy DPO arm re-based into env2 (W=0.5 FJ, lam=0.2,
# no peers) and env3 (env2 + EPS_SOCIAL=0.2), so the adaptation-mechanism
# comparison (SFT-KL vs ICL vs DPO) exists in the same environments as the
# w2f/ws2f forward waves. ICL is frozen -- no loss, KL-direction-free. DPO
# is intrinsically reverse-KL (DPO_BETA IS the implicit KL(pi||ref)
# strength), so it sits on the RLHF-practice side by construction; its
# forward counterpart is the w2f/ws2f SFT-KL wave. icls2 adds the k32noai
# arm (exemplar labels from the SYNCHRONIZED no-AI twin): with peers the
# twin MOVES, so noai != pristine for the first time (at es=0 they coincide
# and noai is skipped, as in the W=1 wave). Tag prefixes keep the checker
# families -- pofdicl2_/pofdicls2_ start with "pofdicl", pofdwdpo2_/
# pofdwdpos2_ start with "pofdwdpo" -- env tokens _w0p5_l0p2[_es0p2]_ route
# W/lam/es, and the icl arm token stays directly before _s<seed> (the
# checker's arm regex). The prepared-but-never-submitted pofdwdpo_/pofdwdpon_
# waves (old W-era tags) are SUPERSEDED by pofdwdpo2_ -- do not submit them.
ICLS2_ARMS = ICL_ARMS + [("k32noai", 32, 0, "noai")]
ROW_ICL2 = ("{tag}, frozen, 0, {seed}, 1, replace, 1.0, fixed, ab, "
            "{es}, 0.0, 0.5, loop, 0.0, {eps_ai}, {iclk}, {icldays}, {iclsrc}")
ROW_WDPO2 = ("{tag}, dpo, 0, {seed}, 1, replace, 1.0, fixed, ab, "
             "{es}, 0.0, 0.5, loop, 0.0, {eps_ai}, {rlhf}, {dpobeta}")

ESF_W05_ES = [0.10, 0.15, 0.25]     # es=0 -> w2f, es=0.20 -> ws2f (reuse)
ESF_W1_ES = [0.0, 0.20, 0.30, 0.40]
# s0 scan results (2026-07-29): W=0.5 attributable 234/127/2/3/61 at es
# 0/0.1/0.15/0.2/0.25 -- locked at 0.1, SLOW dissolution at 0.15 (camp ~130
# for 18 rounds, gone by r29), fast at 0.2, RE-ENTRANT partial capture at
# 0.25 (churning ~60-agent camp at mean 0.37: the wide radius also ferries
# mainstream agents into the platform's gate). W=1: 287/347/345/309 at es
# 0/0.2/0.3/0.4 -- no rescue at any dose, camp GROWS with peers. Repl
# points: the locked/marginal/re-entrant W=0.5 doses + both W=1 endpoints
# + the W=0.5 baseline (within-seed references; es=0.2 W=0.5 is covered by
# the eventual ws2f seed gates).
ESF_REPL_POINTS = [(0.5, 0.0), (0.5, 0.10), (0.5, 0.15), (0.5, 0.25),
                   (1.0, 0.0), (1.0, 0.40)]
ESF_REPL_SEEDS = [42, 43]
ROW_ESF = ("{tag}, sft_kl, 1, {seed}, 1, replace, 1.0, fixed, ab, "
           "{es}, 0.0, {w}, loop, 0.0, {eps_ai}")

# corrected-env port of the DATA-side regularizer (2026-07-30, user): the
# pfrac wave (beta=0, style sft, DATA_REGIME=accumulate + PRISTINE_FRAC)
# re-based into env2 (W=0.5 FJ, lam=0.2, no peers) and env3 (env2 +
# EPS_SOCIAL=0.2). NO KL term by design (user: the data regularizer runs
# without the weight anchor) -> KL-direction-free, like the ICL waves.
# Science: w2f/ws2f anchor to the model's PRIOR (KL_BETA); at b1_ea0p4 that
# anchor CAUSES 234 attributable captures in env2, which peers dissolve in
# env3 (3). This wave anchors to the ORIGINAL DATA instead -- every retrain
# batch pins round(pf*723) rows to the round-0 innate seed
# (mix_pristine_data) -- asking whether a data anchor prevents capture
# rather than causing it, and how it composes with peer repair/delivery.
# pf=0 is the plain-accumulate control: the data REGIME differs from the
# fresh-replace w2f rows, so pf=0-accumulate (not w2f_b0) is the honest
# no-anchor column. Tag prefixes pofdpf2_/pofdpfs2_/pofdpfs2smk_ keep the
# checker family (is_pfrac now matches "pofdpf"); env tokens _w0p5_l0p2
# [_es0p2]_ route W/lam/es and the _pf token is config-checked as before.
# Smoke: accumulate buffer x pristine hold x live peer step (+forced twin)
# is a NEW combination (pfrac/bp were W=1 no-peer) -> 1-job pfs2 smoke.
# The no-peer pf2 side needs none (accumulate x pf proven by the W=1 pfrac
# wave, the env2 operator by w2f/icl2).
ROW_PF2 = ("{tag}, sft, 0, {seed}, 1, accumulate, 1.0, fixed, ab, "
           "{es}, 0.0, 0.5, loop, 0.0, {eps_ai}, {pfrac}")

# feature-endogenization RERUNS (2026-07-30, user): the Tree-3 gender arms
# (natural / dropped / permuted / frozen) re-based into the CURRENT
# corrected peer environment (W=0.5, lam=0.2, es=0.2, ea=0.4, forward KL,
# replace + fresh adapter each round, nested operator), seeds {0, 42, 43}.
# The old mlawD_/mlaaD_/icl_*_gdrop/gperm runs predate the corrected
# operator -- hence reruns. AUDIT 2026-07-30: the four s0 anchors already
# exist and are REUSED, not re-run -- natural b0 = pofdws2_qwen7b_b0_ea0p4
# (reverse-era tag; b0 has no KL term so it is direction-free), natural
# b0p5/b1 = pofdws2f_ s0, frozen = pofdicls2_ ea0p4_k0_s0. Only the 14
# missing cells run:
#   fes  (6): natural b {0, 0.5, 1} x s {42, 43} -- ws2f-family tags
#   fegd (3): PROFILE_DROP_COLS=gender,    b1, s {0,42,43} -- pofdfegd_
#   fegp (3): PROFILE_PERMUTE_COLS=gender, b1, s {0,42,43} -- pofdfegp_
#   fef  (2): frozen k0 x s {42, 43} -- pofdicls2_-family tags
# fegd/fegp prefixes match no special checker family -> generic social
# branch (env tokens carry W/lam/es); fes/fef reuse their families'
# branches. Twins forced everywhere (es>0). The drop/permute knobs act
# once at profile-build time, are config-recorded (profile_drop_cols/
# profile_permute_cols -- gate on those at pull), and permutation is
# seeded by the run seed (saved to permute_cols.json). No smoke: both
# knobs are proven on movielens by the old waves and touch nothing in
# the loop.
# env2 mirror (2026-07-31, user): the same four arms at ES=0 (W=0.5,
# lam=0.2, ea=0.4, NO peer step) -- does the platform loop endogenize
# gender on its own, without the peer current? AUDIT 2026-07-31: the
# four env2 s0 anchors exist and are REUSED -- natural b0 = pofdw2_
# b0_ea0p4 (reverse-era tag, direction-free), natural b0p5/b1 =
# pofdw2f_ s0, frozen = pofdicl2_ ea0p4_k0_s0. Only the 14 missing
# cells run (fes2 6 / fegd2 3 / fegp2 3 / fef2 2), composite key
# qwen7b_fe2. No twin at es=0 (the runner forces it only when eps>0);
# the no-platform counterfactual is innate, as in every env2 wave.
# es=0 -> the checker's exact-replay branch for all four subs. Same
# fegd/fegp permutation per seed as env3 (seeded by run seed only) --
# deliberate, keeps the env2/env3 pair matched within seed.
# reverse-KL mirror of the env3 fe wave (2026-07-31, user): the peer-
# environment fe matrix on the RLHF-practice side of the KL ledger.
# Direction-free cells are REUSED, not re-run: natural b0 all 3 seeds
# (pofdws2_ b0 s0 + pofdws2f_ b0 s42/s43 -- style sft, no KL term) and
# frozen k0 all 3 seeds (pofdicls2_, never trains). AUDIT 2026-07-31:
# reverse natural b0p5/b1 s0 anchors exist from the ws2 wave and are
# REUSED (pofdws2_; that wave predates the kl_direction config field,
# so those configs record kl_direction=None -- physics reverse, the
# only implementation then; the new rows set KL_DIRECTION=reverse
# explicitly and record 'reverse' -- same physics, gate accepts both).
# Only 10 missing cells run, composite key qwen7b_fer:
#   fesr  (4): natural b {0.5, 1} x s {42, 43} -- pofdws2_ replication
#              seeds (reverse-era family, no direction token)
#   fegdr (3): PROFILE_DROP_COLS=gender,    b1, s {0,42,43} -- pofdfegdr_
#   fegpr (3): PROFILE_PERMUTE_COLS=gender, b1, s {0,42,43} -- pofdfegpr_
# fegdr/fegpr permutation is seeded by the run seed ONLY -- SAME
# per-seed permutation as forward fegp and env2 fegp2, deliberate: the
# forward/reverse pair is matched within seed. Checker: pofdws2_ ->
# the ws weaker gate (peer-alive + twin); pofdfegdr_/pofdfegpr_ match
# no special family -> generic social branch via env tokens. Twins
# forced everywhere (es>0). Never flip the runner code default
# ('reverse') -- forward stays env-only in the *f subs.
# CONTINUAL-weights mirror of the env3 forward fe wave (2026-08-01,
# user): the same experiment with FRESH_EACH_ROUND=0 -- the adapter
# persists and keeps training across rounds (the runner's native
# continual-SFT default; forward KL still anchors every round to the
# FIXED pristine base, ref_model_name=base). The archived pre-correction
# fe runs (mlawD_/mlaaD_) were continual, which is WHY the fresh
# protocol exists (rules out weight-drift compounding); this wave
# re-asks the fe question with weight drift back ON, under the
# corrected operator + forward KL, so fresh-vs-continual is a clean
# within-environment contrast. NO anchors exist (no continual training
# run in the corrected era) -> all cells run, seeds {0, 42, 43}:
#   fesc  (9): natural b {0, 0.5, 1} x 3 seeds -- pofdws2fc_ (new
#              family: env3 forward continual)
#   fegdc (3): PROFILE_DROP_COLS=gender,    b1 x 3 seeds -- pofdfegdc_
#   fegpc (3): PROFILE_PERMUTE_COLS=gender, b1 x 3 seeds -- pofdfegpc_
# The frozen arm is SHARED with the fresh matrix (pofdicls2_ k0: no
# weights, fresh/continual meaningless). The _fresh_data tag suffix
# names the DATA protocol (replace, n_train=723 every round -- the
# _fresh_errs gate still applies); only the WEIGHTS are continual.
# Continual is a NEW physics combination in the corrected era -> smoke
# first (pofdws2fcsmk_ b1_ea0p4 s0, 3 rounds, key qwen7b_fec_smoke),
# gate with check_pofd_sanity, THEN submit qwen7b_fec (15 jobs).
# check_pofd_sanity gained an is_cont branch (pofdws2fc_/pofdfegdc_/
# pofdfegpc_ -> fresh_each_round=False expected); permutation seeded by
# run seed ONLY -> same per-seed permutation as fegp/fegp2/fegpr.
# JENSEN-SHANNON mirror of the env3 fe wave (2026-08-02, user): the
# same peer-environment fe matrix with KL_DIRECTION=js -- JS(pi, ref) =
# 0.5*KL(pi||m) + 0.5*KL(ref||m), m the even mixture: the symmetric
# midpoint between mode-seeking (reverse) and mass-covering (forward).
# Bounded by log 2 per token, so at equal beta the anchor SATURATES
# once the policy is far from base -- beta values are not
# strength-comparable across divergences; the grid stays parallel
# anyway ({0.5, 1}) for design symmetry. Direction-free cells REUSED:
# natural b0 all 3 seeds (style sft) and frozen k0 all 3 seeds
# (pofdicls2_). NO trained anchors exist (JS never ran anywhere) ->
# 12 cells, composite key qwen7b_fej:
#   fesj  (6): natural b {0.5, 1} x s {0, 42, 43} -- pofdws2j_ (new
#              family: env3 JS)
#   fegdj (3): PROFILE_DROP_COLS=gender,    b1 x 3 seeds -- pofdfegdj_
#   fegpj (3): PROFILE_PERMUTE_COLS=gender, b1 x 3 seeds -- pofdfegpj_
# Permutation seeded by the run seed ONLY -> same per-seed permutation
# as every other fe wave. Checker: no new branch -- fresh runs, so
# pofdws2j_/pofdfegdj_/pofdfegpj_ route to the generic social branch
# via the _es token (no prefix collision with pofdws2fc_/pofdfegdc_/
# pofdfegpc_). JS is NEW physics (kl_sft.py gained the 'js' branch,
# never trained) -> smoke first (pofdws2jsmk_ b1_ea0p4 s0, 3 rounds,
# key qwen7b_fej_smoke), gate with check_pofd_sanity, THEN qwen7b_fej.
# KL_DIRECTION stays env-only; the runner code default ('reverse')
# never flips.
FE_SEEDS = [0, 42, 43]

# controlled TEACHER feature-endogenization wave (2026-08-04, user): does a
# gender signal that lives ONLY in the KL teacher's weights transfer into
# the population? Two-stage design.
#   Stage 1 (qwen7b_tch, 2 jobs): train two fixed teachers by one-round SFT
#   on TRANSFORMED pristine labels
#     y_i_teacher = clip(y_i + delta * (2*1[gender_i == M] - 1), 0, 1)
#   at delta {+0.08, -0.08} (runner knob TEACHER_LABEL_DELTA, sft-only,
#   touches initial_data["y"] ONLY -- innate/population/twin/buffer
#   untouched; recorded in config). Same budget as every round-0 teacher
#   (1 epoch, LoRA r512, lr 5e-5, batch 4, 723 rows, seed 0, N_ROUNDS=1).
#   The delta-0 (neutral) teacher is REUSED: pofdw2_ b0 ea0p4 round0_adapter
#   (the w2fpt pristine teacher; corr(pred, innate) 0.879).
#   GATE before stage 2 (registered; full thresholds in the tch sub header):
#   opposite signs, |gap| >= 0.05 each, magnitude ratio <= 2, no collapsed
#   prediction distribution (pred_std/corr/eff_support floors), and the
#   neutral teacher's gap BETWEEN the signed pair. Innate gender gap is
#   +0.0021 (~zero, measured 2026-08-04), so the +/-0.08 label gaps are
#   near-symmetric (+0.162/-0.158) and teacher gaps are pure injection.
#   Stage 2 (qwen7b_tfe, 15 jobs): env3 forward SFT-KL loops (b1, ea0p4,
#   W=0.5, lam=0.2, es=0.2, fresh + replace, 30 rounds) x seeds {0,42,43},
#   five arms, the teacher entering ONLY through KL_REF_ADAPTER:
#     tpos    KL ref = +0.08 teacher
#     tneu    KL ref = neutral teacher      (reused pofdw2_ b0 adapter)
#     tneg    KL ref = -0.08 teacher
#     tposgd  +0.08 teacher, PROFILE_DROP_COLS=gender    (student prompts)
#     tposgp  +0.08 teacher, PROFILE_PERMUTE_COLS=gender (randomized-feature
#             test: does the association follow the DISPLAYED feature?)
#   Teacher prompts are never changed (token-level KL needs identical
#   teacher/student prompts); only the student-side profile controls move.
#   LOG_GENDER_GAPS=1 everywhere: per-round gg_pred/op/twin gaps by true +
#   displayed gender, the fixed teacher's gap (gg_teacher), and incremental
#   gender R^2 (gg_r2_inc_*) land in trajectory rows. check_pofd_sanity
#   gained is_tch (delta/label/round0_batch gates) and is_tfe (ref-adapter
#   path, profile controls, forward b1, gg telemetry) branches. Smoke first:
#   pofdtfesmk_ tpos s0, 3 rounds (new teacher adapter x peers x gg keys),
#   then the wave. Tags: pofdtch_qwen7b_d{p,m}0p08_ea0p4_w0p5_l0p2_s0;
#   pofdtfe_qwen7b_b1_ea0p4_<arm>_w0p5_l0p2_es0p2_s<seed>_fresh_data.
TCH_DELTAS = [("dp0p08", "0.08"), ("dm0p08", "-0.08")]
ROW_TCH = ("{tag}, sft, 0, {seed}, 1, replace, 1.0, fixed, ab, "
           "0.0, 0.0, 0.5, loop, 0.0, {eps_ai}, {tdelta}")
ROW_TFE = ("{tag}, sft_kl, 1, {seed}, 1, replace, 1.0, fixed, ab, "
           "0.2, 0.0, 0.5, loop, 0.0, {eps_ai}, {refadapter}")
TFE_RUNS = "/home/gsmithline/perfsim/runs/pokec_gated_lm"
TFE_REFS = {
    "tpos": f"{TFE_RUNS}/pofdtch_qwen7b_dp0p08_ea0p4_w0p5_l0p2_s0/round0_adapter",
    "tneu": f"{TFE_RUNS}/pofdw2_qwen7b_b0_ea0p4_w0p5_l0p2_s0_fresh_data/round0_adapter",
    "tneg": f"{TFE_RUNS}/pofdtch_qwen7b_dm0p08_ea0p4_w0p5_l0p2_s0/round0_adapter",
}
# CI EXTENSION (2026-08-10, user): seeds {44,45} for the three teacher
# arms ONLY (tpos/tneu/tneg; the gd/gp mechanism arms stay at 3 seeds).
# Identical configuration to qwen7b_tfem -- same ROW_TFE, same fixed
# teacher adapters (never retrained), same env3 forward SFT-KL loop --
# so the 6 new runs join the existing 9 to give 5 independent seeds per
# arm. Statistical use: seeds are the inferential replicates (n=5,
# df=4); round-29 student-prediction and population-opinion gaps get
# Student-t 95% CIs plus paired within-seed contrasts (pos-neu, neg-neu,
# pos-neg); teacher gaps are deterministic shared interventions (dashed
# constants, no across-seed CI). Sub is GENERATED (unlike the
# hand-written tfem sub) and carries the g106 guard.
TFE_CI_SEEDS = [44, 45]
TFE_CI_SUB = """\
# HTCondor: CONTROLLED-TEACHER CI SEED EXTENSION (tfe_ci), Qwen-7B --
# GENERATED by gen_pofd_sweep.py from the TFE_ block (2026-08-10). Never
# edit this file by hand: edit the TFE_ block and rerun the script.
# 6 runs: KL ref {tpos +0.08 teacher, tneu neutral, tneg -0.08 teacher}
# (queue col 16 = adapter path) x NEW seeds {44, 45} -- brings every
# teacher arm to 5 seeds for Student-t CIs and paired within-seed
# contrasts. Env3 forward SFT-KL, byte-identical to at_pofd_qwen7b_tfem
# .sub: b1, ea0p4, W_PLAT=0.5, INNATE_LAMBDA=0.2, EPS_SOCIAL=0.2 (queue
# col 10), fresh adapter + replace data, 30 rounds, LOG_GENDER_GAPS=1.
# Teachers are the EXISTING fixed adapters (round0_adapter dirs on the
# purge keep-list) -- never retrained. PRE-FLIGHT: confirm all three
# adapter dirs exist before submitting.
# GATE after pull: check_pofd_sanity is_tfe (ref path vs arm, forward
# b1, profile controls, gg telemetry) + universal _s/_ea/slug gates.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> qwen7b_tfe_ci
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = 128G
request_disk      = 40G
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action EPS_AI=$(eps_ai) KL_DIRECTION=forward KL_REF_ADAPTER=$(refadapter) LOG_GENDER_GAPS=1 INNATE_LAMBDA=0.2 ANS_SAMPLE_K=16 ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 FRESH_EACH_ROUND=1 TRAIN_CAP=723 N_ROUNDS=30 EPOCH_SIZE=100 BASE_MODEL=Qwen/Qwen2.5-7B-Instruct SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 USE_LORA=1 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=64 SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_qwen7b_lora512_pofdtfe"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, refadapter from experiments/condor/configs_pofd_qwen7b_tfe_ci.txt
"""

# DPO CI EXTENSION (2026-08-12, user): seed replicates for the DPO causal
# story, staged SEQUENTIALLY -- each stage submitted only after the previous
# stage's read (ci1 fails to replicate -> STOP, DPO stays an exploratory
# appendix result):
#   ci1 (8):  CONFIRM -- sharp grid pofddpo_, db0.5 x ea {0.2,0.4} x
#             {closed,open}, NEW seeds {42,43} (s0 exists -> 3 seeds/cell).
#   ci2 (24): COMPLETE the sharp grid -- db0.5 x ea {0.05,0.1} (8) +
#             db0.1 x all four gates (16), seeds {42,43}.
#   ci3 (20): PEER BREADTH, full-epoch W=0.5/l=0.2 families -- es=0
#             (pofdwdpo2e_) db0.5 x ea {0.2,0.4} seeds {42,43} (8) + NEW
#             es=0.3 dose (pofdwdpos2e_ prefix, es0p3 token -- the checker's
#             es gate is token-driven) db0.5 x ea {0.2,0.4} seeds {0,42,43}
#             (12; s0 is new too, the dose never ran).
# Rows are byte-identical to the parent families except seed/es. The sharp
# stages keep DPO_MAX_STEPS=3 ON PURPOSE: they replicate the s0 finding
# AS-RUN; the training-budget question is answered separately by the 2e
# full-epoch wave (ci3 inherits max_steps=0 from its parents). All new runs
# carry the counterfactual preference-flip telemetry (runner marks
# dpo_flip_telemetry in config.json, 2026-08-12; the checker requires the
# per-round keys). The telemetry is arithmetic-only -- no extra RNG draws --
# so the new seeds are design-identical replicates of the s0 code path.
DPO_CI_SEEDS = [42, 43]
DPO_CI1_CELLS = [(0.5, 0.2), (0.5, 0.4)]
DPO_CI2_CELLS = ([(0.5, 0.05), (0.5, 0.1)]
                 + [(0.1, ea) for ea in (0.05, 0.1, 0.2, 0.4)])
DPO_CI3_EAS = [0.2, 0.4]
DPO_CI3_ES3_SEEDS = [0, 42, 43]

_DPO_CI_SHARP_TAIL = """\
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = 160G
request_disk      = 40G
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action EPS_AI=$(eps_ai) RLHF_FEEDBACK=$(rlhf) DPO_BETA=$(dpobeta) DPO_TAU=12.0 DPO_GEN_TEMP=1.0 DPO_MAX_STEPS=3 DPO_N_PAIRS=0 SFT_EPOCHS=0 SFT_LR=5e-5 LORA_R=128 USE_LORA=1 DO_SAMPLE=0 FRESH_EACH_ROUND=1 TRAIN_CAP=723 N_ROUNDS=30 EPOCH_SIZE=100 BASE_MODEL=Qwen/Qwen2.5-7B-Instruct SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=64 SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_qwen7b_pofddpo"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, rlhf, dpobeta from experiments/condor/configs_pofd_qwen7b_dpo_ci{n}.txt
"""

DPO_CI1_SUB = ("""\
# HTCondor: DPO CI STAGE 1 -- CONFIRMATION (dpo_ci1), Qwen-7B.
# GENERATED by gen_pofd_sweep.py from the DPO CI EXTENSION block
# (2026-08-12). Never edit by hand: edit the block and rerun the script.
# 8 runs: sharp-grid pofddpo_ cells db0.5 x eps_AI {0.2,0.4} x
# {closed,open} at NEW seeds {42,43} -- with s0, 3 seeds per cell.
# Env byte-identical to at_pofd_qwen7b_dpo.sub (W=1, es=0, lam=0,
# DPO_MAX_STEPS=3 kept AS-RUN; see the generator block for why) except
# the g106 guard. New runs carry the preference-flip telemetry
# (dpo_flip_telemetry; arithmetic-only, replay-identical).
# GATE: read vs s0 BEFORE submitting ci2/ci3. If ci1 fails to
# replicate, STOP -- DPO stays an exploratory appendix result.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> qwen7b_dpo_ci1
""" + _DPO_CI_SHARP_TAIL.format(n=1))

DPO_CI2_SUB = ("""\
# HTCondor: DPO CI STAGE 2 -- COMPLETE THE SHARP GRID (dpo_ci2), Qwen-7B.
# GENERATED by gen_pofd_sweep.py from the DPO CI EXTENSION block
# (2026-08-12). Never edit by hand: edit the block and rerun the script.
# 24 runs at NEW seeds {42,43}: db0.5 x eps_AI {0.05,0.1} (8) + db0.1 x
# eps_AI {0.05,0.1,0.2,0.4} (16) x {closed,open}. With ci1 + s0 the full
# 2x4x2 sharp design has 3 seeds throughout.
# Env byte-identical to at_pofd_qwen7b_dpo.sub except the g106 guard.
# PREREQ: ci1 replicated (do NOT submit otherwise).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> qwen7b_dpo_ci2
""" + _DPO_CI_SHARP_TAIL.format(n=2))

DPO_CI3_SUB = """\
# HTCondor: DPO CI STAGE 3 -- PEER-INFLUENCE BREADTH (dpo_ci3), Qwen-7B.
# GENERATED by gen_pofd_sweep.py from the DPO CI EXTENSION block
# (2026-08-12). Never edit by hand: edit the block and rerun the script.
# 20 runs, FULL-EPOCH W=0.5/l=0.2 families (DPO_MAX_STEPS=0), db0.5 x
# eps_AI {0.2,0.4} x {closed,open}:
#   - es=0 (pofdwdpo2e_ tags), NEW seeds {42,43}: 8 runs (s0 exists).
#   - es=0.3 NEW dose (pofdwdpos2e_ prefix, es0p3 token), seeds
#     {0,42,43}: 12 runs. es rides queue col 10; the checker's es gate
#     is token-driven, so no checker change.
# Env byte-identical to at_pofd_qwen7b_wdpo2e.sub except the g106 guard.
# Existing es0p2 s0 cells (pofdwdpos2e_) stay the middle dose, 1 seed.
# PREREQ: ci1 replicated (do NOT submit otherwise).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> qwen7b_dpo_ci3
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = 128G
request_disk      = 40G
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action EPS_AI=$(eps_ai) RLHF_FEEDBACK=$(rlhf) DPO_BETA=$(dpobeta) DPO_TAU=12.0 DPO_GEN_TEMP=1.0 DPO_MAX_STEPS=0 DPO_N_PAIRS=0 INNATE_LAMBDA=0.2 ANS_SAMPLE_K=16 ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 SFT_EPOCHS=0 SFT_LR=5e-5 LORA_R=128 USE_LORA=1 DO_SAMPLE=0 FRESH_EACH_ROUND=1 TRAIN_CAP=723 N_ROUNDS=30 EPOCH_SIZE=100 BASE_MODEL=Qwen/Qwen2.5-7B-Instruct SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=64 SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_qwen7b_pofdwdpo2e"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, rlhf, dpobeta from experiments/condor/configs_pofd_qwen7b_dpo_ci3.txt
"""

# MISTRAL MAIN-ENV WAVE (2026-08-12, user): the exact canonical main
# environment on the 4th model -- pofdws2f_ family, W=0.5, INNATE_LAMBDA=0.2,
# EPS_SOCIAL=0.2, ea=0.4, forward KL, fresh adapters, 30 rounds, matched
# no-platform twin (instantiated automatically at es>0). b {0,1} x seeds
# {0,42,43} = 6 jobs. Rows byte-identical to the qwen ws2f grammar with the
# model slot swapped; the checker's ws2f branch + universal SLUG_BASE gate
# cover the new slug with no branch changes. SMOKE FIRST (1 job, 3 rounds,
# DEBUG_GEN): first-ever Mistral run through the TRAINING path -- exercises
# the "[/INST]" assistant marker in the SFT collator and the forward-KL pair
# on a new tokenizer before the 6x30-round wave.
MISTRAL_WS2F_SEEDS = [0, 42, 43]

_MISTRAL_WS2F_ENV = (
    'environment       = "REPO=/home/gsmithline/perfsim '
    "CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh "
    "ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key "
    "WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action "
    "HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 "
    "EPS_AI=$(eps_ai) KL_DIRECTION=forward INNATE_LAMBDA=0.2 "
    "ANS_SAMPLE_K=16 ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 FRESH_EACH_ROUND=1 "
    "TRAIN_CAP=723 N_ROUNDS={nr} EPOCH_SIZE=100 "
    "BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.3 SFT_EPOCHS=1 "
    "SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 USE_LORA=1 SFT_LR=5e-5 "
    "N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 "
    "LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=64 SEED_BASE_DATA=1 "
    'WANDB_RUN_SUFFIX={suffix}"')

_MISTRAL_WS2F_TAIL = """\
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = 128G
request_disk      = 40G
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
{env}

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai from experiments/condor/{cfg}
"""

MISTRAL_WS2F_SUB = ("""\
# HTCondor: MISTRAL MAIN-ENV WAVE (pofdws2f_ family), 4th model.
# GENERATED by gen_pofd_sweep.py from the MISTRAL MAIN-ENV block
# (2026-08-12). Never edit by hand: edit the block and rerun the script.
# 6 runs: b {0 sft, 1 forward sft_kl} x seeds {0,42,43} at the canonical
# main environment (W=0.5, INNATE_LAMBDA=0.2, EPS_SOCIAL=0.2 queue col
# 10, ea0p4, fresh adapters, 30 rounds; no-AI twin auto-instantiated at
# es>0 -> twin_raw in trajectory.pt). Env matches at_pofd_qwen7b_ws2f
# .sub with the model slot swapped + /lustre HF cache + g106 guard.
# Promoted off the mlatZ_cand screen: smooth prior, 38 distinct values,
# corr(pred,innate) 0.414. Runner SFT marker "[/INST]" pre-existing.
# PREREQ: mistral7b_ws2f_smoke PASSED + gated (first Mistral training
# run -- collator marker + forward-KL path on a new tokenizer).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> mistral7b_ws2f
""" + _MISTRAL_WS2F_TAIL.format(
    env=_MISTRAL_WS2F_ENV.format(nr=30, suffix="_mistral7b_lora512_pofdws2f"),
    cfg="configs_pofd_mistral7b_ws2f.txt"))

MISTRAL_WS2F_SMOKE_SUB = ("""\
# HTCondor: MISTRAL MAIN-ENV SMOKE, 1 job, 3 rounds (ws2f b1_ea0p4 s0).
# GENERATED by gen_pofd_sweep.py from the MISTRAL MAIN-ENV block
# (2026-08-12). Never edit by hand: edit the block and rerun the script.
# Gates mistral7b_ws2f: FIRST-EVER Mistral run through the training path.
# Verify in the .out: prompts/answers sane (DEBUG_GEN), finite preds in
# [0,1], SFT loss moves (the "[/INST]" collator marker masks correctly),
# forward-KL term finite at b1, peer step + twin complete, then
# check_pofd_sanity PASS on the pulled dir (pofdws2fsmk_ branch).
# Env identical to at_pofd_mistral7b_ws2f.sub except N_ROUNDS=3 +
# DEBUG_GEN + smoke WANDB suffix.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> mistral7b_ws2f_smoke
""" + _MISTRAL_WS2F_TAIL.format(
    env=_MISTRAL_WS2F_ENV.format(
        nr=3, suffix="_mistral7b_lora512_pofdws2fsmk").replace(
        'environment       = "', 'environment       = "DEBUG_GEN=1 DEBUG_GEN_N=12 '),
    cfg="configs_pofd_mistral7b_ws2f_smoke.txt"))

# MATCHED-RANDOMNESS DPO PAIRS (2026-08-13, user), key qwen7b_dpo_mr:
# separates initial stochastic branch selection from causal closed-loop
# preference feedback. ONE Condor job = ONE PAIR: the closed arm (bank
# WRITER) then the open arm (bank READER) run sequentially on the SAME GPU
# via run_one_dpo_pair_idempotent.sh. Both arms share round 0 (trained once
# by the writer, forked: adapter + served preds + population state loaded by
# the reader) and every later round's candidate strings + BT uniforms; the
# reader recomputes preference ORIENTATION with its own judge. From round 1
# the only designed difference is closed judge = deployment-shaped
# population vs open judge = matched no-platform twin (= innate at es=0).
# Setting = the sharp DPO confirmation cell: db0.5, DPO_TAU=12,
# GEN_TEMP=1.0, MAX_STEPS=3, all 723 agents, W=1, lam=0, es=0, gamma=0,
# fresh each round, 30 rounds, pop/base-data seed FIXED at 0.
# DPO_TRAIN_SEED fixed at 777 across arms AND bank seeds -- the wave varies
# preference sampling, never optimizer randomness. Production: ea0.4 x
# bk {100..109} (10 pairs) + ea0.2 x bk {100..104} (5 pairs,
# expected-null control) = 15 pair jobs = 30 arm trajectories. Smoke = 1
# pair, 3 rounds, bk900. Idempotence is PAIR-level (both arms + bank +
# pair_meta.json, see the wrapper); gate arms with check_pofd_sanity and
# pairs with check_dpo_pair.py.
DPO_MR_TRAIN_SEED = 777
DPO_MR_EA_BANKS = [(0.4, range(100, 110)), (0.2, range(100, 105))]
ROW_MR = ("{tag}, dpo, 0, 0, 1, replace, 1.0, fixed, ab, "
          "0.0, 0.0, 1.0, loop, 0.0, {eps_ai}, {bankseed}")

_DPO_MR_TAIL = """\
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_dpo_pair_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = 160G
request_disk      = 40G
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action EPS_AI=$(eps_ai) DPO_BANK_SEED=$(bankseed) DPO_TRAIN_SEED=777 DPO_BETA=0.5 DPO_TAU=12.0 DPO_GEN_TEMP=1.0 DPO_MAX_STEPS=3 DPO_N_PAIRS=0 SFT_EPOCHS=0 SFT_LR=5e-5 LORA_R=128 USE_LORA=1 DO_SAMPLE=0 FRESH_EACH_ROUND=1 TRAIN_CAP=723 N_ROUNDS={nr} EPOCH_SIZE=100 BASE_MODEL=Qwen/Qwen2.5-7B-Instruct SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=64 SEED_BASE_DATA=1 WANDB_RUN_SUFFIX={suffix}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, bankseed from experiments/condor/{cfg}
"""

DPO_MR_SUB = ("""\
# HTCondor: MATCHED-RANDOMNESS DPO PAIRS (pofddpomr_), Qwen-7B.
# GENERATED by gen_pofd_sweep.py from the DPO MR block (2026-08-13).
# Never edit by hand: edit the block and rerun the script.
# 15 PAIR jobs (= 30 arm trajectories): ea0.4 x bank seeds {100..109} +
# ea0.2 x {100..104} (expected-null control), all db0.5 sharp-env cells
# at pop seed 0. Each job runs closed (bank writer) then open (bank
# reader) on the SAME GPU; shared round-0 fork + shared candidates/BT
# uniforms; only the judge differs from round 1. RLHF_FEEDBACK and
# DPO_BANK_MODE are set by the WRAPPER per arm -- never here.
# NOTE $(tag) is the CLOSED tag; open/bank names derive from it.
# PREREQ: qwen7b_dpo_mr_smoke pair PASSED check_dpo_pair.py.
# GATE per pull: check_pofd_sanity on each arm dir + check_dpo_pair.py
# per pair (bank bit-identity, round-0 fork, twin==judge).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> qwen7b_dpo_mr
""" + _DPO_MR_TAIL.format(nr=30, suffix="_qwen7b_pofddpomr",
                          cfg="configs_pofd_qwen7b_dpo_mr.txt"))

DPO_MR_SMOKE_SUB = ("""\
# HTCondor: MATCHED-RANDOMNESS DPO PAIR SMOKE, 1 pair, 3 rounds (bk900).
# GENERATED by gen_pofd_sweep.py from the DPO MR block (2026-08-13).
# Never edit by hand: edit the block and rerun the script.
# First run of the writer/reader bank path end-to-end: verify in the
# .out that the writer logs bank=write + round-0 state save, the reader
# logs the round-0 fork + pristine asserts, and check_dpo_pair.py PASSES
# (the wrapper runs it; nonzero exit -> hold). Then check_pofd_sanity on
# both pulled arm dirs.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> qwen7b_dpo_mr_smoke
""" + _DPO_MR_TAIL.format(nr=3, suffix="_qwen7b_pofddpomrsmk",
                          cfg="configs_pofd_qwen7b_dpo_mr_smoke.txt"))

# MISTRAL SFT CUBE (2026-08-13, user): the three-seed cube on the 4th model,
# comparable to the qwen/olmo cubes but on MODEL-SPECIFIC grids --
#   beta {0, 0.5, 1} x ea {0.05,0.1,0.2,0.4}
#   x es {0,0.10,0.15,0.20,0.25,0.30} x seeds {0,42,43} = 216 cells.
# The qwen/olmo cube constants (CUBE_BETAS/CUBE_SEEDS/CUBE_MODELS/
# CUBE_EXISTING) are UNTOUCHED; their generated files stay byte-identical
# (gated by --verify). Tags reuse cube_tag() -- per-dose home families
# (es=0 -> pofdw2f_, es=0.2 -> pofdws2f_, else pofdesf_) -- and ROW_CUBE;
# fixed dials: W=0.5, INNATE_LAMBDA=0.2, gamma=0, corrected nested
# operator, 30 rounds, fresh LoRA-512 + replace, SFT_EPOCHS=1, LR 5e-5,
# 723 agents, WITH_TWIN=1 (es=0 rows save the matched twin; es>0
# instantiates it automatically), forward SFT-KL at b>0 / ordinary sft at
# b0. Resources/env reuse the validated mistral7b_ws2f settings (lustre
# offline cache, 128G/40G, PPL_BATCH 64, g106 guard). No smoke: the
# mistral SFT and forward-KL production paths passed via the ws2f smoke +
# 6-job wave; b0p5 only changes the queue-fed KL weight.
# AUDIT 2026-08-13 (config fields + trajectory completeness over EVERY
# mistral dir, local == cluster 8/8 dirs, script in session scratchpad --
# NOT tag-matching): exactly 6 cells complete, the ws2f wave b {0,1} x
# ea0.4 x es0.2 x s {0,42,43}. The ws2f smoke (3 rounds) and
# mlatZ_mistral7b_s0 (frozen, W=0.3, 1 round) were REJECTED by the audit.
# 216 - 6 = 210 to queue: mistral7b_cube_s0 (70) -> gate -> then
# mistral7b_cube_repl (140); umbrella key mistral7b_cube = both.
MISTRAL_CUBE_BETAS = [0.0, 0.5, 1.0]
MISTRAL_CUBE_REPL_SEEDS = [42, 43]
MISTRAL_CUBE_EXISTING = {(b, 0.4, 0.20, s) for b in (0.0, 1.0)
                         for s in (0, 42, 43)}
assert len(MISTRAL_CUBE_EXISTING) == 6


def mistral_cube_rows(seeds):
    return [ROW_CUBE.format(
        tag=cube_tag("mistral7b", b, ea, es, s),
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=s,
        es=f"{es:g}", eps_ai=f"{ea:g}")
        for b in MISTRAL_CUBE_BETAS for ea in CUBE_EAS for es in CUBE_ESS
        for s in seeds if (b, ea, es, s) not in MISTRAL_CUBE_EXISTING]


_MISTRAL_CUBE_TAIL = """\
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = 128G
request_disk      = 40G
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 EPS_AI=$(eps_ai) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 ANS_SAMPLE_K=16 ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 FRESH_EACH_ROUND=1 TRAIN_CAP=723 N_ROUNDS=30 EPOCH_SIZE=100 BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.3 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 USE_LORA=1 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=64 SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_mistral7b_lora512_pofdcube"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai from experiments/condor/{cfg}
"""

MISTRAL_CUBE_S0_SUB = ("""\
# HTCondor: MISTRAL SFT CUBE, SEED-0 SLAB (mistral7b_cube_s0).
# GENERATED by gen_pofd_sweep.py from the MISTRAL SFT CUBE block
# (2026-08-13). Never edit by hand: edit the block and rerun the script.
# 70 audited-missing seed-0 cells of beta {0,0.5,1} x ea
# {0.05,0.1,0.2,0.4} x es {0,0.10,0.15,0.20,0.25,0.30} (72 minus the 2
# reused ws2f cells b{0,1}_ea0p4_es0p2_s0; the 2026-08-13 config-field
# audit is recorded as MISTRAL_CUBE_EXISTING). Forward SFT-KL at b>0,
# ordinary sft at b0; W0.5/l0.2/gamma0, corrected operator, 30 rounds,
# fresh LoRA-512 + replace, WITH_TWIN=1, movielens Action, 723 agents.
# Tags in the per-dose home families (pofdw2f_/pofdws2f_/pofdesf_).
# GATE every pull with check_pofd_sanity (es=0 exact replay, es>0 peer
# gate + twin; _b token style/beta, forward at b>0, _ea/_es/_s tokens,
# slug->base). Inspect completeness BEFORE releasing the repl slab.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> mistral7b_cube_s0
#         (umbrella mistral7b_cube = s0 + repl -- only after s0 gates)
""" + _MISTRAL_CUBE_TAIL.format(cfg="configs_pofd_mistral7b_cube_s0.txt"))

MISTRAL_CUBE_REPL_SUB = ("""\
# HTCondor: MISTRAL SFT CUBE, REPLICATION SLAB (mistral7b_cube_repl).
# GENERATED by gen_pofd_sweep.py from the MISTRAL SFT CUBE block
# (2026-08-13). Never edit by hand: edit the block and rerun the script.
# 140 audited-missing seed-42/43 cells (144 minus the 4 reused ws2f
# cells b{0,1}_ea0p4_es0p2_s{42,43}); same design as the s0 slab --
# together they complete the 216-cell three-seed mistral cube with the
# 6 ws2f cells reused, per the MISTRAL_CUBE_EXISTING audit record.
# PREREQ: mistral7b_cube_s0 pulled + gated + completeness-inspected.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> mistral7b_cube_repl
""" + _MISTRAL_CUBE_TAIL.format(cfg="configs_pofd_mistral7b_cube_repl.txt"))
# SFT-ICL REACH (2026-08-13, user), keys sft_icl_reach[_smoke|_baseline]
# + sft_icl_reach_{qwen,olmo,mistral}: with peer interaction OFF, how do
# shared weight updates vs fixed vs refreshed prompt context change the
# set of agents the platform can REACH? Four arms x 3 models x 6 gates
# x 5 seeds = 360 conceptual main trajectories:
#   b0  = ordinary SFT (beta 0, 1 epoch/round, fresh LoRA-512 + replace)
#   b1  = forward SFT-KL (beta 1), otherwise identical to b0
#   fz0 = frozen weights, K=8 random live ICL context FROZEN verbatim at
#         round 0 (ICL_SNAPSHOT_ROUND=0)
#   dyn = frozen weights, K=8 context REBUILT from the live population
#         every round (ICL_SNAPSHOT_ROUND=-1); matched fz0/dyn cells
#         share bit-identical round-0 contexts by the selection-RNG
#         construction (checker-enforced)
# Gates: ea {0.05, 0.1, 0.2, 0.4, 0.7} + the EXPLICIT all-open gate
# (AI_GATE_MODE=all_open, tag token _eaopen_). all_open is NOT eps_ai=1:
# the threshold gate is strict-<, so it must never be disguised as a
# numeric dose. The mode is a new opt-in config field; "threshold" is
# byte-identical (same expression, RNG-free) to every earlier run, and
# runner + checker share ONE gate definition (_gated_pop.ai_gate).
# Env: W=0.5, lam=0.2, es=0 (no peer step ever), gamma=0, greedy
# serving, 30 rounds, WITH_TWIN=1, movielens Action, 723 agents.
# BASELINES (15 = 3 models x 5 seeds, pofdreachbase_): one-round
# frozen-weight NO-context K=0 probes at EPS_AI=0 -- the strict gate
# never opens, so the probe cannot update opinions; pred_raw[0] is the
# frozen no-context prediction vector m_base defining the shared
# pre-intervention cohort U_common(eps) = {i : |m_base_i - innate_i|
# >= eps} that analyze_sft_icl_reach.py derives OFFLINE (never in the
# population-update path).
# REUSE AUDIT (2026-08-13, audit_sft_icl_reach_reuse.py -- BY CONFIG
# FIELDS + 30-round completeness over runs/pokec_gated_lm +
# notes/pofd/cluster, never tag similarity): EXACTLY 33 cells exist --
# qwen 21 (b0 6: w2 s0 plane + fes s42/43 @ea0p4; b1 8: w2f s0 plane +
# s42/43 @ea{0p05,0p4}; fz0 3 + dyn 3: iclf es0 s0; dyn ea0p05 s0:
# icl2 k8live) and olmo 12 (b0/b1 w2f s0 planes; dyn 4: icl2 k8live
# s0). 327 main trajectories queue (qwen 99, olmo 108, mistral 120).
# The audited map is experiments/condor/manifest_sft_icl_reach.json --
# the generator consumes it and HARD-ASSERTS the audited counts; rerun
# the audit script if the local corpus changes.
# HARDWARE: borderline generations flip across GPU architectures
# (2026-08-07 iclf finding). The wave stays schedulable (>=80GB, g106
# excluded, no capability pin -- pinning one architecture at BID-25
# prices risks starving 342 jobs); instead every run records a config
# "hardware" block (host/GPU/CC/CUDA/torch/transformers) and the
# analysis marks heterogeneous-hardware blocks for paired use. The
# checker's cross-run identity checks (fz0<->dyn round-0 context) are
# CPU-derived and hold across GPUs -- never weakened for hardware.
# Smoke (6 jobs, 3 rounds): qwen b0 + dyn at all-open (gates the new
# AI_GATE_MODE path), olmo/mistral fz0 + dyn at ea0p1 (first-ever
# OLMo/Mistral ICL + the never-run-on-these-models snapshot path,
# fixed/dynamic round-0 identity, frozen twin, new checker branches).
# Flow: smoke -> gate -> baselines -> gate -> full production key.
REACH_MANIFEST_PATH = os.path.join(HERE, "manifest_sft_icl_reach.json")
REACH_EXPECT_REUSED = 33
REACH_EXPECT_NEW = 327
REACH_EXPECT_NEW_PER_MODEL = {"qwen7b": 99, "olmo7b": 108, "mistral7b": 120}
REACH_SEEDS = [0, 42, 43, 44, 45]
REACH_GATES = [0.05, 0.1, 0.2, 0.4, 0.7, "open"]
REACH_ARMS = ["b0", "b1", "fz0", "dyn"]
REACH_KEY = {"qwen7b": "sft_icl_reach_qwen", "olmo7b": "sft_icl_reach_olmo",
             "mistral7b": "sft_icl_reach_mistral"}
# resource/env deltas per model (mirrors CUBE_MODELS + the mistral cube
# tail, which are defined later in this file -- kept literal here so the
# block is self-contained at import time)
REACH_MODELS = {
    "qwen7b": {"base_model": "Qwen/Qwen2.5-7B-Instruct",
               "mem": "128G", "disk": "40G", "ppl_batch": 64,
               "extra_env": ""},
    "olmo7b": {"base_model": "allenai/OLMo-2-1124-7B-Instruct",
               "mem": "160G", "disk": "60G", "ppl_batch": 16,
               "extra_env": "HF_HOME=/lustre/fast/fast/gsmithline/hf_cache "
                            "HF_HUB_OFFLINE=1 "},
    "mistral7b": {"base_model": "mistralai/Mistral-7B-Instruct-v0.3",
                  "mem": "128G", "disk": "40G", "ppl_batch": 64,
                  "extra_env": "HF_HOME=/lustre/fast/fast/gsmithline/"
                               "hf_cache HF_HUB_OFFLINE=1 "},
}
# per-arm queue payloads (cols 16-23): ICL knobs, adapter mode, telemetry.
# gg mirrors the family each arm extends (cube waves ran without gender
# gaps, iclf with them) so reused and new cells share telemetry surfaces.
REACH_ARM_COLS = {
    "b0": dict(style="sft", beta="0", iclk=0, snap=-1, uselora=1, fresh=1,
               ansk=16, gg=0),
    # b0xa (mistral_clamp_exclude_a wave, 2026-08-18): the exact b0
    # envelope; "xa" = fixed cohort A eXcluded from Adaptation. The
    # exclusion itself is NOT a queue col -- SFT_EXCLUDE_CLAMPED=1 is
    # pinned in the CLAMP_XA sub template env (every row of that key
    # is the exclusion arm)
    "b0xa": dict(style="sft", beta="0", iclk=0, snap=-1, uselora=1,
                 fresh=1, ansk=16, gg=0),
    "b1": dict(style="sft_kl", beta="1", iclk=0, snap=-1, uselora=1,
               fresh=1, ansk=16, gg=0),
    # b0p5 (fig2 family-prior scout, 2026-08-17): forward-KL SFT at the
    # intermediate beta=0.5 dose; otherwise the exact b1 envelope
    "b0p5": dict(style="sft_kl", beta="0.5", iclk=0, snap=-1, uselora=1,
                 fresh=1, ansk=16, gg=0),
    # b2/b4/b8 (fig2 beta scout, 2026-08-18): the high-retention forward-
    # KL doses; everything except the coefficient copies the b1 envelope
    "b2": dict(style="sft_kl", beta="2", iclk=0, snap=-1, uselora=1,
               fresh=1, ansk=16, gg=0),
    "b4": dict(style="sft_kl", beta="4", iclk=0, snap=-1, uselora=1,
               fresh=1, ansk=16, gg=0),
    "b8": dict(style="sft_kl", beta="8", iclk=0, snap=-1, uselora=1,
               fresh=1, ansk=16, gg=0),
    "fz0": dict(style="frozen", beta="0", iclk=8, snap=0, uselora=0,
                fresh=0, ansk=0, gg=1),
    "dyn": dict(style="frozen", beta="0", iclk=8, snap=-1, uselora=0,
                fresh=0, ansk=0, gg=1),
    # k0 (sft_k0_nopeer wave, 2026-08-14): frozen NO-context prompting --
    # repeated zero-shot serving; no LoRA, no demonstrations, no memory
    "k0": dict(style="frozen", beta="0", iclk=0, snap=-1, uselora=0,
               fresh=0, ansk=0, gg=0),
    # K=32 context depth (sft_icl_ctxgrid wave, 2026-08-15): the same two
    # context regimes as fz0/dyn at 4x the exemplar count. f32 freezes
    # each agent's context verbatim after round 0; d32 rebuilds it every
    # round. run_one_pokec_gated.sh drops GEN_BATCH_SIZE to 8 for
    # ICL_K >= 16 on its own (the ~2k-token prompts overflow the 80GB
    # KV-cache at batch 32) -- no per-arm resource override needed here.
    "f32": dict(style="frozen", beta="0", iclk=32, snap=0, uselora=0,
                fresh=0, ansk=0, gg=1),
    "d32": dict(style="frozen", beta="0", iclk=32, snap=-1, uselora=0,
                fresh=0, ansk=0, gg=1),
    # d8 (clamp-graph personal-history wave, 2026-08-17): frozen weights,
    # ZERO cross-user exemplars (ICL_K=0) -- each agent's prompt carries
    # only its OWN latest eight recorded opinions, oldest to newest
    # (ICL_DAYS=8 rides the d8 sub template's icldays queue col). Fixed
    # agents therefore see only repetitions of their own innate opinion.
    "d8": dict(style="frozen", beta="0", iclk=0, snap=-1, uselora=0,
               fresh=0, ansk=0, gg=1),
}
ROW_REACH = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, ab, "
             "0.0, 0.0, 0.5, loop, 0.0, {eps_ai}, {gatemode}, {iclk}, "
             "{snap}, {uselora}, {fresh}, {ansk}, {gg}, {nrounds}")
REACH_SMOKES = [("qwen7b", "b0", "open"), ("qwen7b", "dyn", "open"),
                ("olmo7b", "fz0", 0.1), ("olmo7b", "dyn", 0.1),
                ("mistral7b", "fz0", 0.1), ("mistral7b", "dyn", 0.1)]


def _reach_gate_tok(gate):
    return "eaopen" if gate == "open" else f"ea{_num(gate)}"


def reach_tag(model, arm, gate, seed, prefix="pofdreach"):
    return (f"{prefix}_{model}_{arm}_{_reach_gate_tok(gate)}_{w_tok()}"
            f"_es0_s{seed}")


def reach_base_tag(model, seed):
    return f"pofdreachbase_{model}_{w_tok()}_es0_s{seed}"


def reach_row(model, arm, gate, seed, nrounds=30, prefix="pofdreach"):
    return ROW_REACH.format(
        tag=reach_tag(model, arm, gate, seed, prefix), seed=seed,
        eps_ai="0" if gate == "open" else f"{gate:g}",
        gatemode="all_open" if gate == "open" else "threshold",
        nrounds=nrounds, **REACH_ARM_COLS[arm])


def reach_base_row(model, seed):
    # 1-round frozen K=0 probe; EPS_AI=0 under the strict threshold gate
    # -> no contacts, opinions untouched (the no-update guarantee)
    return ROW_REACH.format(
        tag=reach_base_tag(model, seed), style="frozen", beta="0",
        seed=seed, eps_ai="0", gatemode="threshold", iclk=0, snap=-1,
        uselora=0, fresh=0, ansk=0, gg=0, nrounds=1)


def _reach_manifest():
    with open(REACH_MANIFEST_PATH) as fh:
        return json.load(fh)


def reach_rows(model):
    """Rows for the audited-MISSING cells of one model, manifest-driven.
    The generated tag must equal the manifest's recorded tag per cell."""
    rows = []
    for c in _reach_manifest()["cells"]:
        if c["model"] == model and c["status"] == "new":
            r = reach_row(model, c["arm"], c["gate"], c["seed"])
            assert r.split(",")[0] == c["run_tag"], (r, c["run_tag"])
            rows.append(r)
    return rows


def reach_smoke_rows(model):
    return [reach_row(m, arm, gate, 0, nrounds=3, prefix="pofdreachsmk")
            for m, arm, gate in REACH_SMOKES if m == model]


# SEED-0 EXPLORATORY SLAB (2026-08-13, user: "do not release all 327
# yet ... run a seed-0 exploratory grid first"): the reach question is
# no longer the paper's central claim, so the five-seed grid waits on
# what seed 0 shows. Slab = 3 models x 4 arms x gates
# {0.05, 0.1, 0.2, 0.4, open} x seed 0 = 60 conceptual cells, of which
# 27 are audited-reused (qwen 15, olmo 12, mistral 0) -> 33 jobs
# (qwen 5: fz0@ea0p05 + the 4 all-open arms; olmo 8: fz0 x 4 numeric
# gates + 4 all-open arms; mistral 20: everything). ea0p7 is EXCLUDED
# from the slab (baseline probes showed its common cohort empty for
# olmo/mistral). Rows are the SAME rows (same tags) as the full
# per-model production files -- NEVER co-submit sft_icl_reach_s0 with
# sft_icl_reach (double-queue write race); releasing the full key
# later no-ops these 33 via the idempotent executable.
REACH_S0_GATES = [0.05, 0.1, 0.2, 0.4, "open"]
REACH_S0_EXPECT = {"qwen7b": 5, "olmo7b": 8, "mistral7b": 20}


def reach_s0_rows(model):
    rows = []
    for c in _reach_manifest()["cells"]:
        if (c["model"] == model and c["status"] == "new"
                and c["seed"] == 0 and c["gate"] in REACH_S0_GATES):
            r = reach_row(model, c["arm"], c["gate"], 0)
            assert r.split(",")[0] == c["run_tag"], (r, c["run_tag"])
            rows.append(r)
    return rows


# SFT vs FROZEN NO-CONTEXT PROMPTING, NO PEERS (2026-08-14, user), keys
# sft_k0_nopeer[_smoke] + sft_k0_nopeer_{qwen,olmo,mistral}: with peers
# off, does SFT transmit population information GLOBALLY through shared
# weights while frozen no-context prompting (k0 = repeated zero-shot
# serving; NOT adaptive ICL -- ICL_K=0, USE_LORA=0, nothing changes
# between rounds) acts only as a FIXED external signal? Descriptive
# seed-0 grid: 3 models x arms {b0, b1, k0} x numeric gates
# {0.05, 0.1, 0.2, 0.4, 0.7, 1.0} = 54 conceptual cells, EVERY cell
# AI_GATE_MODE=threshold. ea=1.0 is the STRICT numeric threshold
# (token _ea1_), deliberately distinct from all_open -- the completed
# _eaopen_ scout cells are NOT reusable for it, and an agent at
# distance exactly 1 stays rejected.
# REUSE AUDIT (audit_sft_k0_nopeer_reuse.py, config fields + 30-round
# completeness): EXACTLY 32 reused / 22 missing -- qwen/olmo b0+b1+k0
# at the four <=0.4 gates reuse the w2/w2f planes + icl2/iclf k0 runs;
# mistral b0/b1 <=0.4 reuse the sft_icl_reach s0 slab; missing = the
# ea0p7/ea1 ladder tops (b0/b1/k0 x 3 models = 18... of which b0/b1
# ea0p7 (6) carry the SAME pofdreach_ tags as the parked full reach
# grid so its eventual release no-ops them) + mistral k0 at the four
# low gates (4). Per model 6/6/10, per arm b0 6 / b1 6 / k0 10, per
# gate 1/1/1/1/9/9 -- all hard-asserted from
# manifest_sft_k0_nopeer.json.
# NEVER co-submit sft_k0_nopeer with sft_icl_reach (the 6 shared
# b0/b1 ea0p7 tags would double-queue). Baselines: the completed
# seed-0 pofdreachbase_ probes are reused (nothing new queues).
# Smokes (2 x 3 rounds): mistral k0 ea0p1 (the never-run no-context
# trajectory path with rejected agents) + qwen b0 ea1 (the _ea1_
# numeric-threshold tag path).
K0_MANIFEST_PATH = os.path.join(HERE, "manifest_sft_k0_nopeer.json")
K0_EXPECT_REUSED = 32
K0_EXPECT_NEW = 22
K0_EXPECT_NEW_PER_MODEL = {"qwen7b": 6, "olmo7b": 6, "mistral7b": 10}
K0_EXPECT_NEW_PER_ARM = {"b0": 6, "b1": 6, "k0": 10}
K0_EXPECT_NEW_PER_GATE = {"0.05": 1, "0.1": 1, "0.2": 1, "0.4": 1,
                          "0.7": 9, "1.0": 9}
K0_KEY = {"qwen7b": "sft_k0_nopeer_qwen", "olmo7b": "sft_k0_nopeer_olmo",
          "mistral7b": "sft_k0_nopeer_mistral"}
K0_SMOKES = [("mistral7b", "k0", 0.1), ("qwen7b", "b0", 1.0)]


def _k0_manifest():
    with open(K0_MANIFEST_PATH) as fh:
        return json.load(fh)


def k0_rows(model):
    rows = []
    for c in _k0_manifest()["cells"]:
        if c["model"] == model and c["status"] == "new":
            r = reach_row(model, c["arm"], c["gate"], c["seed"])
            assert r.split(",")[0] == c["run_tag"], (r, c["run_tag"])
            rows.append(r)
    return rows


def k0_smoke_rows(model):
    return [reach_row(m, arm, gate, 0, nrounds=3, prefix="pofdreachsmk")
            for m, arm, gate in K0_SMOKES if m == model]


# THREE-SEED NO-PEER SFT/ICL GATE GRID (2026-08-14, user), keys
# sft_icl_nopeer_grid3[_smoke] + sft_icl_nopeer_grid3_{qwen,olmo,
# mistral}: complete the channel comparison -- k0 (plain prompting,
# K=0) / fz0 (fixed round-0 K=8 context) / dyn (refreshed live K=8
# context) / b0 (ordinary SFT) -- over 3 models x numeric gates
# {0.05, 0.1, 0.2, 0.4, 1.0} x seeds {0, 42, 43} = 180 conceptual
# cells, all threshold mode (_ea1_ strict numeric, never _eaopen_),
# canonical no-peer env (W0.5/l0.2/es0/gamma0, 30 rounds, WITH_TWIN=1,
# greedy, corrected operator).
# AUDIT (audit_sft_icl_nopeer_grid3_reuse.py ->
# manifest_sft_icl_nopeer_grid3.json): 58 complete (the prediction was
# 56 -- two GENUINE field-exact k0 cells existed unpredicted:
# pofdicl2_qwen7b ea0p4 k0 s42/s43 from the icls2x replicates; queue
# scope unaffected), 122 literally missing of which 28 are k0 seed
# repetitions mapped as DETERMINISTIC REFERENCES to the seed-0 k0 runs
# (k0 draws nothing the trajectory consumes; re-running would
# manufacture artificial replication) -> EXACTLY 94 informative jobs:
# b0 28 / fz0 33 / dyn 33 / k0 0; qwen 30 / olmo 32 / mistral 32.
# Tags: the reach family -- cells shared with the HELD full reach wave
# (numeric gates <= 0.4) carry byte-identical pofdreach_ tags so a
# later broad release no-ops them; the _ea1_ cells (fz0/dyn all seeds
# + b0 s42/43) are new tags. NEVER submit concurrently with
# sft_icl_reach[_s0] (shared tags = write race). Smokes (2 x 3
# rounds, OUTSIDE the 94): mistral fz0 + dyn at the never-run
# threshold-ea1 x context combination.
GRID3_MANIFEST_PATH = os.path.join(HERE,
                                   "manifest_sft_icl_nopeer_grid3.json")
GRID3_EXPECT_REUSED = 58
GRID3_EXPECT_NEW = 94
GRID3_EXPECT_REFERENCE = 28
GRID3_EXPECT_NEW_PER_MODEL = {"qwen7b": 30, "olmo7b": 32, "mistral7b": 32}
GRID3_EXPECT_NEW_PER_ARM = {"k0": 0, "fz0": 33, "dyn": 33, "b0": 28}
GRID3_KEY = {"qwen7b": "sft_icl_nopeer_grid3_qwen",
             "olmo7b": "sft_icl_nopeer_grid3_olmo",
             "mistral7b": "sft_icl_nopeer_grid3_mistral"}
GRID3_SMOKES = [("mistral7b", "fz0", 1.0), ("mistral7b", "dyn", 1.0)]


def _grid3_manifest():
    with open(GRID3_MANIFEST_PATH) as fh:
        return json.load(fh)


def grid3_rows(model):
    rows = []
    for c in _grid3_manifest()["cells"]:
        if c["model"] == model and c["status"] == "new":
            r = reach_row(model, c["arm"], c["gate"], c["seed"])
            assert r.split(",")[0] == c["run_tag"], (r, c["run_tag"])
            rows.append(r)
    return rows


def grid3_smoke_rows():
    return [reach_row(m, arm, gate, 0, nrounds=3, prefix="pofdreachsmk")
            for m, arm, gate in GRID3_SMOKES]


def grid3_sub(model, kind):
    """kind: 'main' | 'smoke' -- rides the REACH sub template."""
    key = (GRID3_KEY[model] if kind == "main"
           else "sft_icl_nopeer_grid3_smoke")
    n_jobs = (len(grid3_rows(model)) if kind == "main"
              else len(grid3_smoke_rows()))
    what = {"main": ("THREE-SEED NO-PEER SFT/ICL GATE GRID -- "
                     "audited-informative cells (30 rounds; _ea1_ strict "
                     "numeric threshold; k0 seed repetitions are "
                     "deterministic references, never queued; NEVER "
                     "co-submit with sft_icl_reach[_s0]: numeric-gate "
                     "tags are shared)"),
            "smoke": ("sft_icl_nopeer_grid3 SMOKE (3 rounds; threshold "
                      "ea1 x fixed/refreshed context, first exercise)")
            }[kind]
    return REACH_SUB_TEMPLATE.format(model=model, key=key, n_jobs=n_jobs,
                                     what=what, **REACH_MODELS[model])


# EPS_SOCIAL=0.2 SFT/ICL CHANNEL TABLE (2026-08-14, user), keys
# sft_icl_peer02[_smoke] + {qwen7b,olmo7b,mistral7b}_sft_icl_peer02:
# complete the peer-live channel comparison -- b0 (ordinary SFT) / k0
# (frozen plain prompting) / fz0 (fixed round-0 K=8 context) / dyn
# (refreshed live K=8 context) -- over 3 models x gates {0.1, 0.4} x
# ES=0.2 x seeds {0, 42, 43} = 72 conceptual cells, all threshold mode.
# Canonical env otherwise (W0.5/l0.2/gamma0, 30 rounds, WITH_TWIN=1,
# greedy, replace + 723 labels, corrected operator, one peer sweep).
# AUDIT (audit_sft_icl_peer02_reuse.py -> manifest_sft_icl_peer02.json,
# config fields + completeness + per-run checker validation): EXACTLY
# 27 reused (qwen 16 / olmo 8 / mistral 3, all validation PASS) / 45
# missing (qwen 8 / olmo 16 / mistral 21; b0 9 / k0 10 / fz0 16 /
# dyn 10), every count hard-asserted. NEW family pofdpeer2_ (es0p2
# token; no tags shared with any other wave -- collision-asserted).
# Smokes (2 x 3 rounds, seed 991, OUTSIDE the 45): mistral fz0 + dyn
# at ea0p1 es0p2 -- the never-run Mistral peer-context paths.
PEER2_MANIFEST_PATH = os.path.join(HERE, "manifest_sft_icl_peer02.json")
PEER2_EXPECT_REUSED = 27
PEER2_EXPECT_NEW = 45
PEER2_EXPECT_NEW_PER_MODEL = {"qwen7b": 8, "olmo7b": 16, "mistral7b": 21}
PEER2_EXPECT_REUSED_PER_MODEL = {"qwen7b": 16, "olmo7b": 8, "mistral7b": 3}
PEER2_KEY = {m: f"{m}_sft_icl_peer02" for m in
             ("qwen7b", "olmo7b", "mistral7b")}
PEER2_SMOKES = [("mistral7b", "fz0", 0.1), ("mistral7b", "dyn", 0.1)]
PEER2_SMOKE_SEED = 991
ROW_PEER2 = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, ab, "
             "0.2, 0.0, 0.5, loop, 0.0, {eps_ai}, threshold, {iclk}, "
             "{snap}, {uselora}, {fresh}, {ansk}, {gg}, {nrounds}")


def peer2_tag(model, arm, gate, seed, prefix="pofdpeer2"):
    return (f"{prefix}_{model}_{arm}_ea{_num(gate)}_{w_tok()}_es0p2"
            f"_s{seed}")


def peer2_row(model, arm, gate, seed, nrounds=30, prefix="pofdpeer2"):
    a = REACH_ARM_COLS[arm]
    return ROW_PEER2.format(
        tag=peer2_tag(model, arm, gate, seed, prefix), style=a["style"],
        beta=a["beta"], seed=seed, eps_ai=f"{gate:g}", iclk=a["iclk"],
        snap=a["snap"], uselora=a["uselora"], fresh=a["fresh"],
        ansk=a["ansk"], gg=a["gg"], nrounds=nrounds)


def _peer2_manifest():
    with open(PEER2_MANIFEST_PATH) as fh:
        return json.load(fh)


def peer2_rows(model):
    rows = []
    for c in _peer2_manifest()["cells"]:
        if c["model"] == model and c["status"] == "new":
            r = peer2_row(model, c["arm"], c["gate"], c["seed"])
            assert r.split(",")[0] == c["run_tag"], (r, c["run_tag"])
            rows.append(r)
    return rows


def peer2_smoke_rows():
    return [peer2_row(m, arm, gate, PEER2_SMOKE_SEED, nrounds=3,
                      prefix="pofdpeer2smk")
            for m, arm, gate in PEER2_SMOKES]


def peer2_sub(model, kind):
    """kind: 'main' | 'smoke' -- rides the REACH sub template (the row
    schema is identical; es rides queue col 10)."""
    key = PEER2_KEY[model] if kind == "main" else "sft_icl_peer02_smoke"
    n_jobs = (len(peer2_rows(model)) if kind == "main"
              else len(peer2_smoke_rows()))
    what = {"main": ("EPS_SOCIAL=0.2 SFT/ICL CHANNEL TABLE -- audited-"
                     "missing cells (30 rounds, peer step LIVE, twin "
                     "moves; pofdpeer2_ family, no shared tags)"),
            "smoke": ("sft_icl_peer02 SMOKE (3 rounds, seed 991; first "
                      "Mistral peer-context exercise)")}[kind]
    return REACH_SUB_TEMPLATE.format(model=model, key=key, n_jobs=n_jobs,
                                     what=what, **REACH_MODELS[model])


def k0_sub(model, kind):
    """kind: 'main' | 'smoke' -- rides the REACH sub template."""
    short = K0_KEY[model].split("_")[-1]
    key = K0_KEY[model] if kind == "main" else f"sft_k0_nopeer_smoke_{short}"
    n_jobs = (len(k0_rows(model)) if kind == "main"
              else len(k0_smoke_rows(model)))
    what = {"main": ("SFT vs FROZEN NO-CONTEXT PROMPTING (k0), no peers "
                     "-- audited-missing seed-0 cells (30 rounds; _ea1_ "
                     "is the strict numeric threshold, never all_open; "
                     "NEVER co-submit with sft_icl_reach: 6 b0/b1 ea0p7 "
                     "tags are shared)"),
            "smoke": "sft_k0_nopeer SMOKE (3 rounds)"}[kind]
    return REACH_SUB_TEMPLATE.format(model=model, key=key, n_jobs=n_jobs,
                                     what=what, **REACH_MODELS[model])


# mistral_sft_icl_gate2d[_smoke] (2026-08-15): the Mistral TWO-DIMENSIONAL
# gate grid behind the diagonal-split heatmap -- ordinary SFT (b0,
# beta=0) vs live-context ICL (dyn, frozen weights, K=8 refreshed every
# round) across eps_AI {0.05,0.1,0.2,0.4,1.0} x eps_social
# {0,0.2,0.4,1.0} x seeds {0,42,43} = 120 conceptual cells, mistral7b
# only. BOTH axes are real numeric thresholds: _ea1_ is EPS_AI=1.0 under
# the strict-< gate (never all_open), _es1_ is EPS_SOCIAL=1.0, the real
# peer-confidence gate.
# AUDIT (audit_sft_icl_gate2d_reuse.py -> manifest_sft_icl_gate2d.json,
# per-reused-cell config fingerprint + trajectory sha256 + live checker
# verdict): 42 reused (all 30 es=0 cells via reach/grid3 + 12 es=0.2
# cells at ea {0.1,0.4} via peer02/ws2f) -> 78 new (es0p2 18 + es0p4 30
# + es1 30; 39 per arm), every count hard-asserted. NEW family
# pofdgate2d_ -- zero tags shared with any other wave (collision-
# asserted). Smokes (2 x 3 rounds, seed 991, OUTSIDE the 78): b0 + dyn
# at ea1 es1 -- the never-run full-open corner, both arms.
GATE2D_MANIFEST_PATH = os.path.join(HERE, "manifest_sft_icl_gate2d.json")
GATE2D_EXPECT_REUSED = 42
GATE2D_EXPECT_NEW = 78
GATE2D_EXPECT_NEW_PER_ES = {"0": 0, "0p2": 18, "0p4": 30, "1": 30}
GATE2D_EXPECT_REUSED_PER_ES = {"0": 30, "0p2": 12, "0p4": 0, "1": 0}
GATE2D_EXPECT_NEW_PER_ARM = {"b0": 39, "dyn": 39}
GATE2D_KEY = "mistral_sft_icl_gate2d"
GATE2D_SMOKE_KEY = "mistral_sft_icl_gate2d_smoke"
GATE2D_SMOKES = [("b0", 1.0, 1.0), ("dyn", 1.0, 1.0)]  # (arm, ea, es)
GATE2D_SMOKE_SEED = 991
ROW_GATE2D = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
              "ab, {es}, 0.0, 0.5, loop, 0.0, {eps_ai}, threshold, "
              "{iclk}, {snap}, {uselora}, {fresh}, {ansk}, {gg}, "
              "{nrounds}")


def gate2d_tag(arm, gate, es, seed, prefix="pofdgate2d"):
    return (f"{prefix}_mistral7b_{arm}_ea{_num(gate)}_{w_tok()}"
            f"_es{_num(es)}_s{seed}")


def gate2d_row(arm, gate, es, seed, nrounds=30, prefix="pofdgate2d"):
    a = REACH_ARM_COLS[arm]
    return ROW_GATE2D.format(
        tag=gate2d_tag(arm, gate, es, seed, prefix), style=a["style"],
        beta=a["beta"], seed=seed, es=f"{es:g}", eps_ai=f"{gate:g}",
        iclk=a["iclk"], snap=a["snap"], uselora=a["uselora"],
        fresh=a["fresh"], ansk=a["ansk"], gg=a["gg"], nrounds=nrounds)


def _gate2d_manifest():
    with open(GATE2D_MANIFEST_PATH) as fh:
        return json.load(fh)


def gate2d_rows():
    """Rows for the audited-MISSING cells, manifest-driven. The generated
    tag must equal the manifest's recorded tag per cell."""
    rows = []
    for c in _gate2d_manifest()["cells"]:
        if c["status"] == "new":
            r = gate2d_row(c["arm"], c["gate"], c["eps_social"],
                           c["seed"])
            assert r.split(",")[0] == c["run_tag"], (r, c["run_tag"])
            rows.append(r)
    return rows


def gate2d_smoke_rows():
    return [gate2d_row(arm, gate, es, GATE2D_SMOKE_SEED, nrounds=3,
                       prefix="pofdgate2dsmk")
            for arm, gate, es in GATE2D_SMOKES]


def gate2d_sub(kind):
    """kind: 'main' | 'smoke' -- rides the REACH sub template (the row
    schema is identical; es rides queue col 10)."""
    key = GATE2D_KEY if kind == "main" else GATE2D_SMOKE_KEY
    n_jobs = (len(gate2d_rows()) if kind == "main"
              else len(gate2d_smoke_rows()))
    what = {"main": ("MISTRAL 2-D GATE GRID -- audited-missing cells "
                     "(30 rounds; SFT b0 vs live-ICL dyn x eps_AI x "
                     "eps_social, BOTH real numeric thresholds; "
                     "pofdgate2d_ family, no shared tags)"),
            "smoke": ("sft_icl_gate2d SMOKE (3 rounds, seed 991; the "
                      "ea1/es1 full-open corner, both arms)")}[kind]
    return REACH_SUB_TEMPLATE.format(model="mistral7b", key=key,
                                     n_jobs=n_jobs, what=what,
                                     **REACH_MODELS["mistral7b"])


# sft_icl_ctxgrid[_smoke] + sft_icl_ctxgrid_{qwen,olmo,mistral}
# (2026-08-15): the ONE-SEED context-depth x dual-gate grid. 3 models x 6
# adaptation channels x eps_AI {0.05,0.1,0.2,0.4,1.0} x eps_social
# {0,0.2,0.4,1.0} x seed 0 = 360 conceptual cells. Channels: b0 ordinary
# SFT (beta=0) / k0 frozen no-context prompting / fz0 fixed K=8 (snapshot
# round 0) / dyn live K=8 / f32 fixed K=32 (snapshot round 0) / d32 live
# K=32. Both gate axes are REAL numeric thresholds -- eps_AI=1.0 is the
# strict-< gate (never all_open) and eps_social=1.0 is the real peer-
# confidence gate.
# AUDIT (audit_sft_icl_ctxgrid_reuse.py ->
# manifest_sft_icl_ctxgrid.json): the spec's predicted 126/234 did NOT
# hold -- exact field matching finds 139 reusable / 221 new (user
# accepted the audited truth 2026-08-15). Reuse spans reach/grid3 33,
# icl2 18, iclf 18, icls2 18, peer02 9, gate2d 26, cube w2/w2f/ws2/ws2f
# 17. FIXED K=32 has ZERO prior runs (nothing pairs icl_k=32 with
# icl_snapshot_round=0), so all 60 f32 cells are new; the archived
# k32pri/k32noai runs are NOT fixed-K=32 context (icl_ctx_source
# pristine/noai) and the audit guards against ever admitting them.
# NEW family pofdctxgrid_ -- zero tags shared with any other wave
# (collision-asserted). Smokes (2 x 3 rounds, seed 991, OUTSIDE the
# 221): mistral f32 + d32 at ea1 es1 -- the never-run K=32 corner.
CTXGRID_MANIFEST_PATH = os.path.join(HERE, "manifest_sft_icl_ctxgrid.json")
# mistral7b K=32 (f32 + d32, 40 cells) is EXCLUDED, never queued: the
# seed-993 diagnostic measured parse_fail_frac=1.0 -- 100% digit-free
# prose generations -- so _parse serves its 0.5 default to every agent
# and the channel carries no signal. The cells stay in the manifest with
# status "excluded" so the grid still accounts for all 360.
CTXGRID_EXPECT_CELLS = 360
CTXGRID_EXPECT_REUSED = 139
CTXGRID_EXPECT_NEW = 181
CTXGRID_EXPECT_EXCLUDED = 40
CTXGRID_EXPECT_NEW_PER_MODEL = {"qwen7b": 77, "olmo7b": 78, "mistral7b": 26}
CTXGRID_EXPECT_NEW_PER_ARM = {"b0": 22, "k0": 35, "fz0": 38, "dyn": 22,
                              "f32": 40, "d32": 24}
CTXGRID_KEY = {m: f"sft_icl_ctxgrid_{m[:-2]}" for m in
               ("qwen7b", "olmo7b", "mistral7b")}
CTXGRID_SMOKE_KEY = "sft_icl_ctxgrid_smoke"
CTXGRID_SMOKES = [("mistral7b", "f32", 1.0, 1.0),
                  ("mistral7b", "d32", 1.0, 1.0)]
CTXGRID_SMOKE_SEED = 991
ROW_CTXGRID = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
               "ab, {es}, 0.0, 0.5, loop, 0.0, {eps_ai}, threshold, "
               "{iclk}, {snap}, {uselora}, {fresh}, {ansk}, {gg}, "
               "{nrounds}")


def ctxgrid_tag(model, arm, gate, es, seed, prefix="pofdctxgrid"):
    return (f"{prefix}_{model}_{arm}_ea{_num(gate)}_{w_tok()}"
            f"_es{_num(es)}_s{seed}")


def ctxgrid_row(model, arm, gate, es, seed, nrounds=30,
                prefix="pofdctxgrid"):
    a = REACH_ARM_COLS[arm]
    return ROW_CTXGRID.format(
        tag=ctxgrid_tag(model, arm, gate, es, seed, prefix),
        style=a["style"], beta=a["beta"], seed=seed, es=f"{es:g}",
        eps_ai=f"{gate:g}", iclk=a["iclk"], snap=a["snap"],
        uselora=a["uselora"], fresh=a["fresh"], ansk=a["ansk"],
        gg=a["gg"], nrounds=nrounds)


def _ctxgrid_manifest():
    with open(CTXGRID_MANIFEST_PATH) as fh:
        return json.load(fh)


def ctxgrid_rows(model):
    """Rows for the audited-MISSING cells of one model, manifest-driven.
    The generated tag must equal the manifest's recorded tag per cell."""
    rows = []
    for c in _ctxgrid_manifest()["cells"]:
        if c["model"] == model and c["status"] == "new":
            r = ctxgrid_row(model, c["arm"], c["gate"], c["eps_social"],
                            c["seed"])
            assert r.split(",")[0] == c["run_tag"], (r, c["run_tag"])
            rows.append(r)
    return rows


def ctxgrid_smoke_rows():
    return [ctxgrid_row(m, arm, gate, es, CTXGRID_SMOKE_SEED, nrounds=3,
                        prefix="pofdctxgridsmk")
            for m, arm, gate, es in CTXGRID_SMOKES]


# DIAGNOSTIC re-smoke (2026-08-15): the first pofdctxgridsmk_ pair came
# back with EVERY prediction at exactly 0.5 -- the runner's silent
# parse-failure default -- for mistral7b at K=32, while the same model at
# K=8 in the same ea1/es1 environment serves pred_std ~0.098 and the
# archived qwen/olmo K=32 runs parse fine. Generation is capped at
# MAX_NEW_TOKENS (default 6), so the leading hypothesis is that mistral
# prefixes its answer under a 32-exemplar prompt and the number never
# lands inside the budget. These 2 jobs re-run the same corner with
# DEBUG_GEN=1 (prints the per-round parse-failure fraction + raw decoded
# strings) and a widened budget. Greedy decoding is prefix-deterministic
# and the parser takes the first number, so a run that already parsed
# inside 6 tokens yields the IDENTICAL value at 24 -- widening cannot
# retroactively change any archived cell.
# NOTE: the executable is IDEMPOTENT -- it exits 0 the moment
# runs/<tag>/trajectory.pt exists. A diagnostic RERUN must therefore
# carry a FRESH tag or it silently no-ops (seed 992 did exactly that on
# 2026-08-15: the .out was rewritten empty, the trajectory kept its
# original timestamp). Bump this seed for every new diagnostic round.
#   s991 -> first smokes (no DEBUG_GEN)
#   s992 -> DEBUG_GEN + MAX_NEW_TOKENS=24, but the ICL path did not yet
#           populate the telemetry, so it printed stale __init__ values
#   s993 -> first run with working ICL-path DEBUG_GEN telemetry
CTXGRID_DBG_KEY = "sft_icl_ctxgrid_debug"
CTXGRID_DBG_SEED = 993
CTXGRID_DBG_MAX_NEW_TOKENS = 24


def ctxgrid_dbg_rows():
    return [ctxgrid_row("mistral7b", arm, 1.0, 1.0, CTXGRID_DBG_SEED,
                        nrounds=3, prefix="pofdctxgridsmkdbg")
            for arm in ("f32", "d32")]


def ctxgrid_dbg_sub():
    mdl = dict(REACH_MODELS["mistral7b"])
    mdl["extra_env"] = (mdl["extra_env"] + "DEBUG_GEN=1 DEBUG_GEN_N=12 "
                        f"MAX_NEW_TOKENS={CTXGRID_DBG_MAX_NEW_TOKENS} ")
    what = ("sft_icl_ctxgrid K=32 PARSE DIAGNOSTIC (3 rounds, seed 992; "
            "DEBUG_GEN=1 dumps the parse-failure fraction + raw decoded "
            f"strings, MAX_NEW_TOKENS={CTXGRID_DBG_MAX_NEW_TOKENS}). The "
            "seed-991 smokes served a constant 0.5 -- total silent parse "
            "failure -- for mistral at K=32")
    return REACH_SUB_TEMPLATE.format(model="mistral7b",
                                     key=CTXGRID_DBG_KEY, n_jobs=2,
                                     what=what, **mdl)


def ctxgrid_sub(model, kind):
    """kind: 'main' | 'smoke' -- rides the REACH sub template (the row
    schema is identical; es rides queue col 10)."""
    key = CTXGRID_KEY[model] if kind == "main" else CTXGRID_SMOKE_KEY
    n_jobs = (len(ctxgrid_rows(model)) if kind == "main"
              else len(ctxgrid_smoke_rows()))
    what = {"main": ("ONE-SEED CONTEXT-DEPTH x DUAL-GATE GRID -- "
                     "audited-missing cells (30 rounds; SFT / K=0 / "
                     "fixed+live K=8 / fixed+live K=32 x eps_AI x "
                     "eps_social, BOTH real numeric thresholds; "
                     "pofdctxgrid_ family, no shared tags). 181 jobs of "
                     "360 conceptual cells: 139 audited-reused, and the "
                     "40 mistral7b K=32 cells EXCLUDED (100% digit-free "
                     "generations -> no served signal). K=32 arms run "
                     "~1.9h vs ~0.9h at K=8 (wrapper drops "
                     "GEN_BATCH_SIZE to 8 for ICL_K>=16)"),
            "smoke": ("sft_icl_ctxgrid SMOKE (3 rounds, seed 991; the "
                      "never-run fixed/live K=32 corner at ea1 es1)")
            }[kind]
    return REACH_SUB_TEMPLATE.format(model=model, key=key, n_jobs=n_jobs,
                                     what=what, **REACH_MODELS[model])


# fig2_provider (2026-08-15): the SIX-JOB targeted replication that
# completes three-seed coverage for Figure 2. The figure asks whether
# RETAINING THE ENTERING MODEL SIGNAL (SFT-KL at beta=1, anchored to the
# base weights) preserves more provider-specific population separation
# than ordinary SFT (beta=0), so the cell set is providers x {b0, b1} x
# ea 0.4 x es {0, 0.2} x seeds {0, 42, 43} = 36 conceptual cells.
# AUDIT (audit_fig2_provider_reuse.py -> manifest_fig2_provider.json):
# 30 complete, 6 missing -- exactly the b1 cells at seeds 42/43 for
# olmo (both doses) and mistral (no-peer only). Everything else already
# has three-seed coverage.
# Cells carry the CANONICAL tag of their environment rather than a new
# family, so a later broad release idempotently no-ops them and the
# existing checker branches apply unchanged (REACH for _es0_,
# the cube _b-token branch for pofdws2f_):
#   es=0    pofdreach_<model>_b1_ea0p4_w0p5_l0p2_es0_s<seed>
#   es=0.2  pofdws2f_olmo7b_b1_ea0p4_w0p5_l0p2_es0p2_s<seed>_fresh_data
# The four es=0 tags are SHARED WITH the (still unreleased) 327-job
# sft_icl_reach production files by design -- NEVER co-submit this key
# with sft_icl_reach / sft_icl_reach_{olmo,mistral} (double-queue write
# race); releasing reach later no-ops these four.
# No smoke: every one of these six is a b1 cell whose seed-0 twin has
# already run and gated in this exact environment.
FIG2_MANIFEST_PATH = os.path.join(HERE, "manifest_fig2_provider.json")
FIG2_EXPECT_NEW = 6
FIG2_EXPECT_PER_MODEL = {"olmo7b": 4, "mistral7b": 2}
FIG2_KEY = {m: f"fig2_provider_{m[:-2]}" for m in ("olmo7b", "mistral7b")}


def fig2_tag(model, es, seed):
    if es == 0.0:
        return f"pofdreach_{model}_b1_ea0p4_{w_tok()}_es0_s{seed}"
    return (f"pofdws2f_{model}_b1_ea0p4_{w_tok()}_es0p2_s{seed}"
            f"_fresh_data")


def fig2_row(model, es, seed):
    a = REACH_ARM_COLS["b1"]
    return ROW_CTXGRID.format(
        tag=fig2_tag(model, es, seed), style=a["style"], beta=a["beta"],
        seed=seed, es=f"{es:g}", eps_ai="0.4", iclk=a["iclk"],
        snap=a["snap"], uselora=a["uselora"], fresh=a["fresh"],
        ansk=a["ansk"], gg=a["gg"], nrounds=30)


def _fig2_manifest():
    with open(FIG2_MANIFEST_PATH) as fh:
        return json.load(fh)


def fig2_rows(model):
    rows = []
    for c in _fig2_manifest()["cells"]:
        if c["model"] == model and c["status"] == "new":
            r = fig2_row(model, c["eps_social"], c["seed"])
            assert r.split(",")[0] == c["run_tag"], (r, c["run_tag"])
            rows.append(r)
    return rows


def fig2_sub(model):
    what = ("FIGURE-2 PROVIDER REPLICATION -- the audited-missing b1 "
            "(SFT-KL beta=1, entering-model signal retained) cells at "
            "ea 0.4, seeds 42/43, completing three-seed coverage for the "
            "provider-separation figure. 30 rounds. es=0 cells carry "
            "pofdreach_ tags SHARED with the unreleased 327-job reach "
            "production -- never co-submit with sft_icl_reach")
    return REACH_SUB_TEMPLATE.format(model=model, key=FIG2_KEY[model],
                                     n_jobs=len(fig2_rows(model)),
                                     what=what, **REACH_MODELS[model])


# mistral_innate_clamp_nopeer[_smoke] (2026-08-17): the NO-PEER
# INNATE-CLAMP wave -- ordinary fresh SFT (b0) vs frozen-weight live K=8
# ICL (dyn) when 20% of the population is PERMANENTLY pinned to its
# innate opinions (INNATE_CLAMP_MODE, run_pokec_gated_lm.py). Cohorts:
# _strat_ = stratified_random (proportional sampling within innate
# quintiles, seed-varying but identical across arms/gates within a
# seed) and _bottom_ = the 145 lowest innate opinions (id tie-break,
# seed-invariant). Frozen agents are still served, gated, labeled
# (their innate labels stay in the SFT batch) and exemplar-eligible;
# only their recorded opinion is pinned, bit-exactly, in BOTH the
# deployed population and the matched twin. es=0 BY CONSTRUCTION (the
# runner hard-fails the clamp under a live peer step -- no
# reset-after-peer approximation exists). Numeric threshold gates
# everywhere: _ea1_ is EPS_AI=1.0 under the strict-< gate, never
# all_open. Everything else clones the canonical mistral gate2d
# b0/dyn surface. NEW family pofdclamp_ -- no audit (the intervention
# is new, so no archived run can hold it); 2 modes x 2 arms x 5 gates
# x 3 seeds = 60 new jobs, zero tags shared with any other wave
# (collision-asserted). Smokes (4 x 3 rounds, seed 991, OUTSIDE the
# 60): both modes x both arms at ea0p2.
CLAMP_KEY = "mistral_innate_clamp_nopeer"
CLAMP_SMOKE_KEY = "mistral_innate_clamp_nopeer_smoke"
CLAMP_MODE_OF_TOK = {"strat": "stratified_random", "bottom": "bottom"}
CLAMP_MODE_TOKS = ["strat", "bottom"]
CLAMP_ARMS = ["b0", "dyn"]
CLAMP_GATES = [0.05, 0.1, 0.2, 0.4, 1.0]
CLAMP_SEEDS = [0, 42, 43]
CLAMP_FRAC = "0.2"
CLAMP_EXPECT_NEW = 60
CLAMP_SMOKE_SEED = 991
CLAMP_SMOKE_GATE = 0.2
ROW_CLAMP = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
             "ab, 0, 0.0, 0.5, loop, 0.0, {eps_ai}, threshold, "
             "{iclk}, {snap}, {uselora}, {fresh}, {ansk}, {gg}, "
             "{nrounds}, {cmode}")


def clamp_tag(arm, mode_tok, gate, seed, prefix="pofdclamp"):
    return (f"{prefix}_mistral7b_{arm}_{mode_tok}_ea{_num(gate)}"
            f"_{w_tok()}_es0_s{seed}")


def clamp_row(arm, mode_tok, gate, seed, nrounds=30, prefix="pofdclamp"):
    a = REACH_ARM_COLS[arm]
    return ROW_CLAMP.format(
        tag=clamp_tag(arm, mode_tok, gate, seed, prefix),
        style=a["style"], beta=a["beta"], seed=seed,
        eps_ai=f"{gate:g}", iclk=a["iclk"], snap=a["snap"],
        uselora=a["uselora"], fresh=a["fresh"], ansk=a["ansk"],
        gg=a["gg"], nrounds=nrounds,
        cmode=CLAMP_MODE_OF_TOK[mode_tok])


def clamp_rows():
    return [clamp_row(arm, tok, gate, seed)
            for tok in CLAMP_MODE_TOKS
            for arm in CLAMP_ARMS
            for gate in CLAMP_GATES
            for seed in CLAMP_SEEDS]


def clamp_smoke_rows():
    return [clamp_row(arm, tok, CLAMP_SMOKE_GATE, CLAMP_SMOKE_SEED,
                      nrounds=3, prefix="pofdclampsmk")
            for tok in CLAMP_MODE_TOKS
            for arm in CLAMP_ARMS]


def clamp_sub(kind):
    """kind: 'main' | 'smoke'."""
    key = CLAMP_KEY if kind == "main" else CLAMP_SMOKE_KEY
    n_jobs = (len(clamp_rows()) if kind == "main"
              else len(clamp_smoke_rows()))
    what = {"main": ("60 production cells (30 rounds; 2 clamp modes x "
                     "b0/dyn x 5 numeric gates x seeds 0/42/43)"),
            "smoke": ("SMOKE (4 x 3 rounds, seed 991; both clamp modes "
                      "x both arms at ea0p2)")}[kind]
    return CLAMP_SUB_TEMPLATE.format(key=key, n_jobs=n_jobs, what=what,
                                     **REACH_MODELS["mistral7b"])


CLAMP_SUB_TEMPLATE = """\
# HTCondor: NO-PEER INNATE-CLAMP, mistral7b -- {what}
# GENERATED by gen_pofd_sweep.py from the CLAMP block. Never edit by
# hand: rerun the script. {n_jobs} job(s).
# 20% of the population (145 of 723 agents) permanently pinned to its
# innate opinions -- INNATE_CLAMP_MODE rides the queue (col 24:
# stratified_random | bottom), INNATE_CLAMP_SEED == the run seed (the
# cohort is identical across arms and gates within a seed),
# INNATE_CLAMP_FRAC fixed at 0.2. Frozen agents are still served,
# gated, labeled and exemplar-eligible; their recorded opinions are
# bit-exact innate in BOTH the deployed population and the matched
# twin. es=0 by construction (the runner hard-fails the clamp under a
# live peer step). Numeric threshold gates everywhere -- _ea1_ is
# EPS_AI=1.0 under the strict-< gate, never all_open. W=0.5, lam=0.2,
# gamma=0, greedy serving, WITH_TWIN=1, movielens Action, 30-round
# mains / 3-round smokes via queue col 23.
# Gate every pull with check_pofd_sanity (CLAMP section: mask
# reconstruction + hash, exactly 145 frozen, frozen rows bit-exact on
# innate in op_raw AND twin_raw, responsive exact platform blend,
# frozen-eligible ICL exemplars, all-723-label SFT batches).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
#   (flow: mistral_innate_clamp_nopeer_smoke -> pull + gate ->
#    mistral_innate_clamp_nopeer)
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 INNATE_CLAMP_MODE=$(cmode) INNATE_CLAMP_FRAC=0.2 INNATE_CLAMP_SEED=$(seed) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_mistral7b_pofdclamp"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, cmode from experiments/condor/configs_pofd_{key}.txt
"""


# mistral_innate_clamp_graph_s0[_smoke] (2026-08-17): the GRAPH-
# PLACEMENT innate-clamp wave -- two DETERMINISTICALLY PRE-BUILT 145-
# agent fixed cohorts (build_clamp_graph_masks.py ->
# clamp_graph_masks.json, committed BEFORE any GPU job) with the peer
# step LIVE under the one-sided STUBBORN operator:
#   _gclump_ graph_clumped   -- a concentrated low-cut region (cut 709,
#            conductance 0.33, ONE 145-agent induced component, 41.5%
#            responsive exposure)
#   _gscat_  graph_scattered -- distributed across the graph (cut 2059
#            = 2.9x clumped, 61 internal edges, 100% responsive
#            exposure)
# Both masks draw IDENTICAL quotas from joint innate-quintile x
# degree-tercile strata (d_mean 0.0003, d_sd ~0, W1 0.0068 -- all
# acceptance criteria hard-asserted from the artifact below). Fixed
# agents participate fully in pairing; an accepted F-R pair moves ONLY
# the responsive endpoint; fixed agents never move even transiently;
# the matched twin runs the IDENTICAL mask + operator on its mirrored
# generator. NO isolated condition exists.
# Grid: 2 masks x 2 arms x ea {0.1,0.2,0.4,1.0} x es
# {0,0.05,0.1,0.2,0.4,1.0} x seed 0 = 96 jobs -- es=0 baselines are
# IN-WAVE (the old no-peer runs used different cohorts and are NOT
# exact baselines for these masks). Same pofdclamp_ family (the
# _gclump_/_gscat_ tokens disambiguate; zero collisions, asserted).
# Smokes (4 x 3 rounds, seed 991, OUTSIDE the 96): both masks x both
# arms at ea0p4 es0p4.
CLAMP_GRAPH_KEY = "mistral_innate_clamp_graph_s0"
CLAMP_GRAPH_SMOKE_KEY = "mistral_innate_clamp_graph_smoke"
CLAMP_GRAPH_ARTIFACT = os.path.join(HERE, "clamp_graph_masks.json")
CLAMP_GRAPH_OF_TOK = {"gclump": "graph_clumped",
                      "gscat": "graph_scattered"}
CLAMP_GRAPH_TOKS = ["gclump", "gscat"]
CLAMP_GRAPH_GATES = [0.1, 0.2, 0.4, 1.0]
CLAMP_GRAPH_ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
CLAMP_GRAPH_EXPECT_NEW = 96
CLAMP_GRAPH_SMOKE_SEED = 991
CLAMP_GRAPH_SMOKE_GATE = 0.4
CLAMP_GRAPH_SMOKE_ES = 0.4
ROW_CLAMP_GRAPH = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, "
                   "fixed, ab, {es}, 0.0, 0.5, loop, 0.0, {eps_ai}, "
                   "threshold, {iclk}, {snap}, {uselora}, {fresh}, "
                   "{ansk}, {gg}, {nrounds}, {cmode}")


def clamp_graph_tag(arm, gtok, gate, es, seed, prefix="pofdclamp"):
    return (f"{prefix}_mistral7b_{arm}_{gtok}_stub"
            f"_ea{_num(gate)}_{w_tok()}_es{_num(es)}_s{seed}")


def clamp_graph_row(arm, gtok, gate, es, seed, nrounds=30,
                    prefix="pofdclamp"):
    a = REACH_ARM_COLS[arm]
    return ROW_CLAMP_GRAPH.format(
        tag=clamp_graph_tag(arm, gtok, gate, es, seed, prefix),
        style=a["style"], beta=a["beta"], seed=seed, es=f"{es:g}",
        eps_ai=f"{gate:g}", iclk=a["iclk"], snap=a["snap"],
        uselora=a["uselora"], fresh=a["fresh"], ansk=a["ansk"],
        gg=a["gg"], nrounds=nrounds,
        cmode=CLAMP_GRAPH_OF_TOK[gtok])


def clamp_graph_rows():
    return [clamp_graph_row(arm, gtok, gate, es, 0)
            for gtok in CLAMP_GRAPH_TOKS
            for arm in CLAMP_ARMS
            for gate in CLAMP_GRAPH_GATES
            for es in CLAMP_GRAPH_ESS]


def clamp_graph_smoke_rows():
    return [clamp_graph_row(arm, gtok, CLAMP_GRAPH_SMOKE_GATE,
                            CLAMP_GRAPH_SMOKE_ES,
                            CLAMP_GRAPH_SMOKE_SEED,
                            nrounds=3, prefix="pofdclampsmk")
            for gtok in CLAMP_GRAPH_TOKS
            for arm in CLAMP_ARMS]


def clamp_graph_sub(kind):
    """kind: 'main' | 'smoke'."""
    key = {"main": CLAMP_GRAPH_KEY,
           "smoke": CLAMP_GRAPH_SMOKE_KEY}[kind]
    n_jobs = {"main": len(clamp_graph_rows()),
              "smoke": len(clamp_graph_smoke_rows())}[kind]
    what = {"main": ("96 seed-0 production cells (30 rounds; "
                     "gclump/gscat x b0/dyn x 4 AI gates x 6 social "
                     "gates incl. in-wave es=0 baselines, one-sided "
                     "stubborn peer operator)"),
            "smoke": ("SMOKE (4 x 3 rounds, seed 991; both masks x "
                      "both arms at ea0p4 es0p4)")}[kind]
    return CLAMP_PEER_SUB_TEMPLATE.format(key=key, n_jobs=n_jobs,
                                          what=what,
                                          **REACH_MODELS["mistral7b"])


CLAMP_PEER_SUB_TEMPLATE = """\
# HTCondor: INNATE-CLAMP GRAPH-PLACEMENT WAVE, mistral7b -- {what}
# GENERATED by gen_pofd_sweep.py from the CLAMP_GRAPH block. Never
# edit by hand: rerun the script. {n_jobs} job(s).
# Two PRE-BUILT 145-agent fixed cohorts (clamp_graph_masks.json,
# deterministic local search over the kNN graph, committed before any
# job; INNATE_CLAMP_MODE rides queue col 24): graph_clumped = one
# connected low-cut block (cut 709, 41.5% exposure) vs
# graph_scattered = distributed (cut 2059, 100% exposure), matched on
# innate x degree strata with identical quotas. Peer step under the
# one-sided STUBBORN operator (INNATE_CLAMP_PEER_MODE=stubborn, fixed
# in this env): fixed agents participate fully in pairing and
# influence responsive neighbors; an accepted F-R pair moves ONLY the
# responsive endpoint to the midpoint; fixed agents never move even
# transiently. NO isolated condition exists. Fixed agents are pinned
# BEFORE the sweep (peers see innate, never a transient blend) and
# re-pinned after; the matched twin runs the IDENTICAL mask +
# operator on its mirrored generator. Telemetry per round: cut
# (F-R pair) sampling and acceptance, cumulative responsive reach
# through the cut, responsive dispersion vs twin, responsive-to-fixed
# mean/W1 gaps + closure, responsive displacement
# (+ clamp_fr_touch_raw).
# es=0 baselines are IN-WAVE (the old no-peer cohorts differ -- NOT
# exact baselines for these masks). W=0.5, lam=0.2, gamma=0, greedy
# serving, WITH_TWIN=1, movielens Action, 30-round mains / 3-round
# smokes.
# Gate every pull with check_pofd_sanity (CLAMP section, graph-mask
# invariants: artifact ids + hash + recomputed graph stats, fixed
# bit-exact in pop AND twin, F-pair participation mandatory,
# touch/reach consistency, responsive-mean conservation replay on
# one-sided-move-free rounds).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
#   (flow: mistral_innate_clamp_graph_smoke -> pull + gate ->
#    mistral_innate_clamp_graph_s0)
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 INNATE_CLAMP_MODE=$(cmode) INNATE_CLAMP_FRAC=0.2 INNATE_CLAMP_SEED=$(seed) INNATE_CLAMP_PEER_MODE=stubborn TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_mistral7b_pofdclamppeer"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, cmode from experiments/condor/configs_pofd_{key}.txt
"""


# mistral_innate_clamp_graph_d8[_smoke] (2026-08-17): the CORRECTED
# graph-clamp comparison. The old cross-user live K=8 arm (_dyn_) is
# REPLACED by personal-history ICL (_d8_): frozen weights, ICL_K=0,
# ICL_DAYS=8 -- each agent's prompt carries only its OWN latest eight
# recorded opinions, oldest to newest (early rounds use the available
# history beginning with innate); NO other agent's profile or opinion
# ever appears, so fixed agents see only repetitions of their own
# innate opinion. Same seed-0 grid as the main graph wave (masks
# gclump/gscat, ea {0.1,0.2,0.4,1}, es {0,0.05,0.1,0.2,0.4,1}) = 96
# CONCEPTUAL cells: every complete _b0_ SFT cell from
# mistral_innate_clamp_graph_s0 is REUSED unchanged; this key holds
# the 48 new _d8_ cells PLUS the 3 _b0_ cells still missing at the
# 2026-08-17 cluster audit (i104 casualties, never completed) = 51
# jobs. The old _dyn_ runs are EXCLUDED from the corrected analysis.
# The 3 backfill tags are SHARED with the main key BY DESIGN -- NEVER
# co-submit with mistral_innate_clamp_graph_s0 (double-queue write
# race); the idempotent exec no-ops anything already complete.
# Smokes (2 x 3 rounds, seed 991, OUTSIDE the 96): d8 x both masks at
# ea0p4 es0p4. Runners write icl_days_log.json.gz (the exact rendered
# personal-history context per agent per round); the checker replays
# it byte-for-byte against (innate, op_raw).
CLAMP_GRAPH_D8_KEY = "mistral_innate_clamp_graph_d8"
CLAMP_GRAPH_D8_SMOKE_KEY = "mistral_innate_clamp_graph_d8_smoke"
# the b0 cells with no trajectory on the cluster (audited 2026-08-17
# after the i104 kills; the retry key that briefly carried them is
# retired -- its 2 dyn cells are no longer part of the design)
CLAMP_GRAPH_B0_BACKFILL_CELLS = [
    ("b0", "gscat", 0.1, 0.05),
    ("b0", "gscat", 0.2, 1.0),
    ("b0", "gscat", 1.0, 0.0),
]
ROW_CLAMP_GRAPH_D8 = ROW_CLAMP_GRAPH + ", {icldays}"


def clamp_graph_d8_row(arm, gtok, gate, es, seed, nrounds=30,
                       prefix="pofdclamp"):
    a = REACH_ARM_COLS[arm]
    return ROW_CLAMP_GRAPH_D8.format(
        tag=clamp_graph_tag(arm, gtok, gate, es, seed, prefix),
        style=a["style"], beta=a["beta"], seed=seed, es=f"{es:g}",
        eps_ai=f"{gate:g}", iclk=a["iclk"], snap=a["snap"],
        uselora=a["uselora"], fresh=a["fresh"], ansk=a["ansk"],
        gg=a["gg"], nrounds=nrounds,
        cmode=CLAMP_GRAPH_OF_TOK[gtok],
        icldays=8 if arm == "d8" else 0)


def clamp_graph_d8_rows():
    rows = [clamp_graph_d8_row("d8", gtok, gate, es, 0)
            for gtok in CLAMP_GRAPH_TOKS
            for gate in CLAMP_GRAPH_GATES
            for es in CLAMP_GRAPH_ESS]
    rows += [clamp_graph_d8_row(arm, gtok, gate, es, 0)
             for arm, gtok, gate, es in CLAMP_GRAPH_B0_BACKFILL_CELLS]
    return rows


def clamp_graph_d8_smoke_rows():
    return [clamp_graph_d8_row("d8", gtok, CLAMP_GRAPH_SMOKE_GATE,
                               CLAMP_GRAPH_SMOKE_ES,
                               CLAMP_GRAPH_SMOKE_SEED,
                               nrounds=3, prefix="pofdclampsmk")
            for gtok in CLAMP_GRAPH_TOKS]


def clamp_graph_d8_sub(kind):
    """kind: 'main' | 'smoke'."""
    key = {"main": CLAMP_GRAPH_D8_KEY,
           "smoke": CLAMP_GRAPH_D8_SMOKE_KEY}[kind]
    n_jobs = {"main": len(clamp_graph_d8_rows()),
              "smoke": len(clamp_graph_d8_smoke_rows())}[kind]
    what = {"main": ("51 seed-0 jobs (30 rounds): the 48 NEW _d8_ "
                     "personal-history cells (gclump/gscat x 4 AI "
                     "gates x 6 social gates) + the 3 _b0_ SFT cells "
                     "missing at the 2026-08-17 audit"),
            "smoke": ("SMOKE (2 x 3 rounds, seed 991; d8 x both "
                      "masks at ea0p4 es0p4)")}[kind]
    return CLAMP_PEER_D8_SUB_TEMPLATE.format(key=key, n_jobs=n_jobs,
                                             what=what,
                                             **REACH_MODELS["mistral7b"])


CLAMP_PEER_D8_SUB_TEMPLATE = """\
# HTCondor: INNATE-CLAMP GRAPH WAVE, PERSONAL-HISTORY ARM, mistral7b -- {what}
# GENERATED by gen_pofd_sweep.py from the CLAMP_GRAPH_D8 block. Never
# edit by hand: rerun the script. {n_jobs} job(s).
# The CORRECTED graph-clamp comparison: ordinary SFT (_b0_) vs
# PERSONAL-HISTORY ICL (_d8_: frozen weights, ICL_K=0, ICL_DAYS=8 --
# each agent's prompt carries ONLY its own latest eight recorded
# opinions, oldest to newest, innate-first on early rounds; no other
# agent's profile or opinion ever appears, so fixed agents see only
# repetitions of their own innate opinion; every run writes
# icl_days_log.json.gz for the byte-exact offline replay). The old
# cross-user _dyn_ arm is EXCLUDED from this analysis. Masks, peer
# operator and telemetry are IDENTICAL to the main graph wave: two
# PRE-BUILT 145-agent cohorts (clamp_graph_masks.json, queue col 24)
# under the one-sided STUBBORN operator (fixed agents keep pairing;
# an accepted F-R pair moves ONLY the responsive endpoint; NO
# isolated condition), fixed pinned before AND after the sweep, the
# matched twin on its mirrored generator. ICL_DAYS rides queue col 25
# (8 on _d8_ rows, 0 on the _b0_ backfill rows -- those 3 tags are
# SHARED with mistral_innate_clamp_graph_s0 BY DESIGN; never
# co-submit the two keys). W=0.5, lam=0.2, gamma=0, greedy serving,
# WITH_TWIN=1, movielens Action, 30-round mains / 3-round smokes.
# Gate every pull with check_pofd_sanity (CLAMP section: graph-mask
# invariants + the d8 personal-history replay -- exact sequence per
# agent per round, ICL_K=0, no cross-user exemplar artifacts, frozen
# weights).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
#   (flow: mistral_innate_clamp_graph_d8_smoke -> pull + gate ->
#    mistral_innate_clamp_graph_d8)
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=$(icldays) ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 INNATE_CLAMP_MODE=$(cmode) INNATE_CLAMP_FRAC=0.2 INNATE_CLAMP_SEED=$(seed) INNATE_CLAMP_PEER_MODE=stubborn TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_mistral7b_pofdclampd8"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, cmode, icldays from experiments/condor/configs_pofd_{key}.txt
"""


# mistral_clamp_exclude_a[_smoke] (2026-08-18): the CAUSAL SOURCE-
# EXCLUSION arm of the graph-clamp wave. The completed _b0_ SFT runs
# train on ALL 723 labels every round, so the 145 fixed agents' innate
# labels enter every shared weight update. The _b0xa_ arm ("xa" =
# cohort A eXcluded from Adaptation) keeps cohort A FULLY present in
# the environment -- served, gated, pinned bit-exact, stubborn peer
# pairing, mirrored in the matched twin -- and changes EXACTLY ONE
# thing: SFT_EXCLUDE_CLAMPED=1 drops A's rows from every SFT batch
# (round 0 included) and keeps the training VOLUME matched at 723
# rows by duplicating 145 responsive agents picked once per run by a
# dedicated run-seeded generator (seed + 723_145; reused every round,
# no simulation RNG advanced), so each round trains on 578 responsive
# agents once + those 145 twice, all with current live opinions and
# never a fixed id. Comparing _b0_
# vs _b0xa_ at a matched cell therefore measures whether information
# from A reaches the responsive cohort THROUGH SHARED MODEL WEIGHTS;
# the personal-history _d8_ arm (each agent sees only its own history)
# is the structural null and is REUSED, never rerun. Same seed-0 grid
# as the completed wave (masks gclump/gscat x ea {0.1,0.2,0.4,1} x es
# {0,0.05,0.1,0.2,0.4,1}) = 48 NEW jobs; all 48 _b0_ and all 48 _d8_
# cells are reused unchanged. Runners persist sft_idx_raw/sft_y_raw/
# sft_dup_idx (the exact ordered training ids + labels per round and
# the run-fixed duplicate selection); the checker proves exclusion +
# volume match against the reconstructed mask complement and the
# replayed seeded selection.
# Smokes (2 x 3 rounds, seed 991, OUTSIDE the 48): b0xa x both masks
# at ea0p4 es0p2.
CLAMP_XA_KEY = "mistral_clamp_exclude_a"
CLAMP_XA_SMOKE_KEY = "mistral_clamp_exclude_a_smoke"
CLAMP_XA_SMOKE_GATE = 0.4
CLAMP_XA_SMOKE_ES = 0.2


def clamp_xa_rows():
    return [clamp_graph_row("b0xa", gtok, gate, es, 0)
            for gtok in CLAMP_GRAPH_TOKS
            for gate in CLAMP_GRAPH_GATES
            for es in CLAMP_GRAPH_ESS]


def clamp_xa_smoke_rows():
    return [clamp_graph_row("b0xa", gtok, CLAMP_XA_SMOKE_GATE,
                            CLAMP_XA_SMOKE_ES, CLAMP_GRAPH_SMOKE_SEED,
                            nrounds=3, prefix="pofdclampsmk")
            for gtok in CLAMP_GRAPH_TOKS]


def clamp_xa_sub(kind):
    """kind: 'main' | 'smoke'."""
    key = {"main": CLAMP_XA_KEY, "smoke": CLAMP_XA_SMOKE_KEY}[kind]
    n_jobs = {"main": len(clamp_xa_rows()),
              "smoke": len(clamp_xa_smoke_rows())}[kind]
    what = {"main": ("48 seed-0 production cells (30 rounds; "
                     "gclump/gscat x b0xa x 4 AI gates x 6 social "
                     "gates incl. in-wave es=0 baselines)"),
            "smoke": ("SMOKE (2 x 3 rounds, seed 991; b0xa x both "
                      "masks at ea0p4 es0p2)")}[kind]
    return CLAMP_XA_SUB_TEMPLATE.format(key=key, n_jobs=n_jobs,
                                        what=what,
                                        **REACH_MODELS["mistral7b"])


CLAMP_XA_SUB_TEMPLATE = """\
# HTCondor: INNATE-CLAMP GRAPH WAVE, SOURCE-EXCLUSION ARM, mistral7b -- {what}
# GENERATED by gen_pofd_sweep.py from the CLAMP_XA block. Never edit
# by hand: rerun the script. {n_jobs} job(s).
# The CAUSAL source-exclusion comparison: ordinary SFT with cohort A's
# labels INCLUDED (_b0_, completed + reused) vs the IDENTICAL runs
# with SFT_EXCLUDE_CLAMPED=1 (_b0xa_): the 145 fixed agents stay
# fully present -- served, gated, pinned bit-exact in population AND
# twin, stubborn peer pairing unchanged -- but their rows are dropped
# from every SFT batch (round 0 included) while the training VOLUME
# stays matched at 723 rows: 145 responsive agents, picked ONCE per
# run by a dedicated run-seeded generator (seed + 723_145, no
# simulation RNG advanced) and reused every round, are each included
# once more, so every round trains on 578 responsive agents once +
# those 145 twice, all with current live opinions and never a fixed
# id. The b0-vs-b0xa difference at a matched cell is the
# weight-mediated pathway from A to the responsive cohort; the
# reused _d8_ personal-history arm is the structural null. Masks,
# peer operator, serving, generation and telemetry are IDENTICAL to
# the completed graph wave (clamp_graph_masks.json rides queue col
# 24, stubborn operator fixed in this env). Every run persists
# sft_idx_raw/sft_y_raw/sft_dup_idx -- the exact ordered training
# ids + labels per round and the run-fixed duplicate selection --
# for the offline exclusion + volume-match proof. W=0.5, lam=0.2,
# gamma=0, greedy serving, WITH_TWIN=1, movielens Action, 30-round
# mains / 3-round smokes.
# Gate every pull with check_pofd_sanity (CLAMP section: graph-mask
# invariants + the b0xa provenance replay -- 723 rows every round ==
# the ascending responsive complement then the replayed seeded
# duplicates, labels == the live opinions, no fixed id ever trains).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
#   (flow: mistral_clamp_exclude_a_smoke -> pull + gate ->
#    mistral_clamp_exclude_a)
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 INNATE_CLAMP_MODE=$(cmode) INNATE_CLAMP_FRAC=0.2 INNATE_CLAMP_SEED=$(seed) INNATE_CLAMP_PEER_MODE=stubborn SFT_EXCLUDE_CLAMPED=1 TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_mistral7b_pofdclampxa"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, cmode from experiments/condor/configs_pofd_{key}.txt
"""


# mistral_bottom20_source_impact (2026-08-18, FULL-GRID revision --
# supersedes the never-submitted 24-job no-peer-only version): the
# BOTTOM-20% SOURCE-IMPACT wave. Cohort A = the 145 agents with the
# LOWEST innate Action opinions (INNATE_CLAMP_MODE=bottom:
# deterministic ranking by innate then agent id -- NOT
# representative, stratified, or graph-optimized), pinned bit-exact
# at innate for all 30 rounds in population AND twin; cohort B = the
# other 578 evolves normally. SEED 0 ONLY. Three arms x ea
# {0.1,0.2,0.4,1} (numeric threshold) x es {0,0.05,0.1,0.2,0.4,1}
# = 72 conceptual cells:
#   b0   ordinary fresh SFT on all 723 labels (A's unchanged innate
#        labels included every round) -- the 4 completed seed-0
#        no-peer bottom cells (es=0, tokenless) REUSE (field-level
#        audit, audit_bottom20_reuse.py ->
#        manifest_bottom20_source_impact.json; hard-fails on any
#        other split); the 20 es>0 cells are NEW
#   b0xa identical SFT with A excluded from every batch, volume-
#        matched at 723 rows via the existing run-seeded 145-
#        duplicate procedure (seed + 723_145) -- 24 NEW
#   d8   frozen weights, personal-history ICL (ICL_K=0, ICL_DAYS=8:
#        only the recipient's own opinions, innate-first) -- 24 NEW
# = 68 NEW jobs. Every new row carries the _stub_ token and runs the
# one-sided STUBBORN operator (fixed agents keep pairing; A-A pairs
# do nothing, in A-B pairs only B moves, B-B updates normally; fixed
# never move even transiently; the IDENTICAL operator runs on the
# matched twin) -- inert at the es=0 in-wave baselines, the graph-
# wave precedent. The old global live-K=8 dyn arm and both graph
# cohorts are NEVER reused (matched audit fields). NO smoke: the
# bottom clamp, stubborn peers, volume-matched exclusion, and
# personal-history paths all carry production gates already.
# ICL_DAYS and SFT_EXCLUDE_CLAMPED ride the queue (cols 25/26: d8
# rows 8/0, b0/b0xa rows 0/{0,1}).
B20_KEY = "mistral_bottom20_source_impact"
B20_MANIFEST_PATH = os.path.join(
    HERE, "manifest_bottom20_source_impact.json")
B20_ARMS = ["b0", "b0xa", "d8"]
B20_GATES = [0.1, 0.2, 0.4, 1.0]
B20_ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
B20_SEEDS = [0]
ROW_B20 = ROW_CLAMP_GRAPH + ", {icldays}, {sftexcl}"


def b20_tag(arm, gate, es, seed=0, prefix="pofdclamp"):
    # only the four reused b0 no-peer cells are tokenless; every NEW
    # cell declares the stubborn operator (inert at es=0)
    stub = "" if (arm == "b0" and es == 0.0) else "_stub"
    return (f"{prefix}_mistral7b_{arm}_bottom{stub}_ea{_num(gate)}"
            f"_{w_tok()}_es{_num(es)}_s{seed}")


def b20_row(arm, gate, es, seed=0, nrounds=30, prefix="pofdclamp"):
    a = REACH_ARM_COLS[arm]
    return ROW_B20.format(
        tag=b20_tag(arm, gate, es, seed, prefix),
        style=a["style"], beta=a["beta"], seed=seed, es=f"{es:g}",
        eps_ai=f"{gate:g}", iclk=a["iclk"], snap=a["snap"],
        uselora=a["uselora"], fresh=a["fresh"], ansk=a["ansk"],
        gg=a["gg"], nrounds=nrounds, cmode="bottom",
        icldays=8 if arm == "d8" else 0,
        sftexcl=1 if arm == "b0xa" else 0)


def b20_rows():
    """The 68 genuinely-missing jobs, straight from the audited
    manifest -- counts are asserted for CONSISTENCY with the expected
    4-reused/68-new split, never forced."""
    mf = json.load(open(B20_MANIFEST_PATH))
    cells = mf["cells"]
    assert mf["n_cells"] == 72 and len(cells) == 72, mf["n_cells"]
    assert {(c["arm"], c["gate"], c["es"]) for c in cells} == \
        {(a, g, e) for a in B20_ARMS for g in B20_GATES
         for e in B20_ESS}
    assert all(c["seed"] == 0 for c in cells), "seed 0 only"
    reused = [c for c in cells if c["status"] == "reused"]
    new = [c for c in cells if c["status"] == "new"]
    assert len(reused) == 4 and len(new) == 68, \
        (len(reused), len(new))
    assert all(c["arm"] == "b0" and c["es"] == 0.0 for c in reused), \
        "only the completed seed-0 b0 bottom no-peer cells may reuse"
    assert all(c.get("verdict") == "PASS" for c in reused)
    rows = []
    for c in sorted(new,
                    key=lambda c: (c["arm"], c["gate"], c["es"])):
        r = b20_row(c["arm"], c["gate"], c["es"])
        assert r.split(",")[0].strip() == c["new_tag"], \
            (r.split(",")[0], c["new_tag"])
        rows.append(r)
    return rows


def b20_sub():
    return B20_SUB_TEMPLATE.format(key=B20_KEY,
                                   n_jobs=len(b20_rows()),
                                   **REACH_MODELS["mistral7b"])


B20_SUB_TEMPLATE = """\
# HTCondor: BOTTOM-20% SOURCE-IMPACT WAVE (full ea x es grid),
# mistral7b -- 68 NEW seed-0 jobs (b0 es>0 20 + b0xa 24 + d8 24; the
# 4 completed b0 bottom no-peer cells REUSE per the audited
# manifest). GENERATED by gen_pofd_sweep.py from the B20 block.
# Never edit by hand: rerun the script. {n_jobs} job(s).
# Cohort A = the 145 LOWEST-innate agents (INNATE_CLAMP_MODE=bottom,
# deterministic innate-then-id ranking), pinned bit-exact at innate
# in population AND twin; B = the other 578. Full grid ea
# 0.1/0.2/0.4/1 x es 0/0.05/0.1/0.2/0.4/1: every row runs the
# one-sided STUBBORN operator (A-A pairs do nothing, in A-B pairs
# only B moves, B-B updates normally, fixed never move even
# transiently, IDENTICAL operator on the matched twin; inert at the
# es=0 in-wave baselines). b0 trains on all 723 labels; b0xa drops
# A's rows from every batch and volume-matches at 723 rows (578
# responsive once + 145 run-seeded duplicates,
# sft_idx_raw/sft_y_raw/sft_dup_idx persisted); d8 is the structural
# null (frozen weights, each prompt carries only the recipient's own
# last 8 opinions, icl_days_log persisted).
# ICL_DAYS rides queue col 25 (8 on d8, else 0); the exclusion flag
# rides col 26 (1 on b0xa, else 0). W=0.5, lam=0.2, gamma=0, greedy
# serving, WITH_TWIN=1, movielens Action, 30 rounds, SEED 0 ONLY.
# NO smoke (all paths carry production gates).
# Gate every pull with check_pofd_sanity (CLAMP section: bottom-mask
# reconstruction, fixed bit-exact in pop AND twin, stubborn peer
# invariants, b0xa provenance replay, d8 personal-history replay).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=$(icldays) ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 INNATE_CLAMP_MODE=$(cmode) INNATE_CLAMP_FRAC=0.2 INNATE_CLAMP_SEED=$(seed) INNATE_CLAMP_PEER_MODE=stubborn SFT_EXCLUDE_CLAMPED=$(sftexcl) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_mistral7b_pofdclampb20"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, cmode, icldays, sftexcl from experiments/condor/configs_pofd_{key}.txt
"""


# mistral_bottom20_evolving (2026-08-18): the FULLY-EVOLVING
# comparison grid for the completed bottom-20%-fixed wave. NO innate
# clamp, NO fixed agents, NO exclusion: all 723 agents evolve
# normally under standard SYMMETRIC peer dynamics with the matched
# no-platform twin. NEW pofdevo_ family (never reuses reach / peer /
# gate2d / global-K8 trajectories -- the point is a same-code-path
# companion to the fixed wave). Two arms x ea 0.1/0.2/0.4/1 (numeric
# threshold) x es 0/0.05/0.1/0.2/0.4/1 x SEED 0 = 48 jobs:
#   b0  ordinary fresh SFT (beta 0) on all 723 current opinions
#       every round
#   d8  frozen weights, personal-history ICL (ICL_K=0, ICL_DAYS=8:
#       each agent sees ONLY its own eight most recent opinions;
#       icl_days_log persisted for the byte replay)
# Serving is MATCHED to the completed fixed wave (the current path,
# per the 2026-08-18 decision -- no eval-mode change), so the
# fixed-vs-evolving contrast isolates the clamp. Analysis defines
# cohort A (the 145 lowest-innate agents) ANALYTICALLY only.
# NO smoke: both paths carry production gates.
EVO_KEY = "mistral_bottom20_evolving"
EVO_ARMS = ["b0", "d8"]
EVO_GATES = [0.1, 0.2, 0.4, 1.0]
EVO_ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
ROW_EVO = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
           "ab, {es}, 0.0, 0.5, loop, 0.0, {eps_ai}, threshold, "
           "{iclk}, {snap}, {uselora}, {fresh}, {ansk}, {gg}, "
           "{nrounds}, {icldays}")


def evo_tag(arm, gate, es, seed=0, prefix="pofdevo"):
    return (f"{prefix}_mistral7b_{arm}_ea{_num(gate)}"
            f"_{w_tok()}_es{_num(es)}_s{seed}")


def evo_row(arm, gate, es, seed=0, nrounds=30):
    a = REACH_ARM_COLS[arm]
    return ROW_EVO.format(
        tag=evo_tag(arm, gate, es, seed),
        style=a["style"], beta=a["beta"], seed=seed, es=f"{es:g}",
        eps_ai=f"{gate:g}", iclk=a["iclk"], snap=a["snap"],
        uselora=a["uselora"], fresh=a["fresh"], ansk=a["ansk"],
        gg=a["gg"], nrounds=nrounds,
        icldays=8 if arm == "d8" else 0)


def evo_rows():
    return [evo_row(arm, gate, es)
            for arm in EVO_ARMS
            for gate in EVO_GATES
            for es in EVO_ESS]


def evo_sub():
    return EVO_SUB_TEMPLATE.format(key=EVO_KEY, n_jobs=len(evo_rows()),
                                   **REACH_MODELS["mistral7b"])


EVO_SUB_TEMPLATE = """\
# HTCondor: FULLY-EVOLVING COMPARISON WAVE, mistral7b -- 48 seed-0
# jobs (b0 ordinary SFT + d8 personal-history ICL x 4 AI gates x 6
# social gates). GENERATED by gen_pofd_sweep.py from the EVO block.
# Never edit by hand: rerun the script. {n_jobs} job(s).
# The no-clamp companion to the completed bottom-20%-fixed wave: ALL
# 723 agents evolve normally (no fixed cohort, no exclusion),
# standard symmetric peer dynamics, matched no-platform twin
# (WITH_TWIN=1). b0 trains fresh each round on all 723 current
# opinions; d8 serves frozen weights with each agent's OWN last 8
# opinions as context (ICL_K=0, ICL_DAYS=8, icl_days_log.json.gz for
# the byte-exact replay -- no other agent's opinion may enter a
# prompt). Serving path MATCHED to the completed fixed runs. Cohort
# A (145 lowest-innate) exists only in the ANALYSIS mask -- never in
# the simulation. W=0.5, lam=0.2, gamma=0, greedy serving, movielens
# Action, 30 rounds, SEED 0 ONLY. NO smoke (both paths carry
# production gates). ICL_DAYS rides queue col 24 (8 on d8, 0 on b0).
# Gate every pull with check_pofd_sanity (EVO section: no clamp
# artifacts, mandatory twin, d8 personal-history byte replay,
# all-723-label FRESH on b0).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=$(icldays) ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_mistral7b_pofdevo"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, icldays from experiments/condor/configs_pofd_{key}.txt
"""


# mistral_bottom20_section4_repl (2026-08-19): the SECTION-4 THREE-
# SEED REPLICATION -- the completed seed-0 fixed-vs-evolving surface
# (b20 fixed b0/d8 + pofdevo) extended to seeds 42 and 43. Target:
# seeds {42, 43} x conditions {fixed, evolving} x arms {b0, d8} x ea
# {0.1, 0.2, 0.4, 1} x es {0, 0.05, 0.1, 0.2, 0.4, 1} = 192
# conceptual cells (96 per seed with seed 0, 288 across all three).
# Reuse is by EXACT FIELD-LEVEL AUDIT over the cluster archive
# (audit_bottom20_section4_repl.py ->
# manifest_bottom20_section4_repl.json), never tag similarity, and
# the split was NOT forced: the audit found 40 reused / 152 new --
#   8  fixed b0 es0: the tokenless mistral_innate_clamp_nopeer
#      originals at s42/s43 (all four gates, both seeds)
#   32 evolving b0 at es {0, 0.2, 0.4, 1}: archived pofdreach /
#      pofdpeer2 / pofdgate2d / pofdws2f cells matching the complete
#      no-clamp surface (config fields + 30 rounds + twin)
# The 152 new jobs split into two schemas under ONE umbrella key:
#   _fixed  88 rows (b0 es>0 20 + d8 24 per seed) on the b20 queue
#           (26 cols, bottom clamp + stubborn peers ride the env)
#   _evo    64 rows (b0 es {0.05, 0.1} 8 + d8 24 per seed) on the
#           evo queue (24 cols, NO clamp env anywhere)
# Every NEW fixed cell declares the stubborn operator (the b20
# full-grid precedent; inert at es=0 -- and no new fixed b0 es0 cell
# exists, so the reused tokenless tags never re-queue). b0xa, the
# global live-K=8 dyn arm and the graph cohorts are excluded by
# design. NO smoke: both code paths carry production gates.
B20R_KEY = "mistral_bottom20_section4_repl"
B20R_FIXED_KEY = B20R_KEY + "_fixed"
B20R_EVO_KEY = B20R_KEY + "_evo"
B20R_MANIFEST_PATH = os.path.join(
    HERE, "manifest_bottom20_section4_repl.json")
B20R_ARMS = ["b0", "d8"]
B20R_CONDS = ["fixed", "evolving"]
B20R_SEEDS = [42, 43]


def b20r_rows():
    """(fixed_rows, evo_rows): the 152 genuinely-missing jobs,
    straight from the audited manifest -- counts are asserted for
    CONSISTENCY with the 2026-08-19 audit (40 reused / 152 new),
    never forced."""
    mf = json.load(open(B20R_MANIFEST_PATH))
    cells = mf["cells"]
    assert mf["n_cells"] == 192 and len(cells) == 192, mf["n_cells"]
    assert {(c["cond"], c["arm"], c["gate"], c["es"], c["seed"])
            for c in cells} == \
        {(cd, a, g, e, s) for cd in B20R_CONDS for a in B20R_ARMS
         for g in B20_GATES for e in B20_ESS for s in B20R_SEEDS}
    reused = [c for c in cells if c["status"] == "reused"]
    new = [c for c in cells if c["status"] == "new"]
    assert len(reused) == 40 and len(new) == 152, \
        (len(reused), len(new))
    assert all(c.get("verdict") == "PASS" for c in reused)
    # audited reuse composition: only b0 cells reuse -- the 8 fixed
    # no-peer originals (es=0) + 32 archived evolving cells
    assert all(c["arm"] == "b0" for c in reused)
    assert sum(1 for c in reused
               if c["cond"] == "fixed" and c["es"] == 0.0) == 8
    assert sum(1 for c in reused if c["cond"] == "evolving") == 32
    fixed_rows, evo_rows_ = [], []
    for c in sorted(new, key=lambda c: (c["seed"], c["cond"],
                                        c["arm"], c["gate"],
                                        c["es"])):
        if c["cond"] == "fixed":
            # every fixed b0 es0 slot reused, so b20_tag's tokenless
            # rule never fires here -- new fixed cells all carry
            # _stub_ (asserted: a change means the archive moved and
            # the wave must be re-planned, not forced)
            assert not (c["arm"] == "b0" and c["es"] == 0.0), c
            r = b20_row(c["arm"], c["gate"], c["es"], seed=c["seed"])
            fixed_rows.append(r)
        else:
            r = evo_row(c["arm"], c["gate"], c["es"], seed=c["seed"])
            evo_rows_.append(r)
        assert r.split(",")[0].strip() == c["new_tag"], \
            (r.split(",")[0], c["new_tag"])
    return fixed_rows, evo_rows_


def b20r_fixed_sub():
    return B20R_FIXED_SUB_TEMPLATE.format(
        key=B20R_FIXED_KEY, n_jobs=len(b20r_rows()[0]),
        **REACH_MODELS["mistral7b"])


def b20r_evo_sub():
    return B20R_EVO_SUB_TEMPLATE.format(
        key=B20R_EVO_KEY, n_jobs=len(b20r_rows()[1]),
        **REACH_MODELS["mistral7b"])


B20R_FIXED_SUB_TEMPLATE = """\
# HTCondor: SECTION-4 THREE-SEED REPLICATION, FIXED condition --
# 88 NEW jobs at seeds 42/43 (per seed: b0 es>0 20 + d8 24; the 8
# tokenless b0 es0 no-peer originals REUSE per the audited
# manifest_bottom20_section4_repl.json). GENERATED by
# gen_pofd_sweep.py from the B20R block. Never edit by hand: rerun
# the script. {n_jobs} job(s).
# Identical environment to the completed seed-0 b20 full grid:
# cohort A = the 145 LOWEST-innate agents (INNATE_CLAMP_MODE=bottom,
# deterministic innate-then-id ranking, cohort seed = run seed),
# pinned bit-exact at innate in population AND twin; B = the other
# 578. Every row runs the one-sided STUBBORN operator (inert at the
# es=0 in-wave baselines). b0 trains fresh on all 723 labels; d8 is
# frozen personal-history ICL (ICL_K=0, ICL_DAYS=8). ICL_DAYS rides
# queue col 25 (8 on d8, else 0); the exclusion flag rides col 26
# (always 0 -- b0xa is NOT part of this wave). W=0.5, lam=0.2,
# gamma=0, greedy serving, WITH_TWIN=1, movielens Action, 30 rounds,
# SEEDS 42/43. NO smoke (both paths carry production gates).
# Gate every pull with check_pofd_sanity (CLAMP section: bottom-mask
# reconstruction, fixed bit-exact in pop AND twin, stubborn peer
# invariants, d8 personal-history replay -- the checker is
# seed-generic).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=$(icldays) ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 INNATE_CLAMP_MODE=$(cmode) INNATE_CLAMP_FRAC=0.2 INNATE_CLAMP_SEED=$(seed) INNATE_CLAMP_PEER_MODE=stubborn SFT_EXCLUDE_CLAMPED=$(sftexcl) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_mistral7b_pofdclampb20"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, cmode, icldays, sftexcl from experiments/condor/configs_pofd_{key}.txt
"""


B20R_EVO_SUB_TEMPLATE = """\
# HTCondor: SECTION-4 THREE-SEED REPLICATION, EVOLVING condition --
# 64 NEW jobs at seeds 42/43 (per seed: b0 es 0.05/0.1 8 + d8 24;
# the 32 archived evolving b0 cells at es 0/0.2/0.4/1 REUSE per the
# audited manifest_bottom20_section4_repl.json: pofdreach /
# pofdpeer2 / pofdgate2d / pofdws2f field-level matches). GENERATED
# by gen_pofd_sweep.py from the B20R block. Never edit by hand:
# rerun the script. {n_jobs} job(s).
# Identical environment to the completed seed-0 pofdevo wave: ALL
# 723 agents evolve normally (no fixed cohort, no exclusion, NO
# clamp env anywhere), standard symmetric peer dynamics, matched
# no-platform twin (WITH_TWIN=1). b0 trains fresh each round on all
# 723 current opinions; d8 serves frozen weights with each agent's
# OWN last 8 opinions as context (ICL_K=0, ICL_DAYS=8,
# icl_days_log.json.gz for the byte-exact replay). Cohort A (145
# lowest-innate) exists only in the ANALYSIS mask. W=0.5, lam=0.2,
# gamma=0, greedy serving, movielens Action, 30 rounds, SEEDS 42/43.
# NO smoke (both paths carry production gates). ICL_DAYS rides queue
# col 24 (8 on d8, 0 on b0).
# Gate every pull with check_pofd_sanity (EVO section: no clamp
# artifacts, mandatory twin, d8 personal-history byte replay,
# all-723-label FRESH on b0 -- the checker is seed-generic).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=$(icldays) ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_mistral7b_pofdevo"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, icldays from experiments/condor/configs_pofd_{key}.txt
"""


# feature_endogenization_n5 (2026-08-19): the FIVE-SEED EXTENSION of
# the main feature-endogenization figure (panels a/b of
# plot_feature_endogenization_main.py). The six established Qwen
# conditions run at seeds {0, 42, 43}; this key adds seeds 44 and 45
# ONLY -- 12 conceptual cells, all NEW per the field-level audit
# (audit_feature_endogenization_n5.py ->
# manifest_feature_endogenization_n5.json, which self-verifies its
# want surface against all 18 established cells before deciding any
# reuse, and hard-fails if a new tag is already occupied):
#   nat_l0    ordinary SFT, KL coefficient lambda=0    2 (pofdws2f_ b0)
#   nat_l0p5  forward-KL SFT, lambda=0.5               2 (pofdws2f_)
#   nat_l1    forward-KL SFT, lambda=1                 2 (pofdws2f_)
#   frozen    frozen weights, K=0                      2 (pofdicls2_)
#   removed   lambda=1 + PROFILE_DROP_COLS=gender      2 (pofdfegd_)
#   permuted  lambda=1 + PROFILE_PERMUTE_COLS=gender   2 (pofdfegp_)
# The runner spells the KL coefficient kl_beta and the tags spell it
# b<...> -- the established convention, kept byte-identical here so
# the new seeds land in the same families. Every DISPLAYED analysis
# label calls it lambda.
# Four queue schemas ride one umbrella key (the environments differ:
# natural / frozen-ICL / drop / permute), exactly mirroring the
# established fes / fef / fegd / fegp subs with two changes: the
# seeds, and GPU PINNING. Every established run executed on an
# A100-SXM4-80GB (11 distinct hosts, verified from the condor logs;
# those configs predate the hardware block), and greedy generation is
# only bit-reproducible within one architecture, so the new rows pin
# CUDADeviceName to the same A100 rather than inheriting the pool's
# H100/B200 mix.
# NO smoke: all six execution paths are already production-validated.
FE5_KEY = "feature_endogenization_n5"
FE5_MANIFEST_PATH = os.path.join(
    HERE, "manifest_feature_endogenization_n5.json")
FE5_SEEDS = [44, 45]
FE5_LAMBDAS = [0.0, 0.5, 1.0]
FE5_A100 = 'NVIDIA A100-SXM4-80GB'
FE5_ARMS = {"nat": ("nat_l0", "nat_l0p5", "nat_l1"),
            "frozen": ("frozen",), "gd": ("removed",),
            "gp": ("permuted",)}


def fe5_tag(cond, seed):
    """Byte-identical to the established families at the new seeds.
    (The seed-0 lambda=0 anchor lives in the reverse-era pofdws2_
    family; seeds 44/45 are pofdws2f_ like 42/43.)"""
    if cond == "nat_l0":
        return ("pofdws2f_qwen7b_b0_ea0p4_w0p5_l0p2_es0p2"
                f"_s{seed}_fresh_data")
    if cond == "nat_l0p5":
        return ("pofdws2f_qwen7b_b0p5_ea0p4_w0p5_l0p2_es0p2"
                f"_s{seed}_fresh_data")
    if cond == "nat_l1":
        return ("pofdws2f_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2"
                f"_s{seed}_fresh_data")
    if cond == "frozen":
        return f"pofdicls2_qwen7b_w0p5_l0p2_es0p2_ea0p4_k0_s{seed}"
    if cond == "removed":
        return ("pofdfegd_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2"
                f"_s{seed}_fresh_data")
    return ("pofdfegp_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2"
            f"_s{seed}_fresh_data")


def fe5_row(cond, seed):
    """The established row grammar: ROW_WS (15 cols) for the trained
    arms, ROW_ICL2 (18 cols) for the frozen arm."""
    if cond == "frozen":
        return ROW_ICL2.format(
            tag=fe5_tag(cond, seed), seed=seed, es="0.2",
            eps_ai="0.4", iclk=0, icldays=0, iclsrc="live")
    lam = {"nat_l0": 0.0, "nat_l0p5": 0.5}.get(cond, 1.0)
    return ROW_WS.format(
        tag=fe5_tag(cond, seed),
        style="sft" if lam == 0.0 else "sft_kl",
        beta=f"{lam:g}", seed=seed, eps_ai="0.4")


def fe5_rows(arm):
    """The genuinely-missing rows for one queue schema, straight from
    the audited manifest -- counts are asserted for CONSISTENCY with
    the 2026-08-19 audit (0 reused / 12 new), never forced."""
    mf = json.load(open(FE5_MANIFEST_PATH))
    cells = mf["cells"]
    assert mf["n_cells"] == 12 and len(cells) == 12, mf["n_cells"]
    assert {(c["cond"], c["seed"]) for c in cells} == \
        {(cd, s) for cd in
         [c for arms in FE5_ARMS.values() for c in arms]
         for s in FE5_SEEDS}
    # the established seeds must never re-queue under this key
    assert all(c["seed"] in FE5_SEEDS for c in cells)
    rows = []
    for c in sorted((c for c in cells if c["cond"] in FE5_ARMS[arm]),
                    key=lambda c: (c["cond"], c["seed"])):
        if c["status"] != "new":
            continue
        r = fe5_row(c["cond"], c["seed"])
        assert r.split(",")[0].strip() == c["new_tag"], \
            (r.split(",")[0], c["new_tag"])
        rows.append(r)
    return rows


def fe5_sub(arm):
    return FE5_SUB_TEMPLATE.format(
        key=f"{FE5_KEY}_{arm}", n_jobs=len(fe5_rows(arm)),
        what=FE5_SUB_WHAT[arm], extra_env=FE5_SUB_ENV[arm],
        suffix=FE5_SUB_SUFFIX[arm], gpu=FE5_A100,
        cols=FE5_SUB_COLS[arm])


FE5_SUB_WHAT = {
    "nat": ("natural-gender arm: ordinary SFT (lambda=0) + forward-KL "
            "SFT at lambda 0.5 and 1"),
    "frozen": "frozen-weights control (K=0, never trains)",
    "gd": "gender-REMOVED control (PROFILE_DROP_COLS=gender)",
    "gp": "gender-PERMUTED control (PROFILE_PERMUTE_COLS=gender)",
}
# byte-identical to the established fes / fef / fegd / fegp envs
_FE5_TRAINED_ENV = (
    "KL_DIRECTION=forward INNATE_LAMBDA=0.2 ANS_SAMPLE_K=16 "
    "ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 FRESH_EACH_ROUND=1 "
    "TRAIN_CAP=723 N_ROUNDS=30 EPOCH_SIZE=100 "
    "BASE_MODEL=Qwen/Qwen2.5-7B-Instruct SFT_EPOCHS=1 "
    "SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 USE_LORA=1 "
    "SFT_LR=5e-5")
FE5_SUB_ENV = {
    "nat": f"EPS_AI=$(eps_ai) {_FE5_TRAINED_ENV}",
    "frozen": ("EPS_AI=$(eps_ai) ICL_K=$(iclk) ICL_DAYS=$(icldays) "
               "ICL_SELECT=random ICL_CTX_SOURCE=$(iclsrc) "
               "INNATE_LAMBDA=0.2 USE_LORA=0 FRESH_EACH_ROUND=0 "
               "TRAIN_CAP=723 N_ROUNDS=30 EPOCH_SIZE=100 "
               "BASE_MODEL=Qwen/Qwen2.5-7B-Instruct "
               "GEN_BATCH_SIZE=32"),
    "gd": f"PROFILE_DROP_COLS=gender EPS_AI=$(eps_ai) "
          f"{_FE5_TRAINED_ENV}",
    "gp": f"PROFILE_PERMUTE_COLS=gender EPS_AI=$(eps_ai) "
          f"{_FE5_TRAINED_ENV}",
}
FE5_SUB_SUFFIX = {"nat": "_qwen7b_lora512_pofdfes",
                  "frozen": "_qwen7b_pofdfef",
                  "gd": "_qwen7b_lora512_pofdfegd",
                  "gp": "_qwen7b_lora512_pofdfegp"}
_FE5_COLS = ("tag, style, beta, seed, deploy_every, regime, pscale, "
             "anchor, pop, eps, gamma, wplat, mode, canary, eps_ai")
FE5_SUB_COLS = {"nat": _FE5_COLS, "gd": _FE5_COLS, "gp": _FE5_COLS,
                "frozen": _FE5_COLS + ", iclk, icldays, iclsrc"}


FE5_SUB_TEMPLATE = """\
# HTCondor: FEATURE-ENDOGENIZATION FIVE-SEED EXTENSION, {what}.
# Qwen-7B, seeds 44/45 ONLY. GENERATED by gen_pofd_sweep.py from the
# FE5 block. Never edit by hand: rerun the script. {n_jobs} job(s).
# Byte-identical environment to the established seeds {{0,42,43}} of
# the main feature figure: movielens Action (723 agents), EPS_AI=0.4,
# EPS_SOCIAL=0.2, W_PLAT=0.5, INNATE_LAMBDA=0.2, 30 rounds, replace +
# fresh adapter each round, forward KL against the FIXED pristine
# base on every KL arm, nested AI-then-peer operator, twin forced
# (eps>0), greedy serving. The runner spells the KL coefficient
# kl_beta (tags say b<...>); displayed analysis labels call it
# lambda.
# GPU PINNED to {gpu}: every established run of this experiment
# executed on that architecture (verified from the condor logs -- the
# configs predate the hardware block), and greedy generation is only
# bit-reproducible within one architecture, so an unpinned job could
# land on H100/B200 and add cross-architecture noise to a five-seed
# mean.
# The 12 cells are ALL new per the audited manifest; no established
# tag appears here, and the exec is idempotent besides.
# NO smoke: all six execution paths are production-validated.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = 128G
request_disk      = 40G
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env} N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=64 SEED_BASE_DATA=1 WANDB_RUN_SUFFIX={suffix}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue {cols} from experiments/condor/configs_pofd_{key}.txt
"""


# feature_lambda1_seeds (2026-08-20): REPLICATION of the natural
# lambda=1 feature-endogenization cell at 10 further seeds (46-55).
# WHY: across the five existing seeds the lambda=1 condition is
# BIMODAL, not noisy-around-a-mean. Four seeds show a transient
# incremental-R^2 spike at an arbitrary round that decays back to
# ~0 by round 29 (late-window means 0.005 / 0.014 / 0.010 / 0.000);
# one seed (45) locks in late and STAYS there (late-window 0.105,
# round-29 gender gap +0.067 vs +0.000..+0.009 elsewhere). One
# lock-in in five draws is consistent with anything from ~5% to
# ~50%, so the rate is the thing worth measuring.
# Scope: the lambda=1 natural cell ONLY -- 1 job per seed. The
# controls do not need re-replication: across the same five seeds
# lambda=0 and lambda=0.5 sit at the noise floor (|peak| <= 0.003)
# and the frozen arm is stable (peak 0.011-0.015, final
# 0.008-0.013). Panels (a)/(b) of the main figure need MATCHED
# seeds across all six conditions, so these seeds extend the RATE
# estimate, not the figure; extending the figure would be 6x the
# jobs.
# Byte-identical to the existing natural cells apart from the seed
# (same pofdws2f_ family, same fes environment, A100-pinned like the
# seed-44/45 runs).
FL1_KEY = "feature_lambda1_seeds"
FL1_SEEDS = list(range(46, 56))
FL1_EXISTING_SEEDS = [0, 42, 43, 44, 45]

# feature_lambda1_repro (2026-08-20): the REPRODUCIBILITY control the
# lock-in analysis has been missing, plus four more independent draws.
#   3 x seed 0 replicates (_rep1/_rep2/_rep3, identical config and
#     identical RNG seed) -- does a fixed seed reproduce its own
#     trajectory at all? LoRA training on GPU is not bitwise
#     deterministic by default, and greedy generation is only
#     bit-reproducible within one GPU architecture (2026-08-19
#     finding), so 30 rounds of closed-loop feedback could amplify
#     numerical noise into a completely different outcome. If the
#     replicates land near seed 0's late-window 0.006, run-to-run
#     variation is genuinely seed-driven and the ~20% lock-in rate
#     means what it says. If they scatter across 0.00-0.10 like
#     different seeds do, then "seed" is the wrong frame and the
#     rate is measuring chaotic amplification, which would also
#     explain why NOTHING at initialisation predicts the outcome
#     (innate is bit-identical across seeds; round-0 served R^2
#     correlates -0.16 with the final value).
#   4 x seeds 1-4 -- four more independent draws, n 15 -> 19. NOTE:
#     seed values have no metric structure; these are not "closer
#     to" seed 0 than 46-55 are, they are just four more draws.
# All 7 are A100-pinned, matching seed 0's original host (g144) --
# so a replicate that diverges cannot be blamed on architecture.
FL1R_KEY = "feature_lambda1_repro"
FL1R_REPLICATES = [1, 2, 3]        # of seed 0
FL1R_NEW_SEEDS = [1, 2, 3, 4]

# feature_lambda_match (2026-08-20, user): fill in lambda=0.5 and
# lambda=0 at the three runs that exist only for lambda=1 -- seed 1
# and seed-0 replicates 2 and 3 -- so Figure 3 can draw every lambda
# curve from the same set of runs. 2 conditions x 3 runs = 6 jobs.
# After this, lambda=0.5 and lambda=0 each have s0, s1, s0_rep2,
# s0_rep3, s42, s43, s44, s45 available (8 runs), all of which also
# exist for lambda=1.
# The replicate tags reuse the _rep<N>_ infix proven on the lambda=1
# repro wave (gates clean, universal seed regex still reads _s0_).
# Same family, same environment, A100-pinned; only the KL arm and the
# run identity differ.
FLM_KEY = "feature_lambda_match"
FLM_ARMS = [("b0p5", "sft_kl", "0.5"), ("b0", "sft", "0")]
FLM_RUNS = [(1, None), (0, 2), (0, 3)]   # (seed, replicate-or-None)


def flm_tag(arm, seed, rep):
    rep_tok = f"_rep{rep}" if rep else ""
    return (f"pofdws2f_qwen7b_{arm}_ea0p4_w0p5_l0p2_es0p2"
            f"_s{seed}{rep_tok}_fresh_data")


def flm_rows():
    return [ROW_WS.format(tag=flm_tag(arm, s, r), style=style,
                          beta=beta, seed=s, eps_ai="0.4")
            for arm, style, beta in FLM_ARMS
            for s, r in FLM_RUNS]


def flm_sub():
    return FE5_SUB_TEMPLATE.format(
        key=FLM_KEY, n_jobs=len(flm_rows()),
        what=("lambda=0.5 and lambda=0 at seed 1 and seed-0 "
              "replicates 2/3, to match the lambda=1 run set"),
        extra_env=FE5_SUB_ENV["nat"], suffix=FE5_SUB_SUFFIX["nat"],
        gpu=FE5_A100, cols=FE5_SUB_COLS["nat"])


def fl1r_tag(seed, rep=None):
    rep_tok = f"_rep{rep}" if rep else ""
    return ("pofdws2f_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2"
            f"_s{seed}{rep_tok}_fresh_data")


def fl1r_rows():
    rows = [ROW_WS.format(tag=fl1r_tag(0, r), style="sft_kl",
                          beta="1", seed=0, eps_ai="0.4")
            for r in FL1R_REPLICATES]
    rows += [ROW_WS.format(tag=fl1r_tag(s), style="sft_kl", beta="1",
                           seed=s, eps_ai="0.4")
             for s in FL1R_NEW_SEEDS]
    return rows


def fl1r_sub():
    return FE5_SUB_TEMPLATE.format(
        key=FL1R_KEY, n_jobs=len(fl1r_rows()),
        what=("lambda=1 REPRODUCIBILITY control: 3 seed-0 replicates "
              "+ seeds 1-4"),
        extra_env=FE5_SUB_ENV["nat"], suffix=FE5_SUB_SUFFIX["nat"],
        gpu=FE5_A100, cols=FE5_SUB_COLS["nat"])


def fl1_tag(seed):
    return ("pofdws2f_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2"
            f"_s{seed}_fresh_data")


def fl1_rows():
    return [ROW_WS.format(tag=fl1_tag(s), style="sft_kl", beta="1",
                          seed=s, eps_ai="0.4")
            for s in FL1_SEEDS]


def fl1_sub():
    return FE5_SUB_TEMPLATE.format(
        key=FL1_KEY, n_jobs=len(fl1_rows()),
        what=("natural lambda=1 REPLICATION seeds 46-55, for the "
              "lock-in rate"),
        extra_env=FE5_SUB_ENV["nat"], suffix=FE5_SUB_SUFFIX["nat"],
        gpu=FE5_A100, cols=FE5_SUB_COLS["nat"])


# qwen_k1_grid (2026-08-19, user via Celestine): the FULL-ANCHOR
# (k=1) replica of the Section-3 equilibrium grid for Qwen2.5-7B.
# Byte-identical to the completed k=0.2 Qwen2.5 cells except
# INNATE_LAMBDA 0.2 -> 1. Arms b0 (ordinary SFT) and b1 (forward-KL
# SFT, lambda_KL=1) x eps_AI {0.1, 0.2, 0.4, 1} x eps_social
# {0, 0.05, 0.2} x seed 0 = 24 jobs, all NEW (no run in the archive
# has ever carried innate_lambda=1).
# k=1 means the FJ anchor is total: each round the opinion re-anchors
# fully to innate, h = 1*innate + 0*x, and the platform blend still
# moves gated agents (x = (1-W)*innate + W*served on contact). The
# dynamics become memoryless rather than degenerate -- there is no
# division by (1-k) anywhere in the runner, so k=1 is numerically
# safe; it is simply a regime that has never been run.
# POP_RESET stays OFF: the ONLY change vs the k=0.2 grid is k.
# TAGS: the innate anchor is spelled by the established _l<k>_ token
# (_l1_ here vs _l0p2_ there), NOT a bare _k1_ token -- _k<N>_ is
# already the ICL-K grammar (_k0_, _k8live_, _k32noai_) and "k0" is
# even in the fam arm regex, so a _k1_ token would be ambiguous. The
# family prefix pofdfamk1_ carries the k=1 identity instead. Because
# it still starts with "pofdfam" the checker's FAM branch applies
# (arm/gate/es surface, save_raw_gen) and the generic _w/_l/_es token
# gate pins innate_lambda=1; the ONE checker change needed was adding
# pofdfamk1_ to the fam slug lookup, which is anchored to the exact
# prefix. Verified end to end against a synthetic run with GENUINE
# k=1 dynamics -- the checker replays the nested update using the
# config anchor, so a k=0.2 trajectory relabelled as k=1 fails.
# NO smoke: the code path is the validated Section-3 path; only the
# anchor value is new.
QK1_KEY = "qwen_k1_grid"
# es=1 full-peer column (2026-08-19, added AFTER the 24-job base grid
# was already submitted). It ships as its OWN key so the in-flight
# cells are never re-queued: the idempotent exec only no-ops runs
# that are already COMPLETE, so re-submitting the base key mid-flight
# would be a write race, not a no-op.
QK1_ES1_KEY = "qwen_k1_grid_es1"
QK1_ARMS = ["b0", "b1"]
QK1_GATES = [0.1, 0.2, 0.4, 1.0]
QK1_ESS = [0.0, 0.05, 0.2]
QK1_ES1 = [1.0]
QK1_LAMBDA = 1.0
QK1_MODEL = "qwen7b"


def qk1_tok():
    """w/l token pair at the FULL anchor (k=1)."""
    return f"w{_num(W_WPLAT)}_l{_num(QK1_LAMBDA)}"


def qk1_tag(arm, gate, es, seed=0):
    return (f"pofdfamk1_{QK1_MODEL}_{arm}_ea{_num(gate)}_{qk1_tok()}"
            f"_es{_num(es)}_s{seed}")


def qk1_row(arm, gate, es, seed=0, nrounds=30):
    a = REACH_ARM_COLS[arm]
    m = FAM_MODELS[QK1_MODEL]
    return ROW_FAMG.format(
        tag=qk1_tag(arm, gate, es, seed),
        style=a["style"], beta=a["beta"], seed=seed, es=f"{es:g}",
        eps_ai=f"{gate:g}", iclk=a["iclk"], snap=a["snap"],
        uselora=a["uselora"], fresh=a["fresh"], ansk=a["ansk"],
        gg=a["gg"], nrounds=nrounds, basemodel=m["base_model"],
        chatthink=m["chatthink"], mem=m["mem"], disk=m["disk"],
        pplbatch=m["pplbatch"])


def qk1_rows(ess=None):
    return [qk1_row(arm, gate, es)
            for arm in QK1_ARMS
            for gate in QK1_GATES
            for es in (QK1_ESS if ess is None else ess)]


def qk1_sub(key=None, ess=None):
    key = key or QK1_KEY
    return QK1_SUB_TEMPLATE.format(key=key,
                                   n_jobs=len(qk1_rows(ess)))


QK1_SUB_TEMPLATE = """\
# HTCondor: QWEN2.5 FULL-ANCHOR (k=1) SECTION-3 GRID -- {n_jobs}
# jobs. GENERATED by gen_pofd_sweep.py from the QK1 block. Never
# edit by hand: rerun the script.
# The completed k=0.2 Qwen2.5 Section-3 grid with INNATE_LAMBDA
# changed 0.2 -> 1 and NOTHING else: Qwen/Qwen2.5-7B-Instruct, arms
# b0 (ordinary SFT) + b1 (forward KL against the FIXED pristine
# base), eps_AI 0.1/0.2/0.4/1 on the numeric strict-< threshold
# gate, eps_social 0/0.05/0.2, W_PLAT=0.5, seed 0, 30 rounds,
# movielens Action (723 agents), fresh LoRA r512 each round, nested
# AI-then-peer operator, matched twin (WITH_TWIN=1), greedy serving,
# SAVE_RAW_GEN=1 -- every one of these matches the k=0.2 cells
# byte-for-byte.
# POP_RESET is deliberately NOT set: k is the only difference.
# The anchor rides the established _l1_ tag token (a bare _k1_ token
# would collide with the ICL-K grammar); the pofdfamk1_ prefix keeps
# the FAM checker branch while the generic _l gate pins
# innate_lambda=1.
# NO smoke: validated Section-3 path, new anchor value only.
# Gate every pull with check_pofd_sanity (FAM section + the generic
# _w/_l/_es token gate).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=1 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_qwen_k1_grid"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


# qwen_gate_sweep (2026-08-19): the SEED-0 GATE SWEEP for the two
# Qwen checkpoints. Complete conceptual grid = 2 models x eps_AI
# {0.05, 0.1, 0.2, 0.4, 1} (numeric strict-< threshold) x eps_social
# {0, 0.05, 0.1, 0.2, 0.4, 1} x seed 0 = 60 cells of regularized SFT
# at lambda=1 (the runner spells the KL coefficient kl_beta and the
# tags spell it b1; forward direction against the FIXED pristine
# base) on the canonical Action environment: movielens Action, 723
# agents, 30 rounds, W=0.5, innate anchor 0.2, gamma=0, fresh LoRA
# r512 each round, nested AI-then-peer operator, greedy serving,
# matched no-platform twin. Qwen3-8B runs CHAT_THINKING=0 (hybrid
# reasoning pinned OFF; the runner's qwen3 marker keeps
# completion-only SFT masking correct).
# Field-level audit (audit_qwen_gate_sweep.py ->
# manifest_qwen_gate_sweep.json) found 30 reused / 30 new -- the grid
# is half-covered by completed cells of the family-prior / gate-
# ablation (pofdfam_, 12 per model), eps_social scan (pofdesf_) and
# main-environment (pofdw2f_ / pofdws2f_) waves, all exact matches on
# the full config surface. Only the 30 genuinely-missing cells queue,
# under the NEW collision-safe pofdqgs_ family. Counts come from the
# manifest and are asserted for CONSISTENCY, never forced.
# SAVE_RAW_GEN is deliberately NOT set: the reused cells are split on
# that field (the pofdfam_ ones have it, the others do not), it is
# pure output plumbing that cannot touch the simulation, and this
# sweep has no use for raw generations -- so it is not part of the
# audited match surface either.
# The twin: WITH_TWIN=1 writes one for every new cell including
# es=0, whereas the reused es=0 cells predate that flag and have
# none. The analyzer therefore reads NOTHING from the twin -- every
# reported statistic (equilibrium mean, equilibrium SD, W1 from the
# initial population) is a property of the population alone.
# NO smoke: both checkpoints and this training path are validated.
QGS_KEY = "qwen_gate_sweep"
QGS_MANIFEST_PATH = os.path.join(
    HERE, "manifest_qwen_gate_sweep.json")
QGS_MODELS = ["qwen7b", "qwen3_8b"]
QGS_GATES = [0.05, 0.1, 0.2, 0.4, 1.0]
QGS_ESS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]


def qgs_tag(model, gate, es, seed=0):
    return (f"pofdqgs_{model}_b1_ea{_num(gate)}_{w_tok()}"
            f"_es{_num(es)}_s{seed}")


def qgs_row(model, gate, es, seed=0, nrounds=30):
    a = REACH_ARM_COLS["b1"]
    m = FAM_MODELS[model]
    return ROW_FAMG.format(
        tag=qgs_tag(model, gate, es, seed),
        style=a["style"], beta=a["beta"], seed=seed, es=f"{es:g}",
        eps_ai=f"{gate:g}", iclk=a["iclk"], snap=a["snap"],
        uselora=a["uselora"], fresh=a["fresh"], ansk=a["ansk"],
        gg=a["gg"], nrounds=nrounds, basemodel=m["base_model"],
        chatthink=m["chatthink"], mem=m["mem"], disk=m["disk"],
        pplbatch=m["pplbatch"])


def qgs_rows():
    """The genuinely-missing cells, straight from the audited
    manifest -- counts are asserted for CONSISTENCY with the
    2026-08-19 audit (30 reused / 30 new), never forced."""
    mf = json.load(open(QGS_MANIFEST_PATH))
    cells = mf["cells"]
    assert mf["n_cells"] == 60 and len(cells) == 60, mf["n_cells"]
    assert {(c["model"], c["gate"], c["es"]) for c in cells} == \
        {(m, g, e) for m in QGS_MODELS for g in QGS_GATES
         for e in QGS_ESS}
    assert all(c["seed"] == 0 for c in cells), "seed 0 only"
    reused = [c for c in cells if c["status"] == "reused"]
    new = [c for c in cells if c["status"] == "new"]
    assert len(reused) == 30 and len(new) == 30, \
        (len(reused), len(new))
    assert all(c.get("verdict") == "PASS" for c in reused)
    rows = []
    for c in sorted(new, key=lambda c: (c["model"], c["gate"],
                                        c["es"])):
        r = qgs_row(c["model"], c["gate"], c["es"])
        assert r.split(",")[0].strip() == c["new_tag"], \
            (r.split(",")[0], c["new_tag"])
        rows.append(r)
    return rows


def qgs_sub():
    return QGS_SUB_TEMPLATE.format(key=QGS_KEY,
                                   n_jobs=len(qgs_rows()))


QGS_SUB_TEMPLATE = """\
# HTCondor: QWEN GATE SWEEP (seed 0) -- {n_jobs} NEW cells of the
# 60-cell grid (2 models x eps_AI 0.05/0.1/0.2/0.4/1 x eps_social
# 0/0.05/0.1/0.2/0.4/1); the other 30 REUSE completed runs per the
# audited manifest_qwen_gate_sweep.json and are NOT queued here.
# GENERATED by gen_pofd_sweep.py from the QGS block. Never edit by
# hand: rerun the script.
# Regularized SFT at lambda=1 (tags say b1; forward KL against the
# FIXED pristine base) on the canonical Action environment: movielens
# Action 723 agents, 30 rounds, W=0.5, INNATE_LAMBDA=0.2, gamma=0,
# numeric strict-< threshold gate, nested AI-then-peer operator,
# matched twin (WITH_TWIN=1), greedy serving, fresh LoRA r512 each
# round. BASE_MODEL / CHAT_THINKING / mem / disk / PPL_BATCH ride the
# queue: Qwen3-8B runs CHAT_THINKING=0 (thinking pinned OFF; the
# runner's qwen3 marker keeps completion-only masking correct),
# Qwen2.5-7B keeps its default template. Both load from the offline
# lustre cache.
# NO smoke: both checkpoints and this training path are already
# production-validated.
# Gate every pull with check_pofd_sanity (QGS section: exact model id
# per slug, qwen3 chat_thinking=False, lambda=1 forward from the _b
# token, gate/es/seed surface from the tag tokens).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_qwen_gate_sweep"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


# ===========================================================================
# QWEN2.5 MECHANISM DIAGNOSTIC (2026-08-20). Two production keys, both
# small, plus one smoke. Everything else the diagnostic needs -- the
# perfect-prediction oracle, the long-horizon oracle grid, the offline
# frozen population replays and the two beta_eff identity checks -- is
# CPU-only and never touches Condor.
#
# THE QUESTION. What actually changes when a population and a platform
# are put in a feedback loop, and how much of it is the pretrained model
# rather than the loop itself? The decomposition is a ladder of five
# conditions at matched (k, eps_social):
#
#   twin                     no platform at all
#   perfect prediction       m(t) = x(t): an exact population echo
#   frozen Qwen  K = D = 0   the STATIC entering prediction map
#   ordinary SFT lambda = 0  parametric retraining on the loop's own data
#   regularized SFT lambda=1 the same, plus explicit forward-KL retention
#
# and the contrasts are read strictly as
#
#   perfect - twin        effect of CLOSING an exact echo loop
#   frozen - perfect      effect of replacing that echo with the static
#                         entering Qwen map
#   ordinary SFT - perfect   AGGREGATE parametric-retraining gap. NOT
#                         "optimizer error": ordinary SFT still starts
#                         from pretrained Qwen weights and a finite-rank
#                         LoRA, so this contrast bundles pretrained
#                         initialization, limited capacity, finite
#                         optimization, shared parameters across agents,
#                         greedy decoding, parsing, and generalization
#                         across profiles.
#   regularized - ordinary   explicit forward-KL reference retention
#   regularized - frozen     how learning from the EVOLVING population
#                         changes an already-retained model signal
#
# THEORY MAPPING (Wu et al., "Reaching a Consensus in Predictive Loops",
# arXiv:2603.12137). Our pre-peer update is
#     z = (1 - W)[k x_innate + (1 - k) x] + W m.
# Under perfect prediction m = x this is FJ with
#     z = (1 - beta_eff) x_innate + beta_eff x,  beta_eff = 1 - (1 - W) k,
# so: the paper setting (k=.2, W=.5) has beta_eff=.9; (k=1, W=.5) has
# beta_eff=.5, i.e. k=1 ALONE IS NOT A CONSENSUS LIMIT; with imperfect
# Qwen predictions k=1 gives the direct Wu form z = (1-W)x_innate + W m;
# the real high-susceptibility boundary is k=1, W=1, where the pre-peer
# population EQUALS the served vector and k drops out algebraically.
# Wu et al.'s consensus result needs perfect prediction AND susceptibility
# -> 1; perfect prediction at finite susceptibility can keep a
# heterogeneous equilibrium. Our randomized Deffuant midpoint process is
# not their deterministic FJ operator -- with a connected graph and
# genuinely open peer interaction it is a randomized-gossip ANALOGUE that
# should converge under perfect prediction at W=1. Qualitative
# limiting-case correspondence, not a replication of their theorem.
#
# WARNING carried into the analysis: comparing k=.2 with k=1 at fixed
# W=.5 changes beta_eff (.9 -> .5) as well as innate/state anchoring. It
# is NOT a pure memory ablation.
# ===========================================================================

# --- Part A: the five missing/superseding frozen cells ---------------------
# qwen_mechanism_frozen. The 32-cell paper-regime grid is
# k {.2, 1} x eps_social {0, .05, .2, 1} x 4 platform arms; 8 of those
# are the CPU oracle, leaving 24 GPU cells. The field-level audit
# (audit_qwen_mechanism.py -> manifest_qwen_mechanism.json) found
# 19 reused / 5 new:
#   * all 16 ordinary/regularized SFT cells already exist (pofdfam_,
#     pofdctxgrid_, pofdqgs_ at k=.2; pofdfamk1_ at k=1) -- exact matches
#     on the full config surface;
#   * three frozen H100 cells are reusable (k=.2, es .05/.2/1);
#   * five frozen cells are missing and queue here.
# THE A100 REFUSAL. The archived k=.2, es=0 frozen cell
# pofdreach_qwen7b_k0_ea1_w0p5_l0p2_es0_s0 matches on every config field
# but ran on an A100, and its parsed prediction vector differs from the
# H100 frozen prior in 17 of 723 agents (MAE .0091, max .5). A frozen
# K=D=0 model never sees the population, so that vector is a CONSTANT
# that the whole grid is compared against -- letting one corner carry a
# different constant would contaminate exactly the k-comparison this
# diagnostic exists to make. It is superseded by a hardware-matched
# rerun, not reused.
# Hence the H100 pin below, and hence the checker requirement that all
# eight frozen cells share ONE canonical prediction sha256
# (1674ee5f...da30bb, DERIVED by the audit, not hand-entered) and that
# every frozen cell's predictions are constant across rounds.
# NO smoke: frozen K=D=0 Qwen2.5 serving is the most-exercised path in
# the project; only the anchor value and the hardware pin are new.
QMECH_KEY = "qwen_mechanism_frozen"
QMECH_MANIFEST_PATH = os.path.join(HERE, "manifest_qwen_mechanism.json")
QMECH_MODEL = "qwen7b"
QMECH_ARM = "k0"                 # frozen, K = D = 0
QMECH_GATE = 1.0                 # numeric strict-< eps_AI, NOT all_open
QMECH_KS = [0.2, 1.0]
QMECH_ESS = [0.0, 0.05, 0.2, 1.0]
# exact architecture string, not a family: the pool also reports a bare
# "NVIDIA H100" for a different SKU, and the point of the pin is that
# every frozen cell decodes on the SAME silicon
QMECH_H100 = "NVIDIA H100 80GB HBM3"
QMECH_CANONICAL_PRED_SHA = (
    "1674ee5f8d833f46de672791d933e1d3bdeefb07484c2d110dec84ce71da30bb")

# KNOWN-BAD NODES. g106 and i104 are excluded by ~88 other .sub files in
# this directory; the GPU-pinned template these two waves were built from
# (FE5) did not carry the list, so the first Wu-limit submission put two
# jobs straight onto bad silicon -- one of them i104 -- and both died in
# 9s with "CUDA unavailable ... exiting 17 for retry" (2026-08-20).
# i101 joins the list from that same submission: same failure mode.
# Note (TARGET.Machine =!= MY.LastRemoteHost) does NOT help here -- it
# compares a slot-qualified name against a bare hostname and is inert --
# so a retry can land right back on the node that just failed. Explicit
# exclusion is the only thing that actually keeps them off.
BAD_NODES = ("g106", "i104", "i101")
BAD_NODE_REQ = "".join(
    f' && (TARGET.Machine != "{h}.internal.cluster.is.localnet")'
    for h in BAD_NODES)

# k rides the QUEUE here (unlike every earlier family, which pinned one
# INNATE_LAMBDA in the sub env) because this grid spans k=.2 AND k=1
ROW_QMECH = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
             "ab, {es}, 0.0, 0.5, loop, 0.0, {eps_ai}, threshold, "
             "{lam}, {iclk}, {snap}, {uselora}, {fresh}, {ansk}, {gg}, "
             "{nrounds}, {basemodel}, {chatthink}, {mem}, {disk}, "
             "{pplbatch}")


def qmech_tag(arm, k, es, seed=0):
    """The anchor rides the established _l<k>_ token: a bare _k1_ token
    would collide with the ICL-K grammar, where _k0_ already spells the
    frozen arm."""
    return (f"pofdqmech_{QMECH_MODEL}_{arm}_ea{_num(QMECH_GATE)}_w"
            f"{_num(W_WPLAT)}_l{_num(k)}_es{_num(es)}_s{seed}")


def qmech_row(arm, k, es, seed=0, nrounds=30):
    a = REACH_ARM_COLS[arm]
    m = FAM_MODELS[QMECH_MODEL]
    return ROW_QMECH.format(
        tag=qmech_tag(arm, k, es, seed), style=a["style"], beta=a["beta"],
        seed=seed, es=f"{es:g}", eps_ai=f"{QMECH_GATE:g}", lam=f"{k:g}",
        iclk=a["iclk"], snap=a["snap"], uselora=a["uselora"],
        fresh=a["fresh"], ansk=a["ansk"], gg=a["gg"], nrounds=nrounds,
        basemodel=m["base_model"], chatthink=m["chatthink"], mem=m["mem"],
        disk=m["disk"], pplbatch=m["pplbatch"])


def qmech_rows():
    """ONLY the cells the audited manifest marks new. Counts come from
    the manifest and are asserted for CONSISTENCY below, never forced:
    if the archive changes, the assertion fires instead of silently
    queueing a different grid."""
    mf = json.load(open(QMECH_MANIFEST_PATH))
    new = [c for c in mf["cells"] if c["status"] == "new"]
    return [qmech_row(c["arm"], c["innate_k"], c["eps_social"], c["seed"])
            for c in sorted(new, key=lambda c: (c["innate_k"],
                                                c["eps_social"]))]


def qmech_sub():
    return QMECH_SUB_TEMPLATE.format(key=QMECH_KEY,
                                     n_jobs=len(qmech_rows()),
                                     gpu=QMECH_H100, bad=BAD_NODE_REQ)


QMECH_SUB_TEMPLATE = """\
# HTCondor: QWEN2.5 MECHANISM DIAGNOSTIC -- FROZEN CELLS ({n_jobs}
# jobs). GENERATED by gen_pofd_sweep.py from the QMECH block. Never
# edit by hand: rerun the script.
# The 32-cell paper-regime grid is k {{.2, 1}} x eps_social
# {{0, .05, .2, 1}} x {{perfect prediction, frozen, ordinary SFT,
# regularized SFT}}. Perfect prediction is a CPU oracle (8 cells) and
# all 16 SFT cells already exist, so the only GPU work is the frozen
# arm: 3 H100 cells reuse, {n_jobs} queue here, per the audited
# manifest_qwen_mechanism.json.
# Frozen = Qwen/Qwen2.5-7B-Instruct, K = D = 0 (plain zero-shot
# prompting, no LoRA, no memory, nothing trains), eps_AI = 1 on the
# NUMERIC strict-< threshold gate (never all_open), W = 0.5, gamma = 0,
# 30 rounds, movielens Action 723 agents, seed 0, nested AI-then-peer
# operator, matched twin (WITH_TWIN=1), greedy serving. INNATE_LAMBDA
# rides the QUEUE ($(lam)) because this grid spans k = .2 AND k = 1.
# GPU PINNED to {gpu}. A frozen K=D=0 model never sees the population,
# so its prediction vector is a CONSTANT the entire grid is compared
# against -- and that constant is hardware-specific. The archived A100
# cell differs from the H100 prior in 17 of 723 agents (MAE .0091, max
# .5), which is why the k=.2/es=0 corner is RERUN here rather than
# reused: a mixed-silicon corner would contaminate exactly the
# k-comparison this diagnostic exists to make. The pin is the exact SKU
# string, not a family -- the pool also reports a bare "NVIDIA H100".
# Gate every pull with check_pofd_sanity (QMECH section: constant
# predictions across rounds, the one canonical prediction sha256 shared
# by all eight frozen cells, H100 hardware, and the k/gate/es/seed
# surface read from the tag tokens).
# NO smoke: frozen Qwen2.5 serving is the most-exercised path here;
# only the anchor value and the hardware pin are new.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) PEER_GATE_MODE=threshold INNATE_LAMBDA=$(lam) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_qwen_mechanism_frozen"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, lam, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


# --- Part C: the exact Wu-style boundary with Qwen -------------------------
# qwen_wu_limit[_smoke]. Four trained-LLM cells at the TRUE limiting
# boundary: k = 1, W in {.5, 1}, arms b0 (ordinary SFT) and b1
# (regularized, forward KL lambda = 1), 100 rounds, seed 0, movielens
# Action 723 agents, fresh LoRA r512 every round, the same optimizer /
# rank / training surface / reference model as the paper, matched twin,
# one peer sweep, gamma = 0, H100-80GB only, raw responses + parsed
# predictions + training losses + populations + twins all saved.
#
# GENUINELY OPEN GATES, AND WHY NOT "1". At the boundary BOTH channels
# must be truly open. Both gates are STRICT inequalities, so the numeric
# value 1 does NOT open them: an agent at 0 served 1, or a peer pair at
# (0, 1), sits at distance exactly 1 and is still REJECTED under a
# threshold of 1. Representing "open" as 1 would therefore quietly drop
# exactly the extreme pairs the consensus limit is about. So the AI side
# uses the existing AI_GATE_MODE=all_open and the peer side uses the NEW
# PEER_GATE_MODE=all_open (2026-08-20, gp.peer_gate). The tags spell it
# _eaopen_ / _esopen_, never _ea1_ / _es1_, and the checker rejects any
# numeric-threshold job wearing an open tag (and vice versa).
# PEER_GATE_MODE defaults to "threshold" and is applied AFTER pair
# selection, so every archived run and every threshold-mode run stays
# byte-identical and consumes identical RNG.
# EPS is still 0.2 in these rows: it is inert for ACCEPTANCE under
# all_open, but eps_social = 0 is how "no peer step" is spelled
# everywhere else in this project, so the runner refuses that
# combination rather than let one run mean two things.
#
# WHAT THIS DOES AND DOES NOT ASSERT. The CPU oracle at k=1, W=1 with
# both gates open must reach consensus (mean preserved, SD < 1e-5 by
# round 300) and check_perfect_predictor enforces that. The four Qwen
# arms here are NOT required to reach consensus -- whether practical
# retraining does or does not is the phenomenon being measured.
# THE SMOKE. This wave introduces the first genuinely open PEER path in
# production, so one short 3-round lambda=1, W=1 cell runs first to
# exercise the new gate mode, training, adapter re-serving, finite CE/KL
# losses, complete raw outputs and the twin. It is a SEPARATE key and is
# NOT part of the four-job production count.
QWU_KEY = "qwen_wu_limit"
QWU_SMOKE_KEY = "qwen_wu_limit_smoke"
QWU_MODEL = "qwen7b"
QWU_ARMS = ["b0", "b1"]
QWU_WS = [0.5, 1.0]
QWU_K = 1.0
QWU_EPS_SOCIAL = 0.2      # inert under all_open; see the note above
QWU_ROUNDS = 100
QWU_SMOKE_ROUNDS = 3
QWU_H100 = QMECH_H100

# W and k both ride the queue; both gate modes are pinned open in the
# sub env (every row of this key is the open condition)
ROW_QWU = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
           "ab, {es}, 0.0, {wplat}, loop, 0.0, {lam}, {iclk}, {snap}, "
           "{uselora}, {fresh}, {ansk}, {gg}, {nrounds}, {basemodel}, "
           "{chatthink}, {mem}, {disk}, {pplbatch}")


# Smoke generation. "smoke" (v1) was submitted before the serving fix of
# 2026-08-20, so any run wearing that tag decoded with LoRA dropout still
# active (Trainer.train() leaves the model in training mode and
# _generate never forced eval), which makes its "greedy" predictions
# non-deterministic. The idempotent executable no-ops COMPLETED runs, so
# reusing the tag would silently keep that stale result. Bump the token
# instead: a new tag can never be satisfied by the old directory.
QWU_SMOKE_TOKEN = "smoke2"


def qwu_tag(arm, w, seed=0, rounds=QWU_ROUNDS, smoke=False):
    """_eaopen_/_esopen_ spell the genuinely open gates -- never _ea1_ or
    _es1_, which are numeric strict-< thresholds that reject a
    distance-1 pair. The horizon is in the tag because 100-round and
    3-round cells of the same condition are different objects."""
    sm = QWU_SMOKE_TOKEN if smoke else ""
    return (f"pofdqwu_{QWU_MODEL}_{arm}_eaopen_w{_num(w)}_l{_num(QWU_K)}"
            f"_esopen_s{seed}_r{rounds}{sm}")


def qwu_row(arm, w, seed=0, rounds=QWU_ROUNDS, smoke=False):
    a = REACH_ARM_COLS[arm]
    m = FAM_MODELS[QWU_MODEL]
    return ROW_QWU.format(
        tag=qwu_tag(arm, w, seed, rounds, smoke), style=a["style"],
        beta=a["beta"], seed=seed, es=f"{QWU_EPS_SOCIAL:g}",
        wplat=f"{w:g}", lam=f"{QWU_K:g}", iclk=a["iclk"], snap=a["snap"],
        uselora=a["uselora"], fresh=a["fresh"], ansk=a["ansk"],
        gg=a["gg"], nrounds=rounds, basemodel=m["base_model"],
        chatthink=m["chatthink"], mem=m["mem"], disk=m["disk"],
        pplbatch=m["pplbatch"])


def qwu_rows():
    return [qwu_row(arm, w) for w in QWU_WS for arm in QWU_ARMS]


# --- Part C addendum: the PERSONAL-HISTORY ICL arm at the boundary --------
# qwen_wu_limit_icl (2026-08-21). Two more cells of the SAME Wu-boundary
# experiment: k=1, W in {.5, 1}, both gates genuinely open, 100 rounds,
# seed 0, H100, matched twin -- everything identical to the four trained
# cells except the platform arm.
#
# THE ARM. d8 = personal-history in-context learning: D=8, K=0. Each
# agent's prompt carries THEIR OWN last 8 opinions, oldest to newest, and
# NOTHING about anyone else (K=0, so no cross-user exemplars at all).
# Weights are frozen -- no SFT, no LoRA, no gradients. So this is a third
# kind of platform, distinct from the three already in the figure:
#   frozen K=D=0   a STATIC map: the same prediction forever
#   d8 (this)      a MEMORY map: no learning, but the served value tracks
#                  the agent's own recent trajectory
#   b0 / b1        PARAMETRIC retraining on the population's labels
# It isolates whether the loop needs weight updates at all, or whether
# per-agent history alone reproduces the feedback dynamics.
#
# ITS OWN KEY, deliberately. The four trained cells are COMPLETE; adding
# these rows to qwen_wu_limit would make that key queue six jobs, and
# while the idempotent exec no-ops completed runs, a separate key means
# the finished wave cannot be touched at all -- no reliance on the
# no-op, and one submit command that queues exactly these two.
# NO SMOKE: the all-open production path is already validated by the four
# completed cells, and frozen personal-history ICL is the established d8
# path from the bottom-20% waves. Nothing here is a new code path.
# =========================================================================
# OBSERVATION-RATE SUBSAMPLING (2026-08-21, qwen_subsample[_smoke]).
# Celestine's hypothesis: ordinary SFT leans MORE on the pretrained model
# when it observes less of the population. This tests it directly by
# varying only how much data the optimizer sees.
#
# THE DESIGN. Take the completed Wu-boundary b0 cell exactly as it is --
# Qwen2.5-7B-Instruct, ordinary SFT (lambda=0), k=1, W=1, both gates
# genuinely all_open, fresh LoRA every round, 100 rounds, seed 0, H100,
# greedy eval-mode serving -- and change ONE thing: the number of agents
# whose labels enter the SFT batch each round.
#     2%   5%   10%   25%   50%   75%   100%
#     14   36    72   181   362   542    723
# The platform still SERVES all 723 agents every round in every arm; only
# the training batch is cut. So the population dynamics see the same
# closed loop and the arms differ purely in observation.
# (75% -> 542 = round(.75*723); the grid is the spec's, and 542 was added
# 2026-08-21 before any production job was submitted, so it joins the
# existing key rather than needing one of its own.)
#
# THE 100% ARM IS NOT RERUN. It IS the completed
# pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100 -- same config, no
# sampling -- so six observation arms queue, not seven.
#
# WHY A NEW KNOB. N_LABELED takes a fixed PREFIX of agents: the same
# people every round, which asks "who is observed", not "how many".
# TRAIN_CAP is applied only in the t>0 branch, so round 0 would train on
# all 723 and the first adapter would not be subsampled at all. Hence
# SFT_SAMPLE_N, opt-in and absent-by-default so every archived run is
# byte-identical.
#
# NESTED SUBSETS. Each round draws ONE permutation of all 723 agents from
# a dedicated stream seeded (SFT_SAMPLE_SEED + round) -- independent of
# the sample size -- and takes its prefix. So within a round the 14 are
# inside the 36 are inside the 72, and the arms are strictly nested
# rather than independent draws. That removes "different people" as an
# explanation for any difference between arms.
#
# THE COMPUTE-MATCHED CELL. A small arm also takes fewer optimizer steps
# (14 rows at batch 4 is 4 steps; 723 rows is 181), so a raw comparison
# confounds "saw less unique data" with "trained less". The 10% arm is
# therefore ALSO run with SFT_SAMPLE_REPEAT_TO=723: the same 72 sampled
# agents, tiled to exactly 723 rows, giving 181 steps -- identical
# compute to the full-data arm, on 72 distinct agents.
# =========================================================================

# =========================================================================
# SFT TRAINING-DOSE FAMILIES (2026-08-21). Three one-round STATIC scouts
# that ask the same question three ways: does a weaker SFT fit leave the
# served vector closer to the entering Qwen model?
#
#   qwen_sft_update_dose   U   optimizer updates    {1,5,20,50,100,181}
#   qwen_sft_lr_dose       LR  learning rate        {1e-6,3e-6,1e-5,3e-5}
#   qwen_sft_rank_dose     r   LoRA rank            {1,4,8,32,128}
#
# ZERO IS FROZEN QWEN in all three families -- U=0, LR=0 and rank=0 all
# mean "no adaptation happened", which is the entering model exactly, so
# none of them needs a GPU job. The upper endpoint of each family is the
# paper's current setting. Note rank 512 is the PAPER DEFAULT and is
# unusually large for a LoRA, which is why the informative band is
# 8-32 rather than the top end.
#
# WHY ONE ROUND. The 100-round subsample wave showed the closed loop
# drives the population to an absorbing constant, after which the
# projection a is 1 BY CONSTRUCTION and measures nothing. These cells
# train ONCE on the innate labels and serve once, so the served vector is
# read before any feedback exists. That is the only regime in which
# "closer to frozen Qwen" is a statement about the model rather than
# about the loop.
#
# THE SHARED ENDPOINT. U=181 / LR=5e-5 / rank=512 is the standard
# complete one-epoch SFT fit for 723 examples at batch 4, and it is the
# full-dose end of ALL THREE families. It is queued exactly once, in the
# update-dose key; the other two reuse it. U=0, LR=0 and (conceptually)
# rank=0 are all the canonical frozen-Qwen vector and need no GPU job.
#
# WHAT THE ARMS ACTUALLY VARY -- state it plainly rather than overclaim:
#   U   is a TRAINING-DOSE intervention, not a pure optimizer-step one:
#       fewer updates also means fewer EXAMPLES were processed (U steps
#       at batch 4 sees 4U rows of the 723).
#   LR  limits how far the weights move. It does NOT test preservation
#       of broad semantic capability -- only of the entering prediction
#       map on these prompts.
#   r   limits WHAT the update can represent, with alpha = 2r so the
#       LoRA scaling alpha/r is constant across ranks. This is the
#       cleanest capacity test of the three; a small adapter may instead
#       learn only a global scalar shift, which is why the analysis
#       reports prediction SD, unique values and max mode share.
#
# TAG GRAMMAR. _l1_ keeps its established meaning (innate anchor k=1) --
# it is NOT reused for the KL coefficient. The dose dials get their own
# unambiguous tokens: _u<updates>_, _lr<rate>_, _rank<r>_. The trailing
# _r1 is the ROUND count, which is why the rank token is spelled "rank"
# and not "r".
# =========================================================================
SFTD_UPDATE_KEY = "qwen_sft_update_dose"
SFTD_LR_KEY = "qwen_sft_lr_dose"
SFTD_RANK_KEY = "qwen_sft_rank_dose"
SFTD_SMOKE_KEY = "qwen_sft_dose_smoke"
SFTD_MODEL = "qwen7b"
SFTD_W = 1.0
SFTD_K = 1.0
SFTD_EPS_SOCIAL = 0.2          # inert under all_open
SFTD_ROUNDS = 1                # ONE adaptation round: static diagnostic
SFTD_H100 = QMECH_H100
SFTD_STD_U = 181               # ceil(723/4): one complete epoch
SFTD_STD_LR = "5e-5"
SFTD_STD_RANK = 512
SFTD_UPDATES = [1, 5, 20, 50, 100, 181]
SFTD_LRS = ["1e-6", "3e-6", "1e-5", "3e-5"]
SFTD_RANKS = [1, 4, 8, 32, 128]


def _lrtok(lr):
    """5e-5 -> 5em5, 1.25e-5 -> 1p25em5. Unambiguous and filename-safe."""
    return str(lr).replace("-", "m").replace(".", "p")


def sftd_tag(u, lr, rank, rounds=SFTD_ROUNDS, smoke=False):
    return (f"pofdsftdose_{SFTD_MODEL}_u{u}_lr{_lrtok(lr)}_rank{rank}"
            f"_eaopen_w{_num(SFTD_W)}_l{_num(SFTD_K)}_esopen_s0"
            f"_r{rounds}{'smoke' if smoke else ''}")


def sftd_row(u, lr, rank, rounds=SFTD_ROUNDS, smoke=False):
    a = REACH_ARM_COLS["b0"]          # ordinary SFT, lambda_KL = 0
    m = FAM_MODELS[SFTD_MODEL]
    return ROW_SFTD.format(
        tag=sftd_tag(u, lr, rank, rounds, smoke), style=a["style"],
        beta=a["beta"], seed=0, es=f"{SFTD_EPS_SOCIAL:g}",
        wplat=f"{SFTD_W:g}", lam=f"{SFTD_K:g}", steps=u, lr=lr, rank=rank,
        iclk=a["iclk"], snap=a["snap"], uselora=a["uselora"],
        fresh=a["fresh"], ansk=a["ansk"], gg=a["gg"], nrounds=rounds,
        basemodel=m["base_model"], chatthink=m["chatthink"], mem=m["mem"],
        disk=m["disk"], pplbatch=m["pplbatch"])


ROW_SFTD = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
            "ab, {es}, 0.0, {wplat}, loop, 0.0, {lam}, {steps}, {lr}, "
            "{rank}, {iclk}, {snap}, {uselora}, {fresh}, {ansk}, {gg}, "
            "{nrounds}, {basemodel}, {chatthink}, {mem}, {disk}, "
            "{pplbatch}")


def sftd_update_rows():
    """U sweep at the standard LR and rank. Includes the shared U=181
    endpoint, which the LR and rank families REUSE rather than re-queue."""
    return [sftd_row(u, SFTD_STD_LR, SFTD_STD_RANK) for u in SFTD_UPDATES]


def sftd_lr_rows():
    """LR sweep at the standard U and rank. LR=5e-5 is the shared
    endpoint and is deliberately absent."""
    return [sftd_row(SFTD_STD_U, lr, SFTD_STD_RANK) for lr in SFTD_LRS]


def sftd_rank_rows():
    """Rank sweep at the standard U and LR. rank=512 is the shared
    endpoint and is deliberately absent."""
    return [sftd_row(SFTD_STD_U, SFTD_STD_LR, r) for r in SFTD_RANKS]


def sftd_smoke_rows():
    """ONE tiny cell exercising the new SFT_MAX_STEPS + SAVE_SFT_ORDER
    path end to end before any production job runs."""
    return [sftd_row(5, SFTD_STD_LR, SFTD_STD_RANK, smoke=True)]


def sftd_sub(key):
    rows = {SFTD_UPDATE_KEY: sftd_update_rows,
            SFTD_LR_KEY: sftd_lr_rows,
            SFTD_RANK_KEY: sftd_rank_rows,
            SFTD_SMOKE_KEY: sftd_smoke_rows}[key]()
    return SFTD_SUB_TEMPLATE.format(key=key, n_jobs=len(rows),
                                    gpu=SFTD_H100, bad=BAD_NODE_REQ)


SFTD_SUB_TEMPLATE = """\
# HTCondor: SFT TRAINING-DOSE SCOUT -- {n_jobs} jobs, ONE adaptation
# round each. GENERATED by gen_pofd_sweep.py from the SFTD block. Never
# edit by hand: rerun the script.
# Asks whether a WEAKER SFT fit leaves the served vector closer to the
# entering Qwen model, three ways: optimizer updates (U), learning rate,
# and LoRA rank. Shared surface = the QWU boundary configuration:
# Qwen/Qwen2.5-7B-Instruct, ordinary SFT (lambda_KL=0), k=1, W=1, BOTH
# gates all_open, one peer sweep, gamma=0, fresh LoRA, batch 4, greedy
# eval-mode serving, replace-only data, no ICL/replay/pristine/reference
# adapter, movielens Action 723 agents, seed 0, matched twin.
# ONE ROUND ON PURPOSE. The 100-round subsample wave showed the closed
# loop drives the population to an absorbing constant, after which the
# projection a equals 1 by construction and measures nothing. These
# cells train once on the INNATE labels and serve once, so the served
# vector is read before any feedback exists.
# SFT_EPOCHS=0 + SFT_MAX_STEPS=$(steps) is the EXISTING step-cap path
# (the pofdbud_ budget wave used it); no new knob was invented for it.
# SAVE_SFT_ORDER=1 persists the ordered (ids, labels) handed to the
# learner, and the learner records the optimizer steps that ACTUALLY
# ran plus the sampler seed that fixes the minibatch order.
# U is a TRAINING-DOSE dial, not a pure step dial: U updates at batch 4
# processes 4U of the 723 rows. LoRA alpha = 2r throughout, so alpha/r
# is constant across ranks.
# Gate every pull with check_pofd_sanity (SFTD section: exact step
# count, one round, shared dataset order, eval-mode serving, finite
# in-range predictions, zero parse failures).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 AI_GATE_MODE=all_open PEER_GATE_MODE=all_open EPS_AI=1 INNATE_LAMBDA=$(lam) SFT_EPOCHS=0 SFT_MAX_STEPS=$(steps) SFT_LR=$(lr) LORA_R=$(rank) SAVE_SFT_ORDER=1 ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{key}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, lam, steps, lr, rank, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


# =========================================================================
# ADAPTER KL / SOFT-DECODE PROBE (AKL, 2026-08-21) -- ONE job, no training
#
# WHY. The dose wave measured distance in SERVED-OPINION space, and
# served opinions come out of GREEDY decoding. Greedy is an argmax, so a
# tiny change in the token distribution can flip the served number and a
# large change can leave it alone. Frozen Qwen serves only five distinct
# values here (98.9% of agents get 0.25 or 0.65), so the whole served map
# turns on one token-level decision -- precisely the regime where argmax
# amplifies. The dose result therefore establishes that weak SFT does not
# preserve Qwen's GREEDY map, and says nothing yet about the token
# distribution. This probe measures the distributional statement.
#
# NOT A POPULATION RUN. It trains nothing and serves no loop: it loads
# the base model once and scores each saved round-0 LoRA adapter against
# it with teacher-forced forwards, so it uses its OWN executable rather
# than run_one_pokec_gated_idempotent.sh (there is no trajectory.pt to be
# idempotent about). One job covers all 15 adapters: the base is loaded
# once and peft's disable_adapter() supplies the reference inside the
# same forward, which is what makes the two distributions comparable.
#
# H100 ONLY. The gate tying this probe to the archived runs is the
# canonical frozen-Qwen served hash, and greedy generation is only
# bit-reproducible within one GPU architecture.
# =========================================================================
# =========================================================================
# FRIEDKIN-JOHNSEN ROBUSTNESS WAVE (FJR, 2026-08-21)
#
# THE QUESTION. Does the main ordinary-SFT (b0) vs forward-KL-SFT (b1)
# result survive when the Deffuant bounded-confidence peer process is
# replaced by a LINEAR FJ operator? Smallest useful appendix comparison:
# six models x two arms, one seed, one configuration. Not a new sweep.
#
# THE OPERATOR: Jiduan Wu's model, homogeneous-parameter specialization
# (FJ_UPDATE_VERSION=wu1, opt-in; "legacy" is untouched):
#     x_init^(t) = (1 - beta) x^innate + beta m_t
#     u^(0)      = x_init^(t)
#     u^(l+1)    = (1 - alpha) x_init^(t) + alpha P u^(l),  l = 0..K-1
# beta = W_PLAT * platform_sus * PLATFORM_SUS_SCALE; alpha = .9 is the
# PEER SUSCEPTIBILITY and K = 100 inner steps run per outer round. The
# human component is the raw innate opinion with no carryover: k = 1.
#
# THE ONLY POPULATION-SIDE QUANTITY THAT VARIES IS beta, whose
# complement 1 - beta is the innate weight: .5 for the core wave, 1 for
# the optional boundary wave (where the innate term drops out entirely).
# No Deffuant memory or gate parameters enter anywhere.
#
# ALPHA vs THE INTERNAL COEFFICIENT. FJWorld.peer_sus is STUBBORNNESS,
# i.e. 1 - alpha. The wave records fj_peer_alpha=.9 and passes .1 to the
# internal field; FJWorld.run_wu refuses the pair if they disagree,
# because passing .9 straight in would run peer susceptibility .1 -- the
# near-opposite dynamics, with every downstream number still well-formed.
#
# WHY THE ARCHIVED FJ PATH COULD NOT BE USED AS-IS. Four blockers, all
# verified in source before this wave was written:
#   * W_PLAT never reached FJ -- it is applied only in the ab branch, so
#     a tag claiming beta=.5 would have run at beta=1.
#   * MovieLens ships peer_sus=1, and peer_sus is the ANCHOR weight, so
#     the archived operator does no neighbour mixing at all.
#   * world.run() re-queried the model, so the vector recorded in
#     pred_raw was not guaranteed to be the vector applied.
#   * the inner loop started from the PREVIOUS round's state, carrying
#     population memory the Wu recurrence does not have.
#
# NO GATE TOKENS IN THE TAG. FJ has no confidence gates -- platform
# exposure and graph mixing are unconditional -- so _ea / _es would name
# something the operator never applies. The runner REFUSES a non-default
# gate mode under wu1 rather than letting it sit inert.
# =========================================================================
FJR_KEY = "fj_robustness"
FJR_SMOKE_KEY = "fj_robustness_smoke"
FJR_FROZEN_KEY = "fj_robustness_frozen"
FJR_BETA1_KEY = "fj_robustness_beta1"      # optional, NOT part of the core
FJR_KL_KEY = "fj_robustness_kl_ladder"     # beta=1 KL-dose ladder
FJR_H100 = QMECH_H100
FJR_BETA = 0.5             # core; the optional boundary wave uses 1.0
FJR_BETA_BOUNDARY = 1.0
FJR_ALPHA = 0.9            # PEER SUSCEPTIBILITY (internal field gets .1)
FJR_INNER = 100            # K inner FJ steps per outer round
FJR_ROUNDS = 30
FJR_SMOKE_ROUNDS = 3
FJR_ARMS = ["b0", "b1"]
# KL-DOSE LADDER (2026-08-21). The beta=1 wave showed that the frozen
# (lambda -> infinity) endpoint does NOT collapse -- it retains SD
# .014-.054 through the same operator -- while both lambda=0 and
# lambda=1 do. Compared at a MATCHED round, lambda=1 preserves LESS
# spread than plain SFT in four of six models, so the collapse is not
# monotone in the KL dose and nothing between lambda=1 and infinity has
# been measured. b2/b8 bracket that gap. beta=1 only: it is the only
# surface where anything collapses.
FJR_KL_ARMS = ["b2", "b8"]
FJR_MODELS = ["qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b",
              "mistral7b", "ministral8b"]
# frozen zero-shot vectors verified field by field (model, dataset,
# n_labeled, profiles, style=frozen, no LoRA, icl_k=icl_days=0, constant
# across rounds, H100 80GB HBM3, transformers 5.5.4, torch 2.5.1) and
# found to carry ONE distinct prediction hash per model. These need no
# job; the FJ replay is model-independent and runs locally on CPU.
FJR_FROZEN_REUSE = {
    "qwen7b": "pofdqmech_qwen7b_k0_ea1_w0p5_l1_es0p05_s0",
    "qwen3_8b": "pofdzsprior_qwen3_8b_w0p5_l0p2_es0_s0",
    "olmo7b": "pofdctxgrid_olmo7b_k0_ea0p05_w0p5_l0p2_es0p4_s0",
    "olmo3_7b": "pofdzsprior_olmo3_7b_w0p5_l0p2_es0_s0",
    "ministral8b": "pofdzsprior_ministral8b_w0p5_l0p2_es0_s0",
}
# Mistral-7B has 393 frozen runs and NOT ONE qualifies: the 20 that are
# K=0 and constant ran on A100, and the H100 ones report the SKU
# "NVIDIA H100" (not "NVIDIA H100 80GB HBM3") and carry context. Greedy
# decoding is only bit-reproducible within one architecture -- the
# archived A100 frozen cell differs from the H100 prior in 17/723 agents
# -- so reusing an A100 vector would measure this one model against a
# different-hardware reference. Hence exactly one extraction job.
FJR_FROZEN_NEW = ["mistral7b"]


def fjr_tag(model, arm, rounds=FJR_ROUNDS, beta=FJR_BETA, alpha=FJR_ALPHA,
            inner=FJR_INNER, seed=0, smoke=False):
    """model, arm, beta, alpha, inner steps, k, seed, horizon -- no gate
    tokens, and a family prefix shared with nothing else."""
    return (f"pofdfj_{model}_{arm}_beta{_num(beta)}_alpha{_num(alpha)}"
            f"_in{inner}_k1_s{seed}_r{rounds}{'smoke' if smoke else ''}")


def fjr_frozen_tag(model, seed=0):
    """A zero-shot EXTRACTION, not an FJ run: it produces the static
    prediction vector, and the FJ replay is model-independent and happens
    locally on CPU. So the tag carries no beta/alpha/inner/k tokens --
    naming FJ parameters here would claim the job applied them."""
    return f"pofdfjzs_{model}_s{seed}_r1"


ROW_FJR = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
           "fj, 0.0, 0.0, {wplat}, loop, 0.0, {alpha}, {inner}, "
           "{uselora}, {fresh}, {ansk}, {gg}, {nrounds}, {basemodel}, "
           "{chatthink}, {mem}, {disk}, {pplbatch}")


def fjr_row(model, arm, rounds=FJR_ROUNDS, beta=FJR_BETA, smoke=False,
            seed=0):
    a = REACH_ARM_COLS[arm]
    m = FAM_MODELS[model]
    return ROW_FJR.format(
        tag=fjr_tag(model, arm, rounds, beta=beta, smoke=smoke, seed=seed),
        style=a["style"], beta=a["beta"], seed=seed, wplat=f"{beta:g}",
        alpha=f"{FJR_ALPHA:g}", inner=FJR_INNER, uselora=a["uselora"],
        fresh=a["fresh"], ansk=a["ansk"], gg=a["gg"], nrounds=rounds,
        basemodel=m["base_model"], chatthink=m["chatthink"], mem=m["mem"],
        disk=m["disk"], pplbatch=m["pplbatch"])


def fjr_rows(beta=FJR_BETA, arms=None, seed=0):
    return [fjr_row(mo, ar, beta=beta, seed=seed)
            for mo in FJR_MODELS for ar in (arms or FJR_ARMS)]


# SEED REPLICATES (2026-08-22). The seed-0 wave produced counts across
# models -- b0 retains more spread than b1 in 6/6 at beta=.5, b1 is
# closer to the frozen model in 5/6 at beta=1 -- and a count has no error
# bar. These add seeds 42/43/44.
#
# WHAT A SEED PERTURBS HERE. The FJ operator is DETERMINISTIC: no peer
# RNG, unlike the Deffuant process. So the seed moves only the learner
# (data order, sampler, LoRA init), and these replicates measure exactly
# one thing -- whether the b0 vs b1 gap is larger than training noise.
# That is the right test for a 6/6 count and the wrong one for a claim
# about population stochasticity, which this wave does not make.
FJR_SEEDS = [42, 43, 44]
FJR_SEED_KEY = "fj_robustness_seeds"
FJR_BETA1_SEED_KEY = "fj_robustness_beta1_seeds"


def fjr_seed_rows(beta=FJR_BETA):
    return [r for sd in FJR_SEEDS for r in fjr_rows(beta=beta, seed=sd)]


def fjr_kl_rows():
    """The beta=1 KL ladder: lambda in {2, 8} across all six models, so
    the shape is not a one-model story."""
    return fjr_rows(beta=FJR_BETA_BOUNDARY, arms=FJR_KL_ARMS)


def fjr_smoke_rows():
    """One 3-round Qwen2.5 forward-KL cell: exercises the whole wu1 path
    (beta applied, alpha mixing, inner reset, served-vector identity)
    before any 30-round job runs."""
    return [fjr_row("qwen7b", "b1", rounds=FJR_SMOKE_ROUNDS, smoke=True)]


ROW_FJR_FROZEN = ("{tag}, frozen, 0, 0, 1, replace, 1.0, fixed, ab, 0, "
                  "0.0, 0.5, loop, 0.0, 0, threshold, 0, -1, 0, 0, 0, 0, "
                  "1, {basemodel}, {chatthink}")


def fjr_frozen_rows():
    """Zero-shot extraction for the models with no qualifying archived
    vector. Uses the ZSPRIOR envelope exactly -- frozen, no LoRA, no
    context, EPS_AI=0 under the strict-< gate so no agent is ever
    contacted -- so pred_raw[0] IS the zero-shot prior. It is an
    EXTRACTION, not an FJ run: the FJ replay is model-independent and
    happens locally on CPU."""
    return [ROW_FJR_FROZEN.format(tag=fjr_frozen_tag(mo),
                                  basemodel=FAM_MODELS[mo]["base_model"],
                                  chatthink=FAM_MODELS[mo]["chatthink"])
            for mo in FJR_FROZEN_NEW]


FJR_FROZEN_SUB_TEMPLATE = """\
# HTCondor: FJ ROBUSTNESS -- ZERO-SHOT EXTRACTION, {n_jobs} job(s).
# GENERATED by gen_pofd_sweep.py from the FJR block. Never edit by hand.
# Produces the frozen (lambda -> infinity) prediction vector for the
# model(s) with NO qualifying archived vector. Mistral-7B-v0.3 has 393
# frozen runs and none qualify: the K=0 constant ones are A100, and the
# H100 ones report SKU "NVIDIA H100" and carry context. Greedy decoding
# is only bit-reproducible within an architecture, so reusing an A100
# vector would measure one model against a different-hardware reference.
# ENVELOPE = the ZSPRIOR screen exactly (frozen, no LoRA, no context,
# EPS_AI=0 under the strict-< gate so no agent is ever contacted), which
# is what makes pred_raw[0] the zero-shot prior AND keeps this vector
# comparable to the four zsprior-derived ones. INNATE_LAMBDA rides along
# at 0.2 for that byte-comparability; it is inert here because no agent
# is ever contacted and the population cannot move.
# The one deliberate difference from ZSPRIOR: this pins the EXACT H100
# SKU, which the zsprior sub does not.
# NOT an FJ run -- the FJ replay is model-independent and local.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = 128G
request_disk      = 40G
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward INNATE_LAMBDA=0.2 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=0 SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{key}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink from experiments/condor/configs_pofd_{key}.txt
"""


def fjr_frozen_sub(key, rows):
    return FJR_FROZEN_SUB_TEMPLATE.format(
        key=key, n_jobs=len(rows), gpu=FJR_H100, bad=BAD_NODE_REQ)


def fjr_sub(key, rows, smoke=False):
    return FJR_SUB_TEMPLATE.format(
        key=key, n_jobs=len(rows), gpu=FJR_H100, bad=BAD_NODE_REQ,
        rounds=FJR_SMOKE_ROUNDS if smoke else FJR_ROUNDS)


FJR_SUB_TEMPLATE = """\
# HTCondor: FRIEDKIN-JOHNSEN ROBUSTNESS WAVE -- {n_jobs} jobs, {rounds}
# rounds each. GENERATED by gen_pofd_sweep.py from the FJR block. Never
# edit by hand: rerun the script.
# Asks whether the ordinary-SFT (b0) vs forward-KL-SFT (b1) result
# survives when Deffuant bounded-confidence peers are replaced by a
# LINEAR FJ operator. Six models x two arms, seed 0, MovieLens Action,
# 723 agents, replace-only data, fresh LoRA r=512 each round, 1 epoch,
# batch 4, lr 5e-5, greedy eval-mode serving.
# JIDUAN WU'S MODEL, homogeneous-parameter specialization.
# FJ_UPDATE_VERSION=wu1 is the OPT-IN operator; "legacy" is the archived
# one and is untouched, so no existing FJ artifact changes.
#   x_init^(t) = (1-beta) x^innate + beta m_t
#   u^(0)      = x_init^(t)
#   u^(l+1)    = (1-alpha) x_init^(t) + alpha P u^(l),  l = 0..K-1
# beta = W_PLAT * platform_sus * PLATFORM_SUS_SCALE and is the ONLY
# population-side quantity that varies (.5 core, 1 boundary); its
# complement 1-beta is the innate weight. alpha=.9 is the PEER
# SUSCEPTIBILITY and K=100. u^(0)=x_init makes the human component
# stateless (k=1), so the ONLY channel between rounds is the model.
# ALPHA vs THE INTERNAL FIELD. FJWorld.peer_sus is STUBBORNNESS = 1-alpha.
# The run records fj_peer_alpha=.9 and passes .1 internally; run_wu
# REFUSES the pair if they disagree, because passing .9 straight in would
# run peer susceptibility .1 -- near-opposite dynamics, with every
# downstream number still well-formed.
# NO GATES, NO DEFFUANT PARAMETERS. FJ platform exposure and graph mixing
# are unconditional, so there are no _ea/_es tokens and the runner
# REFUSES a non-default gate mode under wu1 rather than letting it sit
# inert in the environment.
# THE INNER LOOP CONVERGES at alpha=.9, K=100 (~0.9^100), which is Wu's
# intent -- but that is the FJ fixed point of ONE round's anchor and says
# nothing about the OUTER model-population loop. A round-{rounds} state
# must not be called an equilibrium without the outer convergence check.
# Because convergence erases the inner loop's starting point, each round
# also saves u^(1), which is what lets the checker prove u^(0)=x_init.
# Gate every pull with check_pofd_sanity (FJR section) and hard-fail the
# analyzer until the whole conceptual grid is present.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 POP_MODEL=fj FJ_UPDATE_VERSION=wu1 FJ_ALPHA=$(alpha) FJ_INNER_STEPS=$(inner) PLATFORM_SUS_SCALE=1.0 KL_DIRECTION=forward SFT_EPOCHS=1 SFT_LR=5e-5 LORA_R=512 USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{key}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, alpha, inner, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


# =========================================================================
# JIDUAN WU / POKEC REPLICATION (WU, 2026-08-22)
#
# THE QUESTION. Wu's opinion-dynamics setup on its OWN dataset, with its
# OWN heterogeneous parameters, and a language model in the platform
# slot. Pokec LCC: N = 2163 agents, the FIRST 1730 rows are the OBSERVED
# set O (y_label2163.pk) and the LAST 433 are the HELD-OUT set U
# (y_unlabel_label2163.pk). The platform observes O and predicts U; the
# held-out truth exists only so the analysis can score the prediction and
# must never enter a prompt, a training batch or a served value.
#
# THE OPERATOR (FJ_UPDATE_VERSION=wu1, the opt-in operator; "legacy" is
# untouched), now with PER-AGENT parameters instead of the homogeneous
# specialization the MovieLens FJR wave used:
#     served^(t)   = m_t on U, and x^(t) on O          (passthrough)
#     x_init^(t)_i = (1 - beta_i) x^innate_i + beta_i served^(t)_i
#     u^(0)        = x_init^(t)
#     u^(l+1)_i    = (1 - alpha_i) x_init^(t)_i
#                    + alpha_i (P u^(l))_i,   l = 0 .. K-1
# alpha_i is PEER SUSCEPTIBILITY, shipped by the dataset in
# hetero_peer_sus2163.pkl (mean .8909); beta_i is PLATFORM
# SUSCEPTIBILITY, hetero_platform_sus2163.pkl (mean .8890). K = 100 inner
# steps, T = 50 outer rounds, seed 0 primary. The human component is the
# raw innate opinion with no carryover (k = 1), so the ONLY channel
# between outer rounds is the platform.
#
# THE ALPHA TRAP, WHICH THIS DATASET WALKS STRAIGHT INTO.
# FJWorld.peer_sus is STUBBORNNESS = 1 - alpha, but the file is NAMED
# peer_sus and holds alpha (mean .8909). Passing it through unchanged
# runs peer susceptibility .1091 -- the near-opposite dynamics -- and
# every downstream number stays finite, ordered and plausible. The
# checker (check_jiduan_pokec.py) REPLAYS all K inner steps with the
# per-agent complement and refuses a run whose trajectory matches the
# inverted convention.
#
# WHY THE SCALES EXIST. FJ_ALPHA_SCALE (c_alpha) and FJ_BETA_SCALE
# (c_beta) multiply the DATASET vectors; c = 1 is the exact
# heterogeneous dataset, c = 0 switches that channel off. W_PLAT stays
# 1.0 in every row of this family so beta is never scaled twice -- the
# dose lives in exactly one place, and the checker recomputes
# beta_realized = c_beta * beta_raw from the repo dataset.
# NO SCALAR FJ_ALPHA / FJ_BETA IS EMITTED ANYWHERE IN THIS FAMILY: a
# scalar sitting next to a dataset source is precisely the ambiguity the
# wave exists to avoid, so the sub carries neither and the checker
# hard-fails a run that records one as operative.
#
# NO GATE TOKENS IN THE TAG. FJ has no confidence gates -- platform
# exposure and graph mixing are unconditional -- so _ea / _es would name
# something the operator never applies.
#
# TAG GRAMMAR (one home, read by the checker and the analyzer; never
# re-derived elsewhere):
#   pofdwu_<model>_<arm>_pa<src><c_alpha>_pb<src><c_beta>_in<K>
#          [_rt<T|C>]_s<seed>_r<T>[smoke]
# src is "d" for the dataset vectors and "h" for a homogeneous scalar
# (reserved for the CPU controls; no GPU key uses it). The ARM token
# carries the whole platform channel -- KL dose for the trained arms,
# ICL mode and depth for the frozen ones -- so the ICL mode is named in
# the tag exactly once, by WU_ARM_COLS below.
# =========================================================================
WU_SMOKE_KEY = "jiduan_pokec_smoke"
WU_CONTROLS_KEY = "jiduan_pokec_controls"      # CPU ONLY -- zero Condor jobs
WU_PRIOR_KEY = "jiduan_pokec_prior"
WU_PRIOR_SEEDS_KEY = "jiduan_pokec_prior_seeds"
WU_LADDER_KEY = "jiduan_pokec_lambda_ladder"
WU_ICL_KEY = "jiduan_pokec_icl"
WU_ENV_KEY = "jiduan_pokec_environment"
WU_ROUTE_SMOKE_KEY = "jiduan_pokec_routing_smoke"
WU_ROUTE_SEEDS_KEY = "jiduan_pokec_routing_seeds"
WU_FROZEN_KEY = "jiduan_pokec_frozen"
WU_H100 = QMECH_H100
WU_ROUNDS = 50             # T, outer rounds
WU_SMOKE_ROUNDS = 3
WU_INNER = 100             # K_FJ, inner FJ steps per outer round
WU_N = 2163                # Pokec LCC
WU_N_OBSERVED = 1730       # O = the first 1730 rows (y_label2163.pk)
WU_N_HELDOUT = 433         # U = the last 433 rows (y_unlabel_label2163.pk)
WU_MODELS = ["qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b",
             "mistral7b", "ministral8b"]
WU_PRIOR_ARMS = ["b0", "b1"]
# Qwen2.5 is the project's reference checkpoint; OLMo-2 is the second
# provider family. Two models x two arms x two seeds is the smallest set
# that can say whether a b0-vs-b1 gap clears TRAINING noise -- the FJ
# operator is deterministic, so a seed moves the learner and nothing else.
WU_SEEDS = [42, 43]
WU_SEED_MODELS = ["qwen7b", "olmo7b"]
WU_LADDER_MODEL = "qwen7b"
# lambda -> arm token. 0 and 1 ARE the prior wave's b0/b1 cells; they are
# reused byte-for-byte and never re-queued. The FROZEN model is a
# SEPARATE endpoint of this family, not the lambda -> infinity end of
# this ladder: a frozen run never trains, so it is not "SFT at an
# infinite KL weight", and pretending otherwise would put a
# different-code-path point on a dose axis.
WU_LADDER = [(0.0, "b0"), (0.1, "b0p1"), (0.5, "b0p5"),
             (1.0, "b1"), (10.0, "b10")]
WU_ICL_MODELS = ["qwen7b", "mistral7b"]
# phist8 and ehist8 are BOTH here on purpose and must not be conflated.
# phist8 (strict Wu) replays the platform's own past PREDICTIONS -- the
# platform legitimately has those. ehist8 shows the agent its own past
# POST-FJ opinions, which Wu's platform never observes, and is the
# personal-history mechanism our Section 4 studies. Keeping both is what
# makes the strict-vs-extension contrast measurable rather than asserted.
WU_ICL_ARMS = ["b0", "b1", "phist8", "ehist8", "octx8", "frz"]
WU_ENV_MODEL = "qwen7b"
WU_ENV_ARMS = ["b0", "b1", "phist8"]
WU_ENV_SCALES = [0.0, 0.5, 1.0]
WU_ROUTE_MODEL = "qwen7b"
# ROUTING CARRIES BOTH HISTORY MECHANISMS, NEVER POOLED.
# ehist8 is the Section 4 mechanism (realized post-peer opinions), so it
# is the arm that tests whether shared SFT weights open a cross-agent
# route that agent-local history removes. phist8 is the strict
# Wu-compatible comparison: the platform recalling only what it served.
# The contrast is itself informative -- WITH peers, ehist8 can absorb
# peer-induced change into later prompts and phist8 cannot; WITHOUT
# peers, neither agent-local channel should open a cross-agent route.
# Reporting them separately is the point; averaging them would destroy
# exactly the distinction the stage exists to draw.
WU_ROUTE_ARMS = ["b0", "phist8", "ehist8", "frz"]
WU_ROUTE_CAS = [0.0, 1.0]
# 10% of the OBSERVED pool -> 173 agents, matching Wu's modified-label
# construction. NOT 25%: a larger cohort is a coarser instrument, and the
# point of this stage is a localized source perturbation.
WU_ROUTE_FRAC = 0.10
WU_ROUTE_SEED = 7          # cohort RNG, run-seed-independent by design
# The injected innate value is 1.0, again matching Wu. This is NOT a free
# choice. Observed innate runs [0.040, 0.917] with mean .5273, so:
#   * 1.0 sits OUTSIDE the entire observed range -- a clean, unambiguous
#     injection, and ZERO cohort agents are already at it, so every
#     treated agent actually moves.
#   * 0.5 would sit essentially AT the observed mean (.5273), making the
#     "treatment" a near-null perturbation, and 96 observed agents are
#     already at exactly 0.5 -- ~30 of a 432-cohort would not move at
#     all, silently diluting the effect the stage exists to measure.
WU_ROUTE_VALUE = 1.0
# WHERE THE INTERVENTION LANDS. The runner draws the cohort from the
# OBSERVED pool and rewrites those agents' INNATE opinion before anything
# reads it -- so the treatment reaches x(0), the FJ anchor, the round-0
# SFT labels and (through the passthrough) the served vector, all
# consistently. It is a SOURCE injection at agents the platform can see,
# and the question is whether it reaches the held-out set.
# THE CONTROL TWIN IS frac = 0. The cohort is a deterministic function of
# (ROUTING_TREAT_SEED, frac, |O|) and of nothing else -- not of the run
# seed -- so the checker recomputes it from the treatment's parameters
# and proves the two runs differ on EXACTLY that set and nowhere else.
# That is a stronger statement than comparing two stored masks, and it
# needs no out-of-range sentinel: the runner requires the injected value
# to lie in [0, 1], so there is no "route but change nothing" value.
WU_ROUTE_CONTROL_FRAC = 0.0


def _wu_src_tok(src, scale):
    """d = the dataset vector, h = a homogeneous scalar. The scale rides
    the same token so a tag can never name a source without its dose."""
    if src not in ("dataset", "homogeneous"):
        raise ValueError(f"bad source {src!r}")
    return f"{'d' if src == 'dataset' else 'h'}{_num(scale)}"


def wu_tag(model, arm, *, ca=1.0, cb=1.0, peer_src="dataset",
           plat_src="dataset", seed=0, rounds=WU_ROUNDS, inner=WU_INNER,
           route=None, smoke=False):
    """model, arm (= KL dose or ICL channel), alpha source+scale, beta
    source+scale, K, routing side, seed, horizon. No gate tokens."""
    rt = "" if route is None else f"_rt{route}"
    return (f"pofdwu_{model}_{arm}_pa{_wu_src_tok(peer_src, ca)}"
            f"_pb{_wu_src_tok(plat_src, cb)}_in{inner}{rt}"
            f"_s{seed}_r{rounds}{'smoke' if smoke else ''}")


def wu_frozen_tag(model, seed=0):
    """A zero-shot EXTRACTION, not a Wu run: one round, frozen weights, no
    context, so model_pred_raw[0] IS the frozen prediction map that every
    'distance to the frozen model' in the analysis is measured against.
    It carries no alpha/beta/K tokens because it applies none of them."""
    return f"pofdwuzs_{model}_s{seed}_r1"


def _wu_sft_arm(dose_arm, kl):
    """A trained arm, taking its ENVELOPE from REACH_ARM_COLS so this wave
    cannot drift from the rest of the project on LoRA/fresh/telemetry.
    dose_arm names the reach entry whose KL weight is already `kl`."""
    a = REACH_ARM_COLS[dose_arm]
    assert float(a["beta"]) == kl, (dose_arm, a["beta"], kl)
    return dict(style=a["style"], beta=a["beta"], uselora=a["uselora"],
                fresh=a["fresh"], ansk=a["ansk"], gg=a["gg"],
                iclmode="none", iclk=0, icld=0)


def _wu_new_dose(kl):
    """A KL dose the reach table does not carry (lambda .1 and 10). Copies
    the b1 envelope exactly; only the coefficient differs."""
    a = REACH_ARM_COLS["b1"]
    return dict(style=a["style"], beta=f"{kl:g}", uselora=a["uselora"],
                fresh=a["fresh"], ansk=a["ansk"], gg=a["gg"],
                iclmode="none", iclk=0, icld=0)


def _wu_frozen_arm(mode, k, d):
    """A frozen platform channel: no LoRA, no SFT, no adapter. Envelope
    from REACH_ARM_COLS["k0"] (the established frozen no-context arm);
    only the Wu context knobs differ."""
    a = REACH_ARM_COLS["k0"]
    assert a["style"] == "frozen" and a["uselora"] == 0 and a["fresh"] == 0
    return dict(style=a["style"], beta=a["beta"], uselora=a["uselora"],
                fresh=a["fresh"], ansk=a["ansk"], gg=a["gg"],
                iclmode=mode, iclk=k, icld=d)


# THE ARM TABLE IS THE ONE HOME FOR WHAT AN ARM TOKEN MEANS. The tag
# carries the token, this table carries the semantics, and
# check_jiduan_pokec.py reads the token back and gates the config
# against these exact values -- so a tag can never name a channel the
# run did not use.
#   b0/b0p1/b0p5/b1/b10  ordinary SFT (lambda 0) and forward-KL SFT at
#                        lambda .1/.5/1/10; fresh LoRA every round
#   frz                  frozen weights, NO context (K = D = 0)
#   octx8                frozen, OBSERVED-CONTEXT K=8 -- eight exemplars
#                        drawn from O, showing x_j(t)
#   phist8               frozen, PREDICTION-HISTORY D=8 -- the agent's own
#                        last eight SERVED values
#   ehist8               frozen, EXPRESSED-HISTORY D=8 -- the agent's own
#                        last eight POST-FJ opinions. Defined here so the
#                        vocabulary is complete; NO key queues it.
#
# STRICT vs EXTENSION IS wu_context.is_extension()'s CALL, NOT THIS
# FILE'S. The classification has exactly one home
# (experiments/scripts/cluster_pipelines/wu_context.py) and the mirror
# below is asserted against it by tests/test_jiduan_pokec_infra.py.
# Its rule: a mechanism is an EXTENSION when it shows the model something
# Wu's platform cannot observe. The platform obviously has its own past
# OUTPUTS, so prediction_history is strict; it does NOT observe held-out
# agents' realised opinions, so expressed_history is the extension.
WU_ARM_COLS = {
    "b0": _wu_sft_arm("b0", 0.0),
    "b0p1": _wu_new_dose(0.1),
    "b0p5": _wu_sft_arm("b0p5", 0.5),
    "b1": _wu_sft_arm("b1", 1.0),
    "b10": _wu_new_dose(10.0),
    "frz": _wu_frozen_arm("none", 0, 0),
    "octx8": _wu_frozen_arm("observed_context", 8, 0),
    "phist8": _wu_frozen_arm("prediction_history", 0, 8),
    "ehist8": _wu_frozen_arm("expressed_history", 0, 8),
}
WU_EXTENSION_ARMS = ("ehist8",)
WU_STRICT_ARMS = tuple(a for a in WU_ARM_COLS if a not in WU_EXTENSION_ARMS)
WU_TRAINED_ARMS = ("b0", "b0p1", "b0p5", "b1", "b10")

ROW_WU = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
          "fj, 0.0, 0.0, 1.0, loop, 0.0, {peersrc}, {platsrc}, {ascale}, "
          "{bscale}, {inner}, {iclmode}, {iclk}, {icld}, {rtfrac}, "
          "{rtseed}, {rtval}, {uselora}, {fresh}, {ansk}, {gg}, "
          "{nrounds}, {basemodel}, {chatthink}, {mem}, {disk}, "
          "{pplbatch}")


def wu_row(model, arm, *, ca=1.0, cb=1.0, peer_src="dataset",
           plat_src="dataset", seed=0, rounds=WU_ROUNDS, inner=WU_INNER,
           route=None, smoke=False):
    a = WU_ARM_COLS[arm]
    m = FAM_MODELS[model]
    if route is None:
        frac, rseed, val = 0.0, 0, 0.0
    else:
        # the twins differ in ONE column: the treated run injects, the
        # control does not. Seed and value ride both rows identically so
        # the pair is legible as a pair and the cohort is recomputable
        # from either side.
        frac = WU_ROUTE_FRAC if route == "T" else WU_ROUTE_CONTROL_FRAC
        rseed, val = WU_ROUTE_SEED, WU_ROUTE_VALUE
    return ROW_WU.format(
        tag=wu_tag(model, arm, ca=ca, cb=cb, peer_src=peer_src,
                   plat_src=plat_src, seed=seed, rounds=rounds,
                   inner=inner, route=route, smoke=smoke),
        style=a["style"], beta=a["beta"], seed=seed,
        peersrc=peer_src, platsrc=plat_src, ascale=f"{ca:g}",
        bscale=f"{cb:g}", inner=inner, iclmode=a["iclmode"],
        iclk=a["iclk"], icld=a["icld"], rtfrac=f"{frac:g}", rtseed=rseed,
        rtval=f"{val:g}", uselora=a["uselora"], fresh=a["fresh"],
        ansk=a["ansk"], gg=a["gg"], nrounds=rounds,
        basemodel=m["base_model"], chatthink=m["chatthink"],
        mem=m["mem"], disk=m["disk"], pplbatch=m["pplbatch"])


# ---- the conceptual grids, and the arithmetic that turns them into jobs

def wu_smoke_rows():
    """One 3-round Qwen2.5 forward-KL cell at the exact heterogeneous
    dataset parameters: exercises the whole wu1-on-Pokec path (per-agent
    alpha complement, per-agent beta, observed passthrough, u^(1) save,
    train-on-O-only) before any 50-round job runs."""
    return [wu_row("qwen7b", "b1", rounds=WU_SMOKE_ROUNDS, smoke=True)]


def wu_prior_rows():
    """6 models x 2 arms, seed 0, EXACT heterogeneous dataset parameters
    (c_alpha = c_beta = 1), T = 50. 12 jobs."""
    return [wu_row(mo, ar) for mo in WU_MODELS for ar in WU_PRIOR_ARMS]


def wu_prior_seed_rows():
    """Seeds 42/43 x {Qwen2.5, OLMo-2} x {b0, b1}. 8 jobs."""
    return [wu_row(mo, ar, seed=sd)
            for sd in WU_SEEDS for mo in WU_SEED_MODELS
            for ar in WU_PRIOR_ARMS]


def wu_ladder_reused():
    """The ladder cells that ALREADY EXIST as prior-wave rows: lambda 0
    (b0) and lambda 1 (b1) for Qwen2.5 at seed 0. Returned as tags so the
    caller can assert they are byte-identical to the prior key's."""
    return [wu_tag(WU_LADDER_MODEL, ar) for lam, ar in WU_LADDER
            if ar in WU_PRIOR_ARMS]


def wu_ladder_rows():
    """Qwen2.5 only, lambda in {0, .1, .5, 1, 10} = 5 conceptual cells.
    lambda 0 and 1 are the prior wave's b0/b1 cells and are REUSED, so
    only 3 queue: 5 - 2 = 3."""
    return [wu_row(WU_LADDER_MODEL, ar) for lam, ar in WU_LADDER
            if ar not in WU_PRIOR_ARMS]


def wu_icl_reused():
    """{Qwen2.5, Mistral} x {b0, b1} = 4 prior-wave cells."""
    return [wu_tag(mo, ar) for mo in WU_ICL_MODELS for ar in WU_PRIOR_ARMS]


def wu_icl_rows():
    """{Qwen2.5, Mistral} x {b0, b1, phist8, octx8, frz} = 10 conceptual
    cells; the 4 b0/b1 cells are the prior wave's, so 10 - 4 = 6 queue."""
    return [wu_row(mo, ar) for mo in WU_ICL_MODELS for ar in WU_ICL_ARMS
            if ar not in WU_PRIOR_ARMS]


def wu_env_pairs():
    """The dose grid: (c_alpha in {0,.5,1} with c_beta=1) UNION
    (c_beta in {0,.5,1} with c_alpha=1). The centre (1,1) belongs to both
    axes and is ONE cell, not two."""
    pairs = [(ca, 1.0) for ca in WU_ENV_SCALES]
    for cb in WU_ENV_SCALES:
        if (1.0, cb) not in pairs:
            pairs.append((1.0, cb))
    return pairs


def wu_env_cells():
    """(arm, ca, cb) for the environment grid, with the two collapses
    applied:

    * THE CENTRE IS SHARED. (1,1) is the exact heterogeneous
      configuration, i.e. the prior wave's cell for b0/b1 and the ICL
      wave's cell for phist8. Counted once, queued never.
    * c_beta = 0 MAKES THE MODEL IRRELEVANT. beta_i = 0 for everyone, so
      x_init = x^innate for every round and the population trajectory is
      the same object whatever the platform serves. Queued ONCE under the
      canonical b0 arm rather than three times. The SERVED side still
      differs by arm -- that is why the analyzer reports nothing
      arm-specific from this cell beyond b0's own serving.
    """
    cells = []
    for ca, cb in wu_env_pairs():
        if cb == 0.0:
            cells.append(("b0", ca, cb))
        else:
            cells += [(ar, ca, cb) for ar in WU_ENV_ARMS]
    return cells


def wu_env_reused():
    """The centre column: Qwen2.5 x {b0, b1} from the prior key and
    Qwen2.5 x phist8 from the ICL key. 3 tags."""
    return [wu_tag(WU_ENV_MODEL, ar) for ar in WU_ENV_ARMS]


def wu_env_rows():
    """13 conceptual cells (see wu_env_cells), 3 reused at the centre,
    so 13 - 3 = 10 queue."""
    reused = set(wu_env_reused())
    rows = [wu_row(WU_ENV_MODEL, ar, ca=ca, cb=cb)
            for ar, ca, cb in wu_env_cells()]
    return [r for r in rows if r.split(",")[0] not in reused]


def wu_route_rows(seeds=(0,)):
    """PAIRED treatment/control twins: {b0, phist8, frz} x
    {c_alpha 0, c_alpha 1} x {treat, control} = 12 per seed.

    NOTHING IS REUSED HERE. A control twin is not the same run as a
    prior cell: it carries the routing cohort (same frac, same cohort
    seed) so the pair's masks can be compared, and a prior cell carries
    no mask at all. Reusing one would leave the comparison with nothing
    to check."""
    return [wu_row(WU_ROUTE_MODEL, ar, ca=ca, seed=sd, route=side)
            for sd in seeds for ar in WU_ROUTE_ARMS
            for ca in WU_ROUTE_CAS for side in ("T", "C")]


# FROZEN EXTRACTION -- NO REUSE WAS FOUND, SO EVERY MODEL EXTRACTS.
# The analyzer's primary held-out estimand is distance to the FROZEN
# model's prediction map on U, so one static vector per model is
# required before any cell can be scored. A scan of runs/pokec_gated_lm
# (1180 dirs, 2026-08-22) found 847 movielens runs, 311 pre-DATASET-field
# runs and ZERO frozen Pokec runs: the 162 Pokec-shaped dirs are all
# Qwen2.5 sft / sft_kl. Nothing qualifies, so nothing is reused -- an
# unverified reuse here would silently score five of six models against
# a vector from a different dataset or a different SKU.
WU_FROZEN_MODELS = list(WU_MODELS)
ROW_WU_FROZEN = ("{tag}, frozen, 0, 0, 1, replace, 1.0, fixed, ab, 0, "
                 "0.0, 0.5, loop, 0.0, 0, threshold, 0, -1, 0, 0, 0, 0, "
                 "1, {basemodel}, {chatthink}")


def wu_frozen_rows():
    """One 1-round zero-shot extraction per model. EPS_AI=0 under the
    strict-< gate means no agent is ever contacted and no opinion moves,
    so pred_raw[0] IS the frozen prediction map on all 2163 agents --
    including the 433 held-out ones the analysis needs. It is an
    EXTRACTION, not a Wu run: no FJ parameter is applied, and the
    frozen population replay is model-independent and runs on CPU."""
    return [ROW_WU_FROZEN.format(tag=wu_frozen_tag(mo),
                                 basemodel=FAM_MODELS[mo]["base_model"],
                                 chatthink=FAM_MODELS[mo]["chatthink"])
            for mo in WU_FROZEN_MODELS]


def wu_sub(key, rows, rounds=WU_ROUNDS):
    return WU_SUB_TEMPLATE.format(
        key=key, n_jobs=len(rows), gpu=WU_H100, bad=BAD_NODE_REQ,
        rounds=rounds, n=WU_N, nobs=WU_N_OBSERVED, nheld=WU_N_HELDOUT,
        inner=WU_INNER)


def wu_frozen_sub(key, rows):
    return WU_FROZEN_SUB_TEMPLATE.format(
        key=key, n_jobs=len(rows), gpu=WU_H100, bad=BAD_NODE_REQ,
        n=WU_N, nobs=WU_N_OBSERVED, nheld=WU_N_HELDOUT)


WU_SUB_TEMPLATE = """\
# HTCondor: JIDUAN WU / POKEC REPLICATION -- {n_jobs} jobs, {rounds}
# outer rounds each. GENERATED by gen_pofd_sweep.py from the WU block.
# Never edit by hand: rerun the script.
# Pokec LCC, N={n}: the FIRST {nobs} rows are the OBSERVED set O and the
# LAST {nheld} are the HELD-OUT set U. N_LABELED=TRAIN_CAP={nobs}, so the
# SFT batch is drawn from O alone and a held-out opinion can never reach
# the optimizer. FJ_OBSERVED_PASSTHROUGH=1 serves O its own current
# opinion instead of a model prediction; the model's job is U.
# THE OPERATOR: FJ_UPDATE_VERSION=wu1, PER-AGENT parameters.
#   served^(t)   = m_t on U, x^(t) on O
#   x_init^(t)_i = (1-beta_i) innate_i + beta_i served^(t)_i
#   u^(0)        = x_init^(t)
#   u^(l+1)_i    = (1-alpha_i) x_init_i + alpha_i (P u^(l))_i, l=0..K-1
# alpha_i = FJ_ALPHA_SCALE * hetero_peer_sus2163 (dataset mean .8909);
# beta_i  = FJ_BETA_SCALE  * hetero_platform_sus2163 (mean .8890);
# K = {inner}. W_PLAT is pinned to 1.0 so beta is never scaled twice, and
# NO scalar FJ_ALPHA / FJ_BETA is emitted: a scalar next to a dataset
# source is exactly the ambiguity this wave exists to remove.
# ALPHA IS SUSCEPTIBILITY, NOT STUBBORNNESS. The dataset file is named
# peer_sus but holds alpha; FJWorld.peer_sus is 1-alpha. Feeding it
# through unchanged would run susceptibility .109 -- near-opposite
# dynamics with every downstream number still well-formed. The checker
# replays all K inner steps per round with the per-agent complement.
# NO GATES, NO DEFFUANT PARAMETERS: FJ platform exposure and graph mixing
# are unconditional, so there are no _ea/_es tokens and no gate mode.
# THE INNER LOOP CONVERGES at alpha ~ .89, K={inner}. That is the FJ
# fixed point of ONE round's anchor and says NOTHING about the outer
# model-population loop; because convergence erases the inner loop's
# starting point, each round also saves u^(1), which is the only thing
# that can still prove u^(0) = x_init.
# Gate every pull with check_jiduan_pokec.py and hard-fail the analyzer
# until the whole conceptual grid is present.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=pokec POKEC_DIR=examples/pokec HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 POP_MODEL=fj FJ_UPDATE_VERSION=wu1 FJ_PEER_SOURCE=$(peersrc) FJ_PLATFORM_SOURCE=$(platsrc) FJ_ALPHA_SCALE=$(ascale) FJ_BETA_SCALE=$(bscale) FJ_INNER_STEPS=$(inner) FJ_OBSERVED_PASSTHROUGH=1 WU_ICL_MODE=$(iclmode) WU_ICL_K=$(iclk) WU_ICL_D=$(icld) ROUTING_TREAT_FRAC=$(rtfrac) ROUTING_TREAT_SEED=$(rtseed) ROUTING_TREAT_VALUE=$(rtval) PLATFORM_SUS_SCALE=1.0 KL_DIRECTION=forward SFT_EPOCHS=1 SFT_LR=5e-5 LORA_R=512 USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) SAVE_RAW_GEN=1 SAVE_WU_CTX_LOG=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP={nobs} N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 N_LABELED={nobs} HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{key}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, peersrc, platsrc, ascale, bscale, inner, iclmode, iclk, icld, rtfrac, rtseed, rtval, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


WU_FROZEN_SUB_TEMPLATE = """\
# HTCondor: JIDUAN WU / POKEC -- FROZEN PREDICTION-MAP EXTRACTION,
# {n_jobs} job(s). GENERATED by gen_pofd_sweep.py from the WU block.
# Never edit by hand.
# Produces the frozen (untrained, no-context) prediction map on all {n}
# Pokec agents -- including the {nheld} HELD-OUT ones, which is the
# reference every primary held-out estimand is measured against.
# NO REUSE IS CLAIMED. A scan of runs/pokec_gated_lm on 2026-08-22 found
# no frozen Pokec run for ANY of the six checkpoints, so all six extract
# here rather than borrowing a vector whose dataset, node order or GPU
# SKU could not be verified.
# ONE ROUND, EPS_AI=0 under the strict-< gate: no agent is ever
# contacted, no opinion moves, and pred_raw[0] is therefore the frozen
# map itself. It is an EXTRACTION, not a Wu run -- no FJ parameter is
# applied here and the frozen population replay is model-independent and
# runs locally on CPU.
# The EXACT H100 SKU is pinned: greedy decoding is only bit-reproducible
# within one GPU architecture, and this vector is a constant that every
# other cell is compared against.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = 160G
request_disk      = 60G
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=pokec POKEC_DIR=examples/pokec HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward INNATE_LAMBDA=0.0 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP={nobs} N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED={nobs} HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=0 SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{key}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink from experiments/condor/configs_pofd_{key}.txt
"""


AKL_KEY = "qwen_adapter_kl_probe"
AKL_SMOKE_KEY = "qwen_adapter_kl_probe_smoke"
AKL_H100 = QMECH_H100
AKL_MEM = "128G"
AKL_DISK = "40G"


def akl_rows(smoke=False):
    """One row: the probe's mode, spelled exactly as
    run_one_adapter_kl_probe.sh accepts it so the file and a hand-run of
    the executable cannot disagree. The adapter LIST is not encoded here
    -- the probe reads it from configs_pofd_qwen_sft_*_dose.txt, the
    files the dose jobs actually ran from, so the tag grammar keeps
    exactly one home."""
    return ["smoke" if smoke else "full"]


def akl_sub(key, smoke=False):
    return AKL_SUB_TEMPLATE.format(
        key=key, gpu=AKL_H100, bad=BAD_NODE_REQ, mem=AKL_MEM, disk=AKL_DISK,
        n_jobs=1, what=("2 adapters x 32 agents" if smoke
                        else "all 15 adapters x 723 agents"))


AKL_SUB_TEMPLATE = """\
# HTCondor: ADAPTER KL / SOFT-DECODE PROBE -- {n_jobs} job, {what}.
# GENERATED by gen_pofd_sweep.py from the AKL block. Never edit by hand:
# rerun the script.
# Scores every saved round-0 LoRA adapter from the SFT training-dose
# wave against the frozen base in DISTRIBUTION space, to separate two
# readings of the dose result's flat greedy distance to Qwen:
#   greedy flat AND soft flat   -> the entering model is not retained at
#                                  any dose; the implicit anchor fails.
#   greedy flat BUT soft rising -> the greedy departure is an ARGMAX
#                                  artifact and the anchor is internal.
# Reports KL(base||adapter) and KL(adapter||base) over the base's own
# greedy answer span, the soft-decoded value (expectation of the parsed
# number under the model, in a value map defined once by the base), the
# base's top-1/top-2 margin at the decision position, and the flip rate.
# KL RISES WITH DOSE ALMOST BY CONSTRUCTION -- it is the SHAPE against
# the greedy curve that carries the information, not KL alone.
# No training, no serving loop, no trajectory.pt: hence its own
# executable, not run_one_pokec_gated_idempotent.sh.
# Gate with check_adapter_kl_probe.py, which enforces the canonical
# frozen-Qwen hash, one shared base reference across adapters, support
# coverage, and adapter distinctness.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_adapter_kl_probe.sh
arguments         = $(mode)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 TF_BATCH=8 SMOKE_AGENTS=32 SMOKE_ADAPTERS=2"

output            = /home/gsmithline/perfsim/experiments/condor/logs/{key}.$(mode).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/{key}.$(mode).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/{key}.$(mode).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue mode from experiments/condor/configs_pofd_{key}.txt
"""


QSS_KEY = "qwen_subsample"
QSS_SMOKE_KEY = "qwen_subsample_smoke"
QSS_MODEL = "qwen7b"
QSS_ARM = "b0"                 # ordinary SFT, lambda = 0
QSS_W = 1.0
QSS_K = 1.0
QSS_ROUNDS = 100
QSS_SMOKE_ROUNDS = 3
QSS_N_AGENTS = 723
# exact counts, given rather than derived: round(.02*723)=14 etc, but the
# grid is the spec's, not a rounding rule
QSS_COUNTS = [14, 36, 72, 181, 362, 542]   # 723 = the reused QWU cell
QSS_FULL = 723
QSS_CM_N = 72                              # compute-matched arm
QSS_CM_REPEAT = 723
QSS_H100 = QWU_H100
# the completed full-data cell this wave hangs off
QSS_REUSED_TAG = "pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100"

ROW_QSS = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
           "ab, {es}, 0.0, {wplat}, loop, 0.0, {lam}, {samplen}, "
           "{repeatto}, {iclk}, {snap}, {uselora}, {fresh}, {ansk}, "
           "{gg}, {nrounds}, {basemodel}, {chatthink}, {mem}, {disk}, "
           "{pplbatch}")


def qss_tag(count, repeat_to=0, rounds=QSS_ROUNDS, smoke=False):
    """The observation count rides an _n token. The compute-matched cell
    spells its tiling explicitly (_n72rep723_) so it can never be
    mistaken for the plain 72-agent arm."""
    tok = f"n{count}" + (f"rep{repeat_to}" if repeat_to else "")
    sm = "smoke" if smoke else ""
    return (f"pofdqss_{QSS_MODEL}_{QSS_ARM}_eaopen_w{_num(QSS_W)}"
            f"_l{_num(QSS_K)}_esopen_{tok}_s0_r{rounds}{sm}")


def qss_row(count, repeat_to=0, rounds=QSS_ROUNDS, smoke=False):
    a = REACH_ARM_COLS[QSS_ARM]
    m = FAM_MODELS[QSS_MODEL]
    return ROW_QSS.format(
        tag=qss_tag(count, repeat_to, rounds, smoke), style=a["style"],
        beta=a["beta"], seed=0, es=f"{QWU_EPS_SOCIAL:g}",
        wplat=f"{QSS_W:g}", lam=f"{QSS_K:g}", samplen=count,
        repeatto=repeat_to, iclk=a["iclk"], snap=a["snap"],
        uselora=a["uselora"], fresh=a["fresh"], ansk=a["ansk"],
        gg=a["gg"], nrounds=rounds, basemodel=m["base_model"],
        chatthink=m["chatthink"], mem=m["mem"], disk=m["disk"],
        pplbatch=m["pplbatch"])


def qss_rows():
    """Six observation arms + one compute-matched control = 7 jobs.
    The 100% arm is the REUSED QWU cell and is deliberately absent."""
    return ([qss_row(c) for c in QSS_COUNTS]
            + [qss_row(QSS_CM_N, QSS_CM_REPEAT)])


def qss_smoke_rows():
    """ONE 3-round cell exercising the new sampling path end to end."""
    return [qss_row(QSS_CM_N, rounds=QSS_SMOKE_ROUNDS, smoke=True)]


def qss_sub(smoke=False):
    key = QSS_SMOKE_KEY if smoke else QSS_KEY
    rows = qss_smoke_rows() if smoke else qss_rows()
    return QSS_SUB_TEMPLATE.format(
        key=key, n_jobs=len(rows), gpu=QSS_H100, bad=BAD_NODE_REQ,
        rounds=QSS_SMOKE_ROUNDS if smoke else QSS_ROUNDS,
        kind=("SMOKE (3 rounds, NOT production)" if smoke
              else "PRODUCTION (100 rounds)"))


QSS_SUB_TEMPLATE = """\
# HTCondor: QWEN2.5 OBSERVATION-RATE SUBSAMPLING -- {kind}, {n_jobs}
# jobs. GENERATED by gen_pofd_sweep.py from the QSS block. Never edit by
# hand: rerun the script.
# Tests whether ordinary SFT leans MORE on the pretrained model when it
# observes less of the population. The completed Wu-boundary b0 cell,
# unchanged in every respect -- Qwen2.5-7B-Instruct, ordinary SFT
# (lambda=0), k=1, W=1, BOTH gates genuinely all_open, fresh LoRA r512
# every round, {rounds} rounds, seed 0, movielens Action 723 agents,
# matched twin, greedy eval-mode serving -- except how many agents'
# labels reach the optimizer: 14/36/72/181/362/542
# (2/5/10/25/50/75%).
# The 100% arm is NOT rerun: it IS the completed
# pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100.
# SERVING IS UNTOUCHED: all 723 agents are served every round in every
# arm. Only the SFT batch is cut, so the loop the population experiences
# is the same and the arms differ purely in observation.
# NEW KNOB, opt-in. N_LABELED takes a fixed PREFIX (same people every
# round -- a different question), and TRAIN_CAP is applied only for t>0,
# so round 0 would be unsubsampled. SFT_SAMPLE_N is absent by default,
# leaving every archived run byte-identical.
# NESTED: one permutation of all 723 per round from a dedicated stream
# seeded (SFT_SAMPLE_SEED + round), independent of sample size, prefix
# taken -- so the 14 sit inside the 36 sit inside the 72. "Different
# people" cannot explain a difference between arms.
# COMPUTE-MATCHED CONTROL: the 10% arm also runs with
# SFT_SAMPLE_REPEAT_TO=723 -- the same 72 agents tiled to exactly 723
# rows, hence 181 optimizer steps, identical compute to the full-data
# arm on 72 distinct agents. That separates limited unique data from
# merely taking fewer gradient steps.
# Gate every pull with check_pofd_sanity (QSS section: every subset and
# label reconstructed exactly from (seed, round), uniqueness, nesting,
# and n_train equal to the requested count in ALL rounds).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 AI_GATE_MODE=all_open PEER_GATE_MODE=all_open EPS_AI=1 INNATE_LAMBDA=$(lam) SFT_SAMPLE_N=$(samplen) SFT_SAMPLE_REPEAT_TO=$(repeatto) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{key}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, lam, samplen, repeatto, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


QWU_ICL_KEY = "qwen_wu_limit_icl"
QWU_ICL_ARM = "d8"
QWU_ICL_DAYS = 8


def qwu_icl_rows():
    return [qwu_row(QWU_ICL_ARM, w) for w in QWU_WS]


def qwu_icl_sub():
    return QWU_SUB_TEMPLATE.format(
        key=QWU_ICL_KEY, n_jobs=len(qwu_icl_rows()), gpu=QWU_H100,
        bad=BAD_NODE_REQ, rounds=QWU_ROUNDS, icldays=QWU_ICL_DAYS,
        kind=(f"PERSONAL-HISTORY ICL (D={QWU_ICL_DAYS}, K=0), "
              f"{QWU_ROUNDS} rounds"))


def qwu_smoke_rows():
    """ONE 3-round lambda=1, W=1 cell -- the hardest corner of the new
    path (regularized training AND both gates open)."""
    return [qwu_row("b1", 1.0, rounds=QWU_SMOKE_ROUNDS, smoke=True)]


def qwu_sub(smoke=False):
    key = QWU_SMOKE_KEY if smoke else QWU_KEY
    rows = qwu_smoke_rows() if smoke else qwu_rows()
    return QWU_SUB_TEMPLATE.format(
        key=key, n_jobs=len(rows), gpu=QWU_H100, bad=BAD_NODE_REQ,
        icldays=0,
        rounds=QWU_SMOKE_ROUNDS if smoke else QWU_ROUNDS,
        kind=("SMOKE (3 rounds, NOT production)" if smoke
              else "PRODUCTION (100 rounds)"))


QWU_SUB_TEMPLATE = """\
# HTCondor: QWEN2.5 AT THE WU CONSENSUS BOUNDARY -- {kind}, {n_jobs}
# jobs. GENERATED by gen_pofd_sweep.py from the QWU block. Never edit
# by hand: rerun the script.
# k = 1, W rides the queue, BOTH gates GENUINELY OPEN, {rounds} rounds,
# seed 0, movielens Action 723 agents, Qwen/Qwen2.5-7B-Instruct, fresh
# LoRA r512 every round, same optimizer / rank / training surface /
# reference model as the paper, matched twin (WITH_TWIN=1), one peer
# sweep, gamma = 0, greedy serving, SAVE_RAW_GEN=1.
# WHY MODES AND NOT THE NUMBER 1. Both gates are STRICT inequalities,
# so eps = 1 does NOT open them: an agent at 0 served 1, or a peer pair
# at (0, 1), sits at distance exactly 1 and is still REJECTED. Spelling
# "open" as 1 would silently drop exactly the extreme pairs the
# consensus limit is about. AI_GATE_MODE=all_open is the established
# 2026-08-13 mode; PEER_GATE_MODE=all_open is NEW (2026-08-20,
# gp.peer_gate), defaults to "threshold", and is applied AFTER pair
# selection so every archived run stays byte-identical and consumes
# identical RNG. Tags spell _eaopen_/_esopen_ and the checker rejects
# any numeric-threshold job wearing an open tag.
# EPS=0.2 is inert for acceptance under all_open but is set anyway:
# eps_social=0 is how "no peer step" is spelled everywhere else, and
# the runner refuses that combination rather than let one run mean two
# things.
# THEORY. beta_eff = 1 - (1-W)k, so k=1 alone is NOT a consensus limit
# (at W=.5 it gives beta_eff=.5); the boundary is k=1 AND W=1, where
# the pre-peer population equals the served vector. The CPU oracle must
# reach consensus there and check_perfect_predictor enforces it. These
# Qwen arms are NOT required to -- whether practical retraining
# converges is the phenomenon being measured.
# Gate every pull with check_pofd_sanity (QWU section: both gate modes
# genuinely open, no numeric threshold masquerading as open, no
# rejected peer pair, the declared horizon, finite SFT losses, adapter
# re-served in eval mode).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 AI_GATE_MODE=all_open PEER_GATE_MODE=all_open EPS_AI=1 INNATE_LAMBDA=$(lam) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS={icldays} ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{key}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, lam, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


# qwen_kl_direction[_smoke] (2026-08-22). FORWARD vs REVERSE KL on the
# QWU surface, 10 rounds. The question is whether raising lambda moves
# the population toward frozen Qwen while PRESERVING heterogeneity, and
# whether reverse KL shows that more clearly without collapsing the
# served map. A mean shift alone does not answer it, so the analyzer
# reports SD, distinct served values and largest-mode share beside the
# mean, and the figure carries mean and SD as separate rows.
#
# THESE ARE NOT EQUILIBRIA. Ten rounds on this surface is a short
# transient -- the QWU cells were still moving at round 10 of 100. Every
# artifact, axis label and table caption says "10-round post-peer
# dynamics". Only cells that look informative earn 30-50 rounds and
# replication seeds, as a later decision.
#
# NAMING, because two different quantities are called beta and gamma in
# this project and mixing them up has already cost one wave:
#   Celestine's beta  = platform susceptibility = W_PLAT = KD_WS
#   Celestine's gamma = the innate anchor coefficient k = INNATE_LAMBDA
#                       = KD_K = 1  (NOT the homophily gamma, which is
#                       the positional column and stays 0 as always)
#   lambda            = the KL weight = kl_beta = the {beta} ROW column
# The tag spells the KL weight as _lam<x>_ and never as _b<x>_, so it
# cannot be read as the w_plat token.
#
# DIRECTION IS A QUEUE COLUMN, NOT A PINNED ENV VALUE. Every other wave
# hard-codes KL_DIRECTION in the sub env; this one compares the two, so
# it rides the queue as $(kldir) and the tag records it. The checker
# rejects any run whose recorded kl_direction disagrees with its tag.
# Convention, verified against perfsim/learners/lm/kl_sft.py:
#   forward = KL(pi_ref || pi_theta)   mass-covering
#   reverse = KL(pi_theta || pi_ref)   mode-seeking, the RLHF penalty
# In _anchor_divergence_per_token, logp is the POLICY and logq the
# frozen reference, so "forward" computes sum q (log q - log p) --
# KL(ref || policy) -- which is the mapping above.
#
# WHAT IS QUEUED AND WHAT IS REUSED. Ten GPU jobs, exactly:
#   forward lambda in {.1, 10} x W in {.5, 1}   4
#   reverse lambda in {.1, 1, 10} x W in {.5, 1} 6
# Four more conceptual arms are NOT queued because they already exist
# or need no GPU:
#   ordinary SFT lambda=0   pofdqwu_qwen7b_b0_* first 10 rounds
#   forward lambda=1        pofdqwu_qwen7b_b1_* first 10 rounds
#   perfect prediction      sim_perfect_predictor.py, CPU
#   frozen Qwen             replay_frozen_offline.py, CPU, from the
#                           canonical H100 vector
# The two reused GPU cells were audited field-by-field against this
# block (audit_kl_direction_reuse.py): same checkpoint, seed, dataset,
# 723 agents, k, both gate MODES, eps, one peer sweep, fresh LoRA r512,
# lr, epochs, epoch size, max_steps, train cap, icl_k/days, H100 SKU,
# and kl_direction=forward with an EMPTY kl_ref_adapter (the reference
# is the raw base model, which is what the new cells anchor to as well).
# Their population_update marker reads the v1 string, which is INERT
# here: both gates are all_open and gp.ai_gate returns before it ever
# consults the anchor, so v1 and v2 are the same operator on this
# surface. Their serve_eval_mode field is absent because that field
# postdates them by a day -- but the smoke2 TAG was created by the same
# commit as the eval-mode fix (9ee5136, 2026-08-20 16:31) and its run
# directory exists from 16:36, so the cluster tree was already post-fix
# when these four launched at 17:07. They decoded with dropout off.
KD_KEY = "qwen_kl_direction"
KD_SMOKE_KEY = "qwen_kl_direction_smoke"
KD_MODEL = "qwen7b"
KD_WS = [0.5, 1.0]              # Celestine's beta = W_PLAT
KD_K = 1.0                      # Celestine's gamma = INNATE_LAMBDA
KD_EPS_SOCIAL = QWU_EPS_SOCIAL  # 0.2, inert under all_open, set anyway
KD_ROUNDS = 10
KD_SMOKE_ROUNDS = 3
KD_SEED = 0
KD_H100 = QWU_H100
# (direction token, KL weight lambda). forward lambda=1 is REUSED from
# the QWU b1 cells and is deliberately absent here.
KD_CELLS = [("fwd", 0.1), ("fwd", 10.0),
            ("rev", 0.1), ("rev", 1.0), ("rev", 10.0)]
KD_DIR_ENV = {"fwd": "forward", "rev": "reverse"}
# arm label -> the archived run whose first KD_ROUNDS rounds stand in
KD_REUSED = {
    ("sft0", 0.5): "pofdqwu_qwen7b_b0_eaopen_w0p5_l1_esopen_s0_r100",
    ("sft0", 1.0): "pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100",
    ("fwdlam1", 0.5): "pofdqwu_qwen7b_b1_eaopen_w0p5_l1_esopen_s0_r100",
    ("fwdlam1", 1.0): "pofdqwu_qwen7b_b1_eaopen_w1_l1_esopen_s0_r100",
}
# the frozen vector b for the lambda -> infinity endpoint (CPU replay).
# Spelled literally rather than aliased to RR_REF_RUN: that constant is
# defined ~1800 lines BELOW this block, and the ref_replay wave is free
# to repoint it without silently repointing this one.
KD_FROZEN_REF_RUN = "pofdqmech_qwen7b_k0_ea1_w0p5_l1_es0p05_s0"
KD_FROZEN_REF_SHA = QMECH_CANONICAL_PRED_SHA

# kldir rides the queue right after lam; everything else copies ROW_QWU
ROW_KD = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
          "ab, {es}, 0.0, {wplat}, loop, 0.0, {lam}, {kldir}, {iclk}, "
          "{snap}, {uselora}, {fresh}, {ansk}, {gg}, {nrounds}, "
          "{basemodel}, {chatthink}, {mem}, {disk}, {pplbatch}")


def kd_lam_tok(lam):
    """.1 -> 'lam0p1', 1 -> 'lam1', 10 -> 'lam10'. Spelled lam<x>, never
    b<x>: b<x> is this project's forward-KL ARM token and would collide
    with the w_plat reading of the tag."""
    return f"lam{_num(lam)}"


def kd_tag(direction, lam, w, seed=KD_SEED, rounds=KD_ROUNDS, smoke=False):
    """pofdkd_qwen7b_revlam0p1_eaopen_w0p5_l1_esopen_s0_r10.

    The direction token is REQUIRED and adjacent to the weight, so no
    tag can be read without committing to a direction. The smoke wears
    its own PREFIX (pofdkdsmk_) so check_kl_direction can enforce the
    10-round horizon on production without a truncated run sneaking
    through under a trailing token."""
    if direction not in KD_DIR_ENV:
        raise ValueError(f"direction must be fwd/rev; got {direction!r}")
    pre = "pofdkdsmk" if smoke else "pofdkd"
    return (f"{pre}_{KD_MODEL}_{direction}{kd_lam_tok(lam)}"
            f"_eaopen_w{_num(w)}_l{_num(KD_K)}_esopen_s{seed}_r{rounds}")


def kd_row(direction, lam, w, seed=KD_SEED, rounds=KD_ROUNDS, smoke=False):
    a = REACH_ARM_COLS["b1"]        # the sft_kl envelope; beta overridden
    m = FAM_MODELS[KD_MODEL]
    return ROW_KD.format(
        tag=kd_tag(direction, lam, w, seed, rounds, smoke),
        style="sft_kl", beta=f"{lam:g}", seed=seed,
        es=f"{KD_EPS_SOCIAL:g}", wplat=f"{w:g}", lam=f"{KD_K:g}",
        kldir=KD_DIR_ENV[direction], iclk=a["iclk"], snap=a["snap"],
        uselora=a["uselora"], fresh=a["fresh"], ansk=a["ansk"],
        gg=a["gg"], nrounds=rounds, basemodel=m["base_model"],
        chatthink=m["chatthink"], mem=m["mem"], disk=m["disk"],
        pplbatch=m["pplbatch"])


def kd_rows():
    return [kd_row(d, lam, w) for w in KD_WS for d, lam in KD_CELLS]


def kd_smoke_rows():
    """One 3-round reverse lambda=1, W=1 cell. Reverse is the NEW path in
    this wave and W=1 is the boundary, so this is its hardest corner."""
    return [kd_row("rev", 1.0, 1.0, rounds=KD_SMOKE_ROUNDS, smoke=True)]


def kd_sub(smoke=False):
    rows = kd_smoke_rows() if smoke else kd_rows()
    key = KD_SMOKE_KEY if smoke else KD_KEY
    return KD_SUB_TEMPLATE.format(
        key=key, n_jobs=len(rows), gpu=KD_H100, bad=BAD_NODE_REQ,
        rounds=(KD_SMOKE_ROUNDS if smoke else KD_ROUNDS),
        kind=("3-ROUND REVERSE-KL SMOKE" if smoke
              else "FORWARD vs REVERSE KL, 10-ROUND POST-PEER DYNAMICS"))


KD_SUB_TEMPLATE = """\
# HTCondor: QWEN2.5 FORWARD vs REVERSE KL -- {kind}, {n_jobs} jobs.
# GENERATED by gen_pofd_sweep.py from the KD block. Never edit by hand:
# rerun the script.
# The QWU surface exactly: k = 1, W rides the queue, BOTH gates
# GENUINELY OPEN, {rounds} rounds, seed 0, movielens Action 723 agents,
# Qwen/Qwen2.5-7B-Instruct, fresh LoRA r512 every round, one peer sweep,
# homophily gamma = 0, greedy serving in eval mode, SAVE_RAW_GEN=1.
# KL_DIRECTION RIDES THE QUEUE. Every other wave pins it in this env;
# this one is the comparison, so it is $(kldir) and the tag carries it.
#   forward = KL(pi_ref || pi_theta)    reverse = KL(pi_theta || pi_ref)
# check_kl_direction rejects any run whose recorded direction disagrees
# with its tag, and refuses a wave that is missing either direction.
# THESE ARE 10-ROUND POST-PEER DYNAMICS, NOT EQUILIBRIA.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 AI_GATE_MODE=all_open PEER_GATE_MODE=all_open EPS_AI=1 INNATE_LAMBDA=$(lam) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=$(kldir) WITH_TWIN=1 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{key}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, lam, kldir, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


# --- Section 3: the RETENTION table (S3) -----------------------------------
# section3_retention[_smoke] (2026-08-22). The Section-3 headline object:
# how much of the PRETRAINED PRIOR survives 100 rounds of closed-loop
# retraining, as a joint function of
#   (a) the KL anchor DOSE      lambda in {0, .1, .5, 1, 2, 4, 8}
#   (b) the KL DIRECTION        forward (canonical) vs reverse (RLHF)
#   (c) the ENVIRONMENT         (W_PLAT, k) -- three of them, below
# on TWO checkpoints (Qwen2.5-7B-Instruct and Qwen3-8B, thinking OFF).
#
# NAMING, because three different quantities in this project are called
# beta and two are called gamma, and mixing them up has cost a wave:
#   Celestine's beta = platform susceptibility = W_PLAT   = the {wplat}
#                      queue column (0.5 or 1)
#   Celestine's gamma = the innate anchor coefficient k
#                      = INNATE_LAMBDA = the {lam} queue column (1 or .2)
#                      -- NOT the homophily gamma, which is the
#                      positional column and stays 0.0 as always
#   lambda            = the KL weight = kl_beta = the {beta} ROW column
# The tag spells the KL weight only inside the ARM token (_fwdlam2_,
# _revlam8_) and never as a bare _b<x>_, so it cannot be read as W_PLAT.
#
# THE THREE ENVIRONMENTS (exactly three, never four):
#   env1 "main"  W = 0.5, k = 1     the canonical Section-3 surface
#   env2 "wu"    W = 1,   k = 1     the exact Wu-style limiting boundary
#   env3 "mem"   W = 0.5, k = 0.2   weak innate anchor -- the "memory"
#                                   environment
# (W = 1, k = 0.2) IS DELIBERATELY ABSENT. At beta = W_PLAT = 1 the
# post-AI opinion is the served value outright and the innate term drops
# out of the update algebraically, so a k = 0.2 cell there is the SAME
# operator as the k = 1 cell -- it would be a duplicate run wearing a
# different tag. s3_cells() cannot emit it (S3_ENVS has three entries)
# and the registration asserts no row carries w=1 with k=0.2.
#
# REVERSE KL LIVES ONLY IN env1 AND env2. Reverse is the comparison arm,
# not a dose ladder: two points ({1, 8}) on the two k = 1 environments.
# The weak-anchor environment gets the forward ladder only.
#
# DIRECTION IS A QUEUE COLUMN, NOT A PINNED ENV VALUE -- and THIS is the
# bit that bit an earlier wave. This key MIXES forward and reverse rows,
# so KL_DIRECTION cannot be hard-coded in the sub environment the way
# every single-direction wave does it. It rides the queue as $(kldir),
# exactly as the qwen_kl_direction (KD) block does, which means BOTH of
# these must be true at once:
#   1. the environment string contains  KL_DIRECTION=$(kldir)
#   2. the queue line DECLARES a kldir column, in the slot the row
#      schema puts it in (right after lam)
# If (1) holds and (2) does not, Condor expands $(kldir) to the EMPTY
# STRING, the runner's _env_or("KL_DIRECTION", "reverse") never sees it
# and silently falls back to REVERSE -- on the forward cells too, which
# would look like a perfectly clean wave and quietly answer the wrong
# question. Both are asserted in the registration block below.
# Convention, verified against perfsim/learners/lm/kl_sft.py:
#   forward = KL(pi_ref || pi_theta)   mass-covering (canonical here)
#   reverse = KL(pi_theta || pi_ref)   mode-seeking, the RLHF penalty
#
# THE DIRECTION-NEUTRAL sft ARM, and what goes in its kldir column.
# lambda = 0 is ordinary SFT: TRAINING_STYLE=sft, KL_BETA=0, and the
# runner never constructs a divergence term at all, so kl_direction is
# INERT for it. But the column is not optional -- every row of a key
# must have the same arity, and an empty field would re-open exactly the
# empty-string failure above. So the sft rows carry the literal
# "forward" as an INERT PLACEHOLDER (S3_SFT_KLDIR), chosen because it is
# this project's canonical direction and therefore the least surprising
# value to find in an archived config.json for a run that used neither.
# The TAG, by contrast, carries NO direction token whatsoever: the arm
# token is the bare "sft", so nothing downstream can read a scientific
# direction claim off a cell that has none. The registration asserts
# both halves: sft rows are style "sft" with kl_beta "0", and their tags
# contain neither "fwd" nor "rev".
#
# TAG GRAMMAR
#   pofds3_{model}_{arm}_eaopen_w{W}_k{k}_esopen_anch2_s0_r100
#   arm in {sft, fwdlam0p1, fwdlam0p5, fwdlam1, fwdlam2, fwdlam4,
#           fwdlam8, revlam1, revlam8}
# Two grammar notes:
#  * _eaopen_/_esopen_ spell the GENUINELY open gates (AI_GATE_MODE and
#    PEER_GATE_MODE = all_open), never _ea1_/_es1_: both gates are
#    strict inequalities, so the numeric value 1 rejects a distance-1
#    pair and would silently drop the extreme pairs this table is about.
#  * the anchor k rides a _k{k}_ token here, whereas QWU/KD/QMECH spell
#    it _l{k}_ and reserve _k<n>_ for ICL depth. That is the pinned
#    Section-3 grammar and it is unambiguous WITHIN this family because
#    every pofds3_ row has ICL_K = 0 (no in-context arm exists here at
#    all), but it is a deliberate departure from the older families and
#    any cross-family tag parser must key off the pofds3_ prefix first.
#
# THE "anch2" OPERATOR-PROVENANCE TOKEN (user decision, 2026-08-22).
# The {optok} slot carries the fixed token "anch2", and it is TRUE BY
# CONSTRUCTION: it names exactly the round operator these runs record.
#   tag token  anch2
#   config     population_update == "nested_ai_anchored_then_social_v2"
# That marker comes from _POP_UPDATE_MARKER in run_pokec_gated_lm.py,
# keyed on AI_GATE_REFERENCE, whose default has been "anchor" since
# 2026-08-22. This block deliberately does NOT set AI_GATE_REFERENCE:
# the default already produces the v2 marker, so pinning it in the sub
# env would change the environment string for no behavioural gain.
# An earlier draft of this wave used "hg2", short for a
# "nested_ai_then_social_hgate_v2" operator. THAT STRING EXISTS NOWHERE
# IN THIS REPO -- the runner can emit only "nested_ai_then_social_v1"
# (AI_GATE_REFERENCE=x0, every run archived before 2026-08-22) or
# "nested_ai_anchored_then_social_v2" (anchor). "anch2" is therefore the
# only token that can be checked against a config.json, which is the
# whole point of putting it in the tag.
# WHAT THE TOKEN DOES AND DOES NOT MEAN. It records PROVENANCE, not a
# behavioural difference on this surface: with AI_GATE_MODE=all_open the
# v1 and v2 operators are NUMERICALLY IDENTICAL. gp.ai_gate returns an
# all-ones mask at _gated_pop.py:205-206 and never reaches line 209,
# where the gate reference (x0 vs the anchored x') is the only thing
# that differs between the two semantics. So no pofds3_ result depends
# on which marker it carries -- but every pofds3_ result can be SHOWN to
# carry the v2 one, and archived runs cannot be mistaken for these
# cells: none of them carry an operator token in the tag at all.
#
# WHAT IS QUEUED AND WHAT IS REUSED. 50 conceptual cells:
#   2 models x 3 envs x 7 arms (sft + 6 forward)          = 42
#   2 models x 2 envs (k=1) x 2 reverse arms              =  8
# Four of the 50 are ALREADY ON DISK as archived Qwen2.5 QWU cells --
# the same 100-round, seed-0, both-gates-open, r512-fresh-LoRA,
# 723-agent movielens Action surface at k = 1 -- so they are declared in
# S3_REUSED and NEVER queued:
#   qwen7b sft      W=0.5  ->  pofdqwu_qwen7b_b0_eaopen_w0p5_l1_...
#   qwen7b sft      W=1    ->  pofdqwu_qwen7b_b0_eaopen_w1_l1_...
#   qwen7b fwdlam1  W=0.5  ->  pofdqwu_qwen7b_b1_eaopen_w0p5_l1_...
#   qwen7b fwdlam1  W=1    ->  pofdqwu_qwen7b_b1_eaopen_w1_l1_...
# => 50 - 4 = 46 NEW GPU production jobs, asserted explicitly below.
# The registration also asserts that each declared reuse tag is a tag
# some OTHER generated key actually produces, so a rename in the QWU
# block can never orphan a Section-3 cell silently.
# NOTE: field-level reuse eligibility (the audit that these four really
# are the same experiment) is Agent A's manifest, not this generator's
# claim. What this file guarantees is only: the tags exist, and they are
# never double-queued.
S3_KEY = "section3_retention"
S3_SMOKE_KEY = "section3_retention_smoke"
S3_MODELS = ["qwen7b", "qwen3_8b"]      # keys into FAM_MODELS
# (W_PLAT = Celestine's beta, INNATE_LAMBDA = k = Celestine's gamma)
S3_ENV_MAIN = (0.5, 1.0)
S3_ENV_WU = (1.0, 1.0)
S3_ENV_MEM = (0.5, 0.2)
S3_ENVS = [S3_ENV_MAIN, S3_ENV_WU, S3_ENV_MEM]
S3_ENV_NAMES = {S3_ENV_MAIN: "main", S3_ENV_WU: "wu", S3_ENV_MEM: "mem"}
S3_FWD_LAMS = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
S3_REV_LAMS = [1.0, 8.0]
S3_REV_ENVS = [S3_ENV_MAIN, S3_ENV_WU]  # reverse never runs in env3
S3_EPS_SOCIAL = 0.2       # inert under all_open; the runner refuses 0
S3_ROUNDS = 100
S3_SMOKE_ROUNDS = 3
S3_SEED = 0
S3_H100 = QMECH_H100
# the operator-provenance token: "anch2" <-> config population_update
# "nested_ai_anchored_then_social_v2". See the provenance note above.
S3_OP_TOKEN = "anch2"
S3_DIR_ENV = {"fwd": "forward", "rev": "reverse"}
# the inert placeholder the direction-NEUTRAL sft arm puts in its kldir
# column (its tag carries no direction token; see the note above)
S3_SFT_KLDIR = "forward"
S3_N_CONCEPTUAL = 50      # 2 x 3 x 7 + 2 x 2 x 2
S3_N_NEW = 46             # ... minus the 4 archived QWU cells
# per-row queue payload, PINNED here rather than borrowed from
# REACH_ARM_COLS so a later edit to that shared registry cannot move the
# Section-3 surface out from under an in-flight wave. The values are the
# b0/b1 envelope, which is what the four reused QWU cells ran with:
S3_ICL_K = 0              # no in-context arm anywhere in this wave
S3_ICL_SNAP = -1
S3_USE_LORA = 1
S3_FRESH = 1              # fresh LoRA every round
S3_ANS_K = 16
S3_GG = 0                 # gender-gap telemetry off, as in QWU/KD
# (model, arm, W_PLAT, k) -> the archived run that already IS this cell.
# Declared, asserted-to-exist, and never queued.
S3_REUSED = {
    ("qwen7b", "sft", 0.5, 1.0):
        "pofdqwu_qwen7b_b0_eaopen_w0p5_l1_esopen_s0_r100",
    ("qwen7b", "sft", 1.0, 1.0):
        "pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100",
    ("qwen7b", "fwdlam1", 0.5, 1.0):
        "pofdqwu_qwen7b_b1_eaopen_w0p5_l1_esopen_s0_r100",
    ("qwen7b", "fwdlam1", 1.0, 1.0):
        "pofdqwu_qwen7b_b1_eaopen_w1_l1_esopen_s0_r100",
}

# kldir rides the queue right after lam -- the same 28-column schema the
# KD block uses. Columns 0-13 are the executable's POSITIONAL arguments
# and are untouched by the extra column; everything from lam onward
# reaches the runner through the sub's environment string.
ROW_S3 = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
          "ab, {es}, 0.0, {wplat}, loop, 0.0, {lam}, {kldir}, {iclk}, "
          "{snap}, {uselora}, {fresh}, {ansk}, {gg}, {nrounds}, "
          "{basemodel}, {chatthink}, {mem}, {disk}, {pplbatch}")


def s3_arm_token(direction, lam):
    """('fwd', 0.1) -> 'fwdlam0p1'; ('rev', 8) -> 'revlam8'.

    The direction is FUSED to the weight in a single token so no trained
    arm can be named without committing to a direction. The
    direction-neutral lambda = 0 arm is spelled "sft" and never passes
    through here."""
    if direction not in S3_DIR_ENV:
        raise ValueError(f"direction must be fwd/rev; got {direction!r}")
    return f"{direction}lam{_num(lam)}"


def s3_arms(w, k):
    """The arms of one environment, as (arm_token, style, kl_beta, kldir).

    Ordinary SFT first, then the forward dose ladder, then -- only in the
    two k = 1 environments -- the two reverse comparison points."""
    out = [("sft", "sft", "0", S3_SFT_KLDIR)]
    out += [(s3_arm_token("fwd", lam), "sft_kl", f"{lam:g}", "forward")
            for lam in S3_FWD_LAMS]
    if (w, k) in S3_REV_ENVS:
        out += [(s3_arm_token("rev", lam), "sft_kl", f"{lam:g}", "reverse")
                for lam in S3_REV_LAMS]
    return out


def s3_tag(model, arm, w, k, seed=S3_SEED, rounds=S3_ROUNDS, smoke=False):
    """pofds3_qwen7b_fwdlam2_eaopen_w0p5_k1_esopen_anch2_s0_r100.

    The horizon is in the tag because a 100-round and a 3-round cell of
    the same condition are different objects, and the smoke wears its own
    PREFIX (pofds3smk_) rather than a trailing token so a truncated run
    can never be mistaken for -- or satisfy -- a production cell."""
    pre = "pofds3smk" if smoke else "pofds3"
    return (f"{pre}_{model}_{arm}_eaopen_w{_num(w)}_k{_num(k)}"
            f"_esopen_{S3_OP_TOKEN}_s{seed}_r{rounds}")


def s3_row(model, arm, style, kl_beta, kldir, w, k,
           seed=S3_SEED, rounds=S3_ROUNDS, smoke=False):
    m = FAM_MODELS[model]       # base_model / chatthink / mem / disk / ppl
    return ROW_S3.format(
        tag=s3_tag(model, arm, w, k, seed, rounds, smoke),
        style=style, beta=kl_beta, seed=seed, es=f"{S3_EPS_SOCIAL:g}",
        wplat=f"{w:g}", lam=f"{k:g}", kldir=kldir,
        iclk=S3_ICL_K, snap=S3_ICL_SNAP, uselora=S3_USE_LORA,
        fresh=S3_FRESH, ansk=S3_ANS_K, gg=S3_GG, nrounds=rounds,
        basemodel=m["base_model"], chatthink=m["chatthink"],
        mem=m["mem"], disk=m["disk"], pplbatch=m["pplbatch"])


def s3_cells():
    """All 50 CONCEPTUAL cells, reuse included, in a stable order.

    Returned as (model, arm, style, kl_beta, kldir, w, k) so the
    registration can do its arithmetic on the DESIGN rather than on the
    queued subset."""
    return [(model, arm, style, kl_beta, kldir, w, k)
            for model in S3_MODELS
            for (w, k) in S3_ENVS
            for arm, style, kl_beta, kldir in s3_arms(w, k)]


def s3_rows():
    """The NEW GPU jobs: the 50 conceptual cells minus the 4 declared
    archived reuses = 46."""
    return [s3_row(model, arm, style, kl_beta, kldir, w, k)
            for model, arm, style, kl_beta, kldir, w, k in s3_cells()
            if (model, arm, w, k) not in S3_REUSED]


def s3_smoke_rows():
    """Exactly ONE 3-round cell: Qwen3-8B, REVERSE KL, lambda = 1, the
    Wu-boundary environment (W = 1, k = 1), both gates open.

    That is this wave's hardest new corner in one job -- the NEW
    checkpoint (thinking template off), the NEW direction on this key,
    and the boundary environment -- so if $(kldir) ever failed to reach
    the runner, or the Qwen3 chat template broke completion-only SFT
    masking, this is the cell that shows it in three rounds."""
    return [s3_row("qwen3_8b", "revlam1", "sft_kl", "1", "reverse",
                   *S3_ENV_WU, rounds=S3_SMOKE_ROUNDS, smoke=True)]


def s3_sub(smoke=False):
    rows = s3_smoke_rows() if smoke else s3_rows()
    key = S3_SMOKE_KEY if smoke else S3_KEY
    return S3_SUB_TEMPLATE.format(
        key=key, n_jobs=len(rows), gpu=S3_H100, bad=BAD_NODE_REQ,
        rounds=(S3_SMOKE_ROUNDS if smoke else S3_ROUNDS),
        kind=("3-ROUND QWEN3 REVERSE-KL SMOKE" if smoke
              else "SECTION-3 RETENTION TABLE, 100 ROUNDS"))


# ---------------------------------------------------------------------
# section3_peer_sweeps[_smoke] (2026-08-23). MATCHED PEER-SWEEP STRENGTH.
#
# QUESTION. Ordinary SFT holds a POSITIVE post-peer SD plateau on the Wu
# boundary. Is that an artifact of running only ONE peer sweep between
# retraining rounds? If the plateau is set by how much contraction the
# peer process gets per round, more sweeps should push it down.
# Stated as a HYPOTHESIS, not a gating expectation: the analyzer reports
# what the plateau does, including "it did not move".
#
# The Section 3 Qwen3 surface is held FIXED -- W = 1, k = 1, both gates
# all_open, movielens/Action 723 agents, seed 0, fresh LoRA r512 every
# round, forward KL only -- and the ONLY thing that varies is
# AB_SWEEPS. 60 rounds.
#
# AB_SWEEPS RIDES THE QUEUE. It is an ordinary env var (default 1) that
# no Section 3 sub sets, so it is added as a $(sweeps) column here. One
# sweep keeps its existing meaning: approximately one population-wide
# set of sampled pair interactions. NO new population operator.
#
# REUSE. S = 1 already exists at 100 rounds for all three arms; rounds
# 1-60 are a prefix of those runs, and a field-level audit found every
# scientific field identical (model, thinking off, dataset, 723 agents,
# seed, forward direction, EMPTY reference adapter, W, k, both gate
# modes, eps, ab_sweeps = 1, fresh LoRA, rank, lr, epochs, cap, no ICL,
# serve_eval_mode, anch2 marker). So S = 1 is NOT re-run: 3 arms x 3 new
# sweep counts = 9 production jobs.
# CAVEAT recorded rather than hidden: a 100-round prefix equals a
# 60-round run only if the population and peer streams are stateless in
# (seed, round). That is the documented design, but it is an assumption
# about the stream, not a measurement.
# DEFAULT HORIZON FOR NEW CLOSED-LOOP WAVES: 30 ROUNDS (2026-08-23).
# The outcome that sets the horizon is POPULATION convergence, and it
# settles early. The 100-round S=1 Qwen3 cells were stationary from
# round ~40 with late SD slopes under .004/100rd; the peer-sweep cells
# had SD slopes of ~1e-5/round by round 16; and the balance is a genuine
# fixed point, sigma* = c*a*sigma_innate / [1 - c(1-a)] with a = 0.5k at
# W = 0.5, approached geometrically at ratio c(1-a) -- about .17/round
# at S = 20, so ~99% of the way there in three rounds.
# TWO EXCEPTIONS. (1) MODEL-side quantities converge more slowly than
# the population: ordinary SFT's served SD was still falling at round 16
# and lambda=1's equilibrium mean was still drifting, so a served-map or
# retention claim wants the longer horizon -- say so rather than
# shortening silently. (2) Never kill a RUNNING wave to save rounds:
# trajectory.pt and raw_gen_log.json.gz are written only at completion,
# so stopping mid-run forfeits the entire artifact and costs more than
# finishing.
# The PS and MEM waves below keep 60 because they were already RUNNING
# when this default was set; changing their constants would retag cells
# whose directories are on the cluster under the old horizon.
NEW_WAVE_ROUNDS = 30

PS_KEY = "section3_peer_sweeps"
PS_SMOKE_KEY = "section3_peer_sweeps_smoke"
PS_MODEL = "qwen3_8b"
PS_W = 1.0
PS_K = 1.0
PS_ROUNDS = 60
PS_SMOKE_ROUNDS = 3
PS_SEED = 0
PS_H100 = S3_H100
PS_EPS_SOCIAL = S3_EPS_SOCIAL
# (arm token, training_style, lambda). Forward KL only -- no reverse.
PS_ARMS = [("sft", "sft", "0"), ("fwdlam1", "sft_kl", "1"),
           ("fwdlam8", "sft_kl", "8")]
PS_SWEEPS_NEW = [5, 20, 100]      # queued
PS_SWEEPS_REUSED = 1              # served by the archived S3 100-round cells
PS_REUSED = {
    "sft": "pofds3_qwen3_8b_sft_eaopen_w1_k1_esopen_anch2_s0_r100",
    "fwdlam1": "pofds3_qwen3_8b_fwdlam1_eaopen_w1_k1_esopen_anch2_s0_r100",
    "fwdlam8": "pofds3_qwen3_8b_fwdlam8_eaopen_w1_k1_esopen_anch2_s0_r100",
}

ROW_PS = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
          "ab, {es}, 0.0, {wplat}, loop, 0.0, {lam}, {kldir}, {sweeps}, "
          "{iclk}, {snap}, {uselora}, {fresh}, {ansk}, {gg}, {nrounds}, "
          "{basemodel}, {chatthink}, {mem}, {disk}, {pplbatch}")


def ps_tag(arm, sweeps, rounds=PS_ROUNDS, smoke=False):
    """pofdps_qwen3_8b_fwdlam8_sw100_eaopen_w1_k1_esopen_anch2_s0_r60.

    The sweep count AND the horizon are both in the tag: a 60-round
    S = 20 cell and the 100-round S = 1 Section 3 cell of the same arm
    are different objects, and neither may be mistaken for the other."""
    pre = "pofdpssmk" if smoke else "pofdps"
    return (f"{pre}_{PS_MODEL}_{arm}_sw{sweeps}_eaopen_w{_num(PS_W)}"
            f"_k{_num(PS_K)}_esopen_{S3_OP_TOKEN}_s{PS_SEED}_r{rounds}")


def ps_row(arm, style, lam, sweeps, rounds=PS_ROUNDS, smoke=False):
    a = REACH_ARM_COLS["b1"]
    m = FAM_MODELS[PS_MODEL]
    return ROW_PS.format(
        tag=ps_tag(arm, sweeps, rounds, smoke), style=style, beta=lam,
        seed=PS_SEED, es=f"{PS_EPS_SOCIAL:g}", wplat=f"{PS_W:g}",
        lam=f"{PS_K:g}", kldir="forward", sweeps=sweeps,
        iclk=a["iclk"], snap=a["snap"], uselora=a["uselora"],
        fresh=a["fresh"], ansk=a["ansk"], gg=a["gg"], nrounds=rounds,
        basemodel=m["base_model"], chatthink=m["chatthink"],
        mem=m["mem"], disk=m["disk"], pplbatch=m["pplbatch"])


def ps_rows():
    return [ps_row(arm, style, lam, S)
            for S in PS_SWEEPS_NEW for arm, style, lam in PS_ARMS]


def ps_smoke_rows():
    """The most demanding path in one job: forward lambda = 8 at S = 100,
    i.e. the strongest anchor and 100 peer sweeps per round."""
    return [ps_row("fwdlam8", "sft_kl", "8", 100,
                   rounds=PS_SMOKE_ROUNDS, smoke=True)]


def ps_sub(smoke=False):
    rows = ps_smoke_rows() if smoke else ps_rows()
    key = PS_SMOKE_KEY if smoke else PS_KEY
    return PS_SUB_TEMPLATE.format(
        key=key, n_jobs=len(rows), gpu=PS_H100, bad=BAD_NODE_REQ,
        rounds=(PS_SMOKE_ROUNDS if smoke else PS_ROUNDS),
        kind=("3-ROUND S=100 FORWARD-lambda8 SMOKE" if smoke
              else "PEER-SWEEP STRENGTH, 60 ROUNDS"))


# ---------------------------------------------------------------------
# section3_memory[_smoke] (2026-08-23). THE MEMORY EXTENSION.
#
# QUESTION. k is the innate RE-ANCHOR strength: each round the human
# component is h = k*innate + (1-k)*x, so k = 1 is Jiduan's stateless
# setup (re-anchored to the innate opinion every round) and k < 1 adds
# direct state carryover. Does weakening that recurring innate anchor
# move the population equilibrium CLOSER to the frozen-model
# equilibrium? And does the ordering
#     perfect prediction -> plain SFT -> lambda=1 -> lambda=8 -> frozen
# survive as k falls? Mean/location and heterogeneity/SD are kept as
# SEPARATE outcomes -- a shift of the mean is not preservation of spread.
#
# Everything except k is held at the Section 3 Qwen3 surface: W = 0.5,
# both gates all_open, ONE peer sweep, seed 0, movielens/Action 723,
# fresh LoRA r512, forward KL only, 60 rounds.
#
# REUSE. Section 3 already ran this exact surface at k = 1 and k = 0.2
# for all three trained arms, 100 rounds; rounds 1-60 are a prefix and a
# field-level audit found every scientific field identical (W = 0.5,
# ab_sweeps = 1, Qwen3 thinking off, seed, forward, EMPTY reference
# adapter, both gate modes, rank, lr, cap, no ICL, serve_eval_mode,
# anch2 marker). So 6 of the 9 trained cells are FREE and only k = 0.5
# is queued: 3 GPU jobs.
# The two endpoint arms need no GPU at all -- perfect prediction is
# sim_perfect_predictor.py and the lambda = infinity frozen model is
# replay_frozen_offline.py, 3 k values each = 6 CPU artifacts.
MEM_KEY = "section3_memory"
MEM_SMOKE_KEY = "section3_memory_smoke"
MEM_MODEL = "qwen3_8b"
MEM_W = 0.5
MEM_KS = [1.0, 0.5, 0.2]          # k = INNATE_LAMBDA (re-anchor strength)
# S = number of COMPLETE Deffuant sweeps between retraining rounds, NOT
# individual pair interactions. S = 20 is the primary condition; S = 1 is
# the Section 3 setting and is mostly already run.
MEM_SWEEPS = [1, 20]
# (S, k) pairs already served by archived Section 3 cells -- never queued
MEM_REUSED_SK = [(1, 1.0), (1, 0.2)]
# 30, per the NEW_WAVE_ROUNDS default above: the outcome here is
# POPULATION convergence and it settles well before 30. The wave had
# been submitted at 60 and was cancelled a few minutes in, so nothing is
# forfeited by retagging -- the "never kill a running wave" caveat
# applies to a wave far enough along that its artifacts are worth more
# than the remaining rounds, which was not the case here.
MEM_ROUNDS = 30
MEM_SMOKE_ROUNDS = 3
MEM_SEED = 0
MEM_H100 = S3_H100
MEM_EPS_SOCIAL = S3_EPS_SOCIAL
MEM_ARMS = [("sft", "sft", "0"), ("fwdlam1", "sft_kl", "1"),
            ("fwdlam8", "sft_kl", "8")]
def mem_reused():
    """(arm, S, k) -> the archived Section 3 cell that serves that trained
    cell. A FUNCTION, not a module-level dict: _num() is defined further
    down this file, so a comprehension here would run before it exists.
    Only S = 1 can be reused -- Section 3 ran one sweep per round."""
    return {(arm, 1, k): f"pofds3_qwen3_8b_{arm}_eaopen_w0p5_k{_num(k)}"
                         f"_esopen_{S3_OP_TOKEN}_s0_r100"
            for arm, _s, _l in MEM_ARMS
            for (_S, k) in MEM_REUSED_SK}


def mem_tag(arm, S, k, rounds=MEM_ROUNDS, smoke=False):
    """pofdmem_qwen3_8b_fwdlam8_sw20_eaopen_w0p5_k0p5_esopen_anch2_s0_r60.

    BOTH dials are in the tag. An (S, k) cell and the Section 3 cell of
    the same arm differ in sweep count, horizon, or both, and neither may
    be mistaken for the other."""
    pre = "pofdmemsmk" if smoke else "pofdmem"
    return (f"{pre}_{MEM_MODEL}_{arm}_sw{S}_eaopen_w{_num(MEM_W)}"
            f"_k{_num(k)}_esopen_{S3_OP_TOKEN}_s{MEM_SEED}_r{rounds}")


def mem_row(arm, style, lam, S, k, rounds=MEM_ROUNDS, smoke=False):
    a = REACH_ARM_COLS["b1"]
    m = FAM_MODELS[MEM_MODEL]
    return ROW_PS.format(
        tag=mem_tag(arm, S, k, rounds, smoke), style=style, beta=lam,
        seed=MEM_SEED, es=f"{MEM_EPS_SOCIAL:g}", wplat=f"{MEM_W:g}",
        lam=f"{k:g}", kldir="forward", sweeps=S, iclk=a["iclk"],
        snap=a["snap"], uselora=a["uselora"], fresh=a["fresh"],
        ansk=a["ansk"], gg=a["gg"], nrounds=rounds,
        basemodel=m["base_model"], chatthink=m["chatthink"],
        mem=m["mem"], disk=m["disk"], pplbatch=m["pplbatch"])


def mem_cells():
    """Every (arm, S, k) trained cell MINUS the archived Section 3 ones."""
    return [(arm, style, lam, S, k)
            for S in MEM_SWEEPS for k in MEM_KS
            for arm, style, lam in MEM_ARMS
            if (S, k) not in MEM_REUSED_SK]


def mem_rows():
    return [mem_row(arm, style, lam, S, k)
            for arm, style, lam, S, k in mem_cells()]


def mem_smoke_rows():
    """The primary condition at its hardest corner: S = 20, forward
    lambda = 8, and the strongest state carryover k = 0.2."""
    return [mem_row("fwdlam8", "sft_kl", "8", 20, 0.2,
                    rounds=MEM_SMOKE_ROUNDS, smoke=True)]


def mem_sub(smoke=False):
    rows = mem_smoke_rows() if smoke else mem_rows()
    key = MEM_SMOKE_KEY if smoke else MEM_KEY
    return MEM_SUB_TEMPLATE.format(
        key=key, n_jobs=len(rows), gpu=MEM_H100, bad=BAD_NODE_REQ,
        rounds=(MEM_SMOKE_ROUNDS if smoke else MEM_ROUNDS),
        kind=("3-ROUND k=0.5 SMOKE" if smoke
              else "MEMORY EXTENSION, 60 ROUNDS"))


MEM_SUB_TEMPLATE = """\
# HTCondor: SECTION-3 MEMORY EXTENSION -- {kind}, {n_jobs} jobs.
# GENERATED by gen_pofd_sweep.py from the MEM block. Never edit by hand.
# Qwen3-8B, movielens/Action 723, W = 0.5, BOTH gates all_open, ONE peer
# sweep, seed 0, fresh LoRA r512, forward KL only, {rounds} rounds.
# BOTH dials ride the queue: k as $(lam) and S -- the number of COMPLETE
# Deffuant sweeps per round, not pair interactions -- as $(sweeps).
# (S=1, k=1) and (S=1, k=0.2) are REUSED from the archived Section 3
# cells (audited field by field) and are not queued here.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 AI_GATE_MODE=all_open PEER_GATE_MODE=all_open EPS_AI=1 INNATE_LAMBDA=$(lam) AB_SWEEPS=$(sweeps) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=$(kldir) WITH_TWIN=1 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{key}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, lam, kldir, sweeps, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


PS_SUB_TEMPLATE = """\
# HTCondor: SECTION-3 PEER-SWEEP STRENGTH -- {kind}, {n_jobs} jobs.
# GENERATED by gen_pofd_sweep.py from the PS block. Never edit by hand.
# The Section 3 Qwen3 surface held FIXED (W = 1, k = 1, both gates
# all_open, movielens/Action 723, seed 0, fresh LoRA r512, forward KL
# only, {rounds} rounds); ONLY AB_SWEEPS varies, and it rides the queue
# as $(sweeps). S = 1 is NOT queued -- the archived 100-round Section 3
# cells serve it, audited field by field.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 AI_GATE_MODE=all_open PEER_GATE_MODE=all_open EPS_AI=1 INNATE_LAMBDA=$(lam) AB_SWEEPS=$(sweeps) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=$(kldir) WITH_TWIN=1 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{key}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, lam, kldir, sweeps, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


S3_SUB_TEMPLATE = """\
# HTCondor: SECTION-3 RETENTION -- {kind}, {n_jobs} jobs.
# GENERATED by gen_pofd_sweep.py from the S3 block. Never edit by hand:
# rerun the script.
# Surface: movielens Action, 723 agents, seed 0, {rounds} rounds, BOTH
# GATES GENUINELY OPEN (all_open, not the numeric threshold 1), homophily
# gamma = 0, fresh LoRA r512 every round, one peer sweep, matched twin,
# greedy serving in eval mode, SAVE_RAW_GEN=1.
# W_PLAT (Celestine's beta) and INNATE_LAMBDA (k, Celestine's gamma) BOTH
# ride the queue rather than being pinned here, so the wave's three
# environments share this one sub template --
#   (W=0.5, k=1) main   (W=1, k=1) wu   (W=0.5, k=0.2) mem
# (W=1, k=0.2) is not generated: k drops out algebraically at W=1.
# (The smoke key renders the same template over its single row.)
# BASE_MODEL / CHAT_THINKING / memory / disk / PPL_BATCH ride the queue
# too, so Qwen2.5-7B-Instruct and Qwen3-8B (CHAT_THINKING=0, hybrid
# reasoning explicitly OFF) share this sub.
# KL_DIRECTION RIDES THE QUEUE. This key mixes forward and reverse rows,
# so direction is $(kldir) and the tag's arm token records it.
#   forward = KL(pi_ref || pi_theta)    reverse = KL(pi_theta || pi_ref)
# The lambda = 0 "sft" arm is direction-NEUTRAL: TRAINING_STYLE=sft with
# KL_BETA=0 builds no divergence term, so its kldir field is an inert
# placeholder and its tag carries no direction token.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 AI_GATE_MODE=all_open PEER_GATE_MODE=all_open EPS_AI=1 INNATE_LAMBDA=$(lam) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=$(kldir) WITH_TWIN=1 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{key}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, lam, kldir, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


# fig2_family_prior_scout[_smoke] (2026-08-17): the FIGURE-2 FAMILY-
# PRIOR SCOUT -- six checkpoints (Qwen2.5-7B-Instruct, Qwen3-8B
# thinking OFF, OLMo-2-1124-7B-Instruct, Olmo-3-7B-Instruct,
# Mistral-7B-Instruct-v0.3, Ministral-8B-Instruct-2410) x arms
# {b0 = ordinary SFT, b0p5 = forward-KL SFT beta=0.5, b1 = forward-KL
# SFT beta=1, k0 = frozen plain prompting} x EPS_AI=1 (numeric
# strict-< threshold, never all_open) x es {0.05, 0.2} x seed 0
# = 48 CONCEPTUAL cells on the canonical Action environment (W=0.5,
# lam=0.2, 30 rounds, nested AI-then-peer operator, matched twin,
# greedy serving, fresh LoRA each round on trained arms,
# SAVE_RAW_GEN=1). Reuse is by EXACT FIELD-LEVEL AUDIT
# (audit_fig2_family_prior_reuse.py -> manifest_fig2_family_prior.json,
# config fields + 30-round completeness + twin presence, never tag
# similarity); only cells the manifest marks "new" queue here. ONE sub
# serves all six checkpoints: BASE_MODEL / CHAT_THINKING / memory /
# disk / PPL_BATCH ride the queue (cols 24-28); every job runs against
# the offline lustre cache (PRE-CACHE Qwen/Qwen2.5-7B-Instruct there
# before submitting -- the other five are already cached).
# CHAT_THINKING=0 pins Qwen3's hybrid-reasoning template OFF; the
# runner's qwen3 masking marker ("</think>\n\n") keeps completion-only
# SFT masking correct after the non-thinking assistant prefix.
# Smokes (3 x 3 rounds, seed 0, production configuration): b1 at
# es0p2 for each NEW checkpoint (qwen3_8b / olmo3_7b / ministral8b).
FAM_KEY = "fig2_family_prior_scout"
FAM_SMOKE_KEY = "fig2_family_prior_smoke"
FAM_MANIFEST_PATH = os.path.join(HERE, "manifest_fig2_family_prior.json")
FAM_MODELS = {
    "qwen7b": {"base_model": "Qwen/Qwen2.5-7B-Instruct",
               "chatthink": "default", "mem": "128G", "disk": "40G",
               "pplbatch": 64},
    "qwen3_8b": {"base_model": "Qwen/Qwen3-8B",
                 "chatthink": "0", "mem": "128G", "disk": "40G",
                 "pplbatch": 64},
    "olmo7b": {"base_model": "allenai/OLMo-2-1124-7B-Instruct",
               "chatthink": "default", "mem": "160G", "disk": "60G",
               "pplbatch": 16},
    "olmo3_7b": {"base_model": "allenai/Olmo-3-7B-Instruct",
                 "chatthink": "default", "mem": "160G", "disk": "60G",
                 "pplbatch": 16},
    "mistral7b": {"base_model": "mistralai/Mistral-7B-Instruct-v0.3",
                  "chatthink": "default", "mem": "128G", "disk": "40G",
                  "pplbatch": 64},
    "ministral8b": {"base_model": "mistralai/Ministral-8B-Instruct-2410",
                    "chatthink": "default", "mem": "128G", "disk": "40G",
                    "pplbatch": 64},
}
FAM_ARMS = ["b0", "b0p5", "b1", "k0"]
FAM_ESS = [0.05, 0.2]
FAM_SMOKE_MODELS = ["qwen3_8b", "olmo3_7b", "ministral8b"]
ROW_FAM = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, ab, "
           "{es}, 0.0, 0.5, loop, 0.0, 1, threshold, {iclk}, {snap}, "
           "{uselora}, {fresh}, {ansk}, {gg}, {nrounds}, {basemodel}, "
           "{chatthink}, {mem}, {disk}, {pplbatch}")


def fam_tag(model, arm, es, seed=0, prefix="pofdfam"):
    return (f"{prefix}_{model}_{arm}_ea1_{w_tok()}"
            f"_es{_num(es)}_s{seed}")


def fam_row(model, arm, es, nrounds=30, prefix="pofdfam", seed=0):
    a = REACH_ARM_COLS[arm]
    m = FAM_MODELS[model]
    return ROW_FAM.format(
        tag=fam_tag(model, arm, es, seed, prefix),
        style=a["style"], beta=a["beta"], seed=seed, es=f"{es:g}",
        iclk=a["iclk"], snap=a["snap"], uselora=a["uselora"],
        fresh=a["fresh"], ansk=a["ansk"], gg=a["gg"], nrounds=nrounds,
        basemodel=m["base_model"], chatthink=m["chatthink"],
        mem=m["mem"], disk=m["disk"], pplbatch=m["pplbatch"])


def _fam_manifest():
    with open(FAM_MANIFEST_PATH) as fh:
        return json.load(fh)


def fam_rows():
    """Rows for the cells the field-level audit marks 'new'. The
    manifest is the single source of truth; counts are asserted for
    CONSISTENCY (48 conceptual, reused+new==48, full grid coverage),
    never forced to a particular reuse split."""
    mf = _fam_manifest()
    cells = mf["cells"]
    assert len(cells) == 48, len(cells)
    keys = {(c["model"], c["arm"], c["eps_social"]) for c in cells}
    assert keys == {(m, a, e) for m in FAM_MODELS for a in FAM_ARMS
                    for e in FAM_ESS}, "manifest grid incomplete"
    reused = [c for c in cells if c["status"] == "reused"]
    new = [c for c in cells if c["status"] == "new"]
    assert len(reused) + len(new) == 48
    assert mf["counts"]["reused"] == len(reused)
    assert mf["counts"]["new"] == len(new)
    bad_val = [c["run_tag"] for c in reused
               if c.get("validation") not in ("PASS", "SKIPPED")]
    assert not bad_val, f"reused cells failing validation: {bad_val}"
    return [fam_row(c["model"], c["arm"], c["eps_social"])
            for c in new]


def fam_smoke_rows():
    return [fam_row(m, "b1", 0.2, nrounds=3, prefix="pofdfamsmk")
            for m in FAM_SMOKE_MODELS]


# qwen retry (2026-08-18): exactly the 8 qwen7b cells. The scout
# released before Qwen/Qwen2.5-7B-Instruct reached the offline lustre
# cache, so all 8 died at model load (HF_HUB_OFFLINE=1 refused the
# download) and the retry policy removed them; the other 39 completed
# and passed the gate. SAME tags as the scout key BY DESIGN -- NEVER
# co-submit with fig2_family_prior_scout while these are queued there
# (double-queue write race); the idempotent exec no-ops anything
# already complete. PRE-CACHE the checkpoint before submitting.
FAM_QWEN_RETRY_KEY = "fig2_family_prior_qwen_retry"


def fam_qwen_retry_rows():
    return [r for r in fam_rows()
            if r.split(",")[0].startswith("pofdfam_qwen7b_")]


# beta scout + confirmation keys (2026-08-18): extend the COMPLETED
# 48-cell fam grid along the forward-KL coefficient ONLY. Arms b2/b4/b8
# copy the b1 envelope exactly (fresh LoRA, forward KL, canonical
# Action, ea1 numeric threshold) at es=0.05, seed 0 -- 6 checkpoints x
# 3 betas = 18 jobs, brand-new _b2_/_b4_/_b8_ tags (collision-asserted,
# no smokes: the b1 envelope smoked clean on every new checkpoint).
# The analyzer (analyze_fig2_family_prior.py) then selects the SMALLEST
# shared beta whose late populations sit closest in W1 to their own
# frozen k0 endpoints for >= 5 of 6 checkpoints; the matching
# confirmation key (12 jobs: 6 checkpoints x seeds 42/43 at that beta)
# is the ONLY one submitted.
FAM_BETA_KEY = "fig2_family_prior_beta_scout"
FAM_BETA_ARMS = ["b2", "b4", "b8"]
FAM_BETA_ES = 0.05
FAM_CONFIRM_SEEDS = [42, 43]
FAM_CONFIRM_KEY = {arm: f"fig2_family_prior_{arm}_confirm"
                   for arm in FAM_BETA_ARMS}


def fam_beta_rows():
    return [fam_row(model, arm, FAM_BETA_ES)
            for arm in FAM_BETA_ARMS
            for model in FAM_MODELS]


def fam_confirm_rows(arm):
    return [fam_row(model, arm, FAM_BETA_ES, seed=seed)
            for model in FAM_MODELS
            for seed in FAM_CONFIRM_SEEDS]


def fam_sub(kind):
    """kind: 'main' | 'smoke' | 'qwen_retry' | 'beta_scout' |
    'b2_confirm' | 'b4_confirm' | 'b8_confirm'."""
    key = {"main": FAM_KEY, "smoke": FAM_SMOKE_KEY,
           "qwen_retry": FAM_QWEN_RETRY_KEY,
           "beta_scout": FAM_BETA_KEY,
           "b2_confirm": FAM_CONFIRM_KEY["b2"],
           "b4_confirm": FAM_CONFIRM_KEY["b4"],
           "b8_confirm": FAM_CONFIRM_KEY["b8"]}[kind]
    n_jobs = {"main": len(fam_rows()),
              "smoke": len(fam_smoke_rows()),
              "qwen_retry": len(fam_qwen_retry_rows()),
              "beta_scout": len(fam_beta_rows()),
              "b2_confirm": len(fam_confirm_rows("b2")),
              "b4_confirm": len(fam_confirm_rows("b4")),
              "b8_confirm": len(fam_confirm_rows("b8"))}[kind]
    what = {"main": (f"{n_jobs} seed-0 production cells (30 rounds; the "
                     f"48-cell 6-checkpoint x b0/b0p5/b1/k0 x ea1 x "
                     f"es 0.05/0.2 grid minus the field-audited reuse)"),
            "smoke": ("SMOKE (3 x 3 rounds, seed 0, production "
                      "configuration; b1 es0p2 for qwen3_8b / "
                      "olmo3_7b / ministral8b)"),
            "qwen_retry": ("QWEN RETRY -- exactly the 8 qwen7b cells "
                           "that died uncached (SAME tags as the scout "
                           "key; never co-submit while they are queued "
                           "there; pre-cache Qwen2.5-7B-Instruct "
                           "first)"),
            "beta_scout": ("BETA SCOUT -- 18 seed-0 cells (6 "
                           "checkpoints x forward-KL beta {2,4,8} at "
                           "es0p05; everything else copies the "
                           "completed b1 configuration)"),
            "b2_confirm": ("BETA=2 CONFIRMATION -- 6 checkpoints x "
                           "seeds 42/43 at es0p05 (submit ONLY the "
                           "analyzer-selected beta's key)"),
            "b4_confirm": ("BETA=4 CONFIRMATION -- 6 checkpoints x "
                           "seeds 42/43 at es0p05 (submit ONLY the "
                           "analyzer-selected beta's key)"),
            "b8_confirm": ("BETA=8 CONFIRMATION -- 6 checkpoints x "
                           "seeds 42/43 at es0p05 (submit ONLY the "
                           "analyzer-selected beta's key)")}[kind]
    return FAM_SUB_TEMPLATE.format(key=key, n_jobs=n_jobs, what=what)


FAM_SUB_TEMPLATE = """\
# HTCondor: FIGURE-2 FAMILY-PRIOR SCOUT -- {what}
# GENERATED by gen_pofd_sweep.py from the FAM block. Never edit by
# hand: rerun the script. {n_jobs} job(s).
# Six checkpoints on the canonical Action environment: EPS_AI=1 under
# the numeric strict-< threshold gate, es 0.05/0.2, W=0.5, lam=0.2,
# gamma=0, 30 rounds, nested AI-then-peer operator, matched twin,
# greedy serving, fresh LoRA each round on trained arms (b0 sft /
# b0p5 + b1 forward-KL sft_kl), frozen plain prompting on k0,
# SAVE_RAW_GEN=1. BASE_MODEL / CHAT_THINKING / mem / disk / PPL_BATCH
# ride the queue -- Qwen3-8B runs CHAT_THINKING=0 (thinking pinned
# OFF; the runner's qwen3 marker keeps completion-only masking
# correct). All checkpoints load from the offline lustre cache:
# PRE-CACHE Qwen/Qwen2.5-7B-Instruct there before submitting.
# Cells already satisfied by the field-level reuse audit
# (manifest_fig2_family_prior.json) are NOT queued here.
# Gate every pull with check_pofd_sanity (FAM section: exact model id
# per slug, qwen3 chat_thinking=False, arm/gate/es surface).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
#   (flow: fig2_family_prior_smoke -> pull + gate ->
#    fig2_family_prior_scout)
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_fig2fam"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


# fam_gate_ablation (2026-08-18): the SECTION-3 FAMILY-GATE ABLATION
# -- the six family-prior checkpoints x SFT arms {b0 = beta 0, b1 =
# forward-KL beta 1} across the AI-gate axis ea 0.1/0.2/0.4/1
# (numeric strict-< threshold) at the FIXED es=0.05, lam=0.2, W=0.5
# canonical Action surface, seed 0, 30 rounds = 48 conceptual cells
# on the EXACT completed family-prior-scout code path (same serving,
# training, peer and twin paths; SAVE_RAW_GEN=1; Qwen3 thinking OFF;
# same pofdfam_ family and sub surface -- eps_ai already rides the
# queue). The 12 completed ea=1 scout cells REUSE by field-level
# audit (audit_fam_gate_reuse.py -> manifest_fam_gate_ablation.json;
# save_raw_gen is a MATCHED field so the pofdevo_ wave's mistral b0
# runs at these gates can never shadow a cell); the 36 cells at ea
# {0.1, 0.2, 0.4} queue here. NO smoke: every checkpoint already
# carries a production gate on this path.
FAMG_KEY = "fam_gate_ablation"
FAMG_MANIFEST_PATH = os.path.join(
    HERE, "manifest_fam_gate_ablation.json")
FAMG_ARMS = ["b0", "b1"]
FAMG_GATES = [0.1, 0.2, 0.4, 1.0]
FAMG_ES = 0.05
ROW_FAMG = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
            "ab, {es}, 0.0, 0.5, loop, 0.0, {eps_ai}, threshold, "
            "{iclk}, {snap}, {uselora}, {fresh}, {ansk}, {gg}, "
            "{nrounds}, {basemodel}, {chatthink}, {mem}, {disk}, "
            "{pplbatch}")


def famg_tag(model, arm, gate, seed=0, es=FAMG_ES):
    return (f"pofdfam_{model}_{arm}_ea{_num(gate)}_{w_tok()}"
            f"_es{_num(es)}_s{seed}")


def famg_row(model, arm, gate, nrounds=30, seed=0, es=FAMG_ES):
    a = REACH_ARM_COLS[arm]
    m = FAM_MODELS[model]
    return ROW_FAMG.format(
        tag=famg_tag(model, arm, gate, seed, es),
        style=a["style"], beta=a["beta"], seed=seed,
        es=f"{es:g}", eps_ai=f"{gate:g}",
        iclk=a["iclk"], snap=a["snap"], uselora=a["uselora"],
        fresh=a["fresh"], ansk=a["ansk"], gg=a["gg"], nrounds=nrounds,
        basemodel=m["base_model"], chatthink=m["chatthink"],
        mem=m["mem"], disk=m["disk"], pplbatch=m["pplbatch"])


def famg_rows():
    """The 36 genuinely-missing jobs, straight from the audited
    manifest -- counts asserted for CONSISTENCY with the expected
    12-reused/36-new split, never forced."""
    mf = json.load(open(FAMG_MANIFEST_PATH))
    cells = mf["cells"]
    assert mf["n_cells"] == 48 and len(cells) == 48, mf["n_cells"]
    assert {(c["model"], c["arm"], c["gate"]) for c in cells} == \
        {(m, a, g) for m in FAM_MODELS for a in FAMG_ARMS
         for g in FAMG_GATES}
    reused = [c for c in cells if c["status"] == "reused"]
    new = [c for c in cells if c["status"] == "new"]
    assert len(reused) == 12 and len(new) == 36, \
        (len(reused), len(new))
    assert all(c["gate"] == 1.0 for c in reused), \
        "only the completed ea=1 scout cells may reuse"
    assert all(c.get("verdict") == "PASS" for c in reused)
    rows = []
    for c in sorted(new,
                    key=lambda c: (c["model"], c["arm"], c["gate"])):
        r = famg_row(c["model"], c["arm"], c["gate"])
        assert r.split(",")[0].strip() == c["new_tag"], \
            (r.split(",")[0], c["new_tag"])
        rows.append(r)
    return rows


def famg_sub():
    return FAM_SUB_TEMPLATE.format(
        key=FAMG_KEY, n_jobs=len(famg_rows()),
        what=("SECTION-3 GATE ABLATION: 36 seed-0 cells (30 rounds; "
              "6 checkpoints x b0/b1 x ea 0.1/0.2/0.4 at es0p05; "
              "the 12 ea1 scout cells REUSE per the audited "
              "manifest -- never co-submit with the scout keys)"))


# fam_gate_social (2026-08-18): the SOCIAL-GATE EXTENSION of the
# Section-3 gate ablation -- the same six checkpoints x b0/b1 x ea
# 0.1/0.2/0.4/1 surface EXTENDED across es 0/0.05/0.2 (lam=0.2,
# W=0.5, seed 0, 30 rounds) = 144 conceptual cells on the exact fam
# code path. Field-level audit (audit_fam_gate_reuse.py, full-surface
# mode -> manifest_fam_gate_social.json): the es0p05 cells are the
# 48-cell ablation wave (12 complete ea1 scout cells + 36 queued in
# fam_gate_ablation -- NEVER re-queued here); es0p2 keeps the 12
# completed ea1 scout cells (incl. the pofdgate2d occupant inherited
# from the fam manifest); the 48 es0 cells + the 36 es0p2 cells
# below ea1 = 84 NEW jobs. NO smoke.
FAMGS_KEY = "fam_gate_social"
FAMGS_MANIFEST_PATH = os.path.join(
    HERE, "manifest_fam_gate_social.json")
FAMGS_ESS = [0.0, 0.05, 0.2]


def famgs_rows():
    """The 84 genuinely-missing jobs from the audited full-surface
    manifest -- counts asserted for CONSISTENCY with the expected
    split (every es0p05 cell reused-or-covered, es0p2 new below ea1,
    es0 all new), never forced."""
    mf = json.load(open(FAMGS_MANIFEST_PATH))
    cells = mf["cells"]
    assert mf["n_cells"] == 144 and len(cells) == 144, mf["n_cells"]
    assert {(c["model"], c["arm"], c["gate"], c["es"])
            for c in cells} == \
        {(m, a, g, e) for m in FAM_MODELS for a in FAMG_ARMS
         for g in FAMG_GATES for e in FAMGS_ESS}
    new = [c for c in cells if c["status"] == "new"]
    assert len(new) == 84 and mf["n_new"] == 84, len(new)
    assert all(c["status"] != "new" for c in cells
               if c["es"] == 0.05), \
        "the es0p05 surface belongs to fam_gate_ablation"
    assert all(c["es"] in (0.0, 0.2) for c in new)
    assert sum(1 for c in new if c["es"] == 0.0) == 48
    assert sum(1 for c in new if c["es"] == 0.2) == 36
    assert not any(c["es"] == 0.2 and c["gate"] == 1.0 for c in new)
    reused = [c for c in cells if c["status"] == "reused"]
    assert all(c.get("verdict") == "PASS" for c in reused)
    rows = []
    for c in sorted(new, key=lambda c: (c["model"], c["arm"],
                                        c["gate"], c["es"])):
        r = famg_row(c["model"], c["arm"], c["gate"], es=c["es"])
        assert r.split(",")[0].strip() == c["new_tag"], \
            (r.split(",")[0], c["new_tag"])
        rows.append(r)
    return rows


def famgs_sub():
    return FAM_SUB_TEMPLATE.format(
        key=FAMGS_KEY, n_jobs=len(famgs_rows()),
        what=("SOCIAL-GATE EXTENSION: 84 seed-0 cells (30 rounds; 6 "
              "checkpoints x b0/b1 x the 48 es0 cells + the 36 es0p2 "
              "cells below ea1; es0p05 lives in fam_gate_ablation and "
              "the ea1 cells reuse -- never co-submit with the scout "
              "or ablation keys)"))


# zsprior_screen (2026-08-17): ZERO-SHOT PRIOR SCREEN for four candidate
# checkpoints -- Qwen/Qwen3-8B (CHAT_THINKING=0: the hybrid-reasoning
# template's enable_thinking switch is pinned OFF so the answer lands
# inside the 6-token budget), allenai/Olmo-3-7B-Instruct,
# mistralai/Ministral-8B-Instruct-2410 and
# mistralai/Mistral-Nemo-Instruct-2407. One job per checkpoint on the
# reachbase probe pattern: the paper's 723 MovieLens Action profiles,
# seed 0, greedy decoding, ONE round, frozen weights, no LoRA, no
# context, es=0, EPS_AI=0 under the strict-< threshold gate (no agent
# is ever contacted, opinions cannot update) -- pred_raw[0] IS the
# zero-shot prior. SAVE_RAW_GEN=1 persists every raw decoded response
# next to the parsed values (raw_gen_log.json.gz). BASE_MODEL and
# CHAT_THINKING ride the queue (cols 24/25). NEW family pofdzsprior_
# -- zero tags shared with any other wave (collision-asserted).
# PRE-CACHE each checkpoint on the cluster before submitting (the sub
# runs HF_HUB_OFFLINE=1 against the lustre cache):
#   HF_HOME=/lustre/fast/fast/gsmithline/hf_cache \
#     hf download <checkpoint-id>
ZSPRIOR_KEY = "zsprior_screen"
ZSPRIOR_MODELS = {
    "qwen3_8b": {"base_model": "Qwen/Qwen3-8B", "chatthink": "0"},
    "olmo3_7b": {"base_model": "allenai/Olmo-3-7B-Instruct",
                 "chatthink": "default"},
    "ministral8b": {"base_model": "mistralai/Ministral-8B-Instruct-2410",
                    "chatthink": "default"},
    "mistralnemo": {"base_model": "mistralai/Mistral-Nemo-Instruct-2407",
                    "chatthink": "default"},
}
ZSPRIOR_EXPECT_NEW = 4
ROW_ZSPRIOR = ("{tag}, frozen, 0, 0, 1, replace, 1.0, fixed, ab, 0, "
               "0.0, 0.5, loop, 0.0, 0, threshold, 0, -1, 0, 0, 0, 0, "
               "1, {basemodel}, {chatthink}")


def zsprior_tag(slug):
    return f"pofdzsprior_{slug}_{w_tok()}_es0_s0"


def zsprior_rows():
    return [ROW_ZSPRIOR.format(tag=zsprior_tag(slug),
                               basemodel=m["base_model"],
                               chatthink=m["chatthink"])
            for slug, m in ZSPRIOR_MODELS.items()]


def zsprior_sub():
    return ZSPRIOR_SUB_TEMPLATE.format(key=ZSPRIOR_KEY,
                                       n_jobs=len(zsprior_rows()))


ZSPRIOR_SUB_TEMPLATE = """\
# HTCondor: ZERO-SHOT PRIOR SCREEN -- 4 candidate checkpoints, one
# 1-round frozen probe each (Qwen3-8B thinking-off / Olmo-3-7B-Instruct
# / Ministral-8B-Instruct-2410 / Mistral-Nemo-Instruct-2407).
# GENERATED by gen_pofd_sweep.py from the ZSPRIOR block. Never edit by
# hand: rerun the script. {n_jobs} job(s).
# The paper's 723 MovieLens Action profiles, seed 0, greedy decoding,
# frozen weights, no LoRA, no context, es=0, EPS_AI=0 under the
# strict-< threshold gate: no agent is ever contacted, opinions cannot
# update, and pred_raw[0] is the checkpoint's zero-shot prior.
# SAVE_RAW_GEN=1 writes raw_gen_log.json.gz (every raw decoded
# response + parsed values); BASE_MODEL and CHAT_THINKING ride the
# queue (Qwen3 runs enable_thinking=False explicitly).
# HF_HUB_OFFLINE=1: PRE-CACHE all four checkpoints into
# /lustre/fast/fast/gsmithline/hf_cache before submitting
# (HF_HOME=... hf download <id>).
# Gate every pull with check_pofd_sanity (ZSPRIOR section: 723 raw
# responses present, every one carrying a digit, parsed==served,
# finite in-range predictions, opinions untouched, gate all-closed).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = 128G
request_disk      = 60G
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward INNATE_LAMBDA=0.2 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=0 SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_zsprior_screen"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink from experiments/condor/configs_pofd_{key}.txt
"""


REACH_SUB_TEMPLATE = """\
# HTCondor: SFT-ICL REACH, {model} -- {what}
# GENERATED by gen_pofd_sweep.py from the REACH block + the audited
# manifest_sft_icl_reach.json (2026-08-13). Never edit by hand: rerun
# audit_sft_icl_reach_reuse.py (if the local corpus changed) and then
# this script. {n_jobs} job(s).
# No-peer reach study -- arms b0 (ordinary SFT) / b1 (forward SFT-KL)
# / fz0 (frozen round-0 K=8 ICL context) / dyn (live refreshed K=8
# context) x gates ea {{0.05,0.1,0.2,0.4,0.7}} + the explicit all-open
# gate (AI_GATE_MODE=all_open; NEVER eps_ai=1 -- the threshold gate is
# strict-<). W=0.5, lam=0.2, es=0, gamma=0, greedy serving,
# WITH_TWIN=1, movielens Action, 723 agents. N_ROUNDS / style / gate
# mode / ICL & adapter knobs ride the queue (cols 15-23): mains 30
# rounds, pofdreachbase_ probes 1 round (frozen K=0 at EPS_AI=0: the
# strict gate never opens, opinions cannot update; pred_raw[0] is the
# frozen no-context baseline m_base), smokes 3 rounds.
# Gate every pull with check_pofd_sanity (REACH section: shared
# ai_gate bit-replay, twin==innate to 1 ulp, mandatory gate_raw +
# hardware provenance, all-open all-true + contact exactly 1, fz0
# constant gate/pred, fz0<->dyn round-0 context identity). GPU
# heterogeneity is recorded per run (config hardware block) and
# handled downstream by analyze_sft_icl_reach.py -- no capability pin
# here (it would starve the wave at low bids).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
#   (umbrellas: sft_icl_reach_smoke -> gate -> sft_icl_reach_baseline
#    -> gate -> sft_icl_reach)
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) AI_GATE_MODE=$(gatemode) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{model}_pofdreach"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, gatemode, iclk, snap, uselora, fresh, ansk, gg, nrounds from experiments/condor/configs_pofd_{key}.txt
"""


def reach_sub(model, kind):
    """kind: 'main' | 'base' | 'smoke' | 's0'."""
    short = REACH_KEY[model].split("_")[-1]
    key = {"main": REACH_KEY[model],
           "base": f"sft_icl_reach_base_{short}",
           "smoke": f"sft_icl_reach_smoke_{short}",
           "s0": f"sft_icl_reach_s0_{short}"}[kind]
    n_jobs = {"main": len(reach_rows(model)),
              "base": len(REACH_SEEDS),
              "smoke": len(reach_smoke_rows(model)),
              "s0": len(reach_s0_rows(model))}[kind]
    what = {"main": "audited-missing main trajectories (30 rounds)",
            "base": "1-round frozen K=0 baseline probes (EPS_AI=0)",
            "smoke": "SMOKE (3 rounds)",
            "s0": ("SEED-0 EXPLORATORY SLAB (30 rounds; gates 0.05-0.4 + "
                   "all-open, no ea0p7; SAME tags as the full key -- never "
                   "co-submit with sft_icl_reach)")}[kind]
    return REACH_SUB_TEMPLATE.format(model=model, key=key, n_jobs=n_jobs,
                                     what=what, **REACH_MODELS[model])


# RANDOM-EVEN-SPLIT twin of the controlled-teacher wave (2026-08-04, user):
# same two-stage design with the favored set a SYNTHETIC balanced random
# split (A/B, 361/362 of 723) instead of gender. No prompt feature marks
# membership, so a teacher can carry the signal ONLY by memorizing
# individual profiles -- separating feature-mediated endogenization (the
# gender wave) from instance-memorization transfer, with the group-size
# asymmetry (520/203) removed. TEACHER_LABEL_COL=random_even, FAV=A,
# TEACHER_GROUP_SEED=0: split seeded 52100+group_seed, run-seed-INDEPENDENT
# (one split per wave; it lives in the teacher weights), saved to
# random_group.json + trajectory.pt gender_true in every run.
#   Stage 1 (qwen7b_tchr, 2 jobs): +/-0.08 teachers keyed on the split.
#   GATE: signs opposite + no collapse + neutral between are HARD gates;
#   the magnitude here is stage 1's MEASUREMENT (a small gap = the
#   memorization limit, a result, not a pipeline failure). Proceed to
#   stage 2 only if |gap| >= 0.05 (else report and stop).
#   Stage 2 (qwen7b_tfer, 6 jobs after tfer_smoke): tpos/tneg x seeds
#   {0,42,43}. tneu is NOT re-run -- the pofdtfe_ tneu trajectories are
#   physics-identical (same neutral teacher, seeds, env; telemetry grouping
#   does not touch dynamics) and their random-group gaps are recomputed
#   offline from saved op/pred/twin_raw + the deterministic split. No
#   gdrop/gperm arms: nothing displayed marks the group.
# Tags: pofdtchr_qwen7b_d{p,m}0p08_ea0p4_w0p5_l0p2_s0;
# pofdtfer_qwen7b_b1_ea0p4_{tpos,tneg}_w0p5_l0p2_es0p2_s<seed>_fresh_data.
# Checker: pofdtchr/pofdtfer prefixes ride the is_tch/is_tfe branches with
# col=random_even, fav=A, tchr ref paths, and no _disp key requirement.
TFER_REFS = {
    "tpos": f"{TFE_RUNS}/pofdtchr_qwen7b_dp0p08_ea0p4_w0p5_l0p2_s0/round0_adapter",
    "tneg": f"{TFE_RUNS}/pofdtchr_qwen7b_dm0p08_ea0p4_w0p5_l0p2_s0/round0_adapter",
}

# OLMo ROMANCE mirror of the env3 fe wave (2026-08-05, user): the corrected
# feature-endogenization experiment (natural gender, W=0.5, lam=0.2, es=0.2,
# ea=0.4, forward KL, replace + fresh adapter, 30 rounds, seeds {0,42,43})
# re-run with BASE_MODEL=allenai/OLMo-2-1124-7B-Instruct and
# ML_TARGET=Romance -- a second-model x second-target replication of the
# qwen/Action fes+fef matrix. Same movielens LCC population (measured
# 2026-08-05: the Romance kNN graph keeps the same n=723, 520 M / 203 F),
# so TRAIN_CAP/N_LABELED stay 723. Romance flips the baseline: innate M-F
# gap -0.0149 (women rate Romance higher; Action was +0.0021), innate mean
# 0.6641 std 0.1243. Profiles/prompts exclude Romance and include Action
# (loader feats = core minus target -- automatic). NO controlled teachers /
# transformed labels / gdrop / gperm arms in this wave.
# Arms x seeds {0,42,43} (12 jobs, composite key olmo7brom_fe = fes + fef):
#   fes (9): natural b {0, 0.5, 1} -- pofdws2f_-family tags, b0 style sft
#            (direction-free), b>0 sft_kl + KL_DIRECTION=forward
#   fef (3): frozen no-context k0 -- pofdicls2_-family tags
# Matched no-platform twins ride es>0 (twin_raw in every run; within-seed
# twins identical across trained arms, as in the qwen wave). AUDIT
# 2026-08-05: no Romance-target run exists anywhere -- ML_TARGET has never
# left Action in any wave, and these tags are newly minted -- so ALL 12
# cells run; nothing is reusable (the qwen fes-style s0 anchors are Action
# runs). The MODEL SLOT carries the target token (olmo7brom): the _w/_l/
# _es and icl-arm regexes parse unchanged, and check_pofd_sanity gates
# ml_target=Romance + an OLMo-2 base_model on that token. Smoke FIRST
# (key olmo7brom_fe_smoke: ws2f b1_ea0p4 s0, 3 rounds, DEBUG_GEN):
# ML_TARGET=Romance is a new production dial (the loader is generic but
# never ran off Action) and OLMo's zero-shot Romance behavior is
# unmeasured; everything else is validated (env3 operator + forward KL on
# olmo by its ws2f wave, frozen k0 + peers by olmo icls2, the seed
# protocol by qwen fes/fef).

# NARROW-GATE SOCIAL-DOSE matrix (2026-08-05, user), keys qwen7b_esfn /
# olmo7b_esfn: the eps-social dose-response at NARROW AI gates, matched
# across models. Forward SFT-KL b1, W=0.5, lam=0.2, gamma=0, corrected
# nested operator, fresh + replace, 30 rounds, WITH_TWIN=1 everywhere:
#   ea {0.05, 0.10} x es {0, 0.10, 0.15, 0.20, 0.25} x s {0, 42, 43}
#   x models {qwen7b, olmo7b} on movielens Action = 60 cells.
# The esf wave swept es at the WIDE gate (ea0p4, where the anchor's camp
# is large); this matrix asks how peer repair composes with a NARROW gate
# that rarely opens -- the low-contact corner of the (ea, es) surface --
# and whether the answer is model-general (qwen's out-of-range 0.25 prior
# vs olmo's in-range 0.75 spike). Tags stay in the per-dose HOME families
# (fes/fex convention -- each (es, ea) series is family-uniform across
# seeds): es=0 -> pofdw2f_, es=0.2 -> pofdws2f_, mid doses -> pofdesf_
# (first olmo7b tags in the esf family; the checker is model-agnostic and
# routes by the _w/_l/_es tokens: es=0 exact replay, es>0 peer gate +
# twin). AUDIT 2026-08-05 (on-cluster, all 60 tags): exactly 8 EXIST --
# both models' b1_ea0p05/b1_ea0p1 s0 at es=0 (w2f) and es=0.2 (ws2f),
# the validated gate cells -- REUSED, not re-run. 52 cells run (26 per
# model). The reused es=0 s0 runs predate WITH_TWIN (their no-platform
# twin is the deterministic innate drift, computable offline -- esf
# precedent); new es=0 rows save it via WITH_TWIN=1. No smoke: env
# identical to the validated esf sub (qwen) / esf env + olmo deltas
# validated by olmo w2f/ws2f/w2fx (olmo); only queue-fed dials differ.
NDS_EAS = [0.05, 0.1]
NDS_ESS = [0.0, 0.10, 0.15, 0.20, 0.25]
NDS_MODELS = ["qwen7b", "olmo7b"]


def nds_tag(model, ea, es, seed):
    # b=1 slice of the cube tag map (kept so the superseded esfn configs
    # stay byte-identical; cube_tag is defined in the CUBE block below)
    return cube_tag(model, 1.0, ea, es, seed)


# FULL PARAMETER-CUBE (2026-08-05, user), keys qwen7b_cube / olmo7b_cube:
# every corrected-loop dial crossed on movielens Action --
#   beta {0, 0.1, 0.2, 0.5, 1} x ea {0.05, 0.1, 0.2, 0.4}
#   x es {0, 0.10, 0.15, 0.20, 0.25, 0.30} x seed 0
#   x models {qwen7b, olmo7b} = 240 cells.
# RESCOPED 2026-08-05 (user, same day): ONE seed per cell -- the original
# 3-seed spec (s {0, 42, 43} = 720 cells / 617 jobs) was too many jobs.
# Seed 0 is the project's canonical scan seed and maximizes reuse (every
# existing s0 plane counts). To add replicates later: extend CUBE_SEEDS
# and rerun -- the grid-minus-existing logic and the audit set below
# already carry the s42/s43 cells.
# Fixed dials: W=0.5, lam=0.2, gamma=0, corrected nested operator
# (population_update=nested_ai_then_social_v1), 30 rounds, fresh adapter
# each round + replace data, WITH_TWIN=1 (matched no-platform twin in
# every run), b0 -> ordinary sft (direction-free), b>0 -> forward SFT-KL.
# es=0.30 is a NEW dose (the esf s0 scan found re-entrant capture at
# 0.25 -- the wide radius ferries mainstream agents into the gate; 0.30
# extends the axis one step); b {0.1, 0.2} at mid doses are the first
# non-b1 pofdesf_ cells.
# AUDIT 2026-08-05 (on-cluster, BY CONFIG FIELDS, not tags): every
# config.json under runs/pokec_gated_lm scanned and matched on the full
# dial surface (dataset/target/base_model/population_update/W/lam/gamma/
# regime/fresh/pristine/canary/sweeps/eps/ea/beta/seed/style + LoRA-512
# budget; kl_direction=forward required at b>0; kl_ref_adapter, profile
# drop/permute/shuffle/sort, teacher deltas all absent; completeness =
# trajectory.pt present + 30 trajectory rounds). Legacy reverse-KL,
# continual-adapter and feature-ablation runs are NOT counted, by those
# same fields. EXACTLY 103 cells exist -- qwen7b 63, olmo7b 40 --
# enumerated wave-by-wave in CUBE_EXISTING (kept COMPLETE, including the
# s42/s43 cells now outside the rescoped grid: it is the audit record,
# and the replicate extension reuses it as-is); the qwen b0 s0 cells
# live under reverse-era pofdw2_/pofdws2_ tags (b0 has no KL term, the
# fes/fesr reuse precedent), everything else under pofdw2f_/pofdws2f_/
# pofdesf_ tags. Inside the 1-seed grid 45 qwen / 40 olmo cells exist,
# so 155 cells queue (qwen 75, olmo 80); nothing existing is ever
# overwritten (grid-minus-existing configs + the idempotent executable).
# Of the 57 zero-byte shells left by the 2026-08-05 cluster incident
# (no results in any), the 12 s0 esfn shells are cube tags and re-run;
# the 40 s42/s43 shells wait for the replicate extension.
# Tags stay in the per-dose HOME families (fes/fex/esfn convention):
# es=0 -> pofdw2f_, es=0.2 -> pofdws2f_, every other dose -> pofdesf_.
# SUPERSEDES qwen7b_esfn / olmo7b_esfn (never ran -- the incident killed
# all 52 before round 1): the cube owns the (ea, es) matrix now, its s0
# slice queues here with BYTE-IDENTICAL tags (6 per model), and the esfn
# s42/s43 cells are descoped with every other replicate --
# submit_pofd_sweep.sh refuses the esfn keys and points here
# (co-submission would double-queue tags into the same run dirs).
# The .sub files are GENERATED from CUBE_MODELS: one spec per model
# (slug -> base model + resource/env deltas). Adding a registered model
# = adding ONE entry and rerunning this script; do not hand-edit the
# generated at_pofd_<slug>_cube.sub.
# gate_raw telemetry (runner, 2026-08-05): every new run saves the
# per-round per-agent AI gate/contact mask to trajectory.pt (direct
# contact vs indirect peer transmission separate offline). The 103
# reused cells predate the key; their masks reconstruct exactly as
# |pred_raw[t] - x(t)| < ea on x(t) = innate / op_raw[t-1] -- the same
# derivation check_pofd_sanity now verifies on runs that carry gate_raw.
# No smoke: the sub env is byte-identical to the validated esfn subs but
# for the config filename + wandb suffix, every moving dial is
# queue-fed over combinations validated by the w2f/ws2f/w2fx/ws2fx/esf/
# fes waves, and the gate_raw runner change is telemetry-only (validated
# by the local pure-torch mock + the checker fixture suite).
CUBE_BETAS = [0.0, 0.1, 0.2, 0.5, 1.0]
CUBE_EAS = [0.05, 0.1, 0.2, 0.4]
CUBE_ESS = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]
# rescoped 2026-08-05: one seed per cell (extend + rerun for replicates)
CUBE_SEEDS = [0]
# slug -> spec: ONE entry drives configs_pofd_<slug>_cube.txt AND
# at_pofd_<slug>_cube.sub. extra_env sits between ML_TARGET and EPS_AI
# in the environment line (olmo: lustre HF cache, offline hub).
CUBE_MODELS = {
    "qwen7b": {"base_model": "Qwen/Qwen2.5-7B-Instruct",
               "mem": "128G", "disk": "40G", "ppl_batch": 64,
               "extra_env": ""},
    "olmo7b": {"base_model": "allenai/OLMo-2-1124-7B-Instruct",
               "mem": "160G", "disk": "60G", "ppl_batch": 16,
               "extra_env": "HF_HOME=/lustre/fast/fast/gsmithline/hf_cache "
                            "HF_HUB_OFFLINE=1 "},
}
ROW_CUBE = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, ab, "
            "{es}, 0.0, 0.5, loop, 0.0, {eps_ai}")


def cube_tag(model, b, ea, es, seed):
    """Per-dose HOME family: es=0 -> pofdw2f_, es=0.2 -> pofdws2f_,
    every other dose -> pofdesf_ (fes/fex/esfn convention)."""
    if es == 0.0:
        return (f"pofdw2f_{model}_b{_num(b)}_ea{_num(ea)}_{w_tok()}"
                f"_s{seed}_fresh_data")
    if es == 0.20:
        return (f"pofdws2f_{model}_b{_num(b)}_ea{_num(ea)}_{ws_tok()}"
                f"_s{seed}_fresh_data")
    return (f"pofdesf_{model}_b{_num(b)}_ea{_num(ea)}_{w_tok()}_es{_num(es)}"
            f"_s{seed}_fresh_data")


# AUDIT 2026-08-05: the 103 complete config-matching cells, wave by wave.
CUBE_EXISTING = set()
# qwen es {0, 0.2} s0 planes, full beta x ea: b>0 = w2f/ws2f forward
# waves; b0 = the reverse-era pofdw2_/pofdws2_ sft runs (direction-free)
CUBE_EXISTING |= {("qwen7b", b, ea, es, 0) for b in CUBE_BETAS
                  for ea in CUBE_EAS for es in (0.0, 0.20)}
# qwen fes2/fes replication seeds: natural b {0, 0.5, 1} at ea0p4,
# es 0 (pofdw2f_) and 0.2 (pofdws2f_)
CUBE_EXISTING |= {("qwen7b", b, 0.4, es, s) for b in (0.0, 0.5, 1.0)
                  for es in (0.0, 0.20) for s in (42, 43)}
# qwen esf/fex mid doses: b1 ea0p4 es {0.1, 0.15, 0.25} x s {0, 42, 43}
# + b1 ea0p2 es {0.1, 0.25} s0 (pofdesf_). Seeds literal on purpose --
# this is the audit record, independent of the rescoped CUBE_SEEDS.
CUBE_EXISTING |= {("qwen7b", 1.0, 0.4, es, s)
                  for es in (0.10, 0.15, 0.25) for s in (0, 42, 43)}
CUBE_EXISTING |= {("qwen7b", 1.0, 0.2, es, 0) for es in (0.10, 0.25)}
# olmo es {0, 0.2} s0 planes, full beta x ea (w2f/ws2f + w2fx/ws2fx fill)
CUBE_EXISTING |= {("olmo7b", b, ea, es, 0) for b in CUBE_BETAS
                  for ea in CUBE_EAS for es in (0.0, 0.20)}
assert len(CUBE_EXISTING) == 103, len(CUBE_EXISTING)


def cube_rows(model):
    return [ROW_CUBE.format(
        tag=cube_tag(model, b, ea, es, s),
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=s,
        es=f"{es:g}", eps_ai=f"{ea:g}")
        for b in CUBE_BETAS for ea in CUBE_EAS for es in CUBE_ESS
        for s in CUBE_SEEDS if (model, b, ea, es, s) not in CUBE_EXISTING]


CUBE_SUB_TEMPLATE = """\
# HTCondor: FULL PARAMETER-CUBE, {model} half -- GENERATED by
# gen_pofd_sweep.py from the CUBE_MODELS spec (2026-08-05). Never edit
# this file by hand: edit the spec / CUBE_EXISTING and rerun the script.
# {n_jobs} audited-missing cells of beta {{0,0.1,0.2,0.5,1}} x
# ea {{0.05,0.1,0.2,0.4}} x es {{0,0.10,0.15,0.20,0.25,0.30}} at seed 0
# (replicates descoped 2026-08-05: extend CUBE_SEEDS + rerun to add
# them) -- forward SFT-KL (b0 rows ordinary sft), W_PLAT=0.5,
# INNATE_LAMBDA=0.2, gamma=0, corrected nested operator, fresh +
# replace, 30 rounds, movielens Action, WITH_TWIN=1. beta/style/es/ea/
# seed all ride the queue (es col 10, eps_AI col 15); tags stay in the
# per-dose home families (es=0 -> pofdw2f_, es=0.2 -> pofdws2f_, else
# pofdesf_). The 103 cells the 2026-08-05 config-field audit found
# complete are NOT queued (CUBE_EXISTING in gen_pofd_sweep.py).
# SUPERSEDES at_pofd_{model}_esfn.sub -- byte-identical tags; never
# co-submit both. Every run saves twin_raw + the per-agent gate_raw
# contact mask (runner 2026-08-05). Gate every pull with
# check_pofd_sanity (es=0 exact replay, es>0 peer gate + twin; _b token,
# forward direction at b>0, and gate_raw are cross-checked).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {model}_cube
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 ANS_SAMPLE_K=16 ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 FRESH_EACH_ROUND=1 TRAIN_CAP=723 N_ROUNDS=30 EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 USE_LORA=1 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{model}_lora512_pofdcube"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai from experiments/condor/configs_pofd_{model}_cube.txt
"""


def cube_sub(model):
    return CUBE_SUB_TEMPLATE.format(model=model, n_jobs=len(cube_rows(model)),
                                    **CUBE_MODELS[model])


# EXACT INITIAL-DATA REPLAY (2026-08-06, user), key qwen7b_replay1:
# does an exact anchor of ORIGINAL round-0 data in every retrain arrest
# the closed-loop drift? One-seed slice at the corner dose:
#   beta {0, 1} x replay_frac {0, 0.25, 0.5, 0.75, 1} at ea=0.4,
#   es=0.1, seed 0, qwen7b -- W=0.5, lam=0.2, gamma=0, corrected nested
#   operator, 30 rounds, fresh adapter + replace data, WITH_TWIN=1,
#   b0 -> ordinary sft (direction-free), b1 -> forward SFT-KL.
# Mechanism (runner 2026-08-06, REPLAY_FRAC env -> config replay_frac):
# every deploy round's FIXED-SIZE 723-row batch holds exactly
# round(rf*723) rows with their ORIGINAL round-0 labels (initial_data)
# and the rest with the LATEST round's population labels -- a per-row
# label-source partition of the same one-row-per-agent batch, redrawn
# per round from a dedicated RNG stream, NEVER accumulated history.
# Distinct dial from pristine_frac (accumulate-only pool resampling,
# behavior untouched); the runner refuses rf>0 off the replace regime
# or combined with pristine_frac.
# rf=0 REUSE (2 cells): the whole replay path, its RNG included, is
# guarded by replay_frac > 0, so an rf=0 run is byte-identical to the
# plain replace loop -- and the (b, ea0.4, es0.1, s0) qwen cells
# already exist, complete + config-field-audited + gated (2026-08-05):
#   b0 -> pofdesf_qwen7b_b0_ea0p4_w0p5_l0p2_es0p1_s0_fresh_data (cube)
#   b1 -> pofdesf_qwen7b_b1_ea0p4_w0p5_l0p2_es0p1_s0_fresh_data (esf)
# Same env surface, seed, KL config and data process -> reused as the
# rf=0 cells; 8 jobs queue.
# Telemetry: every rf>0 run saves replay_raw (per-round label-source
# mask), train_y_raw (the ACTUAL labels trained on) and row n_replay;
# check_pofd_sanity's REPLAY section verifies the exact composition
# bit-exactly, plus _b/_rf tokens, style, forward direction and seed.
# Tag family pofdrpl_ keeps the cube token grammar (_b before _ea) so
# the cube-family beta/style/direction gates apply unchanged.
# The .sub is GENERATED from the CUBE_MODELS registry: another model =
# one spec entry + one key here. No smoke: the env line is the
# validated cube env + REPLAY_FRAC=$(rfrac), and the mechanism is
# validated by the local pure-torch fixture suite.
RPL_BETAS = [0.0, 1.0]
RPL_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
RPL_EA = 0.4
RPL_ES = 0.1
RPL_SEEDS = [0]
RPL_MODELS = ["qwen7b"]          # slugs into the CUBE_MODELS registry
ROW_RPL = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, ab, "
           "{es}, 0.0, 0.5, loop, 0.0, {eps_ai}, {rfrac}")


def rpl_tag(model, b, rf, seed):
    return (f"pofdrpl_{model}_b{_num(b)}_ea{_num(RPL_EA)}_rf{_num(rf)}"
            f"_{w_tok()}_es{_num(RPL_ES)}_s{seed}_fresh_data")


def rpl_rows(model):
    # rf=0 cells are the audited pre-existing replace runs (see block
    # comment) -- never re-queued, never overwritten
    return [ROW_RPL.format(
        tag=rpl_tag(model, b, rf, s),
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=s,
        es=f"{RPL_ES:g}", eps_ai=f"{RPL_EA:g}", rfrac=f"{rf:g}")
        for b in RPL_BETAS for rf in RPL_FRACS for s in RPL_SEEDS
        if rf > 0]


RPL_SUB_TEMPLATE = """\
# HTCondor: EXACT INITIAL-DATA REPLAY, {model} -- GENERATED by
# gen_pofd_sweep.py from the CUBE_MODELS spec (2026-08-06). Never edit
# this file by hand: edit the RPL_ block and rerun the script.
# {n_jobs} cells of beta {{0,1}} x replay_frac {{0.25,0.5,0.75,1}} at
# ea=0.4, es=0.1, seed 0 -- the rf=0 cells REUSE the audited complete
# replace runs (pofdesf_ b0/b1 ea0p4 es0p1 s0; rf=0 is byte-identical
# to the plain loop, the replay path is guarded by replay_frac > 0).
# Every deploy round trains on a FIXED 723-row batch holding exactly
# round(rf*723) ORIGINAL round-0 labels + the rest from the LATEST
# round only (never accumulated history); forward SFT-KL at b1,
# ordinary sft at b0, W_PLAT=0.5, INNATE_LAMBDA=0.2, gamma=0, fresh +
# replace, 30 rounds, movielens Action, WITH_TWIN=1. REPLAY_FRAC rides
# the queue (col 16) next to beta/style/eps_ai. Gate every pull with
# check_pofd_sanity (REPLAY section: bit-exact batch composition via
# replay_raw/train_y_raw; _b/_rf tokens, style, forward direction and
# seed cross-checked).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {model}_replay1
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) REPLAY_FRAC=$(rfrac) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 ANS_SAMPLE_K=16 ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 FRESH_EACH_ROUND=1 TRAIN_CAP=723 N_ROUNDS=30 EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 USE_LORA=1 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{model}_lora512_pofdreplay1"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, rfrac from experiments/condor/configs_pofd_{model}_replay1.txt
"""


def rpl_sub(model):
    return RPL_SUB_TEMPLATE.format(model=model, n_jobs=len(rpl_rows(model)),
                                   **CUBE_MODELS[model])


# ORDINARY-SFT TRAINING-BUDGET SWEEP (2026-08-06, user), keys
# qwen7b_budget / olmo7b_budget (+ qwen7b_budget_smoke): does the drift
# depend on how HARD the platform fits each round's data? Undertrained
# arms of the b0 corner at the wide peer dose: both models, ea=0.4,
# es=0.3, s0 -- W=0.5, lam=0.2, gamma=0, nested operator, 30 rounds,
# fresh + replace, WITH_TWIN=1, ordinary sft (b0, direction-free).
# Budget dial: SFT_EPOCHS=0 hands the trainer to SFT_MAX_STEPS (runner
# already wires both; config records max_steps/sft_epochs), steps
# {18, 45, 90} = {0.10, 0.25, 0.50} of the 181-step epoch (723 rows /
# batch 4). The FULL-training endpoint is REUSED, not re-run: the cube
# cells pofdesf_{qwen7b,olmo7b}_b0_ea0p4_w0p5_l0p2_es0p3_s0_fresh_data
# (SFT_EPOCHS=1 = 181 steps/round, complete + gated 2026-08-05, carry
# twin_raw + gate_raw) -> 6 new jobs (2 models x 3 step counts).
# Tags carry the step count (_st token) and the config carries
# max_steps; check_pofd_sanity's BUDGET branch gates style=sft, b0,
# sft_epochs=0, max_steps vs _st, base model vs slug, Action, natural
# labels, plus the shared twin/gate_raw/fresh/n_train sections.
# SMOKE REQUIRED (unlike cube/replay): the SFT max_steps path
# (sft_epochs=0 + max_steps>1 with style=sft) has NEVER run on the
# cluster -- every prior SFT/KL wave was epoch-based (audit of all 831
# pulled configs: (sft_epochs, max_steps) in {(1,1), (0,1)-dpo-only}).
# 1-job 3-round smoke at st18, gate with check_pofd_sanity, then the
# wave. .subs GENERATED from the CUBE_MODELS registry.
BUD_STEPS = [18, 45, 90]
BUD_MODELS = ["qwen7b", "olmo7b"]   # slugs into the CUBE_MODELS registry
BUD_EA = 0.4
BUD_ES = 0.3
BUD_SEEDS = [0]
ROW_BUD = ("{tag}, sft, 0, {seed}, 1, replace, 1.0, fixed, ab, "
           "{es}, 0.0, 0.5, loop, 0.0, {eps_ai}, {steps}")


def bud_tag(model, st, seed, prefix="pofdbud"):
    return (f"{prefix}_{model}_b0_ea{_num(BUD_EA)}_st{st}_{w_tok()}"
            f"_es{_num(BUD_ES)}_s{seed}_fresh_data")


def bud_rows(model):
    return [ROW_BUD.format(tag=bud_tag(model, st, s), seed=s,
                           es=f"{BUD_ES:g}", eps_ai=f"{BUD_EA:g}", steps=st)
            for st in BUD_STEPS for s in BUD_SEEDS]


BUD_SUB_TEMPLATE = """\
# HTCondor: ORDINARY-SFT TRAINING-BUDGET SWEEP, {model}{smk_note} --
# GENERATED by gen_pofd_sweep.py from the CUBE_MODELS spec (2026-08-06).
# Never edit this file by hand: edit the BUD_ block and rerun the script.
# {n_jobs} job(s): b0 ordinary sft with SFT_EPOCHS=0 and a per-round
# optimizer-step cap SFT_MAX_STEPS (queue col 16; steps {{18,45,90}} =
# {{0.10,0.25,0.50}} of the 181-step epoch) at ea=0.4, es=0.3, seed 0,
# W_PLAT=0.5, INNATE_LAMBDA=0.2, gamma=0, nested operator, fresh +
# replace, {n_rounds} rounds, movielens Action, WITH_TWIN=1. The
# full-training endpoint is the REUSED cube cell
# pofdesf_{model}_b0_ea0p4_w0p5_l0p2_es0p3_s0_fresh_data (181 steps) --
# never re-queued. Gate every pull with check_pofd_sanity (BUDGET
# branch: style/b0/sft_epochs=0/max_steps-vs-_st/base-model-vs-slug +
# shared twin/gate_raw/fresh sections).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) SFT_MAX_STEPS=$(steps) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 ANS_SAMPLE_K=16 ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 FRESH_EACH_ROUND=1 TRAIN_CAP=723 N_ROUNDS={n_rounds} EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=0 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 USE_LORA=1 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{model}_lora512_pofdbudget"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, steps from experiments/condor/configs_pofd_{key}.txt
"""


def bud_sub(model, smoke=False):
    key = f"{model}_budget_smoke" if smoke else f"{model}_budget"
    n_jobs = 1 if smoke else len(bud_rows(model))
    return BUD_SUB_TEMPLATE.format(
        model=model, key=key, n_jobs=n_jobs, n_rounds=3 if smoke else 30,
        smk_note=" SMOKE (3 rounds)" if smoke else "",
        **CUBE_MODELS[model])


# PER-AGENT FROZEN-CONTEXT ICL (2026-08-07, user), key qwen7b_iclf
# (+ qwen7b_iclf_smoke): is the context-mediated drift channel driven by
# the exemplars UPDATING, or by their mere presence? Qwen-only frozen
# loops on movielens Action with the runner's new ICL_SNAPSHOT_ROUND:
#   arms  k0  = no context        (ICL_K=0, snapshot unset)
#         dyn = dynamic context   (ICL_K=8, snapshot -1: legacy rebuild
#               every round, code path byte-identical to icl2/icls2)
#         fz0 = freeze at round 0 (ICL_K=8, ICL_SNAPSHOT_ROUND=0)
#         fz8 = freeze at round 8 (ICL_K=8, ICL_SNAPSHOT_ROUND=8:
#               dynamic through round 8, then each agent's round-8
#               context is reused VERBATIM -- identities, order,
#               profiles, displayed values; rounds <= 8 replay the
#               dynamic selection stream exactly, so fz8 and dyn are
#               bit-identical through round 8 at matched seeds)
#   x ea {0.1, 0.2, 0.4} x es {0, 0.1, 0.2, 0.3} x seeds
#   (RESCOPED 2026-08-07, user: "144 jobs?" -> one seed; the original
#   3-seed spec {0, 42, 43} = 144 cells was too many jobs. Seed 0 =
#   48 cells; extend ICLF_SEEDS + rerun for replicates)
#   W=0.5, lam=0.2, gamma=0, K=8 random live exemplars,
#   30 rounds, frozen weights (nothing trains), LOG_GENDER_GAPS=1,
#   WITH_TWIN=1. Every ICL_K>0 run saves icl_idx_raw/icl_val_raw
#   (exact exemplar ids + displayed values) and icl_ctx_log.json.gz
#   (the rendered context text), so the freeze guarantee is audited on
#   the actual prompt text.
# AUDIT 2026-08-07 (on-cluster, BY CONFIG FIELDS): 18 complete cells
# match the dial surface (icl2/icls2 k0+k8live at es {0, 0.2},
# ea {0.1,0.2,0.4}, seeds mostly 0) -- but ALL 18 lack the gender-gap
# telemetry this wave requires (LOG_GENDER_GAPS was never set in any
# icl wave) and the es=0 ones also lack the twin. NOT equivalent (the
# replay-wave exact-equivalence standard), so NOTHING is reused: all
# 144 cells queue. The old runs stay untouched under their own tags.
# SMOKE (2 jobs, 10 rounds, ea0.2 es0.2 s0): dyn + fz8 -- gates the
# never-run snapshot path AND the cross-run bit-identity-through-
# round-8 check (check_pofd_sanity compares a fz8 run to its _dyn_
# sibling automatically when both dirs are pulled side by side).
ICLF_ARMS = [("k0", 0, -1), ("fz0", 8, 0), ("fz8", 8, 8), ("dyn", 8, -1)]
ICLF_EAS = [0.1, 0.2, 0.4]
ICLF_ESS = [0.0, 0.1, 0.2, 0.3]
# rescoped 2026-08-07: one seed per cell (extend + rerun for replicates)
ICLF_SEEDS = [0]
ICLF_MODELS = ["qwen7b"]        # slugs into the CUBE_MODELS registry
ROW_ICLF = ("{tag}, frozen, 0, {seed}, 1, replace, 1.0, fixed, ab, "
            "{es}, 0.0, 0.5, loop, 0.0, {eps_ai}, {iclk}, {snap}")


def iclf_tag(model, arm, ea, es, seed, prefix="pofdiclf"):
    return (f"{prefix}_{model}_{arm}_ea{_num(ea)}_{w_tok()}_es{_num(es)}"
            f"_s{seed}")


def iclf_rows(model):
    return [ROW_ICLF.format(
        tag=iclf_tag(model, arm, ea, es, s), seed=s, es=f"{es:g}",
        eps_ai=f"{ea:g}", iclk=k, snap=snap)
        for arm, k, snap in ICLF_ARMS for ea in ICLF_EAS
        for es in ICLF_ESS for s in ICLF_SEEDS]


ICLF_SUB_TEMPLATE = """\
# HTCondor: PER-AGENT FROZEN-CONTEXT ICL, {model}{smk_note} -- GENERATED
# by gen_pofd_sweep.py from the CUBE_MODELS spec (2026-08-07). Never
# edit this file by hand: edit the ICLF_ block and rerun the script.
# {n_jobs} job(s): frozen-weights loops, arms k0 (no context) / dyn
# (rebuilt every round) / fz0 / fz8 (context frozen verbatim at the
# snapshot round), K=8 random live exemplars, {n_rounds} rounds,
# W_PLAT=0.5, INNATE_LAMBDA=0.2, gamma=0, movielens Action,
# LOG_GENDER_GAPS=1, WITH_TWIN=1. ICL_K and ICL_SNAPSHOT_ROUND ride
# the queue (cols 16-17). Saves icl_idx_raw/icl_val_raw +
# icl_ctx_log.json.gz. The 2026-08-07 audit found 18 dial-matching
# icl2/icls2 cells but NONE carry the required gg telemetry -- nothing
# reused, the full grid queues. Gate every pull with check_pofd_sanity
# (ICL-CTX section: self-exclusion, frozen-cache immutability on ids/
# vals AND rendered text, constant perplexity, gg + twin, fz8-vs-dyn
# prefix bit-identity).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action EPS_AI=$(eps_ai) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live LOG_GENDER_GAPS=1 WITH_TWIN=1 INNATE_LAMBDA=0.2 USE_LORA=0 FRESH_EACH_ROUND=0 TRAIN_CAP=723 N_ROUNDS={n_rounds} EPOCH_SIZE=100 BASE_MODEL={base_model} GEN_BATCH_SIZE=32 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{model}_pofdiclf"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, iclk, snap from experiments/condor/configs_pofd_{key}.txt
"""


def iclf_sub(model, smoke=False):
    key = f"{model}_iclf_smoke" if smoke else f"{model}_iclf"
    n_jobs = 2 if smoke else len(iclf_rows(model))
    return ICLF_SUB_TEMPLATE.format(
        model=model, key=key, n_jobs=n_jobs, n_rounds=10 if smoke else 30,
        smk_note=" SMOKE (dyn vs fz8, 10 rounds)" if smoke else "",
        **CUBE_MODELS[model])


# SFT-TO-ICL CONTEXT TRANSFER (2026-08-07, user), key qwen7b_ctf
# (+ qwen7b_ctf_smoke): does a FROZEN model transported into an SFT-
# shaped information environment reproduce the drift channel? Frozen
# Qwen recipients whose K=8 random exemplar context (frozen verbatim at
# round 0, ICL_CTX_SOURCE=donor) displays a DONOR population's
# opinions; the recipient population always starts from its own innate
# opinions. Canonical peer dose: ea=0.4, es=0.2, W=0.5, lam=0.2, 30
# rounds, gg telemetry + matched twin on. Matched arms differ ONLY in
# the displayed values (identities ride the recipient's own selection
# stream -- identical across arms at a matched seed, checker-enforced):
#   pri  = donor innate (pristine round-0 opinions; round -1)
#   b0r9 = the b0 (ordinary-SFT) fes donor's op_raw[9]
#   b1r9 = the b1 (forward SFT-KL) fes donor's op_raw[9]
# DONOR ROUND 9, FIXED FOR EVERY SEED (user spec): the three-seed mean
# population incremental gender R^2 of the b1 fes donors peaks at round
# 9 under the paper's cross-fitted taste-only protocol
# (plot_feature_endogenization_beta_final.incremental_r2; VERIFIED
# 2026-08-07: mean argmax = 9, per-seed argmaxes 9/15/23 -- no
# per-seed peak selection). NOTE the runner-protocol OLS full-design
# variant peaks at 13; the paper protocol is canonical for this
# quantity. Donors (audited complete on cluster + locally):
#   b1: pofdws2f_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2_s{seed}_fresh_data
#   b0: s0 under the reverse-era pofdws2_ tag (b0 = sft,
#       direction-free -- the fes reuse precedent); s42/43 pofdws2f_.
# RESCOPED AT BIRTH (user: "only do 1 seed for this as well"): the
# 3-seed spec (9 jobs) runs at seed 0 only -> 3 full jobs; extend
# CTF_SEEDS + rerun for replicates. Smoke = the same 3 arms at seed 0,
# 10 rounds (gates the donor-load path + cross-arm identity check).
# Provenance saved per run: donor tag/round/sha256 in config.json, the
# full donor vector + donor gg/incremental-R^2 + realized context gap
# in trajectory.pt, per-agent ids/values/rendered text as in the iclf
# wave. check_pofd_sanity CTF-DONOR section verifies hash, exact
# value-indexing, pri==innate, donor re-derivation, cross-arm
# identity, and diagnostics.
CTF_ARMS = [("pri", "b1", -1), ("b0r9", "b0", 9), ("b1r9", "b1", 9)]
CTF_SEEDS = [0]   # rescoped at birth; spec grid was {0, 42, 43}
CTF_EA = 0.4
CTF_ES = 0.2
ROW_CTF = ("{tag}, frozen, 0, {seed}, 1, replace, 1.0, fixed, ab, "
           "{es}, 0.0, 0.5, loop, 0.0, {eps_ai}, 8, 0, {donor}, {dround}")


def ctf_donor_tag(which, seed):
    if which == "b1":
        return (f"pofdws2f_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2_s{seed}"
                f"_fresh_data")
    fam = "pofdws2_" if seed == 0 else "pofdws2f_"
    return f"{fam}qwen7b_b0_ea0p4_w0p5_l0p2_es0p2_s{seed}_fresh_data"


def ctf_tag(arm, seed, prefix="pofdctf"):
    return (f"{prefix}_qwen7b_{arm}_ea{_num(CTF_EA)}_{w_tok()}"
            f"_es{_num(CTF_ES)}_s{seed}")


def ctf_rows(prefix="pofdctf"):
    return [ROW_CTF.format(
        tag=ctf_tag(arm, s, prefix), seed=s, es=f"{CTF_ES:g}",
        eps_ai=f"{CTF_EA:g}", donor=ctf_donor_tag(which, s), dround=dr)
        for arm, which, dr in CTF_ARMS for s in CTF_SEEDS]


CTF_SUB_TEMPLATE = """\
# HTCondor: SFT-TO-ICL CONTEXT TRANSFER, qwen7b{smk_note} -- GENERATED
# by gen_pofd_sweep.py from the CUBE_MODELS spec (2026-08-07). Never
# edit this file by hand: edit the CTF_ block and rerun the script.
# {n_jobs} job(s): frozen Qwen recipients, K=8 random context FROZEN at
# round 0 displaying DONOR opinions (ICL_CTX_SOURCE=donor; arms pri =
# donor innate, b0r9/b1r9 = the b0/b1 fes donor's op_raw[9] -- round 9
# = the verified three-seed-mean peak of population incremental gender
# R^2, fixed for every seed). ea=0.4, es=0.2, W=0.5, lam=0.2,
# {n_rounds} rounds, movielens Action, LOG_GENDER_GAPS=1, WITH_TWIN=1.
# Donor dir + round ride the queue (cols 18-19); donor tag/round/sha256
# land in config.json, the donor vector + gg diagnostics in
# trajectory.pt. Recipient populations start from their OWN innate
# opinions -- only the displayed context values change across arms.
# Gate every pull with check_pofd_sanity (CTF-DONOR + ICL-CTX
# sections); pull all three arms side by side so the cross-arm
# identity check runs.
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action EPS_AI=$(eps_ai) ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_CTX_DONOR=/home/gsmithline/perfsim/runs/pokec_gated_lm/$(donor) ICL_CTX_DONOR_ROUND=$(dround) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=donor LOG_GENDER_GAPS=1 WITH_TWIN=1 INNATE_LAMBDA=0.2 USE_LORA=0 FRESH_EACH_ROUND=0 TRAIN_CAP=723 N_ROUNDS={n_rounds} EPOCH_SIZE=100 BASE_MODEL={base_model} GEN_BATCH_SIZE=32 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_qwen7b_pofdctf"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, iclk, snap, donor, dround from experiments/condor/configs_pofd_{key}.txt
"""


def ctf_sub(smoke=False):
    key = "qwen7b_ctf_smoke" if smoke else "qwen7b_ctf"
    return CTF_SUB_TEMPLATE.format(
        key=key, n_jobs=len(ctf_rows()), n_rounds=10 if smoke else 30,
        smk_note=" SMOKE (3 arms, 10 rounds)" if smoke else "",
        **CUBE_MODELS["qwen7b"])


# ICLX RETRY (2026-08-07): the 11 cells the g106 black hole killed in the
# first iclx submission (each burned 5 starts alternating between broken
# g106 and transient failures, then periodic_remove deleted it; g106 is
# now hard-excluded from every generated sub). The user did not want the
# 40 completed cells to re-queue even as idempotent no-ops, so these keys
# queue EXACTLY the missing cells: qwen7b_iclf_retry (10) +
# qwen7b_ctf_retry (1), umbrella qwen7b_iclx_retry. Rows are FILTERED
# from the canonical generators, so dials stay byte-identical to the
# main wave; once the cells complete, resubmitting is a pure no-op.
ICLX_RETRY_TAGS = frozenset({
    "pofdiclf_qwen7b_k0_ea0p1_w0p5_l0p2_es0p2_s0",
    "pofdiclf_qwen7b_k0_ea0p4_w0p5_l0p2_es0p1_s0",
    "pofdiclf_qwen7b_fz0_ea0p1_w0p5_l0p2_es0_s0",
    "pofdiclf_qwen7b_fz0_ea0p1_w0p5_l0p2_es0p2_s0",
    "pofdiclf_qwen7b_fz0_ea0p1_w0p5_l0p2_es0p3_s0",
    "pofdiclf_qwen7b_fz0_ea0p4_w0p5_l0p2_es0p2_s0",
    "pofdiclf_qwen7b_fz8_ea0p2_w0p5_l0p2_es0_s0",
    "pofdiclf_qwen7b_dyn_ea0p1_w0p5_l0p2_es0p2_s0",
    "pofdiclf_qwen7b_dyn_ea0p2_w0p5_l0p2_es0p3_s0",
    "pofdiclf_qwen7b_dyn_ea0p4_w0p5_l0p2_es0p2_s0",
    "pofdctf_qwen7b_b0r9_ea0p4_w0p5_l0p2_es0p2_s0",
})


def iclx_retry_rows(which):
    rows = iclf_rows("qwen7b") if which == "iclf" else ctf_rows()
    return [r for r in rows if r.split(",")[0] in ICLX_RETRY_TAGS]


def iclx_retry_sub(which):
    if which == "iclf":
        return ICLF_SUB_TEMPLATE.format(
            model="qwen7b", key="qwen7b_iclf_retry",
            n_jobs=len(iclx_retry_rows("iclf")), n_rounds=30,
            smk_note=" RETRY (g106 casualties only)",
            **CUBE_MODELS["qwen7b"])
    return CTF_SUB_TEMPLATE.format(
        key="qwen7b_ctf_retry", n_jobs=len(iclx_retry_rows("ctf")),
        n_rounds=30, smk_note=" RETRY (g106 casualties only)",
        **CUBE_MODELS["qwen7b"])


# QWEN/OLMO SEED-REPLICATION CORE (2026-08-07, user), umbrella key
# qwen_olmo_seedcore = qwen7b_seedcore + olmo7b_seedcore (+ the 1-job
# qwen7b_seedcore_smoke): seed replicates for the three headline
# surfaces, so the key claims carry across-seed means/CIs instead of
# single-seed curves. EXACTLY 34 audited-missing jobs (qwen 16 + olmo
# 18); one config file + generated .sub per model, with N_ROUNDS /
# INNATE_LAMBDA / WITH_TWIN / ANS_SAMPLE_K queue-fed (cols 16-19) so the
# three sub-families share a sub. AUDIT 2026-08-07 (local corpus, BY
# CONFIG FIELDS -- all 895 pulled run dirs scanned on model/beta/kdir/
# seed/rounds/W/lam/ea/es/style/epochs/regime):
#   1) ONE-UPDATE RETENTION (12): both models x b {0.1,0.2,0.5} x
#      s {42,43}, N_ROUNDS=1, in the Figure-1b environment (ea=0.4,
#      W=0.5, lam=0.2, es=0.1, forward KL, fresh + replace, sft_kl,
#      same training settings as the cube). NO 1-round run exists
#      anywhere (audit: zero configs with n_rounds=1 besides pofdtch_
#      teachers, a different protocol) -> all 12 queue. The seed-0
#      endpoints are REUSED, not re-run: op_raw[0] of the 30-round
#      pofdesf_{model}_b{0p1,0p2,0p5}_ea0p4_w0p5_l0p2_es0p1_s0 runs
#      (round 0 of a 30-round run IS the one-update state -- trajectory
#      prefix property, validated by the iclf fz8/dyn bit-identity).
#      The b0/b1 retention endpoints at s42-45 come free the same way
#      from the main-peer replicates below. NEW prefix pofdret_ (not
#      pofdesf_): a 1-round trajectory under an esf tag would collide
#      with any future 30-round run of the same cell (the idempotent
#      exec skips on trajectory presence, so the full run would no-op
#      against the stub). _rt is not a token -- n_rounds=1 is pinned by
#      the prefix + config and gated by the checker's RETENTION branch.
#   2) DIRECT TRANSMISSION (8): both models x b {0,1} x s {42,43},
#      30 rounds, W=1, lam=0, es=0 (no peer step ever), ea=0.4, b0 ->
#      ordinary sft (direction-free), b1 -> forward SFT-KL. Tags join
#      the existing HOME families at new seeds (fes convention):
#      b0 -> pofd_ (anchors pofd_{model}_b0_ea0p4_s0_fresh_data), b1 ->
#      pofdw1f_ (anchors pofdw1f_{model}_b1_ea0p4_s0_fresh_data).
#      Anchor-matching env: ANS_SAMPLE_K=0 (probe off in the w1f wave;
#      predates the probe in the b0 wave) and WITH_TWIN=0 (both anchors
#      carry no twin -- at W=1/lam=0/es=0 the no-AI twin is frozen
#      innate, reconstructible offline). KL_DIRECTION=forward rides the
#      shared env; b0 rows ignore it (no KL term -- cube precedent, the
#      config records it either way).
#   3) MAIN PEER ENVIRONMENT (14): W=0.5, lam=0.2, ea=0.4, es=0.1,
#      30 rounds -- every model x b {0,1} cell brought to seeds
#      {0,42,43,44,45}. On-disk audit: qwen b1 has s0 (esf) + s42/s43
#      (fex wave), qwen b0 / olmo b0 / olmo b1 have s0 only (cube wave)
#      -> missing = qwen b1 {44,45}, qwen b0 {42..45}, olmo b0 {42..45},
#      olmo b1 {42..45}. Tags stay pofdesf_ via cube_tag (the es=0.1
#      home family); seeds 44/45 are FIRST USE project-wide.
# Statistical use (logged in BATCHES.md): (1) per-beta one-update
# retention mean +/- CI over 3 seeds per model; (2)/(3) across-seed
# means and paired within-seed contrasts (b1 - b0) for the direct and
# main-peer drift/gap endpoints, n=3 (direct) / n=5 (main peer) seeds.
# No smoke for sub-families 2/3 (every dial combination re-runs a
# validated env at a new seed); sub-family 1 needs one: N_ROUNDS=1 has
# NEVER run on the cluster -- the 1-round pofdretsmk_ smoke doubles as
# a physics check, since its round 0 should match op_raw[0] of the
# existing 30-round b0p5 s0 esf run (same seed/env; bit-exact only on
# matching GPU hardware, the 2026-08-07 iclf finding).
SC_MODELS = ["qwen7b", "olmo7b"]    # slugs into the CUBE_MODELS registry
SC_RET_BETAS = [0.1, 0.2, 0.5]
SC_RET_SEEDS = [42, 43]
SC_DIR_BETAS = [0.0, 1.0]
SC_DIR_SEEDS = [42, 43]
SC_MAIN_SEEDS = [0, 42, 43, 44, 45]
# main-peer cells already complete on disk (audit 2026-08-07, config
# fields + 30 trajectory rounds): (model, beta, seed)
SC_MAIN_EXISTING = {("qwen7b", 1.0, 0), ("qwen7b", 1.0, 42),
                    ("qwen7b", 1.0, 43), ("qwen7b", 0.0, 0),
                    ("olmo7b", 0.0, 0), ("olmo7b", 1.0, 0)}
ROW_SC = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, ab, "
          "{es}, 0.0, {w}, loop, 0.0, {eps_ai}, {nrounds}, {lam}, "
          "{wtwin}, {ansk}")


def sc_ret_tag(model, b, seed, prefix="pofdret"):
    return (f"{prefix}_{model}_b{_num(b)}_ea0p4_{w_tok()}_es0p1_s{seed}"
            f"_fresh_data")


def sc_rows(model):
    rows = []
    # 1) one-update retention curve (N_ROUNDS=1, Figure-1b environment)
    for b in SC_RET_BETAS:
        for s in SC_RET_SEEDS:
            rows.append(ROW_SC.format(
                tag=sc_ret_tag(model, b, s), style="sft_kl", beta=f"{b:g}",
                seed=s, es="0.1", w="0.5", eps_ai="0.4", nrounds=1,
                lam="0.2", wtwin=1, ansk=16))
    # 2) direct transmission (W=1, lam=0, es=0 -- anchor-matching env)
    for b in SC_DIR_BETAS:
        for s in SC_DIR_SEEDS:
            rows.append(ROW_SC.format(
                tag=tag_of(model, b, 0.4, s,
                           prefix="pofd" if b == 0 else "pofdw1f"),
                style="sft" if b == 0 else "sft_kl", beta=f"{b:g}",
                seed=s, es="0", w="1.0", eps_ai="0.4", nrounds=30,
                lam="0.0", wtwin=0, ansk=0))
    # 3) main peer environment replicates (cube env, es=0.1 home family)
    for b in SC_DIR_BETAS:
        for s in SC_MAIN_SEEDS:
            if (model, b, s) in SC_MAIN_EXISTING:
                continue
            rows.append(ROW_SC.format(
                tag=cube_tag(model, b, 0.4, 0.1, s),
                style="sft" if b == 0 else "sft_kl", beta=f"{b:g}",
                seed=s, es="0.1", w="0.5", eps_ai="0.4", nrounds=30,
                lam="0.2", wtwin=1, ansk=16))
    return rows


SC_SUB_TEMPLATE = """\
# HTCondor: QWEN/OLMO SEED-REPLICATION CORE, {model}{smk_note} --
# GENERATED by gen_pofd_sweep.py from the CUBE_MODELS spec (2026-08-07).
# Never edit this file by hand: edit the SC_ block and rerun the script.
# {n_jobs} job(s) in three sub-families sharing this sub via queue-fed
# N_ROUNDS/INNATE_LAMBDA/WITH_TWIN/ANS_SAMPLE_K (cols 16-19):
#   1) one-update retention: b {{0.1,0.2,0.5}} x s {{42,43}}, N_ROUNDS=1,
#      Figure-1b env (ea0p4, W=0.5, lam=0.2, es=0.1, forward KL) --
#      pofdret_ tags; seed-0 endpoints reused from the 30-round esf runs
#   2) direct transmission: b {{0,1}} x s {{42,43}}, 30 rounds, W=1,
#      lam=0, es=0, ea0p4, twin/probe OFF (anchor-matching) -- tags join
#      pofd_ (b0, ordinary sft) / pofdw1f_ (b1, forward SFT-KL)
#   3) main peer env: b {{0,1}} to seeds {{0,42,43,44,45}}, 30 rounds,
#      ea0p4/es0p1 cube env -- pofdesf_ tags; existing cells not queued
# Every trained row: fresh adapter + replace data, movielens Action,
# LoRA-512, SFT_EPOCHS=1. Gate every pull with check_pofd_sanity
# (RETENTION branch: n_rounds=1 + slug/beta/style/env; DIRECT: W=1
# exact-copy replay + b-token style gate; main peer: cube _b gate +
# SOCIAL twin/peer-alive; seed + ea tokens and base-model slug are now
# gated on every family).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) KL_DIRECTION=forward WITH_TWIN=$(wtwin) INNATE_LAMBDA=$(lam) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 FRESH_EACH_ROUND=1 TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 USE_LORA=1 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{model}_lora512_pofdseedcore"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, nrounds, lam, wtwin, ansk from experiments/condor/configs_pofd_{key}.txt
"""


def sc_sub(model, smoke=False):
    key = f"{model}_seedcore_smoke" if smoke else f"{model}_seedcore"
    n_jobs = 1 if smoke else len(sc_rows(model))
    return SC_SUB_TEMPLATE.format(
        model=model, key=key, n_jobs=n_jobs,
        smk_note=(" SMOKE (1-round pofdretsmk_ cell; round 0 comparable "
                  "to the 30-round esf b0p5 s0 run)" if smoke else ""),
        **CUBE_MODELS[model])


# FINAL REPLICATION WAVE (2026-08-07, user), umbrella key
# qwen_olmo_finalfill = qwen7b_finalfill_sfticl (18) +
# qwen7b_finalfill_replay (16) + qwen7b_finalfill_corners (8) +
# the UNCHANGED olmo7brom_fe wave (olmo7brom_fes 9 + olmo7brom_fef 3)
# = 54 queued jobs. CONFIG-FIELD AUDIT 2026-08-07 (local corpus 902
# dirs + on-cluster tag/trajectory scan): 46 of the 54 are genuinely
# missing; 8 queue as instant idempotent no-ops BY USER DECISION
# (2026-08-07): the 2 corner cells pofdw2f_qwen7b_b1_ea0p4_..._s{42,43}
# (complete since the fes wave, gated) and the 6 complete Romance cells
# (b0 x3 + k0 x3, found complete on cluster, pulled + gated 7/7 with
# their smoke). NO DPO replication in this wave by design.
#   1) qwen7b_finalfill_sfticl (18): the SFT-vs-ICL gate-dose overlay
#      (plot_sft_icl_gate_overlay.py) at replication seeds {42,43} --
#      env3 (W=0.5, lam=0.2, es=0.2, 30 rounds, movielens Action):
#      6 forward SFT-KL b1 cells at ea {0.05,0.1,0.2} (pofdws2f_ tags,
#      fes env: LoRA-512 fresh+replace, ANS_SAMPLE_K=16) + 12 frozen
#      live-ICL cells k {8,32} x ea {0.05,0.1,0.2} (pofdicls2_ tags,
#      icls2 env: USE_LORA=0, FRESH_EACH_ROUND=0, dynamic context,
#      ICL_SELECT=random, probe off). ea=0.4 cells REUSED at both
#      seeds: ws2f b1 = fes wave (the ctf donors), k8live = feik wave,
#      k32live = icls2x wave -- all audited config-equivalent + gated.
#      One sub serves both families: USE_LORA / FRESH_EACH_ROUND /
#      ICL_K / ICL_DAYS / ICL_CTX_SOURCE / ANS_SAMPLE_K ride the queue
#      (cols 16-21). eps>0 instantiates the matched twin in every run.
#   2) qwen7b_finalfill_replay (16): the replay1 grid at seeds {42,43}
#      -- b {0,1} x rf {0.25,0.5,0.75,1}, ea=0.4, es=0.1, byte-identical
#      dials to the seed-0 wave (rows share ROW_RPL/rpl_tag; env copies
#      the replay1 sub). rf=0 cells REUSED: at s42/43 they are the
#      seedcore pofdesf_ replicates (completed + gated 2026-08-07),
#      at s0 the cube/esf cells -- rf=0 is byte-identical to the plain
#      replace loop (replay path guarded by replay_frac > 0).
#   3) qwen7b_finalfill_corners (8): the (ea, es) corner cells of the
#      b1 forward surface at seeds {42,43} -- ea {0.05,0.4} x
#      es {0,0.3}, cube-family tags via cube_tag (es=0 -> pofdw2f_,
#      es=0.3 -> pofdesf_), cube env (WITH_TWIN=1). The two
#      (ea0p4, es0) tags already hold complete runs -- queued anyway
#      per user decision; the idempotent exec no-ops them.
#   4) olmo7brom_fe: NOT modified, NOT duplicated -- the umbrella
#      expands to the same olmo7brom_fes/olmo7brom_fef targets, whose
#      configs/tags/validation stay byte-identical. AUDIT: 6/12 cells
#      complete on cluster (b0 s{0,42,43} + k0 s{0,42,43}) + the smoke,
#      all pulled + gated 7/7 PASS 2026-08-07; the b0p5/b1 trios are
#      6-file shells with no trajectory (they re-run; the idempotent
#      exec fills the shells). Romance smoke ALREADY PASSED -- the
#      smoke-first step is satisfied; resubmitting it is a no-op.
# Statistical use (BATCHES.md): across-seed (n=3) means/CIs for the
# gate-dose overlay curves (SFT vs live-ICL K8/K32), the replay-dose
# displacement curves with paired within-seed b1-b0 contrasts, the
# corner cells of the (ea, es) surface, and the Romance fe beta axis
# (b {0,0.5,1} + frozen control); no per-seed selection anywhere.
# Checker: NO new logic needed -- every cell lands in a fully-gated
# existing family (cube-prefix _b gates, icl arm gates, REPLAY section,
# Romance branch, universal seed/ea/slug gates); validated by fresh
# fixtures exercising the NEW tag shapes + the full-corpus re-gate.
FF_SEEDS = [42, 43]
FF_SFT_EAS = [0.05, 0.1, 0.2]     # ea=0.4 reused (fes / feik / icls2x)
FF_ICL_KS = [8, 32]
FF_CORNER_EAS = [0.05, 0.4]
FF_CORNER_ESS = [0.0, 0.3]
ROW_FFSI = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, ab, "
            "0.2, 0.0, 0.5, loop, 0.0, {eps_ai}, {iclk}, {icldays}, "
            "{iclsrc}, {uselora}, {fresh}, {ansk}")


def ffsi_rows():
    rows = []
    for ea in FF_SFT_EAS:
        for s in FF_SEEDS:
            rows.append(ROW_FFSI.format(
                tag=(f"pofdws2f_qwen7b_b1_ea{_num(ea)}_{ws_tok()}"
                     f"_s{s}_fresh_data"),
                style="sft_kl", beta="1", seed=s, eps_ai=f"{ea:g}",
                iclk=0, icldays=0, iclsrc="live", uselora=1, fresh=1,
                ansk=16))
    for ea in FF_SFT_EAS:
        for k in FF_ICL_KS:
            for s in FF_SEEDS:
                rows.append(ROW_FFSI.format(
                    tag=f"pofdicls2_qwen7b_{ws_tok()}_ea{_num(ea)}_k{k}live_s{s}",
                    style="frozen", beta="0", seed=s, eps_ai=f"{ea:g}",
                    iclk=k, icldays=0, iclsrc="live", uselora=0, fresh=0,
                    ansk=0))
    return rows


def ffrp_rows():
    return [ROW_RPL.format(
        tag=rpl_tag("qwen7b", b, rf, s),
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=s,
        es=f"{RPL_ES:g}", eps_ai=f"{RPL_EA:g}", rfrac=f"{rf:g}")
        for b in RPL_BETAS for rf in RPL_FRACS for s in FF_SEEDS if rf > 0]


def ffc_rows():
    return [ROW_CUBE.format(
        tag=cube_tag("qwen7b", 1.0, ea, es, s), style="sft_kl", beta="1",
        seed=s, es=f"{es:g}", eps_ai=f"{ea:g}")
        for ea in FF_CORNER_EAS for es in FF_CORNER_ESS for s in FF_SEEDS]


FFSI_SUB_TEMPLATE = """\
# HTCondor: FINAL-FILL SFT-vs-ICL GATE-DOSE REPLICATES, qwen7b --
# GENERATED by gen_pofd_sweep.py from the FF_ block (2026-08-07). Never
# edit this file by hand: edit the FF_ block and rerun the script.
# {n_jobs} jobs at seeds {{42,43}} in env3 (W=0.5, lam=0.2, es=0.2,
# 30 rounds): 6 forward SFT-KL b1 cells at ea {{0.05,0.1,0.2}}
# (pofdws2f_ tags, fes env) + 12 frozen live-ICL cells k {{8,32}} x the
# same gates (pofdicls2_ tags, icls2 env). The ea=0.4 cells at both
# seeds are REUSED (fes / feik / icls2x waves), never re-queued. One
# sub serves both families: USE_LORA / FRESH_EACH_ROUND / ICL_K /
# ICL_DAYS / ICL_CTX_SOURCE / ANS_SAMPLE_K ride the queue (cols 16-21).
# These cells complete the plot_sft_icl_gate_overlay.py dose curves at
# 3 seeds per point. Gate every pull with check_pofd_sanity (ws2f: cube
# _b gate + SOCIAL twin/peer-alive; icls2: frozen + arm gates).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> qwen7b_finalfill_sfticl
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) ICL_K=$(iclk) ICL_DAYS=$(icldays) ICL_SELECT=random ICL_CTX_SOURCE=$(iclsrc) KL_DIRECTION=forward USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 INNATE_LAMBDA=0.2 TRAIN_CAP=723 N_ROUNDS=30 EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_qwen7b_pofdffill"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, iclk, icldays, iclsrc, uselora, fresh, ansk from experiments/condor/configs_pofd_qwen7b_finalfill_sfticl.txt
"""

FFRP_SUB_TEMPLATE = """\
# HTCondor: FINAL-FILL REPLAY REPLICATES, qwen7b -- GENERATED by
# gen_pofd_sweep.py from the FF_ block (2026-08-07). Never edit this
# file by hand: edit the FF_ block and rerun the script.
# {n_jobs} cells: the replay1 grid (beta {{0,1}} x replay_frac
# {{0.25,0.5,0.75,1}}, ea=0.4, es=0.1, W=0.5, lam=0.2, 30 rounds,
# fresh + replace, WITH_TWIN=1) at replication seeds {{42,43}} --
# dials byte-identical to the seed-0 wave (shared ROW_RPL/rpl_tag; env
# copies the replay1 sub). The rf=0 cells are REUSED: at s42/43 the
# seedcore pofdesf_ replicates (complete + gated 2026-08-07), at s0
# the cube/esf cells. Gate every pull with check_pofd_sanity (REPLAY
# section: bit-exact batch composition; _b/_rf tokens, style, forward
# direction and seed cross-checked).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> qwen7b_finalfill_replay
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) REPLAY_FRAC=$(rfrac) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 ANS_SAMPLE_K=16 ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 FRESH_EACH_ROUND=1 TRAIN_CAP=723 N_ROUNDS=30 EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 USE_LORA=1 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_qwen7b_lora512_pofdreplay1"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai, rfrac from experiments/condor/configs_pofd_qwen7b_finalfill_replay.txt
"""

FFC_SUB_TEMPLATE = """\
# HTCondor: FINAL-FILL (ea, es) CORNER REPLICATES, qwen7b -- GENERATED
# by gen_pofd_sweep.py from the FF_ block (2026-08-07). Never edit this
# file by hand: edit the FF_ block and rerun the script.
# {n_jobs} cells: forward SFT-KL b1 at ea {{0.05,0.4}} x es {{0,0.3}} x
# seeds {{42,43}} -- W=0.5, lam=0.2, 30 rounds, fresh + replace,
# WITH_TWIN=1, cube env and cube-family tags (es=0 -> pofdw2f_,
# es=0.3 -> pofdesf_). The two (ea0p4, es0) tags already hold complete
# fes-wave runs -- queued anyway per user decision 2026-08-07; the
# idempotent exec no-ops them in seconds. Gate every pull with
# check_pofd_sanity (cube _b gate; es=0 exact replay, es>0 peer gate +
# twin).
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> qwen7b_finalfill_corners
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = {mem}
request_disk      = {disk}
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.Machine =!= MY.LastRemoteHost) && (TARGET.Machine != "g106.internal.cluster.is.localnet") && (TARGET.Machine != "i104.internal.cluster.is.localnet")

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action {extra_env}EPS_AI=$(eps_ai) KL_DIRECTION=forward WITH_TWIN=1 INNATE_LAMBDA=0.2 ANS_SAMPLE_K=16 ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 FRESH_EACH_ROUND=1 TRAIN_CAP=723 N_ROUNDS=30 EPOCH_SIZE=100 BASE_MODEL={base_model} SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 USE_LORA=1 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH={ppl_batch} SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_qwen7b_lora512_pofdffcorners"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, eps_ai from experiments/condor/configs_pofd_qwen7b_finalfill_corners.txt
"""


SMOKE = ("qwen7b", 0.5, 0.1, 0)   # model, beta, eps_ai, seed -- exercises sft_kl


def _num(v):
    """0.05 -> '0p05', 0.5 -> '0p5', 1.0 -> '1', 0.0 -> '0'"""
    s = f"{v:g}"
    return s.replace(".", "p")


def tag_of(model, beta, eps_ai, seed, prefix="pofd"):
    return f"{prefix}_{model}_b{_num(beta)}_ea{_num(eps_ai)}_s{seed}_fresh_data"


def rows_for(model, seeds, prefix="pofd"):
    out = []
    for seed in seeds:
        for beta in BETAS:
            for eps_ai in EPS_AIS:
                out.append(ROW.format(
                    tag=tag_of(model, beta, eps_ai, seed, prefix),
                    style="sft" if beta == 0 else "sft_kl",
                    beta=f"{beta:g}", seed=seed, eps_ai=f"{eps_ai:g}"))
    return out


def esf_tag(w, es, seed, ea=0.4):
    tok = f"w{_num(w)}_l0p2" + (f"_es{_num(es)}" if es > 0 else "")
    return f"pofdesf_qwen7b_b1_ea{_num(ea)}_{tok}_s{seed}_fresh_data"


def esf_rows(points, seeds, ea=0.4):
    return [ROW_ESF.format(tag=esf_tag(w, es, s, ea), seed=s,
                           es=f"{es:g}", w=f"{w:.1f}", eps_ai=f"{ea:g}")
            for (w, es) in points for s in seeds]


def pfrac_rows(model):
    out = []
    for seed in SEEDS:
        for pf in PFRACS:
            for eps_ai in EPS_AIS:
                tag = (f"pofdpf_{model}_b0_ea{_num(eps_ai)}"
                       f"_pf{_num(pf)}_s{seed}_fresh_data")
                out.append(ROW_PF.format(
                    tag=tag, seed=seed, eps_ai=f"{eps_ai:g}", pfrac=f"{pf:g}"))
    return out


def bp_rows():
    out = []
    for seed in SEEDS:
        for beta in BP_BETAS:
            for pf in PFRACS:
                for eps_ai in BP_EPS:
                    tag = (f"pofdbp_{BP_MODEL}_b{_num(beta)}_ea{_num(eps_ai)}"
                           f"_pf{_num(pf)}_s{seed}_fresh_data")
                    out.append(ROW_BP.format(tag=tag, beta=f"{beta:g}", seed=seed,
                                             eps_ai=f"{eps_ai:g}", pfrac=f"{pf:g}"))
    return out


def icl_rows(model=ICL_MODEL):
    out = []
    for seed in SEEDS:
        for arm, k, days, src in ICL_ARMS:
            for eps_ai in EPS_AIS:
                tag = f"pofdicl_{model}_ea{_num(eps_ai)}_{arm}_s{seed}"
                out.append(ROW_ICL.format(tag=tag, seed=seed, eps_ai=f"{eps_ai:g}",
                                          iclk=k, icldays=days, iclsrc=src))
    return out


def dpo_rows(model=DPO_MODEL):
    out = []
    for seed in SEEDS:
        for db in DPO_BETAS:
            for fb in DPO_FEEDBACKS:
                for eps_ai in EPS_AIS:
                    tag = (f"pofddpo_{model}_db{_num(db)}_ea{_num(eps_ai)}"
                           f"_{fb}_s{seed}_fresh")
                    out.append(ROW_DPO.format(tag=tag, seed=seed,
                                              eps_ai=f"{eps_ai:g}",
                                              rlhf=fb, dpobeta=f"{db:g}"))
    return out


def dpon_rows():
    out = []
    for seed in SEEDS:
        for db in DPO_BETAS:
            for fb in DPO_FEEDBACKS:
                for eps_ai in DPON_EPS:
                    tag = (f"pofddpon_{DPO_MODEL}_db{_num(db)}_ea{_num(eps_ai)}"
                           f"_{fb}_s{seed}_fresh")
                    out.append(ROW_DPO.format(tag=tag, seed=seed,
                                              eps_ai=f"{eps_ai:g}",
                                              rlhf=fb, dpobeta=f"{db:g}"))
    return out


def w_tok():
    return f"w{_num(W_WPLAT)}_l{_num(W_LAMBDA)}"


def w_rows(prefix="pofdw", model=W_MODEL):
    out = []
    for seed in SEEDS:
        for beta in BETAS:
            for eps_ai in EPS_AIS:
                tag = (f"{prefix}_{model}_b{_num(beta)}_ea{_num(eps_ai)}"
                       f"_{w_tok()}_s{seed}_fresh_data")
                out.append(ROW_W.format(
                    tag=tag, style="sft" if beta == 0 else "sft_kl",
                    beta=f"{beta:g}", seed=seed, eps_ai=f"{eps_ai:g}"))
    return out


def wdpo_rows(eps_list, prefix):
    out = []
    for seed in SEEDS:
        for db in DPO_BETAS:
            for fb in DPO_FEEDBACKS:
                for eps_ai in eps_list:
                    tag = (f"{prefix}_{W_MODEL}_db{_num(db)}_ea{_num(eps_ai)}"
                           f"_{fb}_{w_tok()}_s{seed}_fresh")
                    out.append(ROW_WDPO.format(tag=tag, seed=seed,
                                               eps_ai=f"{eps_ai:g}",
                                               rlhf=fb, dpobeta=f"{db:g}"))
    return out


def ws_tok():
    return f"{w_tok()}_es{_num(W_EPS_SOCIAL)}"


def ws_rows(prefix="pofdws", model=W_MODEL):
    out = []
    for seed in SEEDS:
        for beta in BETAS:
            for eps_ai in EPS_AIS:
                tag = (f"{prefix}_{model}_b{_num(beta)}_ea{_num(eps_ai)}"
                       f"_{ws_tok()}_s{seed}_fresh_data")
                out.append(ROW_WS.format(
                    tag=tag, style="sft" if beta == 0 else "sft_kl",
                    beta=f"{beta:g}", seed=seed, eps_ai=f"{eps_ai:g}"))
    return out


# =========================================================================
# REFERENCE REPLAY AT THE WU BOUNDARY (2026-08-22, ref_replay[_smoke]).
# A Qwen2.5-7B pilot on the EXISTING clean QWU surface: every round the
# training set is rebuilt FULL SIZE (all 723 rows), but only a fraction q
# of the rows carry the LIVE population value. The remaining 1-q rows
# carry a PINNED frozen-Qwen prediction b_i for that same agent.
#
# WHAT IT ASKS. The subsample wave cut how MANY agents the optimizer sees
# (14..542 of 723) and confounded two things: less data AND fewer
# optimizer steps. Reference replay cuts neither. Every round trains on
# 723 rows with the same 181 steps at batch 4; the only change is WHOSE
# label a row carries -- the live loop, or the model's own entering
# prior. So q dials the amount of FEEDBACK in the training signal while
# holding data volume and compute exactly fixed, which is the comparison
# the subsample arms could not make.
#
# THE SURFACE IS NOT NEW. It is the completed Wu-boundary b0 cell,
# unchanged: movielens Action, 723 agents, Qwen/Qwen2.5-7B-Instruct,
# ordinary SFT (lambda_KL = 0), W_PLAT = 1, INNATE_LAMBDA = 1, BOTH
# gates genuinely all_open, fresh LoRA r512 every round, 1 epoch at
# batch 4, LR 5e-5, 100 rounds, seed 0, matched twin, greedy eval-mode
# serving, gamma = 0, the exact H100 SKU. Only REF_REPLAY_* is new.
#
# q = 1 IS NOT QUEUED. At q = 1 every row is live, which is EXACTLY
# ordinary SFT -- byte-identical to the completed cell
# pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100 (the same cell the
# subsample wave hangs its 100% arm off). Queuing it would re-run a
# finished job to reproduce a result that already exists, so the ladder
# queues four cells and NAMES the fifth.
#
# THE FROZEN VECTOR IS NOT RE-EXTRACTED EITHER. b is the canonical
# frozen Qwen2.5-7B-Instruct K = D = 0 served vector, sha256
# 1674ee5f...da30bb -- the same constant the mechanism audit derived and
# that check_pofd_sanity, the dose analyzer, the subsample analyzer and
# the adapter-KL probe already pin. A frozen K=D=0 model never sees the
# population, so its prediction is a CONSTANT across rounds and across
# every k / eps_social / W cell -- which is why one archived run can
# supply b for a wave run at a different k. REF_REPLAY_REF_RUN names
# that run: the k=1 mechanism cell, which is the FIRST entry of
# FROZEN_SOURCES in both analyzers and is itself generated by this
# script (configs_pofd_qwen_mechanism_frozen.txt). Its identical-hash
# twin pofdfam_qwen7b_k0_ea1_w0p5_l0p2_es0p05_s0 is the audited fallback
# (manifest_qwen_mechanism.json, verdict PASS, complete).
#
# n PER ARM, HALF-UP AND CHECKED. The live count is
# n = floor(q * 723 + 0.5): 72 / 145 / 362 / 542 / 723 for
# q = .10 / .20 / .50 / .75 / 1. These are pinned literally AND
# recomputed below, because round() is banker's-rounding in Python and
# round(361.5) landing on 362 is luck, not a rule.
#
# TAG GRAMMAR. New family prefix pofdrr_. Both gates are GENUINELY open
# on this surface, so the tags carry the established _eaopen_/_esopen_
# MODE tokens -- never _ea1_/_es1_, which are numeric strict-<
# thresholds that still REJECT a distance-1 pair. q rides a _q<frac>_
# token through the project's _num() spelling (.10 -> q0p1, .75 ->
# q0p75), the SELECTION seed rides _ss<seed>_ and the RUN seed keeps the
# established trailing _s<seed>_r<rounds>. Two seeds, two tokens: they
# are different objects and a single _s0_ could not say which is which.
#
# THE SMOKE. One 3-round q=0.10 cell, its own key, outside the four:
# q=0.10 is the arm where the replay path does the most work (651 of 723
# rows come from b), so it exercises row substitution, the reference
# vector load, training, adapter re-serving, the twin and complete raw
# outputs in the least forgiving corner. It wears the pofdrrsmk_ PREFIX
# rather than a trailing token, because check_ref_replay decides whether
# to enforce the 100-round horizon off that prefix -- a 3-round cell
# under pofdrr_ would be gated as a truncated production run.
# =========================================================================
RR_KEY = "ref_replay"
RR_SMOKE_KEY = "ref_replay_smoke"
RR_MODEL = "qwen7b"
RR_ARM = "b0"                   # ordinary SFT, lambda_KL = 0
RR_W = 1.0                      # W_PLAT
RR_K = 1.0                      # INNATE_LAMBDA
RR_EPS_SOCIAL = QWU_EPS_SOCIAL  # 0.2 -- inert under all_open, set anyway
RR_ROUNDS = QWU_ROUNDS          # 100
RR_SMOKE_ROUNDS = QWU_SMOKE_ROUNDS   # 3
RR_SEED = 0                     # RUN seed
RR_SEL_SEED = 0                 # REF_REPLAY_SEED: the row-selection stream
RR_N_AGENTS = 723
RR_H100 = QMECH_H100
RR_QS = [0.10, 0.20, 0.50, 0.75]     # queued; 1.0 is the reused QWU cell
RR_Q_FULL = 1.0
RR_SMOKE_Q = 0.10
# live-row counts, pinned literally and recomputed by rr_n() below
RR_N = {0.10: 72, 0.20: 145, 0.50: 362, 0.75: 542, 1.0: 723}
# q = 1 IS this completed cell -- same arm, same surface, all-live rows
RR_REUSED_Q1_TAG = "pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100"
# the run supplying the frozen vector b (its pred_raw[0])
RR_REF_RUN = "pofdqmech_qwen7b_k0_ea1_w0p5_l1_es0p05_s0"
RR_REF_RUN_TWIN = "pofdfam_qwen7b_k0_ea1_w0p5_l0p2_es0p05_s0"
RR_REF_SHA = QMECH_CANONICAL_PRED_SHA

# q rides the QUEUE (col 16): one sub serves the whole ladder. The
# selection seed and the reference run are pinned in the sub env --
# every row of this key shares them.
ROW_RR = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, "
          "ab, {es}, 0.0, {wplat}, loop, 0.0, {lam}, {refq}, {iclk}, "
          "{snap}, {uselora}, {fresh}, {ansk}, {gg}, {nrounds}, "
          "{basemodel}, {chatthink}, {mem}, {disk}, {pplbatch}")


def rr_n(q):
    """Live rows at fraction q, HALF-UP: int() truncates toward zero and
    q*N is positive here, so int(x + .5) is exactly floor(x + .5).
    Python's round() is banker's rounding -- round(361.5) == 362 only
    because 362 happens to be even -- so the rule is spelled out."""
    return int(q * RR_N_AGENTS + 0.5)


def rr_q_tok(q):
    """.10 -> 'q0p1', .75 -> 'q0p75', 1 -> 'q1' (project _num grammar)."""
    return f"q{_num(q)}"


def rr_tag(q, rounds=RR_ROUNDS, smoke=False):
    """_eaopen_/_esopen_ spell the genuinely open gates (never _ea1_ /
    _es1_: both gates are strict inequalities, so the number 1 still
    rejects a distance-1 pair). _ss<n>_ is the REPLAY-SELECTION seed and
    the trailing _s<n>_ is the RUN seed. The horizon is in the tag
    because a 3-round and a 100-round cell of one condition are
    different objects.

    The smoke wears its own PREFIX rather than a trailing token:
    check_ref_replay reads the horizon rule off the prefix
    (pofdrrsmk_ = a short smoke, pofdrr_ = the declared 100 rounds), so
    a 3-round production-prefixed tag would be gated as a truncated
    run. Both prefixes still start with pofdrr, which is what the
    checker uses to claim the run at all."""
    return (f"{'pofdrrsmk' if smoke else 'pofdrr'}_{RR_MODEL}"
            f"_{rr_q_tok(q)}_ss{RR_SEL_SEED}_eaopen"
            f"_w{_num(RR_W)}_l{_num(RR_K)}_esopen_s{RR_SEED}_r{rounds}")


def rr_row(q, rounds=RR_ROUNDS, smoke=False):
    a = REACH_ARM_COLS[RR_ARM]
    m = FAM_MODELS[RR_MODEL]
    return ROW_RR.format(
        tag=rr_tag(q, rounds, smoke), style=a["style"], beta=a["beta"],
        seed=RR_SEED, es=f"{RR_EPS_SOCIAL:g}", wplat=f"{RR_W:g}",
        lam=f"{RR_K:g}", refq=f"{q:g}", iclk=a["iclk"], snap=a["snap"],
        uselora=a["uselora"], fresh=a["fresh"], ansk=a["ansk"],
        gg=a["gg"], nrounds=rounds, basemodel=m["base_model"],
        chatthink=m["chatthink"], mem=m["mem"], disk=m["disk"],
        pplbatch=m["pplbatch"])


def rr_rows():
    """Four cells. q = 1 is NOT here: it is ordinary SFT exactly, i.e.
    the completed RR_REUSED_Q1_TAG."""
    return [rr_row(q) for q in RR_QS]


def rr_smoke_rows():
    """ONE 3-round q=0.10 cell -- the corner where the replay path
    substitutes the most rows (651 of 723)."""
    return [rr_row(RR_SMOKE_Q, rounds=RR_SMOKE_ROUNDS, smoke=True)]


def rr_sub(smoke=False):
    key = RR_SMOKE_KEY if smoke else RR_KEY
    rows = rr_smoke_rows() if smoke else rr_rows()
    qs = [RR_SMOKE_Q] if smoke else RR_QS
    return RR_SUB_TEMPLATE.format(
        key=key, n_jobs=len(rows), gpu=RR_H100, bad=BAD_NODE_REQ,
        rounds=RR_SMOKE_ROUNDS if smoke else RR_ROUNDS,
        selseed=RR_SEL_SEED, refrun=RR_REF_RUN, reused=RR_REUSED_Q1_TAG,
        sha=RR_REF_SHA,
        ladder=" ".join(f"q={q:g} n={rr_n(q)}" for q in qs),
        kind=("SMOKE [3 rounds, NOT production]" if smoke
              else "PRODUCTION [100 rounds]"))


# NO BRACES in this template body except the format fields themselves.
RR_SUB_TEMPLATE = """\
# HTCondor: QWEN2.5 REFERENCE REPLAY AT THE WU BOUNDARY --
# {kind}, {n_jobs} jobs. GENERATED by gen_pofd_sweep.py from the RR
# block. Never edit by hand: rerun the script.
# Each round the SFT set is rebuilt FULL SIZE [all 723 rows, one epoch,
# batch 4 -> 181 steps], and a fraction q of those rows carries the LIVE
# population value while the rest carry a PINNED frozen-Qwen prediction
# b_i for the same agent. Data volume and optimizer compute are
# therefore IDENTICAL across arms and across the reused q=1 cell: only
# the amount of FEEDBACK in the training signal changes. That is what
# the subsample wave could not separate, since a smaller sample also
# took fewer steps.
# This ladder: {ladder}
# THE SURFACE IS THE COMPLETED WU-BOUNDARY b0 CELL, UNCHANGED --
# movielens Action 723 agents, Qwen/Qwen2.5-7B-Instruct, ordinary SFT
# [lambda_KL = 0], W = 1, k = 1, BOTH gates genuinely all_open, fresh
# LoRA r512 every round, SFT_EPOCHS=1, batch 4, LR 5e-5, {rounds}
# rounds, seed 0, gamma = 0, matched twin [WITH_TWIN=1], greedy
# eval-mode serving, SAVE_RAW_GEN=1, H100 pinned to the exact SKU.
# WHY MODES AND NOT THE NUMBER 1. Both gates are STRICT inequalities, so
# eps = 1 does NOT open them: an agent at 0 served 1, or a peer pair at
# [0, 1], sits at distance exactly 1 and is still REJECTED. The tags
# spell _eaopen_/_esopen_ and the checker rejects any numeric-threshold
# job wearing an open tag. EPS=0.2 is inert for acceptance under
# all_open but is set anyway: eps_social=0 is how "no peer step" is
# spelled everywhere else, and the runner refuses that combination.
# q = 1 IS NOT QUEUED. All-live rows IS ordinary SFT, so that arm is the
# already completed
#   {reused}
# -- the same cell the subsample wave reuses for its 100% arm. The
# production key therefore queues FOUR cells, not five.
# THE REFERENCE VECTOR IS NOT RE-EXTRACTED EITHER. b is the canonical
# frozen K=D=0 served vector: constant across rounds and across every
# k / eps / W cell, which is why an archived run at a different k can
# supply it.
#   REF_REPLAY_REF_RUN={refrun}
#   sha256 {sha}
#   REF_REPLAY_SEED={selseed} selects WHICH rows stay live.
# Gate every pull with check_pofd_sanity [RR section: both gate modes
# genuinely open, the declared horizon, the live-row count matching
# floor[q*723+.5], the reference vector hashing to the canonical
# frozen prior, finite SFT losses, adapter re-served in eval mode].
# Submit: bash experiments/condor/submit_pofd_sweep.sh <BID> {key}
universe          = vanilla
executable        = /home/gsmithline/perfsim/experiments/condor/run_one_pokec_gated_idempotent.sh
arguments         = $(tag) $(style) $(beta) $(seed) $(deploy_every) $(regime) $(pscale) $(anchor) $(pop) $(eps) $(gamma) $(wplat) $(mode) $(canary)

request_cpus      = 4
request_memory    = $(mem)
request_disk      = $(disk)
request_gpus      = 1
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000) && (TARGET.CUDADeviceName == "{gpu}"){bad}

getenv            = False
environment       = "REPO=/home/gsmithline/perfsim CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh ENV_NAME=opdyn WANDB_KEY_FILE=/home/gsmithline/.wandb_key WANDB_PROJECT=perfsim-gated-lm DATASET=movielens ML_TARGET=Action HF_HOME=/lustre/fast/fast/gsmithline/hf_cache HF_HUB_OFFLINE=1 AI_GATE_MODE=all_open PEER_GATE_MODE=all_open EPS_AI=1 INNATE_LAMBDA=$(lam) REF_REPLAY_Q=$(refq) REF_REPLAY_SEED={selseed} REF_REPLAY_REF_RUN={refrun} ICL_K=$(iclk) ICL_SNAPSHOT_ROUND=$(snap) ICL_DAYS=0 ICL_SELECT=random ICL_CTX_SOURCE=live USE_LORA=$(uselora) FRESH_EACH_ROUND=$(fresh) ANS_SAMPLE_K=$(ansk) ANS_SAMPLE_N=64 ANS_SAMPLE_T=1.0 LOG_GENDER_GAPS=$(gg) KL_DIRECTION=forward WITH_TWIN=1 SAVE_RAW_GEN=1 CHAT_THINKING=$(chatthink) BASE_MODEL=$(basemodel) TRAIN_CAP=723 N_ROUNDS=$(nrounds) EPOCH_SIZE=100 SFT_EPOCHS=1 SFT_BATCH_SIZE=4 GEN_BATCH_SIZE=32 LORA_R=512 SFT_LR=5e-5 N_LABELED=723 HIST_BINS=50 LOG_PERPLEXITY=1 N_PERPLEXITY=64 LOG_PPL_DIST=1 PPL_DIST_CAP=0 PPL_BATCH=$(pplbatch) SEED_BASE_DATA=1 WANDB_RUN_SUFFIX=_{key}"

output            = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).out
error             = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).err
log               = /home/gsmithline/perfsim/experiments/condor/logs/$(tag).log

notification      = Complete
notify_user       = gabriel.smithline@tue.ellis.eu
on_exit_hold      = (ExitCode =!= 0)
periodic_release  = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
periodic_remove   = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)

queue tag, style, beta, seed, deploy_every, regime, pscale, anchor, pop, eps, gamma, wplat, mode, canary, lam, refq, iclk, snap, uselora, fresh, ansk, gg, nrounds, basemodel, chatthink, mem, disk, pplbatch from experiments/condor/configs_pofd_{key}.txt
"""


def main():
    verify = "--verify" in sys.argv
    files, expected = {}, {}
    for model in ACTIVE_MODELS:
        p = os.path.join(HERE, f"configs_pofd_{model}.txt")
        files[p] = rows_for(model, SEEDS)
        expected[p] = len(BETAS) * len(EPS_AIS) * len(SEEDS)
    for model in PFRAC_MODELS:
        p = os.path.join(HERE, f"configs_pofd_{model}_pfrac.txt")
        files[p] = pfrac_rows(model)
        expected[p] = len(PFRACS) * len(EPS_AIS) * len(SEEDS)
    p = os.path.join(HERE, f"configs_pofd_{BP_MODEL}_bp.txt")
    files[p] = bp_rows()
    expected[p] = len(BP_BETAS) * len(PFRACS) * len(BP_EPS) * len(SEEDS)
    files[os.path.join(HERE, f"configs_pofd_{BP_MODEL}_bp_smoke.txt")] = [ROW_BP.format(
        tag="pofdbpsmk_qwen7b_b0p5_ea0p2_pf0p5_s0_fresh_data", beta="0.5", seed=0,
        eps_ai="0.2", pfrac="0.5")]
    p = os.path.join(HERE, "configs_pofd_qwen7b_icl.txt")
    files[p] = icl_rows()
    expected[p] = len(ICL_ARMS) * len(EPS_AIS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_icl_smoke.txt")] = [ROW_ICL.format(
        tag="pofdiclsmk_qwen7b_ea0p1_k32pri_s0", seed=0, eps_ai="0.1",
        iclk=32, icldays=0, iclsrc="pristine")]
    p = os.path.join(HERE, "configs_pofd_olmo7b_icl.txt")
    files[p] = icl_rows("olmo7b")
    expected[p] = len(ICL_ARMS) * len(EPS_AIS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_olmo7b_icl_smoke.txt")] = [ROW_ICL.format(
        tag="pofdiclsmk_olmo7b_ea0p1_k32pri_s0", seed=0, eps_ai="0.1",
        iclk=32, icldays=0, iclsrc="pristine")]
    p = os.path.join(HERE, "configs_pofd_qwen7b_dpo.txt")
    files[p] = dpo_rows()
    expected[p] = len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(EPS_AIS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_dpo_smoke.txt")] = [ROW_DPO.format(
        tag="pofddposmk_qwen7b_db0p1_ea0p1_open_s0_fresh", seed=0, eps_ai="0.1",
        rlhf="open", dpobeta="0.1")]
    p = os.path.join(HERE, "configs_pofd_olmo7b_dpo.txt")
    files[p] = dpo_rows("olmo7b")
    expected[p] = len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(EPS_AIS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_olmo7b_dpo_smoke.txt")] = [ROW_DPO.format(
        tag="pofddposmk_olmo7b_db0p1_ea0p1_open_s0_fresh", seed=0, eps_ai="0.1",
        rlhf="open", dpobeta="0.1")]
    p = os.path.join(HERE, "configs_pofd_qwen7b_dpon.txt")
    files[p] = dpon_rows()
    expected[p] = len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(DPON_EPS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_dpon_smoke.txt")] = [ROW_DPO.format(
        tag="pofddponsmk_qwen7b_db0p1_ea0p4_closed_s0_fresh", seed=0, eps_ai="0.4",
        rlhf="closed", dpobeta="0.1")]
    p = os.path.join(HERE, "configs_pofd_qwen7b_w.txt")
    files[p] = w_rows()
    expected[p] = len(BETAS) * len(EPS_AIS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_w_smoke.txt")] = [ROW_W.format(
        tag=f"pofdwsmk_qwen7b_b0p5_ea0p2_{w_tok()}_s0_fresh_data",
        style="sft_kl", beta="0.5", seed=0, eps_ai="0.2")]
    p = os.path.join(HERE, "configs_pofd_qwen7b_wdpo.txt")
    files[p] = wdpo_rows(EPS_AIS, "pofdwdpo")
    expected[p] = len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(EPS_AIS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_wdpo_smoke.txt")] = [ROW_WDPO.format(
        tag=f"pofdwdposmk_qwen7b_db0p1_ea0p4_closed_{w_tok()}_s0_fresh",
        seed=0, eps_ai="0.4", rlhf="closed", dpobeta="0.1")]
    p = os.path.join(HERE, "configs_pofd_qwen7b_wdpon.txt")
    files[p] = wdpo_rows(DPON_EPS, "pofdwdpon")
    expected[p] = len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(DPON_EPS) * len(SEEDS)
    p = os.path.join(HERE, "configs_pofd_qwen7b_ws.txt")
    files[p] = ws_rows()
    expected[p] = len(BETAS) * len(EPS_AIS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_ws_smoke.txt")] = [ROW_WS.format(
        tag=f"pofdwssmk_qwen7b_b0p5_ea0p2_{ws_tok()}_s0_fresh_data",
        style="sft_kl", beta="0.5", seed=0, eps_ai="0.2")]

    # ---- corrected-dynamics re-runs (population_update=nested_ai_then_social_v1)
    # Identical grids to pofdw_/pofdws_; ONLY the round operator differs, so the
    # tags must differ too -- these write to new run dirs and never overwrite the
    # superseded ones. See the pofdw_ block above for the operator.
    p = os.path.join(HERE, "configs_pofd_qwen7b_w2.txt")
    files[p] = w_rows("pofdw2")
    expected[p] = len(BETAS) * len(EPS_AIS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_w2_smoke.txt")] = [ROW_W.format(
        tag=f"pofdw2smk_qwen7b_b0p5_ea0p2_{w_tok()}_s0_fresh_data",
        style="sft_kl", beta="0.5", seed=0, eps_ai="0.2")]
    p = os.path.join(HERE, "configs_pofd_qwen7b_ws2.txt")
    files[p] = ws_rows("pofdws2")
    expected[p] = len(BETAS) * len(EPS_AIS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_ws2_smoke.txt")] = [ROW_WS.format(
        tag=f"pofdws2smk_qwen7b_b0p5_ea0p2_{ws_tok()}_s0_fresh_data",
        style="sft_kl", beta="0.5", seed=0, eps_ai="0.2")]
    # olmo7b twins of the corrected-operator W ladder (2026-07-28): identical
    # 20-cell grids to the qwen7b w2/ws2 waves; model deltas (separate HF
    # cache, PPL_BATCH=16, 160G/60G) live in the .sub files.
    p = os.path.join(HERE, "configs_pofd_olmo7b_w2.txt")
    files[p] = w_rows("pofdw2", model="olmo7b")
    expected[p] = len(BETAS) * len(EPS_AIS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_olmo7b_w2_smoke.txt")] = [ROW_W.format(
        tag=f"pofdw2smk_olmo7b_b0p5_ea0p2_{w_tok()}_s0_fresh_data",
        style="sft_kl", beta="0.5", seed=0, eps_ai="0.2")]
    p = os.path.join(HERE, "configs_pofd_olmo7b_ws2.txt")
    files[p] = ws_rows("pofdws2", model="olmo7b")
    expected[p] = len(BETAS) * len(EPS_AIS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_olmo7b_ws2_smoke.txt")] = [ROW_WS.format(
        tag=f"pofdws2smk_olmo7b_b0p5_ea0p2_{ws_tok()}_s0_fresh_data",
        style="sft_kl", beta="0.5", seed=0, eps_ai="0.2")]
    # forward-KL probe: see the FKL_ comment block above. The pofdw1f_ prefix
    # keeps run dirs distinct from the reverse-KL pofd_ wave; base ROW ->
    # W=1 environment (wplat=1.0 queue col, eps_social=0, no innate anchor).
    p = os.path.join(HERE, "configs_pofd_qwen7b_w1f.txt")
    files[p] = [ROW.format(tag=tag_of("qwen7b", b, 0.4, 0, prefix="pofdw1f"),
                           style="sft_kl", beta=f"{b:g}", seed=0, eps_ai="0.4")
                for b in FKL_BETAS]
    expected[p] = len(FKL_BETAS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_w1f_smoke.txt")] = [ROW.format(
        tag=tag_of("qwen7b", 0.5, 0.2, 0, prefix="pofdw1fsmk"),
        style="sft_kl", beta="0.5", seed=0, eps_ai="0.2")]
    # olmo twin of the w1f wave (2026-07-29): same 4 cells, single-spike prior
    # at 0.75 ABOVE the population -- the forward-vs-reverse threshold test on
    # a prior whose mode overlaps the data (gate |0.75 - x| < 0.4 is open for
    # nearly everyone, unlike qwen's 0.25 mode). Reverse twins: pofd_olmo7b_
    # b*_ea0p4_s0. Capture analysis needs the HIGH-side mirror of <0.45.
    p = os.path.join(HERE, "configs_pofd_olmo7b_w1f.txt")
    files[p] = [ROW.format(tag=tag_of("olmo7b", b, 0.4, 0, prefix="pofdw1f"),
                           style="sft_kl", beta=f"{b:g}", seed=0, eps_ai="0.4")
                for b in FKL_BETAS]
    expected[p] = len(FKL_BETAS)
    # w2f/ws2f: forward-KL canon for the W=0.5 environments (no smokes -- the
    # forward loss path was validated by the w1f smoke+wave, and these envs
    # are validated under reverse; the direction only touches the loss).
    p = os.path.join(HERE, "configs_pofd_qwen7b_w2f.txt")
    files[p] = [ROW_W.format(
        tag=f"pofdw2f_qwen7b_b{_num(b)}_ea{_num(ea)}_{w_tok()}_s0_fresh_data",
        style="sft_kl", beta=f"{b:g}", seed=0, eps_ai=f"{ea:g}")
        for b in FKL_BETAS for ea in EPS_AIS]
    expected[p] = len(FKL_BETAS) * len(EPS_AIS)
    p = os.path.join(HERE, "configs_pofd_qwen7b_ws2f.txt")
    files[p] = [ROW_WS.format(
        tag=f"pofdws2f_qwen7b_b{_num(b)}_ea{_num(ea)}_{ws_tok()}_s0_fresh_data",
        style="sft_kl", beta=f"{b:g}", seed=0, eps_ai=f"{ea:g}")
        for b in FKL_BETAS for ea in EPS_AIS]
    expected[p] = len(FKL_BETAS) * len(EPS_AIS)
    # pristine-teacher KL reference (see the PT_ comment block above)
    p = os.path.join(HERE, "configs_pofd_qwen7b_w2fpt.txt")
    files[p] = [ROW_W.format(
        tag=f"pofdw2fpt_qwen7b_b{_num(b)}_ea{_num(ea)}_{w_tok()}_s0_fresh_data",
        style="sft_kl", beta=f"{b:g}", seed=0, eps_ai=f"{ea:g}")
        for b in FKL_BETAS for ea in PT_EPS]
    expected[p] = len(FKL_BETAS) * len(PT_EPS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_w2fpt_smoke.txt")] = [ROW_W.format(
        tag=f"pofdw2fptsmk_qwen7b_b0p5_ea0p2_{w_tok()}_s0_fresh_data",
        style="sft_kl", beta="0.5", seed=0, eps_ai="0.2")]
    # olmo twins of w2f/ws2f (2026-07-29): the peer-dissolution test on the
    # in-range 0.75 prior (see the FKL_ comment block). Reverse twins:
    # pofdw2_/pofdws2_ olmo7b. Model deltas live in the .sub files; capture
    # analysis needs the HIGH-side mirror (op>0.70) as in the olmo w1f wave.
    # 2026-07-30 (user): trimmed from the full 16-cell grids (never
    # submitted) to the prioritized union -- (1) wide-gate dose response
    # ea0p4 x beta {0,0.1,0.2,0.5,1} and (2) strong-regularization gate
    # response b1 x ea {0.05,0.1,0.2} -- 8 cells per env, 16 jobs. b0 runs
    # HERE (style sft, direction-free): unlike qwen, the olmo reverse
    # w2/ws2 waves never ran, so there are no b0 runs to reuse.
    OW2F_POINTS = ([(b, 0.4) for b in [0.0] + FKL_BETAS]
                   + [(1.0, ea) for ea in [0.05, 0.1, 0.2]])
    p = os.path.join(HERE, "configs_pofd_olmo7b_w2f.txt")
    files[p] = [ROW_W.format(
        tag=f"pofdw2f_olmo7b_b{_num(b)}_ea{_num(ea)}_{w_tok()}_s0_fresh_data",
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=0,
        eps_ai=f"{ea:g}")
        for b, ea in OW2F_POINTS]
    expected[p] = len(OW2F_POINTS)
    p = os.path.join(HERE, "configs_pofd_olmo7b_ws2f.txt")
    files[p] = [ROW_WS.format(
        tag=f"pofdws2f_olmo7b_b{_num(b)}_ea{_num(ea)}_{ws_tok()}_s0_fresh_data",
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=0,
        eps_ai=f"{ea:g}")
        for b, ea in OW2F_POINTS]
    expected[p] = len(OW2F_POINTS)
    # regularization-excess heatmap fill (2026-08-02, user), keys
    # olmo7b_w2fx / olmo7b_ws2fx, composite olmo7b_rex: the missing
    # low-beta x tight-gate block of the olmo forward grids -- es
    # {0, 0.2} x ea {0.05, 0.1, 0.2} x beta {0, 0.1, 0.2, 0.5}, seed 0,
    # 24 jobs. Tags stay IN the pofdw2f_/pofdws2f_ families (grid-fill;
    # the existing 8+8 cross cells untouched, separate configs so only
    # the missing cells are queued). b0 rows style sft (direction-free;
    # no reverse b0 to reuse -- the olmo w2/ws2 reverse waves never
    # ran). AUDIT 2026-08-02: all 24 tags absent on cluster, no
    # partials, no other-family equivalents. The new subs add
    # WITH_TWIN=1 (user spec: save the matched no-platform twin;
    # telemetry-only per the esf precedent -- es0p2 rows force the twin
    # in the runner anyway, es=0 rows get the deterministic innate
    # twin). No smoke: env identical to the validated olmo w2f/ws2f
    # subs except WITH_TWIN; only queue-fed dials differ.
    OW2FX_BETAS = [0.0, 0.1, 0.2, 0.5]
    OW2FX_EAS = [0.05, 0.1, 0.2]
    p = os.path.join(HERE, "configs_pofd_olmo7b_w2fx.txt")
    files[p] = [ROW_W.format(
        tag=f"pofdw2f_olmo7b_b{_num(b)}_ea{_num(ea)}_{w_tok()}_s0_fresh_data",
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=0,
        eps_ai=f"{ea:g}")
        for b in OW2FX_BETAS for ea in OW2FX_EAS]
    expected[p] = len(OW2FX_BETAS) * len(OW2FX_EAS)
    p = os.path.join(HERE, "configs_pofd_olmo7b_ws2fx.txt")
    files[p] = [ROW_WS.format(
        tag=f"pofdws2f_olmo7b_b{_num(b)}_ea{_num(ea)}_{ws_tok()}_s0_fresh_data",
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=0,
        eps_ai=f"{ea:g}")
        for b in OW2FX_BETAS for ea in OW2FX_EAS]
    expected[p] = len(OW2FX_BETAS) * len(OW2FX_EAS)
    # eps-social dose-response (see the ESF_ comment block): seed-0 scan of
    # the repair channel at the forward headline cell; the W=0.5 es {0, 0.2}
    # cells are reused from w2f/ws2f, not regenerated here.
    p = os.path.join(HERE, "configs_pofd_qwen7b_esf.txt")
    files[p] = esf_rows([(0.5, es) for es in ESF_W05_ES]
                        + [(1.0, es) for es in ESF_W1_ES], [0])
    expected[p] = len(ESF_W05_ES) + len(ESF_W1_ES)
    if ESF_REPL_POINTS:
        p = os.path.join(HERE, "configs_pofd_qwen7b_esf_repl.txt")
        files[p] = esf_rows(ESF_REPL_POINTS, ESF_REPL_SEEDS)
        expected[p] = len(ESF_REPL_POINTS) * len(ESF_REPL_SEEDS)
    # targeted fe expansion (see the qwen7b_fex note in the ESF_ block)
    p = os.path.join(HERE, "configs_pofd_qwen7b_fex.txt")
    files[p] = (esf_rows([(0.5, es) for es in ESF_W05_ES], [42, 43])
                + esf_rows([(0.5, 0.10), (0.5, 0.25)], [0], ea=0.2))
    expected[p] = 2 * len(ESF_W05_ES) + 2
    # corrected-env ICL + DPO ports (see the ICLS2_/ROW_ICL2 comment block)
    p = os.path.join(HERE, "configs_pofd_qwen7b_icl2.txt")
    files[p] = [ROW_ICL2.format(
        tag=f"pofdicl2_qwen7b_{w_tok()}_ea{_num(ea)}_{arm}_s0",
        seed=0, es="0.0", eps_ai=f"{ea:g}", iclk=k, icldays=d, iclsrc=src)
        for arm, k, d, src in ICL_ARMS for ea in EPS_AIS]
    expected[p] = len(ICL_ARMS) * len(EPS_AIS)
    p = os.path.join(HERE, "configs_pofd_qwen7b_icls2.txt")
    files[p] = [ROW_ICL2.format(
        tag=f"pofdicls2_qwen7b_{ws_tok()}_ea{_num(ea)}_{arm}_s0",
        seed=0, es="0.2", eps_ai=f"{ea:g}", iclk=k, icldays=d, iclsrc=src)
        for arm, k, d, src in ICLS2_ARMS for ea in EPS_AIS]
    expected[p] = len(ICLS2_ARMS) * len(EPS_AIS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_icls2_smoke.txt")] = [ROW_ICL2.format(
        tag=f"pofdicls2smk_qwen7b_{ws_tok()}_ea0p1_k32noai_s0",
        seed=0, es="0.2", eps_ai="0.1", iclk=32, icldays=0, iclsrc="noai")]
    # olmo twins of icl2/icls2 (2026-07-29, user): same arms x eps_AI grids
    # on the in-range 0.75 prior spike -- qwen showed frozen weights flip
    # social mixing from repair to delivery (k0 198->259 with peers) and
    # live exemplars amplify capture 1.8x; olmo asks whether the d-arm
    # (personal memory) protection survives when the prior overlaps the
    # data. Capture analysis: HIGH-side mirror (op>0.70). Model deltas in
    # the .sub. No smokes: frozen x twin x peers validated by the qwen
    # icls2 smoke + wave, olmo model path by its W=1 icl wave.
    p = os.path.join(HERE, "configs_pofd_olmo7b_icl2.txt")
    files[p] = [ROW_ICL2.format(
        tag=f"pofdicl2_olmo7b_{w_tok()}_ea{_num(ea)}_{arm}_s0",
        seed=0, es="0.0", eps_ai=f"{ea:g}", iclk=k, icldays=d, iclsrc=src)
        for arm, k, d, src in ICL_ARMS for ea in EPS_AIS]
    expected[p] = len(ICL_ARMS) * len(EPS_AIS)
    p = os.path.join(HERE, "configs_pofd_olmo7b_icls2.txt")
    files[p] = [ROW_ICL2.format(
        tag=f"pofdicls2_olmo7b_{ws_tok()}_ea{_num(ea)}_{arm}_s0",
        seed=0, es="0.2", eps_ai=f"{ea:g}", iclk=k, icldays=d, iclsrc=src)
        for arm, k, d, src in ICLS2_ARMS for ea in EPS_AIS]
    expected[p] = len(ICLS2_ARMS) * len(EPS_AIS)
    # icls2 seed replicates (2026-08-03, user): seeds 42/43 for the mixed-env
    # ea0p4 endpoint cells still at n=1 -- qwen k32live/d5 (k0 s42/s43
    # already ran via fef) and olmo k0/k32live/d5. Completes the 3-seed
    # cross-seed Student-t CIs for the crossmodel ICL endpoint figure.
    # Same ROW_ICL2 env3 rows as the icls2 wave; grid-fill configs so only
    # the missing cells queue. AUDIT 2026-08-03: all 10 tags absent locally
    # and on the cluster (no partials) -- nothing overwritten or resubmitted.
    ICLX_SEEDS = FE_SEEDS[1:]
    ICLX_QWEN_ARMS = [("k32live", 32, 0, "live"), ("d5", 0, 5, "live")]
    ICLX_OLMO_ARMS = [("k0", 0, 0, "live")] + ICLX_QWEN_ARMS
    p = os.path.join(HERE, "configs_pofd_qwen7b_icls2x.txt")
    files[p] = [ROW_ICL2.format(
        tag=f"pofdicls2_qwen7b_{ws_tok()}_ea0p4_{arm}_s{s}",
        seed=s, es="0.2", eps_ai="0.4", iclk=k, icldays=d, iclsrc=src)
        for arm, k, d, src in ICLX_QWEN_ARMS for s in ICLX_SEEDS]
    expected[p] = len(ICLX_QWEN_ARMS) * len(ICLX_SEEDS)
    p = os.path.join(HERE, "configs_pofd_olmo7b_icls2x.txt")
    files[p] = [ROW_ICL2.format(
        tag=f"pofdicls2_olmo7b_{ws_tok()}_ea0p4_{arm}_s{s}",
        seed=s, es="0.2", eps_ai="0.4", iclk=k, icldays=d, iclsrc=src)
        for arm, k, d, src in ICLX_OLMO_ARMS for s in ICLX_SEEDS]
    expected[p] = len(ICLX_OLMO_ARMS) * len(ICLX_SEEDS)
    # ICL feature endogenization (2026-08-04, user): the fe gender matrix at
    # the IN-CONTEXT mechanism -- does the k8live exemplar channel endogenize
    # gender under the corrected env3 dynamics (W=0.5, lam=0.2, es=0.2,
    # ea=0.4, 30 rounds, frozen weights)? Mirrors fegd/fegp (SFT-KL b1) with
    # adaptation running through the prompt instead of the weights.
    # PROFILE_DROP_COLS removes gender from the served profile AND every
    # exemplar line; PROFILE_PERMUTE_COLS shuffles displayed gender jointly,
    # seeded by the run seed ONLY -- same per-seed permutation as every fe
    # wave, deliberate. Natural s0 anchor = pofdicls2_ ea0p4_k8live_s0
    # (icls2 wave; REUSED, not re-run). Only the 8 missing cells run,
    # composite key qwen7b_fei:
    #   feik  (2): natural k8live x s {42, 43} -- pofdicls2_-family tags
    #   feigd (3): PROFILE_DROP_COLS=gender,    k8live x s {0, 42, 43}
    #              -- pofdicls2gd_
    #   feigp (3): PROFILE_PERMUTE_COLS=gender, k8live x s {0, 42, 43}
    #              -- pofdicls2gp_
    # gd/gp prefixes keep the "pofdicl" checker family (frozen + arm gates;
    # the arm token stays directly before _s<seed>); check_pofd_sanity
    # gained profile_drop/permute gates keyed on the new prefixes. No smoke:
    # gdrop/gperm x ICL exemplars proven by the old icl_endog waves (the
    # knobs act once at profile build, before the loop); frozen x twin x
    # peers by the icls2 smoke + wave. AUDIT 2026-08-04: all 8 tags absent
    # on cluster (no partials); the s0 k8live anchor has trajectory.pt.
    p = os.path.join(HERE, "configs_pofd_qwen7b_feik.txt")
    files[p] = [ROW_ICL2.format(
        tag=f"pofdicls2_qwen7b_{ws_tok()}_ea0p4_k8live_s{s}",
        seed=s, es="0.2", eps_ai="0.4", iclk=8, icldays=0, iclsrc="live")
        for s in FE_SEEDS[1:]]
    expected[p] = len(FE_SEEDS[1:])
    for key, tagpre in (("feigd", "pofdicls2gd"), ("feigp", "pofdicls2gp")):
        p = os.path.join(HERE, f"configs_pofd_qwen7b_{key}.txt")
        files[p] = [ROW_ICL2.format(
            tag=f"{tagpre}_qwen7b_{ws_tok()}_ea0p4_k8live_s{s}",
            seed=s, es="0.2", eps_ai="0.4", iclk=8, icldays=0, iclsrc="live")
            for s in FE_SEEDS]
        expected[p] = len(FE_SEEDS)
    # controlled teacher fe wave (see the TCH_/TFE_ comment block)
    p = os.path.join(HERE, "configs_pofd_qwen7b_tch.txt")
    files[p] = [ROW_TCH.format(
        tag=f"pofdtch_qwen7b_{tok}_ea0p4_{w_tok()}_s0",
        seed=0, eps_ai="0.4", tdelta=dv)
        for tok, dv in TCH_DELTAS]
    expected[p] = len(TCH_DELTAS)
    p = os.path.join(HERE, "configs_pofd_qwen7b_tfem.txt")
    files[p] = [ROW_TFE.format(
        tag=f"pofdtfe_qwen7b_b1_ea0p4_{arm}_{ws_tok()}_s{s}_fresh_data",
        seed=s, eps_ai="0.4", refadapter=TFE_REFS[arm])
        for arm in ("tpos", "tneu", "tneg") for s in FE_SEEDS]
    expected[p] = 3 * len(FE_SEEDS)
    for key, arm in (("tfegd", "tposgd"), ("tfegp", "tposgp")):
        p = os.path.join(HERE, f"configs_pofd_qwen7b_{key}.txt")
        files[p] = [ROW_WS.format(
            tag=f"pofdtfe_qwen7b_b1_ea0p4_{arm}_{ws_tok()}_s{s}_fresh_data",
            style="sft_kl", beta="1", seed=s, eps_ai="0.4")
            for s in FE_SEEDS]
        expected[p] = len(FE_SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_tfe_smoke.txt")] = [ROW_TFE.format(
        tag=f"pofdtfesmk_qwen7b_b1_ea0p4_tpos_{ws_tok()}_s0_fresh_data",
        seed=0, eps_ai="0.4", refadapter=TFE_REFS["tpos"])]
    # tfe CI seed extension (see the CI EXTENSION note in the TFE_ block)
    p = os.path.join(HERE, "configs_pofd_qwen7b_tfe_ci.txt")
    files[p] = [ROW_TFE.format(
        tag=f"pofdtfe_qwen7b_b1_ea0p4_{arm}_{ws_tok()}_s{s}_fresh_data",
        seed=s, eps_ai="0.4", refadapter=TFE_REFS[arm])
        for arm in ("tpos", "tneu", "tneg") for s in TFE_CI_SEEDS]
    expected[p] = 3 * len(TFE_CI_SEEDS)
    assert expected[p] == 6, "tfe_ci must queue exactly 6 jobs"
    # random-even-split twin (see the TFER_ comment block)
    p = os.path.join(HERE, "configs_pofd_qwen7b_tchr.txt")
    files[p] = [ROW_TCH.format(
        tag=f"pofdtchr_qwen7b_{tok}_ea0p4_{w_tok()}_s0",
        seed=0, eps_ai="0.4", tdelta=dv)
        for tok, dv in TCH_DELTAS]
    expected[p] = len(TCH_DELTAS)
    p = os.path.join(HERE, "configs_pofd_qwen7b_tfer.txt")
    files[p] = [ROW_TFE.format(
        tag=f"pofdtfer_qwen7b_b1_ea0p4_{arm}_{ws_tok()}_s{s}_fresh_data",
        seed=s, eps_ai="0.4", refadapter=TFER_REFS[arm])
        for arm in ("tpos", "tneg") for s in FE_SEEDS]
    expected[p] = 2 * len(FE_SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_tfer_smoke.txt")] = [ROW_TFE.format(
        tag=f"pofdtfersmk_qwen7b_b1_ea0p4_tpos_{ws_tok()}_s0_fresh_data",
        seed=0, eps_ai="0.4", refadapter=TFER_REFS["tpos"])]
    p = os.path.join(HERE, "configs_pofd_qwen7b_wdpo2.txt")
    files[p] = [ROW_WDPO2.format(
        tag=f"pofdwdpo2_qwen7b_db{_num(db)}_ea{_num(ea)}_{fb}_{w_tok()}_s0_fresh",
        seed=0, es="0.0", eps_ai=f"{ea:g}", rlhf=fb, dpobeta=f"{db:g}")
        for db in DPO_BETAS for fb in DPO_FEEDBACKS for ea in EPS_AIS]
    expected[p] = len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(EPS_AIS)
    p = os.path.join(HERE, "configs_pofd_qwen7b_wdpos2.txt")
    files[p] = [ROW_WDPO2.format(
        tag=f"pofdwdpos2_qwen7b_db{_num(db)}_ea{_num(ea)}_{fb}_{ws_tok()}_s0_fresh",
        seed=0, es="0.2", eps_ai=f"{ea:g}", rlhf=fb, dpobeta=f"{db:g}")
        for db in DPO_BETAS for fb in DPO_FEEDBACKS for ea in EPS_AIS]
    expected[p] = len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(EPS_AIS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_wdpos2_smoke.txt")] = [ROW_WDPO2.format(
        tag=f"pofdwdpos2smk_qwen7b_db0p1_ea0p4_closed_{ws_tok()}_s0_fresh",
        seed=0, es="0.2", eps_ai="0.4", rlhf="closed", dpobeta="0.1")]
    # full-epoch DPO rerun (2026-08-03, user): the wdpo2/wdpos2 wave trained on
    # DPO_MAX_STEPS=3 x batch 4 = 12 of the ~270-320 valid pairs built each
    # round (~4% of the preference signal; cluster logs: pairs=270-323 of 723,
    # ties the rest). Small-budget confound -> rerun the IDENTICAL 2x2x4 grids
    # with DPO_MAX_STEPS=0: one full epoch per round, every valid preference
    # pair consumed exactly once (~70-80 optimizer steps/round). Tags keep the
    # pofdwdpo prefix (checker is_dpo) with family token 2e/s2e; dpo_beta /
    # dpo_max_steps now land in config.json (runner dump added 2026-08-03) and
    # the checker gates them. Smoke first (3 rounds, closed db0p1 ea0p4,
    # peers on): the new full-epoch budget path x dpo x peers combo.
    p = os.path.join(HERE, "configs_pofd_qwen7b_wdpo2e.txt")
    files[p] = [ROW_WDPO2.format(
        tag=f"pofdwdpo2e_qwen7b_db{_num(db)}_ea{_num(ea)}_{fb}_{w_tok()}_s0_fresh",
        seed=0, es="0.0", eps_ai=f"{ea:g}", rlhf=fb, dpobeta=f"{db:g}")
        for db in DPO_BETAS for fb in DPO_FEEDBACKS for ea in EPS_AIS]
    expected[p] = len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(EPS_AIS)
    p = os.path.join(HERE, "configs_pofd_qwen7b_wdpos2e.txt")
    files[p] = [ROW_WDPO2.format(
        tag=f"pofdwdpos2e_qwen7b_db{_num(db)}_ea{_num(ea)}_{fb}_{ws_tok()}_s0_fresh",
        seed=0, es="0.2", eps_ai=f"{ea:g}", rlhf=fb, dpobeta=f"{db:g}")
        for db in DPO_BETAS for fb in DPO_FEEDBACKS for ea in EPS_AIS]
    expected[p] = len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(EPS_AIS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_wdpos2e_smoke.txt")] = [ROW_WDPO2.format(
        tag=f"pofdwdpos2esmk_qwen7b_db0p1_ea0p4_closed_{ws_tok()}_s0_fresh",
        seed=0, es="0.2", eps_ai="0.4", rlhf="closed", dpobeta="0.1")]
    # DPO CI seed extension, staged (see the DPO CI EXTENSION block above)
    p = os.path.join(HERE, "configs_pofd_qwen7b_dpo_ci1.txt")
    files[p] = [ROW_DPO.format(
        tag=f"pofddpo_qwen7b_db{_num(db)}_ea{_num(ea)}_{fb}_s{s}_fresh",
        seed=s, eps_ai=f"{ea:g}", rlhf=fb, dpobeta=f"{db:g}")
        for db, ea in DPO_CI1_CELLS for fb in DPO_FEEDBACKS
        for s in DPO_CI_SEEDS]
    expected[p] = len(DPO_CI1_CELLS) * len(DPO_FEEDBACKS) * len(DPO_CI_SEEDS)
    assert expected[p] == 8, "dpo_ci1 must queue exactly 8 jobs"
    p = os.path.join(HERE, "configs_pofd_qwen7b_dpo_ci2.txt")
    files[p] = [ROW_DPO.format(
        tag=f"pofddpo_qwen7b_db{_num(db)}_ea{_num(ea)}_{fb}_s{s}_fresh",
        seed=s, eps_ai=f"{ea:g}", rlhf=fb, dpobeta=f"{db:g}")
        for db, ea in DPO_CI2_CELLS for fb in DPO_FEEDBACKS
        for s in DPO_CI_SEEDS]
    expected[p] = len(DPO_CI2_CELLS) * len(DPO_FEEDBACKS) * len(DPO_CI_SEEDS)
    assert expected[p] == 24, "dpo_ci2 must queue exactly 24 jobs"
    p = os.path.join(HERE, "configs_pofd_qwen7b_dpo_ci3.txt")
    files[p] = ([ROW_WDPO2.format(
        tag=f"pofdwdpo2e_qwen7b_db0p5_ea{_num(ea)}_{fb}_{w_tok()}_s{s}_fresh",
        seed=s, es="0.0", eps_ai=f"{ea:g}", rlhf=fb, dpobeta="0.5")
        for ea in DPO_CI3_EAS for fb in DPO_FEEDBACKS for s in DPO_CI_SEEDS]
        + [ROW_WDPO2.format(
            tag=(f"pofdwdpos2e_qwen7b_db0p5_ea{_num(ea)}_{fb}_{w_tok()}"
                 f"_es0p3_s{s}_fresh"),
            seed=s, es="0.3", eps_ai=f"{ea:g}", rlhf=fb, dpobeta="0.5")
           for ea in DPO_CI3_EAS for fb in DPO_FEEDBACKS
           for s in DPO_CI3_ES3_SEEDS])
    expected[p] = (len(DPO_CI3_EAS) * len(DPO_FEEDBACKS)
                   * (len(DPO_CI_SEEDS) + len(DPO_CI3_ES3_SEEDS)))
    assert expected[p] == 20, "dpo_ci3 must queue exactly 20 jobs"
    # Mistral main-env wave (see the MISTRAL MAIN-ENV block above)
    p = os.path.join(HERE, "configs_pofd_mistral7b_ws2f.txt")
    files[p] = [
        (f"pofdws2f_mistral7b_b{b}_ea0p4_{ws_tok()}_s{s}_fresh_data, "
         f"{'sft' if b == 0 else 'sft_kl'}, {b}, {s}, 1, replace, 1.0, "
         f"fixed, ab, 0.2, 0.0, 0.5, loop, 0.0, 0.4")
        for b in (0, 1) for s in MISTRAL_WS2F_SEEDS]
    expected[p] = 2 * len(MISTRAL_WS2F_SEEDS)
    assert expected[p] == 6, "mistral7b_ws2f must queue exactly 6 jobs"
    files[os.path.join(HERE, "configs_pofd_mistral7b_ws2f_smoke.txt")] = [
        (f"pofdws2fsmk_mistral7b_b1_ea0p4_{ws_tok()}_s0_fresh_data, "
         "sft_kl, 1, 0, 1, replace, 1.0, fixed, ab, 0.2, 0.0, 0.5, "
         "loop, 0.0, 0.4")]
    # matched-randomness DPO pairs (see the DPO MR block above)
    p = os.path.join(HERE, "configs_pofd_qwen7b_dpo_mr.txt")
    files[p] = [ROW_MR.format(
        tag=(f"pofddpomr_qwen7b_db0p5_ea{_num(ea)}_closed_s0_bk{bk}_fresh"),
        eps_ai=f"{ea:g}", bankseed=bk)
        for ea, banks in DPO_MR_EA_BANKS for bk in banks]
    expected[p] = sum(len(list(b)) for _, b in DPO_MR_EA_BANKS)
    assert expected[p] == 15, "dpo_mr must queue exactly 15 pair jobs"
    files[os.path.join(HERE, "configs_pofd_qwen7b_dpo_mr_smoke.txt")] = [
        ROW_MR.format(
            tag="pofddpomrsmk_qwen7b_db0p5_ea0p4_closed_s0_bk900_fresh",
            eps_ai="0.4", bankseed=900)]
    # Mistral SFT cube (see the MISTRAL SFT CUBE block above)
    _mc_s0 = mistral_cube_rows([0])
    _mc_repl = mistral_cube_rows(MISTRAL_CUBE_REPL_SEEDS)
    assert len(_mc_s0) == 70, f"mistral cube s0 must be 70, got {len(_mc_s0)}"
    assert len(_mc_repl) == 140, \
        f"mistral cube repl must be 140, got {len(_mc_repl)}"
    assert len(_mc_s0) + len(_mc_repl) == 210
    _mc_tags = {r.split(",")[0] for r in _mc_s0 + _mc_repl}
    assert len(_mc_tags) == 210, "duplicate tags inside the mistral cube"
    _mc_reused = {cube_tag("mistral7b", b, ea, es, s)
                  for (b, ea, es, s) in MISTRAL_CUBE_EXISTING}
    assert not (_mc_tags & _mc_reused), \
        f"mistral cube queues reused cells: {_mc_tags & _mc_reused}"
    _prior_tags = {r.split(",")[0] for rows in files.values() for r in rows}
    assert not (_mc_tags & _prior_tags), \
        f"mistral cube collides with existing configs: {_mc_tags & _prior_tags}"
    p = os.path.join(HERE, "configs_pofd_mistral7b_cube_s0.txt")
    files[p] = _mc_s0
    expected[p] = 70
    p = os.path.join(HERE, "configs_pofd_mistral7b_cube_repl.txt")
    files[p] = _mc_repl
    expected[p] = 140
    # corrected-env data-regularizer port (see the ROW_PF2 comment block)
    p = os.path.join(HERE, "configs_pofd_qwen7b_pf2.txt")
    files[p] = [ROW_PF2.format(
        tag=f"pofdpf2_qwen7b_{w_tok()}_b0_ea{_num(ea)}_pf{_num(pf)}_s0_fresh_data",
        seed=0, es="0.0", eps_ai=f"{ea:g}", pfrac=f"{pf:g}")
        for pf in PFRACS for ea in EPS_AIS]
    expected[p] = len(PFRACS) * len(EPS_AIS)
    p = os.path.join(HERE, "configs_pofd_qwen7b_pfs2.txt")
    files[p] = [ROW_PF2.format(
        tag=f"pofdpfs2_qwen7b_{ws_tok()}_b0_ea{_num(ea)}_pf{_num(pf)}_s0_fresh_data",
        seed=0, es="0.2", eps_ai=f"{ea:g}", pfrac=f"{pf:g}")
        for pf in PFRACS for ea in EPS_AIS]
    expected[p] = len(PFRACS) * len(EPS_AIS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_pfs2_smoke.txt")] = [ROW_PF2.format(
        tag=f"pofdpfs2smk_qwen7b_{ws_tok()}_b0_ea0p4_pf0p5_s0_fresh_data",
        seed=0, es="0.2", eps_ai="0.4", pfrac="0.5")]
    # feature-endogenization reruns (see the FE_ comment block)
    p = os.path.join(HERE, "configs_pofd_qwen7b_fes.txt")
    files[p] = [ROW_WS.format(
        tag=f"pofdws2f_qwen7b_b{_num(b)}_ea0p4_{ws_tok()}_s{s}_fresh_data",
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=s,
        eps_ai="0.4")
        for b in [0.0, 0.5, 1.0] for s in FE_SEEDS[1:]]
    expected[p] = 3 * len(FE_SEEDS[1:])
    for key, tagpre in (("fegd", "pofdfegd"), ("fegp", "pofdfegp")):
        p = os.path.join(HERE, f"configs_pofd_qwen7b_{key}.txt")
        files[p] = [ROW_WS.format(
            tag=f"{tagpre}_qwen7b_b1_ea0p4_{ws_tok()}_s{s}_fresh_data",
            style="sft_kl", beta="1", seed=s, eps_ai="0.4")
            for s in FE_SEEDS]
        expected[p] = len(FE_SEEDS)
    p = os.path.join(HERE, "configs_pofd_qwen7b_fef.txt")
    files[p] = [ROW_ICL2.format(
        tag=f"pofdicls2_qwen7b_{ws_tok()}_ea0p4_k0_s{s}",
        seed=s, es="0.2", eps_ai="0.4", iclk=0, icldays=0, iclsrc="live")
        for s in FE_SEEDS[1:]]
    expected[p] = len(FE_SEEDS[1:])
    # env2 mirror of the fe wave (see the FE_ comment block)
    p = os.path.join(HERE, "configs_pofd_qwen7b_fes2.txt")
    files[p] = [ROW_W.format(
        tag=f"pofdw2f_qwen7b_b{_num(b)}_ea0p4_{w_tok()}_s{s}_fresh_data",
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=s,
        eps_ai="0.4")
        for b in [0.0, 0.5, 1.0] for s in FE_SEEDS[1:]]
    expected[p] = 3 * len(FE_SEEDS[1:])
    for key, tagpre in (("fegd2", "pofdfegd"), ("fegp2", "pofdfegp")):
        p = os.path.join(HERE, f"configs_pofd_qwen7b_{key}.txt")
        files[p] = [ROW_W.format(
            tag=f"{tagpre}_qwen7b_b1_ea0p4_{w_tok()}_s{s}_fresh_data",
            style="sft_kl", beta="1", seed=s, eps_ai="0.4")
            for s in FE_SEEDS]
        expected[p] = len(FE_SEEDS)
    p = os.path.join(HERE, "configs_pofd_qwen7b_fef2.txt")
    files[p] = [ROW_ICL2.format(
        tag=f"pofdicl2_qwen7b_{w_tok()}_ea0p4_k0_s{s}",
        seed=s, es="0.0", eps_ai="0.4", iclk=0, icldays=0, iclsrc="live")
        for s in FE_SEEDS[1:]]
    expected[p] = len(FE_SEEDS[1:])
    # continual-weights mirror of the env3 forward fe wave (see FE_ block)
    p = os.path.join(HERE, "configs_pofd_qwen7b_fesc.txt")
    files[p] = [ROW_WS.format(
        tag=f"pofdws2fc_qwen7b_b{_num(b)}_ea0p4_{ws_tok()}_s{s}_fresh_data",
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=s,
        eps_ai="0.4")
        for b in [0.0, 0.5, 1.0] for s in FE_SEEDS]
    expected[p] = 3 * len(FE_SEEDS)
    for key, tagpre in (("fegdc", "pofdfegdc"), ("fegpc", "pofdfegpc")):
        p = os.path.join(HERE, f"configs_pofd_qwen7b_{key}.txt")
        files[p] = [ROW_WS.format(
            tag=f"{tagpre}_qwen7b_b1_ea0p4_{ws_tok()}_s{s}_fresh_data",
            style="sft_kl", beta="1", seed=s, eps_ai="0.4")
            for s in FE_SEEDS]
        expected[p] = len(FE_SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_fec_smoke.txt")] = [ROW_WS.format(
        tag=f"pofdws2fcsmk_qwen7b_b1_ea0p4_{ws_tok()}_s0_fresh_data",
        style="sft_kl", beta="1", seed=0, eps_ai="0.4")]
    # reverse-KL mirror of the env3 fe wave (see the FE_ comment block)
    p = os.path.join(HERE, "configs_pofd_qwen7b_fesr.txt")
    files[p] = [ROW_WS.format(
        tag=f"pofdws2_qwen7b_b{_num(b)}_ea0p4_{ws_tok()}_s{s}_fresh_data",
        style="sft_kl", beta=f"{b:g}", seed=s, eps_ai="0.4")
        for b in [0.5, 1.0] for s in FE_SEEDS[1:]]
    expected[p] = 2 * len(FE_SEEDS[1:])
    for key, tagpre in (("fegdr", "pofdfegdr"), ("fegpr", "pofdfegpr")):
        p = os.path.join(HERE, f"configs_pofd_qwen7b_{key}.txt")
        files[p] = [ROW_WS.format(
            tag=f"{tagpre}_qwen7b_b1_ea0p4_{ws_tok()}_s{s}_fresh_data",
            style="sft_kl", beta="1", seed=s, eps_ai="0.4")
            for s in FE_SEEDS]
        expected[p] = len(FE_SEEDS)
    # Jensen-Shannon mirror of the env3 fe wave (see the FE_ comment block)
    p = os.path.join(HERE, "configs_pofd_qwen7b_fesj.txt")
    files[p] = [ROW_WS.format(
        tag=f"pofdws2j_qwen7b_b{_num(b)}_ea0p4_{ws_tok()}_s{s}_fresh_data",
        style="sft_kl", beta=f"{b:g}", seed=s, eps_ai="0.4")
        for b in [0.5, 1.0] for s in FE_SEEDS]
    expected[p] = 2 * len(FE_SEEDS)
    for key, tagpre in (("fegdj", "pofdfegdj"), ("fegpj", "pofdfegpj")):
        p = os.path.join(HERE, f"configs_pofd_qwen7b_{key}.txt")
        files[p] = [ROW_WS.format(
            tag=f"{tagpre}_qwen7b_b1_ea0p4_{ws_tok()}_s{s}_fresh_data",
            style="sft_kl", beta="1", seed=s, eps_ai="0.4")
            for s in FE_SEEDS]
        expected[p] = len(FE_SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_fej_smoke.txt")] = [ROW_WS.format(
        tag=f"pofdws2jsmk_qwen7b_b1_ea0p4_{ws_tok()}_s0_fresh_data",
        style="sft_kl", beta="1", seed=0, eps_ai="0.4")]
    # OLMo Romance mirror of the env3 fe wave (see the OLMo ROMANCE block)
    p = os.path.join(HERE, "configs_pofd_olmo7brom_fes.txt")
    files[p] = [ROW_WS.format(
        tag=f"pofdws2f_olmo7brom_b{_num(b)}_ea0p4_{ws_tok()}_s{s}_fresh_data",
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=s,
        eps_ai="0.4")
        for b in [0.0, 0.5, 1.0] for s in FE_SEEDS]
    expected[p] = 3 * len(FE_SEEDS)
    p = os.path.join(HERE, "configs_pofd_olmo7brom_fef.txt")
    files[p] = [ROW_ICL2.format(
        tag=f"pofdicls2_olmo7brom_{ws_tok()}_ea0p4_k0_s{s}",
        seed=s, es="0.2", eps_ai="0.4", iclk=0, icldays=0, iclsrc="live")
        for s in FE_SEEDS]
    expected[p] = len(FE_SEEDS)
    files[os.path.join(HERE, "configs_pofd_olmo7brom_fe_smoke.txt")] = [ROW_WS.format(
        tag=f"pofdws2fsmk_olmo7brom_b1_ea0p4_{ws_tok()}_s0_fresh_data",
        style="sft_kl", beta="1", seed=0, eps_ai="0.4")]
    # narrow-gate social-dose matrix (see the NDS_ comment block): only the
    # 26 audited-missing cells per model -- the es {0, 0.2} s0 gate cells
    # exist (w2f/ws2f, both models) and are REUSED, not re-run.
    for model in NDS_MODELS:
        p = os.path.join(HERE, f"configs_pofd_{model}_esfn.txt")
        files[p] = [ROW_ESF.format(
            tag=nds_tag(model, ea, es, s), seed=s, es=f"{es:g}", w="0.5",
            eps_ai=f"{ea:g}")
            for ea in NDS_EAS for es in NDS_ESS for s in FE_SEEDS
            if not (es in (0.0, 0.20) and s == 0)]
        expected[p] = 26
    # full parameter cube (see the CUBE_ comment block): grid minus the 103
    # audited-existing cells; the .sub files are generated from CUBE_MODELS
    cube_subs = {}
    for model in CUBE_MODELS:
        p = os.path.join(HERE, f"configs_pofd_{model}_cube.txt")
        files[p] = cube_rows(model)
        expected[p] = (len(CUBE_BETAS) * len(CUBE_EAS) * len(CUBE_ESS)
                       * len(CUBE_SEEDS)
                       - sum(1 for c in CUBE_EXISTING
                             if c[0] == model and c[4] in CUBE_SEEDS))
        cube_subs[os.path.join(HERE, f"at_pofd_{model}_cube.sub")] = \
            cube_sub(model)
    # exact initial-data replay (see the RPL_ comment block): rf>0 cells
    # only, the rf=0 cells reuse audited complete replace runs; the .sub
    # is generated from the same CUBE_MODELS registry
    for model in RPL_MODELS:
        p = os.path.join(HERE, f"configs_pofd_{model}_replay1.txt")
        files[p] = rpl_rows(model)
        expected[p] = (len(RPL_BETAS) * len(RPL_FRACS) * len(RPL_SEEDS)
                       - len(RPL_BETAS) * len(RPL_SEEDS))   # minus rf=0
        cube_subs[os.path.join(HERE, f"at_pofd_{model}_replay1.sub")] = \
            rpl_sub(model)
    # ordinary-SFT training-budget sweep (see the BUD_ comment block):
    # 3 step counts per model; the full-epoch endpoint is a reused cube
    # cell. Plus the REQUIRED 1-job smoke (the SFT max_steps path has
    # never run on the cluster).
    for model in BUD_MODELS:
        p = os.path.join(HERE, f"configs_pofd_{model}_budget.txt")
        files[p] = bud_rows(model)
        expected[p] = len(BUD_STEPS) * len(BUD_SEEDS)
        cube_subs[os.path.join(HERE, f"at_pofd_{model}_budget.sub")] = \
            bud_sub(model)
    files[os.path.join(HERE, "configs_pofd_qwen7b_budget_smoke.txt")] = [
        ROW_BUD.format(tag=bud_tag("qwen7b", 18, 0, prefix="pofdbudsmk"),
                       seed=0, es=f"{BUD_ES:g}", eps_ai=f"{BUD_EA:g}",
                       steps=18)]
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_budget_smoke.sub")] = \
        bud_sub("qwen7b", smoke=True)
    # frozen-context icl wave (see the ICLF_ comment block): audit found
    # ZERO equivalent existing cells (18 dial matches, none with gg
    # telemetry) -- full grid; smoke = dyn + fz8 at the mid dose.
    for model in ICLF_MODELS:
        p = os.path.join(HERE, f"configs_pofd_{model}_iclf.txt")
        files[p] = iclf_rows(model)
        expected[p] = (len(ICLF_ARMS) * len(ICLF_EAS) * len(ICLF_ESS)
                       * len(ICLF_SEEDS))
        cube_subs[os.path.join(HERE, f"at_pofd_{model}_iclf.sub")] = \
            iclf_sub(model)
    files[os.path.join(HERE, "configs_pofd_qwen7b_iclf_smoke.txt")] = [
        ROW_ICLF.format(tag=iclf_tag("qwen7b", arm, 0.2, 0.2, 0,
                                     prefix="pofdiclfsmk"),
                        seed=0, es="0.2", eps_ai="0.2", iclk=k, snap=snap)
        for arm, k, snap in ICLF_ARMS if arm in ("dyn", "fz8")]
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_iclf_smoke.sub")] = \
        iclf_sub("qwen7b", smoke=True)
    # SFT-to-ICL context transfer (see the CTF_ comment block): 3 arms x
    # seed 0; donor dirs ride the queue
    p = os.path.join(HERE, "configs_pofd_qwen7b_ctf.txt")
    files[p] = ctf_rows()
    expected[p] = len(CTF_ARMS) * len(CTF_SEEDS)
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_ctf.sub")] = ctf_sub()
    files[os.path.join(HERE, "configs_pofd_qwen7b_ctf_smoke.txt")] = \
        ctf_rows(prefix="pofdctfsmk")
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_ctf_smoke.sub")] = \
        ctf_sub(smoke=True)
    # iclx retry: exactly the 11 g106-killed cells (see ICLX_RETRY_TAGS)
    p = os.path.join(HERE, "configs_pofd_qwen7b_iclf_retry.txt")
    files[p] = iclx_retry_rows("iclf")
    expected[p] = 10
    p = os.path.join(HERE, "configs_pofd_qwen7b_ctf_retry.txt")
    files[p] = iclx_retry_rows("ctf")
    expected[p] = 1
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_iclf_retry.sub")] = \
        iclx_retry_sub("iclf")
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_ctf_retry.sub")] = \
        iclx_retry_sub("ctf")
    # qwen/olmo seed-replication core (see the SC_ comment block): exactly
    # the 34 audited-missing cells (retention 12 + direct 8 + main peer
    # 14); smoke = the 1-round path, never exercised on the cluster
    assert sum(len(sc_rows(m)) for m in SC_MODELS) == 34, \
        [len(sc_rows(m)) for m in SC_MODELS]
    for model in SC_MODELS:
        p = os.path.join(HERE, f"configs_pofd_{model}_seedcore.txt")
        files[p] = sc_rows(model)
        expected[p] = (len(SC_RET_BETAS) * len(SC_RET_SEEDS)
                       + len(SC_DIR_BETAS) * len(SC_DIR_SEEDS)
                       + len(SC_DIR_BETAS) * len(SC_MAIN_SEEDS)
                       - sum(1 for c in SC_MAIN_EXISTING if c[0] == model))
        cube_subs[os.path.join(HERE, f"at_pofd_{model}_seedcore.sub")] = \
            sc_sub(model)
    files[os.path.join(HERE, "configs_pofd_qwen7b_seedcore_smoke.txt")] = [
        ROW_SC.format(tag=sc_ret_tag("qwen7b", 0.5, 0, prefix="pofdretsmk"),
                      style="sft_kl", beta="0.5", seed=0, es="0.1", w="0.5",
                      eps_ai="0.4", nrounds=1, lam="0.2", wtwin=1, ansk=16)]
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_seedcore_smoke.sub")] = \
        sc_sub("qwen7b", smoke=True)
    # final replication wave (see the FF_ comment block): three qwen keys
    # + the untouched olmo7brom_fe wave under the qwen_olmo_finalfill
    # umbrella. HARD job-count asserts: 18 + 16 + 8 here, 9 + 3 in the
    # olmo7brom_fes/fef files above -> the umbrella queues exactly 54.
    assert len(ffsi_rows()) == 18, len(ffsi_rows())
    assert len(ffrp_rows()) == 16, len(ffrp_rows())
    assert len(ffc_rows()) == 8, len(ffc_rows())
    assert len(ffsi_rows()) + len(ffrp_rows()) + len(ffc_rows()) \
        + 3 * len(FE_SEEDS) + len(FE_SEEDS) == 54
    p = os.path.join(HERE, "configs_pofd_qwen7b_finalfill_sfticl.txt")
    files[p] = ffsi_rows()
    expected[p] = (len(FF_SFT_EAS) * len(FF_SEEDS)
                   + len(FF_SFT_EAS) * len(FF_ICL_KS) * len(FF_SEEDS))
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_finalfill_sfticl.sub")] = \
        FFSI_SUB_TEMPLATE.format(n_jobs=len(ffsi_rows()),
                                 **CUBE_MODELS["qwen7b"])
    p = os.path.join(HERE, "configs_pofd_qwen7b_finalfill_replay.txt")
    files[p] = ffrp_rows()
    expected[p] = (len(RPL_BETAS) * len(RPL_FRACS) * len(FF_SEEDS)
                   - len(RPL_BETAS) * len(FF_SEEDS))   # minus rf=0
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_finalfill_replay.sub")] = \
        FFRP_SUB_TEMPLATE.format(n_jobs=len(ffrp_rows()),
                                 **CUBE_MODELS["qwen7b"])
    p = os.path.join(HERE, "configs_pofd_qwen7b_finalfill_corners.txt")
    files[p] = ffc_rows()
    expected[p] = len(FF_CORNER_EAS) * len(FF_CORNER_ESS) * len(FF_SEEDS)
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_finalfill_corners.sub")] = \
        FFC_SUB_TEMPLATE.format(n_jobs=len(ffc_rows()),
                                **CUBE_MODELS["qwen7b"])
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_tfe_ci.sub")] = TFE_CI_SUB
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_dpo_ci1.sub")] = DPO_CI1_SUB
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_dpo_ci2.sub")] = DPO_CI2_SUB
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_dpo_ci3.sub")] = DPO_CI3_SUB
    cube_subs[os.path.join(HERE, "at_pofd_mistral7b_ws2f.sub")] = \
        MISTRAL_WS2F_SUB
    cube_subs[os.path.join(HERE, "at_pofd_mistral7b_ws2f_smoke.sub")] = \
        MISTRAL_WS2F_SMOKE_SUB
    cube_subs[os.path.join(HERE, "at_pofd_mistral7b_cube_s0.sub")] = \
        MISTRAL_CUBE_S0_SUB
    cube_subs[os.path.join(HERE, "at_pofd_mistral7b_cube_repl.sub")] = \
        MISTRAL_CUBE_REPL_SUB
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_dpo_mr.sub")] = DPO_MR_SUB
    cube_subs[os.path.join(HERE, "at_pofd_qwen7b_dpo_mr_smoke.sub")] = \
        DPO_MR_SMOKE_SUB
    # SFT-ICL reach wave (see the REACH block): manifest-driven -- only
    # audited-missing cells queue. HARD ASSERTS pin the independently
    # audited counts (33 reused / 327 missing of 360; per-model 99/108/120;
    # 15 baselines; 6 smokes); a changed corpus must re-run
    # audit_sft_icl_reach_reuse.py, which refuses to write a manifest that
    # contradicts its --expect counts.
    _rman = _reach_manifest()
    assert _rman["counts"]["cells"] == 360, _rman["counts"]
    assert _rman["counts"]["reused"] == REACH_EXPECT_REUSED == 33, \
        _rman["counts"]
    assert _rman["counts"]["new"] == REACH_EXPECT_NEW == 327, _rman["counts"]
    assert _rman["counts"]["new_per_model"] == REACH_EXPECT_NEW_PER_MODEL, \
        _rman["counts"]["new_per_model"]
    assert _rman["counts"]["baselines"] == 15, _rman["counts"]
    assert sum(_rman["counts"]["new_per_arm"].values()) == 327
    assert len(_rman["cells"]) == 360 and len(_rman["baselines"]) == 15
    _reach_reused = {c["run_tag"] for c in _rman["cells"]
                     if c["status"] == "reused"}
    assert len(_reach_reused) == 33
    _prior_tags_reach = {r.split(",")[0]
                         for rows in files.values() for r in rows}
    _reach_all_tags = set()
    for model in REACH_MODELS:
        rows_m = reach_rows(model)
        assert len(rows_m) == REACH_EXPECT_NEW_PER_MODEL[model], \
            (model, len(rows_m))
        base_m = [reach_base_row(model, s) for s in REACH_SEEDS]
        smk_m = reach_smoke_rows(model)
        assert len(base_m) == 5 and len(smk_m) == 2, (model, len(base_m),
                                                      len(smk_m))
        for rows_k in (rows_m, base_m, smk_m):
            tags_k = {r.split(",")[0] for r in rows_k}
            assert not (tags_k & _reach_reused), \
                f"reach queues a reused cell: {tags_k & _reach_reused}"
            assert not (tags_k & _prior_tags_reach), \
                f"reach collides with existing configs: " \
                f"{tags_k & _prior_tags_reach}"
            assert not (tags_k & _reach_all_tags), "duplicate reach tags"
            _reach_all_tags |= tags_k
        short = REACH_KEY[model].split("_")[-1]
        p = os.path.join(HERE, f"configs_pofd_{REACH_KEY[model]}.txt")
        files[p] = rows_m
        expected[p] = REACH_EXPECT_NEW_PER_MODEL[model]
        p = os.path.join(HERE, f"configs_pofd_sft_icl_reach_base_{short}.txt")
        files[p] = base_m
        expected[p] = 5
        p = os.path.join(HERE,
                         f"configs_pofd_sft_icl_reach_smoke_{short}.txt")
        files[p] = smk_m
        expected[p] = 2
        # seed-0 exploratory slab (see the REACH_S0 block): a SUBSET of
        # rows_m with byte-identical rows/tags, so the eventual full
        # release no-ops these cells; counts hard-asserted 5/8/20.
        s0_m = reach_s0_rows(model)
        assert len(s0_m) == REACH_S0_EXPECT[model], (model, len(s0_m))
        assert set(s0_m) <= set(rows_m), \
            f"s0 slab rows not a subset of the {model} production rows"
        p = os.path.join(HERE, f"configs_pofd_sft_icl_reach_s0_{short}.txt")
        files[p] = s0_m
        expected[p] = REACH_S0_EXPECT[model]
        for kind in ("main", "base", "smoke", "s0"):
            key = {"main": REACH_KEY[model],
                   "base": f"sft_icl_reach_base_{short}",
                   "smoke": f"sft_icl_reach_smoke_{short}",
                   "s0": f"sft_icl_reach_s0_{short}"}[kind]
            cube_subs[os.path.join(HERE, f"at_pofd_{key}.sub")] = \
                reach_sub(model, kind)
    assert len(_reach_all_tags) == 327 + 15 + 6
    assert sum(REACH_S0_EXPECT.values()) == 33
    # sft_k0_nopeer wave (see the K0 block): manifest-driven -- only the
    # audited-missing cells queue. The 6 b0/b1 ea0p7 tags are
    # DELIBERATELY shared with the parked full reach production files
    # (same tags -> the eventual reach release no-ops them; never
    # co-submit); every other tag must be globally fresh.
    _kman = _k0_manifest()
    assert _kman["counts"]["cells"] == 54, _kman["counts"]
    assert _kman["counts"]["reused"] == K0_EXPECT_REUSED == 32, \
        _kman["counts"]
    assert _kman["counts"]["new"] == K0_EXPECT_NEW == 22, _kman["counts"]
    assert _kman["counts"]["new_per_model"] == K0_EXPECT_NEW_PER_MODEL
    assert _kman["counts"]["new_per_arm"] == K0_EXPECT_NEW_PER_ARM
    assert _kman["counts"]["new_per_gate"] == K0_EXPECT_NEW_PER_GATE
    assert len(_kman["cells"]) == 54 and len(_kman["baselines"]) == 3
    _k_reused = {c["run_tag"] for c in _kman["cells"]
                 if c["status"] == "reused"}
    assert len(_k_reused) == 32
    _reach_prod_tags = set()
    for _m_r in REACH_MODELS:
        _reach_prod_tags |= {r.split(",")[0] for r in reach_rows(_m_r)}
    _prior_before_k0 = {r.split(",")[0]
                        for rows in files.values() for r in rows}
    for model in REACH_MODELS:
        rows_k = k0_rows(model)
        assert len(rows_k) == K0_EXPECT_NEW_PER_MODEL[model], \
            (model, len(rows_k))
        tags_k = {r.split(",")[0] for r in rows_k}
        assert not (tags_k & _k_reused), \
            f"k0 wave queues a reused cell: {tags_k & _k_reused}"
        _shared_k = {t for t in tags_k
                     if "_ea0p7_" in t and "_k0_" not in t}
        assert len(_shared_k) == 2 and _shared_k <= _reach_prod_tags, \
            f"{model}: b0/b1 ea0p7 tags must equal the reach production " \
            f"tags ({_shared_k})"
        assert not ((tags_k - _shared_k) & _prior_before_k0), \
            f"k0 wave collides: {(tags_k - _shared_k) & _prior_before_k0}"
        short = K0_KEY[model].split("_")[-1]
        p = os.path.join(HERE, f"configs_pofd_{K0_KEY[model]}.txt")
        files[p] = rows_k
        expected[p] = K0_EXPECT_NEW_PER_MODEL[model]
        cube_subs[os.path.join(HERE, f"at_pofd_{K0_KEY[model]}.sub")] = \
            k0_sub(model, "main")
        smk_k = k0_smoke_rows(model)
        if smk_k:
            _smk_tags = {r.split(",")[0] for r in smk_k}
            assert not (_smk_tags & _prior_before_k0)
            p = os.path.join(HERE,
                             f"configs_pofd_sft_k0_nopeer_smoke_{short}.txt")
            files[p] = smk_k
            expected[p] = 1
            cube_subs[os.path.join(
                HERE, f"at_pofd_sft_k0_nopeer_smoke_{short}.sub")] = \
                k0_sub(model, "smoke")
    assert sum(len(k0_smoke_rows(m)) for m in REACH_MODELS) == 2
    # three-seed no-peer grid (see the GRID3 block): manifest-driven;
    # numeric-gate (<= 0.4) tags are DELIBERATELY shared with the held
    # full reach production files; _ea1_ tags are fresh everywhere.
    _gman = _grid3_manifest()
    assert _gman["counts"]["cells"] == 180, _gman["counts"]
    assert _gman["counts"]["reused"] == GRID3_EXPECT_REUSED == 58, \
        _gman["counts"]
    assert _gman["counts"]["new"] == GRID3_EXPECT_NEW == 94, _gman["counts"]
    assert _gman["counts"]["reference_k0"] == GRID3_EXPECT_REFERENCE == 28
    assert _gman["counts"]["missing"] == 122
    assert _gman["counts"]["new_per_model"] == GRID3_EXPECT_NEW_PER_MODEL
    assert _gman["counts"]["new_per_arm"] == GRID3_EXPECT_NEW_PER_ARM
    assert len(_gman["cells"]) == 180 and len(_gman["baselines"]) == 3
    _g_reused = {c["run_tag"] for c in _gman["cells"]
                 if c["status"] == "reused"}
    _g_refs = [c for c in _gman["cells"] if c["status"] == "reference"]
    assert all(c["arm"] == "k0" and c["seed"] in (42, 43)
               for c in _g_refs)
    _prior_before_g3 = {r.split(",")[0]
                        for rows in files.values() for r in rows}
    _g3_total = 0
    for model in REACH_MODELS:
        rows_g = grid3_rows(model)
        assert len(rows_g) == GRID3_EXPECT_NEW_PER_MODEL[model], \
            (model, len(rows_g))
        _g3_total += len(rows_g)
        tags_g = {r.split(",")[0] for r in rows_g}
        assert not (tags_g & _g_reused)
        _shared_g = {t for t in tags_g if "_ea1_" not in t}
        assert _shared_g <= _reach_prod_tags, \
            f"{model}: non-ea1 grid3 tags must be reach production " \
            f"tags: {_shared_g - _reach_prod_tags}"
        _fresh_g = tags_g - _shared_g
        assert all("_ea1_" in t for t in _fresh_g)
        assert not (_fresh_g & _prior_before_g3), \
            f"grid3 ea1 collision: {_fresh_g & _prior_before_g3}"
        p = os.path.join(HERE, f"configs_pofd_{GRID3_KEY[model]}.txt")
        files[p] = rows_g
        expected[p] = GRID3_EXPECT_NEW_PER_MODEL[model]
        cube_subs[os.path.join(HERE, f"at_pofd_{GRID3_KEY[model]}.sub")] = \
            grid3_sub(model, "main")
    assert _g3_total == 94
    _g3_smk = grid3_smoke_rows()
    assert len(_g3_smk) == 2
    assert not ({r.split(",")[0] for r in _g3_smk} & _prior_before_g3)
    p = os.path.join(HERE, "configs_pofd_sft_icl_nopeer_grid3_smoke.txt")
    files[p] = _g3_smk
    expected[p] = 2
    cube_subs[os.path.join(HERE,
                           "at_pofd_sft_icl_nopeer_grid3_smoke.sub")] = \
        grid3_sub("mistral7b", "smoke")
    # eps_social=0.2 channel table (see the PEER2 block): manifest-driven,
    # NEW pofdpeer2_ family -- zero shared tags with any other wave.
    _pman = _peer2_manifest()
    assert _pman["counts"]["cells"] == 72, _pman["counts"]
    assert _pman["counts"]["reused"] == PEER2_EXPECT_REUSED == 27, \
        _pman["counts"]
    assert _pman["counts"]["new"] == PEER2_EXPECT_NEW == 45, _pman["counts"]
    assert _pman["counts"]["new_per_model"] == PEER2_EXPECT_NEW_PER_MODEL
    assert _pman["counts"]["reused_per_model"] == \
        PEER2_EXPECT_REUSED_PER_MODEL
    assert len(_pman["cells"]) == 72 and len(_pman["baselines"]) == 3
    assert all(c.get("validation") in ("PASS", "SKIPPED")
               for c in _pman["cells"] if c["status"] == "reused"), \
        "peer2 manifest carries a reused cell that FAILED validation"
    _p_reused = {c["run_tag"] for c in _pman["cells"]
                 if c["status"] == "reused"}
    _prior_before_p2 = {r.split(",")[0]
                        for rows in files.values() for r in rows}
    _p2_total = 0
    for model in REACH_MODELS:
        rows_p = peer2_rows(model)
        assert len(rows_p) == PEER2_EXPECT_NEW_PER_MODEL[model], \
            (model, len(rows_p))
        _p2_total += len(rows_p)
        tags_p = {r.split(",")[0] for r in rows_p}
        assert not (tags_p & _p_reused)
        assert not (tags_p & _prior_before_p2), \
            f"peer2 collision: {tags_p & _prior_before_p2}"
        p = os.path.join(HERE, f"configs_pofd_{PEER2_KEY[model]}.txt")
        files[p] = rows_p
        expected[p] = PEER2_EXPECT_NEW_PER_MODEL[model]
        cube_subs[os.path.join(HERE, f"at_pofd_{PEER2_KEY[model]}.sub")] = \
            peer2_sub(model, "main")
    assert _p2_total == 45
    _p2_smk = peer2_smoke_rows()
    assert len(_p2_smk) == 2
    assert not ({r.split(",")[0] for r in _p2_smk} & _prior_before_p2)
    p = os.path.join(HERE, "configs_pofd_sft_icl_peer02_smoke.txt")
    files[p] = _p2_smk
    expected[p] = 2
    cube_subs[os.path.join(HERE, "at_pofd_sft_icl_peer02_smoke.sub")] = \
        peer2_sub("mistral7b", "smoke")
    # mistral 2-D gate grid (see the GATE2D block): manifest-driven,
    # NEW pofdgate2d_ family -- zero shared tags with any other wave.
    _gman = _gate2d_manifest()
    assert _gman["counts"]["cells"] == 120, _gman["counts"]
    assert _gman["counts"]["reused"] == GATE2D_EXPECT_REUSED == 42, \
        _gman["counts"]
    assert _gman["counts"]["new"] == GATE2D_EXPECT_NEW == 78, \
        _gman["counts"]
    assert _gman["counts"]["new_per_es"] == GATE2D_EXPECT_NEW_PER_ES
    assert _gman["counts"]["reused_per_es"] == GATE2D_EXPECT_REUSED_PER_ES
    assert _gman["counts"]["new_per_arm"] == GATE2D_EXPECT_NEW_PER_ARM
    assert all(c.get("validation") in ("PASS", "SKIPPED")
               for c in _gman["cells"] if c["status"] == "reused"), \
        "gate2d manifest carries a reused cell that FAILED validation"
    _g2_reused = {c["run_tag"] for c in _gman["cells"]
                  if c["status"] == "reused"}
    _prior_before_g2d = {r.split(",")[0]
                         for rows in files.values() for r in rows}
    rows_g2 = gate2d_rows()
    assert len(rows_g2) == GATE2D_EXPECT_NEW == 78
    tags_g2 = {r.split(",")[0] for r in rows_g2}
    assert not (tags_g2 & _g2_reused)
    assert not (tags_g2 & _prior_before_g2d), \
        f"gate2d collision: {tags_g2 & _prior_before_g2d}"
    p = os.path.join(HERE, f"configs_pofd_{GATE2D_KEY}.txt")
    files[p] = rows_g2
    expected[p] = GATE2D_EXPECT_NEW
    cube_subs[os.path.join(HERE, f"at_pofd_{GATE2D_KEY}.sub")] = \
        gate2d_sub("main")
    _g2_smk = gate2d_smoke_rows()
    assert len(_g2_smk) == 2
    assert not ({r.split(",")[0] for r in _g2_smk}
                & (_prior_before_g2d | tags_g2))
    p = os.path.join(HERE, f"configs_pofd_{GATE2D_SMOKE_KEY}.txt")
    files[p] = _g2_smk
    expected[p] = 2
    cube_subs[os.path.join(HERE, f"at_pofd_{GATE2D_SMOKE_KEY}.sub")] = \
        gate2d_sub("smoke")
    # one-seed context-depth x dual-gate grid (see the CTXGRID block):
    # manifest-driven, NEW pofdctxgrid_ family -- zero shared tags.
    _cman = _ctxgrid_manifest()
    assert _cman["counts"]["cells"] == CTXGRID_EXPECT_CELLS == 360, \
        _cman["counts"]
    assert _cman["counts"]["reused"] == CTXGRID_EXPECT_REUSED == 139, \
        _cman["counts"]
    assert _cman["counts"]["new"] == CTXGRID_EXPECT_NEW == 181, \
        _cman["counts"]
    assert _cman["counts"]["excluded"] == CTXGRID_EXPECT_EXCLUDED == 40, \
        _cman["counts"]
    assert (_cman["counts"]["reused"] + _cman["counts"]["new"]
            + _cman["counts"]["excluded"]) == 360, _cman["counts"]
    assert _cman["counts"]["new_per_model"] == CTXGRID_EXPECT_NEW_PER_MODEL
    assert _cman["counts"]["new_per_arm"] == CTXGRID_EXPECT_NEW_PER_ARM
    # the excluded channel must never reach the queue
    assert not [c for c in _cman["cells"]
                if c["status"] == "new" and c["model"] == "mistral7b"
                and c["arm"] in ("f32", "d32")], \
        "mistral7b K=32 is excluded (no parseable signal) -- must not queue"
    assert all(c.get("validation") in ("PASS", "SKIPPED")
               for c in _cman["cells"] if c["status"] == "reused"), \
        "ctxgrid manifest carries a reused cell that FAILED validation"
    # fixed K=32 must be entirely NEW: no archived run pairs icl_k=32
    # with icl_snapshot_round=0, and k32pri/k32noai are a different
    # context source entirely
    assert not [c for c in _cman["cells"]
                if c["arm"] == "f32" and c["status"] == "reused"], \
        "fixed K=32 cannot reuse any archived run"
    _c_reused = {c["run_tag"] for c in _cman["cells"]
                 if c["status"] == "reused"}
    _prior_before_cg = {r.split(",")[0]
                        for rows in files.values() for r in rows}
    _cg_total = 0
    for model in REACH_MODELS:
        rows_c = ctxgrid_rows(model)
        assert len(rows_c) == CTXGRID_EXPECT_NEW_PER_MODEL[model], \
            (model, len(rows_c))
        _cg_total += len(rows_c)
        tags_c = {r.split(",")[0] for r in rows_c}
        assert not (tags_c & _c_reused)
        assert not (tags_c & _prior_before_cg), \
            f"ctxgrid collision: {tags_c & _prior_before_cg}"
        p = os.path.join(HERE, f"configs_pofd_{CTXGRID_KEY[model]}.txt")
        files[p] = rows_c
        expected[p] = CTXGRID_EXPECT_NEW_PER_MODEL[model]
        cube_subs[os.path.join(HERE,
                               f"at_pofd_{CTXGRID_KEY[model]}.sub")] = \
            ctxgrid_sub(model, "main")
    assert _cg_total == CTXGRID_EXPECT_NEW == 181
    _cg_smk = ctxgrid_smoke_rows()
    assert len(_cg_smk) == 2
    assert not ({r.split(",")[0] for r in _cg_smk} & _prior_before_cg)
    p = os.path.join(HERE, f"configs_pofd_{CTXGRID_SMOKE_KEY}.txt")
    files[p] = _cg_smk
    expected[p] = 2
    cube_subs[os.path.join(HERE, f"at_pofd_{CTXGRID_SMOKE_KEY}.sub")] = \
        ctxgrid_sub("mistral7b", "smoke")
    _cg_dbg = ctxgrid_dbg_rows()
    assert len(_cg_dbg) == 2
    assert not ({r.split(",")[0] for r in _cg_dbg}
                & (_prior_before_cg | tags_c
                   | {r.split(",")[0] for r in _cg_smk}))
    p = os.path.join(HERE, f"configs_pofd_{CTXGRID_DBG_KEY}.txt")
    files[p] = _cg_dbg
    expected[p] = 2
    cube_subs[os.path.join(HERE, f"at_pofd_{CTXGRID_DBG_KEY}.sub")] = \
        ctxgrid_dbg_sub()
    # Figure-2 provider replication (see the FIG2 block): 6 jobs
    # completing three-seed b1 coverage. The four es=0 rows DELIBERATELY
    # share tags with the unreleased reach production, so they are
    # exempted from the cross-wave collision assert below.
    _fman = _fig2_manifest()
    assert _fman["counts"]["cells"] == 36, _fman["counts"]
    assert _fman["counts"]["new"] == FIG2_EXPECT_NEW == 6, _fman["counts"]
    assert _fman["counts"]["reused"] == 30, _fman["counts"]
    assert all(c.get("validation") in ("PASS", "SKIPPED")
               for c in _fman["cells"] if c["status"] == "reused"), \
        "fig2 manifest carries a reused cell that FAILED validation"
    _fig2_total = 0
    _fig2_shared = set()
    for model in FIG2_KEY:
        rows_f = fig2_rows(model)
        assert len(rows_f) == FIG2_EXPECT_PER_MODEL[model], \
            (model, len(rows_f))
        _fig2_total += len(rows_f)
        tags_f = {r.split(",")[0] for r in rows_f}
        # es=0 tags are the reach production's by construction; es=0.2
        # tags must be genuinely new
        shared = {t for t in tags_f if t.startswith("pofdreach_")}
        assert len(shared) == 2, (model, shared)
        _fig2_shared |= shared
        fresh = tags_f - shared
        prior = {r.split(",")[0] for rows in files.values() for r in rows}
        assert not (fresh & prior), f"fig2 collision: {fresh & prior}"
        p = os.path.join(HERE, f"configs_pofd_{FIG2_KEY[model]}.txt")
        files[p] = rows_f
        expected[p] = FIG2_EXPECT_PER_MODEL[model]
        cube_subs[os.path.join(HERE,
                               f"at_pofd_{FIG2_KEY[model]}.sub")] = \
            fig2_sub(model)
    assert _fig2_total == FIG2_EXPECT_NEW == 6
    # the shared es=0 tags must be exactly the reach rows they claim to be
    _reach_all = set()
    for _m in REACH_MODELS:
        _reach_all |= {r.split(",")[0] for r in reach_rows(_m)}
    assert _fig2_shared <= _reach_all, \
        f"fig2 es=0 tags not found in the reach production: " \
        f"{_fig2_shared - _reach_all}"
    # no-peer innate-clamp wave (see the CLAMP block): 60 new jobs, no
    # audit -- the intervention is brand new, so no archived run can
    # hold it, and the NEW pofdclamp_ family shares zero tags with any
    # other wave.
    rows_clamp = clamp_rows()
    assert len(rows_clamp) == CLAMP_EXPECT_NEW == 60, len(rows_clamp)
    _cl_tags = {r.split(",")[0] for r in rows_clamp}
    assert len(_cl_tags) == 60
    for _tok, _n_want in (("_strat_", 30), ("_bottom_", 30),
                          ("_b0_", 30), ("_dyn_", 30)):
        _n_got = sum(1 for t in _cl_tags if _tok in t)
        assert _n_got == _n_want, (_tok, _n_got)
    for _sd in CLAMP_SEEDS:
        assert sum(1 for t in _cl_tags if t.endswith(f"_s{_sd}")) == 20
    for _g in CLAMP_GATES:
        assert sum(1 for t in _cl_tags
                   if f"_ea{_num(_g)}_" in t) == 12, _g
    assert all("_es0_" in t for t in _cl_tags), \
        "clamp is no-peer only: every tag must carry _es0_"
    assert all(r.rstrip().endswith((" stratified_random", " bottom"))
               for r in rows_clamp), "queue col 24 must carry the mode"
    _prior_before_cl = {r.split(",")[0]
                       for rows in files.values() for r in rows}
    assert not (_cl_tags & _prior_before_cl), \
        f"clamp collision: {_cl_tags & _prior_before_cl}"
    p = os.path.join(HERE, f"configs_pofd_{CLAMP_KEY}.txt")
    files[p] = rows_clamp
    expected[p] = CLAMP_EXPECT_NEW
    cube_subs[os.path.join(HERE, f"at_pofd_{CLAMP_KEY}.sub")] = \
        clamp_sub("main")
    _cl_smk = clamp_smoke_rows()
    assert len(_cl_smk) == 4
    assert all("_ea0p2_" in r.split(",")[0] and "_s991" in r
               for r in _cl_smk)
    p = os.path.join(HERE, f"configs_pofd_{CLAMP_SMOKE_KEY}.txt")
    files[p] = _cl_smk
    expected[p] = 4
    cube_subs[os.path.join(HERE, f"at_pofd_{CLAMP_SMOKE_KEY}.sub")] = \
        clamp_sub("smoke")
    # innate-clamp GRAPH-PLACEMENT wave (see the CLAMP_GRAPH block):
    # 96 seed-0 jobs, same pofdclamp_ family with mandatory
    # _gclump_/_gscat_ + _stub_ tokens. HARD GATE: the mask artifact
    # must exist with every acceptance criterion true BEFORE a single
    # job row is emitted.
    with open(CLAMP_GRAPH_ARTIFACT) as _gm_fh:
        _gm_art = json.load(_gm_fh)
    assert _gm_art["n_fixed"] == 145 and _gm_art["n"] == 723
    assert all(_gm_art["criteria"].values()), \
        f"mask artifact criteria not met: {_gm_art['criteria']}"
    for _mname in ("graph_clumped", "graph_scattered"):
        assert len(_gm_art["masks"][_mname]["ids"]) == 145
    rows_cg2 = clamp_graph_rows()
    assert len(rows_cg2) == CLAMP_GRAPH_EXPECT_NEW == 96, len(rows_cg2)
    _cg2_tags = {r.split(",")[0] for r in rows_cg2}
    assert len(_cg2_tags) == 96
    for _tok, _n_want in (("_gclump_", 48), ("_gscat_", 48),
                          ("_b0_", 48), ("_dyn_", 48),
                          ("_stub_", 96)):
        _n_got = sum(1 for t in _cg2_tags if _tok in t)
        assert _n_got == _n_want, (_tok, _n_got)
    assert not any("_iso_" in t for t in _cg2_tags), \
        "there is no isolated condition in this design"
    for _g in CLAMP_GRAPH_GATES:
        assert sum(1 for t in _cg2_tags
                   if f"_ea{_num(_g)}_" in t) == 24, _g
    for _e in CLAMP_GRAPH_ESS:
        assert sum(1 for t in _cg2_tags
                   if f"_es{_num(_e)}_" in t) == 16, _e
    assert all(t.endswith("_s0") for t in _cg2_tags)
    _prior_before_cg2 = {r.split(",")[0]
                        for rows in files.values() for r in rows}
    assert not (_cg2_tags & _prior_before_cg2), \
        f"clamp_graph collision: {_cg2_tags & _prior_before_cg2}"
    p = os.path.join(HERE, f"configs_pofd_{CLAMP_GRAPH_KEY}.txt")
    files[p] = rows_cg2
    expected[p] = CLAMP_GRAPH_EXPECT_NEW
    cube_subs[os.path.join(HERE, f"at_pofd_{CLAMP_GRAPH_KEY}.sub")] = \
        clamp_graph_sub("main")
    _cg2_smk = clamp_graph_smoke_rows()
    assert len(_cg2_smk) == 4
    assert all(("_gclump_" in r.split(",")[0]
                or "_gscat_" in r.split(",")[0])
               and "_stub_" in r.split(",")[0]
               and "_ea0p4_" in r.split(",")[0]
               and "_es0p4_" in r.split(",")[0]
               and "_s991" in r.split(",")[0] for r in _cg2_smk)
    p = os.path.join(HERE, f"configs_pofd_{CLAMP_GRAPH_SMOKE_KEY}.txt")
    files[p] = _cg2_smk
    expected[p] = 4
    cube_subs[os.path.join(HERE,
                           f"at_pofd_{CLAMP_GRAPH_SMOKE_KEY}.sub")] = \
        clamp_graph_sub("smoke")
    # personal-history D8 wave (see the CLAMP_GRAPH_D8 block): the 48
    # NEW _d8_ cells + the 3 _b0_ cells missing at the 2026-08-17
    # cluster audit = 51 jobs. The 3 backfill tags are DELIBERATELY
    # shared with the main graph key (no collision assert for them --
    # the retry-key precedent), never co-submit the two keys. The 48
    # _d8_ tags must collide with NOTHING.
    rows_cg8 = clamp_graph_d8_rows()
    assert len(rows_cg8) == 51, len(rows_cg8)
    _cg8_tags = {r.split(",")[0] for r in rows_cg8}
    assert len(_cg8_tags) == 51
    _cg8_d8 = {t for t in _cg8_tags if "_d8_" in t}
    _cg8_b0 = _cg8_tags - _cg8_d8
    assert len(_cg8_d8) == 48 and len(_cg8_b0) == 3
    assert all("_stub_" in t and t.endswith("_s0") for t in _cg8_tags)
    assert not any("_dyn_" in t for t in _cg8_tags), \
        "the corrected wave has no cross-user arm"
    for _tok, _n_want in (("_gclump_", 24), ("_gscat_", 24)):
        assert sum(1 for t in _cg8_d8 if _tok in t) == _n_want, _tok
    for _g in CLAMP_GRAPH_GATES:
        assert sum(1 for t in _cg8_d8
                   if f"_ea{_num(_g)}_" in t) == 12, _g
    for _e in CLAMP_GRAPH_ESS:
        assert sum(1 for t in _cg8_d8
                   if f"_es{_num(_e)}_" in t) == 8, _e
    assert _cg8_b0 <= _cg2_tags, \
        "b0 backfill cells must be a subset of the main graph wave"
    # every d8 row carries icldays=8 (last queue col), every b0 row 0
    for r in rows_cg8:
        _tail = ", 8" if "_d8_" in r.split(",")[0] else ", 0"
        assert r.rstrip().endswith(_tail), r
    _prior_before_cg8 = {r.split(",")[0]
                         for rows in files.values() for r in rows}
    assert not (_cg8_d8 & _prior_before_cg8), \
        f"d8 collision: {_cg8_d8 & _prior_before_cg8}"
    p = os.path.join(HERE, f"configs_pofd_{CLAMP_GRAPH_D8_KEY}.txt")
    files[p] = rows_cg8
    expected[p] = 51
    cube_subs[os.path.join(HERE,
                           f"at_pofd_{CLAMP_GRAPH_D8_KEY}.sub")] = \
        clamp_graph_d8_sub("main")
    _cg8_smk = clamp_graph_d8_smoke_rows()
    assert len(_cg8_smk) == 2
    assert all("_d8_" in r.split(",")[0]
               and "_stub_ea0p4_" in r.split(",")[0]
               and "_es0p4_s991" in r.split(",")[0] for r in _cg8_smk)
    assert sum(1 for r in _cg8_smk if "_gclump_" in r) == 1
    p = os.path.join(HERE,
                     f"configs_pofd_{CLAMP_GRAPH_D8_SMOKE_KEY}.txt")
    files[p] = _cg8_smk
    expected[p] = 2
    cube_subs[os.path.join(HERE,
                           f"at_pofd_{CLAMP_GRAPH_D8_SMOKE_KEY}.sub")] = \
        clamp_graph_d8_sub("smoke")
    # source-exclusion wave (see the CLAMP_XA block): 48 NEW _b0xa_
    # cells on the completed graph grid; the 48 _b0_ and 48 _d8_ cells
    # are REUSED (never re-queued, never relabeled), so the _b0xa_
    # tags must collide with NOTHING anywhere.
    rows_cxa = clamp_xa_rows()
    assert len(rows_cxa) == 48, len(rows_cxa)
    _cxa_tags = {r.split(",")[0] for r in rows_cxa}
    assert len(_cxa_tags) == 48
    assert all("_b0xa_" in t and "_stub_" in t and t.endswith("_s0")
               for t in _cxa_tags)
    for _tok, _n_want in (("_gclump_", 24), ("_gscat_", 24)):
        assert sum(1 for t in _cxa_tags if _tok in t) == _n_want, _tok
    for _g in CLAMP_GRAPH_GATES:
        assert sum(1 for t in _cxa_tags
                   if f"_ea{_num(_g)}_" in t) == 12, _g
    for _e in CLAMP_GRAPH_ESS:
        assert sum(1 for t in _cxa_tags
                   if f"_es{_num(_e)}_" in t) == 8, _e
    # every row keeps the exact b0 queue surface (sft, beta 0, no ICL)
    for r in rows_cxa:
        _cols = [c.strip() for c in r.split(",")]
        assert _cols[1] == "sft" and _cols[2] == "0" \
            and _cols[16] == "0", r
    _prior_before_cxa = {r.split(",")[0]
                         for rows in files.values() for r in rows}
    assert not (_cxa_tags & _prior_before_cxa), \
        f"b0xa collision: {_cxa_tags & _prior_before_cxa}"
    assert "SFT_EXCLUDE_CLAMPED=1" in clamp_xa_sub("main")
    assert "SFT_EXCLUDE_CLAMPED=1" in clamp_xa_sub("smoke")
    p = os.path.join(HERE, f"configs_pofd_{CLAMP_XA_KEY}.txt")
    files[p] = rows_cxa
    expected[p] = 48
    cube_subs[os.path.join(HERE, f"at_pofd_{CLAMP_XA_KEY}.sub")] = \
        clamp_xa_sub("main")
    _cxa_smk = clamp_xa_smoke_rows()
    assert len(_cxa_smk) == 2
    assert all("_b0xa_" in r.split(",")[0]
               and "_stub_ea0p4_" in r.split(",")[0]
               and "_es0p2_s991" in r.split(",")[0] for r in _cxa_smk)
    assert sum(1 for r in _cxa_smk if "_gclump_" in r) == 1
    p = os.path.join(HERE, f"configs_pofd_{CLAMP_XA_SMOKE_KEY}.txt")
    files[p] = _cxa_smk
    expected[p] = 2
    cube_subs[os.path.join(HERE,
                           f"at_pofd_{CLAMP_XA_SMOKE_KEY}.sub")] = \
        clamp_xa_sub("smoke")
    # bottom-20% source-impact wave (see the B20 block, full-grid
    # revision): 68 NEW seed-0 jobs from the audited manifest; the 4
    # completed b0 bottom no-peer cells reuse and must NEVER re-queue
    # (their tokenless tags stay with mistral_innate_clamp_nopeer).
    rows_b20 = b20_rows()
    assert len(rows_b20) == 68, len(rows_b20)
    _b20_tags = {r.split(",")[0] for r in rows_b20}
    assert len(_b20_tags) == 68
    assert all("_bottom_stub_" in t and t.endswith("_s0")
               and "mistral7b" in t for t in _b20_tags)
    for _arm_tok, _nw in (("_b0_", 20), ("_b0xa_", 24), ("_d8_", 24)):
        assert sum(1 for t in _b20_tags if _arm_tok in t) == _nw, \
            _arm_tok
    assert not any("_dyn_" in t for t in _b20_tags)
    # the reused tokenless b0 es0 tags never appear here
    assert not any("_b0_bottom_ea" in t for t in _b20_tags)
    for _g in B20_GATES:
        assert sum(1 for t in _b20_tags
                   if f"_ea{_num(_g)}_" in t) == 17, _g
    for _e in B20_ESS:
        _nw = 8 if _e == 0.0 else 12
        assert sum(1 for t in _b20_tags
                   if f"_es{_num(_e)}_" in t) == _nw, _e
    # queue tails: (icldays, sftexcl) = (8, 0) on d8, (0, 1) on
    # b0xa, (0, 0) on the b0 peer cells
    for r in rows_b20:
        _t = r.split(",")[0]
        _want_tail = (", 8, 0" if "_d8_" in _t
                      else ", 0, 1" if "_b0xa_" in _t else ", 0, 0")
        assert r.rstrip().endswith(_want_tail), r
    assert "INNATE_CLAMP_PEER_MODE=stubborn" in b20_sub()
    _prior_b20 = {r.split(",")[0]
                  for rows in files.values() for r in rows}
    assert not (_b20_tags & _prior_b20), \
        f"b20 collision: {_b20_tags & _prior_b20}"
    p = os.path.join(HERE, f"configs_pofd_{B20_KEY}.txt")
    files[p] = rows_b20
    expected[p] = 68
    cube_subs[os.path.join(HERE, f"at_pofd_{B20_KEY}.sub")] = b20_sub()
    # fully-evolving comparison wave (see the EVO block): 48 seed-0
    # jobs in the brand-new pofdevo_ family -- by design NOTHING is
    # reused, so the tags must collide with nothing anywhere.
    rows_evo = evo_rows()
    assert len(rows_evo) == 48, len(rows_evo)
    _evo_tags = {r.split(",")[0] for r in rows_evo}
    assert len(_evo_tags) == 48
    assert all(t.startswith("pofdevo_mistral7b_")
               and t.endswith("_s0") for t in _evo_tags)
    for _arm_tok, _nw in (("_b0_", 24), ("_d8_", 24)):
        assert sum(1 for t in _evo_tags if _arm_tok in t) == _nw, \
            _arm_tok
    assert not any("bottom" in t or "_stub_" in t for t in _evo_tags)
    for _g in EVO_GATES:
        assert sum(1 for t in _evo_tags
                   if f"_ea{_num(_g)}_" in t) == 12, _g
    for _e in EVO_ESS:
        assert sum(1 for t in _evo_tags
                   if f"_es{_num(_e)}_" in t) == 8, _e
    # icldays tail: 8 on d8, 0 on b0
    for r in rows_evo:
        _want_tail = ", 8" if "_d8_" in r.split(",")[0] else ", 0"
        assert r.rstrip().endswith(_want_tail), r
    assert "INNATE_CLAMP" not in evo_sub(), \
        "the fully-evolving sub must not carry any clamp env"
    _prior_evo = {r.split(",")[0]
                  for rows in files.values() for r in rows}
    assert not (_evo_tags & _prior_evo), \
        f"evo collision: {_evo_tags & _prior_evo}"
    p = os.path.join(HERE, f"configs_pofd_{EVO_KEY}.txt")
    files[p] = rows_evo
    expected[p] = 48
    cube_subs[os.path.join(HERE, f"at_pofd_{EVO_KEY}.sub")] = evo_sub()
    # Section-4 three-seed replication (see the B20R block): the 152
    # audited-missing seed-42/43 cells in two schemas under one
    # umbrella key. Counts come from the manifest and are asserted
    # for CONSISTENCY with the 2026-08-19 audit (40 reused / 152
    # new), never forced.
    rows_b20r_f, rows_b20r_e = b20r_rows()
    assert len(rows_b20r_f) == 88 and len(rows_b20r_e) == 64, \
        (len(rows_b20r_f), len(rows_b20r_e))
    _b20r_f_tags = {r.split(",")[0] for r in rows_b20r_f}
    _b20r_e_tags = {r.split(",")[0] for r in rows_b20r_e}
    assert len(_b20r_f_tags) == 88 and len(_b20r_e_tags) == 64
    assert all("_bottom_stub_" in t and "mistral7b" in t
               and (t.endswith("_s42") or t.endswith("_s43"))
               for t in _b20r_f_tags)
    assert all(t.startswith("pofdevo_mistral7b_")
               and (t.endswith("_s42") or t.endswith("_s43"))
               for t in _b20r_e_tags)
    assert not any("bottom" in t or "_stub_" in t
                   for t in _b20r_e_tags)
    assert not any("_b0xa_" in t or "_dyn_" in t
                   for t in _b20r_f_tags | _b20r_e_tags)
    for _sd in B20R_SEEDS:
        assert sum(1 for t in _b20r_f_tags
                   if t.endswith(f"_s{_sd}")) == 44, _sd
        assert sum(1 for t in _b20r_e_tags
                   if t.endswith(f"_s{_sd}")) == 32, _sd
    for _arm_tok, _nf, _ne in (("_b0_", 40, 16), ("_d8_", 48, 48)):
        assert sum(1 for t in _b20r_f_tags if _arm_tok in t) == _nf
        assert sum(1 for t in _b20r_e_tags if _arm_tok in t) == _ne
    # the reused cells never re-queue: fixed b0 stays off es0 (the
    # tokenless originals), evolving b0 runs only the es 0.05/0.1
    # cells the archive lacks
    assert not any("_b0_" in t and "_es0_" in t for t in _b20r_f_tags)
    assert all("_es0p05_" in t or "_es0p1_" in t
               for t in _b20r_e_tags if "_b0_" in t)
    # queue tails: fixed (icldays, sftexcl) = (8, 0) on d8, (0, 0)
    # on b0 (b0xa is not part of this wave); evo icldays only
    for r in rows_b20r_f:
        _want_tail = ", 8, 0" if "_d8_" in r.split(",")[0] else ", 0, 0"
        assert r.rstrip().endswith(_want_tail), r
    for r in rows_b20r_e:
        _want_tail = ", 8" if "_d8_" in r.split(",")[0] else ", 0"
        assert r.rstrip().endswith(_want_tail), r
    assert "INNATE_CLAMP_PEER_MODE=stubborn" in b20r_fixed_sub()
    assert "INNATE_CLAMP" not in b20r_evo_sub(), \
        "the evolving repl sub must not carry any clamp env"
    _prior_b20r = {r.split(",")[0]
                   for rows in files.values() for r in rows}
    assert not ((_b20r_f_tags | _b20r_e_tags) & _prior_b20r), \
        f"b20r collision: {(_b20r_f_tags | _b20r_e_tags) & _prior_b20r}"
    p = os.path.join(HERE, f"configs_pofd_{B20R_FIXED_KEY}.txt")
    files[p] = rows_b20r_f
    expected[p] = 88
    cube_subs[os.path.join(HERE, f"at_pofd_{B20R_FIXED_KEY}.sub")] = \
        b20r_fixed_sub()
    p = os.path.join(HERE, f"configs_pofd_{B20R_EVO_KEY}.txt")
    files[p] = rows_b20r_e
    expected[p] = 64
    cube_subs[os.path.join(HERE, f"at_pofd_{B20R_EVO_KEY}.sub")] = \
        b20r_evo_sub()
    # feature-endogenization five-seed extension (see the FE5 block):
    # exactly 12 conceptual cells (6 conditions x seeds 44/45), all
    # NEW per the audited manifest, split across the four established
    # queue schemas. The established seeds {0,42,43} must NEVER
    # re-queue here.
    _fe5_all = {}
    for _arm in ("nat", "frozen", "gd", "gp"):
        _rows = fe5_rows(_arm)
        _tags = {r.split(",")[0] for r in _rows}
        assert len(_tags) == len(_rows), _arm
        _fe5_all[_arm] = (_rows, _tags)
    _fe5_tags = set().union(*(t for _, t in _fe5_all.values()))
    _fe5_n = sum(len(r) for r, _ in _fe5_all.values())
    # exactly 12 conceptual cells, unique tags, correct per-arm split
    assert _fe5_n == 12, _fe5_n
    assert len(_fe5_tags) == 12, len(_fe5_tags)
    assert len(_fe5_all["nat"][0]) == 6
    assert all(len(_fe5_all[a][0]) == 2 for a in
               ("frozen", "gd", "gp"))
    assert all("_qwen7b_" in t for t in _fe5_tags)
    assert all(t.endswith("_s44") or t.endswith("_s45")
               or t.endswith("_s44_fresh_data")
               or t.endswith("_s45_fresh_data") for t in _fe5_tags)
    # no established seed may appear anywhere in this key
    for _sd in (0, 42, 43):
        assert not any(f"_s{_sd}_" in t or t.endswith(f"_s{_sd}")
                       for t in _fe5_tags), _sd
    # the six conditions, one family each
    for _pre, _nw in (("pofdws2f_qwen7b_b0_", 2),
                      ("pofdws2f_qwen7b_b0p5_", 2),
                      ("pofdws2f_qwen7b_b1_", 2),
                      ("pofdicls2_qwen7b_", 2),
                      ("pofdfegd_qwen7b_b1_", 2),
                      ("pofdfegp_qwen7b_b1_", 2)):
        assert sum(1 for t in _fe5_tags
                   if t.startswith(_pre)) == _nw, _pre
    # every established feature tag stays untouched by this key
    _fe5_base = {fe5_tag(_c, _s)
                 for _c in ("nat_l0", "nat_l0p5", "nat_l1", "frozen",
                            "removed", "permuted")
                 for _s in (0, 42, 43)}
    assert not (_fe5_tags & _fe5_base), \
        f"fe5 would re-queue established runs: {_fe5_tags & _fe5_base}"
    # and collides with nothing anywhere else in the generator
    _prior_fe5 = {r.split(",")[0]
                  for rows in files.values() for r in rows}
    assert not (_fe5_tags & _prior_fe5), \
        f"fe5 collision: {_fe5_tags & _prior_fe5}"
    for _arm, (_rows, _tags) in _fe5_all.items():
        _sub = fe5_sub(_arm)
        # GPU pinned to the established architecture on every schema
        assert f'CUDADeviceName == "{FE5_A100}"' in _sub, _arm
        assert "KL_DIRECTION=forward" in _sub or _arm == "frozen"
        if _arm == "gd":
            assert "PROFILE_DROP_COLS=gender" in _sub
        if _arm == "gp":
            assert "PROFILE_PERMUTE_COLS=gender" in _sub
        if _arm in ("nat", "frozen"):
            assert "PROFILE_DROP_COLS" not in _sub \
                and "PROFILE_PERMUTE_COLS" not in _sub, _arm
        p = os.path.join(HERE, f"configs_pofd_{FE5_KEY}_{_arm}.txt")
        files[p] = _rows
        expected[p] = len(_rows)
        cube_subs[os.path.join(
            HERE, f"at_pofd_{FE5_KEY}_{_arm}.sub")] = _sub
    # Qwen gate sweep (see the QGS block): the 30 audited-missing
    # cells of the 60-cell seed-0 grid, in the new pofdqgs_ family.
    rows_qgs = qgs_rows()
    assert len(rows_qgs) == 30, len(rows_qgs)
    _qgs_tags = {r.split(",")[0] for r in rows_qgs}
    assert len(_qgs_tags) == 30
    assert all(t.startswith("pofdqgs_") and "_b1_" in t
               and t.endswith("_s0") for t in _qgs_tags)
    for _m in QGS_MODELS:
        assert sum(1 for t in _qgs_tags
                   if t.startswith(f"pofdqgs_{_m}_")) == \
            (12 if _m == "qwen7b" else 18), _m
    # every queued cell is inside the declared grid
    _qgs_grid = {qgs_tag(_m, _g, _e) for _m in QGS_MODELS
                 for _g in QGS_GATES for _e in QGS_ESS}
    assert _qgs_tags <= _qgs_grid, _qgs_tags - _qgs_grid
    # queue tails: model resources ride cols 24-28; qwen3 pins
    # thinking OFF, qwen7b keeps its default template
    for r in rows_qgs:
        _cols = [c.strip() for c in r.split(",")]
        assert _cols[1] == "sft_kl" and _cols[2] == "1", r
        assert _cols[15] == "threshold", r
        if "_qwen3_8b_" in _cols[0]:
            assert _cols[23] == "Qwen/Qwen3-8B", r
            assert _cols[24] == "0", r
        else:
            assert _cols[23] == "Qwen/Qwen2.5-7B-Instruct", r
            assert _cols[24] == "default", r
    _qgs_sub = qgs_sub()
    assert "CHAT_THINKING=$(chatthink)" in _qgs_sub
    assert "KL_DIRECTION=forward" in _qgs_sub
    assert "WITH_TWIN=1" in _qgs_sub
    assert "SAVE_RAW_GEN" not in _qgs_sub
    _prior_qgs = {r.split(",")[0]
                  for rows in files.values() for r in rows}
    assert not (_qgs_tags & _prior_qgs), \
        f"qgs collision: {_qgs_tags & _prior_qgs}"
    p = os.path.join(HERE, f"configs_pofd_{QGS_KEY}.txt")
    files[p] = rows_qgs
    expected[p] = 30
    cube_subs[os.path.join(HERE, f"at_pofd_{QGS_KEY}.sub")] = _qgs_sub
    # lambda=1 replication seeds (see the FL1 block): 10 new seeds of
    # the natural lambda=1 cell, for the lock-in RATE.
    rows_fl1 = fl1_rows()
    assert len(rows_fl1) == 10, len(rows_fl1)
    _fl1_tags = {r.split(",")[0] for r in rows_fl1}
    assert len(_fl1_tags) == 10
    assert all(t.startswith("pofdws2f_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2_")
               and t.endswith("_fresh_data") for t in _fl1_tags)
    # the five EXISTING seeds must never re-queue -- they are done
    # and four of them are load-bearing for the published figure
    for _sd in FL1_EXISTING_SEEDS:
        assert fl1_tag(_sd) not in _fl1_tags, _sd
    assert {int(t.split("_s")[-1].split("_")[0])
            for t in _fl1_tags} == set(FL1_SEEDS)
    # same row grammar as the established natural cells
    for r in rows_fl1:
        _cols = [c.strip() for c in r.split(",")]
        assert _cols[1] == "sft_kl" and _cols[2] == "1", r
        assert _cols[9] == "0.2" and _cols[14] == "0.4", r
    _fl1_sub = fl1_sub()
    _fl1_env = next(ln for ln in _fl1_sub.splitlines()
                    if ln.startswith("environment"))
    assert "KL_DIRECTION=forward" in _fl1_env
    assert "INNATE_LAMBDA=0.2" in _fl1_env      # k stays at 0.2 here
    assert f'CUDADeviceName == "{FE5_A100}"' in _fl1_sub
    _prior_fl1 = {r.split(",")[0]
                  for rows in files.values() for r in rows}
    assert not (_fl1_tags & _prior_fl1), \
        f"fl1 collision: {_fl1_tags & _prior_fl1}"
    p = os.path.join(HERE, f"configs_pofd_{FL1_KEY}.txt")
    files[p] = rows_fl1
    expected[p] = 10
    cube_subs[os.path.join(HERE, f"at_pofd_{FL1_KEY}.sub")] = _fl1_sub
    # reproducibility control + seeds 1-4 (see the FL1R block)
    rows_fl1r = fl1r_rows()
    assert len(rows_fl1r) == 7, len(rows_fl1r)
    _fl1r_tags = [r.split(",")[0] for r in rows_fl1r]
    assert len(set(_fl1r_tags)) == 7
    _reps = [t for t in _fl1r_tags if "_rep" in t]
    _news = [t for t in _fl1r_tags if "_rep" not in t]
    assert len(_reps) == 3 and len(_news) == 4
    # the replicates carry seed 0 in BOTH the tag and the queue column
    for r in rows_fl1r:
        _cols = [c.strip() for c in r.split(",")]
        if "_rep" in _cols[0]:
            assert "_s0_rep" in _cols[0], r
            assert _cols[3] == "0", r
        assert _cols[1] == "sft_kl" and _cols[2] == "1", r
        assert _cols[9] == "0.2" and _cols[14] == "0.4", r
    assert {int(t.split("_s")[-1].split("_")[0]) for t in _news} == \
        set(FL1R_NEW_SEEDS)
    # never re-queue anything already run: the 15 completed lambda=1
    # seeds, and in particular seed 0 itself (its plain tag is the
    # published cell)
    _done = {fl1_tag(s) for s in FL1_EXISTING_SEEDS} | \
        {fl1_tag(s) for s in FL1_SEEDS}
    assert not (set(_fl1r_tags) & _done), \
        f"repro wave would re-queue completed runs: " \
        f"{set(_fl1r_tags) & _done}"
    assert fl1_tag(0) not in _fl1r_tags
    _fl1r_sub = fl1r_sub()
    assert f'CUDADeviceName == "{FE5_A100}"' in _fl1r_sub
    _prior_fl1r = {r.split(",")[0]
                   for rows in files.values() for r in rows}
    assert not (set(_fl1r_tags) & _prior_fl1r), \
        f"fl1r collision: {set(_fl1r_tags) & _prior_fl1r}"
    p = os.path.join(HERE, f"configs_pofd_{FL1R_KEY}.txt")
    files[p] = rows_fl1r
    expected[p] = 7
    cube_subs[os.path.join(HERE, f"at_pofd_{FL1R_KEY}.sub")] = \
        _fl1r_sub
    # lambda=0.5 / lambda=0 at the three lambda=1-only runs (FLM)
    rows_flm = flm_rows()
    assert len(rows_flm) == 6, len(rows_flm)
    _flm_tags = [r.split(",")[0] for r in rows_flm]
    assert len(set(_flm_tags)) == 6
    for _arm, _n in (("_b0p5_", 3), ("_b0_", 3)):
        assert sum(1 for t in _flm_tags if _arm in t) == _n, _arm
    # exactly the three run identities that exist only for lambda=1
    for _arm in ("b0p5", "b0"):
        assert flm_tag(_arm, 1, None) in _flm_tags
        assert flm_tag(_arm, 0, 2) in _flm_tags
        assert flm_tag(_arm, 0, 3) in _flm_tags
    # arm surface: b0p5 = forward-KL at 0.5, b0 = plain SFT
    for r in rows_flm:
        _cols = [c.strip() for c in r.split(",")]
        if "_b0p5_" in _cols[0]:
            assert _cols[1] == "sft_kl" and _cols[2] == "0.5", r
        else:
            assert _cols[1] == "sft" and _cols[2] == "0", r
        assert _cols[9] == "0.2" and _cols[14] == "0.4", r
        # the replicates carry seed 0 in the queue column too
        if "_rep" in _cols[0]:
            assert "_s0_rep" in _cols[0] and _cols[3] == "0", r
    # never touch a completed lambda=1 run or the published cells
    assert not any("_b1_" in t for t in _flm_tags)
    _prior_flm = {r.split(",")[0]
                  for rows in files.values() for r in rows}
    assert not (set(_flm_tags) & _prior_flm), \
        f"flm collision: {set(_flm_tags) & _prior_flm}"
    p = os.path.join(HERE, f"configs_pofd_{FLM_KEY}.txt")
    files[p] = rows_flm
    expected[p] = 6
    cube_subs[os.path.join(HERE, f"at_pofd_{FLM_KEY}.sub")] = flm_sub()
    # Qwen2.5 full-anchor (k=1) Section-3 grid (see the QK1 block):
    # 24 brand-new cells -- the completed k=0.2 grid with
    # INNATE_LAMBDA 0.2 -> 1 and nothing else.
    rows_qk1 = qk1_rows()
    assert len(rows_qk1) == 24, len(rows_qk1)
    _qk1_tags = {r.split(",")[0] for r in rows_qk1}
    assert len(_qk1_tags) == 24
    assert all(t.startswith("pofdfamk1_qwen7b_") and "_w0p5_l1_" in t
               and t.endswith("_s0") for t in _qk1_tags)
    # the anchor token is _l1_, never a bare _k1_ (that grammar is
    # ICL-K: _k0_, _k8live_, _k32noai_)
    assert not any("_k1_" in t for t in _qk1_tags)
    # never the k=0.2 anchor: these must not shadow the Section-3 grid
    assert not any("_l0p2_" in t for t in _qk1_tags)
    for _arm_tok, _nw in (("_b0_", 12), ("_b1_", 12)):
        assert sum(1 for t in _qk1_tags if _arm_tok in t) == _nw, \
            _arm_tok
    for _g in QK1_GATES:
        assert sum(1 for t in _qk1_tags
                   if f"_ea{_num(_g)}_" in t) == 6, _g
    for _e in QK1_ESS:
        assert sum(1 for t in _qk1_tags
                   if f"_es{_num(_e)}_" in t) == 8, _e
    # queue surface: b0 = plain sft, b1 = forward-KL sft_kl, ICL off,
    # qwen2.5 default chat template, 30 rounds
    for r in rows_qk1:
        _cols = [c.strip() for c in r.split(",")]
        assert _cols[1] == ("sft" if "_b0_" in _cols[0] else "sft_kl"), r
        assert _cols[3] == "0" and _cols[15] == "threshold", r
        assert _cols[16] == "0", r
        assert _cols[22] == "30", r
        assert _cols[23] == "Qwen/Qwen2.5-7B-Instruct", r
        assert _cols[24] == "default", r
    _qk1_sub = qk1_sub()
    # the ONE dial that changes, and the ones that must not. Assert
    # against the ENVIRONMENT line only -- the comment block above it
    # names POP_RESET to record that it is deliberately absent.
    _qk1_env = next(ln for ln in _qk1_sub.splitlines()
                    if ln.startswith("environment"))
    assert "INNATE_LAMBDA=1 " in _qk1_env
    assert "INNATE_LAMBDA=0.2" not in _qk1_env
    assert "POP_RESET" not in _qk1_env, \
        "k=1 must change ONLY the anchor -- no population reset"
    assert "SAVE_RAW_GEN=1" in _qk1_env and "WITH_TWIN=1" in _qk1_env
    assert "KL_DIRECTION=forward" in _qk1_env
    _prior_qk1 = {r.split(",")[0]
                  for rows in files.values() for r in rows}
    assert not (_qk1_tags & _prior_qk1), \
        f"qk1 collision: {_qk1_tags & _prior_qk1}"
    p = os.path.join(HERE, f"configs_pofd_{QK1_KEY}.txt")
    files[p] = rows_qk1
    expected[p] = 24
    cube_subs[os.path.join(HERE, f"at_pofd_{QK1_KEY}.sub")] = _qk1_sub
    # es=1 full-peer column, added after the base grid was submitted:
    # its OWN key, 8 cells, disjoint from the in-flight 24.
    rows_qk1e = qk1_rows(QK1_ES1)
    assert len(rows_qk1e) == 8, len(rows_qk1e)
    _qk1e_tags = {r.split(",")[0] for r in rows_qk1e}
    assert len(_qk1e_tags) == 8
    assert all(t.startswith("pofdfamk1_qwen7b_") and "_w0p5_l1_" in t
               and t.endswith("_es1_s0") for t in _qk1e_tags)
    for _arm_tok in ("_b0_", "_b1_"):
        assert sum(1 for t in _qk1e_tags if _arm_tok in t) == 4
    for _g in QK1_GATES:
        assert sum(1 for t in _qk1e_tags
                   if f"_ea{_num(_g)}_" in t) == 2, _g
    # MUST be disjoint from the already-submitted base grid
    assert not (_qk1e_tags & _qk1_tags), \
        f"es1 column would re-queue in-flight cells: " \
        f"{_qk1e_tags & _qk1_tags}"
    _qk1e_sub = qk1_sub(QK1_ES1_KEY, QK1_ES1)
    _qk1e_env = next(ln for ln in _qk1e_sub.splitlines()
                     if ln.startswith("environment"))
    assert "INNATE_LAMBDA=1 " in _qk1e_env
    assert "POP_RESET" not in _qk1e_env
    _prior_qk1e = {r.split(",")[0]
                   for rows in files.values() for r in rows}
    assert not (_qk1e_tags & _prior_qk1e), \
        f"qk1 es1 collision: {_qk1e_tags & _prior_qk1e}"
    p = os.path.join(HERE, f"configs_pofd_{QK1_ES1_KEY}.txt")
    files[p] = rows_qk1e
    expected[p] = 8
    cube_subs[os.path.join(HERE, f"at_pofd_{QK1_ES1_KEY}.sub")] = \
        _qk1e_sub
    # ---- Qwen2.5 mechanism diagnostic, Part A frozen cells (QMECH) ----
    # ONLY the cells the audited manifest marks new. Counts come from the
    # manifest and are asserted for CONSISTENCY, never forced: if the
    # archive changes, these fire instead of queueing a different grid.
    _qmech_mf = json.load(open(QMECH_MANIFEST_PATH))
    rows_qmech = qmech_rows()
    assert _qmech_mf["n_gpu_cells"] == 24, _qmech_mf["n_gpu_cells"]
    assert _qmech_mf["n_reused"] + _qmech_mf["n_new"] == 24
    assert _qmech_mf["n_conceptual_cells"] == 32
    assert _qmech_mf["n_perfect_prediction_cells"] == 8
    assert len(rows_qmech) == _qmech_mf["n_new"], \
        f"{len(rows_qmech)} rows != manifest n_new {_qmech_mf['n_new']}"
    _qmech_tags = {r.split(",")[0] for r in rows_qmech}
    assert len(_qmech_tags) == len(rows_qmech)
    # every new cell is the FROZEN arm: the 16 SFT cells all exist
    assert all(t.startswith("pofdqmech_qwen7b_k0_ea1_w0p5_l")
               and t.endswith("_s0") for t in _qmech_tags), _qmech_tags
    # the anchor rides _l<k>_; a bare _k1_ token would collide with the
    # ICL-K grammar (where _k0_ already spells the frozen arm)
    assert not any("_k1_" in t for t in _qmech_tags)
    # the canonical frozen prediction hash must be DERIVED by the audit
    # and must match the constant this block pins
    assert _qmech_mf["canonical_frozen_pred_sha256"] == \
        QMECH_CANONICAL_PRED_SHA, \
        (f"manifest canonical frozen hash "
         f"{_qmech_mf['canonical_frozen_pred_sha256']} != pinned "
         f"{QMECH_CANONICAL_PRED_SHA}")
    # the A100 k=.2/es=0 frozen cell must be REFUSED, not reused
    _c0 = next(c for c in _qmech_mf["cells"] if c["arm"] == "k0"
               and c["innate_k"] == 0.2 and c["eps_social"] == 0.0)
    assert _c0["status"] == "new", _c0
    assert any("A100" in r["why"] for r in _c0.get("rejected_matches", [])), \
        "the k=.2/es=0 frozen cell must be superseded for HARDWARE"
    # queue surface: frozen K=D=0, no LoRA, nothing trains, numeric gate
    for r in rows_qmech:
        _cols = [c.strip() for c in r.split(",")]
        assert _cols[1] == "frozen" and _cols[2] == "0", r
        assert _cols[3] == "0", r                      # seed 0
        assert _cols[11] == "0.5", r                   # W
        assert _cols[14] == "1" and _cols[15] == "threshold", r
        assert _cols[16] in ("0.2", "1"), r            # k rides the queue
        assert _cols[17] == "0" and _cols[19] == "0", r  # ICL_K, USE_LORA
        assert _cols[23] == "30", r
        assert _cols[24] == "Qwen/Qwen2.5-7B-Instruct", r
    assert {c.split(",")[16].strip() for c in rows_qmech} == {"0.2", "1"}
    _qmech_sub = qmech_sub()
    _qmech_env = next(ln for ln in _qmech_sub.splitlines()
                      if ln.startswith("environment"))
    assert "INNATE_LAMBDA=$(lam)" in _qmech_env
    assert "AI_GATE_MODE=$(gatemode)" in _qmech_env
    assert "PEER_GATE_MODE=threshold" in _qmech_env
    assert "WITH_TWIN=1" in _qmech_env and "SAVE_RAW_GEN=1" in _qmech_env
    assert "POP_RESET" not in _qmech_env
    assert f'CUDADeviceName == "{QMECH_H100}"' in _qmech_sub
    _prior_qmech = {r.split(",")[0]
                    for rows in files.values() for r in rows}
    assert not (_qmech_tags & _prior_qmech), \
        f"qmech collision: {_qmech_tags & _prior_qmech}"
    p = os.path.join(HERE, f"configs_pofd_{QMECH_KEY}.txt")
    files[p] = rows_qmech
    expected[p] = _qmech_mf["n_new"]
    cube_subs[os.path.join(HERE, f"at_pofd_{QMECH_KEY}.sub")] = _qmech_sub
    # ---- Qwen2.5 at the Wu consensus boundary, Part C (QWU) -----------
    rows_qwu = qwu_rows()
    assert len(rows_qwu) == 4, len(rows_qwu)
    _qwu_tags = {r.split(",")[0] for r in rows_qwu}
    assert len(_qwu_tags) == 4
    # genuinely open gates are spelled as MODES in the tag, never as the
    # numeric value 1 (both gates are strict inequalities, so a
    # distance-1 pair would still be rejected under a threshold of 1)
    assert all("_eaopen_" in t and "_esopen_" in t for t in _qwu_tags)
    assert not any("_ea1_" in t or "_es1_" in t for t in _qwu_tags)
    assert all(t.startswith("pofdqwu_qwen7b_") and "_l1_" in t
               and t.endswith(f"_s0_r{QWU_ROUNDS}") for t in _qwu_tags)
    for _w in QWU_WS:
        assert sum(1 for t in _qwu_tags if f"_w{_num(_w)}_" in t) == 2, _w
    for _arm_tok in ("_b0_", "_b1_"):
        assert sum(1 for t in _qwu_tags if _arm_tok in t) == 2, _arm_tok
    for r in rows_qwu:
        _cols = [c.strip() for c in r.split(",")]
        assert _cols[1] == ("sft" if "_b0_" in _cols[0] else "sft_kl"), r
        assert _cols[3] == "0", r                       # seed 0
        assert _cols[9] == "0.2", r                     # eps, inert here
        assert _cols[11] in ("0.5", "1"), r             # W rides the queue
        assert _cols[14] == "1", r                      # k = 1
        assert _cols[15] == "0", r                      # ICL_K: no context
        # both arms TRAIN: LoRA on, fresh adapter every round
        assert _cols[17] == "1" and _cols[18] == "1", r
        assert _cols[21] == str(QWU_ROUNDS), r
        assert _cols[22] == "Qwen/Qwen2.5-7B-Instruct", r
    _qwu_sub = qwu_sub()
    _qwu_env = next(ln for ln in _qwu_sub.splitlines()
                    if ln.startswith("environment"))
    assert "AI_GATE_MODE=all_open" in _qwu_env
    assert "PEER_GATE_MODE=all_open" in _qwu_env
    assert "INNATE_LAMBDA=$(lam)" in _qwu_env
    assert "KL_DIRECTION=forward" in _qwu_env
    assert "WITH_TWIN=1" in _qwu_env and "SAVE_RAW_GEN=1" in _qwu_env
    assert "POP_RESET" not in _qwu_env
    assert f'CUDADeviceName == "{QWU_H100}"' in _qwu_sub
    _prior_qwu = {r.split(",")[0]
                  for rows in files.values() for r in rows}
    assert not (_qwu_tags & _prior_qwu), \
        f"qwu collision: {_qwu_tags & _prior_qwu}"
    p = os.path.join(HERE, f"configs_pofd_{QWU_KEY}.txt")
    files[p] = rows_qwu
    expected[p] = 4
    cube_subs[os.path.join(HERE, f"at_pofd_{QWU_KEY}.sub")] = _qwu_sub
    # the 3-round smoke for the NEW open-peer path -- a SEPARATE key, and
    # deliberately NOT part of the four-job production count
    rows_qwus = qwu_smoke_rows()
    assert len(rows_qwus) == 1, len(rows_qwus)
    _qwus_tags = {r.split(",")[0] for r in rows_qwus}
    # exactly b1, W=1, k=1, 3 rounds -- the hardest corner of the new
    # open-peer path, at the boundary itself
    assert all(t.endswith(f"_s0_r{QWU_SMOKE_ROUNDS}{QWU_SMOKE_TOKEN}")
               and "_b1_" in t and "_w1_" in t and "_l1_" in t
               and "_eaopen_" in t and "_esopen_" in t
               for t in _qwus_tags), _qwus_tags
    # the pre-fix smoke tag must never reappear: its run decoded with
    # LoRA dropout active, and the idempotent exec would no-op on it
    assert not any(t.endswith(f"_r{QWU_SMOKE_ROUNDS}smoke")
                   for t in _qwus_tags), _qwus_tags
    for r in rows_qwus:
        _cols = [c.strip() for c in r.split(",")]
        assert _cols[1] == "sft_kl" and _cols[2] == "1", r
        assert _cols[11] == "1" and _cols[14] == "1", r   # W=1, k=1
        assert _cols[21] == str(QWU_SMOKE_ROUNDS), r
    assert not (_qwus_tags & _qwu_tags), \
        f"smoke would shadow a production cell: {_qwus_tags & _qwu_tags}"
    _qwus_sub = qwu_sub(smoke=True)
    _qwus_env = next(ln for ln in _qwus_sub.splitlines()
                     if ln.startswith("environment"))
    assert "AI_GATE_MODE=all_open" in _qwus_env
    assert "PEER_GATE_MODE=all_open" in _qwus_env
    _prior_qwus = {r.split(",")[0]
                   for rows in files.values() for r in rows}
    assert not (_qwus_tags & _prior_qwus), \
        f"qwu smoke collision: {_qwus_tags & _prior_qwus}"
    p = os.path.join(HERE, f"configs_pofd_{QWU_SMOKE_KEY}.txt")
    files[p] = rows_qwus
    expected[p] = 1
    cube_subs[os.path.join(HERE, f"at_pofd_{QWU_SMOKE_KEY}.sub")] = \
        _qwus_sub
    # ---- Wu-boundary PERSONAL-HISTORY ICL arm (QWU_ICL) ---------------
    # Two cells, own key, so the four COMPLETED trained cells are never
    # re-queued -- not even as idempotent no-ops.
    rows_qwui = qwu_icl_rows()
    assert len(rows_qwui) == 2, len(rows_qwui)
    _qwui_tags = {r.split(",")[0] for r in rows_qwui}
    assert len(_qwui_tags) == 2
    assert all(t.startswith("pofdqwu_qwen7b_d8_eaopen_w") and "_l1_" in t
               and t.endswith(f"_s0_r{QWU_ROUNDS}")
               for t in _qwui_tags), _qwui_tags
    # open gates as MODES, never the numeric 1 -- same rule as the
    # trained cells
    assert all("_eaopen_" in t and "_esopen_" in t for t in _qwui_tags)
    assert not any("_ea1_" in t or "_es1_" in t for t in _qwui_tags)
    for _w in QWU_WS:
        assert sum(1 for t in _qwui_tags if f"_w{_num(_w)}_" in t) == 1, _w
    # MUST be disjoint from the completed four and from the smoke
    assert not (_qwui_tags & _qwu_tags), \
        f"ICL arm would re-queue trained cells: {_qwui_tags & _qwu_tags}"
    assert not (_qwui_tags & _qwus_tags)
    # queue surface: FROZEN, no LoRA, nothing trains, no cross-user
    # exemplars (ICL_K=0 -- personal history only)
    for r in rows_qwui:
        _cols = [c.strip() for c in r.split(",")]
        assert _cols[1] == "frozen" and _cols[2] == "0", r
        assert _cols[3] == "0", r                        # seed 0
        assert _cols[9] == "0.2", r                      # eps, inert
        assert _cols[11] in ("0.5", "1"), r              # W rides the queue
        assert _cols[14] == "1", r                       # k = 1
        assert _cols[15] == "0", r                       # ICL_K = 0
        assert _cols[17] == "0" and _cols[18] == "0", r  # no LoRA, no fresh
        assert _cols[21] == str(QWU_ROUNDS), r
        assert _cols[22] == "Qwen/Qwen2.5-7B-Instruct", r
    _qwui_sub = qwu_icl_sub()
    _qwui_env = next(ln for ln in _qwui_sub.splitlines()
                     if ln.startswith("environment"))
    assert f"ICL_DAYS={QWU_ICL_DAYS}" in _qwui_env
    assert "ICL_K=$(iclk)" in _qwui_env      # the queue pins it to 0
    assert "AI_GATE_MODE=all_open" in _qwui_env
    assert "PEER_GATE_MODE=all_open" in _qwui_env
    assert "INNATE_LAMBDA=$(lam)" in _qwui_env
    assert "WITH_TWIN=1" in _qwui_env and "SAVE_RAW_GEN=1" in _qwui_env
    assert "POP_RESET" not in _qwui_env
    assert f'CUDADeviceName == "{QWU_H100}"' in _qwui_sub
    # the trained keys must STILL render with ICL_DAYS=0
    for _s in (qwu_sub(), qwu_sub(smoke=True)):
        assert "ICL_DAYS=0 " in next(
            ln for ln in _s.splitlines() if ln.startswith("environment"))
    _prior_qwui = {r.split(",")[0]
                   for rows in files.values() for r in rows}
    assert not (_qwui_tags & _prior_qwui), \
        f"qwu icl collision: {_qwui_tags & _prior_qwui}"
    p = os.path.join(HERE, f"configs_pofd_{QWU_ICL_KEY}.txt")
    files[p] = rows_qwui
    expected[p] = 2
    cube_subs[os.path.join(HERE, f"at_pofd_{QWU_ICL_KEY}.sub")] = _qwui_sub
    # ---- forward vs reverse KL (KD) -----------------------------------
    rows_kd = kd_rows()
    assert len(rows_kd) == 10, len(rows_kd)
    _kd_tags = [r.split(",")[0] for r in rows_kd]
    assert len(set(_kd_tags)) == 10, _kd_tags
    # both directions present, in the counts the design calls for
    assert sum("_fwd" in t for t in _kd_tags) == 4, _kd_tags
    assert sum("_rev" in t for t in _kd_tags) == 6, _kd_tags
    # forward lambda=1 is REUSED from QWU b1 and must never be queued
    assert not any("_fwdlam1_" in t for t in _kd_tags), _kd_tags
    for r in rows_kd:
        _c = [c.strip() for c in r.split(",")]
        assert _c[1] == "sft_kl", r              # every cell trains with KL
        assert float(_c[2]) in (0.1, 1.0, 10.0), r   # lambda
        assert _c[3] == str(KD_SEED), r
        assert _c[10] == "0.0", r                # homophily gamma stays 0
        assert float(_c[11]) in (0.5, 1.0), r    # W = Celestine's beta
        assert _c[14] == "1", r                  # k = Celestine's gamma
        assert _c[15] in ("forward", "reverse"), r   # kldir column
        # the recorded direction must agree with the tag token
        assert (("_fwd" in _c[0]) == (_c[15] == "forward")), r
        assert _c[16] == "0" and _c[17] == "-1", r   # no ICL
        assert _c[18] == "1" and _c[19] == "1", r    # LoRA, fresh each round
        assert _c[22] == str(KD_ROUNDS), r
    _kd_sub = kd_sub()
    _kd_env = next(ln for ln in _kd_sub.splitlines()
                   if ln.startswith("environment"))
    assert "KL_DIRECTION=$(kldir)" in _kd_env, _kd_env
    assert "KL_DIRECTION=forward" not in _kd_env, _kd_env
    assert "AI_GATE_MODE=all_open" in _kd_env
    assert "PEER_GATE_MODE=all_open" in _kd_env
    assert "INNATE_LAMBDA=$(lam)" in _kd_env
    assert "WITH_TWIN=1" in _kd_env and "SAVE_RAW_GEN=1" in _kd_env
    assert "TRAIN_CAP=723" in _kd_env and "LORA_R=512" in _kd_env
    # the queue line must actually declare the new column, in the right
    # slot -- a sub whose env reads $(kldir) but never queues it would
    # silently expand to the empty string and the runner would fall back
    # to its "reverse" default on EVERY cell, forward ones included
    _kd_q = next(ln for ln in _kd_sub.splitlines() if ln.startswith("queue "))
    assert ", lam, kldir, iclk," in _kd_q, _kd_q
    assert f'CUDADeviceName == "{KD_H100}"' in _kd_sub
    _prior_kd = {r.split(",")[0] for rows in files.values() for r in rows}
    assert not (set(_kd_tags) & _prior_kd), \
        f"kd collision: {set(_kd_tags) & _prior_kd}"
    # the four REUSED tags must already exist as generated rows -- if a
    # rename ever orphans one, this fails here rather than in the
    # analyzer months later
    for _arm, _tag in KD_REUSED.items():
        assert _tag in _prior_kd, f"KD_REUSED {_arm} -> {_tag} is not generated"
    p = os.path.join(HERE, f"configs_pofd_{KD_KEY}.txt")
    files[p] = rows_kd
    expected[p] = 10
    cube_subs[os.path.join(HERE, f"at_pofd_{KD_KEY}.sub")] = _kd_sub
    # the 3-round reverse smoke: a SEPARATE key, not part of the ten
    rows_kds = kd_smoke_rows()
    assert len(rows_kds) == 1, len(rows_kds)
    _kds_tags = {r.split(",")[0] for r in rows_kds}
    assert all(t.startswith("pofdkdsmk_") for t in _kds_tags), _kds_tags
    assert all("_revlam1_" in t and "_w1_" in t and "_l1_" in t
               and t.endswith(f"_s0_r{KD_SMOKE_ROUNDS}")
               for t in _kds_tags), _kds_tags
    # a smoke must never be able to satisfy a production tag
    assert not (_kds_tags & set(_kd_tags)), _kds_tags
    assert not any(t.startswith("pofdkd_") for t in _kds_tags), _kds_tags
    _kds_sub = kd_sub(smoke=True)
    _kds_env = next(ln for ln in _kds_sub.splitlines()
                    if ln.startswith("environment"))
    assert "KL_DIRECTION=$(kldir)" in _kds_env, _kds_env
    for r in rows_kds:
        _c = [c.strip() for c in r.split(",")]
        assert _c[1] == "sft_kl" and _c[2] == "1", r
        assert _c[15] == "reverse", r
        assert _c[22] == str(KD_SMOKE_ROUNDS), r
    p = os.path.join(HERE, f"configs_pofd_{KD_SMOKE_KEY}.txt")
    files[p] = rows_kds
    expected[p] = 1
    cube_subs[os.path.join(HERE, f"at_pofd_{KD_SMOKE_KEY}.sub")] = _kds_sub
    # ---- observation-rate subsampling (QSS) ---------------------------
    rows_qss = qss_rows()
    assert len(rows_qss) == len(QSS_COUNTS) + 1 == 7, len(rows_qss)
    _qss_tags = [r.split(",")[0] for r in rows_qss]
    assert len(set(_qss_tags)) == 7
    # the 100% arm is the REUSED QWU cell and must NOT be queued
    assert QSS_FULL not in QSS_COUNTS
    assert not any(f"_n{QSS_FULL}_" in t for t in _qss_tags), _qss_tags
    assert QSS_REUSED_TAG not in _qss_tags
    for _c in QSS_COUNTS:
        assert sum(1 for t in _qss_tags if f"_n{_c}_" in t) == 1, _c
    # the compute-matched cell spells its tiling so it can never be
    # confused with the plain 72-agent arm
    _cm = qss_tag(QSS_CM_N, QSS_CM_REPEAT)
    assert _cm in _qss_tags and f"_n{QSS_CM_N}rep{QSS_CM_REPEAT}_" in _cm
    assert _cm != qss_tag(QSS_CM_N)
    # queue surface: ordinary SFT, k=1, W=1, both gates open via the sub
    for r in rows_qss:
        _c = [x.strip() for x in r.split(",")]
        assert _c[1] == "sft" and _c[2] == "0", r        # lambda = 0
        assert _c[3] == "0", r                            # seed 0
        assert _c[11] == "1", r                           # W = 1
        assert _c[14] == "1", r                           # k = 1
        assert int(_c[15]) in QSS_COUNTS + [QSS_CM_N], r  # SFT_SAMPLE_N
        assert int(_c[16]) in (0, QSS_CM_REPEAT), r       # REPEAT_TO
        assert _c[19] == "1" and _c[20] == "1", r         # LoRA, fresh
        assert _c[23] == str(QSS_ROUNDS), r
        assert _c[24] == "Qwen/Qwen2.5-7B-Instruct", r
    # exactly ONE row carries the tiling
    assert sum(1 for r in rows_qss
               if int(r.split(",")[16]) > 0) == 1
    _qss_sub = qss_sub()
    _qss_env = next(ln for ln in _qss_sub.splitlines()
                    if ln.startswith("environment"))
    assert "SFT_SAMPLE_N=$(samplen)" in _qss_env
    assert "SFT_SAMPLE_REPEAT_TO=$(repeatto)" in _qss_env
    assert "AI_GATE_MODE=all_open" in _qss_env
    assert "PEER_GATE_MODE=all_open" in _qss_env
    assert "INNATE_LAMBDA=$(lam)" in _qss_env
    assert "WITH_TWIN=1" in _qss_env
    assert "POP_RESET" not in _qss_env
    assert f'CUDADeviceName == "{QSS_H100}"' in _qss_sub
    _prior_qss = {r.split(",")[0]
                  for rows in files.values() for r in rows}
    assert not (set(_qss_tags) & _prior_qss), \
        f"qss collision: {set(_qss_tags) & _prior_qss}"
    p = os.path.join(HERE, f"configs_pofd_{QSS_KEY}.txt")
    files[p] = rows_qss
    expected[p] = 7
    cube_subs[os.path.join(HERE, f"at_pofd_{QSS_KEY}.sub")] = _qss_sub
    # 3-round smoke for the NEW sampling path
    rows_qsss = qss_smoke_rows()
    assert len(rows_qsss) == 1
    _qsss_tags = {r.split(",")[0] for r in rows_qsss}
    assert all(t.endswith(f"_s0_r{QSS_SMOKE_ROUNDS}smoke")
               for t in _qsss_tags), _qsss_tags
    assert not (_qsss_tags & set(_qss_tags))
    _prior_qsss = {r.split(",")[0]
                   for rows in files.values() for r in rows}
    assert not (_qsss_tags & _prior_qsss)
    p = os.path.join(HERE, f"configs_pofd_{QSS_SMOKE_KEY}.txt")
    files[p] = rows_qsss
    expected[p] = 1
    cube_subs[os.path.join(HERE, f"at_pofd_{QSS_SMOKE_KEY}.sub")] = \
        qss_sub(smoke=True)
    # ---- SFT training-dose scouts (SFTD): U / LR / rank -------------
    _sftd = {SFTD_UPDATE_KEY: (sftd_update_rows(), 6),
             SFTD_LR_KEY: (sftd_lr_rows(), 4),
             SFTD_RANK_KEY: (sftd_rank_rows(), 5),
             SFTD_SMOKE_KEY: (sftd_smoke_rows(), 1)}
    _all_sftd = set()
    for _key, (_rows, _n) in _sftd.items():
        assert len(_rows) == _n, (_key, len(_rows))
        _tags = {r.split(",")[0] for r in _rows}
        assert len(_tags) == _n
        # every cell is ONE round, ordinary SFT, k=1, W=1, both gates open
        for r in _rows:
            _c = [x.strip() for x in r.split(",")]
            assert _c[1] == "sft" and _c[2] == "0", r      # lambda_KL = 0
            assert _c[3] == "0", r                          # seed 0
            assert _c[11] == "1" and _c[14] == "1", r       # W = 1, k = 1
            assert _c[24] == str(SFTD_ROUNDS) == "1", r     # ONE round
            assert _c[25] == "Qwen/Qwen2.5-7B-Instruct", r
            assert int(_c[15]) > 0, r                       # SFT_MAX_STEPS
        # exactly ONE dial moves per family; the other two stay standard
        if _key == SFTD_UPDATE_KEY:
            assert {c.split(",")[15].strip() for c in _rows} == \
                {str(u) for u in SFTD_UPDATES}
            assert {c.split(",")[16].strip() for c in _rows} == {SFTD_STD_LR}
            assert {c.split(",")[17].strip() for c in _rows} == \
                {str(SFTD_STD_RANK)}
        elif _key == SFTD_LR_KEY:
            assert {c.split(",")[16].strip() for c in _rows} == set(SFTD_LRS)
            assert SFTD_STD_LR not in {c.split(",")[16].strip()
                                       for c in _rows}, "shared endpoint"
            assert {c.split(",")[15].strip() for c in _rows} == \
                {str(SFTD_STD_U)}
        elif _key == SFTD_RANK_KEY:
            assert {c.split(",")[17].strip() for c in _rows} == \
                {str(r) for r in SFTD_RANKS}
            assert str(SFTD_STD_RANK) not in {c.split(",")[17].strip()
                                              for c in _rows}
            assert {c.split(",")[16].strip() for c in _rows} == {SFTD_STD_LR}
        _sub = sftd_sub(_key)
        _env = next(ln for ln in _sub.splitlines()
                    if ln.startswith("environment"))
        assert "SFT_EPOCHS=0 SFT_MAX_STEPS=$(steps)" in _env
        assert "SFT_LR=$(lr)" in _env and "LORA_R=$(rank)" in _env
        assert "SAVE_SFT_ORDER=1" in _env
        assert "AI_GATE_MODE=all_open" in _env
        assert "PEER_GATE_MODE=all_open" in _env
        assert f'CUDADeviceName == "{SFTD_H100}"' in _sub
        _prior = {r.split(",")[0] for rows in files.values() for r in rows}
        assert not (_tags & _prior), f"sftd collision {_key}: {_tags & _prior}"
        assert not (_tags & _all_sftd), f"sftd internal collision {_key}"
        _all_sftd |= _tags
        p = os.path.join(HERE, f"configs_pofd_{_key}.txt")
        files[p] = _rows
        expected[p] = _n
        cube_subs[os.path.join(HERE, f"at_pofd_{_key}.sub")] = _sub
    # the shared full-dose endpoint is queued EXACTLY once, in the
    # update-dose key -- the LR and rank families reuse it
    _shared = sftd_tag(SFTD_STD_U, SFTD_STD_LR, SFTD_STD_RANK)
    assert sum(1 for t in _all_sftd if t == _shared) == 1
    assert _shared in {r.split(",")[0] for r in sftd_update_rows()}
    # Figure-2 family-prior scout (see the FAM block): the 48-cell
    # 6-checkpoint grid minus whatever the field-level audit reused.
    # Counts come from the manifest and are asserted for CONSISTENCY
    # (fam_rows), never forced to a particular reuse split.
    rows_fam = fam_rows()
    _fam_tags = {r.split(",")[0] for r in rows_fam}
    assert len(_fam_tags) == len(rows_fam)
    assert all(t.startswith("pofdfam_") and "_ea1_" in t
               and t.endswith("_s0") for t in _fam_tags)
    _prior_before_fam = {r.split(",")[0]
                         for rows in files.values() for r in rows}
    assert not (_fam_tags & _prior_before_fam), \
        f"fam collision: {_fam_tags & _prior_before_fam}"
    p = os.path.join(HERE, f"configs_pofd_{FAM_KEY}.txt")
    files[p] = rows_fam
    expected[p] = len(rows_fam)
    cube_subs[os.path.join(HERE, f"at_pofd_{FAM_KEY}.sub")] = \
        fam_sub("main")
    _fam_smk = fam_smoke_rows()
    assert len(_fam_smk) == 3
    assert all("_b1_ea1_" in r.split(",")[0]
               and r.split(",")[0].startswith("pofdfamsmk_")
               and "_es0p2_s0" in r.split(",")[0] for r in _fam_smk)
    assert {r.split(",")[0].split("_b1_")[0].replace("pofdfamsmk_", "")
            for r in _fam_smk} == set(FAM_SMOKE_MODELS)
    p = os.path.join(HERE, f"configs_pofd_{FAM_SMOKE_KEY}.txt")
    files[p] = _fam_smk
    expected[p] = 3
    cube_subs[os.path.join(HERE, f"at_pofd_{FAM_SMOKE_KEY}.sub")] = \
        fam_sub("smoke")
    # qwen retry: exactly the 8 uncached-death qwen7b cells; tags
    # DELIBERATELY shared with the scout key (no collision assert --
    # the retry-key precedent), never co-submit.
    _fam_rt = fam_qwen_retry_rows()
    assert len(_fam_rt) == 8, len(_fam_rt)
    assert {r.split(",")[0] for r in _fam_rt} <= _fam_tags, \
        "qwen retry cells must be a subset of the scout wave"
    p = os.path.join(HERE, f"configs_pofd_{FAM_QWEN_RETRY_KEY}.txt")
    files[p] = _fam_rt
    expected[p] = 8
    cube_subs[os.path.join(HERE,
                           f"at_pofd_{FAM_QWEN_RETRY_KEY}.sub")] = \
        fam_sub("qwen_retry")
    # beta scout: 18 brand-new seed-0 cells along the forward-KL axis
    rows_fb = fam_beta_rows()
    assert len(rows_fb) == 18, len(rows_fb)
    _fb_tags = {r.split(",")[0] for r in rows_fb}
    assert len(_fb_tags) == 18
    for _a in FAM_BETA_ARMS:
        assert sum(1 for t in _fb_tags if f"_{_a}_" in t) == 6, _a
    assert all("_ea1_" in t and "_es0p05_s0" in t for t in _fb_tags)
    _prior_before_fb = {r.split(",")[0]
                        for rows in files.values() for r in rows}
    assert not (_fb_tags & _prior_before_fb), \
        f"beta scout collision: {_fb_tags & _prior_before_fb}"
    p = os.path.join(HERE, f"configs_pofd_{FAM_BETA_KEY}.txt")
    files[p] = rows_fb
    expected[p] = 18
    cube_subs[os.path.join(HERE, f"at_pofd_{FAM_BETA_KEY}.sub")] = \
        fam_sub("beta_scout")
    # confirmation keys: 12 jobs each (6 checkpoints x seeds 42/43);
    # only the analyzer-selected beta's key is ever submitted
    for _a in FAM_BETA_ARMS:
        rows_fc = fam_confirm_rows(_a)
        assert len(rows_fc) == 12, (_a, len(rows_fc))
        _fc_tags = {r.split(",")[0] for r in rows_fc}
        assert len(_fc_tags) == 12
        assert all(f"_{_a}_" in t and "_es0p05_" in t
                   and (t.endswith("_s42") or t.endswith("_s43"))
                   for t in _fc_tags)
        _prior_now = {r.split(",")[0]
                      for rows in files.values() for r in rows}
        assert not (_fc_tags & _prior_now), \
            f"{_a} confirm collision: {_fc_tags & _prior_now}"
        p = os.path.join(HERE,
                         f"configs_pofd_{FAM_CONFIRM_KEY[_a]}.txt")
        files[p] = rows_fc
        expected[p] = 12
        cube_subs[os.path.join(HERE,
                               f"at_pofd_{FAM_CONFIRM_KEY[_a]}.sub")] = \
            fam_sub(f"{_a}_confirm")
    # Section-3 family-gate ablation (see the FAMG block): 36 new
    # seed-0 cells at ea {0.1, 0.2, 0.4}; the 12 ea=1 scout cells
    # reuse and must NEVER re-queue (their tags stay with
    # fig2_family_prior_scout).
    rows_fg = famg_rows()
    assert len(rows_fg) == 36, len(rows_fg)
    _fg_tags = {r.split(",")[0] for r in rows_fg}
    assert len(_fg_tags) == 36
    assert all(t.startswith("pofdfam_") and "_es0p05_" in t
               and t.endswith("_s0") for t in _fg_tags)
    assert not any("_ea1_" in t for t in _fg_tags), \
        "the ea=1 cells reuse -- they never queue here"
    for _arm_tok, _nw in (("_b0_", 18), ("_b1_", 18)):
        assert sum(1 for t in _fg_tags if _arm_tok in t) == _nw, \
            _arm_tok
    for _g in (0.1, 0.2, 0.4):
        assert sum(1 for t in _fg_tags
                   if f"_ea{_num(_g)}_" in t) == 12, _g
    for _m in FAM_MODELS:
        assert sum(1 for t in _fg_tags
                   if f"_{_m}_" in t) == 6, _m
    # qwen3 rides thinking OFF; every other checkpoint the default
    for r in rows_fg:
        _cols = [c.strip() for c in r.split(",")]
        assert _cols[24] == ("0" if "_qwen3_8b_" in _cols[0]
                             else "default"), r
    _prior_fg = {r.split(",")[0]
                 for rows in files.values() for r in rows}
    assert not (_fg_tags & _prior_fg), \
        f"famg collision: {_fg_tags & _prior_fg}"
    p = os.path.join(HERE, f"configs_pofd_{FAMG_KEY}.txt")
    files[p] = rows_fg
    expected[p] = 36
    cube_subs[os.path.join(HERE, f"at_pofd_{FAMG_KEY}.sub")] = \
        famg_sub()
    # social-gate extension (see the FAMGS block): 84 new seed-0
    # cells (48 es0 + 36 es0p2 below ea1); the es0p05 surface stays
    # with fam_gate_ablation and the ea1 cells reuse -- their tags
    # never queue here.
    rows_fgs = famgs_rows()
    assert len(rows_fgs) == 84, len(rows_fgs)
    _fgs_tags = {r.split(",")[0] for r in rows_fgs}
    assert len(_fgs_tags) == 84
    assert all(t.startswith("pofdfam_") and t.endswith("_s0")
               for t in _fgs_tags)
    assert not any("_es0p05_" in t for t in _fgs_tags)
    assert sum(1 for t in _fgs_tags if "_es0_" in t) == 48
    assert sum(1 for t in _fgs_tags if "_es0p2_" in t) == 36
    assert not any("_ea1_" in t and "_es0p2_" in t
                   for t in _fgs_tags), \
        "the es0p2 ea1 scout cells reuse -- they never queue"
    for _arm_tok, _nw in (("_b0_", 42), ("_b1_", 42)):
        assert sum(1 for t in _fgs_tags if _arm_tok in t) == _nw, \
            _arm_tok
    for _m in FAM_MODELS:
        assert sum(1 for t in _fgs_tags
                   if f"_{_m}_" in t) == 14, _m
    for r in rows_fgs:
        _cols = [c.strip() for c in r.split(",")]
        assert _cols[24] == ("0" if "_qwen3_8b_" in _cols[0]
                             else "default"), r
    _prior_fgs = {r.split(",")[0]
                  for rows in files.values() for r in rows}
    assert not (_fgs_tags & _prior_fgs), \
        f"famgs collision: {_fgs_tags & _prior_fgs}"
    p = os.path.join(HERE, f"configs_pofd_{FAMGS_KEY}.txt")
    files[p] = rows_fgs
    expected[p] = 84
    cube_subs[os.path.join(HERE, f"at_pofd_{FAMGS_KEY}.sub")] = \
        famgs_sub()
    # zero-shot prior screen (see the ZSPRIOR block): exactly 4 one-round
    # probes, one per candidate checkpoint, NEW pofdzsprior_ family.
    rows_zsp = zsprior_rows()
    assert len(rows_zsp) == ZSPRIOR_EXPECT_NEW == 4, len(rows_zsp)
    _zsp_tags = {r.split(",")[0] for r in rows_zsp}
    assert len(_zsp_tags) == 4
    assert all(t.endswith("_es0_s0") for t in _zsp_tags)
    _qwen3_row = next(r for r in rows_zsp if "_qwen3_8b_" in r)
    assert _qwen3_row.rstrip().endswith(", 0"), \
        "Qwen3 must run CHAT_THINKING=0 (thinking explicitly disabled)"
    assert all(r.rstrip().endswith(", default")
               for r in rows_zsp if "_qwen3_8b_" not in r)
    _prior_before_zsp = {r.split(",")[0]
                        for rows in files.values() for r in rows}
    assert not (_zsp_tags & _prior_before_zsp), \
        f"zsprior collision: {_zsp_tags & _prior_before_zsp}"
    p = os.path.join(HERE, f"configs_pofd_{ZSPRIOR_KEY}.txt")
    files[p] = rows_zsp
    expected[p] = ZSPRIOR_EXPECT_NEW
    cube_subs[os.path.join(HERE, f"at_pofd_{ZSPRIOR_KEY}.sub")] = \
        zsprior_sub()
    # ---- Friedkin-Johnsen robustness wave (FJR) ---------------------
    _fjr = {FJR_KEY: (fjr_rows(), 12),
            FJR_SMOKE_KEY: (fjr_smoke_rows(), 1),
            FJR_FROZEN_KEY: (fjr_frozen_rows(), 1),
            # OPTIONAL boundary wave, not part of the core: the same 12
            # trained cells at beta=1, where 1-beta=0 removes the innate
            # component from x_init entirely. Same alpha=.9 and K=100.
            # Generated so it is ready, never bundled into any umbrella.
            FJR_BETA1_KEY: (fjr_rows(beta=FJR_BETA_BOUNDARY), 12),
            FJR_KL_KEY: (fjr_kl_rows(), 12),
            FJR_SEED_KEY: (fjr_seed_rows(), 36),
            FJR_BETA1_SEED_KEY: (fjr_seed_rows(beta=FJR_BETA_BOUNDARY), 36)}
    _all_fjr = set()
    for _key, (_rows, _n) in _fjr.items():
        assert len(_rows) == _n, (_key, len(_rows), _n)
        _tags = {r.split(",")[0] for r in _rows}
        assert len(_tags) == _n, f"duplicate tags in {_key}"
        # no gate tokens: FJ applies neither an AI nor a bounded-confidence
        # gate, so a tag carrying _ea/_es would name something that never ran
        for _t in _tags:
            assert "_ea" not in _t and "_es" not in _t, _t
        if _key != FJR_FROZEN_KEY:
            for r in _rows:
                _c = [x.strip() for x in r.split(",")]
                assert _c[8] == "fj", r                 # POP_MODEL
                assert _c[3] in (["0"] if _key not in
                                 (FJR_SEED_KEY, FJR_BETA1_SEED_KEY)
                                 else [str(x) for x in FJR_SEEDS]), r
                assert _c[5] == "replace", r            # replace-only data
                assert _c[14] == f"{FJR_ALPHA:g}", r    # peer susceptibility
                assert _c[15] == str(FJR_INNER), r      # K inner steps
                assert _c[1] in ("sft", "sft_kl"), r
            # arm is read from the tag's _<arm>_beta token, NOT by
            # splitting on "_": model slugs contain underscores
            # (qwen3_8b, olmo3_7b) so positional splitting picks "8b"
            _all_arms = FJR_ARMS + FJR_KL_ARMS
            def _arm_of(tag):
                hits = [a for a in _all_arms if f"_{a}_beta" in tag]
                assert len(hits) == 1, (tag, hits)
                return hits[0]
            if _key in (FJR_KEY, FJR_BETA1_KEY, FJR_KL_KEY,
                        FJR_SEED_KEY, FJR_BETA1_SEED_KEY):
                _models = {t.split("_")[1] + ("_" + t.split("_")[2]
                                              if t.split("_")[2] not in _all_arms
                                              else "")
                           for t in _tags}
                assert len(_models) == 6, _models
                _lo = FJR_KL_ARMS[0] if _key == FJR_KL_KEY else "b0"
                assert sum(1 for t in _tags if _arm_of(t) == _lo) == _n // 2
            # b0 is ordinary SFT at KL weight 0; b1 is forward KL at 1
            for r in _rows:
                _c = [x.strip() for x in r.split(",")]
                assert (_arm_of(_c[0]), _c[1], _c[2]) in (
                    ("b0", "sft", "0"), ("b1", "sft_kl", "1"),
                    ("b2", "sft_kl", "2"), ("b8", "sft_kl", "8")), r
            _sub = fjr_sub(_key, _rows,
                           smoke=(_key == FJR_SMOKE_KEY))
            _env = next(ln for ln in _sub.splitlines()
                        if ln.startswith("environment"))
            assert "POP_MODEL=fj" in _env and "FJ_UPDATE_VERSION=wu1" in _env
            assert "FJ_ALPHA=$(alpha)" in _env
            assert "FJ_INNER_STEPS=$(inner)" in _env
            assert "KL_DIRECTION=forward" in _env
            assert "KL_REF_ADAPTER" not in _env      # no reference adapter
            assert "AI_GATE_MODE" not in _env and "PEER_GATE_MODE" not in _env
            assert "LORA_R=512" in _env and "SFT_EPOCHS=1" in _env
            assert "SFT_BATCH_SIZE=4" in _env and "SFT_LR=5e-5" in _env
            assert "N_LABELED=723" in _env and "TRAIN_CAP=723" in _env
        else:
            _sub = fjr_frozen_sub(_key, _rows)
            assert "FJ_" not in _sub.split("environment")[1].split("\n")[0]
        assert f'CUDADeviceName == "{FJR_H100}"' in _sub
        _prior = {r.split(",")[0] for rows in files.values() for r in rows}
        assert not (_tags & _prior), f"fjr collision {_key}: {_tags & _prior}"
        assert not (_tags & _all_fjr), f"fjr internal collision {_key}"
        _all_fjr |= _tags
        p = os.path.join(HERE, f"configs_pofd_{_key}.txt")
        files[p] = _rows
        expected[p] = _n
        cube_subs[os.path.join(HERE, f"at_pofd_{_key}.sub")] = _sub
    # the beta=1 key must be DISJOINT from the core wave: same models and
    # arms, different beta, so the tags must differ or submitting both
    # would double-queue one set of run dirs
    assert not ({r.split(",")[0] for r in fjr_rows()}
                & {r.split(",")[0] for r in fjr_rows(beta=FJR_BETA_BOUNDARY)})
    # the corrected configuration must not leave any row from the earlier
    # unsubmitted alpha=.5 / K=1 draft behind under a primary key
    # seed replicates must be disjoint from the seed-0 waves
    assert not ({r.split(",")[0] for r in fjr_seed_rows()}
                & {r.split(",")[0] for r in fjr_rows()})
    assert not ({r.split(",")[0] for r in fjr_seed_rows(beta=FJR_BETA_BOUNDARY)}
                & {r.split(",")[0] for r in fjr_rows(beta=FJR_BETA_BOUNDARY)})
    for _k in (FJR_KEY, FJR_SMOKE_KEY, FJR_BETA1_KEY, FJR_KL_KEY,
               FJR_SEED_KEY, FJR_BETA1_SEED_KEY):
        for _r in files[os.path.join(HERE, f"configs_pofd_{_k}.txt")]:
            assert "alpha0p5" not in _r and "_in1_" not in _r, (_k, _r)

    # ---- Jiduan Wu / Pokec replication (WU) -------------------------
    # EVERY count below is the reuse arithmetic made explicit, and every
    # one is asserted rather than described:
    #   smoke            1   Qwen2.5 b1, 3 rounds
    #   controls         0   CPU ONLY -- no Condor job, no config, no sub
    #   prior           12   6 models x {b0,b1}, seed 0, exact hetero
    #   prior_seeds      8   seeds {42,43} x {qwen7b,olmo7b} x {b0,b1}
    #   lambda_ladder    3   5 conceptual - 2 reused (lambda 0 and 1 ARE
    #                        the prior wave's b0/b1 Qwen2.5 cells)
    #   icl              6   10 conceptual - 4 reused (2 models x b0/b1)
    #   environment     10   13 conceptual - 3 reused; 13 rather than 15
    #                        because c_beta=0 collapses 3 arms into 1 cell
    #   routing_smoke   12   3 arms x 2 c_alpha x 2 sides, seed 0
    #   routing_seeds   24   the same 12 at seeds 42 and 43
    #   frozen           6   one extraction per model; NOTHING reused
    _wu = {WU_SMOKE_KEY: (wu_smoke_rows(), 1, WU_SMOKE_ROUNDS),
           WU_PRIOR_KEY: (wu_prior_rows(), 12, WU_ROUNDS),
           WU_PRIOR_SEEDS_KEY: (wu_prior_seed_rows(), 8, WU_ROUNDS),
           WU_LADDER_KEY: (wu_ladder_rows(), 3, WU_ROUNDS),
           WU_ICL_KEY: (wu_icl_rows(), 8, WU_ROUNDS),
           WU_ENV_KEY: (wu_env_rows(), 10, WU_ROUNDS),
           WU_ROUTE_SMOKE_KEY: (wu_route_rows(), 16, WU_ROUNDS),
           WU_ROUTE_SEEDS_KEY: (wu_route_rows(WU_SEEDS), 32, WU_ROUNDS)}
    # THE CONTROLS KEY QUEUES NOTHING. Perfect prediction, no-platform
    # (c_beta=0) and the frozen replay are LINEAR maps of vectors that
    # already exist -- they depend on no language model, so a GPU job
    # would only re-derive arithmetic. It is kept as a NAMED key because
    # it is part of the conceptual design and someone will type it: the
    # submit script routes it to an explicit refusal that points at the
    # CPU path, which is strictly better than a missing-file error.
    assert WU_CONTROLS_KEY not in _wu
    for _suffix in ("configs_pofd_", "at_pofd_"):
        assert not os.path.exists(
            os.path.join(HERE, f"{_suffix}{WU_CONTROLS_KEY}"
                               f"{'.txt' if 'configs' in _suffix else '.sub'}")), \
            f"{WU_CONTROLS_KEY} must have NO generated file: it queues 0 jobs"
    _all_wu = set()
    for _key, (_rows, _n, _rounds) in _wu.items():
        assert len(_rows) == _n, (_key, len(_rows), _n)
        _tags = {r.split(",")[0] for r in _rows}
        assert len(_tags) == _n, f"duplicate tags in {_key}"
        for _t in _tags:
            # FJ has neither an AI gate nor a bounded-confidence gate, so
            # a tag carrying _ea/_es would name something that never ran
            assert "_ea" not in _t and "_es" not in _t, _t
            assert _t.startswith("pofdwu_"), _t
            assert f"_in{WU_INNER}_" in _t, _t
            assert _t.endswith(f"_r{_rounds}"
                               + ("smoke" if _key == WU_SMOKE_KEY else "")), _t
        for r in _rows:
            _c = [x.strip() for x in r.split(",")]
            assert _c[8] == "fj", r                    # POP_MODEL
            assert _c[5] == "replace", r               # replace-only data
            # W_PLAT is pinned: the platform dose lives in FJ_BETA_SCALE
            # alone, so it can never be applied twice
            assert _c[11] == "1.0", r
            assert _c[14] == "dataset" and _c[15] == "dataset", r
            assert _c[18] == str(WU_INNER), r          # K_FJ
            assert int(_c[29]) == _rounds, r           # outer horizon
            assert _c[1] in ("sft", "sft_kl", "frozen"), r
            # the arm token and the queue payload must agree, or a tag
            # would name a channel the run did not use
            _hits = [a for a in WU_ARM_COLS if f"_{a}_pa" in _c[0]]
            assert len(_hits) == 1, (_c[0], _hits)
            _a = WU_ARM_COLS[_hits[0]]
            assert (_c[1], _c[2], _c[19], _c[20], _c[21]) == (
                _a["style"], _a["beta"], _a["iclmode"], str(_a["iclk"]),
                str(_a["icld"])), r
            assert (int(_c[25]), int(_c[26])) == (_a["uselora"],
                                                  _a["fresh"]), r
            # the scale in the tag must be the scale on the queue
            assert f"_pad{_num(float(_c[16]))}_" in _c[0], r
            assert f"_pbd{_num(float(_c[17]))}_" in _c[0], r
            # the seed rides BOTH the tag and the queue column; the tag's
            # last two tokens are always s<seed> and r<rounds>, so this
            # needs no regex and no positional model split
            _stok = _c[0].split("_")[-2]
            assert _stok[0] == "s" and int(_c[3]) == int(_stok[1:]), r
            _is_route = "_rtT_" in _c[0] or "_rtC_" in _c[0]
            if _is_route:
                assert float(_c[22]) == (WU_ROUTE_FRAC if "_rtT_" in _c[0]
                                         else WU_ROUTE_CONTROL_FRAC), r
                assert int(_c[23]) == WU_ROUTE_SEED, r
                # the runner REFUSES an injected value outside [0, 1], so
                # there is no out-of-range "no-op" sentinel to reach for
                assert 0.0 <= float(_c[24]) <= 1.0, r
                assert float(_c[24]) == WU_ROUTE_VALUE, r
            else:
                assert (float(_c[22]), int(_c[23]), float(_c[24])) == \
                    (0.0, 0, 0.0), r
        _sub = wu_sub(_key, _rows, rounds=_rounds)
        _env = next(ln for ln in _sub.splitlines()
                    if ln.startswith("environment"))
        assert "DATASET=pokec" in _env and "ML_TARGET" not in _env
        assert "POP_MODEL=fj" in _env and "FJ_UPDATE_VERSION=wu1" in _env
        assert "FJ_PEER_SOURCE=$(peersrc)" in _env
        assert "FJ_PLATFORM_SOURCE=$(platsrc)" in _env
        assert "FJ_ALPHA_SCALE=$(ascale)" in _env
        assert "FJ_BETA_SCALE=$(bscale)" in _env
        assert "FJ_INNER_STEPS=$(inner)" in _env
        assert "FJ_OBSERVED_PASSTHROUGH=1" in _env
        assert "WU_ICL_MODE=$(iclmode)" in _env
        assert "ROUTING_TREAT_FRAC=$(rtfrac)" in _env
        assert "KL_DIRECTION=forward" in _env
        # NO SCALAR alpha/beta anywhere: a scalar next to a dataset
        # source is the ambiguity this family exists to remove
        assert "FJ_ALPHA=" not in _env and "FJ_BETA=" not in _env
        assert "FJ_PEER_ALPHA" not in _env
        # train on the OBSERVED set only -- a held-out opinion can never
        # reach the optimizer if the batch is drawn from a 1730 prefix
        assert f"N_LABELED={WU_N_OBSERVED}" in _env
        assert f"TRAIN_CAP={WU_N_OBSERVED}" in _env
        assert "AI_GATE_MODE" not in _env and "PEER_GATE_MODE" not in _env
        assert "KL_REF_ADAPTER" not in _env
        assert f'CUDADeviceName == "{WU_H100}"' in _sub
        _prior_tags = {r.split(",")[0] for rows in files.values() for r in rows}
        assert not (_tags & _prior_tags), f"wu collision {_key}"
        assert not (_tags & _all_wu), f"wu internal collision {_key}"
        _all_wu |= _tags
        p = os.path.join(HERE, f"configs_pofd_{_key}.txt")
        files[p] = _rows
        expected[p] = _n
        cube_subs[os.path.join(HERE, f"at_pofd_{_key}.sub")] = _sub
    # frozen extraction: its own row schema and sub, no FJ parameters
    _wuzs = wu_frozen_rows()
    assert len(_wuzs) == 6 and len(WU_FROZEN_MODELS) == len(WU_MODELS)
    _wuzs_tags = {r.split(",")[0] for r in _wuzs}
    assert len(_wuzs_tags) == 6
    for _t in _wuzs_tags:
        # an extraction applies no FJ parameter, so it must name none
        for _tok in ("_pa", "_pb", "_in", "_rt"):
            assert _tok not in _t, (_tok, _t)
    _wuzs_sub = wu_frozen_sub(WU_FROZEN_KEY, _wuzs)
    _wuzs_env = next(ln for ln in _wuzs_sub.splitlines()
                     if ln.startswith("environment"))
    assert "FJ_" not in _wuzs_env and "POP_MODEL" not in _wuzs_env
    assert "DATASET=pokec" in _wuzs_env
    assert "EPS_AI=$(eps_ai)" in _wuzs_env      # 0 under the strict gate
    assert f'CUDADeviceName == "{WU_H100}"' in _wuzs_sub
    assert not (_wuzs_tags & {r.split(",")[0]
                              for rows in files.values() for r in rows})
    files[os.path.join(HERE, f"configs_pofd_{WU_FROZEN_KEY}.txt")] = _wuzs
    expected[os.path.join(HERE, f"configs_pofd_{WU_FROZEN_KEY}.txt")] = 6
    cube_subs[os.path.join(HERE, f"at_pofd_{WU_FROZEN_KEY}.sub")] = _wuzs_sub

    # ---- the reuse arithmetic, asserted rather than described --------
    _wu_prior_tags = {r.split(",")[0] for r in wu_prior_rows()}
    # LADDER: 5 conceptual, 2 reused, 3 queued. The reused tags must be
    # BYTE-IDENTICAL to the prior key's rows -- if they were merely
    # similar, the ladder would be missing its two anchors and nobody
    # would notice until the analyzer scored a 3-point ladder.
    _lad_reuse = set(wu_ladder_reused())
    assert len(_lad_reuse) == 2, _lad_reuse
    assert _lad_reuse <= _wu_prior_tags, _lad_reuse - _wu_prior_tags
    assert not (_lad_reuse & {r.split(",")[0] for r in wu_ladder_rows()})
    assert len(WU_LADDER) == len(_lad_reuse) + len(wu_ladder_rows()) == 5
    # ICL: 12 conceptual (2 models x 6 arms), 4 reused (both models'
    # b0/b1 from the prior key), 8 queued. Six arms because BOTH history
    # mechanisms are carried: phist8 (strict Wu, the platform's own past
    # predictions) and ehist8 (the Section 4 personal-history extension,
    # realized post-peer opinions). They are never pooled.
    _icl_reuse = set(wu_icl_reused())
    assert len(_icl_reuse) == 4, _icl_reuse
    assert _icl_reuse <= _wu_prior_tags, _icl_reuse - _wu_prior_tags
    assert not (_icl_reuse & {r.split(",")[0] for r in wu_icl_rows()})
    assert (len(WU_ICL_MODELS) * len(WU_ICL_ARMS)
            == len(_icl_reuse) + len(wu_icl_rows()) == 12)
    # both mechanisms are present and distinguishable in the queued set
    _icl_tags = {r.split(",")[0] for r in wu_icl_rows()}
    assert sum(1 for t in _icl_tags if "_phist8_" in t) == 2, _icl_tags
    assert sum(1 for t in _icl_tags if "_ehist8_" in t) == 2, _icl_tags
    # ENVIRONMENT: the two collapses, each asserted on the cell list
    # itself rather than on a count someone wrote down.
    _pairs = wu_env_pairs()
    assert len(_pairs) == 5, _pairs                    # centre shared once
    assert _pairs.count((1.0, 1.0)) == 1
    _cells = wu_env_cells()
    assert len(_cells) == 13, len(_cells)              # 15 - 2 (c_beta=0)
    assert sum(1 for a, ca, cb in _cells if cb == 0.0) == 1
    assert [a for a, ca, cb in _cells if cb == 0.0] == ["b0"]
    _env_reuse = set(wu_env_reused())
    assert len(_env_reuse) == 3, _env_reuse
    # two of the three centre cells come from the prior key, the third
    # (phist8) from the ICL key -- named, not assumed
    assert len(_env_reuse & _wu_prior_tags) == 2
    assert len(_env_reuse & {r.split(",")[0] for r in wu_icl_rows()}) == 1
    assert len(_cells) - len(_env_reuse) == len(wu_env_rows()) == 10
    # ROUTING: paired twins, nothing reused, and the pair differs ONLY in
    # the routed value and the side token.
    for _seeds, _n in (((0,), 16), (tuple(WU_SEEDS), 32)):
        _rr = wu_route_rows(_seeds)
        assert len(_rr) == _n and len({r.split(",")[0] for r in _rr}) == _n
        _byside = {}
        for r in _rr:
            _c = [x.strip() for x in r.split(",")]
            _byside.setdefault(_c[0].replace("_rtT_", "_rt_")
                                    .replace("_rtC_", "_rt_"), []).append(_c)
        assert len(_byside) == _n // 2
        for _k2, _pair in _byside.items():
            assert len(_pair) == 2, _k2
            _t, _cn = sorted(_pair, key=lambda c: float(c[22]))[::-1]
            # the tags differ ONLY in the side token, and every queue
            # column except the routed FRACTION is identical -- that is
            # what makes the two runs a twin pair rather than two runs.
            # Same cohort seed and same injected value on both sides, so
            # the cohort is recomputable from either row.
            assert _t[0].replace("_rtT_", "") == _cn[0].replace("_rtC_", "")
            assert _t[1:22] == _cn[1:22], _k2     # everything up to rtfrac
            assert _t[23:] == _cn[23:], _k2       # ... and everything after
            assert float(_t[22]) == WU_ROUTE_FRAC
            assert float(_cn[22]) == WU_ROUTE_CONTROL_FRAC
    assert not ({r.split(",")[0] for r in wu_route_rows()} & _wu_prior_tags)
    # the seeded routing key is disjoint from the seed-0 one
    assert not ({r.split(",")[0] for r in wu_route_rows(WU_SEEDS)}
                & {r.split(",")[0] for r in wu_route_rows()})
    # the seed replicates change NOTHING but the seed
    def _wu_stem(tag):
        """everything before the trailing _s<seed>_r<rounds> pair"""
        return "_".join(tag.split("_")[:-2])
    _s0 = {}
    for r in wu_prior_rows():
        _c = [x.strip() for x in r.split(",")]
        _s0[_wu_stem(_c[0])] = _c[1:3] + _c[14:29]
    for r in wu_prior_seed_rows():
        _c = [x.strip() for x in r.split(",")]
        assert _c[1:3] + _c[14:29] == _s0[_wu_stem(_c[0])], r
    # every WU key's job count, printed with the arithmetic so the report
    # and the files cannot disagree
    # 96 = 1 smoke + 12 prior + 8 prior_seeds + 3 lambda + 8 icl
    #      + 10 environment + 16 routing_smoke + 32 routing_seeds
    #      + 6 frozen extractions
    assert sum(_n for _, _n, _ in _wu.values()) + len(_wuzs) == 96

    # ---- adapter KL / soft-decode probe (AKL) -----------------------
    # One job per key, no grid. The checks that matter here are not about
    # a grid at all: the executable must be the probe's own (there is no
    # trajectory to be idempotent about), and the probe must be pointed
    # at the dose configs rather than at a re-derived tag list.
    for _key, _smoke in ((AKL_KEY, False), (AKL_SMOKE_KEY, True)):
        _rows = akl_rows(smoke=_smoke)
        assert _rows == (["smoke"] if _smoke else ["full"]), _rows
        _sub = akl_sub(_key, smoke=_smoke)
        # pin the EXECUTABLE line, not a substring of the whole file: the
        # header comment legitimately names run_one_pokec_gated to say
        # this probe is not it
        _exe = next(ln for ln in _sub.splitlines()
                    if ln.startswith("executable"))
        assert _exe.endswith("run_one_adapter_kl_probe.sh"), _exe
        assert f'CUDADeviceName == "{AKL_H100}"' in _sub
        assert "queue mode from" in _sub
        _prior = {r.split(",")[0] for rows in files.values() for r in rows}
        assert not (set(_rows) & _prior), f"akl collision {_key}"
        p = os.path.join(HERE, f"configs_pofd_{_key}.txt")
        files[p] = _rows
        expected[p] = 1
        cube_subs[os.path.join(HERE, f"at_pofd_{_key}.sub")] = _sub
    # the probe reads its adapter list from the dose configs, so those
    # files must exist in this same generated set -- otherwise the probe
    # would fail at runtime on the cluster instead of here
    for _dc in ("qwen_sft_update_dose", "qwen_sft_lr_dose",
                "qwen_sft_rank_dose"):
        assert os.path.join(HERE, f"configs_pofd_{_dc}.txt") in files, _dc

    # ---- reference replay at the Wu boundary (RR) --------------------
    # Full-size training set every round; only the fraction q of rows
    # that carry the LIVE population value changes. Compute and data
    # volume are held fixed, so the arms differ purely in feedback.
    rows_rr = rr_rows()
    assert len(rows_rr) == len(RR_QS) == 4, len(rows_rr)
    _rr_tags = [r.split(",")[0] for r in rows_rr]
    assert len(set(_rr_tags)) == 4
    # THE n ARITHMETIC, recomputed rather than trusted. round() is
    # banker's rounding in Python, so round(.5*723) landing on 362 is
    # luck; rr_n spells floor(q*723 + .5) and must reproduce the pinned
    # counts exactly.
    assert RR_N_AGENTS == 723
    assert sorted(RR_N) == [0.10, 0.20, 0.50, 0.75, 1.0], sorted(RR_N)
    assert [RR_N[q] for q in sorted(RR_N)] == [72, 145, 362, 542, 723]
    for _q, _n in RR_N.items():
        assert rr_n(_q) == _n, (_q, rr_n(_q), _n)
    assert RR_N[RR_Q_FULL] == RR_N_AGENTS      # q=1 keeps every row live
    assert all(0 < RR_N[q] < RR_N_AGENTS for q in RR_QS)
    # q = 1 IS ORDINARY SFT and is therefore NOT queued: it is the
    # COMPLETED Wu-boundary b0 cell, named here so the reuse is explicit
    # and checkable rather than implied by its absence.
    assert RR_Q_FULL not in RR_QS
    assert RR_REUSED_Q1_TAG == "pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100"
    assert RR_REUSED_Q1_TAG == qwu_tag(RR_ARM, RR_W), \
        "the reused q=1 cell must be the generator's own QWU b0 W=1 tag"
    assert RR_REUSED_Q1_TAG == QSS_REUSED_TAG, \
        "the subsample wave reuses the SAME completed cell for its 100% arm"
    assert RR_REUSED_Q1_TAG not in _rr_tags
    assert not any(f"_{rr_q_tok(RR_Q_FULL)}_" in t for t in _rr_tags), \
        f"q=1 must never be queued: {_rr_tags}"
    # the frozen vector b is REUSED too -- one archived run supplies it
    _rr_prior = {r.split(",")[0] for rows in files.values() for r in rows}
    assert RR_REF_RUN in _rr_prior, \
        f"REF_REPLAY_REF_RUN {RR_REF_RUN} is not a generated run"
    assert RR_REF_SHA == QMECH_CANONICAL_PRED_SHA == \
        _qmech_mf["canonical_frozen_pred_sha256"], RR_REF_SHA
    # ... and its identical-hash twin is the AUDITED archive fallback
    _rr_twin = next(c for c in _qmech_mf["cells"]
                    if c.get("run_tag") == RR_REF_RUN_TWIN)
    assert _rr_twin["pred_sha256"] == RR_REF_SHA, _rr_twin
    assert _rr_twin["verdict"] == "PASS" and _rr_twin["arm"] == "k0", _rr_twin
    assert _rr_twin["gpu_name"] == RR_H100, _rr_twin
    # tag grammar: new pofdrr_ family, open gates as MODES, both seeds
    # spelled with their own token, horizon in the tag
    for t in _rr_tags:
        assert t.startswith(f"pofdrr_{RR_MODEL}_q"), t
        assert "_eaopen_" in t and "_esopen_" in t, t
        assert "_ea1_" not in t and "_es1_" not in t, t
        assert "_w1_" in t and "_l1_" in t, t
        assert f"_ss{RR_SEL_SEED}_" in t, t
        assert t.endswith(f"_s{RR_SEED}_r{RR_ROUNDS}"), t
        assert "smoke" not in t, t
    assert {t.split("_")[2] for t in _rr_tags} == \
        {rr_q_tok(q) for q in RR_QS} == {"q0p1", "q0p2", "q0p5", "q0p75"}
    for _q in RR_QS:
        assert sum(1 for t in _rr_tags if f"_{rr_q_tok(_q)}_" in t) == 1, _q
    # queue surface: the QWU b0 cell exactly -- ordinary SFT, W=1, k=1,
    # LoRA on and fresh every round, 100 rounds, seed 0, Qwen2.5-7B
    for r in rows_rr:
        _c = [x.strip() for x in r.split(",")]
        assert _c[1] == "sft" and _c[2] == "0", r     # lambda_KL = 0
        assert _c[3] == "0", r                         # run seed 0
        assert _c[9] == "0.2", r                       # eps, inert here
        assert _c[10] == "0.0", r                      # gamma = 0
        assert _c[11] == "1", r                        # W_PLAT = 1
        assert _c[14] == "1", r                        # INNATE_LAMBDA = 1
        assert float(_c[15]) in RR_QS, r               # REF_REPLAY_Q
        assert _c[16] == "0", r                        # ICL_K: no context
        assert _c[18] == "1" and _c[19] == "1", r      # LoRA, fresh
        assert _c[22] == str(RR_ROUNDS) == "100", r
        assert _c[23] == "Qwen/Qwen2.5-7B-Instruct", r
    assert {float(r.split(",")[15]) for r in rows_rr} == set(RR_QS)
    _rr_sub = rr_sub()
    _rr_env = next(ln for ln in _rr_sub.splitlines()
                   if ln.startswith("environment"))
    # the three new knobs, exactly as the runner names them
    assert "REF_REPLAY_Q=$(refq) " in _rr_env
    assert f"REF_REPLAY_SEED={RR_SEL_SEED} " in _rr_env
    assert f"REF_REPLAY_REF_RUN={RR_REF_RUN} " in _rr_env
    # the surface, pinned in the env rather than riding the queue
    assert "AI_GATE_MODE=all_open" in _rr_env
    assert "PEER_GATE_MODE=all_open" in _rr_env
    assert "INNATE_LAMBDA=$(lam)" in _rr_env
    assert "LORA_R=512" in _rr_env and "SFT_LR=5e-5" in _rr_env
    assert "SFT_EPOCHS=1" in _rr_env and "SFT_BATCH_SIZE=4" in _rr_env
    assert "TRAIN_CAP=723" in _rr_env and "N_LABELED=723" in _rr_env
    assert "KL_DIRECTION=forward" in _rr_env
    assert "WITH_TWIN=1" in _rr_env and "SAVE_RAW_GEN=1" in _rr_env
    assert "DATASET=movielens" in _rr_env and "ML_TARGET=Action" in _rr_env
    assert "POP_RESET" not in _rr_env
    # the training set is rebuilt FULL SIZE: the subsample knob must be
    # absent, or the two waves would be cutting rows at the same time
    assert "SFT_SAMPLE_N" not in _rr_env and "SFT_MAX_STEPS" not in _rr_env
    assert f'CUDADeviceName == "{RR_H100}"' in _rr_sub
    assert not (set(_rr_tags) & _rr_prior), \
        f"rr collision: {set(_rr_tags) & _rr_prior}"
    p = os.path.join(HERE, f"configs_pofd_{RR_KEY}.txt")
    files[p] = rows_rr
    expected[p] = 4
    cube_subs[os.path.join(HERE, f"at_pofd_{RR_KEY}.sub")] = _rr_sub
    # the 3-round q=0.10 smoke: its OWN key, deliberately outside the
    # four-job production count
    rows_rrs = rr_smoke_rows()
    assert len(rows_rrs) == 1, len(rows_rrs)
    _rrs_tags = {r.split(",")[0] for r in rows_rrs}
    # the smoke's own PREFIX: check_ref_replay enforces the 100-round
    # horizon for pofdrr_ and exempts pofdrrsmk_, so a short cell under
    # the production prefix would be gated as a truncated run
    assert all(t.startswith(f"pofdrrsmk_{RR_MODEL}_")
               and f"_{rr_q_tok(RR_SMOKE_Q)}_" in t
               and t.endswith(f"_s{RR_SEED}_r{RR_SMOKE_ROUNDS}")
               and "_eaopen_" in t and "_esopen_" in t
               and "_w1_" in t and "_l1_" in t
               for t in _rrs_tags), _rrs_tags
    # ... and it still starts with pofdrr, which is how that checker
    # claims the run in the first place
    assert all(t.startswith("pofdrr") for t in _rrs_tags), _rrs_tags
    assert not any(t.startswith("pofdrrsmk") for t in _rr_tags), _rr_tags
    assert RR_SMOKE_Q == 0.10 and RR_SMOKE_ROUNDS == 3
    for r in rows_rrs:
        _c = [x.strip() for x in r.split(",")]
        assert _c[1] == "sft" and _c[2] == "0", r
        assert _c[11] == "1" and _c[14] == "1", r      # W = 1, k = 1
        assert float(_c[15]) == RR_SMOKE_Q, r
        assert _c[22] == str(RR_SMOKE_ROUNDS), r
    assert not (_rrs_tags & set(_rr_tags)), \
        f"smoke would shadow a production cell: {_rrs_tags & set(_rr_tags)}"
    _rrs_sub = rr_sub(smoke=True)
    _rrs_env = next(ln for ln in _rrs_sub.splitlines()
                    if ln.startswith("environment"))
    assert "REF_REPLAY_Q=$(refq) " in _rrs_env
    assert f"REF_REPLAY_SEED={RR_SEL_SEED} " in _rrs_env
    assert f"REF_REPLAY_REF_RUN={RR_REF_RUN} " in _rrs_env
    _prior_rrs = {r.split(",")[0] for rows in files.values() for r in rows}
    assert not (_rrs_tags & _prior_rrs), \
        f"rr smoke collision: {_rrs_tags & _prior_rrs}"
    p = os.path.join(HERE, f"configs_pofd_{RR_SMOKE_KEY}.txt")
    files[p] = rows_rrs
    expected[p] = 1
    cube_subs[os.path.join(HERE, f"at_pofd_{RR_SMOKE_KEY}.sub")] = _rrs_sub

    # ---- Section-3 retention table (S3) ------------------------------
    # Registered LAST on purpose: the reuse assertion below requires that
    # every key that could supply an archived tag (QWU in particular) is
    # already in `files`.
    _s3_cells = s3_cells()
    rows_s3 = s3_rows()
    # THE ARITHMETIC, recomputed rather than trusted.
    #   50 conceptual = 2 models x 3 envs x 7 (sft + 6 forward)   [42]
    #                 + 2 models x 2 k=1 envs x 2 reverse          [8]
    #   46 new        = 50 - the 4 archived Qwen2.5 QWU cells
    assert len(_s3_cells) == S3_N_CONCEPTUAL == 50, len(_s3_cells)
    assert (len(S3_MODELS) * len(S3_ENVS) * (1 + len(S3_FWD_LAMS))
            + len(S3_MODELS) * len(S3_REV_ENVS) * len(S3_REV_LAMS)
            == S3_N_CONCEPTUAL), "the design arithmetic changed"
    assert len(S3_REUSED) == 4, S3_REUSED
    assert len(rows_s3) == S3_N_NEW == 46 == \
        len(_s3_cells) - len(S3_REUSED), len(rows_s3)
    _s3_tags = [r.split(",")[0] for r in rows_s3]
    assert len(set(_s3_tags)) == S3_N_NEW, "duplicate S3 tag"
    # every declared reuse must name a cell the DESIGN actually contains,
    # or the "50 - 4" would be subtracting something that was never there
    _s3_cell_keys = {(_m, _a, _w, _k)
                     for _m, _a, _st, _b, _d, _w, _k in _s3_cells}
    assert len(_s3_cell_keys) == S3_N_CONCEPTUAL
    for _rk in S3_REUSED:
        assert _rk in _s3_cell_keys, \
            f"S3_REUSED names a cell the design does not contain: {_rk}"
    # ... and no reused cell may ever be queued
    for _rk in S3_REUSED:
        assert s3_tag(_rk[0], _rk[1], _rk[2], _rk[3]) not in _s3_tags, _rk
    # BOTH models present, in the counts the reuse split implies
    # (qwen7b 25 - 4 = 21 new; qwen3_8b 25, nothing reused for it)
    assert len(S3_MODELS) == 2 and set(S3_MODELS) == {"qwen7b", "qwen3_8b"}
    for _m, _n in (("qwen7b", 21), ("qwen3_8b", 25)):
        assert sum(1 for t in _s3_tags if f"_{_m}_" in t) == _n, (_m, _n)
    # EXACTLY THREE ENVIRONMENTS, and (W=1, k=0.2) is not one of them
    assert len(S3_ENVS) == 3 and len(set(S3_ENVS)) == 3, S3_ENVS
    assert set(S3_ENVS) == {(0.5, 1.0), (1.0, 1.0), (0.5, 0.2)}, S3_ENVS
    assert (1.0, 0.2) not in S3_ENVS, "k drops out at W=1: no such cell"
    assert {(_w, _k) for _m, _a, _st, _b, _d, _w, _k in _s3_cells} \
        == set(S3_ENVS)
    assert {(c.split(",")[11].strip(), c.split(",")[14].strip())
            for c in rows_s3} == {("0.5", "1"), ("1", "1"), ("0.5", "0.2")}
    # REVERSE: 8 rows, only in the two k=1 environments, never in env3
    _s3_rev = [r for r in rows_s3 if "_revlam" in r.split(",")[0]]
    assert len(_s3_rev) == 8 == \
        len(S3_MODELS) * len(S3_REV_ENVS) * len(S3_REV_LAMS), len(_s3_rev)
    assert S3_ENV_MEM not in S3_REV_ENVS
    for r in _s3_rev:
        _c = [x.strip() for x in r.split(",")]
        assert _c[15] == "reverse", r
        assert _c[14] == "1", r                    # k = 1 only
        assert (float(_c[11]), float(_c[14])) in S3_REV_ENVS, r
    for _lam in S3_REV_LAMS:
        assert sum(1 for t in _s3_tags
                   if f"_revlam{_num(_lam)}_" in t) == 4, _lam
    # SFT: direction-NEUTRAL. style "sft", kl_beta 0, no direction token
    # in the tag, and an inert placeholder in the kldir column.
    _s3_sft = [r for r in rows_s3 if "_sft_" in r.split(",")[0]]
    assert len(_s3_sft) == 4, len(_s3_sft)     # 2 models x 3 envs - 2 reused
    for r in _s3_sft:
        _c = [x.strip() for x in r.split(",")]
        assert _c[1] == "sft" and _c[2] == "0", r
        assert "fwd" not in _c[0] and "rev" not in _c[0], \
            f"the sft arm must carry NO direction token: {r}"
        assert _c[15] == S3_SFT_KLDIR == "forward", r
    # FORWARD ladder: 6 doses x 2 models x 3 envs, minus the 2 reused
    # qwen7b lambda=1 cells
    for _lam in S3_FWD_LAMS:
        _n_exp = len(S3_MODELS) * len(S3_ENVS) - (2 if _lam == 1.0 else 0)
        assert sum(1 for t in _s3_tags
                   if f"_fwdlam{_num(_lam)}_" in t) == _n_exp, _lam
    assert (len(_s3_sft) + sum(1 for t in _s3_tags if "_fwdlam" in t)
            + len(_s3_rev) == S3_N_NEW)
    # the per-row surface, and the tag <-> queue agreement that the
    # empty-$(kldir) failure mode would break
    for r in rows_s3:
        _c = [x.strip() for x in r.split(",")]
        assert _c[3] == str(S3_SEED) == "0", r
        assert _c[5] == "replace", r                  # replace-only data
        assert _c[8] == "ab", r
        assert _c[9] == f"{S3_EPS_SOCIAL:g}" == "0.2", r  # inert, never 0
        assert _c[10] == "0.0", r                     # homophily gamma = 0
        assert _c[11] in ("0.5", "1"), r              # W = Celestine beta
        assert _c[14] in ("1", "0.2"), r              # k = Celestine gamma
        assert not (_c[11] == "1" and _c[14] == "0.2"), \
            f"(W=1, k=0.2) must never be generated: {r}"
        assert _c[15] in ("forward", "reverse"), r    # kldir column
        # every NON-sft row's tag direction token agrees with its kldir
        if "_sft_" not in _c[0]:
            assert _c[1] == "sft_kl", r
            assert ("_fwdlam" in _c[0]) == (_c[15] == "forward"), r
            assert ("_revlam" in _c[0]) == (_c[15] == "reverse"), r
            # ... and the lambda in the tag is the lambda on the queue
            assert f"lam{_num(float(_c[2]))}_" in _c[0], r
        assert _c[16] == "0" and _c[17] == "-1", r    # no ICL anywhere
        assert _c[18] == "1" and _c[19] == "1", r     # LoRA, fresh/round
        assert _c[22] == str(S3_ROUNDS) == "100", r
        assert _c[23] in ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B"), r
        # Qwen3 rides hybrid reasoning explicitly OFF; Qwen2.5 the
        # family default (byte-identical template call to every archive)
        assert _c[24] == ("0" if "_qwen3_8b_" in _c[0] else "default"), r
    # tag grammar
    for t in _s3_tags:
        assert t.startswith("pofds3_"), t
        assert "_eaopen_" in t and "_esopen_" in t, t
        assert "_ea1_" not in t and "_es1_" not in t, t
        # the operator token, spelled BOTH through the constant and as a
        # literal: the literal is the half that catches a silent rename
        assert f"_{S3_OP_TOKEN}_" in t and "_anch2_" in t, t
        assert "_hg2_" not in t, t      # the retired draft token
        assert t.endswith(f"_s{S3_SEED}_r{S3_ROUNDS}"), t
        assert "smoke" not in t and "smk" not in t, t
    _s3_sub = s3_sub()
    _s3_env = next(ln for ln in _s3_sub.splitlines()
                   if ln.startswith("environment"))
    # THE FAILURE MODE THIS WAVE MUST NOT REPEAT: an env that reads
    # $(kldir) while the queue line never declares the column expands to
    # the EMPTY STRING, and the runner's _env_or("KL_DIRECTION",
    # "reverse") silently makes every cell reverse -- forward ones too.
    # Both halves are checked, and the column's SLOT is pinned.
    assert "KL_DIRECTION=$(kldir)" in _s3_env, _s3_env
    assert "KL_DIRECTION=forward" not in _s3_env, _s3_env
    assert "KL_DIRECTION=reverse" not in _s3_env, _s3_env
    _s3_q = next(ln for ln in _s3_sub.splitlines()
                 if ln.startswith("queue "))
    assert ", lam, kldir, iclk," in _s3_q, _s3_q
    assert f"configs_pofd_{S3_KEY}.txt" in _s3_q, _s3_q
    # the pinned environment of every trained cell
    assert "DATASET=movielens" in _s3_env and "ML_TARGET=Action" in _s3_env
    assert "AI_GATE_MODE=all_open" in _s3_env
    assert "PEER_GATE_MODE=all_open" in _s3_env
    assert "EPS_AI=1 " in _s3_env
    assert "INNATE_LAMBDA=$(lam)" in _s3_env       # k rides the queue
    assert "ICL_K=$(iclk)" in _s3_env and "ICL_DAYS=0 " in _s3_env
    assert "USE_LORA=$(uselora)" in _s3_env
    assert "FRESH_EACH_ROUND=$(fresh)" in _s3_env
    assert "LORA_R=512" in _s3_env and "SFT_LR=5e-5" in _s3_env
    assert "SFT_EPOCHS=1" in _s3_env and "SFT_BATCH_SIZE=4" in _s3_env
    assert "EPOCH_SIZE=100" in _s3_env
    assert "N_LABELED=723" in _s3_env and "TRAIN_CAP=723" in _s3_env
    assert "WITH_TWIN=1" in _s3_env and "SAVE_RAW_GEN=1" in _s3_env
    assert "SEED_BASE_DATA=1" in _s3_env
    assert "N_ROUNDS=$(nrounds)" in _s3_env
    assert "BASE_MODEL=$(basemodel)" in _s3_env
    assert "CHAT_THINKING=$(chatthink)" in _s3_env
    # knobs that belong to OTHER waves and must not leak in here
    assert "POP_RESET" not in _s3_env
    assert "KL_REF_ADAPTER" not in _s3_env         # the raw base is the ref
    assert "SFT_SAMPLE_N" not in _s3_env and "SFT_MAX_STEPS" not in _s3_env
    assert "REF_REPLAY" not in _s3_env
    assert "POP_MODEL=fj" not in _s3_env
    # H100 80GB only
    assert f'CUDADeviceName == "{S3_H100}"' in _s3_sub
    assert "CUDAGlobalMemoryMb >= 80000" in _s3_sub
    # no collision with ANY previously generated key ...
    _prior_s3 = {r.split(",")[0] for rows in files.values() for r in rows}
    assert not (set(_s3_tags) & _prior_s3), \
        f"s3 collision: {set(_s3_tags) & _prior_s3}"
    # ... and every DECLARED reuse tag must be a tag some OTHER key
    # actually generates, so a rename can never orphan one silently
    for _rk, _rt in S3_REUSED.items():
        assert _rt in _prior_s3, \
            f"S3_REUSED {_rk} -> {_rt} is generated by no key"
        assert _rt not in _s3_tags, f"{_rt} is reused, it must not queue"
    # the two sft reuses ARE the QWU b0 cells and the two fwdlam1 reuses
    # ARE the QWU b1 cells -- named through the generator's own tag
    # builder rather than by string similarity
    for (_m, _a, _w, _k), _rt in S3_REUSED.items():
        assert _rt == qwu_tag("b0" if _a == "sft" else "b1", _w), (_rt, _a)
        assert _k == QWU_K == 1.0, (_rt, _k)
        # the reuses are ARCHIVED QWU tags and must stay byte-identical:
        # no operator token, ever. Adding one would rename a directory
        # that already exists on the cluster and orphan the reuse.
        assert _rt.startswith("pofdqwu_"), _rt
        assert S3_OP_TOKEN not in _rt and "anch2" not in _rt, _rt
        assert "hg2" not in _rt, _rt
    p = os.path.join(HERE, f"configs_pofd_{S3_KEY}.txt")
    files[p] = rows_s3
    expected[p] = S3_N_NEW
    cube_subs[os.path.join(HERE, f"at_pofd_{S3_KEY}.sub")] = _s3_sub
    # the 3-round Qwen3 reverse smoke: a SEPARATE key, deliberately
    # outside the 46-job production count
    rows_s3s = s3_smoke_rows()
    assert len(rows_s3s) == 1, len(rows_s3s)
    _s3s_tags = {r.split(",")[0] for r in rows_s3s}
    assert _s3s_tags == \
        {"pofds3smk_qwen3_8b_revlam1_eaopen_w1_k1_esopen_anch2_s0_r3"}, \
        _s3s_tags
    # A SMOKE MUST BE IMPOSSIBLE TO CONFUSE WITH -- OR TO SATISFY -- A
    # PRODUCTION CELL. Different prefix (pofds3smk_ never starts with
    # pofds3_), and neither tag may be a prefix of the other in either
    # direction, so no prefix-matching consumer can pair them up.
    assert not (_s3s_tags & set(_s3_tags)), _s3s_tags
    for _st in _s3s_tags:
        assert _st.startswith("pofds3smk_"), _st
        assert not _st.startswith("pofds3_"), _st
        for t in _s3_tags:
            assert not _st.startswith(t) and not t.startswith(_st), (_st, t)
    for r in rows_s3s:
        _c = [x.strip() for x in r.split(",")]
        assert _c[1] == "sft_kl" and _c[2] == "1", r      # reverse lambda 1
        assert _c[3] == "0", r                             # seed 0
        assert _c[11] == "1" and _c[14] == "1", r          # W = 1, k = 1
        assert _c[15] == "reverse", r
        assert _c[22] == str(S3_SMOKE_ROUNDS) == "3", r
        assert _c[23] == "Qwen/Qwen3-8B" and _c[24] == "0", r
    _s3s_sub = s3_sub(smoke=True)
    _s3s_env = next(ln for ln in _s3s_sub.splitlines()
                    if ln.startswith("environment"))
    assert "KL_DIRECTION=$(kldir)" in _s3s_env, _s3s_env
    _s3s_q = next(ln for ln in _s3s_sub.splitlines()
                  if ln.startswith("queue "))
    assert ", lam, kldir, iclk," in _s3s_q, _s3s_q
    assert f"configs_pofd_{S3_SMOKE_KEY}.txt" in _s3s_q, _s3s_q
    assert f'CUDADeviceName == "{S3_H100}"' in _s3s_sub
    assert "AI_GATE_MODE=all_open" in _s3s_env
    assert "PEER_GATE_MODE=all_open" in _s3s_env
    _prior_s3s = {r.split(",")[0] for rows in files.values() for r in rows}
    assert not (_s3s_tags & _prior_s3s), \
        f"s3 smoke collision: {_s3s_tags & _prior_s3s}"
    p = os.path.join(HERE, f"configs_pofd_{S3_SMOKE_KEY}.txt")
    files[p] = rows_s3s
    expected[p] = 1
    cube_subs[os.path.join(HERE, f"at_pofd_{S3_SMOKE_KEY}.sub")] = _s3s_sub
    # ---- peer-sweep strength (PS) --------------------------------------
    rows_ps = ps_rows()
    assert len(rows_ps) == 9, len(rows_ps)          # 3 arms x S in {5,20,100}
    _ps_tags = [r.split(",")[0] for r in rows_ps]
    assert len(set(_ps_tags)) == 9, _ps_tags
    assert all(t.startswith("pofdps_") for t in _ps_tags), _ps_tags
    # S = 1 is REUSED and must never be queued
    assert not any("_sw1_" in t for t in _ps_tags), _ps_tags
    for S in PS_SWEEPS_NEW:
        assert sum(f"_sw{S}_" in t for t in _ps_tags) == 3, (S, _ps_tags)
    for r in rows_ps:
        _c = [c.strip() for c in r.split(",")]
        assert _c[3] == "0", r                       # seed
        assert _c[10] == "0.0", r                    # homophily gamma
        assert _c[11] == "1", r                      # W = 1
        assert _c[14] == "1", r                      # k = 1
        assert _c[15] == "forward", r                # forward KL only
        assert int(_c[16]) in PS_SWEEPS_NEW, r       # sweeps column
        assert _c[23] == str(PS_ROUNDS), r
        assert _c[25] == "0", r                      # Qwen3 thinking OFF
        # the sweep count in the TAG must equal the sweeps COLUMN
        assert f"_sw{_c[16]}_" in _c[0], r
    _ps_sub = ps_sub()
    _ps_env = next(l for l in _ps_sub.splitlines() if l.startswith("environment"))
    assert "AB_SWEEPS=$(sweeps)" in _ps_env, _ps_env
    assert "KL_DIRECTION=$(kldir)" in _ps_env
    assert "POP_MODEL=fj" not in _ps_env             # AB_SWEEPS>1 needs ab
    _ps_q = next(l for l in _ps_sub.splitlines() if l.startswith("queue "))
    # env referencing a column the queue never declares expands to "" and
    # AB_SWEEPS silently falls back to 1 -- on every cell
    assert ", kldir, sweeps, iclk," in _ps_q, _ps_q
    _prior_ps = {r.split(",")[0] for rows in files.values() for r in rows}
    assert not (set(_ps_tags) & _prior_ps), set(_ps_tags) & _prior_ps
    for _arm, _t in PS_REUSED.items():
        assert _t in _prior_ps, f"PS_REUSED {_arm} -> {_t} is not generated"
    p = os.path.join(HERE, f"configs_pofd_{PS_KEY}.txt")
    files[p] = rows_ps
    expected[p] = 9
    cube_subs[os.path.join(HERE, f"at_pofd_{PS_KEY}.sub")] = _ps_sub
    rows_pss = ps_smoke_rows()
    assert len(rows_pss) == 1
    _pss = [c.strip() for c in rows_pss[0].split(",")]
    assert _pss[0].startswith("pofdpssmk_") and "_sw100_" in _pss[0]
    assert _pss[1] == "sft_kl" and _pss[2] == "8" and _pss[16] == "100"
    assert _pss[23] == str(PS_SMOKE_ROUNDS)
    assert _pss[0] not in set(_ps_tags)
    _pss_sub = ps_sub(smoke=True)
    assert "AB_SWEEPS=$(sweeps)" in next(
        l for l in _pss_sub.splitlines() if l.startswith("environment"))
    p = os.path.join(HERE, f"configs_pofd_{PS_SMOKE_KEY}.txt")
    files[p] = rows_pss
    expected[p] = 1
    cube_subs[os.path.join(HERE, f"at_pofd_{PS_SMOKE_KEY}.sub")] = _pss_sub
    # ---- memory extension (MEM), an S x k grid --------------------------
    rows_mem = mem_rows()
    # 2 sweep counts x 3 k x 3 arms = 18 conceptual, minus the 6 archived
    # Section 3 cells at (S=1, k=1) and (S=1, k=0.2) = 12 queued
    assert len(rows_mem) == 12, len(rows_mem)
    _mem_tags = [r.split(",")[0] for r in rows_mem]
    assert len(set(_mem_tags)) == 12, _mem_tags
    # the reused (S, k) corners must never be queued
    for _S, _k in MEM_REUSED_SK:
        assert not any(f"_sw{_S}_" in t and f"_k{_num(_k)}_" in t
                       for t in _mem_tags), (_S, _k)
    assert sum("_sw1_" in t for t in _mem_tags) == 3, _mem_tags    # k=0.5 only
    assert sum("_sw20_" in t for t in _mem_tags) == 9, _mem_tags   # all three k
    for r in rows_mem:
        _c = [c.strip() for c in r.split(",")]
        assert _c[10] == "0.0", r                    # homophily gamma
        assert _c[11] == "0.5", r                    # W = 0.5
        assert float(_c[14]) in (1.0, 0.5, 0.2), r   # k
        assert _c[15] == "forward", r                # forward KL only
        assert int(_c[16]) in MEM_SWEEPS, r          # S = complete sweeps
        assert "rev" not in _c[0], r
        assert _c[23] == str(MEM_ROUNDS), r
        assert _c[25] == "0", r                      # Qwen3 thinking OFF
        # both dials in the tag must equal their columns
        assert f"_sw{_c[16]}_" in _c[0], r
        assert f"_k{_num(float(_c[14]))}_" in _c[0], r
    _mem_sub = mem_sub()
    _mem_env = next(l for l in _mem_sub.splitlines() if l.startswith("environment"))
    assert "INNATE_LAMBDA=$(lam)" in _mem_env
    assert "AB_SWEEPS=$(sweeps)" in _mem_env, _mem_env
    _mem_q = next(l for l in _mem_sub.splitlines() if l.startswith("queue "))
    assert ", lam, kldir, sweeps, iclk," in _mem_q, _mem_q
    _prior_mem = {r.split(",")[0] for rows in files.values() for r in rows}
    assert not (set(_mem_tags) & _prior_mem), set(_mem_tags) & _prior_mem
    for _slot, _t in mem_reused().items():
        assert _t in _prior_mem, f"MEM_REUSED {_slot} -> {_t} is not generated"
    assert len(mem_reused()) == 6, len(mem_reused())
    p = os.path.join(HERE, f"configs_pofd_{MEM_KEY}.txt")
    files[p] = rows_mem
    expected[p] = 12
    cube_subs[os.path.join(HERE, f"at_pofd_{MEM_KEY}.sub")] = _mem_sub
    rows_mems = mem_smoke_rows()
    assert len(rows_mems) == 1
    _ms = [c.strip() for c in rows_mems[0].split(",")]
    assert _ms[0].startswith("pofdmemsmk_") and "_sw20_" in _ms[0]
    assert _ms[1] == "sft_kl" and _ms[2] == "8"
    assert _ms[14] == "0.2" and _ms[16] == "20"
    assert _ms[23] == str(MEM_SMOKE_ROUNDS)
    assert _ms[0] not in set(_mem_tags)
    p = os.path.join(HERE, f"configs_pofd_{MEM_SMOKE_KEY}.txt")
    files[p] = rows_mems
    expected[p] = 1
    cube_subs[os.path.join(HERE, f"at_pofd_{MEM_SMOKE_KEY}.sub")] = mem_sub(smoke=True)

    m, b, e, s = SMOKE
    files[os.path.join(HERE, "configs_pofd_smoke.txt")] = [ROW.format(
        tag=tag_of(m, b, e, s, prefix="pofdsmk"),
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=s, eps_ai=f"{e:g}")]

    ok = True
    for path, rows in files.items():
        body = "\n".join(rows) + "\n"
        tags = [r.split(",")[0] for r in rows]
        assert len(tags) == len(set(tags)), f"duplicate tags in {path}"
        if path in expected:
            assert len(rows) == expected[path], \
                f"{path}: {len(rows)} rows != {expected[path]}"
        if verify:
            on_disk = open(path).read() if os.path.exists(path) else None
            status = "OK" if on_disk == body else "MISMATCH"
            ok &= status == "OK"
            print(f"[verify] {os.path.basename(path)}: {len(rows)} jobs, {status}")
        else:
            with open(path, "w") as fh:
                fh.write(body)
            print(f"[write] {os.path.basename(path)}: {len(rows)} jobs")
    for path, body in cube_subs.items():
        if verify:
            on_disk = open(path).read() if os.path.exists(path) else None
            status = "OK" if on_disk == body else "MISMATCH"
            ok &= status == "OK"
            print(f"[verify] {os.path.basename(path)}: generated sub, {status}")
        else:
            with open(path, "w") as fh:
                fh.write(body)
            print(f"[write] {os.path.basename(path)}: generated sub")
    print(f"[grid] {len(ACTIVE_MODELS)} model(s) x {len(BETAS)} beta x "
          f"{len(EPS_AIS)} eps_ai x {len(SEEDS)} seed(s) = "
          f"{len(ACTIVE_MODELS) * len(BETAS) * len(EPS_AIS) * len(SEEDS)} sweep jobs"
          f" + {len(PFRAC_MODELS) * len(PFRACS) * len(EPS_AIS) * len(SEEDS)} pfrac "
          f"(data-regime) jobs across {len(PFRAC_MODELS)} model(s)"
          f" + {len(BP_BETAS) * len(PFRACS) * len(BP_EPS) * len(SEEDS)} bp (beta x pfrac)"
          f" + {2 * len(PFRACS) * len(EPS_AIS)} pf2/pfs2 (env2/env3 data-regime)"
          f" + {len(ICL_ARMS) * len(EPS_AIS) * len(SEEDS)} icl"
          f" + {len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(EPS_AIS) * len(SEEDS)} dpo"
          f" + {len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(DPON_EPS) * len(SEEDS)} dpon (noisy)"
          f" + {len(BETAS) * len(EPS_AIS) * len(SEEDS)} w (W=0.5 FJ lam=0.2)"
          f" + {len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(EPS_AIS) * len(SEEDS)} wdpo"
          f" + {len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(DPON_EPS) * len(SEEDS)} wdpon"
          f" + {len(BETAS) * len(EPS_AIS) * len(SEEDS)} ws (+eps_social=0.2)"
          f" + {sum(len(rpl_rows(m)) for m in RPL_MODELS)} replay1 "
          f"(exact initial-data replay; rf=0 reused)"
          f" + {sum(len(bud_rows(m)) for m in BUD_MODELS)} budget "
          f"(SFT step cap; full epoch reused)"
          f" + {sum(len(iclf_rows(m)) for m in ICLF_MODELS)} iclf "
          f"(frozen-context icl; audit reused 0)"
          f" + {len(ctf_rows())} ctf (SFT-to-ICL context transfer)"
          f" + {sum(len(sc_rows(m)) for m in SC_MODELS)} seedcore "
          f"(seed replicates: retention 12 + direct 8 + main peer 14)"
          f" + {len(ffsi_rows()) + len(ffrp_rows()) + len(ffc_rows())} "
          f"finalfill (sfticl 18 + replay 16 + corners 8; umbrella adds "
          f"the untouched 12-job olmo7brom_fe wave -> 54 queued)"
          f" + {3 * len(TFE_CI_SEEDS)} tfe_ci (controlled-teacher seeds "
          f"{{44,45}} -> 5 seeds/arm)"
          f" + 52 dpo_ci (staged: ci1 8 confirm + ci2 24 sharp-grid "
          f"+ ci3 20 peer breadth incl. new es0p3 dose)"
          f" + {2 * len(MISTRAL_WS2F_SEEDS)} mistral7b_ws2f (4th model, "
          f"canonical main env b{{0,1}} x 3 seeds)"
          f" + 210 mistral7b_cube (3-seed cube: 70 s0 + 140 repl; "
          f"6 ws2f cells reused per the config-field audit)"
          f" + 15 dpo_mr pairs (matched-randomness closed/open, "
          f"= 30 arm trajectories)"
          f" + 327 sft_icl_reach mains (360 conceptual cells, 33 audited "
          f"reused: qwen 99 + olmo 108 + mistral 120 queue) + 15 reach "
          f"baseline probes"
          f" + 22 sft_k0_nopeer (54 conceptual seed-0 cells, 32 audited "
          f"reused: qwen 6 + olmo 6 + mistral 10 queue; ea1 = strict "
          f"numeric threshold)"
          f" + {sum(1 for f in files if 'smoke' in f)} smokes")
    if verify and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
