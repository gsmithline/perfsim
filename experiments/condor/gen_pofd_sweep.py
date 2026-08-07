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
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000)

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
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000)

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
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000)

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
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000)

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
requirements      = (TARGET.CUDAGlobalMemoryMb >= 80000)

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
          f" + {sum(1 for f in files if 'smoke' in f)} smokes")
    if verify and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
