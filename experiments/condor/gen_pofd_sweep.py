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


def w_rows(prefix="pofdw", betas=BETAS):
    out = []
    for seed in SEEDS:
        for beta in betas:
            for eps_ai in EPS_AIS:
                tag = (f"{prefix}_{W_MODEL}_b{_num(beta)}_ea{_num(eps_ai)}"
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


def ws_rows(prefix="pofdws", betas=BETAS):
    out = []
    for seed in SEEDS:
        for beta in betas:
            for eps_ai in EPS_AIS:
                tag = (f"{prefix}_{W_MODEL}_b{_num(beta)}_ea{_num(eps_ai)}"
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
    # Same environments as pofdw_/pofdws_ under the corrected round operator;
    # the tags must differ so these write to new run dirs and never overwrite
    # the superseded ones. Start-with beta grid drops 0.1 (user 2026-07-28:
    # beta in {0, 0.2, 0.5, 1}).
    w2_betas = [0.0, 0.2, 0.5, 1.0]
    p = os.path.join(HERE, "configs_pofd_qwen7b_w2.txt")
    files[p] = w_rows("pofdw2", w2_betas)
    expected[p] = len(w2_betas) * len(EPS_AIS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_w2_smoke.txt")] = [ROW_W.format(
        tag=f"pofdw2smk_qwen7b_b0p5_ea0p2_{w_tok()}_s0_fresh_data",
        style="sft_kl", beta="0.5", seed=0, eps_ai="0.2")]
    p = os.path.join(HERE, "configs_pofd_qwen7b_ws2.txt")
    files[p] = ws_rows("pofdws2", w2_betas)
    expected[p] = len(w2_betas) * len(EPS_AIS) * len(SEEDS)
    files[os.path.join(HERE, "configs_pofd_qwen7b_ws2_smoke.txt")] = [ROW_WS.format(
        tag=f"pofdws2smk_qwen7b_b0p5_ea0p2_{ws_tok()}_s0_fresh_data",
        style="sft_kl", beta="0.5", seed=0, eps_ai="0.2")]
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
    print(f"[grid] {len(ACTIVE_MODELS)} model(s) x {len(BETAS)} beta x "
          f"{len(EPS_AIS)} eps_ai x {len(SEEDS)} seed(s) = "
          f"{len(ACTIVE_MODELS) * len(BETAS) * len(EPS_AIS) * len(SEEDS)} sweep jobs"
          f" + {len(PFRAC_MODELS) * len(PFRACS) * len(EPS_AIS) * len(SEEDS)} pfrac "
          f"(data-regime) jobs across {len(PFRAC_MODELS)} model(s)"
          f" + {len(BP_BETAS) * len(PFRACS) * len(BP_EPS) * len(SEEDS)} bp (beta x pfrac)"
          f" + {len(ICL_ARMS) * len(EPS_AIS) * len(SEEDS)} icl"
          f" + {len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(EPS_AIS) * len(SEEDS)} dpo"
          f" + {len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(DPON_EPS) * len(SEEDS)} dpon (noisy)"
          f" + {len(BETAS) * len(EPS_AIS) * len(SEEDS)} w (W=0.5 FJ lam=0.2)"
          f" + {len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(EPS_AIS) * len(SEEDS)} wdpo"
          f" + {len(DPO_BETAS) * len(DPO_FEEDBACKS) * len(DPON_EPS) * len(SEEDS)} wdpon"
          f" + {len(BETAS) * len(EPS_AIS) * len(SEEDS)} ws (+eps_social=0.2)"
          f" + 8 smokes")
    if verify and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
