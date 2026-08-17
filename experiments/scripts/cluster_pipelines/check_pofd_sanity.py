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

  GATE-MASK   runs that carry gate_raw (runner 2026-08-05, cube wave on):
              per-round per-agent AI contact mask must be bit-equal to
              |served - x(t)| < eps_AI on the start-of-round opinion, with
              shape == op_raw and per-round mean == row['contact']. Older
              runs (key absent/empty) skip it -- the mask reconstructs
              offline from pred_raw/op_raw the same way. Cube-family tags
              (pofdw2f_/pofdws2f_/pofdesf_, plus the replay wave pofdrpl_)
              additionally cross-check the _b token against kl_beta, style
              (sft iff b==0), and kl_direction=forward at b>0.

  REPLAY      pofdrpl_ runs (replay_frac>0, runner 2026-08-06): the _rf token
              must match config replay_frac, and the saved replay_raw /
              train_y_raw must show, per deploy round, a fixed-size batch with
              EXACTLY round(rf*n) rows carrying their original round-0 labels
              (raw innate; teacher shift config-gated to 0) and the rest the
              LATEST round's labels (op_raw[t-1]) -- bit-exact, never
              accumulated history; round 0 all-original; row n_replay must
              match the mask. replay_raw present at replay_frac==0 is an
              error (that path must not exist -- alpha=0 runs are the plain
              replace runs).

  ICL-CTX     pofdiclf_ runs (frozen-context icl wave, incl. pofdiclfsmk_):
              arm token (k0/fz0/fz8/dyn) must match (icl_k,
              icl_snapshot_round); icl_idx_raw/icl_val_raw log the exact
              exemplar ids + displayed values, icl_ctx_log.json.gz the
              rendered text. Checks: no agent in its own context; after the
              snapshot round the cached (ids, vals) AND the rendered text
              never change; fixed-text perplexity constant every round
              (weights frozen); gg telemetry + matched twin present; a fz8
              run's SELECTION stream (icl_idx_raw, model-free RNG) must be
              bit-identical to its _dyn_ sibling through round 8 when the
              sibling run dir exists -- divergence there is a code bug.
              Generated values (pred/op/icl_val) are bit-identical only on
              matching GPU models; across heterogeneous hardware they can
              flip borderline generations from round 0, so value divergence
              is reported as a WARN with stats, never a failure.

  CTF-DONOR   pofdctf_ runs (SFT-to-ICL context transfer, incl.
              pofdctfsmk_): frozen recipient whose frozen-at-round-0 K=8
              context displays a DONOR run's opinions (arm pri = donor
              innate; b0r9/b1r9 = the b0/b1 fes donor's op_raw[9]; donor
              tag+round+sha256 in config). Checks: saved donor vector
              hashes to the config hash; displayed values == donor[ids]
              exactly every round; pri donor == recipient innate; vector
              re-derives bit-exactly from the donor run when pulled
              alongside; matched arms share exemplar identities; donor
              gg / incremental-R^2 / realized-context-gap diagnostics and
              per-row gg_ctx_true present. Shares the ICL-CTX section's
              frozen-context checks (self-exclusion, cache immutability,
              constant perplexity, gg + twin).

  BUDGET      pofdbud_ runs (training-budget wave, incl. pofdbudsmk_):
              ordinary SFT at b0 with a per-round optimizer-step cap --
              config must show training_style=sft, kl_beta=0, sft_epochs=0,
              max_steps matching the _st token, the base model matching the
              model slug, movielens Action, natural labels. Environment/twin/
              gate-mask/fresh checks ride the shared sections (es token ->
              peer gate + twin + mean conservation, gate_raw bit-replay,
              n_train fixed at TRAIN_CAP on every deploy round).

  SEEDCORE    the 2026-08-07 seed-replication wave adds three gate groups:
              (a) RETENTION -- pofdret_ runs (incl. pofdretsmk_): the
              project's first 1-round loops (one deploy+train+update in the
              Figure-1b environment). Config must show n_rounds=1, movielens
              Action, sft_epochs=1, natural labels, the _b token grammar
              (forward SFT-KL at b>0) and the trajectory must hold exactly
              one round. Twin/peer/fresh ride the shared SOCIAL sections
              (es=0.1 -> peer-alive on round 0, twin_raw mandatory).
              (b) DIRECT -- pofd_/pofdw1f_ tags now get the _b token gate
              (kl_beta + style sft iff b==0) and n_rounds=30; pofdw1f_ b>0
              additionally requires kl_direction=forward (pofd_ b>0 is the
              reverse-era wave, which predates the key -- no direction gate).
              W=1/lam=0/es=0 exact-copy replay + NO-PEER already applied.
              (c) EVERY family: the _s seed token, the _ea dose token and
              the model slug (_qwen7b_/_olmo7b_/_llama8b_/_gemma12b_) are
              checked against config seed/eps_ai/base_model, and
              pofd_/pofdw1f_/pofdesf_/pofdret trajectories must hold exactly
              config n_rounds rounds (completeness).

  REACH       pofdreach_ runs (SFT-ICL reach wave 2026-08-13; incl.
              pofdreachsmk_ 3-round smokes and pofdreachbase_ 1-round
              frozen K=0 baseline probes at eps_ai=0): no-peer (_es0_)
              reach study over arms b0 (sft) / b1 (forward SFT-KL) /
              fz0 (frozen round-0 K=8 context) / dyn (live refreshed
              K=8 context) / k0 (frozen NO-context prompting, the
              sft_k0_nopeer wave: no exemplars/log allowed, constant
              perplexity, static-serving gate stability) x gates ea
              {0.05,0.1,0.2,0.4,0.7} + numeric _ea1_ (strict threshold
              at 1.0 -- NOT all_open) + the explicit all-open gate
              (_eaopen_ <-> AI_GATE_MODE=all_open, never a numeric
              threshold). Gate replays go
              through the SHARED gp.ai_gate (one definition for runner
              + checker). Mandatory per run: exact n_rounds
              completeness, twin_raw == innate (<= 1 float32 ulp, bit-
              stable from round 1 -- lam*v+(1-lam)*v rounds 1 ulp off
              v, so literal bit-identity is unattainable), gate_raw
              present + bit-replayed, finite in-range op/pred (no
              parse-fail NaN), hardware provenance in config, model
              slug/seed/ea/arm tokens vs config. all_open: every gate
              bit true + contact exactly 1.0. fz0: constant gate_raw
              AND constant pred_raw within the run (gate flips
              tolerated ONLY when boundary-explained: the flipped
              agent's round-0 margin ||m-innate|-eps| within 4 float32
              ulp -- the lam-anchor's 1-ulp rounding can flip an agent
              sitting exactly on the strict boundary; observed on the
              s0 slab); matched fz0/dyn
              siblings share bit-identical round-0 context ids+values
              (CPU-derived -- never weakened for GPU heterogeneity).
              Baseline probes cross-check innate identity when pulled
              alongside. NO expected scientific outcome is a validity
              condition (no arm must recruit anyone).

  GATE2D      pofdgate2d_ runs (Mistral 2-D gate grid 2026-08-15; incl.
              pofdgate2dsmk_ 3-round smokes): mistral7b-only sweep of
              arms b0 (ordinary SFT, beta=0) vs dyn (frozen weights,
              live K=8 context refreshed every round) over numeric AI
              gates x numeric SOCIAL gates. _ea1_ = EPS_AI=1.0 under
              the strict-< threshold (never all_open); _es1_ =
              EPS_SOCIAL=1.0, the real peer-confidence gate. Every run
              in this family is es>0 (the es=0 column is audited-reused
              under the reach family): the twin MOVES, gates ride the
              shared gp.ai_gate bit-replay (gate_raw + twin_raw
              mandatory), peer-alive / mean-conservation / contact ride
              the SOCIAL section, dyn exemplar ids must genuinely
              refresh, frozen perplexity constant, hardware provenance
              mandatory, model slug/seed/ea/es/arm tokens vs config. NO
              expected scientific outcome is a validity condition.

  CTXGRID     pofdctxgrid_ runs (context-depth x dual-gate grid
              2026-08-15; incl. pofdctxgridsmk_ 3-round smokes): the
              one-seed 6-channel grid -- b0 ordinary SFT / k0 frozen
              no-context prompting / fz0 fixed K=8 (snapshot round 0) /
              dyn live K=8 / f32 fixed K=32 / d32 live K=32 -- across
              numeric AI gates x numeric SOCIAL gates. Uniquely spans
              BOTH peer regimes, so invariants split on the tokenized
              social dose: es=0 requires twin == innate (<= 1 ulp) and
              frozen gate masks for the static-serving channels (k0 /
              fz0 / f32, 4-ulp boundary tolerance), while es>0 lets the
              twin move and lets peers churn those masks -- only the
              served predictions stay bit-constant. Always mandatory:
              gate_raw + twin_raw, exact n_rounds, finite in-range
              op/pred, hardware provenance, k0 exemplar/log absence,
              live channels genuinely re-drawing exemplar ids, and the
              rendered exemplar count matching the icl_k dial (the gate
              that keeps a K=8 context from masquerading as K=32). NO
              expected scientific outcome is a validity condition.

  CLAMP       pofdclamp_ runs (innate-clamp waves 2026-08-17; incl.
              pofdclampsmk_ 3-round smokes): mistral-only b0 (ordinary
              SFT) vs dyn (frozen weights, live K=8 cross-user context)
              vs d8 (frozen weights, ICL_K=0, ICL_DAYS=8 PERSONAL
              history: each agent's prompt carries only its OWN latest
              eight recorded opinions, oldest to newest -- graph-mask
              cohorts only) with a fixed 145-agent cohort permanently
              pinned to its innate opinions. Tag grammar
              pofdclamp_mistral7b_<arm>_<strat|
              bottom|gclump|gscat>[_stub]_ea<g>_w0p5_l0p2_es<e>_s<seed>:
              tokenless strat/bottom = the no-peer wave (es=0);
              strat/bottom+_stub_ = one-sided stubborn peers at es>0;
              gclump/gscat (graph-placement masks, ALWAYS _stub_, es>=0)
              reconstruct from the committed clamp_graph_masks.json
              artifact with hashes and graph statistics recomputed from
              its edge list. Stubborn: fixed agents keep pairing, only
              responsive endpoints of accepted F-R pairs move (touch/
              reach logs verified; NO isolated condition exists), and
              the responsive mean must replay the nested blend on every
              one-sided-move-free round. Verified: exact
              config surface + tag grammar, exactly round(0.2*n) frozen
              agents (145 of 723), the stored mask reconstructing
              bit-identically from (innate, mode, frac, seed) with a
              matching sha256, frozen rows bit-exact on innate at EVERY
              round in BOTH op_raw and twin_raw, responsive rows on the
              exact platform blend (section 3, frozen-aware) with the
              responsive twin <= 1 ulp of innate, SFT consuming all 723
              labels (FRESH), live-ICL contexts free to carry frozen-
              agent exemplars (excluding them is the failure), no peer
              interaction (es=0 token + accepted==0), and numeric
              eps_AI=1.0 staying threshold mode, never all_open. d8
              adds the PERSONAL-HISTORY replay: icl_days_log.json.gz
              (the exact rendered context per agent per round) must
              byte-match the sentence rebuilt from (innate, op_raw) --
              own values only, oldest to newest, innate-first early
              rounds, window == icl_days -- with ICL_K=0, zero
              cross-user exemplar artifacts (no icl_idx_raw/
              icl_val_raw/icl_ctx_log), frozen weights, and fixed
              agents seeing only their own innate repeated. NO
              expected scientific outcome is a validity condition.

  FAM         pofdfam_ runs (Figure-2 family-prior scout 2026-08-17;
              incl. pofdfamsmk_ 3-round b1 smokes): six checkpoints
              (qwen7b, qwen3_8b thinking-off, olmo7b, olmo3_7b,
              mistral7b, ministral8b) x arms b0 (ordinary SFT) / b0p5 +
              b1 (forward-KL SFT) / k0 (frozen plain prompting) at the
              wide-open numeric gate (EPS_AI=1 strict-< threshold,
              never all_open) with a live peer step (es 0.05/0.2),
              seed 0, canonical Action loop (W=0.5, lam=0.2, matched
              twin, fresh LoRA on trained arms, SAVE_RAW_GEN=1).
              Verified: exact per-slug base_model, qwen3_8b REQUIRES
              chat_thinking=False (all others must lack the key), arm
              training surface incl. kl_direction=forward on the KL
              arms, es/seed/round tag surface, hardware provenance.
              Reused cells keep their original tags and are gated
              under their own families.

  ZSPRIOR     pofdzsprior_ runs (zero-shot prior screen 2026-08-17):
              one 1-round frozen probe per CANDIDATE checkpoint --
              qwen3_8b (Qwen/Qwen3-8B, chat_thinking=False REQUIRED),
              olmo3_7b, ministral8b, mistralnemo -- on the paper's 723
              Action profiles at seed 0, greedy decoding, EPS_AI=0
              under the strict-< threshold gate. Verified: exact config
              surface, exactly one round over 723 agents, opinions
              within 1 ulp of innate (the probe cannot update anyone),
              gate mask all-closed, finite in-[0,1] predictions,
              raw_gen_log.json.gz present with all 723 raw responses,
              EVERY response carrying a digit (parse_fail_frac 0.0),
              and the logged parsed values equal to the served
              pred_raw[0]. A constant prior WARNs (degenerate but
              internally consistent). NO expected scientific outcome
              is a validity condition.

Usage:
  python3 check_pofd_sanity.py <run_dir> [<run_dir> ...]
  python3 check_pofd_sanity.py runs/pokec_gated_lm/pofd_*_fresh_data
Exit 0 iff every run passes every check.
"""
import glob
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys

import torch

# the ONE AI-gate definition (gp.ai_gate) is shared with the runner via
# _gated_pop.py so the deployed update and this replay can never diverge
# (sft_icl_reach wave, 2026-08-13). Loaded like the runner loads it.
_GP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_gated_pop.py")
_spec_gp = importlib.util.spec_from_file_location("_gated_pop_check", _GP_PATH)
gp = importlib.util.module_from_spec(_spec_gp)
_spec_gp.loader.exec_module(gp)


def _bit_eq(a, b):
    """Elementwise bit-identity that treats NaN == NaN (pred_raw carries NaN
    for parse-fail agents; two identical runs share the NaN pattern)."""
    if a.shape != b.shape:
        return False
    return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())

ATOL = 2e-6   # float32 blend arithmetic; W=1 makes the copy exact but the
              # stored tensors round-trip through cpu float32

# model slug -> base checkpoint (seedcore wave, 2026-08-07): the slug is
# tag-encoded in every family, so the pairing is gated universally.
# _olmo7brom_ never matches _olmo7b_ (the token match requires the
# trailing underscore right after the slug).
SLUG_BASE = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo7brom": "allenai/OLMo-2-1124-7B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma12b": "google/gemma-3-12b-it",
    # 4th model, promoted 2026-08-12 off the mlatZ_cand prior screen
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
}


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
    # frozen-CONTEXT icl wave (2026-08-07, pofdiclf_/pofdiclfsmk_) gets its
    # own branch -- exclude it from the legacy icl arm-token gate
    is_iclf = name.startswith("pofdiclf")
    is_icl = (name.startswith("pofdicl")   # covers pofdiclsmk_ too
              and not is_iclf)
    # SFT-to-ICL context-transfer wave (2026-08-07, pofdctf_/pofdctfsmk_):
    # frozen recipient whose K=8 context displays a DONOR run's opinions
    is_ctf = name.startswith("pofdctf")
    # families sharing the frozen-context mandatory checks in section 1d
    is_ctx_fam = is_iclf or is_ctf
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
    # exact initial-data replay wave (2026-08-06): replace-regime loops whose
    # per-round batch holds an exact _rf fraction of ORIGINAL round-0 labels
    is_rpl = name.startswith("pofdrpl_")
    # ordinary-SFT training-budget wave (2026-08-06): per-round step cap via
    # SFT_EPOCHS=0 + SFT_MAX_STEPS (_st token); covers pofdbudsmk_ too
    is_bud = name.startswith("pofdbud")
    # one-update retention family (2026-08-07, seedcore wave): the project's
    # first 1-round loops -- one deploy+train+population update in the
    # Figure-1b environment (W=0.5, lam=0.2, es=0.1, ea=0.4, forward KL at
    # b>0). Covers pofdretsmk_ too. NEW prefix on purpose: a 1-round
    # trajectory under a pofdesf_ tag would collide with future 30-round
    # runs of the same cell (the idempotent exec skips on presence).
    is_ret = name.startswith("pofdret")
    # SFT-ICL REACH family (2026-08-13): no-peer reach study -- shared
    # weight updates (b0/b1) vs fixed round-0 ICL context (fz0) vs live
    # refreshed ICL context (dyn) at gates ea {0.05..0.7} + the explicit
    # all-open gate (_eaopen_, AI_GATE_MODE=all_open). pofdreachbase_ =
    # the one-round frozen no-context K=0 baseline probe (eps_ai=0: the
    # strict threshold gate never opens, so the probe cannot update
    # opinions); pofdreachsmk_ = the 3-round smoke. NOTE "pofdreach" and
    # "pofdret" share only the "pofdre" prefix -- neither startswith
    # check can capture the other family.
    is_reach = name.startswith("pofdreach")
    is_reach_base = name.startswith("pofdreachbase")
    is_reach_smk = name.startswith("pofdreachsmk")
    # PEER-CHANNEL grid (2026-08-14, sft_icl_peer02 wave): the es=0.2
    # completion of the SFT/ICL channel table -- arms b0 (ordinary SFT) /
    # k0 (frozen plain prompting) / fz0 (frozen round-0 K=8 context) /
    # dyn (refreshed live K=8 context) at numeric gates with the peer
    # step LIVE. Unlike the no-peer reach family: the twin MOVES (never
    # compared to innate), gate masks may churn even for static-serving
    # arms (peers move agents across a fixed prediction's gate line), and
    # the exact-copy replay is impossible (peer moves are RNG-pairwise)
    # -- the SOCIAL section's mean-conservation + contact gates apply
    # instead. Covers pofdpeer2smk_ (3-round smokes) too.
    is_peer2 = name.startswith("pofdpeer2")
    is_peer2_smk = name.startswith("pofdpeer2smk")
    # MISTRAL 2-D GATE grid (2026-08-15, sft_icl_gate2d wave): the
    # diagonal-split heatmap family -- mistral7b only, arms b0 (ordinary
    # SFT, beta=0) vs dyn (frozen weights, live K=8 context refreshed
    # every round) at numeric AI gates x numeric SOCIAL gates. BOTH axes
    # are real thresholds: _ea1_ is EPS_AI=1.0 under the strict-< gate
    # (never all_open) and _es1_ is EPS_SOCIAL=1.0 under the real peer-
    # confidence gate. New cells exist only at es>0 (the es=0 column is
    # audited-reused under the reach family), so the twin MOVES and the
    # peer-alive / mean-conservation / contact gates ride the SOCIAL
    # section. Covers pofdgate2dsmk_ (3-round smokes) too.
    is_gate2d = name.startswith("pofdgate2d")
    is_gate2d_smk = name.startswith("pofdgate2dsmk")
    # CONTEXT-DEPTH x DUAL-GATE grid (2026-08-15, sft_icl_ctxgrid wave):
    # one-seed 6-channel grid -- b0 ordinary SFT / k0 frozen no-context
    # prompting / fz0 fixed K=8 (snapshot round 0) / dyn live K=8 / f32
    # fixed K=32 / d32 live K=32 -- across numeric AI gates x numeric
    # SOCIAL gates. Unlike every prior family this one spans BOTH peer
    # regimes: es=0 cells are no-peer (twin == innate, exact-copy replay
    # applies) while es>0 cells run the live peer step (twin MOVES, gate
    # masks may churn even for static-serving arms). "pofdctxgrid" and
    # "pofdctf" share only "pofdct" -- neither prefix test captures the
    # other. Covers pofdctxgridsmk_ (3-round smokes) too.
    is_ctxgrid = name.startswith("pofdctxgrid")
    is_ctxgrid_smk = name.startswith("pofdctxgridsmk")
    # NO-PEER INNATE-CLAMP wave (2026-08-17, mistral_innate_clamp_nopeer):
    # a fixed 20% cohort (bottom / stratified_random by innate) is
    # permanently pinned to its innate opinions in BOTH the deployed
    # population and the matched twin; everyone is still served, gated,
    # labeled and exemplar-eligible. es=0 by construction (the clamp
    # hard-fails under a live peer step), so the exact-copy replay
    # applies with the frozen rows overridden to innate. Covers
    # pofdclampsmk_ (3-round smokes) too.
    is_clamp = name.startswith("pofdclamp")
    is_clamp_smk = name.startswith("pofdclampsmk")
    # ZERO-SHOT PRIOR SCREEN (2026-08-17, zsprior_screen): 1-round frozen
    # probes of CANDIDATE checkpoints (Qwen3-8B thinking-off, Olmo-3-7B-
    # Instruct, Ministral-8B, Mistral-Nemo) on the paper's 723 Action
    # profiles. EPS_AI=0 under the strict-< threshold gate: no agent is
    # ever contacted and opinions cannot update; pred_raw[0] is the
    # checkpoint's zero-shot prior, with every raw decoded response
    # persisted (SAVE_RAW_GEN=1 -> raw_gen_log.json.gz).
    is_zsprior = name.startswith("pofdzsprior")
    # FIGURE-2 FAMILY-PRIOR SCOUT (2026-08-17, fig2_family_prior_scout):
    # six checkpoints (Qwen2.5-7B / Qwen3-8B thinking-off / OLMo-2 /
    # OLMo-3 / Mistral-7B / Ministral-8B) x arms b0/b0p5/b1/k0 at the
    # wide-open numeric gate (EPS_AI=1 strict-< threshold) with a live
    # peer step (es 0.05/0.2), seed 0, canonical Action loop. Covers
    # pofdfamsmk_ (3-round b1 smokes) too.
    is_fam = name.startswith("pofdfam")
    is_fam_smk = name.startswith("pofdfamsmk")
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
    elif is_iclf:
        # frozen-context wave: same frozen surface as icl, plus live ctx,
        # exemplar-only (no days), qwen-only, gg telemetry + twin mandatory.
        # The arm token maps to (icl_k, icl_snapshot_round):
        #   k0 = no context; dyn = rebuilt every round; fz0/fz8 = dynamic
        #   through round 0/8 then each agent's context frozen verbatim.
        want.update({"kl_beta": 0.0, "training_style": "frozen",
                     "pristine_frac": 0.0, "fresh_each_round": False,
                     "use_lora": 0, "icl_select": "random", "icl_days": 0,
                     "icl_ctx_source": "live", "ml_target": "Action",
                     "log_gender_gaps": True, "teacher_label_delta": 0.0,
                     "base_model": "Qwen/Qwen2.5-7B-Instruct"})
        ICLF_ARM_WANT = {"k0": (0, -1), "fz0": (8, 0),
                         "fz8": (8, 8), "dyn": (8, -1)}
        m_arm = re.search(r"_(k0|fz0|fz8|dyn)_ea", name)
        if m_arm is None:
            errs.append(f"CONFIG unknown iclf arm token in dirname {name!r}")
        else:
            k_w, snap_w = ICLF_ARM_WANT[m_arm.group(1)]
            got = (cfg.get("icl_k"),
                   cfg.get("icl_snapshot_round", -1))
            if got != (k_w, snap_w):
                errs.append(f"CONFIG icl (k, snapshot)={got!r} (arm "
                            f"{m_arm.group(1)} wants {(k_w, snap_w)!r})")
    elif is_ctf:
        # context-transfer: frozen recipient, donor-valued context frozen at
        # round 0, canonical peer dose. Arm token maps to (donor round,
        # donor tag): pri = donor innate (round -1) read from the b1 donor;
        # b0r9 / b1r9 = the b0 / b1 fes donor's op_raw[9] (round 9 fixed for
        # every seed -- the three-seed mean population incremental gender
        # R^2 peak under the paper's cross-fitted taste protocol, verified
        # 2026-08-07). The b0 s0 donor lives under the reverse-era pofdws2_
        # tag (b0 = sft, direction-free -- the fes reuse precedent).
        want.update({"kl_beta": 0.0, "training_style": "frozen",
                     "pristine_frac": 0.0, "fresh_each_round": False,
                     "use_lora": 0, "icl_select": "random", "icl_days": 0,
                     "icl_ctx_source": "donor", "icl_k": 8,
                     "icl_snapshot_round": 0, "ml_target": "Action",
                     "log_gender_gaps": True, "teacher_label_delta": 0.0,
                     "base_model": "Qwen/Qwen2.5-7B-Instruct"})
        m_arm = re.search(r"_(pri|b0r9|b1r9)_ea", name)
        m_sd = re.search(r"_s(\d+)(?:_|$)", name)
        if m_arm is None or m_sd is None:
            errs.append(f"CONFIG no ctf arm/seed token in dirname {name!r}")
        else:
            sd = int(m_sd.group(1))
            b1_tag = (f"pofdws2f_qwen7b_b1_ea0p4_w0p5_l0p2_es0p2_s{sd}"
                      f"_fresh_data")
            b0_fam = "pofdws2_" if sd == 0 else "pofdws2f_"
            b0_tag = (f"{b0_fam}qwen7b_b0_ea0p4_w0p5_l0p2_es0p2_s{sd}"
                      f"_fresh_data")
            CTF_ARM_WANT = {"pri": (b1_tag, -1), "b0r9": (b0_tag, 9),
                            "b1r9": (b1_tag, 9)}
            tag_w, rnd_w = CTF_ARM_WANT[m_arm.group(1)]
            got = (cfg.get("icl_ctx_donor_tag"),
                   cfg.get("icl_ctx_donor_round"))
            if got != (tag_w, rnd_w):
                errs.append(f"CONFIG donor (tag, round)={got!r} (arm "
                            f"{m_arm.group(1)} s{sd} wants "
                            f"{(tag_w, rnd_w)!r})")
    elif is_reach:
        # shared surface: movielens Action, greedy serving, natural labels,
        # no replay/pristine anchors, gamma off. eps=0 rides the _es0_
        # token (mandatory below); W/lam ride the _w/_l tokens.
        want.update({"ml_target": "Action", "gamma_bias": 0.0,
                     "teacher_label_delta": 0.0, "pristine_frac": 0.0,
                     "replay_frac": 0.0, "do_sample": False,
                     "deploy_every": 1, "n_labeled": 723, "icrh": False,
                     "feedback_mode": "none"})
        if m_es is None or want_es != 0.0:
            errs.append(f"CONFIG reach tag must carry the _es0_ token "
                        f"(got {name!r})")
        REACH_ARM_WANT = {
            "b0": {"training_style": "sft", "kl_beta": 0.0, "use_lora": 1,
                   "lora_r": 512, "sft_epochs": 1, "fresh_each_round": True,
                   "icl_k": 0, "icl_days": 0, "train_cap": 723},
            "b1": {"training_style": "sft_kl", "kl_beta": 1.0,
                   "kl_direction": "forward", "use_lora": 1, "lora_r": 512,
                   "sft_epochs": 1, "fresh_each_round": True, "icl_k": 0,
                   "icl_days": 0, "train_cap": 723},
            "fz0": {"training_style": "frozen", "kl_beta": 0.0,
                    "use_lora": 0, "fresh_each_round": False, "icl_k": 8,
                    "icl_days": 0, "icl_select": "random",
                    "icl_ctx_source": "live", "icl_snapshot_round": 0},
            "dyn": {"training_style": "frozen", "kl_beta": 0.0,
                    "use_lora": 0, "fresh_each_round": False, "icl_k": 8,
                    "icl_days": 0, "icl_select": "random",
                    "icl_ctx_source": "live", "icl_snapshot_round": -1},
            # k0 = frozen NO-CONTEXT prompting (sft_k0_nopeer wave,
            # 2026-08-14): repeated zero-shot serving of the frozen base
            # model. NOT adaptive ICL -- no demonstrations, no memory, no
            # population context; the only channel into the population is
            # the fixed prediction vector through the gate. icl_select /
            # ctx_source / snapshot are inert at K=0 and not gated.
            "k0": {"training_style": "frozen", "kl_beta": 0.0,
                   "use_lora": 0, "fresh_each_round": False, "icl_k": 0,
                   "icl_days": 0},
        }
        if is_reach_base:
            # one-round frozen K=0 probe: eps_ai=0 under the strict
            # threshold gate -> no agent is ever contacted, opinions
            # cannot update; pred_raw[0] is the frozen no-context
            # prediction vector m_base the common cohorts derive from.
            want.update({"training_style": "frozen", "kl_beta": 0.0,
                         "use_lora": 0, "fresh_each_round": False,
                         "icl_k": 0, "icl_days": 0, "n_rounds": 1,
                         "eps_ai": 0.0, "ai_gate_mode": "threshold"})
        else:
            m_arm = re.search(r"_(b0|b1|fz0|dyn|k0)_ea", name)
            if m_arm is None:
                errs.append(f"CONFIG unknown reach arm token in dirname "
                            f"{name!r}")
            else:
                want.update(REACH_ARM_WANT[m_arm.group(1)])
            want["n_rounds"] = 3 if is_reach_smk else 30
            # gate token <-> gate mode: _eaopen_ is the explicit all-open
            # condition and must NEVER be disguised as a numeric
            # threshold; numeric _ea tokens require threshold mode (the
            # value itself rides the universal _ea gate below).
            if "_eaopen_" in name:
                want["ai_gate_mode"] = "all_open"
            else:
                want["ai_gate_mode"] = "threshold"
                if re.search(r"_ea(\d+(?:p\d+)?)_", name) is None:
                    errs.append(f"CONFIG reach tag carries neither a "
                                f"numeric _ea token nor _eaopen_ ({name!r})")
        if not any(f"_{s}_" in name for s in
                   ("qwen7b", "olmo7b", "mistral7b")):
            errs.append(f"CONFIG no known reach model slug in {name!r}")
        # hardware provenance is mandatory in this family (GPU-dependent
        # borderline generations, 2026-08-07 finding)
        hw = cfg.get("hardware") or {}
        hw_missing = [k for k in ("hostname", "gpu_name", "gpu_cc",
                                  "cuda_version", "torch_version",
                                  "transformers_version") if k not in hw]
        if hw_missing:
            errs.append(f"CONFIG hardware metadata missing keys "
                        f"{hw_missing}")
        elif not (hw.get("hostname") and hw.get("gpu_name")):
            errs.append("CONFIG hardware metadata empty hostname/gpu_name")
    elif is_peer2:
        # shared surface: movielens Action, greedy, natural labels, no
        # anchors/interventions, threshold gating. eps=0.2 rides the
        # MANDATORY _es0p2_ token; W/lam ride the _w/_l tokens.
        want.update({"ml_target": "Action", "gamma_bias": 0.0,
                     "teacher_label_delta": 0.0, "pristine_frac": 0.0,
                     "replay_frac": 0.0, "do_sample": False,
                     "deploy_every": 1, "n_labeled": 723, "icrh": False,
                     "feedback_mode": "none", "ai_gate_mode": "threshold"})
        if m_es is None or abs(want_es - 0.2) > 1e-9:
            errs.append(f"CONFIG peer2 tag must carry the _es0p2_ token "
                        f"(got {name!r})")
        PEER2_ARM_WANT = {
            "b0": {"training_style": "sft", "kl_beta": 0.0, "use_lora": 1,
                   "lora_r": 512, "sft_lr": 5e-5, "sft_epochs": 1,
                   "sft_batch_size": 4, "fresh_each_round": True,
                   "icl_k": 0, "icl_days": 0, "train_cap": 723},
            "k0": {"training_style": "frozen", "kl_beta": 0.0,
                   "use_lora": 0, "fresh_each_round": False, "icl_k": 0,
                   "icl_days": 0},
            "fz0": {"training_style": "frozen", "kl_beta": 0.0,
                    "use_lora": 0, "fresh_each_round": False, "icl_k": 8,
                    "icl_days": 0, "icl_select": "random",
                    "icl_ctx_source": "live", "icl_snapshot_round": 0},
            "dyn": {"training_style": "frozen", "kl_beta": 0.0,
                    "use_lora": 0, "fresh_each_round": False, "icl_k": 8,
                    "icl_days": 0, "icl_select": "random",
                    "icl_ctx_source": "live", "icl_snapshot_round": -1},
        }
        m_arm = re.search(r"_(b0|k0|fz0|dyn)_ea", name)
        if m_arm is None:
            errs.append(f"CONFIG unknown peer2 arm token in dirname "
                        f"{name!r}")
        else:
            want.update(PEER2_ARM_WANT[m_arm.group(1)])
        want["n_rounds"] = 3 if is_peer2_smk else 30
        if re.search(r"_ea(\d+(?:p\d+)?)_", name) is None:
            errs.append(f"CONFIG peer2 tag needs a NUMERIC _ea token "
                        f"({name!r}); all-open is not part of this design")
        if not any(f"_{s}_" in name for s in
                   ("qwen7b", "olmo7b", "mistral7b")):
            errs.append(f"CONFIG no known peer2 model slug in {name!r}")
    elif is_gate2d:
        # shared surface: movielens Action, greedy, natural labels, no
        # anchors/interventions, threshold gating EVERYWHERE. The 2-D
        # sweep rides the universal _ea/_es tokens; mistral-only.
        want.update({"ml_target": "Action", "gamma_bias": 0.0,
                     "teacher_label_delta": 0.0, "pristine_frac": 0.0,
                     "replay_frac": 0.0, "do_sample": False,
                     "deploy_every": 1, "n_labeled": 723, "icrh": False,
                     "feedback_mode": "none", "ai_gate_mode": "threshold",
                     "base_model": "mistralai/Mistral-7B-Instruct-v0.3"})
        if m_es is None or not any(abs(want_es - v) < 1e-9
                                   for v in (0.2, 0.4, 1.0)):
            errs.append(f"CONFIG gate2d tag must carry an _es token in "
                        f"{{0p2, 0p4, 1}} (got {name!r}) -- the es=0 "
                        f"column is audited-reused under the reach "
                        f"family, never rerun here")
        GATE2D_ARM_WANT = {
            "b0": {"training_style": "sft", "kl_beta": 0.0, "use_lora": 1,
                   "lora_r": 512, "sft_lr": 5e-5, "sft_epochs": 1,
                   "sft_batch_size": 4, "fresh_each_round": True,
                   "icl_k": 0, "icl_days": 0, "train_cap": 723},
            "dyn": {"training_style": "frozen", "kl_beta": 0.0,
                    "use_lora": 0, "fresh_each_round": False, "icl_k": 8,
                    "icl_days": 0, "icl_select": "random",
                    "icl_ctx_source": "live", "icl_snapshot_round": -1},
        }
        m_arm = re.search(r"_(b0|dyn)_ea", name)
        if m_arm is None:
            errs.append(f"CONFIG unknown gate2d arm token in dirname "
                        f"{name!r} (only b0/dyn exist in this family)")
        else:
            want.update(GATE2D_ARM_WANT[m_arm.group(1)])
        want["n_rounds"] = 3 if is_gate2d_smk else 30
        if re.search(r"_ea(\d+(?:p\d+)?)_", name) is None:
            errs.append(f"CONFIG gate2d tag needs a NUMERIC _ea token "
                        f"({name!r}); all-open is not part of this design")
        if "_mistral7b_" not in name:
            errs.append(f"CONFIG gate2d is mistral-only; no _mistral7b_ "
                        f"slug in {name!r}")
        # hardware provenance mandatory (GPU-dependent borderline
        # generations, 2026-08-07 finding; every gate2d run post-dates
        # the hardware patch)
        hw2 = cfg.get("hardware") or {}
        hw2_missing = [k for k in ("hostname", "gpu_name", "gpu_cc",
                                   "cuda_version", "torch_version",
                                   "transformers_version") if k not in hw2]
        if hw2_missing:
            errs.append(f"CONFIG hardware metadata missing keys "
                        f"{hw2_missing}")
        elif not (hw2.get("hostname") and hw2.get("gpu_name")):
            errs.append("CONFIG hardware metadata empty hostname/gpu_name")
    elif is_ctxgrid:
        # shared surface: movielens Action, greedy, natural labels, no
        # anchors/interventions, threshold gating EVERYWHERE. Both the AI
        # dose and the SOCIAL dose ride their universal tag tokens.
        want.update({"ml_target": "Action", "gamma_bias": 0.0,
                     "teacher_label_delta": 0.0, "pristine_frac": 0.0,
                     "replay_frac": 0.0, "do_sample": False,
                     "deploy_every": 1, "n_labeled": 723, "icrh": False,
                     "feedback_mode": "none",
                     "ai_gate_mode": "threshold"})
        if m_es is None or not any(abs(want_es - v) < 1e-9
                                   for v in (0.0, 0.2, 0.4, 1.0)):
            errs.append(f"CONFIG ctxgrid tag must carry an _es token in "
                        f"{{0, 0p2, 0p4, 1}} (got {name!r})")
        # the six adaptation channels. fz0/f32 freeze each agent's
        # context verbatim after round 0; dyn/d32 rebuild it every round.
        # K=32 is the SAME live population context as K=8, 4x deeper --
        # never the pristine/noai/donor sources of the archived k32pri /
        # k32noai runs, which is why icl_ctx_source is pinned to "live".
        CTXGRID_ARM_WANT = {
            "b0": {"training_style": "sft", "kl_beta": 0.0, "use_lora": 1,
                   "lora_r": 512, "sft_lr": 5e-5, "sft_epochs": 1,
                   "sft_batch_size": 4, "fresh_each_round": True,
                   "icl_k": 0, "icl_days": 0, "train_cap": 723},
            "k0": {"training_style": "frozen", "kl_beta": 0.0,
                   "use_lora": 0, "fresh_each_round": False, "icl_k": 0,
                   "icl_days": 0},
            "fz0": {"training_style": "frozen", "kl_beta": 0.0,
                    "use_lora": 0, "fresh_each_round": False, "icl_k": 8,
                    "icl_days": 0, "icl_select": "random",
                    "icl_ctx_source": "live", "icl_snapshot_round": 0},
            "dyn": {"training_style": "frozen", "kl_beta": 0.0,
                    "use_lora": 0, "fresh_each_round": False, "icl_k": 8,
                    "icl_days": 0, "icl_select": "random",
                    "icl_ctx_source": "live", "icl_snapshot_round": -1},
            "f32": {"training_style": "frozen", "kl_beta": 0.0,
                    "use_lora": 0, "fresh_each_round": False, "icl_k": 32,
                    "icl_days": 0, "icl_select": "random",
                    "icl_ctx_source": "live", "icl_snapshot_round": 0},
            "d32": {"training_style": "frozen", "kl_beta": 0.0,
                    "use_lora": 0, "fresh_each_round": False, "icl_k": 32,
                    "icl_days": 0, "icl_select": "random",
                    "icl_ctx_source": "live", "icl_snapshot_round": -1},
        }
        m_arm = re.search(r"_(b0|k0|fz0|dyn|f32|d32)_ea", name)
        if m_arm is None:
            errs.append(f"CONFIG unknown ctxgrid arm token in dirname "
                        f"{name!r}")
        else:
            want.update(CTXGRID_ARM_WANT[m_arm.group(1)])
        want["n_rounds"] = 3 if is_ctxgrid_smk else 30
        if re.search(r"_ea(\d+(?:p\d+)?)_", name) is None:
            errs.append(f"CONFIG ctxgrid tag needs a NUMERIC _ea token "
                        f"({name!r}); all-open is not part of this design")
        if not any(f"_{s}_" in name for s in
                   ("qwen7b", "olmo7b", "mistral7b")):
            errs.append(f"CONFIG no known ctxgrid model slug in {name!r}")
        hw3 = cfg.get("hardware") or {}
        hw3_missing = [k for k in ("hostname", "gpu_name", "gpu_cc",
                                   "cuda_version", "torch_version",
                                   "transformers_version") if k not in hw3]
        if hw3_missing:
            errs.append(f"CONFIG hardware metadata missing keys "
                        f"{hw3_missing}")
        elif not (hw3.get("hostname") and hw3.get("gpu_name")):
            errs.append("CONFIG hardware metadata empty hostname/gpu_name")
    elif is_clamp:
        # shared surface: canonical no-peer Action environment, threshold
        # gating EVERYWHERE (a numeric _ea1_ stays the strict-< gate,
        # NEVER all_open), mistral-only, matched twin mandatory. The
        # clamp dials are tag-encoded: the _strat_/_bottom_ token maps to
        # INNATE_CLAMP_MODE, the fraction is fixed at 0.2, and
        # INNATE_CLAMP_SEED equals the run seed.
        want.update({"ml_target": "Action", "gamma_bias": 0.0,
                     "teacher_label_delta": 0.0, "pristine_frac": 0.0,
                     "replay_frac": 0.0, "do_sample": False,
                     "deploy_every": 1, "n_labeled": 723, "icrh": False,
                     "feedback_mode": "none", "ai_gate_mode": "threshold",
                     "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
                     "innate_clamp_frac": 0.2})
        CLAMP_ARM_WANT = {
            "b0": {"training_style": "sft", "kl_beta": 0.0, "use_lora": 1,
                   "lora_r": 512, "sft_lr": 5e-5, "sft_epochs": 1,
                   "sft_batch_size": 4, "fresh_each_round": True,
                   "icl_k": 0, "icl_days": 0, "train_cap": 723},
            "dyn": {"training_style": "frozen", "kl_beta": 0.0,
                    "use_lora": 0, "fresh_each_round": False, "icl_k": 8,
                    "icl_days": 0, "icl_select": "random",
                    "icl_ctx_source": "live", "icl_snapshot_round": -1},
            # d8 (corrected graph wave, 2026-08-17): PERSONAL-history
            # ICL -- frozen weights, ZERO cross-user exemplars, each
            # agent's context is its own last 8 recorded opinions
            "d8": {"training_style": "frozen", "kl_beta": 0.0,
                   "use_lora": 0, "fresh_each_round": False, "icl_k": 0,
                   "icl_days": 8, "icl_select": "random",
                   "icl_ctx_source": "live", "icl_snapshot_round": -1},
        }
        CLAMP_MODE_OF_TOK = {"strat": "stratified_random",
                             "bottom": "bottom",
                             "gclump": "graph_clumped",
                             "gscat": "graph_scattered"}
        # peer grammar (2026-08-17): an optional _stub_ token after the
        # cohort selects the one-sided STUBBORN operator (fixed agents
        # keep pairing; only responsive endpoints move). There is
        # deliberately NO isolated condition. Rules by cohort:
        #   strat/bottom + _stub_    -> es must be NONZERO
        #   strat/bottom tokenless   -> the no-peer wave (es=0, no key)
        #   gclump/gscat             -> the GRAPH wave: _stub_ token
        #                               MANDATORY, es may be 0 (in-wave
        #                               baselines; the sweep is inert)
        m_arm = re.search(
            r"_(b0|dyn|d8)_(strat|bottom|gclump|gscat)(?:_(stub))?_ea",
            name)
        if m_arm is None:
            errs.append(f"CONFIG clamp tag needs an _<b0|dyn|d8>_<strat|"
                        f"bottom|gclump|gscat>[_stub]_ token run before "
                        f"_ea ({name!r})")
        else:
            want.update(CLAMP_ARM_WANT[m_arm.group(1)])
            _cl_tok = m_arm.group(2)
            want["innate_clamp_mode"] = CLAMP_MODE_OF_TOK[_cl_tok]
            _is_graph_mask = _cl_tok in ("gclump", "gscat")
            if m_arm.group(1) == "d8" and not _is_graph_mask:
                errs.append(f"CONFIG the personal-history d8 arm exists "
                            f"only in the graph-placement wave "
                            f"(gclump/gscat), got {name!r}")
            if m_arm.group(3):
                want["innate_clamp_peer_mode"] = "stubborn"
                if not _is_graph_mask and \
                        (m_es is None or want_es <= 0.0):
                    errs.append(f"CONFIG peer-clamp tag must carry a "
                                f"NONZERO _es token ({name!r}) -- the "
                                f"peer operator is meaningless at es=0")
                if _is_graph_mask and m_es is None:
                    errs.append(f"CONFIG graph-mask clamp tag must "
                                f"carry an _es token ({name!r})")
            else:
                if _is_graph_mask:
                    errs.append(f"CONFIG graph-mask clamp tags must "
                                f"carry the _stub_ token ({name!r}) -- "
                                f"the graph wave always declares the "
                                f"stubborn operator")
                if "innate_clamp_peer_mode" in cfg:
                    errs.append(f"CONFIG innate_clamp_peer_mode="
                                f"{cfg.get('innate_clamp_peer_mode')!r} "
                                f"recorded on a tokenless (no-peer) "
                                f"clamp tag {name!r}")
                if m_es is None or want_es != 0.0:
                    errs.append(f"CONFIG clamp tag without a peer token "
                                f"must carry the _es0_ token (got "
                                f"{name!r}) -- no reset-after-peer "
                                f"approximation exists")
        want["n_rounds"] = 3 if is_clamp_smk else 30
        if "_eaopen_" in name:
            errs.append(f"CONFIG clamp: all_open is not part of this "
                        f"design ({name!r}); eps_AI=1.0 rides the numeric "
                        f"threshold gate")
        elif re.search(r"_ea(\d+(?:p\d+)?)_", name) is None:
            errs.append(f"CONFIG clamp tag needs a NUMERIC _ea token "
                        f"({name!r})")
        if "_mistral7b_" not in name:
            errs.append(f"CONFIG clamp is mistral-only; no _mistral7b_ "
                        f"slug in {name!r}")
        m_sd_cl = re.search(r"_s(\d+)$", name)
        if m_sd_cl is None:
            errs.append(f"CONFIG clamp tag must end in its seed token "
                        f"({name!r})")
        else:
            # the cohort seed is the RUN seed: paired arms/gates at one
            # seed must share the bit-identical mask
            want["innate_clamp_seed"] = int(m_sd_cl.group(1))
        hw4 = cfg.get("hardware") or {}
        hw4_missing = [k for k in ("hostname", "gpu_name", "gpu_cc",
                                   "cuda_version", "torch_version",
                                   "transformers_version") if k not in hw4]
        if hw4_missing:
            errs.append(f"CONFIG hardware metadata missing keys "
                        f"{hw4_missing}")
        elif not (hw4.get("hostname") and hw4.get("gpu_name")):
            errs.append("CONFIG hardware metadata empty hostname/gpu_name")
    elif is_fam:
        # FIGURE-2 FAMILY-PRIOR SCOUT: canonical Action loop at the
        # wide-open numeric gate with a live peer step. The checkpoint
        # rides its own slug; the arm token maps to the training
        # surface; Qwen3 MUST record chat_thinking=False and every
        # other checkpoint must NOT carry the key.
        want.update({"ml_target": "Action", "gamma_bias": 0.0,
                     "teacher_label_delta": 0.0, "pristine_frac": 0.0,
                     "replay_frac": 0.0, "do_sample": False,
                     "deploy_every": 1, "n_labeled": 723, "icrh": False,
                     "feedback_mode": "none", "ai_gate_mode": "threshold",
                     "eps_ai": 1.0, "train_cap": 723,
                     "anchor_mode": "fixed",
                     "population_update": "nested_ai_then_social_v1",
                     "save_raw_gen": True})
        FAM_BASE = {
            "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
            "qwen3_8b": "Qwen/Qwen3-8B",
            "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
            "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
            "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
            "ministral8b": "mistralai/Ministral-8B-Instruct-2410",
        }
        FAM_ARM_WANT = {
            "b0": {"training_style": "sft", "kl_beta": 0.0,
                   "use_lora": 1, "lora_r": 512, "sft_lr": 5e-5,
                   "sft_epochs": 1, "sft_batch_size": 4,
                   "fresh_each_round": True, "icl_k": 0, "icl_days": 0},
            "b0p5": {"training_style": "sft_kl", "kl_beta": 0.5,
                     "kl_direction": "forward", "use_lora": 1,
                     "lora_r": 512, "sft_lr": 5e-5, "sft_epochs": 1,
                     "sft_batch_size": 4, "fresh_each_round": True,
                     "icl_k": 0, "icl_days": 0},
            "b1": {"training_style": "sft_kl", "kl_beta": 1.0,
                   "kl_direction": "forward", "use_lora": 1,
                   "lora_r": 512, "sft_lr": 5e-5, "sft_epochs": 1,
                   "sft_batch_size": 4, "fresh_each_round": True,
                   "icl_k": 0, "icl_days": 0},
            "k0": {"training_style": "frozen", "kl_beta": 0.0,
                   "use_lora": 0, "fresh_each_round": False,
                   "icl_k": 0, "icl_days": 0},
        }
        slug_f = next((s for s in FAM_BASE
                       if name.startswith(f"pofdfam_{s}_")
                       or name.startswith(f"pofdfamsmk_{s}_")), None)
        if slug_f is None:
            errs.append(f"CONFIG unknown fam checkpoint slug in {name!r}")
        else:
            want["base_model"] = FAM_BASE[slug_f]
            if slug_f == "qwen3_8b":
                want["chat_thinking"] = False
            elif "chat_thinking" in cfg:
                errs.append(f"CONFIG chat_thinking="
                            f"{cfg.get('chat_thinking')!r} recorded for "
                            f"{slug_f} -- only Qwen3 carries the "
                            f"thinking directive")
        m_arm_f = re.search(r"_(b0p5|b0|b1|k0)_ea1_", name)
        if m_arm_f is None:
            errs.append(f"CONFIG fam tag needs an "
                        f"_<b0|b0p5|b1|k0>_ea1_ token run ({name!r}) "
                        f"-- the scout runs ONLY the wide-open numeric "
                        f"gate")
        else:
            want.update(FAM_ARM_WANT[m_arm_f.group(1)])
            if is_fam_smk and m_arm_f.group(1) != "b1":
                errs.append(f"CONFIG fam smokes are b1-only ({name!r})")
        if m_es is None or want_es not in (0.05, 0.2):
            errs.append(f"CONFIG fam tag needs an _es0p05_ or _es0p2_ "
                        f"token ({name!r})")
        want["n_rounds"] = 3 if is_fam_smk else 30
        m_sd_f = re.search(r"_s(\d+)$", name)
        if m_sd_f is None or int(m_sd_f.group(1)) != 0:
            errs.append(f"CONFIG fam is a seed-0 wave; tag must end in "
                        f"_s0 ({name!r})")
        else:
            want["seed"] = 0
        hw_f = cfg.get("hardware") or {}
        hw_f_missing = [k for k in ("hostname", "gpu_name", "gpu_cc",
                                    "cuda_version", "torch_version",
                                    "transformers_version")
                        if k not in hw_f]
        if hw_f_missing:
            errs.append(f"CONFIG hardware metadata missing keys "
                        f"{hw_f_missing}")
        elif not (hw_f.get("hostname") and hw_f.get("gpu_name")):
            errs.append("CONFIG hardware metadata empty hostname/gpu_name")
    elif is_zsprior:
        # 1-round frozen zero-shot probe: nothing trains, nothing enters
        # the prompt beyond the profile, the strict EPS_AI=0 gate never
        # opens, and the raw generations are mandatory artifacts. The
        # candidate checkpoint rides its own slug; Qwen3 MUST record
        # chat_thinking=False (thinking explicitly disabled) and every
        # other checkpoint must NOT carry the key (no directive given).
        want.update({"ml_target": "Action", "gamma_bias": 0.0,
                     "teacher_label_delta": 0.0, "pristine_frac": 0.0,
                     "replay_frac": 0.0, "do_sample": False,
                     "deploy_every": 1, "n_labeled": 723, "icrh": False,
                     "feedback_mode": "none", "ai_gate_mode": "threshold",
                     "eps_ai": 0.0, "n_rounds": 1,
                     "training_style": "frozen", "kl_beta": 0.0,
                     "use_lora": 0, "fresh_each_round": False,
                     "icl_k": 0, "icl_days": 0, "save_raw_gen": True})
        ZSPRIOR_BASE = {
            "qwen3_8b": "Qwen/Qwen3-8B",
            "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
            "ministral8b": "mistralai/Ministral-8B-Instruct-2410",
            "mistralnemo": "mistralai/Mistral-Nemo-Instruct-2407",
        }
        slug_z = next((s for s in ZSPRIOR_BASE if f"_{s}_" in name), None)
        if slug_z is None:
            errs.append(f"CONFIG unknown zsprior checkpoint slug in "
                        f"{name!r}")
        else:
            want["base_model"] = ZSPRIOR_BASE[slug_z]
            if slug_z == "qwen3_8b":
                want["chat_thinking"] = False
            elif "chat_thinking" in cfg:
                errs.append(f"CONFIG chat_thinking="
                            f"{cfg.get('chat_thinking')!r} recorded for "
                            f"{slug_z} -- only the Qwen3 probe carries "
                            f"the thinking directive")
        if m_es is None or want_es != 0.0:
            errs.append(f"CONFIG zsprior tag must carry the _es0_ token "
                        f"(got {name!r})")
        hw5 = cfg.get("hardware") or {}
        hw5_missing = [k for k in ("hostname", "gpu_name", "gpu_cc",
                                   "cuda_version", "torch_version",
                                   "transformers_version") if k not in hw5]
        if hw5_missing:
            errs.append(f"CONFIG hardware metadata missing keys "
                        f"{hw5_missing}")
        elif not (hw5.get("hostname") and hw5.get("gpu_name")):
            errs.append("CONFIG hardware metadata empty hostname/gpu_name")
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
    # cube families (2026-08-05, full-parameter-cube wave): pofdw2f_/
    # pofdws2f_/pofdesf_ tags carry a _b token -- kl_beta must match it,
    # style follows it (sft iff b==0), and every trained b>0 cell is
    # forward-KL by construction (these are the forward-era families; the
    # reverse counterparts are pofdw2_/pofdws2_). b0 rows are
    # direction-free, so no direction gate there. The trailing underscore
    # keeps pofdw2fpt_/pofdws2fc_/pofdws2fsmk_ on their own branches.
    # pofdrpl_ (replay wave) shares the token grammar (_b before _ea), so it
    # rides the same gate. pofdret (no trailing underscore: covers
    # pofdretsmk_ too) is the seedcore retention family -- same grammar,
    # forward at b>0 by construction.
    if name.startswith(("pofdw2f_", "pofdws2f_", "pofdesf_", "pofdrpl_",
                        "pofdret")):
        m_b = re.search(r"_b(\d+(?:p\d+)?)_ea", name)
        if m_b is None:
            errs.append(f"CONFIG no _b token in dirname {name!r}")
        else:
            want_b = float(m_b.group(1).replace("p", "."))
            got_b = float(cfg.get("kl_beta", -1.0))
            if abs(got_b - want_b) > 1e-9:
                errs.append(f"CONFIG kl_beta={got_b!r} (tag says {want_b!r})")
            want["training_style"] = "sft" if want_b == 0 else "sft_kl"
            if want_b > 0:
                want["kl_direction"] = "forward"
    if is_rpl:
        # replay family: _rf token vs config replay_frac; labels are compared
        # to raw innate in section 1c, so the teacher shift must be OFF
        want["teacher_label_delta"] = 0.0
        m_rf = re.search(r"_rf(\d+(?:p\d+)?)_", name)
        if m_rf is None:
            errs.append(f"CONFIG no _rf token in dirname {name!r}")
        else:
            want_rf = float(m_rf.group(1).replace("p", "."))
            got_rf = float(cfg.get("replay_frac", -1.0))
            if abs(got_rf - want_rf) > 1e-9:
                errs.append(f"CONFIG replay_frac={got_rf!r} "
                            f"(tag says {want_rf!r})")
    if is_bud:
        # training-budget family: ordinary SFT at b0 with a per-round
        # optimizer-step cap (SFT_EPOCHS=0 hands control to SFT_MAX_STEPS,
        # recorded as config max_steps). _st token pins the step count, the
        # model slug pins the base checkpoint. Labels natural, movielens
        # Action only. Smoke tags (pofdbudsmk_) share the branch -- they
        # differ only in n_rounds, which is not gated here.
        want.update({"kl_beta": 0.0, "training_style": "sft",
                     "sft_epochs": 0, "ml_target": "Action",
                     "teacher_label_delta": 0.0})
        m_st = re.search(r"_st(\d+)_", name)
        if m_st is None:
            errs.append(f"CONFIG no _st token in dirname {name!r}")
        elif int(cfg.get("max_steps") or -1) != int(m_st.group(1)):
            errs.append(f"CONFIG max_steps={cfg.get('max_steps')!r} "
                        f"(tag says {m_st.group(1)})")
        BUD_BASE = {"qwen7b": "Qwen/Qwen2.5-7B-Instruct",
                    "olmo7b": "allenai/OLMo-2-1124-7B-Instruct"}
        slug = next((s for s in BUD_BASE if f"_{s}_" in name), None)
        if slug is None:
            errs.append(f"CONFIG unknown model slug in dirname {name!r}")
        elif cfg.get("base_model") != BUD_BASE[slug]:
            errs.append(f"CONFIG base_model={cfg.get('base_model')!r} "
                        f"({slug} wants {BUD_BASE[slug]!r})")
    if is_ret:
        # retention family (seedcore): exactly one round, epoch-trained,
        # natural labels, movielens Action; the _b grammar (style + forward
        # at b>0) rides the cube-prefix gate above, W/lam/es the tokens,
        # twin/peer/fresh the shared SOCIAL sections.
        want.update({"n_rounds": 1, "ml_target": "Action", "sft_epochs": 1,
                     "teacher_label_delta": 0.0})
    if name.startswith(("pofd_", "pofdw1f_")):
        # direct-transmission families (seedcore, 2026-08-07): the base
        # pofd_ wave and its forward twin pofdw1f_ carry the same _b
        # grammar as the cube families -- gate kl_beta + style against the
        # token. pofdw1f_ b>0 is forward-KL by construction; pofd_ b>0 is
        # the reverse-era wave (predates the KL_DIRECTION key), so only
        # style is gated there. Both are 30-round families.
        m_b = re.search(r"_b(\d+(?:p\d+)?)_ea", name)
        if m_b is None:
            errs.append(f"CONFIG no _b token in dirname {name!r}")
        else:
            want_b = float(m_b.group(1).replace("p", "."))
            got_b = float(cfg.get("kl_beta", -1.0))
            if abs(got_b - want_b) > 1e-9:
                errs.append(f"CONFIG kl_beta={got_b!r} (tag says {want_b!r})")
            want["training_style"] = "sft" if want_b == 0 else "sft_kl"
            if want_b > 0 and name.startswith("pofdw1f_"):
                want["kl_direction"] = "forward"
        want["n_rounds"] = 30
    if name.startswith("pofdesf_"):
        # main-peer home family: 30-round loops without exception
        want["n_rounds"] = 30
    if is_ret or name.startswith(("pofd_", "pofdw1f_", "pofdesf_")):
        # completeness: the trajectory must hold exactly config n_rounds
        # rounds (a partial pull or an over-run both fail)
        nr = int(cfg.get("n_rounds") or 0)
        if len(traj) != nr or int(op_raw.shape[0]) != nr:
            errs.append(f"CONFIG trajectory holds {len(traj)} rows / "
                        f"op_raw {tuple(op_raw.shape)} (config n_rounds="
                        f"{nr} -- incomplete or over-run)")
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
        # counterfactual preference-flip telemetry (2026-08-12, dpo CI waves
        # onward): the runner marks dpo_flip_telemetry in config; every round
        # row must then carry the pair/flip keys (a 0-pair round keeps the
        # keys, frac None). Older DPO runs lack the marker and skip this gate.
        if cfg.get("dpo_flip_telemetry"):
            for r_ in traj:
                miss = [k for k in ("dpo_pairs", "dpo_ties", "dpo_parse_fail",
                                    "dpo_flip_n", "dpo_flip_frac",
                                    "dpo_judge_disp_mean") if k not in r_]
                if miss:
                    errs.append(f"TRAJ round {r_.get('round')} missing dpo "
                                f"flip telemetry keys {miss}")
                    break
                ff = r_["dpo_flip_frac"]
                if ff is None and int(r_["dpo_pairs"]) > 0:
                    errs.append(f"TRAJ round {r_.get('round')} dpo_flip_frac "
                                f"None with dpo_pairs>0")
                    break
                if ff is not None and not 0.0 <= float(ff) <= 1.0:
                    errs.append(f"TRAJ round {r_.get('round')} "
                                f"dpo_flip_frac={ff!r} outside [0,1]")
                    break
        # matched-randomness arms (pofddpomr_, 2026-08-13): _bk token vs
        # dpo_bank_seed, matched marker, and the closed=write / open=read
        # pairing. The PAIRED invariants (bank bit-identity, round-0 fork,
        # twin==judge) live in check_dpo_pair.py, run per pair.
        if "_bk" in name:
            m_bk = re.search(r"_bk(\d+)_", name)
            if m_bk is None:
                errs.append(f"CONFIG malformed _bk token in dirname {name!r}")
            elif int(cfg.get("dpo_bank_seed", -1)) != int(m_bk.group(1)):
                errs.append(f"CONFIG dpo_bank_seed="
                            f"{cfg.get('dpo_bank_seed')!r} "
                            f"(tag says {m_bk.group(1)})")
            if not cfg.get("dpo_matched"):
                errs.append("CONFIG _bk tag without the dpo_matched marker")
            want_mode = {"closed": "write", "open": "read"}.get(fb)
            if want_mode and cfg.get("dpo_bank_mode") != want_mode:
                errs.append(f"CONFIG dpo_bank_mode="
                            f"{cfg.get('dpo_bank_mode')!r} "
                            f"({fb} arm wants {want_mode!r})")
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
    # universal tag-token gates (2026-08-07, seedcore wave -- previously
    # only the tch/tfe/rpl/bud/iclf/ctf families checked the seed): every
    # family tag-encodes the seed, the eps_AI dose and the model slug, and
    # the configs are queue-generated from the tags, so all three must
    # match on every run. Validated against the full local corpus (895
    # pulled run dirs) before adoption.
    m_s = re.search(r"_s(\d+)(?:_|$)", name)
    if m_s and cfg.get("seed") != int(m_s.group(1)):
        errs.append(f"CONFIG seed={cfg.get('seed')!r} "
                    f"(tag says {m_s.group(1)})")
    m_ea_t = re.search(r"_ea(\d+(?:p\d+)?)_", name)
    if m_ea_t and abs(float(cfg.get("eps_ai", -1.0))
                      - float(m_ea_t.group(1).replace("p", "."))) > 1e-9:
        errs.append(f"CONFIG eps_ai={cfg.get('eps_ai')!r} "
                    f"(tag says {m_ea_t.group(1)})")
    for _slug, _bm in SLUG_BASE.items():
        if f"_{_slug}_" in name and cfg.get("base_model") != _bm:
            errs.append(f"CONFIG base_model={cfg.get('base_model')!r} "
                        f"({_slug} wants {_bm!r})")
    eps_ai = float(cfg["eps_ai"])
    # AI gate mode (2026-08-13): absent in older configs -> "threshold",
    # which reproduces the pre-mode inline gate expression byte-for-byte.
    # Every gate replay below goes through the SHARED gp.ai_gate definition.
    gate_mode = cfg.get("ai_gate_mode") or "threshold"

    # -- 1b GATE-MASK (gate_raw, runner 2026-08-05) --------------------------
    # When present, the saved per-agent AI gate/contact mask must be exactly
    # the gate on the start-of-round opinion: shape == op_raw, per-round mean
    # == row['contact'], and bit-equal to |served - x(t)| < eps_ai (canary is
    # config-gated to 0 in every family here; ab_retain serves the retained
    # winner instead, so it is exempt from the replay). Absent / empty ->
    # older run, skipped -- the mask reconstructs offline the same way.
    gr = d.get("gate_raw")
    if gr is not None and gr.numel() > 0 and not cfg.get("ab_retain"):
        if tuple(gr.shape) != tuple(op_raw.shape):
            errs.append(f"GATE gate_raw shape {tuple(gr.shape)} != "
                        f"op_raw {tuple(op_raw.shape)}")
        else:
            gr = gr.bool()
            for t in range(op_raw.shape[0]):
                x0 = innate if t == 0 else op_raw[t - 1]
                served = pred_raw[t].clamp(0.0, 1.0)
                expect_g = gp.ai_gate(served, x0, eps_ai, gate_mode)
                if not torch.equal(gr[t], expect_g):
                    errs.append(f"GATE round {t}: gate_raw != gate on the "
                                f"start-of-round opinion "
                                f"({int((gr[t] ^ expect_g).sum())} agents off)")
                logged = traj[t].get("contact")
                if logged is not None and \
                        abs(logged - float(gr[t].float().mean())) > 1e-6:
                    errs.append(f"GATE round {t}: contact {logged:.6f} != "
                                f"mean(gate_raw) "
                                f"{float(gr[t].float().mean()):.6f}")

    # -- 1c REPLAY (replay_raw/train_y_raw, runner 2026-08-06) ---------------
    # replay_frac>0 (exact initial-data replay, replace regime): the saved
    # label-source mask and the ACTUAL labels fed to the learner must show a
    # fixed-size batch of exactly round(rf*n) ORIGINAL round-0 labels (raw
    # innate; the teacher shift is config-gated to 0) and n - round(rf*n)
    # labels from the LATEST round only (op_raw[t-1] -- never accumulated
    # history). Round 0 is all-original by definition. Bit-exact: the runner
    # saves the very tensors it trains on.
    rr = d.get("replay_raw")
    ty = d.get("train_y_raw")
    rf = float(cfg.get("replay_frac") or 0.0)
    if rf > 0:
        n_lab = int(cfg.get("train_cap") or 0) or int(op_raw.shape[1])
        dep_rounds = [r["round"] for r in traj if r.get("is_deploy")]
        if rr is None or rr.numel() == 0 or ty is None or ty.numel() == 0:
            errs.append("REPLAY replay_frac>0 but replay_raw/train_y_raw "
                        "missing from trajectory.pt")
        elif (tuple(rr.shape) != (len(dep_rounds), n_lab)
              or tuple(ty.shape) != (len(dep_rounds), n_lab)):
            errs.append(f"REPLAY shapes replay_raw={tuple(rr.shape)} "
                        f"train_y_raw={tuple(ty.shape)} (want "
                        f"{(len(dep_rounds), n_lab)!r} -- one fixed-size "
                        f"batch per deploy round)")
        else:
            rr = rr.bool()
            ty = ty.float()
            innate_lab = innate[:n_lab]
            for di, t in enumerate(dep_rounds):
                n_rep = int(rr[di].sum())
                want_rep = n_lab if t == 0 else int(round(rf * n_lab))
                if n_rep != want_rep:
                    errs.append(f"REPLAY round {t}: {n_rep} original-label "
                                f"rows (want exactly {want_rep})")
                want_y = (innate_lab if t == 0 else
                          torch.where(rr[di], innate_lab,
                                      op_raw[t - 1][:n_lab]))
                if not torch.equal(ty[di], want_y):
                    errs.append(f"REPLAY round {t}: trained labels differ "
                                f"from where(mask, round-0, latest) by max "
                                f"{float((ty[di] - want_y).abs().max()):.2e}")
                logged = traj[t].get("n_replay")
                if logged is not None and int(logged) != n_rep:
                    errs.append(f"REPLAY round {t}: row n_replay={logged} != "
                                f"mask sum {n_rep}")
    elif rr is not None and rr.numel() > 0:
        errs.append("REPLAY replay_raw present but config replay_frac is "
                    "0/absent")

    # -- 1d ICL-CTX (icl_idx_raw/icl_val_raw + icl_ctx_log.json.gz, runner
    # 2026-08-07) ------------------------------------------------------------
    # ICL_K>0 runs log the exact exemplar (agent-id, displayed-value) pairs
    # each agent's context carried, plus the rendered text blocks. Checks:
    # no agent in its own context; with ICL_SNAPSHOT_ROUND=R the (ids, vals)
    # AND the rendered text never change after round R; iclf runs must carry
    # constant fixed-text perplexity (weights frozen), gg telemetry, the
    # matched twin, and a freeze-8 run must be bit-identical to its _dyn_
    # sibling through round R when the sibling exists on disk.
    ii = d.get("icl_idx_raw")
    iv = d.get("icl_val_raw")
    icl_k_cfg = int(cfg.get("icl_k") or 0)
    snap = cfg.get("icl_snapshot_round")
    snap = -1 if snap is None else int(snap)
    if is_ctx_fam and icl_k_cfg > 0 and (ii is None or ii.numel() == 0):
        errs.append("ICL-CTX icl_k>0 but icl_idx_raw missing/empty (runner "
                    "predates the snapshot patch?)")
    if ii is not None and ii.numel() > 0:
        n_ag = op_raw.shape[1]
        n_dep = len([r for r in traj if r.get("is_deploy")])
        if (iv is None or tuple(ii.shape) != (n_dep, n_ag, icl_k_cfg)
                or tuple(iv.shape) != tuple(ii.shape)):
            errs.append(f"ICL-CTX shapes idx={tuple(ii.shape)} "
                        f"val={None if iv is None else tuple(iv.shape)} "
                        f"(want {(n_dep, n_ag, icl_k_cfg)!r})")
        else:
            ii = ii.long()
            iv = iv.float()
            own = torch.arange(n_ag).view(1, n_ag, 1)
            if bool((ii == own).any()):
                bad_t = int((ii == own).flatten(1).any(dim=1).nonzero()[0])
                errs.append(f"ICL-CTX round {bad_t}: an agent appears in its "
                            f"OWN context (self-exclusion violated)")
            if (not torch.isfinite(iv).all() or float(iv.min()) < -1e-6
                    or float(iv.max()) > 1 + 1e-6):
                errs.append("ICL-CTX displayed values non-finite or out of "
                            "[0,1]")
            if snap >= 0:
                if snap >= ii.shape[0]:
                    errs.append(f"ICL-CTX snapshot round {snap} was never "
                                f"reached ({ii.shape[0]} logged rounds)")
                else:
                    for tt in range(snap + 1, ii.shape[0]):
                        if not (torch.equal(ii[tt], ii[snap])
                                and torch.equal(iv[tt], iv[snap])):
                            errs.append(f"ICL-CTX round {tt}: cached context "
                                        f"CHANGED after snapshot round {snap}")
                            break
            gzp = os.path.join(run_dir, "icl_ctx_log.json.gz")
            if os.path.exists(gzp):
                rows_ctx = [json.loads(l) for l in gzip.open(gzp, "rt")]
                if (len(rows_ctx) != ii.shape[0]
                        or any(len(r["ctx"]) != n_ag for r in rows_ctx)):
                    errs.append("ICL-CTX icl_ctx_log rows do not match "
                                "icl_idx_raw shape")
                elif snap >= 0 and snap < len(rows_ctx):
                    ref = rows_ctx[snap]["ctx"]
                    for r in rows_ctx[snap + 1:]:
                        if r["ctx"] != ref:
                            errs.append(f"ICL-CTX round {r['round']}: rendered "
                                        f"context TEXT changed after snapshot")
                            break
            elif is_ctx_fam:
                errs.append("ICL-CTX icl_ctx_log.json.gz missing")
    if is_ctx_fam:
        ppls = [r["perplexity"] for r in traj if "perplexity" in r]
        if ppls and len(set(ppls)) != 1:
            errs.append(f"ICL-CTX fixed-text perplexity varies across rounds "
                        f"({len(set(ppls))} distinct values) -- weights not "
                        f"frozen?")
        need_gg = ["gg_pred_true", "gg_op_true", "gg_r2_inc_true"]
        bad = [r["round"] for r in traj if any(k not in r for k in need_gg)]
        if bad:
            errs.append(f"ICL-CTX gg telemetry missing in rounds {bad[:5]}")
        tw = d.get("twin_raw")
        if tw is None or tw.numel() == 0 or tuple(tw.shape) != tuple(op_raw.shape):
            errs.append("ICL-CTX twin_raw missing/short (WITH_TWIN=1 is "
                        "mandatory in this family)")
        elif any("gg_twin_true" not in r for r in traj):
            errs.append("ICL-CTX gg_twin_true missing from trajectory rows")
        if "_fz8_" in name and snap >= 0:
            sib = os.path.join(os.path.dirname(run_dir.rstrip("/")) or ".",
                               name.replace("_fz8_", "_dyn_"))
            sib_pt = os.path.join(sib, "trajectory.pt")
            if os.path.exists(sib_pt):
                ds = torch.load(sib_pt, map_location="cpu", weights_only=False)
                hi = snap + 1
                # icl_idx_raw is pure RNG (no model in the loop): divergence
                # there is a CODE bug -> hard error. The generated VALUES
                # (pred_raw -> op_raw -> icl_val_raw) are bit-identical only
                # when both jobs ran on the same GPU model -- across
                # heterogeneous hardware, fp-kernel differences flip
                # borderline generations from round 0 (observed 2026-08-07:
                # g204-vs-g145 pairs diverge, g146-vs-g128 pairs are
                # bit-equal). Value divergence is therefore reported as a
                # WARN with stats, not a failure.
                for key in ("icl_idx_raw",):
                    a, b = d.get(key), ds.get(key)
                    if (a is None or b is None or a.numel() == 0
                            or b.numel() == 0 or a.shape[0] < hi
                            or b.shape[0] < hi):
                        errs.append(f"ICL-CTX dyn-twin prefix: {key} "
                                    f"missing/short in one of the pair")
                    elif not _bit_eq(a[:hi].float(), b[:hi].float()):
                        errs.append(f"ICL-CTX {key} differs from the _dyn_ "
                                    f"twin inside rounds 0..{snap} (the "
                                    f"selection stream is model-free -- "
                                    f"this is an RNG/code bug)")
                for key in ("pred_raw", "op_raw", "icl_val_raw"):
                    a, b = d.get(key), ds.get(key)
                    if (a is None or b is None or a.numel() == 0
                            or b.numel() == 0 or a.shape[0] < hi
                            or b.shape[0] < hi):
                        continue
                    a, b = a[:hi].float(), b[:hi].float()
                    if not _bit_eq(a, b):
                        neq = ~((a == b) | (torch.isnan(a) & torch.isnan(b)))
                        t0 = int(neq.reshape(hi, -1).any(dim=1).nonzero()[0])
                        dd = (a - b).abs()
                        mx = float(dd[neq & torch.isfinite(dd)].max()) \
                            if bool((neq & torch.isfinite(dd)).any()) else 0.0
                        print(f"WARN {name}: {key} differs from the _dyn_ "
                              f"twin from round {t0} "
                              f"({int(neq.reshape(hi, -1)[t0].sum())} entries"
                              f", max |diff| {mx:.4f}) -- expected across "
                              f"heterogeneous GPUs; selection stream is "
                              f"bit-identical")

    # -- 1e CTF-DONOR (icl_ctx_source=donor, runner 2026-08-07) --------------
    # Context-transfer provenance: the saved donor vector must hash to the
    # config hash, the displayed context values must equal donor[ids] EXACTLY
    # in every round, a pri arm's donor vector must BE the recipient innate,
    # the vector must re-derive bit-exactly from the donor run when it is
    # pulled alongside, matched arms must share exemplar identities, and the
    # donor gg / realized-context-gap diagnostics must be present.
    if cfg.get("icl_ctx_source") == "donor":
        dv = d.get("icl_donor_vec")
        if dv is None or dv.numel() == 0:
            errs.append("CTF icl_donor_vec missing from trajectory.pt")
        else:
            dv = dv.float()
            h = hashlib.sha256(
                dv.contiguous().numpy().tobytes()).hexdigest()
            if h != cfg.get("icl_ctx_donor_hash"):
                errs.append("CTF donor vector sha256 != config "
                            "icl_ctx_donor_hash")
            d_rnd = cfg.get("icl_ctx_donor_round")
            d_rnd = -1 if d_rnd is None else int(d_rnd)
            if d_rnd < 0 and not torch.equal(dv, innate):
                errs.append("CTF pri arm: donor vector != recipient innate "
                            "(round -1 must display pristine round-0 "
                            "opinions)")
            ii_c = d.get("icl_idx_raw")
            iv_c = d.get("icl_val_raw")
            if (ii_c is not None and ii_c.numel() > 0 and iv_c is not None
                    and iv_c.shape == ii_c.shape):
                want_v = dv[ii_c.long()]
                if not torch.equal(iv_c.float(), want_v):
                    errs.append("CTF displayed context values != donor "
                                "vector at the logged exemplar ids")
            # provenance re-derivation from the donor run, when pulled
            parent = os.path.dirname(run_dir.rstrip("/")) or "."
            d_tag = cfg.get("icl_ctx_donor_tag") or ""
            d_pt = os.path.join(parent, d_tag, "trajectory.pt")
            if d_tag and os.path.exists(d_pt):
                dd = torch.load(d_pt, map_location="cpu", weights_only=False)
                src = (dd["innate"] if d_rnd < 0 else dd["op_raw"][d_rnd])
                if not torch.equal(dv, src.float()):
                    errs.append(f"CTF donor vector != {d_tag}"
                                f"{' innate' if d_rnd < 0 else f' op_raw[{d_rnd}]'}"
                                f" (provenance re-derivation failed)")
            # matched arms share exemplar identities (values-only manipulation)
            m_arm_c = re.search(r"_(pri|b0r9|b1r9)_ea", name)
            if m_arm_c and ii_c is not None and ii_c.numel() > 0:
                for other in ("pri", "b0r9", "b1r9"):
                    if other == m_arm_c.group(1):
                        continue
                    sib_pt = os.path.join(
                        parent, name.replace(f"_{m_arm_c.group(1)}_",
                                             f"_{other}_"), "trajectory.pt")
                    if os.path.exists(sib_pt):
                        ds = torch.load(sib_pt, map_location="cpu",
                                        weights_only=False)
                        ii_s = ds.get("icl_idx_raw")
                        if (ii_s is None or ii_s.numel() == 0
                                or not torch.equal(ii_c.long(), ii_s.long())):
                            errs.append(f"CTF exemplar identities differ "
                                        f"from the _{other}_ arm (matched "
                                        f"arms must differ ONLY in "
                                        f"displayed values)")
        if (d.get("icl_donor_gg") is None or d.get("icl_donor_r2_inc") is None
                or d.get("icl_ctx_gg") is None):
            errs.append("CTF donor gg / r2 / realized-context-gap "
                        "diagnostics missing")
        elif any("gg_ctx_true" not in r for r in traj):
            errs.append("CTF gg_ctx_true missing from trajectory rows")
    elif d.get("icl_donor_vec") is not None and d["icl_donor_vec"].numel():
        errs.append("CTF icl_donor_vec present but icl_ctx_source != donor")

    # -- 1f REACH (sft_icl_reach wave, 2026-08-13) ---------------------------
    # Mandatory artifacts + invariants for every pofdreach* run. The gate
    # bit-replay itself rides section 1b (gate_raw is REQUIRED here, so 1b
    # always fires) through the SHARED gp.ai_gate; the exact per-agent
    # opinion replay rides section 3. Nothing here encodes an expected
    # scientific outcome -- no arm is required to recruit anyone.
    if is_reach:
        # completeness: exactly config n_rounds rounds
        nr_r = int(cfg.get("n_rounds") or 0)
        if len(traj) != nr_r or int(op_raw.shape[0]) != nr_r:
            errs.append(f"REACH trajectory holds {len(traj)} rows / op_raw "
                        f"{tuple(op_raw.shape)} (config n_rounds={nr_r} -- "
                        f"incomplete or over-run)")
        # matched no-platform twin: with no peers and an initially innate
        # twin, x_cf(t+1) = lam*innate + (1-lam)*x_cf(t) has innate as its
        # fixed point -- but float32 rounding of lam*v + (1-lam)*v sits up
        # to ONE ULP off v (max 5.96e-8, verified on the archive), so the
        # spec's "bit-exact" is enforced as <= 1 float32 ulp PLUS bit-
        # stability from round 1 on (the iterate is bit-stationary).
        tw_r = d.get("twin_raw")
        if tw_r is None or tw_r.numel() == 0:
            errs.append("REACH twin_raw missing/empty (WITH_TWIN=1 is "
                        "mandatory in this family)")
        elif tuple(tw_r.shape) != tuple(op_raw.shape):
            errs.append(f"REACH twin_raw shape {tuple(tw_r.shape)} != "
                        f"op_raw {tuple(op_raw.shape)}")
        else:
            tw_r = tw_r.float()
            dtw = float((tw_r - innate).abs().max())
            if dtw > 6.0e-8:
                errs.append(f"REACH twin drifts off innate (max |diff| "
                            f"{dtw:.2e} > 1 float32 ulp; no-peer twin must "
                            f"stay innate)")
            for tt in range(2, tw_r.shape[0]):
                if not torch.equal(tw_r[tt], tw_r[1]):
                    errs.append(f"REACH twin_raw round {tt} not bit-stable "
                                f"(differs from round 1)")
                    break
        if any("twin_mean" not in r for r in traj):
            errs.append("REACH twin telemetry missing from trajectory rows")
        # gate mask: REQUIRED (never skip-if-absent in this family)
        gr_r = d.get("gate_raw")
        if gr_r is None or gr_r.numel() == 0:
            errs.append("REACH gate_raw missing/empty (mandatory in this "
                        "family)")
        elif tuple(gr_r.shape) != tuple(op_raw.shape):
            errs.append(f"REACH gate_raw shape {tuple(gr_r.shape)} != "
                        f"op_raw {tuple(op_raw.shape)}")
        else:
            gr_r = gr_r.bool()
            if gate_mode == "all_open":
                if not bool(gr_r.all()):
                    bad_t = int((~gr_r).flatten(1).any(dim=1)
                                .nonzero()[0])
                    errs.append(f"REACH all_open: gate_raw has False bits "
                                f"(first at round {bad_t})")
                bad_c = [r["round"] for r in traj
                         if r.get("contact") != 1.0]
                if bad_c:
                    errs.append(f"REACH all_open: contact != 1.0 exactly "
                                f"in rounds {bad_c[:5]}")
            _static_arm = ("fz0" if "_fz0_" in name
                           else "k0" if "_k0_" in name else None)
            if _static_arm is not None and gate_mode == "threshold":
                # STATIC-SERVING arms -- fz0 (frozen round-0 context) and
                # k0 (frozen no-context prompting): frozen weights + fixed
                # (or absent) context + greedy serving means the gate mask
                # and the served predictions must be constant (an
                # initially rejected agent never enters, entrants/exits
                # zero). ONE exception, observed on the s0 slab (mistral
                # fz0 ea0p05, agent at |m-innate| exactly 1.2e-8 outside
                # the strict gate): the lam-anchor rounds x by 1 float32
                # ulp, which can flip an agent sitting within ~1 ulp of
                # the boundary. The per-round bit-replay (1b) still
                # verifies every mask exactly, so here we only require
                # that any flip is BOUNDARY-EXPLAINED: the flipped
                # agent's round-0 margin ||m - innate| - eps| must be
                # within 4 float32 ulp (2.4e-7). Real context/serving
                # drift moves gates by far more and still fails.
                served0 = pred_raw[0].clamp(0.0, 1.0)
                margin0 = ((served0 - innate).abs() - eps_ai).abs()
                for tt in range(1, gr_r.shape[0]):
                    flipped = gr_r[tt] ^ gr_r[0]
                    if not bool(flipped.any()):
                        continue
                    off = margin0[flipped]
                    if float(off.max()) > 2.4e-7:
                        errs.append(
                            f"REACH {_static_arm}: gate_raw round {tt} != "
                            f"round 0 ({int(flipped.sum())} agents "
                            f"flipped, max round-0 boundary margin "
                            f"{float(off.max()):.2e} > 4 ulp -- static "
                            f"serving must freeze the gate set)")
                        break
            if _static_arm is not None:
                for tt in range(1, pred_raw.shape[0]):
                    if not _bit_eq(pred_raw[tt], pred_raw[0]):
                        neq = ~((pred_raw[tt] == pred_raw[0])
                                | (torch.isnan(pred_raw[tt])
                                   & torch.isnan(pred_raw[0])))
                        errs.append(f"REACH {_static_arm}: pred_raw round "
                                    f"{tt} != round 0 ({int(neq.sum())} "
                                    f"agents, max |diff| "
                                    f"{float((pred_raw[tt] - pred_raw[0])[neq].abs().max()):.2e}) "
                                    f"-- deterministic static serving must "
                                    f"be constant within a run")
                        break
        # finite, in-range opinions AND predictions (parse-fail NaN is not
        # acceptable in this family -- it would silently shrink the gate)
        if not torch.isfinite(op_raw).all():
            errs.append("REACH non-finite opinions")
        elif float(op_raw.min()) < -1e-6 or float(op_raw.max()) > 1 + 1e-6:
            errs.append(f"REACH opinions outside [0,1]: "
                        f"[{float(op_raw.min()):.4f}, {float(op_raw.max()):.4f}]")
        if not torch.isfinite(pred_raw).all():
            errs.append("REACH non-finite predictions (parse-fail NaN)")
        elif float(pred_raw.min()) < -1e-6 or float(pred_raw.max()) > 1 + 1e-6:
            errs.append(f"REACH predictions outside [0,1]: "
                        f"[{float(pred_raw.min()):.4f}, "
                        f"{float(pred_raw.max()):.4f}]")
        # ICL arms: exemplar artifacts + rendered log + constant fixed-text
        # perplexity (weights frozen). Structural context checks (self-
        # exclusion, snapshot immutability on ids/vals AND text) ride the
        # generic section 1d above.
        if int(cfg.get("icl_k") or 0) > 0:
            ii_r = d.get("icl_idx_raw")
            if ii_r is None or ii_r.numel() == 0:
                errs.append("REACH icl_k>0 but icl_idx_raw missing/empty")
            elif "_dyn_" in name and ii_r.shape[0] > 1:
                # refreshed-context provenance (grid3 wave, 2026-08-14): a
                # dyn arm REBUILDS its context from the live population
                # every round -- 8-of-722 random draws per agent cannot
                # repeat identically, so bit-identical exemplar ids across
                # every round mean the refresh never ran. Structural, not
                # scientific: no direction/amount of context CHANGE in
                # values is required, only that selection re-drew.
                if all(torch.equal(ii_r[tt], ii_r[0])
                       for tt in range(1, ii_r.shape[0])):
                    errs.append("REACH dyn: exemplar ids bit-identical "
                                "across all rounds -- context was never "
                                "refreshed")
            if not os.path.exists(os.path.join(run_dir,
                                               "icl_ctx_log.json.gz")):
                errs.append("REACH icl_ctx_log.json.gz missing")
            ppls_r = [r["perplexity"] for r in traj if "perplexity" in r]
            if ppls_r and len(set(ppls_r)) != 1:
                errs.append(f"REACH fixed-text perplexity varies across "
                            f"rounds ({len(set(ppls_r))} distinct) -- "
                            f"weights not frozen?")
            # matched fixed/dynamic pair: round-0 exemplar identities AND
            # displayed values must be bit-identical when the sibling run
            # is pulled alongside. Both are CPU-derived (RNG selection +
            # innate round-0 labels), so this holds across heterogeneous
            # GPUs -- never weakened for hardware.
            for a_tok, b_tok in (("_fz0_", "_dyn_"), ("_dyn_", "_fz0_")):
                if a_tok in name:
                    sib = os.path.join(
                        os.path.dirname(run_dir.rstrip("/")) or ".",
                        name.replace(a_tok, b_tok), "trajectory.pt")
                    if os.path.exists(sib):
                        ds_r = torch.load(sib, map_location="cpu",
                                          weights_only=False)
                        for key in ("icl_idx_raw", "icl_val_raw"):
                            a_v, b_v = d.get(key), ds_r.get(key)
                            if (a_v is None or b_v is None
                                    or a_v.numel() == 0 or b_v.numel() == 0):
                                errs.append(f"REACH matched pair: {key} "
                                            f"missing in one of fz0/dyn")
                            elif not torch.equal(a_v[0].float(),
                                                 b_v[0].float()):
                                errs.append(f"REACH matched pair: round-0 "
                                            f"{key} differs from the "
                                            f"{b_tok.strip('_')} sibling "
                                            f"(fixed and dynamic arms must "
                                            f"share the round-0 context)")
                    break
        # k0 (frozen no-context prompting): population information must
        # have NO path into the prompt -- no exemplar ids/values, no
        # rendered context log -- and the fixed-text perplexity must be
        # constant every round (weights frozen, no LoRA/optimizer).
        if "_k0_" in name and not is_reach_base:
            for key in ("icl_idx_raw", "icl_val_raw"):
                t_k = d.get(key)
                if t_k is not None and t_k.numel() > 0:
                    errs.append(f"REACH k0: {key} non-empty -- no-context "
                                f"prompting must carry no exemplars")
            if os.path.exists(os.path.join(run_dir, "icl_ctx_log.json.gz")):
                errs.append("REACH k0: icl_ctx_log.json.gz present -- no "
                            "rendered context allowed")
            ppls_k = [r["perplexity"] for r in traj if "perplexity" in r]
            if ppls_k and len(set(ppls_k)) != 1:
                errs.append(f"REACH k0: fixed-text perplexity varies "
                            f"({len(set(ppls_k))} distinct values) -- "
                            f"weights not frozen?")
        # baseline <-> arm innate identity when the matched baseline probe
        # is pulled alongside (the analysis re-verifies this globally)
        if not is_reach_base:
            m_slug_r = re.search(r"_(qwen7b|olmo7b|mistral7b)_", name)
            m_seed_r = re.search(r"_s(\d+)(?:_|$)", name)
            if m_slug_r and m_seed_r:
                base_dir = os.path.join(
                    os.path.dirname(run_dir.rstrip("/")) or ".",
                    f"pofdreachbase_{m_slug_r.group(1)}_w0p5_l0p2_es0"
                    f"_s{m_seed_r.group(1)}", "trajectory.pt")
                if os.path.exists(base_dir):
                    db_r = torch.load(base_dir, map_location="cpu",
                                      weights_only=False)
                    if not torch.equal(db_r["innate"].float(), innate):
                        errs.append("REACH innate vector differs from the "
                                    "matched baseline probe's (population "
                                    "mismatch)")

    # -- 1g PEER2 (sft_icl_peer02 wave, 2026-08-14) --------------------------
    # Mandatory artifacts + invariants for pofdpeer2* runs. The gate
    # bit-replay rides section 1b (gate_raw REQUIRED here, so 1b always
    # fires); peer-alive / twin / mean-conservation / contact ride the
    # SOCIAL section (es=0.2). Static-serving arms (k0/fz0) must keep
    # their PREDICTIONS bit-constant -- but NOT their gate masks: with
    # peers live, agents move across a fixed prediction's gate line by
    # peer dynamics alone. No scientific outcome is a validity condition.
    if is_peer2:
        nr_p = int(cfg.get("n_rounds") or 0)
        if len(traj) != nr_p or int(op_raw.shape[0]) != nr_p:
            errs.append(f"PEER2 trajectory holds {len(traj)} rows / "
                        f"op_raw {tuple(op_raw.shape)} (config n_rounds="
                        f"{nr_p} -- incomplete or over-run)")
        if pred_raw.shape != op_raw.shape:
            errs.append(f"PEER2 pred_raw shape {tuple(pred_raw.shape)} != "
                        f"op_raw {tuple(op_raw.shape)}")
        gr_p = d.get("gate_raw")
        if gr_p is None or gr_p.numel() == 0:
            errs.append("PEER2 gate_raw missing/empty (mandatory in this "
                        "family)")
        elif tuple(gr_p.shape) != tuple(op_raw.shape):
            errs.append(f"PEER2 gate_raw shape {tuple(gr_p.shape)} != "
                        f"op_raw {tuple(op_raw.shape)}")
        if torch.isfinite(pred_raw).all():
            if float(pred_raw.min()) < -1e-6 or \
                    float(pred_raw.max()) > 1 + 1e-6:
                errs.append(f"PEER2 predictions outside [0,1]: "
                            f"[{float(pred_raw.min()):.4f}, "
                            f"{float(pred_raw.max()):.4f}]")
        else:
            errs.append("PEER2 non-finite predictions (parse-fail NaN)")
        _p2_static = ("k0" if "_k0_" in name
                      else "fz0" if "_fz0_" in name else None)
        if _p2_static is not None:
            for tt in range(1, pred_raw.shape[0]):
                if not _bit_eq(pred_raw[tt], pred_raw[0]):
                    neq = ~((pred_raw[tt] == pred_raw[0])
                            | (torch.isnan(pred_raw[tt])
                               & torch.isnan(pred_raw[0])))
                    errs.append(f"PEER2 {_p2_static}: pred_raw round {tt} "
                                f"!= round 0 ({int(neq.sum())} agents) -- "
                                f"static serving must be constant within "
                                f"a run (gate masks MAY churn via peers)")
                    break
        if "_k0_" in name:
            for key in ("icl_idx_raw", "icl_val_raw"):
                t_k = d.get(key)
                if t_k is not None and t_k.numel() > 0:
                    errs.append(f"PEER2 k0: {key} non-empty -- plain "
                                f"prompting must carry no exemplars")
            if os.path.exists(os.path.join(run_dir,
                                           "icl_ctx_log.json.gz")):
                errs.append("PEER2 k0: icl_ctx_log.json.gz present -- no "
                            "rendered context allowed")
        if int(cfg.get("icl_k") or 0) > 0:
            ii_p = d.get("icl_idx_raw")
            if ii_p is None or ii_p.numel() == 0:
                errs.append("PEER2 icl_k>0 but icl_idx_raw missing/empty")
            elif "_dyn_" in name and ii_p.shape[0] > 1:
                if all(torch.equal(ii_p[tt], ii_p[0])
                       for tt in range(1, ii_p.shape[0])):
                    errs.append("PEER2 dyn: exemplar ids bit-identical "
                                "across all rounds -- context was never "
                                "refreshed")
            if not os.path.exists(os.path.join(run_dir,
                                               "icl_ctx_log.json.gz")):
                errs.append("PEER2 icl_ctx_log.json.gz missing")
        if cfg.get("training_style") == "frozen":
            ppls_p = [r["perplexity"] for r in traj if "perplexity" in r]
            if ppls_p and len(set(ppls_p)) != 1:
                errs.append(f"PEER2 fixed-text perplexity varies "
                            f"({len(set(ppls_p))} distinct values) -- "
                            f"weights not frozen?")

    # -- 1h GATE2D (sft_icl_gate2d wave, 2026-08-15) -------------------------
    # Mandatory artifacts + invariants for pofdgate2d* runs. The numeric
    # AI-gate bit-replay rides section 1b (gate_raw REQUIRED here, so 1b
    # always fires) through the SHARED gp.ai_gate; peer-alive / twin /
    # mean-conservation / contact ride the SOCIAL section (es>0 in every
    # cell of this family). Neither arm serves statically -- b0 retrains
    # each round, dyn rebuilds its context -- so no constancy gate
    # applies. No scientific outcome is a validity condition.
    if is_gate2d:
        nr_g2 = int(cfg.get("n_rounds") or 0)
        if len(traj) != nr_g2 or int(op_raw.shape[0]) != nr_g2:
            errs.append(f"GATE2D trajectory holds {len(traj)} rows / "
                        f"op_raw {tuple(op_raw.shape)} (config n_rounds="
                        f"{nr_g2} -- incomplete or over-run)")
        if pred_raw.shape != op_raw.shape:
            errs.append(f"GATE2D pred_raw shape {tuple(pred_raw.shape)} "
                        f"!= op_raw {tuple(op_raw.shape)}")
        gr_g2 = d.get("gate_raw")
        if gr_g2 is None or gr_g2.numel() == 0:
            errs.append("GATE2D gate_raw missing/empty (mandatory in "
                        "this family)")
        elif tuple(gr_g2.shape) != tuple(op_raw.shape):
            errs.append(f"GATE2D gate_raw shape {tuple(gr_g2.shape)} != "
                        f"op_raw {tuple(op_raw.shape)}")
        # the matched no-platform twin runs at the SAME social gate, so
        # it moves -- required in shape here, never compared to innate
        tw_g2 = d.get("twin_raw")
        if tw_g2 is None or tw_g2.numel() == 0:
            errs.append("GATE2D twin_raw missing/empty (WITH_TWIN=1 is "
                        "mandatory in this family)")
        elif tuple(tw_g2.shape) != tuple(op_raw.shape):
            errs.append(f"GATE2D twin_raw shape {tuple(tw_g2.shape)} != "
                        f"op_raw {tuple(op_raw.shape)}")
        if torch.isfinite(pred_raw).all():
            if float(pred_raw.min()) < -1e-6 or \
                    float(pred_raw.max()) > 1 + 1e-6:
                errs.append(f"GATE2D predictions outside [0,1]: "
                            f"[{float(pred_raw.min()):.4f}, "
                            f"{float(pred_raw.max()):.4f}]")
        else:
            errs.append("GATE2D non-finite predictions (parse-fail NaN)")
        if int(cfg.get("icl_k") or 0) > 0:
            ii_g2 = d.get("icl_idx_raw")
            if ii_g2 is None or ii_g2.numel() == 0:
                errs.append("GATE2D icl_k>0 but icl_idx_raw missing/empty")
            elif "_dyn_" in name and ii_g2.shape[0] > 1:
                # refreshed-context provenance: a dyn arm REBUILDS its
                # context from the live population every round -- 8-of-722
                # random draws per agent cannot repeat identically, so
                # bit-identical exemplar ids across every round mean the
                # refresh never ran. Structural, not scientific.
                if all(torch.equal(ii_g2[tt], ii_g2[0])
                       for tt in range(1, ii_g2.shape[0])):
                    errs.append("GATE2D dyn: exemplar ids bit-identical "
                                "across all rounds -- context was never "
                                "refreshed")
            if not os.path.exists(os.path.join(run_dir,
                                               "icl_ctx_log.json.gz")):
                errs.append("GATE2D icl_ctx_log.json.gz missing")
        if cfg.get("training_style") == "frozen":
            ppls_g2 = [r["perplexity"] for r in traj if "perplexity" in r]
            if ppls_g2 and len(set(ppls_g2)) != 1:
                errs.append(f"GATE2D fixed-text perplexity varies "
                            f"({len(set(ppls_g2))} distinct values) -- "
                            f"weights not frozen?")

    # -- 1i CTXGRID (sft_icl_ctxgrid wave, 2026-08-15) -----------------------
    # Mandatory artifacts + invariants for pofdctxgrid* runs. The numeric
    # AI-gate bit-replay rides section 1b (gate_raw REQUIRED here, so 1b
    # always fires) through the SHARED gp.ai_gate. This family spans BOTH
    # peer regimes, so the invariants split on the tokenized social dose:
    # at es=0 the twin must sit on innate and static-serving arms must
    # freeze their gate masks; at es>0 the twin MOVES and gate masks may
    # churn via peers (only the served predictions stay constant). No
    # scientific outcome is a validity condition.
    if is_ctxgrid:
        nr_c = int(cfg.get("n_rounds") or 0)
        if len(traj) != nr_c or int(op_raw.shape[0]) != nr_c:
            errs.append(f"CTXGRID trajectory holds {len(traj)} rows / "
                        f"op_raw {tuple(op_raw.shape)} (config n_rounds="
                        f"{nr_c} -- incomplete or over-run)")
        if pred_raw.shape != op_raw.shape:
            errs.append(f"CTXGRID pred_raw shape {tuple(pred_raw.shape)} "
                        f"!= op_raw {tuple(op_raw.shape)}")
        gr_c = d.get("gate_raw")
        if gr_c is None or gr_c.numel() == 0:
            errs.append("CTXGRID gate_raw missing/empty (mandatory in "
                        "this family)")
        elif tuple(gr_c.shape) != tuple(op_raw.shape):
            errs.append(f"CTXGRID gate_raw shape {tuple(gr_c.shape)} != "
                        f"op_raw {tuple(op_raw.shape)}")
        tw_c = d.get("twin_raw")
        if tw_c is None or tw_c.numel() == 0:
            errs.append("CTXGRID twin_raw missing/empty (WITH_TWIN=1 is "
                        "mandatory in this family)")
        elif tuple(tw_c.shape) != tuple(op_raw.shape):
            errs.append(f"CTXGRID twin_raw shape {tuple(tw_c.shape)} != "
                        f"op_raw {tuple(op_raw.shape)}")
        elif not is_social:
            # no peers: x_cf(t+1) = lam*innate + (1-lam)*x_cf(t) has
            # innate as its fixed point, but float32 rounding of
            # lam*v+(1-lam)*v sits up to ONE ulp off v -- same tolerance
            # the REACH family uses.
            dtw_c = float((tw_c.float() - innate).abs().max())
            if dtw_c > 6.0e-8:
                errs.append(f"CTXGRID es=0 twin drifts off innate (max "
                            f"|diff| {dtw_c:.2e} > 1 float32 ulp)")
        if torch.isfinite(pred_raw).all():
            if float(pred_raw.min()) < -1e-6 or \
                    float(pred_raw.max()) > 1 + 1e-6:
                errs.append(f"CTXGRID predictions outside [0,1]: "
                            f"[{float(pred_raw.min()):.4f}, "
                            f"{float(pred_raw.max()):.4f}]")
        else:
            errs.append("CTXGRID non-finite predictions (parse-fail NaN)")
        if not torch.isfinite(op_raw).all():
            errs.append("CTXGRID non-finite opinions")
        # DEGENERATE CONSTANT SERVING at exactly 0.5. Two very different
        # things look identical in a trajectory: HFCausalLMModel._parse
        # returns a 0.5 default when a generation holds no digit at all,
        # and a model can also genuinely answer "0.5" for everyone. The
        # trajectory alone CANNOT distinguish them.
        # Observed 2026-08-15 on mistral7b at K=32 (723 agents x 3 rounds
        # all 0.5, pred_std 0, unchanged at MAX_NEW_TOKENS 6 and 24),
        # while the same model at K=8 in the same environment serves
        # pred_std ~0.098 and archived qwen/olmo K=32 runs are healthy.
        # CAUSE RESOLVED for mistral7b K=32 (2026-08-15, seed-993
        # diagnostic, once the ICL path actually populated the
        # telemetry): parse_fail_frac = 1.0000 in every round -- 100% of
        # generations hold NO DIGIT. The raw text is prose, "To estimate
        # the user's p...": under a 32-exemplar prompt mistral stops
        # obeying the "Respond with only the number" instruction and
        # starts explaining, so no number appears inside MAX_NEW_TOKENS
        # (checked at 6 and 24) and every agent takes the 0.5 default.
        # An instruction-following failure at long context, not a
        # per-agent opinion. Note a much larger token budget is NOT an
        # obvious fix: _parse takes the FIRST number, which in prose can
        # be a rating quoted back from the profile -- silently wrong
        # values instead of an honest constant.
        # Severity stays WARN, never a failure: the trajectory is
        # internally consistent and what it records is a real property
        # of the model under this prompt.
        if pred_raw.numel() and torch.isfinite(pred_raw).all() \
                and float(pred_raw.min()) == 0.5 \
                and float(pred_raw.max()) == 0.5:
            print(f"WARN {name}: every prediction is exactly 0.5 for all "
                  f"{int(pred_raw.shape[1])} agents in all "
                  f"{int(pred_raw.shape[0])} rounds (pred_std 0) -- a "
                  f"degenerate constant signal, i.e. NO served signal. "
                  f"Known cause for mistral7b at K=32: 100% digit-free "
                  f"generations (prose) taking the _parse 0.5 default. "
                  f"Confirm with DEBUG_GEN=1 (parse_fail_frac).")
        # static-serving channels: frozen weights + fixed (or absent)
        # context + greedy serving => the SERVED predictions are constant
        # in every peer regime. The gate mask is additionally constant
        # only with no peers (peers move agents across a fixed
        # prediction's gate line), with the same 4-ulp boundary tolerance
        # the reach family established.
        _cg_static = next((a for a in ("k0", "fz0", "f32")
                           if f"_{a}_" in name), None)
        if _cg_static is not None:
            for tt in range(1, pred_raw.shape[0]):
                if not _bit_eq(pred_raw[tt], pred_raw[0]):
                    neq = ~((pred_raw[tt] == pred_raw[0])
                            | (torch.isnan(pred_raw[tt])
                               & torch.isnan(pred_raw[0])))
                    errs.append(f"CTXGRID {_cg_static}: pred_raw round "
                                f"{tt} != round 0 ({int(neq.sum())} "
                                f"agents) -- deterministic static serving "
                                f"must be constant within a run")
                    break
            if not is_social and gr_c is not None and gr_c.numel() and \
                    tuple(gr_c.shape) == tuple(op_raw.shape):
                gr_cb = gr_c.bool()
                served0 = pred_raw[0].clamp(0.0, 1.0)
                margin0 = ((served0 - innate).abs() - eps_ai).abs()
                for tt in range(1, gr_cb.shape[0]):
                    flipped = gr_cb[tt] ^ gr_cb[0]
                    if not bool(flipped.any()):
                        continue
                    if float(margin0[flipped].max()) > 2.4e-7:
                        errs.append(
                            f"CTXGRID {_cg_static}: gate_raw round {tt} "
                            f"!= round 0 ({int(flipped.sum())} agents, "
                            f"max round-0 boundary margin "
                            f"{float(margin0[flipped].max()):.2e} > 4 ulp "
                            f"-- no-peer static serving must freeze the "
                            f"gate set)")
                        break
        if "_k0_" in name:
            for key in ("icl_idx_raw", "icl_val_raw"):
                t_k = d.get(key)
                if t_k is not None and t_k.numel() > 0:
                    errs.append(f"CTXGRID k0: {key} non-empty -- "
                                f"no-context prompting must carry no "
                                f"exemplars")
            if os.path.exists(os.path.join(run_dir,
                                           "icl_ctx_log.json.gz")):
                errs.append("CTXGRID k0: icl_ctx_log.json.gz present -- "
                            "no rendered context allowed")
        if int(cfg.get("icl_k") or 0) > 0:
            ii_c = d.get("icl_idx_raw")
            if ii_c is None or ii_c.numel() == 0:
                errs.append("CTXGRID icl_k>0 but icl_idx_raw "
                            "missing/empty")
            elif ii_c.shape[0] > 1 and any(f"_{a}_" in name
                                           for a in ("dyn", "d32")):
                # refreshed-context provenance: a live arm REBUILDS its
                # context every round -- K-of-722 random draws per agent
                # cannot repeat identically, so bit-identical exemplar
                # ids across every round mean the refresh never ran.
                if all(torch.equal(ii_c[tt], ii_c[0])
                       for tt in range(1, ii_c.shape[0])):
                    errs.append("CTXGRID live context: exemplar ids "
                                "bit-identical across all rounds -- "
                                "context was never refreshed")
            # the exemplar count actually rendered must match the dial
            if ii_c is not None and ii_c.numel() > 0 and ii_c.dim() >= 2:
                got_k = int(ii_c.shape[-1])
                want_k = int(cfg.get("icl_k") or 0)
                if got_k != want_k:
                    errs.append(f"CTXGRID icl_idx_raw carries {got_k} "
                                f"exemplars per agent (config icl_k="
                                f"{want_k})")
            if not os.path.exists(os.path.join(run_dir,
                                               "icl_ctx_log.json.gz")):
                errs.append("CTXGRID icl_ctx_log.json.gz missing")
        if cfg.get("training_style") == "frozen":
            ppls_c = [r["perplexity"] for r in traj if "perplexity" in r]
            if ppls_c and len(set(ppls_c)) != 1:
                errs.append(f"CTXGRID fixed-text perplexity varies "
                            f"({len(set(ppls_c))} distinct values) -- "
                            f"weights not frozen?")

    # -- 1j CLAMP (innate-clamp wave, 2026-08-17) ----------------------------
    # Mandatory artifacts + invariants for pofdclamp* runs. The AI-gate
    # bit-replay rides section 1b (gate_raw REQUIRED, so 1b always fires),
    # the responsive-agent exact platform blend rides section 3 with the
    # frozen rows overridden to innate. Nothing here encodes an expected
    # scientific outcome.
    cl_mask = None
    if is_clamp:
        nr_cl = int(cfg.get("n_rounds") or 0)
        if len(traj) != nr_cl or int(op_raw.shape[0]) != nr_cl:
            errs.append(f"CLAMP trajectory holds {len(traj)} rows / "
                        f"op_raw {tuple(op_raw.shape)} (config n_rounds="
                        f"{nr_cl} -- incomplete or over-run)")
        if pred_raw.shape != op_raw.shape:
            errs.append(f"CLAMP pred_raw shape {tuple(pred_raw.shape)} != "
                        f"op_raw {tuple(op_raw.shape)}")
        gr_cl = d.get("gate_raw")
        if gr_cl is None or gr_cl.numel() == 0:
            errs.append("CLAMP gate_raw missing/empty (frozen agents must "
                        "still have gates recorded -- mandatory)")
        elif tuple(gr_cl.shape) != tuple(op_raw.shape):
            errs.append(f"CLAMP gate_raw shape {tuple(gr_cl.shape)} != "
                        f"op_raw {tuple(op_raw.shape)}")
        if not torch.isfinite(op_raw).all():
            errs.append("CLAMP non-finite opinions")
        elif float(op_raw.min()) < -1e-6 or float(op_raw.max()) > 1 + 1e-6:
            errs.append(f"CLAMP opinions outside [0,1]: "
                        f"[{float(op_raw.min()):.4f}, "
                        f"{float(op_raw.max()):.4f}]")
        if not torch.isfinite(pred_raw).all():
            errs.append("CLAMP non-finite predictions (parse-fail NaN)")
        elif float(pred_raw.min()) < -1e-6 or \
                float(pred_raw.max()) > 1 + 1e-6:
            errs.append(f"CLAMP predictions outside [0,1]: "
                        f"[{float(pred_raw.min()):.4f}, "
                        f"{float(pred_raw.max()):.4f}]")
        # cohort mask: present, boolean, exactly round(frac*n) agents
        # (145 of 723 at 0.20), count/mode/seed/hash keys consistent, and
        # RECONSTRUCTING bit-identically from (innate, mode, frac, seed)
        cm_t = d.get("innate_clamp_mask")
        cl_mode = cfg.get("innate_clamp_mode")
        cl_frac = float(cfg.get("innate_clamp_frac") or 0.0)
        cl_seed = cfg.get("innate_clamp_seed")
        if cm_t is None or cm_t.numel() == 0:
            errs.append("CLAMP innate_clamp_mask missing/empty in "
                        "trajectory.pt")
        elif cm_t.dtype != torch.bool or \
                tuple(cm_t.shape) != (innate.numel(),):
            errs.append(f"CLAMP innate_clamp_mask dtype/shape "
                        f"{cm_t.dtype}/{tuple(cm_t.shape)} (want bool "
                        f"[{innate.numel()}])")
        else:
            cl_mask = cm_t.bool()
            n_cl = innate.numel()
            want_frozen = int(round(cl_frac * n_cl))
            got_frozen = int(cl_mask.sum())
            if got_frozen != want_frozen:
                errs.append(f"CLAMP mask freezes {got_frozen} agents "
                            f"(frac {cl_frac:g} of {n_cl} wants exactly "
                            f"{want_frozen})")
            if int(d.get("innate_clamp_count", -1)) != got_frozen:
                errs.append(f"CLAMP innate_clamp_count="
                            f"{d.get('innate_clamp_count')!r} != mask sum "
                            f"{got_frozen}")
            for key, cfg_v in (("innate_clamp_mode", cl_mode),
                               ("innate_clamp_seed", cl_seed)):
                if d.get(key) != cfg_v:
                    errs.append(f"CLAMP trajectory {key}={d.get(key)!r} "
                                f"!= config {cfg_v!r}")
            want_hash = gp.innate_clamp_hash(cl_mask)
            if d.get("innate_clamp_hash") != want_hash:
                errs.append(f"CLAMP innate_clamp_hash="
                            f"{str(d.get('innate_clamp_hash'))[:16]!r}... "
                            f"does not match the stored mask "
                            f"({want_hash[:16]!r}...) -- mask corrupted "
                            f"or tampered")
            if cl_mode in ("graph_clumped", "graph_scattered"):
                # GRAPH-PLACEMENT masks: reconstruct from the committed
                # artifact and RECOMPUTE every graph statistic from the
                # artifact's own edge list (cut, ff, exposure,
                # conductance, largest fixed component) -- the artifact
                # cannot self-certify a stat it did not earn.
                _art_p = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "condor", "clamp_graph_masks.json")
                if not os.path.exists(_art_p):
                    errs.append("CLAMP clamp_graph_masks.json missing "
                                "-- graph masks must be built before "
                                "any job exists")
                else:
                    _art = json.load(open(_art_p))
                    _am = _art["masks"].get(cl_mode) or {}
                    if sorted(cl_mask.nonzero().flatten().tolist()) != \
                            _am.get("ids"):
                        errs.append(f"CLAMP {cl_mode} mask ids differ "
                                    f"from the committed artifact")
                    if d.get("innate_clamp_hash") != _am.get("hash"):
                        errs.append(f"CLAMP {cl_mode} hash differs from "
                                    f"the artifact's")
                    if not all(_art.get("criteria", {}).values()):
                        errs.append("CLAMP artifact acceptance criteria "
                                    "not all true")
                    _a_inn = torch.tensor(_art["innate"],
                                          dtype=torch.float32)
                    if innate.numel() != _a_inn.numel() or \
                            float((innate - _a_inn).abs().max()) > 1e-6:
                        errs.append("CLAMP dataset innate differs from "
                                    "the mask artifact's population")
                    if _am.get("ids"):
                        adj_cl = torch.zeros(n_cl, n_cl,
                                             dtype=torch.bool)
                        for _e1, _e2 in _art["edges"]:
                            adj_cl[_e1][_e2] = True
                            adj_cl[_e2][_e1] = True
                        f_cl = cl_mask
                        r_cl = ~f_cl
                        cut_g = int(adj_cl[f_cl][:, r_cl].sum())
                        ff_g = int(adj_cl[f_cl][:, f_cl].sum()) // 2
                        exp_g = int((adj_cl[r_cl][:, f_cl].sum(1) > 0)
                                    .sum())
                        st_rec = _am.get("stats", {})
                        if cut_g != st_rec.get("cut_edges") or \
                                ff_g != st_rec.get("ff_edges") or \
                                exp_g != st_rec.get(
                                    "exposed_responsive"):
                            errs.append(
                                f"CLAMP {cl_mode} graph stats do not "
                                f"recompute from the artifact edges "
                                f"(cut {cut_g} vs "
                                f"{st_rec.get('cut_edges')}, ff {ff_g} "
                                f"vs {st_rec.get('ff_edges')}, exposed "
                                f"{exp_g} vs "
                                f"{st_rec.get('exposed_responsive')})")
                        _q = _art.get("quotas")
                        _strat_l = _art.get("stratum")
                        if _q and _strat_l:
                            _st_t = torch.tensor(_strat_l)
                            cnt_g = [int((_st_t[cl_mask] == s).sum())
                                     for s in range(len(_q))]
                            if cnt_g != _q:
                                errs.append(f"CLAMP {cl_mode} stratum "
                                            f"counts != shared quotas")
            else:
                try:
                    rec = gp.innate_clamp_mask(innate, cl_mode, cl_frac,
                                               int(cl_seed))
                    if not torch.equal(rec, cl_mask):
                        errs.append(f"CLAMP mask does not reconstruct "
                                    f"from (innate, {cl_mode!r}, "
                                    f"{cl_frac:g}, {cl_seed!r}) -- "
                                    f"{int((rec ^ cl_mask).sum())} "
                                    f"agents differ")
                except (ValueError, TypeError) as e:
                    errs.append(f"CLAMP mask reconstruction impossible: "
                                f"{e}")
            # mode-specific cohort properties, checked independently of
            # the reconstruction (a helper bug cannot self-certify)
            order_cl = sorted(range(n_cl),
                              key=lambda i: (float(innate[i]), i))
            if cl_mode == "bottom":
                want_ids = set(order_cl[:want_frozen])
                got_ids = set(cl_mask.nonzero().flatten().tolist())
                if got_ids != want_ids:
                    errs.append(f"CLAMP bottom cohort is not the "
                                f"{want_frozen} lowest innate opinions "
                                f"(id tie-break): "
                                f"{len(got_ids ^ want_ids)} ids differ")
            elif cl_mode == "stratified_random":
                cuts_cl = gp._largest_remainder([1] * 5, n_cl)
                quotas_cl = gp._largest_remainder(cuts_cl, want_frozen)
                lo_cl = 0
                for b_i, (c_sz, q_w) in enumerate(zip(cuts_cl,
                                                      quotas_cl)):
                    b_ids = torch.tensor(order_cl[lo_cl:lo_cl + c_sz])
                    lo_cl += c_sz
                    q_got = int(cl_mask[b_ids].sum())
                    if q_got != q_w:
                        errs.append(f"CLAMP stratified quintile {b_i} "
                                    f"freezes {q_got} agents (want {q_w} "
                                    f"-- cohort must represent the "
                                    f"innate distribution)")
            # frozen rows: BIT-EXACT copies of innate at every recorded
            # round in BOTH the deployed population and the matched twin
            fro = op_raw[:, cl_mask]
            if not bool((fro == innate[cl_mask].unsqueeze(0)).all()):
                bad_n = int((fro != innate[cl_mask].unsqueeze(0))
                            .any(dim=0).sum())
                errs.append(f"CLAMP {bad_n} frozen agents drift off "
                            f"innate in op_raw (max |diff| "
                            f"{float((fro - innate[cl_mask]).abs().max()):.2e}"
                            f" -- the clamp must be bit-exact)")
            tw_cl = d.get("twin_raw")
            if tw_cl is None or tw_cl.numel() == 0:
                errs.append("CLAMP twin_raw missing/empty (the matched "
                            "twin is mandatory in this family)")
            elif tuple(tw_cl.shape) != tuple(op_raw.shape):
                errs.append(f"CLAMP twin_raw shape {tuple(tw_cl.shape)} "
                            f"!= op_raw {tuple(op_raw.shape)}")
            else:
                tw_cl = tw_cl.float()
                tw_fro = tw_cl[:, cl_mask]
                if not bool((tw_fro
                             == innate[cl_mask].unsqueeze(0)).all()):
                    errs.append(f"CLAMP frozen agents drift off innate in "
                                f"twin_raw (max |diff| "
                                f"{float((tw_fro - innate[cl_mask]).abs().max()):.2e})")
                if not cfg.get("innate_clamp_peer_mode"):
                    # responsive twin: no peers, so lam*v+(1-lam)*v holds
                    # it on innate to <= 1 float32 ulp (the reach
                    # tolerance). With a live clamp-peer operator the
                    # responsive twin MOVES -- never compared to innate.
                    dtw_cl = float((tw_cl[:, ~cl_mask]
                                    - innate[~cl_mask]).abs().max())
                    if dtw_cl > 6.0e-8:
                        errs.append(f"CLAMP responsive twin drifts off "
                                    f"innate (max |diff| {dtw_cl:.2e} > "
                                    f"1 float32 ulp; no-peer twin must "
                                    f"stay innate)")
            # responsive agents must still update normally: if any
            # responsive agent was ever gated onto a served value
            # meaningfully away from its state, the responsive block
            # cannot be identically innate (a clamp applied beyond its
            # mask would freeze them too)
            if gr_cl is not None and gr_cl.numel() and \
                    tuple(gr_cl.shape) == tuple(op_raw.shape):
                gr_b = gr_cl.bool()
                moved = bool((op_raw[:, ~cl_mask]
                              != innate[~cl_mask].unsqueeze(0)).any())
                pulled = False
                for tt in range(op_raw.shape[0]):
                    x_bef = innate if tt == 0 else op_raw[tt - 1]
                    dist = (pred_raw[tt].clamp(0.0, 1.0) - x_bef).abs()
                    if bool((gr_b[tt] & ~cl_mask & (dist > 1e-4)).any()):
                        pulled = True
                        break
                if pulled and not moved:
                    errs.append("CLAMP responsive agents never move off "
                                "innate despite gated contact with a "
                                "distinct served value -- clamp applied "
                                "beyond its mask?")
            # live-ICL runs: frozen agents REMAIN exemplar-eligible.
            # 145-of-723 cohort x K=8 x n x rounds makes zero overlap
            # astronomically improbable unless they were excluded.
            if int(cfg.get("icl_k") or 0) > 0:
                ii_cl = d.get("icl_idx_raw")
                if ii_cl is None or ii_cl.numel() == 0:
                    errs.append("CLAMP icl_k>0 but icl_idx_raw "
                                "missing/empty")
                else:
                    froz_ids = cl_mask.nonzero().flatten()
                    if not bool(torch.isin(ii_cl.long(),
                                           froz_ids).any()):
                        errs.append("CLAMP live ICL contexts never carry "
                                    "a frozen-agent exemplar -- frozen "
                                    "agents must remain eligible")
                    if ii_cl.shape[0] > 1 and "_dyn_" in name and \
                            all(torch.equal(ii_cl[tt], ii_cl[0])
                                for tt in range(1, ii_cl.shape[0])):
                        errs.append("CLAMP dyn: exemplar ids "
                                    "bit-identical across all rounds -- "
                                    "context was never refreshed")
                    if not os.path.exists(os.path.join(
                            run_dir, "icl_ctx_log.json.gz")):
                        errs.append("CLAMP icl_ctx_log.json.gz missing")
            # -- d8 PERSONAL-HISTORY replay (corrected graph wave) -------
            # Every agent's rendered context must be EXACTLY the sentence
            # rebuilt from (innate, op_raw): its OWN recorded opinions,
            # oldest to newest, innate-first on early rounds, window ==
            # icl_days. Byte-equality simultaneously proves the sequence
            # and that NO other agent's profile or opinion entered the
            # context; cross-user exemplar artifacts must not exist at
            # all; fixed agents may see only their own innate repeated.
            icl_d_cl = int(cfg.get("icl_days") or 0)
            if icl_d_cl > 0:
                if int(cfg.get("icl_k") or 0) != 0:
                    errs.append("CLAMP personal-history arm carries "
                                "icl_k>0 -- cross-user exemplars are "
                                "forbidden in d8")
                for _bad_k in ("icl_idx_raw", "icl_val_raw"):
                    _bad_v = d.get(_bad_k)
                    if _bad_v is not None and _bad_v.numel():
                        errs.append(f"CLAMP d8: {_bad_k} non-empty -- "
                                    f"cross-user exemplar artifacts must "
                                    f"not exist")
                if os.path.exists(os.path.join(run_dir,
                                               "icl_ctx_log.json.gz")):
                    errs.append("CLAMP d8: icl_ctx_log.json.gz present "
                                "-- no cross-user context may be "
                                "rendered")
                dl_path = os.path.join(run_dir, "icl_days_log.json.gz")
                if not os.path.exists(dl_path):
                    errs.append("CLAMP d8: icl_days_log.json.gz missing "
                                "(the rendered personal-history "
                                "contexts are mandatory)")
                else:
                    with gzip.open(dl_path, "rt") as fh:
                        dl_rows = [json.loads(line) for line in fh]
                    n_rounds_cl = op_raw.shape[0]
                    tgt_cl = cfg.get("ml_target") or "Action"
                    if [r.get("round") for r in dl_rows] != \
                            list(range(n_rounds_cl)):
                        errs.append(f"CLAMP d8: icl_days_log holds "
                                    f"rounds "
                                    f"{[r.get('round') for r in dl_rows][:5]}"
                                    f"... (want 0..{n_rounds_cl - 1})")
                    else:
                        hist_cl = [innate]
                        bad_hist = None
                        for tt, dr in enumerate(dl_rows):
                            ctxs = dr.get("ctx") or []
                            if len(ctxs) != n_cl:
                                bad_hist = (tt, None,
                                            f"{len(ctxs)} contexts "
                                            f"(want {n_cl})")
                                break
                            win = hist_cl[-icl_d_cl:]
                            for i in range(n_cl):
                                days_s = ", ".join(f"{float(h[i]):.2f}"
                                                   for h in win)
                                want_s = (
                                    "This user's own opinion of "
                                    f"{tgt_cl} movies over the most "
                                    "recent days (oldest to newest): "
                                    f"{days_s}.")
                                if ctxs[i] != want_s:
                                    bad_hist = (tt, i,
                                                f"got {ctxs[i][:90]!r} "
                                                f"want {want_s[:90]!r}")
                                    break
                            if bad_hist is not None:
                                break
                            hist_cl.append(op_raw[tt])
                        if bad_hist is not None:
                            errs.append(f"CLAMP d8: personal-history "
                                        f"context off the (innate, "
                                        f"op_raw) replay at round "
                                        f"{bad_hist[0]} agent "
                                        f"{bad_hist[1]}: {bad_hist[2]}")
                        else:
                            # fixed agents: nothing but their own innate,
                            # stated directly on the final rendered round
                            for i in cl_mask.nonzero().flatten().tolist():
                                iv = f"{float(innate[i]):.2f}"
                                seq_s = dl_rows[-1]["ctx"][i] \
                                    .rsplit(": ", 1)[1].rstrip(".")
                                if any(v != iv
                                       for v in seq_s.split(", ")):
                                    errs.append(
                                        f"CLAMP d8: fixed agent {i} "
                                        f"history is not pure innate "
                                        f"repetition ({seq_s!r} vs "
                                        f"{iv})")
                                    break
            # -- clamp-PEER invariants (mistral_innate_clamp_peer_s0) ----
            cl_peer = cfg.get("innate_clamp_peer_mode")
            if cl_peer:
                if d.get("innate_clamp_peer_mode") != cl_peer:
                    errs.append(f"CLAMP trajectory innate_clamp_peer_mode"
                                f"={d.get('innate_clamp_peer_mode')!r} "
                                f"!= config {cl_peer!r}")
                need_k = ("clamp_fr_sampled", "clamp_fr_accepted",
                          "clamp_fr_reach", "clamp_gap_mean",
                          "clamp_gap_w1", "clamp_resp_std_ratio",
                          "clamp_resp_disp")
                bad_k = [r["round"] for r in traj
                         if any(k not in r for k in need_k)]
                if bad_k:
                    errs.append(f"CLAMP peer telemetry keys missing in "
                                f"rounds {bad_k[:5]}")
                tr_cl = d.get("clamp_fr_touch_raw")
                if tr_cl is None or tr_cl.numel() == 0:
                    errs.append("CLAMP clamp_fr_touch_raw missing/empty "
                                "(mandatory under a peer mode)")
                elif tuple(tr_cl.shape) != tuple(op_raw.shape):
                    errs.append(f"CLAMP clamp_fr_touch_raw shape "
                                f"{tuple(tr_cl.shape)} != op_raw "
                                f"{tuple(op_raw.shape)}")
                else:
                    tr_cl = tr_cl.bool()
                    if bool((tr_cl & cl_mask.unsqueeze(0)).any()):
                        errs.append("CLAMP touch marks a FIXED agent -- "
                                    "only responsive endpoints of "
                                    "accepted F-R pairs may be touched")
                    # recorded cumulative reach must equal the touch log
                    cum = torch.zeros_like(cl_mask)
                    for tt, r in enumerate(traj):
                        cum = cum | tr_cl[tt]
                        want_rc = float(cum[~cl_mask].float().mean())
                        got_rc = r.get("clamp_fr_reach")
                        if got_rc is None or abs(got_rc - want_rc) > 1e-6:
                            errs.append(f"CLAMP round {tt}: "
                                        f"clamp_fr_reach={got_rc!r} != "
                                        f"touch-log cumulative "
                                        f"{want_rc:.6f}")
                            break
                fr_s = [int(r.get("clamp_fr_sampled") or 0) for r in traj]
                fr_a = [int(r.get("clamp_fr_accepted") or 0)
                        for r in traj]
                if any(a > s for s, a in zip(fr_s, fr_a)):
                    errs.append("CLAMP fr_accepted exceeds fr_sampled in "
                                "some round")
                if sum(fr_s) == 0:
                    # accidental isolated-peer behavior: with hundreds of
                    # selections per round and a 20% fixed cohort, zero
                    # sampled F-pairs means fixed agents were EXCLUDED
                    # from pairing -- there is no isolated condition in
                    # this design
                    errs.append("CLAMP stubborn but no fixed agent was "
                                "ever sampled as a pair endpoint -- "
                                "fixed agents must participate in "
                                "pairing (no isolated condition exists)")
                # clamp-aware conservation replay: the RESPONSIVE mean is
                # conserved by the sweep in every round WITHOUT an
                # accepted one-sided F-R move -- op[resp] must then equal
                # the nested blend's z[resp] in mean.
                w_cl = float(cfg.get("w_plat", 1.0))
                lam_cl = float(cfg.get("innate_lambda", 0.0))
                for tt in range(op_raw.shape[0]):
                    if fr_a[tt] > 0:
                        continue
                    served_t = pred_raw[tt].clamp(0.0, 1.0)
                    x0_t = innate if tt == 0 else op_raw[tt - 1]
                    h_t = lam_cl * innate + (1.0 - lam_cl) * x0_t
                    g_t = gp.ai_gate(served_t, x0_t, eps_ai, gate_mode)
                    z_t = torch.where(g_t, (1.0 - w_cl) * h_t
                                      + w_cl * served_t, h_t)
                    dm = abs(float(op_raw[tt][~cl_mask].mean())
                             - float(z_t[~cl_mask].mean()))
                    if dm > 1e-4:
                        errs.append(f"CLAMP round {tt}: responsive mean "
                                    f"off the nested blend by {dm:.2e} "
                                    f"with no one-sided F-R move -- "
                                    f"peer-mode operator not applied as "
                                    f"declared ({cl_peer})")
                        break
                # recorded structure telemetry must match the artifacts
                for tt, r in enumerate(traj):
                    gm = r.get("clamp_gap_mean")
                    want_gm = float(op_raw[tt][~cl_mask].mean()
                                    - op_raw[tt][cl_mask].mean())
                    if gm is None or abs(gm - want_gm) > 1e-5:
                        errs.append(f"CLAMP round {tt}: clamp_gap_mean="
                                    f"{gm!r} != recomputed "
                                    f"{want_gm:.6f}")
                        break
        if cfg.get("training_style") == "frozen":
            ppls_cl = [r["perplexity"] for r in traj if "perplexity" in r]
            if ppls_cl and len(set(ppls_cl)) != 1:
                errs.append(f"CLAMP fixed-text perplexity varies "
                            f"({len(set(ppls_cl))} distinct values) -- "
                            f"weights not frozen?")

    # -- 1k ZSPRIOR (zero-shot prior screen, 2026-08-17) ---------------------
    # Every candidate checkpoint must have loaded, served all 723
    # profiles, parsed to a number for EVERY agent, and left the
    # population untouched. The raw responses are first-class artifacts.
    if is_zsprior:
        n_z = innate.numel()
        if len(traj) != 1 or tuple(op_raw.shape) != (1, n_z):
            errs.append(f"ZSPRIOR trajectory holds {len(traj)} rows / "
                        f"op_raw {tuple(op_raw.shape)} (want exactly one "
                        f"round over {n_z} agents)")
        if tuple(pred_raw.shape) != (1, n_z):
            errs.append(f"ZSPRIOR pred_raw shape {tuple(pred_raw.shape)} "
                        f"-- all {n_z} profiles must be served once")
        elif not torch.isfinite(pred_raw).all():
            errs.append("ZSPRIOR non-finite predictions")
        elif float(pred_raw.min()) < 0.0 or float(pred_raw.max()) > 1.0:
            errs.append(f"ZSPRIOR predictions outside [0,1]: "
                        f"[{float(pred_raw.min()):.4f}, "
                        f"{float(pred_raw.max()):.4f}]")
        elif float(pred_raw.std()) == 0.0:
            print(f"WARN {name}: constant zero-shot prior (every "
                  f"prediction exactly {float(pred_raw[0][0]):g}, "
                  f"pred_std 0) -- a degenerate prior carries no "
                  f"per-agent signal even though parsing succeeded")
        # opinions untouched: the strict EPS_AI=0 gate never opens, so
        # the single recorded round is the lam-anchored innate (<= 1
        # float32 ulp, the reach twin precedent)
        if op_raw.numel():
            d_op_z = float((op_raw[0] - innate).abs().max())
            if d_op_z > 6.0e-8:
                errs.append(f"ZSPRIOR opinions moved off innate (max "
                            f"|diff| {d_op_z:.2e} > 1 float32 ulp; the "
                            f"EPS_AI=0 probe cannot update opinions)")
        gr_z = d.get("gate_raw")
        if gr_z is None or gr_z.numel() == 0:
            errs.append("ZSPRIOR gate_raw missing/empty")
        elif bool(gr_z.bool().any()):
            errs.append(f"ZSPRIOR gate opened for "
                        f"{int(gr_z.bool().sum())} agents -- the strict "
                        f"EPS_AI=0 threshold can never open")
        # raw responses: mandatory, one line, 723 strings, every one
        # carrying a digit, parsed values matching the served vector
        rg_path = os.path.join(run_dir, "raw_gen_log.json.gz")
        if not os.path.exists(rg_path):
            errs.append("ZSPRIOR raw_gen_log.json.gz missing "
                        "(SAVE_RAW_GEN=1 is mandatory in this family)")
        else:
            with gzip.open(rg_path, "rt") as fh:
                rg_rows = [json.loads(line) for line in fh]
            if len(rg_rows) != 1:
                errs.append(f"ZSPRIOR raw_gen_log holds {len(rg_rows)} "
                            f"rounds (want exactly 1)")
            if rg_rows:
                rg = rg_rows[0]
                raw_z = rg.get("raw") or []
                parsed_z = rg.get("parsed") or []
                if len(raw_z) != n_z:
                    errs.append(f"ZSPRIOR {len(raw_z)} raw responses "
                                f"(want {n_z} -- every profile served)")
                no_digit = [i for i, s in enumerate(raw_z)
                            if re.search(r"\d", s) is None]
                if no_digit:
                    errs.append(
                        f"ZSPRIOR numeric parsing FAILED for "
                        f"{len(no_digit)} of {len(raw_z)} responses "
                        f"(digit-free -> served the 0.5 _parse default); "
                        f"first: agent {no_digit[0]} "
                        f"{raw_z[no_digit[0]][:60]!r}")
                if rg.get("parse_fail_frac") not in (0, 0.0):
                    errs.append(f"ZSPRIOR parse_fail_frac="
                                f"{rg.get('parse_fail_frac')!r} (want "
                                f"0.0 -- every response must parse)")
                if len(parsed_z) != n_z:
                    errs.append(f"ZSPRIOR {len(parsed_z)} parsed values "
                                f"(want {n_z})")
                elif tuple(pred_raw.shape) == (1, n_z) and \
                        not torch.equal(
                            torch.tensor(parsed_z,
                                         dtype=torch.float32),
                            pred_raw[0]):
                    errs.append("ZSPRIOR parsed values in "
                                "raw_gen_log.json.gz differ from the "
                                "served pred_raw[0] -- raw/parsed "
                                "provenance broken")

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
                gate = gp.ai_gate(served, x0, eps_ai, gate_mode)
                z = torch.where(gate, (1.0 - w) * h + w * served, h)
                dmean = abs(float(op_raw[t].mean()) - float(z.mean()))
                # clamp-peer runs break TOTAL-mean conservation by design
                # (fixed rows re-pinned to innate off their blend; stubborn
                # F-R moves are one-sided) -- section 1j runs the
                # clamp-aware conservation checks instead
                if dmean > 1e-4 and not is_clamp:
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
        if is_icl or is_iclf or is_ctf or \
                ((is_peer2 or is_gate2d or is_ctxgrid or is_clamp
                  or is_fam)
                 and cfg.get("training_style") == "frozen"):
            # frozen weights: nothing trains, no n_train ever (same skip as
            # the no-peer path below) -- peer-env icl runs (pofdicls2_),
            # the peer2 wave's frozen arms (k0/fz0/dyn), the gate2d
            # wave's dyn arm, the clamp peer/graph waves' dyn arm and
            # the fam scout's k0 arm; sft arms fall through to FRESH
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
        gate = gp.ai_gate(served, x_before, eps_ai, gate_mode)
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
        if cl_mask is not None:
            # innate-clamp runs (section 1j): the runner pins the frozen
            # cohort AFTER the blend, so the exact-copy expectation is
            # innate on frozen rows and the untouched blend elsewhere --
            # this is what "responsive agents still update normally" means
            expect = torch.where(cl_mask, innate, expect)
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
    # icl (frozen): nothing trains, no n_train ever -- skip. Reach frozen
    # arms (fz0/dyn/base) share the skip; reach b0/b1 arms get the check.
    # Clamp b0 falls through: SFT must consume all 723 labels (the frozen
    # cohort's innate labels INCLUDED) every deploy round.
    if is_icl or is_iclf or is_ctf or is_zsprior or \
            ((is_reach or is_ctxgrid or is_clamp)
             and cfg.get("training_style") == "frozen"):
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
        # _es0_ is a no-peer tag (reach family): the peer-alive flavor text
        # must only print when the tokenized dose is actually > 0
        elif (m_es_p := re.search(r"_es(\d+(?:p\d+)?)_", name)) and \
                float(m_es_p.group(1).replace("p", ".")) > 0:
            print(f"PASS {name}  (peer step live, twin simulated, fresh data only)")
        else:
            print(f"PASS {name}  (no peer updates, exact platform blend, fresh data only)")
    print(f"[check_pofd_sanity] {len(dirs) - n_fail}/{len(dirs)} runs pass")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
