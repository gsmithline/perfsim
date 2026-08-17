#!/usr/bin/env bash
# Master submit for the pofd_ platform-only fresh-data sweep. RUN ON THE CLUSTER.
#
#   bash experiments/condor/submit_pofd_sweep.sh <BID> smoke   # 1 gate job, 3 rounds
#   bash experiments/condor/submit_pofd_sweep.sh <BID> full    # the 20-job sweep
#
# Flow: smoke first; when it finishes, gate with
#   python experiments/scripts/cluster_pipelines/check_pofd_sanity.py \
#       runs/pokec_gated_lm/pofdsmk_qwen7b_b0p5_ea0p1_s0_fresh_data
# and submit 'full' ONLY on PASS. Before either submit this script re-verifies
# the on-disk configs against gen_pofd_sweep.py (grid completeness: every
# model x beta x eps_AI x seed row present exactly once) and refuses to submit
# a tag whose run dir already holds a finished trajectory.pt (no overwrites;
# the idempotent executable makes accidental resubmits no-ops anyway).
set -euo pipefail

BID="${1:?usage: submit_pofd_sweep.sh <BID> smoke|qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac|qwen7b_bp[_smoke]|qwen7b_icl[_smoke]|qwen7b_dpo[_smoke]|qwen7b_dpo_ci1|qwen7b_dpo_mr[_smoke]|sft_icl_reach[_smoke|_baseline]|sft_icl_reach_qwen|sft_icl_reach_olmo|sft_icl_reach_mistral|sft_icl_reach_s0|sft_k0_nopeer[_smoke]|sft_k0_nopeer_qwen|sft_k0_nopeer_olmo|sft_k0_nopeer_mistral|sft_icl_nopeer_grid3[_smoke]|sft_icl_nopeer_grid3_qwen|sft_icl_nopeer_grid3_olmo|sft_icl_nopeer_grid3_mistral|sft_icl_peer02[_smoke]|qwen7b_sft_icl_peer02|olmo7b_sft_icl_peer02|mistral7b_sft_icl_peer02|qwen7b_dpo_ci2|qwen7b_dpo_ci3|qwen7b_w[_smoke]|qwen7b_wdpo[_smoke]|qwen7b_wdpon|qwen7b_ws[_smoke]|qwen7b_w2[_smoke]|qwen7b_ws2[_smoke]|qwen7b_w1f[_smoke]|qwen7b_w2f|qwen7b_w2fpt[_smoke]|qwen7b_ws2f|qwen7b_esf[_repl]|qwen7b_fex|qwen7b_pf2|qwen7b_pfs2[_smoke]|qwen7b_fe[s]|qwen7b_fegd|qwen7b_fegp|qwen7b_fef|qwen7b_fe2|qwen7b_fes2|qwen7b_fegd2|qwen7b_fegp2|qwen7b_fef2|qwen7b_fer|qwen7b_fesr|qwen7b_fegdr|qwen7b_fegpr|qwen7b_fec[_smoke]|qwen7b_fesc|qwen7b_fegdc|qwen7b_fegpc|qwen7b_fej[_smoke]|qwen7b_fesj|qwen7b_fegdj|qwen7b_fegpj|qwen7b_fei|qwen7b_feik|qwen7b_feigd|qwen7b_feigp|qwen7b_tch|qwen7b_tfe[_smoke]|qwen7b_tfem|qwen7b_tfe_ci|qwen7b_tfegd|qwen7b_tfegp|qwen7b_tchr|qwen7b_tfer[_smoke]|qwen7b_icl2|qwen7b_icls2[_smoke]|qwen7b_wdpo2|qwen7b_wdpos2[_smoke]|qwen7b_wdpo2e|qwen7b_wdpos2e[_smoke]|qwen7b_wdpoe|mistral7b_ws2f[_smoke]|mistral7b_cube[_s0]|mistral7b_cube_repl|olmo7b_dpo[_smoke]|olmo7b_icl[_smoke]|olmo7b_w1f|olmo7b_w2[_smoke]|olmo7b_ws2[_smoke]|olmo7b_w2f|olmo7b_ws2f|olmo7b_rex|olmo7b_w2fx|olmo7b_ws2fx|olmo7b_icl2|olmo7b_icls2|qwen7b_icls2x|olmo7b_icls2x|icls2x|olmo7brom_fe[s]|olmo7brom_fef|olmo7brom_fe_smoke|qwen7b_cube|olmo7b_cube|qwen7b_replay1|qwen7b_budget[_smoke]|olmo7b_budget|qwen7b_iclf[_smoke]|qwen7b_ctf[_smoke]|qwen7b_iclx[_smoke]|qwen7b_iclx_retry|qwen7b_seedcore[_smoke]|olmo7b_seedcore|qwen_olmo_seedcore|qwen7b_finalfill_sfticl|qwen7b_finalfill_replay|qwen7b_finalfill_corners|qwen_olmo_finalfill|mistral_sft_icl_gate2d[_smoke]|sft_icl_ctxgrid[_smoke|_debug]|sft_icl_ctxgrid_qwen|sft_icl_ctxgrid_olmo|sft_icl_ctxgrid_mistral|fig2_provider|fig2_provider_olmo|fig2_provider_mistral|mistral_innate_clamp_nopeer[_smoke]|full}"
WHAT="${2:?usage: submit_pofd_sweep.sh <BID> smoke|qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac|qwen7b_bp[_smoke]|qwen7b_icl[_smoke]|qwen7b_dpo[_smoke]|qwen7b_dpo_ci1|qwen7b_dpo_mr[_smoke]|sft_icl_reach[_smoke|_baseline]|sft_icl_reach_qwen|sft_icl_reach_olmo|sft_icl_reach_mistral|sft_icl_reach_s0|sft_k0_nopeer[_smoke]|sft_k0_nopeer_qwen|sft_k0_nopeer_olmo|sft_k0_nopeer_mistral|sft_icl_nopeer_grid3[_smoke]|sft_icl_nopeer_grid3_qwen|sft_icl_nopeer_grid3_olmo|sft_icl_nopeer_grid3_mistral|sft_icl_peer02[_smoke]|qwen7b_sft_icl_peer02|olmo7b_sft_icl_peer02|mistral7b_sft_icl_peer02|qwen7b_dpo_ci2|qwen7b_dpo_ci3|qwen7b_w[_smoke]|qwen7b_wdpo[_smoke]|qwen7b_wdpon|qwen7b_ws[_smoke]|qwen7b_w2[_smoke]|qwen7b_ws2[_smoke]|qwen7b_w1f[_smoke]|qwen7b_w2f|qwen7b_w2fpt[_smoke]|qwen7b_ws2f|qwen7b_esf[_repl]|qwen7b_fex|qwen7b_pf2|qwen7b_pfs2[_smoke]|qwen7b_fe[s]|qwen7b_fegd|qwen7b_fegp|qwen7b_fef|qwen7b_fe2|qwen7b_fes2|qwen7b_fegd2|qwen7b_fegp2|qwen7b_fef2|qwen7b_fer|qwen7b_fesr|qwen7b_fegdr|qwen7b_fegpr|qwen7b_fec[_smoke]|qwen7b_fesc|qwen7b_fegdc|qwen7b_fegpc|qwen7b_fej[_smoke]|qwen7b_fesj|qwen7b_fegdj|qwen7b_fegpj|qwen7b_fei|qwen7b_feik|qwen7b_feigd|qwen7b_feigp|qwen7b_tch|qwen7b_tfe[_smoke]|qwen7b_tfem|qwen7b_tfe_ci|qwen7b_tfegd|qwen7b_tfegp|qwen7b_tchr|qwen7b_tfer[_smoke]|qwen7b_icl2|qwen7b_icls2[_smoke]|qwen7b_wdpo2|qwen7b_wdpos2[_smoke]|qwen7b_wdpo2e|qwen7b_wdpos2e[_smoke]|qwen7b_wdpoe|mistral7b_ws2f[_smoke]|mistral7b_cube[_s0]|mistral7b_cube_repl|olmo7b_dpo[_smoke]|olmo7b_icl[_smoke]|olmo7b_w1f|olmo7b_w2[_smoke]|olmo7b_ws2[_smoke]|olmo7b_w2f|olmo7b_ws2f|olmo7b_rex|olmo7b_w2fx|olmo7b_ws2fx|olmo7b_icl2|olmo7b_icls2|qwen7b_icls2x|olmo7b_icls2x|icls2x|olmo7brom_fe[s]|olmo7brom_fef|olmo7brom_fe_smoke|qwen7b_cube|olmo7b_cube|qwen7b_replay1|qwen7b_budget[_smoke]|olmo7b_budget|qwen7b_iclf[_smoke]|qwen7b_ctf[_smoke]|qwen7b_iclx[_smoke]|qwen7b_iclx_retry|qwen7b_seedcore[_smoke]|olmo7b_seedcore|qwen_olmo_seedcore|qwen7b_finalfill_sfticl|qwen7b_finalfill_replay|qwen7b_finalfill_corners|qwen_olmo_finalfill|mistral_sft_icl_gate2d[_smoke]|sft_icl_ctxgrid[_smoke|_debug]|sft_icl_ctxgrid_qwen|sft_icl_ctxgrid_olmo|sft_icl_ctxgrid_mistral|fig2_provider|fig2_provider_olmo|fig2_provider_mistral|mistral_innate_clamp_nopeer[_smoke]|full}"
REPO="${REPO:-/home/gsmithline/perfsim}"
cd "$REPO"

mkdir -p experiments/condor/logs runs/pokec_gated_lm
chmod +x experiments/condor/run_one_pokec_gated.sh \
         experiments/condor/run_one_pokec_gated_idempotent.sh

python3 experiments/condor/gen_pofd_sweep.py --verify

case "$WHAT" in
  smoke)                                 TARGETS="smoke" ;;
  qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac)   TARGETS="$WHAT" ;;
  qwen7b_dpo_ci1|qwen7b_dpo_ci2|qwen7b_dpo_ci3)   TARGETS="$WHAT" ;;
  qwen7b_dpo_mr|qwen7b_dpo_mr_smoke)     TARGETS="$WHAT" ;;
  # SFT-ICL reach wave (2026-08-13): per-model production keys + generated
  # per-model baseline/smoke targets. Flow: sft_icl_reach_smoke (6 jobs)
  # -> gate -> sft_icl_reach_baseline (15 probes) -> gate ->
  # sft_icl_reach (umbrella: baselines idempotently no-op + all 327
  # audited-missing mains). Counts are hard-asserted in gen_pofd_sweep.py
  # against manifest_sft_icl_reach.json.
  sft_icl_reach_qwen|sft_icl_reach_olmo|sft_icl_reach_mistral) TARGETS="$WHAT" ;;
  sft_icl_reach_base_qwen|sft_icl_reach_base_olmo|sft_icl_reach_base_mistral) TARGETS="$WHAT" ;;
  sft_icl_reach_smoke_qwen|sft_icl_reach_smoke_olmo|sft_icl_reach_smoke_mistral) TARGETS="$WHAT" ;;
  sft_icl_reach_smoke)    TARGETS="sft_icl_reach_smoke_qwen sft_icl_reach_smoke_olmo sft_icl_reach_smoke_mistral" ;;
  sft_icl_reach_baseline) TARGETS="sft_icl_reach_base_qwen sft_icl_reach_base_olmo sft_icl_reach_base_mistral" ;;
  # seed-0 exploratory slab (2026-08-13): 33 jobs (qwen 5 + olmo 8 +
  # mistral 20), gates 0.05-0.4 + all-open, seed 0 only. SAME tags as
  # the full production files -- NEVER co-submit with sft_icl_reach /
  # the per-model production keys (double-queue write race). Releasing
  # the full key later no-ops these cells via the idempotent exec.
  sft_icl_reach_s0_qwen|sft_icl_reach_s0_olmo|sft_icl_reach_s0_mistral) TARGETS="$WHAT" ;;
  sft_icl_reach_s0)       TARGETS="sft_icl_reach_s0_qwen sft_icl_reach_s0_olmo sft_icl_reach_s0_mistral" ;;
  # SFT vs frozen no-context prompting, no peers (2026-08-14): 22 jobs
  # (qwen 6 + olmo 6 + mistral 10), seed 0, numeric gates incl. the
  # strict _ea1_ threshold (never all_open). Smokes: mistral k0 ea0p1 +
  # qwen b0 ea1. NEVER co-submit with sft_icl_reach / the per-model
  # reach production keys: the 6 b0/b1 ea0p7 tags are shared by design
  # (the eventual full reach release no-ops them).
  sft_k0_nopeer_qwen|sft_k0_nopeer_olmo|sft_k0_nopeer_mistral) TARGETS="$WHAT" ;;
  sft_k0_nopeer_smoke_qwen|sft_k0_nopeer_smoke_mistral) TARGETS="$WHAT" ;;
  sft_k0_nopeer_smoke)    TARGETS="sft_k0_nopeer_smoke_mistral sft_k0_nopeer_smoke_qwen" ;;
  sft_k0_nopeer)          TARGETS="sft_k0_nopeer_qwen sft_k0_nopeer_olmo sft_k0_nopeer_mistral" ;;
  # three-seed no-peer SFT/ICL gate grid (2026-08-14): 94 informative
  # jobs (qwen 30 + olmo 32 + mistral 32; b0 28 / fz0 33 / dyn 33 /
  # k0 0 -- the k0 seed repetitions are deterministic references and
  # never queue). Smokes: mistral fz0 + dyn at threshold ea1 (outside
  # the 94). NEVER co-submit with sft_icl_reach[_s0] or the per-model
  # reach keys: the numeric-gate (<=0.4) tags are shared by design so
  # a later broad release no-ops these cells (write race otherwise).
  sft_icl_nopeer_grid3_qwen|sft_icl_nopeer_grid3_olmo|sft_icl_nopeer_grid3_mistral) TARGETS="$WHAT" ;;
  sft_icl_nopeer_grid3_smoke) TARGETS="$WHAT" ;;
  sft_icl_nopeer_grid3)   TARGETS="sft_icl_nopeer_grid3_qwen sft_icl_nopeer_grid3_olmo sft_icl_nopeer_grid3_mistral" ;;
  # eps_social=0.2 SFT/ICL channel table (2026-08-14): 45 jobs (qwen 8
  # + olmo 16 + mistral 21; b0 9 / k0 10 / fz0 16 / dyn 10). NEW
  # pofdpeer2_ family -- no shared tags with any other wave, safe to
  # submit alongside anything. Smokes: mistral fz0 + dyn at ea0p1
  # es0p2 seed 991 (outside the 45).
  qwen7b_sft_icl_peer02|olmo7b_sft_icl_peer02|mistral7b_sft_icl_peer02) TARGETS="$WHAT" ;;
  sft_icl_peer02_smoke)   TARGETS="$WHAT" ;;
  sft_icl_peer02)         TARGETS="qwen7b_sft_icl_peer02 olmo7b_sft_icl_peer02 mistral7b_sft_icl_peer02" ;;
  # mistral 2-D gate grid (2026-08-15): 78 jobs -- SFT b0 vs live-ICL
  # dyn x eps_AI 0.05-1.0 x eps_social 0.2/0.4/1.0, seeds 0/42/43;
  # ea1/es1 are REAL numeric thresholds (never all_open). NEW
  # pofdgate2d_ family -- no shared tags with any other wave, safe to
  # submit alongside anything. Smokes: b0 + dyn at ea1 es1 seed 991
  # (outside the 78). Flow: mistral_sft_icl_gate2d_smoke (2 jobs) ->
  # pull + gate -> mistral_sft_icl_gate2d (78 jobs).
  mistral_sft_icl_gate2d|mistral_sft_icl_gate2d_smoke) TARGETS="$WHAT" ;;
  # one-seed context-depth x dual-gate grid (2026-08-15): 181 jobs
  # (qwen 77 + olmo 78 + mistral 26; b0 22 / k0 35 / fz0 38 / dyn 22 /
  # f32 40 / d32 24) over 360 conceptual cells, 139 audited-reused and
  # 40 EXCLUDED -- mistral7b K=32 serves no parseable signal (100%
  # digit-free generations, parse_fail_frac=1.0), so those cells are
  # recorded in the manifest but never queued.
  # BOTH gate axes are real numeric thresholds. NEW pofdctxgrid_ family
  # -- no shared tags with any other wave, safe alongside anything.
  # K=32 arms run ~1.9h vs ~0.9h at K=8 (the wrapper drops
  # GEN_BATCH_SIZE to 8 for ICL_K>=16); whole wave ~310 GPU-h.
  # Smokes: mistral f32 + d32 at ea1 es1 seed 991 (outside the 221).
  # Flow: sft_icl_ctxgrid_smoke -> pull + gate -> sft_icl_ctxgrid.
  sft_icl_ctxgrid_qwen|sft_icl_ctxgrid_olmo|sft_icl_ctxgrid_mistral) TARGETS="$WHAT" ;;
  sft_icl_ctxgrid_smoke)  TARGETS="$WHAT" ;;
  # K=32 parse diagnostic (2026-08-15): the seed-991 smokes served a
  # constant 0.5 (silent parse failure) for mistral at K=32. 2 jobs,
  # seed 992, DEBUG_GEN=1 + MAX_NEW_TOKENS=24.
  sft_icl_ctxgrid_debug)  TARGETS="$WHAT" ;;
  sft_icl_ctxgrid)        TARGETS="sft_icl_ctxgrid_qwen sft_icl_ctxgrid_olmo sft_icl_ctxgrid_mistral" ;;
  # Figure-2 provider replication (2026-08-15): 6 jobs completing
  # three-seed b1 (SFT-KL beta=1) coverage at ea 0.4 -- olmo 4 (es 0
  # and 0.2, seeds 42/43) + mistral 2 (es 0, seeds 42/43). No smoke:
  # every cell's seed-0 twin already ran and gated in this exact
  # environment. The four es=0 cells carry pofdreach_ tags SHARED with
  # the unreleased 327-job reach production -- NEVER co-submit with
  # sft_icl_reach or sft_icl_reach_{olmo,mistral}.
  fig2_provider_olmo|fig2_provider_mistral) TARGETS="$WHAT" ;;
  fig2_provider)          TARGETS="fig2_provider_olmo fig2_provider_mistral" ;;
  # no-peer innate-clamp wave (2026-08-17): mistral-only b0/dyn with
  # 20% of the population (145/723) permanently pinned to innate --
  # cohorts stratified_random + bottom, numeric gates {0.05,0.1,0.2,
  # 0.4,1.0} (ea1 = strict-< threshold, never all_open), seeds
  # 0/42/43 = 60 jobs. NEW pofdclamp_ family, no shared tags, es=0 by
  # construction (the runner hard-fails the clamp under a live peer
  # step). Smokes: 4 x 3 rounds seed 991, both modes x both arms at
  # ea0p2. Flow: mistral_innate_clamp_nopeer_smoke -> pull + gate ->
  # mistral_innate_clamp_nopeer.
  mistral_innate_clamp_nopeer|mistral_innate_clamp_nopeer_smoke) TARGETS="$WHAT" ;;
  sft_icl_reach)          TARGETS="sft_icl_reach_base_qwen sft_icl_reach_base_olmo sft_icl_reach_base_mistral sft_icl_reach_qwen sft_icl_reach_olmo sft_icl_reach_mistral" ;;
  qwen7b_icl|qwen7b_icl_smoke|qwen7b_dpo|qwen7b_dpo_smoke)   TARGETS="$WHAT" ;;
  qwen7b_dpon|qwen7b_dpon_smoke)         TARGETS="$WHAT" ;;
  qwen7b_bp|qwen7b_bp_smoke)             TARGETS="$WHAT" ;;
  qwen7b_w|qwen7b_w_smoke|qwen7b_wdpo|qwen7b_wdpo_smoke|qwen7b_wdpon)   TARGETS="$WHAT" ;;
  qwen7b_ws|qwen7b_ws_smoke)             TARGETS="$WHAT" ;;
  qwen7b_w2|qwen7b_w2_smoke)             TARGETS="$WHAT" ;;
  qwen7b_ws2|qwen7b_ws2_smoke)           TARGETS="$WHAT" ;;
  qwen7b_w1f|qwen7b_w1f_smoke|qwen7b_w2f|qwen7b_w2fpt|qwen7b_w2fpt_smoke|qwen7b_ws2f)   TARGETS="$WHAT" ;;
  qwen7b_esf|qwen7b_esf_repl|qwen7b_fex) TARGETS="$WHAT" ;;
  qwen7b_pf2|qwen7b_pfs2|qwen7b_pfs2_smoke)   TARGETS="$WHAT" ;;
  qwen7b_fes|qwen7b_fegd|qwen7b_fegp|qwen7b_fef)   TARGETS="$WHAT" ;;
  qwen7b_fes2|qwen7b_fegd2|qwen7b_fegp2|qwen7b_fef2) TARGETS="$WHAT" ;;
  qwen7b_fesr|qwen7b_fegdr|qwen7b_fegpr) TARGETS="$WHAT" ;;
  qwen7b_fesc|qwen7b_fegdc|qwen7b_fegpc|qwen7b_fec_smoke) TARGETS="$WHAT" ;;
  qwen7b_fesj|qwen7b_fegdj|qwen7b_fegpj|qwen7b_fej_smoke) TARGETS="$WHAT" ;;
  qwen7b_fe)                             TARGETS="qwen7b_fes qwen7b_fegd qwen7b_fegp qwen7b_fef" ;;
  qwen7b_fe2)                            TARGETS="qwen7b_fes2 qwen7b_fegd2 qwen7b_fegp2 qwen7b_fef2" ;;
  qwen7b_fer)                            TARGETS="qwen7b_fesr qwen7b_fegdr qwen7b_fegpr" ;;
  qwen7b_fec)                            TARGETS="qwen7b_fesc qwen7b_fegdc qwen7b_fegpc" ;;
  qwen7b_fej)                            TARGETS="qwen7b_fesj qwen7b_fegdj qwen7b_fegpj" ;;
  qwen7b_feik|qwen7b_feigd|qwen7b_feigp) TARGETS="$WHAT" ;;
  qwen7b_fei)                            TARGETS="qwen7b_feik qwen7b_feigd qwen7b_feigp" ;;
  qwen7b_tch|qwen7b_tfem|qwen7b_tfe_ci|qwen7b_tfegd|qwen7b_tfegp|qwen7b_tfe_smoke) TARGETS="$WHAT" ;;
  qwen7b_tfe)                            TARGETS="qwen7b_tfem qwen7b_tfegd qwen7b_tfegp" ;;
  qwen7b_tchr|qwen7b_tfer|qwen7b_tfer_smoke) TARGETS="$WHAT" ;;
  qwen7b_icl2|qwen7b_icls2|qwen7b_icls2_smoke)   TARGETS="$WHAT" ;;
  qwen7b_wdpo2|qwen7b_wdpos2|qwen7b_wdpos2_smoke)   TARGETS="$WHAT" ;;
  mistral7b_ws2f|mistral7b_ws2f_smoke)   TARGETS="$WHAT" ;;
  mistral7b_cube_s0|mistral7b_cube_repl)   TARGETS="$WHAT" ;;
  mistral7b_cube)                        TARGETS="mistral7b_cube_s0 mistral7b_cube_repl" ;;
  qwen7b_wdpo2e|qwen7b_wdpos2e|qwen7b_wdpos2e_smoke)   TARGETS="$WHAT" ;;
  qwen7b_wdpoe)                          TARGETS="qwen7b_wdpo2e qwen7b_wdpos2e" ;;
  olmo7b_icl2|olmo7b_icls2)              TARGETS="$WHAT" ;;
  qwen7b_icls2x|olmo7b_icls2x)           TARGETS="$WHAT" ;;
  icls2x)                                TARGETS="qwen7b_icls2x olmo7b_icls2x" ;;
  olmo7b_dpo|olmo7b_dpo_smoke)           TARGETS="$WHAT" ;;
  olmo7b_icl|olmo7b_icl_smoke)           TARGETS="$WHAT" ;;
  olmo7b_w1f)                            TARGETS="$WHAT" ;;
  olmo7b_w2|olmo7b_w2_smoke)             TARGETS="$WHAT" ;;
  olmo7b_ws2|olmo7b_ws2_smoke)           TARGETS="$WHAT" ;;
  olmo7b_w2f|olmo7b_ws2f)                TARGETS="$WHAT" ;;
  olmo7b_w2fx|olmo7b_ws2fx)              TARGETS="$WHAT" ;;
  olmo7b_rex)                            TARGETS="olmo7b_w2fx olmo7b_ws2fx" ;;
  olmo7brom_fes|olmo7brom_fef|olmo7brom_fe_smoke) TARGETS="$WHAT" ;;
  olmo7brom_fe)                          TARGETS="olmo7brom_fes olmo7brom_fef" ;;
  qwen7b_cube|olmo7b_cube)               TARGETS="$WHAT" ;;
  qwen7b_replay1)                        TARGETS="$WHAT" ;;
  qwen7b_budget|olmo7b_budget|qwen7b_budget_smoke)   TARGETS="$WHAT" ;;
  qwen7b_iclf|qwen7b_iclf_smoke)         TARGETS="$WHAT" ;;
  qwen7b_ctf|qwen7b_ctf_smoke)           TARGETS="$WHAT" ;;
  # umbrella: the frozen-context icl wave + the context-transfer wave in
  # one submission (user 2026-08-07: "add this to the [iclf] jobs")
  qwen7b_iclx)                           TARGETS="qwen7b_iclf qwen7b_ctf" ;;
  qwen7b_iclx_smoke)                     TARGETS="qwen7b_iclf_smoke qwen7b_ctf_smoke" ;;
  qwen7b_iclf_retry|qwen7b_ctf_retry)    TARGETS="$WHAT" ;;
  # exactly the 11 g106-killed cells; nothing completed re-queues
  qwen7b_iclx_retry)                     TARGETS="qwen7b_iclf_retry qwen7b_ctf_retry" ;;
  qwen7b_seedcore|olmo7b_seedcore|qwen7b_seedcore_smoke) TARGETS="$WHAT" ;;
  # umbrella: both models' seed-replication cores (34 jobs total --
  # retention 12 + direct transmission 8 + main peer replicates 14)
  qwen_olmo_seedcore)                    TARGETS="qwen7b_seedcore olmo7b_seedcore" ;;
  qwen7b_finalfill_sfticl|qwen7b_finalfill_replay|qwen7b_finalfill_corners) TARGETS="$WHAT" ;;
  # umbrella: the final replication wave -- three qwen finalfill keys +
  # the UNCHANGED olmo7brom_fe wave (54 queued; the 2 complete corner
  # cells + 6 complete Romance cells no-op via the idempotent exec).
  # Flow: olmo7brom_fe_smoke first (already complete + gated 2026-08-07
  # -- resubmitting is a no-op), then this key.
  qwen_olmo_finalfill)                   TARGETS="qwen7b_finalfill_sfticl qwen7b_finalfill_replay qwen7b_finalfill_corners olmo7brom_fes olmo7brom_fef" ;;
  qwen7b_esfn|olmo7b_esfn)
    # SUPERSEDED 2026-08-05 by the full parameter cube: every esfn cell is a
    # cube cell with a BYTE-IDENTICAL tag, so co-submitting both keys would
    # double-queue 52 tags into the same run dirs (a write race, not a no-op:
    # the idempotent exec only skips COMPLETED trajectories).
    echo "[submit_pofd] ${WHAT} is SUPERSEDED by ${WHAT%_esfn}_cube -- not submitting." >&2
    echo "[submit_pofd] use: bash experiments/condor/submit_pofd_sweep.sh ${BID} ${WHAT%_esfn}_cube" >&2
    exit 2 ;;
  full)                                  TARGETS="qwen7b gemma12b olmo7b" ;;
  *) echo "usage: submit_pofd_sweep.sh <BID> smoke|qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac|qwen7b_bp[_smoke]|qwen7b_icl[_smoke]|qwen7b_dpo[_smoke]|qwen7b_dpo_ci1|qwen7b_dpo_mr[_smoke]|sft_icl_reach[_smoke|_baseline]|sft_icl_reach_qwen|sft_icl_reach_olmo|sft_icl_reach_mistral|sft_icl_reach_s0|sft_k0_nopeer[_smoke]|sft_k0_nopeer_qwen|sft_k0_nopeer_olmo|sft_k0_nopeer_mistral|sft_icl_nopeer_grid3[_smoke]|sft_icl_nopeer_grid3_qwen|sft_icl_nopeer_grid3_olmo|sft_icl_nopeer_grid3_mistral|sft_icl_peer02[_smoke]|qwen7b_sft_icl_peer02|olmo7b_sft_icl_peer02|mistral7b_sft_icl_peer02|qwen7b_dpo_ci2|qwen7b_dpo_ci3|qwen7b_w[_smoke]|qwen7b_wdpo[_smoke]|qwen7b_wdpon|qwen7b_ws[_smoke]|qwen7b_w2[_smoke]|qwen7b_ws2[_smoke]|qwen7b_w1f[_smoke]|qwen7b_w2f|qwen7b_w2fpt[_smoke]|qwen7b_ws2f|qwen7b_esf[_repl]|qwen7b_fex|qwen7b_pf2|qwen7b_pfs2[_smoke]|qwen7b_fe[s]|qwen7b_fegd|qwen7b_fegp|qwen7b_fef|qwen7b_fe2|qwen7b_fes2|qwen7b_fegd2|qwen7b_fegp2|qwen7b_fef2|qwen7b_fer|qwen7b_fesr|qwen7b_fegdr|qwen7b_fegpr|qwen7b_fec[_smoke]|qwen7b_fesc|qwen7b_fegdc|qwen7b_fegpc|qwen7b_fej[_smoke]|qwen7b_fesj|qwen7b_fegdj|qwen7b_fegpj|qwen7b_fei|qwen7b_feik|qwen7b_feigd|qwen7b_feigp|qwen7b_tch|qwen7b_tfe[_smoke]|qwen7b_tfem|qwen7b_tfe_ci|qwen7b_tfegd|qwen7b_tfegp|qwen7b_tchr|qwen7b_tfer[_smoke]|qwen7b_icl2|qwen7b_icls2[_smoke]|qwen7b_wdpo2|qwen7b_wdpos2[_smoke]|qwen7b_wdpo2e|qwen7b_wdpos2e[_smoke]|qwen7b_wdpoe|mistral7b_ws2f[_smoke]|mistral7b_cube[_s0]|mistral7b_cube_repl|olmo7b_dpo[_smoke]|olmo7b_icl[_smoke]|olmo7b_w1f|olmo7b_w2[_smoke]|olmo7b_ws2[_smoke]|olmo7b_w2f|olmo7b_ws2f|olmo7b_rex|olmo7b_w2fx|olmo7b_ws2fx|olmo7b_icl2|olmo7b_icls2|qwen7b_icls2x|olmo7b_icls2x|icls2x|olmo7brom_fe[s]|olmo7brom_fef|olmo7brom_fe_smoke|qwen7b_cube|olmo7b_cube|qwen7b_replay1|qwen7b_budget[_smoke]|olmo7b_budget|qwen7b_iclf[_smoke]|qwen7b_ctf[_smoke]|qwen7b_iclx[_smoke]|qwen7b_iclx_retry|qwen7b_seedcore[_smoke]|olmo7b_seedcore|qwen_olmo_seedcore|qwen7b_finalfill_sfticl|qwen7b_finalfill_replay|qwen7b_finalfill_corners|qwen_olmo_finalfill|mistral_sft_icl_gate2d[_smoke]|sft_icl_ctxgrid[_smoke|_debug]|sft_icl_ctxgrid_qwen|sft_icl_ctxgrid_olmo|sft_icl_ctxgrid_mistral|fig2_provider|fig2_provider_olmo|fig2_provider_mistral|mistral_innate_clamp_nopeer[_smoke]|full" >&2; exit 2 ;;
esac

for T in $TARGETS; do
  SUB="experiments/condor/at_pofd_${T}.sub"
  CFG="experiments/condor/configs_pofd_${T}.txt"
  done_n=0
  while IFS=, read -r tag _; do
    if [ -f "runs/pokec_gated_lm/${tag}/trajectory.pt" ]; then
      echo "[submit_pofd] NOTE: ${tag} already has trajectory.pt (idempotent no-op if complete)"
      done_n=$((done_n + 1))
    fi
  done < "$CFG"
  echo "[submit_pofd] ${T}: $(wc -l < "$CFG") jobs in $CFG (${done_n} with existing results)"
  condor_submit_bid "$BID" "$SUB"
done
