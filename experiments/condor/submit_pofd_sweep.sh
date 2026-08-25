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

BID="${1:?usage: submit_pofd_sweep.sh <BID> smoke|qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac|qwen7b_bp[_smoke]|qwen7b_icl[_smoke]|qwen7b_dpo[_smoke]|qwen7b_dpo_ci1|qwen7b_dpo_mr[_smoke]|sft_icl_reach[_smoke|_baseline]|sft_icl_reach_qwen|sft_icl_reach_olmo|sft_icl_reach_mistral|sft_icl_reach_s0|sft_k0_nopeer[_smoke]|sft_k0_nopeer_qwen|sft_k0_nopeer_olmo|sft_k0_nopeer_mistral|sft_icl_nopeer_grid3[_smoke]|sft_icl_nopeer_grid3_qwen|sft_icl_nopeer_grid3_olmo|sft_icl_nopeer_grid3_mistral|sft_icl_peer02[_smoke]|qwen7b_sft_icl_peer02|olmo7b_sft_icl_peer02|mistral7b_sft_icl_peer02|qwen7b_dpo_ci2|qwen7b_dpo_ci3|qwen7b_w[_smoke]|qwen7b_wdpo[_smoke]|qwen7b_wdpon|qwen7b_ws[_smoke]|qwen7b_w2[_smoke]|qwen7b_ws2[_smoke]|qwen7b_w1f[_smoke]|qwen7b_w2f|qwen7b_w2fpt[_smoke]|qwen7b_ws2f|qwen7b_esf[_repl]|qwen7b_fex|qwen7b_pf2|qwen7b_pfs2[_smoke]|qwen7b_fe[s]|qwen7b_fegd|qwen7b_fegp|qwen7b_fef|qwen7b_fe2|qwen7b_fes2|qwen7b_fegd2|qwen7b_fegp2|qwen7b_fef2|qwen7b_fer|qwen7b_fesr|qwen7b_fegdr|qwen7b_fegpr|qwen7b_fec[_smoke]|qwen7b_fesc|qwen7b_fegdc|qwen7b_fegpc|qwen7b_fej[_smoke]|qwen7b_fesj|qwen7b_fegdj|qwen7b_fegpj|qwen7b_fei|qwen7b_feik|qwen7b_feigd|qwen7b_feigp|qwen7b_tch|qwen7b_tfe[_smoke]|qwen7b_tfem|qwen7b_tfe_ci|qwen7b_tfegd|qwen7b_tfegp|qwen7b_tchr|qwen7b_tfer[_smoke]|qwen7b_icl2|qwen7b_icls2[_smoke]|qwen7b_wdpo2|qwen7b_wdpos2[_smoke]|qwen7b_wdpo2e|qwen7b_wdpos2e[_smoke]|qwen7b_wdpoe|mistral7b_ws2f[_smoke]|mistral7b_cube[_s0]|mistral7b_cube_repl|olmo7b_dpo[_smoke]|olmo7b_icl[_smoke]|olmo7b_w1f|olmo7b_w2[_smoke]|olmo7b_ws2[_smoke]|olmo7b_w2f|olmo7b_ws2f|olmo7b_rex|olmo7b_w2fx|olmo7b_ws2fx|olmo7b_icl2|olmo7b_icls2|qwen7b_icls2x|olmo7b_icls2x|icls2x|olmo7brom_fe[s]|olmo7brom_fef|olmo7brom_fe_smoke|qwen7b_cube|olmo7b_cube|qwen7b_replay1|qwen7b_budget[_smoke]|olmo7b_budget|qwen7b_iclf[_smoke]|qwen7b_ctf[_smoke]|qwen7b_iclx[_smoke]|qwen7b_iclx_retry|qwen7b_seedcore[_smoke]|olmo7b_seedcore|qwen_olmo_seedcore|qwen7b_finalfill_sfticl|qwen7b_finalfill_replay|qwen7b_finalfill_corners|qwen_olmo_finalfill|mistral_sft_icl_gate2d[_smoke]|sft_icl_ctxgrid[_smoke|_debug]|sft_icl_ctxgrid_qwen|sft_icl_ctxgrid_olmo|sft_icl_ctxgrid_mistral|fig2_provider|fig2_provider_olmo|fig2_provider_mistral|mistral_innate_clamp_nopeer[_smoke]|mistral_innate_clamp_graph_s0|mistral_innate_clamp_graph_smoke|mistral_innate_clamp_graph_d8[_smoke]|mistral_clamp_exclude_a[_smoke]|mistral_bottom20_source_impact|mistral_bottom20_evolving|mistral_bottom20_section4_repl[_fixed|_evo]|feature_endogenization_n5[_nat|_frozen|_gd|_gp]|qwen_gate_sweep|qwen_k1_grid[_es1]|feature_lambda1_seeds|feature_lambda1_repro|feature_lambda_match|qwen_mechanism_frozen|qwen_wu_limit[_smoke|_icl]|qwen_subsample[_smoke]|qwen_sft_update_dose|qwen_sft_lr_dose|qwen_sft_rank_dose|qwen_sft_dose_smoke|qwen_adapter_kl_probe[_smoke]|qwen_kl_direction[_smoke]|ref_replay[_smoke]|fj_robustness[_smoke|_frozen|_beta1|_kl_ladder|_seeds|_beta1_seeds]|jiduan_pokec_smoke|jiduan_pokec_controls|jiduan_pokec_prior[_seeds]|jiduan_pokec_lambda_ladder|jiduan_pokec_icl|jiduan_pokec_environment|jiduan_pokec_routing_smoke|jiduan_pokec_routing_seeds|jiduan_pokec_frozen|fig2_family_prior_scout|fig2_family_prior_smoke|fig2_family_prior_qwen_retry|fig2_family_prior_beta_scout|fig2_family_prior_b2_confirm|fig2_family_prior_b4_confirm|fig2_family_prior_b8_confirm|fam_gate_ablation|fam_gate_social|zsprior_screen|section3_retention[_smoke]|section3_peer_sweeps[_smoke]|section3_memory[_smoke]|section3_label_mix[_smoke]|section3_lambda_fill[_smoke]|section4_gate_anch2[_smoke][_fixed|_evo]|section4_gate_anch2_fig6[_smoke|_ext|_s0|_s42_43][_fixed|_evo]|fig4_family_prior_repl30|fig3_full_loop[_smoke|_ext]|sft_update_dose_loop[_smoke]|fig4_anchor_tradeoff[_smoke|_ext|_zsprior]|full}"
WHAT="${2:?usage: submit_pofd_sweep.sh <BID> smoke|qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac|qwen7b_bp[_smoke]|qwen7b_icl[_smoke]|qwen7b_dpo[_smoke]|qwen7b_dpo_ci1|qwen7b_dpo_mr[_smoke]|sft_icl_reach[_smoke|_baseline]|sft_icl_reach_qwen|sft_icl_reach_olmo|sft_icl_reach_mistral|sft_icl_reach_s0|sft_k0_nopeer[_smoke]|sft_k0_nopeer_qwen|sft_k0_nopeer_olmo|sft_k0_nopeer_mistral|sft_icl_nopeer_grid3[_smoke]|sft_icl_nopeer_grid3_qwen|sft_icl_nopeer_grid3_olmo|sft_icl_nopeer_grid3_mistral|sft_icl_peer02[_smoke]|qwen7b_sft_icl_peer02|olmo7b_sft_icl_peer02|mistral7b_sft_icl_peer02|qwen7b_dpo_ci2|qwen7b_dpo_ci3|qwen7b_w[_smoke]|qwen7b_wdpo[_smoke]|qwen7b_wdpon|qwen7b_ws[_smoke]|qwen7b_w2[_smoke]|qwen7b_ws2[_smoke]|qwen7b_w1f[_smoke]|qwen7b_w2f|qwen7b_w2fpt[_smoke]|qwen7b_ws2f|qwen7b_esf[_repl]|qwen7b_fex|qwen7b_pf2|qwen7b_pfs2[_smoke]|qwen7b_fe[s]|qwen7b_fegd|qwen7b_fegp|qwen7b_fef|qwen7b_fe2|qwen7b_fes2|qwen7b_fegd2|qwen7b_fegp2|qwen7b_fef2|qwen7b_fer|qwen7b_fesr|qwen7b_fegdr|qwen7b_fegpr|qwen7b_fec[_smoke]|qwen7b_fesc|qwen7b_fegdc|qwen7b_fegpc|qwen7b_fej[_smoke]|qwen7b_fesj|qwen7b_fegdj|qwen7b_fegpj|qwen7b_fei|qwen7b_feik|qwen7b_feigd|qwen7b_feigp|qwen7b_tch|qwen7b_tfe[_smoke]|qwen7b_tfem|qwen7b_tfe_ci|qwen7b_tfegd|qwen7b_tfegp|qwen7b_tchr|qwen7b_tfer[_smoke]|qwen7b_icl2|qwen7b_icls2[_smoke]|qwen7b_wdpo2|qwen7b_wdpos2[_smoke]|qwen7b_wdpo2e|qwen7b_wdpos2e[_smoke]|qwen7b_wdpoe|mistral7b_ws2f[_smoke]|mistral7b_cube[_s0]|mistral7b_cube_repl|olmo7b_dpo[_smoke]|olmo7b_icl[_smoke]|olmo7b_w1f|olmo7b_w2[_smoke]|olmo7b_ws2[_smoke]|olmo7b_w2f|olmo7b_ws2f|olmo7b_rex|olmo7b_w2fx|olmo7b_ws2fx|olmo7b_icl2|olmo7b_icls2|qwen7b_icls2x|olmo7b_icls2x|icls2x|olmo7brom_fe[s]|olmo7brom_fef|olmo7brom_fe_smoke|qwen7b_cube|olmo7b_cube|qwen7b_replay1|qwen7b_budget[_smoke]|olmo7b_budget|qwen7b_iclf[_smoke]|qwen7b_ctf[_smoke]|qwen7b_iclx[_smoke]|qwen7b_iclx_retry|qwen7b_seedcore[_smoke]|olmo7b_seedcore|qwen_olmo_seedcore|qwen7b_finalfill_sfticl|qwen7b_finalfill_replay|qwen7b_finalfill_corners|qwen_olmo_finalfill|mistral_sft_icl_gate2d[_smoke]|sft_icl_ctxgrid[_smoke|_debug]|sft_icl_ctxgrid_qwen|sft_icl_ctxgrid_olmo|sft_icl_ctxgrid_mistral|fig2_provider|fig2_provider_olmo|fig2_provider_mistral|mistral_innate_clamp_nopeer[_smoke]|mistral_innate_clamp_graph_s0|mistral_innate_clamp_graph_smoke|mistral_innate_clamp_graph_d8[_smoke]|mistral_clamp_exclude_a[_smoke]|mistral_bottom20_source_impact|mistral_bottom20_evolving|mistral_bottom20_section4_repl[_fixed|_evo]|feature_endogenization_n5[_nat|_frozen|_gd|_gp]|qwen_gate_sweep|qwen_k1_grid[_es1]|feature_lambda1_seeds|feature_lambda1_repro|feature_lambda_match|qwen_mechanism_frozen|qwen_wu_limit[_smoke|_icl]|qwen_subsample[_smoke]|qwen_sft_update_dose|qwen_sft_lr_dose|qwen_sft_rank_dose|qwen_sft_dose_smoke|qwen_adapter_kl_probe[_smoke]|qwen_kl_direction[_smoke]|ref_replay[_smoke]|fj_robustness[_smoke|_frozen|_beta1|_kl_ladder|_seeds|_beta1_seeds]|jiduan_pokec_smoke|jiduan_pokec_controls|jiduan_pokec_prior[_seeds]|jiduan_pokec_lambda_ladder|jiduan_pokec_icl|jiduan_pokec_environment|jiduan_pokec_routing_smoke|jiduan_pokec_routing_seeds|jiduan_pokec_frozen|fig2_family_prior_scout|fig2_family_prior_smoke|fig2_family_prior_qwen_retry|fig2_family_prior_beta_scout|fig2_family_prior_b2_confirm|fig2_family_prior_b4_confirm|fig2_family_prior_b8_confirm|fam_gate_ablation|fam_gate_social|zsprior_screen|section3_retention[_smoke]|section3_peer_sweeps[_smoke]|section3_memory[_smoke]|section3_label_mix[_smoke]|section3_lambda_fill[_smoke]|section4_gate_anch2[_smoke][_fixed|_evo]|section4_gate_anch2_fig6[_smoke|_ext|_s0|_s42_43][_fixed|_evo]|fig4_family_prior_repl30|fig3_full_loop[_smoke|_ext]|sft_update_dose_loop[_smoke]|fig4_anchor_tradeoff[_smoke|_ext|_zsprior]|full}"
REPO="${REPO:-/home/gsmithline/perfsim}"
cd "$REPO"

mkdir -p experiments/condor/logs runs/pokec_gated_lm
chmod +x experiments/condor/run_one_pokec_gated.sh \
         experiments/condor/run_one_pokec_gated_idempotent.sh \
         experiments/condor/run_one_adapter_kl_probe.sh

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
  # innate-clamp GRAPH-PLACEMENT wave (2026-08-17): 96 seed-0 jobs --
  # two PRE-BUILT 145-agent cohorts (clamp_graph_masks.json,
  # deterministic; graph_clumped = one connected low-cut block, 41.5%
  # exposure vs graph_scattered = distributed, 100% exposure, cut
  # 2.9x; matched innate x degree strata) under the one-sided
  # STUBBORN operator (fixed agents keep pairing; only responsive
  # endpoints move; NO isolated condition). gclump/gscat x b0/dyn x
  # ea {0.1,0.2,0.4,1} x es {0,0.05,0.1,0.2,0.4,1} -- es=0 baselines
  # IN-WAVE (old no-peer cohorts are different masks, not baselines).
  # Same pofdclamp_ family, _gclump_/_gscat_ tokens disambiguate.
  # Smokes: both masks x both arms at ea0p4 es0p4 seed 991. Flow:
  # mistral_innate_clamp_graph_smoke -> pull + gate ->
  # mistral_innate_clamp_graph_s0.
  mistral_innate_clamp_graph_s0|mistral_innate_clamp_graph_smoke) TARGETS="$WHAT" ;;
  # CORRECTED graph-clamp wave (2026-08-17): personal-history ICL
  # (_d8_: frozen weights, ICL_K=0, ICL_DAYS=8 -- each agent sees only
  # its OWN latest eight opinions, oldest to newest; fixed agents see
  # only their own innate repeated) REPLACES the cross-user _dyn_ arm.
  # 51 jobs = the 48 new _d8_ cells + the 3 _b0_ cells missing at the
  # 2026-08-17 audit (those 3 tags are SHARED with
  # mistral_innate_clamp_graph_s0 -- NEVER co-submit the two keys:
  # double-queue write race). Smokes: 2 x 3 rounds seed 991, d8 x both
  # masks at ea0p4 es0p4. Flow: mistral_innate_clamp_graph_d8_smoke ->
  # pull + gate -> mistral_innate_clamp_graph_d8.
  mistral_innate_clamp_graph_d8|mistral_innate_clamp_graph_d8_smoke) TARGETS="$WHAT" ;;
  # causal source-exclusion wave (2026-08-18): 48 NEW seed-0 _b0xa_
  # cells -- the exact completed graph-clamp b0 grid (gclump/gscat x
  # ea {0.1,0.2,0.4,1} x es {0,0.05,0.1,0.2,0.4,1}) with
  # SFT_EXCLUDE_CLAMPED=1: cohort A (145 fixed agents) stays fully
  # present, served, pinned and peer-paired, but its rows never enter
  # an SFT batch; the volume stays MATCHED at 723 rows/round (578
  # responsive once + 145 run-seeded responsive duplicates, fixed per
  # run). The completed _b0_ and _d8_ cells are REUSED (no shared
  # tags with any key -- safe to submit alone). Smokes: 2 x 3 rounds
  # seed 991, b0xa x both masks at ea0p4 es0p2. Flow:
  # mistral_clamp_exclude_a_smoke -> pull + gate ->
  # mistral_clamp_exclude_a.
  mistral_clamp_exclude_a|mistral_clamp_exclude_a_smoke) TARGETS="$WHAT" ;;
  # bottom-20% source-impact wave (2026-08-18): 24 NEW jobs -- the
  # bottom cohort (145 LOWEST-innate agents, deterministic ranking)
  # pinned in pop + twin, es=0, ea {0.1,0.2,0.4,1} x seeds {0,42,43}:
  # 12 b0xa (volume-matched source exclusion) + 12 d8 (personal-
  # history ICL). The 12 completed b0 bottom cells REUSE per the
  # audited manifest (tags stay with mistral_innate_clamp_nopeer --
  # zero shared tags here, safe to submit alone). NO smoke: all three
  # paths already carry production gates.
  mistral_bottom20_source_impact) TARGETS="$WHAT" ;;
  # fully-evolving comparison wave (2026-08-18): 48 seed-0 jobs in
  # the NEW pofdevo_ family -- the no-clamp companion to the
  # completed bottom-20% grid (b0 SFT + d8 personal-history x 4 AI
  # gates x 6 social gates; all 723 agents evolve, symmetric peers,
  # matched twin; serving path matched to the fixed runs). Nothing
  # reused, no shared tags, NO smoke.
  mistral_bottom20_evolving) TARGETS="$WHAT" ;;
  # Section-4 three-seed replication (2026-08-19): the seed-0
  # fixed-vs-evolving surface extended to seeds 42/43. Field-level
  # audit (manifest_bottom20_section4_repl.json) reused 40 of the
  # 192 target cells -- 8 tokenless fixed-SFT no-peer originals + 32
  # archived evolving b0 cells (pofdreach/pofdpeer2/pofdgate2d/
  # pofdws2f); 152 NEW jobs queue in two schemas: _fixed 88 (bottom
  # clamp + stubborn peers, seeds ride the queue) + _evo 64 (no
  # clamp env). Reused/completed tags never re-queue (zero shared
  # tags; the exec is idempotent besides). NO smoke: both code paths
  # carry production gates. The umbrella submits both sub-keys.
  mistral_bottom20_section4_repl_fixed|mistral_bottom20_section4_repl_evo) TARGETS="$WHAT" ;;
  mistral_bottom20_section4_repl) TARGETS="mistral_bottom20_section4_repl_fixed mistral_bottom20_section4_repl_evo" ;;
  # feature-endogenization five-seed extension (2026-08-19): seeds 44
  # and 45 for the six established Qwen conditions of the main
  # feature figure = 12 jobs, ALL new per the field-level audit
  # (manifest_feature_endogenization_n5.json: 0 reused / 12 new; the
  # audit self-verifies its want surface against all 18 established
  # cells first). Four queue schemas because the environments differ
  # (natural 6 / frozen 2 / gender-removed 2 / gender-permuted 2);
  # the umbrella submits all four. No established tag appears in any
  # of them, so nothing completed can be re-queued, and the exec is
  # idempotent besides. GPU pinned to the A100 the established runs
  # used. NO smoke: all six paths are production-validated.
  feature_endogenization_n5_nat|feature_endogenization_n5_frozen|feature_endogenization_n5_gd|feature_endogenization_n5_gp) TARGETS="$WHAT" ;;
  feature_endogenization_n5) TARGETS="feature_endogenization_n5_nat feature_endogenization_n5_frozen feature_endogenization_n5_gd feature_endogenization_n5_gp" ;;
  # Qwen gate sweep (2026-08-19): the seed-0 eps_AI x eps_social grid
  # for Qwen2.5-7B and Qwen3-8B at lambda=1 = 60 conceptual cells, of
  # which 30 REUSE completed runs per the field-level audit
  # (manifest_qwen_gate_sweep.json: pofdfam_ 24 + pofdesf_ 4 +
  # pofdw2f_ 1 + pofdws2f_ 1) and 30 are NEW in the pofdqgs_ family.
  # Reused tags are never queued here and the exec is idempotent
  # besides. Qwen3 rides CHAT_THINKING=0 off the queue. NO smoke:
  # both checkpoints and this training path are validated.
  qwen_gate_sweep) TARGETS="$WHAT" ;;
  # Qwen2.5 FULL-ANCHOR (k=1) Section-3 grid (2026-08-19): the
  # completed k=0.2 Qwen2.5 equilibrium grid with INNATE_LAMBDA
  # 0.2 -> 1 and nothing else changed (verified by env diff: exactly
  # one token differs). b0/b1 x ea {.1,.2,.4,1} x es {0,.05,.2},
  # seed 0 = 24 NEW jobs in the pofdfamk1_ family. The anchor rides
  # the established _l1_ tag token; POP_RESET stays off. No existing
  # tag is touched. NO smoke: validated path, new anchor value.
  qwen_k1_grid) TARGETS="$WHAT" ;;
  # es=1 full-peer column of the k=1 grid (2026-08-19), added AFTER
  # the 24-cell base grid was already submitted. SEPARATE key with
  # only the 8 new cells: re-running qwen_k1_grid while those 24 are
  # in flight would double-queue them (the idempotent exec no-ops
  # only COMPLETED runs), so never substitute one key for the other.
  qwen_k1_grid_es1) TARGETS="$WHAT" ;;
  # lambda=1 replication seeds 46-55 (2026-08-20): 10 jobs, the
  # natural lambda=1 feature cell only, to measure the LOCK-IN RATE.
  # Across seeds 0/42/43/44/45 that cell is bimodal -- four transient
  # spikes that decay to the noise floor, one persistent lock-in --
  # and 1-in-5 gives a 95% interval of [0.5%, 72%]. Byte-identical to
  # the existing cells apart from the seed (k stays 0.2 here; this is
  # NOT the k=1 grid). The five completed seeds are never re-queued.
  feature_lambda1_seeds) TARGETS="$WHAT" ;;
  # reproducibility control + seeds 1-4 (2026-08-20): 7 jobs. THREE
  # seed-0 replicates (identical seed and config, distinct _rep<N>_
  # tag) answer a question the lock-in rate depends on -- does a
  # fixed seed reproduce its own trajectory at all? LoRA training is
  # not bitwise deterministic and greedy generation diverges across
  # GPU architectures, so 30 rounds of feedback could amplify
  # numerical noise into a different outcome. A100-pinned so a
  # divergent replicate cannot be blamed on hardware. Plus seeds 1-4
  # as four more independent draws (seed values have no metric
  # structure -- these are NOT "closer to" seed 0).
  feature_lambda1_repro) TARGETS="$WHAT" ;;
  # lambda=0.5 / lambda=0 at the three run identities that so far
  # exist only for lambda=1 -- seed 1 and seed-0 replicates 2 and 3
  # (2026-08-20). 6 jobs, so Figure 3 can draw every lambda curve
  # from the same run set. Same family/environment/A100 pin; only
  # the KL arm and run identity differ.
  feature_lambda_match) TARGETS="$WHAT" ;;
  # Qwen2.5 MECHANISM DIAGNOSTIC -- frozen cells (2026-08-20). The
  # 32-cell paper-regime grid is k {.2,1} x es {0,.05,.2,1} x
  # {perfect prediction, frozen, ordinary SFT, regularized SFT}.
  # Perfect prediction is a CPU oracle (8 cells, sim_perfect_predictor)
  # and all 16 SFT cells already exist, so the only GPU work is the
  # frozen arm: the field-level audit (manifest_qwen_mechanism.json)
  # found 19 reused / 5 new of the 24 GPU cells. 5 jobs.
  # The k=.2/es=0 frozen cell is RERUN rather than reused: the
  # archived match ran on an A100 and its parsed prediction vector
  # differs from the H100 frozen prior in 17 of 723 agents (MAE .0091,
  # max .5). A frozen K=D=0 model never sees the population, so that
  # vector is a CONSTANT the whole grid is compared against, and a
  # mixed-silicon corner would contaminate exactly the k-comparison
  # this diagnostic exists to make. Hence the exact H100-80GB pin and
  # the checker's one-canonical-hash requirement.
  # NO smoke: frozen Qwen2.5 serving is the most-exercised path here.
  qwen_mechanism_frozen) TARGETS="$WHAT" ;;
  # Qwen2.5 at the WU CONSENSUS BOUNDARY (2026-08-20): k=1, W {.5,1},
  # ordinary + regularized SFT, 100 rounds, seed 0 = 4 jobs, H100 only.
  # beta_eff = 1 - (1-W)k, so k=1 ALONE is not a consensus limit (at
  # W=.5 it gives beta_eff=.5); the boundary is k=1 AND W=1, where the
  # pre-peer population equals the served vector.
  # BOTH gates are GENUINELY OPEN via modes, never via the number 1:
  # both are strict inequalities, so eps=1 still REJECTS a distance-1
  # pair -- exactly the extreme pairs the consensus limit is about.
  # AI_GATE_MODE=all_open is the 2026-08-13 mode; PEER_GATE_MODE=
  # all_open is NEW (2026-08-20) and defaults to "threshold", applied
  # AFTER pair selection so archived runs stay byte-identical.
  # RUN THE SMOKE FIRST: this is the first genuinely open PEER path in
  # production. It is a separate key and is NOT one of the four.
  qwen_wu_limit|qwen_wu_limit_smoke) TARGETS="$WHAT" ;;
  # PERSONAL-HISTORY ICL arm of the Wu-boundary experiment (2026-08-21):
  # 2 jobs. Same k=1, W {.5,1}, both gates all_open, 100 rounds, seed 0,
  # H100, matched twin as the four completed trained cells -- only the
  # platform arm differs. d8 = D=8 personal history, K=0 (each agent
  # sees ONLY their own last 8 opinions, no cross-user exemplars),
  # frozen weights, no SFT, no LoRA.
  # It is a THIRD kind of platform: frozen K=D=0 is a static map, b0/b1
  # are parametric retraining, and this is memory without learning --
  # it isolates whether the loop needs weight updates at all.
  # Its OWN key on purpose: the four trained cells are complete, and a
  # separate key means they cannot be touched even as idempotent no-ops.
  # NO smoke: the all-open path is validated by those four, and frozen
  # d8 is the established path from the bottom-20% waves.
  qwen_wu_limit_icl) TARGETS="$WHAT" ;;
  # OBSERVATION-RATE SUBSAMPLING (2026-08-21): does ordinary SFT lean
  # more on the pretrained model when it sees less of the population?
  # The completed Wu-boundary b0 cell unchanged except how many agents'
  # labels reach the optimizer: 14/36/72/181/362/542 (2/5/10/25/50/75%),
  # plus
  # one compute-matched control (72 agents tiled to 723 rows = 181
  # steps, identical compute to full data on 72 distinct agents).
  # 7 jobs. The 100% arm is NOT rerun -- it IS the completed
  # pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100.
  # Serving is untouched: all 723 agents are served every round in
  # every arm; only the SFT batch is cut. Subsets are NESTED across
  # arms (one permutation per round, prefix taken), so "different
  # people" cannot explain a between-arm difference.
  # RUN THE SMOKE FIRST: SFT_SAMPLE_N is a new code path.
  qwen_subsample|qwen_subsample_smoke) TARGETS="$WHAT" ;;
  # SFT TRAINING-DOSE SCOUTS (2026-08-21): three one-round static
  # families asking whether a WEAKER SFT fit leaves the served vector
  # closer to the entering Qwen model -- by optimizer updates (5 jobs),
  # learning rate (4), and LoRA rank (6). 15 jobs total.
  # ONE ROUND on purpose: the 100-round subsample wave showed the loop
  # drives the population to an absorbing constant, after which the
  # projection a is 1 by construction and measures nothing.
  # The shared full-dose endpoint (U=181, LR=5e-5, rank=512) is queued
  # once in the update-dose key; the other two families reuse it.
  # U=0 / LR=0 are the canonical frozen-Qwen vector -- no GPU job.
  # RUN THE SMOKE FIRST: SFT_EPOCHS=0 + SFT_MAX_STEPS is an existing
  # path but SAVE_SFT_ORDER and the step-count provenance are new.
  qwen_sft_update_dose|qwen_sft_lr_dose|qwen_sft_rank_dose|qwen_sft_dose_smoke) TARGETS="$WHAT" ;;
  # adapter KL / soft-decode probe (2026-08-21): ONE job, no training.
  # Reads its adapter list from the three dose config files, so it must
  # run AFTER those 15 cells have finished and saved round0_adapter/.
  # Its own executable -- no trajectory.pt, so nothing to be idempotent
  # about; re-running just overwrites the probe directory.
  qwen_adapter_kl_probe|qwen_adapter_kl_probe_smoke) TARGETS="$WHAT" ;;
  # REFERENCE REPLAY at the Wu boundary (2026-08-22): 4 jobs, the
  # completed Wu-boundary b0 cell with ONE change -- each round the SFT
  # set is rebuilt FULL SIZE [723 rows, 181 steps at batch 4] but only a
  # fraction q of rows carries the LIVE population value; the rest carry
  # the pinned frozen-Qwen prediction for that same agent.
  # q = .10 / .20 / .50 / .75 -> live rows 72 / 145 / 362 / 542.
  # Data volume and compute are held FIXED across arms, so the ladder
  # varies feedback alone -- the confound the subsample wave could not
  # remove, since a smaller sample also took fewer optimizer steps.
  # q = 1 IS NOT QUEUED: all-live rows IS ordinary SFT, i.e. the
  # completed pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100.
  # The frozen vector is REUSED, not re-extracted: REF_REPLAY_REF_RUN
  # names the canonical frozen K=D=0 run [sha256 1674ee5f...da30bb].
  # RUN THE SMOKE FIRST: REF_REPLAY_* is a new code path. The smoke is
  # 3 rounds at q=0.10, the corner that substitutes the most rows.
  ref_replay|ref_replay_smoke)           TARGETS="$WHAT" ;;
  # forward-vs-reverse KL (2026-08-22). Smoke FIRST: reverse is the new
  # path in this wave, and a direction that silently defaulted would
  # still produce a clean-looking run.
  qwen_kl_direction|qwen_kl_direction_smoke) TARGETS="$WHAT" ;;
  # Friedkin-Johnsen robustness wave (2026-08-21): does the b0 vs b1
  # result survive a LINEAR peer operator? 12 trained cells (6 models x
  # 2 arms), 30 rounds, seed 0, FJ_UPDATE_VERSION=wu1. Flow:
  # fj_robustness_smoke (1 cell, 3 rounds) -> gate -> fj_robustness.
  # fj_robustness_frozen extracts the ONE missing zero-shot vector
  # (mistral7b); the other five are reused from archived runs and the
  # FJ replay itself is model-independent and runs locally on CPU.
  # fj_robustness_beta1 is OPTIONAL and deliberately excluded from every
  # umbrella: same models and arms at beta=1, distinct tags, submit only
  # if explicitly wanted.
  # fj_robustness_kl_ladder (2026-08-21): beta=1, lambda in {2,8}, six
  # models. The beta=1 wave showed lambda=0 and lambda=1 both collapse
  # while the frozen lambda->infinity endpoint does NOT, and that at a
  # matched round lambda=1 preserves LESS spread than plain SFT in four
  # of six models -- so the dose response is not monotone and the gap
  # between lambda=1 and infinity is unmeasured. Own key, no umbrella.
  # fj_robustness_seeds / _beta1_seeds (2026-08-22): seeds 42/43/44,
  # 36 jobs each. The seed-0 wave reported COUNTS across models (b0
  # retains more spread than b1 in 6/6 at beta=.5; b1 closer to frozen
  # in 5/6 at beta=1) and a count has no error bar. The FJ operator is
  # deterministic, so a seed perturbs only the learner -- these measure
  # whether the b0-vs-b1 gap exceeds training noise, which is exactly
  # what those counts need. Separate keys: submit one or both.
  fj_robustness|fj_robustness_smoke|fj_robustness_frozen|fj_robustness_beta1|fj_robustness_kl_ladder|fj_robustness_seeds|fj_robustness_beta1_seeds) TARGETS="$WHAT" ;;
  # Jiduan Wu / Pokec replication (2026-08-22): Wu's setup on Wu's own
  # dataset -- Pokec LCC N=2163, the first 1730 rows OBSERVED, the last
  # 433 HELD OUT -- with per-agent alpha_i (peer susceptibility, dataset
  # mean .8909) and beta_i (platform, mean .8890), K_FJ=100 inner steps,
  # T=50 outer rounds, FJ_UPDATE_VERSION=wu1, DATASET=pokec.
  # 82 GPU jobs across nine keys. Flow:
  #   jiduan_pokec_frozen  6 one-round extractions -- the frozen
  #     prediction map on all 2163 agents, which is the reference every
  #     held-out estimand is measured against. NOTHING is reused: the
  #     2026-08-22 scan of runs/pokec_gated_lm found no frozen Pokec run
  #     for any of the six checkpoints.
  #   jiduan_pokec_smoke   1 cell, Qwen2.5 forward-KL, 3 rounds -- gate
  #     it with check_jiduan_pokec.py BEFORE any 50-round job.
  #   jiduan_pokec_prior  12 -- 6 checkpoints x ordinary SFT / forward-KL
  #     SFT, seed 0, exact heterogeneous dataset parameters.
  #   jiduan_pokec_prior_seeds  8 -- seeds 42/43 for Qwen2.5 and OLMo-2,
  #     which is what turns a per-model count into a noise comparison.
  #   jiduan_pokec_lambda_ladder  3 -- Qwen2.5 at lambda .1/.5/10;
  #     lambda 0 and 1 ARE the prior key's b0/b1 cells and are reused
  #     byte-for-byte, never re-queued. The FROZEN model is a separate
  #     endpoint of this family, NOT the lambda -> infinity end of this
  #     ladder: it never trains, so it is not on the dose axis.
  #   jiduan_pokec_icl     6 -- Qwen2.5 and Mistral at frozen no-context,
  #     observed-context K=8 and prediction-history D=8; the b0/b1 cells
  #     of both models are reused from the prior key.
  #   jiduan_pokec_environment  10 -- the c_alpha and c_beta dose cross
  #     for Qwen2.5 x b0/b1/prediction-history. The centre c_alpha=
  #     c_beta=1 is the exact heterogeneous cell and is reused once; at
  #     c_beta=0 the platform cannot reach the population at all, so the
  #     three arms are the SAME trajectory and that cell is queued once.
  #   jiduan_pokec_routing_smoke  12 / _routing_seeds 24 -- paired
  #     treatment/control twins. Both sides carry the SAME routed cohort
  #     (same frac, same cohort seed) and differ only in the served
  #     value, so the checker can prove the two runs routed the identical
  #     agents. NOTHING is reused: a prior cell carries no cohort at all.
  # NEVER co-submit two of these keys expecting reuse to protect you --
  # the reused cells live in jiduan_pokec_prior / _icl and appear in no
  # other config file, so no tag is ever queued twice.
  jiduan_pokec_smoke|jiduan_pokec_prior|jiduan_pokec_prior_seeds|jiduan_pokec_lambda_ladder|jiduan_pokec_icl|jiduan_pokec_environment|jiduan_pokec_routing_smoke|jiduan_pokec_routing_seeds|jiduan_pokec_frozen) TARGETS="$WHAT" ;;
  jiduan_pokec_controls)
    # A DELIBERATE ZERO-JOB KEY. The Wu controls -- perfect prediction
    # m_t = x_t-1, no platform c_beta=0, and the frozen replay -- are
    # linear maps of vectors that already exist. None of them depends on
    # what a language model would say, so a GPU job would only re-derive
    # arithmetic. The key is registered anyway because it is part of the
    # conceptual design and someone will type it; routing it here beats
    # a missing-config error that looks like a generator bug.
    echo "[submit_pofd] jiduan_pokec_controls queues ZERO jobs: the Wu controls are CPU-only." >&2
    echo "[submit_pofd] run: python3 experiments/scripts/cluster_pipelines/jiduan_controls.py" >&2
    echo "[submit_pofd] the analyzer also carries a minimal in-line copy: analyze_jiduan_pokec.py --controls-only" >&2
    exit 2 ;;
  # Figure-2 family-prior scout (2026-08-17): six checkpoints
  # (Qwen2.5-7B / Qwen3-8B thinking-off / OLMo-2 / OLMo-3 /
  # Mistral-7B / Ministral-8B) x b0/b0p5/b1/k0 x ea1 threshold x
  # es {0.05,0.2}, seed 0 = 48 conceptual cells; the field-audited
  # manifest (manifest_fig2_family_prior.json) reuses completed cells
  # and only the missing ones queue. BASE_MODEL/CHAT_THINKING/mem/
  # disk/PPL_BATCH ride the queue; offline lustre cache -- PRE-CACHE
  # Qwen/Qwen2.5-7B-Instruct there first. Smokes: 3 x 3 rounds
  # (b1 es0p2, one per NEW checkpoint). Flow: fig2_family_prior_smoke
  # -> pull + gate -> fig2_family_prior_scout.
  fig2_family_prior_scout|fig2_family_prior_smoke) TARGETS="$WHAT" ;;
  # exactly the 8 qwen7b scout cells that died before
  # Qwen/Qwen2.5-7B-Instruct reached the lustre cache (SAME tags as
  # fig2_family_prior_scout -- NEVER co-submit while they are queued
  # there; PRE-CACHE the checkpoint first)
  fig2_family_prior_qwen_retry) TARGETS="$WHAT" ;;
  # beta extension (2026-08-18): 18 seed-0 cells -- 6 checkpoints x
  # forward-KL beta {2,4,8} (_b2_/_b4_/_b8_ tags) at ea1/es0p05,
  # everything else copies the completed fam b1 configuration. The
  # analyzer (analyze_fig2_family_prior.py) selects the SMALLEST beta
  # with >=5/6 own-family W1 matching; submit ONLY that beta's
  # confirmation key (12 jobs: 6 checkpoints x seeds 42/43).
  fig2_family_prior_beta_scout) TARGETS="$WHAT" ;;
  fig2_family_prior_b2_confirm|fig2_family_prior_b4_confirm|fig2_family_prior_b8_confirm) TARGETS="$WHAT" ;;
  # Section-3 family-gate ablation (2026-08-18): 36 seed-0 cells --
  # 6 checkpoints x b0/b1 x ea {0.1,0.2,0.4} at es0p05, the exact
  # fam-scout code path (same pofdfam_ family + sub surface). The 12
  # completed ea1 scout cells REUSE per manifest_fam_gate_ablation
  # .json (tags stay with fig2_family_prior_scout -- no shared tags
  # here, safe to submit alone). NO smoke.
  fam_gate_ablation) TARGETS="$WHAT" ;;
  # social-gate extension (2026-08-18): 84 new seed-0 cells -- the
  # 48 es0 cells (all gates) + the 36 es0p2 cells below ea1 for the
  # 6 checkpoints x b0/b1 surface. es0p05 lives in fam_gate_ablation
  # and the ea1 es0p2 cells reuse (incl. the inherited gate2d
  # occupant) -- no shared tags with any key, safe to submit alone,
  # but NEVER co-submit while fam_gate_ablation is queued IF that
  # key was regenerated (it was not: tags are disjoint by es). NO
  # smoke.
  fam_gate_social) TARGETS="$WHAT" ;;
  # zero-shot prior screen (2026-08-17): 4 one-round frozen probes, one
  # per candidate checkpoint (Qwen3-8B thinking-off / Olmo-3-7B-Instruct
  # / Ministral-8B / Mistral-Nemo) on the paper's 723 Action profiles,
  # seed 0, greedy, EPS_AI=0 strict threshold (no contacts, opinions
  # untouched), SAVE_RAW_GEN=1. NEW pofdzsprior_ family, no shared
  # tags. PRE-CACHE all four checkpoints into
  # /lustre/fast/fast/gsmithline/hf_cache first (HF_HUB_OFFLINE=1).
  zsprior_screen) TARGETS="$WHAT" ;;
  # Section-3 retention table (2026-08-22): 46 NEW 100-round GPU cells --
  # 50 conceptual (2 checkpoints x 3 environments x {sft + forward KL
  # lambda .1/.5/1/2/4/8}, plus reverse KL lambda {1,8} in the two k=1
  # environments) minus 4 archived Qwen2.5 QWU cells that already ARE
  # those cells and are declared in S3_REUSED, never queued. Both gates
  # genuinely open, seed 0, movielens Action, 723 agents, fresh LoRA
  # r512 every round, H100-80GB only.
  # KL_DIRECTION RIDES THE QUEUE on this key (it mixes forward and
  # reverse), so SMOKE FIRST: section3_retention_smoke is one 3-round
  # Qwen3-8B reverse lambda=1 cell at (W=1, k=1). A direction that
  # silently defaulted, or a Qwen3 chat template that broke
  # completion-only masking, would still produce a clean-looking run.
  # PRE-CACHE Qwen/Qwen3-8B and Qwen/Qwen2.5-7B-Instruct into
  # /lustre/fast/fast/gsmithline/hf_cache first (HF_HUB_OFFLINE=1).
  section3_retention|section3_retention_smoke) TARGETS="$WHAT" ;;
  section3_peer_sweeps|section3_peer_sweeps_smoke) TARGETS="$WHAT" ;;
  section3_memory|section3_memory_smoke) TARGETS="$WHAT" ;;
  section3_label_mix|section3_label_mix_smoke) TARGETS="$WHAT" ;;
  section3_lambda_fill|section3_lambda_fill_smoke) TARGETS="$WHAT" ;;
  # Section-4 CORRECTED-GATE wave (2026-08-24): the bottom-20 fixed-vs-
  # evolving Section 4 experiment re-run with AI_GATE_REFERENCE=anchor
  # (|m - x'| <= eps_AI on the anchored opinion; config population_update
  # = nested_ai_anchored_then_social_v2). 72 production jobs = 36 fixed +
  # 36 evolving. TWO queue schemas under one umbrella name, exactly as the
  # B20R replication does it: the fixed sub carries the clamp env and the
  # evolving sub must carry NONE, so they cannot share a schema. The
  # umbrella targets submit both halves. New pofds4g_ tags -- disjoint
  # from every archived Section-4 tag, so nothing is overwritten and the
  # idempotent exec no-ops anything already complete.
  # Flow: section4_gate_anch2_smoke (4 x 3 rounds) -> pull + gate with
  # check_section4_gate.py --smoke -> section4_gate_anch2.
  section4_gate_anch2_fixed|section4_gate_anch2_evo) TARGETS="$WHAT" ;;
  section4_gate_anch2_smoke_fixed|section4_gate_anch2_smoke_evo) TARGETS="$WHAT" ;;
  section4_gate_anch2)       TARGETS="section4_gate_anch2_fixed section4_gate_anch2_evo" ;;
  section4_gate_anch2_smoke) TARGETS="section4_gate_anch2_smoke_fixed section4_gate_anch2_smoke_evo" ;;
  # FIGURE-6 GRID (2026-08-25): the corrected-gate Section 4 on the
  # matched 4 x 4 gate grid ea, es in {0,.1,.3,1}; 146 GPU jobs (72 fixed
  # + 74 evolving, incl. 2 ea=0 witnesses; the other 46 ea=0 cells are
  # twin-derived). REUSED = 0: nothing corrected-gate exists on disk.
  # SHARES 24 tags (ea=1, es in {0,1}) with the UNRUN section4_gate_anch2
  # key -- NEVER co-submit the two; whichever runs first no-ops the other.
  # Flow: section4_gate_anch2_fig6_smoke (4 x 3 rounds at ea=.1 es=.3)
  # -> pull + gate with check_section4_gate.py --wave fig6 --smoke ->
  # section4_gate_anch2_fig6 -> analyze -> commit the extension manifest
  # if any pair is unsettled -> section4_gate_anch2_fig6_ext.
  section4_gate_anch2_fig6_fixed|section4_gate_anch2_fig6_evo) TARGETS="$WHAT" ;;
  # seed-staged release: identical rows/tags/env, partitioned by seed;
  # gate seed 0 with check_section4_gate.py --wave fig6 --seeds 0 first
  section4_gate_anch2_fig6_s0_fixed|section4_gate_anch2_fig6_s0_evo|section4_gate_anch2_fig6_s42_43_fixed|section4_gate_anch2_fig6_s42_43_evo) TARGETS="$WHAT" ;;
  section4_gate_anch2_fig6_s0)     TARGETS="section4_gate_anch2_fig6_s0_fixed section4_gate_anch2_fig6_s0_evo" ;;
  section4_gate_anch2_fig6_s42_43) TARGETS="section4_gate_anch2_fig6_s42_43_fixed section4_gate_anch2_fig6_s42_43_evo" ;;
  section4_gate_anch2_fig6_smoke_fixed|section4_gate_anch2_fig6_smoke_evo) TARGETS="$WHAT" ;;
  section4_gate_anch2_fig6_ext_fixed|section4_gate_anch2_fig6_ext_evo) TARGETS="$WHAT" ;;
  section4_gate_anch2_fig6)       TARGETS="section4_gate_anch2_fig6_fixed section4_gate_anch2_fig6_evo" ;;
  section4_gate_anch2_fig6_smoke) TARGETS="section4_gate_anch2_fig6_smoke_fixed section4_gate_anch2_fig6_smoke_evo" ;;
  section4_gate_anch2_fig6_ext)   TARGETS="section4_gate_anch2_fig6_ext_fixed section4_gate_anch2_fig6_ext_evo" ;;
  # FIGURE-4 REPLICATION + CONVERGENCE (2026-08-24): the exact condition
  # Figure 4 displays (forward-KL lambda=1, ea1, es0p05, W=0.5, lam=0.2)
  # on all six checkpoints at seeds 0/42/43 for 30 rounds = 18 jobs. New
  # pofdf4r_ tags: the displayed 30-round pofdfam_ cells are never
  # re-queued. NO smoke -- the b1 envelope is already production-validated
  # on every checkpoint by the completed family-prior wave; only the
  # seeds change. ~2.5h/job (measured on the cells it replicates).
  fig4_family_prior_repl30) TARGETS="$WHAT" ;;
  # REDESIGNED FIGURE 3, UNIFIED 108-CELL GRID (2026-08-24 v2): the
  # missing (beta x gamma x lambda) cells of
  # reference_retention_memory_equilibrium. 63 NEW GPU jobs (56 at beta
  # {.25,.75} x 4 gammas x 7 finite lambdas + 7 at beta=.5, gamma=0); 28
  # further GPU cells are already on disk (pofdmem_ 9 + pofdlam_ 16 +
  # pofdps_ 3, audited tag by tag) and are NEVER re-queued. The rest of
  # the figure needs no GPU at all: beta=0 is twin_raw, already in every
  # run, and lambda=inf is the CPU offline replay
  # (experiments/scripts/cluster_pipelines/gen_fig3_frozen_replays.sh).
  # fig3_full_loop_ext runs TARGETED 60/100-round horizon extensions for
  # cells the analyzer marks unsettled, driven by the committed
  # fig3_extension_request.json (the generator validates every entry).
  # Flow: fig3_full_loop_smoke (1 x 3 rounds, beta=.25 gamma=0 lambda=8)
  # -> pull + gate with check_fig3_full_loop.py --smoke -> fig3_full_loop
  # -> analyze -> commit updated fig3_extension_request.json if needed ->
  # fig3_full_loop_ext.
  fig3_full_loop|fig3_full_loop_smoke|fig3_full_loop_ext) TARGETS="$WHAT" ;;
  # Open-gate cross-model consensus comparison for Section 3: six models x
  # three seeds, all freshly run under one provenance (18 production jobs).
  # Mistral-7B strict-parse rerun (provenance exemption, 3 jobs + smoke):
  # the legacy parser read Mistral's ".64 (" as 1.0; see the S3M_RERUN note.
  section3_model_equilibria|section3_model_equilibria_smoke|section3_model_equilibria_mistral_rerun|section3_model_equilibria_mistral_rerun_smoke) TARGETS="$WHAT" ;;
  # RECURSIVE UPDATE-DOSE (2026-08-25): plain SFT at the Section 3
  # no-memory/full-adoption surface with ONLY optimizer-step frequency
  # varied via SFT_GRAD_ACCUM (u1/u5/u19; u181 = the archived pofdps_
  # SFT cell, reused; pp control = archived CPU artifact). 3 jobs + 1
  # two-round smoke. Flow: smoke -> check_update_dose.py --smoke -> main.
  sft_update_dose_loop|sft_update_dose_loop_smoke) TARGETS="$WHAT" ;;
  # FIGURE-4 ANCHOR TRADE-OFF (2026-08-25): beta = W_PLAT {0,.25,.5,.75,1}
  # x gamma = INNATE_LAMBDA {1,.5,.2,0} x es {.05,.2} x {Qwen2.5-7B,
  # Qwen3-8B} under AI_GATE_REFERENCE=anchor (anch2), AI gate all_open
  # (the threshold form is the STRICT |m - x'| < eps_AI), peer gate
  # threshold, ONE Deffuant sweep per round, forward-KL lambda=2, fresh
  # r512 LoRA, 30 rounds, seed 0. 80 cells = 60 GPU jobs + 20 exact
  # algebraic dups (beta=1: gamma drops out; beta=0: population == twin,
  # Qwen3-8B only). PARSE_MODE=strict SAVE_RAW_GEN=1 WITH_TWIN=1
  # TRAIN_WITNESS=1 on every job. fig4_anchor_tradeoff_zsprior is the
  # Qwen2.5-7B zero-shot prior (1 frozen round, zsprior_screen env);
  # the smoke umbrella queues it beside the two 3-round smoke cells.
  # Flow: fig4_anchor_tradeoff_smoke -> check_fig4_anchor.py --smoke ->
  # fig4_anchor_tradeoff -> pull + check_fig4_anchor.py -> frozen
  # replays -> analyze -> commit fig4_anchor_extension_request.json if
  # needed -> fig4_anchor_tradeoff_ext.
  fig4_anchor_tradeoff|fig4_anchor_tradeoff_ext|fig4_anchor_tradeoff_zsprior) TARGETS="$WHAT" ;;
  fig4_anchor_tradeoff_smoke) TARGETS="fig4_anchor_tradeoff_smoke fig4_anchor_tradeoff_zsprior" ;;
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
  *) echo "usage: submit_pofd_sweep.sh <BID> smoke|qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac|qwen7b_bp[_smoke]|qwen7b_icl[_smoke]|qwen7b_dpo[_smoke]|qwen7b_dpo_ci1|qwen7b_dpo_mr[_smoke]|sft_icl_reach[_smoke|_baseline]|sft_icl_reach_qwen|sft_icl_reach_olmo|sft_icl_reach_mistral|sft_icl_reach_s0|sft_k0_nopeer[_smoke]|sft_k0_nopeer_qwen|sft_k0_nopeer_olmo|sft_k0_nopeer_mistral|sft_icl_nopeer_grid3[_smoke]|sft_icl_nopeer_grid3_qwen|sft_icl_nopeer_grid3_olmo|sft_icl_nopeer_grid3_mistral|sft_icl_peer02[_smoke]|qwen7b_sft_icl_peer02|olmo7b_sft_icl_peer02|mistral7b_sft_icl_peer02|qwen7b_dpo_ci2|qwen7b_dpo_ci3|qwen7b_w[_smoke]|qwen7b_wdpo[_smoke]|qwen7b_wdpon|qwen7b_ws[_smoke]|qwen7b_w2[_smoke]|qwen7b_ws2[_smoke]|qwen7b_w1f[_smoke]|qwen7b_w2f|qwen7b_w2fpt[_smoke]|qwen7b_ws2f|qwen7b_esf[_repl]|qwen7b_fex|qwen7b_pf2|qwen7b_pfs2[_smoke]|qwen7b_fe[s]|qwen7b_fegd|qwen7b_fegp|qwen7b_fef|qwen7b_fe2|qwen7b_fes2|qwen7b_fegd2|qwen7b_fegp2|qwen7b_fef2|qwen7b_fer|qwen7b_fesr|qwen7b_fegdr|qwen7b_fegpr|qwen7b_fec[_smoke]|qwen7b_fesc|qwen7b_fegdc|qwen7b_fegpc|qwen7b_fej[_smoke]|qwen7b_fesj|qwen7b_fegdj|qwen7b_fegpj|qwen7b_fei|qwen7b_feik|qwen7b_feigd|qwen7b_feigp|qwen7b_tch|qwen7b_tfe[_smoke]|qwen7b_tfem|qwen7b_tfe_ci|qwen7b_tfegd|qwen7b_tfegp|qwen7b_tchr|qwen7b_tfer[_smoke]|qwen7b_icl2|qwen7b_icls2[_smoke]|qwen7b_wdpo2|qwen7b_wdpos2[_smoke]|qwen7b_wdpo2e|qwen7b_wdpos2e[_smoke]|qwen7b_wdpoe|mistral7b_ws2f[_smoke]|mistral7b_cube[_s0]|mistral7b_cube_repl|olmo7b_dpo[_smoke]|olmo7b_icl[_smoke]|olmo7b_w1f|olmo7b_w2[_smoke]|olmo7b_ws2[_smoke]|olmo7b_w2f|olmo7b_ws2f|olmo7b_rex|olmo7b_w2fx|olmo7b_ws2fx|olmo7b_icl2|olmo7b_icls2|qwen7b_icls2x|olmo7b_icls2x|icls2x|olmo7brom_fe[s]|olmo7brom_fef|olmo7brom_fe_smoke|qwen7b_cube|olmo7b_cube|qwen7b_replay1|qwen7b_budget[_smoke]|olmo7b_budget|qwen7b_iclf[_smoke]|qwen7b_ctf[_smoke]|qwen7b_iclx[_smoke]|qwen7b_iclx_retry|qwen7b_seedcore[_smoke]|olmo7b_seedcore|qwen_olmo_seedcore|qwen7b_finalfill_sfticl|qwen7b_finalfill_replay|qwen7b_finalfill_corners|qwen_olmo_finalfill|mistral_sft_icl_gate2d[_smoke]|sft_icl_ctxgrid[_smoke|_debug]|sft_icl_ctxgrid_qwen|sft_icl_ctxgrid_olmo|sft_icl_ctxgrid_mistral|fig2_provider|fig2_provider_olmo|fig2_provider_mistral|mistral_innate_clamp_nopeer[_smoke]|mistral_innate_clamp_graph_s0|mistral_innate_clamp_graph_smoke|mistral_innate_clamp_graph_d8[_smoke]|mistral_clamp_exclude_a[_smoke]|mistral_bottom20_source_impact|mistral_bottom20_evolving|mistral_bottom20_section4_repl[_fixed|_evo]|feature_endogenization_n5[_nat|_frozen|_gd|_gp]|qwen_gate_sweep|qwen_k1_grid[_es1]|feature_lambda1_seeds|feature_lambda1_repro|feature_lambda_match|qwen_mechanism_frozen|qwen_wu_limit[_smoke|_icl]|qwen_subsample[_smoke]|qwen_sft_update_dose|qwen_sft_lr_dose|qwen_sft_rank_dose|qwen_sft_dose_smoke|qwen_adapter_kl_probe[_smoke]|qwen_kl_direction[_smoke]|ref_replay[_smoke]|fj_robustness[_smoke|_frozen|_beta1|_kl_ladder|_seeds|_beta1_seeds]|jiduan_pokec_smoke|jiduan_pokec_controls|jiduan_pokec_prior[_seeds]|jiduan_pokec_lambda_ladder|jiduan_pokec_icl|jiduan_pokec_environment|jiduan_pokec_routing_smoke|jiduan_pokec_routing_seeds|jiduan_pokec_frozen|fig2_family_prior_scout|fig2_family_prior_smoke|fig2_family_prior_qwen_retry|fig2_family_prior_beta_scout|fig2_family_prior_b2_confirm|fig2_family_prior_b4_confirm|fig2_family_prior_b8_confirm|zsprior_screen|section3_retention[_smoke]|section3_peer_sweeps[_smoke]|section3_memory[_smoke]|section3_label_mix[_smoke]|section3_lambda_fill[_smoke]|section4_gate_anch2[_smoke][_fixed|_evo]|section4_gate_anch2_fig6[_smoke|_ext|_s0|_s42_43][_fixed|_evo]|fig4_family_prior_repl30|fig3_full_loop[_smoke|_ext]|sft_update_dose_loop[_smoke]|fig4_anchor_tradeoff[_smoke|_ext|_zsprior]|full" >&2; exit 2 ;;
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
