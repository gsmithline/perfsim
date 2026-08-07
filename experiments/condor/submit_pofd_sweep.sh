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

BID="${1:?usage: submit_pofd_sweep.sh <BID> smoke|qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac|qwen7b_bp[_smoke]|qwen7b_icl[_smoke]|qwen7b_dpo[_smoke]|qwen7b_w[_smoke]|qwen7b_wdpo[_smoke]|qwen7b_wdpon|qwen7b_ws[_smoke]|qwen7b_w2[_smoke]|qwen7b_ws2[_smoke]|qwen7b_w1f[_smoke]|qwen7b_w2f|qwen7b_w2fpt[_smoke]|qwen7b_ws2f|qwen7b_esf[_repl]|qwen7b_fex|qwen7b_pf2|qwen7b_pfs2[_smoke]|qwen7b_fe[s]|qwen7b_fegd|qwen7b_fegp|qwen7b_fef|qwen7b_fe2|qwen7b_fes2|qwen7b_fegd2|qwen7b_fegp2|qwen7b_fef2|qwen7b_fer|qwen7b_fesr|qwen7b_fegdr|qwen7b_fegpr|qwen7b_fec[_smoke]|qwen7b_fesc|qwen7b_fegdc|qwen7b_fegpc|qwen7b_fej[_smoke]|qwen7b_fesj|qwen7b_fegdj|qwen7b_fegpj|qwen7b_fei|qwen7b_feik|qwen7b_feigd|qwen7b_feigp|qwen7b_tch|qwen7b_tfe[_smoke]|qwen7b_tfem|qwen7b_tfegd|qwen7b_tfegp|qwen7b_tchr|qwen7b_tfer[_smoke]|qwen7b_icl2|qwen7b_icls2[_smoke]|qwen7b_wdpo2|qwen7b_wdpos2[_smoke]|qwen7b_wdpo2e|qwen7b_wdpos2e[_smoke]|qwen7b_wdpoe|olmo7b_dpo[_smoke]|olmo7b_icl[_smoke]|olmo7b_w1f|olmo7b_w2[_smoke]|olmo7b_ws2[_smoke]|olmo7b_w2f|olmo7b_ws2f|olmo7b_rex|olmo7b_w2fx|olmo7b_ws2fx|olmo7b_icl2|olmo7b_icls2|qwen7b_icls2x|olmo7b_icls2x|icls2x|olmo7brom_fe[s]|olmo7brom_fef|olmo7brom_fe_smoke|qwen7b_cube|olmo7b_cube|qwen7b_replay1|qwen7b_budget[_smoke]|olmo7b_budget|qwen7b_iclf[_smoke]|qwen7b_ctf[_smoke]|qwen7b_iclx[_smoke]|qwen7b_iclx_retry|qwen7b_seedcore[_smoke]|olmo7b_seedcore|qwen_olmo_seedcore|full}"
WHAT="${2:?usage: submit_pofd_sweep.sh <BID> smoke|qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac|qwen7b_bp[_smoke]|qwen7b_icl[_smoke]|qwen7b_dpo[_smoke]|qwen7b_w[_smoke]|qwen7b_wdpo[_smoke]|qwen7b_wdpon|qwen7b_ws[_smoke]|qwen7b_w2[_smoke]|qwen7b_ws2[_smoke]|qwen7b_w1f[_smoke]|qwen7b_w2f|qwen7b_w2fpt[_smoke]|qwen7b_ws2f|qwen7b_esf[_repl]|qwen7b_fex|qwen7b_pf2|qwen7b_pfs2[_smoke]|qwen7b_fe[s]|qwen7b_fegd|qwen7b_fegp|qwen7b_fef|qwen7b_fe2|qwen7b_fes2|qwen7b_fegd2|qwen7b_fegp2|qwen7b_fef2|qwen7b_fer|qwen7b_fesr|qwen7b_fegdr|qwen7b_fegpr|qwen7b_fec[_smoke]|qwen7b_fesc|qwen7b_fegdc|qwen7b_fegpc|qwen7b_fej[_smoke]|qwen7b_fesj|qwen7b_fegdj|qwen7b_fegpj|qwen7b_fei|qwen7b_feik|qwen7b_feigd|qwen7b_feigp|qwen7b_tch|qwen7b_tfe[_smoke]|qwen7b_tfem|qwen7b_tfegd|qwen7b_tfegp|qwen7b_tchr|qwen7b_tfer[_smoke]|qwen7b_icl2|qwen7b_icls2[_smoke]|qwen7b_wdpo2|qwen7b_wdpos2[_smoke]|qwen7b_wdpo2e|qwen7b_wdpos2e[_smoke]|qwen7b_wdpoe|olmo7b_dpo[_smoke]|olmo7b_icl[_smoke]|olmo7b_w1f|olmo7b_w2[_smoke]|olmo7b_ws2[_smoke]|olmo7b_w2f|olmo7b_ws2f|olmo7b_rex|olmo7b_w2fx|olmo7b_ws2fx|olmo7b_icl2|olmo7b_icls2|qwen7b_icls2x|olmo7b_icls2x|icls2x|olmo7brom_fe[s]|olmo7brom_fef|olmo7brom_fe_smoke|qwen7b_cube|olmo7b_cube|qwen7b_replay1|qwen7b_budget[_smoke]|olmo7b_budget|qwen7b_iclf[_smoke]|qwen7b_ctf[_smoke]|qwen7b_iclx[_smoke]|qwen7b_iclx_retry|qwen7b_seedcore[_smoke]|olmo7b_seedcore|qwen_olmo_seedcore|full}"
REPO="${REPO:-/home/gsmithline/perfsim}"
cd "$REPO"

mkdir -p experiments/condor/logs runs/pokec_gated_lm
chmod +x experiments/condor/run_one_pokec_gated.sh \
         experiments/condor/run_one_pokec_gated_idempotent.sh

python3 experiments/condor/gen_pofd_sweep.py --verify

case "$WHAT" in
  smoke)                                 TARGETS="smoke" ;;
  qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac)   TARGETS="$WHAT" ;;
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
  qwen7b_tch|qwen7b_tfem|qwen7b_tfegd|qwen7b_tfegp|qwen7b_tfe_smoke) TARGETS="$WHAT" ;;
  qwen7b_tfe)                            TARGETS="qwen7b_tfem qwen7b_tfegd qwen7b_tfegp" ;;
  qwen7b_tchr|qwen7b_tfer|qwen7b_tfer_smoke) TARGETS="$WHAT" ;;
  qwen7b_icl2|qwen7b_icls2|qwen7b_icls2_smoke)   TARGETS="$WHAT" ;;
  qwen7b_wdpo2|qwen7b_wdpos2|qwen7b_wdpos2_smoke)   TARGETS="$WHAT" ;;
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
  qwen7b_esfn|olmo7b_esfn)
    # SUPERSEDED 2026-08-05 by the full parameter cube: every esfn cell is a
    # cube cell with a BYTE-IDENTICAL tag, so co-submitting both keys would
    # double-queue 52 tags into the same run dirs (a write race, not a no-op:
    # the idempotent exec only skips COMPLETED trajectories).
    echo "[submit_pofd] ${WHAT} is SUPERSEDED by ${WHAT%_esfn}_cube -- not submitting." >&2
    echo "[submit_pofd] use: bash experiments/condor/submit_pofd_sweep.sh ${BID} ${WHAT%_esfn}_cube" >&2
    exit 2 ;;
  full)                                  TARGETS="qwen7b gemma12b olmo7b" ;;
  *) echo "usage: submit_pofd_sweep.sh <BID> smoke|qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac|qwen7b_bp[_smoke]|qwen7b_icl[_smoke]|qwen7b_dpo[_smoke]|qwen7b_w[_smoke]|qwen7b_wdpo[_smoke]|qwen7b_wdpon|qwen7b_ws[_smoke]|qwen7b_w2[_smoke]|qwen7b_ws2[_smoke]|qwen7b_w1f[_smoke]|qwen7b_w2f|qwen7b_w2fpt[_smoke]|qwen7b_ws2f|qwen7b_esf[_repl]|qwen7b_fex|qwen7b_pf2|qwen7b_pfs2[_smoke]|qwen7b_fe[s]|qwen7b_fegd|qwen7b_fegp|qwen7b_fef|qwen7b_fe2|qwen7b_fes2|qwen7b_fegd2|qwen7b_fegp2|qwen7b_fef2|qwen7b_fer|qwen7b_fesr|qwen7b_fegdr|qwen7b_fegpr|qwen7b_fec[_smoke]|qwen7b_fesc|qwen7b_fegdc|qwen7b_fegpc|qwen7b_fej[_smoke]|qwen7b_fesj|qwen7b_fegdj|qwen7b_fegpj|qwen7b_fei|qwen7b_feik|qwen7b_feigd|qwen7b_feigp|qwen7b_tch|qwen7b_tfe[_smoke]|qwen7b_tfem|qwen7b_tfegd|qwen7b_tfegp|qwen7b_tchr|qwen7b_tfer[_smoke]|qwen7b_icl2|qwen7b_icls2[_smoke]|qwen7b_wdpo2|qwen7b_wdpos2[_smoke]|qwen7b_wdpo2e|qwen7b_wdpos2e[_smoke]|qwen7b_wdpoe|olmo7b_dpo[_smoke]|olmo7b_icl[_smoke]|olmo7b_w1f|olmo7b_w2[_smoke]|olmo7b_ws2[_smoke]|olmo7b_w2f|olmo7b_ws2f|olmo7b_rex|olmo7b_w2fx|olmo7b_ws2fx|olmo7b_icl2|olmo7b_icls2|qwen7b_icls2x|olmo7b_icls2x|icls2x|olmo7brom_fe[s]|olmo7brom_fef|olmo7brom_fe_smoke|qwen7b_cube|olmo7b_cube|qwen7b_replay1|qwen7b_budget[_smoke]|olmo7b_budget|qwen7b_iclf[_smoke]|qwen7b_ctf[_smoke]|qwen7b_iclx[_smoke]|qwen7b_iclx_retry|qwen7b_seedcore[_smoke]|olmo7b_seedcore|qwen_olmo_seedcore|full" >&2; exit 2 ;;
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
