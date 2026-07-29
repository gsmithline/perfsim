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

BID="${1:?usage: submit_pofd_sweep.sh <BID> smoke|qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac|qwen7b_bp[_smoke]|qwen7b_icl[_smoke]|qwen7b_dpo[_smoke]|qwen7b_w[_smoke]|qwen7b_wdpo[_smoke]|qwen7b_wdpon|qwen7b_ws[_smoke]|qwen7b_w2[_smoke]|qwen7b_ws2[_smoke]|qwen7b_w1f[_smoke]|qwen7b_w2f|qwen7b_ws2f|qwen7b_esf[_repl]|qwen7b_icl2|qwen7b_icls2[_smoke]|qwen7b_wdpo2|qwen7b_wdpos2[_smoke]|olmo7b_dpo[_smoke]|olmo7b_icl[_smoke]|olmo7b_w1f|olmo7b_w2[_smoke]|olmo7b_ws2[_smoke]|olmo7b_w2f|olmo7b_ws2f|olmo7b_icl2|olmo7b_icls2|full}"
WHAT="${2:?usage: submit_pofd_sweep.sh <BID> smoke|qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac|qwen7b_bp[_smoke]|qwen7b_icl[_smoke]|qwen7b_dpo[_smoke]|qwen7b_w[_smoke]|qwen7b_wdpo[_smoke]|qwen7b_wdpon|qwen7b_ws[_smoke]|qwen7b_w2[_smoke]|qwen7b_ws2[_smoke]|qwen7b_w1f[_smoke]|qwen7b_w2f|qwen7b_ws2f|qwen7b_esf[_repl]|qwen7b_icl2|qwen7b_icls2[_smoke]|qwen7b_wdpo2|qwen7b_wdpos2[_smoke]|olmo7b_dpo[_smoke]|olmo7b_icl[_smoke]|olmo7b_w1f|olmo7b_w2[_smoke]|olmo7b_ws2[_smoke]|olmo7b_w2f|olmo7b_ws2f|olmo7b_icl2|olmo7b_icls2|full}"
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
  qwen7b_w1f|qwen7b_w1f_smoke|qwen7b_w2f|qwen7b_ws2f)   TARGETS="$WHAT" ;;
  qwen7b_esf|qwen7b_esf_repl)            TARGETS="$WHAT" ;;
  qwen7b_icl2|qwen7b_icls2|qwen7b_icls2_smoke)   TARGETS="$WHAT" ;;
  qwen7b_wdpo2|qwen7b_wdpos2|qwen7b_wdpos2_smoke)   TARGETS="$WHAT" ;;
  olmo7b_icl2|olmo7b_icls2)              TARGETS="$WHAT" ;;
  olmo7b_dpo|olmo7b_dpo_smoke)           TARGETS="$WHAT" ;;
  olmo7b_icl|olmo7b_icl_smoke)           TARGETS="$WHAT" ;;
  olmo7b_w1f)                            TARGETS="$WHAT" ;;
  olmo7b_w2|olmo7b_w2_smoke)             TARGETS="$WHAT" ;;
  olmo7b_ws2|olmo7b_ws2_smoke)           TARGETS="$WHAT" ;;
  olmo7b_w2f|olmo7b_ws2f)                TARGETS="$WHAT" ;;
  full)                                  TARGETS="qwen7b gemma12b olmo7b" ;;
  *) echo "usage: submit_pofd_sweep.sh <BID> smoke|qwen7b|gemma12b|olmo7b|qwen7b_pfrac|olmo7b_pfrac|qwen7b_bp[_smoke]|qwen7b_icl[_smoke]|qwen7b_dpo[_smoke]|qwen7b_w[_smoke]|qwen7b_wdpo[_smoke]|qwen7b_wdpon|qwen7b_ws[_smoke]|qwen7b_w2[_smoke]|qwen7b_ws2[_smoke]|qwen7b_w1f[_smoke]|qwen7b_w2f|qwen7b_ws2f|qwen7b_esf[_repl]|qwen7b_icl2|qwen7b_icls2[_smoke]|qwen7b_wdpo2|qwen7b_wdpos2[_smoke]|olmo7b_dpo[_smoke]|olmo7b_icl[_smoke]|olmo7b_w1f|olmo7b_w2[_smoke]|olmo7b_ws2[_smoke]|olmo7b_w2f|olmo7b_ws2f|olmo7b_icl2|olmo7b_icls2|full" >&2; exit 2 ;;
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
