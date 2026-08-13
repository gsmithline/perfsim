#!/usr/bin/env bash
# MATCHED-RANDOMNESS DPO PAIR wrapper (2026-08-13): one Condor job runs the
# CLOSED arm (bank writer) then the OPEN arm (bank reader) sequentially on
# the SAME allocated GPU, then validates the paired invariants.
#
#   $1 = the CLOSED arm tag (contains "_closed_"); the open tag and the
#        shared bank directory are derived from it. Remaining args are the
#        standard run_one_pokec_gated.sh columns.
#   env: DPO_BANK_SEED (from the queue), DPO_TRAIN_SEED (fixed in the .sub),
#        plus the usual DPO_* knobs. RLHF_FEEDBACK and DPO_BANK_MODE are set
#        HERE per arm -- never in the .sub.
#
# Idempotence: the pair counts as complete ONLY when both arm trajectories,
# the shared bank, and the paired validation metadata (pair_meta.json,
# written by check_dpo_pair.py on success) all exist. Partial pairs resume:
# a finished closed arm + bank is reused, only the missing pieces run.
set -eo pipefail
TAG_CLOSED="$1"; shift
case "$TAG_CLOSED" in
  *_closed_*) ;;
  *) echo "[dpo_pair] tag $TAG_CLOSED lacks _closed_" >&2; exit 2 ;;
esac
TAG_OPEN="${TAG_CLOSED/_closed_/_open_}"
REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
RUNS="$REPO/runs/pokec_gated_lm"
BANK="$RUNS/${TAG_CLOSED/_closed_/_bank_}"
N_ROUNDS="${N_ROUNDS:-30}"
: "${DPO_BANK_SEED:?run_one_dpo_pair needs DPO_BANK_SEED}"
: "${DPO_TRAIN_SEED:?run_one_dpo_pair needs DPO_TRAIN_SEED}"

# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "${ENV_NAME:-opdyn}"

complete() {  # $1 = run dir
  [ -f "$1/trajectory.pt" ] || return 1
  python - "$1/trajectory.pt" "$N_ROUNDS" <<'EOF'
import sys
import torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sys.exit(0 if len(d.get("trajectory", [])) >= int(sys.argv[2]) else 1)
EOF
}

if [ -f "$BANK/pair_meta.json" ] && complete "$RUNS/$TAG_CLOSED" \
    && complete "$RUNS/$TAG_OPEN"; then
  echo "[dpo_pair] $TAG_CLOSED pair already complete + validated -- exiting 0"
  exit 0
fi

if ! complete "$RUNS/$TAG_CLOSED"; then
  echo "[dpo_pair] running CLOSED arm (bank writer): $TAG_CLOSED"
  RLHF_FEEDBACK=closed DPO_BANK_MODE=write DPO_BANK_DIR="$BANK" \
    "$REPO/experiments/condor/run_one_pokec_gated.sh" "$TAG_CLOSED" "$@"
fi

if ! complete "$RUNS/$TAG_OPEN"; then
  [ -f "$BANK/bank_meta.json" ] || {
    echo "[dpo_pair] closed arm complete but bank meta missing" >&2; exit 1; }
  echo "[dpo_pair] running OPEN arm (bank reader): $TAG_OPEN"
  RLHF_FEEDBACK=open DPO_BANK_MODE=read DPO_BANK_DIR="$BANK" \
    "$REPO/experiments/condor/run_one_pokec_gated.sh" "$TAG_OPEN" "$@"
fi

echo "[dpo_pair] validating paired invariants"
python "$REPO/experiments/scripts/cluster_pipelines/check_dpo_pair.py" \
  "$RUNS/$TAG_CLOSED" --write-meta
echo "[dpo_pair] pair complete + validated"
