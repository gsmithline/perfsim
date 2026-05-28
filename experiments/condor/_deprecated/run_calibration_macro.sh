#!/usr/bin/env bash
# Single macro calibration job for HTCondor.
# Args: RUN_TAG START_MONTH
set -eo pipefail

RUN_TAG="$1"
START_MONTH="$2"

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

N_ITERS="${N_ITERS:-50}"
N_STEPS="${N_STEPS:-9}"
LR="${LR:-0.1}"
SEED="${SEED:-0}"
N_AGENTS="${N_AGENTS:-100}"
MACRO_YAML="${MACRO_YAML:-config_100_agents.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-perfsim-macro-calibration}"

echo "[run_calibration_macro] host=$(hostname) tag=$RUN_TAG start=$START_MONTH iters=$N_ITERS lr=$LR"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi
export WANDB_DIR="${WANDB_DIR:-$REPO/wandb}"

cd "$REPO"

env \
    RUN_TAG="$RUN_TAG" \
    START_MONTH="$START_MONTH" \
    N_STEPS="$N_STEPS" \
    N_ITERS="$N_ITERS" \
    LR="$LR" \
    SEED="$SEED" \
    N_AGENTS="$N_AGENTS" \
    MACRO_YAML="$MACRO_YAML" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    OUT_DIR="${OUT_DIR:-$REPO/runs/calibration_macro/$RUN_TAG}" \
    python experiments/scripts/calibrate_macro_single.py
