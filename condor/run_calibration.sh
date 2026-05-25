#!/usr/bin/env bash
# Single calibration job for HTCondor.
# Args: RUN_TAG SEED_FRAC START_WEEK
set -eo pipefail

RUN_TAG="$1"
SEED_FRAC="$2"
START_WEEK="${3:-17}"

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

N_ITERS="${N_ITERS:-50}"
LR="${LR:-0.5}"
START_WEEK="${START_WEEK:-17}"
N_WEEKS="${N_WEEKS:-3}"
SEED="${SEED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-perfsim-calibration}"

echo "[run_calibration] host=$(hostname) tag=$RUN_TAG frac=$SEED_FRAC iters=$N_ITERS lr=$LR"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi
export WANDB_DIR="${WANDB_DIR:-$REPO/wandb}"
export HF_HOME="${HF_HOME:-/home/gsmithline/.cache/huggingface}"

cd "$REPO"

env \
    RUN_TAG="$RUN_TAG" \
    SEED_FRAC="$SEED_FRAC" \
    START_WEEK="${START_WEEK}" \
    N_WEEKS="$N_WEEKS" \
    N_ITERS="$N_ITERS" \
    LR="$LR" \
    SEED="$SEED" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    OUT_DIR="${OUT_DIR:-$REPO/runs/calibration/$RUN_TAG}" \
    python scripts/calibrate_covid_single.py
