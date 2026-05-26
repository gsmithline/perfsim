#!/usr/bin/env bash
set -eo pipefail

RUN_TAG="$1"
ISOLATION_LEVEL="$2"

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

source "$CONDA_SH"
conda activate "$ENV_NAME"

if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$REPO"

env \
    RUN_TAG="$RUN_TAG" \
    ISOLATION_LEVEL="$ISOLATION_LEVEL" \
    N_ROUNDS="${N_ROUNDS:-10}" \
    K_STEPS="${K_STEPS:-21}" \
    SEED_FRAC="${SEED_FRAC:-0.05}" \
    SEED="${SEED:-0}" \
    CALIBRATED_R2="${CALIBRATED_R2:-}" \
    WANDB_PROJECT="${WANDB_PROJECT:-}" \
    OUT_DIR="${OUT_DIR:-$REPO/runs/at_covid_nomodel/$RUN_TAG}" \
    python scripts/run_covid_nomodel.py
