#!/usr/bin/env bash
# Single-config runner for HTCondor.
# Args: RUN_TAG TRAINING_STYLE KL_BETA
#
# Mirrors Opinion-dynamics-post-training/condor/run_one.sh.
# Sets up the conda env + wandb key, then invokes
# scripts/run_covid_lm.py with the (style, beta) for this job.

set -eo pipefail

RUN_TAG="$1"
TRAINING_STYLE="$2"
KL_BETA="$3"

# ---- machine-specific paths (edit before submitting) ----
REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

# ---- run knobs (overridable) ----
N_ROUNDS="${N_ROUNDS:-5}"
K_STEPS="${K_STEPS:-3}"
SEED_FRAC="${SEED_FRAC:-0.05}"
SEED="${SEED:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-50}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-32}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-perfsim-at-covid-lm}"
CALIBRATED_R2="${CALIBRATED_R2:-}"

echo "[run_one] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[run_one] tag=$RUN_TAG style=$TRAINING_STYLE beta=$KL_BETA n_rounds=$N_ROUNDS K=$K_STEPS model=$BASE_MODEL R2=$CALIBRATED_R2 frac=$SEED_FRAC"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi
export WANDB_DIR="${WANDB_DIR:-$REPO/wandb}"
export HF_HOME="${HF_HOME:-/home/gsmithline/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$REPO"

env \
    RUN_TAG="$RUN_TAG" \
    TRAINING_STYLE="$TRAINING_STYLE" \
    KL_BETA="$KL_BETA" \
    BASE_MODEL="$BASE_MODEL" \
    N_ROUNDS="$N_ROUNDS" \
    K_STEPS="$K_STEPS" \
    SEED_FRAC="$SEED_FRAC" \
    SEED="$SEED" \
    SFT_MAX_STEPS="$SFT_MAX_STEPS" \
    SFT_BATCH_SIZE="$SFT_BATCH_SIZE" \
    GEN_BATCH_SIZE="$GEN_BATCH_SIZE" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    CALIBRATED_R2="$CALIBRATED_R2" \
    OUT_DIR="${OUT_DIR:-$REPO/runs/at_covid_lm/$RUN_TAG}" \
    python scripts/run_covid_lm.py
