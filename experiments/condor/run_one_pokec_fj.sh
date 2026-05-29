#!/usr/bin/env bash
# Single Pokec FJ + LM run.
# Args: RUN_TAG TRAINING_STYLE KL_BETA SEED DEPLOY_EVERY DATA_REGIME

set -eo pipefail

RUN_TAG="$1"
TRAINING_STYLE="$2"
KL_BETA="$3"
SEED="${4:-0}"
DEPLOY_EVERY="${5:-1}"
DATA_REGIME="${6:-replace}"

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

N_ROUNDS="${N_ROUNDS:-12}"
EPOCH_SIZE="${EPOCH_SIZE:-100}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-1}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"
WANDB_PROJECT="${WANDB_PROJECT:-perfsim-pokec-fj-lm}"
LORA_R="${LORA_R:-8}"
USE_LORA="${USE_LORA:-1}"
SFT_LR="${SFT_LR:-5e-5}"
N_LABELED="${N_LABELED:-1730}"
POKEC_DIR="${POKEC_DIR:-$REPO/examples/pokec}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-6}"
HIST_BINS="${HIST_BINS:-50}"
LOG_PERPLEXITY="${LOG_PERPLEXITY:-1}"
N_PERPLEXITY="${N_PERPLEXITY:-64}"
WANDB_RUN_SUFFIX="${WANDB_RUN_SUFFIX:-}"

echo "[run_one_pokec_fj] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[run_one_pokec_fj] tag=$RUN_TAG style=$TRAINING_STYLE beta=$KL_BETA seed=$SEED deploy_every=$DEPLOY_EVERY regime=$DATA_REGIME model=$BASE_MODEL"

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
    SEED="$SEED" \
    DEPLOY_EVERY="$DEPLOY_EVERY" \
    DATA_REGIME="$DATA_REGIME" \
    BASE_MODEL="$BASE_MODEL" \
    N_ROUNDS="$N_ROUNDS" \
    EPOCH_SIZE="$EPOCH_SIZE" \
    SFT_MAX_STEPS="$SFT_MAX_STEPS" \
    SFT_EPOCHS="$SFT_EPOCHS" \
    SFT_BATCH_SIZE="$SFT_BATCH_SIZE" \
    GEN_BATCH_SIZE="$GEN_BATCH_SIZE" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    LORA_R="$LORA_R" \
    USE_LORA="$USE_LORA" \
    SFT_LR="$SFT_LR" \
    N_LABELED="$N_LABELED" \
    POKEC_DIR="$POKEC_DIR" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    HIST_BINS="$HIST_BINS" \
    LOG_PERPLEXITY="$LOG_PERPLEXITY" \
    N_PERPLEXITY="$N_PERPLEXITY" \
    WANDB_RUN_SUFFIX="$WANDB_RUN_SUFFIX" \
    OUT_DIR="${OUT_DIR:-$REPO/runs/pokec_fj_lm/$RUN_TAG}" \
    python experiments/scripts/run_pokec_fj_lm.py
