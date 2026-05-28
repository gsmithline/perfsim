#!/usr/bin/env bash
# Single Schelling LM run.
# Args: RUN_TAG TRAINING_STYLE KL_BETA

set -eo pipefail

RUN_TAG="$1"
TRAINING_STYLE="$2"
KL_BETA="$3"

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

N_ROUNDS="${N_ROUNDS:-10}"
K_STEPS="${K_STEPS:-3}"
SEED="${SEED:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-50}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-16}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"
WANDB_PROJECT="${WANDB_PROJECT:-perfsim-at-schelling-lm}"
SFT_FULL_EPOCH="${SFT_FULL_EPOCH:-0}"
LORA_R="${LORA_R:-32}"
USE_LORA="${USE_LORA:-1}"
SFT_LR="${SFT_LR:-1e-5}"
NUM_AGENTS="${NUM_AGENTS:-200}"
GRID_SIZE="${GRID_SIZE:-20}"
BASELINE_THRESHOLD="${BASELINE_THRESHOLD:-0.30}"
LAMBDA="${LAMBDA:-0.15}"
NEIGHBORHOOD_RADIUS="${NEIGHBORHOOD_RADIUS:-1}"
WANDB_RUN_SUFFIX="${WANDB_RUN_SUFFIX:-}"

echo "[run_one_schelling] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[run_one_schelling] tag=$RUN_TAG style=$TRAINING_STYLE beta=$KL_BETA grid=${GRID_SIZE}x${GRID_SIZE} agents=$NUM_AGENTS H_0=$BASELINE_THRESHOLD"

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
    SEED="$SEED" \
    SFT_MAX_STEPS="$SFT_MAX_STEPS" \
    SFT_BATCH_SIZE="$SFT_BATCH_SIZE" \
    GEN_BATCH_SIZE="$GEN_BATCH_SIZE" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    SFT_FULL_EPOCH="$SFT_FULL_EPOCH" \
    LORA_R="$LORA_R" \
    USE_LORA="$USE_LORA" \
    SFT_LR="$SFT_LR" \
    NUM_AGENTS="$NUM_AGENTS" \
    GRID_SIZE="$GRID_SIZE" \
    BASELINE_THRESHOLD="$BASELINE_THRESHOLD" \
    LAMBDA="$LAMBDA" \
    NEIGHBORHOOD_RADIUS="$NEIGHBORHOOD_RADIUS" \
    WANDB_RUN_SUFFIX="$WANDB_RUN_SUFFIX" \
    OUT_DIR="${OUT_DIR:-$REPO/runs/at_schelling_lm/$RUN_TAG}" \
    python experiments/scripts/run_schelling_lm.py
