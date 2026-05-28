#!/usr/bin/env bash
# Single in-context macro performative loop job.
# Args: RUN_TAG STATIC HISTORY_WINDOW

set -eo pipefail

RUN_TAG="$1"
STATIC="$2"
HISTORY_WINDOW="$3"

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

N_ROUNDS="${N_ROUNDS:-20}"
K_STEPS="${K_STEPS:-3}"
SEED="${SEED:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
WANDB_PROJECT="${WANDB_PROJECT:-perfsim-at-macro-in-context}"
MACRO_YAML="${MACRO_YAML:-config_100_agents.yaml}"
N_AGENTS="${N_AGENTS:-100}"
GROUP_PROMPTING="${GROUP_PROMPTING:-1}"
CONSUMPTION_NOISE="${CONSUMPTION_NOISE:-0.05}"
WANDB_RUN_SUFFIX="${WANDB_RUN_SUFFIX:-}"
CALIBRATED_UAC="${CALIBRATED_UAC:-}"

echo "[run_one_in_context] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[run_one_in_context] tag=$RUN_TAG static=$STATIC window=$HISTORY_WINDOW model=$BASE_MODEL"

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
    STATIC="$STATIC" \
    HISTORY_WINDOW="$HISTORY_WINDOW" \
    BASE_MODEL="$BASE_MODEL" \
    N_ROUNDS="$N_ROUNDS" \
    K_STEPS="$K_STEPS" \
    SEED="$SEED" \
    GEN_BATCH_SIZE="$GEN_BATCH_SIZE" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    MACRO_YAML="$MACRO_YAML" \
    N_AGENTS="$N_AGENTS" \
    GROUP_PROMPTING="$GROUP_PROMPTING" \
    CONSUMPTION_NOISE="$CONSUMPTION_NOISE" \
    WANDB_RUN_SUFFIX="$WANDB_RUN_SUFFIX" \
    CALIBRATED_UAC="$CALIBRATED_UAC" \
    OUT_DIR="${OUT_DIR:-$REPO/runs/at_macro_in_context/$RUN_TAG}" \
    python experiments/scripts/run_macro_in_context.py
