#!/usr/bin/env bash
# Args: RUN_TAG REGIME LABEL CONDITION SEED MODEL SCREENER
set -eo pipefail

RUN_TAG="$1"
DATA_REGIME="${2:-replace}"
LABEL="${3:-experience}"
CONDITION="${4:-phenomenon}"
SEED="${5:-0}"
MODEL="${6:-${MODEL:-linear}}"
SCREENER="${7:-${SCREENER:-classification}}"

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
N_ROUNDS="${N_ROUNDS:-15}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
ANCHOR_ALPHA="${ANCHOR_ALPHA:-0.3}"
WANDB_PROJECT="${WANDB_PROJECT:-perfsim-resume-llm}"
WANDB_RUN_SUFFIX="${WANDB_RUN_SUFFIX:-}"

echo "[run_one_resume_llm] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[run_one_resume_llm] tag=$RUN_TAG regime=$DATA_REGIME label=$LABEL seed=$SEED model=$BASE_MODEL"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi
export HF_HOME="${HF_HOME:-/home/gsmithline/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$REPO"

env \
    RUN_TAG="$RUN_TAG" \
    DATA_REGIME="$DATA_REGIME" \
    LABEL="$LABEL" \
    CONDITION="$CONDITION" \
    SEED="$SEED" \
    BASE_MODEL="$BASE_MODEL" \
    N_ROUNDS="$N_ROUNDS" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    GEN_BATCH_SIZE="$GEN_BATCH_SIZE" \
    ANCHOR_ALPHA="$ANCHOR_ALPHA" \
    MODEL="$MODEL" \
    SCREENER="$SCREENER" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_RUN_SUFFIX="$WANDB_RUN_SUFFIX" \
    OUT_DIR="${OUT_DIR:-$REPO/runs/resume_llm/$RUN_TAG}" \
    python experiments/scripts/ai_mediated/run_resume_llm_loop.py
