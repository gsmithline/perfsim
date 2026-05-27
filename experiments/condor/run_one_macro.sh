#!/usr/bin/env bash
# Single-config runner for HTCondor (macro_economics version).
# Args: RUN_TAG TRAINING_STYLE KL_BETA
#
# Mirror of experiments/condor/run_one.sh but dispatches to scripts/run_macro_lm.py
# instead of run_covid_lm.py. Passes through all the env-var knobs the
# macro run script reads.

set -eo pipefail

RUN_TAG="$1"
TRAINING_STYLE="$2"
KL_BETA="$3"

# ---- machine-specific paths ----
REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

# ---- run knobs (overridable) ----
N_ROUNDS="${N_ROUNDS:-5}"
K_STEPS="${K_STEPS:-3}"
SEED="${SEED:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-20}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-16}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-perfsim-at-macro-lm}"
MACRO_YAML="${MACRO_YAML:-config_100_agents.yaml}"
SFT_FULL_EPOCH="${SFT_FULL_EPOCH:-0}"
SFT_SANITY="${SFT_SANITY:-0}"
LORA_R="${LORA_R:-32}"
USE_LORA="${USE_LORA:-1}"
SFT_LR="${SFT_LR:-1e-5}"
WANDB_RUN_SUFFIX="${WANDB_RUN_SUFFIX:-}"

echo "[run_one_macro] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[run_one_macro] tag=$RUN_TAG style=$TRAINING_STYLE beta=$KL_BETA n_rounds=$N_ROUNDS K=$K_STEPS model=$BASE_MODEL yaml=$MACRO_YAML"

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
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    MACRO_YAML="$MACRO_YAML" \
    SFT_FULL_EPOCH="$SFT_FULL_EPOCH" \
    SFT_SANITY="$SFT_SANITY" \
    LORA_R="$LORA_R" \
    USE_LORA="$USE_LORA" \
    SFT_LR="$SFT_LR" \
    WANDB_RUN_SUFFIX="$WANDB_RUN_SUFFIX" \
    CALIBRATED_UAC="${CALIBRATED_UAC:-}" \
    OUT_DIR="${OUT_DIR:-$REPO/runs/at_macro_lm/$RUN_TAG}" \
    python experiments/scripts/run_macro_lm.py
