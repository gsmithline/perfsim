#!/usr/bin/env bash
# Single Pokec gated-population + LM run.
# Args: RUN_TAG TRAINING_STYLE KL_BETA SEED DEPLOY_EVERY DATA_REGIME [PLATFORM_SUS_SCALE] [ANCHOR_MODE] [POP_MODEL] [EPS] [GAMMA_BIAS] [W_PLAT] [RUN_MODE] [CANARY_DELTA]

set -eo pipefail

RUN_TAG="$1"
TRAINING_STYLE="$2"
KL_BETA="$3"
SEED="${4:-0}"
DEPLOY_EVERY="${5:-1}"
DATA_REGIME="${6:-replace}"
PLATFORM_SUS_SCALE="${7:-${PLATFORM_SUS_SCALE:-1.0}}"
ANCHOR_MODE="${8:-${ANCHOR_MODE:-fixed}}"
POP_MODEL="${9:-${POP_MODEL:-fj}}"
EPS="${10:-${EPS:-0.3}}"
GAMMA_BIAS="${11:-${GAMMA_BIAS:-1.5}}"
W_PLAT="${12:-${W_PLAT:-0.3}}"
RUN_MODE="${13:-${RUN_MODE:-loop}}"
CANARY_DELTA="${14:-${CANARY_DELTA:-0.0}}"

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
WANDB_PROJECT="${WANDB_PROJECT:-perfsim-pokec-gated-lm}"
LORA_R="${LORA_R:-8}"
USE_LORA="${USE_LORA:-1}"
SFT_LR="${SFT_LR:-5e-5}"
N_LABELED="${N_LABELED:-1730}"
POKEC_DIR="${POKEC_DIR:-$REPO/examples/pokec}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-6}"
HIST_BINS="${HIST_BINS:-50}"
LOG_PERPLEXITY="${LOG_PERPLEXITY:-1}"
N_PERPLEXITY="${N_PERPLEXITY:-64}"
SEED_BASE_DATA="${SEED_BASE_DATA:-1}"
TRAIN_CAP="${TRAIN_CAP:-0}"
N_PROBE="${N_PROBE:-64}"
TEL_EVAL_CAP="${TEL_EVAL_CAP:-64}"
GRAD_NORM_N="${GRAD_NORM_N:-8}"
WANDB_RUN_SUFFIX="${WANDB_RUN_SUFFIX:-}"

echo "[run_one_pokec_gated] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[run_one_pokec_gated] tag=$RUN_TAG style=$TRAINING_STYLE beta=$KL_BETA seed=$SEED deploy_every=$DEPLOY_EVERY regime=$DATA_REGIME pscale=$PLATFORM_SUS_SCALE anchor=$ANCHOR_MODE pop=$POP_MODEL eps=$EPS gamma=$GAMMA_BIAS w=$W_PLAT mode=$RUN_MODE canary=$CANARY_DELTA model=$BASE_MODEL"

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
    SEED_BASE_DATA="$SEED_BASE_DATA" \
    TRAIN_CAP="$TRAIN_CAP" \
    PLATFORM_SUS_SCALE="$PLATFORM_SUS_SCALE" \
    ANCHOR_MODE="$ANCHOR_MODE" \
    POP_MODEL="$POP_MODEL" \
    EPS="$EPS" \
    GAMMA_BIAS="$GAMMA_BIAS" \
    W_PLAT="$W_PLAT" \
    RUN_MODE="$RUN_MODE" \
    CANARY_DELTA="$CANARY_DELTA" \
    N_PROBE="$N_PROBE" \
    TEL_EVAL_CAP="$TEL_EVAL_CAP" \
    GRAD_NORM_N="$GRAD_NORM_N" \
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
    OUT_DIR="${OUT_DIR:-$REPO/runs/pokec_gated_lm/$RUN_TAG}" \
    python experiments/scripts/cluster_pipelines/run_pokec_gated_lm.py
