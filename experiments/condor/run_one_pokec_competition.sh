#!/usr/bin/env bash
# Single Pokec FJ + 3-LLM competition run.
# Args: RUN_TAG TRAINING_STYLE KL_BETA TAU SEED

set -eo pipefail

RUN_TAG="$1"
TRAINING_STYLE="$2"
KL_BETA="$3"
TAU="${4:-0.05}"
SEED="${5:-0}"

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

N_ROUNDS="${N_ROUNDS:-12}"
EPOCH_SIZE="${EPOCH_SIZE:-100}"
BASE_MODELS="${BASE_MODELS:-Qwen/Qwen2.5-1.5B-Instruct,HuggingFaceTB/SmolLM2-1.7B-Instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PLACEMENTS="${PLACEMENTS:-}"
PLACE_PASSES="${PLACE_PASSES:-0}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-1}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"
WANDB_PROJECT="${WANDB_PROJECT:-perfsim-pokec-fj-competition}"
LORA_R="${LORA_R:-8}"
USE_LORA="${USE_LORA:-1}"
SFT_LR="${SFT_LR:-5e-5}"
N_LABELED="${N_LABELED:-1730}"
POKEC_DIR="${POKEC_DIR:-$REPO/examples/pokec}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-6}"
HIST_BINS="${HIST_BINS:-50}"

echo "[run_one_pokec_competition] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$REPO"

if [ -f "$WANDB_KEY_FILE" ]; then
  export WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")"
fi

export RUN_TAG TRAINING_STYLE KL_BETA TAU SEED \
  N_ROUNDS EPOCH_SIZE BASE_MODELS PLACEMENTS PLACE_PASSES \
  SFT_MAX_STEPS SFT_EPOCHS SFT_BATCH_SIZE GEN_BATCH_SIZE WANDB_PROJECT \
  LORA_R USE_LORA SFT_LR N_LABELED POKEC_DIR MAX_NEW_TOKENS HIST_BINS

python experiments/scripts/cluster_pipelines/run_pokec_fj_competition.py
