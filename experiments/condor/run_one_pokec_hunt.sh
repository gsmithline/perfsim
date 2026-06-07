#!/usr/bin/env bash
# Single Pokec FJ hunter/describer run.
# Args: RUN_TAG PLATFORM_TYPES TAU HUNT_LR SEED

set -eo pipefail

RUN_TAG="$1"
PLATFORM_TYPES="$2"
TAU="${3:-0.05}"
HUNT_LR="${4:-1e-5}"
SEED="${5:-0}"

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

N_ROUNDS="${N_ROUNDS:-12}"
EPOCH_SIZE="${EPOCH_SIZE:-100}"
BASE_MODELS="${BASE_MODELS:-Qwen/Qwen2.5-1.5B-Instruct,HuggingFaceTB/SmolLM2-1.7B-Instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
HUNT_STEPS="${HUNT_STEPS:-8}"
HUNT_BATCH="${HUNT_BATCH:-64}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
WANDB_PROJECT="${WANDB_PROJECT:-perfsim-pokec-fj-hunt}"
LORA_R="${LORA_R:-8}"
SFT_LR="${SFT_LR:-5e-5}"
N_LABELED="${N_LABELED:-1730}"
POKEC_DIR="${POKEC_DIR:-$REPO/examples/pokec}"

echo "[run_one_pokec_hunt] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$REPO"

if [ -f "$WANDB_KEY_FILE" ]; then
  export WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")"
fi

export RUN_TAG PLATFORM_TYPES TAU HUNT_LR SEED \
  N_ROUNDS EPOCH_SIZE BASE_MODELS HUNT_STEPS HUNT_BATCH \
  SFT_EPOCHS SFT_BATCH_SIZE WANDB_PROJECT LORA_R SFT_LR N_LABELED POKEC_DIR

python experiments/scripts/cluster_pipelines/run_pokec_fj_hunt.py
