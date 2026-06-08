#!/usr/bin/env bash
# Single Pokec FJ hunter run WITH a KL leash on the hunters to their own prior.
# Args: RUN_TAG PLATFORM_TYPES TAU HUNT_LR SEED HUNT_KL_BETA

set -eo pipefail

RUN_TAG="$1"
PLATFORM_TYPES="$2"
TAU="${3:-0.05}"
HUNT_LR="${4:-1e-5}"
SEED="${5:-0}"
HUNT_KL_BETA="${6:-0.0}"

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

N_ROUNDS="${N_ROUNDS:-25}"
EPOCH_SIZE="${EPOCH_SIZE:-100}"
BASE_MODELS="${BASE_MODELS:-Qwen/Qwen2.5-1.5B-Instruct,HuggingFaceTB/SmolLM2-1.7B-Instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
HUNT_STEPS="${HUNT_STEPS:-8}"
HUNT_BATCH="${HUNT_BATCH:-64}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
WANDB_PROJECT="${WANDB_PROJECT:-perfsim-pokec-fj-hunt-kl}"
LORA_R="${LORA_R:-8}"
SFT_LR="${SFT_LR:-5e-5}"
N_LABELED="${N_LABELED:-1730}"
POKEC_DIR="${POKEC_DIR:-$REPO/examples/pokec}"
OUT_DIR="${OUT_DIR:-$REPO/runs/pokec_fj_hunt_kl/$RUN_TAG}"

echo "[run_one_pokec_hunt_kl] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none) kl_beta=$HUNT_KL_BETA"

source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$REPO"

if [ -f "$WANDB_KEY_FILE" ]; then
  export WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")"
fi

export RUN_TAG PLATFORM_TYPES TAU HUNT_LR SEED HUNT_KL_BETA \
  N_ROUNDS EPOCH_SIZE BASE_MODELS HUNT_STEPS HUNT_BATCH \
  SFT_EPOCHS SFT_BATCH_SIZE WANDB_PROJECT LORA_R SFT_LR N_LABELED POKEC_DIR OUT_DIR

python experiments/scripts/cluster_pipelines/run_pokec_fj_hunt.py
