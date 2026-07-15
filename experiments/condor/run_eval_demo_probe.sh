#!/usr/bin/env bash
# Tree-2 frozen counterfactual probe runner (no training, no dynamics).
# Env: BASE_MODEL, TARGETS, AGES, OUT_DIR, GEN_CHUNK.

set -eo pipefail

# shard target passed as $1 (one condor proc per genre); optional $2-$4
# override BASE_MODEL, OUT_DIR, GEN_CHUNK for cross-model sweeps
if [ -n "${1:-}" ]; then
  export TARGETS="$1"
fi
if [ -n "${2:-}" ]; then
  export BASE_MODEL="$2"
fi
if [ -n "${3:-}" ]; then
  export OUT_DIR="$3"
fi
if [ -n "${4:-}" ]; then
  export GEN_CHUNK="$4"
fi
# $5: HF cache override ("-" = keep default). OLMo lives in the lustre
# cache; Qwen/Llama/Gemma in the default ~/.cache (BATCHES.md gotcha 3).
if [ -n "${5:-}" ] && [ "${5}" != "-" ]; then
  export HF_HOME="$5"
fi

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"

source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$REPO"

exec python experiments/scripts/cluster_pipelines/eval_demo_probe.py
