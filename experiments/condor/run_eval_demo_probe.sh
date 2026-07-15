#!/usr/bin/env bash
# Tree-2 frozen counterfactual probe runner (no training, no dynamics).
# Env: BASE_MODEL, TARGETS, AGES, OUT_DIR, GEN_CHUNK.

set -eo pipefail

# shard target passed as $1 (one condor proc per genre)
if [ -n "${1:-}" ]; then
  export TARGETS="$1"
fi

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"

source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$REPO"

exec python experiments/scripts/cluster_pipelines/eval_demo_probe.py
