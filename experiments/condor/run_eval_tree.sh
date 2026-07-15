#!/usr/bin/env bash
# Tree-3 nested-consistency probe runner.
# Args: $1 BASE_MODEL, $2 OUT json, $3 HF_HOME override ("-" = default),
#       $4 ADAPTER_PATH ("-" = frozen base), $5 GEN_CHUNK (optional).

set -eo pipefail

export BASE_MODEL="$1"
export OUT="$2"
if [ -n "${3:-}" ] && [ "${3}" != "-" ]; then
  export HF_HOME="$3"
fi
if [ -n "${4:-}" ] && [ "${4}" != "-" ]; then
  export ADAPTER_PATH="$4"
fi
if [ -n "${5:-}" ]; then
  export GEN_CHUNK="$5"
fi

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"

source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$REPO"

exec python experiments/scripts/cluster_pipelines/eval_tree_consistency.py
