#!/usr/bin/env bash
# Q6 diagnostic runner: frozen-reference perplexity on saved round corpora.
# Env: TAGS (comma-separated run tags), BASE_MODEL, PPL_BATCH.

set -eo pipefail

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"

source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$REPO"

exec python experiments/scripts/cluster_pipelines/eval_ref_ppl.py
