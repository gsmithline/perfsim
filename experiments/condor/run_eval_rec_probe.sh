#!/usr/bin/env bash
set -eo pipefail
REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
source "$CONDA_SH"; conda activate "$ENV_NAME"; cd "$REPO"
exec python experiments/scripts/cluster_pipelines/eval_rec_probe.py
