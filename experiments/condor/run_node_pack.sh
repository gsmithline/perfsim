#!/usr/bin/env bash
# Whole-node HTCondor executable: runs ONE manifest group (<= 8 single-GPU
# cells, one per assigned GPU) via node_pack.py. Each cell is launched
# exactly as its own single-GPU job would be (env + args rebuilt from the
# cell's generated sub and configs row, same idempotent wrapper), so the
# artifacts and logs are the ordinary per-tag ones.
#
# Why (2026-08-26): the H100 pool disappeared and the free A100-80GB nodes
# (g181-g184) only accept whole-node jobs. See node_pack.py.
#
# Usage (from the sub): run_node_pack.sh <manifest.json> <group_id>
set -eo pipefail
MANIFEST="$1"
GROUP="$2"
REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$ENV_NAME"
cd "$REPO"
echo "[run_node_pack] $(date '+%F %T') host=$(hostname) group=$GROUP CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/[run_node_pack] gpu /' || true
exec python "$REPO/experiments/condor/node_pack.py" "$MANIFEST" "$GROUP" --repo "$REPO"
