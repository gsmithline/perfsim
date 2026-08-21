#!/usr/bin/env bash
# Adapter KL / soft-decode probe over the SFT training-dose adapters.
# Args: MODE  (full | smoke)
#
# Not a population run: no training, no serving loop, no trajectory.pt.
# It loads the base model once, then scores every saved round-0 LoRA
# adapter against it with teacher-forced forwards. That is why it does
# NOT go through run_one_pokec_gated_idempotent.sh -- there is no
# trajectory to be idempotent about. Re-running simply overwrites the
# probe directory, which is cheap and deterministic.
#
# H100 only: the gate that ties this probe to the archived runs is the
# canonical frozen-Qwen served hash, and greedy generation is only
# bit-reproducible within one GPU architecture.

set -eo pipefail

MODE="${1:-full}"

REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
OUT_DIR="${OUT_DIR:-$REPO/runs/adapter_kl_probe}"
TF_BATCH="${TF_BATCH:-8}"

cd "$REPO"
# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$ENV_NAME"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

ARGS=(--out-dir "$OUT_DIR" --tf-batch "$TF_BATCH")
if [ "$MODE" = "smoke" ]; then
  # A truncated agent set cannot reproduce the canonical 723-agent hash,
  # so --smoke downgrades that one gate to a warning. Everything else --
  # teacher-forcing self-check, support coverage, adapter distinctness --
  # still runs, which is the point of the smoke.
  OUT_DIR="${OUT_DIR}_smoke"
  ARGS=(--out-dir "$OUT_DIR" --tf-batch "$TF_BATCH"
        --limit-agents "${SMOKE_AGENTS:-32}"
        --max-adapters "${SMOKE_ADAPTERS:-2}" --smoke)
elif [ "$MODE" != "full" ]; then
  echo "usage: run_one_adapter_kl_probe.sh full|smoke" >&2
  exit 2
fi

echo "[akl] mode=$MODE out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python experiments/scripts/cluster_pipelines/probe_adapter_kl.py "${ARGS[@]}"

echo "[akl] probe done; gate with:"
echo "  python experiments/scripts/cluster_pipelines/check_adapter_kl_probe.py --dir $OUT_DIR"
