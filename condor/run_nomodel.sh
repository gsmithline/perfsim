#!/usr/bin/env bash
set -euo pipefail

export RUN_TAG="$1"
export ISOLATION_LEVEL="$2"

cd /home/gsmithline/perfsim
source /home/gsmithline/miniconda3/etc/profile.d/conda.sh
conda activate opdyn

if [[ -f /home/gsmithline/.wandb_key ]]; then
    export WANDB_API_KEY=$(cat /home/gsmithline/.wandb_key)
fi

python scripts/run_covid_nomodel.py
