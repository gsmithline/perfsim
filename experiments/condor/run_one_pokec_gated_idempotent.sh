#!/usr/bin/env bash
# Idempotent wrapper around run_one_pokec_gated.sh: exits 0 immediately if the
# run is already complete (trajectory.pt holds >= N_ROUNDS rounds).
#
# Why (the babysitting fix, 2026-07-17): completed jobs sometimes exit through
# an NFS ESTALE and land on hold; periodic_release then RESTARTS them from
# round 0 (this wasted a full re-run in the ICL batch). With this wrapper a
# released-but-already-done job becomes a cheap no-op and leaves the queue on
# its own, so hold/release cycles need no human attention. Pair it in the .sub
# with:
#   on_exit_hold     = (ExitCode =!= 0)
#   periodic_release = (NumJobStarts < 5) && ((time() - EnteredCurrentStatus) > 180)
#   periodic_remove  = (JobStatus == 5) && (NumJobStarts >= 5) && ((time() - EnteredCurrentStatus) > 600)
# so genuine failures retry up to 5 starts and then self-remove.

set -eo pipefail
RUN_TAG="$1"
REPO="${REPO:-/home/gsmithline/perfsim}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
CHECK_ENV="${ENV_NAME:-opdyn}"
N_ROUNDS="${N_ROUNDS:-30}"
OUT_DIR="${OUT_DIR:-$REPO/runs/pokec_gated_lm/$RUN_TAG}"

if [ -f "$OUT_DIR/trajectory.pt" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH" && conda activate "$CHECK_ENV"
    if python - "$OUT_DIR/trajectory.pt" "$N_ROUNDS" <<'EOF'
import sys
import torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sys.exit(0 if len(d.get("trajectory", [])) >= int(sys.argv[2]) else 1)
EOF
    then
        echo "[idempotent] $RUN_TAG already complete (>= $N_ROUNDS rounds) -- exiting 0"
        exit 0
    fi
    echo "[idempotent] $RUN_TAG present but incomplete -- rerunning"
fi

exec "$REPO/experiments/condor/run_one_pokec_gated.sh" "$@"
