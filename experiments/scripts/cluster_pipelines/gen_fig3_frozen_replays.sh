#!/usr/bin/env bash
# The lambda = infinity column of the redesigned Figure 3.  CPU ONLY --
# no GPU, no Condor, no model load.  Safe to run on a laptop.
#
# WHY THIS IS NOT A GPU JOB.  A frozen Qwen3 prompted with ICL_K = D = 0
# never sees the population: the prompt is the agent's static profile and
# nothing trains, so its parsed prediction vector is a CONSTANT, identical
# in every round and independent of eps_AI and eps_social.  The archived
# cells verify that empirically (one shared sha256 across all rounds).
# replay_frozen_offline.py therefore loads that real vector from an
# archived H100 run and replays the population and peer process around it
# through sim_perfect_predictor.simulate -- the IDENTICAL operator path
# the GPU runs take, not a second copy of the dynamics.  The population
# loop still runs fully recursively; only the model is frozen, which is
# exactly what lambda = infinity means.
#
# All 13 cells of the column share ONE source vector.  Four already exist
# (k in {1,.5,.2} at W=.5, and k=1 at W=1); the nine below are the ones
# the redesign adds.  --expect-sha makes a wrong source run a hard error
# rather than a silently different figure.
#
# Re-runnable: replay_frozen_offline.py refuses to clobber an existing
# artifact unless --force is passed, so this never overwrites the four
# that are already on disk.
set -euo pipefail
cd "$(dirname "$0")/../../.."

SRC="notes/pofd/cluster/pofdzsprior_qwen3_8b_w0p5_l0p2_es0_s0"
SHA="fdfdeab7466345159cd7ae16ee487d4982d686cfdb93287780ae4d109ccba3f7"
ROUNDS=30
SWEEPS=100
SEED=0

# (W_PLAT=beta, INNATE_LAMBDA=gamma) pairs the redesign adds
PAIRS=(
  "0.25 0"    "0.25 0.2"  "0.25 0.5"  "0.25 1"
  "0.5  0"
  "0.75 0"    "0.75 0.2"  "0.75 0.5"  "0.75 1"
)

num () { printf '%s' "$1" | sed 's/\./p/'; }

for pair in "${PAIRS[@]}"; do
  set -- $pair
  W="$1"; K="$2"
  OUT="notes/pofd/frozen_replay/frz_k$(num "$K")_w$(num "$W")_eaopen_esopen_sw${SWEEPS}_s${SEED}_r${ROUNDS}.pt"
  if [ -f "$OUT" ]; then
    echo "[fig3-frozen] SKIP (exists) $OUT"
    continue
  fi
  echo "[fig3-frozen] W=$W k=$K sweeps=$SWEEPS rounds=$ROUNDS"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 \
  python3 experiments/scripts/cluster_pipelines/replay_frozen_offline.py \
    --from-run "$SRC" \
    --expect-sha "$SHA" \
    --innate-k "$K" \
    --w-plat "$W" \
    --eps-social .2 \
    --ai-gate-mode all_open \
    --peer-gate-mode all_open \
    --sweeps "$SWEEPS" \
    --rounds "$ROUNDS" \
    --seed "$SEED"
done
echo "[fig3-frozen] done -- 9 artifacts under notes/pofd/frozen_replay/"
