#!/usr/bin/env bash
# All CPU-only cells of the Qwen2.5 mechanism diagnostic (2026-08-20).
# No GPU, no Condor, no LLM -- perfect-prediction oracles, the long-horizon
# oracle grid, and the offline frozen population replays.
#
#   bash experiments/scripts/cluster_pipelines/gen_qwen_mechanism_cpu.sh \
#       [FROZEN_SOURCE_RUN_DIR]
#
# FROZEN_SOURCE_RUN_DIR must be an archived H100 frozen K=D=0 Qwen2.5 run;
# it defaults to the canonical one below. replay_frozen_offline.py refuses
# anything on other silicon and checks the vector's sha256 against the
# canonical hash, because a frozen model's served vector is a CONSTANT the
# whole grid is compared against.
#
# Artifacts are refused rather than overwritten when they already exist, so
# re-running this is a no-op. Gate the result with:
#   python experiments/scripts/cluster_pipelines/check_perfect_predictor.py \
#       --dir notes/pofd/perfect_prediction --dir notes/pofd/frozen_replay
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
PP="python3 experiments/scripts/cluster_pipelines/sim_perfect_predictor.py"
FRZ="python3 experiments/scripts/cluster_pipelines/replay_frozen_offline.py"

CANON_SHA="1674ee5f8d833f46de672791d933e1d3bdeefb07484c2d110dec84ce71da30bb"
FROZEN_SRC="${1:-notes/pofd/cluster/pofdfam_qwen7b_k0_ea1_w0p5_l0p2_es0p05_s0}"

echo "=== Part A: 8 paper-regime perfect-prediction cells (30 rounds) ==="
# k {.2, 1} x eps_social {0, .05, .2, 1} at W=.5, numeric eps_AI=1 --
# matched to the 24 GPU cells of the same grid.
for k in 0.2 1; do
  for es in 0 0.05 0.2 1; do
    $PP --innate-k "$k" --w-plat 0.5 --eps-social "$es" --eps-ai 1 \
        --rounds 30 --seed 0 --quiet
  done
done

echo "=== Part E: 12-cell oracle grid isolating bounded-confidence (300 rounds) ==="
# k=1, W {.5, .9, 1} x peer {no peers, threshold .05, threshold .2,
# TRUE all_open}, AI gate genuinely open throughout. This separates
# platform susceptibility from nonlinear peer selection: residual
# dispersion at W=1 under a NARROW peer gate exists with no pretrained
# signal at all and is attributable to bounded confidence.
# W is population susceptibility to the served output -- NOT a
# regularization dial. Do not read W=.5 vs W=1 as a regularization
# comparison.
for w in 0.5 0.9 1; do
  # no peers: eps_social=0 IS the no-peer condition (threshold mode)
  $PP --innate-k 1 --w-plat "$w" --eps-social 0 --eps-ai 1 \
      --ai-gate-mode all_open --rounds 300 --seed 0 --quiet
  for es in 0.05 0.2; do
    $PP --innate-k 1 --w-plat "$w" --eps-social "$es" --eps-ai 1 \
        --ai-gate-mode all_open --rounds 300 --seed 0 --quiet
  done
  # genuinely open peers -- a MODE, never eps_social=1: the Deffuant
  # test is a strict inequality, so a pair at (0,1) sits at distance
  # exactly 1 and eps_social=1 would still REJECT it.
  $PP --innate-k 1 --w-plat "$w" --eps-social 0.2 --eps-ai 1 \
      --ai-gate-mode all_open --peer-gate-mode all_open \
      --rounds 300 --seed 0 --quiet
done

echo "=== Part D: frozen offline replays at the Wu boundary (300 rounds) ==="
# k=1, W {.5, 1}, BOTH gates truly open. Served values are real H100
# Qwen2.5 K=D=0 outputs, held constant -- which the archived run
# demonstrates rather than assumes. The matching perfect-prediction
# controls are the all_open cells of the Part E grid above (same
# parameters, same filenames, already generated).
for w in 0.5 1; do
  $FRZ --from-run "$FROZEN_SRC" --expect-sha "$CANON_SHA" \
       --innate-k 1 --w-plat "$w" --eps-social 0.2 --eps-ai 1 \
       --ai-gate-mode all_open --peer-gate-mode all_open \
       --rounds 300 --seed 0 --quiet
done

echo
echo "perfect-prediction artifacts: $(ls notes/pofd/perfect_prediction | wc -l)"
echo "frozen-replay artifacts:      $(ls notes/pofd/frozen_replay | wc -l)"
echo
echo "gate with:"
echo "  python experiments/scripts/cluster_pipelines/check_perfect_predictor.py \\"
echo "      --dir notes/pofd/perfect_prediction --dir notes/pofd/frozen_replay"
