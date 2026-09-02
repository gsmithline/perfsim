#!/usr/bin/env bash
# Run ONE section3_frontier_icl cell locally.
#
# WHY LOCAL. This wave is a network client: the computation happens in the
# provider's datacentre, so the machine running it needs no GPU and
# essentially no CPU. Running it here removes every cluster-specific risk
# (worker-node egress, a key on a shared filesystem, a Condor slot held
# idle for an hour of network I/O) and buys nothing back.
#
# SAFETY. Defaults to the 3-round SMOKE. The 30-round production run
# requires --production explicitly, and both are bounded by a hard cost
# cap enforced in-process that RAISES rather than truncating the science.
#
#   ./run_frontier_local.sh --model google/gemini-3.7-flash \
#       --provider "Google AI Studio" --seed 0 [--production] [--max-cost 5]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL=""; PROVIDER=""; SEED=0; ROUNDS=3; MAXCOST=2.0; TAGPRE="pofds3fsmk"
CONC=8; RPS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)      MODEL="$2"; shift 2 ;;
    --provider)   PROVIDER="$2"; shift 2 ;;
    --seed)       SEED="$2"; shift 2 ;;
    --max-cost)   MAXCOST="$2"; shift 2 ;;
    --concurrency) CONC="$2"; shift 2 ;;
    --rps)        RPS="$2"; shift 2 ;;
    --production) ROUNDS=30; TAGPRE="pofds3f"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL" && -n "$PROVIDER" ]] || { echo "need --model and --provider" >&2; exit 2; }

# THE KEY IS NEVER AN ARGUMENT. It comes from the environment or a file,
# so it cannot land in shell history, a process listing, or this script.
if [[ -z "${OPENROUTER_API_KEY:-}" && -z "${OPENROUTER_API_KEY_FILE:-}" ]]; then
  echo "no key: export OPENROUTER_API_KEY, or set OPENROUTER_API_KEY_FILE" >&2
  exit 2
fi

# PREFLIGHT. Resolve the (possibly canonical-slug) model to a routable id
# and derive the decoding policy from LIVE endpoint metadata, so a knob the
# endpoint does not expose is never sent -- with require_parameters=true
# that would make the pinned endpoint ineligible and the route fail. Free:
# public metadata, no key. Fails hard on a non-ZDR provider.
PF="$(python "${REPO}/experiments/scripts/cluster_pipelines/or_preflight.py" \
        --model "$MODEL" --provider "$PROVIDER" --seed "$SEED")" || {
  echo "[local] preflight failed; not spending anything" >&2; exit 2; }
echo "$PF" | grep '^#' || true
eval "$(echo "$PF" | grep '^export ')"
MODEL="$OR_MODEL"

# the model token must match gen_frontier_icl.model_token exactly, or the
# local run and the Condor manifest would disagree about which cell this is
MTOK="$(python - "$MODEL" "$PROVIDER" <<'PY'
import re, sys
slug, prov = sys.argv[1], sys.argv[2]
print(f"{re.sub(r'[^a-z0-9]+','',slug.split('/')[-1].lower())}-"
      f"{re.sub(r'[^a-z0-9]+','',prov.lower())}")
PY
)"
TAG="${TAGPRE}_${MTOK}_d8_greedy_sw100_eaopen_w1_k1_esopen_anch2_s${SEED}_r${ROUNDS}"
OUT="${REPO}/runs/pokec_gated_lm/${TAG}"

# +2% head-room on the request cap; the COST cap is the real control
MAXREQ=$(python -c "print(int(723*${ROUNDS}*1.02))")

echo "[local] cell   : ${TAG}"
echo "[local] rounds : ${ROUNDS}  agents: 723  requests: $((723*ROUNDS))"
echo "[local] caps   : ${MAXREQ} requests / \$${MAXCOST} realized"
echo "[local] out    : ${OUT}"
if [[ -f "${OUT}/trajectory.pt" ]]; then
  echo "[local] already complete; delete the dir to re-run." ; exit 0
fi

cd "$REPO"
RUN_TAG="$TAG" OUT_DIR="$OUT" \
DATASET=movielens ML_TARGET=Action \
MODEL_BACKEND=openrouter OR_MODEL="$MODEL" OR_PROVIDER="$PROVIDER" \
OR_MAX_TOKENS=16 OR_TEMPERATURE="$OR_TEMPERATURE" OR_SEED="$OR_SEED" \
OR_TOP_P=1 OR_REASONING_DISABLED=1 \
OR_CONCURRENCY="$CONC" OR_RPS="$RPS" OR_MAX_REQUESTS="$MAXREQ" \
OR_MAX_COST="$MAXCOST" OR_CACHE="${OUT}/or_cache.sqlite" \
TRAINING_STYLE=frozen SFT_EPOCHS=0 USE_LORA=0 KL_BETA=0 \
FRESH_EACH_ROUND=0 LOG_PERPLEXITY=0 LOG_ANSWER_DIST=0 ANS_SAMPLE_K=0 \
PARSE_MODE=strict SAVE_RAW_GEN=1 \
ICL_K=0 ICL_DAYS=8 ICL_SELECT=random ICL_CTX_SOURCE=live \
POP_MODEL=ab AI_GATE_MODE=all_open PEER_GATE_MODE=all_open \
EPS_AI=1.0 EPS_SOCIAL=0.2 W_PLAT=1 INNATE_LAMBDA=1 DEFFUANT_ALPHA=0.5 \
AB_SWEEPS=100 N_ROUNDS="$ROUNDS" WITH_TWIN=1 \
TRAIN_CAP=723 N_LABELED=723 SEED="$SEED" SEED_BASE_DATA=1 \
WANDB_MODE=disabled WANDB_DISABLED=true USE_TF=0 \
python experiments/scripts/cluster_pipelines/run_pokec_gated_lm.py

echo "[local] done. Gate it with:"
echo "  python experiments/scripts/cluster_pipelines/check_section3_frontier_icl.py \\"
echo "      --runs-root ${REPO}/runs/pokec_gated_lm $([[ $ROUNDS == 3 ]] && echo --smoke)"
