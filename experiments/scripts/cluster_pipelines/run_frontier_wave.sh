#!/usr/bin/env bash
# THE ONE production launcher for the frontier ICL wave.
#
# Every rule here exists because its absence caused a real failure on
# 2026-09-05, when the wave was launched twice against a single cache:
#   * an ATOMIC LOCK (mkdir is atomic on POSIX) refuses a second launch;
#   * set -euo pipefail, so an OS-killed runner cannot be reported as
#     success by a grep at the end of a pipe;
#   * the wave ABORTS on the first cell failure rather than continuing;
#   * stdout AND stderr are kept UNFILTERED in the log -- the previous
#     launcher piped through grep and threw the diagnosis away;
#   * PRODUCTION TAGS (pofds3f_), never the smoke prefix;
#   * cells run SEQUENTIALLY, never concurrently;
#   * the resource guard runs before every cell;
#   * a $275 wave cap is checked against LIVE account usage before each
#     cell, charging the cell's worst case;
#   * the key is read from the Keychain per process and never printed,
#     never an argument, never written to the log.
set -euo pipefail

REPO=/Users/gabesmithline/Desktop/ellis_work.nosync/perfsim
LOCK="${REPO}/runs/pokec_gated_lm/.frontier_wave.lock"
LOG="${WAVE_LOG:-${REPO}/runs/pokec_gated_lm/frontier_wave.log}"
CAP="${WAVE_CAP:-275}"
OVERRUN="${OVERRUN_FACTOR:-1.7}"
CONC="${CONC:-12}"
RPS="${RPS:-8}"
ROUNDS="${ROUNDS:-20}"

# ---- atomic single-instance lock ---------------------------------------
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "REFUSING TO LAUNCH: $LOCK exists (pid $(cat "$LOCK/pid" 2>/dev/null))." >&2
  echo "A wave is already running, or a previous one died. If it is dead," >&2
  echo "remove the lock deliberately: rmdir --ignore-fail-on-non-empty $LOCK" >&2
  exit 1
fi
echo $$ > "$LOCK/pid"
cleanup () { rm -rf "$LOCK"; }
trap cleanup EXIT

cd "$REPO"
source experiments/scripts/cluster_pipelines/resource_gate.sh

key () { security find-generic-password -w -s openrouter -a "$USER"; }
usage_now () {
  OPENROUTER_API_KEY="$(key)" python -c "
import sys; sys.path.insert(0,'$REPO')
from perfsim.models.openrouter_client import validate_key
print(validate_key()['usage'])" 2>/dev/null
}

START="$(usage_now)"
[ -n "$START" ] || { echo "cannot read account usage; refusing to spend" >&2; exit 2; }
echo "[wave] pid $$  start usage \$$START  cap \$$CAP  rounds $ROUNDS  conc $CONC rps $RPS"

cell () {   # slug provider depth est_usd
  local now spent room worst
  now="$(usage_now)"
  spent="$(python -c "print(f'{float($now)-float($START):.4f}')")"
  room="$(python -c "print(f'{float($CAP)-float($spent):.4f}')")"
  worst="$(python -c "print(f'{float($4)*float($OVERRUN):.4f}')")"
  echo "[wave] spent \$$spent of \$$CAP | next $1 D=$3 est \$$4 worst \$$worst"
  if python -c "import sys; sys.exit(0 if float('$worst')>float('$room') else 1)"; then
    echo "[wave] STOP: worst case \$$worst exceeds remaining \$$room" >&2
    return 9
  fi
  resource_ok "$1 D=$3" || { echo "[guard] refusing to start" >&2; return 8; }
  echo "=== CELL $1 D=$3 via $2 ($ROUNDS rounds) ==="
  OPENROUTER_API_KEY="$(key)" OR_MAX_RETRIES=12 \
  experiments/scripts/cluster_pipelines/run_frontier_local.sh \
    --model "$1" --provider "$2" --seed 0 --production --rounds "$ROUNDS" \
    --icl-days "$3" --max-cost "$worst" --concurrency "$CONC" --rps "$RPS" \
    --allow-no-max-tokens
  echo "=== CELL DONE $1 D=$3 ==="
}

cell moonshotai/kimi-k3-20260715      "Morph"          0 25
cell moonshotai/kimi-k3-20260715      "Morph"          8 25
cell openai/gpt-5.6-sol-20260709      "Azure"          0 35
cell openai/gpt-5.6-sol-20260709      "Azure"          8 35
cell anthropic/claude-opus-5-20260723 "Amazon Bedrock" 0 40
cell anthropic/claude-opus-5-20260723 "Amazon Bedrock" 8 40
echo "[wave] ALL SIX CELLS COMPLETE"
