#!/usr/bin/env bash
# Run a list of cells under a WAVE-LEVEL cost cap.
#
# WHY THIS EXISTS. OR_MAX_COST bounds ONE cell. Six per-cell caps summing
# to $37 ran a $20 balance $11 over on 2026-08-31 without a single cap
# firing, because nothing bounded the TOTAL. This checks the authoritative
# figure -- the account's own usage -- before each cell, and refuses to
# start one that could carry the wave past the cap.
#
# It also charges the cell's WORST CASE against the cap, not its estimate:
# billed-but-uncached requests (retries, and in-flight requests billed when
# the OS kills a run under memory pressure) have historically made actual
# spend ~1.65x the cached-response estimate.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WAVE_CAP="${WAVE_CAP:?set WAVE_CAP to the maximum additional USD for this wave}"
OVERRUN_FACTOR="${OVERRUN_FACTOR:-1.7}"

usage_now () {
  OPENROUTER_API_KEY="$(security find-generic-password -w -s openrouter -a "$USER")" \
  python -c "
import sys; sys.path.insert(0,'$REPO')
from perfsim.models.openrouter_client import validate_key
print(validate_key()['usage'])" 2>/dev/null
}

START="$(usage_now)"
[ -n "$START" ] || { echo "cannot read account usage; refusing to spend" >&2; exit 2; }
echo "[wave] start usage \$$START   cap +\$$WAVE_CAP   overrun factor ${OVERRUN_FACTOR}x"

run_cell () {  # slug provider depth est_usd [reasoning]
  local now spent room worst
  now="$(usage_now)"
  spent="$(python -c "print(f'{float($now)-float($START):.4f}')")"
  room="$(python -c "print(f'{float($WAVE_CAP)-float($spent):.4f}')")"
  worst="$(python -c "print(f'{float($4)*float($OVERRUN_FACTOR):.4f}')")"
  echo "[wave] spent \$$spent of \$$WAVE_CAP; next cell est \$$4 (worst \$$worst)"
  if python -c "import sys; sys.exit(0 if float('$worst') > float('$room') else 1)"; then
    echo "[wave] STOP: worst case \$$worst exceeds remaining \$$room. Not starting"
    echo "[wave]       $1 D=$3. Nothing further will be spent."
    return 9
  fi
  OPENROUTER_API_KEY="$(security find-generic-password -w -s openrouter -a "$USER")" \
  OR_MAX_RETRIES=10 caffeinate -i \
  "$REPO/experiments/scripts/cluster_pipelines/run_frontier_local.sh" \
    --model "$1" --provider "$2" --seed 0 --rounds "${ROUNDS:-3}" \
    --icl-days "$3" --max-cost "$worst" --concurrency "${CONC:-3}" \
    --rps "${RPS:-2}" ${5:+--reasoning-mode "$5"} --allow-no-max-tokens 2>&1 \
  | grep -E "round [0-9]|realized|loop done|Retry|Budget|Error"
}
