#!/usr/bin/env bash
# Refuse to start PAID work when the machine cannot carry it.
#
# WHY. Every OS kill under memory pressure bills the requests that were
# in flight and never reached the cache: on 2026-08-31 that was most of
# the gap between $11.76 of cached responses and ~$19.5 of actual spend.
# A resource check before each model is therefore a COST control, not
# just a hygiene check.
#
#   resource_ok "label"   -> 0 to proceed, 1 to stop
MIN_FREE_GB="${MIN_FREE_GB:-8}"
MIN_MEM_FREE_PCT="${MIN_MEM_FREE_PCT:-15}"
MIN_SWAP_FREE_MB="${MIN_SWAP_FREE_MB:-256}"

resource_ok () {
  local label="${1:-}" free_gb mem_pct swap_free ok=0
  free_gb=$(df -g /System/Volumes/Data | tail -1 | awk '{print $4}')
  mem_pct=$(memory_pressure 2>/dev/null | awk -F': *' '/free percentage/{gsub(/%/,"",$2); print $2}')
  # "total = X  used = Y  free = Z  (encrypted)" -- take the field AFTER
  # the literal "free", not a fixed column, so the trailing "(encrypted)"
  # cannot be parsed as the number.
  swap_free=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="free"){gsub(/M/,"",$(i+2)); print $(i+2); exit}}')
  [ -n "$mem_pct" ] || mem_pct=100
  [ -n "$swap_free" ] || swap_free=99999
  echo "[guard] $label: disk ${free_gb}GB free | mem ${mem_pct}% free | swap ${swap_free}MB free"
  if [ "${free_gb%.*}" -lt "$MIN_FREE_GB" ]; then
    echo "[guard] STOP: free disk ${free_gb}GB is below the ${MIN_FREE_GB}GB floor"; ok=1
  fi
  if [ "${mem_pct%.*}" -lt "$MIN_MEM_FREE_PCT" ]; then
    echo "[guard] STOP: memory pressure critical (${mem_pct}% free)"; ok=1
  fi
  if [ "${swap_free%.*}" -lt "$MIN_SWAP_FREE_MB" ]; then
    echo "[guard] STOP: swap nearly exhausted (${swap_free}MB free) -- a kill"
    echo "[guard]       mid-round bills requests that never reach the cache"; ok=1
  fi
  [ "$ok" -eq 0 ] && echo "[guard] ok to proceed"
  return $ok
}
