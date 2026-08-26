#!/usr/bin/env bash
# The lambda = infinity column of Figure 4 (anchor trade-off wave, F4A).
# CPU ONLY -- no GPU, no Condor, no model load.  Safe to run on a laptop.
#
# WHY THIS IS NOT A GPU JOB.  A frozen model prompted with ICL_K = D = 0
# never sees the population: the prompt is the agent's static profile and
# nothing trains, so its parsed prediction vector is a CONSTANT, identical
# in every round and independent of eps_AI and eps_social.  The archived
# zero-shot-prior runs verify that empirically (one sha256 per checkpoint;
# the Qwen2.5-7B prior is additionally bit-identical across seven archived
# H100 K=0 cells on five different hosts).  replay_frozen_offline.py loads
# that real vector from the archived H100 zsprior run and replays the
# population and peer process around it through
# sim_perfect_predictor.simulate -- the IDENTICAL operator path
# (gp.nested_presocial_update with gate_on="anchor", then gp.ab_sweep) the
# GPU runs take, not a second copy of the dynamics.
#
# F4A OPERATOR (per experiments/condor/gen_pofd_sweep.py F4A block):
#   AI gate all_open (EPS_AI inert), social gate THRESHOLD at es in
#   {0.05, 0.2}, ONE Deffuant sweep per round, alpha 0.5, 30 rounds,
#   seed 0, beta = W_PLAT in {0, .25, .5, .75, 1}, gamma = INNATE_LAMBDA
#   in {1, .5, .2, 0}.
#
# 60 ARTIFACTS, named per gen_pofd_sweep.f4a_frozen_name:
#   frozen_f4a_{model}_w{beta}_k{gamma}_es{es}_sw100_r30.pt
#   per es (30): per model 12 (beta in {.25,.5,.75} x 4 gammas) + 1
#   (beta = 1, gamma = 1: z = served exactly, gamma drops out, so ONE
#   artifact stands in for every gamma) + 4 SHARED beta = 0 cells
#   (model "shared": z = h exactly, the served vector is never read, so
#   the replay is model-independent and equals the no-AI twin).  The
#   shared cells are computed from the Qwen3-8B vector; when the Qwen2.5-7B
#   zsprior run dir is present they are RECOMPUTED from that vector and
#   asserted bit-identical (op_raw AND twin_raw), which is the operator-
#   level proof that beta = 0 ignores the model.
#
# SOURCES.  qwen3_8b: notes/pofd/cluster/pofdzsprior_qwen3_8b_w0p5_l0p2_es0_s0
# (sha pinned below, --expect-sha makes a wrong source a hard error).
# qwen7b: notes/pofd/cluster/pofdzsprior_qwen7b_w0p5_l0p2_es0_s0 -- does
# not exist until the F4A smoke sub runs and the dir is pulled; every
# qwen7b-sourced artifact is SKIPPED with a message until then.  Once it
# is pulled, set QWEN7B_SHA below to the sha256 recorded by
# check_fig4_anchor.py (zsprior verdict).  The archived H100 Qwen2.5-7B
# K=0 prior (seven pofdqmech_/pofdfam_ k0 cells, 2026-08-1x) hashed to
#   1674ee5f8d833f46de672791d933e1d3bdeefb07484c2d110dec84ce71da30bb
# so that is the value the fresh run is EXPECTED to reproduce; it is
# deliberately NOT pre-filled -- pin what the fresh run actually served.
#
# NAMING.  replay_frozen_offline.py writes its own canonical name
# (frz_k{K}_w{W}_eaopen_es{ES}_sw1_s0_r30.pt) and refuses to overwrite,
# so each cell is generated into a per-source staging dir and MOVED to the
# f4a_frozen_name; replay_frozen_offline.py's defaults are not touched.
#
# Re-runnable: every cell whose final artifact exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/../../.."

OUT="${OUT:-notes/pofd/fig4_anchor/frozen}"     # sw1 sibling: OUT=notes/pofd/fig4_anchor/frozen_sw1 SWEEPS=1
ROUNDS=30
SWEEPS="${SWEEPS:-100}"   # primary wave = 100 Deffuant sweeps; SWEEPS=1 builds the sibling
SEED=0
ALPHA=0.5
EPS_AI=1.0

# HARDWARE CLASS (2026-08-26): the H100 pool vanished from the cluster, so the
# whole Figure-4 anchor wave -- both zero-shot priors included -- runs on
# A100-SXM4-80GB; replay_frozen_offline refuses a source served elsewhere.
EXPECT_GPU="NVIDIA A100-SXM4-80GB"
QWEN3_SRC="notes/pofd/cluster/pofdzsprior_qwen3_8b_w0p5_l0p2_es0_a100_s0"
QWEN3_SHA="8d63ac2cc99b3aa83fb4ac87b9486fa62baff7a64f39f8509f32e52fb527005c"        # fill from the F4A checker's zsprior verdict once the A100 prior is pulled
QWEN7B_SRC="notes/pofd/cluster/pofdzsprior_qwen7b_w0p5_l0p2_es0_a100_s0"
QWEN7B_SHA="4f9822a13eff8c457fcfe5037baffae2a87975b178d65b1f9343f7d404044303"       # fill from the F4A checker's zsprior verdict once pulled

ES_LIST="0.05 0.2"
MODELS="qwen3_8b qwen7b"
BETAS_GAMMA_GRID="0.25 0.5 0.75"     # beta > 0, < 1: all four gammas
GAMMAS="1 0.5 0.2 0"
BETA0_SHARED_MODEL="qwen3_8b"        # the vector the shared cells are computed from

REPLAY="experiments/scripts/cluster_pipelines/replay_frozen_offline.py"
RUN="env USE_TF=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3"

num () { printf '%s' "$1" | sed 's/\./p/'; }

src_of () {
  case "$1" in
    qwen3_8b) printf '%s' "$QWEN3_SRC" ;;
    qwen7b)   printf '%s' "$QWEN7B_SRC" ;;
    *) echo "[f4a-frozen] unknown model slug $1" >&2; exit 2 ;;
  esac
}
sha_of () {
  case "$1" in
    qwen3_8b) printf '%s' "$QWEN3_SHA" ;;
    qwen7b)   printf '%s' "$QWEN7B_SHA" ;;
  esac
}
# 0 = usable, 1 = skip (absent), 2 = present but sha unpinned (hard error)
source_state () {
  local m="$1" d
  d="$(src_of "$m")"
  if [ ! -f "$d/trajectory.pt" ] || [ ! -f "$d/config.json" ]; then
    return 1
  fi
  if [ -z "$(sha_of "$m")" ]; then
    return 2
  fi
  return 0
}

final_name () {   # model beta gamma es  -> f4a_frozen_name
  printf 'frozen_f4a_%s_w%s_k%s_es%s_sw%s_r%s.pt' "$1" "$(num "$2")" "$(num "$3")" "$(num "$4")" "$SWEEPS" "$ROUNDS"
}
default_name () { # beta gamma es -> replay_frozen_offline's own artifact name
  printf 'frz_k%s_w%s_eaopen_es%s_sw%s_s%s_r%s.pt' "$(num "$2")" "$(num "$1")" "$(num "$3")" "$SWEEPS" "$SEED" "$ROUNDS"
}

# replay_one <model-source> <beta> <gamma> <es> <stage-dir>  -> path of the produced file
replay_one () {
  local m="$1" W="$2" K="$3" ES="$4" stage="$5" d sha def
  d="$(src_of "$m")"; sha="$(sha_of "$m")"; def="$(default_name "$W" "$K" "$ES")"
  mkdir -p "$stage"
  rm -f "$stage/$def"
  $RUN "$REPLAY" \
    --from-run "$d" \
    --expect-sha "$sha" \
    --expect-gpu "$EXPECT_GPU" \
    --innate-k "$K" \
    --w-plat "$W" \
    --eps-social "$ES" \
    --eps-ai "$EPS_AI" \
    --ai-gate-mode all_open \
    --peer-gate-mode threshold \
    --sweeps "$SWEEPS" \
    --alpha "$ALPHA" \
    --rounds "$ROUNDS" \
    --seed "$SEED" \
    --out-dir "$stage" \
    --quiet
  if [ ! -f "$stage/$def" ]; then
    echo "[f4a-frozen] ERROR: expected $stage/$def after the replay; not found" >&2
    exit 3
  fi
  printf '%s' "$stage/$def"
}

mkdir -p "$OUT"
n_made=0; n_skip_exist=0; n_skip_src=0
t0=$(date +%s)

for m in $MODELS; do
  st=0; source_state "$m" || st=$?
  case "$st" in
    0) echo "[f4a-frozen] source $m: $(src_of "$m") sha $(sha_of "$m")" ;;
    1) echo "[f4a-frozen] source $m: $(src_of "$m") ABSENT -- every $m-sourced artifact is skipped (pull the zsprior run dir, pin its sha, re-run)" ;;
    2) echo "[f4a-frozen] ERROR: source $m present at $(src_of "$m") but its sha is not pinned in this script; set it from the checker's zsprior verdict" >&2; exit 4 ;;
  esac
done

for ES in $ES_LIST; do
  # ---- beta in {.25,.5,.75} x 4 gammas, plus beta = 1 at gamma = 1, per model
  for m in $MODELS; do
    st=0; source_state "$m" || st=$?
    for W in $BETAS_GAMMA_GRID 1; do
      if [ "$W" = "1" ]; then klist="1"; else klist="$GAMMAS"; fi
      for K in $klist; do
        fin="$OUT/$(final_name "$m" "$W" "$K" "$ES")"
        if [ -f "$fin" ]; then
          echo "[f4a-frozen] SKIP (exists) $fin"; n_skip_exist=$((n_skip_exist+1)); continue
        fi
        if [ "$st" != "0" ]; then
          echo "[f4a-frozen] SKIP (no $m source) $fin"; n_skip_src=$((n_skip_src+1)); continue
        fi
        echo "[f4a-frozen] $m es=$ES beta=$W gamma=$K"
        p="$(replay_one "$m" "$W" "$K" "$ES" "$OUT/.stage_$m")"
        mv "$p" "$fin"; n_made=$((n_made+1))
      done
    done
  done
  # ---- beta = 0: model-independent (z = h; the served vector is never read)
  st=0; source_state "$BETA0_SHARED_MODEL" || st=$?
  st7=0; source_state qwen7b || st7=$?
  for K in $GAMMAS; do
    fin="$OUT/$(final_name shared 0 "$K" "$ES")"
    if [ -f "$fin" ]; then
      echo "[f4a-frozen] SKIP (exists) $fin"; n_skip_exist=$((n_skip_exist+1))
    elif [ "$st" != "0" ]; then
      echo "[f4a-frozen] SKIP (no $BETA0_SHARED_MODEL source) $fin"; n_skip_src=$((n_skip_src+1)); continue
    else
      echo "[f4a-frozen] shared es=$ES beta=0 gamma=$K (from $BETA0_SHARED_MODEL)"
      p="$(replay_one "$BETA0_SHARED_MODEL" 0 "$K" "$ES" "$OUT/.stage_shared")"
      mv "$p" "$fin"; n_made=$((n_made+1))
    fi
    # model-independence assertion: recompute from the OTHER vector and
    # demand bit-identical op_raw and twin_raw
    if [ "$st7" = "0" ]; then
      echo "[f4a-frozen]   beta=0 cross-check from qwen7b vector"
      p7="$(replay_one qwen7b 0 "$K" "$ES" "$OUT/.stage_qwen7b_b0")"
      $RUN - "$fin" "$p7" <<'PY'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
assert a["config"]["w_plat"] == 0.0 == b["config"]["w_plat"]
assert a["config"]["base_model"] != b["config"]["base_model"], "cross-check must use a different source model"
assert torch.equal(a["op_raw"], b["op_raw"]), "beta=0 op_raw depends on the served vector -- operator is NOT model-independent at beta=0"
assert torch.equal(a["twin_raw"], b["twin_raw"]), "twin differs between sources"
assert torch.equal(a["op_raw"], a["twin_raw"]), "beta=0 population != twin"
print(f"[f4a-frozen]   OK: beta=0 replay identical from {a['config']['base_model']} and {b['config']['base_model']}")
PY
      rm -f "$p7"
    else
      echo "[f4a-frozen]   beta=0 cross-check from qwen7b vector SKIPPED (no qwen7b source)"
    fi
  done
done

rmdir "$OUT/.stage_qwen3_8b" "$OUT/.stage_qwen7b" "$OUT/.stage_shared" "$OUT/.stage_qwen7b_b0" 2>/dev/null || true
t1=$(date +%s)

# ---- verify every artifact on disk carries the F4A pins the analyzer checks
$RUN - "$OUT" "$ROUNDS" "$SWEEPS" "$SEED" "$ALPHA" <<'PY'
import sys, glob, os, re, torch
out, rounds, sweeps, seed, alpha = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
rx = re.compile(r"^frozen_f4a_(qwen3_8b|qwen7b|shared)_w(0|0p25|0p5|0p75|1)_k(1|0p5|0p2|0)_es(0p05|0p2)_sw(\d+)_r(\d+)\.pt$")
src_model = {"qwen3_8b": "Qwen/Qwen3-8B", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
def f(tok): return float(tok.replace("p", "."))
bad = 0; n = 0
for p in sorted(glob.glob(os.path.join(out, "frozen_f4a_*.pt"))):
    m = rx.match(os.path.basename(p))
    if not m: print("[f4a-frozen] BAD NAME", p); bad += 1; continue
    slug, w, k, es, sw, r = m.groups()
    d = torch.load(p, map_location="cpu", weights_only=False); c = d["config"]
    errs = []
    if c.get("platform") != "frozen_offline_replay": errs.append("platform")
    if c.get("population_update") != "nested_ai_anchored_then_social_v2": errs.append("population_update")
    if slug != "shared" and c.get("base_model") != src_model[slug]: errs.append(f"base_model={c.get('base_model')}")
    if slug == "shared" and c.get("base_model") not in src_model.values(): errs.append("base_model")
    if c.get("w_plat") != f(w): errs.append("w_plat")
    if c.get("innate_k") != f(k): errs.append("innate_k")
    if c.get("eps_social") != f(es): errs.append("eps_social")
    if c.get("ab_sweeps") != int(sw) or int(sw) != sweeps: errs.append("ab_sweeps")
    if c.get("rounds") != int(r) or int(r) != rounds: errs.append("rounds")
    if c.get("seed") != seed: errs.append("seed")
    if c.get("ai_gate_mode") != "all_open": errs.append("ai_gate_mode")
    if c.get("peer_gate_mode") != "threshold": errs.append("peer_gate_mode")
    if c.get("deffuant_alpha") != alpha: errs.append("deffuant_alpha")
    if c.get("gamma_bias") != 0.0: errs.append("gamma_bias")
    if "replay_note" not in c or "frozen_pred_sha256" not in c: errs.append("provenance")
    for key in ("op_raw", "twin_raw", "pred_raw"):
        t = d[key]
        if tuple(t.shape) != (rounds, 723) or not bool(torch.isfinite(t).all()): errs.append(key)
    if not bool((d["pred_raw"] == d["pred_raw"][0]).all()): errs.append("pred not constant")
    if slug == "shared" and not torch.equal(d["op_raw"], d["twin_raw"]): errs.append("beta0 op!=twin")
    if float(d["op_raw"].min()) < 0 or float(d["op_raw"].max()) > 1: errs.append("op range")
    n += 1
    if errs: bad += 1; print("[f4a-frozen] BAD", os.path.basename(p), errs)
print(f"[f4a-frozen] verified {n} artifacts, {bad} bad")
sys.exit(1 if bad else 0)
PY

echo "[f4a-frozen] done: generated $n_made, skipped-existing $n_skip_exist, skipped-no-source $n_skip_src, $((t1 - t0)) s -- $(ls "$OUT"/frozen_f4a_*.pt 2>/dev/null | wc -l | tr -d ' ') / 60 artifacts under $OUT"
