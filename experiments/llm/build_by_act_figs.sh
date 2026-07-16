#!/usr/bin/env bash
# Regenerate experiments/llm/figs/by_act/ -- a browsable tree of RELATIVE
# symlinks grouping existing figures by the two-act paper structure
# (see notes/PAPER_MAP.md). Zero script changes: the real files stay in
# figs/qwen/ and figs/. Idempotent: wipes and rebuilds by_act/ each run.
# Re-run whenever new figures land (ICL, the serve x train hinge, etc.).
#
# Run from repo root:  bash experiments/llm/build_by_act_figs.sh
set -euo pipefail

FIGS="experiments/llm/figs"
ROOT="$FIGS/by_act"
rm -rf "$ROOT"

# link <section-dir> <fig-file-relative-to-$FIGS>...
link() {
  local section="$1"; shift
  local dir="$ROOT/$section"
  mkdir -p "$dir"
  for f in "$@"; do
    if [[ -e "$FIGS/$f" ]]; then
      # $dir is $FIGS/by_act/<act>/<section> = 3 levels below $FIGS
      ln -s "../../../$f" "$dir/$(basename "$f")"
    else
      echo "  MISSING (skipped): $f" >&2
    fi
  done
}

# ---- ACT I: the non-closed loop ----
link act1_open_loop/I1_population_alone \
  fig_pop_alone.png
link act1_open_loop/I2_frozen_serve \
  qwen/frozen_attractor_dists.png qwen/crossmodel_drift.png qwen/olmo_frozen_retrain.png
link act1_open_loop/I3_nf_train_only \
  qwen/nf_ladder.png
# the serve x train 2x2 that ties Act I -> Act II
link hinge/serve_x_train \
  fig_hinge_2x2.png

# ---- ACT II: the closed loop ----
link act2_closed_loop/II1_prior \
  qwen/frozen_attractor_dists.png qwen/crossmodel_drift.png qwen/olmo_frozen_retrain.png \
  qwen/demo_probe_map.png qwen/demo_probe_map_llama.png qwen/demo_probe_map_olmo.png \
  qwen/demo_probe_map_gemma.png
link act2_closed_loop/II2a_kl_anchor \
  qwen/two_object_plane.png qwen/two_object_plane_e010.png qwen/two_object_plane_regimes.png \
  qwen/ref_ppl_diagnostic.png qwen/knob_ppl_lines.png
link act2_closed_loop/II2b_data_regime \
  qwen/ppl_lines_median.png qwen/unanchored_regime_bars.png qwen/regime_vs_beta.png \
  qwen/ppl_vs_feature_strength.png
link act2_closed_loop/II2c_dials_mixing \
  qwen/gate_boundary.png qwen/homophily_resists.png qwen/cleaneps_opinion_dists.png \
  qwen/fdial_llm_dr_vr.png qwen/fdial_llm_dr_vr_a010.png qwen/fdial_llm_dr_vr_continual.png
link act2_closed_loop/II2d_mediation \
  qwen/stag_mediation.png
link act2_closed_loop/II2e_social_structure \
  qwen/dr2_emergence.png qwen/tree3_gap_lines.png \
  qwen/demo_probe_age_curves.png qwen/demo_probe_age_curves_llama.png \
  qwen/demo_probe_age_curves_olmo.png qwen/demo_probe_age_curves_gemma.png
# II3_in_context: GAP -- ICL figures not built yet (data pending). Placeholder dir:
mkdir -p "$ROOT/act2_closed_loop/II3_in_context"

echo "Built $ROOT:"
find "$ROOT" -type l | sort | sed 's/^/  /'
echo "Sections: $(find "$ROOT" -type d -mindepth 2 | wc -l | tr -d ' '); links: $(find "$ROOT" -type l | wc -l | tr -d ' ')"
