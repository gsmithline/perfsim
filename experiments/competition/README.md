# Mediated opinion dynamics experiments

Active project: a nonlinear opinion-dynamics population (NDlib bounded
confidence: Hegselmann-Krause or Deffuant with algorithmic bias) coupled in
closed loop to a learning platform — an MLP retrained every round on the
current (platform-mediated) opinions, anchored to a fixed prior P0 (the KL
analogue), feeding back through the same confidence gate (agents absorb a
prediction only if it is within epsilon of their view). Questions: phase
transitions in the population showing up in the model's retraining dynamics,
collapse thresholds, contamination (s_t), defenses, and platform-side
telemetry. Run everything from the repo root.

| File | What it does |
|------|--------------|
| `_ndlib_fixed.py` | corrected NDlib HK iteration (upstream writes only the last node per sweep) |
| `_ab_probe.py` | pure AlgorithmicBias population probe: epsilon x gamma fragmentation map |
| `08_ndlib_mlp_loop.py` | HK + MLP loop: epsilon x w sweep; staircase in model loss; platform lowers critical epsilon |
| `08c_hysteresis.py` | epsilon ramps: HK alone has one-way hysteresis, platform closes the loop |
| `08d_constant_anchor.py` | constant-P0 anchor: capture vs one-round detachment, boundary anti-monotone in anchor weight |
| `09_ab_mlp_loop.py` | AlgorithmicBias + MLP loop: gamma=1.5 fragmentation healed at exactly the prior |
| `10_collapse_metrics.py` | generation curves: collapse threshold in rate, slow leak below threshold |
| `11_self_consuming.py` | direct self-consumption vs mediated: mediation creates the threshold and the diversity floor |
| `12_accumulate.py` | accumulate vs replace vs round0-mix: only a fixed pristine share protects |
| `13b_two_priors_hard.py` | two platforms, hard assignment: distinct priors restore pluralism (3/4 cells) |
| `14_pokec_real.py` | real Pokec graph + innate: epsilon axis survives, gamma neutered by sparsity |
| `15_politisky.py` | PolitiSky24 (real polarized Bluesky data): centrist healing, partisan flip vs strand |
| `16_model_mass.py` | torch-vectorized instrumented population with provenance tags; s_t law, exposure clock, defenses, canary corners |
| `17_three_figs.py` | polished figures: master-clock search (negative), lagged-crawl defense, canary flow meter |
| `18_platform_telemetry.py` | training-side panel (L_init, displacement, 2x2 matrix, leash); (s_t, eps) recoverable from telemetry alone |
| `fig_ndlib_spaghetti.py` | per-node opinion trajectories via NDlib's OpinionEvolution |
| `fig_diffusion_3d.py` | diffusion-style snapshots: scatter + 3D density over (x_t, x0) |

Figures and JSON land in `figs/`. PolitiSky24 data in `experiments/data/politisky24/`.

`circle/` — parked earlier project: K MLP platforms competing on a circular
opinion space (Salop-style) over a circular-FJ population. Phase grids,
simplex/flow portraits, speed-drift toy. Self-contained, paths updated; its
figures in `circle/figs/`.

`superseded/` — kept as records, do not build on: `07_ndlib_one_platform.py`
(built on NDlib's ARWHK, which has a reset-to-zero bug), `08b_extreme_prior.py`
(prediction-space anchor artifact, fixed by 08d), `13_two_priors.py`
(averaging feedback rule merges two priors into one centrist platform,
fixed by 13b).
