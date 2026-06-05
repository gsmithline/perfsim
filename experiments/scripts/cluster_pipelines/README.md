# cluster_pipelines

Earlier experiment scripts (covid, Pokec FJ, Schelling, FJ/MLP locals, calibration).
The current AI-mediated / market-collapse work lives in `../ai_mediated/`.

Run from the repo root (`python experiments/scripts/cluster_pipelines/<file>.py`). The
matching Condor wrappers in `experiments/condor/run_one_*.sh` point here.

- `run_pokec_fj_lm.py` — Pokec FJ + LM deployment-schedule loop (uses `_collapse_metrics.py`)
- `run_covid_lm.py`, `run_covid_perfgd.py`, `run_covid_nomodel.py`, `run_covid_sensitivity.py`, `covid_abm_sweep.py` — covid AT loop variants
- `run_schelling_lm.py` — Schelling LM loop
- `calibrate_covid.py`, `calibrate_covid_single.py`, `calibrate_schelling.py` — calibration
- `fj_baselines_local.py`, `fj_schedules_local.py`, `fj_beta_dissociation.py`, `fj_mlp_homogenize.py`, `mlp_beta_sweep.py` — local FJ / MLP analyses
- `grad_diagnostics.py` — gradient checks
- `_collapse_metrics.py` — shared collapse-metric helpers (imported `__file__`-relative)
