# cluster_pipelines

Run from the repo root (`python experiments/scripts/cluster_pipelines/<file>.py`). The
matching Condor wrappers in `experiments/condor/run_one_*.sh` point here.

- `run_pokec_fj_lm.py` — Pokec FJ + LM deployment-schedule loop (uses `_collapse_metrics.py`)
- `run_pokec_fj_competition.py`, `run_pokec_fj_hunt.py` — competing-platform loops
- `run_covid_lm.py`, `run_covid_perfgd.py`, `run_covid_nomodel.py`, `run_covid_sensitivity.py`, `covid_abm_sweep.py` — covid AT loop variants
- `calibrate_covid.py`, `calibrate_covid_single.py` — calibration
- `fj_baselines_local.py`, `fj_schedules_local.py`, `fj_beta_dissociation.py`, `fj_mlp_homogenize.py` — local FJ / MLP analyses
- `grad_diagnostics.py` — gradient checks
- `_collapse_metrics.py` — shared collapse-metric helpers (imported `__file__`-relative)
