# Archived / parked experiment lines

Superseded or parked work moved out of the active tree to reduce clutter.
**Nothing is deleted** — files are moved here and full git history is preserved
(git records the moves as renames on the next commit). Archived 2026-06-16.

The **active** project is the gated NDlib mediated-collapse loop:
- `experiments/competition/` — MLP + theory blocks (08–23), `theory_contamination.*`, `findings_empirical.tex`
- `experiments/scripts/cluster_pipelines/run_pokec_gated_lm.py` (+ `_gated_pop.py`, `_collapse_metrics.py`, `_mock_gated_test.py`)
- `experiments/condor/at_pokec_gated_*` + `run_one_pokec_gated.sh`
- `runs/pokec_gated_lm/` (LLM results, incl. the innate λ×ε×β batch)

## Contents

### `circle/`
Earlier parked line: circular Salop competition + circular (Kuramoto-like) FJ population.

### `fj/`  (archived 2026-06-16)
The deprecated Friedkin–Johnsen (FJ) project, superseded by the gated NDlib loop.
- `scripts/` — the `experiments/fj/` run scripts (beach, mw_mlps, mlp_hunters, hotelling, …)
- `make_scripts/` — FJ hunt/competition figure scripts (`make_hunt*`, `make_competition*`, `analyze_matrix.py`)
- `cluster_pipelines/` — `run_pokec_fj_{lm,competition,hunt}.py`, `fj_{beta_dissociation,schedules_local,baselines_local,mlp_homogenize}.py`, `grad_diagnostics.py`
- `condor/` — `at_pokec_fj_*.sub` (16) + their `configs_pokec_{fj,competition,hunt,mw,diverge}*.txt` + runners (`run_one_pokec_{fj,hunt,hunt_kl,competition}.sh`)

Note: the gated pipeline still supports FJ population via `POP_MODEL=fj` (uses
`perfsim.environments.dynamics.FJWorld`, which stays in the package); only the
standalone FJ experiment scripts are archived here.

## Deleted 2026-06-16 (removed from disk, recoverable from git history at old paths)
These were judged not worth keeping on disk. They are still in git history (all
were tracked), so `git log --all --full-history -- <path>` recovers them.
- **covid** ABM calibration line — `experiments/condor/at_covid_*.sub`, `at_calibration.sub`, their configs + runners (`run_{calibration,nomodel,one}.sh`); and the result files `experiments/runs/{alpha_*_history.pt, covid_abm_sweep.json, mlp_beta_sweep_covid.json}`. (The covid pipeline `run_covid_*.py` scripts were already gone before this.)
- **recommender** — `experiments/recommender/` (13 scripts), early recommender-market line.
- **ai_mediated** — `experiments/scripts/ai_mediated/` (12 scripts: resume-mediation harness dropped 2026-06-10 + bandit-market deprioritized) and `experiments/condor/{at_resume_llm_*.sub, configs_resume_*.txt, run_one_resume_llm.sh}`.

## Still in place
- `runs/_archive_fj_project/` — already-archived FJ LLM results (left as-is)
