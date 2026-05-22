# perfsim HTCondor setup (covid + LM + KL-SFT sweep)

Mirrors the layout in `Opinion-dynamics-post-training/condor/`. One submit
file (`at_covid_lm.sub`), one wrapper shell (`run_one.sh`), one config
sweep file (`configs_betas.txt`), one python entry (`../scripts/run_covid_lm.py`).

## 0. Status

This pipeline has **never been run end-to-end**. The first cluster submit
is the smoke test. Expected things to debug on first run:

- `HFCausalLMModel` + `KLSFTLearner` are wired but untested end-to-end (the
  development box hits a tokenizers thread hang on macOS).
- 37,518 LM generations per round on Astoria is expensive. With Qwen-0.5B
  bf16 on A100 + batched generate at `GEN_BATCH_SIZE=64`, very rough
  estimate: 5-10 min of inference per round, plus 1-3 min of SFT per round.
  ~5 rounds × 5 betas → 4-8 hours of A100 time.

Start with `N_ROUNDS=2 K_STEPS=2` and one beta as a sanity job before
the full sweep.

## 1. Install on cluster

```bash
# Login node
cd ~
git clone <REPO_URL> perfsim
cd perfsim

# Create env (any python>=3.10)
conda create -n perfsim python=3.11 -y
conda activate perfsim

# Core
pip install -e .

# CUDA torch (replace cu121 with the right CUDA your cluster has)
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121

# Optional extras
pip install -e ".[lm,agenttorch]"

# Sanity
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
python -c "from perfsim.scenarios.at_covid import make_covid_env; print('at_covid OK')"
python -c "from perfsim.models.hf_causal_lm import HFCausalLMModel; print('LM stack imports')"
```

## 2. Wandb (optional)

```bash
echo "<YOUR_WANDB_API_KEY>" > ~/.wandb_key
chmod 600 ~/.wandb_key
```

`run_one.sh` reads this file and exports `WANDB_API_KEY`.

## 3. Edit machine-specific paths

In `condor/run_one.sh` and `condor/at_covid_lm.sub`, replace the four
defaults:

```
REPO=/home/gsmithline/perfsim                                # your clone
CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh  # your conda
ENV_NAME=perfsim                                              # your env name
WANDB_KEY_FILE=/home/gsmithline/.wandb_key                    # your key path
```

Also fix the `notify_user` email if you want completion emails.

The `condor/logs/` directory is tracked via `.gitkeep`. If you ever delete
it or use a fresh clone where the directory is missing, jobs will land in
held immediately with "Failed to open ... standard output: No such file or
directory". Fix: `mkdir -p condor/logs && condor_release <jobid>`.

## 4. Sanity submit (one beta, short)

Before the full sweep, run a single config to make sure the LM stack works
on your cluster.

Create `condor/configs_sanity.txt`:

```
at_covid_sanity     sft_kl     1.0
```

Submit:

```bash
N_ROUNDS=2 K_STEPS=2 condor_submit_bid <BID> \
    condor/at_covid_lm.sub -append "queue tag, style, beta from condor/configs_sanity.txt"
```

Or temporarily change the `queue` line in `at_covid_lm.sub` to point at
`configs_sanity.txt`.

Wall time target: ~10-15 minutes on A100. If it succeeds and
`runs/at_covid_lm/at_covid_sanity/trajectory.json` exists with two rounds,
you can submit the full sweep.

## 5. Full sweep

```bash
mkdir -p condor/logs
condor_submit_bid <BID> condor/at_covid_lm.sub
```

This queues 5 jobs (one per row in `configs_betas.txt`). Each job calls
`run_one.sh tag style beta`. Outputs land in:

- `condor/logs/<tag>.{out,err,log}` for the condor logs
- `runs/at_covid_lm/<tag>/{trajectory.json,history.pt,config.json}` for the
  run outputs
- wandb run `perfsim-at-covid-lm/<tag>` if wandb is configured

## 6. Knobs (env vars, all overridable)

Set on the `condor_submit_bid` command line via `-append "environment = ..."`,
or hard-code in `at_covid_lm.sub`'s `environment` string.

| Var | Default | Notes |
|---|---|---|
| `N_ROUNDS` | 5 | outer rounds (train → deploy → rollout cycles) |
| `K_STEPS` | 3 | AT substeps per round |
| `SEED_FRAC` | 0.05 | initial infected fraction (>0 needed for non-zero gradient if you ever switch to grad_run) |
| `SEED` | 0 | torch seed for reproducibility |
| `BASE_MODEL` | Qwen/Qwen2.5-0.5B-Instruct | HF model id |
| `SFT_MAX_STEPS` | 50 | SFTTrainer steps per round |
| `GEN_BATCH_SIZE` | 64 | LM generation batch size |
| `MAX_NEW_TOKENS` | 8 | LM gen budget per agent |
| `WANDB_PROJECT` | perfsim-at-covid-lm | "" to disable wandb |

## 7. After-run analysis

Each `runs/at_covid_lm/<tag>/trajectory.json` has per-round records:

```
[
  {"round": 0, "theta_norm": null, "daily_infected_sum": 2253.4,
   "fraction_non_S": 0.084, "stability_gap": null},
  {"round": 1, "theta_norm": 12.31, "daily_infected_sum": 1980.7, ...}
]
```

For comparing betas, plot `daily_infected_sum` and `theta_norm` per round
across tags. No notebook for this yet; add one under `examples/` once you
have real data.
