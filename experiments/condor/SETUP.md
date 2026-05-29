# perfsim HTCondor setup

## 1. Install on cluster

```bash
cd ~
git clone <REPO_URL> perfsim
cd perfsim

conda create -n opdyn python=3.11 -y
conda activate opdyn

pip install -e .
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[lm,agenttorch]"

# Sanity
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
python -c "from perfsim.scenarios.at_covid import make_covid_env; print('at_covid OK')"
python -c "from perfsim.scenarios.at_macro import make_macro_env; print('at_macro OK')"
```

## 2. W&B

```bash
echo "<YOUR_WANDB_API_KEY>" > ~/.wandb_key
chmod 600 ~/.wandb_key
```

## 3. Machine-specific paths

Edit these defaults in `condor/run_one.sh`, `condor/run_calibration.sh`,
`condor/run_calibration_macro.sh`:

```
REPO=/home/gsmithline/perfsim
CONDA_SH=/home/gsmithline/miniconda3/etc/profile.d/conda.sh
ENV_NAME=opdyn
WANDB_KEY_FILE=/home/gsmithline/.wandb_key
```

---

## 4. COVID experiments

### 4.1 Calibrated ABM parameters

Calibrated to real Astoria weekly case data (`county_data.csv` bundled
in agent_torch). Population: 37,518 agents from Astoria, Queens.

| Season | Start week | Dates | Real cases (3wk) | Seed frac | Calibrated R2 | Ratio |
|--------|-----------|-------|-------------------|-----------|---------------|-------|
| Alpha | 17 | Dec 2020 - Jan 2021 | 353 | 0.005 | 0.60 | 1.000 |
| Delta | 52 | Aug - Sep 2021 | 184 | 0.001 | 1.13 | 0.995 |
| Omicron | 71 | Dec 2021 - Jan 2022 | 3,317 | 0.05 | 1.35 | 1.001 |

R2 = transmission rate in AT's `NewTransmission` substep.
Seed frac = fraction of agents initially set to INFECTED (disease_stage=2).
Calibration script: `scripts/calibrate_covid_single.py`.
W&B project: `perfsim-calibration`.

### 4.2 COVID calibration jobs

Submit file: `condor/at_calibration.sub`
Runner: `condor/run_calibration.sh`
Configs: `condor/configs_calibration.txt`
Python: `scripts/calibrate_covid_single.py`

```bash
mkdir -p condor/logs
condor_submit_bid <BID> condor/at_calibration.sub
```

After jobs finish, pick best per season:
```bash
python scripts/calibrate_covid.py pick-best
```

### 4.3 COVID performative loop (LM + KL-SFT beta sweep)

Submit file: `condor/at_covid_calibrated.sub`
Runner: `condor/run_one.sh`
Configs: `condor/configs_alpha_sweep.txt` (Alpha only) or
         `condor/configs_calibrated_sweep.txt` (all 3 seasons)
Python: `scripts/run_covid_lm.py`

Current experiment parameters:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | 0.5B does not produce usable results |
| `N_ROUNDS` | 10 | outer performative loop rounds |
| `K_STEPS` | 21 | AT timesteps per round (3 weeks, matches calibration) |
| `SEED_FRAC` | per-season | 0.005 (Alpha), 0.001 (Delta), 0.05 (Omicron) |
| `CALIBRATED_R2` | per-season | 0.60 (Alpha), 1.13 (Delta), 1.35 (Omicron) |
| `SFT_MAX_STEPS` | 50 | TRL SFTTrainer steps per round |
| `SFT_LR` | 1e-5 | SFT learning rate |
| `LORA_R` | 32 | LoRA rank (r=8 causes collapse on Qwen) |
| `USE_LORA` | 1 | 1=LoRA, 0=full fine-tuning |
| `GEN_BATCH_SIZE` | 64 | LM generation batch size |
| `MAX_NEW_TOKENS` | 8 | LM generation budget per agent |
| `TARGET_KIND` | `exposed_binary` | SFT target: (disease_stage > 0) |

Training styles:
- `sft` (beta=0): pure SFT, no KL anchor
- `sft_kl` (beta>0): SFT + KL(pi_theta || pi_ref), loads frozen reference model
- `frozen`: no training, pretrained model deployed as-is (reference baseline)

Resource requirements:
- `sft` and `frozen`: ~32G RAM, 1 GPU (one 7B model)
- `sft_kl`: ~96G RAM, 1 GPU (two 7B models: fine-tuned + frozen reference)

```bash
mkdir -p condor/logs
condor_submit_bid <BID> condor/at_covid_calibrated.sub
```

W&B project: `perfsim-covid-calibrated`

### 4.4 Per-round metrics tracked

Epidemic state:
- `daily_infected_sum` — total new infections this round
- `fraction_non_S` — fraction of population no longer susceptible
- `n_susceptible`, `n_exposed`, `n_infected`, `n_recovered`, `n_dead`

Model behavior:
- `pred_mean`, `pred_std`, `pred_min`, `pred_max` — LM isolation recommendations
- `pred_age0` through `pred_age5` — per-age-group recommendations
- `train_loss` — loss on data the model's own policy produced (performative risk proxy)
- `theta_norm`, `stability_gap` — parameter trajectory

Subgroup effects:
- `burden_age0` through `burden_age5` — per-age-group disease burden

### 4.5 COVID gradient diagnostics

Validated AT autodiff through SEIRM epidemic sim:
- Autograd/finite-difference ratio: 0.98-1.10
- Gradient variance (CV): 0.02 across 10 seeds
- Script: `scripts/grad_diagnostics.py` (runs locally, ~8 min)

### 4.6 Outputs

Each run produces:
```
runs/at_covid_lm/<tag>/
  config.json              — resolved config
  history.pt               — torch-pickled per-round records
  trajectory.json          — per-round summary (logged to W&B)
  recommendations.json     — final per-profile LM recommendations
  diagnostic_pre_sft.json  — 20 sample LM outputs before training
  diagnostic_post_sft.json — 20 sample LM outputs after training
```

---

## 5. Macro economics experiments

### 5.1 ABM status

Macro substeps patched and functional:
- `_PatchedMacroRates` — fixes shape mismatch in bundled `UpdateMacroRates`
  (unemployment_adaptation_coefficient: YAML shape [1] vs code expects [num_timesteps, 3])
- `_PatchedFinancialMarket` — re-implements Taylor-rule interest rate,
  price adjustment, inflation calculation

Macro indicators that evolve: unemployment rate, hourly wages, price of
goods, inflation rate, labor force participation.

### 5.2 Macro calibration

Target: Queens County monthly unemployment rates, 2019-2023 (bundled in
`agent_torch/models/macro_economics/data/unemployment_rate_csvs/Queens-Table.csv`).

| Period | Start month | Dates | Unemployment |
|--------|------------|-------|-------------|
| Pre-COVID baseline | 0 | Jan-Sep 2019 | 2.9-4.2% |
| COVID shock | 15 | Apr-Dec 2020 | 23.2→11.5% |
| Recovery | 24 | Jan-Sep 2021 | 12.5→8.3% |
| Post-recovery | 36 | Jan-Sep 2022 | 7.4→4.1% |

Parameter being calibrated: `unemployment_adaptation_coefficient` (UAC),
a (num_timesteps, 3) regression coefficient matrix.

Full population: 2,712,360 agents (full NYC census data, bundled in
agent_torch). Uses `config.yaml` (not `config_100_agents.yaml`).

Submit file: `condor/at_macro_calibration.sub`
Runner: `condor/run_calibration_macro.sh`
Configs: `condor/configs_macro_calibration.txt`
Python: `scripts/calibrate_macro_single.py`

```bash
mkdir -p condor/logs
condor_submit_bid <BID> condor/at_macro_calibration.sub
```

W&B project: `perfsim-macro-calibration-full`

---

## 6. Config files (Condor queue-from)

Condor's `queue from` does NOT support `#` comments. All config files
must contain only data rows.

| File | Format | Used by |
|------|--------|---------|
| `configs_alpha_sweep.txt` | tag, style, beta, frac, r2 | `at_covid_calibrated.sub` |
| `configs_calibrated_sweep.txt` | tag, style, beta, frac, r2 | `at_covid_calibrated.sub` (all seasons) |
| `configs_calibration.txt` | tag, frac, week | `at_calibration.sub` |
| `configs_macro_calibration.txt` | tag, start, lr | `at_macro_calibration.sub` |
| `configs_betas.txt` | tag, style, beta | `at_covid_lm.sub` (legacy, uncalibrated) |
| `configs_baseline.txt` | tag, style, beta | `at_covid_lm.sub` (legacy) |

---

## 7. Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_covid_lm.py` | Performative loop: LM + AT covid + KL-SFT |
| `scripts/calibrate_covid_single.py` | Calibrate R2 for one (seed_frac, season) on the cluster |
| `scripts/calibrate_covid.py` | Local calibration: `r2` / `surge` / `pick-best` subcommands |
| `scripts/grad_diagnostics.py` | Validate AT autodiff (sign, FD, variance) |
| `scripts/run_macro_lm.py` | Performative loop: LM + AT macro + KL-SFT |
