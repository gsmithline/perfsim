"""No-model baseline: run the COVID ABM with keep_trajectory=True and NO LM.

Platform signal stays at zero (sigmoid(0) = 0.5 isolation for all agents).
Records the same metrics as run_covid_lm.py so results are directly comparable.

Environment variables:
  RUN_TAG              required.
  N_ROUNDS             outer rounds. Default 10.
  K_STEPS              inner AT substeps per round. Default 21.
  SEED_FRAC            initial infected fraction. Default 0.05.
  CALIBRATED_R2        transmission rate. If set, overrides AT default.
  SEED                 random seed. Default 0.
  OUT_DIR              output dir. Default runs/at_covid_nomodel/$RUN_TAG.
  WANDB_PROJECT        optional.
  ISOLATION_LEVEL      uniform isolation level [0,1]. Default 0.0 (no isolation).
                       Note: platform_signal is in logit space, sigmoid applied by ABM.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _env_or(name, default=None):
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var {name!r} not set")
    return val


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def main() -> int:
    import torch

    run_tag = _env_or("RUN_TAG")
    n_rounds = _env_int("N_ROUNDS", 10)
    k_steps = _env_int("K_STEPS", 21)
    seed_frac = _env_float("SEED_FRAC", 0.05)
    seed = _env_int("SEED", 0)
    calibrated_r2 = os.environ.get("CALIBRATED_R2")
    isolation_level = _env_float("ISOLATION_LEVEL", 0.0)
    out_dir = Path(os.environ.get("OUT_DIR", f"runs/at_covid_nomodel/{run_tag}"))
    wandb_project = os.environ.get("WANDB_PROJECT")

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_tag": run_tag,
        "n_rounds": n_rounds,
        "k_steps": k_steps,
        "seed_frac": seed_frac,
        "seed": seed,
        "calibrated_r2": calibrated_r2,
        "isolation_level": isolation_level,
        "host": os.uname().nodename,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[run] {json.dumps(config)}", flush=True)

    wandb = None
    if wandb_project:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=wandb_project, name=run_tag, config=config)

    from perfsim.scenarios.at_covid import make_covid_env
    from perfsim.scenarios.at_covid.env import build_covid_runner, seed_initial_infections

    torch.manual_seed(seed)

    # Set platform_signal to a fixed logit value so sigmoid gives the desired isolation
    p = max(0.01, min(0.99, isolation_level))
    logit_val = torch.log(torch.tensor(p / (1.0 - p)))

    def fixed_signal_writer(runner, preds):
        n = runner.state["agents"]["citizens"]["age"].shape[0]
        runner.state["agents"]["citizens"]["platform_signal"] = torch.full((n,), logit_val.item())

    env = make_covid_env(
        init_seed=seed,
        initial_infections_fraction=seed_frac,
        signal_writer=fixed_signal_writer,
        keep_trajectory=True,
    )

    if calibrated_r2:
        r2_val = float(calibrated_r2)
        transmission = env.runner.initializer.transition_function["0"].new_transmission
        with torch.no_grad():
            transmission.calibrate_R2.fill_(r2_val)
        print(f"[run] set R2 = {r2_val}", flush=True)

    n_agents = env.runner.state["agents"]["citizens"]["age"].shape[0]
    print(f"[run] {n_agents} agents, isolation={isolation_level}, seed_frac={seed_frac}", flush=True)

    history = []
    for rnd in range(n_rounds):
        t0 = time.time()

        # Write the fixed signal before each round
        fixed_signal_writer(env.runner, None)

        # Run ABM inner loop
        env.runner.step(num_steps=k_steps)

        # Collect metrics
        citizens = env.runner.state["agents"]["citizens"]
        ds = citizens["disease_stage"].squeeze()
        age = citizens["age"].squeeze().long()
        di = env.runner.state["environment"]["daily_infected"]

        stage_counts = {
            "n_susceptible": int((ds == 0).sum().item()),
            "n_exposed": int((ds == 1).sum().item()),
            "n_infected": int((ds == 2).sum().item()),
            "n_recovered": int((ds == 3).sum().item()),
            "n_dead": int((ds == 4).sum().item()),
        }

        sick = (ds >= 1).float()
        subgroup_burden = {}
        for bucket in range(6):
            mask = age == bucket
            if mask.sum() > 0:
                subgroup_burden[f"burden_age{bucket}"] = float(sick[mask].mean().item())

        row = {
            "round": rnd,
            "daily_infected_sum": float(di.sum().item()),
            "fraction_non_S": float((ds > 0).float().mean().item()),
            "stage_counts": stage_counts,
            "subgroup_burden": subgroup_burden,
            "isolation_level": isolation_level,
            "elapsed_s": time.time() - t0,
        }
        history.append(row)

        di_val = row["daily_infected_sum"]
        frac = row["fraction_non_S"]
        print(f"[round {rnd}] di={di_val:.0f} frac_nonS={frac:.4f} "
              f"S={stage_counts['n_susceptible']} I={stage_counts['n_infected']} "
              f"R={stage_counts['n_recovered']} D={stage_counts['n_dead']}",
              flush=True)

        if wandb:
            flat = {"round": rnd, "daily_infected_sum": di_val, "fraction_non_S": frac}
            flat.update(stage_counts)
            flat.update(subgroup_burden)
            wandb.log(flat, step=rnd)

    torch.save(history, out_dir / "history.pt")
    (out_dir / "trajectory.json").write_text(json.dumps(history, indent=2, default=str))
    print(f"[run] outputs in {out_dir}", flush=True)

    if wandb:
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
