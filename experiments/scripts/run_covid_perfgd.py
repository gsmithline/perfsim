"""PerfGD on AT covid: minimize total infections via backprop through the differentiable ABM."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from perfsim.learners.perfgd import PerfGDLearner, PerfGDFiniteDiffLearner
from perfsim.losses import MSELoss
from perfsim.models.linear import LinearModel
from perfsim.scenarios.at_covid import (
    default_signal_writer_grad,
    make_covid_env,
    seed_initial_infections,
)
from perfsim.simulator import Simulator

try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:
    _wandb = None
    _HAS_WANDB = False


def _env_or(name, default=None):
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var {name!r} not set")
    return val


def main() -> int:
    run_tag = _env_or("RUN_TAG", "covid_perfgd_test")
    method = _env_or("PERFGD_METHOD", "backprop")
    n_rounds = int(_env_or("N_ROUNDS", "20"))
    env_steps = int(_env_or("K_STEPS", "5"))
    lr = float(_env_or("LR", "0.01"))
    seed = int(_env_or("SEED", "0"))
    infection_frac = float(_env_or("INFECTION_FRAC", "0.05"))
    out_dir = Path(_env_or("OUT_DIR", f"runs/covid_perfgd/{run_tag}"))
    wandb_project = os.environ.get("WANDB_PROJECT", "")

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_tag": run_tag,
        "method": method,
        "n_rounds": n_rounds,
        "env_steps": env_steps,
        "lr": lr,
        "seed": seed,
        "infection_frac": infection_frac,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[perfgd] {json.dumps(config)}", flush=True)

    wandb = None
    if wandb_project and _HAS_WANDB:
        wandb = _wandb
        wandb.init(project=wandb_project, name=run_tag, config=config)

    model = LinearModel(in_features=1, out_features=1, bias=True)
    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        model.linear.bias.fill_(0.0)

    env = make_covid_env(
        init_seed=seed,
        signal_writer=default_signal_writer_grad,
        initial_infections_fraction=infection_frac,
    )
    n_agents = env.runner.state["agents"]["citizens"]["age"].shape[0]
    print(f"[perfgd] env ready: {n_agents} agents", flush=True)

    def population_loss(env_obj):
        return env_obj.runner.state["environment"]["daily_infected"].sum()

    loss = MSELoss()

    if method == "backprop":
        learner = PerfGDLearner(
            model=model,
            loss=loss,
            env=env,
            population_loss_fn=population_loss,
            lr=lr,
            env_steps=env_steps,
            optimizer="adam",
        )
    elif method == "fd":
        learner = PerfGDFiniteDiffLearner(
            model=model,
            loss=loss,
            env=env,
            population_loss_fn=population_loss,
            lr=lr,
            eps=float(_env_or("FD_EPS", "0.5")),
            env_steps=env_steps,
            n_seeds=int(_env_or("FD_SEEDS", "3")),
        )
    else:
        raise ValueError(f"unknown PERFGD_METHOD: {method!r}")

    def m_infected(sim):
        di = sim.env.runner.state["environment"].get("daily_infected")
        if di is None:
            return 0.0
        return float(di.sum().item())

    def m_params(sim):
        w = float(sim.predictor.model.linear.weight.item())
        b = float(sim.predictor.model.linear.bias.item())
        return {"weight": w, "bias": b}

    trajectory = []

    def _on_round(t, record):
        row = {
            "round": t,
            "total_infected": float(record["total_infected"]),
        }
        params = record.get("params", {})
        row.update(params)
        gap = record.get("stability_gap")
        if hasattr(gap, "item"):
            row["stability_gap"] = float(gap.item())
        trajectory.append(row)
        if wandb is not None:
            wandb.log(row)
        print(f"[round {t}] infected={row['total_infected']:.0f} w={row.get('weight', 0):.4f} b={row.get('bias', 0):.4f}", flush=True)

    sim = Simulator(env=env, learner=learner, loss=loss, metrics={
        "total_infected": m_infected,
        "params": m_params,
    })

    print(f"[perfgd] starting: {n_rounds} rounds, method={method}", flush=True)
    t0 = time.time()
    hist = sim.run(n_rounds=n_rounds, epoch_size=env_steps, seed=seed, on_round=_on_round)
    print(f"[perfgd] done in {time.time() - t0:.1f}s", flush=True)

    (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    torch.save([dict(r) for r in hist.records], out_dir / "history.pt")

    if wandb is not None:
        wandb.finish()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
