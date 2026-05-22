"""Measure gradient through a covid epoch via `AgentTorchEnvironment.grad_run`.

End-to-end demonstration that the deployed perfsim model's parameters
receive non-zero gradient from a loss computed on AT's evolved state.

Pipeline:
  1. Build the covid env.
  2. Seed ~5% of agents as infected so transmission has something to do.
  3. Call `env.grad_run(model, n_steps=K)` (NOT `env.run` — that uses
     `torch.no_grad`).
  4. Loss = sum of `daily_infected` across the rollout.
  5. Backprop. Report `|dL/dw|` and `|dL/db|`.

Why all four conditions matter:
  - Seeded infections: starting all-susceptible gives gradient = 0 because
    every Bernoulli draws on near-zero probability.
  - K > 1: cumulative loss over a multi-step rollout amplifies signal.
  - grad_run (not run): `run` wraps `model(X)` in `torch.no_grad`.
  - non-detaching signal_writer: `at_covid` default uses `.clone()` only
    (no `.detach()`), so the graph survives the write to runner.state.

Requires `pip install 'perfsim[agenttorch]'`. ~30s wall clock on CPU at
37,518 agents with K=5.
"""

from __future__ import annotations

import time

import torch

from perfsim.scenarios.at_covid import (
    default_signal_writer_grad,
    make_covid_env,
    seed_initial_infections,
)


def main():
    env = make_covid_env(
        init_seed=0,
        signal_writer=default_signal_writer_grad,
    )

    n_seeded = seed_initial_infections(env, fraction=0.05, seed=0)
    n_agents = env.runner.state["agents"]["citizens"]["age"].shape[0]
    print(f":: seeded {n_seeded} / {n_agents} agents as infected")

    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.05)
        model.bias.fill_(-1.0)

    print(":: grad_run with K=5 inner steps")
    t0 = time.time()
    final_data = env.grad_run(model, n_steps=5)
    print(f":: forward done in {time.time() - t0:.1f}s")

    daily_infected = env.runner.state["environment"]["daily_infected"]
    disease_stage = env.runner.state["agents"]["citizens"]["disease_stage"]

    print(f":: daily_infected.sum() = {daily_infected.sum().item():.1f}")
    print(f":: disease_stage.sum()  = {disease_stage.sum().item():.1f}")
    print(f":: fraction non-S       = {(disease_stage > 0).float().mean().item():.4f}")

    loss = daily_infected.sum()
    print(f":: loss = {loss.item():.4f}; requires_grad = {loss.requires_grad}")

    t0 = time.time()
    loss.backward()
    print(f":: backward done in {time.time() - t0:.1f}s")

    gw = model.weight.grad
    gb = model.bias.grad
    print(f":: |dL/dw| = {gw.norm().item():.4f}")
    print(f":: |dL/db| = {gb.norm().item():.4f}")
    if (gw.abs() > 1e-6).any():
        print(":: gradient is non-zero — usable for performative optimization")
    else:
        print(":: gradient is numerically zero — check seeding fraction and K")


if __name__ == "__main__":
    main()
