"""Measure gradient through a covid epoch via `AgentTorchEnvironment.grad_run`.

End-to-end demonstration that the deployed perfsim model's parameters
receive non-zero gradient from a loss computed on AT's evolved state.

Requires `pip install 'perfsim[agenttorch]'`. ~10s wall clock on CPU at
37,518 agents with K=5.
"""

import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def imports():
    import time

    import torch

    from perfsim.scenarios.at_covid import (
        default_signal_writer_grad,
        make_covid_env,
        seed_initial_infections,
    )
    return (
        default_signal_writer_grad,
        make_covid_env,
        seed_initial_infections,
        time,
        torch,
    )


@app.cell
def _intro():
    import marimo as mo
    mo.md(
        """
        # Gradient measurement through AT covid

        Pipeline:
        1. Build covid env with a grad-preserving signal writer.
        2. Seed ~5% of agents as infected so transmission has signal.
        3. `env.grad_run(model, n_steps=5)`. No `torch.no_grad` and no detach.
        4. Loss = `daily_infected.sum()` across the rollout.
        5. backward() -- gradient lands on `model.weight` and `model.bias`.

        Four conditions all matter (any one missing gives gradient = 0):
        seeded infections, K > 1, `grad_run` not `run`, non-detaching writer.
        """
    )
    return (mo,)


@app.cell
def build_env(default_signal_writer_grad, make_covid_env, seed_initial_infections):
    env = make_covid_env(
        init_seed=0,
        signal_writer=default_signal_writer_grad,
    )
    n_seeded = seed_initial_infections(env, fraction=0.05, seed=0)
    n_agents = env.runner.state["agents"]["citizens"]["age"].shape[0]
    print(f":: seeded {n_seeded} / {n_agents} agents as infected")
    return env, n_agents, n_seeded


@app.cell
def build_model(torch):
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.05)
        model.bias.fill_(-1.0)
    return (model,)


@app.cell
def rollout(env, model, time):
    print(":: grad_run with K=5 inner steps")
    _t0 = time.time()
    final_data = env.grad_run(model, n_steps=5)
    print(f":: forward done in {time.time() - _t0:.1f}s")
    return (final_data,)


@app.cell
def inspect_state(env):
    daily_infected = env.runner.state["environment"]["daily_infected"]
    disease_stage = env.runner.state["agents"]["citizens"]["disease_stage"]
    print(f":: daily_infected.sum() = {daily_infected.sum().item():.1f}")
    print(f":: disease_stage.sum()  = {disease_stage.sum().item():.1f}")
    print(f":: fraction non-S       = {(disease_stage > 0).float().mean().item():.4f}")
    return daily_infected, disease_stage


@app.cell
def backward(daily_infected, model, time):
    loss = daily_infected.sum()
    print(f":: loss = {loss.item():.4f}; requires_grad = {loss.requires_grad}")

    _t0 = time.time()
    loss.backward()
    print(f":: backward done in {time.time() - _t0:.1f}s")
    return (loss,)


@app.cell
def report(model):
    _gw = model.weight.grad
    _gb = model.bias.grad
    print(f":: |dL/dw| = {_gw.norm().item():.4f}")
    print(f":: |dL/db| = {_gb.norm().item():.4f}")
    if (_gw.abs() > 1e-6).any():
        print(":: gradient is non-zero -- usable for performative optimization")
    else:
        print(":: gradient is numerically zero -- check seeding fraction and K")
    return


if __name__ == "__main__":
    app.run()
