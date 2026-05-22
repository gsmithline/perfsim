"""Performative optimization worked example.

Demonstrates `grad_run` used as the search direction for an outer
optimization loop over perfsim's deployed predictor params. AT's
substep-level learnables (R2, M) are frozen; we vary only the predictor.

Objective: minimize total daily_infected over a 5-step rollout. The
predictor maps age -> isolation score (per-agent). Higher isolation
reduces transmission, so the optimizer should push the per-agent
sigmoid output toward 1.

Run interactively: `marimo edit examples/notebooks/at_covid_perf_opt.py`
Run as script:    `python examples/notebooks/at_covid_perf_opt.py`
"""

import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def imports():
    import copy
    import time

    import torch

    from perfsim.scenarios.at_covid import (
        default_signal_writer_grad,
        make_covid_env,
        seed_initial_infections,
    )
    return (
        copy,
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
        # Performative optimization on the AT covid sim

        Gradient B in action. We use `env.grad_run(model, n_steps=K)` to roll
        out the AT sim under the deployed model, compute a downstream
        objective on AT state, and backprop to model params. AT's own
        learnables (R2, M) stay frozen.

        Objective: minimize the total daily_infected over a 5-step rollout.
        The model output (sigmoid'd in PerfsimIsolationDecision) becomes the
        per-agent isolation probability; higher isolation -> fewer
        transmissions -> lower loss. We expect the optimizer to push the
        sigmoid output toward 1 for most agents.

        This is the kind of loop you would use to do performative
        optimization (Mendler-Dunner et al.): find theta* that minimizes a
        downstream objective on the induced population.
        """
    )
    return (mo,)


@app.cell
def build_env(default_signal_writer_grad, make_covid_env, seed_initial_infections, torch):
    env = make_covid_env(
        init_seed=0,
        signal_writer=default_signal_writer_grad,
    )
    seed_initial_infections(env, fraction=0.05, seed=0)
    n_agents = env.runner.state["agents"]["citizens"]["age"].shape[0]
    print(f"n_agents = {n_agents}")
    print(f"seeded infections at iter 0: {(env.runner.state['agents']['citizens']['disease_stage'] == 2.0).sum().item()}")
    return env, n_agents


@app.cell
def _snap(copy, env, torch):
    def snapshot_state(state):
        out = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().clone()
            elif isinstance(v, dict):
                out[k] = snapshot_state(v)
            else:
                out[k] = copy.deepcopy(v)
        return out

    def restore_state(target, snap):
        for k, v in snap.items():
            if isinstance(v, dict):
                if k not in target or not isinstance(target[k], dict):
                    target[k] = {}
                restore_state(target[k], v)
            elif isinstance(v, torch.Tensor):
                target[k] = v.detach().clone()
            else:
                target[k] = copy.deepcopy(v)

    initial_snap = snapshot_state(env.runner.state)
    print("snapshotted initial state")
    return initial_snap, restore_state, snapshot_state


@app.cell
def build_model(torch):
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)
    print(f"init: w={model.weight.item():.4f}, b={model.bias.item():.4f}")
    print(f"init: sigmoid(0) -> isolation prob = 0.5 for all agents")
    return (model,)


@app.cell
def _md_loop(mo):
    mo.md(
        """
        ## Outer loop

        For each iteration:
        1. Restore the env to the seeded initial state.
        2. `env.grad_run(model, n_steps=5)` with autograd live.
        3. loss = total daily_infected.
        4. backward + Adam step on model params.

        Watch the bias drift positive (raising sigmoid output toward 1) and
        total infections fall.
        """
    )
    return


@app.cell
def perf_optimize(env, initial_snap, model, restore_state, time, torch):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    K = 5
    n_iters = 15

    history = []
    t0 = time.time()
    for it in range(n_iters):
        restore_state(env.runner.state, initial_snap)
        env.runner.reset_state_before_episode()

        optimizer.zero_grad()
        env.grad_run(model, n_steps=K)
        loss = env.runner.state["environment"]["daily_infected"].sum()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _avg_iso = torch.sigmoid(model(env.runner.state["agents"]["citizens"]["age"].float())).mean().item()

        history.append({
            "iter": it,
            "loss": float(loss.detach()),
            "w": model.weight.item(),
            "b": model.bias.item(),
            "avg_isolation_prob": _avg_iso,
        })

        if it % 3 == 0 or it == n_iters - 1:
            print(
                f"iter {it:2d}  loss={loss.item():.1f}  "
                f"w={model.weight.item():.4f}  b={model.bias.item():.4f}  "
                f"avg P(isolate)={_avg_iso:.3f}"
            )
    print(f"\ntotal time: {time.time() - t0:.1f}s")
    return K, history, n_iters, optimizer


@app.cell
def plot(history):
    import matplotlib.pyplot as plt

    _po_iters = [h["iter"] for h in history]
    _po_losses = [h["loss"] for h in history]
    _po_bs = [h["b"] for h in history]
    _po_ws = [h["w"] for h in history]
    _po_iso = [h["avg_isolation_prob"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(_po_iters, _po_losses)
    axes[0].set_xlabel("iter")
    axes[0].set_ylabel("daily_infected.sum()")
    axes[0].set_title("Objective")

    axes[1].plot(_po_iters, _po_ws, label="w")
    axes[1].plot(_po_iters, _po_bs, label="b")
    axes[1].set_xlabel("iter")
    axes[1].set_ylabel("param")
    axes[1].set_title("Predictor params")
    axes[1].legend()

    axes[2].plot(_po_iters, _po_iso)
    axes[2].set_ylim(0, 1)
    axes[2].axhline(0.5, color="gray", linestyle=":", label="init")
    axes[2].set_xlabel("iter")
    axes[2].set_ylabel("P(isolate)")
    axes[2].set_title("Avg isolation probability")
    axes[2].legend()

    fig.tight_layout()
    fig
    return axes, fig, plt


if __name__ == "__main__":
    app.run()
