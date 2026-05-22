"""Joint training worked example.

Demonstrates simultaneously updating an AT-side learnable parameter (R2,
the transmission rate inside NewTransmission) AND a perfsim-side
predictor (the isolation-policy Linear model). Two optimizers, two
parameter groups, one shared backward pass through `env.grad_run`.

Objective: a weighted combination of
  1. Matching a target daily_infected trajectory (drives R2 toward truth).
  2. Penalizing total infections (drives the predictor toward high
     isolation).

R2 and the predictor pull in different directions. R2 wants to reproduce
the higher-infection target trajectory; the predictor wants to suppress
infections via aggressive isolation. The two optimizers fight, and we
plot both trajectories.

Run interactively: `marimo edit examples/notebooks/at_covid_joint.py`
Run as script:    `python examples/notebooks/at_covid_joint.py`
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
        build_covid_runner,
        default_signal_writer_grad,
        make_covid_env,
        seed_initial_infections,
    )
    return (
        build_covid_runner,
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
        # Joint training: AT's R2 + perfsim's predictor

        Two parameter groups, two optimizers, one shared `env.grad_run`. AT's
        learnable transmission rate (R2) and our isolation-policy predictor
        are optimized simultaneously against a combined objective:

            loss = mse(daily_infected, y_target) + lambda * total_infected

        The MSE term wants to reproduce a target trajectory generated at
        R2=4.75 with default policy; this pulls R2 back toward 4.75. The
        total-infected term wants to minimize infections; this pulls the
        predictor toward high isolation. The two pressures fight each other:
        if the predictor isolates aggressively, infections drop, MSE rises,
        so R2 has to rise even further to match the (higher) target. The
        equilibrium depends on lambda.
        """
    )
    return (mo,)


@app.cell
def target_traj(build_covid_runner, seed_initial_infections, torch):
    # Target rollout: default R2=4.75, model=0 -> isolation_prob=0.5.
    target_runner = build_covid_runner(seed=0)
    seed_initial_infections(target_runner, fraction=0.05, seed=0)
    target_runner.state["agents"]["citizens"]["platform_signal"] = torch.zeros(
        target_runner.state["agents"]["citizens"]["age"].shape[0]
    )
    K = 5
    with torch.no_grad():
        target_runner.step(num_steps=K)
        y_target = target_runner.state["environment"]["daily_infected"].detach().clone()
    print(f"target daily_infected.sum() = {y_target.sum().item():.0f}")
    return K, y_target


@app.cell
def build_env(default_signal_writer_grad, make_covid_env, seed_initial_infections, torch):
    env = make_covid_env(
        init_seed=0,
        signal_writer=default_signal_writer_grad,
    )
    seed_initial_infections(env, fraction=0.05, seed=0)
    return (env,)


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
    return initial_snap, restore_state, snapshot_state


@app.cell
def setup_params(env, torch):
    # AT-side: calibrate_R2 (perturbed from default 4.75).
    transmission_substep = env.runner.initializer.transition_function["0"].new_transmission
    r2_param = transmission_substep.calibrate_R2
    with torch.no_grad():
        r2_param.fill_(2.5)

    # perfsim-side: Linear predictor.
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)

    print(f"R2 init: {r2_param.detach().flatten()[:3].tolist()}")
    print(f"model init: w={model.weight.item():.4f} b={model.bias.item():.4f}")
    return model, r2_param, transmission_substep


@app.cell
def _md_loop(mo):
    mo.md(
        """
        ## Joint outer loop

        Each iteration:
        1. Restore state, reset trajectory log.
        2. `env.grad_run(model, n_steps=K)` -- one shared graph through both
           param groups.
        3. loss = mse(daily_infected, y_target) + lambda * total_infected.
        4. backward() -- populates `.grad` on both `r2_param` and `model.parameters()`.
        5. opt_r2.step()  and  opt_model.step().

        The two optimizers each see only their own params (set up by the
        parameter group list), but they read gradient from the same shared
        backward call.
        """
    )
    return


@app.cell
def joint_train(K, env, initial_snap, model, r2_param, restore_state, time, torch, y_target):
    LAMBDA = 0.05
    opt_r2 = torch.optim.Adam([r2_param], lr=0.3)
    opt_model = torch.optim.Adam(model.parameters(), lr=0.2)
    n_iters = 20

    history = []
    t0 = time.time()
    for it in range(n_iters):
        restore_state(env.runner.state, initial_snap)
        env.runner.reset_state_before_episode()

        opt_r2.zero_grad()
        opt_model.zero_grad()

        env.grad_run(model, n_steps=K)
        pred = env.runner.state["environment"]["daily_infected"]
        mse = ((pred - y_target) ** 2).mean()
        total_inf = pred.sum()
        loss = mse + LAMBDA * total_inf

        loss.backward()

        opt_r2.step()
        opt_model.step()

        with torch.no_grad():
            r2_param.clamp_(min=0.1)
            _avg_iso = torch.sigmoid(
                model(env.runner.state["agents"]["citizens"]["age"].float())
            ).mean().item()

        history.append({
            "iter": it,
            "loss": float(loss.detach()),
            "mse": float(mse.detach()),
            "total_inf": float(total_inf.detach()),
            "R2": r2_param.detach().flatten()[0].item(),
            "w": model.weight.item(),
            "b": model.bias.item(),
            "avg_iso": _avg_iso,
        })

        if it % 4 == 0 or it == n_iters - 1:
            print(
                f"iter {it:2d}  loss={loss.item():.1f}  mse={mse.item():.1f}  "
                f"inf={total_inf.item():.0f}  R2[0]={history[-1]['R2']:.3f}  "
                f"P(iso)={_avg_iso:.3f}"
            )
    print(f"\ntotal time: {time.time() - t0:.1f}s")
    return LAMBDA, history, n_iters, opt_model, opt_r2


@app.cell
def plot(history):
    import matplotlib.pyplot as plt

    _iters = [h["iter"] for h in history]
    _loss = [h["loss"] for h in history]
    _mse = [h["mse"] for h in history]
    _inf = [h["total_inf"] for h in history]
    _r2 = [h["R2"] for h in history]
    _iso = [h["avg_iso"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(_iters, _loss, label="total")
    axes[0, 0].plot(_iters, _mse, label="MSE term")
    axes[0, 0].plot(_iters, _inf, label="total_inf")
    axes[0, 0].set_xlabel("iter")
    axes[0, 0].set_title("Loss decomposition")
    axes[0, 0].legend()

    axes[0, 1].plot(_iters, _r2)
    axes[0, 1].axhline(4.75, color="red", linestyle="--", label="target R2=4.75")
    axes[0, 1].set_xlabel("iter")
    axes[0, 1].set_ylabel("R2[0]")
    axes[0, 1].set_title("AT param: transmission rate")
    axes[0, 1].legend()

    axes[1, 0].plot(_iters, _iso)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel("iter")
    axes[1, 0].set_ylabel("P(isolate)")
    axes[1, 0].set_title("Predictor: avg isolation prob")

    axes[1, 1].plot(_iters, _inf, label="achieved")
    axes[1, 1].axhline(sum([h["total_inf"] for h in history[:1]]), color="gray",
                      linestyle=":", label="init")
    axes[1, 1].set_xlabel("iter")
    axes[1, 1].set_ylabel("total infections")
    axes[1, 1].set_title("Total infections per rollout")
    axes[1, 1].legend()

    fig.tight_layout()
    fig
    return axes, fig, plt


if __name__ == "__main__":
    app.run()
