"""AT calibration worked example.

Demonstrates AgentTorch's gradient model in isolation. Fits a substep-level
learnable parameter (R2, the transmission rate in NewTransmission) against
a synthetic target trajectory.

Run interactively: `marimo edit examples/notebooks/at_covid_calibrate.py`
Run as script:    `python examples/notebooks/at_covid_calibrate.py`
"""

import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def imports():
    import time

    import torch

    from perfsim.scenarios.at_covid import build_covid_runner, seed_initial_infections
    return build_covid_runner, seed_initial_infections, time, torch


@app.cell
def _intro():
    import marimo as mo
    mo.md(
        """
        # AT calibration: recover R2 by gradient descent

        AgentTorch's training pattern is one backward per episode. An episode is
        K timesteps x num_substeps substeps, all chained into one computation
        graph. The optimizer steps substep-level learnable parameters.

        Here we:

        1. Run the bundled covid sim with the default R2=4.75 to produce a
           synthetic target trajectory of daily_infected.
        2. Build a fresh runner with R2 perturbed to 3.0.
        3. Train R2 by minimizing MSE between the rollout's daily_infected and
           the target.
        4. Watch R2 climb back toward 4.75.

        perfsim's predictor is NOT involved. This isolates AT's own gradient mechanism.
        """
    )
    return (mo,)


@app.cell
def find_r2(build_covid_runner):
    # Probe the runner to find R2.  AT exposes substep-level learnable params
    # via runner.named_parameters() (Runner is an nn.Module).  We surface a few
    # so the user can see what's optimizable.
    probe_runner = build_covid_runner(seed=0)
    _params_seen = []
    for _pname, _p in probe_runner.named_parameters():
        _params_seen.append((_pname, tuple(_p.shape), _p.detach().flatten()[:1].tolist()))
    params_seen = _params_seen
    params_seen[:8]
    return (params_seen, probe_runner)


@app.cell
def run_target(build_covid_runner, seed_initial_infections, torch):
    # Roll out with default R2 and capture daily_infected as the target.
    target_runner = build_covid_runner(seed=0)
    seed_initial_infections(target_runner, fraction=0.05, seed=0)
    K = 5

    target_runner.state["agents"]["citizens"]["platform_signal"] = torch.zeros(
        target_runner.state["agents"]["citizens"]["age"].shape[0]
    )

    with torch.no_grad():
        target_runner.step(num_steps=K)
        y_target = target_runner.state["environment"]["daily_infected"].detach().clone()
    print(f"target daily_infected: total cases = {y_target.sum().item():.0f}")
    return K, y_target


@app.cell
def _snapshot(calib_runner, torch):
    # Snapshot the runner's state right after init+seeding so we can restore
    # it before each calibration iteration without paying for a full init().
    # Tensors only; non-tensor leaves are deep-copied via copy.deepcopy.
    import copy

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
        # Walk both dicts and copy snap into target *in-place* so any
        # external references to target.state remain valid. For tensors
        # we use copy_ so the dtype/device match.
        for k, v in snap.items():
            if isinstance(v, dict):
                if k not in target or not isinstance(target[k], dict):
                    target[k] = {}
                restore_state(target[k], v)
            elif isinstance(v, torch.Tensor):
                target[k] = v.detach().clone()
            else:
                target[k] = copy.deepcopy(v)

    initial_snap = snapshot_state(calib_runner.state)
    print("snapshotted initial state for restore between iterations")
    return initial_snap, restore_state, snapshot_state


@app.cell
def build_to_calibrate(build_covid_runner, seed_initial_infections, torch):
    # Fresh runner.  We'll modify R2 to a perturbed value before optimizing.
    calib_runner = build_covid_runner(seed=0)
    seed_initial_infections(calib_runner, fraction=0.05, seed=0)
    calib_runner.state["agents"]["citizens"]["platform_signal"] = torch.zeros(
        calib_runner.state["agents"]["citizens"]["age"].shape[0]
    )

    # AT subtlety: when `simulation_metadata.calibration: true` (covid's
    # default), substep.forward reads `self.calibrate_R2` (a plain leaf
    # tensor set via setattr), NOT `self.learnable_args["R2"]` (the
    # nn.ParameterDict entry that named_parameters() surfaces). Optimizing
    # learnable_args.R2 would do nothing. We target calibrate_R2 directly.
    transmission_substep = calib_runner.initializer.transition_function["0"].new_transmission
    r2_param = transmission_substep.calibrate_R2
    r2_name = "initializer.transition_function.0.new_transmission.calibrate_R2"
    default_value = r2_param.detach().clone()

    # Perturb R2 to a wrong value to start.
    with torch.no_grad():
        r2_param.fill_(2.5)
    print(f"R2 located at: {r2_name}")
    print(f"R2 shape: {tuple(r2_param.shape)}")
    print(f"R2 init (perturbed): {r2_param.detach().flatten()[:3].tolist()}")
    print(f"R2 target (default): {default_value.flatten()[:3].tolist()}")
    return calib_runner, default_value, r2_name, r2_param, transmission_substep


@app.cell
def _md_loop(mo):
    mo.md(
        """
        ## Calibration loop

        Each iteration:
        1. Reset the runner state (without re-creating the Runner -- so R2
           keeps its current value).
        2. Re-seed the same initial infections (deterministic).
        3. Rollout K=5 steps with autograd live.
        4. Compute MSE against y_target.
        5. backward + optimizer.step on R2.

        Watch R2 climb toward the target value.
        """
    )
    return


@app.cell
def calibrate(K, calib_runner, initial_snap, r2_param, restore_state, time, torch, y_target):
    optimizer = torch.optim.Adam([r2_param], lr=0.3)
    n_iters = 20

    history = []
    t0 = time.time()
    for it in range(n_iters):
        # Restore initial state in place. R2 is held in the substep module,
        # not in state, so its current value persists across the restore.
        restore_state(calib_runner.state, initial_snap)
        calib_runner.reset_state_before_episode()

        optimizer.zero_grad()
        calib_runner.step(num_steps=K)
        pred = calib_runner.state["environment"]["daily_infected"]
        loss = ((pred - y_target) ** 2).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            r2_param.clamp_(min=0.1)

        history.append({
            "iter": it,
            "loss": float(loss.detach()),
            "R2": r2_param.detach().flatten()[0].item(),
        })

        if it % 4 == 0 or it == n_iters - 1:
            print(f"iter {it:2d}  loss={loss.item():.3f}  R2[0]={history[-1]['R2']:.4f}")
    print(f"\ntotal time: {time.time() - t0:.1f}s")
    return history, n_iters, optimizer


@app.cell
def plot(history):
    import matplotlib.pyplot as plt

    iters = [h["iter"] for h in history]
    losses = [h["loss"] for h in history]
    r2s = [h["R2"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(iters, losses)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("MSE loss")
    ax1.set_yscale("log")
    ax1.set_title("Calibration loss")

    ax2.plot(iters, r2s)
    ax2.axhline(4.75, color="red", linestyle="--", label="target R2=4.75")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("R2")
    ax2.set_title("R2 trajectory")
    ax2.legend()

    fig.tight_layout()
    fig
    return ax1, ax2, fig, iters, losses, plt, r2s


if __name__ == "__main__":
    app.run()
