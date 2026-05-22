"""End-to-end smoke test: perfsim drives the bundled agent_torch covid sim.

Wires the bundled covid model on the bundled Astoria population (37,518
agents) through perfsim's Simulator. The deployed LinearModel emits a
per-agent isolation score from age; that score becomes the per-agent
isolation probability used by AT's transmission substep.

Requires `pip install 'perfsim[agenttorch]'`. ~12s wall clock on CPU.
"""

import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def imports():
    import time

    import torch

    from perfsim.learners.erm import ERMLearner
    from perfsim.losses import MSELoss
    from perfsim.models.linear import LinearModel
    from perfsim.scenarios.at_covid import make_covid_env
    from perfsim.simulator import Simulator
    return (
        ERMLearner,
        LinearModel,
        MSELoss,
        Simulator,
        make_covid_env,
        time,
        torch,
    )


@app.cell
def _intro():
    import marimo as mo
    mo.md(
        """
        # AT covid smoke test

        Drives the bundled covid sim through perfsim's epoch loop. Each round
        the deployed LinearModel(age -> score) sets per-agent isolation
        probability via `PerfsimIsolationDecision`. AT advances K substeps,
        perfsim extracts `(age, disease_stage)` and retrains the predictor.

        Validates the adapter wiring end-to-end against real AT machinery on
        real bundled astoria data.
        """
    )
    return (mo,)


@app.cell
def build_env(make_covid_env):
    env = make_covid_env(init_seed=0)
    return (env,)


@app.cell
def build_sim(ERMLearner, LinearModel, MSELoss, Simulator, env):
    model = LinearModel(in_features=1, out_features=1)
    loss = MSELoss()
    learner = ERMLearner(model=model, loss=loss, max_iter=20)
    sim = Simulator(env=env, learner=learner, loss=loss)
    return learner, loss, model, sim


@app.cell
def run_loop(sim, time):
    print(":: running 2 rounds, epoch_size=3")
    _t0 = time.time()
    hist = sim.run(n_rounds=2, epoch_size=3, seed=0)
    print(f":: wall time = {time.time() - _t0:.1f}s")
    return (hist,)


@app.cell
def report(env, hist, torch):
    for _i, _record in enumerate(hist):
        print(
            f"  round {_i}: theta = {_record['theta'].tolist()}, "
            f"stability_gap = {_record.get('stability_gap', 'NA')}"
        )

    _citizens = env.runner.state["agents"]["citizens"]
    _counts = torch.bincount(_citizens["disease_stage"].squeeze().long(), minlength=6)
    print(f":: final SEIRM histogram: {_counts.tolist()}")
    return


if __name__ == "__main__":
    app.run()
