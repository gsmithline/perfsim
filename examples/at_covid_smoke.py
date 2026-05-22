"""End-to-end smoke test: perfsim drives the bundled agent_torch covid sim.

Wires the bundled covid model on the bundled Astoria population (37,518
agents) through perfsim's Simulator. The deployed LinearModel emits a
per-agent isolation score from age; that score becomes the per-agent
isolation probability used by AT's transmission substep.

All covid-specific glue (langchain shim, YAML path patch, substep registry,
zero-initialized platform_signal, default callables) lives in
`perfsim.scenarios.at_covid`. This script is the caller.

Run: `python examples/at_covid_smoke.py`. Requires `pip install
'perfsim[agenttorch]'`. ~12s wall clock on CPU.
"""

from __future__ import annotations

import time

import torch

from perfsim.learners.erm import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models.linear import LinearModel
from perfsim.scenarios.at_covid import make_covid_env
from perfsim.simulator import Simulator


def main():
    env = make_covid_env(init_seed=0)

    model = LinearModel(in_features=1, out_features=1)
    loss = MSELoss()
    learner = ERMLearner(model=model, loss=loss, max_iter=20)

    sim = Simulator(env=env, learner=learner, loss=loss)

    print(":: running 2 rounds, epoch_size=3")
    t0 = time.time()
    hist = sim.run(n_rounds=2, epoch_size=3, seed=0)
    print(f":: wall time = {time.time() - t0:.1f}s")

    for i, record in enumerate(hist):
        print(
            f"  round {i}: theta = {record['theta'].tolist()}, "
            f"stability_gap = {record.get('stability_gap', 'NA')}"
        )

    citizens = env.runner.state["agents"]["citizens"]
    counts = torch.bincount(citizens["disease_stage"].squeeze().long(), minlength=6)
    print(f":: final SEIRM histogram: {counts.tolist()}")


if __name__ == "__main__":
    main()
