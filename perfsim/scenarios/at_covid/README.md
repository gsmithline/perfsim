# at_covid

A realistic environment: the bundled AgentTorch COVID model run through perfsim's loop. The model's predictions become an isolation signal that changes how the epidemic spreads, and the resulting outcomes are what it retrains on. Built on the AgentTorch adapter (`perfsim/adapters/README.md`).

## Use it

```python
from perfsim.scenarios.at_covid import make_covid_env

env = make_covid_env(init_seed=0)
sim = Simulator(env=env, learner=ERMLearner(model, loss), loss=loss)
hist = sim.run(n_rounds=10, epoch_size=20, seed=0)
```

Each round: the model scores every agent from their age, that score is written in as an isolation probability, the simulation steps forward (agents isolate, the disease spreads and progresses), and the model retrains on `(age -> disease stage)`. The four wiring functions are overridable keyword args on `make_covid_env`.

Install: `pip install -e ".[agenttorch]"`.

## Files

- `env.py` — `make_covid_env`, `build_covid_runner`, the default wiring functions.
- `action.py` — `PerfsimIsolationDecision`, the substep that reads our signal.
- `_compat.py` — private shims for AgentTorch 0.6.0 quirks.

## Things to know

- This is a **wiring / smoke-grade** scenario: it shows the loop runs end to end, not a calibrated epidemic. Override the default outcome extractor for a sharper target.
- **Fixed ~37.5k agents** (can't subsample without rebuilding the data). Roughly 5s to start, ~1s per step on CPU.
- A few AgentTorch 0.6.0 rough edges (a missing langchain import, a hardcoded population path, resolver re-registration) are handled by the shims in `_compat.py`.

## Gradients

`grad_run` gives you a gradient through one epoch. For it to be nonzero you need: seeded infections (`seed_initial_infections(env, fraction=0.05)`, an all-susceptible population gives zero), at least ~5 steps so effects accumulate, and the non-detaching writer (`default_signal_writer_grad`). The model is frozen across rounds, so this measures one epoch, not the whole loop.

## macro_economics

Not wired up: AgentTorch 0.6.0's macro-economics model doesn't run end to end (needs LLM machinery, has a shape bug, and is missing population files). Those are upstream issues, not a perfsim config problem.
