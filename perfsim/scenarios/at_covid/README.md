# at_covid

Drives the bundled `agent_torch` covid model through perfsim's epoch loop via
the `AgentTorchEnvironment` adapter. Adapter contract: `perfsim/adapters/README.md`.

## Public API

```python
from perfsim.scenarios.at_covid import make_covid_env

env = make_covid_env(init_seed=0)
sim = Simulator(env=env, learner=ERMLearner(model, loss), loss=loss)
hist = sim.run(n_rounds=10, epoch_size=20, seed=0)
```

Per epoch: model scores each agent from age -> `signal_writer` deposits at
`agents/citizens/platform_signal` -> `runner.step(epoch_size)` (our
`PerfsimIsolationDecision` reads the signal as isolation probability; AT's
`NewTransmission` + `SEIRMProgression` evolve `disease_stage`) ->
`state_extractor` returns `(x=age, y=disease_stage)` -> retrain. Example:
`examples/at_covid_smoke.py`.

## Files

- `env.py` -- `build_covid_runner`, `make_covid_env`, default callables.
- `action.py` -- `PerfsimIsolationDecision` (perfsim-controlled substep).
- `_compat.py` -- AT 0.6.0 compatibility shims (private).

Install: `pip install -e ".[agenttorch]"` (pins `agent-torch>=0.6,<0.7`). All four
callables on `make_covid_env(...)` are overridable, keyword-only.

## Caveats

- **langchain shim.** AT 0.6.0's covid `action.py` imports removed langchain
  symbols; stubbed via `install_langchain_shim()` (never called, we register our
  own action).
- **Hardcoded population path** in the bundled YAML; text-replaced to the bundled
  astoria path before `read_config`.
- **OmegaConf resolver non-idempotency.** `read_config` re-registers resolvers
  each call; `should_register_resolvers()` tracks first-call state.
- **Fixed 37,518 agents** (cannot subsample without rewriting the data). ~5s init,
  ~1s/step on CPU.
- **Smoke-grade target.** `disease_stage` cast to float; validates wiring, not
  calibration. Override `default_state_extractor` for a sharper target.

## Gradients (`grad_run`)

`grad_run(model, n_steps)` is `run` without `no_grad` around `model(X)`. Four
conditions for non-zero gradient:

1. Seed infections (`seed_initial_infections(env, fraction=0.05)`) -- all-S gives 0.
2. `n_steps >= ~5` so cumulative quantities build signal.
3. Use `grad_run`, not `run`.
4. Non-detaching writer: pass `default_signal_writer_grad` (clones, no detach).

Chain: `model.weight -> preds -> signal -> will_isolate -> newly_exposed_today ->
update_stages -> disease_stage` (and `daily_infected`); linear in
`newly_exposed_today`, `StraightThroughBernoulli.backward` passes through. theta is
frozen across the epoch loop, so this is one-shot measurement, not end-to-end
rollout differentiation. Demo: `examples/at_covid_grad_smoke.py`.

## Signal pattern

`PerfsimIsolationDecision` reads `platform_signal` every substep, never writes
back (A2 / fixed anchor); the bundled substeps don't touch it either, so
`strict_signal=True` passes. A substep that mutates it raises
`SignalMutationError` (disable with `strict_signal=False`).

## macro_economics

Not wired: AT 0.6.0's `macro_economics` is unrunnable end-to-end --
`WorkConsumptionPropensity` needs LLM machinery + has a dead `will_work` ref;
`labor_market.UpdateMacroRates` has a matmul shape bug on `step(1)`; bundled
`populations/NYC` is missing files the config wants. Upstream bugs, not
config mismatch.
