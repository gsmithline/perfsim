# `perfsim.scenarios.at_covid`

Drives the bundled `agent_torch` covid model through perfsim's epoch loop.
First concrete demonstration of perfsim's `AgentTorchEnvironment` adapter
against a real bundled AT sim. See `perfsim/adapters/README.md` for the
adapter contract; this README covers the covid-specific glue.

## What it does

```python
from perfsim.scenarios.at_covid import make_covid_env
from perfsim.simulator import Simulator
from perfsim.models.linear import LinearModel
from perfsim.learners.erm import ERMLearner
from perfsim.losses import MSELoss

env = make_covid_env(init_seed=0)
model = LinearModel(in_features=1, out_features=1)
loss = MSELoss()
sim = Simulator(env=env, learner=ERMLearner(model=model, loss=loss), loss=loss)
hist = sim.run(n_rounds=10, epoch_size=20, seed=0)
```

Each epoch:

1. perfsim queries the model with per-agent age, gets a scalar score per agent.
2. `signal_writer` deposits the score at `runner.state["agents"]["citizens"]["platform_signal"]`.
3. `runner.step(num_steps=epoch_size)` advances the bundled AT covid sim.
   Our `PerfsimIsolationDecision` substep reads `platform_signal`, sigmoids
   it, and uses it as the per-agent isolation probability. AT's bundled
   `NewTransmission` and `SEIRMProgression` consume the isolation decision
   to evolve `disease_stage`.
4. `state_extractor` returns `(x=age, y=disease_stage)` to perfsim.
5. perfsim retrains the model on that data.

The example caller is `examples/at_covid_smoke.py` (~40 lines).

## Layout

```
perfsim/scenarios/at_covid/
  __init__.py     # public exports
  env.py          # build_covid_runner, make_covid_env, default callables
  action.py       # PerfsimIsolationDecision (perfsim-controlled substep)
  _compat.py      # AT 0.6.0 compatibility shims (private)
  README.md
```

## Install

```bash
pip install -e ".[agenttorch]"
```

Pins `agent-torch>=0.6,<0.7`. The bundled covid model loads against
`agent_torch/populations/astoria/` which ships inside the package.

## Customizing the loop

All four callables on `make_covid_env(...)` are overridable. Keep them
keyword-only:

```python
def my_feature_provider(runner):
    citizens = runner.state["agents"]["citizens"]
    # use age + disease_stage as features
    return torch.stack([citizens["age"].float(),
                        citizens["disease_stage"].float()], dim=-1)

env = make_covid_env(feature_provider=my_feature_provider, ...)
```

You can also pass `keep_trajectory=True` if you want `runner.state_trajectory`
preserved across epochs, or `strict_signal=False` if you specifically want
a substep that mutates `platform_signal` during the inner loop.

## Caveats

Honest list of things that are sharp or non-portable. Each is encapsulated
in `_compat.py` or `env.py`; you do not need to do anything about them
unless agent_torch ships a fix.

1. **langchain shim.** `agent_torch 0.6.0`'s covid `action.py` imports
   `LangchainLLM`, which imports symbols (`langchain.chains.LLMChain`,
   `langchain.prompts.*`) that were removed in langchain 1.x. We stub
   them at module load via `install_langchain_shim()`. The stubs are
   never called because we register our own action substep.
2. **Hardcoded population path.** The bundled covid YAML hardcodes the
   AT authors' machine path
   (`/u/ayushc/projects/GradABM/...`). We text-replace it to the bundled
   astoria path before `read_config`.
3. **OmegaConf resolver non-idempotency.** AT's `read_config` registers
   OmegaConf resolvers on every call. The second call (triggered by
   `Simulator.reset` rebuilding the runner) raises. We track first-call
   state in `_compat.should_register_resolvers()`.
4. **Fixed population size.** The bundled Astoria data has 37,518
   agents. Cannot subsample without rewriting the CSV/pickle files.
   ~5s init, ~1s per AT step on CPU.
5. **Smoke-grade target.** `disease_stage` is an integer 0..5; we cast
   to float and train a linear model on it. This validates the wiring,
   not the calibration. Most agents stay at 0 (Susceptible) for short
   runs; theta drift is small. Override `default_state_extractor` for
   a more sensitive target (e.g., `daily_infected` summed over the
   epoch, or `is_quarantined`).

## Measuring gradients (`grad_run`)

The adapter exposes a parallel `grad_run(model, n_steps)` that does the same
work as `run` but without `torch.no_grad` around `model(X)`. With covid:

```python
from perfsim.scenarios.at_covid import (
    default_signal_writer_grad,
    make_covid_env,
    seed_initial_infections,
)

env = make_covid_env(
    init_seed=0,
    signal_writer=default_signal_writer_grad,   # opt in to non-detaching writer
)
seed_initial_infections(env, fraction=0.05, seed=0)

model = torch.nn.Linear(1, 1)
env.grad_run(model, n_steps=5)
loss = env.runner.state["environment"]["daily_infected"].sum()
loss.backward()
# model.weight.grad and model.bias.grad now hold non-zero gradients.
```

See `examples/at_covid_grad_smoke.py` for the runnable demo.

Four conditions must all hold for non-zero gradient (any one missing gives
gradient = 0 even though the graph survives):

1. **Realistic initial dynamics.** Starting all-susceptible gives 0 because
   `potentially_exposed_today = st_bernoulli(probs) * (1 - will_isolate)`,
   `probs ≈ 0`, and `bernoulli(0) = 0` everywhere. Use
   `seed_initial_infections(env, fraction=0.05)` to fix this.
2. **Multi-step rollout.** `n_steps >= ~5` so cumulative quantities like
   `daily_infected` build up signal.
3. **Use `grad_run`, not `run`.** `run` wraps `model(X)` in `torch.no_grad`.
4. **Non-detaching signal writer.** The default `default_signal_writer`
   does `.detach().clone()` (safe for the non-grad `run` path). For
   gradient measurement, pass `default_signal_writer_grad` (uses
   `.clone()` only) via `make_covid_env(signal_writer=...)`. If you
   write your own, do the same: clone, do not detach.

Mechanically, the gradient flows: `model.weight` → `preds` → `signal` →
`PerfsimIsolationDecision` → `will_isolate` → `(1 - will_isolate) *
st_bernoulli(probs)` → `newly_exposed_today` → `update_stages` →
`disease_stage`, and also through `daily_infected`. AT's `update_stages`
and `update_transition_times` are linear in `newly_exposed_today`, and
`StraightThroughBernoulli.backward` returns `grad_output * ones` (pass-
through), so the chain is intact.

The Simulator's epoch loop still freezes theta across the inner loop
(DESIGN.md §8). `grad_run` is for one-shot gradient measurement, not for
end-to-end differentiable PP loops (that is a v2 question).

## Pattern contract

`PerfsimIsolationDecision` reads `platform_signal` every substep but does
not write back. This is the A2 (fixed anchor) contract. AT's bundled
`NewTransmission` and `SEIRMProgression` do not touch `platform_signal`
either. The adapter's `strict_signal=True` check therefore passes.

If you author a substep that mutates `platform_signal` (B violation), the
adapter raises `SignalMutationError`. Disable with `strict_signal=False`.

## Why not macro_economics yet

agent_torch 0.6.0 ships `macro_economics` but it is unrunnable end-to-end:

- The bundled `WorkConsumptionPropensity` action requires LLM machinery and
  has a dead `will_work` reference (returns an unset name).
- `labor_market.UpdateMacroRates` (both `transition.py` and `transition_nyc.py`
  variants) has a shape bug on `runner.step(1)`:
  `RuntimeError: size mismatch, got input (1), mat (1x10), vec (1)` from
  the matmul against `external_UAC`. This is an upstream code bug, not a
  config mismatch. Cannot be patched from outside without forking the
  substep.
- Bundled `populations/NYC` is missing the `100_sampled_agents` subdir the
  config wants, and `kings_county_monthly_cases` exists only as `.csv`
  (config wants `.pkl`).

When agent_torch ships a fix, a `perfsim/scenarios/at_macro/` package would
mirror this one's layout. Until then, only covid is wired.

## See also

- `perfsim/adapters/README.md` -- adapter contract, three-callable API, A/B taxonomy
- `DESIGN.md` S20 -- design rationale, AT 0.6.0 verification findings
- `examples/at_covid_smoke.py` -- thin caller, ~40 lines
