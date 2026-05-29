# `perfsim.environments.dynamics`

Concrete `Dynamics` environments: the performative map **D(theta)** that produces
training data given a deployed predictor theta. `AgentBased` envs (per-agent
decision rules) live in `perfsim/adapters/`; the only one is the AgentTorch
adapter.

A `Dynamics` env answers: what data does the predictor train on
(`env.step(handle) -> {"x", "y"}`), and what state persists to the next step.
During one outer round `env.step(handle)` runs `epoch_size` times under a frozen
handle; only the final step's data trains the predictor. `epoch_size = 1` is the
classical lockstep PP loop.

## At a glance

| Environment | State | Predictor's role | What it's for |
|---|---|---|---|
| `GaussianShiftWorld` | stateless | shifts the regression target via `Atheta + b` | RRM/RGD gating test against closed-form fixed point `(I − A)⁻¹ b` |
| `StrategicLinearWorld` | fixed population (`x_0`, `y`) | linear strategic shift: `x_t = x_0 + ε·w` | Perdomo (2020) strategic classification with a linear predictor |
| `StrategicGradientWorld` | fixed population (`x_0`, `y`) | gradient strategic shift: `x_t = x_0 + ε·∂f/∂x` | Same setup with arbitrary differentiable predictors |
| `AccumulatingShiftWorld` | drifting `x_0` | gradient strategic shift | Strategic classification where the population internalizes manipulation over time |
| `FJWorld` | per-agent opinions `x_i` | optional platform recommendation | Friedkin-Johnsen opinion dynamics on a graph |
| `ReplicatorWorld` | mixture `p ∈ Δ^K` | enters via caller-supplied `fitness(p, model)` | Discrete Taylor-Jonker replicator on K strategies |

`StrategicLinearWorld` and `StrategicGradientWorld` are one-shot best-response
and set `max_meaningful_epoch_size = 1`; the rest accept any `epoch_size`.

## Base classes

Defined in `perfsim/core/environment.py`. Concrete envs extend:

- **`StatelessDynamics`**: history-independent D(theta), iid per step. Forked-generator
  pattern keeps `sample` (peek) from advancing the `step` RNG.
- **`StatefulPopulationWorld`**: persistent per-agent state. Subclasses implement
  `_step(model) -> (data, next_state)`; the base handles reset/sample/step.

Capability traits (runtime-checkable Protocols) an env may opt into:
`Differentiable` (`grad_sample`), `FullyDifferentiable` (`grad_step` too),
`Rewarding`, `Trajectory`, `ClosedFormFixedPoint` (`closed_form_fp`). Only
`GaussianShiftWorld` opts in so far (`Differentiable`, `ClosedFormFixedPoint`).

## Environments

**`GaussianShiftWorld`** (`gaussian_shift.py`, `StatelessDynamics`). Stateless
regression world: `x ~ N(0, I)`, `y = x·(Atheta + b) + σ·noise`. Under
`LinearModel + MSELoss`, RRM iterates `theta_{t+1} = Atheta_t + b` with fixed
point `theta* = (I − A)⁻¹ b`. Canonical gating test (`‖A‖_2 < 1`). Run with
`epoch_size = 1`.

**`StrategicLinearWorld`** (`strategic_linear.py`). Perdomo (2020 §5.1) strategic
classification: `x_t = x_0 + ε·w`, `w` the predictor's weight vector. Pass
`ε = -μ` for risk-lowering agents. Needs a `.linear.weight` (LinearModel,
LogisticModel). One-shot: `max_meaningful_epoch_size = 1`.

**`StrategicGradientWorld`** (`strategic_gradient.py`). Generalizes the above to
any differentiable predictor: `x_t = x_0 + ε·∂f(x_0; theta)/∂x`. Linear `f`
reduces to `StrategicLinearWorld`. Wraps its autograd in `enable_grad()` so it
works inside a caller's `no_grad`. One-shot.

**`AccumulatingShiftWorld`** (`accumulating_shift.py`). Like the gradient world
but `x_0` drifts toward the strategic position: `x_0^{t+1} = (1−η)·x_0 + η·x_strategic`.
`η=0` recovers the static world; `η=1` fully adopts each round's position. State
persists as `state["x0"]`.

**`FJWorld`** (`fj.py`). Linear Friedkin-Johnsen opinion dynamics on a graph. One
step: `x_zero = (1−σ)·s + σ·predictions`, `x_new = α·x_zero + (1−α)·(W·x)`, where
`s` is innate opinion, `α` peer susceptibility, `σ` platform_sus. `platform_sus=0`
is platform-free FJ. `fj_equilibrium(x_zero)` gives the analytic fixed point.
Canonical `epoch_size > 1` env (opinions settle under fixed theta).
`normalize_adjacency` row-normalizes a raw adjacency.

**`ReplicatorWorld`** (`replicator.py`). Discrete Taylor-Jonker replicator on a
K-strategy mixture: `p_{t+1} = p·(1 + f(p, theta)) / ⟨p, 1 + f(p, theta)⟩`. Caller
supplies `fitness(p, model) -> (K,)`; runs `n_ticks` per round. Emits
`(one-hot strategy id, fitness)`. Simplex preserved exactly.

## Helpers

`_common.py` (internal): `validate_strat_features`, `input_gradient` (autograd
`∂(sum f)/∂x`, `no_grad`-safe), `apply_strategic_shift`. `fj.normalize_adjacency`
row-normalizes a raw adjacency the way `run_free_fj.py` does.
