# `perfsim.environments.dynamics`

Concrete `Dynamics` environments: the performative map **D(θ)** that produces
training data given a deployed predictor θ. The Simulator drives the
`Environment ↔ Predictor` epoch loop; this directory holds the concrete
dynamical-systems environments.

A `Dynamics` environment answers two questions:

1. *What data does the deployed predictor train on?* `env.step(handle) -> {"x", "y"}`
2. *What state, if any, persists into the next inner step?* Handled internally by the subclass.

For per-epoch behavior, see the `Simulator` outer-epoch / inner-step loop:
during one outer round, `env.step(handle)` is called `epoch_size` times under
a frozen handle from `predictor.deploy()`. Only the final step's data is
passed to `predictor.train(...)`. With `epoch_size = 1` this reduces to the
classical lockstep PP loop.

## At a glance

| Environment | State | Predictor's role | What it's for |
|---|---|---|---|
| `GaussianShiftWorld` | stateless | shifts the regression target via `Aθ + b` | RRM/RGD gating test against closed-form fixed point `(I − A)⁻¹ b` |
| `StrategicLinearWorld` | fixed population (`x_0`, `y`) | linear strategic shift: `x_t = x_0 + ε·w` | Perdomo (2020) strategic classification with a linear predictor |
| `StrategicGradientWorld` | fixed population (`x_0`, `y`) | gradient strategic shift: `x_t = x_0 + ε·∂f/∂x` | Same setup but with arbitrary differentiable predictors (MLPs, etc.) |
| `AccumulatingShiftWorld` | drifting `x_0` | gradient strategic shift | Strategic classification with a population that internalizes manipulation over time |
| `FJWorld` | per-agent opinions `x_i` | optional platform recommendation | Linear Friedkin-Johnsen opinion dynamics on a graph (free or platform-coupled) |
| `ReplicatorWorld` | mixture `p ∈ Δ^K` | enters via the caller-supplied `fitness(p, model)` | Discrete Taylor-Jonker replicator on K strategies |

`StrategicLinearWorld` and `StrategicGradientWorld` are inherently one-shot
best-response and declare `max_meaningful_epoch_size = 1`; the Simulator
rejects `epoch_size > 1` against them. All other environments accept
arbitrary `epoch_size`.

## The base classes

The `Environment` ABC and its sibling intermediates live in
`perfsim/core/environment.py`. Concrete dynamics environments extend one of:

**`StatelessDynamics`**: D(θ) is history-independent. Each call samples iid
from D(θ_t). The base provides a forked-generator pattern so `sample`
(peek) does not advance the RNG that `step` uses, which keeps off-policy
evaluation (decoupled performative risk) hermetic.

**`StatefulPopulationWorld`**: persistent per-agent state. Subclasses
implement `_step(model) -> (data, next_state)`; the base handles `reset()`,
`sample` (peek; discards next_state), and `step` (advance; installs
next_state). State is a `dict[str, Tensor]` with subclass-defined keys.

There are also **capability traits** (runtime-checkable Protocols in
`perfsim/core/environment.py`) that an environment can opt into:

- `Differentiable`: exposes `grad_sample(model)` whose output is
  autograd-traceable wrt θ. Required by derivative-aware Learners (Izzo).
- `FullyDifferentiable`: additionally exposes a `grad_step` that is
  autograd-traceable across multiple inner steps.
- `Rewarding`: fills a `reward` field in the data dict. Required by
  RL-family Learners (v2).
- `Trajectory`: produces multi-step trajectory tensors with a leading
  time axis (v2).
- `ClosedFormFixedPoint`: has a `closed_form_fp()` returning the analytic
  RRM fixed point. Used in gating tests.

`GaussianShiftWorld` implements `Differentiable` and `ClosedFormFixedPoint`.
The rest do not opt into capability traits yet.

---

## `GaussianShiftWorld`

**File:** `gaussian_shift.py`. **Base:** `StatelessDynamics`.

Stateless location-shift regression world.

```
x ~ N(0, I_d)
y = x · (A θ + b) + σ · N(0, 1)
```

For a `LinearModel(d, 1, bias=False)` trained with `MSELoss`, the population
risk minimizer at deployed θ is `Aθ + b`, so RRM iterates
`θ_{t+1} = Aθ_t + b` and the closed-form fixed point is

```
θ* = (I − A)⁻¹ b
```

Used as the canonical gating test: any RRM-style Learner must converge to
`θ*` (within sample noise σ) on a contractive choice of `A` (`‖A‖_2 < 1`).
Exposes `closed_form_fp()` and `grad_sample(model)`. This is the only
currently differentiable environment.

Stateless, so `epoch_size > 1` is wasted compute under final-state-only
training: the final step is one iid sample from D(θ), no different from
a single step. Run with `epoch_size = 1`.

**Use when:** verifying a Learner's convergence behavior on a problem with
known mathematics; running gating tests that need an analytic anchor.

---

## `StrategicLinearWorld`

**File:** `strategic_linear.py`. **Base:** `StatefulDynamics` (custom; population
state is fixed, not evolving).

Perdomo et al. (ICML 2020) Section 5.1 strategic classification. Each agent
has fixed initial features `x_0_i` and a fixed label `y_i`. On each round,
agents perform a linear strategic shift:

```
x_t = x_0 + ε · w
```

where `w` is the deployed linear predictor's weight vector. Optionally only a
subset of feature columns (`strat_features`) is shifted; the rest stay at
their initial values.

Sign convention: in Perdomo's strategic-loan setup agents want to *lower*
their predicted default probability, so callers pass `ε = -μ` for `μ > 0`.

Requires the predictor to expose a `.linear.weight` attribute (so
`LinearModel`, `LogisticModel` work; `MLPModel` does not). Use
`StrategicGradientWorld` for arbitrary predictors.

Declares `max_meaningful_epoch_size = 1`: the strategic best-response is
one-shot, so N inner steps under fixed θ either repeat the same shift or
collapse algebraically into a single step. The Simulator rejects requests
with `epoch_size > 1`.

**Use when:** reproducing Perdomo's strategic-loan figure; closed-form
strategic-classification analysis.

---

## `StrategicGradientWorld`

**File:** `strategic_gradient.py`. **Base:** `StatefulDynamics`.

Generalization of `StrategicLinearWorld` to arbitrary differentiable
predictors. The strategic shift is the gradient of the predictor's scalar
output wrt the input:

```
x_t = x_0 + ε · ∂f(x_0; θ) / ∂x
```

For a linear predictor `f(x) = w · x` the gradient is exactly `w`, so this
world reduces to `StrategicLinearWorld` bit-for-bit. For MLP predictors the
gradient is a non-trivial function of `x_0`, giving each agent a
location-dependent strategic response.

The world wraps its autograd in `torch.enable_grad()` so it works correctly
even when callers (such as `performative_risk`) are inside a
`torch.no_grad()` block.

Same one-shot constraint as `StrategicLinearWorld`:
`max_meaningful_epoch_size = 1`.

**Use when:** strategic classification with non-linear predictors; testing
that perfsim's PP loop handles MLP predictors against strategic populations.

---

## `AccumulatingShiftWorld`

**File:** `accumulating_shift.py`. **Base:** `StatefulPopulationWorld`.

Variant of `StrategicGradientWorld` where the population's natural feature
position `x_0` itself drifts over time toward the strategic position:

```
grad         = ∂f(x_0^t; θ_t) / ∂x
x_strategic  = x_0^t + ε · grad                  # this round's training data
x_0^{t+1}    = (1 − η) · x_0^t + η · x_strategic # drift
```

- `η = 0`: no accumulation; identical to `StrategicGradientWorld`.
- `η = 1`: agents fully adopt each round's strategic position as their new
  natural baseline.
- `0 < η < 1`: sticky population that gradually internalizes past manipulations.

State persists as `state["x0"]` across rounds.

**Use when:** studying how RRM convergence is affected by populations that
adapt their baseline over time, not just their per-round strategic
response.

---

## `FJWorld`

**File:** `fj.py`. **Base:** `StatefulPopulationWorld`.

Linear Friedkin-Johnsen opinion dynamics on a graph. Direct torch port of
the inner loop from
`Opinion-dynamics-post-training/run_free_fj.py:94-101` and
`run_cf_fj.py:372-375`.

Each agent has a fixed innate opinion `s_i`, a (possibly heterogeneous)
peer susceptibility `α_i ∈ [0, 1]`, and a current opinion `x_i` that
persists across env steps. One `env.step(model)` performs one FJ update:

```
predictions = model(features)
x_zero      = (1 − σ) · s + σ · predictions       # σ = platform_sus
x_new       = α · x_zero + (1 − α) · (W · x_state)
state["opinion"] = x_new
```

To drive multiple FJ iterations under a fixed deployed model, use the
Simulator's `epoch_size` (or loop `world.step(model)` manually). The
opinion vector persists continuously across both the inner `epoch_size`
loop and the outer round loop.

The graph `W` is typically a (column-normalized) social-network adjacency.
`normalize_adjacency` is provided to convert a raw adjacency in the same
way as the source script.

- `platform_sus = 0`: platform-free FJ. `x_zero = s` every step. Iterating
  to convergence yields `(I − diag(1 − α) W)⁻¹ diag(α) s` per
  `fj_equilibrium`.
- `platform_sus > 0`: predicted-opinion-anchored FJ. Matches `run_cf_fj.py`.

`fj_equilibrium(x_zero)` returns the analytic fixed point of the FJ inner
update at the given anchor. Useful as a gating anchor and for diagnostic
comparison across runs.

This is the canonical environment for `epoch_size > 1`: under fixed θ, the
opinion-mixing converges toward `fj_equilibrium`, so a sufficiently
large `epoch_size` lets the predictor train on the settled distribution
rather than a transient.

**Use when:** running the user's opinion-dynamics-post-training experiments
inside perfsim; FJ baseline against which platform / LM coupling effects are
compared.

---

## `ReplicatorWorld`

**File:** `replicator.py`. **Base:** `StatefulPopulationWorld`.

Discrete-time replicator dynamics on a K-strategy mixture. Direct torch port
of `evolutionary-prediction-games/evoml/dynamics.py::discrete_replicator`
(Taylor-Jonker 1978 eq. 3):

```
p_{t+1} = p_t · (1 + f(p_t, θ_t)) / ⟨p_t, 1 + f(p_t, θ_t)⟩
```

State is the mixture `p ∈ Δ^K`. Per PP round the world runs `n_ticks`
replicator updates; the caller supplies a `fitness(p, model) -> (K,)`
function, typically per-strategy accuracy or utility of θ_t on
strategy-k data. Emitted training data is `(one-hot strategy id, fitness)`.

The simplex constraint is preserved exactly by the update; verified in
`tests/test_replicator_world.py::TestSimplexInvariant`.

**Use when:** evolutionary game theory experiments on the prediction game;
studying how a deployed predictor reshapes the strategy mixture; replicating
`evoml`-based notebooks inside perfsim.

---

## `StatefulPopulationWorld` (base class for stateful population envs)

**File:** `stateful_population.py`. **Base:** `StatefulDynamics` (from `core/environment.py`).

ABC for environments with persistent per-agent state. Subclasses implement
`_step(model) -> (data, next_state)`; the base handles bookkeeping. Used by
`AccumulatingShiftWorld`, `FJWorld`, `ReplicatorWorld`.

State is a `dict[str, Tensor]` with subclass-defined keys (`"x0"`,
`"opinion"`, `"mixture"`, etc.). Convention:

| key | shape | meaning |
|---|---|---|
| `x` | `(N, D)` | observable features |
| `y` | `(N, K)` | labels |
| `latent` | `(N, L)` | hidden agent state (opinions, history, etc.) |
| `agent_type` | `(N,)` | integer type label for heterogeneous populations |

Not every environment uses every key; the base does not enforce field presence.

**Determinism:** `_step` should be a pure function of `(self._state, model)`.
Stochastic subclasses must manage their own seeded RNG; the forked-generator
pattern from `StatelessDynamics` is the recommended way to keep
`sample == step` for the same state.

---

## Helpers (internal)

**`perfsim/environments/dynamics/_common.py`**: utilities shared across
multiple dynamics environments. Internal (underscore prefix); promote to a
public location if external callers need them.

- `validate_strat_features(strat_features, dim)`: canonicalizes the
  `strat_features` argument (None, or a unique LongTensor of in-range
  indices). Used by `StrategicLinearWorld`, `StrategicGradientWorld`, and
  `AccumulatingShiftWorld`.
- `input_gradient(model, x0, expected_n)`: computes `∂(sum f)/∂x` via
  autograd, with `torch.enable_grad()` wrapping for `no_grad`-safe use.
- `apply_strategic_shift(x0, direction, epsilon, strat_features)`: the
  `ε · direction` shift, optionally restricted to a feature subset.

**`perfsim/environments/dynamics/fj.normalize_adjacency(adj)`**:
column-normalize a raw adjacency, zeroing isolated rows. Exported from
`perfsim.environments.dynamics.fj` since it is specifically the convention
`run_free_fj.py` uses.
