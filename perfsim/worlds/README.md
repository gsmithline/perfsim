# `perfsim.worlds`

Concrete `World` implementations: the performative map **D(θ)** that produces
training data given a deployed predictor θ. The Simulator drives the
`World ↔ Learner` loop; this directory holds the worlds themselves.

A **World** answers two questions each round:

1. *What data does the deployed predictor train on?* `world.step(model) -> {"x", "y"}`
2. *What state, if any, persists into next round?* Handled internally by the World subclass.

For the design rationale (stateless vs stateful, capability traits,
modes 1/2/3 of ABM integration), see the project-level `DESIGN.md`.

## At a glance

| World | State | Predictor's role | What it's for | Mode |
|---|---|---|---|---|
| `GaussianShiftWorld` | stateless | shifts the regression target via `Aθ + b` | RRM/RGD gating test against closed-form fixed point `(I − A)⁻¹ b` | 1a |
| `StrategicLinearWorld` | fixed population (`x_0`, `y`) | linear strategic shift: `x_t = x_0 + ε·w` | Perdomo (2020) strategic classification with a linear predictor | 1a\* |
| `StrategicGradientWorld` | fixed population (`x_0`, `y`) | gradient strategic shift: `x_t = x_0 + ε·∂f/∂x` | Same setup but with arbitrary differentiable predictors (MLPs, etc.) | 1a\* |
| `AccumulatingShiftWorld` | drifting `x_0` | gradient strategic shift | Strategic classification with a population that internalizes manipulation over time | 1b |
| `FJWorld` | per-agent opinions `x_i` | optional platform recommendation | Linear Friedkin-Johnsen opinion dynamics on a graph (free or platform-coupled) | 1b |
| `ReplicatorWorld` | mixture `p ∈ Δ^K` | enters via the caller-supplied `fitness(p, model)` | Discrete Taylor-Jonker replicator on K strategies | 1b |

Modes 1a and 1a\* refer to the [mode taxonomy in DESIGN.md](../../DESIGN.md):
1a = stateless rollout; 1a\* = stateful population (fixed) where the
per-agent strategic response is closed-form (collapsible to a single
algebraic step); 1b = stateful rollout with persistent dynamics; 2 = with
calibrated φ; 3 = with inner population adaptation. perfsim currently
implements 1a, 1a\*, and 1b.

## The base classes

`World` (abstract) lives in `perfsim/core/world.py`. Subclasses pick one of
two skeletons:

**`StatelessWorld`**: D(θ) is history-independent. Each call samples iid from
D(θ_t). Base provides a forked-generator pattern so `sample` (peek) does not
advance the RNG that `step` uses, which keeps off-policy evaluation
(decoupled performative risk) hermetic.

**`StatefulPopulationWorld`**: persistent per-agent state. Subclasses
implement `_step(model) -> (data, next_state)`; the base handles `reset()`,
`sample` (peek; discards next_state), and `step` (advance; installs
next_state). State is a `dict[str, Tensor]` with subclass-defined keys.

There are also **capability traits** (runtime-checkable Protocols in
`perfsim/core/world.py`) that a World can opt into:

- `DifferentiableWorld`: exposes `grad_sample(model)` whose output is
  autograd-traceable wrt θ. Required by derivative-aware Learners (Izzo).
- `FullyDifferentiableWorld`: additionally exposes a `grad_step` that is
  autograd-traceable across multiple rounds.
- `RewardingWorld`: fills a `reward` field in the data dict. Required by
  RL-family Learners.
- `TrajectoryWorld`: produces trajectory tensors with a leading time axis.
- `ClosedFormFixedPoint`: has a `closed_form_fp()` returning the analytic
  RRM fixed point. Used in gating tests.

`GaussianShiftWorld` implements `DifferentiableWorld` and
`ClosedFormFixedPoint`. The rest do not opt into capability traits yet.

---

## `GaussianShiftWorld`

**File:** `gaussian_shift.py`. **Base:** `StatelessWorld`.

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
currently differentiable World.

**Use when:** verifying a Learner's convergence behavior on a problem with
known mathematics; running gating tests that need an analytic anchor.

---

## `StrategicLinearWorld`

**File:** `strategic_linear.py`. **Base:** `StatefulWorld` (custom; population
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

**Use when:** reproducing Perdomo's strategic-loan figure; closed-form
strategic-classification analysis.

---

## `StrategicGradientWorld`

**File:** `strategic_gradient.py`. **Base:** `StatefulWorld`.

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
persists across PP rounds. Each round:

```
predictions = model(features)
x_zero      = (1 − σ) · s + σ · predictions       # σ = platform_sus
for _ in range(n_ticks):
    x = α · x_zero + (1 − α) · (W · x)
```

The graph `W` is typically a (column-normalized) social-network adjacency.
`normalize_adjacency` is provided to convert a raw adjacency in the same
way as the source script.

- `platform_sus = 0`: platform-free FJ. `x_zero = s` every round. Converges
  to `(I − diag(1 − α) W)⁻¹ diag(α) s` per `fj_equilibrium`.
- `platform_sus > 0`: predicted-opinion-anchored FJ. Matches `run_cf_fj.py`.

`fj_equilibrium(x_zero)` returns the analytic fixed point of the inner loop
at the given anchor. Useful as a gating anchor and for diagnostic comparison
across runs.

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

## `StatefulPopulationWorld` (base class for stateful worlds)

**File:** `stateful_population.py`. **Base:** `StatefulWorld` (from `core/world.py`).

ABC for worlds with persistent per-agent state. Subclasses implement
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

Not every world uses every key; the base does not enforce field presence.

**Determinism:** `_step` should be a pure function of `(self._state, model)`.
Stochastic subclasses must manage their own seeded RNG; the forked-generator
pattern from `StatelessWorld` is the recommended way to keep
`sample == step` for the same state.

---

## Helpers (internal)

**`perfsim/worlds/_common.py`**: utilities shared across multiple world
implementations. Internal (underscore prefix); promote to a public location
if external callers need them.

- `validate_strat_features(strat_features, dim)`: canonicalizes the
  `strat_features` argument (None, or a unique LongTensor of in-range
  indices). Used by `StrategicLinearWorld`, `StrategicGradientWorld`, and
  `AccumulatingShiftWorld`.
- `input_gradient(model, x0, expected_n)`: computes `∂(sum f)/∂x` via
  autograd, with `torch.enable_grad()` wrapping for `no_grad`-safe use.
- `apply_strategic_shift(x0, direction, epsilon, strat_features)`: the
  `ε · direction` shift, optionally restricted to a feature subset.

**`perfsim/worlds/fj.normalize_adjacency(adj)`**: column-normalize a raw
adjacency, zeroing isolated rows. Exported from `perfsim.worlds.fj` since
it is specifically the convention `run_free_fj.py` uses.
