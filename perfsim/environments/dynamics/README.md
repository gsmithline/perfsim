# `perfsim.environments.dynamics`

Ready-made environments that react to your model. You don't write a map for these; you pick one, pass it straight to the `Simulator`, and run the loop. Most carry internal state and can only be sampled by running them forward (the samples level).

Each round the simulator runs the environment `epoch_size` times under a frozen model, and the model retrains on the last step's data. `epoch_size = 1` is the plain lockstep loop; larger values let the population settle under a fixed model before retraining.

```python
from perfsim import Simulator
from perfsim.environments.dynamics import FJWorld
# world = FJWorld(...); sim = Simulator(env=world, learner=..., loss=...)
```

## The environments

| Environment | What reacts | Use it for |
|---|---|---|
| `GaussianShiftWorld` | the regression target shifts with the model | quick sanity check: RRM/RGD should hit the closed-form fixed point `(I − A)⁻¹ b` |
| `StrategicLinearWorld` | features move toward the classifier's weights | strategic classification with a linear model |
| `StrategicGradientWorld` | features move along the model's input gradient | strategic classification with any differentiable model |
| `AccumulatingShiftWorld` | the population slowly adopts the gamed position | strategic gaming that builds up over rounds |
| `FJWorld` | opinions on a graph drift toward neighbors and the platform | opinion dynamics, recommendation effects |
| `ReplicatorWorld` | a mixture over strategies evolves by fitness | population shares under selection |

The two `Strategic*` worlds are one-shot (set `max_meaningful_epoch_size = 1`); the rest accept any `epoch_size`.

## Notes per environment

- **`GaussianShiftWorld`** — stateless. `x ~ N(0, I)`, `y = x·(Aθ + b) + noise`. With a linear model and MSE, retraining converges to `θ* = (I − A)⁻¹ b` when `‖A‖ < 1`. Run at `epoch_size = 1`.
- **`StrategicLinearWorld`** — fixed population; each round `x = x₀ + ε·w` (w = the model's weights). Pass `ε = -μ` for agents lowering their risk.
- **`StrategicGradientWorld`** — same idea for any differentiable model: `x = x₀ + ε·∂f/∂x`. Reduces to the linear world when the model is linear.
- **`AccumulatingShiftWorld`** — like the gradient world, but `x₀` itself drifts toward the gamed position each round (rate `η`; `η=0` is static, `η=1` fully adopts).
- **`FJWorld`** — Friedkin-Johnsen opinion dynamics on a graph. Each step blends innate opinion, the platform's prediction (weight `platform_sus`), and the neighbor average (weight set by peer susceptibility). `platform_sus=0` is the platform-free baseline. `fj_equilibrium()` gives the analytic fixed point; `normalize_adjacency()` prepares a raw graph.
- **`ReplicatorWorld`** — discrete replicator on a K-strategy simplex; you supply `fitness(p, model)`. Stays on the simplex exactly.

## Base classes

In `perfsim/core/environment.py`: `StatelessDynamics` (no memory, fresh draw each step) and `StatefulPopulationWorld` (persistent per-agent state; subclasses write `_step`). An environment can opt into capability traits, `Differentiable`, `FullyDifferentiable`, `Rewarding`, `Trajectory`, `ClosedFormFixedPoint`, when it supports them.
