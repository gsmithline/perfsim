<p align="center">
  <img src="docs/logo.png" alt="perfsim" width="420">
</p>

A benchmark library for performative prediction (PP): settings where a deployed model changes the data it is then evaluated and retrained on. The central object is the **distribution map** D: Theta -> Delta(Z) (Perdomo et al. 2020), the rule for how the world reacts to your model. perfsim treats that map as the main building block, makes "how much a method is allowed to know about it" part of the types, and gives you the retraining loop, the environments, and the tools to measure what happens.

The design is inspired by the survey of Kehrenberg et al. 2026 (arXiv:2602.10176). 

## The distribution map

A map takes a model and gives back the data that model induces. The common case is a fixed base population plus a rule for how the model shifts it. That is a `TransformationMap`: you write those two pieces and sampling is handled for you.

```python
import torch
from perfsim import TransformationMap
from perfsim.core import SUPERVISED_SCHEMA

class CreditGamingMap(TransformationMap):
    """Applicants shift gameable features toward the classifier's weights."""

    model_channel = "parameters"   # this map reads theta itself

    def __init__(self, x0, y, eps=0.5):   # eps: sensitivity, how far features move per unit of model weight
        self._x0, self._y, self._eps = x0, y, eps

    @property
    def produces_schema(self):
        return SUPERVISED_SCHEMA

    def sample_base(self, n, *, generator):
        idx = torch.randint(self._x0.shape[0], (n,), generator=generator)
        return {"x": self._x0[idx], "y": self._y[idx]}

    def transform(self, z_base, model):
        return {"x": z_base["x"] + self._eps * model.params, "y": z_base["y"]}
```

Maps see the deployed model through a `ModelView`. The prediction channel is always open, the parameter channel opens only when the map declares `model_channel = "parameters"`. Reading an undeclared channel raises `AccessError`. What a component was allowed to see is enforced by the API, not by convention.

## Access levels: how much a method assumes it knows

Methods differ in how much they assume they understand about how the world reacts to the deployed model. perfsim makes that assumption explicit and enforces it, so a method can only use what its level allows.

| Level | What you assume you have | In code |
|---|---|---|
| samples | You can only get data by running the environment. You don't know its density function, you just draw samples from it. This is the realistic case. | `map.sample` |
| mechanism |  A fixed base population plus the rule for how deploying the model shifts it. You can compute the shift, but not the probability of any given outcome. | `TransformationMap` (`sample_base` + `transform`) |
| density | You know the full probabilities: how likely any outcome is under a given model. The strongest assumption, and the rarest in practice. | `DensityMap.log_prob` |

Lower levels assume less, so they apply to more realistic environments but let a method do less. Higher levels let a method optimize more directly, but only apply when you genuinely know that much. `access_levels(map)` reports which a given map offers.

The map side is enforced automatically: a map sees the model through a `ModelView` and cannot read the model's weights unless it declared that it needs them.

On the method side, the standard loop hands a learner materialized samples, so ordinary methods (RRM, RGD) are samples-only just by what they're given. A method that needs more asks for a handle: `env.access("mechanism")` (or `"density"`) lets it sample and apply the shift, but refuses anything above the level it asked for, or any level the map doesn't offer. So a higher-level method commits to its level by the handle it takes, and the handle is what stops it from quietly using more.

## Running the PP loop

Wrap a map in `MapEnvironment` and run it with the `Simulator`. `ERMLearner` retrains to convergence each round (RRM) and the Gaussian-shift map has a known fixed point to check against:

```python
import torch
from perfsim import GaussianShiftMap, MapEnvironment, Simulator
from perfsim.learners import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel

gmap = GaussianShiftMap(A=0.5 * torch.eye(3), b=torch.ones(3), sigma_noise=0.01)
env = MapEnvironment(gmap, batch_size=256)

model = LinearModel(3, 1, bias=False)
loss = MSELoss()
sim = Simulator(env=env, learner=ERMLearner(model, loss), loss=loss)
history = sim.run(n_rounds=20, epoch_size=1, seed=0)

print(model.get_params(), "vs", gmap.closed_form_fp())
```

Each round: the learner trains on last round's data, the model is deployed, the environment produces the next batch. With `epoch_size > 1` the population evolves under a frozen model before the next retraining.

## Map families included

- `StrategicLinearMap`: quadratic-cost strategic classification (Perdomo et al. 2020)
- `LocationScaleMap`: linear-in-theta location and scale shifts (Miller et al. 2021)
- `GaussianShiftMap`: all three access levels plus a closed-form RRM fixed point
- `MixtureShiftMap`: the configurable benchmark map. A Gaussian-mixture base with a theta-dependent shift, exposing all three access levels on one object, with independent knobs for sensitivity (`epsilon`) and modality (`n_modes` / `separation`, which turns mixture dominance on or off). Use it when you want to vary one environment property at a time and hand the same env to a method at any tier.

More families maps and environments are planned. 

## Worlds: stateful simulators

True simulators with internal state stay native `Environment`s: Friedkin-Johnsen opinion dynamics, the recommender ecosystem, AgentTorch ABMs. You can't write down their map analytically, you can only sample from it by running them forward (the samples level). These are the realistic environments, where the conclusions from the controllable maps get stress-tested.

## Stateful PP: transition maps

By default the population reacts fully each round. To make it react gradually instead, wrap a map in `GeometricDecayEnv(map, lam=...)` or `StaggeredResponseEnv(map, k=...)` and run at `epoch_size=1`. `metrics.py` has `limiting_distribution` and `long_term_performative_risk` for the long-run behavior.

## Layout

```
perfsim/
  perfsim/                        # importable library
    maps/                         # THE CENTRAL PACKAGE
      base.py                     # DistributionMap / TransformationMap / DensityMap,
                                  # ModelView, AccessError, access_levels
      access.py                   # MapAccess: tier-restricted handle for learners
      location_scale.py           # canonical families
      strategic.py
      gaussian_shift.py
      mixture_shift.py            # configurable map: sensitivity + modality knobs
    core/                         # ML plumbing: Model, Loss, Learner, Predictor,
                                  # Dataset, schemas, Environment ABCs
    environments/
      map_env.py                  # MapEnvironment adapter (map -> epoch loop)
      dynamics/                   # FJ, replicator, strategic worlds, gaussian shift
    learners/
      erm.py                      # ERM solved to convergence (RRM)
      gradient.py                 # k SGD/Adam steps per round (RGD at k=1)
      lm/                         # SFT and KL-SFT learners for HF causal LMs
    models/                       # linear, logistic, MLP, HFCausalLM
    scenarios/
      perdomo_loan/               # Perdomo 2020 replication
      at_covid/                   # AgentTorch COVID ABM scenario
      recommender/                # performative recommender ecosystem
    adapters/
      agenttorch.py               # wraps agent_torch.Runner as perfsim env
    simulator.py                  # outer epoch loop
    history.py / metrics.py / losses.py
  experiments/                    # NOT part of the package
  tests/
  examples/                       # bundled datasets (pokec, two_tickets)
```

Dependency rule: `maps/` depends only on `core` primitives (types, model), `environments/` depend on `maps/`, never the reverse.

## Install

```bash
pip install -e .                 # core (torch, numpy, pandas)
pip install -e ".[lm]"          # + transformers, peft, trl, accelerate
pip install -e ".[agenttorch]"  # + agent_torch for ABM scenarios
pip install -e ".[dev]"         # + pytest, ruff, mypy
```

See `pyproject.toml` for all extras: `[tabular]`, `[kaggle]`, `[hf]`, `[trl]`, `[vllm]`.

## Capability protocols

Optional protocols an Environment may declare:

- `Differentiable`: you can take a gradient through `grad_sample(model)` (an oracle baseline, stronger than any of the access levels above).
- `FullyDifferentiable`: you can take a gradient through the whole inner rollout.
- `Rewarding`: fills a `reward` field for RL learners.
- `Trajectory`: returns multi-step trajectories with a time axis.
- `ClosedFormFixedPoint`: provides an exact fixed point to check against.

## Roadmap

Headlines: the benchmark runner that runs every method on the same environments and stamps the results (the access enforcement itself is now in, via `MapAccess`, is a method that uses the full-density access is next), fitted maps (`fit(observations)`) so you can approximate a sample-only world with a higher-level map and measure how wrong the approximation is, assumption diagnostics (how sensitive the environment is, whether it has one mode or several) reported as estimates, and the remaining map families from the survey.

