# perfsim

General high-performance simulator for performative prediction (PP). A deployed model's outputs shape the population's behavior, which becomes the model's next training data. perfsim provides the loop, the environments, and the measurement tools.

## Research direction (Under Development and not fully up to date)

Deploying an LM into a population creates a feedback loop. KL regularization, standard in post-training, anchors the model to its pretrained worldview, controlling how much it can adapt. We study how this anchor shapes the population's long-run equilibrium using differentiable agent-based simulators calibrated to real-world data.

**Core question:** When the performative loop converges, where does the population end up, reflecting the population's own dynamics, or the pretrained model's beliefs? How does the KL coefficient control this? 

**What we measure, population impact as a function of beta:**

- **COVID (Astoria, 37k agents):** LM recommends per-agent isolation levels. ABM simulates SEIRM disease transmission. We measure per-subgroup disease burden, model recommendations, and the gap between what the model recommends and what the data says is needed at each beta.

- **Macro economics (Queens County, 100-agent subsample of 2.7M):** LM advises consumption decisions. ABM simulates earning, consumption, labor markets, and inflation. At equilibrium, the population's economic state (unemployment, asset distribution, inflation) is a function of beta.

**Three reference lines:**
1. Equilibrium under the optimal performative policy (gradient descent through the differentiable ABM)
2. Equilibrium under the frozen pretrained model (the reference prior's implied trajectory)
3. Equilibrium under KL-regularized retraining at each beta (the actual performative loop)

Line 3 approaches line 1 as beta approaches 0 (model adapts) and approaches line 2 as beta grows (model locked to prior). The shape of this curve, and how it differs across subgroups, is the main result.

## Core loop

Each outer round is one epoch:

1. `Environment` is queried: model predictions are written into the population state.
2. `Environment` runs `epoch_size` inner steps under fixed parameters. State evolves; model does not.
3. `Learner` trains on the resulting data.
4. Repeat.

With `epoch_size = 1`, this reduces to the classical lockstep PP loop (Perdomo et al. 2020). With `epoch_size > 1`, the population evolves under fixed parameters before each retraining.

## Layout

```
perfsim/
  perfsim/                        # importable library
    core/                         # ABCs and types
    environments/
      dynamics/                   # FJ, replicator, strategic linear/gradient,
                                  # gaussian shift, accumulating shift
    learners/
      erm.py                      # ERM solved to convergence (RRM)
      gradient.py                 # k SGD/Adam steps per round
      lm/                         # SFT and KL-SFT learners for HF causal LMs
    models/                       # linear, logistic, MLP, HFCausalLM
    scenarios/
      perdomo_loan/               # Perdomo 2020 replication
      at_covid/                   # AgentTorch COVID ABM scenario
      at_macro/                   # AgentTorch macro economics scenario
    adapters/
      agenttorch.py               # wraps agent_torch.Runner as perfsim env
    simulator.py                  # outer epoch loop
    history.py / metrics.py / losses.py
  experiments/                    # NOT part of the package
    scripts/                      # run_covid_lm.py, run_macro_lm.py, calibrate_*, etc.
    condor/                       # HTCondor .sub, .sh, sweep configs
    runs/                         # output artifacts
  tests/
  examples/                       # marimo notebooks
```

## Install

```bash
pip install -e .                 # core only (torch, numpy, pydantic)
pip install -e ".[lm]"          # + transformers, peft, trl, accelerate, pandas
pip install -e ".[agenttorch]"  # + agent_torch for ABM scenarios
pip install -e ".[dev]"         # + pytest, ruff, mypy
```

See `pyproject.toml` for all extras: `[tabular]`, `[kaggle]`, `[hf]`, `[trl]`, `[vllm]`, `[a2a]`.

## Minimal example

```python
from perfsim.environments.dynamics import GaussianShiftWorld
from perfsim.learners import GradientLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.simulator import Simulator
import torch

A = 0.5 * torch.eye(3)
b = torch.tensor([1.0, 0.5, -0.5])
env = GaussianShiftWorld(A=A, b=b, sigma_noise=0.01, batch_size=128)

model = LinearModel(in_features=3, out_features=1, bias=False)
loss = MSELoss()
learner = GradientLearner(model, loss, lr=0.05, steps_per_round=1)

sim = Simulator(env=env, learner=learner, loss=loss)
history = sim.run(n_rounds=50, epoch_size=1, seed=0)
```

## Capability protocols

Optional protocols an Environment may declare:

- `Differentiable`: `grad_sample(model)` / `grad_step(model)` are autograd-traceable.
- `FullyDifferentiable`: full inner-loop rollout is autograd-traceable.
- `Rewarding`: fills a `reward` field in the data dict (for RL learners).
- `Trajectory`: produces multi-step trajectory tensors with a leading time axis.
- `ClosedFormFixedPoint`: provides an analytic fixed point for validation.

## TODO

- [ ] vLLM integration for faster LM inference during generation sweeps
- [ ] A2A protocol wire-up for multi-agent communication
- [ ] RL learners (PPO, GRPO, DPO) with trajectory data schema
- [ ] Learned surrogate (D-hat) for PerfGD without running the full ABM
- [ ] Macro ABM cluster validation (beta sweep)
- [ ] Multi-step Coordinator for complex agent orchestration
