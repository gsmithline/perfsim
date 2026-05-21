# perfsim

General high-performance simulator for performative prediction (PP). Built around a `Predictor` (the platform) that interacts with an `Environment` over a configurable number of inner steps per training epoch.

**Status:** v0 in progress. `Predictor` facade, epoch loop, dynamics environments, and supervised learners are in place. Concrete agent-based environments are deferred to v2.

## Core loop

Each outer round is one **epoch**:

1. `Predictor` publishes its parameters via `predictor.deploy()`.
2. `Environment` runs `epoch_size` inner steps under fixed θ. State evolves; θ does not.
3. `Predictor` trains on the final inner step's data via `predictor.train(...)`.
4. Repeat.

With `epoch_size = 1`, this reduces to the classical lockstep PP loop (Perdomo et al. 2020 RRM, Mendler-Dünner RGD). With `epoch_size > 1`, the population evolves under fixed θ before each retraining: the stateful-PP regime in which a sufficiently long inner loop approximates the stationary distribution under θ.

## Layout

```
perfsim/
  core/                         # ABCs and facades
    predictor.py                # Predictor(Model, Loss, Learner)
    environment.py              # Environment, Dynamics, AgentBased, capability Protocols
    model.py / loss.py / learner.py / dataset.py / types.py
    agent_spec.py / executor.py / messages.py   # preserved scaffolding (idle in v0/v1)
  environments/
    dynamics/                   # FJ, replicator, strategic linear, strategic gradient,
                                # gaussian shift, accumulating shift, stateful population
    agent_based/                # v2 stub
  learners/
    erm.py                      # ERM solved to convergence (RRM)
    gradient.py                 # k SGD/Adam steps (RGD with k=1)
    derivative_aware.py         # stub; requires Differentiable env
    proximal.py / rl/ / lm/     # v1.x and v2 stubs
  models/                       # linear, MLP
  scenarios/
    perdomo_loan/               # v1 literature-replication target
  adapters/
    agenttorch.py               # v3 optional adapter
  simulator.py                  # outer epoch loop, inner step loop
  history.py / metrics.py / losses.py
examples/                       # end-to-end runnable scripts
tests/                          # gating tests, epoch-loop tests, predictor-facade tests
```

## Install

```bash
pip install -e .             # core only
pip install -e ".[tabular]"  # adds pandas + pyarrow for TabularDataset
pip install -e ".[kaggle]"   # adds kaggle CLI for KaggleDataset (Perdomo replication)
pip install -e ".[dev]"      # pytest, ruff, mypy
```

Optional extras for later phases: `[hf]`, `[trl]`, `[vllm]`, `[agenttorch]`, `[a2a]`. See `pyproject.toml`.

## Minimal example

```python
from perfsim.core.predictor import Predictor
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
predictor = Predictor(model=model, loss=loss, learner=learner)

sim = Simulator(env=env, predictor=predictor)
history = sim.run(n_rounds=50, epoch_size=1, seed=0)
```

`Simulator` also accepts the legacy triplet `Simulator(env, learner, loss)`; a `Predictor` is constructed internally.

## Phasing

- **v0**: `Predictor` facade; `Environment` ABCs (`Dynamics`, `AgentBased`); epoch loop with `max_meaningful_epoch_size` enforcement; ERM and Gradient Learners; supervised schema; `TensorDataset`; all dynamics environments listed above. Epoch-loop and facade gating tests in place.
- **v1**: Full architecture, supervised only. `TabularDataset` and `KaggleDataset`. Faithful Perdomo replication via GiveMeSomeCredit (`epoch_size = 1`). Metrics (PR, DPR, stability gap). Examples and reproduction scripts.
- **v2**: First concrete `AgentBased` environment (Mesa-backed or hand-rolled). Trajectory data schema finalized alongside the first RL Learner. `learners/rl/` (PG, PPO, GRPO, DPO). Trajectory aggregation modes beyond final-state-only. Multi-step Coordinator. LM-backed `Predictor` (TRL + vLLM). `HFDataset` optional wrapper. Off-policy evaluation under epoch semantics. Mendler-Dünner RGD replication.
- **v3**: A2A wire-up of the preserved `agents/` and `executor.py` scaffolding. Optional CUDA-fused Simulator subclass. AgentTorch adapter (`adapters/agenttorch.py`). Performatively-optimal outer-RL wrapper (stretch).

## What `Predictor.deploy()` means

The environment receives the deployed handle **once** at the start of `env.step(...)` and is contractually forbidden from re-querying it during the rest of the inner loop. This is what makes "θ waits during the epoch" true by construction: only `predictor.train(...)` mutates the model, and the Simulator calls that exactly once per outer round, after the inner loop.

## Capability traits

Optional Protocols an Environment may declare:

- `Differentiable`: `grad_sample(model)` is autograd-traceable wrt θ.
- `FullyDifferentiable`: stronger; the full inner-loop rollout is autograd-traceable.
- `Rewarding`: fills a `reward` field in the data dict (v2; required by RL Learners).
- `Trajectory`: produces multi-step trajectory tensors with a leading time axis (v2).
- `ClosedFormFixedPoint`: provides an analytic RRM fixed point for gating tests.

In v0, `GaussianShiftWorld` is the only environment that satisfies `Differentiable` and `ClosedFormFixedPoint`. Others can opt in as their use cases land.
