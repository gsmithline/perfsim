# perfsim

General high-performance simulator for performative prediction (PP). Built around a `Predictor` (the platform) that interacts with an `Environment` over a configurable number of inner steps per training epoch.

**Status:** v0 in progress. `Predictor` facade, epoch loop, dynamics environments, and supervised learners are in place. Concrete agent-based environments are deferred to v2.

## Research direction

**Thesis:** Deploying an LM into a population creates a feedback loop: the model's outputs shape the population's behavior, which becomes the model's next training data. KL regularization — standard in post-training — anchors the model to its pretrained worldview, controlling how much it can adapt. We study how this anchor shapes the population's long-run equilibrium using differentiable agent-based simulators calibrated to real-world data.

**Core question:** When the performative loop converges, where does the population end up? Specifically: does the equilibrium population state reflect the population's own dynamics, or the pretrained model's beliefs about it? And how does the KL coefficient control this?

**What we measure — population equilibria as a function of beta:**

In FJ opinion dynamics, this was directly visible: the population's opinion vector settles at a fixed point that shifts as beta (KL strength) changes. Low beta: opinions settle where the peer dynamics and fine-tuning signal push them. High beta: opinions settle near where the pretrained model's predictions would push them. The frozen (unretrained) model defines one attractor; the fully adapted model (beta=0) defines another. The KL coefficient interpolates between them.

In the ABM settings (covid, macro), we study the same phenomenon in richer dynamics:

- **COVID (Astoria, 37k agents):** The LM recommends per-agent isolation levels. The ABM simulates SEIRM disease transmission. At equilibrium, the population's disease state distribution (per-subgroup infection rates, recovery rates) is a function of beta. We measure:
  - Per-subgroup disease burden at convergence (burden_age0..5)
  - Per-subgroup model recommendations at convergence (pred_age0..5)
  - The gap between what the model recommends and what the epidemic data says is needed — this gap persists at high beta (model anchored to pretrained prior) and closes at low beta (model adapts)
  - Total infections at equilibrium across betas

- **Macro economics (Queens County, 2.7M agents):** The LM advises work/consumption decisions. The ABM simulates earning, consumption, labor markets, and inflation. At equilibrium, the population's economic state (unemployment rate, asset distribution, per-subgroup income) is a function of beta.

**Three reference lines for the paper's main figure:**
1. The equilibrium under the optimal performative policy (found via gradient descent through the differentiable ABM)
2. The equilibrium under the frozen pretrained model (the reference prior's implied trajectory)
3. The equilibrium under KL-regularized retraining at each beta (the actual performative loop)

Line 3 should approach line 1 as beta→0 (model adapts) and approach line 2 as beta→∞ (model locked to prior). The shape of this curve — and how it differs across subgroups — is the main result.

**Methodological contribution:** Because AgentTorch is differentiable, we can compute exact performative gradients through the ABM (validated: autograd/FD ratio ~1.0, CV ~0.02). This enables gradient-based calibration to real data, performative gradient computation without PerfGrad's assumptions, and potentially gradient-based equilibrium finding.

**Calibrated ABM parameters (COVID):**

| Season | Dates | Real cases (3wk) | Seed frac | R2 | Fit ratio |
|--------|-------|-----------------|-----------|-----|-----------|
| Alpha | Dec 2020 | 353 | 0.005 | 0.60 | 1.000 |
| Delta | Aug 2021 | 184 | 0.001 | 1.13 | 0.995 |
| Omicron | Dec 2021 | 3,317 | 0.05 | 1.35 | 1.001 |

**Calibrated ABM parameters (Macro):** UAC fit to Queens County monthly unemployment, 2019-2023. Full 2.7M agent population. Converged across all 4 economic periods.

**Key prior results:**
- 7B LM shifts the population in AT covid (demonstrated empirically)
- FJ experiments show KL coefficient affects where population equilibrates
- AT autodiff produces accurate performative gradients (ratio ~1.0, CV ~0.02)
- 0.5B model does not produce usable results; 7B required

**Equilibrium concept — performative stability, not population steady state:**

The COVID ABM is transient (SEIRM with no reinfection — the epidemic ends). Unlike FJ, there is no population-state equilibrium. The right concept is **performative stability** (Perdomo et al. 2020): LM parameters θ* such that deploying θ* into the ABM and retraining on the resulting data returns θ*. When `stability_gap → 0`, the model has found a performative stable point. Different betas produce different stable points with different population outcomes.

**Learned surrogate as an analytical tool (future work):**

The ABM's performative map — policy → population response — can be learned as a differentiable surrogate: 6 per-subgroup isolation recommendations → 6 per-subgroup infection rates. This surrogate is an analyzable object:
- **Jacobian:** ∂(infections_group_i) / ∂(isolation_group_j) reveals cross-group coupling from the contact network
- **Fixed points:** Newton's method on the 6→6 surrogate instead of running the full ABM loop
- **Stability:** eigenvalues at fixed points determine convergence of the performative loop
- **Beta sensitivity:** trace how the stable point moves continuously as KL strength changes

The pretrained LM has its own implicit mapping (beliefs about demographics and risk from pretraining). The surrogate learns the actual mapping. The discrepancy between these two objects is what the KL anchor preserves — measurable as the gap between pretrained predictions and surrogate-optimal policy.

**Dreamer-style performative optimization (future work):**

With the surrogate, you can dream the full performative loop: deploy policy → surrogate predicts outcome → simulate KL-SFT retraining → get next policy → repeat — all inside the learned model. Optimizing through this dreamed loop finds the performatively optimal policy at each beta, showing exactly what the KL anchor costs in population outcomes. The key: you're optimizing inside a learned world model while being regularized toward a pretrained world model — two competing models of the world.

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
    agenttorch.py               # optional adapter; see adapters/README.md
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

The `[agenttorch]` extra wires perfsim to an `agent_torch.Runner` as a
concrete `AgentBased` environment. See `perfsim/adapters/README.md` for the
Algorithm 1 contract, the three required user callables, and the A1 / A2 / B
signal-mutation pattern check. Install with `pip install -e ".[agenttorch]"`.

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
