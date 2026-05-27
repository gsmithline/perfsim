# `perfsim.adapters`

Optional adapters that bridge perfsim's `Environment` contract to external
ABM frameworks

- `agenttorch.AgentTorchEnvironment`: wraps an `agent_torch.Runner` as a
  perfsim `AgentBased` environment.

The adapter is loaded only when its extras are installed. Core perfsim has
no runtime dependency on AgentTorch.

---

## `AgentTorchEnvironment`

The first concrete `AgentBased` environment in perfsim. Drives an
[AgentTorch](https://github.com/AgentTorch/AgentTorch) simulation under
perfsim's epoch loop.

### Install

```bash
pip install 'perfsim[agenttorch]'
```

Pins `agent-torch>=0.6,<0.7`. Importing `perfsim.adapters.agenttorch`
without the extra raises a clear `ImportError` with the install hint.

### Algorithm 1 contract

```
env.run(model, n_steps):
    if not keep_trajectory: runner.reset_state_before_episode()
    X     = feature_provider(runner)        # (N, F)
    preds = model(X)                        # queried ONCE
    signal_writer(runner, preds)            # mutates runner.state
    runner.step(num_steps=n_steps)          # AT advances internally
    return state_extractor(runner)          # Data dict incl. agent_idx
```

This matches Algorithm 1 of Wu, Abebe, Mendler-Dünner 2026
([arxiv 2603.12137](https://arxiv.org/abs/2603.12137)): the deployed
predictor is queried exactly once per epoch, its output is installed as a
fixed input to the inner AT loop, and the AT runner advances autonomously
for `n_steps` time steps without re-querying the model.

### Constructor

```python
from perfsim.adapters.agenttorch import AgentTorchEnvironment

env = AgentTorchEnvironment(
    runner_factory   = lambda seed: build_my_at_runner(seed),
    feature_provider = lambda r: r.state["agents"]["citizen"]["features"],
    signal_writer    = lambda r, p: r.state["agents"]["citizen"].__setitem__(
        "platform_signal", p
    ),
    state_extractor  = lambda r: {
        "x": r.state["agents"]["citizen"]["features"],
        "y": r.state["agents"]["citizen"]["opinion"],
        "agent_idx": torch.arange(
            r.state["agents"]["citizen"]["opinion"].shape[0]
        ),
    },
    signal_path = ("agents", "citizen", "platform_signal"),
)
```

Four required callables / paths:

| Arg | Type | What it does |
|---|---|---|
| `runner_factory` | `Callable[[int], agent_torch.Runner]` | Returns a fresh seeded Runner. Called at construction and on every `reset(seed)`. Must call `torch.manual_seed(seed)` itself if substeps draw from torch RNG. |
| `feature_provider` | `Callable[[Runner], Tensor]` | Returns the (N, F) feature matrix fed to perfsim's model. Reads from `runner.state[...]`. |
| `signal_writer` | `Callable[[Runner, Tensor], None]` | Writes the model's per-agent predictions into `runner.state[...]`. perfsim does not assume any particular shape; the writer is responsible for matching the AT sim's expected layout. |
| `state_extractor` | `Callable[[Runner], Data]` | Reads `runner.state` after the inner loop and returns the supervised `Data` dict (`x`, `y`, `agent_idx`) that perfsim's Learner trains on. |
| `signal_path` | `tuple[str, ...]` | Path of dict keys into `runner.state` that the `signal_writer` writes to. The adapter uses this to verify the field was not mutated during the inner loop (see "Pattern contract"). |

Optional keyword args:

- `produces_schema: DataSchema = SUPERVISED_SCHEMA`: declared schema for binding.
- `max_meaningful_epoch_size: int | float = inf`: per-instance cap on `epoch_size`.
- `keep_trajectory: bool = False`: if True, do not truncate `runner.state_trajectory` between epochs. Default truncates to avoid unbounded memory growth.
- `strict_signal: bool = True`: if True, raise `SignalMutationError` when the signal field changes during `runner.step(...)`. See pattern contract below.
- `init_seed: int = 0`: seed passed to `runner_factory` at construction.

### Pattern contract (A1 / A2 / B)

The deployed signal is written once at the top of `env.run`. Three patterns
describe what the AT sim does with it during the inner loop:

| Pattern | Inside the inner loop | Allowed? |
|---|---|---|
| A1 | signal used at step 0 to set state, then dropped | Allowed, atypical |
| **A2** | **signal held constant, read every substep as anchor** | **Default; matches Friedkin-Johnsen** |
| B | signal field overwritten by some substep transition | Forbidden |

Pattern B breaks Algorithm 1: the recorded `theta_t` is no longer a faithful
witness of what drove the epoch, because the signal in `runner.state` at
step 5 is not what perfsim deployed at step 0. The adapter detects B by
snapshotting `runner.state[signal_path]` before `runner.step(...)` and
asserting `torch.allclose` after. A1 and A2 both pass; only B fails.

To opt out (e.g., if you specifically want a decaying-platform model), pass
`strict_signal=False`. perfsim's hot path still records `theta_t`, but
interpretation of that recorded theta is on you.

### What does and does not differentiate

The adapter satisfies `Differentiable` but **not** `FullyDifferentiable`.

- **`grad_run(model, n_steps)`** is the canonical entry point for
  gradient measurement. Same shape as `run` but without `torch.no_grad`
  around `model(X)` and without an explicit `.detach()` on preds. The
  user-supplied `signal_writer` must NOT detach either; the at_covid
  scenario's default uses `.clone()` without `.detach()` for this reason.
- `grad_step(model)` is `grad_run(model, n_steps=1)`.
- Whether gradient actually flows through `runner.step(...)` depends on
  the AT sim's substep code. Covid is differentiable through
  `StraightThroughBernoulli` plus `update_stages`'s linear update; see
  `perfsim/scenarios/at_covid/README.md` for the four conditions needed
  for non-zero gradient.
- Across the full epoch loop, theta is frozen by the Simulator's design
  (see `DESIGN.md` §8). `grad_run` is for one-shot gradient measurement,
  not for end-to-end differentiable PP rollouts where theta varies
  inside the inner loop (that is a v2 question).

`sample(model)` and `grad_sample(model)` raise `NotImplementedError` in v1.
AT runners do not expose a free peek primitive, and the Simulator hot path
uses `run`, not `sample`.

### Authoring AT sims for use with perfsim

Two requirements for the AT-side code:

1. The AT sim's substep that consumes the platform signal must read the
   value at the `signal_path` you pass to the adapter. perfsim does not
   know which substep this is; the contract is purely "write here, you
   read here."
2. No substep transition may overwrite the signal field. Read it freely,
   condition on it freely, use it in computations. Write back to other
   state fields (e.g., `opinion`) but not to `signal_path`. The runtime
   check enforces this.

If your sim genuinely needs a platform field that evolves under the inner
loop (e.g., explicit platform-learning dynamics), pass `strict_signal=False`
and accept that perfsim's `theta_t` records the deployed predictor, not
the signal that drove the epoch end-to-end.

### Example skeleton

```python
import torch
from agent_torch.core.config import build_config
from agent_torch.core.registry import Registry
from agent_torch.core.runner import Runner
from perfsim.adapters.agenttorch import AgentTorchEnvironment
from perfsim.simulator import Simulator
from perfsim.models.linear import LinearModel
from perfsim.learners.erm import ERMLearner
from perfsim.losses import MSELoss


def runner_factory(seed: int) -> Runner:
    torch.manual_seed(seed)
    config   = build_config("my_at_config.yaml")
    registry = Registry()
    runner = Runner(config, registry)
    runner.init()
    return runner


env = AgentTorchEnvironment(
    runner_factory   = runner_factory,
    feature_provider = lambda r: r.state["agents"]["citizen"]["features"],
    signal_writer    = lambda r, p: r.state["agents"]["citizen"].__setitem__(
        "platform_signal", p.squeeze(-1) if p.ndim > 1 else p
    ),
    state_extractor  = lambda r: {
        "x": r.state["agents"]["citizen"]["features"],
        "y": r.state["agents"]["citizen"]["opinion"].unsqueeze(-1),
        "agent_idx": torch.arange(
            r.state["agents"]["citizen"]["opinion"].shape[0]
        ),
    },
    signal_path = ("agents", "citizen", "platform_signal"),
)

model = LinearModel(in_features=4, out_features=1)
loss = MSELoss()
sim = Simulator(env=env, learner=ERMLearner(model=model, loss=loss), loss=loss)
hist = sim.run(n_rounds=10, epoch_size=20, seed=0)
```

The AT-side `my_at_config.yaml` must define a substep whose policy or
transition reads `state["agents"]["citizen"]["platform_signal"]`. perfsim
does not ship a config; the AT sim author writes it.

### Testing

Adapter tests live in `tests/test_agenttorch_adapter.py`. They are gated by
`pytest.importorskip("agent_torch")`, so the core perfsim suite remains
AT-free. The tests use a `FakeRunner` stub that duck-types the AT Runner
interface (`.state`, `.step(num_steps=...)`, `.reset_state_before_episode()`)
so a full AT YAML config is not required to exercise the adapter contract.

### See also

- `DESIGN.md` §20 for the full design rationale, including the four
  verification items resolved against `agent-torch==0.6.0`.
- `perfsim/core/environment.py` for the `AgentBased` ABC and the
  `Differentiable` / `FullyDifferentiable` capability Protocols.
- `arxiv 2603.12137` (Wu, Abebe, Mendler-Dünner 2026) Algorithm 1.
