# `perfsim.adapters`

Bridges perfsim's `Environment` contract to external ABM frameworks. Loaded
only when extras are installed; core perfsim has no runtime dependency on them.

- `agenttorch.AgentTorchEnvironment`: wraps an `agent_torch.Runner` as a perfsim
  `AgentBased` environment.

## `AgentTorchEnvironment`

Drives an [AgentTorch](https://github.com/AgentTorch/AgentTorch) sim under
perfsim's epoch loop. Install: `pip install 'perfsim[agenttorch]'` (pins
`agent-torch>=0.6,<0.7`); importing without the extra raises a clear ImportError.

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

Matches Algorithm 1 of Wu, Abebe, Mendler-Dünner 2026
([arxiv 2603.12137](https://arxiv.org/abs/2603.12137)): the predictor is queried
once per epoch, its output is a fixed input to the inner loop, and the runner
advances `n_steps` without re-querying.

### Constructor

Four required callables / paths:

| Arg | Type | What it does |
|---|---|---|
| `runner_factory` | `Callable[[int], Runner]` | Returns a fresh seeded Runner (seeds torch RNG itself). Called at construction and every `reset(seed)`. |
| `feature_provider` | `Callable[[Runner], Tensor]` | Returns the (N, F) feature matrix fed to the model. |
| `signal_writer` | `Callable[[Runner, Tensor], None]` | Writes per-agent predictions into `runner.state`; responsible for matching the AT sim's layout. |
| `state_extractor` | `Callable[[Runner], Data]` | Reads `runner.state` after the loop, returns the supervised `Data` dict (`x`, `y`, `agent_idx`). |
| `signal_path` | `tuple[str, ...]` | Key path into `runner.state` the writer targets; used for the mutation check below. |

Optional: `produces_schema` (default supervised), `max_meaningful_epoch_size`,
`keep_trajectory` (default False truncates the trajectory), `strict_signal`
(default True), `init_seed`.

```python
env = AgentTorchEnvironment(
    runner_factory   = lambda seed: build_my_at_runner(seed),
    feature_provider = lambda r: r.state["agents"]["citizen"]["features"],
    signal_writer    = lambda r, p: r.state["agents"]["citizen"].__setitem__("platform_signal", p),
    state_extractor  = lambda r: {
        "x": r.state["agents"]["citizen"]["features"],
        "y": r.state["agents"]["citizen"]["opinion"],
        "agent_idx": torch.arange(r.state["agents"]["citizen"]["opinion"].shape[0]),
    },
    signal_path = ("agents", "citizen", "platform_signal"),
)
sim = Simulator(env=env, learner=ERMLearner(model, loss), loss=loss)
hist = sim.run(n_rounds=10, epoch_size=20, seed=0)
```

The AT config must define a substep that reads the `signal_path` field. perfsim
ships no config; the AT author writes it.

### Signal pattern (A1 / A2 / B)

The signal is written once at the top of `env.run`. What the inner loop does
with it:

| Pattern | Inside the inner loop | Allowed? |
|---|---|---|
| A1 | used at step 0, then dropped | allowed, atypical |
| **A2** | **held constant, read every substep as anchor** | **default; matches FJ** |
| B | overwritten by some substep | forbidden |

B breaks Algorithm 1 (the recorded `theta_t` no longer witnesses the epoch). The
adapter snapshots `signal_path` before `runner.step` and asserts `allclose`
after; A1/A2 pass, B fails. Pass `strict_signal=False` to opt out (e.g. a
decaying-platform model) — then `theta_t` records the deployed predictor, not the
signal that drove the epoch.

### Differentiability

Satisfies `Differentiable`, not `FullyDifferentiable`.

- `grad_run(model, n_steps)`: like `run` but without `no_grad`/`.detach()` around
  `model(X)`; the `signal_writer` must not detach either (at_covid's default uses
  `.clone()`, not `.detach()`). `grad_step` is `grad_run(., 1)`.
- Whether grad flows through `runner.step` depends on the AT substeps (covid:
  yes, via `StraightThroughBernoulli`). theta is frozen across the epoch loop, so
  `grad_run` is one-shot measurement, not end-to-end rollout differentiation.
- `sample` / `grad_sample` raise `NotImplementedError` (no peek primitive).

### Testing

`tests/test_agenttorch_adapter.py`, gated by `importorskip("agent_torch")`. Uses
a `FakeRunner` stub duck-typing the Runner interface, so no AT YAML is needed.
