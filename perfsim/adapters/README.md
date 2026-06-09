# `perfsim.adapters`

Run an external agent-based simulation as a perfsim environment. Only loaded when the matching extra is installed; core perfsim doesn't depend on these.

- `AgentTorchEnvironment` wraps an [AgentTorch](https://github.com/AgentTorch/AgentTorch) `Runner`. Install: `pip install 'perfsim[agenttorch]'`.

## How it works

Once per round the adapter does five things: read features out of the sim, run your model on them, write the model's predictions back into the sim, step the sim forward `n_steps`, then read out the training data.

```
read features -> model(features) -> write predictions into the sim
              -> sim.step(n_steps) -> read back {x, y, agent_idx}
```

The model is queried once and its output is held fixed while the sim runs. So you give the adapter four small functions that know your sim's layout:

| Argument | What it does |
|---|---|
| `runner_factory(seed)` | builds a fresh, seeded Runner (also called on every `reset`) |
| `feature_provider(runner)` | returns the `(N, F)` features to feed the model |
| `signal_writer(runner, preds)` | writes the predictions into the sim's state |
| `state_extractor(runner)` | reads the sim after stepping, returns `{x, y, agent_idx}` |

Plus `signal_path` (where in the state the predictions go, used by the safety check below). Optional: `strict_signal`, `keep_trajectory`, `init_seed`.

```python
env = AgentTorchEnvironment(
    runner_factory   = lambda seed: build_my_runner(seed),
    feature_provider = lambda r: r.state["agents"]["citizen"]["features"],
    signal_writer    = lambda r, p: r.state["agents"]["citizen"].__setitem__("platform_signal", p),
    state_extractor  = lambda r: {
        "x": r.state["agents"]["citizen"]["features"],
        "y": r.state["agents"]["citizen"]["opinion"],
        "agent_idx": torch.arange(len(r.state["agents"]["citizen"]["opinion"])),
    },
    signal_path = ("agents", "citizen", "platform_signal"),
)
sim = Simulator(env=env, learner=ERMLearner(model, loss), loss=loss)
hist = sim.run(n_rounds=10, epoch_size=20, seed=0)
```

Your AgentTorch config needs a substep that reads `signal_path`; perfsim doesn't ship one.

## Things to know

- **The prediction must stay fixed during the inner steps.** A substep may read it, but must not overwrite it. The adapter checks this (it compares `signal_path` before and after stepping) and errors if a substep changed it. Pass `strict_signal=False` to allow it (e.g. a decaying-signal model).
- **Gradients.** `grad_run` is `run` without detaching, so you can get a gradient through one epoch (if the sim's substeps are differentiable). The model is frozen across rounds, so this is a one-shot measurement, not differentiation through the whole loop. There's no cheap peek, so `sample` isn't supported.
- **Tests.** `tests/test_agenttorch_adapter.py` uses a fake Runner, so it runs without AgentTorch installed.
