# at_schelling

AT-driven Schelling segregation scenario for perfsim. Built on `agent_torch.core.Runner` with a real YAML config, registry, and `SubstepObservation` / `SubstepAction` / `SubstepTransition` classes.

## Public API

```python
from perfsim.scenarios.at_schelling import make_schelling_env

env = make_schelling_env(
    num_agents=50,
    grid_height=10,
    grid_width=10,
    n_types=4,
    baseline_threshold=0.4,
    lambda_=0.15,
    neighborhood_radius=1,
    move_hardness=8.0,
)
```

`env` is an `AgentTorchEnvironment` (perfsim adapter) wrapping `agent_torch.core.Runner`. Use `env.run(model, n_steps=K)` for the Algorithm-1 loop.

`BinaryLMScorer` is exported lazily via `__getattr__` so that the at_schelling import does not pull `transformers` at module import time.

## Substep order (per round)

1. `compute_neighborhood` -- per-agent same/opp/empty fractions over the Moore neighborhood.
2. `happiness_predict_action` + `write_p_pred` -- read `platform_signal`, expose as `p_pred`.
3. `move_decision` -- `H_i = clip(H_0 - lambda*(p_i - 0.5), 0, 1)`; `move_i = StraightThroughBernoulli(sigmoid(hardness * (H_i - s_i)))`.
4. `execute_moves` -- relocate movers to random empty cells; update `coordinates`, `grid_occupancy`, `grid_type`.
5. `compute_realized_happiness` -- recompute `same_frac` post-move; `y_i = 1[s_new_i >= H_0]` (BASELINE, not LM-modulated).

## Files

- `config.yaml` -- full AT config (simulation_metadata + state + network + substeps).
- `env.py` -- `make_schelling_env` + `build_schelling_runner` + registry + post-init placement.
- `data.py` -- ACS-shaped 4-type demographics + random initial placement.
- `default_callables.py` -- feature_provider, signal_writer, state_extractor.
- `model_scoring.py` -- `BinaryLMScorer` over HAPPY/UNHAPPY token logits.
- `substeps/` -- five `SubstepTransition` (+ one `SubstepAction`) classes.
- `_lightweight_reference/` -- the previous duck-typed runner. Kept for substep-math reference.
