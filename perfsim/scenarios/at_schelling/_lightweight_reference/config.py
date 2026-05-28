"""Programmatic config builder for the at_schelling scenario.

Mirrors the structure of AT YAMLs (simulation_metadata, state, substeps)
but as a plain Python dict. The `SchellingRunner` consumes it directly;
there's no YAML or OmegaConf in the loop.

Key knobs (all overridable via kwargs):
  num_agents             # population size
  grid_height, grid_width  # cell grid dimensions
  n_types                # number of demographic types
  baseline_threshold     # H_0, Schelling happiness threshold
  lambda_                # LM modulation strength (kwarg name; stored as 'lambda')
  neighborhood_radius    # 1 = 8-neighbor Moore
  num_steps_per_episode  # how many Schelling rounds per env.run call
  device                 # torch device string
  move_hardness          # sigmoid sharpness in move_decision (8.0 is plenty for
                         # binary forward agreement up to 1e-3 in [0, 1])
"""

from __future__ import annotations

from typing import Any


def build_schelling_config(
    *,
    num_agents: int = 200,
    grid_height: int = 20,
    grid_width: int = 20,
    n_types: int = 4,
    baseline_threshold: float = 0.4,
    lambda_: float = 0.15,
    neighborhood_radius: int = 1,
    num_steps_per_episode: int = 5,
    device: str = "cpu",
    move_hardness: float = 8.0,
    seed: int = 0,
) -> dict[str, Any]:
    if grid_height * grid_width < num_agents:
        raise ValueError(
            f"grid {grid_height}x{grid_width} too small for {num_agents} agents"
        )
    if n_types < 2:
        raise ValueError(f"n_types must be >= 2; got {n_types}")
    if not 0.0 <= baseline_threshold <= 1.0:
        raise ValueError(
            f"baseline_threshold must be in [0, 1]; got {baseline_threshold}"
        )

    return {
        "simulation_metadata": {
            "num_agents": int(num_agents),
            "grid_height": int(grid_height),
            "grid_width": int(grid_width),
            "n_types": int(n_types),
            "baseline_threshold": float(baseline_threshold),
            # Stored under the dunder-free key so substep code can read
            # `meta["lambda"]` without Python keyword issues.
            "lambda": float(lambda_),
            "neighborhood_radius": int(neighborhood_radius),
            "num_steps_per_episode": int(num_steps_per_episode),
            "device": str(device),
            "move_hardness": float(move_hardness),
            "seed": int(seed),
        },
        # Mirrors AT's `state.agents.<group>.properties` layout. Initial
        # tensors are filled in by the state_initializer, not the config.
        "state": {
            "agents": {
                "residents": {
                    "number": int(num_agents),
                    "properties": [
                        "coordinates",
                        "type",
                        "same_frac",
                        "opp_frac",
                        "empty_frac",
                        "p_pred",
                        "effective_threshold",
                        "move_decision",
                        "realized_happiness",
                        "previous_state",
                        "platform_signal",
                    ],
                }
            },
            "environment": {
                "grid_occupancy_shape": [int(grid_height), int(grid_width)],
                "grid_type_shape": [int(grid_height), int(grid_width)],
            },
        },
        # Substep names are documentary here; the actual callables are
        # passed to `SchellingRunner` directly from env.py. We keep the
        # names in config so a future YAML loader could wire them up via
        # a name->callable registry, mirroring AT's pattern.
        "substeps": {
            "0": {"name": "compute_neighborhood"},
            "1": {"name": "happiness_predict_action"},
            "2": {"name": "move_decision"},
            "3": {"name": "execute_moves"},
            "4": {"name": "compute_realized_happiness"},
        },
    }
