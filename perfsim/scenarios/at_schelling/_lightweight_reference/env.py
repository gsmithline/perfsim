"""Build a perfsim environment for the Schelling segregation scenario.

Public entry point: `make_schelling_env(...) -> AgentTorchEnvironment`.

Design choice (recorded for honesty): we DO use the existing
`AgentTorchEnvironment` adapter from `perfsim.adapters.agenttorch`, but
we pass it a `SchellingRunner` (a lightweight runner with AT-shaped
state) instead of a real `agent_torch.Runner`. The adapter only
duck-types against `.state`, `.step(num_steps=K)`, and
`.reset_state_before_episode()`, so this works without modification.

This buys us:
  - The perfsim-side strict_signal contract check (signal field
    cannot mutate during runner.step) from the AT adapter.
  - The single-model-query-per-epoch Algorithm 1 pattern, identical
    to at_covid.
  - Identical feature_provider / signal_writer / state_extractor API
    surface, so scripts written for at_covid transfer with cosmetic
    edits.

Without dragging in:
  - YAML loading + OmegaConf resolver registration.
  - Registry registration for substeps / networks / initializers.
  - A bundled population that may not match the demographics we want.

If we later want to switch to a real `agent_torch.Runner`, the substep
functions in `perfsim.scenarios.at_schelling.substeps` are pure
state-mutating callables that can be wrapped as
SubstepObservation/Policy/Transition classes with mechanical
rewrites of the input/output_variables plumbing.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import Tensor

from perfsim.adapters.agenttorch import AgentTorchEnvironment
from perfsim.scenarios.at_schelling.config import build_schelling_config
from perfsim.scenarios.at_schelling.data import (
    initial_placement,
    load_nyc_demographics,
)
from perfsim.scenarios.at_schelling.default_callables import (
    default_feature_provider,
    default_signal_writer,
    default_signal_writer_grad,
    default_state_extractor,
)
from perfsim.scenarios.at_schelling.runner import SchellingRunner
from perfsim.scenarios.at_schelling.substeps import (
    compute_neighborhood,
    compute_realized_happiness,
    execute_moves,
    happiness_predict_action,
    move_decision,
)


def build_schelling_substeps() -> list[Callable]:
    """Substep order: neighborhood -> read p_pred -> decide -> move -> realize."""
    return [
        compute_neighborhood,
        happiness_predict_action,
        move_decision,
        execute_moves,
        compute_realized_happiness,
    ]


def _state_initializer_factory(
    *,
    demographics: dict[int, int] | None,
    proportions: tuple[float, ...] | None,
    seed: int,
) -> Callable[[dict], dict]:
    """Returns a state_initializer(config) closure.

    Pulled out so make_schelling_env can capture user-supplied
    demographics / proportions / seed at construction time.
    """

    def _init(config: dict) -> dict:
        meta = config["simulation_metadata"]
        n = int(meta["num_agents"])
        H = int(meta["grid_height"])
        W = int(meta["grid_width"])
        device = torch.device(meta["device"])

        demos = demographics or load_nyc_demographics(
            n_agents=n, proportions=proportions
        )
        coords, types_, grid_occ, grid_type = initial_placement(
            n_agents=n,
            grid_height=H,
            grid_width=W,
            demographics=demos,
            seed=seed,
        )
        coords = coords.to(device)
        types_ = types_.to(device)
        grid_occ = grid_occ.to(device)
        grid_type = grid_type.to(device)

        # All per-agent property tensors must exist before substeps run
        # so that the substeps can write into them by key without
        # KeyError. We initialize with sensible neutral values.
        residents: dict[str, Any] = {
            "coordinates": coords,
            "type": types_,
            "same_frac": torch.full((n,), 0.5, device=device),
            "opp_frac": torch.full((n,), 0.5, device=device),
            "empty_frac": torch.zeros((n,), device=device),
            "p_pred": torch.full((n,), 0.5, device=device),
            "effective_threshold": torch.full(
                (n,), float(meta["baseline_threshold"]), device=device
            ),
            "move_decision": torch.zeros((n,), device=device),
            "realized_happiness": torch.zeros((n,), device=device),
            # `previous_state` starts neutral; round 1's prompt sees 0.5
            # which we render as "unknown" or "unhappy" depending on
            # the prompt builder.
            "previous_state": torch.full((n,), 0.5, device=device),
            "platform_signal": torch.full((n,), 0.5, device=device),
        }

        state: dict[str, Any] = {
            "current_step": 0,
            "current_substep": "0",
            "agents": {"residents": residents},
            "environment": {
                "grid_occupancy": grid_occ,
                "grid_type": grid_type,
            },
        }
        return state

    return _init


def build_schelling_runner(
    seed: int = 0,
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
    demographics: dict[int, int] | None = None,
    proportions: tuple[float, ...] | None = None,
) -> SchellingRunner:
    """Construct and initialize a SchellingRunner."""
    torch.manual_seed(int(seed))
    config = build_schelling_config(
        num_agents=num_agents,
        grid_height=grid_height,
        grid_width=grid_width,
        n_types=n_types,
        baseline_threshold=baseline_threshold,
        lambda_=lambda_,
        neighborhood_radius=neighborhood_radius,
        num_steps_per_episode=num_steps_per_episode,
        device=device,
        move_hardness=move_hardness,
        seed=seed,
    )
    init = _state_initializer_factory(
        demographics=demographics,
        proportions=proportions,
        seed=seed,
    )
    runner = SchellingRunner(
        config=config,
        substeps=build_schelling_substeps(),
        state_initializer=init,
    )
    runner.init()
    return runner


def make_schelling_env(
    *,
    init_seed: int = 0,
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
    demographics: dict[int, int] | None = None,
    proportions: tuple[float, ...] | None = None,
    feature_provider: Optional[Callable[[Any], Tensor]] = None,
    signal_writer: Optional[Callable[[Any, Tensor], None]] = None,
    state_extractor: Optional[Callable[[Any], dict[str, Tensor]]] = None,
    keep_trajectory: bool = False,
    strict_signal: bool = True,
) -> AgentTorchEnvironment:
    """Construct an AgentTorchEnvironment driving the Schelling sim.

    All Schelling knobs (grid, threshold, lambda, demographics) flow
    through to `build_schelling_runner` via a factory closure that
    captures them at make-time and accepts only a seed at reset time
    (matching the perfsim `RunnerFactory` signature).
    """

    def _factory(seed: int):
        return build_schelling_runner(
            seed,
            num_agents=num_agents,
            grid_height=grid_height,
            grid_width=grid_width,
            n_types=n_types,
            baseline_threshold=baseline_threshold,
            lambda_=lambda_,
            neighborhood_radius=neighborhood_radius,
            num_steps_per_episode=num_steps_per_episode,
            device=device,
            move_hardness=move_hardness,
            demographics=demographics,
            proportions=proportions,
        )

    return AgentTorchEnvironment(
        runner_factory=_factory,
        feature_provider=feature_provider or default_feature_provider,
        signal_writer=signal_writer or default_signal_writer,
        state_extractor=state_extractor or default_state_extractor,
        signal_path=("agents", "residents", "platform_signal"),
        keep_trajectory=keep_trajectory,
        strict_signal=strict_signal,
        init_seed=init_seed,
    )
