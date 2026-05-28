"""Build an AgentTorchEnvironment driving the AT Schelling model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from agent_torch.core import Registry, Runner
from agent_torch.core.helpers import grid_network, read_config

from perfsim.adapters.agenttorch import AgentTorchEnvironment
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
from perfsim.scenarios.at_schelling.substeps import (
    ComputeNeighborhood,
    ComputeRealizedHappiness,
    ExecuteMoves,
    HappinessPredictAction,
    MoveDecision,
    WritePPred,
)


CONFIG_YAML = str(Path(__file__).parent / "config.yaml")

_OMEGACONF_RESOLVERS_REGISTERED = False


def _register_resolvers_once() -> bool:
    global _OMEGACONF_RESOLVERS_REGISTERED
    if not _OMEGACONF_RESOLVERS_REGISTERED:
        _OMEGACONF_RESOLVERS_REGISTERED = True
        return True
    return False


def _build_registry() -> Registry:
    # AT's Initializer looks up substep classes by the YAML dict key
    # (e.g. `compute_neighborhood:` under `transition:`), NOT by the
    # `generator:` field underneath that key. Keep names in sync with
    # config.yaml.
    reg = Registry()
    reg.register(ComputeNeighborhood, "compute_neighborhood", key="transition")
    reg.register(HappinessPredictAction, "happiness_predict_action", key="policy")
    reg.register(WritePPred, "write_p_pred", key="transition")
    reg.register(MoveDecision, "move_decision", key="transition")
    reg.register(ExecuteMoves, "execute_moves", key="transition")
    reg.register(ComputeRealizedHappiness, "compute_realized_happiness", key="transition")
    reg.register(grid_network, "grid", key="network")
    return reg


def _apply_initial_placement(
    runner: Runner,
    *,
    n_agents: int,
    grid_height: int,
    grid_width: int,
    demographics: dict[int, int] | None,
    proportions: tuple[float, ...] | None,
    seed: int,
) -> None:
    """Overwrite zero-initialized placement tensors with random valid placement."""
    demos = demographics or load_nyc_demographics(
        n_agents=n_agents, proportions=proportions
    )
    coords, types_, grid_occ, grid_type = initial_placement(
        n_agents=n_agents,
        grid_height=grid_height,
        grid_width=grid_width,
        demographics=demos,
        seed=seed,
    )
    device = torch.device(runner.config["simulation_metadata"]["device"])
    residents = runner.state["agents"]["residents"]
    residents["coordinates"] = coords.to(device)
    residents["type"] = types_.to(device)
    runner.state["environment"]["grid_occupancy"] = grid_occ.to(device)
    runner.state["environment"]["grid_type"] = grid_type.to(device)


def _patched_config(overrides: dict[str, Any]) -> dict[str, Any]:
    """Load YAML and patch simulation_metadata. Caller owns the resulting dict."""
    cfg = read_config(CONFIG_YAML, register_resolvers=_register_resolvers_once())
    meta = cfg["simulation_metadata"]
    meta.update(overrides)
    n = int(meta["num_agents"])
    H = int(meta["grid_height"])
    W = int(meta["grid_width"])
    cfg["state"]["agents"]["residents"]["number"] = n
    # Property shapes interpolate ${state.agents.residents.number} at YAML load,
    # so they freeze to the default before this override propagates. Rewrite
    # them post-load so they match the patched num_agents.
    props = cfg["state"]["agents"]["residents"]["properties"]
    for name, prop in props.items():
        prop["shape"] = [n, 2] if name == "coordinates" else [n]
    for grid_name in ("grid_occupancy", "grid_type"):
        cfg["state"]["environment"][grid_name]["shape"] = [H, W]
    return cfg


def build_schelling_runner(
    seed: int = 0,
    *,
    num_agents: int = 50,
    grid_height: int = 10,
    grid_width: int = 10,
    n_types: int = 4,
    baseline_threshold: float = 0.4,
    baseline_threshold_per_type: list[float] | tuple[float, ...] | None = None,
    lambda_: float = 0.15,
    neighborhood_radius: int = 1,
    num_steps_per_episode: int = 5,
    device: str = "cpu",
    move_hardness: float = 8.0,
    demographics: dict[int, int] | None = None,
    proportions: tuple[float, ...] | None = None,
) -> Runner:
    torch.manual_seed(int(seed))
    overrides = {
        "num_agents": int(num_agents),
        "grid_height": int(grid_height),
        "grid_width": int(grid_width),
        "n_types": int(n_types),
        "baseline_threshold": float(baseline_threshold),
        "lambda_": float(lambda_),
        "neighborhood_radius": int(neighborhood_radius),
        "num_steps_per_episode": int(num_steps_per_episode),
        "device": str(device),
        "move_hardness": float(move_hardness),
        "seed": int(seed),
    }
    if baseline_threshold_per_type is not None:
        overrides["baseline_threshold_per_type"] = list(baseline_threshold_per_type)
    cfg = _patched_config(overrides)
    reg = _build_registry()
    runner = Runner(cfg, reg)
    runner.init()
    _apply_initial_placement(
        runner,
        n_agents=num_agents,
        grid_height=grid_height,
        grid_width=grid_width,
        demographics=demographics,
        proportions=proportions,
        seed=seed,
    )
    return runner


def make_schelling_env(
    *,
    init_seed: int = 0,
    num_agents: int = 50,
    grid_height: int = 10,
    grid_width: int = 10,
    n_types: int = 4,
    baseline_threshold: float = 0.4,
    baseline_threshold_per_type: list[float] | tuple[float, ...] | None = None,
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
    """Construct an AgentTorchEnvironment driving the AT Schelling model."""

    def _factory(seed: int) -> Runner:
        return build_schelling_runner(
            seed,
            num_agents=num_agents,
            grid_height=grid_height,
            grid_width=grid_width,
            n_types=n_types,
            baseline_threshold=baseline_threshold,
            baseline_threshold_per_type=baseline_threshold_per_type,
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
