"""Default feature_provider / signal_writer / state_extractor for Schelling."""

from __future__ import annotations

import torch
from torch import Tensor


def default_feature_provider(runner) -> Tensor:
    """Stack (type, same_frac, opp_frac, empty_frac, previous_state) per agent."""
    r = runner.state["agents"]["residents"]
    return torch.stack(
        [
            r["type"].float(),
            r["same_frac"].float(),
            r["opp_frac"].float(),
            r["empty_frac"].float(),
            r["previous_state"].float(),
        ],
        dim=-1,
    ).detach()


def default_signal_writer(runner, preds: Tensor) -> None:
    if preds.ndim == 2 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    device = runner.state["agents"]["residents"]["type"].device
    runner.state["agents"]["residents"]["platform_signal"] = (
        preds.detach().clone().to(device).float()
    )


def default_signal_writer_grad(runner, preds: Tensor) -> None:
    if preds.ndim == 2 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    runner.state["agents"]["residents"]["platform_signal"] = preds.clone().float()


def default_state_extractor(runner) -> dict[str, Tensor]:
    r = runner.state["agents"]["residents"]
    x = default_feature_provider(runner)
    y = r["realized_happiness"].detach().float().reshape(-1, 1)
    n = y.shape[0]
    return {
        "x": x,
        "y": y,
        "agent_idx": torch.arange(n),
    }
