"""Default feature_provider, signal_writer, state_extractor for at_schelling.

These three are the only Schelling-specific bits the perfsim adapter
needs; everything else (the runner, the substeps, the config) is
hidden inside the env factory.

Feature provider returns a per-agent feature tensor; for the LM scorer
the canonical "feature" is just `agent_idx` because the LM's actual
input is built per-agent inside `BinaryLMScorer.forward` from the
profiles DataFrame. So we emit a (N, 5) feature tensor of
[type, same_frac, opp_frac, empty_frac, previous_state] for callers
that want numeric features (e.g., a linear scorer for sanity checks),
and the LM scorer ignores it except for the leading dimension.

Signal writer writes p_i values onto state["agents"]["residents"]
["platform_signal"]. The move_decision substep reads
state["agents"]["residents"]["p_pred"], populated by
happiness_predict_action from platform_signal -- this two-hop pattern
mirrors at_covid (signal -> action -> consumer-of-action).

State extractor returns the SFT-ready (x, y, agent_idx) triple where
y is the realized happiness from the LAST round of the env.run loop.
"""

from __future__ import annotations

import torch
from torch import Tensor


def default_feature_provider(runner) -> Tensor:
    """Stack (type, same_frac, opp_frac, empty_frac, previous_state)."""
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
    """Write per-agent p_i values to `agents/residents/platform_signal`.

    Detaches before writing (non-grad `run` path is the default). For
    `grad_run` use `default_signal_writer_grad`.
    """
    if preds.ndim == 2 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    runner.state["agents"]["residents"]["platform_signal"] = (
        preds.detach().clone().to(runner.state["agents"]["residents"]["type"].device)
    )


def default_signal_writer_grad(runner, preds: Tensor) -> None:
    """Non-detaching variant for grad_run paths."""
    if preds.ndim == 2 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    runner.state["agents"]["residents"]["platform_signal"] = preds.clone()


def default_state_extractor(runner) -> dict[str, Tensor]:
    """Return (x=features, y=realized_happiness, agent_idx) for SFT.

    y is the post-move realized happiness from the most recent round
    of runner.step(num_steps=K). The SFTLearner formats y as the string
    "HAPPY" or "UNHAPPY" via a target_formatter (see README).
    """
    r = runner.state["agents"]["residents"]
    x = default_feature_provider(runner)
    y = r["realized_happiness"].detach().float().reshape(-1, 1)
    n = y.shape[0]
    return {
        "x": x,
        "y": y,
        "agent_idx": torch.arange(n),
    }
