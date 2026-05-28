"""Substep: read the LM's per-agent happiness probability.

The actual LM call happens OUTSIDE this substep, in
`AgentTorchEnvironment.run` (or its hand-rolled equivalent). That entry
point queries `model(features)` once and writes the result to
`state["agents"]["residents"]["platform_signal"]` via the
`signal_writer` callable. This substep simply lifts that vector into a
canonical name (`p_pred`) so the move-decision substep can consume it
without depending on the writer's naming.

It also pins `p_pred` to (N,) shape and clips into [0, 1] defensively
in case the writer left logits or a probability slightly outside the
unit interval (e.g., from numerical drift in BinaryLMScorer).

Pattern: A2 (fixed anchor). The substep does NOT overwrite
`platform_signal`; the adapter's `strict_signal` check stays green.
"""

from __future__ import annotations

import torch


def happiness_predict_action(state: dict, config: dict) -> None:
    residents = state["agents"]["residents"]
    sig = residents["platform_signal"]
    if sig.ndim == 2 and sig.shape[-1] == 1:
        sig = sig.squeeze(-1)
    p = sig.clamp(0.0, 1.0)
    residents["p_pred"] = p
