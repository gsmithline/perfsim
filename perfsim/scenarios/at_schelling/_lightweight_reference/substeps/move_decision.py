"""Substep: decide which agents will move this round.

Computes the LM-modulated effective threshold per agent:

    H_i = clip(H_0 - lambda * (p_i - 0.5), 0, 1)

then samples the move decision via straight-through Bernoulli on
sigmoid(hardness * (H_i - s_i)), which is `1[s_i < H_i]` at the forward
and a pass-through gradient on the backward.

The straight-through is overkill for the non-grad `run` path (perfsim's
default), but matches AT's `StraightThroughBernoulli` so this substep is
ready for `grad_run` once the user wires a differentiable LM scorer
through it.

Writes:
  state["agents"]["residents"]["effective_threshold"]  (N,)  H_i
  state["agents"]["residents"]["move_decision"]        (N,)  {0, 1}
"""

from __future__ import annotations

import torch

from agent_torch.core.distributions import StraightThroughBernoulli


_st_bernoulli = StraightThroughBernoulli.apply


def move_decision(state: dict, config: dict) -> None:
    meta = config["simulation_metadata"]
    H_0 = float(meta["baseline_threshold"])
    lam = float(meta["lambda"])
    hardness = float(meta.get("move_hardness", 8.0))

    residents = state["agents"]["residents"]
    p = residents["p_pred"]
    s = residents["same_frac"]

    H_eff = (H_0 - lam * (p - 0.5)).clamp(0.0, 1.0)

    # `(H_eff - s)` is positive when the threshold is above the realized
    # same-type fraction -> agent is unhappy -> should move. The hardness
    # factor sharpens the sigmoid so the forward agrees with the hard
    # comparison s < H_eff at the boundary up to numerical noise.
    move_prob = torch.sigmoid(hardness * (H_eff - s))
    move = _st_bernoulli(move_prob)

    residents["effective_threshold"] = H_eff
    residents["move_decision"] = move
