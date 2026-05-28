"""Substep: after moves, recompute neighborhoods and label HAPPY / UNHAPPY.

Two-step:
  1. Recompute `same_frac` using the post-move grid. We reuse the
     `compute_neighborhood` substep so the math is shared.
  2. Set `realized_happiness = 1[same_frac_new >= H_0]` where H_0 is the
     BASELINE threshold (NOT the LM-modulated H_i). This is the SFT
     target -- the LM's job is to predict the realized happiness label
     under the baseline rule, not under the rule it itself perturbed.

Also stores `previous_state` for the next round's prompt context (1.0 if
happy, 0.0 if unhappy at the end of this round). This is what the
prompt builder reads as "Previous state: happy/unhappy" on the next
round's LM call.

Writes:
  state["agents"]["residents"]["same_frac"]            (N,) updated
  state["agents"]["residents"]["opp_frac"]             (N,) updated
  state["agents"]["residents"]["empty_frac"]           (N,) updated
  state["agents"]["residents"]["realized_happiness"]   (N,) 0 or 1 float
  state["agents"]["residents"]["previous_state"]       (N,) 0 or 1 float
"""

from __future__ import annotations

import torch

from perfsim.scenarios.at_schelling.substeps.compute_neighborhood import (
    compute_neighborhood,
)


def compute_realized_happiness(state: dict, config: dict) -> None:
    # Recompute neighborhood fractions on the post-move grid.
    compute_neighborhood(state, config)

    meta = config["simulation_metadata"]
    H_0 = float(meta["baseline_threshold"])

    residents = state["agents"]["residents"]
    s_new = residents["same_frac"]
    realized = (s_new >= H_0).float()

    residents["realized_happiness"] = realized
    # `previous_state` is what the NEXT round's prompt reads. The current
    # round's prediction was made before the move, so the prediction
    # target is `realized_happiness` (this round's outcome). Snapshotting
    # it into `previous_state` here means round t+1's prompt says "your
    # previous state was {realized_happiness from round t}". That matches
    # the user-specified prompt structure.
    residents["previous_state"] = realized.clone()
