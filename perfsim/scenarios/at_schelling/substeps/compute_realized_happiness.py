"""SubstepTransition: recompute same_frac post-move; y = 1[s_new >= H_0]."""

from __future__ import annotations

import re

import torch

from agent_torch.core.helpers import get_by_path
from agent_torch.core.substep import SubstepTransition

from perfsim.scenarios.at_schelling.substeps.compute_neighborhood import (
    _compute_fractions,
)


def _get(state, var: str):
    return get_by_path(state, re.split("/", var))


class ComputeRealizedHappiness(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        meta = self.config["simulation_metadata"]
        self.H = int(meta["grid_height"])
        self.W = int(meta["grid_width"])
        self.r = int(meta.get("neighborhood_radius", 1))
        self.H_0 = float(meta["baseline_threshold"])

    def forward(self, state, action):
        iv = self.input_variables
        coords = _get(state, iv["coordinates"]).long()
        types_ = _get(state, iv["type"]).long()
        grid_occ = _get(state, iv["grid_occupancy"]).long()
        grid_type = _get(state, iv["grid_type"]).long()

        same_frac, opp_frac, empty_frac = _compute_fractions(
            coords, types_, grid_occ, grid_type, self.H, self.W, self.r
        )
        realized = (same_frac >= self.H_0).float()
        return {
            "same_frac": same_frac,
            "opp_frac": opp_frac,
            "empty_frac": empty_frac,
            "realized_happiness": realized,
            # Snapshot for next round's prompt context.
            "previous_state": realized.clone(),
        }
