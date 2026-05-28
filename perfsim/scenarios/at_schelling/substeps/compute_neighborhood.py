"""SubstepTransition: per-agent same/opp/empty neighbor fractions."""

from __future__ import annotations

import re

import torch

from agent_torch.core.helpers import get_by_path
from agent_torch.core.substep import SubstepTransition


def _get(state, var: str):
    return get_by_path(state, re.split("/", var))


def _compute_fractions(
    coords: torch.Tensor,
    types_: torch.Tensor,
    grid_occ: torch.Tensor,
    grid_type: torch.Tensor,
    H: int,
    W: int,
    r: int,
):
    n = coords.shape[0]
    device = coords.device
    same_frac = torch.full((n,), 0.5, device=device)
    opp_frac = torch.full((n,), 0.5, device=device)
    empty_frac = torch.zeros((n,), device=device)

    for i in range(n):
        row = int(coords[i, 0].item())
        col = int(coords[i, 1].item())
        my_type = int(types_[i].item())

        n_same = n_opp = n_empty = n_total = 0
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = row + dr, col + dc
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                n_total += 1
                occ = int(grid_occ[rr, cc].item())
                if occ == 0:
                    n_empty += 1
                else:
                    t = int(grid_type[rr, cc].item())
                    if t == my_type:
                        n_same += 1
                    else:
                        n_opp += 1
        n_occ = n_same + n_opp
        if n_occ > 0:
            same_frac[i] = n_same / n_occ
            opp_frac[i] = n_opp / n_occ
        if n_total > 0:
            empty_frac[i] = n_empty / n_total
    return same_frac, opp_frac, empty_frac


class ComputeNeighborhood(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        meta = self.config["simulation_metadata"]
        self.H = int(meta["grid_height"])
        self.W = int(meta["grid_width"])
        self.r = int(meta.get("neighborhood_radius", 1))

    def forward(self, state, action):
        iv = self.input_variables
        coords = _get(state, iv["coordinates"]).long()
        types_ = _get(state, iv["type"]).long()
        grid_occ = _get(state, iv["grid_occupancy"]).long()
        grid_type = _get(state, iv["grid_type"]).long()

        same_frac, opp_frac, empty_frac = _compute_fractions(
            coords, types_, grid_occ, grid_type, self.H, self.W, self.r
        )
        return {
            "same_frac": same_frac,
            "opp_frac": opp_frac,
            "empty_frac": empty_frac,
        }
