"""SubstepTransition: relocate movers to random empty cells."""

from __future__ import annotations

import re

import torch

from agent_torch.core.helpers import get_by_path
from agent_torch.core.substep import SubstepTransition


def _get(state, var: str):
    return get_by_path(state, re.split("/", var))


class ExecuteMoves(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, action):
        iv = self.input_variables
        coords = _get(state, iv["coordinates"]).long().clone()
        types_ = _get(state, iv["type"]).long()
        move = _get(state, iv["move_decision"])
        grid_occ = _get(state, iv["grid_occupancy"]).long().clone()
        grid_type = _get(state, iv["grid_type"]).long().clone()

        empty_rows, empty_cols = torch.where(grid_occ == 0)
        empty_cells = list(zip(empty_rows.tolist(), empty_cols.tolist()))
        movers = torch.nonzero(move > 0.5, as_tuple=False).flatten().tolist()
        if not movers or not empty_cells:
            return {
                "coordinates": coords,
                "grid_occupancy": grid_occ,
                "grid_type": grid_type,
            }

        # Per-step seed so a re-run with the same Runner seed produces the same shuffle.
        step = int(state.get("current_step", 0))
        g = torch.Generator()
        g.manual_seed(step * 9973 + 1)
        perm = torch.randperm(len(movers), generator=g).tolist()
        cell_perm = torch.randperm(len(empty_cells), generator=g).tolist()
        movers_shuffled = [movers[i] for i in perm]
        cells_shuffled = [empty_cells[i] for i in cell_perm]

        cell_idx = 0
        for agent_idx in movers_shuffled:
            if cell_idx >= len(cells_shuffled):
                break
            new_r, new_c = cells_shuffled[cell_idx]
            cell_idx += 1
            old_r = int(coords[agent_idx, 0].item())
            old_c = int(coords[agent_idx, 1].item())
            grid_occ[old_r, old_c] = 0
            grid_type[old_r, old_c] = -1
            coords[agent_idx, 0] = new_r
            coords[agent_idx, 1] = new_c
            grid_occ[new_r, new_c] = agent_idx + 1
            grid_type[new_r, new_c] = int(types_[agent_idx].item())

        return {
            "coordinates": coords,
            "grid_occupancy": grid_occ,
            "grid_type": grid_type,
        }
