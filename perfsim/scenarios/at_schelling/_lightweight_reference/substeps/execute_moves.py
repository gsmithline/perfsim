"""Substep: relocate agents whose `move_decision == 1` to empty cells.

Assignment strategy (cheap, correct, not gradient-friendly):
  - Collect the list of (row, col) cells that are empty.
  - For each agent flagged to move (in random order, seeded by a
    per-round generator), pick a random empty cell, swap the agent
    into that cell, and mark the vacated cell as empty.
  - If the empty-cell pool runs out (high move density), remaining
    moving agents stay put for this round.

Mutates:
  state["agents"]["residents"]["coordinates"]   in place
  state["environment"]["grid_occupancy"]        in place
  state["environment"]["grid_type"]             in place

This substep is intentionally non-differentiable (cell-pool sampling is
a discrete reassignment, not a soft-attention swap). Differentiable
move execution would require a per-agent soft probability over cells;
deferred to Stage 2.
"""

from __future__ import annotations

import torch


def execute_moves(state: dict, config: dict) -> None:
    meta = config["simulation_metadata"]
    H = int(meta["grid_height"])
    W = int(meta["grid_width"])

    residents = state["agents"]["residents"]
    coords = residents["coordinates"]
    types_ = residents["type"]
    move = residents["move_decision"]

    grid_occ = state["environment"]["grid_occupancy"]
    grid_type = state["environment"]["grid_type"]

    # Build empty-cell pool from grid_occupancy.
    empty_mask = grid_occ == 0
    empty_rows, empty_cols = torch.where(empty_mask)
    empty_cells = list(zip(empty_rows.tolist(), empty_cols.tolist()))

    # Move-order randomization. The per-round seed makes this reproducible
    # under a fixed Simulator seed (the runner gets `manual_seed(seed)` at
    # init_seed time in the factory).
    movers = torch.nonzero(move > 0.5, as_tuple=False).flatten().tolist()
    if not movers or not empty_cells:
        return

    # Use a torch.Generator tied to current_step so two calls at the same
    # step produce the same shuffle (helpful for deterministic replays).
    g = torch.Generator()
    g.manual_seed(int(state.get("current_step", 0)) * 9973 + 1)
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

        # Vacate old cell.
        grid_occ[old_r, old_c] = 0
        grid_type[old_r, old_c] = -1

        # Occupy new cell.
        coords[agent_idx, 0] = new_r
        coords[agent_idx, 1] = new_c
        grid_occ[new_r, new_c] = agent_idx + 1
        grid_type[new_r, new_c] = int(types_[agent_idx].item())

    # No assignment back to state - we mutated coords, grid_occ, grid_type
    # in place. The dict still points at the same tensors.
