"""Substep: compute per-agent same-type / opp-type / empty neighbor fractions.

Inputs read from state:
  state["agents"]["residents"]["coordinates"]  (N, 2) int rows/cols
  state["agents"]["residents"]["type"]         (N,)   int demographic type id
  state["environment"]["grid_occupancy"]       (H, W) int -- agent_idx + 1
                                                 at occupied cells, 0 at empty.
                                                 (We use +1 so 0 unambiguously
                                                 means empty without colliding
                                                 with agent_idx == 0.)
  state["environment"]["grid_type"]            (H, W) int -- type at each cell,
                                                 -1 at empty cells.

Writes to state["agents"]["residents"]:
  same_frac    (N,) float -- fraction of OCCUPIED neighbor cells that match
                              the agent's type. Defined as 0.5 for agents
                              with zero occupied neighbors (no information).
  opp_frac     (N,) float -- 1 - same_frac when occupied-neighbors > 0,
                              else 0.5.
  empty_frac   (N,) float -- (# empty cells in Moore neighborhood) /
                              (total neighborhood size, i.e. 8 for interior
                              cells, fewer on the boundary).

Neighborhood radius is `simulation_metadata.neighborhood_radius` (default 1
= Moore 8-neighborhood). The substep is O(N * (2r+1)^2) per call, which is
fine for the ~200-agent / 20x20 target. No vectorization yet; favors
correctness over speed.
"""

from __future__ import annotations

import torch


def compute_neighborhood(state: dict, config: dict) -> None:
    meta = config["simulation_metadata"]
    H = int(meta["grid_height"])
    W = int(meta["grid_width"])
    r = int(meta.get("neighborhood_radius", 1))

    residents = state["agents"]["residents"]
    coords = residents["coordinates"]
    types_ = residents["type"]
    n = coords.shape[0]
    device = coords.device

    grid_occ = state["environment"]["grid_occupancy"]
    grid_type = state["environment"]["grid_type"]

    same_frac = torch.full((n,), 0.5, device=device)
    opp_frac = torch.full((n,), 0.5, device=device)
    empty_frac = torch.zeros((n,), device=device)

    for i in range(n):
        row = int(coords[i, 0].item())
        col = int(coords[i, 1].item())
        my_type = int(types_[i].item())

        n_same = 0
        n_opp = 0
        n_empty = 0
        n_total = 0
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
        # else: leave at default 0.5 / 0.5 (no information)
        if n_total > 0:
            empty_frac[i] = n_empty / n_total

    residents["same_frac"] = same_frac
    residents["opp_frac"] = opp_frac
    residents["empty_frac"] = empty_frac
