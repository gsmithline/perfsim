"""Demographic data and initial agent placement for Schelling.

Stage 1: hardcode ACS-shaped proportions for NYC (rough match to 2020
American Community Survey 5-year estimates for NYC overall). Stage 2
will load per-tract demographics from bundled NYC ACS pickles.

Reference proportions (NYC, ACS 2020 5-yr, rounded):
  White (non-Hispanic):  ~31%
  Black (non-Hispanic):  ~21%
  Hispanic (any race):   ~29%
  Asian (non-Hispanic):  ~14%
  Other / multiracial:   ~ 5%   (rolled into "White" here for the 4-type
                                 demo so the proportions still sum to 1.0)

Types are encoded as integer ids:
  0 = White, 1 = Black, 2 = Hispanic, 3 = Asian.

`load_nyc_demographics` returns a dict of {type_id: count}. The grid
initializer in `env.py` places agents at random empty cells in
proportion to those counts.
"""

from __future__ import annotations

import torch


# Order: White, Black, Hispanic, Asian. Sum = 1.0.
NYC_ACS_PROPORTIONS_4TYPE: tuple[float, ...] = (0.36, 0.21, 0.29, 0.14)

TYPE_NAMES: tuple[str, ...] = ("White", "Black", "Hispanic", "Asian")


def load_nyc_demographics(
    n_agents: int = 200,
    *,
    proportions: tuple[float, ...] | None = None,
) -> dict[int, int]:
    """Return a {type_id: count} dict that sums to `n_agents`.

    Uses largest-remainder rounding so the rounded counts exactly sum to
    `n_agents`. With the default proportions and `n_agents=200` this
    yields White=72, Black=42, Hispanic=58, Asian=28 (sum=200).
    """
    props = proportions if proportions is not None else NYC_ACS_PROPORTIONS_4TYPE
    if abs(sum(props) - 1.0) > 1e-6:
        raise ValueError(
            f"proportions must sum to 1.0; got {sum(props):.6f} for {props!r}"
        )
    raw = [n_agents * p for p in props]
    counts = [int(x) for x in raw]
    remainders = [r - int(r) for r in raw]
    deficit = n_agents - sum(counts)
    # Distribute the deficit to types with the largest fractional remainders.
    order = sorted(range(len(props)), key=lambda i: remainders[i], reverse=True)
    for i in order[:deficit]:
        counts[i] += 1
    return {i: counts[i] for i in range(len(props))}


def initial_placement(
    n_agents: int,
    grid_height: int,
    grid_width: int,
    demographics: dict[int, int],
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place agents at random empty cells on an HxW grid.

    Returns:
      coordinates  (N, 2) int -- row, col per agent
      types        (N,)  int -- demographic type id per agent
      grid_occupancy (H, W) int -- agent_idx + 1 at occupied, 0 at empty
      grid_type    (H, W) int -- type at occupied cells, -1 at empty

    The "+1" convention on grid_occupancy lets us distinguish empty
    cells (value 0) from agent index 0 (stored as 1).
    """
    if grid_height * grid_width < n_agents:
        raise ValueError(
            f"grid {grid_height}x{grid_width} = {grid_height*grid_width} cells "
            f"is too small for n_agents={n_agents}"
        )
    if sum(demographics.values()) != n_agents:
        raise ValueError(
            f"demographics counts sum to {sum(demographics.values())}, "
            f"expected n_agents={n_agents}"
        )

    g = torch.Generator()
    g.manual_seed(int(seed))

    # Random ordering of cells.
    all_cells = grid_height * grid_width
    perm = torch.randperm(all_cells, generator=g)
    chosen = perm[:n_agents]
    rows = chosen // grid_width
    cols = chosen % grid_width

    # Per-agent types from demographics dict (in stable order).
    type_list: list[int] = []
    for t_id, count in sorted(demographics.items()):
        type_list.extend([t_id] * count)
    types_tensor = torch.tensor(type_list, dtype=torch.long)

    # Shuffle type-to-cell assignment so types are spatially well-mixed at
    # init (Schelling segregation is interesting precisely because it
    # emerges from a well-mixed start).
    type_perm = torch.randperm(n_agents, generator=g)
    types_tensor = types_tensor[type_perm]

    coordinates = torch.stack([rows, cols], dim=-1).long()

    grid_occupancy = torch.zeros((grid_height, grid_width), dtype=torch.long)
    grid_type = torch.full((grid_height, grid_width), -1, dtype=torch.long)
    for i in range(n_agents):
        r, c = int(rows[i].item()), int(cols[i].item())
        grid_occupancy[r, c] = i + 1
        grid_type[r, c] = int(types_tensor[i].item())

    return coordinates, types_tensor, grid_occupancy, grid_type
