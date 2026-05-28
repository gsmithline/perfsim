"""NYC ACS-shaped demographics + initial Schelling-grid placement."""

from __future__ import annotations

import torch


NYC_ACS_PROPORTIONS_4TYPE: tuple[float, ...] = (0.36, 0.21, 0.29, 0.14)
TYPE_NAMES: tuple[str, ...] = ("White", "Black", "Hispanic", "Asian")


def load_nyc_demographics(
    n_agents: int = 200,
    *,
    proportions: tuple[float, ...] | None = None,
) -> dict[int, int]:
    """Return a {type_id: count} dict summing to n_agents."""
    props = proportions if proportions is not None else NYC_ACS_PROPORTIONS_4TYPE
    if abs(sum(props) - 1.0) > 1e-6:
        raise ValueError(
            f"proportions must sum to 1.0; got {sum(props):.6f} for {props!r}"
        )
    raw = [n_agents * p for p in props]
    counts = [int(x) for x in raw]
    remainders = [r - int(r) for r in raw]
    deficit = n_agents - sum(counts)
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
    """Random placement on HxW grid. Returns coords, types, grid_occ, grid_type.

    grid_occupancy stores (agent_idx + 1) at occupied cells (so 0 = empty
    without colliding with agent index 0). grid_type stores type id at
    occupied cells, -1 at empty.
    """
    if grid_height * grid_width < n_agents:
        raise ValueError(
            f"grid {grid_height}x{grid_width} too small for n_agents={n_agents}"
        )
    if sum(demographics.values()) != n_agents:
        raise ValueError(
            f"demographics counts sum to {sum(demographics.values())}, "
            f"expected n_agents={n_agents}"
        )

    g = torch.Generator()
    g.manual_seed(int(seed))

    all_cells = grid_height * grid_width
    perm = torch.randperm(all_cells, generator=g)
    chosen = perm[:n_agents]
    rows = chosen // grid_width
    cols = chosen % grid_width

    type_list: list[int] = []
    for t_id, count in sorted(demographics.items()):
        type_list.extend([t_id] * count)
    types_tensor = torch.tensor(type_list, dtype=torch.long)
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
