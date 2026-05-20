"""Perdomo strategic-loan world.

Builds a `StrategicLinearWorld` from a Dataset of (features, label). Applies
optional per-feature standardization and supports the Perdomo restriction
that only a subset of feature columns can be strategically manipulated.

Perdomo convention (ICML 2020 Section 5.1): the strategic best-response is
`x_t[strat_features] = x_0[strat_features] - mu * theta[strat_features]`.
We implement this by passing `epsilon = -mu` to the base
`StrategicLinearWorld`, which uses `x_t = x_0 + epsilon * w` and restricts
the shift to `strat_features` if provided.
"""

from __future__ import annotations

from typing import Iterable

import torch

from perfsim.core.dataset import Dataset
from perfsim.worlds.strategic_linear import StrategicLinearWorld


def build_world(
    dataset: Dataset,
    *,
    mu: float = 1.0,
    standardize: bool = True,
    robust: bool = False,
    clip: float = 0.0,
    strat_features: Iterable[int] | None = None,
    dtype: torch.dtype = torch.float32,
) -> StrategicLinearWorld:
    """Build a StrategicLinearWorld from a Dataset for the Perdomo loan setup.

    Args:
        dataset: Dataset producing the supervised schema {x, y}.
        mu: Perdomo strategic strength parameter (>= 0).
        standardize: if True, per-feature normalization.
        robust: if True (and `standardize=True`), use median / IQR. Default
            False (Perdomo uses mean / std on balanced data).
        clip: if > 0 (and `standardize=True`), clip post-standardized
            values to `[-clip, clip]`. Default 0 (disabled). Useful with
            unbalanced data; not needed on Perdomo's balanced setup.
        strat_features: if set, only these feature indices are
            strategically shifted. Default None (all features).
        dtype: tensor dtype.

    Returns:
        StrategicLinearWorld configured with `epsilon = -mu`.
    """
    if mu < 0:
        raise ValueError(f"mu must be >= 0 (sign handled internally); got {mu}")
    if clip < 0:
        raise ValueError(f"clip must be >= 0; got {clip}")
    data = dataset.load()
    x0 = data["x"].to(dtype=dtype)
    y = data["y"]
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    if standardize:
        if robust:
            center = x0.median(dim=0, keepdim=True).values
            q75 = x0.quantile(0.75, dim=0, keepdim=True)
            q25 = x0.quantile(0.25, dim=0, keepdim=True)
            scale = (q75 - q25).clamp(min=1e-6)
        else:
            center = x0.mean(dim=0, keepdim=True)
            scale = x0.std(dim=0, keepdim=True).clamp(min=1e-6)
        x0 = (x0 - center) / scale
        if clip > 0:
            x0 = x0.clamp(-clip, clip)
    return StrategicLinearWorld(
        x0=x0,
        y=y,
        epsilon=-float(mu),
        strat_features=strat_features,
        dtype=dtype,
    )
