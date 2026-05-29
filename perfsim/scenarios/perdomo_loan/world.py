"""Perdomo strategic-loan world: a StrategicLinearWorld from a Dataset.

Perdomo (ICML 2020 5.1) best-response x_t = x_0 - mu * theta, implemented by
passing epsilon=-mu to StrategicLinearWorld (which uses x_t = x_0 + epsilon * w).
Supports optional standardization and the strat_features manipulation subset.
"""

from __future__ import annotations

from typing import Iterable

import torch

from perfsim.core.dataset import Dataset
from perfsim.environments.dynamics.strategic_linear import StrategicLinearWorld


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
    """Build a StrategicLinearWorld (epsilon=-mu) from a Dataset.

    standardize uses mean/std, or median/IQR if robust; clip>0 bounds the
    standardized values. strat_features restricts the manipulable columns.
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
