"""Welfare / degradation probes for the social-dilemma PP loop.

The headline these are built to support: a performatively stable predictor can
be behaviorally ACCURATE (low calibration error) while the population it induced
is socially WORSE (low mean cooperation, collapsed diversity, the baseline type
signal crowded out). The four metrics together tell that story:

- mean_cooperation(data):   ybar = mean_i y_i.            Welfare level.
- cooperation_variance(data): Var_i(y_i).                 Homogenization / collapse.
- type_r2(data, type_baseline): R^2 of a linear fit y ~ b(x_i). How much of the
  realized cooperation the exogenous baseline type still explains; -> 0 means the
  platform signal has crowded out the type signal.
- calibration_error(model, data): E_i[(theta(x_i) - y_i)^2]. How accurate the
  platform is on the world it helped create.
"""

from __future__ import annotations

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import Data


def _y(data: Data) -> Tensor:
    return data["y"].detach().reshape(-1).float()


def mean_cooperation(data: Data) -> float:
    return float(_y(data).mean())


def cooperation_variance(data: Data) -> float:
    return float(_y(data).var(unbiased=False))


def type_r2(data: Data, type_baseline: Tensor) -> float:
    """R^2 of the least-squares fit y ~ a + b * type_baseline.

    1.0 = realized cooperation fully explained by exogenous type; 0.0 = type no
    longer predicts behavior (signal crowded out). Returns 0.0 if y is constant.
    """
    y = _y(data)
    b = type_baseline.detach().reshape(-1).float()
    if y.numel() != b.numel():
        raise ValueError(f"type_baseline ({b.numel()}) and y ({y.numel()}) length mismatch")
    y_var = y.var(unbiased=False)
    if float(y_var) < 1e-12:
        return 0.0
    bc = b - b.mean()
    denom = float((bc * bc).sum())
    if denom < 1e-12:
        return 0.0
    slope = float((bc * (y - y.mean())).sum() / denom)
    y_hat = y.mean() + slope * bc
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def calibration_error(model: Model, data: Data) -> float:
    """Mean squared error of the platform prediction on the induced data."""
    y = _y(data)
    with torch.no_grad():
        pred = model(data["x"]).detach().reshape(-1).float()
    return float(((pred - y) ** 2).mean())
