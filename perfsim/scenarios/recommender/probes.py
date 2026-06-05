"""Probes for the recommender ecosystem PP loop.

Headline: a performatively stable ranker becomes well-aligned with the engagement
it induces (alignment_induced -> 1) while drifting away from true relevance
(alignment_truth -> down), as exposure concentrates and tastes homogenize.

corr(theta, induced engagement). NOTE: ~regression fit quality, since theta
    is MSE-trained on this target. Diagnostic only -- NOT a performativity signal.
    Use engagement_faithfulness for the performative measure.
"""

from __future__ import annotations

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import Data


def _pearson(a: Tensor, b: Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom) < 1e-12:
        return 0.0
    return float((a @ b) / denom)


def alignment_induced(model: Model, data: Data) -> float:
    """Correlation of theta's scores with the engagement it induced. -> 1 = fits its own world."""
    with torch.no_grad():
        pred = model(data["x"]).reshape(-1)
    return _pearson(pred, data["y"])


def alignment_truth(model: Model, world) -> float:
    """Correlation of theta's scores with hidden true relevance. Falls under performativity."""
    with torch.no_grad():
        pred = model(world.features).reshape(-1)
    return _pearson(pred, world.hidden_quality())


def induced_calibration(model: Model, data: Data) -> float:
    """MSE(theta(x), engagement). The platform's own accuracy metric."""
    with torch.no_grad():
        pred = model(data["x"]).reshape(-1)
        y = data["y"].reshape(-1)
    return float(((pred - y) ** 2).mean())


def exposure_gini(world) -> float:
    """Gini of the exposure vector. 0 = uniform feed, ->1 = winner-take-all."""
    x = world.last_exposure.reshape(-1).float().clamp_min(0.0)
    total = float(x.sum())
    if total < 1e-12:
        return 0.0
    xs, _ = x.sort()
    n = xs.numel()
    idx = torch.arange(1, n + 1, dtype=xs.dtype)
    return float((2.0 * (idx * xs).sum()) / (n * xs.sum()) - (n + 1) / n)


def engagement_entropy(data: Data) -> float:
    """Shannon entropy (nats) of normalized engagement across items. Diversity of consumption."""
    y = data["y"].reshape(-1).float().clamp_min(0.0)
    total = y.sum()
    if float(total) < 1e-12:
        return 0.0
    p = y / total
    nz = p[p > 0]
    return float(-(nz * nz.log()).sum())


def interest_drift(world) -> float:
    """Mean L2 movement of user interests from their initial values. Population performativity."""
    with torch.no_grad():
        return float((world.current_interest - world.interest0).norm(dim=1).mean())


def engagement_faithfulness(world) -> float:
    """corr(induced engagement, true relevance). The performative DATA distortion.

    Flat under no exposure bias; decays as the exposure-feedback loop runs. This,
    not alignment_induced (which is regression fit quality), is the signal.
    """
    return _pearson(world.last_engagement, world.hidden_quality())


def exposure_faithfulness(world) -> float: #TODO check this
    """corr(exposure, true relevance). Does attention go to truly-relevant items"""
    return _pearson(world.last_exposure, world.hidden_quality())


# ---- competition probes ------------------------------------------------------
def platform_divergence(world) -> float:
    """Mean pairwise (1 - corr) between platform exposure profiles. 0 = clones."""
    u = world.last_exposure
    n = u.shape[0]
    vals = [1.0 - _pearson(u[i], u[j]) for i in range(n) for j in range(i + 1, n)]
    return float(sum(vals) / len(vals))


def share_concentration(world) -> float:
    """Mean max share per user. 1/P = even split, 1 = fully sorted."""
    return float(world.last_shares.max(dim=1).values.mean())


def satisfaction_current(world) -> float:
    """Share-weighted realized affinity under current (drifted) tastes."""
    return float((world.last_shares * world.last_satisfaction).sum(dim=1).mean())


def welfare_innate(world) -> float:
    """Share-weighted realized affinity under ORIGINAL tastes. The second ledger."""
    with torch.no_grad():
        aff0 = world.interest0 @ world.features.t()
        per = (world.last_choice * aff0.unsqueeze(0)).sum(dim=2).t()   # (N, P)
        return float((world.last_shares * per).sum(dim=1).mean())


# ---- producer-ecosystem probes ---------------------------------------------
def producer_diversity(world) -> float:
    """Effective number of live producers = exp(entropy(availability)).

    Collapses under rich-get-richer dynamics. Requires a world exposing
    `last_availability` (ProducerEcosystemWorld).
    """
    a = world.last_availability.reshape(-1).float().clamp_min(0.0)
    total = a.sum()
    if float(total) < 1e-12:
        return 0.0
    p = a / total
    nz = p[p > 0]
    return float((-(nz * nz.log()).sum()).exp())


def surviving_producers(world, *, thresh_ratio: float = 0.1) -> int:
    """Count producers with availability above thresh_ratio * uniform. Topic survival."""
    a = world.last_availability.reshape(-1).float()
    uniform = 1.0 / a.numel()
    return int((a > thresh_ratio * uniform).sum())


def user_welfare(world) -> float:
    """Mean realized TRUE affinity of consumption (uses original interest0).

    Falls when topics users truly prefer die out under rich-get-richer.
    """
    with torch.no_grad():
        aff0 = world.interest0 @ world.features.t()   # (N, K) true preferences
        choice = world.last_choice                     # (N, K)
        return float((choice * aff0).sum(dim=1).mean())