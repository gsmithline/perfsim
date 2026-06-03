"""Behaviors: the DBN transition pieces (RecSim NG "behaviors").

Vectorized over the (N, D) user-population batch. All differentiable.
"""

from __future__ import annotations

import torch
from torch import Tensor


def exposure(scores: Tensor, beta: float) -> Tensor:
    """Slate exposure propensity from deployed scores. (K,) -> (K,) on simplex.

    u = softmax(beta * scores). beta is the ranker sharpness: large beta -> a few
    items get almost all exposure (winner-take-all feed).
    """
    return torch.softmax(beta * scores, dim=0)


def engagement(choice: Tensor, appeal: Tensor) -> Tensor:
    """Observed per-item engagement label. (N, K),(K,) -> (K,).

    Mean consumption probability across users, scaled by item appeal. This is the
    logged signal theta trains on -- biased by exposure through `choice`.
    """
    return choice.mean(dim=0) * appeal


def drift_interest(
    interest: Tensor,
    choice: Tensor,
    features: Tensor,
    eta: float,
) -> Tensor:
    """User interests drift toward consumed items. (N, D),(N, K),(K, D) -> (N, D).

    target_n = E_k[choice_{n,k}] feature_k = (choice @ features)_n.
    interest_next = interest + eta * (target - interest). This is the population
    becoming performative: tastes move toward whatever theta surfaced.
    """
    target = choice @ features
    return interest + eta * (target - interest)


def replicator_step(weights: Tensor, fitness: Tensor) -> Tensor:
    """Taylor-Jonker discrete replicator on a simplex. (K,),(K,) -> (K,), sums to 1.

    weights_{t+1} = weights * (1 + fitness) / <weights, 1 + fitness>.
    """
    shifted = 1.0 + fitness
    return (weights * shifted) / (weights @ shifted).clamp_min(1e-12)


def producer_fitness(engagement: Tensor, *, kappa: float, rate: float) -> Tensor:
    """Relative-engagement fitness for producers. (K,) -> (K,).

    f_k = rate * (y_k / mean(y) - 1) - kappa. Above-average engagement -> grow;
    below-average -> shrink. kappa is a flat production cost.
    """
    mean = engagement.mean().clamp_min(1e-12)
    return rate * (engagement / mean - 1.0) - kappa


def exposure_with_boost(
    scores: Tensor,
    availability: Tensor,
    *,
    beta: float,
    boost: float,
) -> Tensor:
    """Ranker exposure, optionally boosting low-availability producers. (K,),(K,) -> (K,).

    u = softmax(beta*scores - boost*log a). boost=0 is the myopic ranker; boost>0
    is an ecosystem-aware policy that amplifies under-served (small) producers.
    """
    logits = beta * scores
    if boost != 0.0:
        logits = logits - boost * torch.log(availability.clamp_min(1e-12))
    return torch.softmax(logits, dim=0)