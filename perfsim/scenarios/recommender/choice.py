"""Differentiable choice model (RecSim NG MultinomialLogitChoiceModel analogue).

All functions are smooth (matmul, softmax, log) so gradients flow from the
chosen-item distribution back to the deployed scores -> theta. This is what lets
the environment opt into perfsim's FullyDifferentiable trait (PerfGD).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def affinities(interest: Tensor, features: Tensor) -> Tensor:
    """Per-user, per-item affinity = <interest_n, features_k>. (N, D),(K, D)->(N, K)."""
    return interest @ features.t()


def mnl_choice(
    affinity: Tensor,
    log_exposure: Tensor,
    *,
    alpha: float,
    temp: float,
    log_availability: Optional[Tensor] = None,
    gamma_supply: float = 0.0,
) -> Tensor:
    """Multinomial-logit choice over items, exposure-biased (and optionally supply-weighted).

    logits_{n,k} = (affinity + alpha*log u_k + gamma_supply*log a_k) / temp, softmax over items.
    alpha: exposure bias (demand-side, from the ranker). gamma_supply: how much a
    producer's availability a_k raises its consumption odds (supply-side). With
    log_availability=None / gamma_supply=0 this is the plain demand-only choice model.
    Returns (N, K); rows sum to 1. Note alpha and beta reach the dynamics only
    through alpha*beta (the logsumexp in alpha*log softmax(beta*scores) cancels
    here); exposure itself is the only beta-only quantity.
    """
    logits = affinity + alpha * log_exposure.unsqueeze(0)
    if log_availability is not None and gamma_supply != 0.0:
        logits = logits + gamma_supply * log_availability.unsqueeze(0)
    return torch.softmax(logits / temp, dim=1)