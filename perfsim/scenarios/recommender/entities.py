"""Entities for the performative recommender ecosystem.

Following RecSim NG's (Mladenov et al. 2021, arXiv:2103.08057) entity pattern:
each entity models an entire population via batched tensors, not one object 
per agent. The leading tensor dim is the population axis.

- Corpus:         K items, each a feature row + a base appeal scalar. Fixed.
- UserPopulation: N users, each a latent interest vector (N, D), sampled from a
                  community mixture so heterogeneity = per-user parameters drawn
                  from a small set of community centers (RecSim NG "communities").

interest0 is the initial population and doubles as the hidden ground-truth
anchor: the platform never trains on it, but probes compare theta to the true
affinities it implies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Corpus:
    """A fixed item corpus. features: (K, D); appeal: (K,)."""

    features: Tensor
    appeal: Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError(f"features must be (K, D); got {tuple(self.features.shape)}")
        if self.appeal.shape != (self.features.shape[0],):
            raise ValueError(
                f"appeal must be (K,)={(self.features.shape[0],)}; "
                f"got {tuple(self.appeal.shape)}"
            )

    @property
    def n_items(self) -> int:
        return int(self.features.shape[0])

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True)
class UserPopulation:
    """A user population. interest0: (N, D); community: (N,) community id."""

    interest0: Tensor
    community: Tensor

    def __post_init__(self) -> None:
        if self.interest0.ndim != 2:
            raise ValueError(f"interest0 must be (N, D); got {tuple(self.interest0.shape)}")
        if self.community.shape != (self.interest0.shape[0],):
            raise ValueError(
                f"community must be (N,)={(self.interest0.shape[0],)}; "
                f"got {tuple(self.community.shape)}"
            )

    @property
    def n_users(self) -> int:
        return int(self.interest0.shape[0])

    @property
    def dim(self) -> int:
        return int(self.interest0.shape[1])


def sample_corpus(
    n_items: int,
    dim: int,
    *,
    generator: torch.Generator,
    appeal_low: float = 0.5,
    appeal_high: float = 1.5,
    dtype: torch.dtype = torch.float32,
) -> Corpus:
    """Items ~ N(0, I) in feature space; appeal ~ U(appeal_low, appeal_high)."""
    features = torch.randn(n_items, dim, generator=generator, dtype=dtype)
    span = appeal_high - appeal_low
    appeal = appeal_low + span * torch.rand(n_items, generator=generator, dtype=dtype)
    return Corpus(features=features, appeal=appeal)


def sample_user_population(
    n_users: int,
    dim: int,
    *,
    n_communities: int = 4,
    community_spread: float = 0.3,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> UserPopulation:
    """Community mixture: each user belongs to one of `n_communities` clusters.

    Community centers ~ N(0, I); a user's interest = center + spread * N(0, I).
    Smaller `community_spread` -> tighter, more distinct communities.
    """
    centers = torch.randn(n_communities, dim, generator=generator, dtype=dtype)
    community = torch.randint(
        n_communities, (n_users,), generator=generator
    )
    jitter = community_spread * torch.randn(n_users, dim, generator=generator, dtype=dtype)
    interest0 = centers[community] + jitter
    return UserPopulation(interest0=interest0.to(dtype), community=community)