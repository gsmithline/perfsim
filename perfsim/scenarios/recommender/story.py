"""Story: assemble entities into a RecommenderEcosystemWorld."""

from __future__ import annotations

import torch

from perfsim.scenarios.recommender.entities import sample_corpus, sample_user_population
from perfsim.scenarios.recommender.env import RecommenderEcosystemWorld
from perfsim.scenarios.recommender.env_producers import ProducerEcosystemWorld


def build_recommender_env(
    *,
    n_items: int = 30,
    n_users: int = 200,
    dim: int = 8,
    n_communities: int = 4,
    community_spread: float = 0.3,
    alpha: float = 4.0,
    beta: float = 4.0,
    temp: float = 0.5,
    eta: float = 0.15,
    noise: float = 0.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> RecommenderEcosystemWorld:
    """Build a performative recommender world with a community-mixture population.

    alpha: exposure bias (label performativity). beta: ranker sharpness.
    eta: taste-drift rate (population performativity). temp: choice temperature.
    """
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    corpus = sample_corpus(n_items, dim, generator=gen, dtype=dtype)
    users = sample_user_population(
        n_users, dim, n_communities=n_communities,
        community_spread=community_spread, generator=gen, dtype=dtype,
    )
    return RecommenderEcosystemWorld(
        corpus, users, alpha=alpha, beta=beta, temp=temp, eta=eta, noise=noise, dtype=dtype,
    )


def build_producer_env(
    *,
    n_items: int = 30,
    n_users: int = 200,
    dim: int = 8,
    n_communities: int = 4,
    community_spread: float = 0.3,
    alpha: float = 4.0,
    beta: float = 8.0,
    temp: float = 0.5,
    eta: float = 0.0,
    gamma_supply: float = 1.0,
    kappa: float = 0.0,
    repl_rate: float = 0.1,
    boost: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> ProducerEcosystemWorld:
    """Build a producer-ecosystem world (replicating content supply).

    gamma_supply: supply-side feedback (availability -> consumption; the producer
    channel is only performative when this is > 0). repl_rate / kappa: replicator
    speed / production cost. boost: ecosystem-aware exposure that amplifies small
    producers (0 = myopic ranker). eta: user taste-drift (0 isolates the supply
    channel).
    """
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    corpus = sample_corpus(n_items, dim, generator=gen, dtype=dtype)
    users = sample_user_population(
        n_users, dim, n_communities=n_communities,
        community_spread=community_spread, generator=gen, dtype=dtype,
    )
    return ProducerEcosystemWorld(
        corpus, users, alpha=alpha, beta=beta, temp=temp, eta=eta,
        gamma_supply=gamma_supply, kappa=kappa, repl_rate=repl_rate,
        boost=boost, noise=noise, dtype=dtype,
    )