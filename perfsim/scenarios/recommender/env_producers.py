"""ProducerEcosystemWorld: recommender ecosystem with replicating content supply.

Each item is a producer carrying an availability mass on the simplex that
replicates on engagement (rich-get-richer; optional ecosystem-aware boost).
Producer dynamics are replicator-based, inspired by -- not a port of -- RecSim
NG's provider model. Shared plumbing lives in RecommenderWorldBase.
"""

from __future__ import annotations

import torch
from torch import Tensor

from perfsim.core.types import Data
from perfsim.environments.dynamics.stateful_population import State
from perfsim.scenarios.recommender._base import RecommenderWorldBase
from perfsim.scenarios.recommender.behaviors import (
    drift_interest,
    engagement,
    exposure_with_boost,
    producer_fitness,
    replicator_step,
)
from perfsim.scenarios.recommender.choice import affinities, mnl_choice
from perfsim.scenarios.recommender.entities import Corpus, UserPopulation


class ProducerEcosystemWorld(RecommenderWorldBase):
    """State = user interests (N, D) + producer availability (K,) on the simplex."""

    def __init__(
        self,
        corpus: Corpus,
        users: UserPopulation,
        *,
        alpha: float = 4.0,
        beta: float = 8.0,
        temp: float = 0.5,
        eta: float = 0.0,
        gamma_supply: float = 1.0,
        kappa: float = 0.0,
        repl_rate: float = 0.1,
        boost: float = 0.0,
        noise: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        k = corpus.n_items
        availability0 = torch.full((k,), 1.0 / k, dtype=dtype)
        super().__init__(
            corpus, users,
            {"interest": users.interest0.to(dtype).clone(), "availability": availability0},
            dtype=dtype,
        )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temp = float(temp)
        self.eta = float(eta)
        self.gamma_supply = float(gamma_supply)
        self.kappa = float(kappa)
        self.repl_rate = float(repl_rate)
        self.boost = float(boost)
        self.noise = float(noise)

    @property
    def last_availability(self) -> Tensor:
        return self._info["availability"]

    def _forward(self, scores: Tensor, state: State) -> tuple[Data, State, dict[str, Tensor]]:
        interest = state["interest"]
        availability = state["availability"]
        u = exposure_with_boost(scores, availability, beta=self.beta, boost=self.boost)
        log_u = torch.log(u.clamp_min(1e-12))
        log_a = torch.log(availability.clamp_min(1e-12))
        aff = affinities(interest, self._features)
        choice = mnl_choice(
            aff, log_u, alpha=self.alpha, temp=self.temp,
            log_availability=log_a, gamma_supply=self.gamma_supply,
        )
        y = engagement(choice, self._appeal)
        fitness = producer_fitness(y, kappa=self.kappa, rate=self.repl_rate)
        availability_next = replicator_step(availability, fitness)
        interest_next = drift_interest(interest, choice, self._features, self.eta)
        if self.noise > 0.0 and self._gen is not None:
            interest_next = interest_next + self.noise * torch.randn(
                interest_next.shape, generator=self._gen, dtype=self._dtype
            )
        info = {
            "exposure": u, "choice": choice, "engagement": y,
            "scores": scores, "availability": availability, "fitness": fitness,
        }
        data: Data = {
            "x": self._features,
            "y": y.reshape(-1, 1),
            "agent_idx": self._item_idx.clone(),
        }
        return data, {"interest": interest_next, "availability": availability_next}, info
