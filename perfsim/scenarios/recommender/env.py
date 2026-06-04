"""RecommenderEcosystemWorld: theta ranks -> exposure -> MNL choice -> engagement.

Labels (exposure bias) and population (taste drift) are both performative. Shared
plumbing lives in RecommenderWorldBase; this class declares state + transition.
"""

from __future__ import annotations

import torch
from torch import Tensor

from perfsim.core.types import Data
from perfsim.environments.dynamics.stateful_population import State
from perfsim.scenarios.recommender._base import RecommenderWorldBase
from perfsim.scenarios.recommender.behaviors import drift_interest, engagement, exposure
from perfsim.scenarios.recommender.choice import affinities, mnl_choice
from perfsim.scenarios.recommender.entities import Corpus, UserPopulation


class RecommenderEcosystemWorld(RecommenderWorldBase):
    """Performative recommender. State = user interests (N, D); data = (item x, engagement y)."""

    def __init__(
        self,
        corpus: Corpus,
        users: UserPopulation,
        *,
        alpha: float = 4.0,
        beta: float = 4.0,
        temp: float = 0.5,
        eta: float = 0.15,
        noise: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            corpus, users,
            {"interest": users.interest0.to(dtype).clone()},
            dtype=dtype,
        )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temp = float(temp)
        self.eta = float(eta)
        self.noise = float(noise)

    def _forward(self, scores: Tensor, state: State) -> tuple[Data, State, dict[str, Tensor]]:
        interest = state["interest"]
        u = exposure(scores, self.beta)                            # (K,)
        log_u = torch.log(u.clamp_min(1e-12))
        aff = affinities(interest, self._features)                 # (N, K)
        choice = mnl_choice(aff, log_u, alpha=self.alpha, temp=self.temp)
        y = engagement(choice, self._appeal)                       # (K,)
        interest_next = drift_interest(interest, choice, self._features, self.eta)
        if self.noise > 0.0 and self._gen is not None:
            interest_next = interest_next + self.noise * torch.randn(
                interest_next.shape, generator=self._gen, dtype=self._dtype
            )
        info = {"exposure": u, "choice": choice, "engagement": y, "scores": scores}
        data: Data = {
            "x": self._features,
            "y": y.reshape(-1, 1),
            "agent_idx": self._item_idx.clone(),
        }
        return data, {"interest": interest_next}, info
