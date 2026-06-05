"""CompetingRecommendersWorld: P rankers share one drifting user population.

Each platform trains only on the engagement its own attention share induced;
interests drift toward the share-weighted consumption mixture. eta_mob > 0
moves per-user shares by multiplicative weights on realized satisfaction
(eta_mob = 0 freezes the split).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import Data
from perfsim.scenarios.recommender.behaviors import exposure
from perfsim.scenarios.recommender.choice import affinities, mnl_choice
from perfsim.scenarios.recommender.entities import Corpus, UserPopulation


class CompetingRecommendersWorld:
    """State = interest (N, D) + shares (N, P); run() returns one Data per platform."""

    def __init__(
        self,
        corpus: Corpus,
        users: UserPopulation,
        *,
        n_platforms: int = 2,
        alpha: float = 4.0,
        beta: float = 4.0,
        temp: float = 0.5,
        eta: float = 0.15,
        eta_mob: float = 0.0,
        shares0: Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if corpus.dim != users.dim:
            raise ValueError(f"corpus dim {corpus.dim} != user dim {users.dim}")
        if n_platforms < 2:
            raise ValueError(f"n_platforms must be >= 2; got {n_platforms}")
        self.n_platforms = int(n_platforms)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temp = float(temp)
        self.eta = float(eta)
        self.eta_mob = float(eta_mob)
        self._dtype = dtype
        self._features = corpus.features.to(dtype).detach().clone()
        self._appeal = corpus.appeal.to(dtype).detach().clone()
        self._interest0 = users.interest0.to(dtype).detach().clone()
        self._community = users.community.detach().clone()
        self._item_idx = torch.arange(corpus.n_items)
        if shares0 is None:
            shares0 = torch.full((users.n_users, self.n_platforms), 1.0 / self.n_platforms)
        if shares0.shape != (users.n_users, self.n_platforms):
            raise ValueError(
                f"shares0 must be ({users.n_users}, {self.n_platforms}); "
                f"got {tuple(shares0.shape)}"
            )
        if not torch.allclose(shares0.sum(dim=1), torch.ones(users.n_users), atol=1e-5):
            raise ValueError("shares0 rows must sum to 1")
        self._shares0 = shares0.to(dtype).detach().clone()
        self._state: dict[str, Tensor] = {}
        self._info: dict[str, Tensor] = {}
        self.reset()

    @property
    def features(self) -> Tensor:
        return self._features

    @property
    def appeal(self) -> Tensor:
        return self._appeal

    @property
    def interest0(self) -> Tensor:
        return self._interest0

    @property
    def community(self) -> Tensor:
        return self._community

    @property
    def current_interest(self) -> Tensor:
        return self._state["interest"]

    @property
    def shares(self) -> Tensor:
        return self._state["shares"]

    @property
    def last_exposure(self) -> Tensor:
        return self._info["exposure"]                # (P, K)

    @property
    def last_choice(self) -> Tensor:
        return self._info["choice"]                  # (P, N, K)

    @property
    def last_engagement(self) -> Tensor:
        return self._info["engagement"]              # (P, K)

    @property
    def last_satisfaction(self) -> Tensor:
        return self._info["satisfaction"]            # (N, P)

    @property
    def last_shares(self) -> Tensor:
        return self._info["shares"]                  # (N, P), as used this step

    def hidden_quality(self) -> Tensor:
        with torch.no_grad():
            aff0 = self._interest0 @ self._features.t()
            return aff0.mean(dim=0) * self._appeal

    def reset(self, seed: int = 0) -> None:
        self._state = {"interest": self._interest0.clone(), "shares": self._shares0.clone()}
        self._info = {}

    def run(self, models: Sequence[Model], n_steps: int) -> list[Data]:
        """Query each theta once, evolve the population n_steps, return per-platform data."""
        if len(models) != self.n_platforms:
            raise ValueError(f"expected {self.n_platforms} models; got {len(models)}")
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive int; got {n_steps!r}")
        with torch.no_grad():
            scores = torch.stack(
                [m(self._features).reshape(-1).to(self._dtype) for m in models]
            )
            out: list[Data] = []
            for _ in range(n_steps):
                out = self._step(scores)
        return out

    def _step(self, scores: Tensor) -> list[Data]:
        interest = self._state["interest"]
        shares = self._state["shares"]
        aff = affinities(interest, self._features)
        choices, datas = [], []
        for p in range(self.n_platforms):
            u = exposure(scores[p], self.beta)
            choice = mnl_choice(
                aff, torch.log(u.clamp_min(1e-12)), alpha=self.alpha, temp=self.temp
            )
            w = shares[:, p].unsqueeze(1)
            y = (choice * w).sum(dim=0) / w.sum().clamp_min(1e-12) * self._appeal
            choices.append((u, choice))
            datas.append({
                "x": self._features,
                "y": y.reshape(-1, 1),
                "agent_idx": self._item_idx.clone(),
            })
        choice_t = torch.stack([c for _, c in choices])                    # (P, N, K)
        sat = (choice_t * aff.unsqueeze(0)).sum(dim=2).t()                 # (N, P)
        target = (shares.t().unsqueeze(2) * (choice_t @ self._features)).sum(dim=0)
        interest_next = interest + self.eta * (target - interest)
        shares_next = shares
        if self.eta_mob > 0.0:
            shares_next = shares * torch.exp(self.eta_mob * sat)
            shares_next = shares_next / shares_next.sum(dim=1, keepdim=True)
        self._info = {
            "exposure": torch.stack([u for u, _ in choices]),
            "choice": choice_t,
            "engagement": torch.stack([d["y"].reshape(-1) for d in datas]),
            "satisfaction": sat,
            "shares": shares,
        }
        self._state = {"interest": interest_next, "shares": shares_next}
        return datas
