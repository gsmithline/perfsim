"""RecommenderEcosystemWorld: a performative recommender as a perfsim environment.

The deployed model theta is the platform ranker: it scores items, exposure is
allocated by softmax over scores, a batched user population chooses items via a
differentiable MNL biased by exposure, and the logged engagement becomes the
supervised training data for the next round. Meanwhile user interests drift
toward whatever was surfaced so both the labels (exposure bias) and the
population (taste drift).

Algorithm 1: theta is queried once per epoch (frozen), the population evolves
`n_steps` inner ticks under that frozen ranking, and the final engagement trains
theta. Mirrors PublicGoodsDilemma.run / the AgentTorch adapter contract.

Differentiability: all behaviors are smooth, so grad_sample/grad_step/grad_run
keep the autograd path theta -> scores -> exposure -> choice -> engagement,
satisfying FullyDifferentiable (usable with PerfGDLearner).
"""

from __future__ import annotations

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.environments.dynamics.stateful_population import State, StatefulPopulationWorld
from perfsim.scenarios.recommender.behaviors import drift_interest, engagement, exposure
from perfsim.scenarios.recommender.choice import affinities, mnl_choice
from perfsim.scenarios.recommender.entities import Corpus, UserPopulation


class RecommenderEcosystemWorld(StatefulPopulationWorld):
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
        if corpus.dim != users.dim:
            raise ValueError(
                f"corpus dim {corpus.dim} != user dim {users.dim}"
            )
        super().__init__({"interest": users.interest0.to(dtype).clone()}, dtype=dtype)
        self._features = corpus.features.to(dtype).detach().clone()
        self._appeal = corpus.appeal.to(dtype).detach().clone()
        self._interest0 = users.interest0.to(dtype).detach().clone()
        self._community = users.community.detach().clone()
        self._k = corpus.n_items
        self._d = corpus.dim
        self._n_users = users.n_users
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temp = float(temp)
        self.eta = float(eta)
        self.noise = float(noise)
        self._item_idx = torch.arange(self._k)
        self._gen: torch.Generator | None = None
        self._scores: Tensor | None = None          # frozen theta(features) per epoch
        self._info: dict[str, Tensor] = {}           # detached, for probes
        self._grad_info: dict[str, Tensor] = {}       # grad-tracked, for PerfGD

    # ---- contract ----------------------------------------------------------
    @property
    def produces_schema(self) -> DataSchema:
        return SUPERVISED_SCHEMA

    @property
    def n_items(self) -> int:
        return self._k

    @property
    def n_users(self) -> int:
        return self._n_users

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
    def last_exposure(self) -> Tensor:
        return self._info["exposure"]

    @property
    def last_engagement(self) -> Tensor:
        return self._info["engagement"]

    @property
    def last_choice(self) -> Tensor:
        return self._info["choice"]

    @property
    def grad_engagement(self) -> Tensor:
        return self._grad_info["engagement"]

    def hidden_quality(self) -> Tensor:
        """True item relevance implied by the ORIGINAL population (never trained on)."""
        with torch.no_grad():
            aff0 = self._interest0 @ self._features.t()      # (N, K)
            return aff0.mean(dim=0) * self._appeal           # (K,)

    def reset(self, seed: int = 0) -> None:
        super().reset(seed=seed)            # restores interest -> interest0 snapshot
        self._gen = torch.Generator()
        self._gen.manual_seed(int(seed))
        self._scores = None
        self._info = {}
        self._grad_info = {}

    # ---- core transition (pure, differentiable) ----------------------------
    def _forward(self, scores: Tensor, interest: Tensor) -> tuple[Data, Tensor, dict[str, Tensor]]:
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
        return data, interest_next, info

    # ---- non-grad path (used by base sample/step and run) ------------------
    def _step(self, model: Model) -> tuple[Data, State]:
        scores = self._scores
        if scores is None:
            with torch.no_grad():
                scores = model(self._features).reshape(-1).to(self._dtype)
        data, interest_next, info = self._forward(scores, self._state["interest"])
        self._info = {k: v.detach().clone() for k, v in info.items()}
        return data, {"interest": interest_next.detach()}

    def run(self, model: Model, n_steps: int) -> Data:
        """Query theta once, evolve the population n_steps, return final engagement."""
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive int; got {n_steps!r}")
        with torch.no_grad():
            self._scores = model(self._features).reshape(-1).to(self._dtype)
        final: Data | None = None
        for _ in range(n_steps):
            final = self.step(model)     # base.step -> _step -> installs next_state
        self._scores = None
        assert final is not None
        return final

    # ---- differentiable path (FullyDifferentiable / PerfGD) ----------------
    def grad_sample(self, model: Model) -> Data:
        scores = model(self._features).reshape(-1).to(self._dtype)   # grad live
        data, _interest_next, info = self._forward(scores, self._state["interest"])
        self._grad_info = info
        self._info = {k: v.detach().clone() for k, v in info.items()}
        return data

    def grad_step(self, model: Model) -> Data:
        scores = model(self._features).reshape(-1).to(self._dtype)
        data, interest_next, info = self._forward(scores, self._state["interest"])
        self._grad_info = info
        self._info = {k: v.detach().clone() for k, v in info.items()}
        self._state = {"interest": interest_next.detach()}
        return data

    def grad_run(self, model: Model, n_steps: int) -> Data:
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive int; got {n_steps!r}")
        scores = model(self._features).reshape(-1).to(self._dtype)   # grad live, once
        interest = self._state["interest"]
        final: Data | None = None
        info: dict[str, Tensor] = {}
        for _ in range(n_steps):
            final, interest, info = self._forward(scores, interest)
        self._grad_info = info
        self._info = {k: v.detach().clone() for k, v in info.items()}
        self._state = {"interest": interest.detach()}
        assert final is not None
        return final